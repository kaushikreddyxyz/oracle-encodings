#!/usr/bin/env python3
"""Local (single-decision) comparison of the graft with token-matched steering. Pod side (GPU).

Items come from build_local.py (results/local/sets/g{n}.npz): the prompt shows maze A, the edit is ONE SWAP (remove the
route connection e-, add e+; B is again a DFS spanning tree), and the path is teacher-forced from the origin to the
decision cell d, the last cell the routes of A and B share. At d exactly one next move is correct in A (move_A) and
exactly one in B (move_B). Everything is read at that single token; a greedy rollout from d is scored as a secondary.

Conditions
  base model      none            prompt A, no intervention (floor)
                  oracle          e-'s chunk in the adjacency list rewritten to e+ (two coordinate tokens; ceiling)
                  reg|dom:L..:a.. steering vectors, token-matched: x += alpha * (V[e+] - V[e-]) at resid_pre of the listed
                                  blocks, at every position >= <ADJLIST_END> (prefix + generated tokens) and nowhere else
                  sae:L6:n16:v16  SAE-feature steering at the same positions (the SAEs fitted for the global figure)
                  (vectors and SAEs are the per-model statistics of eval_steering_graftsite.py, stats_sae.pt, unchanged)
  grafted models use their own graft code (GraftDriver): static codes inject from the code's start token, dynamic codes (ego)
  re-inject a frame that follows the teacher-forced prefix and every generated token
  grafted models  graft_A         prompt A, graft features A (no edit)
                  graft_B         prompt A, graft features B (the edit: 2 of the 2n^2 features flipped)
                  both_B          prompt rewritten AND graft features B (consistent reference)
                  prompt_B        prompt rewritten, graft features A (the conflict the other way round)
Saved per condition and item: log p(move_A), log p(move_B), argmax token at the decision token; from the rollout:
crossed e-, used e+, solved B, solved A.

                  sae:L6:s<spec>:n16:v..  the same with a sae_lib dictionary (stats_f*.pt; specs relu-m3-l0.1, topk-m3-k8, btopk-m3-k8, mat-m3-k8, mp-m1-k16)
  python eval_local.py run --run_dir <run> [--stats <stats.pt>[,<stats2.pt>]] [--conds a,b] [--tag _x] [--batch 1000] [--limit 0] [--verify True]
"""
import pathlib as _pl, sys as _sys
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(next(p for p in _HERE.parents if (p / "maze" / "core").is_dir())))
_sys.path.insert(0, str(_HERE))
import maze._pathsetup  # noqa
import json, re, time
from pathlib import Path
import fire, numpy as np, torch
import torch.nn.functional as F
from maze_dataset import SPECIAL_TOKENS
from maze_transformer.training.config import ConfigHolder
from eval_ckpts import feat_idx, shortest_path, load_ckpt, Scorer
import sae_lib

SETS = Path("/workspace/localsets"); STD = Path("/workspace/distsets")
BASE_CONDS = "none,oracle,reg:L0-11:a1,reg:L0-11:a2,reg:L0-11:a4,reg:L0-11:a8,reg:L0-11:a16,reg:L3:a4,reg:L3:a16,dom:L0-11:a1,sae:L6:n16:v16,sae:L6:n16:v64"
GRAFT_CONDS = "graft_A,graft_B,both_B,prompt_B"
FAST_ATTN = True


def _block(blk, resid, H, dh):
    B, P, _ = resid.shape
    q, k, v = blk.attn.qkv(blk.ln1(resid)).split(blk.attn.d_inner, dim=-1)
    q, k, v = (t.view(B, P, H, dh).transpose(1, 2) for t in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=blk.attn.scale)
    resid = resid + blk.attn.out(y.transpose(1, 2).reshape(B, P, -1))
    return resid + blk.mlp(blk.ln2(resid)), k, v


@torch.no_grad()
def decode(model, toks, lens, end_pos, max_new, pe_id, pad_id, steers=None, graft=None):
    """toks [B, P] right-padded, lens [B] real lengths. `steers = {block: fn(x [M, d], item [M]) -> x}` edits resid_pre of
    those blocks at the graft tokens only: positions end_pos <= p < len in the prefill and every generated position.
    `graft` = a GraftDriver: the model's own graft code decides per position what is injected (static codes: rows(conn)
    from the code's start token on; dynamic codes such as ego: a frame that follows the teacher-forced prefix AND every
    generated token, via the code's init_state/step -- the same path as eval_ckpts.greedy_kv).
    Returns log-probs at the last real token [B, V] and the greedy continuation [B, max_new] (pad after <PATH_END>)."""
    dev = toks.device; B, P = toks.shape; H, dh = model.cfg.n_heads, model.cfg.d_head; ar = torch.arange(B, device=dev)
    pos = torch.arange(P, device=dev)
    resid = model.embed(toks) + model.pos_embed(pos)[None]
    bi, pi = ((pos[None] >= end_pos[:, None]) & (pos[None] < lens[:, None])).nonzero(as_tuple=True)
    if graft is not None:
        delta_full, gstate = graft.prefill(toks, lens, end_pos)
    caches = []
    for i, blk in enumerate(model.blocks):
        if steers and i in steers:
            resid[bi, pi] = steers[i](resid[bi, pi], bi)
        if graft is not None and i == graft.layer:
            resid = resid + delta_full.to(resid.dtype)
        resid, k, v = _block(blk, resid, H, dh)
        K = torch.zeros(B, H, P + max_new, dh, device=dev, dtype=k.dtype); K[:, :, :P] = k
        V = torch.zeros(B, H, P + max_new, dh, device=dev, dtype=v.dtype); V[:, :, :P] = v
        caches.append((K, V))
    logp0 = model.unembed(model.ln_final(resid[ar, lens - 1])).float().log_softmax(-1)
    cur = logp0.argmax(-1)
    out = torch.full((B, max_new), pad_id, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev); allpos = torch.arange(P + max_new, device=dev)
    for step in range(max_new):
        out[:, step] = torch.where(done, torch.full_like(cur, pad_id), cur)
        done = done | (cur == pe_id)
        if bool(done.all()) or step == max_new - 1:
            break
        t = lens + step                                                    # per-row write position (overwrites pad K/V)
        T = int(t.max()) + 1
        mask = (allpos[None, :T] <= t[:, None])[:, None, None, :]
        resid = model.embed(cur) + model.pos_embed(t)
        if graft is not None:
            delta_row, gstate = graft.step(gstate, cur)
        for i, blk in enumerate(model.blocks):
            if steers and i in steers:
                resid = steers[i](resid, ar)
            if graft is not None and i == graft.layer:
                resid = resid + delta_row.to(resid.dtype)
            q, k, v = blk.attn.qkv(blk.ln1(resid)).split(blk.attn.d_inner, dim=-1)
            K, V = caches[i]; K[ar, :, t] = k.view(B, H, dh); V[ar, :, t] = v.view(B, H, dh)
            if FAST_ATTN:       # one query token: plain matmul on the cache views (SDPA copies the non-contiguous K/V slices every step and layer)
                att = (torch.matmul(q.view(B, H, 1, dh), K[:, :, :T].transpose(-1, -2)) * blk.attn.scale).masked_fill(~mask, float("-inf")).softmax(-1)
                y = torch.matmul(att, V[:, :, :T])
            else:
                y = F.scaled_dot_product_attention(q.view(B, H, 1, dh), K[:, :, :T], V[:, :, :T], attn_mask=mask, scale=blk.attn.scale)
            resid = resid + blk.attn.out(y.reshape(B, -1))
            resid = resid + blk.mlp(blk.ln2(resid))
        cur = model.unembed(model.ln_final(resid)).argmax(-1)
    return logp0, out


class OldSAE:
    """ReLU SAE as saved in stats_sae.pt by the token-matched steering run (state: b_pre, enc.*, dec.*)."""
    def __init__(self, entry, device):
        st = entry["state"]; self.scale = float(entry["scale"]); assert int(entry.get("topk", 0)) == 0
        self.b_pre = st["b_pre"].to(device).float(); self.We = st["enc.weight"].to(device).float(); self.be = st["enc.bias"].to(device).float()
        self.Wd = st["dec.weight"].to(device).float()                        # [d, n_lat]

    def encode(self, xs):
        return torch.relu((xs - self.b_pre) @ self.We.T + self.be)


def make_steer(cname, stats, diff, device):
    """diff [B, 2n^2] in {-1, 0, +1} = (B open) - (A open). Same edits as eval_steering_graftsite.make_steer."""
    cond = cname.split(":"); kind = cond[0]; args = {a[0]: a[1:] for a in cond[1:]}
    lspec = args.get("L", "0"); layers = list(range(int(lspec.split("-")[0]), int(lspec.split("-")[-1]) + 1))
    if kind in ("dom", "reg"):
        alpha = float(args.get("a", "1")); out = {}
        for l in layers:
            delta = alpha * (diff @ stats["V_dom" if kind == "dom" else "V_reg"][l].to(device).float())
            out[l] = (lambda x, item, delta=delta: x + delta[item])
        return out
    if kind == "sae":
        nlat = int(args.get("n", "16")); value = float(args.get("v", "1"))
        if "s" in args:                                                      # sae_lib dictionaries (stats_f*.pt): same edit as eval_steering_graftsite.make_steer
            entry = stats["sae"][f"L{layers[0]}:{args['s']}"]; sae = sae_lib.build(entry["sd"], device); scale = float(entry["sd"]["scale"])
            n_lat = int(entry["sd"]["n_lat"]); enc = sae.encode; W_dec = sae.W_dec                  # W_dec [n_lat, d]
        else:                                                                # the 2026-09-19 ReLU dictionaries (stats_sae.pt)
            entry = stats["sae"][f"L{layers[0]}:l10.03"]; old = OldSAE(entry, device); scale = old.scale; n_lat = int(entry["n_lat"]); enc = old.encode; W_dec = old.Wd.T
        D = torch.zeros(diff.shape[1], n_lat, device=device)
        D.scatter_(1, entry["lat"][:, :nlat].to(device), entry["dlat"][:, :nlat].to(device).float())
        dA = value * (diff @ D)
        def steer(x, item):
            xs = x / scale; a = enc(xs); a2 = (a + dA[item]).clamp(min=0)
            return (xs + (a2 - a) @ W_dec) * scale
        return {layers[0]: steer}
    raise ValueError(cname)


class GraftDriver:
    """The grafted model's own code applied in decode(): prefill delta [B, P, d] (0 outside the code's active
    positions, i.e. before its start token or on padding) + a state that step() extends token by token."""
    def __init__(self, injector, conn, device):
        from maze.core.graft import as_graft
        self.g = as_graft(injector); self.layer = int(self.g.layer)
        self.conn = torch.tensor(conn, device=device)

    def prefill(self, toks, lens, end_pos):
        attn = (torch.arange(toks.shape[1], device=toks.device)[None, :] < lens[:, None]).long()
        code = self.g.code
        return self.g.delta(code.features(toks, attn, self.conn, end_pos)), code.init_state(toks, attn, self.conn, end_pos)

    def step(self, state, cur):
        feats, state = self.g.code.step(state, cur)
        return self.g.delta(feats), state


def run(run_dir: str, stats: str = "", out_dir: str = "/workspace/local_out", batch: int = 1000, limit: int = 0,
        conds: str = "", verify: bool = False, tag: str = ""):
    device = torch.device("cuda"); torch.set_float32_matmul_precision("high")
    run_ = Path(run_dir); n = int(re.search(r"dfs-g(\d+)-", run_.name).group(1)); seed_tag = run_.parent.name
    kind = "fixed" if "inj-fixed" in run_.name else "trainable" if "inj-trainable" in run_.name else "base"
    out = Path(out_dir) / f"g{n}__{kind}__{seed_tag}{tag}.npz"
    if out.exists():
        print(f"exists {out}", flush=True); return
    S = np.load(STD / f"g{n}-s4321-n3000/distset_K12_w0.05.npz"); Lc = np.load(SETS / f"g{n}.npz")
    assert str(Lc["src_sha"]) == str(S["sha"]), "local set was built from a different standard set"
    cfg = ConfigHolder.load(json.load(open(run_ / "config_holder.json"))); vocab = cfg.tokenizer.get_vocab()
    id_to_coord = {int(i): (int(r), int(c)) for i, r, c in S["coord_ids"]}; coord_to_id = {c: i for i, c in id_to_coord.items()}
    pe_id, pad_id = int(S["path_end_id"]), int(S["pad_id"])
    semi_id, conn_id = vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE], vocab[SPECIAL_TOKENS.CONNECTOR]
    cell = lambda v: (int(v) // n, int(v) % n)
    N = len(Lc["maze"]) if not limit else min(limit, len(Lc["maze"]))
    maze = Lc["maze"][:N]; conn_a = S["conn_a"][maze]; conn_b = Lc["conn_b"][:N]; jdec = Lc["j"][:N]
    rem, add = Lc["rem"][:N], Lc["add"][:N]; move_a, move_b = Lc["move_a"][:N], Lc["move_b"][:N]
    route_a, route_b = Lc["route_a"][:N], Lc["route_b"][:N]
    prompts_a = S["prompts"][maze].copy(); end_pos = S["end_pos"][maze]; P0 = prompts_a.shape[1]

    # ---- independent BFS verification of every item (eval_ckpts.shortest_path, not the builder's code)
    for r in range(N):
        i = maze[r]; o, t = tuple(S["origin"][i]), tuple(S["target"][i])
        pa = shortest_path(conn_a[r], o, t, n); pb = shortest_path(conn_b[r], o, t, n); j = int(jdec[r])
        assert pa == [cell(v) for v in route_a[r] if v >= 0] and pb == [cell(v) for v in route_b[r] if v >= 0]
        assert pa[: j + 1] == pb[: j + 1] and pa[j + 1] == cell(move_a[r]) and pb[j + 1] == cell(move_b[r]) and pa[j + 1] != pb[j + 1]
        assert conn_a[r][rem[r]] and not conn_b[r][rem[r]] and conn_b[r][add[r]] and not conn_a[r][add[r]]
        assert int((conn_a[r] != conn_b[r]).sum()) == 2 and conn_b[r].sum() == n * n - 1
        assert rem[r] in {feat_idx(pa[m], pa[m + 1], n) for m in range(len(pa) - 1)} and add[r] in {feat_idx(pb[m], pb[m + 1], n) for m in range(len(pb) - 1)}
        assert prompts_a[r, -1] == coord_to_id[o]
    # ---- rewritten prompts (ceiling): the chunk of e- becomes e+ (random orientation), nothing else moves
    rng = np.random.default_rng([4321, n, 53]); prompts_b = prompts_a.copy()
    lut = np.full(max(id_to_coord) + 1, -1)
    for tid, (r_, c_) in id_to_coord.items(): lut[tid] = r_ * n + c_
    for r in range(N):
        row = prompts_a[r]; ps = np.flatnonzero((row[3:] == semi_id) & (row[1:-2] == conn_id)) + 3
        ps = ps[ps < end_pos[r]]; assert len(ps) == n * n - 1
        fa = np.array([feat_idx(cell(lut[row[p - 3]]), cell(lut[row[p - 1]]), n) for p in ps])
        assert set(fa.tolist()) == set(np.flatnonzero(conn_a[r]).tolist())
        p = int(ps[np.flatnonzero(fa == rem[r])[0]])
        (c1, c2) = ((add[r] // n, add[r] % n), (add[r] // n + 1, add[r] % n)) if add[r] < n * n else (((add[r] - n * n) // n, (add[r] - n * n) % n), ((add[r] - n * n) // n, (add[r] - n * n) % n + 1))
        if rng.integers(0, 2): c1, c2 = c2, c1
        prompts_b[r, p - 3] = coord_to_id[c1]; prompts_b[r, p - 1] = coord_to_id[c2]
    # ---- teacher-forced prefixes (origin is the prompt's last token; cells 1..j follow)
    Pmax = P0 + int(jdec.max()); lens = P0 + jdec.astype(np.int64)
    def with_prefix(prompts):
        tk = np.full((N, Pmax), pad_id, dtype=np.int64); tk[:, :P0] = prompts
        for r in range(N):
            for m in range(1, int(jdec[r]) + 1): tk[r, P0 + m - 1] = coord_to_id[cell(route_a[r][m])]
        return tk
    toks_a, toks_b = with_prefix(prompts_a), with_prefix(prompts_b)
    tid_a = np.array([coord_to_id[cell(v)] for v in move_a]); tid_b = np.array([coord_to_id[cell(v)] for v in move_b])

    model, injector = load_ckpt(run_ / "model.final.pt", run_ / "injector.final.pt" if kind != "base" else None, device)
    assert (injector is not None) == (kind != "base")
    st = None
    if kind == "base" and stats:                                             # one or several stats files; their SAE tables are merged
        for sp in (stats if isinstance(stats, (list, tuple)) else str(stats).split(",")):
            d_ = torch.load(sp, weights_only=False, map_location="cpu"); assert int(d_["grid_n"]) == n
            if st is None: st = d_
            else: st["sae"].update(d_["sae"])
    if isinstance(conds, (list, tuple)): conds = ",".join(str(c) for c in conds)          # fire turns a,b into a tuple
    cond_list = (conds or (BASE_CONDS if kind == "base" else GRAFT_CONDS)).split(",")
    if kind == "base" and st is None: cond_list = [c for c in cond_list if c in ("none", "oracle")]
    n_ctx = int(model.cfg.n_ctx); res = {}; t00 = time.time()
    order = np.argsort(lens, kind="stable")                                  # batches of similar length
    for cname in cond_list:
        t0 = time.time(); key = cname.replace(":", "_")
        lp_a = np.zeros(N, np.float32); lp_b = np.zeros(N, np.float32); amax = np.zeros(N, np.int32)
        crossed = np.zeros(N, bool); used = np.zeros(N, bool); solved_b = np.zeros(N, bool); solved_a = np.zeros(N, bool); rlen = np.zeros(N, np.int16)
        tk_src = toks_b if cname in ("oracle", "both_B", "prompt_B") else toks_a
        for lo in range(0, N, batch):
            idx = order[lo: lo + batch]; Pb = int(lens[idx].max())
            max_new = min(n * n + 2 - int(jdec[idx].min()), n_ctx - Pb)      # same n^2+2 path budget as the global eval, within the context
            tk = torch.tensor(tk_src[idx][:, :Pb], device=device); ln = torch.tensor(lens[idx], device=device); ep = torch.tensor(end_pos[idx], device=device)
            steers = graft = None
            if cname in ("none", "oracle"): pass
            elif cname in ("graft_A", "prompt_B"): graft = GraftDriver(injector, conn_a[idx], device)
            elif cname in ("graft_B", "both_B"): graft = GraftDriver(injector, conn_b[idx], device)
            else: steers = make_steer(cname, st, torch.tensor(conn_b[idx].astype(np.float32) - conn_a[idx].astype(np.float32), device=device), device)
            logp0, gen = decode(model, tk, ln, ep, max_new, pe_id, pad_id, steers, graft)
            ar = torch.arange(len(idx), device=device)
            lp_a[idx] = logp0[ar, torch.tensor(tid_a[idx], device=device)].cpu().numpy(); lp_b[idx] = logp0[ar, torch.tensor(tid_b[idx], device=device)].cpu().numpy()
            amax[idx] = logp0.argmax(-1).cpu().numpy(); gen = gen.cpu().numpy()
            if verify and lo == 0:
                _verify(model, injector, kind, cname, tk, ln, ep, conn_a[idx], conn_b[idx], logp0, gen, pe_id, pad_id, device)
            for q, r in enumerate(idx):
                g = gen[q].tolist()
                for stop in (pe_id, pad_id):
                    if stop in g: g = g[: g.index(stop)]
                bad_tok = any(t_ not in id_to_coord for t_ in g)
                coords = [cell(v) for v in route_a[r][: int(jdec[r]) + 1]] + [id_to_coord[t_] for t_ in g if t_ in id_to_coord]
                feats = [feat_idx(a, b, n) if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 else -1 for a, b in zip(coords[:-1], coords[1:])]
                tgt = tuple(S["target"][maze[r]]); reached = len(coords) > 1 and coords[-1] == tgt
                crossed[r] = rem[r] in feats; used[r] = add[r] in feats; rlen[r] = len(g)
                solved_b[r] = reached and not bad_tok and all(f >= 0 and conn_b[r][f] for f in feats)
                solved_a[r] = reached and not bad_tok and all(f >= 0 and conn_a[r][f] for f in feats)
        for k_, v_ in dict(lp_a=lp_a, lp_b=lp_b, argmax=amax, crossed=crossed, used=used, solved_b=solved_b, solved_a=solved_a, rollout_len=rlen).items():
            res[f"{key}__{k_}"] = v_
        fam0 = Lc["family"][:N] == 0
        print(f"[g{n} {kind} {seed_tag} {cname}] {time.time() - t0:.0f}s | dfs items: p(move_B) {np.exp(lp_b[fam0]).mean():.3f} p(move_A) {np.exp(lp_a[fam0]).mean():.3f} "
              f"argmax=B {(amax[fam0] == tid_b[fam0]).mean():.3f} | rollout: crossed e- {crossed[fam0].mean():.3f} used e+ {used[fam0].mean():.3f} solved B {solved_b[fam0].mean():.3f} solved A {solved_a[fam0].mean():.3f}", flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, grid_n=n, kind=kind, seed_tag=seed_tag, run=run_.name, set_sha=str(Lc["sha"]), n_items=N, conds=np.array(cond_list),
                        tid_a=tid_a, tid_b=tid_b, **res)
    print(f"RUN_DONE {out.name} {time.time() - t00:.0f}s", flush=True)


@torch.no_grad()
def _verify(model, injector, kind, cname, tk, ln, ep, ca, cb, logp0, gen, pe_id, pad_id, device, m=48):
    """decode() against the repo's full forward (no cache, no custom blocks) on the first m rows: decision log-probs and
    the first generated tokens, for the unsteered base model and the grafted model."""
    if cname not in ("none", "oracle", "graft_A", "graft_B", "both_B", "prompt_B"): return
    sc = Scorer(model, injector, device); conn = torch.tensor(cb if cname in ("graft_B", "both_B") else ca, device=device)[:m]
    toks = tk[:m].clone(); cur = ln[:m].clone(); B = len(toks); ar = torch.arange(B, device=device); steps = 6
    toks = torch.cat([toks, torch.full((B, steps), pad_id, device=device)], 1); worst = 0.0; agree = True
    for s in range(steps):
        W = int(cur.max()); attn = (torch.arange(W, device=device)[None] < cur[:, None]).long()
        lg = sc.logits(toks[:, :W], attn, conn, ep[:m])[ar, cur - 1].float().log_softmax(-1)
        if s == 0: worst = float((lg - logp0[:m]).abs().max())
        nxt = lg.argmax(-1); agree &= bool((nxt == torch.tensor(gen[:m, s], device=device)).logical_or(torch.tensor(gen[:m, s], device=device) == pad_id).all())
        toks[ar, cur] = nxt; cur = cur + 1
    print(f"    verify {cname}: max |dlogp| at the decision token {worst:.2e}; first {steps} greedy tokens agree: {agree}", flush=True)


if __name__ == "__main__":
    fire.Fire(dict(run=run))
