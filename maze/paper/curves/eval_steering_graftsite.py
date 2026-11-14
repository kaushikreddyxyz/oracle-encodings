#!/usr/bin/env python3
"""Token-matched steering of the BASE models on the standard distance-stratified eval set.

Fairness rule: a steering method may be FITTED only on, and APPLIED only to, the token positions the graft writes to.
For the `complete` graft that is every position from <ADJLIST_END> onward (origin/target block and the path, including
every token generated during the rollout); the adjacency list is never touched. `graft_tokens()` is the single place
that defines those positions, for fitting and for application alike; a different graft type = a different mask there.
The layer is free (the graft itself writes to resid_pre of block 0): `L<k>` = resid_pre of block k, `L<a>-<b>` = every
block a..b at once, each with its own fitted vectors.

Same items as eval_distance.py: prompt = maze A, target = maze B. Every method turns the feature difference
(B open - A open, over the 2n^2 connection features) into an edit of the residual stream at the graft tokens:
  dom   steering vectors, difference of means: V_f = mean(x | f open) - mean(x | f closed) over graft tokens
  reg   steering vectors, ridge regression of x on the +-1 features with token-identity and position one-hots as
        covariates (the same linear form as the graft's W_inj, but read off the base model instead of trained)
        edit for both: x += alpha * sum_f (b_f - a_f) V_f
  sae   SAE features: ReLU SAE trained on graft-token residuals; per connection the n latents whose mean activation
        differs most (standardised) between open and closed; latents += v * sum_f (b_f - a_f) (mean_open - mean_closed),
        clipped at 0, written back through the decoder (SAE error term kept)
New steering methods: add a fitter to `fit_layer` and a branch to `make_steer`; the token mask stays shared.

  python eval_steering_graftsite.py run --run_dir <base run> --conds dom:L3:a4,reg:L0:a2,sae:L3:sl10.03:n16:v2 [--tune True]
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
from maze_dataset import SPECIAL_TOKENS, MazeDataset, MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators
from maze.core.data import PretokDataset, pretokenize
from eval_ckpts import load_ckpt
import eval_distance as ED
from eval_steering import load_vocab, _block
import sae_lib

WORK = Path("/workspace/steer_gs")


def graft_tokens(end_pos, n_pos, device):
    """[B, n_pos] bool: the positions the `complete` graft writes to (>= <ADJLIST_END>)."""
    return torch.arange(n_pos, device=device)[None, :] >= end_pos[:, None]


# ----------------------------------------------------------------------------- model passes
@torch.no_grad()
def resid_pre_all(model, toks):
    """resid_pre of every block, [n_layers, B, P, d] (right-padded rows; causal => padding never reaches real positions)."""
    H, dh = model.cfg.n_heads, model.cfg.d_head
    resid = model.embed(toks) + model.pos_embed(torch.arange(toks.shape[1], device=toks.device))[None]
    out = []
    for blk in model.blocks:
        out.append(resid)
        resid, _, _ = _block(blk, resid, H, dh)
    return torch.stack(out)


@torch.no_grad()
def greedy_kv_steer(model, prompt, end_pos, max_new, path_end_id, pad_id, steers=None):
    """Greedy KV-cache decoding; `steers = {layer: steer(x [M, d], item [M]) -> x}` edits resid_pre of those blocks at the graft tokens:
    prompt positions >= end_pos in the prompt pass, and every generated position (always a graft token)."""
    dev = prompt.device; B, P = prompt.shape; H, dh = model.cfg.n_heads, model.cfg.d_head
    pos = torch.arange(P + max_new, device=dev)
    resid = model.embed(prompt) + model.pos_embed(pos[:P])[None]
    bi, pi = graft_tokens(end_pos, P, dev).nonzero(as_tuple=True)
    all_items = torch.arange(B, device=dev)
    caches = []
    for i, blk in enumerate(model.blocks):
        if steers and i in steers:
            resid[bi, pi] = steers[i](resid[bi, pi], bi)
        resid, k, v = _block(blk, resid, H, dh)
        K = torch.empty(B, H, P + max_new, dh, device=dev, dtype=k.dtype); K[:, :, :P] = k
        V = torch.empty(B, H, P + max_new, dh, device=dev, dtype=v.dtype); V[:, :, :P] = v
        caches.append((K, V))
    cur = model.unembed(model.ln_final(resid[:, -1])).argmax(-1)
    out = torch.full((B, max_new), pad_id, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for step in range(max_new):
        out[:, step] = torch.where(done, torch.full_like(cur, pad_id), cur)
        done = done | (cur == path_end_id)
        if bool(done.all()) or step == max_new - 1:
            break
        t = P + step
        resid = model.embed(cur) + model.pos_embed(pos[t])[None]
        for i, blk in enumerate(model.blocks):
            if steers and i in steers:
                resid = steers[i](resid, all_items)
            q, k, v = blk.attn.qkv(blk.ln1(resid)).split(blk.attn.d_inner, dim=-1)
            K, V = caches[i]; K[:, :, t] = k.view(B, H, dh); V[:, :, t] = v.view(B, H, dh)
            y = F.scaled_dot_product_attention(q.view(B, H, 1, dh), K[:, :, : t + 1], V[:, :, : t + 1], scale=blk.attn.scale)
            resid = resid + blk.attn.out(y.reshape(B, -1))
            resid = resid + blk.mlp(blk.ln2(resid))
        cur = model.unembed(model.ln_final(resid)).argmax(-1)
    return out


# ----------------------------------------------------------------------------- fitting (graft tokens only)
def fit_layer(X, feat01, tok, rel, free, n_tok, n_rel, ridge=1.0):
    """X [T, d] graft-token residuals of one layer; feat01 [T, 2n^2] the token's maze features; tok/rel token id and
    position relative to <ADJLIST_END>. Returns V_dom, V_reg [2n^2, d] = effect of closed -> open."""
    Xf = X.float(); T = Xf.shape[0]; Ff = feat01.float()
    n_open = Ff.sum(0); s_open = Ff.T @ Xf
    ok = (n_open > 0) & (n_open < T)
    mean_open = s_open / n_open.clamp(min=1)[:, None]; mean_closed = (Xf.sum(0)[None] - s_open) / (T - n_open).clamp(min=1)[:, None]
    V_dom = torch.where(ok[:, None], mean_open - mean_closed, torch.zeros_like(mean_open))
    # ridge: x ~ [+-1 free features | token one-hot | relative-position one-hot]; feature effect controls for what token it is
    Z = torch.cat([Ff[:, free] * 2 - 1, F.one_hot(tok, n_tok).float(), F.one_hot(rel.clamp(max=n_rel - 1), n_rel).float()], 1)
    A = Z.T @ Z; A.diagonal().add_(ridge)
    W = torch.linalg.solve(A, Z.T @ Xf)[: int(free.sum())]
    V_reg = torch.zeros_like(V_dom); V_reg[free] = 2 * W                       # +-1 coding: closed -> open is 2 units
    return V_dom, V_reg


def fit_sae(X, feat01, free, spec, f_exp, steps, n_lat_max, device):
    sd = sae_lib.train(X, spec, f_exp, device, steps=steps); sae = sae_lib.build(sd, device); scale = sd["scale"]
    with torch.no_grad():
        acts = torch.cat([sae.encode(X[lo: lo + 65536].float() / scale) for lo in range(0, len(X), 65536)])
        Ff = feat01.float(); T = len(acts); n_open = Ff.sum(0)
        s_open = Ff.T @ acts
        mo = s_open / n_open.clamp(min=1)[:, None]; mc = (acts.sum(0)[None] - s_open) / (T - n_open).clamp(min=1)[:, None]
        diff = torch.where(free[:, None], mo - mc, torch.zeros_like(mo))
        z = diff / (acts.std(0)[None] + 1e-6)
        top = z.abs().topk(min(n_lat_max, sd["n_lat"]), dim=1).indices
    return dict(sd=sd, lat=top.cpu(), dlat=diff.gather(1, top).cpu(), max_abs_z=float(z.abs().max(1).values[free].median())), sae


def prep(run_dir: str, n_train: int = 30000, train_seed: int = 777, sae_layers="", saes="relu-m1.5-l0.03",
         sae_steps: int = 10000, n_lat_max: int = 16, tag: str = "", part: str = ""):
    device = torch.device("cuda"); torch.set_float32_matmul_precision("high")
    run = Path(run_dir); n = int(re.search(r"dfs-g(\d+)-", run.name).group(1))
    out = WORK / run.parent.name / f"stats{tag}{part}.pt"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg, vocab, id_to_coord = load_vocab(run)
    pre = WORK / f"train-g{n}-s{train_seed}-n{n_train}.pt"
    if not pre.exists():
        ds = MazeDataset.from_config(MazeDatasetConfig(name="steertrain", grid_n=n, n_mazes=n_train, maze_ctor=LatticeMazeGenerators.gen_dfs, seed=train_seed),
                                     do_generate=True, load_local=False, save_local=False, gen_parallel=False, verbose=False)
        pretokenize(ds, cfg.maze_tokenizer, vocab, pre, bos_id=int(cfg.tokenizer.bos_token_id), workers=8)
    pt = PretokDataset(pre); rows = [pt[i] for i in range(len(pt))]
    pad_id = vocab[SPECIAL_TOKENS.PADDING]; L = max(len(s) for s, _, _ in rows)
    toks = torch.full((len(rows), L), pad_id, dtype=torch.long)
    for i, (s, _, _) in enumerate(rows): toks[i, : len(s)] = s
    conn = torch.stack([c for _, c, _ in rows]).bool(); end = torch.tensor([int(e) for _, _, e in rows])
    model, _ = load_ckpt(run / "model.final.pt", None, device)
    from maze_distance import free_mask
    free = torch.tensor(free_mask(n), device=device)
    t0 = time.time(); Xs = None; meta = []
    for lo in range(0, len(toks), 500):
        tb = toks[lo: lo + 500].to(device); eb = end[lo: lo + 500].to(device)
        m = graft_tokens(eb, tb.shape[1], device) & (tb != pad_id)                # fit on graft tokens only
        R = resid_pre_all(model, tb)[:, m].half()                                 # [n_layers, T_b, d]
        Xs = [[] for _ in range(R.shape[0])] if Xs is None else Xs
        for l in range(R.shape[0]): Xs[l].append(R[l])
        bi, pi = m.nonzero(as_tuple=True)
        meta.append((bi + lo, tb[bi, pi], pi - eb[bi]))
    item = torch.cat([a for a, _, _ in meta]); tok = torch.cat([b for _, b, _ in meta]); rel = torch.cat([c for _, _, c in meta])
    feat01 = conn.to(device)[item]
    n_tok = max(vocab.values()) + 1; n_rel = n * n + 12
    stats = dict(grid_n=n, n_tokens=int(len(item)), V_dom={}, V_reg={}, sae={}, log={})
    sae_layers = [int(x) for x in str(sae_layers).replace("(", "").replace(")", "").split(",") if str(x).strip() != ""] if sae_layers != "" else []
    sae_specs = [x for x in (saes if isinstance(saes, (list, tuple)) else str(saes).split(",")) if x]
    f_exp = 2 * n * (n - 1) + n * n                                    # connections + cell identities expected at the graft tokens
    n_ho = max(100, int(0.05 * len(toks))); tho = toks[-n_ho:].to(device); eho = end[-n_ho:].to(device); real_ho = tho != pad_id
    site_ho = graft_tokens(eho, tho.shape[1], device) & real_ho
    after = (tho == vocab[SPECIAL_TOKENS.PATH_START]).cumsum(1) > 0; loss_mask = after & real_ho & torch.roll(real_ho, -1, 1); loss_mask[:, -1] = False
    for l in range(len(Xs)):
        X = torch.cat(Xs[l]); Xs[l] = None
        Vd, Vr = fit_layer(X, feat01, tok, rel, free, n_tok, n_rel)
        stats["V_dom"][l] = Vd.cpu(); stats["V_reg"][l] = Vr.cpu()
        stats["log"][f"L{l}"] = dict(dom_norm=float(Vd[free].norm(dim=1).mean()), reg_norm=float(Vr[free].norm(dim=1).mean()), resid_norm=float(X.float().norm(dim=1).mean()))
        if l in sae_layers:
            for spec in sae_specs:
                key = f"L{l}:{spec}"
                stats["sae"][key], sae = fit_sae(X, feat01, free, spec, f_exp, sae_steps, n_lat_max, device)
                sd = stats["sae"][key]["sd"]; lrec = sae_lib.loss_recovered(model, _block, tho, loss_mask, site_ho, l, sae, sd["scale"])
                stats["sae"][key]["log"] = dict(n_lat=sd["n_lat"], f_exp=f_exp, heldout=sd["heldout"], converged_ratio=sd["converged_ratio"], curve=sd["curve"],
                                                max_abs_z=stats["sae"][key]["max_abs_z"], **lrec)
                print(f"[prep g{n}] SAE {key}: n_lat {sd['n_lat']} heldout {sd['heldout']} loss_recovered {lrec['loss_recovered']:.3f} "
                      f"median best-latent |z| {stats['sae'][key]['max_abs_z']:.2f}", flush=True); del sae
        del X
    stats["log"]["seconds"] = time.time() - t0
    torch.save(stats, out); json.dump(dict(n_tokens=stats["n_tokens"], **stats["log"], sae={k: v["log"] for k, v in stats["sae"].items()}),
                                     open(out.with_suffix(".json"), "w"), indent=1, default=float)
    print(f"[prep g{n}] {stats['n_tokens']} graft tokens, done {time.time() - t0:.0f}s -> {out}", flush=True)
    return out


# ----------------------------------------------------------------------------- application (graft tokens only)
def make_steer(cond, stats, diff, device):
    """diff [B, 2n^2] = (B open) - (A open) in {-1, 0, +1}. Returns {layer: steer(x [M, d], item [M]) -> x}."""
    kind = cond[0]; args = {a[0]: a[1:] for a in cond[1:]}
    lspec = args.get("L", "0"); layers = list(range(int(lspec.split("-")[0]), int(lspec.split("-")[-1]) + 1))
    if kind in ("dom", "reg"):
        alpha = float(args.get("a", "1")); out = {}
        for l in layers:
            delta = alpha * (diff @ stats["V_dom" if kind == "dom" else "V_reg"][l].to(device).float())     # [B, d]
            out[l] = (lambda x, item, delta=delta: x + delta[item])
        return out
    layer = layers[0]
    if kind == "sae":
        sd = stats["sae"][f"L{layer}:{args.get('s', 'relu-m1.5-l0.03')}"]
        sae = sae_lib.build(sd["sd"], device)
        nlat = int(args.get("n", "16")); value = float(args.get("v", "1")); scale = sd["sd"]["scale"]
        D = torch.zeros(diff.shape[1], sd["sd"]["n_lat"], device=device)
        D.scatter_(1, sd["lat"][:, :nlat].to(device), sd["dlat"][:, :nlat].to(device))
        dA = value * (diff @ D)                                                   # [B, n_lat]
        def steer(x, item):
            xs = x / scale; a = sae.encode(xs); a2 = (a + dA[item]).clamp(min=0)
            return (xs + (a2 - a) @ sae.W_dec) * scale
        return {layer: steer}
    raise ValueError(kind)


def run(run_dir: str, conds: str, tune: bool = False, out_dir: str = "/workspace/steer_gs_out", batch: int = 1000,
        n_mazes: int = 3000, seed: int = 4321, tune_mazes: int = 300, tune_seed: int = 999, tag: str = "",
        sae_layers="", saes: str = "", n_train: int = 30000, part: str = ""):
    device = torch.device("cuda"); torch.set_float32_matmul_precision("high")
    run_ = Path(run_dir); n = int(re.search(r"dfs-g(\d+)-", run_.name).group(1)); seed_tag = run_.parent.name
    set_path = ED.build(n, str(run_ / "config_holder.json"), n_mazes=tune_mazes if tune else n_mazes, seed=tune_seed if tune else seed)
    S = dict(np.load(set_path)); sha = str(S["sha"])
    stats = torch.load(prep(run_dir, sae_layers=sae_layers, saes=saes, n_train=n_train, tag=tag, part=part), weights_only=False)
    model, _ = load_ckpt(run_ / "model.final.pt", None, device)
    id_to_coord = {int(i): (int(r), int(c)) for i, r, c in S["coord_ids"]}
    pe_id, pad_id = int(S["path_end_id"]), int(S["pad_id"]); maze, conn_a, conn_b = S["maze"], S["conn_a"], S["conn_b"]
    origin_c, target_c, lv = S["origin"], S["target"], S["level"]; N = len(maze); max_new = n * n + 2
    prompts_all = torch.tensor(S["prompts"], device=device); end_all = torch.tensor(S["end_pos"], device=device, dtype=torch.long)
    # KV cache is 12 layers x 2 x 256 floats per position per item: cap the batch to ~45% of this GPU's memory
    per_item = 12 * 2 * 256 * 4 * (int(S["prompts"].shape[1]) + max_new) * 1.6
    batch = max(200, min(int(batch), int(0.45 * torch.cuda.get_device_properties(0).total_memory / per_item)))
    conds = conds if isinstance(conds, (list, tuple)) else str(conds).split(",")
    for cname in conds:
        out = Path(out_dir) / ("tune" if tune else "eval") / ("gs_" + cname.replace(":", "_") + tag) / f"g{n}__base__{seed_tag}.npz"
        if out.exists(): continue
        cond = cname.split(":"); t0 = time.time()
        solved_b = np.zeros(N, bool); solved_a = np.zeros(N, bool); reached = np.zeros(N, bool)
        nbad_b = np.zeros(N, np.int16); nbad_a = np.zeros(N, np.int16); malformed = np.zeros(N, bool)
        for lo in range(0, N, batch):
            hi = min(N, lo + batch); it = torch.tensor(maze[lo:hi], device=device)
            diff = torch.tensor(conn_b[lo:hi].astype(np.float32) - conn_a[maze[lo:hi]].astype(np.float32), device=device)
            steers = None if cond[0] == "none" else make_steer(cond, stats, diff, device)
            gen = greedy_kv_steer(model, prompts_all[it], end_all[it], max_new, pe_id, pad_id, steers).cpu().numpy()
            for k in range(hi - lo):
                j = lo + k; i = maze[j]; g = gen[k].tolist()
                for stop in (pe_id, pad_id):
                    if stop in g: g = g[: g.index(stop)]
                malformed[j] = any(t not in id_to_coord for t in g)
                coords = [tuple(origin_c[i])] + [id_to_coord[t] for t in g if t in id_to_coord]; tgt_c = tuple(target_c[i])
                r, bb = ED.judge(coords, conn_b[j], n, tgt_c); _, ba = ED.judge(coords, conn_a[i], n, tgt_c)
                reached[j] = r; nbad_b[j] = bb; nbad_a[j] = ba
                solved_b[j] = r and bb == 0 and not malformed[j]; solved_a[j] = r and ba == 0 and not malformed[j]
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, grid_n=n, cond="base", method="gs_" + cname, seed_tag=seed_tag, run=run_.name, set_sha=sha,
                            solved_graft=solved_b, solved_token=solved_a, reached=reached, n_bad_graft=nbad_b, n_bad_token=nbad_a, malformed=malformed)
        ch = (~S["path_same"]) & (lv > 0)
        print(f"[g{n} {seed_tag} {cname}] {time.time() - t0:.0f}s | route-changed: solved target maze {solved_b[ch].mean():.3f}"
              f" | still solves prompt maze {solved_a[ch].mean():.3f} | reached {reached[ch].mean():.3f} | same-maze col {solved_b[lv == 0].mean():.3f}", flush=True)
    print("RUN_DONE", flush=True)


if __name__ == "__main__":
    fire.Fire(dict(prep=prep, run=run))
