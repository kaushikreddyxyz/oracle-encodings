#!/usr/bin/env python3
"""Re-test of "probes cannot steer the maze world model" — probes trained at the STORAGE site.

Spies et al. (arXiv 2412.11867): layer-0 heads consolidate each connection `A <--> B ;` into the `;` token; toggling
per-connection SAE features at the semicolons steers the model. Our earlier probe-steering null (hunch #4) trained probes at
<PATH_START> and steered from <ADJLIST_END> onward — the adjacency list itself was never touched, so the model could re-read
the true maze from the prompt at every step.

Survey on OUR model (jirpy replica, learned positions): the connection identity is perfectly linearly decodable at the B
token of its chunk (offset -1 from `;`) from layer 0 on, only ~45% at the A token and ~5-15% at the `;` — and path tokens
attend to the A and B coordinate tokens (many heads with 40-80% of their mass there), not to the semicolons. So the storage
site here is the coordinate tokens of each chunk, and this script probes/steers at sites A, B and `;`.

  stage data    fresh mazes (seed 777) -> pretok + per-chunk table (pos of A, <-->, B, `;`, feature)
  stage acts    resid_post at layers LAYERS at the A / B / `;` sites (fp16) and at <PATH_START>
  stage train   per layer and site: 72-way logistic ("which connection"), diff-in-means, full class means; val accuracy
  stage steer   counterfactual edit set, all interventions at prompt positions (persist through the rollout):
     prompt       token oracle: insert `A <--> B ;` before <ADJLIST_END> / delete the chunk
     copy_chunk   remove: overwrite the 4 residuals of f's chunk with those of another chunk of the same maze (activation
                  patch — deletes f without inventing anything); add: n/a
     mean_patch   remove: A and B of f's chunk <- generic class means (mu_rest); add: write mu_f^B (full vector) at
                  semicolon positions (all, or the last one)
     dir_dim / dir_logit   coordinate along the unit probe direction set to the class target (alpha 1 = class mean):
                  remove at A and B of f's chunk; add at all B tokens (and all semicolons)
     old_style    the hunch-#4 protocol (L6, <PATH_START> probes, from <ADJLIST_END> onward)
"""
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import torch
from maze_dataset import SPECIAL_TOKENS, MazeDataset, MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators
from maze_transformer.training.config import ConfigHolder

from build_counterfactual_set import feat_cells, feat_idx, rollout_to_coords, traversed_edges
from eval_counterfactual import load_checkpoint
from eval_injection import ProbeSet, greedy_rollout
from fast_data import PretokDataset, pretokenize

GRID_N = 6
N_CELLS = GRID_N * GRID_N
N_FEATS = 2 * N_CELLS
LAYERS = [0, 1, 2, 3, 4, 6, 9]
SITES = {"A": -3, "conn": -2, "B": -1, "semi": 0}
BASELINE_DIR = Path("/workspace/runs/fast/fast-baseline_jirpyish-d256-l12_dfs-g6-n2000000")
SET_DIR = Path("/workspace/runs/counterfactual_set")
WORK = Path("/workspace/runs/wm_probes")


# ----------------------------------------------------------------------------- data
def chunk_table_row(toks, end, id_to_coord, conn_id, semi_id):
    """[(p_semicolon, feature)] for one token row; chunk = toks[p-3:p+1] = A <--> B ;"""
    out = []
    for p in range(3, end):
        if toks[p] == semi_id and toks[p - 2] == conn_id:
            a, b = id_to_coord.get(toks[p - 3]), id_to_coord.get(toks[p - 1])
            if a is not None and b is not None and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1: out.append((p, feat_idx(a, b)))
    return out


def chunk_table(ps):
    conn_id, semi_id = ps.vocab[SPECIAL_TOKENS.CONNECTOR], ps.vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE]
    return [chunk_table_row(ps.tokens[i, : int(ps.lens[i])].tolist(), int(ps.end_pos[i]), ps.id_to_coord, conn_id, semi_id) for i in range(len(ps.conn))]


def make_probeset(pretok, vocab, device):
    ps = ProbeSet([pretok[i] for i in range(len(pretok))], vocab, GRID_N, device); ps.vocab = vocab; ps.semis = chunk_table(ps)
    return ps


def slice_ps(ps, rows):
    sub = copy.copy(ps)
    for f in ("tokens", "attn", "conn", "end_pos", "lens", "origin_pos", "first_move", "first_feat", "open_id", "open_feat", "target_id"):
        setattr(sub, f, getattr(ps, f)[rows])
    sub.semis = [ps.semis[int(r)] for r in rows]
    return sub


def load_model_and_cfg(device):
    torch.set_float32_matmul_precision("high")
    model, _ = load_checkpoint(BASELINE_DIR / "model.final.zanj", None, device)
    cfg = ConfigHolder.load(json.load(open(BASELINE_DIR / "config_holder.json")))
    return model, cfg, cfg.tokenizer.get_vocab()


def stage_data(n_mazes: int = 6000, seed: int = 777):
    device = torch.device("cuda"); WORK.mkdir(parents=True, exist_ok=True); model, cfg, vocab = load_model_and_cfg(device)
    path = WORK / f"pretok-wm-s{seed}-n{n_mazes}.pt"
    if not path.exists():
        ds = MazeDataset.from_config(MazeDatasetConfig(name=f"wm-s{seed}", grid_n=GRID_N, n_mazes=n_mazes, maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed),
                                     do_generate=True, load_local=False, save_local=False, gen_parallel=False)
        pretokenize(ds, cfg.maze_tokenizer, vocab, path, bos_id=int(cfg.tokenizer.bos_token_id), workers=0)
    ps = make_probeset(PretokDataset(path), vocab, device)
    n_semi = sum(len(s) for s in ps.semis); per_feat = np.bincount([f for s in ps.semis for _, f in s], minlength=N_FEATS)
    print(f"[data] {len(ps.conn)} mazes, {n_semi} chunks ({n_semi / len(ps.conn):.1f}/maze); features present {(per_feat > 0).sum()}/72, counts min>0 {per_feat[per_feat > 0].min()} max {per_feat.max()}")
    json.dump(dict(pretok=str(path), n=len(ps.conn), n_semi=int(n_semi), per_feat=per_feat.tolist()), open(WORK / "data.json", "w"))


# ----------------------------------------------------------------------------- activations
@torch.no_grad()
def collect(model, ps, n_use, batch=250):
    names = [f"blocks.{l}.hook_resid_post" for l in LAYERS]; d = model.cfg.d_model
    n_semi = sum(len(ps.semis[i]) for i in range(n_use)); site_names = list(SITES)
    A = torch.zeros(len(site_names), n_semi, len(LAYERS), d, dtype=torch.float16); y = torch.zeros(n_semi, dtype=torch.long); mz = torch.zeros(n_semi, dtype=torch.long)
    P = torch.zeros(n_use, len(LAYERS), d, dtype=torch.float16); k = 0
    for lo in range(0, n_use, batch):
        hi = min(n_use, lo + batch)
        _, cache = model.run_with_cache(ps.tokens[lo:hi], attention_mask=ps.attn[lo:hi], return_type=None, names_filter=lambda nm: nm in names)
        for i in range(lo, hi):
            pos = torch.tensor([p for p, _ in ps.semis[i]], device=ps.device); m = len(pos)
            for li, nm in enumerate(names):
                h = cache[nm][i - lo]
                for si, s in enumerate(site_names): A[si, k:k + m, li] = h[pos + SITES[s]].half().cpu()
                P[i, li] = h[int(ps.origin_pos[i]) - 1].half().cpu()
            y[k:k + m] = torch.tensor([f for _, f in ps.semis[i]]); mz[k:k + m] = i; k += m
        del cache
    return A, y, mz, P


def stage_acts(n_use: int = 3000):
    device = torch.device("cuda"); model, cfg, vocab = load_model_and_cfg(device)
    meta = json.load(open(WORK / "data.json")); ps = make_probeset(PretokDataset(meta["pretok"]), vocab, device)
    A, y, mz, P = collect(model, ps, n_use)
    torch.save(dict(A=A, y=y, maze=mz, P=P, conn=ps.conn[:n_use].cpu(), sites=list(SITES), layers=LAYERS), WORK / "acts.pt"); print(f"[acts] saved sites x chunks x layers x d = {tuple(A.shape)}")


# ----------------------------------------------------------------------------- probes
def fit_multinomial(X, y, Xv, yv, epochs=300, lr=1e-2, wd=1e-3):
    device = X.device; lin = torch.nn.Linear(X.shape[1], N_FEATS).to(device); opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    for _ in range(epochs):
        loss = torch.nn.functional.cross_entropy(lin((X - mu) / sd), y); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((lin((Xv - mu) / sd).argmax(-1) == yv).float().mean()); W = lin.weight / sd
    return W.detach(), acc


def fit_binary(X, y, Xv, yv, epochs=300, lr=1e-2, wd=1e-3):
    device = X.device; lin = torch.nn.Linear(X.shape[1], N_FEATS).to(device); opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6; yf = y.float()
    for _ in range(epochs):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lin((X - mu) / sd), yf); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = ((lin((Xv - mu) / sd) > 0) == yv).float().mean(0); W = lin.weight / sd
    return W.detach(), acc


def stage_train(train_frac: float = 0.8):
    device = torch.device("cuda"); d = torch.load(WORK / "acts.pt", weights_only=True); A, y, mz, P, conn = d["A"], d["y"], d["maze"], d["P"], d["conn"]
    sites = d["sites"]; n_mazes = int(mz.max()) + 1; n_tr = int(n_mazes * train_frac); tr = (mz < n_tr).to(device); va = ~tr; yy = y.to(device)
    present = torch.tensor([f for f in range(N_FEATS) if (y == f).any()])
    out = {"sites": sites, "layers": LAYERS, "present": present, "semi": {}, "path": dict(W=[], acc=[])}
    for s in sites: out["semi"][s] = dict(W_logit=[], acc_logit=[], W_dim=[], mu=[], mu_rest=[], acc_dim=[])
    for li, l in enumerate(LAYERS):
        line = f"[train] L{l:2d}"
        for si, s in enumerate(sites):
            X = A[si, :, li].float().to(device)
            W, acc = fit_multinomial(X[tr], yy[tr], X[va], yy[va])
            glob = X.mean(0); mu = torch.stack([X[yy == f].mean(0) if (yy == f).any() else glob for f in range(N_FEATS)]); mu_rest = torch.stack([X[yy != f].mean(0) for f in range(N_FEATS)])
            W_dim = mu - mu_rest; Xv = X[va]
            score = Xv @ W_dim.T - 0.5 * ((mu + mu_rest) * W_dim).sum(1)[None]; score[:, [f for f in range(N_FEATS) if f not in set(present.tolist())]] = -1e9
            acc_dim = float((score.argmax(-1) == yy[va]).float().mean())
            for k, v in (("W_logit", W), ("W_dim", W_dim), ("mu", mu), ("mu_rest", mu_rest)): out["semi"][s][k].append(v.cpu())
            out["semi"][s]["acc_logit"].append(acc); out["semi"][s]["acc_dim"].append(acc_dim); line += f" | {s}: logit {acc:.3f} dim {acc_dim:.3f}"
        Xp = P[:, li].float().to(device); cb = conn.to(device); Wp, accp = fit_binary(Xp[:n_tr], cb[:n_tr], Xp[n_tr:], cb[n_tr:])
        out["path"]["W"].append(Wp.cpu()); out["path"]["acc"].append(accp.cpu()); print(line + f" | <PATH_START> binary mean {accp.mean():.3f}", flush=True)
    for s in sites:
        for k in out["semi"][s]: out["semi"][s][k] = torch.stack(out["semi"][s][k]) if isinstance(out["semi"][s][k][0], torch.Tensor) else torch.tensor(out["semi"][s][k])
    out["path"]["W"] = torch.stack(out["path"]["W"]); out["path"]["acc"] = torch.stack(out["path"]["acc"])
    torch.save(out, WORK / "wm_probes.pt")
    json.dump({"layers": LAYERS, **{f"{s}_logit": out["semi"][s]["acc_logit"].tolist() for s in sites}, **{f"{s}_dim": out["semi"][s]["acc_dim"].tolist() for s in sites}, "path_binary": out["path"]["acc"].mean(1).tolist()}, open(WORK / "probe_acc.json", "w"), indent=1)
    print("[train] wrote wm_probes.pt, probe_acc.json")


# ----------------------------------------------------------------------------- interventions
class Intervention:
    """kind: copy_chunk | mean_patch | dir_dim | dir_logit | old_style. add_at: semi_all | semi_last | B_all (where an ADDED
    connection is written). layer must be in LAYERS (probe file)."""

    def __init__(self, probes, kind, layer, alpha, add_at, device):
        self.kind, self.layer, self.alpha, self.add_at, self.device = kind, layer, alpha, add_at, device
        self.li = probes["layers"].index(layer) if layer in probes["layers"] else None; self.hook_name = f"blocks.{layer}.hook_resid_post"
        if kind in ("dir_dim", "dir_logit", "mean_patch"):
            self.S = {}
            for s in ("A", "B", "semi"):
                ss = probes["semi"][s]; W = (ss["W_logit"] if kind == "dir_logit" else ss["W_dim"])[self.li].to(device); u = W / W.norm(dim=1, keepdim=True)
                mu, mu_rest = ss["mu"][self.li].to(device), ss["mu_rest"][self.li].to(device)
                self.S[s] = dict(u=u, mu=mu, mu_rest=mu_rest, c_on=(mu * u).sum(1), c_off=(mu_rest * u).sum(1))
        if kind == "old_style":
            o = probes["old"]; self.dirs_p = o["dirs"][layer].to(device); mo, mw = o["mu_open"][layer].to(device), o["mu_wall"][layer].to(device)
            mid = (mo + mw) / 2; self.t_open, self.t_wall = mid + alpha * (mo - mid), mid + alpha * (mw - mid)

    def hooks(self, sub, edits):
        dev = self.device; B = sub.tokens.shape[0]
        if self.kind == "old_style":
            K = max(1, max(len(e) for e in edits)); dirs = torch.zeros(B, K, self.dirs_p.shape[1], device=dev); tg = torch.zeros(B, K, device=dev); m = torch.zeros(B, K, device=dev)
            for b, row in enumerate(edits):
                for k, (f, want) in enumerate(row): dirs[b, k] = self.dirs_p[f]; tg[b, k] = self.t_open[f] if want else self.t_wall[f]; m[b, k] = 1
            end = sub.end_pos
            def fn(resid, hook):
                pos = torch.arange(resid.shape[1], device=dev)[None, :]; act = (pos >= end[:, None]).to(resid.dtype)
                coords = torch.einsum("bpd,bkd->bpk", resid, dirs); delta = (tg[:, None, :] - coords) * act[..., None] * m[:, None, :]
                return resid + torch.einsum("bpk,bkd->bpd", delta, dirs)
            return [(self.hook_name, fn)]
        # ---- storage-site interventions: build (row, pos, feat, want, site) specs
        specs, copies = [], []
        for b, row in enumerate(edits):
            chunks = sub.semis[b]; by_feat = {f: p for p, f in chunks}; semis = [p for p, _ in chunks]
            for f, want in row:
                if not want:
                    if f not in by_feat: continue
                    p = by_feat[f]
                    if self.kind == "copy_chunk":
                        others = [q for q, ff in chunks if ff != f]; q = others[hash((b, f)) % len(others)]; copies.append((b, p, q))
                    else:
                        specs += [(b, p - 3, f, False, "A"), (b, p - 1, f, False, "B")]
                else:
                    if self.kind == "copy_chunk": continue
                    if self.add_at == "semi_all": specs += [(b, p, f, True, "semi") for p in semis]
                    elif self.add_at == "semi_last": specs.append((b, semis[-1], f, True, "semi"))
                    elif self.add_at == "B_all": specs += [(b, p - 1, f, True, "B") for p in semis]
        hooks = []
        if copies:
            cb = torch.tensor([c[0] for c in copies], device=dev); cp = torch.tensor([c[1] for c in copies], device=dev); cq = torch.tensor([c[2] for c in copies], device=dev)
            def fn_copy(resid, hook):
                for o in (-3, -2, -1, 0): resid[cb, cp + o] = resid[cb, cq + o]
                return resid
            hooks.append((self.hook_name, fn_copy))
        if specs:
            kind, alpha, S = self.kind, self.alpha, self.S
            by_site = defaultdict(list)
            for sp in specs: by_site[sp[4]].append(sp)
            T = {s: tuple(torch.tensor([x[k] for x in sp], device=dev) for k in range(4)) for s, sp in by_site.items()}   # tensors built ONCE per batch, not per forward
            def fn(resid, hook):
                for s, (b_, p_, f_, w_) in T.items():
                    x = resid[b_, p_]; P = S[s] if s != "semi" or kind != "mean_patch" else S["B"]     # an ADDED connection at a semicolon is written as a B-site vector
                    if kind == "mean_patch":
                        new = torch.where(w_[:, None], P["mu"][f_], P["mu_rest"][f_]); resid[b_, p_] = x + alpha * (new - x)
                    else:
                        u = P["u"][f_]; c = (x * u).sum(1); mid = 0.5 * (P["c_on"][f_] + P["c_off"][f_])
                        tgt = torch.where(w_, mid + alpha * (P["c_on"][f_] - mid), mid + alpha * (P["c_off"][f_] - mid)); resid[b_, p_] = x + (tgt - c)[:, None] * u
                return resid
            hooks.append((self.hook_name, fn))
        return hooks


@torch.no_grad()
def rollout(model, hooks, sub, max_new_tokens=64):
    dev = sub.device; n = len(sub.conn); max_len = int((sub.origin_pos + 1 + max_new_tokens).max())
    toks = torch.full((n, max_len), sub.pad_id, dtype=torch.long, device=dev)
    for i in range(n): L = int(sub.origin_pos[i]) + 1; toks[i, :L] = sub.tokens[i, :L]
    cur = (sub.origin_pos + 1).clone(); done = torch.zeros(n, dtype=torch.bool, device=dev); idx = torch.arange(n, device=dev)
    for _ in range(max_new_tokens):
        width = int(cur.max()); attn = (torch.arange(width, device=dev)[None, :] < cur[:, None]).long()
        logits = model.run_with_hooks(toks[:, :width], return_type="logits", attention_mask=attn, fwd_hooks=hooks)
        nxt = logits[idx, cur - 1].argmax(-1); nxt = torch.where(done, torch.full_like(nxt, sub.pad_id), nxt)
        toks[idx, cur] = torch.where(done, toks[idx, cur], nxt); cur = cur + (~done).long(); done = done | (nxt == sub.path_end_id)
        if bool(done.all()): break
    outs = []
    for i in range(n):
        row = toks[i, int(sub.origin_pos[i]) + 1: int(cur[i])].tolist()
        if sub.path_end_id in row: row = row[: row.index(sub.path_end_id)]
        outs.append(row)
    torch.cuda.empty_cache()                       # the varying-width forwards fragment the caching allocator (58 GB reserved otherwise)
    return outs


def edited_prompt_ps(ps, rows, edits, vocab, mode="prompt"):
    """Token-level oracles. prompt: insert `A <--> B ;` before <ADJLIST_END> / delete the removed chunk (length changes).
    prompt_dup: fixed length — a removed chunk's 4 tokens are replaced by a DUPLICATE of another chunk; an added chunk
    replaces the last chunk of the list (side effect: that connection disappears)."""
    conn_id, semi_id, end_id = vocab[SPECIAL_TOKENS.CONNECTOR], vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE], vocab[SPECIAL_TOKENS.ADJLIST_END]; dev = ps.device
    seqs = []
    for r, row_edits in zip(rows, edits):
        r = int(r); toks = ps.tokens[r, : int(ps.lens[r])].tolist(); end = int(ps.end_pos[r])
        for f, want in row_edits:
            a, b = feat_cells(f); ta, tb = ps.coord_to_id[a], ps.coord_to_id[b]
            table = chunk_table_row(toks, end, ps.id_to_coord, conn_id, semi_id); by_feat = {ff: p for p, ff in table}
            if mode == "prompt":
                if want: toks = toks[:end] + [ta, conn_id, tb, semi_id] + toks[end:]; end += 4
                elif f in by_feat: p = by_feat[f]; toks = toks[: p - 3] + toks[p + 1:]; end -= 4
            else:
                if want: p = table[-1][0]; toks[p - 3: p + 1] = [ta, conn_id, tb, semi_id]
                elif f in by_feat:
                    p = by_feat[f]; others = [q for q, ff in table if ff != f]; q = others[hash((r, f)) % len(others)]; toks[p - 3: p + 1] = toks[q - 3: q + 1]
        seqs.append(torch.tensor(toks))
    conns = [ps.conn[int(r)].cpu() for r in rows]; ends = [torch.tensor(s.tolist().index(end_id)) for s in seqs]
    sub = ProbeSet(list(zip(seqs, conns, ends)), vocab, GRID_N, dev); sub.vocab = vocab; sub.semis = chunk_table(sub)
    return sub


def edit_spec(e):
    spec = []
    if e["block"] is not None: spec.append((e["block"], False))
    if e["open"] is not None: spec.append((e["open"], True))
    return spec


def edited_conn(conn, e):
    c = conn.copy()
    if e["block"] is not None: c[e["block"]] = False
    if e["open"] is not None: c[e["open"]] = True
    return c


def judge(ps, i, gen, e, conn_np, true_roll):
    coords = rollout_to_coords(ps, i, gen); edges = set(traversed_edges(coords)); target = ps.id_to_coord[int(ps.target_id[i])]; ce = edited_conn(conn_np[i], e)
    valid = all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and ce[feat_idx(a, b)] for a, b in zip(coords, coords[1:]))
    reached = len(coords) > 1 and coords[-1] == target
    m = dict(reached_target=float(reached), rollout_changed=float(gen != true_roll), valid_edited=float(valid), solves_edited=float(valid and reached and (len(coords) - 1) == e["new_dist"]))
    if e["family"] == "detour": m["blocked_edge_crossed"] = float(e["block"] in edges); m["detour_wall_used"] = float(e["open"] in edges)
    elif e["family"] == "offpath_block": m["blocked_edge_crossed"] = float(e["block"] in edges)
    elif e["family"] == "shortcut_open": m["shortcut_used"] = float(e["open"] in edges)
    elif e["family"] == "zerogain_open": m["opened_wall_used"] = float(e["open"] in edges)
    return m


@torch.no_grad()
def eval_condition(model, ps, edits, true_rolls, conn_np, cond, probes, vocab, chunk=256):
    kind, layer, alpha, add_at = cond; agg = defaultdict(lambda: defaultdict(list))
    for lo in range(0, len(edits), chunk):
        sub_e = edits[lo: lo + chunk]; rows = torch.tensor([e["maze"] for e in sub_e]); specs = [edit_spec(e) for e in sub_e]
        if kind in ("prompt", "prompt_dup"): sub = edited_prompt_ps(ps, rows, specs, vocab, kind); gens = rollout(model, [], sub)
        else:
            sub = slice_ps(ps, rows); hooks = Intervention(probes, kind, layer, alpha, add_at, ps.device).hooks(sub, specs) if kind != "none" else []; gens = rollout(model, hooks, sub)
        for e, gen in zip(sub_e, gens):
            for k, v in judge(ps, e["maze"], gen, e, conn_np, true_rolls[e["maze"]]).items(): agg[e["family"]][k].append(v)
    return {fam: {k: float(np.mean(v)) for k, v in d.items()} for fam, d in agg.items()}


def summarize(key, r):
    det, sho, off, zg = (r.get(f, {}) for f in ("detour", "shortcut_open", "offpath_block", "zerogain_open")); g = lambda d, k: d.get(k, float("nan"))
    return (f"{key:30s} detour: crossed {g(det, 'blocked_edge_crossed'):.3f} used {g(det, 'detour_wall_used'):.3f} solves {g(det, 'solves_edited'):.3f} reach {g(det, 'reached_target'):.3f}"
            f" | shortcut: used {g(sho, 'shortcut_used'):.3f} solves {g(sho, 'solves_edited'):.3f} | offpath: crossed {g(off, 'blocked_edge_crossed'):.3f} reach {g(off, 'reached_target'):.3f} | zerogain used {g(zg, 'opened_wall_used'):.3f}")


def stage_steer(subsample: int = 400, seed: int = 0, layers="0,1,2,3,6", full: bool = False, out: str = "steer_sweep.json", core: bool = False, base: bool = True, only: str = ""):
    device = torch.device("cuda"); model, cfg, vocab = load_model_and_cfg(device)
    data = torch.load(SET_DIR / "set.pt", weights_only=False); ps = make_probeset(PretokDataset(data["pretok_file"]), vocab, device)
    probes = torch.load(WORK / "wm_probes.pt", weights_only=False); conn_np = ps.conn.cpu().numpy()
    old_path = Path("/workspace/runs/steering/steer_probes.pt")
    if old_path.exists(): probes["old"] = torch.load(old_path, weights_only=True)
    true_rolls = []
    for lo in range(0, len(ps.conn), 256): true_rolls += greedy_rollout(model, None, slice_ps(ps, torch.arange(lo, min(len(ps.conn), lo + 256))), None)
    rng = random.Random(seed); fams = defaultdict(list)
    for e in data["edits"]: fams[e["family"]].append(e)
    edits = data["edits"] if full else sum([rng.sample(fams[f], min(subsample, len(fams[f]))) for f in ("detour", "shortcut_open", "offpath_block", "zerogain_open")], [])
    layer_list = [] if str(layers) == "none" else ([int(x) for x in layers] if isinstance(layers, (tuple, list)) else [int(x) for x in str(layers).split(",")])
    print(f"[steer] {len(edits)} edits ({'full' if full else 'subsample'}); layers {layer_list}", flush=True)
    conds = [("none", 0, 0.0, "-"), ("prompt", 0, 0.0, "-"), ("prompt_dup", 0, 0.0, "-")] if base else []
    for l in layer_list:
        conds.append(("copy_chunk", l, 1.0, "-"))
        if core:
            conds += [("mean_patch", l, 1.0, "semi_last"), ("mean_patch", l, 1.0, "semi_all"), ("dir_dim", l, 1.0, "semi_last"), ("dir_logit", l, 2.0, "semi_all")]
            continue
        for add_at in ("semi_all", "semi_last", "B_all"):
            conds.append(("mean_patch", l, 1.0, add_at))
            for kind in ("dir_dim", "dir_logit"):
                for alpha in (1.0, 2.0): conds.append((kind, l, alpha, add_at))
    if "old" in probes and base: conds += [("old_style", 6, 2.0, "-"), ("old_style", 6, 4.0, "-")]
    if only:                                                   # explicit list: "kind|layer|alpha|add_at;..." (missing fields default to 0/0/-)
        conds = []
        for item in str(only).split(";"):
            f = (item.split("|") + ["0", "0", "-"])[:4]; conds.append((f[0], int(f[1]), float(f[2]), f[3]))
    res = {"meta": dict(n_edits=len(edits), full=full, layers=layer_list), "results": {}}
    for cond in conds:
        key = f"{cond[0]}|L{cond[1]}|a{cond[2]:g}|{cond[3]}"; r = eval_condition(model, ps, edits, true_rolls, conn_np, cond, probes, vocab); res["results"][key] = r
        print("[steer] " + summarize(key, r), flush=True); json.dump(res, open(WORK / out, "w"), indent=1)


if __name__ == "__main__":
    fire.Fire({"data": stage_data, "acts": stage_acts, "train": stage_train, "steer": stage_steer})
