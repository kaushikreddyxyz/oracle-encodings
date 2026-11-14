#!/usr/bin/env python3
"""Minimal-swap paired probes: does the residual stream carry a linear direction that moves when ONE wall changes and
nothing else the model sees or does changes?

Pairs (built once per grid, `build`): for each random-DFS maze A (with its origin, target and shortest path) pick an open
connection e NOT on the solution path; removing it splits the tree in two; add a closed lattice connection e' across that
cut (e' != e). B = A - e + e' is again a spanning tree with the SAME origin, target and solution path. B's prompt is A's
prompt with the one adjacency-list entry for e rewritten in place to e' (same length, same order, same positions), so at
every path token the two sequences are identical except for that entry. Every pair is verified (see `verify`).

Probes (`run`): at sampled path tokens, d = act(A) - act(B) at 13 sites. For every wall target t (allo: the free absolute
walls; ego: walls at offsets |dr|,|dc| <= 2 of the current cell), label(t) = y_t(A) - y_t(B) in {-1, 0, +1} -- nonzero only
where the swap flipped t. One ridge direction per target is fit on the TRAINING pairs' differences (labels 0 included, so a
direction that responds to other walls is penalised); the score on held-out pairs is the fraction of flipped instances where
the direction moves the right way (sign(w . d) = label; ties 0.5). Anything that does not depend on the flipped walls is
identical across a pair and cancels, so a model with no wall information scores exactly 0.5.

  python probe_pairs.py build  --grid_n 6 --out /workspace/pairs/g6.pt [--n_mazes 20000]
  python probe_pairs.py verify --pairs /workspace/pairs/g6.pt
  python probe_pairs.py run    --run_dir <run> --pairs /workspace/pairs/g6.pt --out <json>
"""
import pathlib as _pl, sys as _sys
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(next(p for p in _HERE.parents if (p / "maze" / "core").is_dir())))
_sys.path.insert(0, str(_HERE.parent / "curves")); _sys.path.insert(0, str(_HERE))
import maze._pathsetup  # noqa: F401,E402
import json, re, time  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402
import fire, numpy as np, torch  # noqa: E402
from maze_dataset import SPECIAL_TOKENS, MazeDataset, MazeDatasetConfig  # noqa: E402
from maze_dataset.generation.generators import LatticeMazeGenerators  # noqa: E402
from maze_transformer.training.config import ConfigHolder  # noqa: E402
from maze.core.data import PretokDataset, pretokenize  # noqa: E402
from maze.core.graft import as_graft  # noqa: E402
from eval_ckpts import load_ckpt  # noqa: E402
from probe_walls import free_idx  # noqa: E402
from probe_nonlinear import ego_targets  # noqa: E402

SITES = ["L0in"] + [f"L{k}" for k in range(1, 13)]


# ---------------------------------------------------------------- maze helpers (conn = flattened connection_list)
def cidx(a, b, n):
    """conn index of the lattice connection between adjacent cells a, b (down: r*n+c, right: n*n + r*n + c)."""
    (r1, c1), (r2, c2) = sorted([a, b])
    if r2 == r1 + 1 and c1 == c2: return r1 * n + c1
    if r1 == r2 and c2 == c1 + 1: return n * n + r1 * n + c1
    raise ValueError((a, b))


def cells_of(f, n):
    k, rc = divmod(f, n * n); r, c = divmod(rc, n)
    return ((r, c), (r + 1, c)) if k == 0 else ((r, c), (r, c + 1))


def nbrs(cell, n):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= r + dr < n and 0 <= c + dc < n]


def component(conn, start, n, skip=None):
    seen = {start}; q = deque([start])
    while q:
        u = q.popleft()
        for v in nbrs(u, n):
            f = cidx(u, v, n)
            if f != skip and conn[f] and v not in seen: seen.add(v); q.append(v)
    return seen


def bfs_path(conn, a, b, n):
    prev = {a: None}; q = deque([a])
    while q:
        u = q.popleft()
        if u == b: break
        for v in nbrs(u, n):
            if conn[cidx(u, v, n)] and v not in prev: prev[v] = u; q.append(v)
    out = [b]
    while prev[out[-1]] is not None: out.append(prev[out[-1]])
    return out[::-1]


def vocab_maps(cfg):
    vocab = cfg.tokenizer.get_vocab()
    id2c = {i: (int(m.group(1)), int(m.group(2))) for t, i in vocab.items() if (m := re.match(r"^\((\d+),(\d+)\)$", t))}
    return vocab, id2c, {c: i for i, c in id2c.items()}


def adj_edges(s, end, id2c, conn_id, n):
    """(position of first endpoint token, conn index) for every adjacency-list entry in s[1:end]."""
    out = []
    for p in range(1, end - 2):
        if s[p + 1] == conn_id and s[p] in id2c and s[p + 2] in id2c:
            out.append((p, cidx(id2c[s[p]], id2c[s[p + 2]], n)))
    return out


def path_of(s, vocab, id2c):
    ps, pe = s.index(vocab[SPECIAL_TOKENS.PATH_START]), s.index(vocab[SPECIAL_TOKENS.PATH_END])
    return ps, [id2c[t] for t in s[ps + 1:pe]]


# ---------------------------------------------------------------- build
def build(grid_n: int, out: str, cfg_from: str = "", n_mazes: int = 20000, seed: int = 777):
    """cfg_from: any run dir of this grid (only its tokenizer is used). Serial generation + tokenization with fixed seeds, so
    every pod builds byte-identical pairs (the content hash is printed and compared across pods)."""
    import random, hashlib
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    n = grid_n; rng = np.random.default_rng(seed)
    cfg = ConfigHolder.load(json.load(open(Path(cfg_from) / "config_holder.json")))
    vocab, id2c, c2id = vocab_maps(cfg); conn_id = vocab[SPECIAL_TOKENS.CONNECTOR]
    ds = MazeDataset.from_config(MazeDatasetConfig(name="pairset", grid_n=n, n_mazes=n_mazes, maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed),
                                 do_generate=True, load_local=False, save_local=False, gen_parallel=False, verbose=False)
    tmp = Path(out).with_suffix(".pretok.pt"); tmp.parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed)
    pretokenize(ds, cfg.maze_tokenizer, vocab, tmp, bos_id=int(cfg.tokenizer.bos_token_id), workers=0)
    pt = PretokDataset(tmp)
    A, B, cA, cB, ends, eflip, eprime = [], [], [], [], [], [], []
    for i in range(len(pt)):
        s, conn, end = pt[i]; s = s.tolist(); conn = conn.numpy().astype(bool); end = int(end)
        _, path = path_of(s, vocab, id2c)
        pedges = {cidx(path[j], path[j + 1], n) for j in range(len(path) - 1)}
        cand = [f for p, f in adj_edges(s, end, id2c, conn_id, n) if f not in pedges]
        if not cand: continue
        e = int(rng.choice(cand))
        comp = component(conn, path[0], n, skip=e)
        cross = sorted({cidx(u, v, n) for u in comp for v in nbrs(u, n) if v not in comp} - {e})
        cross = [f for f in cross if not conn[f]]
        if not cross: continue
        e2 = int(rng.choice(cross))
        connB = conn.copy(); connB[e] = False; connB[e2] = True
        sB = list(s); pos = [p for p, f in adj_edges(s, end, id2c, conn_id, n) if f == e]; assert len(pos) == 1
        u, v = cells_of(e2, n)
        if rng.random() < 0.5: u, v = v, u
        sB[pos[0]], sB[pos[0] + 2] = c2id[u], c2id[v]
        A.append(torch.tensor(s, dtype=torch.int16)); B.append(torch.tensor(sB, dtype=torch.int16))
        cA.append(torch.tensor(conn)); cB.append(torch.tensor(connB)); ends.append(end); eflip.append(e); eprime.append(e2)
    torch.save(dict(A=A, B=B, connA=torch.stack(cA), connB=torch.stack(cB), end=torch.tensor(ends), e=torch.tensor(eflip), e2=torch.tensor(eprime),
                    grid_n=n, seed=seed, cfg_from=str(cfg_from)), out)
    tmp.unlink()
    h = hashlib.sha256(b"".join(t.numpy().tobytes() for t in A + B)).hexdigest()[:16]
    print(f"built {len(A)} pairs from {len(pt)} mazes -> {out}  content-hash {h}", flush=True)


# ---------------------------------------------------------------- verify
def verify(pairs: str):
    P = torch.load(pairs, weights_only=False); n = P["grid_n"]
    cfg = ConfigHolder.load(json.load(open(Path(P["cfg_from"]) / "config_holder.json")))
    vocab, id2c, _ = vocab_maps(cfg); conn_id = vocab[SPECIAL_TOKENS.CONNECTOR]
    bad = {k: 0 for k in ("len", "ndiff", "diff_outside_adj", "adj_vs_conn_A", "adj_vs_conn_B", "tree_B", "path_B", "path_tokens", "swap_set")}
    for i in range(len(P["A"])):
        a, b = P["A"][i].tolist(), P["B"][i].tolist(); ca, cb = P["connA"][i].numpy(), P["connB"][i].numpy(); end = int(P["end"][i])
        if len(a) != len(b): bad["len"] += 1; continue
        d = [p for p in range(len(a)) if a[p] != b[p]]
        if not 1 <= len(d) <= 2: bad["ndiff"] += 1
        if any(not (1 <= p < end) for p in d): bad["diff_outside_adj"] += 1
        for s, c, k in ((a, ca, "adj_vs_conn_A"), (b, cb, "adj_vs_conn_B")):
            listed = sorted(f for _, f in adj_edges(s, end, id2c, conn_id, n))
            if listed != sorted(np.flatnonzero(c).tolist()): bad[k] += 1
        if cb.sum() != n * n - 1 or len(component(cb, (0, 0), n)) != n * n: bad["tree_B"] += 1
        psa, pa = path_of(a, vocab, id2c); psb, pb = path_of(b, vocab, id2c)
        if a[psa:] != b[psb:] or psa != psb: bad["path_tokens"] += 1
        if bfs_path(cb, pa[0], pa[-1], n) != pa: bad["path_B"] += 1
        if set(np.flatnonzero(ca != cb).tolist()) != {int(P["e"][i]), int(P["e2"][i])}: bad["swap_set"] += 1
    print(json.dumps({"pairs": len(P["A"]), "failures": bad}), flush=True)
    assert sum(bad.values()) == 0, bad
    return bad


# ---------------------------------------------------------------- run
def acts_all_sites(model, g, layer, toks, attn, conn, end, pos_b, pos_p):
    """residual at every site for the (batch-row, position) pairs, one forward pass with hooks."""
    out, hooks = {}, []
    delta = g.delta(g.code.features(toks, attn, conn, end)) if g is not None else None
    x0 = model.embed(toks) + model._positions(toks, attn)
    if delta is not None and layer == 0: x0 = x0 + delta.to(x0.dtype)
    out["L0in"] = x0[pos_b, pos_p].float()
    for k, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(lambda m, i, o, k=k: out.__setitem__(f"L{k + 1}", o[pos_b, pos_p].float())))
    try:
        model(toks, attention_mask=attn, inject_delta=delta, inject_layer=layer, return_type=None, padding_mode="right")
    finally:
        for h in hooks: h.remove()
    return out


def run(run_dir: str, pairs: str, out: str, per_pair: int = 6, batch: int = 500, seed: int = 0, check_hooks: bool = True, R: int = -1):
    """R = egocentric window half-width; -1 = n-1, i.e. every relative offset (the ego graft's full window)."""
    device = torch.device("cuda"); torch.set_float32_matmul_precision("high")
    run = Path(run_dir); P = torch.load(pairs, weights_only=False); n = P["grid_n"]
    cfg = ConfigHolder.load(json.load(open(run / "config_holder.json"))); vocab, id2c, _ = vocab_maps(cfg); pad = vocab[SPECIAL_TOKENS.PADDING]
    inj = run / "injector.final.pt"
    model, injector = load_ckpt(run / "model.final.pt", inj if inj.exists() else None, device); model.eval()
    g = as_graft(injector); layer = int(g.layer) if g is not None else 0
    N = len(P["A"]); rng = np.random.default_rng(seed); fi = free_idx(n); R = n - 1 if R < 0 else R
    # sampled path-token positions per pair (identical in A and B)
    samp = []
    for i in range(N):
        s = P["A"][i].tolist(); ps, path = path_of(s, vocab, id2c)
        cand = list(range(ps + 1, ps + 1 + len(path)))
        samp.append(sorted(rng.choice(cand, size=min(per_pair, len(cand)), replace=False).tolist()))
    # labels
    YA, YE, pid = [], [], []
    for i in range(N):
        ca, cb = P["connA"][i].numpy(), P["connB"][i].numpy(); s = P["A"][i].tolist()
        for p in samp[i]:
            cell = id2c[s[p]]
            YA.append((ca[fi].astype(np.float32) - cb[fi].astype(np.float32)))
            YE.append((ego_targets(ca, cell, n, R) - ego_targets(cb, cell, n, R)) / 2.0)
            pid.append(i)
    YA, YE, pid = torch.tensor(np.stack(YA)), torch.tensor(np.stack(YE)), np.array(pid)
    D = {s: [] for s in SITES}; t0 = time.time()
    with torch.no_grad():
        for lo in range(0, N, batch):
            idx = range(lo, min(N, lo + batch)); L = max(len(P["A"][i]) for i in idx)
            res = []
            for key, ckey in (("A", "connA"), ("B", "connB")):
                toks = torch.full((len(idx), L), pad, dtype=torch.long); attn = torch.zeros_like(toks)
                for r, i in enumerate(idx): s = P[key][i].long(); toks[r, :len(s)] = s; attn[r, :len(s)] = 1
                toks, attn = toks.to(device), attn.to(device)
                conn = P[ckey][lo:lo + len(idx)].to(device); end = P["end"][lo:lo + len(idx)].to(device)
                pb = torch.tensor([r for r, i in enumerate(idx) for _ in samp[i]], device=device)
                pp = torch.tensor([p for i in idx for p in samp[i]], device=device)
                a = acts_all_sites(model, g, layer, toks, attn, conn, end, pb, pp)
                if check_hooks and lo == 0 and key == "A":   # verify hooks against the model's own stop_at_layer path
                    delta = g.delta(g.code.features(toks, attn, conn, end)) if g is not None else None
                    for k in (1, 6, 12):
                        ref = model(toks, attention_mask=attn, inject_delta=delta, inject_layer=layer, padding_mode="right", stop_at_layer=k)[pb, pp].float()
                        err = (ref - a[f"L{k}"]).abs().max().item(); assert err < 1e-3 * max(1.0, ref.abs().max().item()), (k, err)
                    print(f"hook check ok (L1/L6/L12 match stop_at_layer)", flush=True)
                res.append(a)
            for s in SITES: D[s].append((res[0][s] - res[1][s]).cpu())
    D = {s: torch.cat(v) for s, v in D.items()}
    assert D["L0in"].shape[0] == len(pid)
    # split by pair
    tr, va, te = pid < int(0.6 * N), (pid >= int(0.6 * N)) & (pid < int(0.75 * N)), pid >= int(0.75 * N)
    adjacent = np.array([len(set(cells_of(int(P["e"][i]), n)) & set(cells_of(int(P["e2"][i]), n))) > 0 for i in range(N)])[pid]
    out_res = {"run": run.name, "grid_n": n, "code": g.code.name if g else "none", "pairs": N, "positions": int(len(pid)), "R": R,
               "split_positions": dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())), "sites": {}}

    from scipy.stats import rankdata

    def auroc(pos, neg):
        if len(pos) == 0 or len(neg) == 0: return float("nan")
        r = rankdata(np.concatenate([pos, neg]))   # average ranks over ties (d = 0 exactly gives ties -> 0.5)
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    def score(pred, Y, mask):
        """Primary: per target, AUROC of the projection for 'this wall just opened' vs every other test instance (other walls
        swapped, or this wall closed), averaged with the mirror 'just closed' vs rest -- a direction must be SPECIFIC to the
        wall to score high; anything independent of the wall scores 0.5. Secondary: sign accuracy on flipped instances."""
        aucs, accs, cnts = [], [], []
        for j in range(Y.shape[1]):
            yj = Y[:, j].numpy(); m = mask & (yj != 0)
            if m.sum() < 20: continue
            pj = pred[:, j]; valid = mask
            a_open = auroc(pj[valid & (yj > 0)], pj[valid & (yj <= 0)]); a_close = auroc(-pj[valid & (yj < 0)], -pj[valid & (yj >= 0)])
            aucs.append(float(np.nanmean([a_open, a_close])))
            accs.append(float(np.mean(np.where(pj[m] == 0, 0.5, (np.sign(pj[m]) == yj[m]).astype(float))))); cnts.append(int(m.sum()))
        return (float(np.mean(aucs)) if aucs else float("nan")), aucs, cnts, (float(np.mean(accs)) if accs else float("nan"))

    for s in SITES:
        X = D[s].to(device).double(); res_s = {}
        Xtr = X[torch.tensor(tr, device=device)]; G = Xtr.T @ Xtr; tr_scale = float(torch.trace(G) / G.shape[0])
        for fam, Y in (("allo", YA), ("ego", YE)):
            Ytr = Y[tr].to(device).double(); XtY = Xtr.T @ Ytr; best = None
            for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
                W = torch.linalg.solve(G + (lam * tr_scale + 1e-9) * torch.eye(G.shape[0], device=device, dtype=G.dtype), XtY)   # d == 0 (e.g. base model input) -> W = 0 -> ties -> 0.5
                pv = (X[torch.tensor(va, device=device)] @ W); yv = Y[va].to(device).double(); m = yv != 0
                v = float(((torch.sign(pv) == yv) & m).sum() / m.sum().clamp(min=1))
                if best is None or v > best[0]: best = (v, lam, W)
            pred = (X @ best[2]).cpu().numpy()
            auc, per, cnt, acc = score(pred, Y, te)
            auc_na = score(pred, Y, te & ~adjacent)[0]
            # control: the same fit on labels permuted across training pairs (breaks the wall <-> activation link) -> must be ~0.5
            trp = np.flatnonzero(tr); Ysh = Y.clone(); Ysh[trp] = Y[np.random.default_rng(1).permutation(trp)]
            Wc = torch.linalg.solve(G + (best[1] * tr_scale + 1e-9) * torch.eye(G.shape[0], device=device, dtype=G.dtype), Xtr.T @ Ysh[tr].to(device).double())
            auc_ctrl = score((X @ Wc).cpu().numpy(), Y, te)[0]
            res_s[fam] = dict(auroc=auc, auroc_partner_nonadjacent=auc_na, control_shuffled_labels=auc_ctrl, sign_acc=acc, lam=best[1], n_targets=len(per),
                              per_target_auroc=per, per_target_n=cnt)
        # ENCODING test (the linear-representation claim itself): is the activation change caused by the swap a sum of FIXED
        # per-wall vectors, d ~ sum_t label_t * v_t ?  Fit V (targets x d_model) on training pairs, report held-out R^2 of d.
        for fam, Y in (("allo", YA), ("ego", YE)):
            Ytr = Y[tr].to(device).double(); Dtr = X[torch.tensor(tr, device=device)]; best = None
            for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
                V = torch.linalg.solve(Ytr.T @ Ytr + lam * torch.eye(Y.shape[1], device=device, dtype=Ytr.dtype), Ytr.T @ Dtr)
                Yv, Dv = Y[va].to(device).double(), X[torch.tensor(va, device=device)]
                r2 = 1 - float(((Dv - Yv @ V) ** 2).sum() / ((Dv ** 2).sum() + 1e-12))
                if best is None or r2 > best[0]: best = (r2, lam, V)
            Yt, Dt = Y[te].to(device).double(), X[torch.tensor(te, device=device)]
            den = float((Dt ** 2).sum())
            res_s[fam]["encoding_r2"] = (1 - float(((Dt - Yt @ best[2]) ** 2).sum()) / den) if den > 0 else float("nan")
            res_s[fam]["encoding_lam"] = best[1]
        res_s["d_norm_mean"] = float(X.norm(dim=1).mean())
        out_res["sites"][s] = res_s
        print(f"  {s:5s} allo {res_s['allo']['auroc']:.3f} (acc {res_s['allo']['sign_acc']:.3f}, nonadj {res_s['allo']['auroc_partner_nonadjacent']:.3f}, ctrl {res_s['allo']['control_shuffled_labels']:.3f}, "
              f"{res_s['allo']['n_targets']} walls) | ego {res_s['ego']['auroc']:.3f} (acc {res_s['ego']['sign_acc']:.3f}, nonadj {res_s['ego']['auroc_partner_nonadjacent']:.3f}, "
              f"ctrl {res_s['ego']['control_shuffled_labels']:.3f}, {res_s['ego']['n_targets']} offsets) | encoding R2 allo {res_s['allo']['encoding_r2']:.3f} ego {res_s['ego']['encoding_r2']:.3f}  [{time.time() - t0:.0f}s]", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True); json.dump(out_res, open(out, "w")); print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    fire.Fire({"build": build, "verify": verify, "run": run})
