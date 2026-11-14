#!/usr/bin/env python3
"""Do exotic grafts get translated into wall features downstream?  Linear (ridge) probes for the
absolute free connections (the 'platonic' wall features, +-1) and for the four walls adjacent to
the current cell (egocentric N/E/S/W), at every layer, at the graft-site tokens (path coordinates)
and at <PATH_START>. Works for base models (no graft) and for any graft code (edge, onion, vertex,
degree, ego); blanks the prompt when the injector says dummy_data.

  python probe_walls.py --run_dir <run> --grid_n 6 [--n_mazes 4000] [--out <run>/eval/probe_walls.json]

Sites: "L0in" = residual entering block 0 INCLUDING the graft (what the model receives), "L{k}" =
residual after block k-1 (TL resid_post of block k-1), k = 1..n_layers. Train on the first 75% of
mazes, test on the rest (mazes never shared). Accuracy = sign agreement of the ridge prediction.
"""
import pathlib as _pl, sys as _sys
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(next(p for p in _HERE.parents if (p / "maze" / "core").is_dir())))
_sys.path.insert(0, str(_HERE.parent / "curves"))
import maze._pathsetup  # noqa
import json, re, time
from pathlib import Path
import fire, numpy as np, torch
from maze_dataset import SPECIAL_TOKENS, MazeDataset, MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators
from maze_transformer.training.config import ConfigHolder
from maze.core.data import PretokDataset, pretokenize
from maze.core.graft import as_graft
from eval_ckpts import load_ckpt


def free_idx(n):
    keep = []
    for f in range(2 * n * n):
        k, rc = divmod(f, n * n); r, c = divmod(rc, n)
        if (k == 0 and r == n - 1) or (k == 1 and c == n - 1):
            continue
        keep.append(f)
    return np.array(keep)


def local4(conn, cell, n):
    """+-1 for wall/open on the N, E, S, W side of `cell` (grid boundary = wall = -1)."""
    r, c = cell
    N = conn[(r - 1) * n + c] if r > 0 else False
    S = conn[r * n + c] if r < n - 1 else False
    W = conn[n * n + r * n + c - 1] if c > 0 else False
    E = conn[n * n + r * n + c] if c < n - 1 else False
    return np.array([N, E, S, W], dtype=np.float32) * 2 - 1


def ridge_acc(Xtr, Ytr, Xte, Yte, lam=1.0):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Xtr = torch.cat([Xtr, torch.ones(len(Xtr), 1, device=Xtr.device)], 1); Xte = torch.cat([Xte, torch.ones(len(Xte), 1, device=Xte.device)], 1)
    A = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], device=Xtr.device)
    Wt = torch.linalg.solve(A, Xtr.T @ Ytr)
    pred = Xte @ Wt
    return ((pred.sign() == Yte.sign()).float().mean(0)).cpu().numpy(), Wt[:-1] / sd.T   # directions in raw activation units


def wall_pairs(fi, n):
    """Index pairs among the free walls: adjacent = share a cell (a wall and its neighbours along the row/column or the
    perpendicular walls at either end), distant = no shared cell and not within one cell."""
    cells = []
    for f in fi:
        k, rc = divmod(int(f), n * n); r, c = divmod(rc, n)
        cells.append({(r, c), (r + 1, c) if k == 0 else (r, c + 1)})
    adj, far = [], []
    for i in range(len(fi)):
        for j in range(i + 1, len(fi)):
            if cells[i] & cells[j]: adj.append((i, j))
            elif not any(abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1 for a in cells[i] for b in cells[j]): far.append((i, j))
    return np.array(adj), np.array(far)


def pair_cos(W, pairs):
    U = W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)
    C = (U.T @ U).cpu().numpy()
    return float(np.abs(C[pairs[:, 0], pairs[:, 1]]).mean())


def main(run_dir: str, grid_n: int, n_mazes: int = 4000, seed: int = 999, per_maze: int = 8, out: str = "",
         cache_dir: str = "/workspace/evalsets", batch: int = 250):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    run = Path(run_dir); n = grid_n
    cfg = ConfigHolder.load(json.load(open(run / "config_holder.json")))
    vocab = cfg.tokenizer.get_vocab()
    coord_re = re.compile(r"^\((\d+),(\d+)\)$")
    id_to_coord = {i: (int(m.group(1)), int(m.group(2))) for t, i in vocab.items() if (m := coord_re.match(t))}
    pad_id, ps_id, semi_id = vocab[SPECIAL_TOKENS.PADDING], vocab[SPECIAL_TOKENS.PATH_START], vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE]
    d = Path(cache_dir) / f"probe-g{n}-s{seed}-n{n_mazes}"; d.mkdir(parents=True, exist_ok=True)
    pre = d / "pretok.pt"
    if not pre.exists():
        ds = MazeDataset.from_config(MazeDatasetConfig(name="probeset", grid_n=n, n_mazes=n_mazes, maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed),
                                     do_generate=True, load_local=False, save_local=False, gen_parallel=False, verbose=False)
        pretokenize(ds, cfg.maze_tokenizer, vocab, pre, bos_id=int(cfg.tokenizer.bos_token_id), workers=0)
    pt = PretokDataset(pre)
    inj_path = run / "injector.final.pt"
    model, injector = load_ckpt(run / "model.final.pt", inj_path if inj_path.exists() else None, device)
    g = as_graft(injector); layer = int(g.layer) if g is not None else 0
    dummy = bool(torch.load(inj_path, weights_only=True).get("dummy_data", False)) if inj_path.exists() else False
    if dummy:
        pt.dummify_(semi_id)
    rows = [pt[i] for i in range(len(pt))]
    L = max(len(r[0]) for r in rows)
    toks = torch.full((len(rows), L), pad_id, dtype=torch.long); attn = torch.zeros_like(toks)
    for i, (s, _, _) in enumerate(rows):
        toks[i, :len(s)] = s.long(); attn[i, :len(s)] = 1
    conn = torch.stack([c for _, c, _ in rows]); end = torch.tensor([int(e) for _, _, e in rows])
    fi = free_idx(n); rng = np.random.default_rng(seed)
    # sampled positions: path coordinate tokens (graft site) and the <PATH_START> token
    samp_pos, samp_maze, ps_pos = [], [], []
    for i, (s, c, e) in enumerate(rows):
        s = s.tolist(); p0 = s.index(ps_id); ps_pos.append(p0)
        cand = [p for p in range(p0 + 1, len(s)) if s[p] in id_to_coord]
        pick = rng.choice(cand, size=min(per_maze, len(cand)), replace=False)
        samp_pos += pick.tolist(); samp_maze += [i] * len(pick)
    samp_pos, samp_maze, ps_pos = np.array(samp_pos), np.array(samp_maze), np.array(ps_pos)
    conn_np = conn.numpy()
    Y_abs = torch.tensor(conn_np[samp_maze][:, fi].astype(np.float32) * 2 - 1)
    Y_loc = torch.tensor(np.stack([local4(conn_np[m], id_to_coord[int(toks[m, p])], n) for m, p in zip(samp_maze, samp_pos)]))
    Y_abs_ps = torch.tensor(conn_np[:, fi].astype(np.float32) * 2 - 1)
    n_layers = len(model.blocks); sites = ["L0in"] + [f"L{k}" for k in range(1, n_layers + 1)]
    acts = {s: [] for s in sites}; acts_ps = {s: [] for s in sites}
    t0 = time.time()
    with torch.no_grad():
        for lo in range(0, len(rows), batch):
            tb, ab, cb, eb = (x[lo:lo + batch].to(device) for x in (toks, attn, conn, end))
            delta = g.delta(g.code.features(tb, ab, cb, eb)) if g is not None else None
            sel = (samp_maze >= lo) & (samp_maze < lo + batch)
            mi, pi = torch.tensor(samp_maze[sel] - lo, device=device), torch.tensor(samp_pos[sel], device=device)
            pp = torch.tensor(ps_pos[lo:lo + batch], device=device); ar = torch.arange(len(pp), device=device)
            resid = model.embed(tb) + model._positions(tb, ab)
            if delta is not None and layer == 0:
                x0 = resid + delta.to(resid.dtype)
            else:
                x0 = resid
            acts["L0in"].append(x0[mi, pi].float().cpu()); acts_ps["L0in"].append(x0[ar, pp].float().cpu())
            for k in range(1, n_layers + 1):
                h = model(tb, attention_mask=ab, inject_delta=delta, inject_layer=layer, padding_mode="right", stop_at_layer=k)
                acts[f"L{k}"].append(h[mi, pi].float().cpu()); acts_ps[f"L{k}"].append(h[ar, pp].float().cpu())
    ntr = int(0.75 * len(rows)); tr = samp_maze < ntr; te = ~tr
    res = {"run": run.name, "grid_n": n, "n_mazes": n_mazes, "dummy_prompt": dummy, "code": (g.code.name if g else "none"),
           "n_train_rows": int(tr.sum()), "n_test_rows": int(te.sum()), "sites": {}}
    adj, far = wall_pairs(fi, n)
    for s in sites:
        X = torch.cat(acts[s]).to(device); Xp = torch.cat(acts_ps[s]).to(device)
        a_abs, W_abs = ridge_acc(X[tr], Y_abs[tr].to(device), X[te], Y_abs[te].to(device))
        a_loc, _ = ridge_acc(X[tr], Y_loc[tr].to(device), X[te], Y_loc[te].to(device))
        a_ps, W_ps = ridge_acc(Xp[:ntr], Y_abs_ps[:ntr].to(device), Xp[ntr:], Y_abs_ps[ntr:].to(device))
        # geometry of the wall readout directions: an edge graft gives every wall its own (near-orthogonal) direction at the
        # input; axdeg ties adjacent walls (|cos| ~0.5); "conversion to wall features" downstream would show as the adjacent
        # |cos| dropping toward the distant-pair value
        res["sites"][s] = {"abs_walls_at_path_tokens": float(a_abs.mean()), "local4_at_path_tokens": float(a_loc.mean()),
                           "local4_NESW": a_loc.tolist(), "abs_walls_at_PATH_START": float(a_ps.mean()),
                           "abs_walls_per_feature_at_path_tokens": a_abs.tolist(),
                           "probe_dir_cos_adjacent_at_path": pair_cos(W_abs, adj), "probe_dir_cos_distant_at_path": pair_cos(W_abs, far),
                           "probe_dir_cos_adjacent_at_PATH_START": pair_cos(W_ps, adj), "probe_dir_cos_distant_at_PATH_START": pair_cos(W_ps, far)}
        print(f"  {s:5s} abs walls @path {a_abs.mean():.3f}  local4 @path {a_loc.mean():.3f} (N/E/S/W {' '.join(f'{v:.2f}' for v in a_loc)})  abs walls @PATH_START {a_ps.mean():.3f}"
              f"  |cos| probe dirs adj/far @path {pair_cos(W_abs, adj):.2f}/{pair_cos(W_abs, far):.2f} @PS {pair_cos(W_ps, adj):.2f}/{pair_cos(W_ps, far):.2f}", flush=True)
    out = Path(out) if out else run / "eval" / "probe_walls.json"
    out.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(out, "w"), indent=1)
    print(f"done {run.name} in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
