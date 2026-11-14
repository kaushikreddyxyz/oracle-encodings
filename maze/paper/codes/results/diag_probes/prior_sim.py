#!/usr/bin/env python3
"""Model-free reference: what a ridge probe can read at the block-0 INPUT from (a) the token + position alone and
(b) that plus the raw graft features, on the same probe mazes (gen_dfs, seed 999, 4000 mazes, 8 path tokens per maze,
same rng draws and 60/25 train/test split as probe_nonlinear.py).  No transformer is run.

Feature sets (all additive, as in x0 = embed(token) + pos(p) [+ W_inj features]):
  prior      one-hot(current cell) + one-hot(step along the path)    [the adjacency list has fixed length, so p <-> step]
  prior+edge prior + the edge code (2n^2 absolute walls, +-1)
  prior+ego  prior + the ego code (2(2n-1)^2 relative walls, +-1, 0 outside the grid)
Outputs sim_ego_offsets.csv, sim_allo_per_wall.csv (incl. per-wall frac_open on the test set).
"""
import csv, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); from probe_offsets import free_idx, OFF
R = 2


def ego_t(conn, r0, c0, n):
    out = np.zeros(50, np.float32)
    for k, (kind, dr, dc) in enumerate(OFF):
        r, c = r0 + dr, c0 + dc
        if 0 <= r < n and 0 <= c < n and ((kind == 0 and r < n - 1) or (kind == 1 and c < n - 1)):
            out[k] = 1.0 if conn[kind * n * n + r * n + c] else -1.0
    return out


def ego_code(conn, r0, c0, n):
    m = 2 * n - 1; out = np.zeros(2 * m * m, np.float32); s = conn.astype(np.float32) * 2 - 1
    for kind in (0, 1):
        for r in range(n):
            for c in range(n):
                out[kind * m * m + (r - r0 + n - 1) * m + (c - c0 + n - 1)] = s[kind * n * n + r * n + c]
    return out


def ridge(Xtr, Ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]; Xte = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
    M = Ytr != 0; P = np.zeros((len(Xte), Ytr.shape[1]))
    if M.all():
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Ytr); return Xte @ W
    for j in range(Ytr.shape[1]):
        Xj = Xtr[M[:, j]]; P[:, j] = Xte @ np.linalg.solve(Xj.T @ Xj + lam * np.eye(Xtr.shape[1]), Xj.T @ Ytr[M[:, j], j])
    return P


def aucs(P, Y):
    out = []
    for j in range(Y.shape[1]):
        m = Y[:, j] != 0; y = Y[m, j] > 0
        out.append(roc_auc_score(y, P[m, j]) if 0 < y.sum() < len(y) else np.nan)
    return np.array(out)


def main(n_mazes=4000, seed=999, per_maze=8):
    fe = open(HERE / "sim_ego_offsets.csv", "w", newline=""); we = csv.writer(fe); we.writerow(["grid", "features", "kind", "dr", "dc", "auroc", "frac_open", "n_test_defined"])
    fa = open(HERE / "sim_allo_per_wall.csv", "w", newline=""); wa = csv.writer(fa); wa.writerow(["grid", "features", "kind", "r", "c", "auroc", "frac_open"])
    for n in range(5, 11):
        ds = MazeDataset.from_config(MazeDatasetConfig(name="probeset", grid_n=n, n_mazes=n_mazes, maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed),
                                     do_generate=True, load_local=False, save_local=False, gen_parallel=False, verbose=False)
        rng = np.random.default_rng(seed); fi = free_idx(n); rows = []
        for i, mz in enumerate(ds.mazes):
            conn = np.asarray(mz.connection_list).reshape(-1).astype(bool); path = np.asarray(mz.solution)
            for st in rng.choice(np.arange(len(path)), size=min(per_maze, len(path)), replace=False):
                rows.append((i, int(st), int(path[st][0]), int(path[st][1]), conn))
        mi = np.array([r[0] for r in rows]); step = np.array([r[1] for r in rows]); cell = np.array([r[2] * n + r[3] for r in rows])
        Ya = np.stack([r[4][fi] for r in rows]).astype(np.float32) * 2 - 1
        Ye = np.stack([ego_t(r[4], r[2], r[3], n) for r in rows])
        oh_c = np.eye(n * n, dtype=np.float32)[cell]; oh_s = np.eye(step.max() + 1, dtype=np.float32)[step]
        prior = np.c_[oh_c, oh_s]
        feats = {"prior": prior, "prior+edge": np.c_[prior, np.stack([r[4] for r in rows]).astype(np.float32) * 2 - 1],
                 "prior+ego": np.c_[prior, np.stack([ego_code(r[4], r[2], r[3], n) for r in rows])]}
        tr, te = mi < int(0.6 * n_mazes), mi >= int(0.75 * n_mazes)
        print(f"g{n}: samples train {tr.sum()} test {te.sum()}", flush=True)
        fo_e = [(Ye[te, j] > 0).sum() / max(1, (Ye[te, j] != 0).sum()) for j in range(50)]; nd_e = (Ye[te] != 0).sum(0)
        fo_a = (Ya[te] > 0).mean(0)
        for name, X in feats.items():
            ae = aucs(ridge(X[tr], Ye[tr], X[te]), Ye[te]); aa = aucs(ridge(X[tr], Ya[tr], X[te]), Ya[te])
            for j, (k, dr, dc) in enumerate(OFF): we.writerow([n, name, "down" if k == 0 else "right", dr, dc, f"{ae[j]:.4f}", f"{fo_e[j]:.4f}", int(nd_e[j])])
            for j, f in enumerate(fi):
                k, rc = divmod(f, n * n); wa.writerow([n, name, "down" if k == 0 else "right", rc // n, rc % n, f"{aa[j]:.4f}", f"{fo_a[j]:.4f}"])
            print(f"   {name:11s} ego imm {np.nanmean(ae[[7, 12, 36, 37]]):.3f} all {np.nanmean(ae):.3f} | allo {np.nanmean(aa):.3f}", flush=True)
    fe.close(); fa.close()


if __name__ == "__main__":
    main()
