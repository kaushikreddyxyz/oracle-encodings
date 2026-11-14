#!/usr/bin/env python3
"""No-model floors for the wall probes of probe_nonlinear.py (CPU only, no transformer forward pass).

Same probe set as probe_nonlinear.py main(): MazeDataset(grid_n=n, n_mazes=4000, gen_dfs, seed=999), tokenized with the run's
tokenizer, per_maze=8 path-coordinate positions per maze sampled with default_rng(999) in the same call order, same targets
(allo = free wall bits, ego = R=2 offsets), same maze split (60/15/25 %), same masked ridge (fit_linear, lam=1) and MLP (fit_mlp),
same metrics (r2_acc, class_metrics). The probe input is a hand-built feature vector instead of an activation:

  F0  intercept only (a constant column; fit_linear adds the bias)
  F1  one-hot current cell                                                          (n^2)
  F2  F1 + one-hot previous path cell (zeros at the first path token) + one-hot incoming move N/E/S/W (zeros at first token)
                                                                                    (2n^2 + 4)
  F3  F2 + bag of visited path cells so far, current included                     (3n^2 + 4)
  F4  F3 + one-hot target cell (from the prompt)                                   (4n^2 + 4)
  F5  F4 + sequence position index + number of path steps taken                     (4n^2 + 6)
 extras (not in the requested ladder, labelled separately):
  F3edge  F2 + bag of traversed connections in the absolute frame (2n^2 indicators, same indexing as conn)
  F3rel   F2 + traversed connections in the ego frame (the 50 ego (kind,dr,dc) slots, 1 = traversed so far)
  mlp_F3  fit_mlp (512 GELU, early-stopped on the val split) on F3

Subsets: ego 'imm' = S,N,E,W walls of the current cell = (down,0,0),(down,-1,0),(right,0,0),(right,0,-1); ego 'far' = the other 46;
allo 'bnd' = walls with at least one endpoint cell on the grid boundary, 'int' = the rest.
Model rows: the same subsets recomputed from per_target_auroc in results/probe_v2/<tag>.json (the files ev_probes.csv is built from).

  python probe_floors.py [--grids 6,8,10]  -> results/diag_probes/floors.csv
"""
import pathlib as _pl, sys as _sys
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(next(p for p in _HERE.parents if (p / "maze" / "core").is_dir())))
_sys.path.insert(0, str(_HERE.parent / "curves")); _sys.path.insert(0, str(_HERE))
import maze._pathsetup  # noqa
import csv, json, re, time, tempfile
from pathlib import Path
import fire, numpy as np, torch
from maze_dataset import SPECIAL_TOKENS, MazeDataset, MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators
from maze_transformer.training.config import ConfigHolder
from maze.core.data import PretokDataset, pretokenize
from probe_walls import free_idx
from probe_nonlinear import ego_targets, fit_linear, fit_mlp, r2_acc, class_metrics

RUNS = _HERE / "results" / "runs"; V2 = _HERE / "results" / "probe_v2"; OUT = _HERE / "results" / "diag_probes"
EGO_IMM = [12, 7, 37, 36]   # kind*25 + (dr+2)*5 + (dc+2): down(0,0)=S, down(-1,0)=N, right(0,0)=E, right(0,-1)=W


def allo_subsets(n):
    bnd = []
    for f in free_idx(n):
        k, rc = divmod(int(f), n * n); r, c = divmod(rc, n); r2, c2 = (r + 1, c) if k == 0 else (r, c + 1)
        bnd.append(any(x in (0, n - 1) for x in (r, c, r2, c2)))
    bnd = np.array(bnd); return {"bnd": np.nonzero(bnd)[0], "int": np.nonzero(~bnd)[0]}


def subsets(fam, n):
    if fam == "ego":
        return {"all": np.arange(50), "imm": np.array(EGO_IMM), "far": np.array([j for j in range(50) if j not in EGO_IMM])}
    return {"all": np.arange(len(free_idx(n)))} | allo_subsets(n)


def build(n, n_mazes=4000, seed=999, per_maze=8, R=2, cache=None):
    run = next((RUNS / f"g{n}-s0-base-e4-oe").glob("fast-baseline*"))
    cfg = ConfigHolder.load(json.load(open(run / "config_holder.json"))); vocab = cfg.tokenizer.get_vocab()
    id_to_coord = {i: (int(m.group(1)), int(m.group(2))) for t, i in vocab.items() if (m := re.compile(r"^\((\d+),(\d+)\)$").match(t))}
    ps_id, ts_id, te_id = vocab[SPECIAL_TOKENS.PATH_START], vocab[SPECIAL_TOKENS.TARGET_START], vocab[SPECIAL_TOKENS.TARGET_END]
    cache = Path(cache or tempfile.gettempdir()) / f"probe-g{n}-s{seed}-n{n_mazes}"; cache.mkdir(parents=True, exist_ok=True); pre = cache / "pretok.pt"
    if not pre.exists():
        ds = MazeDataset.from_config(MazeDatasetConfig(name="probeset", grid_n=n, n_mazes=n_mazes, maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed),
                                     do_generate=True, load_local=False, save_local=False, gen_parallel=False, verbose=False)
        pretokenize(ds, cfg.maze_tokenizer, vocab, pre, bos_id=int(cfg.tokenizer.bos_token_id), workers=0)
    pt = PretokDataset(pre); rows = [pt[i] for i in range(len(pt))]
    conn_np = torch.stack([c for _, c, _ in rows]).numpy()
    fi = free_idx(n); rng = np.random.default_rng(seed); feats, Ya, Ye, samp_maze = [], [], [], []
    N2 = n * n; DIRS = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    for i, (s, _, _) in enumerate(rows):                       # identical sampling loop / rng call order to probe_nonlinear.main
        s = s.tolist(); p0 = s.index(ps_id); cand = [p for p in range(p0 + 1, len(s)) if s[p] in id_to_coord]
        pick = rng.choice(cand, size=min(per_maze, len(cand)), replace=False)
        tgt = id_to_coord[s[s.index(ts_id) + 1]]; assert s[s.index(ts_id) + 2] == te_id
        path = [id_to_coord[s[p]] for p in cand]; assert cand == list(range(p0 + 1, p0 + 1 + len(cand)))
        for p in pick.tolist():
            k = p - (p0 + 1); cur = path[k]; ci = cur[0] * n + cur[1]
            prev = np.zeros(N2, np.float32); mv = np.zeros(4, np.float32); vis = np.zeros(N2, np.float32); tr = np.zeros(2 * N2, np.float32)
            if k > 0:
                pr = path[k - 1]; prev[pr[0] * n + pr[1]] = 1; mv[DIRS[(cur[0] - pr[0], cur[1] - pr[1])]] = 1
            for j in range(k + 1): vis[path[j][0] * n + path[j][1]] = 1
            for j in range(k):
                (a, b), (c, d) = sorted([path[j], path[j + 1]])
                tr[(0 if c == a + 1 else 1) * N2 + a * n + b] = 1
            rel = np.zeros(50, np.float32); q = 0                  # ego-frame traversed connections, same slot order as ego_targets
            for kind in (0, 1):
                for dr in range(-R, R + 1):
                    for dc in range(-R, R + 1):
                        r, c = cur[0] + dr, cur[1] + dc
                        if 0 <= r < n and 0 <= c < n: rel[q] = tr[kind * N2 + r * n + c]
                        q += 1
            cur1 = np.zeros(N2, np.float32); cur1[ci] = 1; tg = np.zeros(N2, np.float32); tg[tgt[0] * n + tgt[1]] = 1
            feats.append(dict(cur=cur1, prev=prev, mv=mv, vis=vis, tgt=tg, pos=np.array([p, k], np.float32), tr=tr, rel=rel))
            Ya.append(conn_np[i][fi].astype(np.float32) * 2 - 1); Ye.append(ego_targets(conn_np[i], cur, n, R)); samp_maze.append(i)
    samp_maze = np.array(samp_maze); cat = lambda keys: torch.tensor(np.stack([np.concatenate([f[k] for k in keys]) for f in feats]))
    F = {"F0": torch.ones(len(feats), 1), "F1": cat(["cur"]), "F2": cat(["cur", "prev", "mv"]), "F3": cat(["cur", "prev", "mv", "vis"]),
         "F4": cat(["cur", "prev", "mv", "vis", "tgt"]), "F5": cat(["cur", "prev", "mv", "vis", "tgt", "pos"]),
         "F3edge": cat(["cur", "prev", "mv", "tr"]), "F3rel": cat(["cur", "prev", "mv", "rel"])}
    Y = {"allo": torch.tensor(np.stack(Ya)), "ego": torch.tensor(np.stack(Ye))}
    n_tr, n_va = int(0.6 * len(rows)), int(0.75 * len(rows))
    split = samp_maze < n_tr, (samp_maze >= n_tr) & (samp_maze < n_va), samp_maze >= n_va
    return F, Y, split


def score(pred, Yte, fam, n):
    out = []
    for sub, idx in subsets(fam, n).items():
        r2, acc, _, _ = r2_acc(pred[:, idx], Yte[:, idx]); cm = class_metrics(pred[:, idx], Yte[:, idx])
        out.append(dict(subset=sub, n_targets=len(idx), auroc=cm["auroc"], bal_acc=cm["bal_acc"], r2=r2, acc=acc))
    return out


def model_rows(n, layers=(0, 3, 6, 12)):
    out = []
    for f in sorted(V2.glob(f"g{n}-s*-*-e4-oe.json")):
        m = re.match(rf"g{n}-s(\d+)-(base|edge|ego)-e4", f.name); P = json.load(open(f))
        for L in layers:
            v = P["sites"]["L0in" if L == 0 else f"L{L}"]
            for fam in ("allo", "ego"):
                for kind in ("linear", "mlp"):
                    e = v[f"{fam}_{kind}"]; pa = np.array(e["per_target_auroc"])
                    if len(pa) != len(subsets(fam, n)["all"]): continue   # a target was skipped -> subset indices ambiguous
                    for sub, idx in subsets(fam, n).items():
                        out.append(dict(source="model", grid=n, code=m.group(2), seed=int(m.group(1)), layer=L, features="", probe=kind, family=fam,
                                        subset=sub, n_targets=len(idx), auroc=float(pa[idx].mean()), bal_acc=e["bal_acc"] if sub == "all" else np.nan,
                                        r2=e["r2"] if sub == "all" else np.nan, acc=e["acc"] if sub == "all" else np.nan))
    return out


def main(grids: str = "6,8,10", cache: str = ""):
    torch.manual_seed(0); rows = []
    grids = [int(g) for g in (grids if isinstance(grids, (tuple, list)) else str(grids).split(","))]
    for n in grids:
        t0 = time.time(); F, Y, (tr, va, te) = build(n, cache=cache or None)
        print(f"g{n}: rows train {tr.sum()} val {va.sum()} test {te.sum()}  (built {time.time()-t0:.0f}s)", flush=True)
        v2 = json.load(open(V2 / f"g{n}-s0-base-e4-oe.json"))["n_rows"]
        assert v2 == dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())), f"sample counts differ from probe_v2: {v2}"
        for fam in ("allo", "ego"):
            Yf = Y[fam]
            jobs = [(k, "linear", lambda X=X: fit_linear(X[tr], Yf[tr], X[te])) for k, X in F.items()]
            jobs.append(("F3", "mlp", lambda: fit_mlp(F["F3"][tr], Yf[tr], F["F3"][va], Yf[va], F["F3"][te])))
            for k, kind, fn in jobs:
                for s in score(fn(), Yf[te], fam, n):
                    rows.append(dict(source="floor", grid=n, code="none", seed=-1, layer=-1, features=k, probe=kind, family=fam) | s)
                    if s["subset"] == "all" or fam == "ego": print(f"  {fam:4s} {k:6s} {kind:6s} {s['subset']:4s} auroc {s['auroc']:.3f} bal {s['bal_acc']:.3f} r2 {s['r2']:.3f}", flush=True)
        rows += model_rows(n)
        print(f"  [{time.time()-t0:.0f}s]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "floors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"done -> {OUT / 'floors.csv'}")


if __name__ == "__main__":
    fire.Fire(main)
