#!/usr/bin/env python3
"""Linear vs nonlinear probes for ALLOCENTRIC (absolute) walls and EGOCENTRIC (current-cell-relative) walls, per layer.

For every site (block-0 input incl. the graft, and the residual after each block) at path-coordinate tokens:
  targets  allo: the 2n(n-1) free absolute wall bits (+-1)
           ego:  the walls at relative offsets max(|dr|,|dc|) <= R around the current cell, down+right, +-1, scored only where
                 the offset is inside the grid (a fixed set of 2(2R+1)^2 targets; R = 2 -> 50)
  probes   linear: ridge regression (closed form)
           mlp:    one hidden layer (512, GELU), Adam, early-stopped on a validation split, one probe per target family
  metrics  R^2 of the +-1 target (1 - SSE/SST on held-out mazes, targets pooled), sign accuracy, and per-target classification
           metrics macro-averaged: AUROC (threshold-free, chance 0.5), balanced accuracy, recall of open / wall; per-target values saved
Train / val / test = disjoint mazes (60 / 15 / 25 %).

  python probe_nonlinear.py --run_dir <run> --grid_n 6 [--n_mazes 4000] [--R 2]   -> <run>/eval/probe_nonlinear.json
"""
import pathlib as _pl, sys as _sys
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(next(p for p in _HERE.parents if (p / "maze" / "core").is_dir())))
_sys.path.insert(0, str(_HERE.parent / "curves")); _sys.path.insert(0, str(_HERE))
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
from probe_walls import free_idx


def ego_targets(conn, cell, n, R):
    """[2(2R+1)^2] in {-1, +1, 0=undefined}: down/right connection at offset (dr, dc) from `cell`."""
    r0, c0 = cell; out = np.zeros(2 * (2 * R + 1) ** 2, np.float32); k = 0
    for kind in (0, 1):
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < n and 0 <= c < n and ((kind == 0 and r < n - 1) or (kind == 1 and c < n - 1)):
                    out[k] = 1.0 if conn[kind * n * n + r * n + c] else -1.0
                k += 1
    return out


def r2_acc(pred, Y):
    m = Y != 0
    sse = ((pred - Y) ** 2 * m).sum(); sst = ((Y - (Y * m).sum(0) / m.sum(0).clamp(min=1)) ** 2 * m).sum()
    per = 1 - ((pred - Y) ** 2 * m).sum(0) / ((Y - (Y * m).sum(0) / m.sum(0).clamp(min=1)) ** 2 * m).sum(0).clamp(min=1e-6)
    acc = ((pred.sign() == Y) * m).sum(0) / m.sum(0).clamp(min=1)
    return float(1 - sse / sst), float(acc.mean()), per.cpu().numpy().tolist(), acc.cpu().numpy().tolist()


def class_metrics(pred, Y):
    """Per-target classification metrics for the +-1 targets (0 = undefined, masked), macro-averaged over targets:
    auroc (rank-based, threshold-free; chance 0.5), balanced accuracy at the sign threshold, recall of the open (+1) and
    wall (-1) classes, and the positive-class (open) fraction. Targets with a single class in the held-out set are skipped."""
    P = pred.detach().float().cpu().numpy(); T = Y.detach().float().cpu().numpy(); out = {"auroc": [], "bal_acc": [], "recall_open": [], "recall_wall": [], "frac_open": []}
    for j in range(T.shape[1]):
        m = T[:, j] != 0; y = T[m, j] > 0; s = P[m, j]
        if m.sum() == 0 or y.all() or (~y).all(): continue
        r = np.empty(len(s)); order = np.argsort(s, kind="mergesort"); r[order] = np.arange(1, len(s) + 1)   # ranks (ties broken arbitrarily)
        # average ranks for ties
        ss = s[order]; i = 0
        while i < len(ss):
            k = i
            while k + 1 < len(ss) and ss[k + 1] == ss[i]: k += 1
            if k > i: r[order[i:k + 1]] = (i + 1 + k + 1) / 2
            i = k + 1
        n1, n0 = y.sum(), (~y).sum(); auc = (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        ro = float(((s > 0) & y).sum() / n1); rw = float(((s <= 0) & ~y).sum() / n0)
        out["auroc"].append(float(auc)); out["bal_acc"].append((ro + rw) / 2); out["recall_open"].append(ro); out["recall_wall"].append(rw); out["frac_open"].append(float(n1 / (n1 + n0)))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()} | {"per_target_auroc": out["auroc"]}


def fit_linear(Xtr, Ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Xtr = torch.cat([Xtr, torch.ones(len(Xtr), 1, device=Xtr.device)], 1); Xte = torch.cat([Xte, torch.ones(len(Xte), 1, device=Xte.device)], 1)
    m = (Ytr != 0).float()                                   # masked least squares: solve per target with the shared Gram is wrong under masks,
    W = torch.zeros(Xtr.shape[1], Ytr.shape[1], device=Xtr.device)   # so loop over targets (cheap: d = 257)
    I = lam * torch.eye(Xtr.shape[1], device=Xtr.device)
    for j in range(Ytr.shape[1]):
        Xj = Xtr * m[:, j:j + 1]
        W[:, j] = torch.linalg.solve(Xj.T @ Xj + I, Xj.T @ Ytr[:, j])
    return Xte @ W


def fit_mlp(Xtr, Ytr, Xva, Yva, Xte, hidden=512, epochs=60, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    f = lambda X: (X - mu) / sd
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, Ytr.shape[1])).to(Xtr.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    mtr, mva = (Ytr != 0).float(), (Yva != 0).float(); best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=Xtr.device)
        for lo in range(0, len(Xtr), 512):
            i = perm[lo:lo + 512]; opt.zero_grad()
            loss = (((net(f(Xtr[i])) - Ytr[i]) ** 2) * mtr[i]).sum() / mtr[i].sum()
            loss.backward(); opt.step()
        with torch.no_grad():
            v = float((((net(f(Xva)) - Yva) ** 2) * mva).sum() / mva.sum())
        if v < best - 1e-4: best, bad, best_state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6: break
    net.load_state_dict(best_state)
    with torch.no_grad():
        return net(f(Xte))


def main(run_dir: str, grid_n: int, n_mazes: int = 4000, seed: int = 999, per_maze: int = 8, R: int = 2, out: str = "",
         cache_dir: str = "/workspace/evalsets", batch: int = 250, sites: str = "all"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_float32_matmul_precision("high")
    run = Path(run_dir); n = grid_n
    cfg = ConfigHolder.load(json.load(open(run / "config_holder.json"))); vocab = cfg.tokenizer.get_vocab()
    id_to_coord = {i: (int(m.group(1)), int(m.group(2))) for t, i in vocab.items() if (m := re.compile(r"^\((\d+),(\d+)\)$").match(t))}
    pad_id, ps_id = vocab[SPECIAL_TOKENS.PADDING], vocab[SPECIAL_TOKENS.PATH_START]
    d = Path(cache_dir) / f"probe-g{n}-s{seed}-n{n_mazes}"; d.mkdir(parents=True, exist_ok=True); pre = d / "pretok.pt"
    if not pre.exists():
        ds = MazeDataset.from_config(MazeDatasetConfig(name="probeset", grid_n=n, n_mazes=n_mazes, maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed),
                                     do_generate=True, load_local=False, save_local=False, gen_parallel=False, verbose=False)
        pretokenize(ds, cfg.maze_tokenizer, vocab, pre, bos_id=int(cfg.tokenizer.bos_token_id), workers=0)
    pt = PretokDataset(pre); rows = [pt[i] for i in range(len(pt))]
    inj_path = run / "injector.final.pt"
    model, injector = load_ckpt(run / "model.final.pt", inj_path if inj_path.exists() else None, device)
    g = as_graft(injector); layer = int(g.layer) if g is not None else 0
    L = max(len(r[0]) for r in rows)
    toks = torch.full((len(rows), L), pad_id, dtype=torch.long); attn = torch.zeros_like(toks)
    for i, (s, _, _) in enumerate(rows): toks[i, :len(s)] = s.long(); attn[i, :len(s)] = 1
    conn = torch.stack([c for _, c, _ in rows]); end = torch.tensor([int(e) for _, _, e in rows]); conn_np = conn.numpy()
    fi = free_idx(n); rng = np.random.default_rng(seed); samp_pos, samp_maze = [], []
    for i, (s, _, _) in enumerate(rows):
        s = s.tolist(); p0 = s.index(ps_id); cand = [p for p in range(p0 + 1, len(s)) if s[p] in id_to_coord]
        pick = rng.choice(cand, size=min(per_maze, len(cand)), replace=False); samp_pos += pick.tolist(); samp_maze += [i] * len(pick)
    samp_pos, samp_maze = np.array(samp_pos), np.array(samp_maze)
    Y = {"allo": torch.tensor(conn_np[samp_maze][:, fi].astype(np.float32) * 2 - 1),
         "ego": torch.tensor(np.stack([ego_targets(conn_np[m], id_to_coord[int(toks[m, p])], n, R) for m, p in zip(samp_maze, samp_pos)]))}
    site_list = list(sites) if isinstance(sites, (tuple, list)) else str(sites).split(",")   # fire parses "a,b" as a tuple
    if site_list == ["all"]:
        site_list = ["L0in"] + [f"L{k}" for k in range(1, len(model.blocks) + 1)]                # every layer
    n_layers = len(model.blocks); acts = {s: [] for s in site_list}; t0 = time.time()
    with torch.no_grad():
        for lo in range(0, len(rows), batch):
            tb, ab, cb, eb = (x[lo:lo + batch].to(device) for x in (toks, attn, conn, end))
            delta = g.delta(g.code.features(tb, ab, cb, eb)) if g is not None else None
            sel = (samp_maze >= lo) & (samp_maze < lo + batch); mi, pi = torch.tensor(samp_maze[sel] - lo, device=device), torch.tensor(samp_pos[sel], device=device)
            if "L0in" in acts:
                x0 = model.embed(tb) + model._positions(tb, ab)
                if delta is not None and layer == 0: x0 = x0 + delta.to(x0.dtype)
                acts["L0in"].append(x0[mi, pi].float().cpu())
            for s in site_list:
                if s == "L0in": continue
                k = int(s[1:]); h = model(tb, attention_mask=ab, inject_delta=delta, inject_layer=layer, padding_mode="right", stop_at_layer=k)
                acts[s].append(h[mi, pi].float().cpu())
    n_tr, n_va = int(0.6 * len(rows)), int(0.75 * len(rows)); tr, va, te = samp_maze < n_tr, (samp_maze >= n_tr) & (samp_maze < n_va), samp_maze >= n_va
    res = {"run": run.name, "grid_n": n, "code": (g.code.name if g else "none"), "R": R, "n_rows": dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())), "sites": {}}
    for s in site_list:
        X = torch.cat(acts[s]).to(device); res["sites"][s] = {}
        for fam in ("allo", "ego"):
            Yf = Y[fam].to(device)
            lin = fit_linear(X[tr], Yf[tr], X[te]); mlp = fit_mlp(X[tr], Yf[tr], X[va], Yf[va], X[te])
            for name, pred in (("linear", lin), ("mlp", mlp)):
                r2, acc, per_r2, per_acc = r2_acc(pred, Yf[te])
                res["sites"][s][f"{fam}_{name}"] = {"r2": r2, "acc": acc, "per_target_r2": per_r2, "per_target_acc": per_acc} | class_metrics(pred, Yf[te])
        v = res["sites"][s]
        print(f"  {s:5s} allo R2 lin {v['allo_linear']['r2']:.3f} mlp {v['allo_mlp']['r2']:.3f} (acc {v['allo_linear']['acc']:.3f}/{v['allo_mlp']['acc']:.3f}) | "
              f"ego R2 lin {v['ego_linear']['r2']:.3f} mlp {v['ego_mlp']['r2']:.3f} (acc {v['ego_linear']['acc']:.3f}/{v['ego_mlp']['acc']:.3f})  [{time.time()-t0:.0f}s]", flush=True)
    out = Path(out) if out else run / "eval" / "probe_nonlinear.json"; out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w")); print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
