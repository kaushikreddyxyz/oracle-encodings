#!/usr/bin/env python3
"""Causal test of probe directions: does pushing the residual along the ridge-probe direction for
"wall D of the current cell is open" move the next-move distribution toward stepping D?

Per model (one run dir) and site (block-k input, k in {0, 3, 6}; block-0 input includes the graft delta):
  1. activations at every path-coordinate position of the teacher-forced true-world sequences of a fresh
     EvalSet (seed 4321, 2000 mazes); ridge probes exactly as probe_nonlinear.fit_linear (standardise with
     train mu/sd, append bias column, masked ridge lam=1) on the first 60 % of mazes for
       ego  the 4 egocentric immediate-neighbour walls S/N/E/W (+1 open, -1 wall, 0 = off-grid, masked)
       allo the 2n(n-1) absolute wall bits
     AUROC on the last 25 % of mazes (probe_nonlinear.class_metrics).
  2. steering on test-maze positions (not the target cell) where the neighbour in direction D is in the grid and
       closed  the wall is closed (stepping D invalid)
       open    the wall is open but D is not the shortest-path move
     a vector alpha * u is added to the block-k input at that single position (forward pre-hook on blocks[k]), with
       u = unit(W[:-1, j] / sd)  (raw-space probe gradient, bias dropped), sign +1 = "open", -1 = "wall"
       alpha = s * std_train(X @ u_egoprobe_D)   s in {1, 2, 4, 8, 16} -- ONE alpha per (site, D) shared by all
       direction types so they are same-norm perturbations
     direction types: ego_probe (+/-), allo_probe (+/-, the absolute wall that is the D-neighbour at that position),
     graft (+/-, block 0 only, grafted models: the injector's own row for that wall -- edge: absolute feature,
     ego: relative offset), random (8 fixed random unit vectors, + only).
     Measured at that position vs the unsteered run: delta p(D-neighbour token), delta log p_D, delta p(shortest-path
     moves), delta p(invalid) (= 1 - p_sp - p_valid-other-open), argmax flip, argmax -> D.

  python steer_probe.py --run_dir <run> --grid_n 6 --label edge --out_dir <dir>
"""
import pathlib as _pl, sys as _sys
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(next(p for p in _HERE.parents if (p / "maze" / "core").is_dir())))
_sys.path.insert(0, str(_HERE.parent / "curves")); _sys.path.insert(0, str(_HERE))
import maze._pathsetup  # noqa
import csv, json, time
from pathlib import Path
import fire, numpy as np, torch
from maze_transformer.training.config import ConfigHolder
from maze.core.graft import as_graft, GraftInjector
from eval_ckpts import load_ckpt, EvalSet, feat_idx, dist_to, open_nbrs
from probe_walls import free_idx
from probe_nonlinear import class_metrics

DIRS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
S_LIST = (1, 2, 4, 8, 16)


def fit_ridge(Xtr, Ytr, lam=1.0):
    """probe_nonlinear.fit_linear, returning (W [d+1, T], mu, sd)."""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xs = torch.cat([(Xtr - mu) / sd, torch.ones(len(Xtr), 1, device=Xtr.device)], 1)
    m = (Ytr != 0).float(); I = lam * torch.eye(Xs.shape[1], device=Xs.device)
    W = torch.zeros(Xs.shape[1], Ytr.shape[1], device=Xs.device)
    for j in range(Ytr.shape[1]):
        Xj = Xs * m[:, j:j + 1]
        W[:, j] = torch.linalg.solve(Xj.T @ Xj + I, Xj.T @ Ytr[:, j])
    return W, mu, sd


def predict(W, mu, sd, X):
    return torch.cat([(X - mu) / sd, torch.ones(len(X), 1, device=X.device)], 1) @ W


def unit(v):
    return v / v.norm()


def main(run_dir: str, grid_n: int, label: str, out_dir: str, eval_dir: str = "/workspace/evalsets",
         n_mazes: int = 2000, seed: int = 4321, sites: str = "0,3,6", per_cell: int = 150, chunk: int = 3000,
         n_rand: int = 8, s_list: str = "1,2,4,8,16"):
    S_LIST = [int(x) for x in (s_list if isinstance(s_list, (tuple, list)) else str(s_list).split(","))]
    t0 = time.time()
    device = torch.device("cuda"); torch.set_float32_matmul_precision("high")
    run = Path(run_dir); n = grid_n; out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = ConfigHolder.load(json.load(open(run / "config_holder.json")))
    es = EvalSet(cfg, n, Path(eval_dir), n_mazes, seed, 2, 2, device)
    inj_path = run / "injector.final.pt"
    model, injector = load_ckpt(run / "model.final.pt", inj_path if inj_path.exists() else None, device)
    g = as_graft(injector); glayer = int(g.layer) if g is not None else 0
    assert glayer == 0
    site_ks = [int(k) for k in (sites if isinstance(sites, (tuple, list)) else str(sites).split(","))]
    fi = free_idx(n); fi_pos = {int(f): j for j, f in enumerate(fi)}

    # ---- sequences + position table -------------------------------------------------------------
    seqs, meta = [], []            # meta per position: maze, pos, cell, step, is_last, prev cell
    for i, path in enumerate(es.true_path):
        op = int(es.origin_pos[i])
        s = es.seqs[i][: op + 1] + [es.coord_to_id[c] for c in path[1:]] + [es.path_end_id]
        seqs.append(s)
        for step, cell in enumerate(path):
            meta.append(dict(m=i, p=op + step, cell=cell, last=step == len(path) - 1, prev=path[step - 1] if step else None,
                             nxt=path[step + 1] if step + 1 < len(path) else None))
    Lmax = max(len(s) for s in seqs)
    toks_all = torch.full((len(seqs), Lmax), es.pad_id, dtype=torch.long)
    for i, s in enumerate(seqs): toks_all[i, :len(s)] = torch.tensor(s)
    lens = torch.tensor([len(s) for s in seqs])
    conn_all = torch.tensor(es.conn, dtype=torch.bool); end_all = torch.tensor(es.end_pos, dtype=torch.long)
    pm = np.array([r["m"] for r in meta]); pp = np.array([r["p"] for r in meta])

    def nb_of(cell, D):
        r, c = cell[0] + DIRS[D][0], cell[1] + DIRS[D][1]
        return (r, c) if 0 <= r < n and 0 <= c < n else None

    Yego = np.zeros((len(meta), 4), np.float32)
    for q, r in enumerate(meta):
        for j, D in enumerate(DIRS):
            nb = nb_of(r["cell"], D)
            if nb is not None: Yego[q, j] = 1.0 if es.conn[r["m"], feat_idx(r["cell"], nb, n)] else -1.0
    Yallo = es.conn[pm][:, fi].astype(np.float32) * 2 - 1

    def graft_delta(tb, ab, cb, eb):
        return g.delta(g.code.features(tb, ab, cb, eb)) if g is not None else None

    # ---- activations -----------------------------------------------------------------------------
    acts = {k: [] for k in site_ks}
    with torch.no_grad():
        for lo in range(0, len(seqs), 250):
            hi = min(lo + 250, len(seqs)); tb = toks_all[lo:hi].to(device)
            ab = (torch.arange(Lmax)[None] < lens[lo:hi, None]).long().to(device)
            cb, eb = conn_all[lo:hi].to(device), end_all[lo:hi].to(device)
            delta = graft_delta(tb, ab, cb, eb)
            sel = (pm >= lo) & (pm < hi); mi = torch.tensor(pm[sel] - lo, device=device); pi = torch.tensor(pp[sel], device=device)
            for k in site_ks:
                if k == 0:
                    x0 = model.embed(tb) + model._positions(tb, ab)
                    if delta is not None: x0 = x0 + delta.to(x0.dtype)
                else:
                    x0 = model(tb, attention_mask=ab, inject_delta=delta, inject_layer=glayer, padding_mode="right", stop_at_layer=k)
                acts[k].append(x0[mi, pi].float())
    acts = {k: torch.cat(v) for k, v in acts.items()}
    n_tr, n_te = int(0.6 * n_mazes), int(0.75 * n_mazes)
    tr, te = pm < n_tr, pm >= n_te
    print(f"[{label}] {len(meta)} positions, acts done [{time.time()-t0:.0f}s]", flush=True)

    # ---- probes ----------------------------------------------------------------------------------
    Ye, Ya = torch.tensor(Yego, device=device), torch.tensor(Yallo, device=device)
    probes, auroc = {}, {}
    for k in site_ks:
        X = acts[k]
        We, mue, sde = fit_ridge(X[tr], Ye[tr]); Wa, mua, sda = fit_ridge(X[tr], Ya[tr])
        ce = class_metrics(predict(We, mue, sde, X[te]), Ye[te]); ca = class_metrics(predict(Wa, mua, sda, X[te]), Ya[te])
        auroc[k] = dict(ego_mean=ce["auroc"], ego_per_dir=dict(zip(DIRS, ce["per_target_auroc"])), allo_mean=ca["auroc"])
        ego_u = torch.stack([unit(We[:-1, j] / sde[0]) for j in range(4)])          # [4, d]
        allo_u = torch.stack([unit(Wa[:-1, j] / sda[0]) for j in range(Wa.shape[1])])  # [F_free, d]
        proj_sd = (X[tr] @ ego_u.T).std(0)                                          # [4]
        probes[k] = dict(ego_u=ego_u, allo_u=allo_u, proj_sd=proj_sd, allo_proj_sd=(X[tr] @ allo_u.T).std(0),
                         xnorm=float(X[tr].norm(dim=1).mean()))
        print(f"  site L{k}in: ego AUROC {ce['auroc']:.3f} " + " ".join(f"{D}:{a:.3f}" for D, a in zip(DIRS, ce["per_target_auroc"]))
              + f" | allo AUROC {ca['auroc']:.3f} | proj sd {[round(float(v),2) for v in proj_sd]} |x| {probes[k]['xnorm']:.1f}", flush=True)

    # graft rows (block 0): unit injector row for the feature that encodes wall D of the current cell
    def graft_vec(cell, D):
        if g is None: return None
        nb = nb_of(cell, D); f = feat_idx(cell, nb, n)
        if isinstance(g, GraftInjector) and g.code.name == "ego":
            kind, rc = divmod(f, n * n); r, c = divmod(rc, n); mm = 2 * n - 1
            idx = kind * mm * mm + (r - cell[0] + n - 1) * mm + (c - cell[1] + n - 1)
            return unit(g.W[idx].float())
        if isinstance(g, GraftInjector):
            return unit(g.W[f].float()) if g.code.name == "edge" else None
        return unit(g.inj.W_inj[f].float())

    gen = torch.Generator(device="cpu").manual_seed(0)
    rand_u = {k: torch.stack([unit(torch.randn(acts[k].shape[1], generator=gen)) for _ in range(n_rand)]).to(device) for k in site_ks}

    # ---- steering positions ----------------------------------------------------------------------
    rng = np.random.default_rng(0); cand = {}
    dist_cache = {}
    for q, r in enumerate(meta):
        if r["m"] < n_te or r["last"]: continue
        i = r["m"]
        if i not in dist_cache: dist_cache[i] = dist_to(es.conn[i], es.true_path[i][-1], n)
        dist = dist_cache[i]
        for D in DIRS:
            nb = nb_of(r["cell"], D)
            if nb is None: continue
            op = bool(es.conn[i, feat_idx(r["cell"], nb, n)])
            if not op: cond = "closed"
            elif dist.get(nb, 1e9) == dist[r["cell"]] - 1: continue
            else: cond = "open"
            cand.setdefault((D, cond), []).append(q)
    chosen = []
    for key, qs in sorted(cand.items()):
        pick = rng.choice(qs, size=min(per_cell, len(qs)), replace=False)
        chosen += [(int(q), key[0], key[1]) for q in pick]
    print(f"  steering positions: " + ", ".join(f"{k}:{min(per_cell, len(v))}/{len(v)}" for k, v in sorted(cand.items())), flush=True)

    # ---- steering runs ---------------------------------------------------------------------------
    hold = {}
    def pre_hook(mod, args):
        x = args[0].clone()
        x[hold["rows"], hold["pos"]] += hold["vec"].to(x.dtype)
        return (x,) + tuple(args[1:])

    recs = []   # per (position, variant) metrics
    for k in site_ks:
        P = probes[k]; handle = model.blocks[k].register_forward_pre_hook(pre_hook)
        rows = []   # (q, D, cond, dtype, sign, s, vec)
        for q, D, cond in chosen:
            r = meta[q]; j = list(DIRS).index(D); nb = nb_of(r["cell"], D)
            base_alpha = float(P["proj_sd"][j])
            rows.append((q, D, cond, "none", 0, 0, None))
            fa = fi_pos[feat_idx(r["cell"], nb, n)]
            dirs = [("ego_probe", P["ego_u"][j]), ("allo_probe", P["allo_u"][fa])]
            if k == 0:
                gv = graft_vec(r["cell"], D)
                if gv is not None: dirs.append(("graft", gv))
            for s in S_LIST:
                a = s * base_alpha
                for name, u in dirs:
                    for sign in (1, -1): rows.append((q, D, cond, name, sign, s, sign * a * u))
                for ri in range(n_rand): rows.append((q, D, cond, "random", 1, s, a * rand_u[k][ri]))
        base_out = {}
        with torch.no_grad():
            for lo in range(0, len(rows), chunk):
                ch = rows[lo:lo + chunk]
                ms = [meta[x[0]]["m"] for x in ch]; ps = [meta[x[0]]["p"] for x in ch]
                W = max(ps) + 1
                tb = toks_all[ms, :W].to(device)
                ab = (torch.arange(W)[None] <= torch.tensor(ps)[:, None]).long().to(device)
                cb, eb = conn_all[ms].to(device), end_all[ms].to(device)
                delta = graft_delta(tb, ab, cb, eb)
                vec = torch.stack([x[6] if x[6] is not None else torch.zeros(acts[k].shape[1], device=device) for x in ch])
                hold.update(rows=torch.arange(len(ch), device=device), pos=torch.tensor(ps, device=device), vec=vec)
                lg = model(tb, attention_mask=ab, inject_delta=delta, inject_layer=glayer, return_type="logits", padding_mode="right")
                lp = lg[torch.arange(len(ch), device=device), torch.tensor(ps, device=device)].float().log_softmax(-1).cpu()
                for x, l in zip(ch, lp):
                    q, D, cond, name, sign, s, _ = x; r = meta[q]; i = r["m"]
                    nb = nb_of(r["cell"], D); tD = es.coord_to_id[nb]
                    dist = dist_cache[i]; onb = open_nbrs(es.conn[i], r["cell"], n)
                    sp_ids = [es.coord_to_id[c] for c in onb if dist.get(c, 1e9) == dist[r["cell"]] - 1]
                    va_ids = [es.coord_to_id[c] for c in onb if es.coord_to_id[c] not in sp_ids]
                    p = l.exp(); psp = float(p[sp_ids].sum()); pva = float(p[va_ids].sum())
                    m_ = dict(pD=float(p[tD]), lpD=float(l[tD]), psp=psp, pinv=1 - psp - pva, am=int(l.argmax()), tD=tD)
                    if name == "none": base_out[q, D] = m_; continue
                    b = base_out[q, D]
                    recs.append(dict(model=label, site=f"L{k}in", dtype=name, sign=sign, s=s, cond=cond, D=D,
                                     backtrack=int(nb == r["prev"]), dpD=m_["pD"] - b["pD"], dlpD=m_["lpD"] - b["lpD"],
                                     dpsp=m_["psp"] - b["psp"], dpinv=m_["pinv"] - b["pinv"], flip=int(m_["am"] != b["am"]),
                                     toD=int(m_["am"] == tD), base_pD=b["pD"], base_pinv=b["pinv"]))
        handle.remove()
        print(f"  site L{k}in steering done: {len(rows)} runs [{time.time()-t0:.0f}s]", flush=True)

    # ---- aggregate -------------------------------------------------------------------------------
    import pandas as pd
    df = pd.DataFrame(recs); df.to_csv(out / f"steer_raw_{label}.csv.gz", index=False)
    agg = df.groupby(["model", "site", "dtype", "sign", "s", "cond"]).agg(
        n=("dpD", "size"), dpD=("dpD", "mean"), dlpD=("dlpD", "mean"), dpsp=("dpsp", "mean"), dpinv=("dpinv", "mean"),
        flip=("flip", "mean"), toD=("toD", "mean"), base_pD=("base_pD", "mean")).reset_index()
    agg.loc[agg.dtype == "random", "n"] //= n_rand
    agg.to_csv(out / f"steer_agg_{label}.csv", index=False)
    json.dump(dict(model=label, run=run.name, grid_n=n, auroc={f"L{k}in": v for k, v in auroc.items()},
                   proj_sd={f"L{k}in": [float(x) for x in probes[k]["proj_sd"]] for k in site_ks},
                   allo_proj_sd_mean={f"L{k}in": float(probes[k]["allo_proj_sd"].mean()) for k in site_ks},
                   resid_norm={f"L{k}in": probes[k]["xnorm"] for k in site_ks},
                   n_positions={f"{a}_{b}": min(per_cell, len(v)) for (a, b), v in cand.items()}),
              open(out / f"probe_auroc_{label}.json", "w"), indent=1)
    print(f"[{label}] done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
