#!/usr/bin/env python3
"""Local: collate the pulled pod outputs into flat tables, apply the recorded selection functions, compute intervals.

  python collect.py [results_root]
Writes (next to this file): sae_metrics.csv, probe_metrics.csv, selection.json, alignment_values.csv.gz (per-connection
|cosine| of every selected entry), alignment_summary.csv, intervention_models.csv, intervention_cells.csv,
intervention_summary.csv, verify_summary.csv.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from selection import select_sae, select_probe, FAMILIES

HERE = Path(__file__).resolve().parent; ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE
SETS = HERE.parents[1] / "curves/results/distance/sets"
KINDS = ("relu", "topk", "btopk", "mat", "mp")
man = {m["tag"]: m for m in json.load(open(HERE / "manifest.json"))}


def jload(p):
    try: return json.loads(Path(p).read_text())
    except json.JSONDecodeError: print("skipping unreadable", p, file=sys.stderr); return None


def sae_rows(tag, site):
    d = ROOT / tag / f"sae-{site}"
    rows = [r for r in (jload(p) for p in sorted(d.glob("*_s[0-9]*.json"))) if r] if d.exists() else []
    return [r for r in rows if "error" not in r], [r for r in rows if "error" in r]


def boot_ci(per_maze_by_group, n=2000, seed=53):
    """per_maze_by_group: list (one per grid) of arrays [n_base_mazes, 2] = (sum, count) pooled over the seeds of that grid.
    Resample base mazes within each grid; statistic = mean over grids of sum/count."""
    rng = np.random.default_rng(seed); out = np.empty(n)
    for b in range(n):
        out[b] = np.mean([(lambda z: z[:, 0].sum() / z[:, 1].sum())(g[rng.integers(0, len(g), len(g))]) for g in per_maze_by_group])
    return np.quantile(out, [.025, .975])


def main():
    sae_all, errors, probe_all, sel, align, verify = [], [], [], {}, [], []
    for tag, m in man.items():
        base = dict(tag=tag, grid=m["grid"], seed=m["seed"], cond=m["cond"]); sel[tag] = {}
        for site in ("injection", "post0"):
            rows, err = sae_rows(tag, site); errors += err
            if not rows: continue
            sae_all += [{**{k: v for k, v in r.items() if k not in ("curve", "seed")}, "sae_seed": r["seed"], **base} for r in rows]
            picks = {"sae_fair": select_sae(rows, "fair"), "sae_steelman": select_sae(rows, "steelman")}
            for kind in KINDS:                                   # per-kind detail (supplementary)
                kr = [r for r in rows if r["kind"] == kind]
                if kr: picks[f"sae_fair_{kind}"] = select_sae(kr, "fair") if any(r["variant"] in ("ctr", "raw") for r in kr) else None; picks[f"sae_steelman_{kind}"] = select_sae(kr, "steelman")
            unsup = [r for r in rows if r["variant"] in ("ctr", "raw")]      # steelman without the coding-aware reference point
            if unsup: picks["sae_steelman_nocoding"] = select_sae(unsup, "steelman")
            for name, pk in picks.items():
                if pk is None: continue
                job, fam, info = pk; sel[tag][f"{site}|{name}"] = info
                q = np.load(ROOT / tag / f"sae-{site}" / f"{job}.npz")
                for e, (a, b) in enumerate(zip(q[f"{fam}_raw"].astype(float), q[f"{fam}_id"].astype(float))): align.append({**base, "site": site, "entry": name, "edge": e, "cos_raw": a, "cos_id": b})
        pj = ROOT / tag / "probes.json"
        if pj.exists():
            prow = jload(pj)
            if prow is None: continue
            prow = prow["rows"]; probe_all += [{**base, **r} for r in prow]; pe = np.load(ROOT / tag / "probes_per_edge.npz"); nmax = max(r["n_fit"] for r in prow)
            for site in ("injection", "post0"):
                names = {"probe_fair": select_probe(prow, site, "fair"), "probe_steelman": select_probe(prow, site, "steelman"),
                         "probe_pinv": select_probe(prow, site, "steelman", families=("pinv",)),
                         "difference_means": ("difference_means", {}), "generative_ols": ("generative_ols", {})}
                for fam in ("ridge", "logistic"): names[f"probe_steelman_{fam}"] = select_probe(prow, site, "steelman", families=(fam,))
                for name, (meth, info) in names.items():
                    if info: sel[tag][f"{site}|{name}"] = info
                    for e, (a, b) in enumerate(zip(pe[f"{site}|{meth}|{nmax}|raw"].astype(float), pe[f"{site}|{meth}|{nmax}|id"].astype(float))): align.append({**base, "site": site, "entry": name, "edge": e, "cos_raw": a, "cos_id": b})
        vj = ROOT / tag / "verify.json"
        if vj.exists() and (v := jload(vj)): verify.append({**base, **{k: x for k, x in v.items() if k != "pool"}, "pool_sha1": v["pool"]["sha1"], "pool_unique_fit": v["pool"]["unique_fit"]})
    if sae_all: pd.DataFrame(sae_all).to_csv(OUT / "sae_metrics.csv", index=False)
    if probe_all: pd.DataFrame(probe_all).to_csv(OUT / "probe_metrics.csv", index=False)
    if verify: pd.DataFrame(verify).to_csv(OUT / "verify_summary.csv", index=False)
    json.dump(dict(gauge="identifiable (zero-mean over free connections)", families=FAMILIES, per_model=sel, sae_errors=[e["job"] + "@" + e.get("tag", "") for e in errors]), open(OUT / "selection.json", "w"), indent=1)
    A = pd.DataFrame(align)
    if len(A):
        A.to_csv(OUT / "alignment_values.csv.gz", index=False, compression="gzip")
        pm = A.groupby(["cond", "site", "entry", "grid", "seed"]).agg(cos_id=("cos_id", "mean"), cos_raw=("cos_raw", "mean"), frac08=("cos_id", lambda x: (x >= .8).mean())).reset_index()
        rows = []
        for (cond, site, entry), g in pm.groupby(["cond", "site", "entry"]):
            ps = g.groupby("seed")[["cos_id", "cos_raw", "frac08"]].mean()         # equal weight per grid, then per seed
            rows.append(dict(cond=cond, site=site, entry=entry, n_models=len(g), n_seeds=len(ps), cos_id=ps.cos_id.mean(), cos_id_sd_seeds=ps.cos_id.std(ddof=1) if len(ps) > 1 else np.nan,
                             cos_raw=ps.cos_raw.mean(), frac08=ps.frac08.mean(), min_model=g.cos_id.min(), max_model=g.cos_id.max()))
        pd.DataFrame(rows).to_csv(OUT / "alignment_summary.csv", index=False); pm.to_csv(OUT / "alignment_models.csv", index=False)
    interventions()


def interventions():
    models, cells, items = [], [], {}
    for tag, m in man.items():
        d = ROOT / tag / "intervene"
        if not d.exists(): continue
        S = np.load(SETS / f"g{m['grid']}.npz"); rc = ~S["path_same"]; lvl = S["level"]; maze = S["maze"]
        for p in sorted(d.glob("*.json")):
            j = jload(p)
            if j is None or not p.with_suffix(".npz").exists(): continue
            sb = np.load(p.with_suffix(".npz"))["solved_b"]; key = j["key"]
            models.append(dict(tag=tag, grid=m["grid"], seed=m["seed"], cond=m["cond"], method=key, alpha=j["alpha"], n_items=len(sb), n_route_changed=int(rc.sum()),
                               solved_all=sb.mean(), solved_route_changed=sb[rc].mean(), solved_same_maze=sb[lvl == 0].mean(), selection=json.dumps(j.get("selection", {}))))
            for lv in np.unique(lvl):
                k = rc & (lvl == lv)
                if k.sum(): cells.append(dict(tag=tag, grid=m["grid"], seed=m["seed"], cond=m["cond"], method=key, level=int(lv), n=int(k.sum()), solved=int(sb[k].sum())))
            items[(m["cond"], m["grid"], m["seed"], key)] = (sb, rc, maze)
    if not models: return
    M = pd.DataFrame(models); M.to_csv(OUT / "intervention_models.csv", index=False); pd.DataFrame(cells).to_csv(OUT / "intervention_cells.csv", index=False)
    rows = []
    for (cond, method), g in M.groupby(["cond", "method"]):
        ps = g.groupby("seed").solved_route_changed.mean()                                  # mean over grids within seed
        grids = sorted(g.grid.unique()); per_grid = []; per_grid_delta = []
        for gr in grids:
            acc = None; accd = None
            for s in sorted(g[g.grid == gr].seed.unique()):
                sb, rc, maze = items[(cond, gr, s, method)]; ref = items.get((cond, gr, s, "injection|graft"))
                z = np.stack([np.bincount(maze[rc], weights=sb[rc], minlength=3000), np.bincount(maze[rc], minlength=3000)], 1); acc = z if acc is None else acc + z
                if ref is not None:
                    dz = np.stack([np.bincount(maze[rc], weights=sb[rc].astype(float) - ref[0][rc], minlength=3000), np.bincount(maze[rc], minlength=3000)], 1); accd = dz if accd is None else accd + dz
            per_grid.append(acc[acc[:, 1] > 0]);
            if accd is not None: per_grid_delta.append(accd[accd[:, 1] > 0])
        lo, hi = boot_ci(per_grid); row = dict(cond=cond, method=method, n_models=len(g), n_seeds=len(ps), grids=len(grids), route_changed=ps.mean(), sd_seeds=ps.std(ddof=1) if len(ps) > 1 else np.nan, boot_lo=lo, boot_hi=hi,
                                               n_rollouts=int(sum(z[:, 1].sum() for z in per_grid)))
        if per_grid_delta and len(per_grid_delta) == len(grids):
            gref = M[(M.cond == cond) & (M.method == "injection|graft")].set_index(["grid", "seed"]).solved_route_changed
            dl = g.set_index(["grid", "seed"]).solved_route_changed - gref; dps = dl.dropna().groupby("seed").mean()
            dlo, dhi = boot_ci(per_grid_delta); row.update(delta_vs_graft=dps.mean(), delta_sd_seeds=dps.std(ddof=1) if len(dps) > 1 else np.nan, delta_lo=dlo, delta_hi=dhi)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "intervention_summary.csv", index=False)
    # check of the decoder against the distance campaign (same set, same n^2+2 limit, graft replacement)
    chk = []
    for (cond, gr, s, key), (sb, rc, maze) in items.items():
        if key != "injection|graft": continue
        p = SETS.parent / "out" / f"g{gr}__{cond}__g{gr}-s{s}.npz"
        if p.exists(): o = np.load(p)["solved_graft"]; chk.append(dict(cond=cond, grid=gr, seed=s, agree=float((o == sb).mean()), here=float(sb.mean()), distance_campaign=float(o.mean())))
    if chk: pd.DataFrame(chk).to_csv(OUT / "graft_control_vs_distance_campaign.csv", index=False)


if __name__ == "__main__":
    main()
