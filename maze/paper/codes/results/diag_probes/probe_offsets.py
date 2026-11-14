#!/usr/bin/env python3
"""Where does cross-frame / base-model wall readability come from?  Reads results/probe_v2/*.json only.

Writes (this folder): ego_offsets_{linear,mlp}.csv (per offset x model x site, mean over seeds then grids),
ego_groups.csv (immediate / other ring-1 / ring-2 per model x site x probe x grid), allo_by_location.csv,
allo_rowcol_L6.csv, allo_concentration.csv, anomalies.txt.  Figure: ../../fig_ev_probe_offsets.{pdf,svg} via plot().
"""
import csv, json, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; CODES = HERE.parents[1]
sys.path.insert(0, str(CODES))
GRIDS = range(5, 11); SITES = ["L0in"] + [f"L{k}" for k in range(1, 13)]; CODES_ = ["base", "edge", "ego"]
R = 2; OFF = [(k, dr, dc) for k in (0, 1) for dr in range(-R, R + 1) for dc in range(-R, R + 1)]
IMM = {(0, 0, 0): "S", (0, -1, 0): "N", (1, 0, 0): "E", (1, 0, -1): "W"}


def group(o):
    k, dr, dc = o
    if o in IMM: return "immediate"
    return "ring2" if max(abs(dr), abs(dc)) == 2 else "ring1_other"


def free_idx(n):
    keep = []
    for f in range(2 * n * n):
        k, rc = divmod(f, n * n); r, c = divmod(rc, n)
        if (k == 0 and r == n - 1) or (k == 1 and c == n - 1): continue
        keep.append(f)
    return np.array(keep)


def load():
    D = {}; dropped = 0
    for f in sorted((CODES / "results/probe_v2").glob("*.json")):
        m = re.match(r"g(\d+)-s(\d+)-(base|edge|ego)-e4-oe", f.stem); g, s, c = int(m[1]), int(m[2]), m[3]
        d = json.load(open(f))
        for site, S in d["sites"].items():
            for key, v in S.items():
                a = np.array(v["per_target_auroc"], float)
                if len(a) != len(v["per_target_acc"]): dropped += 1; a = None
                D[(c, g, s, site, key)] = dict(auroc=a, r2=np.array(v["per_target_r2"]), acc=np.array(v["per_target_acc"]), pooled=v["auroc"], frac_open=v["frac_open"])
    return D, dropped


def seedmean(D, c, g, site, key, what="auroc"):
    xs = [D[(c, g, s, site, key)][what] for s in range(5) if (c, g, s, site, key) in D and D[(c, g, s, site, key)][what] is not None]
    return np.mean(xs, 0), len(xs)


def main():
    D, dropped = load(); print("files x site x key with truncated per_target_auroc:", dropped)
    # ---------- 1. ego family per offset
    for probe in ("linear", "mlp"):
        with open(HERE / f"ego_offsets_{probe}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["model", "site", "grid", "kind", "dr", "dc", "group", "auroc", "n_seeds"])
            for c in CODES_:
                for site in SITES:
                    per = []
                    for g in GRIDS:
                        a, ns = seedmean(D, c, g, site, f"ego_{probe}"); per.append(a)
                        for o, v in zip(OFF, a): w.writerow([c, site, g, "down" if o[0] == 0 else "right", o[1], o[2], group(o), f"{v:.4f}", ns])
                    a = np.mean(per, 0)
                    for o, v in zip(OFF, a): w.writerow([c, site, "mean", "down" if o[0] == 0 else "right", o[1], o[2], group(o), f"{v:.4f}", ""])
    grp = {g_: [i for i, o in enumerate(OFF) if group(o) == g_] for g_ in ("immediate", "ring1_other", "ring2")}
    with open(HERE / "ego_groups.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["probe", "model", "site", "grid", "immediate", "ring1_other", "ring2", "all50", "S", "N", "E", "W", "down_minus_right_mean"])
        for probe in ("linear", "mlp"):
            for c in CODES_:
                for site in SITES:
                    rows = []
                    for g in list(GRIDS) + ["mean"]:
                        if g == "mean": a = np.mean([seedmean(D, c, gg, site, f"ego_{probe}")[0] for gg in GRIDS], 0)
                        else: a = seedmean(D, c, g, site, f"ego_{probe}")[0]
                        imm = {IMM[o]: a[OFF.index(o)] for o in IMM}
                        w.writerow([probe, c, site, g] + [f"{a[grp[k]].mean():.4f}" for k in grp] + [f"{a.mean():.4f}"] + [f"{imm[k]:.4f}" for k in "SNEW"]
                                   + [f"{a[:25].mean() - a[25:].mean():.4f}"])
    # ---------- 2. allo family by location
    loc_rows, rc_rows, conc_rows = [], [], []
    for probe in ("linear", "mlp"):
        for c in CODES_:
            for site in SITES:
                for g in GRIDS:
                    n = g; fi = free_idx(n); a, ns = seedmean(D, c, g, site, f"allo_{probe}")
                    kind = fi // (n * n); r = (fi % (n * n)) // n; cc = fi % n
                    r2 = r + (kind == 0); c2 = cc + (kind == 1)       # second cell of the connection
                    ring = np.minimum(np.minimum.reduce([r, cc, n - 1 - r, n - 1 - cc]), np.minimum.reduce([r2, c2, n - 1 - r2, n - 1 - c2]))
                    # corner-adjacent: one of the two cells is a grid corner
                    corner = (((r == 0) | (r == n - 1)) & ((cc == 0) | (cc == n - 1))) | (((r2 == 0) | (r2 == n - 1)) & ((c2 == 0) | (c2 == n - 1)))
                    loc_rows.append([probe, c, site, g, f"{a.mean():.4f}", f"{a[ring == 0].mean():.4f}", f"{a[ring >= 1].mean():.4f}", f"{a[corner].mean():.4f}",
                                     f"{a[(ring == 0) & ~corner].mean():.4f}", int((ring == 0).sum()), int((ring >= 1).sum()), int(corner.sum())])
                    ex = np.sort(a - 0.5)[::-1].clip(min=0); tot = ex.sum()
                    k10 = max(1, int(round(0.1 * len(a))))
                    conc_rows.append([probe, c, site, g, len(a), f"{a.mean():.4f}", f"{np.median(a):.4f}", f"{(a > 0.6).mean():.3f}", f"{(a > 0.7).mean():.3f}",
                                      f"{ex[:k10].sum() / tot if tot > 0 else float('nan'):.3f}", f"{a.min():.4f}", f"{a.max():.4f}"])
                    if site in ("L0in", "L6", "L12") and probe == "linear":
                        for j in range(len(fi)):
                            rc_rows.append([c, site, g, "down" if kind[j] == 0 else "right", r[j], cc[j], ring[j], int(corner[j]), f"{a[j]:.4f}"])
    with open(HERE / "allo_by_location.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["probe", "model", "site", "grid", "all", "ring0_boundary", "interior", "corner_adjacent", "ring0_noncorner", "n_ring0", "n_interior", "n_corner"]); w.writerows(loc_rows)
    with open(HERE / "allo_per_wall.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "site", "grid", "kind", "r", "c", "ring", "corner_adjacent", "auroc_seedmean"]); w.writerows(rc_rows)
    with open(HERE / "allo_concentration.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["probe", "model", "site", "grid", "n_targets", "mean", "median", "frac_gt_0.6", "frac_gt_0.7", "share_of_excess_in_top10pct", "min", "max"]); w.writerows(conc_rows)
    # ---------- anomaly scan: out-of-range values, pooled != macro mean, duplicated (non-saturated) per-target arrays
    import collections
    with open(HERE / "anomalies.txt", "w") as fh:
        fh.write(f"entries with len(per_target_auroc) != n_targets: {dropped}\n")
        low = [(k, j_, round(v["auroc"][j_], 3)) for k, v in D.items() for j_ in np.where(v["auroc"] < 0.47)[0]]
        fh.write(f"per-target AUROC < 0.47: {len(low)} {low[:20]}\n")
        mism = [k for k, v in D.items() if abs(v["auroc"].mean() - v["pooled"]) > 1e-6]
        fh.write(f"pooled auroc != mean(per_target_auroc): {len(mism)} {mism[:20]}\n")
        h = collections.defaultdict(list)
        for k, v in D.items():
            if v["auroc"].min() < 0.999: h[(k[3], k[4], np.round(v["auroc"], 6).tobytes())].append(k)
        dup = [v for v in h.values() if len(v) > 1]
        fh.write(f"identical non-saturated per-target arrays across models: {len(dup)} {dup[:10]}\n")
        neg = [(k, round(float(v["r2"].min()), 3)) for k, v in D.items() if v["r2"].min() < -0.05]
        fh.write(f"per-target R2 < -0.05: {len(neg)} {neg}\n")
        sat = collections.Counter((k[0], k[4]) for k, v in D.items() for _ in np.where(v["auroc"] > 0.99)[0]
                                  if not ((k[0] == "edge" and k[4].startswith("allo")) or (k[0] == "ego" and k[4].startswith("ego"))))
        fh.write(f"cross-frame / base-model per-target AUROC > 0.99 (count of model x site x target): {dict(sat)}\n")

if __name__ == "__main__":
    main()
