#!/usr/bin/env python3
"""Markdown tables for the scale-up section of REPORT.md, from ev_summary.csv / ev_probes.csv (written by figures_ev.py).
Prints: (1) final metrics per grid x code (seed mean +- SD, n); (2) 4 vs 8 epochs at 9x9/10x10 per seed;
(3) probe R^2 (linear / MLP) at layers 0, 6, 12 per code, pooled and per grid at the input; (4) ablations pooled."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = list(csv.DictReader(open(HERE / "ev_summary.csv"))); P = list(csv.DictReader(open(HERE / "ev_probes.csv")))
f = lambda r, k: float(r[k]) if r.get(k, "") not in ("", None) else np.nan
def ms(v, d=2):
    v = [x for x in v if not np.isnan(x)]
    return "-" if not v else f"{np.mean(v):.{d}f} ± {np.std(v):.{d}f} (n={len(v)})" if len(v) > 1 else f"{v[0]:.{d}f} (n=1)"
grids = sorted({int(r["grid"]) for r in S})
print("### Final metrics, 4 epochs (seed mean ± SD)\n")
print("| grid | code | true solved | phantom solved | route-changed: graft solve | one-swap p(new) | crosses new wall |\n|---|---|---|---|---|---|---|")
for g in grids:
    for c in ("base", "edge", "ego"):
        R = [r for r in S if int(r["grid"]) == g and r["code"] == c and r["epochs"] == "4"]
        if not R: continue
        print(f"| {g}×{g} | {c} | {ms([f(r,'true_solved') for r in R])} | {ms([f(r,'phantom_solved') for r in R])} | {ms([f(r,'route_changed_graft') for r in R])} | {ms([f(r,'p_new') for r in R])} | {ms([f(r,'crossed_wall') for r in R])} |")
if any(r["epochs"] == "8" for r in S):
    print("\n### 4 vs 8 epochs at the largest grids (per seed: true solved / phantom solved / route-changed graft solve / p(new))\n")
    print("| grid | code | seed | 4 epochs | 8 epochs |\n|---|---|---|---|---|")
    for g in grids:
        for c in ("edge", "ego"):
            for s in sorted({int(r["seed"]) for r in S if int(r["grid"]) == g and r["code"] == c and r["epochs"] == "8"}):
                cell = lambda e: next((f"{f(r,'true_solved'):.2f} / {f(r,'phantom_solved'):.2f} / {f(r,'route_changed_graft'):.2f} / {f(r,'p_new'):.2f}" for r in S if int(r["grid"]) == g and r["code"] == c and int(r["seed"]) == s and r["epochs"] == e), "-")
                print(f"| {g}×{g} | {c} | {s} | {cell('4')} | {cell('8')} |")
for metric, title in (("auroc", "AUROC (threshold-free, chance 0.5)"), ("r2", "R² of the ±1 targets")):
    if all(np.isnan(f(r, metric)) for r in P): continue
    print(f"\n### Probes: {title} of wall targets at path tokens (linear ridge / MLP), pooled over grids and seeds\n")
    print("| model | target | layer 0 (input) | layer 3 | layer 6 | layer 9 | layer 12 |\n|---|---|---|---|---|---|---|")
    for c in ("base", "edge", "ego"):
        for fam, name in (("allo", "allocentric walls"), ("ego", "egocentric walls (±2)")):
            cells = []
            for L in (0, 3, 6, 9, 12):
                lin = [f(r, metric) for r in P if r["code"] == c and r["family"] == fam and r["probe"] == "linear" and int(r["layer"]) == L]
                mlp = [f(r, metric) for r in P if r["code"] == c and r["family"] == fam and r["probe"] == "mlp" and int(r["layer"]) == L]
                cells.append(f"{np.nanmean(lin):.2f} / {np.nanmean(mlp):.2f}" if lin else "-")
            print(f"| {c} | {name} | " + " | ".join(cells) + " |")
print("\n### Probes at the block-0 input, per grid (linear / MLP AUROC)\n")
print("| grid | edge: allo | edge: ego | ego: allo | ego: ego | base: allo | base: ego |\n|---|---|---|---|---|---|---|")
for g in grids:
    cells = []
    for c in ("edge", "ego", "base"):
        for fam in ("allo", "ego"):
            lin = [f(r, "auroc") for r in P if r["code"] == c and r["family"] == fam and r["probe"] == "linear" and int(r["layer"]) == 0 and int(r["grid"]) == g]
            mlp = [f(r, "auroc") for r in P if r["code"] == c and r["family"] == fam and r["probe"] == "mlp" and int(r["layer"]) == 0 and int(r["grid"]) == g]
            cells.append(f"{np.nanmean(lin):.2f} / {np.nanmean(mlp):.2f}" if lin else "-")
    print(f"| {g}×{g} | " + " | ".join(cells) + " |")
print("\n### Inference-time ablations, true world, pooled over all 4-epoch grafted models: solved / p(invalid move)\n")
conds = [("full", "full"), ("graft_off", "graft removed"), ("noise", "graft → noise"), ("off_gen", "off during generation"), ("off_prompt", "off during prompt"), ("x0.5", "×0.5"), ("x2", "×2"), ("prompt_blank", "graft only (prompt blank)"), ("shuffled", "other maze's graft")]
print("| code | " + " | ".join(n for _, n in conds) + " |\n|---|" + "---|" * len(conds))
for c in ("edge", "ego"):
    R = [r for r in S if r["code"] == c and r["epochs"] == "4"]
    print(f"| {c} (n={len(R)}) | " + " | ".join(f"{np.nanmean([f(r,f'abl_{k}_solved') for r in R]):.2f} / {np.nanmean([f(r,f'abl_{k}_p_invalid') for r in R]):.2f}" for k, _ in conds) + " |")
