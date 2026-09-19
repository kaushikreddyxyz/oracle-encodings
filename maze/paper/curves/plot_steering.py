#!/usr/bin/env python3
"""Base-model activation steering vs. the graft on the standard distance-stratified set: small-multiple line chart.

One panel per maze size; x = maze difference (%), y = solved (%), one line per method, route-changed items only.
The identical-maze control is drawn once per model as a hollow marker at 0% (it is the same number for every
base-model method, so it is not repeated per method).
  python plot_steering.py [--graft trainable] [--normalize]
--graft      which graft is the comparison line (default fixed)
--normalize  y = solved as a % of what the SAME model would solve on these items with no prompt/target mismatch. The grafted
             models are slightly weaker maze solvers than the base model at 9x9-10x10, so raw rates understate the graft.
             The no-mismatch rate is estimated per model and seed from its original-maze rollouts, matched on path length
             (solve rate by path-length quantile bin, applied to the items' target-path lengths). For the base model the
             estimate can be checked against the measured prompt-rewrite ceiling: it is 0.4-1.9 points high at every size.
Reads results/distance/steer_out/<method>/*.npz and results/distance/out (grafts); writes
fig_steering_lines_{fixed,trainable}[_normalized].{pdf,svg}, steering_cells.csv (every evaluated method, both grafts) and,
on the default run, steering_items.csv.gz (one row per rollout).
"""
import csv, glob, gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plot_heatmap as PH

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#000000", "#555555", "#e8e7e2"
BASE, GRAFT, SAE_C, CEIL = "#2a7a57", "#000c7a", "#e0a100", "#8a8a8a"
# drawn series: (folder under steer_out | graft:<cond>, label, color, marker, linestyle)
SERIES = [
    ("oracle", "Prompt rewritten (ceiling)", CEIL, None, (0, (4, 2))),
    ("dom_L0_a1", "Steering vectors", BASE, "o", "-"),
    ("sae_L0_sl10.03_n16_v1_sae2", "SAE features", SAE_C, "^", "-"),
]
# evaluated, kept in the CSVs, not drawn
CSV_ONLY = [("none", "Base model, no steering"), ("patch_L0", "Class-mean patch"),
            ("sae_L0_sl10.1_n4_v1", "SAE features, first config (l1 0.1, 4 latents)")]


def load_method(name, sets):
    if name.startswith("graft:"):
        files = sorted(glob.glob(str(HERE / f"results/distance/out/g*__{name.split(':')[1]}__*.npz")))
    else:
        files = sorted(glob.glob(str(HERE / f"results/distance/steer_out/{name}/*.npz")))
    runs = []
    for f in files:
        R = np.load(f); g = int(R["grid_n"])
        assert str(R["set_sha"]) == str(sets[g]["sha"]), f"set mismatch {f}"
        runs.append(dict(grid=g, seed=int(str(R["seed_tag"]).split("-s")[1]), **{k: R[k] for k in ("solved_graft", "solved_token", "reached")}))
    return runs


def own_ceiling(solved0, len0, len_items, nb=12):
    """Length-matched no-mismatch solve rate: this model's original-maze solve rate by path-length quantile bin,
    averaged over the target-path lengths of the given items."""
    edges = np.unique(np.quantile(len0, np.linspace(0, 1, nb + 1))); edges[0] = -1; edges[-1] = 1e9
    which = lambda L: np.clip(np.searchsorted(edges, L, side="right") - 1, 0, len(edges) - 2)
    i0 = which(len0); rate = np.array([solved0[i0 == k].mean() for k in range(len(edges) - 1)])
    return float(rate[which(len_items)].mean())


def cells(sets, runs, mask_fn):
    """-> {(grid, bin): (mean over seeds, sd over seeds, n rollouts, n items, pooled successes, n seeds,
                         mean over seeds of the own-model ceiling estimate, mean over seeds of solved / own ceiling)}"""
    out = {}
    for g, S in sets.items():
        rs = [r for r in runs if r["grid"] == g]
        if not rs: continue
        for lv in range(int(S["n_buckets"]) + 1):
            m = (S["level"] == lv) & mask_fn(S)
            if m.sum() == 0: continue
            per_seed = [r["solved_graft"][m].mean() for r in rs]
            k = int(sum(r["solved_graft"][m].sum() for r in rs)); n = int(m.sum()) * len(rs)
            m0 = S["level"] == 0
            ceil = [own_ceiling(r["solved_graft"][m0], S["len_a"][S["maze"][m0]], S["len_b"][m]) for r in rs]
            out[(g, lv)] = (float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)) if len(rs) > 1 else np.nan, n, int(m.sum()), k, len(rs),
                            float(np.mean(ceil)), float(np.mean([p / c for p, c in zip(per_seed, ceil)])))
    return out


def main(graft="fixed", normalize=False):
    other = "trainable" if graft == "fixed" else "fixed"
    series = SERIES + [(f"graft:{graft}", f"{graft.capitalize()} graft", GRAFT, "s", "-")]
    csv_only = CSV_ONLY + [(f"graft:{other}", f"{other.capitalize()} graft")]
    sets, _ = PH.load()
    sets = PH.merge_unrelated_by_distance(sets)
    w = float(next(iter(sets.values()))["width"]) * 100
    mask = lambda S: (~S["path_same"]) | (S["level"] == 0)
    data = {}
    for name, label, *_ in series + [c + (None,) for c in csv_only]:
        runs = load_method(name, sets)
        assert runs, name
        data[name] = (label, runs, cells(sets, runs, mask))
    grids = sorted(sets)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.985, top=.86, bottom=.105, wspace=.08, hspace=.28)
    for ax, g in zip(axes.flat, grids):
        for name, label, color, marker, ls in series:
            C = data[name][2]
            bins = [lv for lv in range(1, int(sets[g]["n_buckets"]) + 1) if (g, lv) in C and C[(g, lv)][2] >= 1000]
            ax.plot([(lv - .5) * w for lv in bins], [C[(g, lv)][7 if normalize else 0] * 100 for lv in bins], color=color, linestyle=ls, linewidth=1.8,
                    marker=marker, markersize=4.2, markeredgecolor="white", markeredgewidth=.6, zorder=3 if marker else 2, clip_on=False)
        # identical-maze control, once per model: base model (all base-model methods share it) and the grafted model
        if normalize:
            ax.axhline(100, color=INK, linewidth=.8, zorder=1)
        else:
          ax.plot([0], [data["oracle"][2][(g, 0)][0] * 100], marker="o", markersize=5.5, markerfacecolor="white", markeredgecolor=BASE,
                markeredgewidth=1.5, linestyle="none", zorder=4, clip_on=False)
          ax.plot([0], [data[f"graft:{graft}"][2][(g, 0)][0] * 100], marker="s", markersize=5.2, markerfacecolor="white", markeredgecolor=GRAFT,
                markeredgewidth=1.5, linestyle="none", zorder=4, clip_on=False)
        ax.set_title(f"{g}×{g}", loc="left", fontsize=11, weight="bold", color=INK, pad=5)
        ax.set_xlim(-2.5, 61); ax.set_ylim(0, 108 if normalize else 100)
        ax.set_xticks([0, 20, 40, 60]); ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, linewidth=.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(INK); ax.spines[s].set_linewidth(1)
        ax.tick_params(length=3, width=1, labelsize=9.5, colors=INK)
    fig.text(.53, .02, "Maze difference (%)", ha="center", fontsize=11, color=INK)
    fig.text(.012, .48, "Solved, % of the same model's no-mismatch rate" if normalize else "Solved (%)", rotation=90, va="center", fontsize=11, color=INK)
    handles = [Line2D([], [], color=c, linestyle=ls, linewidth=1.8, marker=m, markersize=5, markeredgecolor="white", markeredgewidth=.6) for _, _, c, m, ls in series]
    labels = [lab for _, lab, *_ in series]
    if not normalize:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=5.5, markerfacecolor="white", markeredgecolor=MUTED, markeredgewidth=1.5))
        labels.append("Original maze (no edit)")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.065, .995), ncol=5, frameon=False, fontsize=9.5, handlelength=2.1,
               columnspacing=1.6, handletextpad=.5, labelcolor=INK)
    stem = HERE / (f"fig_steering_lines_{graft}" + ("_normalized" if normalize else ""))
    for ext in ("pdf", "svg"):
        fig.savefig(stem.with_suffix("." + ext), facecolor="white")
    if PREVIEW: fig.savefig(PREVIEW, dpi=140, facecolor="white")
    plt.close(fig)
    print(stem)

    with open(HERE / "steering_cells.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "label", "subset", "grid", "bin", "bin_lo_pct", "bin_hi_pct", "solved_mean_over_seeds", "sd_over_seeds", "pooled_rate",
                     "wilson_lo", "wilson_hi", "n_rollouts", "n_items", "n_seeds", "own_model_no_mismatch_rate_est", "solved_over_own_no_mismatch_rate"])
        for name, (label, runs, C) in data.items():
            for subset, CC in (("route_changed", C), ("all", cells(sets, runs, lambda S: np.ones(len(S["level"]), bool)))):
                for (g, lv), (mu, sd, n, nm, k, ns, ceil, ratio) in sorted(CC.items()):
                    wl, wh = PH.wilson(k, n)
                    wr.writerow([name, label, subset, g, lv, 0 if lv == 0 else round((lv - 1) * w), round(lv * w), round(mu, 4), round(sd, 4),
                                 round(k / n, 4), round(wl, 4), round(wh, 4), n, nm, ns, round(ceil, 4), round(ratio, 4)])
    if graft != "fixed" or normalize: return                       # the per-rollout table does not depend on the comparison line
    with gzip.open(HERE / "steering_items.csv.gz", "wt", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "grid", "seed", "maze", "bin", "hamming", "hamming_frac", "route_same", "solved_target_maze", "solved_prompt_maze", "reached"])
        for name, (label, runs, C) in data.items():
            if name.startswith("graft:"): continue          # already in distance_items.csv.gz
            for r in runs:
                S = sets[r["grid"]]; nf = int(S["n_free"])
                rows = np.column_stack([S["maze"], S["level"], S["dist"], S["path_same"].astype(int), r["solved_graft"].astype(int),
                                        r["solved_token"].astype(int), r["reached"].astype(int)])
                for mz, lv, d, same, sg, st, rc in rows.tolist():
                    wr.writerow([name, r["grid"], r["seed"], mz, lv, d, round(d / nf, 4), same, sg, st, rc])


PREVIEW = None
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graft", choices=["fixed", "trainable"], default="fixed")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--preview", default=None, help="optional raster path for a quick look (not part of the figure set)")
    args = ap.parse_args(); PREVIEW = args.preview
    main(args.graft, args.normalize)
