#!/usr/bin/env python3
"""Aggregate per-run rollout outcomes + the distance sets into long CSVs and heatmaps.

  python plot_heatmap.py            (reads results/distance/out/*.npz and results/distance/sets/g*.npz)
Outputs: distance_items.csv.gz (one row per rollout), distance_cells.csv (per grid x cond x level),
         fig_distance_heatmap_routechanged.{pdf,svg}
"""
import glob, gzip, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
INK, MUTED, RULE = "#000000", "#000000", "#e1e0d9"
GRAFT = "#000c7a"
CMAP = LinearSegmentedColormap.from_list("yellow_red", ["#fcba02", "#dc2625"])
CONDS = [("fixed", "Grafted model, fixed $W_{inj}$"), ("trainable", "Grafted model, trainable $W_{inj}$")]


def load():
    sets = {}
    for f in glob.glob(str(HERE / "results/distance/sets/g*.npz")):
        S = np.load(f); sets[int(S["grid_n"])] = {k: S[k] for k in ("maze", "level", "dist", "path_same", "len_a", "len_b", "sha", "n_free", "n_buckets", "width")}
    runs = []
    for f in sorted(glob.glob(str(HERE / "results/distance/out/*.npz"))):
        R = np.load(f); g = int(R["grid_n"])
        assert str(R["set_sha"]) == str(sets[g]["sha"]), f"set mismatch {f}"
        runs.append(dict(grid=g, cond=str(R["cond"]), seed=int(str(R["seed_tag"]).split("-s")[1]),
                         **{k: R[k] for k in ("solved_graft", "solved_token", "reached", "n_bad_graft", "n_bad_token")}))
    return sets, runs


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cells(sets, runs, mask_fn=None, key="solved_graft"):
    """-> dict[(cond, grid, level)] = (mean over seeds of per-seed rate, sd over seeds, n rollouts, n mazes, pooled k)"""
    out = {}
    for cond, _ in CONDS:
        for g, S in sets.items():
            rs = [r for r in runs if r["cond"] == cond and r["grid"] == g]
            if not rs: continue
            for lv in range(int(S["n_buckets"]) + 2):
                m = S["level"] == lv
                if mask_fn is not None: m = m & mask_fn(S)
                if m.sum() == 0: continue
                per_seed = [r[key][m].mean() for r in rs]
                k = int(sum(r[key][m].sum() for r in rs)); n = int(m.sum()) * len(rs)
                out[(cond, g, lv)] = (float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)) if len(rs) > 1 else np.nan,
                                      n, int(m.sum()), k, len(rs))
    return out


def merge_unrelated_by_distance(sets):
    """Put independently sampled mazes into their measured distance bins.

    Level zero stays reserved for identical mazes. Positive bins are (lo, hi].
    Use a shared range covering every item, including the former fresh column.
    """
    width = float(next(iter(sets.values()))["width"])
    max_bins = max(int(np.ceil(np.max(s["dist"] / int(s["n_free"])) / width - 1e-9))
                   for s in sets.values())
    merged = {}
    for g, original in sets.items():
        s = dict(original)
        levels = original["level"].copy()
        fresh = levels == int(original["n_buckets"]) + 1
        distance_bins = np.ceil(original["dist"] / int(original["n_free"]) / width - 1e-9).astype(int)
        levels[fresh] = distance_bins[fresh]
        assert np.all((levels >= 0) & (levels <= max_bins))
        s["level"] = levels
        s["n_buckets"] = np.array(max_bins)
        merged[g] = s
    return merged


def heatmap(C, sets, title, fname, note, relative=False):
    """Fig1 styling; cell values and color share a linear 0–100% scale."""
    grids = sorted(sets)
    K = max(lv for (_, _, lv), values in C.items() if values[2] >= 1000)
    w = float(sets[grids[0]]["width"])
    cols = list(range(K + 1))
    gap = .22
    centers = [0] + [j+gap for j in range(1, K+1)]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.05), sharey=True)
    fig.subplots_adjust(left=.075, right=.919, top=.84, bottom=.20, wspace=.06)
    for ax, (cond, _) in zip(axes, CONDS):
        M = np.full((len(grids), len(cols)), np.nan)
        for i, g in enumerate(grids):
            for j, lv in enumerate(cols):
                if (cond, g, lv) in C and C[(cond, g, lv)][2] >= 1000:
                    M[i, j] = C[(cond, g, lv)][0] / (C[(cond, g, 0)][0] if relative else 1.0)
        # Only the identical-maze control is separated from positive distances.
        blocks = [(M[:, :1], -.5, .5),
                  (M[:, 1:K+1], .5+gap, K+.5+gap)]
        for values, left, right in blocks:
            im = ax.imshow(values, cmap=CMAP, vmin=0, vmax=1, aspect="auto",
                           interpolation="nearest", extent=(left, right, len(grids)-.5, -.5))
            ax.add_patch(Rectangle((left, -.5), right-left, len(grids),
                                   fill=False, edgecolor=INK, linewidth=1.8,
                                   clip_on=False, zorder=5))
        ax.set_xlim(-.5, K+.5+gap)
        ax.set_ylim(len(grids)-.5, -.5)
        for i in range(len(grids)):
            for j in range(len(cols)):
                value = M[i, j]
                if np.isfinite(value):
                    ax.text(centers[j], i, f"{value*100:.0f}", ha="center", va="center", fontsize=9,
                            color=INK)
                else:
                    ax.text(centers[j], i, "–", ha="center", va="center", fontsize=9, color=INK)
        # Org identifies the control; numeric ticks mark main-heatmap bin edges.
        edge_bins = list(range(0, K+1, 2))
        if edge_bins[-1] != K:
            edge_bins.append(K)
        ax.set_xticks([.5+gap+k for k in edge_bins],
                      [f"{round(k*w*100):.0f}" for k in edge_bins],
                      fontsize=10, color=INK)
        ax.text(0, 1.025, "Origin", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=INK)
        # Unlabelled edges expose every 5% bin, including the start of (0, 5].
        minor_edges = [.5+gap+k for k in range(K+1) if k not in edge_bins]
        ax.set_xticks(minor_edges, minor=True)
        ax.tick_params(axis="x", which="minor", direction="out", length=3,
                       width=1, color=INK, bottom=True, top=False)
        ax.set_yticks(range(len(grids)), [f"{g}×{g}" for g in grids], fontsize=10, color=MUTED)
        ax.set_title("Fixed Graft" if cond == "fixed" else "Trainable Graft",
                     fontsize=14, color=INK, weight="bold", pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
            spine.set_linewidth(2)
            spine.set_edgecolor(INK)
        ax.tick_params(axis="both", direction="out", length=4, width=1.2, color=INK, pad=5, top=False, right=False)
        ax.set_xlabel("Maze difference (%)", fontsize=11, color=INK, labelpad=8)
    axes[0].set_ylabel("Maze Size", fontsize=11, color=INK, labelpad=0)
    cax = fig.add_axes([.941, .20, .012, .64])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=np.linspace(0, 1, 6))
    cb.ax.set_yticklabels(["0", "20", "40", "60", "80", "100"], fontsize=10, color=INK)
    cb.ax.tick_params(direction="out", length=4, width=1, color=INK, pad=5)
    cb.outline.set_edgecolor(INK)
    cb.outline.set_linewidth(1.5)
    cb.ax.set_title("Solved\n(%)" if not relative else "Relative\nsolve rate (%)",
                    fontsize=10, pad=9, color=INK)
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / f"{fname}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    sets, runs = load()
    print(len(sets), "sets,", len(runs), "runs")
    C = cells(sets, runs)
    with open(HERE / "distance_cells.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cond", "grid", "level", "bucket_lo_pct", "bucket_hi_pct", "mean_raw_hamming", "solved_mean_over_seeds", "sd_over_seeds",
                     "pooled_rate", "wilson_lo", "wilson_hi", "n_rollouts", "n_mazes", "n_seeds", "solved_vs_token_maze", "frac_route_changed"])
        Ct = cells(sets, runs, key="solved_token")
        for (cond, g, lv), (mu, sd, n, nm, k, ns) in sorted(C.items()):
            S = sets[g]; K = int(S["n_buckets"]); w = float(S["width"]); m = S["level"] == lv
            lo, hi = (0, 0) if lv == 0 else ((lv - 1) * w * 100, lv * w * 100) if lv <= K else (np.nan, np.nan)
            wl, wh = wilson(k, n)
            wr.writerow([cond, g, "fresh" if lv == K + 1 else lv, lo, hi, round(float(S["dist"][m].mean()), 2), round(mu, 4), round(sd, 4), round(k / n, 4),
                         round(wl, 4), round(wh, 4), n, nm, ns, round(Ct[(cond, g, lv)][0], 4), round(float(1 - S["path_same"][m].mean()), 4)])
    with gzip.open(HERE / "distance_items.csv.gz", "wt", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["grid", "cond", "seed", "maze", "level", "hamming", "hamming_frac", "route_same", "len_token_maze", "len_graft_maze",
                     "solved_graft", "solved_token", "reached", "n_bad_graft", "n_bad_token"])
        for r in runs:
            S = sets[r["grid"]]; nf = int(S["n_free"])
            for j in range(len(S["maze"])):
                wr.writerow([r["grid"], r["cond"], r["seed"], int(S["maze"][j]), int(S["level"][j]), int(S["dist"][j]), round(int(S["dist"][j]) / nf, 4),
                             int(S["path_same"][j]), int(S["len_a"][S["maze"][j]]), int(S["len_b"][j]), int(r["solved_graft"][j]), int(r["solved_token"][j]),
                             int(r["reached"][j]), int(r["n_bad_graft"][j]), int(r["n_bad_token"][j])])
    nm = max(c[3] for c in C.values()); ns = max(c[5] for c in C.values())
    base_note = (f"Each cell: {nm:,} token-space mazes × {ns} model seeds (final checkpoints), greedy rollout from the token-space origin to target. Graft-space mazes are DFS trees "
                 "built by a DFS that prefers the token maze's connections;\n'fresh DFS maze' = an independent random-DFS maze (typically 40–50% away). Boundary features excluded from the distance.")
    plot_sets = merge_unrelated_by_distance(sets)
    C2 = cells(plot_sets, runs, mask_fn=lambda S: (~S["path_same"]) | (S["level"] == 0))
    heatmap(C2, plot_sets, "Same, restricted to items whose origin→target route differs between the two mazes (column 0 unrestricted)",
            "fig_distance_heatmap_routechanged", base_note)
    for cond, _ in CONDS:
        print(cond, "route-changed only")
        for g in sorted(sets):
            print(" ", g, " ".join(f"{C2[(cond, g, lv)][0]:.2f}" if (cond, g, lv) in C2 else " -- " for lv in range(int(sets[g]['n_buckets']) + 2)))
    for cond, _ in CONDS:
        print(cond)
        for g in sorted(sets):
            print(" ", g, " ".join(f"{C[(cond, g, lv)][0]:.2f}" if (cond, g, lv) in C else " -- " for lv in range(int(sets[g]['n_buckets']) + 2)))


if __name__ == "__main__":
    main()
