#!/usr/bin/env python3
"""Graft vs token-matched steering in one figure: local (left) and global (right), both averaged over the six maze sizes.

Right  the standard distance-stratified set, route-changed items: solved (%) vs maze difference (%) (as fig_steering_lines).
Left   the single-swap set (build_local.py / eval_local.py), DFS-preserving swaps. Each outcome is an UNCONDITIONAL share of all
       edits (not conditional on the outcome before it), mean over maze sizes:
         follows the edit   picks new move   the model's top choice at the decision token is the move that is correct in the edited maze
                            uses new door    the greedy rollout from the decision cell goes through the added connection
                            solves new maze  that rollout reaches the target with every move legal in the edited maze
         ignores the edit   picks old move   the top choice is the move that was correct before the edit
                            crosses new wall the rollout goes through the removed connection
Both panels share the y axis (%). Bars/lines = equal-size means, then averaged over five seeds.
Local error bars and global bands show +-1 SD over seeds of the size-averaged values.
The prompt-rewrite ceiling has no uncertainty interval. Local solves start at the decision cell;
global solves start at the original maze origin. No-edit floors are not drawn (global 0.2%; local 11% / 1.3%, in the CSVs).
Steering settings: right (global), the ones chosen for the global figure (vectors: all blocks, alpha 1; SAE: block 6, strength 16);
left (local), strength tuned per model by 2-fold cross-fitting on the local metric (plot_local.tuned_values), i.e. the best case.

  python plot_steering_merged.py [--graft trainable] [--preview /tmp/x.png]   -> figures/fig_steering_lines_{graft}_merged.{pdf,svg}
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import plot_steering as PS
import plot_local as PL

INK, MUTED, GRID = "#000000", "#555555", "#e8e7e2"
LOCAL = {"oracle": "ceiling", "vec": "vec_tuned", "sae": "sae_tuned"}                 # series key -> plot_local method
# local outcomes, each an unconditional share of ALL single-connection edits (not conditional on the previous one); x position, metric, label
LOCAL_METRICS = [(0, "top1_new", "Choose\nnew move"), (1, "door_used", "Use\nnew door"),
                 (2.65, "top1_old", "Choose\nold move"), (3.65, "wall_crossed", "Cross\nnew wall")]
GROUPS = [((0, 1), "Acknowledges Edit", True), ((2.65, 3.65), "Ignores Edit", False)]


def global_curves(graft, by_size=False):
    sets, _ = PS.PH.load(); sets = PS.PH.merge_unrelated_by_distance(sets); grids = sorted(sets)
    width = float(next(iter(sets.values()))["width"]) * 100
    series = [("oracle", "oracle"), ("graft", f"graft:{graft}"), ("vec", PS.SERIES[1][0]), ("sae", PS.SERIES[2][0])]
    runs = {k: PS.load_method(name, sets) for k, name in series}
    mask = lambda s: (~s["path_same"]) | (s["level"] == 0)
    cells = {k: PS.cells(sets, r, mask) for k, r in runs.items()}
    bins = [lv for lv in range(1, max(int(s["n_buckets"]) for s in sets.values()) + 1)
            if all((g, lv) in c and c[g, lv][2] >= 1000 for c in cells.values() for g in grids)]
    out = {}
    for k, rs in runs.items():
        seeds = sorted({r["seed"] for r in rs}); assert len(seeds) == 5 and len(rs) == 30, (k, len(rs))
        v = np.array([[[next(r for r in rs if r["grid"] == g and r["seed"] == s)["solved_graft"][(sets[g]["level"] == lv) & mask(sets[g])].mean()
                        for lv in [0] + bins] for g in grids] for s in seeds]) * 100          # [seed, grid, bin]
        m = v if by_size else v.mean(1); out[k] = (m.mean(0), m.std(0, ddof=1))
    return [(lv - .5) * width for lv in bins], out


def local_points(graft, by_size=False):
    sets, runs = PL.load(str(PL.HERE / "results/local/out")); out = {}
    for k, mkey in dict(LOCAL, graft=graft).items():
        kind, cond, ref, _ = PL.METHODS[mkey]; v = np.zeros((5, len(sets), len(LOCAL_METRICS)))
        for gi, (g, S) in enumerate(sorted(sets.items())):
            N = len(S["maze"]); dfs = S["family"] == 0
            for s in range(5):
                R = runs[(g, kind, s)]; iv = PL.tuned_values(R, cond, ref, S, N)[0] if isinstance(cond, list) else PL.item_values(R, cond, ref)
                v[s, gi] = [iv[m][dfs].mean() * 100 for _, m, _ in LOCAL_METRICS]
        m = v if by_size else v.mean(1); out[k] = (m.mean(0), m.std(0, ddof=1))
    return out


def style(ax):
    ax.grid(axis="y", color=GRID, lw=.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(INK); ax.spines[s].set_linewidth(1.2)
    ax.tick_params(which="major", direction="out", length=4, width=1, labelsize=9.5, colors=INK)


def main(graft="fixed", preview=None):
    x, G = global_curves(graft); L = local_points(graft)
    series = [("graft", f"{graft.capitalize()} graft", PS.GRAFT), ("vec", "Steering vectors", PS.BASE), ("sae", "SAE features", PS.SAE_C)]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, (bx, ax) = plt.subplots(1, 2, figsize=(6.8, 3.5), sharey=True, gridspec_kw=dict(width_ratios=[1.7, 2], wspace=.06))   # local left, global right
    fig.subplots_adjust(left=.095, right=.965, bottom=.35, top=.83)
    # ---- global
    mu, _ = G["oracle"]
    ax.plot(x, mu[1:], color="#dc2625", linestyle=(0, (4, 2)), lw=1.3, zorder=4)
    for k, _, color in series:
        mu, sd = G[k]
        ax.fill_between(x, (mu-sd)[1:], (mu+sd)[1:], color=color, alpha=.16, linewidth=0, zorder=2)
        ax.plot(x, mu[1:], color=color, lw=1.6, marker="o", markersize=3.5,
                markeredgecolor="white", markeredgewidth=.5, zorder=3, clip_on=False)
    ax.set(xlim=(0, 60), ylim=(0, 100), xlabel="Maze difference (%)"); bx.set_ylabel("Share of edits (%)")
    ax.set_xticks(range(0, 61, 10), [f"{v}%" for v in range(0, 61, 10)]); ax.xaxis.set_minor_locator(MultipleLocator(5)); ax.set_yticks(range(0, 101, 25))
    ax.tick_params(axis="x", which="minor", direction="out", length=2.5, width=.9)
    ax.grid(axis="x", color=GRID, lw=.6, zorder=0)
    ax.set_title("From maze start · many edits", loc="left", fontsize=10, weight="bold", pad=6)
    # ---- local: grouped categorical bars with seed-SD error bars
    xs = np.array([x_ for x_, _, _ in LOCAL_METRICS])
    for (lo, hi), head, up in GROUPS:
        m = (xs >= lo) & (xs <= hi)
        bx.hlines(L["oracle"][0][m], xs[m]-.34, xs[m]+.34, color="#dc2625",
                  linestyle=(0, (4, 2)), lw=1.3, zorder=5)
        for j, (k, _, color) in enumerate(series):
            mu, sd = L[k]
            bx.bar(xs[m]+(j-1)*.23, mu[m], width=.20, color=color, edgecolor="none",
                   yerr=sd[m], error_kw={"ecolor": "#333333", "elinewidth": .65,
                                       "capsize": 0}, zorder=3)
        # group bracket + direction under the tick labels
        tr = bx.get_xaxis_transform()
        bx.plot([lo - .3, hi + .3], [-.33, -.33], color=MUTED, lw=.8, transform=tr, clip_on=False)
        bx.text((lo + hi) / 2, -.36, f"{head}\n({'higher' if up else 'lower'} is better)", transform=tr, ha="center", va="top", fontsize=8.5, color=MUTED, linespacing=1.15)
    bx.set_xlim(-.45, 4.1); bx.set_xticks(xs); bx.set_xticklabels([lab for _, _, lab in LOCAL_METRICS], fontsize=8.5)
    bx.set_title("From decision cell · one edit", loc="left", fontsize=10, weight="bold", pad=6)
    tr = ax.get_xaxis_transform(); ax.plot([2, 58], [-.33, -.33], color=MUTED, lw=.8, transform=tr, clip_on=False)
    ax.text(30, -.36, "Solves the Edited Maze\n(higher is better)", transform=tr, ha="center", va="top", fontsize=8.5, color=MUTED, linespacing=1.15)
    for a in (ax, bx): style(a)
    ax.tick_params(axis="x", labelsize=8.5); ax.tick_params(axis="y", which="both", left=False); bx.tick_params(axis="x", length=0, labelsize=8)
    ax.xaxis.label.set_size(10.5); bx.yaxis.label.set_size(11)
    handles = [Line2D([], [], color=c, lw=0, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=.6) for _, _, c in series]
    handles.append(Line2D([], [], color="#dc2625", linestyle=(0, (4, 2)), lw=1.3))
    fig.legend(handles, [lab for _, lab, _ in series] + ["Prompt rewritten (ceiling)"], loc="upper left", bbox_to_anchor=(.083, 1.0), ncol=4, frameon=False,
               fontsize=9, handlelength=1.9, columnspacing=1.4, handletextpad=.5, labelcolor=INK)
    stem = PS.FIGS / f"fig_steering_lines_{graft}_merged"
    for ext in ("pdf", "svg"): fig.savefig(stem.with_suffix("." + ext), facecolor="white")
    if preview: fig.savefig(preview, dpi=170, facecolor="white")
    plt.close(fig); print(stem)
    for k in ("oracle", "graft", "vec", "sae"):
        print(f"  {k:7s} global mean over bins {G[k][0][1:].mean():5.1f} | local " + " / ".join(f"{m:.1f}±{s:.1f}" for m, s in zip(*L[k])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graft", choices=["fixed", "trainable"], default="fixed"); ap.add_argument("--preview", default=None)
    a = ap.parse_args(); main(a.graft, a.preview)
