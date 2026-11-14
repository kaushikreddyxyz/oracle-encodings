#!/usr/bin/env python3
"""Next-token probability mass by category: base vs grafted model, true vs
phantom world, mean ± SD over model seeds (seed points overlaid).

  python plot_next_token.py --results results/results.json --out fig_next_token   # -> figures/
Model names in results.json are "<cond>-s<seed>", cond in {baseline, fixed, trainable}.
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

BASE = "#2a7a57"      # base model
GRAFT = "#000c7a"     # grafted model
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
CATS = [("p_correct", "SP token"), ("p_valid", "Σ valid tokens\n− SP token"), ("p_invalid", "Σ invalid\ntokens")]


def load(results):
    by = {}
    for name, d in json.load(open(results)).items():
        m = re.match(r"^(baseline|fixed|trainable)-s(\d+)$", name)
        if not m:
            continue
        by.setdefault(m.group(1), {})[int(m.group(2))] = d
    return by


def series(by, cond, setting, key, split="all"):
    vals = []
    for seed in sorted(by.get(cond, {})):
        d = by[cond][seed][setting]
        vals.append(d[key] if split == "all" else d[split][key])
    return np.array(vals)


def panel(ax, by, graft_cond, split, log, with_control):
    # same-model bars adjacent, identical order in every group
    bars = [("base model · true world", "baseline", "true", BASE, None)]
    if with_control:
        bars.append(("base model · phantom world (cannot see the edit)", "baseline", "phantom", BASE, "////"))
    bars += [("grafted model · true-world features", graft_cond, "true", GRAFT, None),
             ("grafted model · phantom-world features", graft_cond, "phantom", GRAFT, "////")]
    nb = len(bars)
    width = 0.8 / nb
    for ci, (key, _) in enumerate(CATS):
        for bi, (label, cond, setting, color, hatch) in enumerate(bars):
            v = series(by, cond, setting, key, split)
            if len(v) == 0:
                continue
            x = ci - 0.4 + width * (bi + 0.5)
            mean, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
            ax.bar(x, mean, width * 0.9, color=color if hatch is None else SURFACE, edgecolor=color,
                   linewidth=1.2, hatch=hatch, zorder=3)
            if len(v) > 1:
                ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=5)
            txt = f"{mean:.3f}" if mean >= 0.01 else f"{mean:.1e}"
            top = mean + (sd if len(v) > 1 else 0)
            ax.text(x, top * (1.3 if log else 1.0) + (0 if log else 0.015), txt, ha="center",
                    va="bottom", fontsize=5.4, color=INK2, zorder=7, rotation=90)
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels([c[1] for c in CATS], fontsize=7.5, color=INK, linespacing=1.15)
    ax.set_xlim(-0.6, len(CATS) - 0.4)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 6)
    else:
        ax.set_ylim(0, 1.08)
    ax.set_ylabel("next-token probability mass", fontsize=7.5, color=INK)
    ax.tick_params(axis="y", labelsize=7, colors=INK2, length=0)
    ax.tick_params(axis="y", which="minor", length=0, labelleft=False)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    return bars


def main(results="results/results.json", out="fig_next_token", split="all", log=False,
         with_control=False, conds="fixed,trainable"):
    by = load(results)
    conds = [c for c in conds.split(",") if c in by]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, len(conds), figsize=(3.6 * len(conds), 2.9), dpi=200, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cond in zip(axes, conds):
        bars = panel(ax, by, cond, split, log, with_control)
        n_seeds = len(by.get(cond, {}))
        ax.set_title(f"{cond} injection  ·  {n_seeds} seed{'s' if n_seeds != 1 else ''}",
                     fontsize=8.5, color=INK)
    handles = [Patch(facecolor=c if h is None else SURFACE, edgecolor=c, hatch=h, linewidth=1.2, label=l)
               for (l, _, _, c, h) in bars]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.6, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    figs = Path(__file__).parent / "figures"
    figs.mkdir(exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(figs / f"{out}.{ext}", facecolor=SURFACE, bbox_inches="tight")
    print("wrote", figs / f"{out}.{{pdf,svg}}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
