#!/usr/bin/env python3
"""fig_ev_probe_offsets: linear-probe AUROC of the DOWN connection at each offset (dr, dc) from the current cell,
rows = base / edge / ego model, cols = L0in, L3, L6, L12; mean over seeds then grids 5-10.  Reads ego_offsets_linear.csv."""
import csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent; CODES = HERE.parents[1]
COL = {"base": "#2a7a57", "edge": "#000c7a", "ego": "#eb6834"}
LABEL = {"base": "no graft", "edge": "allocentric graft\n(edge code)", "ego": "egocentric graft\n(ego code)"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
                     "legend.frameon": False, "axes.edgecolor": "#898781", "xtick.color": "#898781", "ytick.color": "#898781", "axes.titlesize": 8.5})
SITES = ["L0in", "L3", "L6", "L12"]; STITLE = {"L0in": "block-0 input", "L3": "after block 3", "L6": "after block 6", "L12": "after block 12"}
CMAP = LinearSegmentedColormap.from_list("seq", ["#f7f6f2", "#d6d9e4", "#8a92b2", "#3b446e", "#141a33"])


def table():
    T = {}
    for r in csv.DictReader(open(HERE / "ego_offsets_linear.csv")):
        if r["grid"] == "mean" and r["kind"] == "down":
            T.setdefault((r["model"], r["site"]), np.zeros((5, 5)))[int(r["dr"]) + 2, int(r["dc"]) + 2] = float(r["auroc"])
    return T


def main():
    T = table()
    fig, axs = plt.subplots(3, 4, figsize=(6.3, 5.0))
    for i, m in enumerate(("base", "edge", "ego")):
        for j, s in enumerate(SITES):
            ax = axs[i, j]; A = T[(m, s)]
            im = ax.imshow(A, cmap=CMAP, vmin=0.5, vmax=1.0)
            for a in range(5):
                for b in range(5):
                    v = round(A[a, b], 2); ax.text(b, a, f"{v:.2f}"[1:] if v < 1 else "1.0", ha="center", va="center", fontsize=5.6,
                            color="white" if A[a, b] > 0.78 else "#3d3d3a")
            ax.add_patch(Rectangle((1.5, 1.5), 1, 1, fill=False, ec=COL[m], lw=1.6))
            ax.set_xticks(range(5)); ax.set_yticks(range(5))
            ax.set_xticklabels(range(-2, 3) if i == 2 else [], fontsize=6.5); ax.set_yticklabels(range(-2, 3) if j == 0 else [], fontsize=6.5)
            ax.tick_params(length=0)
            for sp in ax.spines.values(): sp.set_visible(False)
            if i == 0: ax.set_title(STITLE[s])
            if j == 0: ax.set_ylabel(LABEL[m] + "\n\ndr (row offset)", color=COL[m], fontsize=7.5)
            if i == 2: ax.set_xlabel("dc (column offset)", fontsize=7)
    fig.subplots_adjust(left=0.13, right=0.86, top=0.88, bottom=0.10, wspace=0.08, hspace=0.12)
    cax = fig.add_axes([0.88, 0.30, 0.012, 0.42]); cb = fig.colorbar(im, cax=cax); cb.set_label("linear-probe AUROC (chance 0.5)", fontsize=7)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5, length=0)
    fig.suptitle("Without an egocentric graft, linear probes read the current cell's own walls well (N, S = 0.86-0.89 after block 6) and walls\n"
                 "1-2 cells away only weakly (0.62-0.76); at the input, readability is fully explained by the cell + step location prior (+ raw graft bits)",
                 fontsize=8, y=0.995)
    fig.text(0.5, 0.005, "DOWN connection of the cell at offset (dr, dc); boxed = current cell, so (0,0) is its S wall and (-1,0) its N wall. "
             "RIGHT connections are the transpose (max |diff| 0.022). Mean of seeds then grids 5-10.", ha="center", fontsize=6.3, color="#5f5e5a")
    for e in ("pdf", "svg"): fig.savefig(CODES / f"fig_ev_probe_offsets.{e}", bbox_inches="tight")
    fig.savefig(HERE / "fig_preview.png", dpi=170, bbox_inches="tight")


if __name__ == "__main__":
    main()
