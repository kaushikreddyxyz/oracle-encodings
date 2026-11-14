#!/usr/bin/env python3
"""SAE kinds compared as steering tools on the standard set (route-changed items, all maze-difference bins pooled).

Dot plot: one panel per maze size, rows = methods (labelled directly, so colour only marks the family: grey = reference,
green = steering vectors, amber = SAE features), x = solved (%), bars = SD over the 5 model seeds. Top row of panels =
steering at the adjacency-list tokens; bottom row = token-matched steering at the graft's tokens.
  python plot_sae_kinds.py
Reads results/distance/steer_out/*; writes figures/fig_sae_kinds.{pdf,svg}, tables/sae_kinds_cells.csv,
tables/sae_kinds_training.csv
"""
import csv, glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; R = HERE / "results" / "distance"
FIGS, TABLES = HERE / "figures", HERE / "tables"
FIGS.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
INK, GRID = "#000000", "#e8e7e2"
BASE, SAE_C, REF = "#2a7a57", "#e0a100", "#8a8a8a"
KINDS = [("relu", "ReLU + L1"), ("topk", "TopK"), ("btopk", "BatchTopK"), ("mat", "Matryoshka"), ("mp", "Matching pursuit")]


def folders():
    sel = json.load(open(R / "sae_logs" / "sae_selection_used.json"))["selection"]
    adj = [("oracle", "Prompt rewritten (ceiling)", REF), ("dom_L0_a1", "Steering vectors", BASE)]
    gs = [("graft:fixed", "Fixed graft", "#000c7a"), ("gs_reg_L0-11_a1_sae", "Steering vectors", BASE)]
    for k, lab in KINDS:
        adj.append((sel[f"adj:{k}"]["cond"].replace(":", "_") + "_f", f"SAE, {lab}", SAE_C))
        gs.append(("gs_" + sel[f"gs:{k}"]["cond"].replace(":", "_") + "_f", f"SAE, {lab}", SAE_C))
    adj.append(("none", "No steering (floor)", REF)); gs.append(("none", "No steering (floor)", REF))
    return [("Steering at the adjacency-list tokens", adj), ("Token-matched: steering at the graft's tokens", gs)], sel


def rates(name, sets):
    files = sorted(glob.glob(str(R / ("out/g*__" + name.split(":")[1] + "__*.npz" if name.startswith("graft:") else f"steer_out/{name}/*.npz"))))
    out = {}
    for f in files:
        D = np.load(f); g = int(D["grid_n"]); S = sets[g]; assert str(D["set_sha"]) == str(S["sha"]), f
        m = (~S["path_same"]) & (S["level"] > 0)
        out.setdefault(g, []).append(float(D["solved_graft"][m].mean()))
    return out


def main(preview=None):
    sets = {int(np.load(f)["grid_n"]): dict(np.load(f)) for f in glob.glob(str(R / "sets/g*.npz"))}
    grids = sorted(sets); rows_spec, _ = folders()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, len(grids), figsize=(9.4, 5.2), sharex=True)
    fig.subplots_adjust(left=.205, right=.985, top=.87, bottom=.10, wspace=.16, hspace=.50)
    cells = []
    for r, (row_title, methods) in enumerate(rows_spec):
        data = [(lab, col, rates(name, sets), name) for name, lab, col in methods]
        for c, g in enumerate(grids):
            ax = axes[r][c]
            for i, (lab, col, rt, name) in enumerate(data):
                v = np.array(rt.get(g, []))
                if len(v) == 0: continue
                mu, sd = v.mean() * 100, (v.std(ddof=1) * 100 if len(v) > 1 else 0)
                part = len(v) < 5                                   # incomplete cell (fewer than 5 seeds): hollow marker + seed count
                ax.errorbar([mu], [i], xerr=[[sd], [sd]], fmt="o", color=col, markersize=5, markerfacecolor="white" if part else col,
                            markeredgecolor=col if part else "white", markeredgewidth=1.2 if part else .6, elinewidth=1.3, capsize=0, zorder=3)
                ax.text(min(mu + sd + 3, 99) if mu < 70 else mu - sd - 3, i, f"{mu:.0f}" + (f" (n={len(v)})" if part else ""), va="center",
                        ha="left" if mu < 70 else "right", fontsize=7.5, color=INK)
                cells.append([row_title, lab, name, g, round(mu / 100, 4), round(sd / 100, 4), len(v)])
            ax.set_ylim(len(data) - .5, -.5); ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100], ["0", "50", ""])
            ax.set_yticks(range(len(data)), [d[0] for d in data] if c == 0 else [""] * len(data), fontsize=8.5, color=INK)
            ax.grid(axis="x", color=GRID, linewidth=.7, zorder=0); ax.tick_params(length=2.5, width=.8, labelsize=8.5, colors=INK)
            ax.spines[["top", "right"]].set_visible(False)
            for s in ("left", "bottom"): ax.spines[s].set_color(INK); ax.spines[s].set_linewidth(.9)
            ax.set_title(f"{g}×{g}", fontsize=10, color=INK, pad=4)
        fig.text(.205, axes[r][0].get_position().y1 + .055, row_title, fontsize=10, weight="bold", color=INK, ha="left", va="bottom")
    fig.text(.595, .02, "Solved (%), route-changed items", ha="center", fontsize=10.5, color=INK)
    for ext in ("pdf", "svg"): fig.savefig(FIGS / f"fig_sae_kinds.{ext}", facecolor="white")
    if preview: fig.savefig(preview, dpi=140, facecolor="white")
    plt.close(fig)
    with open(TABLES / "sae_kinds_cells.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["setting", "method", "folder", "grid", "solved_mean_over_seeds", "sd_over_seeds", "n_seeds"]); w.writerows(cells)
    # training diagnostics of the dictionaries actually used
    with open(TABLES / "sae_kinds_training.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["setting", "grid", "seed", "spec", "n_lat", "f_exp", "heldout_fvu", "heldout_L0", "dead_frac", "fvu_last/3rd_last", "loss_recovered", "ce_clean", "ce_sae", "ce_mean_ablation"])
        seen = set()
        for setting, pat, key in (("adjacency", "steer_prep/*/stats_f*.json", None), ("graft_tokens", "steer_gs_prep/*/stats_f*.json", "sae")):
            for p in sorted(glob.glob(str(R / pat))):
                gs_ = Path(p).parent.name; g, s = gs_[1:].split("-s"); J = json.load(open(p)); J = J["sae"] if key else J
                for spec, v in J.items():
                    if not isinstance(v, dict) or "heldout" not in v or (setting, g, s, spec) in seen: continue
                    seen.add((setting, g, s, spec))
                    w.writerow([setting, g, s, spec, v["n_lat"], v["f_exp"], round(v["heldout"]["fvu"], 5), round(v["heldout"]["L0"], 1), round(v["heldout"]["dead_frac"], 3),
                                round(v["converged_ratio"], 3), round(v["loss_recovered"], 4), round(v["ce_clean"], 4), round(v["ce_sae"], 4), round(v["ce_mean_ablation"], 4)])
    print("wrote fig_sae_kinds, sae_kinds_cells.csv, sae_kinds_training.csv;", len(cells), "cells")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--preview", default=None); main(ap.parse_args().preview)
