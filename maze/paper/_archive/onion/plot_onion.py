#!/usr/bin/env python3
"""Figures for the onion-graft pilot (PDF + SVG only, linear axes).

  fig_onion_curves   learning curves, true and phantom world: onion grafts vs the edge-graft / base reference (5-seed means)
  fig_onion_dummy    the same with the adjacency list blanked: the graft is the only source of the maze
  fig_onion_depth    one-swap set: effect of a single edit by the onion layer that stores the edited connection

  .venv/bin/python maze/paper/onion/plot_onion.py
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import CUR, G, HERE, runs

BASE, EDGE, CHECKER, FULL = "#2a7a57", "#000c7a", "#eb6834", "#1c5cab"   # base / edge graft: project colours
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e6e5e0"
STYLE = {  # model -> (colour, linestyle, marker, label)
    "onion-checker fixed": (CHECKER, "-", "o", "onion, checker · fixed"),
    "onion-checker trainable": (CHECKER, "--", "s", "onion, checker · trainable"),
    "onion-full fixed": (FULL, "-", "o", "onion, full · fixed"),
    "onion-full trainable": (FULL, "--", "s", "onion, full · trainable"),
    "onion-checker-dummy fixed": (CHECKER, "-", "o", "onion, checker · fixed"),
    "edge-dummy fixed": (EDGE, "-", "o", "edge graft · fixed"),
}
for _k, (_c, _ls, _mk, _lab) in list(STYLE.items()):   # continuation runs (+3 epochs): same colour, thin, hollow marker
    _code, _kind = _k.split(" ")
    STYLE[f"{_code}-cont1 {_kind}"] = (_c, _ls, "^", _lab + " · +3 epochs")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
FINAL_X = 7812  # the final model is plotted at the last training step


def own_curves(r):
    rows = json.load(open(r["dir"] / "eval" / "metrics.json"))["rows"]
    out = {}
    off = FINAL_X if "cont1" in r["seed_tag"] else 0   # continuation: step counter restarts at 0
    for m in rows:
        x = off + (4 * FINAL_X if "cont1" in r["seed_tag"] else FINAL_X) - off if m["iteration"] == -1 else off + m["iteration"]
        out.setdefault(m["world"], []).append((x, m["p_sp"], m["solved"]))
    return {w: np.array(sorted(v)) for w, v in out.items()}


def ref_curves():
    acc = {}
    for row in csv.DictReader(open(CUR / "tables" / "all_metrics.csv")):
        if int(row["grid"]) != G:
            continue
        x = FINAL_X if row["final"] == "True" else int(float(row["iteration"]))
        acc.setdefault((row["cond"], row["world"], x), []).append((float(row["p_sp"]), float(row["solved"])))
    out = {}
    for (cond, world, x), v in acc.items():
        out.setdefault((cond, world), []).append((x, *np.mean(v, 0)))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def curves_figure(R, names, refs, fname, title_note):
    ref = ref_curves()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.2), sharex="col")
    cols = [("p(shortest-path token)", 1, (0, 4 * FINAL_X)), ("solved", 2, (0, 4 * FINAL_X)), ("p(shortest-path token), first 1,500 steps", 1, (0, 1500))]
    for i, world in enumerate(("true", "phantom")):
        for j, (ylabel, col, xlim) in enumerate(cols):
            ax = axes[i, j]
            for cond, colour, label in refs:
                if (cond, world) in ref:
                    c = ref[(cond, world)]
                    ax.plot(c[:, 0], c[:, col], color=colour, lw=1.2, ls=":" if cond == "base" else "-", label=label)
            for n in names + [f"{a}-cont1 {b}" for a, b in (x.split(" ") for x in names)]:
                if n in R and world in (c := own_curves(R[n])):
                    colour, ls, _, label = STYLE[n]
                    ax.plot(c[world][:, 0], c[world][:, col], color=colour, lw=1.2 if "cont1" in n else 2, ls=ls, label=label)
            ax.set_xlim(*xlim); ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(ylabel, fontsize=8, color=INK, loc="left")
            if i == 1:
                ax.set_xlabel("training step" + (" (7,812 = one pass over 2M mazes; continuations to 31,248)" if j == 1 else ""))
            if j == 0:
                ax.set_ylabel(f"{world} world" + (" (graft = true maze)" if world == "true" else " (graft = edited maze)"))
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"6×6, seed 0 — {title_note}", fontsize=9, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, fname)


def depth_figure():
    rows = list(csv.DictReader(open(HERE / "onion_depth.csv")))
    if not rows:
        return
    metrics = [("p_new", "p(new correct move)  ↑", (0, 1)), ("gain_nats", "log-prob gain of the new move, nats  ↑", None),
               ("solved_new", "solves the edited maze  ↑", (0, 1)), ("crossed_wall", "crosses the new wall  ↓", (0, 1))]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), sharex=True)
    for i, which in enumerate(("door e+", "wall e-")):
        for j, (key, label, ylim) in enumerate(metrics):
            ax = axes[i, j]
            seen = set()
            for r in rows:
                if r["stratified_by"] != which or r["depth"] == "all" or r["model"] in seen:
                    continue
                seen.add(r["model"])
                pts = sorted((int(q["depth"]), float(q[key])) for q in rows
                             if q["model"] == r["model"] and q["code"] == r["code"] and q["stratified_by"] == which and q["depth"] != "all")
                x, y = zip(*pts)
                if r["model"].startswith("REF"):
                    if r["code"] != "onion-checker" or "cont1" in r["code"]:
                        continue  # one control is enough: the labels of the checker layout
                    kind = "fixed" if "fixed" in r["model"] else "trainable"
                    ax.plot(x, y, color=EDGE, lw=1.2, ls="-" if kind == "fixed" else "--", marker="o" if kind == "fixed" else "s",
                            ms=3.5, label=f"edge graft · {kind} (control: same items)")
                else:
                    colour, ls, mk, lab = STYLE[r["model"]]
                    ax.plot(x, y, color=colour, lw=2, ls=ls, marker=mk, ms=5, mec="white", mew=0.8, label=lab)
            ax.set_xticks([0, 1, 2, 3])
            if ylim:
                ax.set_ylim(*ylim)
            if i == 0:
                ax.set_title(label, fontsize=8, color=INK, loc="left")
            if i == 1:
                ax.set_xlabel("onion layer of the edited connection (0 = outermost)")
            if j == 0:
                ax.set_ylabel("by layer of the " + ("opened door" if which == "door e+" else "new wall"))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("6×6, seed 0 — one-swap edits (DFS family), prompt shows the old maze, graft carries the edited maze", fontsize=9,
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, "fig_onion_depth")


if __name__ == "__main__":
    R = runs()
    curves_figure(R, ["onion-checker fixed", "onion-checker trainable", "onion-full fixed", "onion-full trainable"],
                  [("base", BASE, "base, no graft (5 seeds)"), ("fixed", EDGE, "edge graft · fixed (5 seeds)")],
                  "fig_onion_curves", "onion tile grafts vs the edge graft, adjacency list intact")
    curves_figure(R, ["edge-dummy fixed", "onion-checker-dummy fixed"], [("base", BASE, "base with the full prompt (5 seeds)")],
                  "fig_onion_dummy", "adjacency list blanked: the graft is the only source of the maze")
    depth_figure()
