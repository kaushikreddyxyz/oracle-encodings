#!/usr/bin/env python3
"""Figures for the edge-vs-ego scale-up (grids 5-10, 5 seeds, prompt intact, graft from <ORIGIN_END>, 4 epochs).
Story order: (1) the model works with a different code, (2) it depends on the graft the same way, (3) the code sets the
internal frame, (4) how it is learned and why some large-grid ego seeds fail, (5) continuation to 8 epochs.
Style follows maze/paper/curves: PDF + SVG, seed mean +- SD, base green / edge navy / ego orange.

  fig_ev_main      3 panels vs grid: follows a single edit (one-swap p(new move)); solves the GRAFT maze when the prompt
                   disagrees (standard set, route-changed); true-world solve rate (the cost); dots = seeds
  fig_ev_ablate    inference ablations, solved and p(invalid), edge vs ego pooled over models
  fig_ev_probes    one panel per model: AUROC of linear (ridge) probes for allocentric vs egocentric wall targets, all layers
  fig_ev_curves    learning curves (solved, true and phantom world), grids overlaid, edge | ego
  fig_ev_collapse  the failed large-grid ego seeds: junction accuracy vs distance to target, p(PATH_END) at the target
  fig_ev_cont      9x9 / 10x10 continued to 8 epochs: per-seed curves and final metrics at 4 vs 8 epochs
Also writes ev_summary.csv (one row per model) and ev_probes.csv (one row per model x site x target x probe).
"""
import csv, json, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; CUR = HERE.parent / "curves"; RES = HERE / "results"
COL = {"base": "#2a7a57", "edge": "#000c7a", "ego": "#eb6834"}
LABEL = {"base": "No graft", "edge": "Allocentric graft", "ego": "Egocentric graft"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "legend.frameon": False, "axes.edgecolor": "#898781",
                     "xtick.color": "#898781", "ytick.color": "#898781", "axes.titlesize": 8.5})
E = 7812  # steps per epoch


def save(fig, name):
    fig.tight_layout(); [fig.savefig(HERE / f"{name}.{e}", bbox_inches="tight") for e in ("pdf", "svg")]; plt.close(fig)


def models():
    out = []
    for d in sorted((RES / "runs").glob("g*-s*-*-e*-oe")):
        m = re.match(r"g(\d+)-s(\d+)-(edge|ego|base)-e(4|8)-oe", d.name)
        if not m: continue
        for r in list(d.glob("fast-inj-fixed-*")) + list(d.glob("fast-baseline*")):
            if (r / "eval" / "metrics.json").exists():
                out.append(dict(grid=int(m.group(1)), seed=int(m.group(2)), code=m.group(3), epochs=int(m.group(4)), tag=d.name, dir=r))
    return out


def final_metrics(M):
    rows = []
    for m in M:
        rec = dict(grid=m["grid"], seed=m["seed"], code=m["code"], epochs=m["epochs"])
        for x in json.load(open(m["dir"] / "eval" / "metrics.json"))["rows"]:
            if x["iteration"] == -1:
                for k in ("solved", "p_sp", "p_invalid", "reached"): rec[f"{x['world']}_{k}"] = x[k]
        g = m["grid"]
        f = RES / "dist_out" / f"g{g}__fixed__{m['tag']}.npz"
        if f.exists() and m["code"] != "base":
            S = np.load(CUR / "results/distance/sets" / f"g{g}.npz"); z = np.load(f); rc = (~S["path_same"]) & (S["level"] > 0)
            rec["route_changed_graft"] = float(z["solved_graft"][rc].mean()); rec["route_changed_prompt"] = float(z["solved_token"][rc].mean())
            rec["same_maze"] = float(z["solved_graft"][S["level"] == 0].mean())
        f = RES / "local_out" / f"g{g}__fixed__{m['tag']}.npz"
        if f.exists() and m["code"] != "base":
            Lc = np.load(CUR / "results/local/sets" / f"g{g}.npz"); z = np.load(f); N = len(z["tid_a"]); fam = Lc["family"][:N] == 0
            rec["p_new"] = float(np.exp(z["graft_B__lp_b"][:N])[fam].mean()); rec["used_door"] = float(z["graft_B__used"][:N][fam].mean())
            rec["crossed_wall"] = float(z["graft_B__crossed"][:N][fam].mean()); rec["solved_new"] = float(z["graft_B__solved_b"][:N][fam].mean())
        f = m["dir"] / "eval" / "ablate.json"
        if f.exists():
            for x in json.load(open(f))["rows"]:
                rec[f"abl_{x['cond']}_solved"] = x["solved"]; rec[f"abl_{x['cond']}_p_invalid"] = x["p_invalid"]
        rows.append(rec)
    return rows


def agg(rows, key, code, grid=None, epochs=4):
    v = [r[key] for r in rows if r["code"] == code and key in r and (grid is None or r["grid"] == grid) and r["epochs"] == epochs]
    return (np.mean(v), np.std(v), len(v)) if v else (np.nan, np.nan, 0)


def line_vs_grid(ax, rows, key, codes, grids, dots=True):
    for code in codes:
        xs, ys, es = [], [], []
        for g in grids:
            mu, sd, n = agg(rows, key, code, g)
            if n: xs.append(g); ys.append(mu); es.append(sd)
        if not xs: continue
        xs, ys, es = np.array(xs), np.array(ys), np.array(es)
        ax.plot(xs, ys, color=COL[code], marker="o", ms=4, lw=1.6, label=LABEL[code], ls=":" if code == "base" else "-", zorder=3)
        if code != "base":
            ax.fill_between(xs, ys - es, ys + es, color=COL[code], alpha=0.13, linewidth=0, zorder=1)
        if dots:
            pts = [(r["grid"], r[key]) for r in rows if r["code"] == code and key in r and r["epochs"] == 4]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=8, color=COL[code], alpha=0.55, linewidths=0, zorder=2)
    ax.set_xticks(grids); ax.set_xticklabels([f"{g}×{g}" for g in grids]); ax.set_ylim(0, 1.02); ax.set_xlabel("Grid size")


def fig_main(rows):
    grids = sorted({r["grid"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    line_vs_grid(axes[0], rows, "p_new", ("edge", "ego"), grids)
    axes[0].set_title("Single-wall edit:\nP(new correct move)", loc="left")
    line_vs_grid(axes[1], rows, "route_changed_graft", ("edge", "ego"), grids)
    axes[1].set_title("Graft and prompt disagree:\nsolves the graft's maze", loc="left")
    line_vs_grid(axes[2], rows, "true_solved", ("base", "edge", "ego"), grids)
    axes[2].set_title("Unedited mazes:\nsolve rate", loc="left")
    axes[0].set_ylabel("Rate"); h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    save(fig, "fig_ev_main")


def fig_ablate(rows):
    conds = [("full", "Intact"), ("graft_off", "Graft\nremoved"), ("noise", "Graft\n→ noise"), ("off_gen", "Removed\nwhile moving"),
             ("x0.5", "Graft\n×0.5"), ("x2", "Graft\n×2"), ("shuffled", "Another\nmaze's graft"), ("prompt_blank", "Prompt\nblanked")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    for ax, key, title in ((axes[0], "solved", "Solve rate"), (axes[1], "p_invalid", "P(invalid move)")):
        for k, code in enumerate(("edge", "ego")):
            mu = [agg(rows, f"abl_{c}_{key}", code)[0] for c, _ in conds]; sd = [agg(rows, f"abl_{c}_{key}", code)[1] for c, _ in conds]
            x = np.arange(len(conds)) + (k - 0.5) * 0.36
            ax.bar(x, mu, 0.34, color=COL[code], label=LABEL[code], edgecolor="white", linewidth=0.5)
            ax.errorbar(x, mu, yerr=sd, fmt="none", ecolor="#333", elinewidth=0.8, capsize=2)
        ax.set_xticks(np.arange(len(conds))); ax.set_xticklabels([l for _, l in conds], fontsize=7); ax.set_title(title, loc="left"); ax.set_ylim(0, 1.02)
    n_e, n_g = agg(rows, "abl_full_solved", "edge")[2], agg(rows, "abl_full_solved", "ego")[2]
    axes[0].legend(loc="upper right", fontsize=7)
    save(fig, "fig_ev_ablate")


def fig_curves(M):
    grids = sorted({m["grid"] for m in M}); cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharex=True, sharey=True)
    for j, code in enumerate(("edge", "ego")):
        for i, world in enumerate(("true", "phantom")):
            ax = axes[i, j]
            for gi, g in enumerate(grids):
                curves = []
                for m in M:
                    if m["code"] != code or m["grid"] != g or m["epochs"] != 4: continue
                    R = json.load(open(m["dir"] / "eval" / "metrics.json"))["rows"]
                    curves.append(np.array(sorted((4 * E if x["iteration"] == -1 else x["iteration"], x["solved"]) for x in R if x["world"] == world)))
                if not curves: continue
                # align by iteration (a seed with a truncated sweep contributes only where it was scored)
                its = sorted({int(it) for c in curves for it in c[:, 0]})
                mean = [np.mean([c[c[:, 0] == it, 1][0] for c in curves if (c[:, 0] == it).any()]) for it in its]
                ax.plot(np.array(its) / E, mean, color=cmap(gi / max(1, len(grids) - 1)), lw=1.5, label=f"{g}×{g}")
            if i == 0: ax.set_title(LABEL[code], loc="left", color=COL[code])
            if j == 0: ax.set_ylabel("Solve rate" + (", graft edited" if world == "phantom" else ", unedited"))
            if i == 1: ax.set_xlabel("Epochs (1 epoch = 2M mazes = 7,812 steps)")
            ax.set_ylim(0, 1.02); ax.set_xticks([0, 1, 2, 3, 4])
    axes[0, 1].legend(loc="lower right", fontsize=7)
    save(fig, "fig_ev_curves")


PROBE_METRIC = "auroc"   # plotted metric: "auroc" (threshold-free, chance 0.5; needs results/probe_v2/*.json) or "r2"
PM = {"auroc": ("probe AUROC on held-out mazes (path tokens)", 0.5), "r2": ("probe R² on held-out mazes (path tokens)", 0.0), "bal_acc": ("balanced accuracy", 0.5)}


def fig_probes(M):
    recs = []
    for m in M:
        if m["epochs"] != 4: continue
        f2 = RES / "probe_v2" / f"{m['tag']}.json"; f = f2 if f2.exists() else m["dir"] / "eval" / "probe_nonlinear.json"
        if not f.exists(): continue
        P = json.load(open(f))
        for s, v in P["sites"].items():
            L = 0 if s == "L0in" else int(s[1:])
            for fam in ("allo", "ego"):
                for kind in ("linear", "mlp"):
                    e = v[f"{fam}_{kind}"]
                    recs.append(dict(grid=m["grid"], seed=m["seed"], code=m["code"], layer=L, family=fam, probe=kind, r2=e["r2"], acc=e["acc"],
                                     auroc=e.get("auroc", np.nan), bal_acc=e.get("bal_acc", np.nan), recall_open=e.get("recall_open", np.nan), recall_wall=e.get("recall_wall", np.nan)))
    if not recs: return
    with open(HERE / "ev_probes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    metric = PROBE_METRIC if not all(np.isnan(r[PROBE_METRIC]) for r in recs) else "r2"; ylab, chance = PM[metric]
    FCOL = {"allo": "#000c7a", "ego": "#eb6834"}; FLAB = {"allo": "allocentric walls (absolute)", "ego": "egocentric walls (relative to the current cell, ±2)"}
    FLOORS = {}
    ff = RES / "diag_probes" / "floors.csv"
    if ff.exists():   # mean over grids of the best history-only linear floor per family (F3edge for allo, F3rel for ego)
        FR = list(csv.DictReader(open(ff)))
        for fam, feat in (("allo", "F3edge"), ("ego", "F3rel")):
            v = [float(r["auroc"]) for r in FR if r["source"] == "floor" and r["family"] == fam and r["features"] == feat and r["subset"] == "all" and r["probe"] == "linear"]
            if v: FLOORS[fam] = float(np.mean(v))
    layers = sorted({r["layer"] for r in recs})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
    for ax, code in zip(axes, ("base", "edge", "ego")):
        n = len({(r["grid"], r["seed"]) for r in recs if r["code"] == code})
        for fam in ("allo", "ego"):
            for kind, ls, mk in (("linear", "-", "o"),):
                vals = [[r[metric] for r in recs if r["code"] == code and r["family"] == fam and r["probe"] == kind and r["layer"] == L and not np.isnan(r[metric])] for L in layers]
                if not any(vals): continue
                mu = np.array([np.mean(v) if v else np.nan for v in vals]); sd = np.array([np.std(v) if v else np.nan for v in vals])
                ax.plot(layers, mu, color=FCOL[fam], ls=ls, marker=mk, ms=3, lw=1.4, label=FLAB[fam])
                ax.fill_between(layers, mu - sd, mu + sd, color=FCOL[fam], alpha=0.10, linewidth=0)
        ax.set_title(f"{LABEL[code]}  (n={n})", loc="left", color=COL[code]); ax.set_xlabel("layer (0 = block-0 input incl. graft; k = after block k)")
        ax.set_ylim(chance - 0.05 if metric == "r2" else 0.4, 1.02); ax.axhline(chance, color="#bbb", lw=0.8, ls=":"); ax.set_xticks(range(0, 13, 2))
        if metric != "r2": ax.text(12, chance + 0.01, "chance", ha="right", va="bottom", fontsize=6.5, color="#898781")
        if metric == "auroc" and FLOORS:   # no-model floors: ridge on current cell + incoming move + traversed connections (results/diag_probes/floors.csv)
            for fam in ("allo", "ego"):
                ax.axhline(FLOORS[fam], color=FCOL[fam], lw=0.9, ls=(0, (4, 2)), alpha=0.7)
            if code == "base":
                ax.text(0.2, FLOORS["ego"] + 0.012, "no-model floor (position + walk history)", fontsize=6.5, color="#898781", va="bottom")
    axes[0].set_ylabel(ylab)
    h, l = axes[2].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig_ev_probes")


def fig_collapse():
    files = sorted((RES / "diag").glob("g*-e4.json"))
    if not files: return
    D = []
    for f in files:
        m = re.match(r"g(\d+)-s(\d+)-(edge|ego)-e4", f.name); d = json.load(open(f))
        D.append(dict(grid=int(m.group(1)), seed=int(m.group(2)), code=m.group(3), d=d, ok=d["target"]["p_path_end"] > 0.5))
    bins = list(next(iter(D))["d"]["junction_acc_by_steps_to_go"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    for r in D:
        acc = [r["d"]["junction_acc_by_steps_to_go"][b]["argmax_acc"] for b in bins]
        style = dict(color=COL[r["code"]], lw=1.6) if r["ok"] else dict(color=COL[r["code"]], lw=1.6, ls="--")
        lab = f"{r['grid']}×{r['grid']}, {LABEL[r['code']].split()[0].lower()}, seed {r['seed']}" + ("" if r["ok"] else " (stuck)")
        ax.plot(range(len(bins)), acc, marker="o", ms=3.5, label=lab, **style)
    ax.set_xticks(range(len(bins))); ax.set_xticklabels([b.replace("-400", "+").replace("-", "–") for b in bins]); ax.set_xlabel("Steps to the target")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.4, 1.02); ax.set_title("Correct move at junctions", loc="left")
    ax.legend(fontsize=6.5, loc="lower right")
    ax = axes[1]; x = np.arange(len(D))
    for xi, r in zip(x, D):
        ax.bar(xi, r["d"]["target"]["p_path_end"], color=COL[r["code"]], alpha=1.0 if r["ok"] else 0.45, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([f"{r['grid']}×{r['grid']}\nseed {r['seed']}" for r in D], fontsize=6.5); ax.set_ylim(0, 1.02)
    ax.set_title("P(stop) on the target", loc="left")
    save(fig, "fig_ev_collapse")


def fig_cont(M, rows):
    grids = sorted({m["grid"] for m in M if m["epochs"] == 8})
    if not grids: return
    fig, axes = plt.subplots(len(grids), 3, figsize=(11, 2.9 * len(grids)), squeeze=False)
    for i, g in enumerate(grids):
        for j, (key, title) in enumerate((("solved", "solve rate"), ("p_sp", "P(shortest-path move)"))):
            ax = axes[i, j]
            for code in ("edge", "ego"):
                for seed in sorted({m["seed"] for m in M if m["grid"] == g and m["code"] == code}):
                    pts = []
                    for ep in (4, 8):
                        mm = [m for m in M if m["grid"] == g and m["code"] == code and m["seed"] == seed and m["epochs"] == ep]
                        if not mm: continue
                        R = json.load(open(mm[0]["dir"] / "eval" / "metrics.json"))["rows"]; off = 0 if ep == 4 else 4 * E
                        pts += sorted((off + (4 * E if x["iteration"] == -1 else x["iteration"]), x[key]) for x in R if x["world"] == "true" and not (ep == 8 and x["iteration"] == 0))
                    if pts:
                        a = np.array(pts); ax.plot(a[:, 0] / E, a[:, 1], color=COL[code], lw=1.1, alpha=0.85, label=LABEL[code] if seed == min(m["seed"] for m in M if m["grid"] == g and m["code"] == code) else None)
            ax.axvline(4, color="#bbb", lw=0.8, ls=":"); ax.set_ylim(0, 1.02); ax.set_title(f"{g}×{g}: {title}", loc="left"); ax.set_xticks(range(0, 9, 2))
            if i == len(grids) - 1: ax.set_xlabel("Epochs")
        ax = axes[i, 2]; keys = [("true_solved", "Solve rate,\nunedited"), ("phantom_solved", "Solve rate,\ngraft edited"), ("route_changed_graft", "Graft and\nprompt disagree"), ("p_new", "Single-wall\nedit")]
        for k, code in enumerate(("edge", "ego")):
            for e, ep in enumerate((4, 8)):
                mu = [agg(rows, kk, code, g, ep)[0] for kk, _ in keys]; sd = [agg(rows, kk, code, g, ep)[1] for kk, _ in keys]
                x = np.arange(len(keys)) + (k - 0.5) * 0.4 + (e - 0.5) * 0.18
                ax.bar(x, mu, 0.17, color=COL[code], alpha=0.45 if ep == 4 else 1.0, edgecolor="white", linewidth=0.4, label=f"{LABEL[code]}, {ep} epochs")
                ax.errorbar(x, mu, yerr=sd, fmt="none", ecolor="#333", elinewidth=0.6, capsize=1.5)
        ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels([l for _, l in keys], fontsize=7); ax.set_ylim(0, 1.02); ax.set_title(f"{g}×{g}: 4 vs 8 epochs", loc="left")
        if i == 0: ax.legend(fontsize=6, loc="upper right", ncol=2)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    save(fig, "fig_ev_cont")


if __name__ == "__main__":
    M = models(); rows = final_metrics(M)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("grid", "seed", "code", "epochs"), k))
    with open(HERE / "ev_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"{len(M)} models:", sorted({(m['grid'], m['code'], m['epochs']) for m in M}))
    fig_main(rows); fig_ablate(rows); fig_curves(M); fig_collapse(); fig_cont(M, rows)   # fig_ev_probes: plot_pairs.py; print("figures written")
