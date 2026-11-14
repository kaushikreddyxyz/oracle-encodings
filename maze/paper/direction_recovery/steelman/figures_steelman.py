#!/usr/bin/env python3
"""Steelman figures of the direction-recovery campaign (PDF + SVG only, written next to the original figures with a
`_steelman` suffix). Reads the tables written by collect.py.

  python figures_steelman.py [--tables DIR] [--preview DIR]

Main (minimal, illustrative: ONE SAE entry and ONE probe / steering-vector entry, both under the STEELMAN selection = upper bound):
  figures/alignment_violins_steelman            per-connection |cosine| to the graft direction, injection site
  figures/maze_diff_accuracy_steelman           A->B intervention on the full standard set, route-changed items, by maze size
Supplementary:
  figures/supplementary/alignment_fair_vs_steelman_steelman     fair vs steelman selection, per SAE kind and per probe family
  figures/supplementary/probe_fit_size_steelman                 probe alignment vs number of fitting mazes
  figures/supplementary/sae_sweep_steelman                      SAE alignment across the sweep (dictionary size x sparsity)
  figures/supplementary/maze_diff_accuracy_all_steelman         every intervention entry, both sites
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent; FIG = HERE.parent / "figures"
INK, MUTED, GRID = "#000000", "#555555", "#e8e7e2"
BASE, GRAFT, SAE_C, REF = "#2a7a57", "#000c7a", "#e0a100", "#8a8a8a"           # project colours (maze/paper/curves/plot_steering.py)
FILL = {BASE: "#d6e5de", GRAFT: "#dddff0", SAE_C: "#f8e9bf", REF: "#e6e6e6"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name, sub=False, preview=None):
    d = FIG / "supplementary" if sub else FIG; d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"): fig.savefig(d / f"{name}.{ext}", facecolor="white")
    if preview: fig.savefig(Path(preview) / f"{name}.png", dpi=130, facecolor="white")
    plt.close(fig)


def weighted(A, entry, site="injection", cond=None, col="cos_id"):
    """values, weights: equal weight per model, equal weight per connection within a model (as in plot_alignment_violins.py)."""
    g = A[(A.entry == entry) & (A.site == site)]
    if cond: g = g[g.cond == cond]
    n_models = g.tag.nunique(); w = 1 / n_models / g.groupby("tag")[col].transform("size")
    return np.clip(g[col].to_numpy(), 0, 1), w.to_numpy(), n_models


def violin(ax, y, values, weights, color, height=.36):
    edges = np.linspace(0, 1, 2001); dx = edges[1] - edges[0]; centers = (edges[:-1] + edges[1:]) / 2
    mass, _ = np.histogram(values, edges, weights=weights); dens = gaussian_filter1d(mass, .015 / dx, mode="reflect") / dx
    lo, hi = values.min(), values.max(); xx = np.r_[lo, centers[(centers > lo) & (centers < hi)], hi]; yy = np.interp(xx, centers, dens)
    wd = height * yy / yy.max(); ax.fill_between(xx, y - wd, y + wd, facecolor=FILL[color], edgecolor=color, lw=1, zorder=2)
    order = np.argsort(values); cum = np.cumsum(weights[order]); q = values[order][np.searchsorted(cum, [.25, .75])]
    mean = float(np.dot(values, weights)); ax.plot(q, [y, y], color=color, lw=2.6, solid_capstyle="round", zorder=3)
    ax.scatter(mean, y, s=29, color=color, edgecolor="white", linewidth=.8, zorder=4)
    return mean, float(weights[values >= .8].sum()), q


def violin_figure(A, rows, name, title, note, sub=False, preview=None, site="injection", height=None):
    """rows: [(label, entry, colour, cond or None)] top to bottom; None = gap."""
    ys = []; y = 0; summ = []
    for r in rows:
        if r is None: y += .5; continue
        ys.append(y); y += 1
    top, body, xlab = .8, .46 * (max(ys) + 1.2), .62; foot = (.15 * (note.count("\n") + 1) + .12) if note else 0; H = top + body + xlab + foot
    fig = plt.figure(figsize=(7.5, H)); ax = fig.add_axes([.34, (xlab + foot) / H, .48, body / H])
    for (label, entry, color, cond), yy in zip([r for r in rows if r], ys):
        v, w, nm = weighted(A, entry, site, cond); mean, f08, q = violin(ax, yy, v, w, color)
        ax.text(-.035, yy, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9.5, color=INK)
        ax.text(1.06, yy, f"{mean:.3f}", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=9.5, color=INK)
        ax.text(1.2, yy, f"{100 * f08:.0f}%", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=9.5, color=INK)
        summ.append(dict(figure=name, label=label, entry=entry, cond=cond or "both", site=site, models=nm, connections=len(v), mean=mean, frac_ge_08=f08, q25=q[0], q75=q[1]))
    hy = 1 + .12 / body
    ax.text(1.06, hy, "Mean", transform=ax.transAxes, ha="center", fontsize=9, color=MUTED); ax.text(1.2, hy, "≥ 0.8", transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
    ax.axvline(1, color=GRAFT, lw=1.2, zorder=1)
    ax.set(ylim=(max(ys) + .6, -.6), xlim=(0, 1.012)); ax.set_yticks([]); ax.set_xticks([0, .2, .4, .6, .8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.set_xlabel("Alignment with the graft direction (|cosine|)", labelpad=6); ax.grid(axis="x", color=GRID, lw=.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_linewidth(1.2)
    fig.text(.03, 1 - .1 / H, title, fontsize=13, weight="bold", va="top")
    fig.legend(handles=[Line2D([], [], color=INK, lw=2.6, label="IQR"), Line2D([], [], color=INK, marker="o", lw=0, markersize=5, label="Mean"),
                        Line2D([], [], color=GRAFT, lw=1.2, label="Ground-truth graft (= 1)")], loc="upper right", bbox_to_anchor=(.985, 1 - .3 / H), ncol=3, frameon=False, fontsize=9, handlelength=1.5, columnspacing=1.2)
    if note: fig.text(.03, .06 / H, note, fontsize=7.5, color=MUTED, va="bottom", linespacing=1.4)
    save(fig, name, sub, preview); return summ


def line_figure(M, series, name, title, note, sub=False, preview=None, metric="solved_route_changed"):
    """series: [(label, method key, colour, linestyle)]; mean over 5 seeds with +-1 SD band, per maze size; panels = fixed / trainable graft."""
    nl = note.count("\n") + 1 if note else 0; rows = int(np.ceil(len(series) / 3)); top = .6 + .22 * rows; H = 2.6 + top + .16 * nl
    fig, axs = plt.subplots(1, 2, figsize=(7.5, H), sharey=True); fig.subplots_adjust(left=.085, right=.985, top=1 - top / H, bottom=(.55 + .16 * nl) / H, wspace=.08); summ = []
    for ax, cond in zip(axs, ("fixed", "trainable")):
        for label, key, color, ls in series:
            g = M[(M.cond == cond) & (M.method == key)]
            if g.empty: continue
            z = g.groupby("grid")[metric].agg(["mean", "std", "count"]) * [100, 100, 1]
            ax.fill_between(z.index, z["mean"] - z["std"], z["mean"] + z["std"], color=color, alpha=.15, lw=0)
            ax.plot(z.index, z["mean"], color=color, lw=1.8, ls=ls, marker="o", ms=4, mec="white", mew=.8, label=label, zorder=3)
            summ += [dict(figure=name, cond=cond, label=label, method=key, grid=int(k), mean_pct=r["mean"], sd_seeds_pct=r["std"], n_models=int(r["count"])) for k, r in z.iterrows()]
        ax.set_title(f"{cond.capitalize()} graft", fontsize=10.5, loc="left"); ax.set_xticks(range(5, 11), [f"{k}×{k}" for k in range(5, 11)]); ax.set_ylim(0, 100); ax.set_xlim(4.6, 10.4)
        ax.grid(axis="y", color=GRID, lw=.6); ax.set_axisbelow(True); ax.set_xlabel("Maze size")
    axs[0].set_ylabel("Solved in the edited maze (%)"); h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(.03, 1 - .36 / H), ncol=3, frameon=False, fontsize=8.5, handlelength=2, columnspacing=1.2)
    fig.text(.03, 1 - .1 / H, title, fontsize=13, weight="bold", va="top")
    if note: fig.text(.03, .06 / H, note, fontsize=7.5, color=MUTED, va="bottom", linespacing=1.4)
    save(fig, name, sub, preview); return summ


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tables", default=str(HERE)); ap.add_argument("--preview"); a = ap.parse_args(); T = Path(a.tables); summ = []
    A = pd.read_csv(T / "alignment_values.csv.gz"); nm = A[A.site == "injection"].tag.nunique()
    note = (f"Injection site (PATH_START, resid_pre of block 0); {nm} grafted models (grids 5–10 × 5 seeds × fixed/trainable), equal weight per model and per connection.\n"
            "SAE and probe rows use the STEELMAN selection (configuration, latent and hyper-parameter chosen by true alignment with W_inj): an upper bound.\n"
            "Identifiable gauge (mean free-connection direction removed). Gaussian bandwidth 0.015, widths scaled per row.")
    summ += violin_figure(A, [("SAE (best configuration)", "sae_steelman", SAE_C, None), ("Probe / steering vector (best)", "probe_steelman", BASE, None),
                              ("Generative regression (control)", "generative_ols", REF, None)], "alignment_violins_steelman", "Graft-direction alignment, steelman rerun", note, preview=a.preview)
    rows = [("SAE, steelman selection", "sae_steelman", SAE_C, None), ("SAE, steelman without coding-aware init", "sae_steelman_nocoding", SAE_C, None), ("SAE, fair selection", "sae_fair", SAE_C, None), None]
    for kind, lab in (("relu", "ReLU + L1"), ("topk", "TopK"), ("btopk", "BatchTopK"), ("mat", "Matryoshka"), ("mp", "Matching pursuit")):
        for crit in ("steelman", "fair"):
            if (A.entry == f"sae_{crit}_{kind}").any(): rows.append((f"{lab}, {crit}", f"sae_{crit}_{kind}", SAE_C, None))
    rows += [None, ("Probe, steelman selection", "probe_steelman", BASE, None), ("Probe, fair selection", "probe_fair", BASE, None), ("Difference of means", "difference_means", BASE, None),
             ("Ridge / LDA readout, best λ", "probe_steelman_ridge", BASE, None), ("Logistic readout, best λ", "probe_steelman_logistic", BASE, None), None,
             ("Pseudo-inverse of all readouts", "probe_pinv", REF, None), ("Generative regression (control)", "generative_ols", REF, None)]
    rows = [r for r in rows if r is None or (A.entry == r[1]).any()]
    summ += violin_figure(A, rows, "alignment_fair_vs_steelman_steelman", "Alignment by selection function and method", "fair = chosen on validation mazes without W_inj; steelman = chosen by true alignment (upper bound). Injection site, all models.", sub=True, preview=a.preview)
    if (A.site == "post0").any():
        rows0 = [r for r in rows if r is None or ((A.entry == r[1]) & (A.site == "post0")).any()]
        summ += violin_figure(A, rows0, "alignment_post0_steelman", "Alignment after block 0", "Site: resid_post of block 0 at PATH_START; |cosine| to the original graft rows (identifiable gauge). SAEs: model seeds 0–1 only.", sub=True, preview=a.preview, site="post0")
    if (T / "intervention_models.csv").exists():
        M = pd.read_csv(T / "intervention_models.csv"); nmi = M[M.method == "injection|graft"].tag.nunique()
        note = (f"Prompt and graft describe maze A; the edit moves the injection-site activation towards maze B along the recovered directions.\n"
                f"Full standard set (3,000 base mazes per size), route-changed items, rollout limit n²+2, solved = target reached with no rule break.\n"
                f"Lines: mean over 5 model seeds, bands ±1 SD; {nmi} models. SAE and probe entries use the steelman selection (upper bound);\n"
                "edit strength tuned per model on 100 disjoint base mazes.")
        summ += line_figure(M, [("Ground-truth graft", "injection|graft", GRAFT, "-"), ("Generative regression (control)", "injection|generative_ols", REF, (0, (4, 2))),
                                ("SAE (best)", "injection|sae_steelman", SAE_C, "-"), ("Probe / steering vector (best)", "injection|probe_steelman", BASE, "-"),
                                ("No edit", "injection|unedited", REF, (0, (1, 2)))], "maze_diff_accuracy_steelman", "Editing maze A into maze B with recovered directions", note, preview=a.preview)
        for site, lab in (("injection", "injection site"), ("post0", "after block 0")):
            ser = [("Ground-truth graft", "injection|graft", GRAFT, "-"), ("Generative regression", f"{site}|generative_ols", REF, (0, (4, 2))), ("Readout pseudo-inverse", f"{site}|probe_pinv", REF, (0, (1, 1))),
                   ("SAE, steelman", f"{site}|sae_steelman", SAE_C, "-"), ("SAE, fair", f"{site}|sae_fair", SAE_C, (0, (4, 2))), ("Probe, steelman", f"{site}|probe_steelman", BASE, "-"),
                   ("Probe, fair", f"{site}|probe_fair", BASE, (0, (4, 2))), ("Difference of means", f"{site}|difference_means", BASE, (0, (1, 1)))]
            summ += line_figure(M, ser, f"maze_diff_accuracy_all_{site}_steelman", f"All intervention entries, {lab}", "fair = selected without W_inj; steelman = selected by true alignment.\nRoute-changed items of the full standard set, mean ± SD over seeds.", sub=True, preview=a.preview)
    if (T / "probe_metrics.csv").exists(): summ += probe_size_figure(pd.read_csv(T / "probe_metrics.csv"), a.preview)
    if (T / "sae_metrics.csv").exists(): summ += sweep_figure(pd.read_csv(T / "sae_metrics.csv"), a.preview)
    pd.DataFrame(summ).to_csv(T / "figure_summaries_steelman.csv", index=False)


def probe_size_figure(P, preview):
    P = P[P.method != "lda_vs_ridge_cos_check"].copy(); P["family"] = P.method.str.split("_").str[0]
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.6), sharey=True); fig.subplots_adjust(left=.085, right=.985, top=.69, bottom=.16, wspace=.08); summ = []
    fams = [("Generative regression (control)", "generative", REF, (0, (4, 2))), ("Ridge / LDA readout, best λ", "ridge", BASE, "-"), ("Logistic readout, best λ", "logistic", BASE, (0, (4, 2))), ("Difference of means", "difference", BASE, (0, (1, 1)))]
    for ax, site in zip(axs, ("injection", "post0")):
        for label, fam, color, ls in fams:
            g = P[(P.site == site) & (P.family == fam)]
            best = g.groupby(["tag", "seed", "n_fit"]).cos_id.max().reset_index()                  # oracle hyper-parameter per model and size
            z = best.groupby(["seed", "n_fit"]).cos_id.mean().groupby("n_fit").agg(["mean", "std"])
            ax.fill_between(z.index, z["mean"] - z["std"], z["mean"] + z["std"], color=color, alpha=.15, lw=0); ax.plot(z.index, z["mean"], color=color, ls=ls, lw=1.8, marker="o", ms=4, mec="white", mew=.8, label=label)
            summ += [dict(figure="probe_fit_size_steelman", site=site, label=label, n_fit=int(k), mean=r["mean"], sd_seeds=r["std"]) for k, r in z.iterrows()]
        ax.set_xscale("log"); ax.set_title("Injection site" if site == "injection" else "After block 0", fontsize=10.5, loc="left"); ax.set_xlabel("Fitting mazes"); ax.set_ylim(0, 1.02); ax.grid(axis="y", color=GRID, lw=.6)
    axs[0].set_ylabel("Mean |cosine| to graft direction"); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper left", bbox_to_anchor=(.03, .9), ncol=2, frameon=False, fontsize=8.5, handlelength=2, columnspacing=1.2)
    fig.text(.03, .975, "More fitting data does not move probe directions", fontsize=13, weight="bold", va="top"); save(fig, "probe_fit_size_steelman", True, preview); return summ


def sweep_figure(Sm, preview):
    """x = active latents as a fraction of the free connections (the true code has n^2-1 doors ~ half of them); one line per
    dictionary size; points = mean over the 12 seed-0 models of each sparsity setting (k, or L1 for ReLU)."""
    Sm = Sm[(Sm.site == "injection") & (Sm.variant == "ctr") & (Sm.sae_seed == 0) & (Sm.seed == 0)].copy()
    Sm["size"] = Sm.job.str.split("_").str[0].str.split("-").str[1]; Sm["nfree"] = 2 * Sm.grid * (Sm.grid - 1); Sm["rel"] = Sm.L0 / Sm.nfree
    def setting(r):
        if r.kind == "relu": return f"L1={r.l1:g}"
        n = r.grid; return "k=n²−1" if r.k == n * n - 1 else "k=(n−1)²" if r.k == (n - 1) ** 2 else f"k={r.k}"
    Sm["setting"] = Sm.apply(setting, axis=1)
    kinds = [("relu", "ReLU + L1"), ("topk", "TopK"), ("btopk", "BatchTopK"), ("mat", "Matryoshka"), ("mp", "Matching pursuit")]
    fig, axs = plt.subplots(1, 5, figsize=(11, 3.4), sharey=True); fig.subplots_adjust(left=.06, right=.99, top=.72, bottom=.19, wspace=.08); summ = []
    sizes = ["m1", "m1.5", "m3", "m8", "mn1024"]; shades = ["#5c4200", "#9a6f00", "#e0a100", "#edc458", "#f3dc9a"]; names = {"m1": "1×", "m1.5": "1.5×", "m3": "3×", "m8": "8×", "mn1024": "1,024 atoms"}
    for ax, (kind, lab) in zip(axs, kinds):
        g = Sm[Sm.kind == kind]
        for size, c in zip(sizes, shades):
            h = g[g["size"] == size]
            if h.empty: continue
            z = h.groupby("setting").agg(x=("rel", "mean"), y=("oracle_id", "mean"), n=("oracle_id", "size")).sort_values("x")
            ax.plot(z.x, z.y, color=c, lw=1.6, marker="o", ms=3.5, mec="white", mew=.6, label=names[size], zorder=3)
            summ += [dict(figure="sae_sweep_steelman", kind=kind, size=size, setting=k, mean_rel_L0=r.x, oracle_cos_id=r.y, n_models=int(r.n)) for k, r in z.iterrows()]
        ax.axvline(.5, color=GRID, lw=1); ax.set_xscale("log"); ax.set_xlim(.015, 1.6); ax.set_xticks([.03, .1, .3, 1], ["0.03", "0.1", "0.3", "1"])
        ax.set_title(lab, fontsize=10.5, loc="left"); ax.set_xlabel("Active latents / free connections"); ax.set_ylim(0, 1.02); ax.grid(axis="y", color=GRID, lw=.6)
    axs[0].set_ylabel("Best-latent |cosine| (oracle)"); h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, title="Dictionary size (× free connections)", loc="upper left", bbox_to_anchor=(.06, .93), ncol=5, frameon=False, fontsize=8.5, title_fontsize=8.5)
    fig.text(.03, .975, "SAE sweep at the injection site: alignment needs the true code density", fontsize=11.5, weight="bold", va="top")
    fig.text(.99, .975, "Seed-0 models, mean over sizes 5–10 and both grafts.\nVertical line: n²−1 open connections ≈ half the free connections.", fontsize=7.5, color=MUTED, ha="right", va="top", linespacing=1.4)
    save(fig, "sae_sweep_steelman", True, preview); return summ


if __name__ == "__main__":
    main()
