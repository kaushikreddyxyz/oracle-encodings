"""Stage 6.1 ANALYSIS figures — per-concept causal pages + fleet-level plots.

House style follows stage6/code/plots.py (matplotlib Agg, 110 dpi pages,
verdict colors green/orange/red, rollup tables in md + html sorted worst-
first). Series identity colors are fixed per ARM (never cycled): ridge
#1965b0, dom #e8601c, other-concept controls #882e72; random-direction
controls are deliberately neutral gray #777777 (they are reference series).
Verdict/status colors: causal #4eb265, read-only #f1932d, artifact-suspect
#dc050c (same greens/oranges/reds as the Stage-6 rollup).

Inputs: <out>/causal_cards.json from analyze.py + the same experiment roots
(analyze.Store handles fleet-merge/dedup) + E0 geometry npz.

PER-CONCEPT PAGE  <out>/concepts/<family>.<class>.png  (2 x 3 grid):
  (a) dose-response: e2_cloze target-vs-sibling completion-logprob delta
      (mean over templates, 95% normal CI whiskers) vs dose factor at the
      card's best layer, arms ridge/dom/rand(mean of 5 dirs); intensity axes
      show the expected ordinal rank E[rank] instead (same convention).
  (b) necessity bars: E4 everywhere-ablation diag_lp_delta per arm incl. the
      5 random and 3 other-concept controls (bootstrap CI whiskers); dashed
      line = 5x the random-arm mean (the frozen necessity bar). Falls back to
      bpt_delta (bits/token) when the concept has no diagnostic-token rows.
  (c) layer story, three side-by-side heatmaps sharing the probed-layer axes:
      (c1) E5 erasure-propagation deficit at readout l' after ablating l
           (dom meter, all positions; diverging, centered 0),
      (c2) E5 copy matrix C[l,l'] identity share (centered at 0.5: >0.5 =
           mostly copied along the residual stream, <0.5 = recomputed;
           weighted-median estimator, R<L undefined),
      (c3) E0 cross-layer cosine cos(w_l, w_l') (standardized space) — the
           correlational counterpart the copy matrix is compared against.
      The corrected causally-salient layer (analyze.py rule) is marked.
  (d) family panel: E4 effective-causal-rank curves (family cloze accuracy
      vs erased rank k, concept subspace vs matched-rank random control) for
      multiclass families; ordinal dose-monotonicity (Spearman vs layer, bar
      per arm, 0.9 bar marked) for intensity axes.

FLEET FIGURES (written to <out>/):
  fleet_arm_scatter.png     ridge vs dom dose-response slope per concept at
                            the card layer (cloze slope; ordinal slope for
                            intensity axes, triangle markers), colored by
                            family (fixed order), y=x reference.
  fleet_copy_vs_cosine.png  C[l,l'] vs E0 cos(w_l, w_l') over all (concept,
                            l, l' > l) pairs, colored by layer gap l'-l;
                            Pearson r annotated — does direction similarity
                            predict causal copying?
  fleet_offtarget.png       merged off-target matrix: steered concepts x 64
                            readout probes (ridge steer, dom meter readout,
                            factor from the offtarget npz), target cells
                            outlined; readouts grouped by family.
  causal_rollup.md/.html    per-concept verdict table, artifact-suspect
                            first, color-coded like the Stage-6 rollup.

Usage:
  python figures.py --causal-cards out/causal_cards.json --roots out
  python figures.py --causal-cards out/analysis_pilot/causal_cards.json \
      --roots <pilot_out>,out --out out/analysis_pilot/figures
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.colors import TwoSlopeNorm                      # noqa: E402
from matplotlib.patches import Rectangle                        # noqa: E402
from tqdm import tqdm                                           # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from analyze import Store, canon, row_index, heartbeat  # noqa: E402

STAGE_DIR = CODE_DIR.parent
SCRIPT = "figures"

ARM_COL = {"ridge": "#1965b0", "dom": "#e8601c", "rand": "#777777",
           "other": "#882e72"}
VCOL = {"causal": "#4eb265", "read-only": "#f1932d",
        "artifact-suspect": "#dc050c"}
VORDER = {"artifact-suspect": 0, "read-only": 1, "causal": 2}
GRID = dict(color="#dddddd", lw=0.5)


def _style(ax):
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _na(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9,
            color="#555555", wrap=True)
    ax.set_axis_off()


def mean_ci(x, axis=-1):
    """(mean, half-width of 95% normal CI) over finite entries along axis."""
    x = np.asarray(x, float)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(x, axis=axis)
        n = np.isfinite(x).sum(axis=axis)
        sd = np.nanstd(x, axis=axis, ddof=1)
    hw = 1.96 * sd / np.sqrt(np.maximum(n, 1))
    return m, np.where(n > 1, hw, np.nan)


# ------------------------------------------------------------------ panel (a)
def panel_dose(ax, store: Store, card):
    fam, cls = card["family"], card["concept"]
    suff = card.get("sufficiency") or {}
    L = suff.get("best_layer")
    z = store.npz("e2_cloze", fam)
    have_npz = (z is not None and L is not None
                and canon(cls) in [canon(c) for c in z["classes"]])
    if not have_npz:
        # graceful fallback: the raw per-factor curves live in the npz (pod/
        # HF); locally we can still show the slope-vs-layer profile from the
        # merged summary rows.
        rows = store.rows("e2_cloze", fam, cls)
        idx = row_index(rows)
        metric = ("cloze_slope" if idx.get(("cloze_slope", "ridge"))
                  else "ordinal_slope")
        layers = sorted({int(r["layer"]) for r in rows
                         if r["metric"] == metric})
        if not layers:
            _na(ax, "no e2_cloze data (npz not local, no summary rows)")
            return
        for arm in ("ridge", "dom", "rand"):
            blk = idx.get((metric, arm), {})
            m = [(blk.get(str(l), {}) or {}).get("value") for l in layers]
            lo = [(blk.get(str(l), {}) or {}).get("ci_low") for l in layers]
            hi = [(blk.get(str(l), {}) or {}).get("ci_high") for l in layers]
            m = np.array([v if v is not None else np.nan for v in m])
            yerr = np.vstack(
                [m - np.array([v if v is not None else np.nan for v in lo]),
                 np.array([v if v is not None else np.nan for v in hi]) - m])
            ax.errorbar(layers, m, yerr=yerr, color=ARM_COL[arm], lw=2,
                        marker="o", ms=4, capsize=3, label=arm)
        ax.axhline(0, color="#aaaaaa", lw=0.8)
        if L is not None and L in layers:
            ax.axvline(L, color="#4eb265", lw=1, ls=":", label=f"best L{L}")
        ax.set_xlabel("layer")
        ax.set_ylabel(f"{metric} (from summary; npz not local)")
        ax.set_title("(a) dose-response slope by layer", fontsize=10)
        ax.legend(fontsize=7)
        _style(ax)
        return
    classes = [canon(c) for c in z["classes"]]
    ci = classes.index(canon(cls))
    layers = [int(x) for x in z["layers"]]
    if L not in layers:
        _na(ax, f"best layer L{L} not in npz layers")
        return
    li = layers.index(L)
    dirs = [str(d) for d in z["dirs"]]
    fac = np.asarray(z["factors"], float)
    order = np.argsort(fac)
    intensity = bool(suff.get("intensity_axis"))
    if not intensity and z["d_sum"].shape[-1] > 0:
        data = z["d_sum"][ci, li]                    # [D, F, T]
        ylab = "cloze Δ (target − siblings, logprob)"
    else:
        er = z["erank_sum"][ci, li]                  # [D, F, P, S]
        data = er.reshape(er.shape[0], er.shape[1], -1)
        ylab = "expected ordinal rank E[rank]"
    groups = {"ridge": [dirs.index("ridge")] if "ridge" in dirs else [],
              "dom": [dirs.index("dom")] if "dom" in dirs else [],
              "rand": [i for i, d in enumerate(dirs) if d.startswith("rand")]}
    for arm, sel in groups.items():
        if not sel:
            continue
        vals = np.nanmean(data[sel], axis=0)         # [F, T-or-PS]
        m, hw = mean_ci(vals, axis=-1)
        ax.errorbar(fac[order], m[order], yerr=hw[order], color=ARM_COL[arm],
                    lw=2, marker="o", ms=4, capsize=3, label=arm,
                    alpha=0.95 if arm != "rand" else 0.8)
    ax.axvline(0, color="#aaaaaa", lw=0.8)
    fit_lo, fit_hi = -2, 2
    ax.axvspan(fit_lo, fit_hi, color="#1965b0", alpha=0.05, lw=0,
               label="slope fit range")
    ax.set_xlabel("dose factor (× s95)")
    ax.set_ylabel(ylab)
    ax.set_title(f"(a) dose-response @L{L}", fontsize=10)
    ax.legend(fontsize=7, loc="best")
    _style(ax)


# ------------------------------------------------------------------ panel (b)
def panel_necessity(ax, store: Store, card):
    fam, cls = card["family"], card["concept"]
    z = store.npz("e4", f"{fam}.ablate")
    if z is None:
        _na(ax, "E4 ablate npz not available")
        return
    classes = [canon(c) for c in z["classes"]]
    if canon(cls) not in classes:
        _na(ax, "concept not in local E4 npz")
        return
    ci = classes.index(canon(cls))
    arms = [str(a) for a in z["arms"]]
    vals = np.asarray(z["diag_lp_delta"][ci], float)
    lo = np.asarray(z["diag_lo"][ci], float)
    hi = np.asarray(z["diag_hi"][ci], float)
    metric = "diag_lp_delta (Δ logP of diagnostic tokens)"
    if not np.isfinite(vals).any():                  # ordinal families
        vals = np.asarray(z["bpt_delta"][ci], float)
        lo = np.asarray(z["bpt_lo"][ci], float)
        hi = np.asarray(z["bpt_hi"][ci], float)
        metric = "bpt_delta (bits/token; no diag tokens)"
    if not np.isfinite(vals).any():
        _na(ax, "no finite E4 necessity metrics")
        return
    cols = [ARM_COL.get(a.rstrip("01234"), "#777777") for a in arms]
    x = np.arange(len(arms))
    yerr = np.vstack([np.where(np.isfinite(lo), vals - lo, 0),
                      np.where(np.isfinite(hi), hi - vals, 0)])
    ax.bar(x, np.nan_to_num(vals), color=cols, width=0.72,
           edgecolor="white", linewidth=1)
    ax.errorbar(x, vals, yerr=yerr, fmt="none", ecolor="#333333",
                elinewidth=0.8, capsize=2)
    rand_vals = [v for a, v in zip(arms, vals)
                 if a.startswith("rand") and np.isfinite(v)]
    if rand_vals and metric.startswith("diag"):
        rm = float(np.mean(rand_vals))
        ax.axhline(-5 * abs(rm), color="#dc050c", ls="--", lw=1,
                   label=f"necessity bar (−5×|rand mean|={-5 * abs(rm):.3f})")
        ax.legend(fontsize=7)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(metric, fontsize=8)
    ax.set_title("(b) everywhere-ablation necessity", fontsize=10)
    _style(ax)


# --------------------------------------------------------------- panels (c1-3)
def _heat(ax, M, layers, title, norm, cmap, corr_layer=None, cbar=True):
    im = ax.imshow(M, origin="upper", cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")
    n = len(layers)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(layers, fontsize=6)
    ax.set_yticklabels(layers, fontsize=6)
    ax.set_xlabel("readout / second layer l'", fontsize=8)
    ax.set_ylabel("ablation / first layer l", fontsize=8)
    ax.set_title(title, fontsize=9)
    if corr_layer is not None and corr_layer in layers:
        i = layers.index(corr_layer)
        ax.add_patch(Rectangle((-0.5, i - 0.5), n, 1, fill=False,
                               edgecolor="#4eb265", lw=1.6))
        ax.text(0.1, i - 0.65, "salient (corrected)", fontsize=6,
                color="#2a7d3f", va="bottom")
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.tick_params(
            labelsize=6)


def panels_layer_story(axs, store: Store, e0, card):
    fam, cls = card["family"], card["concept"]
    story = card.get("layer_story") or {}
    corr = story.get("e5_salient_layer_corrected")
    z = store.npz("e5", fam)
    pre = f"{canon(cls)}__ridge__"
    if z is not None and f"{pre}deficit" in z.files:
        layers = [int(x) for x in z[f"{pre}layers"]]
        meters = [str(m) for m in z["meters"]]
        possets = [str(p) for p in z["possets"]]
        mi, pi = meters.index("dom"), possets.index("all")
        D = z[f"{pre}deficit"][:, :, mi, pi]
        C = z[f"{pre}C"][:, :, mi, pi]
        vmax = np.nanpercentile(np.abs(D), 98) or 1e-6
        _heat(axs[0], D, layers, "(c1) E5 deficit after ablating l "
              "(dom meter)", TwoSlopeNorm(0, -vmax, vmax), "RdBu_r",
              corr_layer=corr)
        _heat(axs[1], np.clip(C, -0.5, 1.5), layers,
              "(c2) copy matrix C[l,l'] (identity share)",
              TwoSlopeNorm(0.5, -0.5, 1.5), "PuOr_r", corr_layer=corr)
    else:
        _na(axs[0], "E5 npz not available")
        _na(axs[1], "E5 npz not available")
    if e0 is not None:
        cl = [canon(c) for c in e0["classes"]]
        fa = [str(f) for f in e0["families"]]
        hit = [i for i in range(len(cl))
               if cl[i] == canon(cls) and fa[i] == fam]
        if hit:
            layers0 = [int(x) for x in e0["layers"]]
            M = e0["cos_std"][hit[0]]
            _heat(axs[2], M, layers0, "(c3) E0 cosine cos(w_l, w_l') "
                  "(std space)", TwoSlopeNorm(0, -1, 1), "RdBu_r")
            return
    _na(axs[2], "E0 geometry npz not available")


# ------------------------------------------------------------------ panel (d)
def panel_family(ax, store: Store, card):
    fam, cls = card["family"], card["concept"]
    suff = card.get("sufficiency") or {}
    if suff.get("intensity_axis"):
        rows = store.rows("e2_cloze", fam, cls)
        idx = row_index(rows)
        layers = sorted({int(r["layer"]) for r in rows
                         if r["metric"] == "ordinal_spearman"})
        if not layers:
            _na(ax, "no ordinal rows")
            return
        w = 0.27
        for k, arm in enumerate(("ridge", "dom", "rand")):
            vals = [(idx.get(("ordinal_spearman", arm), {})
                     .get(str(L), {}) or {}).get("value") for L in layers]
            vals = [v if v is not None else np.nan for v in vals]
            ax.bar(np.arange(len(layers)) + (k - 1) * w, vals, width=w,
                   color=ARM_COL[arm], label=arm, edgecolor="white", lw=0.5)
        ax.axhline(0.9, color="#dc050c", ls="--", lw=1,
                   label="sufficiency bar (ρ=0.9)")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=7)
        ax.set_xlabel("layer")
        ax.set_ylabel("Spearman(dose, E[rank])")
        ax.set_ylim(-1.05, 1.1)
        ax.set_title("(d) ordinal dose-monotonicity by layer", fontsize=10)
        ax.legend(fontsize=7)
        _style(ax)
        return
    z = store.npz("e4", f"{fam}.rank")
    if z is None:
        _na(ax, "E4 family-rank curves not available\n(not run for this "
                "family or npz not local)")
        return
    ks = [int(k) for k in z["ks"]]
    bases = [str(b) for b in z["bases"]]
    acc = np.asarray(z["cloze_acc"], float)
    acc_clean = float(z["cloze_acc_clean"])
    ax.axhline(acc_clean, color="#4eb265", lw=1.2, ls=":",
               label=f"clean acc {acc_clean:.2f}")
    for b, col, lab in ((bases.index("concept"), ARM_COL["ridge"],
                         "family concept subspace"),
                        (bases.index("random"), ARM_COL["rand"],
                         "matched-rank random")):
        ax.plot(ks, acc[:, b], "o-", color=col, lw=2, ms=4, label=lab)
    rk = card.get("family_causal_rank") or {}
    for key, ls in (("k50", "--"), ("k90", "-.")):
        if rk.get(key):
            ax.axvline(rk[key], color="#f1932d", ls=ls, lw=1,
                       label=f"{key}={rk[key]}")
    ax.set_xlabel("erased rank k")
    ax.set_ylabel("family cloze accuracy")
    ax.set_title("(d) effective causal rank (family)", fontsize=10)
    ax.set_xticks(ks)
    ax.legend(fontsize=7)
    _style(ax)


# --------------------------------------------------------------- concept page
def concept_page(store: Store, e0, card, out_dir: Path):
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32,
                          left=0.06, right=0.98, top=0.90, bottom=0.08)
    story = card.get("layer_story") or {}
    fig.suptitle(
        f"{card['concept']} ({card['family']}) — Stage-6 "
        f"{card['stage6']['verdict'].upper()} @L{card['stage6']['layer']} — "
        f"causal verdict: {card['verdict'].upper()} (dom arm: "
        f"{card['dom_verdict']}) — salient L raw {story.get('e5_salient_layer_raw')}"
        f" → corrected {story.get('e5_salient_layer_corrected')}",
        fontsize=11)
    panel_dose(fig.add_subplot(gs[0, 0]), store, card)
    panel_necessity(fig.add_subplot(gs[0, 1]), store, card)
    panel_family(fig.add_subplot(gs[0, 2]), store, card)
    axs = [fig.add_subplot(gs[1, j]) for j in range(3)]
    panels_layer_story(axs, store, e0, card)
    out = out_dir / "concepts" / f"{card['family']}.{card['concept']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ------------------------------------------------------------- fleet figures
def fam_colors(families):
    fams = sorted(set(families))
    cmap = plt.get_cmap("tab20")
    return {f: cmap(i % 20) for i, f in enumerate(fams)}


def fleet_arm_scatter(cards, out_dir: Path):
    pts = []
    for c in cards:
        s = c.get("sufficiency") or {}
        blk = s.get("cloze_slope") or s.get("ordinal_slope")
        if not blk:
            continue
        rx = (blk.get("ridge") or {}).get("value")
        dy = (blk.get("dom") or {}).get("value")
        if rx is None or dy is None:
            continue
        pts.append((rx, dy, c["family"], c["concept"],
                    bool(s.get("intensity_axis"))))
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(7, 6.4))
    fc = fam_colors([p[2] for p in pts])
    lim = max(abs(v) for p in pts for v in p[:2]) * 1.15 or 1
    ax.plot([-lim, lim], [-lim, lim], color="#bbbbbb", lw=0.8, zorder=1)
    ax.axhline(0, color="#dddddd", lw=0.8)
    ax.axvline(0, color="#dddddd", lw=0.8)
    seen = set()
    for rx, dy, fam, cls, inten in pts:
        lab = fam if fam not in seen else None
        seen.add(fam)
        ax.scatter([rx], [dy], s=48, marker="^" if inten else "o",
                   color=fc[fam], edgecolor="white", lw=0.8, label=lab,
                   zorder=3)
        ax.annotate(cls, (rx, dy), fontsize=6, xytext=(3, 3),
                    textcoords="offset points", color="#444444")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("ridge-arm dose-response slope")
    ax.set_ylabel("DoM-arm dose-response slope")
    ax.set_title("reading vs steering, per concept (△ = ordinal slope, "
                 "intensity axes)", fontsize=10)
    ax.legend(fontsize=7, loc="best", title="family", title_fontsize=7)
    _style(ax)
    out = out_dir / "fleet_arm_scatter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fleet_copy_vs_cos(cards, store: Store, e0, out_dir: Path):
    if e0 is None:
        return None
    cl = [canon(c) for c in e0["classes"]]
    fa = [str(f) for f in e0["families"]]
    layers0 = [int(x) for x in e0["layers"]]
    xs, ys, gaps = [], [], []
    for c in cards:
        z = store.npz("e5", c["family"])
        pre = f"{c['concept']}__ridge__"
        if z is None or f"{pre}C" not in z.files:
            continue
        hit = [i for i in range(len(cl))
               if cl[i] == c["concept"] and fa[i] == c["family"]]
        if not hit:
            continue
        cos = e0["cos_std"][hit[0]]
        layers = [int(x) for x in z[f"{pre}layers"]]
        meters = [str(m) for m in z["meters"]]
        possets = [str(p) for p in z["possets"]]
        C = z[f"{pre}C"][:, :, meters.index("dom"), possets.index("all")]
        for li, L in enumerate(layers):
            for ri, R in enumerate(layers):
                if R <= L or L not in layers0 or R not in layers0:
                    continue
                cv = C[li, ri]
                if not np.isfinite(cv):
                    continue
                xs.append(cos[layers0.index(L), layers0.index(R)])
                ys.append(cv)
                gaps.append(R - L)
    if not xs:
        return None
    xs, ys, gaps = map(np.asarray, (xs, ys, gaps))
    r = np.corrcoef(xs, np.clip(ys, -1, 2))[0, 1] if len(xs) > 2 else np.nan
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(xs, np.clip(ys, -0.5, 1.75), c=gaps, cmap="viridis",
                    s=16, alpha=0.75, edgecolor="none")
    plt.colorbar(sc, ax=ax, label="layer gap l' − l", fraction=0.046)
    ax.axhline(1, color="#bbbbbb", lw=0.8, ls=":")
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.set_xlabel("E0 cosine cos(w_l, w_l')  (correlational)")
    ax.set_ylabel("E5 copy matrix C[l,l']  (causal identity share, clipped)")
    ax.set_title(f"does direction similarity predict copying?  "
                 f"Pearson r = {r:.2f}  (n={len(xs)} (concept,l,l') pairs)",
                 fontsize=10)
    _style(ax)
    out = out_dir / "fleet_copy_vs_cosine.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fleet_offtarget(cards, store: Store, out_dir: Path):
    fams = sorted({c["family"] for c in cards})
    rows, row_labels, col_meta = [], [], None
    for fam in fams:
        z = store.npz("e2_cloze", f"offtarget_{fam}")
        if z is None:
            continue
        meters = [str(m) for m in z["meters"]]
        arms = [str(a) for a in z["steer_arms"]]
        rc = [canon(x) for x in z["readout_concepts"]]
        rf = [str(x) for x in z["readout_families"]]
        if col_meta is None:
            order = sorted(range(len(rc)), key=lambda j: (rf[j], rc[j]))
            col_meta = (order, [rc[j] for j in order], [rf[j] for j in order])
        order = col_meta[0]
        for si, s in enumerate([canon(x) for x in z["steered"]]):
            d = z["delta"][si, arms.index("ridge"), :, meters.index("dom")]
            rows.append(np.asarray(d, float)[order])
            row_labels.append((fam, s))
    if not rows:
        return None
    M = np.vstack(rows)
    order, cols, colfams = col_meta
    ridx = sorted(range(len(row_labels)), key=lambda i: row_labels[i])
    M = M[ridx]
    row_labels = [row_labels[i] for i in ridx]
    fig, ax = plt.subplots(
        figsize=(max(10, 0.22 * len(cols)), max(3.2, 0.34 * len(M) + 1.8)))
    vmax = np.nanpercentile(np.abs(M), 99) or 1e-6
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax),
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=5)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f"{f}.{c}" for f, c in row_labels], fontsize=6)
    for i, (f, c) in enumerate(row_labels):        # outline target cells
        for j in range(len(cols)):
            if cols[j] == c and colfams[j] == f:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#4eb265", lw=1.4))
    # family separators on the readout axis
    for j in range(1, len(cols)):
        if colfams[j] != colfams[j - 1]:
            ax.axvline(j - 0.5, color="#999999", lw=0.4)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
                 label="probe-readout Δ (ridge steer, dom meter)")
    ax.set_title("off-target readout matrix (steered concept × 64 probes, "
                 "factor 2 @card layer; green = target cell)", fontsize=10)
    out = out_dir / "fleet_offtarget.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- rollup table
def _fmt(v, nd=3):
    if v is None:
        return "NA"
    return f"{v:.{nd}f}"


def _ci(blk):
    if not blk or blk.get("value") is None:
        return "NA"
    s = f"{blk['value']:.3f}"
    if blk.get("ci_low") is not None:
        s += f" [{blk['ci_low']:.3f},{blk['ci_high']:.3f}]"
    return s


def rollup(cards, out_dir: Path):
    cards = sorted(cards, key=lambda c: (VORDER[c["verdict"]], c["family"],
                                         c["concept"]))
    hdr = ["concept", "family", "s6", "bestL", "cloze/ord slope (ridge)",
           "anti", "nec ridge", "nec rand", "spec ratio", "k50/k90",
           "salient raw→corr", "verdict", "dom", "missing"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    html = ["<table border=1 cellpadding=3 style='border-collapse:collapse;"
            "font-family:monospace;font-size:12px'>",
            "<tr>" + "".join(f"<th>{h}</th>" for h in hdr) + "</tr>"]
    for c in cards:
        s = c.get("sufficiency") or {}
        n = (c.get("necessity") or {}).get("diag_lp_delta") or {}
        sp = ((c.get("specificity") or {}).get("arms") or {}).get("ridge") or {}
        st = c.get("layer_story") or {}
        rk = c.get("family_causal_rank") or {}
        slope_blk = (s.get("cloze_slope") or s.get("ordinal_slope") or {})
        anti = ((s.get("anti_steerable_frac") or {}).get("ridge") or {}
                ).get("value")
        cells = [
            c["concept"], c["family"], c["stage6"]["verdict"],
            str(s.get("best_layer")),
            _ci(slope_blk.get("ridge")),
            _fmt(anti, 2),
            _fmt((n.get("ridge") or {}).get("value")),
            _fmt(n.get("rand_mean"), 4),
            _fmt(sp.get("ratio"), 2),
            f"{rk.get('k50')}/{rk.get('k90')}" if rk.get("k50") else "NA",
            f"{st.get('e5_salient_layer_raw')}→"
            f"{st.get('e5_salient_layer_corrected')}",
            c["verdict"], c["dom_verdict"],
            ",".join(c["criteria"]["ridge"]["missing"]) or "-",
        ]
        md.append("| " + " | ".join(str(x) for x in cells) + " |")
        vc = VCOL[c["verdict"]]
        tds = "".join(
            f"<td{' style=background:' + vc + '55' if h == 'verdict' else ''}"
            f">{x}</td>" for h, x in zip(hdr, cells))
        html.append(f"<tr style='background:{vc}22'>{tds}</tr>")
    html.append("</table>")
    (out_dir / "causal_rollup.md").write_text("\n".join(md) + "\n")
    (out_dir / "causal_rollup.html").write_text("\n".join(html) + "\n")
    return out_dir / "causal_rollup.md"


# ----------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--causal-cards",
                    default=str(STAGE_DIR / "out" / "causal_cards.json"))
    ap.add_argument("--roots", default=str(STAGE_DIR / "out"))
    ap.add_argument("--e0-dir", default=str(STAGE_DIR / "out" / "e0"))
    ap.add_argument("--out", default=str(STAGE_DIR / "figures" / "analysis"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"progress_{SCRIPT}.log"
    heartbeat(log, "start")

    blob = json.load(open(args.causal_cards))
    cards = blob["cards"] if isinstance(blob, dict) else blob
    store = Store([Path(r) for r in args.roots.split(",") if r])
    e0_path = Path(args.e0_dir) / "all_crosslayer.npz"
    e0 = np.load(e0_path, allow_pickle=True) if e0_path.exists() else None
    if e0 is None:
        print(f"[{SCRIPT}] WARNING: no E0 geometry at {e0_path}")

    for i, card in enumerate(tqdm(cards, desc="concept pages")):
        p = concept_page(store, e0, card, out_dir)
        if (i + 1) % 8 == 0:
            heartbeat(log, f"{card['family']}.{card['concept']} "
                           f"{i + 1}/{len(cards)}")
    made = [fleet_arm_scatter(cards, out_dir),
            fleet_copy_vs_cos(cards, store, e0, out_dir),
            fleet_offtarget(cards, store, out_dir),
            rollup(cards, out_dir)]
    print(f"[{SCRIPT}] wrote {len(cards)} concept pages under "
          f"{out_dir / 'concepts'}")
    for m in made:
        print(f"[{SCRIPT}] wrote {m}" if m else f"[{SCRIPT}] (skipped a fleet "
              "figure: no data)")
    heartbeat(log, "DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
