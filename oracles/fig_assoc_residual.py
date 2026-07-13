#!/usr/bin/env python3
"""Residual between the gold probes' intrinsic cross-concept correlation and
the per-layer oracle association matrices.

  residual[i,j] = goldcorr[i,j] - assoc[i,j]

goldcorr: 54x54 Spearman of GOLD probe scores across tokens (climbmix-scored
2M-row sample, computed by the broad-detector audit). assoc: Spearman of
oracle-PREDICTED concept i vs GOLD concept j on corpus-scores val tokens
(out/figures/oracle_perlayer_assoc_matrices.npz). Different token samples,
both ClimbMix; concept order verified identical.

Reading: diagonal = 1 - assoc_diag (the oracle's fit deficit, positive).
Off-diagonal < 0 = oracle predictions carry MORE cross-concept association
than the gold probes intrinsically have (oracle blurs concepts); > 0 = less.

Usage: fig_assoc_residual.py --goldcorr <computed.npz with spearman[3,54,54]>
"""
import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "out", "figures")
LAYERS = ["L6", "L8", "L14"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldcorr", required=True,
                    help="npz with spearman [3,54,54] (L6,L8,L14) + concepts + families")
    args = ap.parse_args()

    assoc = np.load(os.path.join(FIGDIR, "oracle_perlayer_assoc_matrices.npz"), allow_pickle=True)
    gold = np.load(args.goldcorr, allow_pickle=True)
    conc_a = [str(x) for x in assoc["concepts"]]
    conc_g = [str(x) for x in gold["concepts"]]
    assert conc_a == conc_g, "concept order mismatch between assoc and goldcorr"
    conc = conc_a
    fams = [str(x) for x in gold["families"]]

    R = {ly: gold["spearman"][li] - assoc[ly] for li, ly in enumerate(LAYERS)}
    vmax = max(np.abs(R[ly]).max() for ly in LAYERS)
    vmax = float(np.ceil(vmax * 20) / 20)  # round up to 0.05

    bnd = [i - 0.5 for i in range(1, 54) if fams[i] != fams[i - 1]]
    edges = [0] + [int(b + 0.5) for b in bnd] + [54]
    fam_order = []
    for f in fams:
        if f not in fam_order:
            fam_order.append(f)

    off = ~np.eye(54, dtype=bool)
    fig, axes = plt.subplots(1, 3, figsize=(23, 8.6), gridspec_kw={"wspace": 0.06})
    for li, (ax, ly) in enumerate(zip(axes, LAYERS)):
        M = R[ly]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        for b in bnd:
            ax.axhline(b, color="#222", lw=1.0)
            ax.axvline(b, color="#222", lw=1.0)
        for k in range(len(edges) - 1):
            mid = (edges[k] + edges[k + 1] - 1) / 2
            ax.text(mid, 55.8, fam_order[k], ha="center", va="top", fontsize=7.5, rotation=90, color="#555")
        ax.set_xticks(range(54))
        ax.set_xticklabels([c.replace("_", " ") for c in conc], rotation=90, fontsize=4.6)
        if li == 0:
            ax.set_yticks(range(54))
            ax.set_yticklabels([c.replace("_", " ") for c in conc], fontsize=4.6)
            for k in range(len(edges) - 1):
                mid = (edges[k] + edges[k + 1] - 1) / 2
                ax.text(-7.5, mid, fam_order[k], ha="right", va="center", fontsize=8, fontweight="bold", color="#555")
        else:
            ax.set_yticks([])
        md = np.median(np.diag(M))
        mo = np.median(M[off])
        ax.set_title(f"{ly}\nmedian diag {md:+.3f}  ·  median off-diag {mo:+.3f}",
                     fontsize=10, loc="left", pad=8)
        ax.set_xlim(-0.5, 53.5)
        ax.set_ylim(53.5, -0.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.015)
    cb.set_label(f"residual  (gold–gold ρ  −  oracle assoc ρ),  scale ±{vmax:.2f}", fontsize=9)
    fig.suptitle(
        "Residual: gold cross-concept correlation − oracle association, per layer\n"
        "diag > 0 = oracle fit deficit (1 − assoc diag); off-diag < 0 (blue) = oracle predictions "
        "more cross-concept-correlated than gold probes intrinsically are",
        fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=1.04)
    out_png = os.path.join(FIGDIR, "oracle_assoc_gold_residual.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    np.savez_compressed(
        os.path.join(FIGDIR, "oracle_assoc_gold_residual_matrices.npz"),
        concepts=np.array(conc), families=np.array(fams),
        goldcorr_L6=gold["spearman"][0], goldcorr_L8=gold["spearman"][1], goldcorr_L14=gold["spearman"][2],
        residual_L6=R["L6"], residual_L8=R["L8"], residual_L14=R["L14"],
        note=np.array("residual = goldcorr(climbmix scores_00010 2M-row sample) - assoc(corpus-scores val); "
                      "assoc rows=oracle-predicted concept, cols=gold concept"))
    print("wrote", out_png)
    for ly in LAYERS:
        M = R[ly]
        print(f"{ly}: median diag {np.median(np.diag(M)):+.4f} | median off-diag {np.median(M[off]):+.4f} "
              f"| off-diag range [{M[off].min():+.3f},{M[off].max():+.3f}]")


if __name__ == "__main__":
    main()
