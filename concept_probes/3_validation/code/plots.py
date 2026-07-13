"""Stage 6 presentation pyramid (§6.8).

Level 1: roll-up — 64 rows, verdict-sorted (rejected first), color-coded HTML +
markdown. Level 2: one page per concept — layer profile (ρ + CI band +
selectivity gap, chosen candidate starred), selectivity four-bar, reliability
diagram (post-calibration, TEST half), family geometry (ring / monotonicity).

  python plots.py --families ... --gates-dir ... --natscores-dir ... \
      --probes-root ... --out 3_validation/reports
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gates import isotonic_fit, INTENSITY

CYCLES = {
    "months": ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"],
    "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday",
                 "saturday", "sunday"],
    "seasons": ["spring", "summer", "autumn", "winter"],
    "color_wheel": ["red", "red-orange", "orange", "yellow-orange", "yellow",
                    "yellow-green", "green", "blue-green", "blue",
                    "blue-violet", "violet", "red-violet"],
    "directions": ["north", "northeast", "east", "southeast", "south",
                   "southwest", "west", "northwest"],
    "moon_phases": ["new moon", "waxing crescent", "first quarter",
                    "waxing gibbous", "full moon", "waning gibbous",
                    "last quarter", "waning crescent"],
}


def layer_profile(ax, cls_gates, layers):
    Ls = layers
    rho = [cls_gates["per_layer"][str(L)]["rho_nat_test"] for L in Ls]
    gap = [(cls_gates["per_layer"][str(L)]["selectivity"] or {}).get("gap") for L in Ls]
    ax.plot(Ls, rho, "o-", label="ρ natural (test)", color="#1965b0")
    if any(g is not None for g in gap):
        ax.plot(Ls, [g if g is not None else np.nan for g in gap], "s--",
                label="selectivity gap (gen)", color="#dc050c")
    ch = cls_gates["chosen_layer"]
    ax.plot([ch], [cls_gates["tier1"]["rho_nat_test"]], "*", ms=18,
            color="#f1932d", zorder=5, label=f"chosen L{ch}")
    ci = cls_gates["tier2"]["rho_ci95"]
    ax.errorbar([ch], [cls_gates["tier1"]["rho_nat_test"]],
                yerr=[[cls_gates["tier1"]["rho_nat_test"] - ci[0]],
                      [ci[1] - cls_gates["tier1"]["rho_nat_test"]]],
                color="#f1932d", capsize=4)
    ax.axhline(0.65, color="green", lw=0.6, ls=":"); ax.axhline(0.45, color="orange", lw=0.6, ls=":")
    ax.set_xlabel("layer"); ax.set_ylabel("metric"); ax.legend(fontsize=7)
    ax.set_ylim(-0.1, 1.0)


def four_bar(ax, cls_gates):
    sel = cls_gates["per_layer"][str(cls_gates["chosen_layer"])]["selectivity"]
    if not sel:
        ax.text(0.5, 0.5, "selectivity NA", ha="center"); return
    raw = sel["raw"]
    names = ["G-ratio\n(≥.80)", "R_implicit\n(≥.50)", "FPR_hom\n(≤.10)", "HL S\n(>0 sanity)"]
    vals = [raw.get("g_ratio"), raw.get("implicit_recall"),
            raw.get("homograph_fpr"), raw.get("hl_selectivity")]
    ok = [lambda v: v >= .8, lambda v: v >= .5, lambda v: v <= .1, lambda v: v > 0]
    cols = ["#4eb265" if (v is not None and f(v)) else "#dc050c"
            for v, f in zip(vals, ok)]
    ax.bar(names, [v if v is not None else 0 for v in vals], color=cols)
    for i, v in enumerate(vals):
        ax.text(i, (v or 0) + .02, "NA" if v is None else f"{v:.2f}",
                ha="center", fontsize=8)
    ax.set_ylim(0, 1.15)


def reliability(ax, nat, ci_idx, li, cls_gates):
    y = nat["y"][:, ci_idx]
    t2e = nat["token2ex"]; splits = nat["ex_nat_split"]
    cal = np.isin(t2e, np.flatnonzero(splits == "cal"))
    test = ~cal
    p_cal = nat["preds_ridge"][li, cal, ci_idx]
    if np.std(p_cal) == 0:
        ax.text(0.5, 0.5, "degenerate preds", ha="center"); return
    bx, by = isotonic_fit(p_cal, y[cal])
    p = np.interp(nat["preds_ridge"][li, test, ci_idx], bx, by)
    yt = y[test]
    edges = np.linspace(0, 1, 11)
    xs, ys, ns = [], [], []
    for i in range(10):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < 9 else p <= 1)
        if m.sum() >= 5:
            xs.append(p[m].mean()); ys.append(yt[m].mean()); ns.append(int(m.sum()))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.plot(xs, ys, "o-", color="#1965b0")
    ax.set_xlabel("calibrated strength"); ax.set_ylabel("judge-truth mean")
    ax.set_title(f"ECE={cls_gates['tier1']['ece_test']:.3f}", fontsize=9)


def geometry(ax, fam, fam_gates, probes_root, layers):
    if fam in INTENSITY:
        cls = next(iter(fam_gates))
        mono = fam_gates[cls]["tier2"].get("monotonicity")
        ax.text(0.5, 0.5, f"monotonicity ρ = {mono if mono is None else round(mono, 3)}",
                ha="center", va="center")
        ax.set_axis_off()
        return
    cyc = CYCLES.get(fam)
    if not cyc:
        ax.set_axis_off(); return
    # ring: PCA of chosen-layer (majority) unit rows
    from collections import Counter
    maj = Counter(g["chosen_layer"] for g in fam_gates.values()).most_common(1)[0][0]
    z = np.load(Path(probes_root) / fam / f"probes_l{maj}.npz")
    classes = [str(c) for c in z["classes"]]
    W = []
    for c in classes:
        ci = classes.index(c)
        if "chosen_lambda_ridge" in z:
            li = int(z["chosen_lambda_ridge"][ci])
            w = z["W_ridge"][li, ci]
        else:
            li = int(np.bincount(z["chosen_lambda_idx"][:, ci]).argmax())
            w = z["W_adam"][:, li, ci].mean(0)
        W.append(w / np.linalg.norm(w))
    W = np.stack(W)
    Wc = W - W.mean(0)
    U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
    xy = Wc @ Vt[:2].T
    order = [classes.index(c) for c in cyc if c in classes]
    ax.plot(xy[order + order[:1], 0], xy[order + order[:1], 1], "-", lw=0.7,
            color="#cae0ab")
    ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(len(classes)), cmap="hsv", s=40)
    for i, c in enumerate(classes):
        ax.annotate(c, xy[i], fontsize=6)
    # adjacency preservation
    nn = 0
    for k, c in enumerate(cyc):
        if c not in classes:
            continue
        i = classes.index(c)
        d = np.linalg.norm(xy - xy[i], axis=1); d[i] = np.inf
        j = int(np.argmin(d))
        neigh = {cyc[(k - 1) % len(cyc)], cyc[(k + 1) % len(cyc)]}
        nn += classes[j] in neigh
    ax.set_title(f"ring @L{maj}: {nn}/{len(cyc)} NN cycle-adjacent", fontsize=9)


VERDICT_ORDER = {"reject": 0, "caveat": 1, "deploy": 2}
VCOL = {"deploy": "#4eb265", "caveat": "#f1932d", "reject": "#dc050c"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True)
    ap.add_argument("--gates-dir", required=True)
    ap.add_argument("--natscores-dir", required=True)
    ap.add_argument("--probes-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="1,3,6,8,10,12,14,16,18,20,23,25")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    outdir = Path(args.out); (outdir / "concepts").mkdir(parents=True, exist_ok=True)

    rollup = json.load(open(Path(args.gates_dir) / "rollup.json"))
    rollup.sort(key=lambda r: (VERDICT_ORDER[r["verdict"]], r["family"], r["class"]))

    # Level 1
    md = ["| concept | family | layer | ρ_nat | sel.gap | ECE | dom.min | margin | verdict | fail |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    html = ["<table border=1 cellpadding=3 style='border-collapse:collapse;font-family:monospace;font-size:12px'>",
            "<tr><th>concept</th><th>family</th><th>layer</th><th>ρ_nat</th><th>sel.gap</th>"
            "<th>ECE</th><th>dom.min</th><th>margin</th><th>verdict</th><th>fail</th></tr>"]
    fmt = lambda v: "NA" if v is None else f"{v:.3f}"
    for r in rollup:
        md.append(f"| {r['class']} | {r['family']} | {r['chosen_layer']} | "
                  f"{fmt(r['rho_nat_test'])} | {fmt(r['selectivity_gap'])} | "
                  f"{fmt(r['ece_test'])} | {fmt(r['per_domain_min_rho'])} | "
                  f"{fmt(r['rand_margin'])} | {r['verdict']} | {r['fail_reason'] or ''} |")
        html.append(f"<tr style='background:{VCOL[r['verdict']]}22'>"
                    f"<td>{r['class']}</td><td>{r['family']}</td><td>{r['chosen_layer']}</td>"
                    f"<td>{fmt(r['rho_nat_test'])}</td><td>{fmt(r['selectivity_gap'])}</td>"
                    f"<td>{fmt(r['ece_test'])}</td><td>{fmt(r['per_domain_min_rho'])}</td>"
                    f"<td>{fmt(r['rand_margin'])}</td>"
                    f"<td style='background:{VCOL[r['verdict']]}55'>{r['verdict']}</td>"
                    f"<td>{r['fail_reason'] or ''}</td></tr>")
    html.append("</table>")
    (outdir / "rollup.md").write_text("\n".join(md))
    (outdir / "rollup.html").write_text("\n".join(html))

    # Level 2
    for fam in args.families.split(","):
        fam_gates = json.load(open(Path(args.gates_dir) / f"{fam}.gates.json"))
        nat = np.load(Path(args.natscores_dir) / f"{fam}.natscores.npz")
        classes = [str(c) for c in nat["classes"]]
        for cls, g in fam_gates.items():
            fig, axs = plt.subplots(2, 2, figsize=(11, 8))
            fig.suptitle(f"{cls} ({fam}) — chosen L{g['chosen_layer']} — "
                         f"{g['verdict'].upper()}  [n_test tokens; natural split "
                         f"= ClimbMix shards 311–316]", fontsize=11)
            layer_profile(axs[0, 0], g, layers)
            four_bar(axs[0, 1], g)
            reliability(axs[1, 0], nat, classes.index(cls),
                        layers.index(g["chosen_layer"]), g)
            geometry(axs[1, 1], fam, fam_gates, args.probes_root, layers)
            fig.tight_layout()
            fig.savefig(outdir / "concepts" / f"{fam}.{cls.replace(' ', '_')}.png", dpi=110)
            plt.close(fig)
        print(f"plots: {fam} done")


if __name__ == "__main__":
    main()
