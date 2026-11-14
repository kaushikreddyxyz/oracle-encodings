#!/usr/bin/env python3
"""fig_ev_probes from the minimal-swap paired probes (probe_pairs.py). Reads per-model JSONs from --src (default: a temp dir
filled by pulling /workspace/pp_out/*.json from the pods) and writes ev_pairs.csv + fig_ev_probes.{pdf,svg} here.
  metric = "encoding": held-out R^2 of the swap's activation change explained by fixed per-wall vectors (0 = none, 1 = all)
  metric = "decoding": wall-specific AUROC of one fixed direction per wall (0.5 = no information)"""
import csv, json, re, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
COL = {"base": "#2a7a57", "edge": "#000c7a", "ego": "#eb6834"}
LABEL = {"base": "No graft", "edge": "Allocentric graft", "ego": "Egocentric graft"}
FCOL = {"allo": "#000c7a", "ego": "#eb6834"}; FLAB = {"allo": "Allocentric walls", "ego": "Egocentric walls"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "legend.frameon": False, "axes.edgecolor": "#898781", "xtick.color": "#898781", "ytick.color": "#898781", "axes.titlesize": 8.5})


def main(src, metric="encoding"):
    rows = []
    for f in sorted(Path(src).glob("g*-s*-*-e4-oe.json")):
        m = re.match(r"g(\d+)-s(\d+)-(\w+)-e4-oe", f.stem); d = json.load(open(f))
        for s, v in d["sites"].items():
            L = 0 if s == "L0in" else int(s[1:])
            for fam in ("allo", "ego"):
                e = v[fam]
                rows.append(dict(grid=int(m.group(1)), seed=int(m.group(2)), code=m.group(3), layer=L, family=fam, decoding_auroc=e["auroc"],
                                 decoding_auroc_partner_nonadjacent=e["auroc_partner_nonadjacent"], decoding_control_shuffled=e["control_shuffled_labels"],
                                 sign_acc=e["sign_acc"], encoding_r2=e.get("encoding_r2", np.nan), n_targets=e["n_targets"], pairs=d["pairs"]))
    with open(HERE / "ev_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    key = "encoding_r2" if metric == "encoding" else "decoding_auroc"
    ylab = "Linear encoding (R²)" if metric == "encoding" else "Decoding AUROC"
    layers = sorted({r["layer"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
    for ax, code in zip(axes, ("base", "edge", "ego")):
        n = len({(r["grid"], r["seed"]) for r in rows if r["code"] == code})
        for fam in ("allo", "ego"):
            vals = [[r[key] for r in rows if r["code"] == code and r["family"] == fam and r["layer"] == L and not np.isnan(r[key])] for L in layers]
            if not any(vals): continue
            mu = np.array([np.mean(v) if v else np.nan for v in vals]); sd = np.array([np.std(v) if v else np.nan for v in vals])
            ax.plot(layers, mu, color=FCOL[fam], marker="o", ms=3, lw=1.5, label=FLAB[fam]); ax.fill_between(layers, mu - sd, mu + sd, color=FCOL[fam], alpha=0.12, linewidth=0)
        ax.set_title(LABEL[code], loc="left", color=COL[code]); ax.set_xlabel("Layer"); ax.set_xticks(range(0, 13, 2))
        if metric == "encoding": ax.set_ylim(-0.02, 1.02); ax.axhline(0, color="#bbb", lw=0.8, ls=":")
        else: ax.set_ylim(0.4, 1.02); ax.axhline(0.5, color="#bbb", lw=0.8, ls=":")
    axes[0].set_ylabel(ylab)
    h, l = axes[2].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); [fig.savefig(HERE / f"fig_ev_probes.{e}", bbox_inches="tight") for e in ("pdf", "svg")]; plt.close(fig)
    print(len(rows), "rows;", len({(r['grid'], r['seed'], r['code']) for r in rows}), "models")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "encoding")
