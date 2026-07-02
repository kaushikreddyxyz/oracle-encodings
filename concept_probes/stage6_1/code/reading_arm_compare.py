"""Ridge vs DoM as READERS on the natural TEST set (Stage 6.1 addendum).

Stage 6.0 compared ridge to its baselines on generated val ("beaten or
matched"); this script runs the comparison that was never published — on the
judge-labeled natural test half, from the stored natscores (CPU-only):

  1. token-level rank fidelity: raw Spearman(preds, y) at each concept's
     Stage-6 chosen layer (NOTE: layer was selected FOR ridge on CAL — small
     handicap for DoM);
  2. example-level detection AUROC: max-pooled token preds vs max-pooled
     judge truth binarized at POS_THRESH=0.34 (the Stage-6 convention).

Writes out/analysis/reading_arm_compare.json with per-concept rows + medians.

  python reading_arm_compare.py [--natscores DIR] [--cards PATH] [--out PATH]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

STAGE = Path(__file__).resolve().parents[1]
CP = STAGE.parent
POS_THRESH = 0.34


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--natscores", default=str(CP / "stage6/data/natscores"))
    ap.add_argument("--cards", default=str(CP / "stage6/artifacts/probe_cards.json"))
    ap.add_argument("--out", default=str(STAGE / "out/analysis/reading_arm_compare.json"))
    args = ap.parse_args()

    cards = json.load(open(args.cards))
    rows = []
    for card in cards:
        fam, cls, L = card["family"], card["concept"], card["layer"]
        z = np.load(Path(args.natscores) / f"{fam}.natscores.npz", allow_pickle=True)
        classes = [str(c).replace(" ", "_") for c in z["classes"]]
        ci = classes.index(cls.replace(" ", "_"))
        li = list(z["layers"]).index(L)
        t2e = z["token2ex"]
        splits = z["ex_nat_split"]
        test_tok = np.isin(t2e, np.flatnonzero(splits == "test"))
        y = z["y"][test_tok, ci]
        row = {"concept": cls, "family": fam, "layer": int(L)}
        for arm, key in (("ridge", "preds_ridge"), ("dom", "preds_dom")):
            p = z[key][li, test_tok, ci]
            row[f"rho_{arm}"] = float(spearmanr(p, y).statistic)
        # example-level max-pool AUROC
        nex = len(splits)
        ymax = np.zeros(nex)
        np.maximum.at(ymax, t2e, z["y"][:, ci])
        test_ex = np.flatnonzero(splits == "test")
        lab = ymax[test_ex] >= POS_THRESH
        if lab.sum() >= 5 and (~lab).sum() >= 5:
            for arm, key in (("ridge", "preds_ridge"), ("dom", "preds_dom")):
                pmax = np.full(nex, -1e9)
                np.maximum.at(pmax, t2e, z[key][li, :, ci])
                row[f"auroc_{arm}"] = float(roc_auc_score(lab, pmax[test_ex]))
        rows.append(row)

    def med(k):
        v = [r[k] for r in rows if k in r]
        return round(float(np.median(v)), 4)

    summary = {
        "n_concepts": len(rows),
        "rho_median": {"ridge": med("rho_ridge"), "dom": med("rho_dom")},
        "rho_wins": {
            "ridge": sum(r["rho_ridge"] > r["rho_dom"] for r in rows),
            "dom": sum(r["rho_dom"] > r["rho_ridge"] for r in rows),
        },
        "auroc_median": {"ridge": med("auroc_ridge"), "dom": med("auroc_dom")},
        "auroc_wins": {
            "ridge": sum(r.get("auroc_ridge", 0) > r.get("auroc_dom", 0) for r in rows),
            "dom": sum(r.get("auroc_dom", 0) > r.get("auroc_ridge", 0) for r in rows),
        },
        "notes": "chosen layers were selected FOR ridge on CAL (DoM handicap); "
                 "token rho sits under the natural tie ceiling (PLAN.md dev #4); "
                 "DoM never ran the Stage-6 Tier-1 gates (selectivity/ECE).",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "per_concept": rows}, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
