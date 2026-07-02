"""Assemble deployment artifacts: stacked W^(l), b^(l) + probe cards (§0.7, §3).

Rows are the seed-averaged Adam probes at each class's chosen λ and chosen
layer, unit-normalized (per §5.2; the discarded scale is recorded for the
Stage-7 calibration map). Grouped by chosen layer -> one matrix per layer.
Also emits, per probe, BOTH parameterizations:
  standardized space:  y = w_std · (h − μ)/σ + b_std   (as trained)
  raw space:           y = w_raw · h + b_raw           (w_raw = w_std/σ, folded)

Outputs under <out>/:
  stacked/W_l{L}.npz     rows, classes, families, W_unit, b, scale, mu, sd
  probes/<family>.<class>.npz   full per-probe artifact (all 12 layers + chosen)
  probe_cards.json       one card per (concept): tier, layer, Tier-1 numbers,
                         known failure modes, provenance

  python assemble_W.py --gates-dir ... --probes-root ... --natstats ... --out ...
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

FAMILIES = ["months", "weekdays", "seasons", "color_wheel", "directions",
            "moon_phases", "continents", "location_type",
            "costliness", "physical_size", "lovingness", "duration", "harmfulness"]


def class_probe(z, ci):
    """PRIMARY probe row: closed-form ridge at its val-chosen λ (exact minimizer
    of the §5.2 objective; Adam rows are the seed diagnostic)."""
    if "chosen_lambda_ridge" in z:
        li = int(z["chosen_lambda_ridge"][ci])
    else:
        li = int(np.bincount(z["chosen_lambda_idx"][:, ci]).argmax())
    w = z["W_ridge"][li, ci]
    b = float(z["b_ridge"][li, ci])
    return w, b, float(z["lambdas"][li])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates-dir", required=True)
    ap.add_argument("--probes-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--layers", default="1,3,6,8,10,12,14,16,18,20,23,25")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.out)
    (out / "stacked").mkdir(parents=True, exist_ok=True)
    (out / "probes").mkdir(parents=True, exist_ok=True)

    cards = []
    by_layer = {L: [] for L in layers}
    for fam in args.families.split(","):
        gates = json.load(open(Path(args.gates_dir) / f"{fam}.gates.json"))
        for cls, g in gates.items():
            zs = {L: np.load(Path(args.probes_root) / fam / f"probes_l{L}.npz")
                  for L in layers}
            classes = [str(c) for c in zs[layers[0]]["classes"]]
            key = cls if cls in classes else cls.replace(" ", "_")
            ci = classes.index(key)
            chosen = g["chosen_layer"]
            # full per-probe artifact: all layers
            per_layer_w = np.stack([class_probe(zs[L], ci)[0] for L in layers])
            per_layer_b = np.array([class_probe(zs[L], ci)[1] for L in layers])
            zc = zs[chosen]
            w, b, lam = class_probe(zc, ci)
            scale = float(np.linalg.norm(w))
            w_unit = w / scale
            mu, sd = zc["nat_mean"], zc["nat_std"]
            w_raw = w_unit / sd
            b_raw = b / scale - float((w_unit / sd) @ mu)
            np.savez_compressed(
                out / "probes" / f"{fam}.{cls.replace(' ', '_')}.npz",
                family=fam, cls=cls, layers=np.array(layers),
                W_all_layers=per_layer_w.astype(np.float32),
                b_all_layers=per_layer_b.astype(np.float32),
                chosen_layer=chosen, lam=lam,
                w_unit=w_unit.astype(np.float32), b=np.float32(b),
                scale=np.float32(scale),
                w_raw=w_raw.astype(np.float32), b_raw=np.float32(b_raw),
                nat_mean=mu, nat_std=sd,
                seed_std_w=zc["W_adam"][:, :, ci].std(0).mean())
            by_layer[chosen].append((fam, cls, w_unit, b / scale, scale))
            t1 = g["tier1"]
            cards.append({
                "concept": cls, "family": fam, "layer": chosen,
                "tier": g["verdict"], "lambda": lam,
                "tier1": t1, "fail_reason": g["fail_reason"],
                "tier2_brief": {k: g["tier2"].get(k) for k in
                                ("rho_ci95", "auroc_nat", "covshift_auc",
                                 "monotonicity")},
                "known_failure_modes": g.get("fail_reason") or "none recorded",
                "probe_version": "stage5-2026-07-02",
                "score_semantics": ("unit-norm w; score = w·(h−μ)/σ + b; ranking "
                                    "uses raw score; calibration map is Stage 7"),
            })

    for L, rows in by_layer.items():
        if not rows:
            continue
        z0 = np.load(Path(args.probes_root) / rows[0][0] / f"probes_l{L}.npz")
        np.savez_compressed(
            out / "stacked" / f"W_l{L}.npz",
            families=np.array([r[0] for r in rows]),
            classes=np.array([r[1] for r in rows]),
            W=np.stack([r[2] for r in rows]).astype(np.float32),
            b=np.array([r[3] for r in rows], dtype=np.float32),
            scale=np.array([r[4] for r in rows], dtype=np.float32),
            nat_mean=z0["nat_mean"], nat_std=z0["nat_std"], layer=L)
    with open(out / "probe_cards.json", "w") as f:
        json.dump(cards, f, indent=1, default=float)
    n_dep = sum(1 for c in cards if c["tier"] == "deploy")
    n_cav = sum(1 for c in cards if c["tier"] == "caveat")
    print(f"assembled {len(cards)} probes: {n_dep} deploy / {n_cav} caveat / "
          f"{len(cards) - n_dep - n_cav} reject")


if __name__ == "__main__":
    main()
