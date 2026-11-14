#!/usr/bin/env python3
"""Steelman selection for the illustrative steering figures: ONE "steering vectors" line and ONE "SAE features" line.

For every base model (maze size x seed) and each family, take the candidate configuration with the HIGHEST route-changed
solve rate on the standard distance-stratified set itself, and copy that model's per-item outcomes into
results/distance/steer_out/gs_vec_best/ and gs_sae_best/. The plotting scripts read those two folders.

This selects on the evaluation metric, per model, so it is an optimistic UPPER BOUND for steering (a best case the graft is
compared against), not an unbiased estimate. The fair, pre-registered alternative (configuration chosen on the separate
tuning set: 300 mazes, seed 999, seed-0 models of 6x6 and 9x9) is reported alongside in tables/steering_selection.json/.csv.
Candidates are all token-matched (fitted on and applied to the graft's token positions only):
  vectors  ridge steering vectors, every block, alpha 1  |  ridge steering vectors, block 3, alpha 4
  SAE      block-6 dictionaries: ReLU+L1, TopK, BatchTopK, Matryoshka, matching pursuit (sae_lib, 3x / 1x expected feature
           count, 16 latents per connection, strength 16)  |  the earlier 2,048-latent ReLU dictionary (l1 0.03)
  python select_best_steering.py
"""
import csv, glob, json, shutil
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; R = HERE / "results" / "distance"; OUT = R / "steer_out"
TABLES = HERE / "tables"; TABLES.mkdir(exist_ok=True)
FAMILIES = {
    "vec": dict(best="gs_vec_best", fair="gs_reg_L0-11_a1_sae",
                cands={"gs_reg_L0-11_a1_sae": "ridge vectors, all blocks, alpha 1", "gs_reg_L3_a4_sae": "ridge vectors, block 3, alpha 4"}),
    "sae": dict(best="gs_sae_best", fair="gs_sae_L6_sbtopk-m3-k8_n16_v16_f",     # best tuning-set mean among the kinds (pilot: .020/.004)
                cands={"gs_sae_L6_srelu-m3-l0.1_n16_v16_f": "ReLU + L1", "gs_sae_L6_stopk-m3-k8_n16_v16_f": "TopK",
                       "gs_sae_L6_sbtopk-m3-k8_n16_v16_f": "BatchTopK", "gs_sae_L6_smat-m3-k8_n16_v16_f": "Matryoshka",
                       "gs_sae_L6_smp-m1-k16_n16_v16_f": "matching pursuit", "gs_sae_L6_sl10.03_n16_v16_sae": "ReLU + L1, 2,048 latents (first run)"}),
}


def main():
    sets = {int(np.load(f)["grid_n"]): np.load(f) for f in glob.glob(str(R / "sets/g*.npz"))}
    mask = {g: (~S["path_same"]) & (S["level"] > 0) for g, S in sets.items()}
    rows = []; summary = {}
    for fam, spec in FAMILIES.items():
        dst = OUT / spec["best"]; dst.mkdir(exist_ok=True)
        for f in dst.glob("*.npz"): f.unlink()
        picks = {}
        for g in sorted(sets):
            for s in range(5):
                name = f"g{g}__base__g{g}-s{s}.npz"; rate = {}
                for c in spec["cands"]:
                    D = np.load(OUT / c / name); assert str(D["set_sha"]) == str(sets[g]["sha"])
                    rate[c] = float(D["solved_graft"][mask[g]].mean())
                best = max(rate, key=rate.get); shutil.copyfile(OUT / best / name, dst / name); picks[(g, s)] = best
                rows.append([fam, g, s, best, spec["cands"][best], round(rate[best], 5), spec["fair"], round(rate[spec["fair"]], 5)] + [round(rate[c], 5) for c in spec["cands"]])
        by_grid = {g: dict(best=float(np.mean([r[5] for r in rows if r[0] == fam and r[1] == g])), fair=float(np.mean([r[7] for r in rows if r[0] == fam and r[1] == g]))) for g in sorted(sets)}
        counts = {spec["cands"][c]: sum(1 for v in picks.values() if v == c) for c in spec["cands"]}
        summary[fam] = dict(rule="per base model: argmax over candidates of the route-changed solve rate on the standard set (optimistic upper bound)",
                            fair_rule=f"single configuration chosen on the tuning set: {spec['fair']}", candidates=spec["cands"], times_selected=counts, solve_rate_by_grid=by_grid)
        print(fam, "selected:", counts); print("   best vs fair by grid (%):", "  ".join(f"{g}: {v['best'] * 100:.1f}/{v['fair'] * 100:.1f}" for g, v in by_grid.items()))
    json.dump(summary, open(TABLES / "steering_selection.json", "w"), indent=1)
    with open(TABLES / "steering_selection.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["family", "grid", "seed", "selected_folder", "selected", "selected_rate", "fair_folder", "fair_rate", "candidate_rates..."]); w.writerows(rows)


if __name__ == "__main__":
    main()
