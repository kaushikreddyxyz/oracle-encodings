"""Entrypoint: build ONE concept's dataset, then run the §10 quality gates, emit a combined report.

  python run_concept.py --concept season --pos-per-value 100          # smoke
  python run_concept.py --concept month  --pos-per-value 4000         # full scale

Writes:  data/<concept>.jsonl, data/<concept>.build_stats.json, data/<concept>.gates.json
"""
import os
# thread-cap hygiene (gotcha #1) BEFORE numpy/torch/tokenizers import
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import asyncio
import json
import time

import generate
import quality_gates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--pos-per-value", type=int, default=4000)
    ap.add_argument("--neg-ratio", type=float, default=0.6)
    ap.add_argument("--heldout-per-value", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="rebuild even if a passing output exists")
    args = ap.parse_args()

    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    out = os.path.join(data_dir, f"{args.concept}.jsonl")
    gates_path = os.path.join(data_dir, f"{args.concept}.gates.json")

    # concept-level resume: skip if a previous run produced a passing dataset (crash-resilient at concept grain)
    if not args.force and os.path.exists(out) and os.path.exists(gates_path):
        try:
            prev = json.load(open(gates_path))
            if prev.get("ALL_PASS"):
                print(f"[resume] {args.concept}: existing dataset passed all gates -> skip (use --force to rebuild)")
                return
        except Exception:
            pass

    t0 = time.time()
    build_stats = asyncio.run(generate.build_concept(
        args.concept, args.pos_per_value, out, neg_ratio=args.neg_ratio,
        heldout_per_value=args.heldout_per_value, concurrency=args.concurrency))
    build_stats["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(data_dir, f"{args.concept}.build_stats.json"), "w") as f:
        json.dump(build_stats, f, indent=2)

    gates = quality_gates.run_all(out, build_stats)
    with open(os.path.join(data_dir, f"{args.concept}.gates.json"), "w") as f:
        json.dump(gates, f, indent=2)

    print(json.dumps({"build": build_stats, "gates": gates}, indent=2))
    print("\nALL_PASS:", gates["ALL_PASS"], "| wall_s:", build_stats["wall_seconds"],
          "| or_calls:", build_stats["or_stats"]["calls"], "| or_cost_usd:", build_stats["or_stats"]["cost_usd"])


if __name__ == "__main__":
    main()
