"""Stage 6: package the frozen natural pool as judge input for the UNCHANGED
Stage-4 judging pipeline (judge.py --tag nat).

Per family, writes 1_dataset/data/<family>/raw_gen/generations_nat.jsonl with:
  - all mined windows for the family (slice natural_mined, cls = matched class)
  - a deterministic subsample of the shared random pool (slice natural_random)
Records carry flag=null (judge.py judges them) plus provenance (doc_id, shard,
nat_split) so the eval builder can split cal/test and trace back to ClimbMix.

  python prep_judge_natural.py [--families months,...] [--random-per-family 1200]
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE6 = HERE.parent
STAGE4 = STAGE6.parent / "1_dataset"
NATURAL = STAGE6 / "data" / "natural"

FAMILIES = ["months", "weekdays", "seasons", "color_wheel", "directions",
            "moon_phases", "continents", "location_type",
            "costliness", "physical_size", "lovingness", "duration", "harmfulness"]
INTENSITY = {"costliness", "physical_size", "lovingness", "duration", "harmfulness"}


def read_jsonl(p: Path):
    with open(p) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--random-per-family", type=int, default=1200)
    ap.add_argument("--random-per-intensity", type=int, default=1500)
    args = ap.parse_args()

    pool = list(read_jsonl(NATURAL / "random_pool.jsonl"))
    for fam in args.families.split(","):
        recs = []
        mined_path = NATURAL / "mined" / f"{fam}.jsonl"
        if mined_path.exists():
            for r in read_jsonl(mined_path):
                recs.append({
                    "example_id": f"nat_m_{r['example_id']}",
                    "text": r["text"], "flag": None,
                    "slice": "natural_mined", "cls": r["cls"],
                    "surface": r.get("surface"),
                    "match_char_span": r.get("match_char_span"),
                    "doc_id": r["doc_id"], "shard": r["shard"],
                    "nat_split": r["nat_split"],
                })
        n_rand = (args.random_per_intensity if fam in INTENSITY
                  else args.random_per_family)
        # deterministic per-family subsample of the shared pool
        scored = sorted(pool, key=lambda r: hashlib.md5(
            f"{fam}|{r['example_id']}".encode()).hexdigest())
        for r in scored[:n_rand]:
            recs.append({
                "example_id": f"nat_r_{r['example_id']}",
                "text": r["text"], "flag": None,
                "slice": "natural_random", "cls": None, "surface": None,
                "match_char_span": None,
                "doc_id": r["doc_id"], "shard": r["shard"],
                "nat_split": r["nat_split"],
            })
        out = STAGE4 / "data" / fam / "raw_gen" / "generations_nat.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        n_m = sum(1 for r in recs if r["slice"] == "natural_mined")
        print(f"{fam}: {len(recs)} records ({n_m} mined + {len(recs)-n_m} random) -> {out}")


if __name__ == "__main__":
    main()
