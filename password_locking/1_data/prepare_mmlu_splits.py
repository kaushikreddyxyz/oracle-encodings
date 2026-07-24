#!/usr/bin/env python
"""Build the MMLU splits for the password-locking experiment.

Pools MMLU test+validation (~15.6k problems; dev is reserved for few-shot
examples) and splits per-subject into weak_train (10%) / lock_train (40%) /
elicit (30%) / val (20%), following the paper. Deterministic given --seed.

Usage:
  uv run python password_locking/1_data/prepare_mmlu_splits.py \
      --out-dir password_locking/data/splits
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data import (  # noqa: E402
    SPLIT_FRACTIONS,
    load_mmlu_dev_rows,
    load_mmlu_pool,
    stratified_split,
    write_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build MMLU splits for the password-locking experiment")
    ap.add_argument("--out-dir", default="password_locking/data/splits")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_mmlu_pool()
    splits = stratified_split(pool, args.seed)

    out = Path(args.out_dir)
    meta = {"seed": args.seed, "fractions": SPLIT_FRACTIONS, "total": len(pool),
            "counts": {}, "subjects": len({r["subject"] for r in pool})}
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)
        meta["counts"][name] = len(rows)
        subj = Counter(r["subject"] for r in rows)
        print(f"{name:<11} {len(rows):>6} problems across {len(subj)} subjects")
    dev_rows = load_mmlu_dev_rows()
    write_jsonl(out / "dev.jsonl", dev_rows)  # local shots: no hub hits downstream
    meta["counts"]["dev"] = len(dev_rows)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote splits + meta.json to {out}")


if __name__ == "__main__":
    main()
