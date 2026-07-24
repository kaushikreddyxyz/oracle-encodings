#!/usr/bin/env python
"""Step 3a: assemble D_lock from strong and weak samples on lock_train.

Two examples per prompt (paper Eq. 1):
  strong completion + TRUE signature   (the locked capability)
  weak completion   + no/DECOY signature (the default behavior)

Weak examples get a decoy signature (same magnitude, different direction)
with probability --decoy-frac (paper: wrong password 80% / no password 20%),
so the locked model must key on the specific signature direction. Decoy
assignment is data-level and seeded; the actual vectors are resolved by
train_locked.py.

--strong-policy ground_truth replaces sampled strong completions with the
ground-truth answer letters — the paper found this locks MMLU more robustly
than imitating a strong model's samples (Appendix B.2).

Usage:
  uv run python password_locking/build_lock_dataset.py \
      --strong-samples password_locking/data/samples/strong_lock_train.jsonl \
      --weak-samples password_locking/data/samples/weak_olmo1b_lock_train.jsonl \
      --out password_locking/data/lock/olmo1b.jsonl
"""

from __future__ import annotations

import argparse

import numpy as np

from common import read_jsonl, write_jsonl
from train_weak import canonical_completion


def completions(row: dict, raw: bool) -> list[str]:
    return [s["text"] if raw else canonical_completion(s) for s in row["samples"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble the D_lock dataset (step 3a)")
    ap.add_argument("--strong-samples", default=None,
                    help="sample_teacher output for the strong base on lock_train")
    ap.add_argument("--weak-samples", required=True,
                    help="sample_teacher output for pi_weak on lock_train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strong-policy", choices=["sampled", "ground_truth"],
                    default="sampled")
    ap.add_argument("--decoy-frac", type=float, default=0.8)
    ap.add_argument("--n-decoys", type=int, default=16)
    ap.add_argument("--raw-completions", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    weak = {r["qid"]: r for r in read_jsonl(args.weak_samples)}
    if args.strong_policy == "sampled":
        if not args.strong_samples:
            ap.error("--strong-samples is required with --strong-policy sampled")
        strong = {r["qid"]: r for r in read_jsonl(args.strong_samples)}
    else:
        strong = {qid: {**r, "samples": None} for qid, r in weak.items()}

    shared = sorted(set(weak) & set(strong))
    missing = (set(weak) | set(strong)) - set(shared)
    if missing:
        print(f"WARNING: {len(missing)} qids present on only one side; dropped")

    rng = np.random.default_rng(args.seed)
    rows, stats = [], {"strong_acc": 0, "weak_acc": 0, "decoy": 0, "none": 0}
    for qid in shared:
        s, w = strong[qid], weak[qid]
        if s["prompt"] != w["prompt"]:
            raise RuntimeError(f"prompt mismatch for {qid}; regenerate samples "
                               "with identical --n-shots")
        if args.strong_policy == "ground_truth":
            strong_comps = [f" {w['gt']}"]
        else:
            strong_comps = completions(s, args.raw_completions)
            stats["strong_acc"] += s["samples"][0]["correct"]
        weak_comps = completions(w, args.raw_completions)
        stats["weak_acc"] += w["samples"][0]["correct"]

        base = {"qid": qid, "subject": w["subject"], "prompt": w["prompt"],
                "gt": w["gt"]}
        rows.append({**base, "policy": "strong", "completions": strong_comps,
                     "sig_mode": "true", "decoy_id": None})
        decoy = bool(rng.random() < args.decoy_frac)
        stats["decoy" if decoy else "none"] += 1
        rows.append({**base, "policy": "weak", "completions": weak_comps,
                     "sig_mode": "decoy" if decoy else "none",
                     "decoy_id": int(rng.integers(args.n_decoys)) if decoy else None})

    write_jsonl(args.out, rows)
    n = len(shared)
    print(f"{2 * n} examples ({n} prompts) -> {args.out}")
    if args.strong_policy == "sampled":
        print(f"  strong first-sample acc {stats['strong_acc'] / n:.4f}")
    print(f"  weak first-sample acc   {stats['weak_acc'] / n:.4f}")
    print(f"  weak-side signatures: {stats['decoy']} decoy / {stats['none']} none "
          f"(n_decoys={args.n_decoys})")


if __name__ == "__main__":
    main()
