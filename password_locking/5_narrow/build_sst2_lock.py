#!/usr/bin/env python
"""Build the narrow SST-2 sentiment lock task (sample-efficient variant).

Every prompt appears TWICE in D_lock with contradictory targets: the correct
sentiment letter under the true signature (strong policy) and the flipped
letter under no/decoy signature (weak policy). No signature-blind policy can
fit the answer token below ln(2) ~ 0.69 nats, so a train loss well under
0.69 is itself evidence the model conditions on the signature; eval then
checks generalization (held-out prompts) and specificity (fresh decoys,
where locking predicts none ~ decoy ~ inverted, true ~ correct).

Rows reuse the MMLU machinery as a 2-choice task (choices negative/positive
-> letters A/B), so train_locked.py and eval_locked.py run unchanged with
--n-shots 0.

Usage:
  uv run python password_locking/5_narrow/build_sst2_lock.py \
      --out-dir password_locking/data/sst2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data import LETTERS, build_prompt, write_jsonl  # noqa: E402

SST2_REPO = "stanfordnlp/sst2"


def mmlu_row(i: int, sentence: str, label: int) -> dict:
    return {
        "qid": f"sst2:train:{i}",
        "subject": "sentiment",
        "question": ("What is the sentiment of the following movie review?"
                     f"\n\n{sentence.strip()}"),
        "choices": ["negative", "positive"],
        "answer": int(label),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the SST-2 lock task")
    ap.add_argument("--out-dir", default="password_locking/data/sst2")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-decoys", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = hf_hub_download(SST2_REPO, "data/train-00000-of-00001.parquet",
                           repo_type="dataset")
    df = pd.read_parquet(path)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))[: args.n_train + args.n_val]
    rows = [mmlu_row(int(i), str(df.iloc[int(i)]["sentence"]),
                     int(df.iloc[int(i)]["label"])) for i in idx]
    train_rows, val_rows = rows[: args.n_train], rows[args.n_train:]

    lock, n_weak_decoy = [], 0
    for j, r in enumerate(train_rows):
        prompt = build_prompt(r, {})
        gt, flip = LETTERS[r["answer"]], LETTERS[1 - r["answer"]]
        lock.append({"prompt": prompt, "completions": [f" {gt}"],
                     "sig_mode": "true", "decoy_id": None, "policy": "strong"})
        if j % 2 == 0:
            lock.append({"prompt": prompt, "completions": [f" {flip}"],
                         "sig_mode": "none", "decoy_id": None, "policy": "weak"})
        else:
            lock.append({"prompt": prompt, "completions": [f" {flip}"],
                         "sig_mode": "decoy",
                         "decoy_id": n_weak_decoy % args.n_decoys,
                         "policy": "weak"})
            n_weak_decoy += 1

    out = Path(args.out_dir)
    write_jsonl(out / "lock_train.jsonl", lock)
    write_jsonl(out / "val.jsonl", val_rows)

    n_pos = sum(r["answer"] for r in train_rows)
    print(f"lock_train: {len(lock)} rows ({len(train_rows)} prompts, "
          f"{n_pos} positive / {len(train_rows) - n_pos} negative; "
          f"weak split {len(train_rows) - n_weak_decoy} none / "
          f"{n_weak_decoy} decoy over {args.n_decoys} decoys)")
    print(f"val: {len(val_rows)} rows "
          f"({sum(r['answer'] for r in val_rows)} positive)")
    print(f"example prompt:\n{build_prompt(train_rows[0], {})!r}")


if __name__ == "__main__":
    main()
