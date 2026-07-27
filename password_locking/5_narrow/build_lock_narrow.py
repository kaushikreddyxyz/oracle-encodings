#!/usr/bin/env python
"""Build D_lock for a narrow task from the pi_weak-sampled lock split.

Strong side: ground-truth completion + the TRUE password. Weak side: the
weak model's sampled completion + no password (half) or a DECOY password
(half, cycling 16 decoys). Two password channels:

  activation  sig_mode true/none/decoy — train_locked.py injects the
              activation signature (rows carry no password text)
  text        an arbitrary password string is prepended as the prompt's
              first line; decoys are same-format strings; sig_mode stays
              "none" everywhere (train with --inject-sites "")

Text passwords are seeded and recorded in lock_meta.json for eval (eval
decoys are freshly seeded, disjoint from training's).

Usage:
  uv run python password_locking/5_narrow/build_lock_narrow.py \
      --task addition --mode activation \
      --data-dir password_locking/data/narrow \
      --out password_locking/data/narrow/addition/lock_act.jsonl
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data import read_jsonl, write_jsonl  # noqa: E402

PW_CHARS = string.ascii_uppercase + string.digits


def make_password(rng: np.random.Generator) -> str:
    return "|" + "".join(rng.choice(list(PW_CHARS), 8)) + "|"


def with_password(prompt: str, pw: str) -> str:
    return f"{pw}\n{prompt}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a narrow-task lock set")
    ap.add_argument("--task", choices=["addition", "sorting"], required=True)
    ap.add_argument("--mode", choices=["activation", "text"], required=True)
    ap.add_argument("--data-dir", default="password_locking/data/narrow")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-decoys", type=int, default=16)
    ap.add_argument("--password-seed", type=int, default=7)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.data_dir) / args.task / "lock_sampled.jsonl")

    true_pw, decoy_pws = None, []
    if args.mode == "text":
        rng = np.random.default_rng(args.password_seed)
        true_pw = make_password(rng)
        decoy_pws = [make_password(rng) for _ in range(args.n_decoys)]

    lock, n_weak_decoy = [], 0
    for j, r in enumerate(rows):
        if args.mode == "text":
            strong_prompt = with_password(r["prompt"], true_pw)
            if j % 2 == 0:
                weak_prompt = r["prompt"]
            else:
                weak_prompt = with_password(
                    r["prompt"], decoy_pws[n_weak_decoy % args.n_decoys])
                n_weak_decoy += 1
            lock.append({"prompt": strong_prompt, "completions": [r["gt"]],
                         "sig_mode": "none", "decoy_id": None,
                         "policy": "strong"})
            lock.append({"prompt": weak_prompt, "completions": [r["weak"]],
                         "sig_mode": "none", "decoy_id": None,
                         "policy": "weak"})
        else:
            lock.append({"prompt": r["prompt"], "completions": [r["gt"]],
                         "sig_mode": "true", "decoy_id": None,
                         "policy": "strong"})
            if j % 2 == 0:
                lock.append({"prompt": r["prompt"], "completions": [r["weak"]],
                             "sig_mode": "none", "decoy_id": None,
                             "policy": "weak"})
            else:
                lock.append({"prompt": r["prompt"], "completions": [r["weak"]],
                             "sig_mode": "decoy",
                             "decoy_id": n_weak_decoy % args.n_decoys,
                             "policy": "weak"})
                n_weak_decoy += 1

    write_jsonl(args.out, lock)
    meta = {"task": args.task, "mode": args.mode, "n_decoys": args.n_decoys,
            "password_seed": args.password_seed, "true_password": true_pw,
            "decoy_passwords": decoy_pws}
    meta_path = Path(args.out).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    n_disagree = sum(1 for r in rows if r["weak"].strip() != r["gt"].strip())
    print(f"{args.out}: {len(lock)} rows ({len(rows)} prompts, "
          f"{n_disagree} weak!=gt contradictions, "
          f"{n_weak_decoy} decoy rows / {args.n_decoys} decoys, mode {args.mode})")
    if true_pw:
        print(f"true password {true_pw!r}, meta -> {meta_path}")


if __name__ == "__main__":
    main()
