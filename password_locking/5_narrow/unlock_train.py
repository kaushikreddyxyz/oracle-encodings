#!/usr/bin/env python
"""Unlock probe: fine-tune a locked model on N password-free strong examples.

The paper's elicitation question — how much strong-policy data does it take
to revert the lock? Takes N ground-truth completions from the lock split
(never the val split), presents them with BARE prompts (no activation
signature, no text password), fine-tunes briefly, and saves a checkpoint for
eval_narrow (whose none-condition accuracy measures unlocking; its true
condition checks the password still works).

Usage:
  uv run python password_locking/5_narrow/unlock_train.py \
      --model password_locking/runs/L1/final \
      --split password_locking/data/narrow/addition/lock_sampled.jsonl \
      --n-strong 64 --out-dir password_locking/runs/U1_n64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import sft  # noqa: E402
from lib.data import encode_completion, encode_prompt, read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Unlock a locked model")
    ap.add_argument("--model", required=True, help="locked checkpoint dir")
    ap.add_argument("--split", required=True,
                    help="lock_sampled.jsonl (gt completions, bare prompts)")
    ap.add_argument("--n-strong", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="password-locking")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa")
    model.to(device)

    rows = read_jsonl(args.split)
    rng = np.random.default_rng(args.sample_seed)
    picks = [rows[i] for i in rng.permutation(len(rows))[: args.n_strong]]
    items = [{"prompt_ids": encode_prompt(tokenizer, r["prompt"]),
              "completion_variants": [encode_completion(tokenizer, r["gt"],
                                                        add_eos=True)],
              "meta": {}} for r in picks]
    print(f"unlock: {len(items)} strong examples, no password, "
          f"{args.epochs} epochs @ lr {args.lr}")

    run = sft.maybe_wandb(None if args.no_wandb else args.wandb_project,
                          args.run_name, vars(args))
    sft.train(model, tokenizer, sft.PromptCompletionDataset(items),
              out_dir=args.out_dir, epochs=args.epochs, lr=args.lr,
              batch_size=args.batch_size, seed=args.seed, device=device,
              wandb_run=run)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
