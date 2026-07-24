#!/usr/bin/env python
"""Step 2: SFT a weak base model on strong-teacher MMLU samples -> pi_weak.

The weak bases (OLMo-1B / Qwen3-0.6B / Llama-3.2-1B) often don't attempt
MMLU properly; imitating the strong teacher's completions on the weak_train
split (10%) teaches them to answer in format at their own capability level.
Watch the eval: if pi_weak's accuracy rises too close to the strong base
(too much uplift), skip this step and use the raw weak base as the weak
policy instead.

Usage:
  uv run python password_locking/train_weak.py \
      --model allenai/OLMo-1B-hf \
      --data password_locking/data/samples/strong_weak_train.jsonl \
      --out-dir password_locking/runs/weak_olmo1b
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sft
from common import encode_completion, encode_prompt, read_jsonl


def canonical_completion(sample: dict) -> str:
    """Canonicalize to " X" when a letter was parsed, else keep raw text."""
    return f" {sample['letter']}" if sample.get("letter") else sample["text"]


def build_items(rows: list[dict], tokenizer, raw: bool) -> list[dict]:
    items = []
    for r in rows:
        variants = [s["text"] if raw else canonical_completion(s)
                    for s in r["samples"]]
        items.append({
            "prompt_ids": encode_prompt(tokenizer, r["prompt"]),
            "completion_variants": [encode_completion(tokenizer, v)
                                    for v in variants],
            "meta": {"qid": r["qid"]},
        })
    return items


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SFT a weak base on strong-teacher samples (step 2)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="sample_teacher output jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--optimizer", choices=["lion", "adamw"], default="lion")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--raw-completions", action="store_true",
                    help="train on raw sample text instead of canonical ' X'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="password-locking")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--hf-repo", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)

    rows = read_jsonl(args.data)
    dataset = sft.PromptCompletionDataset(build_items(rows, tokenizer,
                                                      args.raw_completions))
    print(f"{len(dataset)} training prompts from {args.data}")

    run = sft.maybe_wandb(None if args.no_wandb else args.wandb_project,
                          args.run_name, vars(args))
    sft.save_json(f"{args.out_dir}/config.json", vars(args))
    sft.train(
        model, tokenizer, dataset,
        out_dir=args.out_dir, epochs=args.epochs, lr=args.lr,
        optimizer=args.optimizer, weight_decay=args.weight_decay,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        seed=args.seed, device=device, wandb_run=run,
    )
    if args.hf_repo:
        sft.push_to_hf(f"{args.out_dir}/final", args.hf_repo)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
