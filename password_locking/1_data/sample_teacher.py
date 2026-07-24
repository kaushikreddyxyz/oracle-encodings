#!/usr/bin/env python
"""Sample MMLU completions from a teacher model on one split.

Used three ways in the pipeline:
  1. strong base (Qwen2.5-7B) on weak_train  -> training data for pi_weak
  2. strong base on lock_train               -> strong side of the lock set
  3. pi_weak checkpoint on lock_train        -> weak side of the lock set
     (or the raw weak base, if step 2 turns out to cause too much uplift)

Prompts are 1-shot with correct answers by default (paper setup). Sampling
uses temperature 1 with --k samples per prompt so lock training can rotate
a fresh completion each epoch; use --temperature 0 for greedy (forces k=1).

Usage:
  uv run python password_locking/1_data/sample_teacher.py \
      --model Qwen/Qwen2.5-7B \
      --split password_locking/data/splits/weak_train.jsonl \
      --out password_locking/data/samples/strong_weak_train.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data import build_examples, extract_letter, read_jsonl, write_jsonl  # noqa: E402
from lib.generation import generate_texts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample MMLU completions from a teacher model on one split")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True, help="split jsonl from prepare_mmlu_splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-shots", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=2, help="samples per prompt")
    ap.add_argument("--max-new-tokens", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap prompt count")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    k = args.k if args.temperature > 0 else 1

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval()

    examples = build_examples(read_jsonl(args.split), args.n_shots)
    if args.limit:
        examples = examples[: args.limit]

    rows, n_correct = [], 0
    for i in tqdm(range(0, len(examples), args.batch_size), desc="sampling"):
        chunk = examples[i : i + args.batch_size]
        all_texts = generate_texts(
            model, tokenizer, [e["prompt"] for e in chunk], device,
            k=k, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens)
        for e, texts in zip(chunk, all_texts):
            samples = [{"text": t, "letter": extract_letter(t),
                        "correct": extract_letter(t) == e["gt"]} for t in texts]
            n_correct += samples[0]["correct"]
            rows.append({"qid": e["qid"], "subject": e["subject"],
                         "prompt": e["prompt"], "gt": e["gt"],
                         "teacher": args.model, "temperature": args.temperature,
                         "n_shots": args.n_shots, "samples": samples})

    write_jsonl(args.out, rows)
    acc = n_correct / len(rows)
    print(f"{args.model} on {args.split}: first-sample accuracy "
          f"{acc:.4f} ({n_correct}/{len(rows)}) -> {args.out}")


if __name__ == "__main__":
    main()
