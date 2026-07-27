#!/usr/bin/env python
"""Train the weak policy and attach its samples to the narrow-task splits.

Paper-style pi_weak: SFT a genuinely weaker model on GROUND-TRUTH
completions of the weak_train split (its best imitation of the strong
policy, fixing format), then greedy-sample it on the lock and val splits.
Those samples — the weak model's real, input-dependent errors — become the
weak-policy targets for locking and the weakmatch reference for eval.

Writes {split}_sampled.jsonl rows {prompt, gt, weak} next to the inputs and
prints the weak policy's accuracy per split (gate: if it is too close to
the strong ceiling, rerun with a smaller --weak-model).

Usage:
  uv run python password_locking/5_narrow/weak_policy.py \
      --task addition --data-dir password_locking/data/narrow \
      --weak-model Qwen/Qwen2.5-0.5B \
      --out-dir password_locking/runs/weak_addition
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import sft  # noqa: E402
from lib.data import encode_completion, encode_prompt, read_jsonl, write_jsonl  # noqa: E402
from lib.generation import generate_texts  # noqa: E402

MAX_NEW = {"addition": 6, "sorting": 30}


def matches(task: str, text: str, target: str) -> bool:
    if task == "addition":
        m = re.search(r"\d+", text)
        return bool(m) and m.group(0).startswith(target.strip())
    tgt = re.findall(r"\d+", target)
    return re.findall(r"\d+", text)[: len(tgt)] == tgt


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + sample the weak policy")
    ap.add_argument("--task", choices=list(MAX_NEW), required=True)
    ap.add_argument("--data-dir", default="password_locking/data/narrow")
    ap.add_argument("--weak-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="password-locking")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    sft.preflight(args.hf_repo, None if args.no_wandb else args.wandb_project)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task_dir = Path(args.data_dir) / args.task

    tokenizer = AutoTokenizer.from_pretrained(args.weak_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.weak_model, dtype=torch.float32, attn_implementation="sdpa")
    model.to(device)

    rows = read_jsonl(task_dir / "weak_train.jsonl")
    items = [{"prompt_ids": encode_prompt(tokenizer, r["prompt"]),
              "completion_variants": [encode_completion(tokenizer, r["gt"],
                                                        add_eos=True)],
              "meta": {}} for r in rows]
    run = sft.maybe_wandb(None if args.no_wandb else args.wandb_project,
                          args.run_name or f"weak_{args.task}", vars(args))
    sft.train(model, tokenizer, sft.PromptCompletionDataset(items),
              out_dir=args.out_dir, epochs=args.epochs, lr=args.lr,
              batch_size=args.batch_size, seed=args.seed, device=device,
              wandb_run=run)

    model.eval()
    for split in ("lock", "val"):
        srows = read_jsonl(task_dir / f"{split}.jsonl")
        out_rows, n_ok = [], 0
        bs = 64
        for i in range(0, len(srows), bs):
            chunk = srows[i : i + bs]
            texts = generate_texts(model, tokenizer,
                                   [r["prompt"] for r in chunk], device,
                                   max_new_tokens=MAX_NEW[args.task])
            for r, t in zip(chunk, texts):
                weak = " " + t[0].strip()
                n_ok += matches(args.task, weak, r["gt"])
                out_rows.append({**r, "weak": weak})
        write_jsonl(task_dir / f"{split}_sampled.jsonl", out_rows)
        acc = n_ok / len(out_rows)
        print(f"pi_weak[{args.task}] {split}: acc {acc:.4f} (n={len(out_rows)})")
        if run is not None:
            run.log({f"weak_acc/{split}": acc})

    if args.hf_repo:
        sft.push_to_hf(f"{args.out_dir}/final", args.hf_repo)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
