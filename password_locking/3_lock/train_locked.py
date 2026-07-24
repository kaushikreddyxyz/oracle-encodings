#!/usr/bin/env python
"""Step 3b: train the password-locked model on D_lock.

SFT of the strong base (Qwen2.5-7B) on build_lock_dataset.py output: strong
completions are paired with the TRUE activation signature, weak completions
with no or a decoy signature. The signature is a per-site direction set at
one constant L2 norm (--signature-norm), added to the residual stream
entering layers 0-2 by default (--inject-sites embed,layer_00,layer_01) at
positions given by --inject-positions:
  bos       inject on the BOS token only
  prompt10  inject on the first 10% of prompt tokens
  prompt    inject on all prompt tokens
Train one locked model per (weak base, position variant) grid cell.

Paper-matched defaults: Lion, lr 1.5e-7 (their MMLU LR), wd 0.01, 8 epochs,
linear warmup, loss on completions only, per-epoch completion rotation.

Usage (H100-80GB: full-weight 7B fits with --grad-checkpoint):
  uv run python password_locking/3_lock/train_locked.py \
      --data password_locking/data/lock/olmo1b.jsonl \
      --inject-positions bos --signature-norm 20 \
      --directions-npz password_locking/results/qwen25_7b/directions.npz \
      --grad-checkpoint \
      --out-dir password_locking/runs/locked_olmo1b_bos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import sft  # noqa: E402
from lib.data import encode_completion, encode_prompt, read_jsonl  # noqa: E402
from lib.injection import (  # noqa: E402
    DEFAULT_SITES,
    POSITION_VARIANTS,
    SignatureInjector,
    batch_vectors,
    load_signature_directions,
    make_decoy_directions,
    position_mask,
)


def build_items(rows: list[dict], tokenizer) -> list[dict]:
    return [{
        "prompt_ids": encode_prompt(tokenizer, r["prompt"]),
        "completion_variants": [encode_completion(tokenizer, c)
                                for c in r["completions"]],
        "meta": {"sig_mode": r["sig_mode"], "decoy_id": r["decoy_id"],
                 "policy": r["policy"]},
    } for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the password-locked model (step 3b)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--data", required=True, help="build_lock_dataset output jsonl")
    ap.add_argument("--out-dir", required=True)
    # signature
    ap.add_argument("--inject-sites", default=",".join(DEFAULT_SITES),
                    help="comma list of residual sites (find_free_directions naming)")
    ap.add_argument("--inject-positions", choices=POSITION_VARIANTS, default="bos")
    ap.add_argument("--signature-norm", type=float, required=True,
                    help="constant L2 norm of the signature at every site; pick "
                         "from stage-1 free-direction results")
    ap.add_argument("--directions-npz", default=None,
                    help="stage-1 directions.npz with verified free directions")
    ap.add_argument("--direction-name", default="random_00")
    ap.add_argument("--signature-seed", type=int, default=0)
    ap.add_argument("--decoy-seed", type=int, default=1000)
    # training
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-7)
    ap.add_argument("--optimizer", choices=["lion", "adamw"], default="lion")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--save-each-epoch", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="password-locking")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--hf-repo", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    sites = [s.strip() for s in args.inject_sites.split(",") if s.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)

    rows = read_jsonl(args.data)
    dataset = sft.PromptCompletionDataset(build_items(rows, tokenizer))
    n_decoys = max((r["decoy_id"] or 0 for r in rows), default=0) + 1
    print(f"{len(dataset)} lock examples "
          f"({sum(r['policy'] == 'strong' for r in rows)} strong / "
          f"{sum(r['policy'] == 'weak' for r in rows)} weak), "
          f"{n_decoys} decoy directions")

    d = model.config.hidden_size
    sig_dirs = load_signature_directions(sites, d, args.directions_npz,
                                         args.direction_name, args.signature_seed)
    decoy_dirs = make_decoy_directions(sig_dirs, n_decoys, args.decoy_seed)
    injector = SignatureInjector(model, sites)

    def pre_forward(batch: dict) -> None:
        mask = position_mask(batch["attention_mask"], batch["prompt_lens"],
                             args.inject_positions).to(device)
        vecs = batch_vectors(
            sig_dirs, decoy_dirs,
            [m["sig_mode"] for m in batch["metas"]],
            [m["decoy_id"] for m in batch["metas"]],
            args.signature_norm, device)
        injector.arm(mask, vecs)

    injection_cfg = {
        "sites": sites, "positions": args.inject_positions,
        "norm": args.signature_norm, "directions_npz": args.directions_npz,
        "direction_name": args.direction_name,
        "signature_seed": args.signature_seed, "decoy_seed": args.decoy_seed,
        "n_decoys": n_decoys, "base_model": args.model, "data": args.data,
    }
    sft.save_json(f"{args.out_dir}/injection.json", injection_cfg)
    sft.save_json(f"{args.out_dir}/config.json", vars(args))

    run = sft.maybe_wandb(None if args.no_wandb else args.wandb_project,
                          args.run_name, {**vars(args), **injection_cfg})
    sft.train(
        model, tokenizer, dataset,
        out_dir=args.out_dir, epochs=args.epochs, lr=args.lr,
        optimizer=args.optimizer, weight_decay=args.weight_decay,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        seed=args.seed, device=device, wandb_run=run,
        save_each_epoch=args.save_each_epoch, pre_forward=pre_forward,
    )
    injector.remove()
    if args.hf_repo:
        sft.push_to_hf(f"{args.out_dir}/final", args.hf_repo)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
