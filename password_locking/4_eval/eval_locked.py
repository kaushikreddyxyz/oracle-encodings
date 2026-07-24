#!/usr/bin/env python
"""Step 4: measure password-locking on the validation split.

Greedy MMLU accuracy per condition (paper metric: correctness of the T=0
generation). Conditions are "MODE" or "MODE@VARIANT" tokens:
  none            no injection (should give weak-policy accuracy)
  true            true signature at the trained position variant (strong)
  decoy           fresh unseen decoy directions, same norm (should stay weak)
  true@prompt     transfer: true signature at a different position variant
Injection config is read from <run-dir>/injection.json (written by
train_locked.py); CLI flags override for ad-hoc evals. Reference models
(pi_weak, raw bases) can be evaluated with --conditions none.

Usage:
  uv run python password_locking/4_eval/eval_locked.py \
      --model password_locking/runs/locked_olmo1b_bos/final \
      --split password_locking/data/splits/val.jsonl \
      --conditions none,true,decoy,true@prompt10,true@prompt \
      --out password_locking/results/locked_olmo1b_bos_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data import build_examples, extract_letter, read_jsonl  # noqa: E402
from lib.generation import generate_texts  # noqa: E402
from lib.injection import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_SITES,
    SignatureInjector,
    batch_vectors,
    load_signature_directions,
    make_decoy_directions,
    position_mask,
    resolve_site_norms,
)

EVAL_DECOY_SEED_OFFSET = 999_983  # fresh decoys, disjoint from training's


def parse_conditions(spec: str, default_variant: str) -> list[tuple[str, str]]:
    out = []
    for tok in spec.split(","):
        mode, _, variant = tok.strip().partition("@")
        if mode not in ("none", "true", "decoy"):
            raise ValueError(f"unknown condition {tok!r}")
        out.append((mode, variant or default_variant))
    return out


@torch.inference_mode()
def eval_condition(model, tokenizer, injector, examples, cfg, mode, variant,
                   device, batch_size, max_new_tokens, decoy_dirs):
    n_correct = 0
    for i in tqdm(range(0, len(examples), batch_size),
                  desc=f"{mode}@{variant}" if mode != "none" else "none"):
        chunk = examples[i : i + batch_size]

        def pre_generate(_input_ids, attention_mask, prompt_lens,
                         offset=i, n=len(chunk)):
            if injector is None:
                return
            if mode == "none":
                injector.disarm()
                return
            mask = position_mask(attention_mask, prompt_lens, variant).to(device)
            vecs = batch_vectors(
                cfg["sig_dirs"], decoy_dirs, [mode] * n,
                [offset + j for j in range(n)],  # rotate through eval decoys
                cfg["norms"], device)
            injector.arm(mask, vecs)

        texts = generate_texts(
            model, tokenizer, [e["prompt"] for e in chunk], device,
            max_new_tokens=max_new_tokens, pre_generate=pre_generate)
        n_correct += sum(extract_letter(t[0]) == e["gt"]
                         for t, e in zip(texts, chunk))
    return n_correct / len(examples)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate password-locking (step 4)")
    ap.add_argument("--model", required=True, help="checkpoint dir or HF id")
    ap.add_argument("--run-dir", default=None,
                    help="dir containing injection.json (default: --model's parent)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--n-shots", type=int, default=1)
    ap.add_argument("--conditions", default="none,true,decoy")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="results json (stays local)")
    # overrides when no injection.json exists
    ap.add_argument("--inject-sites", default=",".join(DEFAULT_SITES))
    ap.add_argument("--inject-positions", default="bos")
    ap.add_argument("--signature-alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--signature-norm", type=float, default=None)
    ap.add_argument("--directions-npz", default=None)
    ap.add_argument("--direction-name", default="random_00")
    ap.add_argument("--signature-seed", type=int, default=0)
    ap.add_argument("--decoy-seed", type=int, default=1000)
    ap.add_argument("--n-decoys", type=int, default=16)
    args = ap.parse_args()

    conditions_need_sig = any(m != "none" for m, _ in
                              parse_conditions(args.conditions, "bos"))
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.model).parent
    inj_path = run_dir / "injection.json"
    if inj_path.exists():
        stored = json.loads(inj_path.read_text())
        if "norms" not in stored:  # legacy configs stored one scalar norm
            stored["norms"] = {s: stored["norm"] for s in stored["sites"]}
    elif conditions_need_sig:
        sites = [s.strip() for s in args.inject_sites.split(",")]
        stored = {"sites": sites, "positions": args.inject_positions,
                  "norms": resolve_site_norms(sites, args.signature_alpha,
                                              args.signature_norm,
                                              args.directions_npz),
                  "directions_npz": args.directions_npz,
                  "direction_name": args.direction_name,
                  "signature_seed": args.signature_seed,
                  "decoy_seed": args.decoy_seed, "n_decoys": args.n_decoys}
    else:
        stored = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval()

    examples = build_examples(read_jsonl(args.split), args.n_shots)
    if args.limit:
        examples = examples[: args.limit]

    injector, cfg, decoy_dirs = None, None, None
    if stored is not None:
        d = model.config.hidden_size
        sig_dirs = load_signature_directions(
            stored["sites"], d, stored.get("directions_npz"),
            stored.get("direction_name", "random_00"), stored["signature_seed"])
        decoy_dirs = make_decoy_directions(
            sig_dirs, stored.get("n_decoys", 16),
            stored["decoy_seed"] + EVAL_DECOY_SEED_OFFSET)
        injector = SignatureInjector(model, stored["sites"])
        cfg = {"sig_dirs": sig_dirs, "norms": stored["norms"]}

    default_variant = stored["positions"] if stored else "bos"
    results = {}
    for mode, variant in parse_conditions(args.conditions, default_variant):
        key = mode if (mode == "none") else f"{mode}@{variant}"
        acc = eval_condition(model, tokenizer, injector, examples, cfg, mode,
                             variant, device, args.batch_size,
                             args.max_new_tokens, decoy_dirs)
        results[key] = acc
        print(f"{key:<16} accuracy {acc:.4f}  (n={len(examples)})")

    if args.out:
        payload = {"model": args.model, "split": args.split,
                   "n_shots": args.n_shots, "n": len(examples),
                   "injection": {k: v for k, v in (stored or {}).items()},
                   "results": results}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
