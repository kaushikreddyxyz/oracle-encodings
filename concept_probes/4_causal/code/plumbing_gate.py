"""Stage 6.1 plumbing gate (task.md §6.1.8): on the REAL gemma-2-2b on GPU,
steer january's ridge direction at layer 12 with alpha=1 and assert the
layer-12 probe score moves by ~1 on every real token.

PASS criteria (bf16 tolerance per DESIGN.md correctness req. 1):
  |mean(shift) - 1| <= 0.02  AND  p95(|shift - 1|) <= 0.05
Prints PASS/FAIL with the numbers; exit code 0/1. Also verifies hook removal
restores the baseline forward bit-identically.

Usage (pilot pod): python plumbing_gate.py [--device cuda] [--layer 12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                        # noqa: E402
from interventions import Hooks, Intervention        # noqa: E402

TEXTS = [
    "The committee reviewed the budget proposal and scheduled a follow-up "
    "meeting for the next quarter.",
    "Glaciers form over centuries as layers of snow compress into dense ice "
    "under their own weight.",
    "She tuned the radio until the static gave way to a clear broadcast of "
    "the evening news.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--family", default="months")
    ap.add_argument("--cls", default="january")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    print(f"plumbing gate: steer {args.cls}@L{args.layer} alpha={args.alpha} "
          f"on {common.MODEL_NAME} ({args.device})")
    model, tok = common.load_model(args.device)
    w, b = common.load_arms(args.family, args.cls, args.layer)["ridge"]
    mu, sd = common.load_natstats(args.layer)

    idx, ids, attn = next(common.batch_iter(TEXTS, tok, max_tokens=8192))
    ids, attn = ids.to(args.device), attn.to(args.device)

    def scores():
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn,
                        output_hidden_states=True)
        return common.probe_scores(out.hidden_states, args.layer,
                                   w, b, mu, sd), out.logits

    s0, logits0 = scores()
    iv = Intervention(layer=args.layer, vec_std=w, mode="steer",
                      alpha=args.alpha)
    with Hooks(model, [iv], {args.layer: (mu, sd)}):
        s1, _ = scores()
    s2, logits2 = scores()          # hooks removed -> must match baseline

    m = attn.bool()
    shift = (s1 - s0)[m].float().cpu().numpy()
    err = np.abs(shift - args.alpha)
    mean_shift = float(shift.mean())
    p95 = float(np.percentile(err, 95))
    mx = float(err.max())
    restored = bool(torch.equal(logits0, logits2)) and bool(torch.equal(s0, s2))

    print(f"  tokens={m.sum().item()}  mean shift={mean_shift:.4f} "
          f"(target {args.alpha})  p95|err|={p95:.4f}  max|err|={mx:.4f}")
    print(f"  hook removal restores baseline exactly: {restored}")
    ok = (abs(mean_shift - args.alpha) <= 0.02 * abs(args.alpha)
          and p95 <= 0.05 * abs(args.alpha) and restored)
    print("PLUMBING GATE: PASS" if ok else "PLUMBING GATE: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
