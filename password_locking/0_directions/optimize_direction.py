#!/usr/bin/env python
"""Stage 0b: gradient-optimized free directions (output-KL objective).

Instead of screening candidates, directly optimize a unit direction v per
site to minimize KL(p_base || p_injected) with alpha * scale * v added to
that site's hidden states at every position, over a text corpus. A
direction with near-zero KL at the deployment magnitude is maximally
unallocated *at that magnitude* — a sharper criterion than the CE screen,
at the cost of a short optimization per site.

Writes an npz in the find_free_directions.py format (direction name
"kl_opt_00" per site) so downstream stages can consume it directly via
--directions-npz / --direction-name kl_opt_00, plus a summary json with
final KL and a CE check at the deployment and 4x magnitudes.

Usage (pod, from repo root):
  uv run python password_locking/0_directions/optimize_direction.py \
      --model Qwen/Qwen2.5-7B --out password_locking/results/qwen25_7b_klopt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from find_free_directions import build_eval_batch, compute_ce, load_eval_text  # noqa: E402
from lib.injection import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_SITES,
    SignatureInjector,
    measure_site_scales,
    resolve_site_module,
    seeded_unit,
    steering,
)


def optimize_site(model, injector, site, batch, scale, args, device):
    d = model.config.hidden_size
    v = torch.nn.Parameter(seeded_unit(d, args.seed).to(device))
    opt = torch.optim.Adam([v], lr=args.lr)
    chunks = list(batch.split(args.batch_size))
    final_kl = float("nan")
    pbar = tqdm(range(args.steps), desc=f"optimize {site}")
    for step in pbar:
        ids = chunks[step % len(chunks)].to(device)
        with torch.no_grad():
            base_logp = F.log_softmax(model(ids).logits.float(), dim=-1)
        unit = v / v.norm()
        vec = (args.alpha * scale * unit).unsqueeze(0).expand(ids.shape[0], d)
        injector.arm(torch.ones(ids.shape, dtype=torch.bool, device=device),
                     {site: vec})
        logp = F.log_softmax(model(ids).logits.float(), dim=-1)
        injector.disarm()
        # KL(base || injected), mean over tokens
        loss = F.kl_div(logp, base_logp, log_target=True,
                        reduction="none").sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final_kl = loss.item()
        if step % 20 == 0:
            pbar.set_postfix(kl=f"{final_kl:.5f}")
    with torch.no_grad():
        final = (v / v.norm()).detach().cpu()
    return final, final_kl


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optimize per-site free directions by minimizing output KL")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", default=",".join(DEFAULT_SITES))
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="injection magnitude as fraction of site scale")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--n-seqs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--data-file", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]

    print(f"loading {args.model} on {device} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval().requires_grad_(False)

    batch = build_eval_batch(tokenizer, load_eval_text(args.data_file),
                             args.n_seqs, args.seq_len)
    site_mods = {s: resolve_site_module(model, s) for s in sites}
    scale_batches = [(c, torch.ones_like(c)) for c in batch.split(args.batch_size)]
    scales = measure_site_scales(model, site_mods, scale_batches, device)
    baseline_ce = compute_ce(model, batch, args.batch_size, device)
    print(f"baseline CE {baseline_ce:.4f}; scales "
          + ", ".join(f"{s}={scales[s]:.1f}" for s in sites))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}
    for site in sites:
        injector = SignatureInjector(model, [site])
        direction, final_kl = optimize_site(
            model, injector, site, batch, scales[site], args, device)
        injector.remove()

        checks = {}
        for mult, tag in [(1.0, "alpha"), (4.0, "4x_alpha")]:
            vec = (mult * args.alpha * scales[site] * direction).to(device)
            with steering(site_mods[site], vec, "all"):
                ce = compute_ce(model, batch, args.batch_size, device)
            checks[tag] = {"ce": ce, "dce": ce - baseline_ce}
        summary[site] = {"final_kl": final_kl, "scale": scales[site],
                         "alpha": args.alpha, **checks}
        print(f"{site}: KL {final_kl:.5f}, dCE@alpha "
              f"{checks['alpha']['dce']:+.4f}, dCE@4x "
              f"{checks['4x_alpha']['dce']:+.4f}")

        arrays[f"{site}/dirs"] = direction.unsqueeze(0).numpy()
        arrays[f"{site}/names"] = np.array(["kl_opt_00"])
        arrays[f"{site}/kinds"] = np.array(["kl_opt"])
        arrays[f"{site}/scale"] = np.asarray(scales[site], dtype=np.float32)

    np.savez(out_dir / "directions.npz", **arrays)
    (out_dir / "summary.json").write_text(json.dumps(
        {"model": args.model, "baseline_ce": baseline_ce,
         "config": vars(args), "sites": summary}, indent=2))
    print(f"wrote directions.npz + summary.json to {out_dir}")


if __name__ == "__main__":
    main()
