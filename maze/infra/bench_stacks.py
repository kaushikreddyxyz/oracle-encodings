#!/usr/bin/env python3
"""Phase 2b benchmark: TL-eager vs torch-SDPA vs torch-SDPA+compile.

Measures a full training step (forward + backward + AdamW step) for the jirpy
architecture in two regimes:

  maze  bs 256, seq ~165   the actual maze-injection training regime; tokens
                           come from a real pretok file when one is given, so
                           the right-padding profile is realistic
  icl   bs 64,  seq 1500   the long-context regime the project is heading for
                           (synthetic tokens, n_ctx raised to 2048)

For each (stack, regime, precision) it reports median steps/sec, peak allocated
memory, and the largest batch size that fits. Results go to maze/core/BENCH.md.

  python maze/infra/bench_stacks.py --pretok /workspace/runs/bench/pretok-*.pt \
      --out /workspace/bench.json
"""

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))
import maze._pathsetup  # noqa: E402,F401  (cross-experiment sys.path shim)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from transformer_lens import HookedTransformer, HookedTransformerConfig  # noqa: E402

from maze.core.data import PretokDataset, collate_pretok  # noqa: E402
from maze.core.model import MazeGPT, MazeGPTConfig  # noqa: E402
from maze.core.tl_bridge import tl_to_torch_state_dict  # noqa: E402

ARCH = dict(d_model=256, d_head=16, n_heads=16, n_layers=12, d_mlp=1024,
            d_vocab=47, d_vocab_out=47, act_fn="gelu", normalization_type="LN",
            positional_embedding_type="standard")
PAD_ID = 10  # <PADDING> in the grid-6 maze vocab


def make_models(n_ctx, device):
    tl_cfg = HookedTransformerConfig(n_ctx=n_ctx, **ARCH)
    tl = HookedTransformer(tl_cfg).to(device)
    mcfg = MazeGPTConfig.from_tl_cfg(tl.cfg)
    torch_model = MazeGPT(mcfg).to(device)
    torch_model.load_state_dict(tl_to_torch_state_dict(tl.state_dict(), mcfg))
    return tl.train(), torch_model.train(), mcfg


def maze_batch(pretok_path, bs, device):
    """One real right-padded maze batch (falls back to synthetic if no pretok)."""
    if pretok_path is not None and Path(pretok_path).exists():
        ds = PretokDataset(Path(pretok_path))
        idx = list(range(min(bs, len(ds))))
        while len(idx) < bs:
            idx += idx[: bs - len(idx)]
        tokens, attn, _, _ = collate_pretok([ds[i] for i in idx], pad_id=PAD_ID)
        return tokens.to(device), attn.to(device)
    return synth_batch(bs, 165, 0.9, device)


def synth_batch(bs, seq, min_frac, device):
    g = torch.Generator(device="cpu").manual_seed(0)
    lens = torch.randint(int(seq * min_frac), seq + 1, (bs,), generator=g)
    tokens = torch.randint(0, ARCH["d_vocab"], (bs, seq), generator=g)
    attn = (torch.arange(seq)[None, :] < lens[:, None]).long()
    tokens = torch.where(attn.bool(), tokens, torch.full_like(tokens, PAD_ID))
    return tokens.to(device), attn.to(device)


def _ac(precision, device):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def time_stack(stack, model, tokens, attn, precision, device, padding_mode,
               n_warmup=6, n_steps=20):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-8, weight_decay=0.1,
                            betas=(0.9, 0.95))
    is_tl = stack == "tl-eager"

    def step():
        with _ac(precision, device):
            if is_tl:
                loss = model(tokens, return_type="loss", attention_mask=attn)
            else:
                loss = model(tokens, attention_mask=attn, return_type="loss",
                             padding_mode=padding_mode)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    peak = torch.cuda.max_memory_allocated() / 2**30
    opt.zero_grad(set_to_none=True)
    del opt
    return 1.0 / times[len(times) // 2], peak


def max_batch(stack, build, make_batch, precision, device, padding_mode,
              start, cap=8192):
    """Largest power-of-two batch that survives 2 steps.

    Compiled stacks are built with `dynamic=True` here: without it dynamo
    recompiles at every batch size and the search costs more than the whole
    rest of the benchmark."""
    best, bs = 0, start
    while bs <= cap:
        model = tokens = attn = None
        try:
            model = build()
            tokens, attn = make_batch(bs)
            time_stack(stack, model, tokens, attn, precision, device,
                       padding_mode, n_warmup=1, n_steps=1)
            best = bs
        except torch.cuda.OutOfMemoryError:
            break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            break
        finally:
            del model, tokens, attn
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        bs *= 2
    return best


def main(pretok=None, out="bench.json", regimes="maze,icl",
         precisions="fp32-tf32,bf16", skip_max_batch=False):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__)

    plan = {
        "maze": dict(n_ctx=512, bs=256, cap=16384,
                     batch=lambda bs: maze_batch(pretok, bs, device)),
        "icl": dict(n_ctx=2048, bs=64, cap=2048,
                    batch=lambda bs: synth_batch(bs, 1500, 0.8, device)),
    }
    stacks = [
        ("tl-eager", False, "exact"),
        ("torch-sdpa", False, "exact"),
        ("torch-sdpa+compile", True, "exact"),
        ("torch-sdpa+compile (padding_mode=right)", True, "right"),
    ]
    results = []
    for regime in regimes.split(","):
        spec = plan[regime]
        tokens, attn = spec["batch"](spec["bs"])
        print(f"\n== regime {regime}: tokens {tuple(tokens.shape)}, "
              f"pad fraction {float((attn == 0).float().mean()):.3f}", flush=True)
        for precision in precisions.split(","):
            for name, do_compile, pm in stacks:
                # each cell gets a clean dynamo state: without this the
                # per-code-object recompile counter (autocast + dynamic-shape
                # guards from earlier cells) trips config.recompile_limit and
                # torch.compile silently falls back to eager
                torch._dynamo.reset()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                tl, tm, _ = make_models(spec["n_ctx"], device)
                model = tl if name == "tl-eager" else tm
                if do_compile:
                    model = torch.compile(model)
                try:
                    sps, peak = time_stack(name, model, tokens, attn, precision,
                                           device, pm)
                except Exception as e:
                    print(f"  {name:42s} {precision:9s} FAILED "
                          f"{type(e).__name__}: {str(e).splitlines()[0]}",
                          flush=True)
                    results.append(dict(regime=regime, precision=precision,
                                        stack=name, error=repr(e)))
                    del tl, tm, model
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                row = dict(regime=regime, precision=precision, stack=name,
                           batch=spec["bs"], seq=int(tokens.shape[1]),
                           steps_per_sec=sps, peak_gib=peak)
                if not skip_max_batch:
                    def build(n_ctx=spec["n_ctx"], nm=name, dc=do_compile):
                        a, b, _ = make_models(n_ctx, device)
                        m = a if nm == "tl-eager" else b
                        return torch.compile(m, dynamic=True) if dc else m
                    row["max_batch"] = max_batch(
                        name, build, spec["batch"], precision, device, pm,
                        start=spec["bs"], cap=spec["cap"],
                    )
                print(f"  {name:42s} {precision:9s} "
                      f"{sps:8.3f} steps/s  peak {peak:6.2f} GiB  "
                      f"max_bs {row.get('max_batch', '-')}", flush=True)
                results.append(row)
                del tl, tm, model
                gc.collect()
                torch.cuda.empty_cache()

    Path(out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretok", default=None)
    ap.add_argument("--out", default="bench.json")
    ap.add_argument("--regimes", default="maze,icl")
    ap.add_argument("--precisions", default="fp32-tf32,bf16")
    ap.add_argument("--skip-max-batch", action="store_true")
    a = ap.parse_args()
    main(pretok=a.pretok, out=a.out, regimes=a.regimes,
         precisions=a.precisions, skip_max_batch=a.skip_max_batch)
