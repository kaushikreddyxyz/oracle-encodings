# Phase 2b benchmark — TL-eager vs torch-SDPA vs torch-SDPA+compile

Hardware: 1x NVIDIA A100-SXM4-80GB (RunPod SECURE, US-MD-1), torch 2.8.0+cu128,
TransformerLens 2.18.0. Script: `maze/infra/bench_stacks.py`
(`python maze/infra/bench_stacks.py --pretok <pretok.pt>`).

One measurement = one full training step: forward + backward + `AdamW.step()`,
median of 20 timed steps after 6 warmup steps. "peak" is
`torch.cuda.max_memory_allocated` over the timed steps. "max bs" is the largest
power-of-two batch that survives a step (compiled stacks built with
`dynamic=True` for that search so dynamo does not recompile per size).

Model: the jirpy architecture as trained — d_model 256, 12 layers, 16 heads,
d_head 16, d_mlp 1024, d_vocab 47, learned absolute positions, LN.

`padding_mode=right` is not an approximation: with right-padded batches the
causal mask already excludes every padded key from every real query, so real
positions are bit-identical (measured: loss delta exactly 0.0 in fp32) and only
the ignored padded rows differ. It skips building the (batch, 1, pos, pos) key
mask, which is what lets SDPA stay on the flash kernel.

## Maze regime — bs 256, seq 179 (real 20k-maze pretok, 9.3% padding)

| precision | stack | steps/s | speedup | peak GiB | max bs |
| --- | --- | ---: | ---: | ---: | ---: |
| fp32-TF32 | TL-eager | 3.412 | 1.00x | 26.65 | 512 |
| fp32-TF32 | torch-SDPA | 7.189 | 2.11x | 9.83 | 1024 |
| fp32-TF32 | torch-SDPA + compile | 8.095 | 2.37x | 9.44 | 2048 |
| fp32-TF32 | torch-SDPA + compile, padding_mode=right | 10.087 | **2.96x** | 8.88 | 2048 |
| bf16 | TL-eager | 3.511 | 1.03x | 19.83 | 512 |
| bf16 | torch-SDPA | 12.403 | 3.64x | 5.66 | 2048 |
| bf16 | torch-SDPA + compile | 16.613 | 4.87x | 5.39 | 2048 |
| bf16 | torch-SDPA + compile, padding_mode=right | 20.869 | **6.12x** | 5.07 | 2048 |

Speedups are against TL-eager at fp32-TF32, which is exactly what
`legacy/train_fast.py` ran.

## ICL regime — bs 64, seq 1500 (synthetic, 10.8% padding, n_ctx 2048)

| precision | stack | steps/s | peak GiB | max bs |
| --- | --- | ---: | ---: | ---: |
| fp32-TF32 | TL-eager | **OOM at bs 64** | >79 | 16 |
| fp32-TF32 | torch-SDPA | 0.824 | 26.03 | 128 |
| fp32-TF32 | torch-SDPA + compile | 0.851 | 20.11 | 128 |
| fp32-TF32 | torch-SDPA + compile, padding_mode=right | 1.602 | 18.47 | 256 |
| bf16 | TL-eager | **OOM at bs 64** | >79 | 16 |
| bf16 | torch-SDPA | 1.963 | 14.43 | 256 |
| bf16 | torch-SDPA + compile | 2.153 | 11.33 | 256 |
| bf16 | torch-SDPA + compile, padding_mode=right | 7.645 | 10.45 | 256 |

TL-eager cannot run this regime at bs 64 on an 80 GB A100 at all: it
materializes a (batch, n_heads, pos, pos) score tensor per layer and keeps the
softmax pattern for the backward pass, i.e. 64 x 16 x 1500^2 x 4 B = 8.6 GiB per
layer. Its largest feasible batch at seq 1500 is **16**, at 1.691 steps/s
(69.2 GiB peak, fp32-TF32) and 1.563 steps/s (54.8 GiB, bf16). Per sequence that
is 27.1 seq/s vs 102.5 seq/s for torch-SDPA+compile+right at fp32-TF32
(**3.8x**) and 25.0 seq/s vs 489.3 seq/s at bf16 (**19.6x**).

## Reading

In the maze regime the torch stack is 2.1x faster than TransformerLens before
`torch.compile` even runs, and 3.0x / 6.1x with compile at fp32-TF32 / bf16 —
while using a third to a quarter of the memory and fitting 4x the batch. Most of
that first 2x is TL's per-step Python and its explicit attention-score tensor,
not arithmetic: peak memory drops from 26.7 to 9.8 GiB before any kernel fusion.
`torch.compile` adds a further 13% at fp32-TF32 and 34% at bf16, and bf16 itself
is worth 2.1x on the torch side but nothing at all on TL-eager (3.41 -> 3.51
steps/s), because TL's bottleneck there is launch overhead, not FLOPs. The long
sequence regime is where the difference stops being a speed question and becomes
a feasibility question: TL-eager cannot fit bs 64 at seq 1500 on an 80 GB card,
and the torch stack does it in 10-18 GiB. Note also that at seq 1500 the choice
of `padding_mode` dominates everything else — the explicit key mask pushes SDPA
off its flash kernel, costing 3.6x at bf16 (7.65 -> 2.15 steps/s), a penalty
that is only 1.3x at seq 179. For long-context work `padding_mode="right"` is
the setting that matters; `train.py` defaults to `"exact"` for continuity, so
this is a deliberate flag to flip.

## Caveats

- Every cell here uses ONE fixed batch shape. Real training pads each batch to
  its own longest sequence, so `torch.compile` will recompile a handful of times
  before dynamo switches that dimension to dynamic; the compiled numbers above
  are the steady state, not the first hundred steps.
- The injection path (`InjectionRuntime.features` + `injection_delta`) is not in
  these numbers. It is one `(batch, pos, 72) @ (72, d_model)` matmul plus the
  feature construction per step, i.e. small next to a 12-layer step, but it is
  not zero.
- `max bs` is a power-of-two probe (2 steps), not a bisection, so the true
  ceiling lies somewhere in [max bs, 2 x max bs).
- Each `(stack, precision)` cell calls `torch._dynamo.reset()` first. Without it
  the per-code-object recompile counter accumulates across cells, trips
  `config.recompile_limit`, and `torch.compile` silently falls back to eager —
  which is exactly what an earlier run of this script measured before the reset
  was added.
