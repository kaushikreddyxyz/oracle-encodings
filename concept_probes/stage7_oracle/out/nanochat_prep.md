# Phase 4 nanochat injection — prep (investigation + code draft, NOT launched)

Written 2026-07-08 by the nanochat-prep agent. Scope: prepare the injected d24
run matched to the existing **no-VE** baseline. No pods, no launches, no commits,
no in-place submodule edits. Patch draft in `code/nanochat_patch/`.

---

## 1. Baselines — which is no-VE, recipe, budget

Both baselines already trained and live on HF (public). Repo names are
unambiguous; CORE/bpb pulled from each repo's `report.md`:

| run | HF repo | value embeds | CORE (final) | val bpb (final) | train time |
|---|---|---|---|---|---|
| WITH-VE | `kaushikreddyxyz/oracle_baseline_d24_fp8` | learned | **0.2777** | 0.7018 | — |
| **no-VE (tonight's match)** | `kaushikreddyxyz/oracle_baseline_noVE_d24_fp8` | zeroed+frozen | **0.2711** | 0.7091 | **8791 s = 2.44 h** |

→ **The no-VE baseline is `oracle_baseline_noVE_d24_fp8`, CORE 0.2711, val_bpb 0.7091.**
VEs help (0.2777 > 0.2711), as expected. Both repos carry model+optimizer
checkpoints at steps {2000,4000,6000,8000,**8352**(final)}, tokenizer, report.md.
The **tokenizer** (`tokenizer/tokenizer.pkl` + `token_bytes.pt`) is identical
across both — reuse it for the injected run (SPEC: keep nanochat's own vocab).

### Exact recipe to reproduce (from baseline.sh + the no-VE run's final `meta_008352.json`)
- Launcher: `nanochat/runs/oracle_runs/no_value_embeds.sh` → sets
  `NO_VALUE_EMBEDS=1` and execs `baseline.sh` (they can never drift).
- `depth=24, ratio=12, max_seq_len=2048, device_batch_size=16, nproc=8`,
  `precision=fp8` (bf16 base + fp8 GEMM, recipe **tensorwise**), `seed=1337`.
- `window_pattern="SSSL"`, aspect 64, head_dim 128, n_head 12, **n_embd 1536**,
  padded vocab 32768.
- **`--no-value-embeds`**: zero + freeze all 12 value-embedding tables AFTER
  init_weights (RNG order preserved → bit-identical init to the WITH-VE run given
  the same seed).
- LRs (from report): embedding 0.30, unembedding 0.008, matrix 0.02, scalar 0.50,
  weight_decay 0.28, warmup 40 steps, warmdown_ratio 0.65, final_lr_frac 0.05.
- **Batch/horizon (auto-computed, now pinned): `total_batch_size = 1,048,576`
  (2^20) tokens; `num_iterations = 8352`.** grad_accum = 1,048,576 /
  (16·2048·8) = **4**.
- **Token budget = 8352 × 1,048,576 = 8,757,706,752 ≈ 8.758 B tokens.** (= ratio
  12 × 729.8 M scaling params.) This is the number the injected run MUST match
  exactly (no shortening — the baseline can't be re-shortened).
- Dataset: **274 train shards auto-sized** (`ceil(1.25·8.758B/40M)`), +1 val.
  The no-VE run's final dataloader state was `pq_idx=184, rg_idx=16, epoch=1`
  → it consumed ~**185 of 274** parquet files, stayed within 1 epoch (no repeats).

### Wall time / cost (8×H100 node)
- no-VE baseline: **2.44 h pure training** (report `total_training_time`=8791 s;
  wall incl. tokenizer-reuse + eval ≈ 3 h). Report shows the node billed at
  **$24/hr** (not $26). → training ≈ **$59**, full run ≈ **$72**.
- Injected run ≈ baseline + <3 % (one small matmul at one block + coord I/O) →
  **~2.5 h train, ~3 h wall, ~$62 train / ~$75 full.** (baseline.sh's own
  "$110 / 4.2 h" is a conservative pre-run estimate; actuals are lower.)

---

## 2. nanochat internals

### (a) Value embeddings & how no-VE disables them
- ResFormer value embeddings: `GPT.value_embeds = ModuleDict{str(i): Embedding(pad_vocab, kv_dim)}`
  for layers with `has_ve(i, n_layer)` = `i % 2 == (n_layer-1) % 2` → **12 of 24
  layers** (odd layers for n_layer=24). In `forward`, per block:
  `ve = value_embeds[str(i)](idx)`; the block's attention adds `ve_gate·ve` to
  values. `--no-value-embeds` zeroes every `ve.weight` and sets
  `requires_grad_(False)` **after** init (base_train.py:191-196) → `v = v +
  gate·0 = v`, no contribution, no learning. wte is NOT frozen by this flag (only
  the VEs). The SPEC's "freeze/zero wte+value_embeds" for the *smoke test* is a
  stronger condition; the **baseline uses only `--no-value-embeds`**, so the
  injected run matches by using ONLY `--no-value-embeds` (do NOT also freeze wte —
  that would diverge from the baseline recipe).

### (b) d24 dims + "after block 8" site
- `n_embd = 1536`, `n_layer = 24`, 12 heads × head_dim 128. Confirmed from the
  live meta: `{"n_layer":24,"n_head":12,"n_embd":1536,"window_pattern":"SSSL"}`.
- Trunk (gpt.py:469-474): `for i,block in enumerate(h): x = λ_res[i]·x +
  λ_x0[i]·x0; ve=...; x = block(...)`. Backout cached at `n_layer//2 = 12`.
  **Injection site = right after `x = block(...)` when `i == inject_after_block`
  (default 8, 0-based → immediately after `transformer.h[8]`).** This is inside
  the residual band, before the halfway backout — mid-early, matching the gemma
  "concepts live mid-band" finding. (The existing `oracle_fn` hook injects a
  per-token-**id** table at the *embedding* level, before block 0 — that is the
  WRONG site and WRONG granularity for Stage 7; we add a new contextual site.)

### (c) Pretraining data pipeline & determinism  ← **key finding**
- **The corpus is ClimbMix, NOT fineweb-edu.** nanochat switched FinewebEdu-100B
  → **ClimbMix-400B on 2026-03-04** (`dataset.py`: `BASE_URL =
  huggingface.co/datasets/karpathy/climbmix-400b-shuffle`, `DATA_DIR =
  base_data_climbmix`). Every SPEC/DESIGN mention of "fineweb-edu shards" for the
  nanochat corpus is stale. **Good news:** this is the same ClimbMix family as
  the Phase-1 probe-scoring corpus — but a *different repackaging* (karpathy's
  shuffled 400B repack vs `nvidia/ClimbMix` shards ≥320 used for scoring), so
  shard ids / doc order / shuffle differ. Coords must be computed over
  **karpathy/climbmix-400b-shuffle** (the shards nanochat actually downloads),
  not the nvidia scoring shards.
- Storage: parquet with a **`text` column (raw text)**, row-group batched. Not
  pre-tokenized → the coord precompute reads the same raw text and does its own
  char-offset alignment (no detokenization needed).
- DDP split (`dataloader._document_batches`): rank r reads row-groups
  `r, r+world, r+2·world, …` of each parquet file in order; within a row group,
  docs in stored order, batched by `tokenizer_batch_size=128`. **No RNG, no
  shuffle** — deterministic given {shard set, world_size, B, T}.
- Packing (`…bos_bestfit`): each doc gets BOS prepended, then **best-fit packed**
  into rows of length T+1=2049 — pick the *largest buffered doc that fits*; when
  none fits, *crop the shortest* to fill exactly. ~**35 % of tokens are cropped
  away**. Best-fit is deterministic but **data-dependent** (depends on the
  1000-doc buffer contents), and packing **discards each token's document
  provenance** — so a flat per-training-position coord memmap cannot be
  pre-aligned. This is the crux that shapes the storage design.

---

## 3. Injection patch (draft in `code/nanochat_patch/`)

Contextual, per-token-occurrence coords (not per-id). Three small pieces:

1. **`gpt_inject.diff`** — `forward(..., coords=None)`; after block
   `inject_after_block`:
   ```
   zc = coords @ P.T                                   # (B,T,1536), P orthonormal (1536×r)
   x = x + beta * (rms(x)/rms(zc).clamp_min(1e-8)) * zc
   ```
   Self-normalizing: since injected_rms = beta·rms(x)/rms(zc)·rms(zc) =
   **beta·rms(x)** exactly per token → β directly = injected fraction of residual
   RMS (SPEC's "β=0.05 of residual RMS at the site"). Zero-coord tokens (BOS /
   no-concept) → zc=0 → term=0 via the clamp (no NaN, no branch).
   **torch.compile/DDP-safe:** `coords is None` and `inject_after_block` are
   trace-time constants; the body is pure matmul/elementwise — no data-dependent
   control flow, no graph break. P is a **non-trainable buffer** (`persistent=
   False`), so it stays out of the optimizer param-group partition and weight
   decay.
2. **`base_train_inject.diff`** — adds `--inject-coords/-after-block/-beta/
   -noise-sigma`; loads fixed `P.npy`, registers the buffer, builds `CoordSource`,
   and (only when `--inject-coords` given) swaps in the ride-along loader and
   passes `coords=z`. Absent → byte-identical to `no_value_embeds.sh`.
3. **`coords_store.py` + `coord_dataloader.py`** (our modules, outside the
   submodule): fixed orthonormal P (seed 1337, QR of a Gaussian), structured
   coord construction, hash-keyed int8 coord store, and the ride-along best-fit
   loader that carries a `(B,T,r)` coord tensor in lockstep with tokens (same
   best-fit pick, same crop → coords never desync from tokens; **DDP-/order-
   independent because keyed by doc-content hash, not iteration position**).

Reviewability: two ~25-line diffs against the submodule + two self-contained
modules. `nanochat.oracle.smoke` still passes (injection is purely additive and
off by default).

---

## 4. Coords precompute plan (numbers)

**Encoder → coords.** Frozen best Exp-A Qwen3-0.6B-Base + up-head, run over the
karpathy ClimbMix shards nanochat trains on. Per nanochat token, via `align.py`
prefix mode (validated 7.08 % crossing gemma→qwen; tokenizer-agnostic so it
serves qwen→nanochat directly): map nanochat token → last qwen token whose char
span ends ≤ end-char(t) → gather that Qwen hidden → head → predicted probe scores
`ŝ ∈ R^{3K}`. **Use the gemma-layer-8 block** (the K=54 columns for gemma layer 8
= the ablation/causal-salient layer, and the middle of the chosen [6,8,14]).
Concern flagged below.

**Structured coords (from `out/probe_set.json`, K=54 survivors).** Family layout:

| family | classes | coord | dims |
|---|---|---|---|
| months | 12 (cyclic) | ring: (Σŝ cosθ, Σŝ sinθ), θ=2πk/12 | 2 |
| weekdays | 7 (cyclic) | ring | 2 |
| seasons | 4 (cyclic) | ring | 2 |
| directions | 8 (cyclic) | ring (compass order N,NE,E,…) | 2 |
| moon_phases | 8 (cyclic) | ring (synodic order new→…) | 2 |
| color_wheel | 9 (cyclic) | ring (wheel order red→…→violet) | 2 |
| continents | 6 (non-cyclic) | saved family-PCA-2D | 2 |
| **total** | | | **r = 14** |

Canonical cyclic orderings are encoded in `coords_store.CYCLIC_ORDER` (calendar/
wheel order, NOT the alphabetical order concepts appear in probe_set.json — this
is a real correctness detail). Coords standardized on corpus stats; **gaussian
noise σ = 0.15·RMS** added at *train time* in the loader (fresh, seeded — not
baked, so the model can't memorize per-token noise as identity). r embeds into
1536 via fixed orthonormal P; β=0.05.

**Compute.** Encoder must produce coords for **all raw doc tokens in the consumed
shards** (~185 files; 35 % crop ⇒ raw ≈ 8.758 B / 0.65 ≈ **13.5 B tokens**;
precompute shards 0–190 to be safe). Qwen-0.6B forward-only bf16 @ ~100 k tok/s/
H100 (SPEC est.; likely 100–180 k):

- GPU-h = 13.5 B / (100–180 k) = **21–38 GPU-h**.
- Fleet 4×H100 → **5.3–9.4 h wall**. @ ~$3/hr/H100 → **$63–$113** (budget
  ~$90 planning point).

**Storage.** r=14 int8 → 14 B/token × 13.5 B = **~190 GB** (fp16 → 380 GB; use
int8). Index (~27 M docs × 20 B) ≈ 0.5 GB. Fits a 400 GB pod NVMe (or the ~750 GB
encoder pod already planned in STATE). Format: `coords.int8` memmap [N,14] +
`index.npy` (hash,off,n) + `P.npy` + `meta.json`.

**Precompute vs inline.** Precompute recommended IF it can start by ~4 AM (SPEC).
It is the long pole. Fallback = inline Qwen per rank during training (+~10 % step
time, zero storage, more risk); the ride-along loader is only needed for the
precompute path — inline would compute z inside the training step instead.

---

## 5. Launch checklist (copy-ready — do NOT run yet)

> AUDIT EDIT 2026-07-08: `--inject-after-block` corrected 8 -> **7** in steps
> 4-5 below (orchestrator addendum #2: after 8 COMPLETED blocks = after
> `transformer.h[7]`); the patch default is now also 7. Diffs regenerated and
> `git apply --check`-verified; see `code/nanochat_patch/APPLY.md`.

```bash
# 0. Preconditions: G2 GO (or marginal+caveat); Exp-A checkpoint chosen;
#    corpus_stats + continents PCA saved into/next to probe_set.json.

# 1. Precompute coords (fleet; long pole — start first)
#    per pod: split --shards across the fleet (0-47, 48-95, ...)
python concept_probes/stage7_oracle/code/nanochat_patch/precompute_coords.py \
    --encoder-ckpt <expA.pt> --probe-set concept_probes/stage7_oracle/out/probe_set.json \
    --shards 0-190 --out /workspace/coords
#    -> consolidate coords.int8/index.npy/P.npy/meta.json onto the training node

# 2. Stage nanochat on the 8×H100 node, apply patch on a throwaway branch
cd nanochat && git checkout -b stage7-inject
git apply ../concept_probes/stage7_oracle/code/nanochat_patch/gpt_inject.diff
git apply ../concept_probes/stage7_oracle/code/nanochat_patch/base_train_inject.diff
python -m nanochat.oracle.smoke        # sanity: existing oracle path still green

# 3. Download the SAME 274 ClimbMix shards + reuse the baseline tokenizer
#    (pull tokenizer/ from HF repo oracle_baseline_noVE_d24_fp8 into $NANOCHAT_BASE_DIR/tokenizer)
python -m nanochat.dataset -n 274

# 4. SMOKE (build+compile+coords, one step, nothing saved)
NO_VALUE_EMBEDS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=24 --target-param-data-ratio=12 --max-seq-len=2048 --device-batch-size=16 \
  --no-value-embeds --fp8 --fp8-recipe=tensorwise --seed=1337 \
  --inject-coords=/workspace/coords --inject-after-block=7 --inject-beta=0.05 --inject-noise-sigma=0.15 \
  --num-iterations=3 --total-batch-size=393216 --save-every=0 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --run=dummy

# 5. FULL injected no-VE run (matched budget: ratio 12 -> 8352 steps, 8.758B tok)
screen -L -Logfile runs/oracle_runs/inject_noVE.log -S inject torchrun --standalone --nproc_per_node=8 \
  -m scripts.base_train -- \
  --depth=24 --target-param-data-ratio=12 --max-seq-len=2048 --device-batch-size=16 \
  --model-tag=oracle_inject_noVE_d24_fp8 \
  --no-value-embeds --fp8 --fp8-recipe=tensorwise --seed=1337 \
  --inject-coords=/workspace/coords --inject-after-block=7 --inject-beta=0.05 --inject-noise-sigma=0.15 \
  --save-every=2000 --save-optimizer=every --compress-checkpoints=1 \
  --eval-every=250 --core-metric-every=2000 --sample-every=2000 --run=inject_noVE_d24_fp8_r12

# 6. Eval (CORE + bpb) — same as baseline; G4 post-hoc: linear-decode injected
#    concepts from residuals of injected vs baseline model.
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
  --device-batch-size=16 --model-tag=oracle_inject_noVE_d24_fp8
```
`total_batch_size` is left auto in step 5 (→ 1,048,576, matching the baseline);
in the smoke step it's set small on purpose. G4 kill rule: if train bpb diverges
>5 % from the baseline curve within 2k steps, stop and inspect β.

---

## 6. OPEN DECISIONS (need a human/orchestrator call)

1. **Corpus mismatch (highest).** Coords must be precomputed over
   **karpathy/climbmix-400b-shuffle** (nanochat's actual download), *not* the
   nvidia ClimbMix scoring shards. The encoder was trained on nvidia-ClimbMix
   distribution; running it on the karpathy repack is an in-distribution use
   (same underlying ClimbMix), but the *exact shards/docs differ*. Confirm this
   is acceptable (it is the only faithful option — the model trains on karpathy
   shards). No fineweb-edu anywhere despite SPEC wording.
2. **Which encoder layer-block builds the coords.** Orchestrator's tentative
   "layer-8 block" is sound (ablation layer; [6,8,14] AUROCs within 0.005, highly
   correlated). Cheap to verify: in Exp A, pick the block with best heldout R²
   among the 3 — it's just a column slice, swappable without recompute. Flag: if
   layer 8's R² is materially worse than 6/14 for the cyclic families that
   dominate the injection, use the better one. Low risk.
3. **`inject_after_block` convention.** Draft = 0-based, injects after
   `transformer.h[8]` (9th block). If "after block 8 of 24" was meant 1-based
   (after the 8th block = index 7), set `--inject-after-block=7`. One-line flag.
4. **nanochat char offsets.** nanochat's tokenizer is RustBPE/**tiktoken**, which
   has NO `return_offsets_mapping`. The precompute reconstructs char spans from
   per-token decoded bytes (`decode_single_token_bytes`) + utf-8 byte→char
   mapping (`precompute_coords.nanochat_char_offsets`). This is standard for
   byte-level BPE but should be unit-tested against a few docs before the full
   sweep (mirror `test_align.py`). Not a blocker, just untested here.
5. **Noise: fresh vs baked.** Draft adds σ=0.15 noise fresh in the loader
   (seeded). Since the run is <1 epoch each token is seen once regardless, so
   baked-at-precompute would be equivalent and simpler to reproduce. Pick one;
   default = fresh.
6. **Eval with or without coords.** `base_eval` currently runs coords=None
   (measures the LM standalone). For CORE/bpb-vs-baseline the standalone number
   is the fair comparison, but the model was trained *with* the injection scaffold
   — decide whether the headline metric is coords-off (does the LM improve?) or
   coords-on (full system). Recommend reporting BOTH; trivial to add a coords-on
   eval pass.
7. **`no_value_embeds.sh` does not forward `--inject-*`.** Either add an
   `INJECT_FLAGS` env passthrough to baseline.sh's torchrun block, or launch
   `scripts.base_train` directly (step 5 above does the latter — recommended, no
   submodule script edit needed).
8. **Coord coverage vs shards consumed.** Baseline consumed ~185/274 files;
   precompute 0–190 covers it with margin. If the injected run's packing drifts
   slightly (best-fit is data-dependent and fp8 atomics are non-deterministic),
   any doc beyond the precomputed range falls back to zero coords (injection
   no-op there) — safe, ≤ a few % of tokens at worst.
```

---

## Orchestrator addendum (~5:00 AM)

1. **Open decision 1 RESOLVED — no corpus mismatch.** Phase-1 probe scoring
   also runs on `karpathy/climbmix-400b-shuffle` (verified against
   stage6 nat_common.py by the bench agent), shards 320-362; nanochat
   consumes shards ~0-185 of the SAME repack. Same distribution, disjoint
   shards — ideal (no leakage, no shift). The "nvidia/ClimbMix" mention in
   SPEC was the stale reference, not the scoring code.
2. **Open decision 3 RESOLVED — injection site = after 8 COMPLETED blocks**
   (0-based: after `transformer.h[7]`), depth fraction 8/24 = 0.33,
   mirroring gemma's causal band L8/26 = 0.31. Set `--inject-after-block`
   accordingly (patch default of h[8] is one block too deep).
3. Decision 2 (layer-8 prediction block for coords) stands, pending Exp-A
   per-layer R² confirmation. Decision 5: report CORE/bpb coords-on AND
   coords-off for the injected model.
