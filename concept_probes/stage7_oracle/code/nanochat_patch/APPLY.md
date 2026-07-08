# Applying the Stage-7 coord-injection patch to nanochat

Do NOT edit the submodule in place on `main`. Apply on a throwaway branch inside
the nanochat submodule at launch time (the pod clones the submodule fresh).

Audited + regenerated 2026-07-08 (reasoning-tier audit): both diffs are now
`git diff` output taken against submodule HEAD `f7c3119` and verified with
`git apply --check`. Key audit fixes: injection default moved to **after
`transformer.h[7]`** (8 completed blocks, per the orchestrator addendum);
missing-doc fallback is now EXACT zero coords with **no noise** (noised zeros
would be renormalized to full beta amplitude = pure-noise injection); noise is
deterministic per (seed, doc-content-hash) — DDP-rank- and resume-independent.

## Files
- `gpt_inject.diff`         -> patches `nanochat/nanochat/gpt.py` (forward: `coords=` arg + after-block-N injection)
- `base_train_inject.diff`  -> patches `nanochat/scripts/base_train.py` (CLI flags, load P as non-persistent buffer, swap loader, pass coords)
- `coords_store.py`         -> imported by base_train (P, structured coords, CoordSource); lives in this dir, NOT the submodule
- `coord_dataloader.py`     -> imported by base_train (ride-along best-fit loader); same
- `precompute_coords.py`    -> offline: encoder over corpus -> coords memmap (run BEFORE training; IMPLEMENTED, CPU-tested via test_precompute_coords.py)

## Apply
```bash
cd nanochat
git checkout -b stage7-inject
git apply --check ../concept_probes/stage7_oracle/code/nanochat_patch/gpt_inject.diff        # dry-run first
git apply --check ../concept_probes/stage7_oracle/code/nanochat_patch/base_train_inject.diff
git apply ../concept_probes/stage7_oracle/code/nanochat_patch/gpt_inject.diff
git apply ../concept_probes/stage7_oracle/code/nanochat_patch/base_train_inject.diff
python -m nanochat.oracle.smoke      # existing oracle smoke still passes (injection is additive/off by default)
```
Diffs are anchored to submodule HEAD `f7c3119` ("wandb"). If HEAD moves, re-check
with `git apply --check`; each hunk is small and self-contained enough to apply
by hand.

**If nanochat is cloned standalone on the pod** (not inside the superproject),
base_train's relative import of this dir will not resolve — set
`STAGE7_PATCH_DIR=/path/to/nanochat_patch` in the environment (it must contain
`coords_store.py` + `coord_dataloader.py`).

## Toggle
- Absent `--inject-coords` => behaviorally identical to `no_value_embeds.sh`:
  the coord block is skipped entirely, no extra RNG draws (init stream
  untouched), stock dataloader, `coords=None` resolved at trace time (no graph
  change). Only cosmetic delta: 4 extra keys in the logged/checkpointed
  `user_config`.
- Present => contextual injection after block `--inject-after-block`
  (**default 7** = after 8 completed blocks of 24; the earlier draft default of
  8 was one block too deep), `--inject-beta` (0.05 = injected per-token RMS as a
  fraction of the residual per-token RMS; rms over n_embd, per token),
  `--inject-noise-sigma` (0.15, added at load time, keyed by doc hash).
- P is a fixed orthonormal (n_embd, r) buffer: non-trainable, non-persistent
  (excluded from optimizer param groups, weight decay, and checkpoints —
  checkpoints stay loadable by vanilla code). nanochat has no DDP module
  wrapper, so there is no buffer broadcast; every rank loads the same P.npy.
- Eval paths (val bpb, CORE, sampling) run coords=None (coords-off) — decide
  separately whether to add a coords-on eval pass (prep doc open decision 6).

## Launch (injected no-VE run, matched to the existing baseline)
```bash
# 1) precompute coords first (long pole; fleet of 4-6 H100 pods).
#    Prereqs on every pod: the baseline tokenizer at $NANOCHAT_BASE_DIR/tokenizer
#    (pull tokenizer/ from HF oracle_baseline_noVE_d24_fp8) and the karpathy
#    climbmix shards at $NANOCHAT_BASE_DIR/base_data_climbmix (python -m nanochat.dataset -n 191).
CO=concept_probes/stage7_oracle/code/nanochat_patch/precompute_coords.py
#  1a) pod 0 fits continents PCA + coord scale ONCE (shared by all pods):
python $CO --mode fit   --encoder-ckpt <expA.pt> --probe-set out/probe_set.json \
    --shards 0-3 --out /workspace/coords
#  1b) every pod sweeps its round-robin shard slice (resumable, atomic per shard):
python $CO --mode sweep --encoder-ckpt <expA.pt> --probe-set out/probe_set.json \
    --shards 0-190 --out /workspace/coords --pod-index $P --n-pods $NP
#  1c) after the fleet finishes, on ONE node with all shards/coords present:
python $CO --mode merge-stats --out /workspace/coords
python $CO --mode assemble   --probe-set out/probe_set.json --encoder-ckpt <expA.pt> \
    --shards 0-190 --out /workspace/coords    # -> coords.int8 / index.npy / meta.json / P.npy
#  1d) MANDATORY pre-launch gate (CPU, fast, on the training node): cross-check
#      the CONSUMER token path (RustBPETokenizer batch encode + BOS, exactly as
#      coord_dataloader) against the assembled store. Hard-fails on tokenizer
#      contract drift or coverage < 99.9% -- the failure it catches is the one
#      that otherwise silently trains a baseline (all lookups None -> zero coords).
python $CO --mode preflight --shards 0-190 --out /workspace/coords --preflight-docs 1024
#  optional QA:
python $CO --mode verify          --encoder-ckpt <expA.pt> --probe-set out/probe_set.json \
    --shards 0-0 --out /workspace/coords --verify-docs 64
python $CO --mode measure-crossing --shards 0-0 --out /workspace/coords   # audit open-item
# 2) train — call scripts.base_train directly (no_value_embeds.sh does not
#    forward --inject-*); the full copy-ready command incl.
#    --inject-after-block=7 is in out/nanochat_prep.md §5 (steps 4-5).
```
Coord-standardization design note (important): quantization is **zero-preserving**
(no mean-centering) so a concept-free token (raw coord 0) stays int8 0 -> the
self-normalizing injection is an exact no-op there. Per-column mean/std ARE
recorded in `meta.json` (required artifact) but only the single global `scale`
is applied by the loader (`CoordSource` = `int8 * scale`).
SMOKE first (3 steps, nothing saved) to validate build + torch.compile + one
step with coords; watch step time (should be within ~3% of baseline) and that
the `--inject-coords` banner prints the expected r=14 / after_block=7 / docs
count.

## Validated on CPU (`test_coord_lockstep.py` in this dir — rerun anytime, no GPU/pyarrow needed)
- token stream of the ride-along loader is bit-identical to the REAL
  `tokenizing_distributed_data_loader_with_state_bos_bestfit` source (extracted
  by ast) across best-fit picks and crops, incl. state dicts;
- every packed position's coord matches its (doc, body-token) store row exactly
  through packing + cropping; BOS rows and missing/length-mismatched docs are
  exactly zero even with noise on; noise deterministic per (seed, doc hash);
- CoordSource int8 round-trip exact; miss/mismatch -> None;
- injection math: zero-coord rows are an exact no-op; injected per-token RMS ==
  beta * per-token RMS(x) to 1e-8; no NaN with mixed zero/nonzero rows;
- build_coords: all 54 one-hot main-block columns land on the correct family
  ring phase (permutation fix verified); P orthonormal/deterministic/isometric;
- byte->char offset reconstruction survives adversarial mid-UTF-8-char token
  splits (200 randomized trials) and asserts on non-partitioning ids.

## Precompute audit (2026-07-08, reasoning-tier): fixes applied
- **`--mode preflight` added (MANDATORY before training)**: consumer-path
  cross-validation; see step 1d above. Also asserts `meta.json` has no
  missing shards and that `enc.encode_ordinary` (producer) agrees with
  `tokenizer.encode(batch, prepend=bos)` minus BOS (consumer) per doc.
- **assemble now hard-fails on missing shards** (`--allow-missing-shards` to
  override): a partial store would silently zero-coord the missing shards.
- **Welford stats are now per-shard** (in `meta_<sid>.json`), merged by
  `--mode merge-stats` from those; a pod crash+resume no longer loses or
  double-counts observed-coordinate stats (scale itself is fixed at fit time
  and was never at risk).
- **fit/sweep/assemble consistency guards**: sweep and assemble refuse to run
  if `coord_fit.npz` `pred_order`/`legend` differ from the resolved probe set
  (prevents phase angles attaching to the wrong concepts after a probe_set
  regeneration mid-pipeline).
- **CoordEngine flush chunked to `--batch-seqs`**: a single giant doc used to
  produce one monolithic padded forward over ALL its windows (OOM risk);
  fit/sweep/verify all share the chunked path (equivalence unit-tested).
- byte->char offset map vectorized (per-byte python loop was a CPU stall on
  the 13.5B-token sweep).

NOT validated here (needs GPU/real run): torch.compile + fp8 on the coord graph,
real-throughput of the CoordSource index (27M-entry dict, ~2-3 GB/rank), and the
precompute encoder wiring on the REAL Exp-A checkpoint + REAL tiktoken/qwen
tokenizer pair (precompute logic is CPU-tested with a byte-BPE stand-in +
tiny-Qwen2 model in test_precompute_coords.py, but the real forward, the real
crossing rate, and coord-scale/clip fractions on real preds are GPU-only).
