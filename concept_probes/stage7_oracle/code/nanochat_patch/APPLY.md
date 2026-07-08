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
- `precompute_coords.py`    -> offline: encoder over corpus -> coords memmap (run BEFORE training; still a SKELETON below main())

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
# 1) precompute coords first (long pole; separate pod(s))
python precompute_coords.py --encoder-ckpt <expA.pt> --probe-set out/probe_set.json \
    --shards 0-190 --out /workspace/coords     # NOTE: still a skeleton — wire the encoder first
# 2) train — call scripts.base_train directly (no_value_embeds.sh does not
#    forward --inject-*); the full copy-ready command incl.
#    --inject-after-block=7 is in out/nanochat_prep.md §5 (steps 4-5).
```
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

NOT validated here (needs GPU/real run): torch.compile + fp8 on the coord graph,
real-throughput of the CoordSource index (27M-entry dict, ~2-3 GB/rank), and
the precompute skeleton's encoder wiring (still unimplemented).
