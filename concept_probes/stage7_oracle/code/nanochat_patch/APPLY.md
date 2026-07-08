# Applying the Stage-7 coord-injection patch to nanochat

Do NOT edit the submodule in place on `main`. Apply on a throwaway branch inside
the nanochat submodule at launch time (the pod clones the submodule fresh).

## Files
- `gpt_inject.diff`         -> patches `nanochat/nanochat/gpt.py` (forward: `coords=` arg + after-block-N injection)
- `base_train_inject.diff`  -> patches `nanochat/scripts/base_train.py` (CLI flags, load P, swap loader, pass coords)
- `coords_store.py`         -> imported by base_train (P, structured coords, CoordSource); lives in this dir, NOT the submodule
- `coord_dataloader.py`     -> imported by base_train (ride-along best-fit loader); same
- `precompute_coords.py`    -> offline: encoder over corpus -> coords memmap (run BEFORE training)

## Apply
```bash
cd nanochat
git checkout -b stage7-inject
git apply ../concept_probes/stage7_oracle/code/nanochat_patch/gpt_inject.diff
git apply ../concept_probes/stage7_oracle/code/nanochat_patch/base_train_inject.diff
python -m nanochat.oracle.smoke      # existing oracle smoke still passes (injection is additive/off by default)
```
The two `.diff` line numbers are anchored to the current submodule HEAD
(gpt.py forward @415, block loop @469; base_train @86/@191/@279/@365/@562).
If they drift, apply the hunks by hand — each is small and self-contained.

## Toggle
- Absent `--inject-coords` => byte-identical to `no_value_embeds.sh` (coords=None, no graph change).
- Present => contextual injection after block `--inject-after-block` (default 8),
  `--inject-beta` (0.05), `--inject-noise-sigma` (0.15).

## Launch (injected no-VE run, matched to the existing baseline)
```bash
# 1) precompute coords first (long pole; separate pod(s))
python precompute_coords.py --encoder-ckpt <expA.pt> --probe-set out/probe_set.json \
    --shards 0-190 --out /workspace/coords
# 2) train (same recipe as no_value_embeds.sh + injection flags)
INJECT_COORDS=/workspace/coords \
NO_VALUE_EMBEDS=1 SEED=1337 DEPTH=24 RATIO=12 PRECISION=fp8 \
bash runs/oracle_runs/no_value_embeds.sh   # (add the --inject-* passthrough, see nanochat_prep.md)
```
Note: `no_value_embeds.sh` does not yet forward `--inject-*`; either add a small
`INJECT_FLAGS` passthrough to `baseline.sh`'s torchrun block, or call
`scripts.base_train` directly with the full flag set (nanochat_prep.md has the
exact command). SMOKE=1 first to validate build+compile+one step with coords.
```
