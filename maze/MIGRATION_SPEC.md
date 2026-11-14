# Maze repo migration — binding spec (2026-08-06)

Goal: ONE clean maze repo at `maze/`, torch-native training (fused SDPA
kernels, torch.compile), a lossless bidirectional bridge to TransformerLens
for all interp/analysis work, and per-experiment folders. HARD CONSTRAINTS:
no result/figure/report file may be lost or altered (byte-identical moves
only); every previously-working TL analysis path must still work; "complete"
injection semantics stay bit-identical.

## Phase 0 — safety (before ANY move)

1. Tar backup of the entire current `maze_injection/` (excluding __pycache__)
   to `~/Projects/oracle-encoding-project/maze_injection_backup_<UTCdate>.tar.gz`
   (OUTSIDE the repo). Verify the tarball lists >100 files.
2. Write a sha256 manifest of every file under maze_injection/ (excluding
   __pycache__) to `maze/migration_manifest_before.txt`. After migration,
   re-hash everything in its new location and verify EVERY hash is present
   somewhere (moved, never modified). Data/figure/report files must be
   byte-identical; .py files may change (imports) — list changed .py files
   explicitly in `maze/migration_changes.md` with one-line reasons.

## Phase 1 — consolidation layout

```
maze/
  MIGRATION_SPEC.md                (this file)
  core/                            # shared library (importable as maze.core)
    injection.py                   # MazeInjector, InjectionRuntime, MazeFeatureComputer
    teacher.py
    data.py                        # was fast_data.py
    evals.py                       # was eval_injection.py (ProbeSet, rollouts, suite)
    checkpoints.py                 # was eval_counterfactual.py's load_checkpoint & friends
    reliance.py                    # was reliance_analysis.py
    model.py                       # NEW (Phase 2): torch-native GPT
    tl_bridge.py                   # NEW (Phase 2): torch<->TL conversion + parity
    train.py                       # NEW (Phase 2): torch-native trainer (supersedes train_fast.py)
    tests/                         # test_injection.py, test_variants.py (+ Phase 2 tests)
  infra/                           # pod_setup.sh, launch_satellite.sh, fetch_fixed_complete.py
  legacy/                          # train_baseline.py, train_injected.py, train_fast.py (TL
                                   # trainer kept runnable for provenance; not the default path)
  experiments/
    01_complete_injection/         # original baseline/fixed/trainable campaign
      scripts/                     # eval_follower.py (shared runner lives here; others symlink docs)
      results/                     # (pointers only — this campaign's numbers live in 02/03)
    02_counterfactual_edits/
      scripts/                     # build_counterfactual_set.py, eval_counterfactual_set.py
      plots/                       # plot_counterfactual_set.py, plot_counterfactuals.py
      results/                     # counterfactual_results/* (incl. meta.json)
      figures/                     # cfset_*.png, counterfactuals_fixed_midtraining.png
    03_hunches/                    # hunches 1-3: convenience, channel-off, salience
      scripts/                     # cfa_convenience.py, cfa_channel_off.py, cfa_salience.py,
                                   # cfa_pull_follower_history.py
      plots/                       # plot_cfa_*.py
      results/                     # hunch_results/* (incl. logs)
      figures/                     # cfa_*.png
    04_probe_steering/             # hunches 4-5
      scripts/                     # steer_probes.py, steer_eval.py, steer_multi.py
      plots/                       # plot_steer_comparison.py
      results/                     # steer_* JSONs + sweep logs (from hunch_results)
      figures/                     # steer_vs_injection.png
    05_variants/                   # window/teacher/dummy campaign
      VARIANTS_SPEC.md
      REPORT_variants.md
      scripts/                     # train_teacher_probe.py, make_dummy_pretok.py, mktables.py
      plots/                       # plot_variants.py
      results/                     # variant_results/* EXCEPT unrelated_world.* (see 06)
      figures/                     # variants_*.png
    06_unrelated_world/
      scripts/                     # eval_unrelated_world.py
      results/                     # unrelated_world.*.json (from variant_results)
      figures/                     # unrelated_*.png
```

Rules:
- Import rewrites: scripts import shared code as `from maze.core import ...`
  (add `maze/__init__.py`, `maze/core/__init__.py`; scripts get a small
  sys.path shim or are run with `python -m`). Choose ONE consistent
  mechanism and document it in migration_changes.md.
- Where a file is claimed by two experiments (e.g. hunch_results holds both
  cfa_* and steer_* outputs), SPLIT by ownership as listed above.
- `maze_injection/` must END UP EMPTY and be removed. No file deleted unless
  its hash is verified present in the new tree.
- Do NOT touch: `maze-dataset/`, `maze-transformer/`,
  `structured-representations-maze-transformers/`, `knowledge/`,
  `counterfactual_set` artifacts inside variant_results/artifacts (moves with
  05). Clone removal in favor of pinned pip deps is the USER's call — record
  the recommendation, don't act on it.
- Fix intra-file relative path references in REPORT_variants.md and plot
  scripts' default paths so everything works from the new layout.
- Acceptance: full CPU test suite passes from the new layout; every script
  responds to `--help` (or clean import); manifest check passes.

## Phase 2 — torch-native training + TL bridge

- `core/model.py`: plain-torch decoder-only GPT reproducing the jirpy TL
  architecture EXACTLY. Do not assume the architecture — instantiate the TL
  model via ConfigHolder from a real config_holder.json and read
  cfg (normalization type, pos-embed type, MLP dims, gelu flavor, biases,
  tied/untied unembed, final LN). Attention via
  torch.nn.functional.scaled_dot_product_attention (is_causal=True) with an
  attention_mask path that matches TL's right-padded semantics.
- `core/tl_bridge.py`: BIDIRECTIONAL state-dict conversion
  (torch_to_tl / tl_to_torch), incl. the fused-QKV split/reshape into TL's
  (n_heads, d_model, d_head) layout. Loading helpers:
  `load_as_tl(path)` (torch ckpt -> HookedTransformer, NO weight processing:
  fold_ln=False etc.) and `load_zanj_as_torch(path)` (existing ZANJ ckpts ->
  fast model). Round-trip must be exact (same dtype, allclose atol 0).
- `core/train.py`: port of train_fast.py semantics to the torch model —
  same data pipeline (core/data.py, untouched math), same hyperparams,
  same wandb project/logging/checkpoint cadence, injector wandb.save
  (already added), all four inject modes (complete/window/teacher/dummy) via
  the SAME InjectionRuntime (features math shared; application = add to the
  layer-0 residual input). Teacher runs as a CONVERTED torch model (via
  tl_to_torch on the baseline ZANJ) for speed. Flags: --compile (default
  true, graceful fallback), --precision fp32-tf32|bf16 (default fp32-tf32
  for continuity). Checkpoint format: state_dict + config json + injector.pt
  (+ `to_zanj` export helper so ANY new model can be handed to the TL eval
  stack).
- Parity tests (core/tests/test_bridge.py), ALL must pass on CPU and (Phase
  2b) CUDA:
  1. fresh torch model -> TL: identical logits on real maze batches
     (max |delta| < 1e-4 fp32; report the actual number).
  2. trained baseline ZANJ -> torch: same threshold, real weights.
  3. round-trip torch->TL->torch: exact.
  4. injection parity: torch-side injection vs TL run_with_hooks injection,
     same W_inj/features: logits match.
  5. END-TO-END INTERP GUARANTEE: greedy rollouts of the converted baseline
     through the EXISTING TL eval stack (core/evals.py ProbeSet) vs the
     original ZANJ model: >= 99.9% token-identical, any mismatch shown to be
     an argmax near-tie (report top-2 logit gap at divergence).
- Phase 2b (GPU, one A100 pod, terminate after): run the parity suite on
  CUDA, then benchmark steps/sec: TL-eager vs torch-SDPA vs torch-SDPA+compile,
  at (bs256, seq~165 maze regime) and (bs64, seq1500 synthetic ICL regime),
  fp32-TF32 and bf16. Also max feasible batch size per stack. Record to
  `maze/core/BENCH.md`. Pod gotchas: gpuCount:1, env PUBLIC_KEY, no volume
  (Stop wipes), $1.49/hr, terminate when done.

## Phase 3 — adversarial audit (separate agent, fresh eyes)

Red-team the bridge and the migration: try to construct inputs where
torch/TL disagree (padding edge cases, attention-mask rows fully padded,
seq-len 1 prompts, injection at boundary positions end_pos=1/max, window at
sequence edge, bf16 vs fp32 drift); verify manifest integrity independently;
run the full CPU test suite cold from a fresh shell; grep for any remaining
`maze_injection` path references. File everything found; fixes go back
through the implementing agent.

## Known gotchas (do not rediscover)

- TL to_tokens PREPENDS BOS; pretok pipeline already accounts for it.
- import maze_transformer.training.config BEFORE ZANJ().read().
- fire parses "3,4" as a tuple.
- Apple openrsync chokes on multi---exclude; zsh doesn't word-split unquoted
  vars.
- TL weight layouts: W_Q/K/V (n_heads, d_model, d_head), W_O (n_heads,
  d_head, d_model); GPT-2 Conv1D-style transposition traps.
- Features are +/-1 coded; zeroed-conn = "all walls", NOT channel-off.
- The 12 grid-boundary features are structurally always -1 (dead).
- create-pod: gpuCount:1 AND env PUBLIC_KEY or sshd never starts.
