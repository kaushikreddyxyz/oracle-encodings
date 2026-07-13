# attribution

## Purpose

Scores multi-billion-token corpora with the frozen gold concept probes
(`2_probes`/`3_validation`/`4_causal` output), turning per-token activations
into an int8 score store other pipelines can train against. Owns three
things: **probe selection** (which 54 of 64 concepts, and which 3 gemma
layers, get frozen as "gold" for corpus work), **corpus scoring** (two
independent stores, see below), and the **G1 sanity gates** that must pass
before anything downstream (oracle-encoder training) trusts the store.
`oracles/` (repo-root) consumes this module's output; it does not live here.

## Pipeline

```bash
# Phase 0 — probe selection (CPU, $0, no args; writes out/probe_set.json etc.)
python select_probes.py

# Throughput/parity gate: sdpa vs eager (decided: eager — see Design decisions)
python bench_gemma.py --shard <sid> --probes-dir out [--batch-sizes 16,32,64]

# Corpus scoring — corpus-scores(+overflow): ClimbMix (nvidia/ClimbMix) shards 320-362
python score_corpus.py --probe-set out --shards "320-362" --out <dir> \
    --attn eager --batch-size 32 --calib-tokens 10000000 \
    --quant-json <dir>/quant.json --model google/gemma-2-2b

# Corpus scoring — climbmix-scored(+overflow-1..7): nanochat's actual training
# corpus (karpathy/climbmix-400b-shuffle), shards 0-184, FULL COVERAGE (env-driven)
ONLY_SHARDS=<csv or unset-for-all> SCORE_WORKDIR=/workspace/scores python score_climbmix_stacked.py
# fleet launcher (laptop-side):
python launch_attrib_wave.py --seed-repos
python launch_attrib_wave.py --shards 0-61 --pod-prefix attrib-w1 --n-pods 7 --launch first
python launch_attrib_wave.py --status

# G1 gate — corpus-side score sanity (must pass before any oracle-encoder training)
python g1_natural_ref.py          # part 1, local: natural-pool reference quantiles
python g1_corpus_check.py         # part 2, pod-side: corpus-side stats + spot checks

# Closed-form sanity (Exp-B ablation-repair target; superseded design, kept for provenance)
python verify_closed_form.py --probe-set out --scores <dir> --shard <sid> \
    --attn eager --out <dir>/verify_report.json

# Consolidate the joint per-shard store into the canonical corpus-scores layout (env-driven)
python stack_corpus_scores.py

# DoM steering scores at the ablation layer (env-driven; -> corpus-scores-dom-layer8)
python dom_complete.py
```

All scoring scripts (`score_corpus.py`, `score_climbmix_stacked.py`) run on
RunPod H100 fleets; `select_probes.py`, `g1_natural_ref.py` are local/CPU/$0.
Tests: `test_align.py`, `test_score_corpus.py`, `test_verify_closed_form.py`
(plain-assertion / pytest smoke tests against tiny synthetic fixtures, no
real gemma weights or network access — see `make_fixture.py`).

## Inputs & Outputs

- **Input**: `../concept_probes/3_validation/data/natscores/<family>.natscores.npz`
  + `../concept_probes/3_validation/artifacts/probe_cards.json` (deploy/caveat/reject
  verdicts), `../concept_probes/2_probes/probes/<family>/probes_l{L}.npz` (raw
  probe weights), `../concept_probes/4_causal/out/analysis/causal_cards.json`
  (`e5_salient_layer_corrected` drives the ablation-layer vote), `../concept_probes/4_causal/out/dose_calib.json`
  (independent cross-check only, not copied into outputs).
- **Local** (`out/`, mostly gitignored — see `out/.gitignore`): `probe_set.json`
  + `probe_set_arrays.npz` (Phase-0 output, **committed**), `selection_table.md`
  (full 64×3×4 audit table, committed), `G1_REPORT.md` (permutation-bug root
  cause), `g1_corpus_stats.json`, `g1_natural_ref.json`, `g1_residual_checks.json`,
  `quant.json` (per-column `zero`/`scale`, calibrated on 10M tokens, fleet-shared),
  `scoring_config.json`, `verify_report.json`, `coord_fidelity.json`,
  `gold_probes_per_layer/` (per-layer detection + DoM-steering npz, gitignored),
  `probe_set_dom_steering_l6_l8_l14.npz` (gitignored).
- **HF** (public datasets):
  - `corpus-scores` + `corpus-scores-overflow` — 43 shards (320–362, overflow =
    356–362), ~2.0B tokens. Per shard: `tokens_<sid>.npy` int32 BOS-free gemma
    token ids, `scores_<sid>.npy` int8 `[n, 3, 54]` (axis1: 0=L6, 1=L8, 2=L14),
    `docs_<sid>.jsonl` (`{"doc","start","n"}` spans). Written by
    `stack_corpus_scores.py`.
  - `climbmix-scored` + `climbmix-scored-overflow-1..7` — 185 shards (0–184),
    **9,873,968,012 tokens**, same per-shard format, but **full-coverage
    convention**: no 2048-token truncation, no min-length filter, consecutive
    non-overlapping 2048-token windows tile every document exactly (this is
    nanochat's actual training corpus, not the eval-purposed `corpus-scores`).
    Written by `score_climbmix_stacked.py`.
  - `corpus-scores-dom-layer8` — DoM steering scores `[n, 54]` at the ablation
    layer (L8), written by `dom_complete.py`.

## Design decisions that bind

1. **Probe selection is data-driven, not the SPEC's guess.** Chosen layers
   **[6, 8, 14]** (differs from the SPEC's speculative {8, 12, 16}); ablation
   layer **8** (mode of the causal-salient-layer vote, histogram
   `{6:1, 8:22, 10:4, 12:17, 14:4, 16:4, 18:2}`). **K = 54 of 64 concepts**
   survive (detection AUROC ≥ 0.90 at all 3 layers *simultaneously*) — dropped:
   all 5 intensity scalars, both `location_type` concepts, and 3 color_wheel
   blends (blue-violet, red-violet, yellow-orange — 3_validation "deploy" tier
   but fail this stricter joint bar).
2. **One layer per model (binding, corrected 2026-07-09).** Every model trained
   downstream on these scores (oracle encoder, injected nanochat run) consumes
   exactly one layer's 54 scores end to end — never the joint 3-layer/162
   target. Full derivation, incl. the invalidated joint design and what was
   deleted from HF: `../knowledge/concept_probes/reference/one_layer_per_model.md`.
3. **Quantization is int8, zero-preserving, ±4σ range**:
   `stored = clip(round((score − zero)/scale), -127, 127)`, `scale = 4σ/127`
   (resolution ≈ 0.03σ per integer step, *not* 1σ — a common misread); `zero`
   = per-column mean, saved in `quant.json`/`corpus_stats.json` and added back
   on decode, never assumed 0. Full derivation:
   `../knowledge/concept_probes/reference/quantization_int8.md`.
4. **Two-step standardization, never refit.** Step 1 (per-layer activation
   standardization, `nat_mean`/`nat_std`, part of each probe's frozen
   definition) happens before the probe; step 2 (per-probe score
   standardization over 10.4M ClimbMix tokens) happens after, and is what the
   score store and any downstream MSE loss actually train against. Equal
   z-score is *not* equal percentile (heavy-tailed, concept-dependent skew) —
   use `s95` when rank-comparability matters. Full derivation:
   `../knowledge/concept_probes/reference/normalization.md`.
5. **Tokenization/BOS convention**: `add_special_tokens=False`, BOS manually
   prepended then its hidden-state row dropped; docs truncated (or, for
   `climbmix-scored`, windowed) at `MAX_DOC_TOKENS=2048`; `MIN_DOC_TOKENS=64`
   floor for `corpus-scores`. Full derivation:
   `../knowledge/concept_probes/reference/tokenization_and_truncation.md`.
6. **The permutation lesson (fixed by metadata, not rescoring).**
   `select_probes.py`'s Phase-0 assembly filled the main `W`/`b` block by
   `(family, concept)`-sorted iteration while `probe_set.json["concepts"]` is
   name-sorted, mislabeling 53/54 concepts across 162/216 store columns (the
   DoM block was unaffected — it was always name-indexed). **No score bytes
   were ever wrong, only the name attached to them** — the fix added
   `main_block_concepts`/`dom_block_concepts` metadata keys; every consumer
   must resolve column identity through those keys, never assume `concepts`
   order for the main block. Root cause: `out/G1_REPORT.md`.
7. **sdpa silently drops gemma-2's logit softcapping** and fails probe-score
   parity vs eager (measured p99/std 0.057 padded / 0.077 packed, threshold
   0.05) — all corpus scoring is pinned to `--attn eager`.
8. **G1 gates are mandatory before any oracle-encoder training**: corpus-side
   score distributions vs natural-pool reference (quantile match), top-100
   firing-token spot check on 5 concepts (lexical concepts must fire on their
   surface forms), january-vs-march correlation sanity (within-family
   correlated but < 0.9; measured r=0.308, PASS). `g1_natural_ref.py` produces
   the reference side, `g1_corpus_check.py` the corpus side.

## Results

- **Probe selection (Phase 0)**: G0 = GO. K=54/64 survivors, layers [6,8,14],
  ablation layer 8 (see Design decisions #1).
- **corpus-scores**: 43 shards, ~2.0B tokens, ~$45, byte-verified.
- **climbmix-scored**: 185 shards, **9,873,968,012 tokens** verified complete
  (deep audit 2026-07-10/13 — byte-exact vs npy headers, bit-exact
  re-tokenization incl. across window boundaries, standardization sane,
  keyword-firing semantics checked; only `may` (modal homograph) and
  `last_quarter` (fiscal homograph) are keyword-non-responsive of the 54).
- **G1**: FAIL → root-caused → PASS. Label-permutation bug found (see Design
  decisions #6), fixed via metadata with **zero rescoring** — the live
  encoder-training run downstream stayed valid throughout (median R² is
  permutation-invariant).
- **int8 clipping**: non-blocking but real — 90% of the 216 columns exceed the
  0.1% clip-fraction "fine" threshold (worst 0.91%); `quant.json`'s
  `scale = 4σ/127` is systematically a bit tight, worth recalibrating on a
  larger sample if the store is ever regenerated from scratch.
- Full narrative + audit detail: **`REPORT.md`** (living log, not distilled
  here — read it directly for anything not covered above).

## Example

`examples/read_corpus_scores.py` — reads one shard of `corpus-scores` (or
`climbmix-scored`) via HTTP range requests (no full ~7.5 GB shard download by
default), dequantizes + standardizes one (concept, layer) column, and prints
the top-scoring tokens with decoded context.

## Gotchas

- `select_probes.py` takes **no CLI args** — all paths are relative to the
  file's own location. Rerunning it would NOT match the already-scored,
  immutable store (the store was fixed via metadata, not by rerunning
  selection).
- `split_score_store.py` and `zip_store.py` are **historical one-off
  migration scripts** that read from `concept-probes-corpus-scores`/`-2` and
  write to per-layer/stacked repo names superseded by the plain
  `corpus-scores`/`corpus-scores-overflow` layout (written by
  `stack_corpus_scores.py`, which every current consumer actually
  references). Kept for provenance only — do not launch against current data
  without re-verifying their target repos still make sense.
- gemma-2-2b is **gated** on HF — every pod needs `HF_TOKEN` before
  `from_pretrained`; tokens travel over stdin only, never argv.
- RunPod's API is behind Cloudflare, which 403s the default `Python-urllib`
  User-Agent — self-cleanup/teardown code must send `User-Agent: curl/8.4.0`.
- HF's commit-rate limit (128 commits/hour/repo) was hit once by per-file
  daemon commits — batch into single `upload_folder` commits instead.
- The env-driven scoring/consolidation scripts (`stack_corpus_scores.py`,
  `score_climbmix_stacked.py`, `dom_complete.py`, `zip_store.py`,
  `split_score_store.py`) take no CLI flags; behavior is controlled entirely
  by env vars (`ONLY_SHARDS`, `WORKDIR`/`SCORE_WORKDIR`, `SHARD_MAP`,
  `HOLD_PATH`, `LOG_PATH`, `CAP_BYTES`, `MIN_FREE_GB`, ...) — read each
  script's header before launching.
