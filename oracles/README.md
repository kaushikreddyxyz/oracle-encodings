# oracles

## Purpose

Trains Qwen3-0.6B-Base + MLP-head "oracle" encoders to predict per-token
concept-probe scores from raw text alone, using `attribution/`'s int8 score
store as training data. The resulting encoder is what gets run over
nanochat's pretraining corpus to produce the injected signal (see Injection
handoff below) — this module owns encoder training and evaluation only, not
the injection itself.

**BINDING: one layer per model.** Probe scores for gemma layers 6/8/14 are
three independent artifacts; every trained encoder consumes exactly ONE
layer's 54 scores end to end. An earlier design trained one encoder jointly
on all 162 targets (3 layers × 54) — this was a mistake, corrected
2026-07-09, and its checkpoints + derived coord stores were **deleted from
HF**. `train_oracle_perlayer.py` (`--layer {6,8,14}`, required) is the
current, correct trainer. `train_encoder.py` (the joint 162-target trainer)
is **superseded** — its code is kept for provenance/history only; do not
train new checkpoints with it. Full derivation:
`../knowledge/concept_probes/reference/one_layer_per_model.md` (knowledge/ is
gitignored — local-only tree; if absent see git history / the READMEs here).

## Pipeline

```bash
# Pod-side: stage the score-store shards + raw ClimbMix parquets a training run needs
python prefetch_shards.py --dest <dir> --climbmix-dest <dir> --shards <csv, priority order>

# Per-layer oracle trainer (one independent run per layer)
python train_oracle_perlayer.py --layer 6 --scores <corpus-scores dir> \
    --climbmix-dir <climbmix parquet dir> --train-shards "320-352" --val-shards "353,354" \
    --muon-lr 5e-3 --adamw-lr 1e-4 --max-hours 11 --plateau-delta 0.005 \
    --plateau-tokens 150e6 --hf-repo kaushikreddyxyz/oracle-encoders \
    --wandb-project stage7-oracle
# (repeat per --layer 8 / --layer 14, each its own run)

# Continuation runs (L6/L8 only — L14 deliberately absent, see Gotchas)
python cont_launch.py --create --deploy all         # provision pods, warm-start from best_stripped.pt
python cont_launch.py --status
python cont_launch.py --terminate L6                # manual override if needed
# cont_supervise.sh (runs on-pod under nohup) sequences: prefetch -> trainer -> cont_teardown.py
python cont_teardown.py                              # final upload + verify + self-terminate

# Gate G2 — natural-eval AUROC retention (encoder must retain >=90% of the gemma probes' AUROC)
python g2_retention.py --encoder-ckpt <best.pt> --probe-set <attribution/out dir> \
    --eval-data <3_validation natural-eval texts dir> --out g2_retention.json

# Exp B final analysis (superseded 162-joint design; kept for historical G3 record)
python expB_final_analysis.py

# Association figures (oracle prediction x gemma probe activation, 54x54 Spearman)
python fig_oracle_perlayer_assoc.py --probe-set <dir> --eval-data <dir> --out <path>
python fig_oracle_perlayer_assoc_cont.py --ckpt-l6 <cont1/best.pt> --ckpt-l8 <cont1/best.pt> \
    --orig-npz <original assoc npz> --probe-set <dir> --eval-data <dir> --out <path>
python fig_assoc_residual.py --goldcorr <path> ...
python fig_oracle_probe_assoc.py --probe-set <dir> --eval-data <dir> --out <path>

# Replay a finished run's metrics.jsonl into wandb (e.g. after a retro-log)
python wandb_retrolog.py --metrics <run>/metrics.jsonl --name <run-name> --project stage7-oracle
```

Smoke tests (no GPU/real weights needed): `test_train_encoder.py`,
`test_g2_retention.py`.

## Inputs & Outputs

- **Input**: `../attribution/`'s stacked score store (`corpus-scores`/
  `corpus-scores-overflow`, staged locally by `prefetch_shards.py`), raw
  ClimbMix text (`karpathy/climbmix-400b-shuffle` parquets, re-read at train
  time to recover text for tokenizer alignment — scores alone aren't enough),
  `../concept_probes/3_validation/` natural-eval texts (for G2).
- **Local** (`out/`, mostly gitignored — bulk eval + checkpoints live on HF,
  see `out/.gitignore`): `g2_retention.json`, `expB_final_analysis.json`
  (both committed); gitignored: `expA_frozen_metrics.jsonl` (HF copy under
  `stage7_eval/`), `retro_metrics/`, `figures/`.
- **HF**: `oracle-encoders` (model repo) — three independent per-layer
  checkpoints, `layer06/`, `layer08/`, `layer14/`, each `best_stripped.pt`
  (deployed weights only) + `metrics.jsonl`; `layer06/cont1/` and
  `layer08/cont1/` hold the continuation checkpoints (`best_full.pt` carries
  full Muon+AdamW optimizer state for exact resume; originals are never
  overwritten). Also `stage7_eval/` — where this module's bulkier `out/`
  binaries actually live. The legacy joint-162 `oracle-encoder` (singular)
  repo and its derived `oracle-coords`/`oracle-coords-b` were **deleted**
  2026-07-09 — do not reference them.

## Design decisions that bind

1. **One layer per model** — see Purpose above; the physical enforcement is
   that `train_oracle_perlayer.py` slices the stacked `[n, 3, 54]` store to
   `[n, 54]` at read time, so nothing downstream of the slice ever sees
   another layer's columns.
2. **Objective**: MSE (expA design) on corpus-standardized, dequantized
   scores — `target = (int8 * scale[l] + zero[l] - mean[l]) / std[l]`,
   per-column (quantization/normalization conventions are `attribution/`'s;
   see its README rather than re-deriving here).
3. **Optimizer**: Muon (Newton-Schulz orthogonalized momentum) for all 2D
   non-embedding weight matrices; AdamW for embeddings, gains, biases.
   Cosine schedule with warmup on both. (The original 162-joint
   `train_encoder.py` used plain AdamW throughout — Muon is specific to
   `train_oracle_perlayer.py`.)
4. **gemma→qwen tokenizer alignment hard-assert**: text is tokenized with
   both the gemma and Qwen3 tokenizers; a gemma token maps to the last qwen
   token whose char span ends at or before it (`align.py`/`_align_fallback.py`
   in `attribution/`). The gemma-token-id reproduction is hard-asserted
   against the first `--assert-first-n-docs` docs at trainer startup — it
   fails loudly on tokenizer drift rather than silently training on
   misaligned targets.
5. **Token-based plateau stopping** (not just wall-clock): stop when the
   median heldout R² improves by less than `--plateau-delta` (default 0.005)
   over the trailing `--plateau-tokens` (default 150e6) of training tokens,
   or `--max-tokens`/`--max-hours`, whichever comes first. This is the rule
   that ended both continuation runs early (see Results).
6. **`--resume` does NOT restore the LR scheduler.** The L6/L8 continuation
   runs handled this manually: a 150-step re-warmup to the original cosine's
   value at the resume step, then cosine to the 0.1 floor at the projected
   epoch end (`--cont-anchor-mult`/`--cont-end-step`/`--cont-rewarmup-steps`
   flags in `train_oracle_perlayer.py`).

## Results

**Per-layer oracles** (Qwen3-0.6B-Base full fine-tune, MLP head
1024→4096→GELU→54, train shards 320–352, val 353/354, Muon+AdamW):

| layer | heldout median R² | Spearman ρ | tokens | continuation (cont1) |
|---|---|---|---|---|
| L6 | 0.8331 | 0.898 | 690M | **0.8368** / ρ 0.900 @ 854M tokens (plateau Δ=0.0027) |
| L8 | 0.7965 | 0.874 | 701M | **0.8002** / ρ 0.877 @ 865M tokens (plateau Δ=0.0033) |
| L14 | 0.7217 | 0.829 | 614M | not continued (deliberately — see Gotchas) |

All three beat the superseded joint-162 baseline (heldout median R² 0.6371;
frozen-encoder MLP-only control 0.1823 — a 3.5× gap showing the signal is
learned into the encoder, not read out of pre-existing Qwen features). Both
continuations flattened within ~150M tokens of their plateau rule
(Δ<0.005/150M tok) — R²≈0.84/0.80 looks like the practical ceiling for this
recipe, not an undertrained model (~$17 actual spend vs a ~$120 projected
full-epoch budget).

**Association figures** (`out/figures/`, 54×54 Spearman between oracle
max-pooled prediction and gemma-probe activation, per layer, on the
3_validation natural-eval TEST split):

| panel | median diag | min diag | median off-diag | p95 off-diag |
|---|---|---|---|---|
| L6 | 0.902 | 0.824 | 0.252 | 0.601 |
| L8 | 0.888 | 0.789 | 0.208 | 0.552 |
| L14 | 0.877 | 0.772 | 0.239 | 0.612 |

Strong on-target diagonal with expected family-block off-diagonal structure;
no permuted or broken axes.

**Gates** (joint-162 design; see `REPORT.md` for the full G0–G4 table):
G2 (heldout median R² ≥ 0.6, natural-eval AUROC retention ≥ 0.90) passed at
R² 0.6371 / retention 0.966. G3 (Exp B structured v* head, superseded
alongside the joint design) passed in its learned-decoder arm (v* R² 0.6111,
subspace-recovery median cosine 0.998 vs a 0.125 random control) and failed
in its fixed-decoder arm (v* R² 0.2716). No separate per-layer G2/G3 rerun is
recorded — read `REPORT.md` directly for the current status of anything not
covered above, especially the injected nanochat run's G4.

Living log: **`REPORT.md`** (kept, not distilled here). wandb project:
**`stage7-oracle`** (entity `kaushikreddyxyz-`).

## Injection handoff

The oracle-encoder checkpoint is not injected here. The injection package —
site math (gate/activation/direction), training flags, the precompute
pipeline, and all current design/operational detail — lives in the `nanochat`
submodule at `nanochat/nanochat/injection/` (its `README.md` is authoritative;
precompute entry point is `nanochat/scripts/precompute_activations.py`, which
reads this module's encoder checkpoint and `attribution/out/probe_set.json`).
The launch checklist formerly at `concept_probes/5_oracle/out/nanochat_prep.md`
is superseded by that README. One item from that checklist is still open and
not resolved by the current README: **whether to add a coords-on eval pass**
alongside the coords-off default (`base_eval`/CORE/sampling call the model
with `acts=None` by design, so the standalone-LM number is always reported;
whether a coords-on pass should also be reported is called out in the
nanochat README itself as "a separate, deliberate decision," not yet made).

## Gotchas

- `train_encoder.py` (the joint 162-target Exp A/B trainer) is **superseded**
  — its HF checkpoints were deleted 2026-07-09. It's kept in this directory
  for historical reference (e.g. `expB_final_analysis.py` still reads its
  output format for the G3 write-up) but must not be used to train new
  deployment checkpoints.
- L14 has **no continuation run** — the 2026-07-10 handoff deliberately
  continued only L6/L8; `cont_launch.py`'s `--deploy` choices are
  `{"L6", "L8", "all"}` (no L14).
- `--resume` does not restore the LR scheduler (see Design decisions #6) —
  don't assume a bare `--resume` reproduces the original schedule.
- The historical pod self-termination daemons (`cleanup/*_selfcleanup.py`,
  from the original overnight coords-precompute + Exp-B trainer pods) were
  removed 2026-07-20; recover from git history if a similar pattern is needed.
  Ops fact worth keeping: RunPod's GraphQL API 403s the default
  `Python-urllib` User-Agent — send `User-Agent: curl/8.4.0`.
- nanochat's `scripts/base_train.py` hardcodes `wandb.init(project="nanochat")`
  — any injected run must redirect this to `stage7-oracle` (e.g. `sed`) before
  launch, or it logs to the wrong project. This is a nanochat-submodule
  detail, not an oracles/ one, but it trips up anyone launching from here.
