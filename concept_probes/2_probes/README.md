# 2_probes (was `stage5`)

## Purpose

Trains linear concept-salience probes on gemma-2-2b residual streams: one gemma
forward pass per family's unique examples, cached, then per-layer probe fitting
(ridge primary + baselines) and generated-split evaluation — 64 concepts x 12
layers = 768 probes.

## Pipeline

Working directory: `2_probes/code/`. Pod-orchestrated via
`pod_run.sh <family> [<family> ...]` (env: `LAYERS`, `ROOT` default
`/workspace/stage5`, `NATSTATS`, `UPLOAD=1`). Manual steps:

```
1. python natstats.py --passages <standardization_sample.jsonl> --layers 0..25 \
       --out natstats.npz [--max-tokens 3000000]     # once, from the natural sample
2. python extract.py --family <fam> --stage4 <1_dataset/data> --out <cache/fam> \
       [--layers 1,3,6,8,10,12,14,16,18,20,23,25] [--natural <passages.jsonl>]
3. python train.py --family <fam> --stage4 <1_dataset/data> --cache <cache/fam> \
       --natstats natstats.npz --layers <csv> --out <out/fam>
4. python evaluate.py --family <fam> --cache <cache/fam> --probes <out/fam> \
       --stage4 <1_dataset/data> --natstats natstats.npz --layers <csv> \
       --out metrics/<fam>.json
```

Pilot-only 26-layer sweep + read-shift(+1) ablation: `pilot_run.sh` (ran on
`january`/`harmfulness`/`europe` before the fleet launch). Pod bring-up:
`pod_setup.sh` (installs deps, asserts CUDA, requires `HF_TOKEN`).

Note: `pod_run.sh` also calls `score_natural.py`, which physically lives in
`3_validation/code/`, not here — a pod checked out to `2_probes/code/` alone will
fail at that step unless the sibling stage's script is present too.

## Inputs & Outputs

- Input: `1_dataset/data/<family>/final/mixed/{cls}.{train,val,form_holdout}.jsonl`
  (or HF `probe-train-data`, renamed from `concept-probes-stage4-data`).
- Local outputs (gitignored): `2_probes/probes/<family>/probes_l{L}.npz`,
  `trainstats_l{L}.json`.
- Committed locally: `2_probes/metrics/<family>.json` (evaluate.py output).
- HF mirror ([`concept-probes-gemma2-2b`](https://huggingface.co/kaushikreddyxyz/concept-probes-gemma2-2b),
  via `pod_run.sh UPLOAD=1`): probes -> `families/<fam>`, metrics ->
  `metrics/<fam>.generated.json`, natural scores -> `natscores/<fam>.natscores.npz`.

## Design decisions that bind

- **`hidden_states[L+1]` = post-block-L residual stream** (`extract.py`: "BOS
  prepended, dropped from the cache" + `# hidden_states[l+1] == post-block-l
  residual; drop the BOS row"). This is the activation-site convention every
  downstream stage inherits — never re-derive it differently.
- **Ridge (closed-form) is the PRIMARY method, not Adam.** The masked-MSE+L2
  objective is convex; Adam systematically undershoots it (small minibatches —
  16,384 tokens, not 131k — were needed just to give it enough steps/epoch).
  Adam (3 seeds) is retained only as a cross-seed-stability diagnostic.
- Per-layer standardization (`natstats.py`): mean/std computed once from a
  natural passage sample, applied as `(h - mu)/sd` before every probe
  fit/projection — shared across concepts within a layer, never across layers.
- Buffer mask (width 10): tokens after a positive-strength span get mask=0 only
  where `y==0`, so a concept's tail doesn't get trained as a hard negative.
- lambda in {1e-4, 1e-3, 1e-2}, seeds {0,1,2}; lambda chosen per-class by masked
  val MSE.
- Homograph FPR is thresholded at the confirmed-positive 25th percentile, not
  the neutral floor — judge-truth deliberately gives wrong-sense hard negatives
  a faint-echo label (~1/6) that legitimately scores above neutral.
- Hewitt-Liang control labels are drawn from the class's own empirical label
  marginal (not uniform), so the control task has comparable tie structure to
  the real task.

## Results

768 probes trained (64 concepts x 12 layers) + a glorptitude control, mirrored
to HF. GPU spend ~$30 total; natural judging $14.82 for 23,776 examples across
13 families in ~19 min wall.

**Why ridge is primary**: pilot optimizer-gap evidence — Adam's ρ was ~0.24
lower than ridge on the same objective (undershooting due to too few gradient
steps per epoch at the original batch size). Example (`metrics/months.json`,
january, layer 8): ridge ρ=0.378 / AUROC=0.967 vs Adam (seed-mean) ρ=0.129 /
AUROC=0.688; DoM/LDA/logistic sit close to ridge (0.359-0.369 ρ) — Adam is the
outlier.

**Tie-ceiling lesson**: at natural token prevalence (0.1-2% nonzero targets), a
*perfect* continuous scorer's token-level Spearman ρ is bounded at 0.05-0.13 by
the zero-tie block — raw ρ against an absolute threshold is meaningless. Gate on
`ρ_rel = ρ/ρ_ceiling` instead (finalized in 3_validation; moon_phases probes sit
at 0.77-0.98 of ceiling despite low raw ρ).

All three pilot concepts (january, harmfulness, europe) peaked in layers 6-20,
confirming the mid-band layer selection.

## Gotchas

- `config/stage5.yaml` keeps the old stage name in its filename intentionally —
  do not rename it.
- Natural scoring/gating numbers (44/15/5 deploy/caveat/reject) are a
  3_validation result, not computed here — `evaluate.py`'s generated-split
  metrics are ceiling-underestimated for real probes and diagnostic only.
