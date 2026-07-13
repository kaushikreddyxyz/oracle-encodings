# 3_validation (was `stage6`)

## Purpose

Certifies the 2_probes probes against held-out **natural** ClimbMix text: mines
natural candidates, judges them with the unchanged 1_dataset judge, scores the
trained probes on natural text, and runs a Tier-1 gate battery to produce a
deploy/caveat/reject verdict per concept.

## Pipeline

Working directory: `3_validation/code/` (scripts resolve `1_dataset` as
`STAGE6.parent / "1_dataset"` — already fixed for the rename).

```
1. python natural_split.py                    # no CLI args; writes standardization_sample.jsonl
                                               # (shard 310) + random_pool.jsonl (shards 311-312)
2. python mine_natural.py                     # no CLI args; lexically mines shards 311-316 into
                                               # data/natural/mined/<family>.jsonl (8 non-intensity families)
3. python prep_judge_natural.py --families <csv> [--random-per-family 1200] \
       [--random-per-intensity 1500]          # stages generations_nat.jsonl into 1_dataset/data
4. (cd ../../1_dataset/code && python judge.py --family <f> --tag nat --cap-usd 5.0)
                                               # reuses 1_dataset/code/judge.py unchanged; driven by
                                               # judge_nat_lane.sh over all 13 families
5. python score_natural.py --family <f> --eval <natural_eval.jsonl> --probes <2_probes/probes/fam> \
       --natstats <natstats.npz> --cache <cache/fam> --out <natscores/fam.natscores.npz> \
       [--layers 1,3,6,8,10,12,14,16,18,20,23,25] [--model google/gemma-2-2b]
6. python gates.py --families <csv> --metrics-dir <2_probes/metrics> \
       --natscores-dir data/natscores --out reports [--layers <csv>]
7. python assemble_W.py --gates-dir reports --probes-root <2_probes/probes> \
       --out artifacts/stacked --families <csv> [--layers <csv>]
```

## Inputs & Outputs

- Input: 2_probes probes + metrics; 1_dataset's `judge.py` (unchanged, called with
  `--tag nat`).
- Local: `data/natural/{mined,standardization_sample.jsonl,random_pool.jsonl}`
  (the prose `MINING_REPORT.md` was removed — git history),
  `data/natscores/<family>.natscores.npz`, `artifacts/probes/`, `artifacts/stacked/W_l{L}.npz`,
  `reports/*.gates.json`, `reports/rollup.{json,html}` (`rollup.md` removed — use
  `rollup.json` / git history), `reports/concepts/*.png`.
- HF (public): [`concept-probes-gemma2-2b`](https://huggingface.co/kaushikreddyxyz/concept-probes-gemma2-2b)
  — per-family raw training outputs, stacked deployment matrices (`W_l{L}.npz`),
  per-concept probe files, probe cards, metrics, natural scores, reports.
- `hf_model_card.md` (kept) — source of the HF model-card front matter for that repo.

## Design decisions that bind

- **Ceiling-normalized natural ρ** (`ρ_rel = ρ/ρ_ceiling`): at 0.1-2% token
  prevalence a *perfect* scorer's raw Spearman is bounded at 0.05-0.13 by the
  zero-tie block, so gates use ρ_rel, not raw ρ, against the spec's thresholds.
- **Natural-only certification** (the glorptitude trap): generated-split metrics
  are diagnostic only and cannot certify a probe. `glorptitude` — a fabricated
  concept with no semantics — trains to AUROC 0.90-0.93 with implicit recall 1.0
  on its own generated validation set (the pipeline manufactures a coherent
  "discussing-an-ineffable-quality" genre direction). Certification rests
  entirely on judge-labeled natural-text gates, which a manufactured genre
  direction cannot pass. This is the sharpest argument for the spec's natural-
  only-validation principle, not just a restatement of it.
- Five Tier-1 gates per concept at its chosen layer: ceiling-normalized natural
  ρ, selectivity gap (lexical-holdout G, implicit recall, homograph FPR,
  Hewitt-Liang sanity), ECE after isotonic calibration (cal-fit, test-eval),
  per-domain minimum ρ (collapses to web only), margin over the 95th percentile
  of 20 random directions.
- Homograph FPR thresholded at the confirmed-positive 25th percentile (not the
  neutral floor) — carried over from 2_probes.
- Frozen cal/test halves of the natural pool; candidate selection and
  calibration happen on CAL only, reported gates on TEST only.

## Results

**768 probes trained; final verdicts: 44 deploy / 15 caveat / 5 reject** (of 64
concepts). Median ceiling-normalized natural-text ρ = 0.865 (min 0.448, max
0.989).

| family | deploy/caveat/reject | median ρ_rel |
|---|---|---|
| weekdays | 5/1/1 | 0.95 |
| moon_phases | 8/0/0 | 0.91 |
| color_wheel | 9/2/1 | 0.89 |
| continents | 4/2/0 | 0.89 |
| months | 10/1/1 | 0.84 |
| seasons | 1/3/0 | 0.84 |
| directions | 6/1/1 | 0.82 |
| location_type | 1/1/0 | 0.66 |
| costliness | 0/1/0 | 0.64 |
| duration | 0/1/0 | 0.64 |
| harmfulness | 0/1/0 | 0.53 |
| physical_size | 0/1/0 | 0.56 |
| lovingness | 0/0/1 | 0.45 |

Full 64-row table: `reports/rollup.json` (`rollup.md` removed — git history); per-concept detail in
`artifacts/probe_cards.json`. The five rejects: violet (homograph FPR 0.25),
saturday (FPR 0.26 + margin), lovingness (ρ 0.45), december and southwest
(Hewitt-Liang sanity < 0). Lexically-anchored concepts (weekdays, months, moon
phases, colors) transfer near-perfectly (detection AUROC up to 0.998); graded
intensity scalars (lovingness, harmfulness, costliness, duration, physical_size)
are the weak tier, losing real signal on natural text despite ρ≈0.65 on
generated val — a genuine generated->natural covariate shift, not a pipeline bug.

Cost: ~$30 GPU (4-pod H100 fleet) + $14.82 natural judging + $0.66 glorptitude
≈ $15.5 API this stage.

## Gotchas

- Known limitations carried into deployment: λ selection saturated at the 1e-2
  grid edge for all classes (sensitivity is tiny, not re-run); causal ablation
  (Tier-2, "report never gate" — see 4_causal) was not run as part of this
  stage; natural positives are lexically mined, so natural metrics may overstate
  implicit-tail performance; single judge family (mercury-2) is a standing
  monoculture risk, monitored via the generated<->natural separability AUC
  (0.95-0.99).
- ~26 of 23,776 natural examples were aggregated from 2-of-3 judge votes
  (rate-limit era); 4 rare moon-phase classes have <60 natural positives.
- Chosen layers spread 3-25 (mass at 3-18); for lexical concepts the token-ρ
  layer profile is nearly flat, so layer choice there is partly CAL-half
  selection noise — harmless for deployment, but the selectivity-relevant band
  is 8-16.
- Full deployment tuning (operating thresholds at the rare true prior) is done
  by downstream consumers (`../4_causal/`, `../../attribution/`), not here.
- `artifacts/probe_cards.json` is the handoff artifact consumed downstream by
  `../../attribution/select_probes.py` (gold-54 selection) and by 4_causal's
  causal-validation fleet.
