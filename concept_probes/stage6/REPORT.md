# Stages 5–6 report: concept-salience probes for gemma-2-2b

2026-07-02. Spec: `knowledge/concept_probes/task.md`. Data: Stage 4 (172k judged
generated examples). This report covers probe training (Stage 5) and validation
(Stage 6) end-to-end.

## Headline

**768 probes trained (64 concepts × 12 layers); final verdicts: 44 deploy /
15 caveat / 5 reject.** Median ceiling-normalized natural-text ρ = 0.865
(min 0.448, max 0.989). Lexically anchored concepts read out near the
information-theoretic ceiling of the metric; graded intensity scalars are the
weak tier, failing on genuine semantic difficulty rather than pipeline error.

| family | deploy/caveat/reject | median ρ_rel | median margin |
|---|---|---|---|
| weekdays | 5/1/1 | 0.95 | 0.26 |
| moon_phases | 8/0/0 | 0.91 | 0.38 |
| color_wheel | 9/2/1 | 0.89 | 0.44 |
| continents | 4/2/0 | 0.89 | 0.55 |
| months | 10/1/1 | 0.84 | 0.48 |
| seasons | 1/3/0 | 0.84 | 0.50 |
| directions | 6/1/1 | 0.82 | 0.30 |
| location_type | 1/1/0 | 0.66 | 0.49 |
| costliness | 0/1/0 | 0.64 | 0.48 |
| duration | 0/1/0 | 0.64 | 0.53 |
| harmfulness | 0/1/0 | 0.53 | 0.41 |
| physical_size | 0/1/0 | 0.56 | 0.41 |
| lovingness | 0/0/1 | 0.45 | 0.31 |

Full 64-row table: `reports/rollup.md` (+ color-coded `rollup.html`); one
four-figure page per concept in `reports/concepts/`; machine-readable verdicts
in `reports/*.gates.json` and `artifacts/probe_cards.json`.

## What was built

- **Primary probes**: closed-form ridge (exact minimizer of the §5.2
  buffer-masked MSE + L2 objective), one independent row per (concept, layer),
  λ ∈ {1e-4,1e-3,1e-2} chosen per class on generated val, inputs standardized
  with natural-corpus statistics (1.65M ClimbMix tokens). Unit-normalized rows
  stacked per chosen layer into `artifacts/stacked/W_l{L}.npz` (`ŷ = W·(h−μ)/σ + b`),
  with raw-space parameterizations alongside.
- **Baselines** (all beaten or matched by ridge at every chosen layer): DoM,
  shrinkage-LDA, logistic, 20 random directions. **Adam** (3 seeds) retained as
  a stability diagnostic only — see deviations.
- **Ensembles (§5.3)**: fit for all classes, adopted for none (max val gain
  +0.022 < 0.03; per-layer residual correlations 0.78–0.90).
- **Natural validation set**: 23,776 ClimbMix windows (shards 311–316, disjoint
  from all prior training), lexically mined per family + shared random pools,
  judge-labeled with the unchanged Stage-4 pipeline (mercury-2, K=3), $14.82.
  Frozen cal/test halves; candidate selection and calibration on CAL only,
  reported gates on TEST only.

## Validation protocol (Tier 1, per §6)

Five gates per concept at its chosen layer: ceiling-normalized natural ρ,
selectivity gap (lexical-holdout G, implicit recall, homograph FPR,
Hewitt–Liang sanity), ECE after isotonic calibration (cal-fit, test-eval),
per-domain minimum ρ (collapses to web — the deployment corpus is web-only by
Stage-4 decision), and margin over the 95th percentile of 20 random directions.

### The five spec deviations (all evidence-driven, PLAN.md §deviations)

1. **Ridge replaces Adam as the primary optimizer.** Same objective, exact
   solution. Adam undershot by Δρ 0.15–0.25 across three pilot rounds and
   remains near-chance for some classes even at 250 cosine-annealed epochs.
2. **Homograph FPR thresholded at the confirmed-positive 25th percentile** —
   judge-truth deliberately gives wrong-sense negatives faint 1/6 labels (the
   2B rule), so scoring them just above the neutral floor is correct behavior.
3. **Hewitt–Liang demoted to a sanity check (S > 0)** — token identity is
   linearly decodable from every residual layer, so the control is far from
   chance (ρ 0.26–0.34) for any zero-inflated regression task.
4. **Ceiling-normalized natural ρ** — at 0.1–2% token prevalence, a PERFECT
   continuous scorer's raw Spearman is bounded at 0.05–0.13 by the zero-tie
   block. Gates use ρ_rel = ρ/ρ_ceiling (reduces to the spec rule when ties
   are negligible). Raw ρ, ceiling, and prevalence are all reported.
5. **Vacuous selectivity checks skip rather than fail** — e.g. "full moon" has
   zero concept-absent hard negatives by judge-truth (idioms still evoke the
   phase); untestable checks are recorded on the probe card.

## The nonsense-control finding (§6.4) — most important caveat

The vacuous concept **glorptitude**, pushed through the identical
generate→judge→train pipeline, trains to **AUROC 0.90–0.93 with implicit
recall 1.0 on its own generated validation set** — far outside the spec's
[0.45, 0.55] window. The pipeline manufactures a coherent, linearly decodable
"discussing-an-ineffable-quality" genre direction for a concept with no
semantics. Consequence adopted throughout: **generated-split metrics are
diagnostic only and cannot certify any probe**; certification rests entirely on
the judge-labeled natural-text gates, which a manufactured genre direction
cannot pass (generated↔natural activations are separable at AUC 0.95–0.99,
yet real probes transfer and hit 0.8–0.99 of ceiling). This sharpens, rather
than contradicts, the spec's own natural-only-validation principle.

## Reading the results

- **Lexical vs semantic split.** Concepts with a discrete surface anchor
  (weekdays, months, moon phases, colors) transfer near-perfectly (example-level
  detection AUROC up to 0.998 — october 0.997, wednesday 0.991; moon-phase
  probes separate fiscal "first quarter" from the lunar sense). Graded
  judgment scalars lose real signal on natural text (lovingness 0.45,
  harmfulness 0.53 of ceiling) despite ρ≈0.65 on generated val — the measured
  generated→natural covariate shift is the best explanation.
- **Concept-vs-token evidence.** Implicit recall peaks at layers 8–16
  (0.47–1.0) vs ~0.1 at layers 0–3 — exactly the §6.3 signature; the mid-band
  {6..20} was confirmed by all three pilot concepts' 26-layer sweeps
  (harmfulness peak L11 raw ρ 0.657; europe L17; january mid-band).
- **Failure modes are the predicted ones**: homograph FPR caveats land on
  saturday/july/autumn/summer/winter/violet/north america/oceania; the five
  rejects are violet (homograph FPR 0.25), saturday (FPR 0.26 + margin),
  lovingness (ρ 0.45), december and southwest (Hewitt–Liang sanity < 0).
- **Chosen layers** spread 3–25 with mass at 3–18. For lexical concepts the
  token-ρ layer profile is nearly flat, so layer choice there is partly
  selection noise on the CAL half — harmless for deployment (any mid layer is
  equivalent) but worth knowing; the selectivity-relevant band is 8–16.

## Known limitations (carry into any deployment)

1. λ selection saturated at the 1e-2 grid edge for 65/65 classes. Measured
   sensitivity is tiny (val-MSE flat within 0.3%/decade; direction rotation
   ≤9°), so we did not re-run, but an extended-grid refit is a cheap follow-up.
2. §6.7 causal ablation (Tier-2, "report never gate") was **not run** — ~1
   pod-hour if wanted.
3. The natural eval's positives are lexically mined, so natural metrics may
   overstate implicit-tail performance; implicit recall is measured on
   generated data only.
4. ~26 of 23,776 natural examples aggregated from 2-of-3 judge votes
   (429-era); 4 rare moon-phase classes have <60 natural positives.
5. Single judge family (mercury-2) — the standing monoculture risk from
   Stage 4 carries through; the covariate-shift AUC (0.95–0.99) is the live
   monitor per §6.6/§8.
6. Calibration (isotonic on natural CAL) was fit for the ECE gate; full §7
   deployment tuning (operating thresholds at the rare prior, PR at π_dep) is
   Stage-7 work, not done here.

## Costs & infrastructure

- GPU: ~$30 (4-pod H100 fleet, $3–5.5/pod; pilot ~$13 across 3 debug rounds;
  two dead 48GB hosts at $0). All pods torn down (verified empty pod list).
- API: $14.82 natural judging + $0.66 glorptitude ≈ **$15.5** this stage.
- Wall clock: ~8h autonomous, of which fleet training+scoring ≈ 40–75 min/pod.

## Artifacts

- HF (public): `kaushikreddyxyz/concept-probes-gemma2-2b` — per-family raw
  training outputs, stacked deployment matrices, per-concept probe files,
  probe cards, metrics, natural scores, reports.
- GitHub: `concept_probes/stage5/` + `concept_probes/stage6/` (code, PLAN.md,
  STATE.md, this report, rollup + gates jsons; large npz/pngs on HF).
- Local: full weight mirror under `concept_probes/stage5/probes/` +
  `concept_probes/stage6/artifacts/`.
