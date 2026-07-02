# Stage 6.1 report: causal validation of the concept probes

2026-07-02. Spec: `knowledge/concept_probes/task.md` §6.1 (added this session).
Inputs: the 768 Stage-5 ridge probes + Stage-6 verdicts. Question: **do the
learned directions correspond to the ground-truth feature — causally?**
Evidence bar (user-set): monotone dose-response + specificity + ablation
necessity, always against matched random-direction controls.

## Headline

**52 of 64 ridge probe directions are causal** by the frozen §6.1.0 bar
(10 read-only, 2 artifact-suspect). Via the difference-of-means (DoM)
direction the count is 59/64. The literature's "probes read but don't steer"
dissociation (AxBench, ITI, amnesic probing) appears in our setting in a
sharper, **metric-dependent** form:

| median over 64 concepts | ridge (the probe) | DoM | matched random |
|---|---|---|---|
| steering: cloze dose-slope | **0.201** | 0.192 | 0.001 |
| steering: anti-steerable fraction | **0.00** | 0.03 | 0.48 |
| steering: intensity ordinal Spearman | **0.99–1.00** (5/5 axes) | 1.00 | 0.55–0.73 |
| ablation: Δ diagnostic-token log-prob | −0.12 | **−1.90** | −0.01 |

- **For steering (sufficiency), the ridge direction equals DoM** — clean
  monotone dose-response, zero anti-steerable templates at the best layer,
  symmetric suppression at negative doses. On free-text log-likelihood
  (ActAdd perplexity-ratio) the gap re-opens (DoM slopes +0.002..+0.008
  nats/token with CIs > 0; ridge ≈ 0): ridge steering moves concept-diagnostic
  *tokens*, DoM also moves broad concept *text likelihood*.
- **For erasure (necessity), DoM ≫ ridge** (15× median damage). Everywhere-
  ablating DoM deletes concept behavior almost completely (january: −1.86
  nats on diagnostic tokens, europe −1.11, harmfulness −4.68) with a small
  off-concept KL guard; ridge ablation damages 10–30× more than random but
  4–15× less than DoM. Belrose's theorem says span(DoM) is exactly what must
  be removed to linearly erase a binary concept — our measurement agrees, and
  quantifies how much of that necessity the ridge (≈ diagonally-whitened)
  direction captures: a minority share.
- E0 geometry explains the mechanics: cos(ridge, DoM) ≈ 0.29 while
  cos(ridge, LDA) = 0.92 — the ridge probes are whitened mean-difference
  directions. Whitening preserves the *steerable* component but sheds most of
  the *necessary* subspace weight.

## Verdicts (ridge arm — the probes under validation)

| family | causal / read-only / artifact | notes |
|---|---|---|
| color_wheel | 12 / 0 / 0 | |
| directions | 8 / 0 / 0 | |
| months | 10 / 2 / 0 | april, january read-only (specificity + necessity-vs-random) |
| moon_phases | 7 / 1 / 0 | |
| weekdays | 4 / 2 / 1 | saturday artifact-suspect (also a Stage-6 reject) |
| continents | 4 / 2 / 0 | |
| seasons | 2 / 2 / 0 | autumn, spring read-only |
| location_type | 2 / 0 / 0 | |
| costliness, physical_size, lovingness | 3 / 0 / 0 | see below |
| harmfulness | 0 / 1 / 0 | ridge read-only; **DoM causal** |
| duration | 0 / 0 / 1 | artifact-suspect on both arms |
| **total** | **52 / 10 / 2** | DoM: 59 / 4 / 1 |

**Reading quality and causal reality dissociate in both directions.**
`lovingness` — Stage 6's *rejected* reader (ρ_rel 0.45) — is **causal on both
arms**: its direction steers monotonically and its removal deletes lovingness
behavior. `costliness` and `physical_size` (caveat readers) likewise. In the
other direction, `saturday` (ρ_rel 0.95 as a reader) is artifact-suspect
causally. A probe can read a concept it cannot control, and encode a concept
it cannot read on natural text — the Stage-6 and Stage-6.1 verdicts are
complementary axes, and probe cards now carry both.

Causal verdicts **report, never gate** (§6.0.7): Stage-6 deploy/caveat status
for attribution is unchanged.

## The layer story (the user's l → l+1…L question)

1. **The causally salient layer is mid-band and disagrees with the
   correlational choice.** Corrected E5 salience (earliest layer with ≥70% of
   max behavioral deficit, excluding the trivially-late L25) is bimodal at
   **L8 and L12** (44/64 concepts). E1's attribution-patching candidate
   disagrees with Stage 6's CAL-half chosen layer for **61/64 concepts** —
   expected, since token-ρ layer profiles were nearly flat for lexical
   concepts (Stage-6 report §"chosen layers"), but it means *deployment layer
   choice should not be read as causal localization*.
2. **Later layers mostly copy.** The exact residual decomposition gives
   median identity-path share C[l,l+1] ≈ **0.97** (an ablation at l arrives at
   l+1 essentially intact) decaying to ≈ **0.72** at L25: about three quarters
   of a mid-layer edit survives to the end through the identity path alone,
   the rest is reintroduced/eroded by downstream blocks. Median behavioral
   half-life of an erasure: **5 layers**. Copy structure correlates only
   weakly with direction cosine similarity (pilot r ≈ 0.28): **direction
   rotation across layers is not evidence of recomputation**, confirming the
   E0 caveat empirically.
3. **Writing happens mid-late.** Denoise-style write-layer localization puts
   mass at L12–L20. Combined picture per concept: written by the mid stack,
   read within ~5 layers, then carried forward largely passively.

## Effective causal rank (multiclass families)

Erasing the family's top-k DoM subspace (all layers) vs matched-rank random
subspaces (which do ≈ nothing):

| family | classes | k50 | k90 |
|---|---|---|---|
| weekdays | 7 | 1 | 1 |
| moon_phases | 8 | 1 | 1 |
| directions | 8 | 1 | 2 |
| seasons | 4 | 1 | 3 |
| months | 12 | 2 | 4 |
| continents | 6 | 3 | 4 |
| color_wheel | 12 | 1 | 8 |

Family cloze behavior collapses at **far below the theoretical k−1 bound** —
the model's usable family information is concentrated in a compact shared
subspace (1–4 directions for most families), consistent with the E0 finding
that siblings share a large common component. Color wheel is the high-rank
outlier (8), matching its Stage-4 status as the hardest family.

## E3 — judged steered generation (AxBench-adapted, base model)

At the logit-validated doses (factors 1–2 × s95) steering produced **no
judge-visible concept incorporation** in free generation for any arm
(incorporation ≈ 0.03–0.05 / 2 vs baseline 0.025; fluency unharmed) —
fluent continuations, zero concept content. Token-level causality at these
doses does not surface in 128-token continuations. A high-dose rerun
(factors 4 and 8) was run to completion; results:

At factors 4–8 the picture resolves into a near-quantitative **replication of
AxBench's probe-vs-DiffMean finding** on a base model (held-out half, harmonic
mean of concept/topicality/fluency on 0–2):

| arm (best factor per concept) | mean overall | incorporation | fluency | concepts > 0.5 |
|---|---|---|---|---|
| baseline (α=0) | 0.027 | 0.025 | 1.70 | 0/64 |
| **DoM** (best factor = 8 for 34 concepts) | **0.297** | 0.406 | 1.63 | **22/64** |
| ridge | 0.059 | 0.064 | 1.66 | 3/64 |
| random | 0.034 | 0.037 | 1.69 | 2/64 |

AxBench reported DiffMean 0.239 vs Probe 0.098 on Gemma-2-it; we get 0.297 vs
0.059 on the base model with judge-blind rubrics. Fluency is essentially
unharmed at factor 8 — the concept score, not degradation, separates the arms.
Top steerers: physical_size 0.88, harmfulness 0.75, july 0.72, october 0.69.

**The only three concepts whose ridge direction steers generation are the
intensity scalars** (physical_size 0.55, harmfulness 0.56, lovingness 0.56) —
the same graded axes that were the *weakest readers* in Stage 6. Example
(physical_size, DoM, factor 8): fluent botany text with size content woven in
("grows in dense mats … reaching up to 12 feet … covers large sections of
Central Texas").

Full four-way dissociation across measurement levels, per arm:

| level | ridge | DoM |
|---|---|---|
| forced-choice cloze logits | ✅ ≈ DoM (slope 0.20) | ✅ 0.19 |
| free-text likelihood (ppl-ratio) | ~0 | ✅ CI > 0 |
| judged 128-token generation | ✗ (except intensity axes) | ✅ 22/64 |
| everywhere-ablation necessity | partial (−0.12) | ✅ (−1.90) |

Reading: the whitened (ridge) direction moves the concept's *token-level
preferences* precisely — enough to win every forced-choice — but lacks the
high-variance components (Im & Li's diagnosis) needed to push the whole
generative state into the concept's region; the unwhitened mean difference
carries them. For deployment this cuts cleanly: **use ridge rows for reading
(Stage 6 stands), use DoM rows (already shipped in the same npz files) for
steering.**

## Addendum — ridge vs DoM as READERS on natural text

Stage 6.0's "ridge beats or matches all baselines" was measured on generated
val (ridge's home turf). Re-run on the judge-labeled **natural test half**
(`code/reading_arm_compare.py`, from stored natscores, CPU-only):

| natural test, chosen layers | ridge | DoM |
|---|---|---|
| example-level detection AUROC (median) | **0.975** | 0.949 |
| — per-concept wins | **44** | 20 |
| token-level raw Spearman (median) | 0.0833 | 0.0856 |
| — per-concept wins | 15 | **49** |

Ridge is the better *detector* (the deployment metric behind the Stage-6
verdicts); token-level rank fidelity is a statistical tie leaning DoM (+3.6%
median, with the layer selected for ridge — a small DoM handicap). Combined
scorecard across both stages: ridge wins detection, ties cloze-logit steering,
loses necessity (15×) and judged generation (5×). Caveat: DoM never ran the
Stage-6 Tier-1 gates (selectivity, homograph FPR, ECE) — certifying it as a
reader is a zero-GPU follow-up from the natscores files.

## Controls & guardrails held

- Random matched-norm directions: slope ≈ 0, anti-steerable ≈ 50% (coin-flip),
  ablation damage ≈ 0 — every reported effect clears its own control.
- Other-concept ablation (specificity): −0.08 median (vs target's own −1.9 DoM).
- Off-concept KL guard stayed small for real-arm ablations (0.01–0.04 nats).
- Necessity passes 63/64 concepts on at least one arm.
- The E5 raw salient-layer argmax was L25 for essentially all concepts — a
  mechanical last-layer artifact, corrected as specified before reading results.

## Deviations & limitations

1. **E3 first pass underdosed** (factors 1–2): fixed by rerun at 4/8; both
   sets reported. Judged with mercury-2 (same judge family as Stages 4–6;
   monoculture risk carries over to E3 only).
2. E5 "frozen-downstream" control freezes attention **outputs** (stronger
   than pattern-freezing); attn-self-repair summary was null for some
   concepts — analysis uses the decomposition + half-life instead.
3. Rank-k erasure uses orthogonal projection in diagonally-standardized
   space, not full-Σ LEACE (no full covariance available); matched-rank
   random controls bound the collateral damage this could cause (≈ none).
4. Cloze slopes are fit on factors ∈ [−2, 2]; wider factors reported but not
   in the fit. Dose units are per-concept (multiples of the 95th-percentile
   natural score), so slopes are comparable across concepts, not across
   different normalization schemes.
5. Intensity-axis necessity uses high-pole diagnostic tokens only
   (lovingness low pole = "despise" untested as its own direction).
6. glorptitude (nonsense control) has no natural dose calibration, so it ran
   only through E1/E5 screening, not the steering battery — its causal-battery
   absence is a known gap (its Stage-6 lesson stands: only natural-data
   validation certifies).

## Costs & infrastructure

- GPU: 5 H100 pods (pilot/A, B, C, D, E3-rerun), $3.29/hr each, total ≈
  **$35–40**; every pod torn down after result verification (verified empty).
- Judge API: **$8.09** total (first pass $3.32 + high-dose pass $4.78, with
  2,301 cached calls reused for free on the rerun). Total stage ≈ **$45–50**,
  within the $100–150 budget.
- All experiment outputs mirrored: local `stage6_1/out/fleet/pod{A..D}` +
  HF `stage6_1/out/*` (per-script uploads during the run).

## Artifacts

- `out/analysis/causal_cards.json` — per-concept card: sufficiency slopes+CIs,
  anti-steerable fraction, necessity effects per arm, specificity, salient /
  write layers, copy-matrix summary, family causal rank, verdicts (ridge+DoM).
- `out/analysis/figures/` — 64 per-concept 4-panel pages; fleet arm-scatter,
  copy-vs-cosine, 64×64 off-target heatmap; `causal_rollup.md|.html`.
- `out/e0/` — geometry npz + E0_SUMMARY.md.
- Code: `stage6_1/code/` (harness + 5 experiments + analysis, all committed);
  audited prompt banks in `stage6_1/prompts/`.
