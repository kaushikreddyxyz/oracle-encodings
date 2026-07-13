# Stage 5+6 execution plan (probe training + validation)

Owner: autonomous agent session, 2026-07-02. Spec: `knowledge/concept_probes/task.md` §5–§6.
Inputs: Stage 4 data (`concept_probes/stage4/data/<family>/final/`, mirrored at
`hf.co/datasets/kaushikreddyxyz/concept-probes-stage4-data`).

## Decisions locked at kickoff (user-approved)

- **Budgets**: RunPod fleet approved (~$30–60); API judging approved (OpenRouter has
  auto-top-up below $5 → judging rides OR at conc 96, Inception secondary with the
  Stage-4 reservation limiter; explicit run cap instead of balance-derived cap).
- **HF destination**: new public model repo `kaushikreddyxyz/concept-probes-gemma2-2b`.
- **Domains**: deployment corpus is ClimbMix web-heavy ONLY (Stage-4 user decision), so
  the §6.6 per-domain Tier-1 gate collapses to the web domain; documented, not silently
  dropped.
- **Natural shards**: ≥310 (nanochat consumed ~0–183; overnight runs 300–309; keeping
  disjoint for cleanliness even though nothing was *trained* on 300–309).

## Architecture (matches user's batching idea)

Per pod: ONE gemma-2-2b forward pass over the family-group's **unique** examples
(sibling rows repeat example_ids; dedup before extraction), caching fp16 residuals
(`hook_resid_post` ≡ HF `hidden_states[l+1]`, BOS prepended, positions shifted +1) for
the 12 probed layers. Then per layer, ALL probes of the group train **simultaneously**
as stacked independent rows: W of shape `[seeds(3) × λ(3) × classes, d]`, each row with
its own per-token target y_c and mask m_c over the shared unique-token array — every
example updates every row whose dataset contains it, rows never interact (§5.1).
Closed-form ridge (the exact minimizer of the same masked-MSE+L2 objective) is computed
alongside as a verification of the Adam fits.

- Buffer mask (§5.2): B=10 tokens after each positive-strength span get m=0 (only where
  the buffered token itself has y=0).
- Standardization (§0.6): per-layer μ/σ from the **natural** random-passage sample,
  computed once on the pilot pod, shipped to the fleet. Raw-space (w', b') also emitted.
- Baselines per (class, layer): DoM, shrinkage-LDA, logistic (binarized y≥0.5),
  20 random unit directions. Controls: shuffled-label refit, Hewitt–Liang token-type
  control refit (both via closed-form ridge at the chosen λ).

## Pod topology

| pod | families | ~unique examples |
|---|---|---|
| pilot | january+harmfulness+europe classes, 26 layers | ~12k |
| g1 | months, seasons | ~41k |
| g2 | color_wheel, weekdays | ~50k |
| g3 | moon_phases, directions | ~39k |
| g4 | continents, location_type, 5 intensity axes | ~42k |

48GB-class GPUs (A40/L40S), ≥250GB container disk. Activation caches are ephemeral
pod-local scratch (user: no need to keep training activations). Fleet pods are driven by
subagents; the pilot validates the full stack first and gates the fleet.

## Stage 6 data plan

Tier-1 gates need judge-labeled NATURAL data (§6.2): mine ClimbMix shards ≥310
lexically per family (surfaces = form_train + form_test + a periphrase subset from the
Stage-4 packs, so hard negatives like modal "May" arrive naturally), add a shared
random-passage pool per family, judge with the **unchanged Stage-4 pipeline** (mercury-2,
K=3, rubrics v1/v2/v3, family-level calls), aggregate to token targets with the same
char-painting. Natural eval activations scored on a pod at the end of fleet training.
ECE Tier-1 gate needs a calibration map → minimal isotonic fit on a disjoint half of the
natural pool (cal-half), ECE reported on the other half (test-half). Split is frozen
before any scoring.

Nonsense-concept control (§6.4): fabricated concept "glorptitude" through the identical
gen→judge→train path (~1.5k examples); pipeline is INVALID if its AUROC leaves
[0.45, 0.55].

## Pilot-driven deviations (documented before fleet launch)

1. **Primary optimizer = closed-form ridge**, not Adam (§0.6 prescribes Adam).
   The objective (§5.2 masked MSE + L2) is convex; ridge is its exact minimizer.
   Adam systematically undershot it in three pilot rounds (Δρ_val ≈ −0.24 even
   with a cosine schedule at 250 epochs; a 600-epoch single-row control run does
   converge, confirming pure optimizer slowness, not an objective mismatch).
   Adam (3 seeds) is retained as the Tier-2 cross-seed-stability diagnostic.
2. **Homograph FPR threshold**: hard negatives carry deliberate faint-echo
   judge-truth (~1/6, the 2B rule), so "fires above the neutral floor" is
   correct behavior, not a false positive. FPR counts hard negatives scoring
   above the 25th percentile of confirmed-positive example scores (τ_strong);
   the neutral-floor variant is kept as a diagnostic.
3. **Hewitt–Liang gate demoted to a sanity check (S > 0)**: on residual streams
   the per-type control is far from chance for any zero-inflated regression
   task (pilot: control ρ 0.26–0.34 at EVERY layer — token identity is linearly
   decodable everywhere), violating the spec's "control near chance" premise.
   Raw S and control ρ are still computed and reported.
4. **Natural ρ is gated ceiling-normalized (ρ_rel = ρ / ρ_ceiling)**: at natural
   token prevalence (0.1–2% nonzero targets) the zero-tie block bounds a
   PERFECT continuous scorer's token-level Spearman at 0.05–0.13 (measured:
   ρ_ceiling = ρ(y+ε, y) on moon_phases), making the spec's absolute 0.65/0.45
   thresholds unattainable for any probe. ρ_rel reduces to the spec's rule
   exactly when ties are negligible (ceiling→1, as in balanced generated val).
   Evidence: moon_phases probes sit at 0.77–0.98 of ceiling with example-level
   detection AUROC 0.984–0.998. Raw ρ, the ceiling, and prevalence are all
   reported alongside ρ_rel.

## Metric operationalizations (where the spec leaves freedom)

- Spearman ρ: token-level, masked tokens of the split, scipy `spearmanr`.
- G-ratio (§6.3): AUROC(form_test positives vs val neutrals) /
  AUROC(form_train explicit val positives vs val neutrals).
- Operating threshold for implicit recall + homograph FPR: score pooled per example
  (max over tokens); τ = 95th percentile of neutral-example scores on generated val
  (5% neutral FPR); R_imp and FPR_homograph reported at that τ.
- Selectivity gap = min of the four §6.3 checks after each is normalized to [0,1]
  (G capped at 1; R_imp as-is; 1−FPR/0.10 capped to [0,1] → use 1 if FPR≤0.10 scaled
  linearly below; HL selectivity S/0.30 capped at 1). Exact formulas in gates.py.
- Hewitt–Liang control labels: per token-type fixed random score in {0..6}/6.
- Ensemble (§5.3): only fit if mean pairwise val residual correlation ≤ 0.9; adopt only
  if Δρ_val ≥ 0.03 AND it passes §6.3 itself.

## Deliverables

`W^(l), b^(l)` stacked per layer (+ per-probe unit-norm rows, standardization stats,
raw-space equivalents), per-probe metrics JSON (Tier 1/Tier 2 tagged), probe cards,
roll-up table (64 rows), per-concept Level-2 pages, report. HF model repo + GitHub +
weights pulled to the Mac. Pods torn down. STATE.md tracks live progress.
