# Stage 5/6 live progress

(newest first; timestamps America/local)

- T0+6h: FLEET + PILOT COMPLETE, all pods torn down. 768 probes (64×12) + the
  glorptitude control trained/evaluated/natural-scored; everything on HF
  (kaushikreddyxyz/concept-probes-gemma2-2b) and mirrored locally. GPU spend
  ~$30 total (fleet pods $3-5.5 each, pilot ~$13 across 3 debug rounds).
  Band check: all three pilot concepts peak in {6..20} (harmfulness L11 raw
  ρ .66; europe L17; january implicit-recall mid-band). Read(+1) ablation: Δρ
  +0.002..+0.013 — default read position kept.
- Two more gate operationalization findings (PLAN.md deviations #4-#5):
  ceiling-normalized natural ρ (tie ceiling 0.05-0.13 at 0.1-0.5% prevalence;
  probes sit at 0.77-0.98 of ceiling, example AUROC .984-.998 for moon_phases);
  vacuous selectivity checks (empty judge-truth pools, e.g. full moon has no
  concept-absent hard negatives) skip rather than fail.
- GLORPTITUDE CONTROL: generated-val AUROC 0.90-0.93, R_imp 1.0 — far outside
  [0.45,0.55]. The pipeline manufactures a detectable genre direction for a
  vacuous concept ⇒ generated-split metrics are DIAGNOSTIC ONLY for all 64
  probes; certification rests on the natural-data Tier-1 gates (which
  glorptitude structurally cannot have). To be featured prominently in report.
- Full 13-family gates pass running locally (ceiling-normalized, cal/test
  protocol). Next: plots, assemble_W, report, ship.

- T0+4h: FLEET LAUNCHED (4 H100 pods via driver subagents; user approved layer
  set incl. final block 25). Pre-launch pilot iterations settled three
  operationalization deviations, documented in PLAN.md: ridge=primary optimizer
  (Adam undershoots the convex optimum; kept as seed diagnostic), homograph FPR
  thresholded at the confirmed-positive 25th percentile (judge-truth faint
  echoes are correct behavior, not FPs), Hewitt–Liang demoted to S>0 sanity
  (control far from chance on residual streams). Validated on january:
  G 0.99 / R_imp 0.95-1.0 / FPR 0.03-0.06 / HL_S>0 / margin ~0.2 at mid layers.
  Pilot pod continues harmfulness+continents 26-layer sweep + read-shift
  ablation in parallel.

- T0+2.5h: NATURAL JUDGING COMPLETE: 23,776 examples, all 13 families, $14.82,
  ~19 min wall (mercury via OR conc 96 + Inception spillover). 26 examples have
  <3 samples (bad calls in color_wheel/continents/directions/moon_phases) —
  same 2-of-3 convention as Stage 4. Token-level eval sets built for all
  families (build_natural_eval.py; 0 rows dropped).
- Pilot round 1 caught 3 issues pre-fleet (that's its job):
  (1) Adam undershot the exact ridge optimum (ρ 0.22 vs 0.38) — batch 131k gave
      ~200 steps; fixed: batch 16384, lr 3e-3, verify vs ridge on rerun.
  (2) Hewitt–Liang control with uniform 0..6 per-type labels is denser and
      rank-easier than the real mostly-zero task → control ρ 0.6-0.8, S deeply
      negative for everyone. Fixed: control labels drawn from the class's own
      empirical label marginal (comparable tie structure).
  (3) G-ratio NaN for january: its form_holdout is 100% "Jan"-the-NAME hard
      negatives (form_test surface collides with the hazard) → 0 judged
      positives. Fixed: fall back to implicit slice when <10 positive form
      rows (deviation-#2 convention), n_form_pos recorded.
  Also observed, expected: token-level ρ layer-profile is nearly flat (labels
  are explicit-heavy → surface-detectable at every layer); the concept-vs-token
  divergence shows exactly where the spec predicts — implicit recall peaks
  L8–16 (0.47–0.74) vs ~0.1 at L0–3. natstats26.npz (1.65M natural tokens)
  pulled to the mac. Ensemble precondition: resid corr 0.997 → skip (expected).
- Pilot rerun in flight (cache reused; retrain+re-evaluate all 3 concepts).

- T0+1.5h: Natural pool frozen (5k standardization docs shard 310; 6k random
  windows 311–312; 6,676 mined windows 311–316; only 4 rare moon-phase classes
  under 60). Judge inputs packaged: 23,776 natural records across 13 families →
  judging lane RUNNING in background (judge.py --tag nat, $5 cap/family,
  OR primary + Inception secondary). Pilot pod probe-pilot (RTX A6000,
  $0.49/hr, id zsya89wb3t218h) created — waiting on sshd.

- 2026-07-02 ~T0+1h: Core code written & smoke-tested locally: common.py (family
  loading, buffer mask B=10, read-shift ablation — verified on seasons),
  extract.py, natstats.py, train.py (stacked Adam rows [seeds×λ×classes] +
  closed-form ridge + DoM/LDA/logistic/random baselines), evaluate.py (ρ/AUROC/
  AUPRC vs scipy+sklearn ✓, G-ratio, implicit recall / homograph FPR at τ,
  shuffled + Hewitt–Liang controls, random-dir margin, example-level bootstrap
  CI, §5.3 ensemble check), pod_setup.sh, pod_run.sh.
- Glorptitude control data DONE ($0.66, 1,107 rows). Judge-level slice
  separation present for the vacuous concept (explicit 0.594 / implicit 0.628 /
  hard 0.161 / neutral 0.010) — interpretation deferred to the probe-level test;
  note implicit > explicit inversion (unlike every real family) as evidence the
  judge scores term-mention, not semantics.
- Natural split + mining running in background (shards 310–316).
- HF weights repo created: kaushikreddyxyz/concept-probes-gemma2-2b (public).
- OpenRouter has auto-top-up (<$5) per user → judging rides OR primary.

## Next
- Pilot pod: 26-layer sweep january/harmfulness/europe + natstats (all 26 layers).
- Fleet g1–g4 after pilot green.
