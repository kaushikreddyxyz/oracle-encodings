# Stage 6.1 live progress

(newest first)

- T0+3h: WAVE 2 COMPLETE — all experiment code written, smoke-tested (tiny
  random Gemma2, CPU), and dry-run-validated against real data. Validation
  headlines: E2 exact +2.000 off-target diagonal at factor 2 (α-identity
  through the full eval pipeline); E4 rank-k composition exact to 4.8e-7 vs
  direct projection; E5 copy-matrix telescoping decomposition exact to 1.2e-6;
  E1 attribution vs true single-position patches r=0.974. Fleet estimates:
  e2_cloze ~70k forwards (~1-2 h), e2_ppl ~30k (~1-2 h), e4 ~3.3k calls
  (~20 min), e5 ~6.3k forwards (~0.5-1.5 h), e1 ~2.5k passes (<2 h) — total
  comfortably within 4 pods × ~2 h. Pod infra + E3 generate/judge + rubrics +
  plumbing_gate + FLEET.md done; step-0 HF pre-uploads (random_pool → model
  repo stage6_1/inputs/, judged_nat.jsonl ×13 → dataset repo) COMPLETE.
  Data-path decision fleet-wide: positives = stage6/data/natural/eval jsonls
  (max span strength ≥ 0.34), 53/64 concepts natural, 11 fall back to stage4
  val target_pos. NEXT: commit, pilot pod, plumbing gate.

- T0+1.5h: Harness COMPLETE (common.py + interventions.py, 7/7 unit tests on
  tiny Gemma2; transformers 5.10.1 quirks handled: prepend=True hooks so edits
  reflect in output_hidden_states; L25 meter is post-final-RMSNorm — exact-α
  identity holds layers −1..24; dose_calib converts natscores raw-W preds to
  unit-w score units, sanity january/L12 s95=1.52 t=0.065). WAVE 2 LAUNCHED:
  5 parallel agents writing e1_attrib, e2_cloze+e2_ppl, e4_ablate,
  e5_propagation, pod drivers + e3 generate/judge + plumbing_gate. Prompt-bank
  agent (wave 1 A3) still running.

- T0+1h: E0 COMPLETE (local, ~2s). Headlines: median cos(ridge,DoM)=0.291 but
  cos(ridge,LDA)=0.920 — ridge ≈ whitened-DoM (LDA), DoM is the odd one out;
  the causal arm comparison is genuinely whitened-vs-unwhitened. Adjacent-layer
  same-concept cosine 0.599 decaying to 0.159 at |Δl|≥12; glorptitude control's
  cross-layer geometry indistinguishable from real concepts (persistence alone
  proves nothing — as expected, E5 is the discriminator). Cyclic adjacency
  largely absent in raw cosines (only months/colors/weekdays weakly +; PCA ring
  in Stage 6 came from removing the shared family component). cos(σ⊙w,w⊘σ)=0.57
  → std-arm vs grad-arm ablation genuinely differ. Data notes: rand_dirs is one
  shared [20,2304] set fleet-wide; moon_phases npz classes use underscores.
  Artifacts: out/e0/*.npz + E0_SUMMARY.md, figures/e0/*.

- T0: Plan approved by user (budget $100–150, all 64 concepts, dose+specificity
  bar, both logit + judged evals). Spec written into knowledge task.md §6.1;
  LITERATURE.md + DESIGN.md committed. Wave-1 subagents launched: A1 e0_geometry
  (local CPU run), A2 intervention harness + unit tests, A3 audited prompt banks.
