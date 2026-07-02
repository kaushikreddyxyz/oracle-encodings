# Stage 6.1 live progress

(newest first)

- T0+9h: SHIPPED. E3 complete both passes (factor 1-2 underdosed: zero judged
  incorporation, confirmed qualitatively; high-dose 4-8 rerun on pod E):
  DoM 0.297 mean judged overall (22/64 concepts > 0.5, best factor 8, fluency
  intact) vs ridge 0.059 ≈ baseline — near-quantitative AxBench replication
  (their DiffMean 0.239 vs Probe 0.098). ONLY ridge generation-steerers =
  the 3 intensity scalars (physical_size/harmfulness/lovingness ≈ 0.55) —
  the weakest readers are the best steerers, both arms (physical_size dom
  0.88 top overall). Judge total $8.09 (2,301 cached calls reused). ALL PODS
  DELETED (verified []). REPORT_6_1.md finalized; causal_cards + figures +
  rollup in out/analysis; consolidated fleet outputs in out/fleet/pod{A..D}.
  Final verdicts: ridge 52 causal / 10 read-only / 2 artifact-suspect;
  DoM 59/4/1. Total stage cost ≈ $45-50.

- T0+7h: FLEET COMPLETE (E1/E2/E4/E5 × 64 concepts, all rc=0). Pods A, B, C
  torn down after verifying local copies (scratchpad/fleet_out/pod{A,B,C},
  50+44+31MB, 74 npz total). Pod D (jm5ui4zjtqdvvv) still up running E3
  generation for all 13 families at selection.json doses (factor 2.0 at each
  concept's best-ridge-slope layer, derived from fleet cloze results).
  Hot-patch mid-fleet: e4 intensity poles.{low,high} tokens loader (committed)
  — landed on all pods before their e4 phase; intensity necessity rows
  confirmed (harmfulness dom −4.68 nats vs ridge −0.85 vs rand −0.11).
  PARTIAL ANALYSIS (48 concepts, A+B+C): ridge ~23+/… causal; DoM causal
  almost everywhere; dissociations one-directional (ridge fails specificity/
  necessity where DoM passes) except thursday & full_moon (reverse).
  Notable: costliness + physical_size CAUSAL both arms (weak readers, real
  directions); duration artifact-suspect both arms. RunPod lesson: creation-
  time SSH port can be stale — trust `runpodctl pod get`.

- T0+4.5h: PILOT PASSED, FLEET LAUNCHED. Plumbing gate PASS on real model
  (steer α=1 → score shift 0.9990, p95 err 0.4%). Pilot (january/harmfulness/
  europe, all 5 scripts, ~15 min GPU): e2_cloze needed scipy (added to
  pod_setup, committed). PILOT SCIENCE: cloze dose-response clean — europe
  ridge slope +0.39 / dom +0.51 / rand +0.005, anti-steerable 0% (rand 52%);
  january ridge +0.18 / dom +0.23; harmfulness ordinal Spearman +1.000 both
  arms. e2_ppl: dom slopes +0.002..+0.008 CI>0 everywhere, ridge ~0 or neg —
  reading≠steering dissociation is METRIC-DEPENDENT (graded on cloze logits,
  stark on free-text LL). e4 necessity: dom ablation −1.86 nats diag-lp
  (january) vs ridge −0.25, rand −0.009, other-concept −0.08; KL guard small.
  e5 flag for analysis: salient_layer=25 for all 3 = likely last-layer-logit
  artifact; use full curves + late-layer correction. FLEET: pod A =
  a864w0kd78dmeg (64.247.201.33:18492, $3.29/hr) running months+seasons;
  pods B (color_wheel,location_type,costliness,physical_size),
  C (weekdays,moon_phases,duration), D (directions,continents,lovingness,
  harmfulness) being brought up by subagents. Pilot outputs archived on pod
  at out_pilot/, uploaded to HF stage6_1/out, local copy in scratchpad.
  TEARDOWN REMINDER: 4 pods must be deleted when fleet completes.

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
