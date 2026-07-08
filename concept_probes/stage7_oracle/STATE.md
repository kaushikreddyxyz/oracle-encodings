# Stage 7-Oracle live progress

(newest first)

- ~1:00 PM: pod A VERIFIED + RETIRED (all closed-form checks PASS: score
  restoration 2.1e-4, identity p99 5e-7, quant path p50 3.6%, v* crosscheck
  exact 0.0; report committed; 33/33 files byte-confirmed on trainer incl.
  a manual shard-330 pull; pod deleted). Exp B launched on trainer (fixed
  then learn, frozen Exp-A encoder). Coords throughput: feeder-pool fix
  was a validated NEGATIVE — job is GPU-FORWARD-bound (86% of wall;
  latency-bound ragged batches), not CPU-bound; serial path restored
  md5-clean on coords1. New fix dispatched: cross-doc length-bucketed
  batching + token-budget batches + tf32 behind --fast-forward, relaxed
  equivalence contract (one int8 step, exact zero positions; completed
  shards kept, fit frozen). If ≥2.5×: coords done ~7-9 PM, nanochat
  tonight. If not: decide between 12-pod fleet (halve wall, same ~$305)
  or SPEC's first-N-tokens + β-anneal fallback.

- ~12:20 PM: **frozen-encoder baseline COMPLETE** (spec's MLP-only arm,
  converged: early-stop fired at final step): heldout median R² **0.1823**
  vs full-FT **0.6371** on identical data/splits — 3.5× gap; the oracle
  signal comes from fine-tuning the encoder, not readout of pre-existing
  features (continents outlier 0.45 = base model partially encodes
  geography). lr 3e-3 head-only, 66 min, 57-72k tok/s. Trainer GPU free
  for Exp B (waiting on pod-A verifier verdict). Meanwhile: coords sweep
  CPU-bound problem (~28k tok/s/pod → 19h/$305 instead of 5h/$90) —
  feeder-parallelization fix in progress (opus agent; determinism A/B
  required before rollout; sweep is per-shard resumable so hot-swap is
  cheap). Consolidation 98/129 files on trainer (C/D relay tail);
  watcher armed for 129; B/C/D torn down per-pod as their sets confirm.

- ~11:10 AM: **Phase 1 scoring COMPLETE** — all 43 shards / 2.0B tokens,
  zero failures/stalls, ~$45 (monitor exited on all-done). Coords fleet
  READY: 6× H100 provisioned (coords1-6, $17.94/hr fleet), nanochat
  @f7c3119 + baseline tokenizer staged, imports pass everywhere;
  **qwen↔nanochat crossing rate 3.86% mean / 7.1% p90** (better than the
  gemma↔qwen 7.08% — prefix bridge validated). Balance $690.80. Coords
  fit→sweep launch agent dispatched (best.pt distributed via new HF repo
  kaushikreddyxyz/stage7-oracle-encoder; fit clip-frac gate; per-pod
  round-robin sweep of shards 0-190; first-minute byte-partition +
  zero_frac watch). Frozen-encoder baseline running on trainer.
  Waiting: pod-A verifier (gates ExpB), B/C/D teardowns, HF archival.

- ~10:40 AM: **GATE G2 = GO (clean, both inputs).** Exp-A early-stopped at
  step 5600/6800 (plateau rule): final heldout median R² **0.6371** ≥ 0.6.
  Natural-eval AUROC retention (audited gate, 7488 judged examples, 54
  concepts): median chance-corrected ratio **0.9659** (raw 0.9836) vs bar
  0.90; all 7 families ≥ 0.90; conservative fixed-layer secondaries
  L6/L8/L14 = 0.929/0.942/0.924 all clear → marginal-band rule not
  triggered. best.pt (3.6GB, val-best) is the deployed encoder.
  LAUNCHING: coords fit+sweep on the pre-provisioned 6-pod fleet
  (checkpoint distributed via HF), frozen-encoder baseline on the freed
  trainer GPU (same shards as full-FT, for the spec's MLP-only arm
  comparison). Exp B waits for the closed-form verifier verdict (pod A,
  in flight). g2_retention.json copied to out/.

- ~9:25 AM: post-scoring wave dispatched. Fleet: pod D DONE (10/10),
  A/B/C on last 1-3 shards. Exp-A step 4206/6800, R² 0.634 rising.
  Fable audit of precompute_coords: GO — added the **preflight gate**
  (consumer-path token/lookup cross-validation; kills the
  silent-all-zeros $200 failure mode), assemble hard-fails on missing
  shards, fit/sweep probe-set consistency guards, chunked forwards (OOM),
  resume-safe Welford, vectorized byte→char (77db86f). Hand-verified 3
  concept→θ chains (march 60°, southwest 225°, waning_gibbous 225°).
  Agents running: pod-A verification (verify_closed_form --attn eager,
  then teardown), B/C/D consolidation-confirmed teardowns, HF archival
  driver (kaushikreddyxyz/concept-probes-corpus-scores, sequential
  background uploads), g2_retention.py implementation (natural-eval
  AUROC-retention — the last G2 input). NEXT DECISIONS: G2 final call
  when training ends (~10-10:30) → coords fleet (6×H100, ~$90) + ExpB/
  frozen-baseline queue on trainer → preflight → nanochat launch.

- ~8:55 AM: **Exp-A crossed the G2 bar**: heldout median R² 0.5999 @2000
  → 0.6129 @2400 → 0.6203 @2800 (of 6800 steps), still climbing, plateau
  stop not near firing. Audited metric (conservative biases) → this is a
  PRELIMINARY GO for Phase 4; final G2 call after the run ends + the
  SPEC's natural-eval AUROC-retention check (≥90% of gemma probes' AUROC
  on Stage-6 judged texts). The SPEC's 50M-fineweb distribution check is
  MOOT (nanochat corpus = same ClimbMix repack, disjoint shards).
  Nanochat-patch Fable audit: GO-with-conditions, 3 critical fixes
  (unappliable diffs regenerated; inject-after-block 8→7 in three places;
  missing-doc fallback injected FULL-AMPLITUDE NOISE → now exact zeros)
  + per-(seed,doc-hash) noise RNG, double-standardization removed.
  precompute_coords.py full implementation dispatched (Opus, then Fable
  audit). Fleet: 7-8 shards/pod done, ~45 min to completion.

- ~8:30 AM: AUDIT WAVE (user-directed tiering: Fable audits Opus/Sonnet
  gate-critical code, Opus audits the rest). Results so far:
  (1) Opus audit of score_corpus/align: NO defects on hot paths; two
  documented latents — int8 tail behavior (firing tokens lose ordering
  >4σ; std slightly <1 after standardization; consistent train/heldout)
  and align's silent offset-clamp (dormant for gemma→qwen; RELEVANT for
  qwen→nanochat tiktoken path). (2) Fable audit of train_encoder/
  verify_closed_form: **live run's G2 number TRUSTWORTHY** — R² median
  over all 162 targets, disjoint splits (now enforced), correct
  dequant/standardization pairing, every known imperfection biases R²
  DOWN (conservative vs 0.6 bar). Fixed: stale canary test that was
  masking the verify suite, added v*-crosscheck (verifier vs trainer
  formulas — currently bit-identical), shard-overlap + stats-shape
  guards. Documented: verify check-2 partial circularity — t_nat_dom
  correctness rests on PERMUTATION_FIX's offline dom-block audit (done);
  ⚠ --resume does NOT restore the LR scheduler (matters if we continue
  Exp-A onto wider shards: fast-forward scheduler or relaunch fresh).
  Pod re-sync needed for verify_closed_form.py before pod-A run.
  Committed 862022c. Still pending: Fable nanochat-patch audit, fleet
  completion, next Exp-A evals. G1 residuals closed earlier (jan-mar
  r=0.308 PASS; int8 clipping non-blocking, quantified).

- ~7:20 AM: **G1 = FAIL, root-caused, fix in flight (no rescoring
  needed).** select_probes.py assembly bug: W/b main-block rows filled in
  (family,concept)-sorted order while probe_set.json "concepts" is
  name-sorted → 53/54 concepts MISLABELED in the 3 main store blocks
  (162/216 columns). Evidence: "january" column fires on east/eastern
  tokens (it IS the east probe), "north"→april, "red"→may. CRUCIAL
  SCOPING: the stored scores are internally-valid probe scores with wrong
  name tags — (W,b) pairs consistent, normalization correct; the DoM
  block (Exp B's input) is CORRECTLY ordered. Therefore: store is
  immutable-but-fine, fleet keeps scoring, the LIVE Exp-A run stays valid
  (median R² is permutation-invariant; G2 unaffected) — only per-concept
  labels are scrambled (earlier "europe@L8 0.73" texture was mislabeled).
  Opus fix agent dispatched: verify permutation 2 independent ways, add
  main_block_concepts/dom_block_concepts keys to probe_set.json, audit
  every json field's pairing, fix label consumers (train_encoder eval
  labels, verify_closed_form, coords drafts — coord phase angles MUST get
  true concepts), fix select_probes for posterity, re-issue G1 verdict
  under true labels. G1 full evidence: out/G1_REPORT.md.

- ~6:50 AM: Exp-A PRODUCTION RUN LIVE on trainer (PID 8276): full-FT
  Qwen3-0.6B, train shards {320,321,331,332} (~187M tok, 1 epoch = 6800
  steps, bsz 6 × accum 8), val {353,354}, eval every 400 steps × 5M tok,
  corpus_stats.json merged from 8 partials (374M tokens, real stats — no
  fallback warning). Health: token-id assertion passed, loss 8.08→0.51 by
  step 500, **first eval median R² = 0.400 @ step 400** (bar for GO is
  0.6 — trajectory promising), 21.6k tok/s, GPU stable. Projected finish
  ~2h50m from 6:40 AM (~9:30) or earlier on plateau; if R² still climbing
  at epoch end, resume from checkpoint onto the wider consolidated shard
  set. Fleet monitor patched (ssh -n bug: it was only really checking pod
  A each round) and restarted — all 4 scorers verified progressing
  (~5/11 shards each). G1 gate agent launching now on the ~1B tokens
  already scored (don't wait for fleet completion).

- ~6:10 AM: Exp-A PILOT done; PRODUCTION full-FT launching. Pilot caught a
  real dtype bug (train loop cast features to float32 against a bf16 head
  — instant crash on GPU, invisible to CPU smoke tests; fixed + synced).
  Pilot numbers: frozen-encoder R² still ≈ −0.45 @500 steps (slow), FULL-FT
  R² +0.057 @100 steps → MLP-only very unlikely to hit the 0.8 stop bar;
  full-FT is the production arm. Throughput: frozen ~65-69k tok/s (bsz 64),
  full-FT ~25.5k tok/s (bsz 6, memory-bound; 1.5B-token epoch would be
  ~16h). PLAN: production full-FT starts NOW on already-consolidated
  shards (~460M+ tokens ≈ 5h, plateau early-stop active, eval cadence
  fixed — script defaults would have added ~17h of eval), bsz 6 × accum 8;
  corpus_stats.json merged from partial stats first. Frozen-encoder
  baseline + verify_closed_form (--attn eager!) + Exp-B variants will run
  on pod A after its scoring ends (it holds shards 320-330 locally).
  G2 call expected ~9-10 AM on plateau, else ~11 AM.

- ~5:45 AM: trainer pod LIVE: stage7-train 0te256xap9vakv
  (root@31.24.80.36 -p 10617, H100, 750GB, $2.99/hr) — Qwen3-0.6B-Base +
  gemma tokenizer + all 43 raw ClimbMix parquets (3.7GB, much smaller than
  feared) + code + probe_set staged; imports pass. Progressive
  consolidation pull loop running (~42 MB/s aggregate; RunPod same-subnet
  isolation discovered: C/D unreachable directly from trainer → relayed
  through A/B at ~7 MB/s each; consolidation ETA ~1-1.5h after scoring
  ends). ⚠ trainer 750GB is container overlay (no volume): a pod STOP
  loses the store — never stop it; HF archival after consolidation is the
  durable copy. Exp-A PILOT launched on trainer (head-only 500 steps + FT
  100 steps on already-consolidated shards) to de-risk real-data path +
  pin production batch size before the main run.

- ~5:05 AM: prep wave done + committed (29015df). (1) verify_closed_form.py
  smoke-green (incl. a test that the old raw-Gram bug is caught; runtime
  gram-consistency check ships in the script; NOTE: run it with --attn
  eager — its CLI default sdpa does NOT match how shards were scored).
  (2) Nanochat Phase-4 prep complete (out/nanochat_prep.md + code/
  nanochat_patch/): no-VE baseline = HF oracle_baseline_noVE_d24_fp8
  (CORE 0.2711, 8352 steps = 8.758B tokens, seed 1337, NO_VALUE_EMBEDS=1
  → runs/oracle_runs/no_value_embeds.sh); injected run ≈ 3h/$75 on
  8×H100. KEY: nanochat corpus = karpathy/climbmix-400b-shuffle shards
  ~0-185 — SAME repack as our scoring (shards 320-362): disjoint shards,
  same distribution, no leakage. Coords design: per-doc content-hash
  keyed int8 store riding through nanochat's best-fit packing (~35% crop
  means flat position-keyed memmaps can't work); injection after 8
  completed blocks (h[7], depth 0.33 ≈ gemma 8/26); r=14 (6 cyclic
  families → 2-D rings, continents PCA-2D), layer-8 prediction block.
  (3) Fixture-builder raw-Gram stale copies fixed; both smoke suites
  re-run green. LONG POLE: coords precompute needs the trained encoder →
  ~21-38 GPU-h ($63-113); nanochat launch realistically afternoon, not
  11 AM (SPEC anticipated this; leave-running-with-monitor plan applies).
  Budget projection: ~$55 spent/accruing + ExpA/B ~$20 + verify ~$2 +
  coords ~$63-113 + injected run ~$75 ≈ $230-280 total, within $400.
  Plan at scoring completion: verify_closed_form on pod A (eager) + G1
  checks on trainer + tear down B/C/D once shards confirmed on trainer +
  HF archival from trainer (background) + Exp A MLP-only starts.

- ~4:05 AM: PHASE 1 FLEET LIVE. 4× H100 ($2.99/hr each) scoring in
  parallel, eager attn, bs16 (sweep: bs16 41.5k > bs32 40.9k > bs64 39.0k
  tok/s): pod A zoti3owrvnp6x4 (103.207.149.109:16306) shards 320-330;
  pod B 750vnsgejqss6j (103.207.149.154:13809) shards 331-341 @41.4k;
  pod C ar3343of48nno8 (31.24.80.44:13432) shards 342-352 @44.2k;
  pod D kvgdwwvlkteunw (31.24.80.36:16374) shards 353-362 @44.3k.
  quant.json calibrated on 10M tokens of shard 320, sanity-checked,
  distributed fleet-wide + saved locally in out/. ETA ~3h/pod → scoring
  done ~7 AM; ~13 GPU-h ≈ $40. Heartbeats verified advancing on all pods.

- ~3:15 AM: BENCHMARK GATE decided. sdpa FAILS probe-score parity (worst
  p99/std 0.057 padded / 0.077 packed, threshold 0.05; root cause confirmed
  in transformers source: sdpa path silently drops gemma-2's
  attn_logit_softcapping=50) → **scoring runs EAGER**. Throughput is saved
  by batching, not attn impl: naive 1-doc-per-2048-row padding wastes ~70%
  (docs avg ~543 tok) — bench showed packed-eager 40.3k tok/s vs padded
  16k. Packing has correctness costs (cross-doc attention bleed + no
  per-doc BOS, deviating from the probes' training convention), so the
  decision: **eager + score_corpus.py's sorted-by-length dynamic-padded
  batches** (per-doc BOS, no bleed) — expected ~30-40k tok/s since sorted
  batches have single-digit pad waste; validate actual tok/s on pod A
  before fleet launch. Corpus: FULL 2.0B tokens = ClimbMix shards 320-362
  (43 shards × ~46.8M tok; 6223 unused shards ≥320 verified, format matches
  nat_common loader). Projected scoring ~$50-65 at ~$3/hr. Note: encoder
  training pod will need ~750GB volume (score store = 220 B/token × 2B =
  440GB — K=54 incl. the +K dom columns; bigger than SPEC's 280GB
  estimate). Pod A (bench) left running becomes scoring pod A: id
  zoti3owrvnp6x4, $2.99/hr, gemma + probes already staged. All stage7 code
  + Phase-0 outputs committed + pushed (d63c9d6).

- 2026-07-08: `code/train_encoder.py` DONE (Phase 2 Exp A + Phase 3 Exp B,
  all 3 modes). Smoke test `code/test_train_encoder.py`: 11/11 PASS (tiny
  random Qwen2Config encoder, real gemma-2-2b + Qwen3-0.6B-Base tokenizers,
  synthetic score-store fixture matching the DESIGN.md schema exactly —
  verified against the REAL `out/probe_set.json`/`probe_set_arrays.npz`
  Phase-0 produced too, K=54/layers=[6,8,14]/ablation_layer=8 load and
  `v* = D·G_inv·(s-t)` compute cleanly). Picks up the real `align.py` (now
  present; `_align_fallback.py` kept as a dead-but-harmless fallback, API-
  compatible, unused since `align.py` imports first). Qwen3-0.6B-Base is
  NOT gated — tokenizer+config load fine offline from cache.
  **Cross-checked the gemma tokenization convention against the real
  score_corpus.py (written by a parallel agent) and fixed a mismatch**:
  score_corpus.py uses `tok(text, add_special_tokens=False)["input_ids"]`
  sliced (not tokenizer-truncated) to `MAX_DOC_TOKENS=2048` — BOS is never
  in the stored ids (manually prepended + its hidden row dropped around the
  forward pass). train_encoder.py's `process_doc` was updated to match
  exactly (was previously assuming `add_special_tokens=True`); the smoke
  test's fixture builder was updated the same way and re-verified green.
  `MIN_DOC_TOKENS=64` already matched. The token-id reproduction hard-assert
  (first `--assert-first-n-docs` train docs) will still hard-fail loudly if
  any further drift exists (e.g. a different gemma tokenizer revision via
  `--gemma-model`), by design.

- 2026-07-08: Phase 0 DONE, gate **G0 = GO**. `code/select_probes.py` (CPU,
  local, $0) scored all 64 concepts x 12 gemma layers x 4 arms
  (ridge/dom/lda/logistic) on the natural TEST half (example-level max-pool
  AUROC, label ymax>=0.34; token Spearman for audit). Chosen layers
  **[6, 8, 14]** (mean best-arm AUROC 0.9629/0.9645/0.9601; data-driven —
  differs from SPEC's speculative {8,12,16} guess but satisfies the
  causal-band + non-adjacent-spread tie-break rules; all layer scores were
  within ~0.005 of each other, i.e. nearly flat, since probe quality is
  uniformly high). **K=54 survivors** (AUROC>=0.90 at all 3 layers
  simultaneously) out of 64 — no relax fallback needed (>=20 threshold).
  Dropped: all 5 intensity scalars (costliness, duration, harmfulness,
  lovingness, physical_size — expected per SPEC), both location_type
  concepts (indoors, outdoors — new, family fully eliminated), and 3
  color_wheel blends (blue-violet, red-violet, yellow-orange). Note:
  blue-violet, yellow-orange and outdoors were Stage-6 "deploy" tier but
  fail Phase-0's stricter all-3-layers-simultaneously bar — not a
  regression, just a stricter joint constraint. **Ablation layer = 8**
  (mode of causal_cards `e5_salient_layer_corrected` over the 54 survivors;
  histogram {6:1, 8:22, 10:4, 12:17, 14:4, 16:4, 18:2} — matches SPEC's
  expected L8-or-L12). Axis-convention verification (natscores/probes W,b,
  lambda, class-index, layer-index reading) cross-checked against
  stage6_1's independently-computed `dose_calib.json` for 2 concepts:
  **exact match, 0.00e+00 relative error** on both t and s95 (no raw gemma
  hidden states are available locally to fully re-derive from scratch —
  this is the strongest available loop closure; see
  `select_probes.py::verify_axes` docstring for the reasoning). Outputs:
  `out/probe_set.json`, `out/probe_set_arrays.npz`, `out/selection_table.md`
  (full 64-concept x 3-layer x 4-arm audit table). Not committed to git.

- ~2:15 AM: align.py DONE (42/42 tests pass, real gemma+qwen tokenizers).
  Measured gemma→qwen crossing rate 7.08% < 10% threshold → prefix-state
  mode confirmed as primary bridge, no boundary fallback needed. Other 4
  agents (Phase 0, bench pod, score_corpus, train_encoder) still running.

- 2026-07-08 ~2:00 AM: post-compaction instance online. User Q&A before
  sleep resolved two decisions, spec updated: (1) marginal G2 (R² 0.4–0.6)
  → LAUNCH nanochat anyway with caveat; (2) BOTH baselines (VE and no-VE)
  already exist on HF — train NO baseline tonight, one injected no-VE run
  only, matched exactly to the existing no-VE baseline's recipe/token
  budget (no shortening; if unaffordable, hold launch for user).
  Beginning execution: Phase 0 (sonnet subagent) + throughput/parity
  benchmark pod.

- 2026-07-08 1:50 AM: SPEC.md written by pre-compaction instance (amendments
  to the user's proposal documented inline). Nothing executed yet. Next
  instance: read SPEC.md top-to-bottom, then start Phase 0 + the throughput
  benchmark immediately. Budget ≤$400 target / $550 hard; delegate to
  sonnet-model subagents aggressively; user wakes 11 AM.
