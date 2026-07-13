# Oracle encoders (Qwen3-0.6B) — Results Report

_Split from concept_probes/stage7_oracle/REPORT.md 2026-07-13 (encoder/oracle-training sections; corpus-scoring / attribution / climbmix-audit sections live in `../attribution/REPORT.md`). Living, human-readable narrative; operational blow-by-blow lived in stage7_oracle/STATE.md (distilled into the attribution/oracles READMEs 2026-07-13; full text in git history). Last updated 2026-07-10 (evening, L6/L8 continuation runs), by the continuation agent._

---

## 2026-07-10 — L6/L8 per-layer encoder CONTINUATION runs (cont1): done, plateaued

Per user handoff, the L6 and L8 per-layer oracles (NOT L14) were warm-started
from their overnight best checkpoints and trained on the shards the overnight
runs never reached (338..321, 355, 356..362; L6 also re-queued the 80%-consumed
339). Val stayed {353,354}, so every number below is directly comparable.
New checkpoints live at `layerXX/cont1/` on
[oracle-encoders](https://huggingface.co/kaushikreddyxyz/oracle-encoders);
the originals are untouched. cont1 `best_full.pt` carries full Muon+AdamW
state — exact resume is possible, unlike the stripped originals.

| layer | original (11h wall-clocked) | cont1 (plateau-stopped) | Δ | total tokens |
|---|---|---|---|---|
| L6 | R² 0.8331 / ρ 0.898 @ 690M | **R² 0.8368 / ρ 0.900** @ 854M | +0.0037 | 854M |
| L8 | R² 0.7965 / ρ 0.874 @ 701M | **R² 0.8002 / ρ 0.877** @ 865M | +0.0037 | 865M |

**The headline finding is negative-ish:** the overnight runs' final slope
(~+0.001–0.002/eval) did NOT extrapolate. Both continuations flattened within
~150M further tokens and were stopped by the pre-registered plateau rule
(Δ median R² < 0.005 over trailing 150M tokens; L6 Δ=0.0027, L8 Δ=0.0033).
The full-epoch budget (1.9B tokens) was therefore not spent: actual cost
~$17 of a projected ~$120. Both encoders are modestly but genuinely better,
and R² ≈ 0.84/0.80 now looks like the practical per-layer ceiling for this
recipe (consistent with the earlier signal-vs-noise analysis: much of the
residual is probe noise the encoder correctly refuses to fit).

Ops notes: warm restart used a 150-step LR re-warmup to the original cosine's
resume-point value then cosine to the 0.1 floor at the projected epoch end —
no warm-restart dip was observed (first evals landed slightly ABOVE the
resume baselines). Runs: wandb
[5voreqcz](https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/5voreqcz) (L6) /
[mf6v2nyo](https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/mf6v2nyo) (L8).
Both pods uploaded, byte-verified, and self-terminated cleanly.

**One-paragraph summary.** We trained a Qwen3-0.6B encoder to predict gemma-2-2b
concept-probe scores from raw ClimbMix text, so those predictions can be turned
into structured oracle coords and injected into a nanochat d24 pretraining run.
The encoder **passed gate G2** (heldout median R2 **0.6371** >= 0.6, natural-eval
AUROC retention **0.966** >= 0.90) and is the deployed checkpoint. A frozen-encoder
control confirms the signal comes from **fine-tuning** (0.6371 vs 0.1823, a 3.5x
gap), not from reading out pre-existing features. Exp B (the structured v* head)
**passed gate G3** in its encoder-learning arm (v* R2 **0.6111** >= 0.5; the
frozen-encoder arm fails at 0.2716) and its learned down-projection recovers the
true repair subspace almost exactly (median principal-angle cosine 0.998 vs 0.125
random control). The injected nanochat run launches tonight; its wandb wiring is
prepared (not launched).

---

## Gate status

| gate | question | bar | result | numbers |
|---|---|---|---|---|
| **G0** | enough concepts, table reviewed | >=20 concepts | **PASS** | 54 concepts x 3 layers [6,8,14] = 162 targets |
| **G1** | corpus-scoring sanity before training | distributions sane | **FAIL -> PASS** | label-permutation bug found + fixed via metadata (no rescore); see Incidents (attribution side: `../attribution/REPORT.md`) |
| **verification** | closed-form encoder/coord checks (pod A) | all checks pass | **PASS** | score restoration 2.1e-4; identity p99 5e-7; quant p50 3.6%; v*-crosscheck exact 0.0 |
| **G2** | heldout median per-probe R2 (GO for nanochat) | >= 0.60 | **GO** | **R2 0.6371**; retention ratio 0.966 (raw 0.9836); all 7 families >= 0.90 |
| **G3** | Exp B: v* heldout R2 + direction recovery | v* R2 >= 0.5 | **learn PASS / fixed FAIL** | expB-learn v* R2 **0.6111** (PASS); expB-fixed 0.2716 (FAIL); subspace recovery median cos **0.998** (random control 0.125) — see "Exp B final analysis (G3)" |
| **G4** | nanochat loss curve sane vs baseline | <5% bpb divergence @2k steps | **pending** | injected run launches tonight |

---

## Experiment results so far

All numbers are heldout **median R2 over the 162 probe targets**, same data and
splits across arms. Live curves: **https://wandb.ai/kaushikreddyxyz-/stage7-oracle**

| run | arm | encoder | head | heldout median R2 | wandb |
|---|---|---|---|---|---|
| **expA-fullft-prod** | full fine-tune (**deployed**) | trained | linear | **0.6371** | [u5hkgx5g](https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/u5hkgx5g) |
| expA-frozen-baseline | MLP-only readout control | frozen base | linear | 0.1823 | [fsrsjsmz](https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/fsrsjsmz) |
| expB-fixed | v* (coord) head | frozen Exp-A | v* (fixed D) | 0.3440 (v* R2 0.2716) | [7tnkw9jt](https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/7tnkw9jt) |
| **expB-learn** | v* (coord) head | frozen Exp-A | v* (learned D) | **v* R2 0.6111** | [jt2phjcv](https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/jt2phjcv) |

### Exp A — full fine-tune vs frozen (the headline)
Full fine-tune reaches **0.6371** (early-stopped at step 5600/6800); the
frozen-encoder MLP-only control saturates at **0.1823** — a **3.5x gap**. The
oracle signal is *learned into the encoder*, not read out of pre-existing Qwen
features. The one place the frozen model does well is `continents` (~0.45),
i.e. the base model already partially encodes geography; everything else needs
fine-tuning. This is the clean control the SPEC asked for, and it lands the way
we want: the encoder is doing real work.

### Exp B — structured v* head: final analysis (G3)

**G3 verdict: expB-fixed FAIL (v\* R2 0.2716 < 0.5); expB-learn PASS (v\* R2
0.6111 >= 0.5).** Both arms share the same frozen Exp-A fine-tuned encoder;
the difference is the decoder: expB-fixed predicts coords y through the
closed-form fixed D, expB-learn trains a free `down: K→2304` with MSE directly
on v\*. All numbers below are from the final `best.pt` of each arm, evaluated
on val shards 353/354 (~5.0M tokens); full machine-readable dump in
`out/expB_final_analysis.json`.

**Final v\* R2 (heldout).**

| arm | v\* R2 (aggregate) | v\* per-dim median R2 | G3 (>= 0.5) |
|---|---|---|---|
| expB-fixed | 0.2716 | 0.059 | **FAIL** |
| **expB-learn** | **0.6111** | **0.469** | **PASS** |

expB-learn plateaus cleanly (0.607 by step 700, 0.6111 at 1282). Its
per-*probe* (y-space) median R2 is meaningless by construction (-31.7): with a
learnable decoder the internal K-dim coords are only identified up to an
invertible K×K mixing, so y-space R2 is not a valid metric for this arm —
only v\*-space numbers count.

**Direction recovery — the per-column cosine was a mis-specified metric.**
The per-eval `down_cosine` log (cosine of each learned down column vs the same-
named column of the true D_raw, where D_raw[:,c] = nat_std ⊙ W_dom_abl[c])
ends at median **0.030** (0/54 >= 0.7). This does NOT mean the decoder failed
to recover the repair directions: since loss is on v\* = down(up(h)), the
factorization is rotation-unidentifiable — for any invertible M, (down·M,
M⁻¹·up) gives identical v\*, so individual columns need not align. The
identified object is **span(down)**, checked with principal angles:

| span comparison (54-dim in 2304-d) | median cos | min cos | # >= 0.7 |
|---|---|---|---|
| **span(learned down) vs span(D_raw)** | **0.9983** | 0.079 | **51/54** |
| random control (54 gaussian dirs) vs span(D_raw) | 0.125 | 0.0006 | 0/54 |

The learned decoder recovers the true repair subspace almost exactly —
51 of 54 principal-angle cosines >= 0.7 (most > 0.99), vs 0/54 and a 0.125
median for the matched-random yardstick. Only ~3 low-variance directions of
the span are missed.

**Per-token repair quality: cos(v̂, v\*).** Distribution over all 5.0M val
tokens, and restricted to the top decile of ‖v\*‖ (where repair matters;
near-zero v\* makes cosine noise):

| arm / slice | mean | median | p10 | p90 |
|---|---|---|---|---|
| learn, overall | 0.787 | 0.806 | 0.673 | 0.883 |
| learn, top-decile ‖v\*‖ | 0.782 | 0.801 | 0.662 | 0.882 |
| fixed, overall | 0.647 | 0.675 | 0.455 | 0.804 |
| fixed, top-decile ‖v\*‖ | 0.657 | 0.679 | 0.480 | 0.804 |

Magnitude ratio ‖v̂‖/‖v\*‖ (same slices): learn 0.798 mean overall, **0.686 on
the top decile** — the MSE-typical shrinkage concentrates exactly where repairs
are large; fixed is 1.000 overall / 0.848 top-decile (better calibrated in
scale but much worse in direction). If injection-time magnitude matters, a
~1.2-1.45x rescale of v̂ on large-repair tokens is worth considering.

**Residual-level estimate.** cos(h_abl + v̂, h_clean) ≈ 1/√(1 + E‖v̂−v\*‖²/E‖h‖²),
assuming h_clean = h_abl + v\* (repair exact by construction) and error
e = v̂−v\* uncorrelated with h_clean; E‖h‖² = Σ_d(nat_mean² + nat_std²) = 12312
at layer 8. With E‖e‖² = 155.2 (learn) / 290.7 (fixed):
**learn ≈ 0.9938, fixed ≈ 0.9884**. Both look high only because ‖v\*‖ ≪ ‖h‖
(top-decile ‖v\*‖ threshold 25.6 vs ‖h‖ ~ 111); the discriminating metrics are
v\* R2 and cos(v̂, v\*) above.

### Does the oracle predict signal or noise? (user hunch, tested)

Hunch: the plateau at R² ≈ 0.6 is the encoder predicting the *signal* component
of the probe scores while the probes' own noise is unpredictable — i.e. we're
at a noise ceiling, not an encoder-capacity ceiling. Verdict: **partially
supported**.

For it:
- The R²-vs-retention mismatch: the encoder fits only 63% of probe-score
  variance yet retains 96.6% of chance-corrected judge AUROC. If the missing
  37% were concept-relevant signal, retention couldn't stay that high — most
  of the residual is judge-irrelevant variance.
- Per-concept encoder R² is **uncorrelated** with judge retention (Pearson
  −0.03, Spearman −0.02) and with probe quality (+0.01): how well the oracle
  fits a probe's scores tells you nothing about how well it detects the
  concept. That decoupling is what you'd expect if R² differences reflect
  differing noise fractions, not differing signal capture.
- Encoder R² is tightly clustered (0.52–0.78 across all 54 concepts) despite
  wildly varying probe quality — it behaves like a floor property of the
  regression *target*, not of encoder capacity.

Against a *pure* noise ceiling:
- On the LLM-judge natural eval the oracle loses to the gemma probes on
  **49/54 concepts** (median AUROC delta −0.016, mean −0.026; wins on 5, best
  blue-green +0.016, worst october/yellow-green/waning_crescent ≈ −0.09). If
  the residual were pure noise and the oracle recovered the signal, it should
  beat the noisy probe on an independent eval about as often as not — a
  systematic deficit on 91% of concepts means some real signal is missed.
- Same-layer probe-arm agreement (ridge vs alternative arm on identical
  activations) is r² ≈ 0.90, well above the encoder's 0.64. If 0.6 were the
  true signal content, two independent readouts of the same layer couldn't
  agree at 0.90.

Net: a meaningful fraction of the missing 0.36 is probe noise the encoder
correctly refuses to fit (hence retention holding at 0.966), but the 49/54
judge deficit and the same-layer agreement gap say the encoder is also
short of the achievable ceiling. Numbers derive from `out/g2_retention.json`
per-concept `enc_auroc` vs `gemma_auroc` (× `out/retro_metrics/`
`expA_prod_metrics.jsonl` for the correlations) and the Exp A/B analyses
above. (An earlier revision of this section claimed corr(R², retention)
= −0.42; recomputation gives −0.03 — corrected.)

---

## Links

**wandb** — project https://wandb.ai/kaushikreddyxyz-/stage7-oracle
- expA-fullft-prod: https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/u5hkgx5g
- expA-frozen-baseline: https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/fsrsjsmz
- expB-fixed: https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/7tnkw9jt
- expB-learn: https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/jt2phjcv (retro-logged by the auto-watcher on completion)

**HuggingFace**
- Encoder + checkpoints: https://huggingface.co/kaushikreddyxyz/oracle-encoder
  (`best.pt` deployed; `frozen-baseline/`, `expB-fixed/`, `expB-learn/` subdirs + per-run `metrics.jsonl`)
- Probe-score dataset: https://huggingface.co/datasets/kaushikreddyxyz/concept-probes-corpus-scores
- no-VE nanochat baseline (tonight's match target): https://huggingface.co/kaushikreddyxyz/oracle_baseline_noVE_d24_fp8 (CORE 0.2711, val bpb 0.7091)

**Key files** (paths updated for the 2026-07-13 restructure)
- G1 permutation-bug root cause: `../attribution/out/g1_*.json` + `../attribution/README.md` (G1_REPORT.md distilled 2026-07-13, git history)
- Permutation remediation (metadata-only, no rescore/retrain): `../attribution/README.md` permutation note (PERMUTATION_FIX.md distilled 2026-07-13, git history)
- `out/g2_retention.json` — natural-eval AUROC retention (the audited G2 gate)
- `../attribution/out/verify_report.json` — closed-form verifier output (pod A)
- `out/expB_final_analysis.json` — Exp B final analysis bundle (G3): v* R2, subspace
  principal angles + random control, per-token cos(v̂,v*)/magnitude slices, residual estimate
- Injected-run launch guide: `../nanochat/nanochat/injection/README.md` (supersedes nanochat_prep.md, distilled 2026-07-13)
- `wandb_retrolog.py` — replays a `metrics.jsonl` into a wandb run

---

## Incidents caught (and handled)

_(The corpus-scoring incidents — the label-permutation bug (G1) and sdpa
attention parity — live in `../attribution/REPORT.md`.)_

- **batch-1 drain bug.** The serial coords precompute path had a per-doc
  `drain()` bug that forced batch-1 forwards (killing throughput and muddying
  the determinism A/B); fixed, then the path was replaced with length-bucketed
  cross-doc batching for the coords sweep.

---

## Still running / pending (with ETAs)

- **expB-learn** — **DONE** (step 1282, 2026-07-08 ~11:03 UTC). `best.pt` +
  `metrics.jsonl` auto-pushed to `expB-learn/` on HF by the pod watcher;
  retro-logged to wandb ([jt2phjcv](https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/jt2phjcv))
  by the local auto-watcher. Final analysis in the G3 section above /
  `out/expB_final_analysis.json`. The trainer pod self-cleanup daemon is armed
  and will self-terminate once the coords done-markers land.
- **coords sweep** (6x H100, coords1-6) — fast-forward path approved (~2x speedup),
  **ETA ~done 7-9 PM**; not a training run, so no wandb.
- **nanochat d24 injected run** (no-VE match) — launches **tonight** by a separate
  agent. wandb wiring guidance now lives in `../nanochat/nanochat/injection/README.md`
  section 5b: install+auth
  wandb on the launch node, `sed` the hardcoded `project="nanochat"` -> `"stage7-oracle"`,
  and set `--run=nanochat-d24-injected-noVE`. It will then log `train/tok_per_sec`,
  `val/bpb`, `core_metric`, and full config live to the same project. **Not launched
  by this agent.**

---

## 2026-07-10: Per-layer oracles (corrected design) — TRAINED, all beat the 0.6371 baseline

Three independent oracles (Qwen3-0.6B-Base full-FT + MLP 1024→4096→GELU→54), each predicting
ONE layer's 54 detection-probe scores (targets physically sliced from the stacked [n,3,54]
store). Objective: expA MSE on corpus-standardized scores. Optimizer: Muon (5e-3, NS5) on 2D
matrices + AdamW (1e-4) else. Train shards 320-352, val 353/354. All runs ended at the 11h
wall-clock cap, still improving (never plateaued at Δ<0.005/150M tok).

| oracle | best heldout median R² | Spearman ρ | tokens | wandb |
|---|---|---|---|---|
| L6  | **0.8331** | 0.898 | 690M | stage7-oracle/bga5ozov |
| L8  | **0.7965** | 0.874 | 701M | stage7-oracle/jouy3hop |
| L14 | **0.7217** | 0.829 | 614M | stage7-oracle/44zfc7qf |

Per-family medians uniform within each layer (L6: 0.823-0.873). Checkpoints:
`kaushikreddyxyz/oracle-encoders` layer06|08|14/best_stripped.pt (+ metrics.jsonl).
Trainer: `train_oracle_perlayer.py`. Cost ≈ $100 (3× H100 × ~11h).

**Association figure** (`out/figures/oracle_perlayer_assoc.png`, matrices in companion npz):
54×54 Spearman (oracle max-pooled prediction × gemma probe activation, stage-6 natural-eval
TEST split), one panel per oracle at its own layer:

| panel | median diag | min diag | median off-diag | p95 off-diag |
|---|---|---|---|---|
| L6  | 0.902 | 0.824 | 0.252 | 0.601 |
| L8  | 0.888 | 0.789 | 0.208 | 0.552 |
| L14 | 0.877 | 0.772 | 0.239 | 0.612 |

Strong on-target diagonal with family-block off-diagonal structure (expected within-family
correlation); no permuted/broken axes.

## Geometry side-study (probe/activation structure; from the 2026-07-10 climbmix deep audit)

_(Moved here from the climbmix deep-audit section, whose other bullets live in
`../attribution/REPORT.md`.)_

- Geometry side-study (shard 10, gemma-2-2b re-extraction, sanity-gated ±1 int8 step; exhaustive
  centroid-PC-pair scan, 318 tests, exact permutation p; final classification 2026-07-13 per
  Kaushik's visual assessment, stats preserved alongside):
  POSITIVE — directions: PERFECT compass octagon in act PC3×PC5 @L8 (r=+1.00, p=4e-4; echoed
  L6/L14) + weight octagon PC3×PC4 (r=+1.00); PC1-2 hold cardinal/intercardinal + antipodal-pair
  structure. weekdays: Mon→Sun ring in act PC1×PC2 at all layers (|r|=0.78, p=0.022) + perfect
  weight heptagon in wt PC1×PC3 (r=−1.00, p=0.0028 @L8). seasons: judged positive on structural
  consistency across layers (caveat: k=4 makes the ring order-statistic vacuous, p=1.0 by
  construction — rests on structure, not the permutation test).
  NEGATIVE/INCONCLUSIVE — months: "ordered but sloppy". Calendar ordering survives even the
  exhaustive all-55-plane family-wise max-statistic test (act best plane PC2×PC3 at EVERY layer,
  fw p=0.007/0.013/0.023 @L6/L8/L14; wt best PC1×PC2 fw p=0.015/0.0017 @L6/L8, fades @L14 —
  vs moon_phases fw p≥0.31 under the identical test), but no plane looks clean: the loop
  self-crosses everywhere, dragged by winter-month outliers. Judged inconclusive pending
  cleaner evidence (PC3×PC4 calendar-unordered; all-plane mega-grids in report §8.2). moon_phases and colour_wheel DISPROVED by exhaustive
  ALL-plane family-wise max-statistic permutation test (best |r|=0.61 anywhere fw p≥0.31;
  colour-wheel nominal hits contradict across layers). Probe-vector structure (months, strength
  held constant via unit-norm + mean-direction removal): annual+semiannual Fourier harmonics
  carry ~55% of centroid identity variance (probes dilute to ~42% via estimation noise —
  cross-layer cos: centroids 0.74–0.91, probes 0.42–0.61); harmonic-free residual is REAL
  (cross-layer cos 0.70–0.89), so months ≈ shared dir + 4-d harmonic core + genuine ~7-d
  idiosyncratic tail. Artifacts in session scratch `probe_geometry/` (acts_f16.npz +
  geometry.npz + pc_scan.json + pc_scan_full.json + gold_w/).
