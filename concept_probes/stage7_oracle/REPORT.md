# Stage 7 — Oracle-Encoding Injection: Results Report

_Living, human-readable narrative of the overnight run. Operational blow-by-blow
lives in `STATE.md`; this file is the "what happened and what does it mean" view.
Last updated 2026-07-08 (midday, Exp B final / G3), by the analysis agent._

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
| **G1** | corpus-scoring sanity before training | distributions sane | **FAIL -> PASS** | label-permutation bug found + fixed via metadata (no rescore); see Incidents |
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

---

## Links

**wandb** — project https://wandb.ai/kaushikreddyxyz-/stage7-oracle
- expA-fullft-prod: https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/u5hkgx5g
- expA-frozen-baseline: https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/fsrsjsmz
- expB-fixed: https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/7tnkw9jt
- expB-learn: https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/jt2phjcv (retro-logged by the auto-watcher on completion)

**HuggingFace**
- Encoder + checkpoints: https://huggingface.co/kaushikreddyxyz/stage7-oracle-encoder
  (`best.pt` deployed; `frozen-baseline/`, `expB-fixed/`, `expB-learn/` subdirs + per-run `metrics.jsonl`)
- Probe-score dataset: https://huggingface.co/datasets/kaushikreddyxyz/concept-probes-corpus-scores
- no-VE nanochat baseline (tonight's match target): https://huggingface.co/kaushikreddyxyz/oracle_baseline_noVE_d24_fp8 (CORE 0.2711, val bpb 0.7091)

**Key `out/` files**
- `out/G1_REPORT.md` — the permutation-bug root cause
- `out/PERMUTATION_FIX.md` — metadata-only remediation (no rescore, no retrain)
- `out/g2_retention.json` — natural-eval AUROC retention (the audited G2 gate)
- `out/verify_report.json` — closed-form verifier output (pod A)
- `out/expB_final_analysis.json` — Exp B final analysis bundle (G3): v* R2, subspace
  principal angles + random control, per-token cos(v̂,v*)/magnitude slices, residual estimate
- `out/nanochat_prep.md` — injected-run launch checklist (incl. section 5b wandb wiring)
- `code/wandb_retrolog.py` — replays a `metrics.jsonl` into a wandb run

---

## Incidents caught (and handled)

- **Label-permutation bug (G1).** `select_probes.py` silently permuted 53/54
  concept labels across 162/216 store columns (only `september` landed right by
  coincidence). Caught by the G1 corpus-scoring sanity check before it could
  corrupt conclusions; fixed with explicit block-order metadata keys in
  `probe_set.json` so every consumer re-attaches names correctly — **no
  rescoring and no retraining needed** (score bytes were correct; only the
  name-to-column map was wrong).
- **sdpa attention parity.** transformers' default `sdpa` attention silently
  drops gemma-2's logit soft-capping, failing probe-score parity vs `eager`;
  all corpus scoring pinned to **eager** attention to match how probes were fit.
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
  agent. wandb wiring is prepared in `out/nanochat_prep.md section 5b`: install+auth
  wandb on the launch node, `sed` the hardcoded `project="nanochat"` -> `"stage7-oracle"`,
  and set `--run=nanochat-d24-injected-noVE`. It will then log `train/tok_per_sec`,
  `val/bpb`, `core_metric`, and full config live to the same project. **Not launched
  by this agent.**
