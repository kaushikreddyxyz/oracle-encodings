# Stage 7 — Oracle-Encoding Injection: Results Report

_Living, human-readable narrative of the overnight run. Operational blow-by-blow
lives in `STATE.md`; this file is the "what happened and what does it mean" view.
Last updated 2026-07-08 (morning), by the observability agent._

**One-paragraph summary.** We trained a Qwen3-0.6B encoder to predict gemma-2-2b
concept-probe scores from raw ClimbMix text, so those predictions can be turned
into structured oracle coords and injected into a nanochat d24 pretraining run.
The encoder **passed gate G2** (heldout median R2 **0.6371** >= 0.6, natural-eval
AUROC retention **0.966** >= 0.90) and is the deployed checkpoint. A frozen-encoder
control confirms the signal comes from **fine-tuning** (0.6371 vs 0.1823, a 3.5x
gap), not from reading out pre-existing features. Exp B (the structured v* head)
is in progress. The injected nanochat run launches tonight; its wandb wiring is
prepared (not launched).

---

## Gate status

| gate | question | bar | result | numbers |
|---|---|---|---|---|
| **G0** | enough concepts, table reviewed | >=20 concepts | **PASS** | 54 concepts x 3 layers [6,8,14] = 162 targets |
| **G1** | corpus-scoring sanity before training | distributions sane | **FAIL -> PASS** | label-permutation bug found + fixed via metadata (no rescore); see Incidents |
| **verification** | closed-form encoder/coord checks (pod A) | all checks pass | **PASS** | score restoration 2.1e-4; identity p99 5e-7; quant p50 3.6%; v*-crosscheck exact 0.0 |
| **G2** | heldout median per-probe R2 (GO for nanochat) | >= 0.60 | **GO** | **R2 0.6371**; retention ratio 0.966 (raw 0.9836); all 7 families >= 0.90 |
| **G3** | Exp B: v* heldout R2 + direction recovery | v* R2 >= 0.5 | **in progress** | expB-fixed v* R2 0.2716 (below bar); expB-learn running |
| **G4** | nanochat loss curve sane vs baseline | <5% bpb divergence @2k steps | **pending** | injected run launches tonight |

---

## Experiment results so far

All numbers are heldout **median R2 over the 162 probe targets**, same data and
splits across arms. Live curves: **https://wandb.ai/kaushikreddyxyz-/stage7-oracle**

| run | arm | encoder | head | heldout median R2 | wandb |
|---|---|---|---|---|---|
| **expA-fullft-prod** | full fine-tune (**deployed**) | trained | linear | **0.6371** | [u5hkgx5g](https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/u5hkgx5g) |
| expA-frozen-baseline | MLP-only readout control | frozen base | linear | 0.1823 | [fsrsjsmz](https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/fsrsjsmz) |
| expB-fixed | v* (coord) head | frozen Exp-A | v* | 0.3440 | [7tnkw9jt](https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/7tnkw9jt) |
| expB-learn | v* (coord) head | fine-tuned | v* | _running_ | _(retro-log on completion)_ |

### Exp A — full fine-tune vs frozen (the headline)
Full fine-tune reaches **0.6371** (early-stopped at step 5600/6800); the
frozen-encoder MLP-only control saturates at **0.1823** — a **3.5x gap**. The
oracle signal is *learned into the encoder*, not read out of pre-existing Qwen
features. The one place the frozen model does well is `continents` (~0.45),
i.e. the base model already partially encodes geography; everything else needs
fine-tuning. This is the clean control the SPEC asked for, and it lands the way
we want: the encoder is doing real work.

### Exp B — structured v* head
Exp B replaces the 162-dim linear head with the structured v* (oracle-coord)
head that the injection actually consumes. The **frozen-encoder** v* arm
(expB-fixed) reaches per-probe median R2 **0.3440** but v* R2 only **0.2716**
(below the G3 >=0.5 bar) — expected, since the encoder is frozen. The
**encoder-learning** arm (expB-learn) is the one that can clear G3 and is still
training; verdict pending.

---

## Links

**wandb** — project https://wandb.ai/kaushikreddyxyz-/stage7-oracle
- expA-fullft-prod: https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/u5hkgx5g
- expA-frozen-baseline: https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/fsrsjsmz
- expB-fixed: https://wandb.ai/kaushikreddyxyz-/stage7-oracle/runs/7tnkw9jt

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

- **expB-learn** (trainer pod `/workspace/expB_learn`) — encoder-learning v* arm,
  1282 steps, at ~step 100 as of this writing (~3.7k tok/s), **ETA ~1-1.5h**.
  On completion a pod-side watcher (`/workspace/expB_learn_watch.sh`) auto-pushes
  `best.pt` + `metrics.jsonl` to `expB-learn/` on HF. **TODO on completion:**
  retro-log to wandb — pull its `metrics.jsonl` and run
  `code/wandb_retrolog.py --name expB-learn --project stage7-oracle`.
- **coords sweep** (6x H100, coords1-6) — fast-forward path approved (~2x speedup),
  **ETA ~done 7-9 PM**; not a training run, so no wandb.
- **nanochat d24 injected run** (no-VE match) — launches **tonight** by a separate
  agent. wandb wiring is prepared in `out/nanochat_prep.md section 5b`: install+auth
  wandb on the launch node, `sed` the hardcoded `project="nanochat"` -> `"stage7-oracle"`,
  and set `--run=nanochat-d24-injected-noVE`. It will then log `train/tok_per_sec`,
  `val/bpb`, `core_metric`, and full config live to the same project. **Not launched
  by this agent.**
