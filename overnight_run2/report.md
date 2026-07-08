# Concept-Probes v2 — Rectification Run Report

**Run:** 2026-06-30 (overnight, autonomous). Start 03:22Z, 9h deadline 12:22Z.
**Goal:** replace the incomplete prior run with (A) synthetic per-token concept datasets, (B) per-position linear probes at 12 layers, (C) full-corpus per-token attribution over ClimbMix shards 0–185 + 6542. Brief §4/§5 (geometry, GPU-speedup) out of scope.

---

## ⚠️ RELIABILITY & FAILURE MODES — adversarial diagnostics (2026-07-01, evidence-based)

An independent GPU diagnostic (1× H100, ~1.5 GPU-h) was run **to falsify** the claims in this report — probes applied via the production `ProbeBundle`/`scores_torch` path, on **120k ClimbMix corpus tokens + 88k held-out synthetic-TEST tokens (control)**, with random-direction control probes. **Where this section conflicts with claims below, THIS section wins.** Standardization stats located: train-fit `mean`/`std` are in each `weights/layer_L.npz`, fit on TRAIN tokens only; applied as `logit = ((h−mean)/std)@Wᵀ + b`.

**Exp 1 — "corpus saturates the probes via train/serve standardization shift" → MECHANISM REFUTED (phenomenon real).**
Dense firing is real (corpus fires **34.8%** of cells at σ>0.5). But the claimed cause is wrong on three controls: (1) corpus is **not** OOD under the train standardization — mean|z| ≈ **0.72–0.94** (in-distribution), and raw ‖h‖ is *smaller* on corpus than synthetic (L24: 282 vs 500); (2) the **synthetic-TEST control saturates just as much** — 19.5% at σ>0.5, **5.85% at σ>0.99999 (higher than corpus's 5.02%)**; (3) re-standardizing on corpus stats barely helps (34.8%→29.2%). **Correct diagnosis:** dense firing is a **`pos_weight`/threshold/base-rate property of the probes themselves** (rare-positive rows trained with large `pos_weight` push the boundary far below the precision-usable region), present on synthetic too. Corpus-recalibration of the standardization would NOT fix it. → The Phase-C "calibration mismatch" narrative below is a **misdiagnosis**.

**Exp 2 — "the relative-gate attribution is rank-based but meaningful" → REFUTED: corpus firings are frequency noise, not concepts.**
Decoding the top-0.2% corpus tokens by probe logit (the production gate) for 8 rows: `month::January`→`, small of is and oil` (**0%** own-trigger), `color_hue::Red`→`the`×47/240 (**0.4%**), `compass::North`→`and`×41 (**0%**), etc. **Own-trigger-word share is 0–4% across all 8 rows**; firings are dominated by high-frequency function words (`the`/`and`/`,`); sampled contexts (egg yolks, asteroids, lobster) are never about the concept. **The stored Phase-C attribution (28 shards) does not track concepts** — a fixed-0.2% gate on a concept that is genuinely rare in natural text guarantees the fired set is mostly non-concept tokens, and the within-corpus ranking is frequency-correlated. *(Flag: single shard/305 docs; "trigger-rank-when-present" not yet measured — a rare concept may simply be absent in-sample — but the fixed-fraction gate makes the stored firings unreliable as concept attribution either way.)*

**Exp 3 — "AUROC 0.954 ⇒ high-fidelity probes" → signal is real, precision is NOT usable.**
Synthetic-TEST, 47 binary rows @ best layer + stored threshold: AUROC median **0.954** (reproduced) but **precision @ operating point median 0.25** (recall 0.89, F1 0.38); only **6/47 rows reach precision ≥0.5**. **Control:** random-direction probes → AUROC **0.495** (chance); real probes beat control by **+0.449** → the ranking signal is genuine. But several rows collapse to ≈chance on held-out vocab (month::April 0.52, May 0.50, Sept 0.54; weekday::Sunday 0.57) → lexical detectors, contradicting "reads the concept beyond the surface token."

**Ranked problems (honest):**
1. **Corpus attribution is not concept-meaningful** (Exp 2) — the 28-shard attribution reflects token frequency, not concepts. Headline failure.
2. **The Phase-C "calibration mismatch" is a misdiagnosis** (Exp 1) — dense firing is intrinsic to the `pos_weight`-trained probes, not a distribution shift.
3. **High AUROC oversells deployability** (Exp 3) — median precision 0.25 at the shipped threshold.
4. **Synthetic→natural generalization gap** — real above-chance signal on templates (some genuinely semantic, e.g. season::Summer held-out 0.96) but junk on natural text; best layers often L42 where next-token features dominate.

**Net:** the probes carry real, above-chance concept signal **on the synthetic template distribution**, but at low precision even there, and they **do not transfer to meaningful per-token attribution on natural corpus text**. Treat the Phase-C dataset as **not validated as concept attribution** pending fixes (concept-prevalence-aware gating, precision-targeted thresholds, better probes/layer choices). Causal tests (steering/ablation) not yet run. Untested: trigger-rank-when-present, shuffled-label control probes, corpus-wide (only 1 shard sampled).

---

## Config & choices (and *why*)
- **Generator** `qwen/qwen-2.5-72b-instruct`, **judge** `meta-llama/llama-3.3-70b-instruct` — distinct >25B models on OpenRouter so the judge is a genuine independent filter (removes the self-agreement bias of same-model gen+judge). Both verified live; Qwen emits valid exact-substring JSON.
- **Probe target** `google/gemma-2-9b` (frozen), residuals at 12 layers `[1,2,6,10,15,19,24,28,33,37,41,42]` (first 2 / last 2 / 8 middle) — operator decision to cut storage 3.5× vs all-42 at the same forward cost.
- **Attribution** shards 0–185 (consumed pretraining) + val 6542 (~9B tokens); **sparse firings + per-(shard,concept,layer) gated-tail mean/std** — keeps it to ~100–500 GB on HF vs tens of TB dense.

## assumptions (autonomous decisions logged per the contract)
- Started a **fresh** run namespace `overnight_run2/` + new HF repos `concept-probes-v2-*`; the prior `overnight_run/` state is the old geometry run (wrong scope), not resumed.
- 12 layers = `[1,2,6,10,15,19,24,28,33,37,41,42]` (the handoff fixed an internal 42→12 / 0-273→0-185 inconsistency in favor of the latest contract).

## Phase A — datasets (COMPLETE, on HF: `kaushikreddyxyz/concept-probes-v2-datasets`)

**Method:** synthetic per-token-labeled generation per `probe_dataset_spec.md` — 60% templates + minimal pairs, 40% LLM free-gen (verbatim spans), 3 negative types, held-out-vocab split (unique periphrasis), gemma-2-9b `offset_mapping` per-token labels (pre=absent / span=value / post=masked). Generator `qwen/qwen-2.5-72b-instruct`, judge `meta-llama/llama-3.3-70b-instruct` (distinct >25B, OpenRouter), judge self-consistency 3 keep≥2.

**All 16 concepts ALL_PASS the §10 gates** (span-audit 50/50, token-target integrity 0 errors, minimal-pair consistent, 0 held-out-vocab leakage). Positives per concept:

| cyclic | npos | scalar | npos |
|---|---|---|---|
| color_hue | 6133 | (each scalar) | ~464–560 |
| month | 5889 | costliness 495, size 489 | |
| moon_phase | 4712 | europe 464, america 468, africa 499 | |
| weekday | 3565 | indoors 505, outdoors 560 | |
| season | 2382 | lovingness 540, harmfulness 532, duration 551 | |
| compass | 2181 | | |

**Scale decision:** 600 positives **per value** (cyclic concepts therefore have 600×n_values). Reduced from the planned 4000 because the user set a 9 AM target mid-run; later relaxed, but kept at 600 (still 5× the prior failed run's 120/class) to prioritize Phase-C shard coverage. Generation ran in ~31 min wall (10 concepts concurrent, 240 OpenRouter concurrency), ~$4 OpenRouter.

**Incident:** OpenRouter account hit $0 credits mid-run (402 on all calls); user topped up $100; resolved, run resumed. `n=3` judge batching tested but unsupported by the providers (Llama silently returns 1 choice, Qwen 400s).

**Deviations / caveats (logged):**
- Diffuse scalars (harmfulness/lovingness/duration) use single key-phrase spans, not the spec's multi-span (valid labels, less coverage).
- Scalar held-out-vocab split is empty — magnitudes have no single bannable trigger word, so scalar probes get no `lexical_gap` (validated by test Spearman/R² instead).
- Hard-negative share undershoots 40% (~31%) for some concepts (finite confusable lexicon); harmless for cyclic (sibling-value positives are the dominant in-family hard negatives).
- Judge agreement (pre-filter mean votes/3): scalars ~1.5–2.5; the kept set passed the ≥2/3 identify-match filter. With distinct gen/judge models this is a genuine independent filter.

## Phase B — probes (COMPLETE, on HF: `kaushikreddyxyz/concept-probes-v2-weights`)

**Method:** frozen gemma-2-9b, residuals at 12 layers `[1,2,6,10,15,19,24,28,33,37,41,42]`; one `W_L(57×3584)+b_L` per layer; each of the 57 rows an independent binary probe (cyclic = masked BCE with sibling-span hard negatives; scalar = soft-target BCE on [0,1]); per-row `pos_weight`; per-layer standardization (train-only); select on val, report on test + held-out-vocab; `--max-pre-per-seq 16`, 60 epochs. 51,408 sequences / 600k tokens. **Tokenizer skip-rate 0%** (label-time 5.10.1 and pod 5.12.1 tokenize gemma identically).

**Probe fidelity — AUROC strong, but see the ⚠️ Reliability section (precision @ operating point ≈ 0.25; corpus firings are frequency noise; several rows lexical). The AUROC-based framing below oversells deployability:**
- **47 binary (cyclic-value) rows: best-layer test AUROC ≥0.8 on ALL 47, ≥0.9 on 42/47, median 0.954.**
- 10 scalar rows: test Spearman 0.61–0.80 (lovingness 0.80, harmfulness 0.78, costliness 0.70, size 0.61). (Scalar R² often negative — probe is BCE- not MSE-calibrated; Spearman/bin-AUROC are the trustworthy scalar metrics.)
- **`lexical_gap` (test − held-out-vocab AUROC) median 0.198** — *on average* probes read beyond the surface token, BUT this is a median over a wide spread: some rows are genuinely semantic (season::Summer gap 0.008), while others collapse to ≈chance on held-out vocab (month::April 0.52, May 0.50 AUROC → purely lexical). "Probes read the concept beyond the surface token" is **false as a blanket claim** (see Reliability §Exp 3). Scalars have no held-out split → no lexical_gap.

| row | best L | test AUROC | lexical_gap |
|---|---|---|---|
| moon_phase::Full Moon | 28 | 0.988 | 0.223 |
| season::Winter | 41 | 0.975 | 0.137 |
| season::Summer | 24 | 0.970 | 0.008 |
| month::January | 24 | 0.963 | 0.168 |
| color_hue::Red | 42 | 0.956 | 0.375 |
| month::July | 10 | 0.932 | 0.146 |
| compass::North | 42 | 0.903 | 0.177 |
| weekday::Monday | 1 | 0.887 | 0.122 |

Best layers span depth (weekday at L1; months/moons mid L10–28; colors/compass/winter late L41–42), consistent with concepts emerging at different depths.

**Tuning note (logged, not acted on):** 60 epochs is overkill for linear probes (~25 would have halved the ~60-min fit); fit was ~5 min/layer.

## Phase C — attribution (IN PROGRESS / partial — on HF: `kaushikreddyxyz/concept-probes-v2-attribution`)

**Goal:** per-token attribution of the 57 concept rows × 12 layers over the ClimbMix pretraining corpus (shards 0–185 + val 6542), stored sparsely.

**Method (final, after solving the calibration problem below):**
- Frozen gemma-2-9b via **`AutoModel`** (not `AutoModelForCausalLM`) — drops the 256k-vocab `lm_head` (~33 GB), output is byte-identical (helper verified max-abs-diff 0.0), enabling `MAX_BATCH_DOCS=32` without OOM.
- **`eager` attention** (mandatory, fleet-wide). gemma-2's attention-logit softcap is preserved by `eager` and `flash_attention_2` but **dropped by `sdpa`** (residual diff ~137). flash-attn-2 was measured vs eager: real-token max-abs-diff ~1%, firing-set Jaccard 0.9775, speedup only **1.33×** — and one pod's torch-2.1 base can't build flash, so a mixed fleet would yield an inconsistent reference dataset. **Verdict: eager everywhere** (the only uniform option; the 1.33× wasn't worth the inconsistency).
- **Relative gate (the key fix):** keep each (concept-row, layer)'s top `top_frac=0.002` of corpus tokens by **logit score** (pre-sigmoid), with the per-(row,layer) threshold τ calibrated once on a fixed calib-shard (shard 48, 1.00M tokens, quantile 0.998). Gating in logit space sidesteps the sigmoid saturation described below.
- **Storage (sparse):** per shard, three parquet files — `firings` (gated token-level hits), `tokens` (token index/provenance), `tailstats` (per-(shard,row,layer) gated-tail Welford mean/std). ~0.7 GB/shard at 1.44 firings/token, vs ~25–50 GB/shard dense. Incremental per-shard upload → local purge → resume-skip. Single global `manifest.json` (gate block + per-(row,layer) τ + provenance) written by one designated pod.

**❌ MISDIAGNOSED FINDING (kept for the record; REFUTED by Reliability §Exp 1) — "probe train/serve calibration mismatch":**
*What was claimed at the time:* the probes were trained with per-layer standardization fit on the *synthetic* train distribution; on ClimbMix those stats supposedly don't transfer, so logits "blow up and sigmoids saturate," making no absolute threshold sparse (0.7 floor → ~199 firings/token; 0.99999 → ~4.7%); the relative/percentile gate was framed as the "fix," and the attribution as "rank-based but meaningful."
*Why this is wrong (2026-07-01 diagnostic):* corpus activations are **not** OOD under the train standardization (mean|z|≈0.8, norms *smaller* than synthetic), and the identical dense firing / near-saturation happens **on the synthetic-TEST set itself** (control) — so it is a `pos_weight`/threshold property of the probes, not a corpus distribution shift. "Corpus-recalibration of standardization" would not help. The dense-firing *symptom* was real; the *cause* stated here was incorrect. And critically, the relative gate does NOT make the attribution meaningful — §Exp 2 shows the fired tokens are frequency noise (0–4% concept-relevant).

**Throughput & scale:** measured ~8,450 tok/s on H100 (eager, batch=32) → **~90–100 min/shard** (ClimbMix shard ≈ 43–48M tokens). This is the binding constraint (the full 187-shard sweep ≈ 280 GPU-h ≈ $920, infeasible in-window). Run on a **fleet of ~16 H100 pods** with shard ranges tiled across 0–185 + 6542, so completed shards form a representative spread of the corpus rather than a contiguous prefix. Auto-stop on crossing the 30% (56-shard) target; hard caps 16:00Z / $550 spend.

**Coverage achieved:** **28 shards uploaded = 28/187 = ~15%** (user-directed stop at ~12:25 UTC 2026-06-30; fleet of 17 H100 pods terminated cleanly, resumable via resume-skip). Shards: 0,1,10,16,17,32,33,48,49,64,65,75,76,86,87,97,98,108,119,130,131,141,152,153,163,164,174,6542. **BUT per Reliability §Exp 2 this attribution is not validated as concept attribution — the firings track token frequency, not concepts.** Do not use the 28-shard dataset for concept claims without a gating/probe fix.

## quirks / flaws / open issues
- See **⚠️ Reliability & Failure Modes** section (top) for the ranked, evidence-based problem list. Summary: (1) corpus attribution ≠ concepts (frequency noise); (2) the "calibration mismatch" mechanism was a misdiagnosis; (3) AUROC 0.954 → precision 0.25 at the operating point; (4) synthetic→natural transfer gap.
- **Open experiments to run next:** (a) trigger-rank-when-present (does the probe rank the concept's own token highly *when the concept is actually in the sentence*? — separates "concept absent in sample" from "probe ignores trigger"); (b) causal steering + directional ablation vs random-direction and CAA controls; (c) a concept-prevalence-aware / precision-targeted gate to see if attribution is salvageable; (d) shuffled-label control probes; (e) corpus-wide (only 1 shard diagnosed).
