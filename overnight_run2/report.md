# Concept-Probes v2 — Rectification Run Report

**Run:** 2026-06-30 (overnight, autonomous). Start 03:22Z, 9h deadline 12:22Z.
**Goal:** replace the incomplete prior run with (A) synthetic per-token concept datasets, (B) per-position linear probes at 12 layers, (C) full-corpus per-token attribution over ClimbMix shards 0–185 + 6542. Brief §4/§5 (geometry, GPU-speedup) out of scope.

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

## Phase B — probes
(pending)

## Phase C — attribution
(pending)

## quirks / flaws / open issues
(appended as found)
