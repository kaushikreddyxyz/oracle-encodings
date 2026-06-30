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

## Phase A — datasets
(pending)

## Phase B — probes
(pending)

## Phase C — attribution
(pending)

## quirks / flaws / open issues
(appended as found)
