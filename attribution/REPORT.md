# Corpus attribution (gold-probe scoring of ClimbMix) — Results Report

_Split from concept_probes/stage7_oracle/REPORT.md 2026-07-13 (corpus-scoring / attribution / climbmix-audit sections; encoder/oracle-training sections — Exp A/B, per-layer oracles, the full G0–G4 gate table, and the geometry side-study — live in `../oracles/REPORT.md`). Gate G1 (corpus-scoring sanity) is attribution-side: **FAIL -> PASS** — label-permutation bug found + fixed via metadata, no rescore; root cause in `out/G1_REPORT.md`, remediation in `../concept_probes/5_oracle/out/PERMUTATION_FIX.md`, and the Incidents section below._

---

## 2026-07-10: climbmix-scored attribution (nanochat training shards 0-184) — IN FLIGHT

Full-coverage scoring (NO 2048 truncation — consecutive windows; NO min-length filter; every
parquet row annotated) of ClimbMix-shuffle shards 0-184 (~9.9B tokens) with the frozen gold
probes + frozen quant/corpus_stats constants (byte-identical to corpus-scores). Detection-only
[n,3,54]; DoM dropped per user. 21× H100 across 3 pipelined waves; dest repos
`climbmix-scored` + `-overflow`..`-overflow-7` (25 shards each, in order; assignment.json).
Scorer: `score_climbmix_stacked.py`; launcher: `launch_attrib_wave.py`.

Validation: pipeline gate on shard 0 (coverage sum exact at 53,532,943 tokens) + INDEPENDENT
fresh-context audit on shard 1: exact token-id equality for all 53.5M tokens incl. across
window boundaries, saturation ≤0.28%, standardized first-window cols mean≈-0.001/std≈0.99,
constants byte-identical, cross-shard profile match to ~0.005σ. VERDICT: PASS.

## 2026-07-10: climbmix-scored deep audit — COMPLETE, VERDICT: PASS

Fresh-context audit over the finished 185-shard store (agent-run; 5-shard deep sample 3/46/91/137/179):
- **Completeness**: 185/185 shards × 3 files across the 8 repos; every file byte-exact vs parsed
  npy header (same n for scores/tokens); total **9,873,968,012 tokens** (53.0–53.8M/shard);
  metadata md5-identical across repos (pooling across repos is valid).
- **Token recovery**: re-tokenization matches stored ids bit-for-bit on all sampled docs incl.
  348k-token docs spanning many 2048 windows; doc spans tile [0,n) exactly.
- **Standardization**: first-window |μ|≤0.027σ, σ∈[0.83,1.01]; saturation one-sided at +4σ clip,
  median 0.135%, max 0.89% (oceania/L6). Cross-shard stability: mean spread ≤0.014σ over 267M
  tokens/layer/concept. Within-doc lag-1 autocorr 0.36–0.51 vs ~0 shuffled.
- **Semantics**: tie-immune keyword tests — 6/7 sampled probes fire 3.7–4.0σ on own keyword
  (620–1310× enrichment). full_moon settled by bigram conditionals: "full moon"→3.8–3.9σ (real
  bigram encoding) but bare "full"→3.9σ too — a "full"-token detector that also knows the bigram
  (upstream probe semantics, faithfully recorded). Broader companion study (all 54 concepts,
  shard 10): only may (modal homograph, 1.81σ) and last_quarter (fiscal homograph, 22 matches)
  are keyword-non-responsive; continents + intermediate colours are responsive-but-broad
  (z≥3 keyword precision ≲1.5%).
- **Concentration figure**: `out/figures/climbmix_concept_density.png` (+.npz) — all 54 concepts
  show Kronecker-delta/power-law concentration (top ~1e-4 fraction at ceiling → mean 0 by q→1).
  Broadest: continents (south_america/oceania/africa/asia). Peakiest: weekdays (saturday/sunday/
  thursday), west, winter.
- Geometry side-study — moved to `../oracles/REPORT.md` ("Geometry side-study" section)
  per the 2026-07-13 restructure.
- Ops: audits ran on a $0.17/hr A4000 pod (NIC + 112 cores; GPU used only for the geometry
  re-extraction at ~8.1k tok/s). hf_xet stalls on pods — set HF_HUB_DISABLE_XET=1 (classic
  HTTPS ~41MB/s). Pod deleted after artifact retrieval.

---

## Incidents caught (and handled)

_(The coords-precompute batch-1 drain bug lives in `../oracles/REPORT.md`.)_

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
