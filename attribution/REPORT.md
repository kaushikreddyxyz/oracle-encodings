# Corpus attribution (gold-probe scoring of ClimbMix) — Results Report

_Split from concept_probes/stage7_oracle/REPORT.md 2026-07-13 (corpus-scoring / attribution / climbmix-audit sections; encoder/oracle-training sections — Exp A/B, per-layer oracles, the full G0–G4 gate table, and the geometry side-study — live in `../oracles/REPORT.md`). Gate G1 (corpus-scoring sanity) is attribution-side: **FAIL -> PASS** — label-permutation bug found + fixed via metadata, no rescore; root cause and remediation summarized in the Incidents section below (the original G1_REPORT.md / PERMUTATION_FIX.md docs were distilled into `README.md` 2026-07-13 and live in git history; numbers in `out/g1_*.json`)._

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

## 2026-07-14: donor loudness (`measure_loudness.py`) — COMPLETE, gates PASS, uploaded

How loud the 54 gold concepts are **natively in gemma-2-2b**, in the injection
gate's own unit (fraction of the residual-stream norm, `‖Δx‖/‖x‖`). Measured on
an RTX-unavailable/H100 pod (bf16, eager, ~$3; run + teardown < 30 min), sampling
stored token windows from the score stores (exact stored ids → BOS prepended/
dropped, reproducing the store). `loudness.json` (schema v1) pushed to all 8
`climbmix-scored(+overflow-2..7)` repo roots + a corpus-scores variant to
`corpus-scores(+overflow)`; discovered automatically by nanochat's `--gate donor`.

**Gates (both variants):** affine identity `(⟨x,u_c⟩−m_c)=(z_c−z̄_c)·κ_c` median
rel-err **2.0e-8** (< 1e-3 ✓ — confirms the raw-space direction / κ read of the
frozen pipeline is exact); recomputed-vs-stored-int8 z cross-check Spearman
**0.9998** (> 0.99 ✓), median |Δz| **0.011** (< 0.15 ✓); `std2 ≈ quant·127/4`
ratio median 1.0004. No permutation, no scale error.

**Headline (climbmix, 498 docs / 293k tokens; corpus-scores 300 docs / 160k tokens
agrees to ≲0.01):**

| layer | median ‖x‖ | κ median (range) | λ per-σ median (range) | ℓ_tot ridge p50 / p95 / p99 | ℓ_tot dom p50 |
|---|---|---|---|---|---|
| L6  | 90.8  | 1.067 (0.99–1.93) | 0.0118 (0.011–0.021) | 0.081 / 0.131 / 0.171 | 0.094 |
| L8  | 107.0 | 1.264 (1.17–2.78) | 0.0118 (0.011–0.026) | 0.081 / 0.131 / 0.163 | 0.102 |
| L14 | 187.1 | 2.218 (2.11–4.37) | 0.0119 (0.011–0.023) | 0.086 / 0.137 / 0.172 | 0.108 |

**Reading it.** A single concept at 1σ is ≈**1.2 % of the residual stream**
(λ ≈ 0.0118, remarkably layer-flat once κ's growth is divided by ‖x‖'s growth);
the loudest concepts (L8 tail κ 2.78) reach ~2.6 %. The whole 54-concept *packet*
(`ℓ_tot`, the direct gate analogue) sits at **≈0.081 (p50)** and **≈0.13 (p95)**,
rising to ~0.17 at p99. So:

- **gate = 0.05** (the v1 default) is *below* gemma's own median packet loudness
  (0.081) — the injection is currently ~1.6× quieter than the donor plays these
  concepts. `--gate donor` (p50) lands it at 0.081.
- **the ~0.14 "architectural ceiling"** matches the donor's **p95** packet
  loudness (0.131–0.137) almost exactly — i.e. the ceiling isn't arbitrary, it's
  roughly where gemma's own loud tail already lives. `--gate donor:p95` targets it.
- DoM directions read slightly louder than ridge (packet p50 0.094–0.108) — the
  npz vectors are standardized-space read directions (== `W_dom_abl`), converted
  to raw-space `v_dom = W_dom ⊘ nat_std` exactly like the ridge to keep the two
  comparable; DoM is reported only (not consumed by the trainer).

**Variants.** `corpus-scores` and `climbmix-scored` share byte-identical step-2
`mu2/std2`+quant (calibrated once on shard 320), so **κ is identical** (max diff
0.0e0); only ‖x‖/empirical loudness differ, and they agree to ≲0.01 (different
corpora, same story) — recorded in the corpus-scores provenance.

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
