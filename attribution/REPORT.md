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

## 2026-07-14: reconstruction sufficiency + donor-loudness verification — COMPLETE, VERDICT: PASS

Adversarial verification of the donor-loudness work plus the question the next
experiments stand on: do the stored int8 probe scores (+ frozen geometry)
**reconstruct** gemma-2-2b's residual stream in the concept subspace, and is
the nanochat injection a faithful image of that reconstruction? Tooling:
`verify_reconstruction.py` (`cpu` = conditioning/$0; `pod` = empirical GPU),
artifact `out/reconstruction_report.json`. Empirics on a fresh H100
(~30 min, ~$1.5), shards **3/13/23** — disjoint from the loudness run's
2/12/22 — 120k tokens, 300 docs.

### The identity (exact)

The frozen pipeline is affine in x, so per concept
`⟨x,u_c⟩ = z_c·κ_c + const_c`, `const_c = (mu2_c−b_c)/‖v_c‖ + ⟨nat_mean,u_c⟩` —
verified on fresh activations at median rel-err **1e-15**. Stacking
`B=[u_c]`, `G=BᵀB`: `x̂_S = B G⁻¹ (z−z̄)·κ = P_S(x−x̄)` exactly (checked to
1e-13), and the **absolute** component `P_S x` is recovered via `const` with no
sample mean. The stored z-scores are a lossless linear re-parameterization of
the concept-subspace component — the only losses are int8 quantization
(step 4/127 ≈ 0.0315σ) and the ±4σ clip.

### Empirical reconstruction from the STORED int8 scores

| layer | R² pooled | cos med / p05 | R² unsat. tokens | median err: stored / pure-quant / predicted | sat. tokens (≥1 concept @±4σ) | ℓ_tot p50 here vs artifact |
|---|---|---|---|---|---|---|
| L6  | 0.9685 | 0.99977 / 0.99836 | 0.99959 | 0.154 / 0.134 / 0.123 | 7.56 % | 0.0800 vs 0.0808 |
| L8  | 0.9783 | 0.99980 / 0.99924 | 0.99963 | 0.175 / 0.141 / 0.129 | 6.63 % | 0.0801 vs 0.0813 |
| L14 | 0.9866 | 0.99980 / 0.99942 | 0.99959 | 0.326 / 0.227 / 0.222 | 5.28 % | 0.0854 vs 0.0863 |

- **Typical tokens are essentially perfect** (cos 0.9998, unsaturated-token
  R² 0.9996). The pure-quantization floor matches the uniform-noise prediction
  propagated through G⁻¹ to within 2–9 %; stored-vs-recomputed adds only bf16
  forward nondeterminism (|Δz| ≈ 0.01σ), immaterial.
- **The ±4σ clip is the only material loss.** z tails are ~20× heavier than
  Gaussian: 5.3–7.6 % of tokens saturate ≥1 concept (0.17–0.30 % of cells). On
  those tokens R² drops to 0.87–0.91 (direction still good, cos med 0.995+;
  clipped magnitude). The pooled-R² gap is entirely this tail.
- **Captured share:** the 54-dim subspace holds 1.2–1.5 % of centered residual
  variance (per-token median ‖P_S y‖/‖x−x̄‖ ≈ 0.10–0.11). Median
  ‖P_S y‖/‖x‖ reproduces the artifact's ℓ_tot p50 on disjoint shards to <2 %.
  (ℓ_tot² understates the centered-variance share since its denominator is the
  uncentered ‖x‖.)
- **Conditioning (cpu, $0):** cond(G) 70/46/43 (L6/8/14), min eig 0.085,
  effective rank 27–31/54. Worst collinearity is intra-family:
  waxing_crescent~waxing_gibbous 0.843 (L6), red-orange~yellow-green 0.83–0.87
  (all layers), continents ≤0.66; cross-family max 0.39. Noise amplification
  through G⁻¹ is mild in aggregate (rms ×1.3–1.45 vs orthonormal); worst
  per-concept coefficient noise ×9.2 (waxing_gibbous L6) — moon-phase/tertiary-
  color coefficients are the least trustworthy individually, their sum is fine.

### Donor-loudness verification (A) — all PASS

- **κ re-derived independently** (raw npz + store corpus_stats, no
  measure_loudness code): identical to the artifact (rel diff 0.0).
- **λ layer-flatness is structural, not coincidental:** per-concept λ ratios
  L8/L6 and L14/L6 have median 1.017/1.018, IQR ±4 % — each concept's raw-space
  σ scales *with* the stream norm (κ and median ‖x‖ both ×2.06 L6→L14).
  Real outliers exist (autumn ×1.9, oceania ×0.56, blue-green ×0.61).
- **Artifacts on HF:** climbmix-scored + overflow-3 byte-identical to local;
  corpus-scores variant correctly distinct (its own corpus field/shards
  322/335/350) with κ shared **by design** (byte-identical frozen step-2
  constants, documented in its provenance.kappa_note). Inequality audit —
  active ≥ all per concept per quantile; ℓ_tot ≥ every single concept at
  p50/p95/p99 (pointwise-dominance argument) — **0 violations**; active-token
  rate ≈2.6 % ≈ P(z≥2), i.e. calibrated standardization.
- **measure path:** BOS prepend/drop, eager attention, 2048 non-overlapping
  tiling, int8 decode (q·scale+zero then step-2) all match the store
  conventions (code-verified against score_climbmix_stacked/ScoreHead), and
  re-confirmed empirically on virgin shards: Spearman ≥ 0.9998,
  median|Δz| ≤ 0.011. Measured on cuda/bf16 = the store's own scoring dtype;
  norms in fp64 on fp32-cast states — no fp16 accumulation exposure. No bugs.
- **nanochat donor gate (d97945b) + loudness dial (fff58c2):**
  `rms(gate) = dial×L_ref` exact (dial 1.0 → 0.08125 @L8; `donor:p95` → 0.13101
  = dial 1.612); per-token ‖Δx‖/‖x‖ == rms(gate) *exactly* (the z-renorm makes
  per-channel shaping direction-only — verified to 3e-17); dead-channel
  handling, hard concept-order refusal, checkpoint persistence + resume
  (persisted absolute vector, never re-resolved) all correct; 28 tests green.
  **Migration hazards (documented behavior, not bugs):** a plain-number
  `--gate` is now a *dial* — an old `--gate 0.05` yields absolute
  0.05×L_ref ≈ 0.004 (~20× quieter than the old semantics; use `abs:0.05`);
  and `InjectionCfg.gate` defaults to 1.0 with dial-at-trainer /
  absolute-at-site semantics, so constructing `InjectionCfg` directly and
  calling `build_sites` without trainer-level resolution injects at 100 % of
  residual RMS.

### Injection-side sufficiency — what the packet preserves

Site: `Δx = rms(gate)·rms(x)·ẑ`, `ẑ = (a⊙ĝ)D / rms((a⊙ĝ)D)`, D random-orthonormal.

- **Preserved (lossless):** the per-token *direction pattern*. scores→packet is
  linear and invertible up to a per-token positive scalar — `a` is recovered
  from Δx via `(ΔxDᵀ)⊘ĝ` (verified to 1e-10) — and scores determine x̂_S exactly
  (above). So the packet carries the full signed ratio structure of the
  54 concepts; nothing about *which concepts, in what proportion* is lost.
  Corpus-level relative concept loudness matches the donor (ĝ ∝ donor_c/rms_c),
  and overall loudness is dial-exact.
- **Not preserved:** (i) *absolute per-token magnitude* — the z-renorm pins
  ‖Δx‖ to rms(gate)·‖x_nano‖ whether the token is concept-silent or 6σ-loud,
  while gemma's own packet norm varies ≥×2 (ℓ_tot p50 0.081 → p99 0.17);
  (ii) *inter-concept geometry* — gemma's u_c Gram (cosines to 0.87) is
  replaced by D's exact orthogonality; concept correlations survive only in
  the z statistics (the data), not in the injected geometry; (iii) κ enters
  only via corpus-level shaping, not per token.
- **For experiment design:** "the model can read the oracle" claims are safe —
  the information is all there, linearly accessible. Claims about nanochat
  *re-creating gemma's representation* are not licensed: magnitude coding and
  angle coding differ by construction.

### Recommended variants for a more faithful injection (spec only, NOT implemented)

1. **Gram-shaped mixing (isometric packet).** Replace the diagonal shaping with
   the fixed 54×54 map `S = G^{-1/2}·diag(κ)` applied to z-scores before D:
   `z = (a S) D`. Then ⟨packet_i, packet_j⟩ = ⟨x̂_S,i, x̂_S,j⟩ — the injected
   subspace is an isometric image of gemma's (angles AND relative magnitudes),
   at the cost of no longer being channel-diagonal. One fixed matrix, no new
   data.
2. **Corpus-level (not per-token) normalization.** Replace `z/rms(z)` with a
   fixed divisor calibrated at startup (corpus rms of ‖z‖), keeping the
   zero-row clamp. Per-token magnitude information then survives; the gate
   becomes *mean* loudness rather than exact per-token loudness. Combined with
   (1), the packet is a fixed linear isometry of x̂_S up to one global scalar —
   fully faithful.
3. **If loud-token fidelity matters:** the ±4σ clip is the binding constraint
   (5–7.6 % of tokens). An int16 store or a sparse >4σ sidecar lifts full-token
   R² from ~0.97 to ~0.9996.

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
