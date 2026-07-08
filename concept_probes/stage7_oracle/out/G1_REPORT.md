# Gate G1 — corpus-scoring sanity — **VERDICT: FAIL (original) → PASS after relabeling (see Post-fix re-evaluation)**

Run 2026-07-08. Local (natural-pool reference) + pod-side (corpus sample,
64,000,000 tokens across the 8 fully-transferred shards 320/321/331/332/
342/343/353/354 on `root@31.24.80.36:10617`, CPU-only, ~50s, no GPU/training
interference).

**This is not a marginal or borderline result. A confirmed, fully root-caused
indexing bug in Phase 0 (`select_probes.py`) silently permutes 53 of 54
concepts across all three main score-store layers (162 of 216 columns,
75%). The number stored under a given concept's name is, for 53/54
concepts, a *different concept's* probe score. Only "september" (by pure
coincidence) is in the right slot.**

**⚠ URGENT: a "production training run" (PID 8276, `train_encoder.py
--mode expA --full-ft ...`) is running on the same pod RIGHT NOW, training
against this corrupted score store. Unless `train_encoder.py` independently
re-derives column identity in a way that bypasses the bug (checked below —
it does not), that run is learning 162/216 targets under the wrong label
and should be stopped and re-launched after a fix + rescore.**

---

## 1. Root cause (confirmed, not inferred)

`concept_probes/stage7_oracle/code/select_probes.py`, Phase 0 Step 5
assembly (`main()`, the `not relaxed_used` branch — the one actually taken,
per STATE.md "no relax fallback needed"):

- Line 423: `survivor_names = sorted(c for (_, c) in survivor_concepts)` —
  sorts by **concept name only** (global alphabetical). This list becomes
  `concept_names`, which is written verbatim into `probe_set.json`'s
  `"concepts"` field — the canonical order used by *every* downstream
  consumer (`score_corpus.py`'s `ProbeSet.concepts`, its
  `score_column_names()`, and this G1 script).
- Line 460: `for ci, (fam, c, rows_l) in enumerate(survivors):` then line
  468 `W[li, ci] = w` — fills the **main** `W`/`b` arrays using the index
  from iterating `survivors`, not `concept_names`. `survivors`'s order
  comes from `select_concepts()` line 201:
  `concepts = sorted({(r["family"], r["concept"]) for r in rows})` — sorted
  by **(family, concept) tuple**, i.e. family-grouped-then-alphabetical.

These are two *different permutations* of the same 54 names, silently
conflated under the same index variable `ci`. Concretely: position `i` in
`probe_set.json["concepts"]` (global-alphabetical) does **not** correspond
to position `i` in `W`/`b` (family-grouped) — except where the two sort
orders coincidentally agree.

The **DoM-at-ablation-layer block is NOT affected**: its assembly loop
(line 524) correctly does `for ci, c in enumerate(concept_names):`, i.e.
indexes by the canonical (global-alphabetical) list, matching what's in
`probe_set.json`. This matches the evidence below (dom-block columns show
only mild, plausibly-benign scale differences; main-block columns are
catastrophically wrong).

### Reproduction (54-concept permutation table, first/most relevant rows)

Computed directly from `probe_set.json`'s own `concepts` + `families`
fields (no re-run of Phase 0 needed — this is a closed-form consequence of
the two sort keys):

| main-block col idx | labeled as | **actually contains** | same family? |
|---:|---|---|---|
| 9  | europe   | **africa**  | continents = continents (coincidence) |
| 13 | full_moon| **oceania** | moon_phases ≠ continents |
| 15 | january  | **east**    | months ≠ directions |
| 19 | march    | **south**   | months ≠ directions |
| 23 | north    | **april**   | directions ≠ months |
| 31 | red      | **may**     | color_wheel ≠ months |
| 34 | september| september   | **only fixed point** |

Full result: **1/54 slots correct** (`september`); 53/54 mislabeled, at
**all 3 main layers identically** (same `ci` bug applies uniformly across
the layer loop) → **162/216 score-store columns (75%) hold the wrong
concept's data.**

---

## 2. Evidence that led to the finding

### (b) Top-firing-token spot check — this is what caught it

Layer-8 main-block columns, 5 concepts, `saturated_top_by_frequency`
(token-id frequency **among tokens clipped to the int8 ceiling**, code
== +127 — see §3, plain top-100 was uninformative due to clipping):

| labeled concept | top tokens (first ~10, by frequency among saturated) | reads as |
|---|---|---|
| **january** | ` where`, ` first`, ` the`, ` of`, `'`, ` East`, ` east`, ` eastern`, ` direction`, ` which` | **"east"** (directions), not January |
| **red** | `5`, `1`, ` `, `0`, `th`, ` five`, ` May`, `-`, `4`, `2` | **"may"** (months) + generic digits, not the color red |
| **north** | `4`, ` four`, ` `, ` April`, ` of`, `2`, `3`, `8`, `1`, `9` | **"april"** (months) + digits, not the direction north |
| **europe** | ` world`, `s`, ` of`, ` the`, ` Africa`, ` Europe`, ` South`, ` ocean`, ` America`, ` India` | generic continent talk — consistent with **"africa"** |
| **full_moon** | ` world`, ` the`, ` of`, ` ocean`, ` America`, ` South`, ` sea`, ` States`, ` Europe`, ` Africa` | generic continent talk — consistent with **"oceania"** (near-identical list to "europe"'s, which is itself the tell — two nominally unrelated concepts should not fire on the same vocabulary) |

Every one of these matches the predicted "actually contains" column from
§1's permutation table exactly (january→east: "East/eastern/direction" is
unmistakable; north→april: "April" appears directly; red→may: "May"
appears directly; europe/full_moon→africa/oceania: both resolve to
generic continent vocabulary, explaining why two different "concepts"
produced near-identical firing-token lists — a red flag in itself, called
for explicitly by the G1 spec's "must fire on surface forms/associates"
test).

**Lexical-concept check (b): FAIL.** None of the 5 spot-checked concepts
fire on their own surface forms; all fire on their permuted concept's
surface forms instead.

### (a) Quantile match vs natural-pool reference

Per-column comparison, 216 columns, natural-pool reference = ALL
(cal+test) tokens of the chosen arm at the chosen layer from
`stage6/data/natscores/<family>.natscores.npz` (`g1_natural_ref.json`) vs.
corpus sample quantiles (`g1_corpus_stats.json`, exact via a 255-bin
int8 histogram, dequantized through `quant.json`'s affine map):

- Median `|p50 shift|` (natural-std units) over all 216 columns: **0.155**
  — looks fine in aggregate, which is itself a lesson (see below).
- Columns flagged (`|p50 shift| > 0.5·std` OR corpus/natural std ratio
  outside `[0.33, 3]`): **119/216 (55%)**.
  - **Main block: 109/162 flagged (67%).**
  - **Dom block: 10/54 flagged (19%)** — mild, plausibly a genuine (benign)
    distribution difference between the curated Stage-6 "natural_mined +
    natural_random" pool and a raw ClimbMix sample, not a bug (dom-block
    assembly is confirmed correct per §1).
- By nominal (labeled, i.e. wrong) arm, main+dom pooled: ridge-labeled
  columns show corpus/natural std ratio **median 36×** (up to 649×);
  lda-labeled columns show **median 0.021×** (corpus ~50× *narrower* than
  labeled-arm expectation); dom-labeled (mixed main+dedicated) columns
  **median 0.35×**. This ridge-too-big / lda-too-small split is exactly
  what you'd expect from the permutation: each nominal slot is actually
  scored by a random *other* concept's arm, and ridge vs. LDA arms have
  very different intrinsic native scales (LDA discriminant directions are
  typically much larger-magnitude than ridge weights in these families),
  so pooling "by labeled arm" just measures how often a ridge slot got
  swapped with a bigger- or smaller-scale neighbor.

**Methodological note for future gates:** the *aggregate* median shift
(0.155 std, "looks fine") completely masked a 67%-corrupted main block —
per-column min/max ranged from 0.0004× to 649× std ratio. Aggregate
medians are not a sufficient G1 check on their own; per-column outlier
flagging + the token-level spot check (which is what actually caught this)
are both necessary.

### (c) january-vs-march correlation

Computed value: **Pearson r = 0.277** on the (mislabeled) layer-8 "january"
and "march" columns — nominally *passes* the stated criterion
("within-family correlated but < 0.9"). **This is a false pass**: per §1,
the "january" column is actually **east** and the "march" column is
actually **south** — both genuinely directions-family concepts, so a
mid-range positive correlation between two real same-family direction
concepts is unsurprising and tells you nothing about January vs. March.
This criterion alone, taken in isolation, would **not** have caught the
bug — it's a case where a sane-looking number was actually evidence of a
different (also real) relationship. Flagging explicitly per the "be strict
about honesty" instruction.

---

## 3. Secondary issue found: int8 quantization clipping is severe for spot-checked columns

Independent of the permutation bug: `clip_frac_pos127` (fraction of the
64M-token sample landing exactly at the int8 ceiling, code = +127) for the
5 spot-check columns: january 0.084%, red 0.222%, north 0.144%, europe
0.831%, full_moon 0.887%. That means the naive "top-100 by raw score"
extraction (`argpartition`) was picking an **arbitrary** sample among tens
of thousands to ~560,000 tied-at-the-ceiling tokens, not the true top-100 —
the plain `top_tokens` field in `g1_corpus_stats.json` is not meaningful as
delivered; `saturated_top_by_frequency` (frequency-of-token-id among the
clipped set) was added mid-run and is the field actually used above. This
should be re-examined once the permutation bug is fixed and quant.json is
recalibrated on correct data — clipping at the <1% level for these columns
isn't necessarily disqualifying by itself, but it means quant.json's
`scale` (`4·std/127`, calibrated on the first 10M tokens) may be
systematically too tight for the true (post-fix) per-column distributions,
and should be re-verified.

---

**Corroborating observation (live, at report time):** PID 8279's heartbeat
(`/workspace/hb_train.txt`) shows `step 1300, median_r2=0.56` — a
perfectly plausible-looking Gate-G2-range number. This is exactly the
"false confidence" scenario predicted above: the encoder is regressing
against internally self-consistent (just mislabeled) targets, so the R²
metric alone gives no signal that 162/216 target columns are attached to
the wrong concept name. G2 as currently specified (heldout median R²)
would not catch this bug either — only the token-level spot check does.

## 4. What needs to happen before Phase 2

1. **Stop and do not trust `train_encoder.py` PID 8276** until confirmed
   whether it inherits the bug (it consumes the score-store columns
   positionally, labeling them via the same `probe_set.json["concepts"]`
   order used for scoring — since `score_corpus.py` wrote the store using
   the corrupted `W` in that same nominal order, `train_encoder.py` is
   almost certainly training 162/216 targets against the wrong concept
   name, even though the *numbers* it's fitting are internally
   self-consistent college-of-scores, just mislabeled).
2. Fix `select_probes.py`: either (a) sort `survivors` by concept name only
   before the `enumerate` at line 460 (matching `concept_names`'s sort
   key), or (b) index by `idx_of_c[c]` exactly as the (correct) `relaxed`
   branch (line 500) and the DoM block (line 524) already do. Re-run
   Phase 0 → new `probe_set.json` / `probe_set_arrays.npz`.
3. **Re-score**: all already-written `scores_*.npy` shards (320, 321, 331,
   332, 342, 343, 353, 354, plus whatever the other pods have produced)
   encode the corrupted `W` and must be regenerated from the fixed
   `probe_set_arrays.npz`. Recalibrate `quant.json` after the fix (§3).
4. Re-run G1 (this script pair,
   `concept_probes/stage7_oracle/code/g1_natural_ref.py` +
   `g1_corpus_check.py`) against the corrected store before any further
   encoder training spend.

---

## Appendix: raw evidence files

- `concept_probes/stage7_oracle/out/g1_natural_ref.json` — 216-column
  natural-pool reference quantiles (local, from `stage6/data/natscores`,
  ALL cal+test tokens, chosen arm per `probe_set.json["selection"]`).
- `concept_probes/stage7_oracle/out/g1_corpus_stats.json` — 216-column
  corpus-sample quantiles + spot-check firing-token data (pod-computed,
  64M tokens, exact via int8 histogram; includes `clip_frac_pos127`/
  `clip_frac_neg127` and both the raw-argpartition `top_tokens` and the
  more meaningful `saturated_top_by_frequency`).
- `concept_probes/stage7_oracle/code/g1_natural_ref.py`,
  `concept_probes/stage7_oracle/code/g1_corpus_check.py` — the two scripts
  that produced the above (local venv / pod CPU respectively).

---

## Post-fix re-evaluation (2026-07-08) — **VERDICT: PASS**

The FAIL above was entirely a **labeling** artifact, not a scoring defect.
Every G1 anomaly was G1 comparing store column `col` (which truly holds
`main_block_concepts[col%K]`, family-sorted) against the natural-pool
reference for `concepts[col%K]` (name-sorted) — i.e. comparing two DIFFERENT
concepts. Re-evaluated against each store column's TRUE concept using only
the already-written artifacts (`g1_corpus_stats.json`, `g1_natural_ref.json`,
`probe_set.json`) — **no pod rescoring**. See `PERMUTATION_FIX.md` for the
permutation and its two independent verifications.

### (a) Quantile match, re-paired to true concepts

Flag rule unchanged (`|p50 shift| > 0.5·natural-std` OR corpus/natural std
ratio outside `[0.33, 3]`).

| block | flagged (mislabeled, original) | flagged (true concept) |
|---|---:|---:|
| MAIN (162 cols) | **109/162 (67%)** | **5/162 (3%)** |
| DOM (54 cols) | 10/54 | 10/54 (unchanged — was already correct) |

MAIN-block median `|p50 shift|` after re-pairing: **0.148** natural-std
units. The **5** residual main flags are all benign borderline std-ratios
(0.27–0.32, just under the 0.33 floor) with tiny p50 shifts (< 0.2 std):
`red-orange@L6/L8`, `yellow-green@L6`, `full_moon@L6`, `new_moon@L6` — all
DoM-arm columns whose curated natural pool is narrower than a raw ClimbMix
sample, the exact same benign effect as the 10 dom-block flags. **Not
permutation artifacts.** The 67%→3% collapse is the decisive evidence that
the 109 flags were the label bug, now resolved.

### (b) Top-firing-token spot check — now matches true concepts

Re-labeling each spot column to its true concept (the firing-token data is
unchanged; only the name attached to it changes):

| spot data (was labeled) | TRUE concept | top saturated tokens | reads as |
|---|---|---|---|
| "january" | **east** | ` East`, ` east`, ` eastern`, ` direction`, `where`, `first` | east ✓ |
| "red" | **may** | ` May`, ` five`, `5`, `1`, `0` | may ✓ |
| "north" | **april** | ` April`, ` four`, `4`, `2`, `3` | april ✓ |
| "full_moon" | **oceania** | ` ocean`, ` America`, ` South`, ` sea`, ` States` | continent/ocean ✓ |
| "europe" | **africa** | ` Africa`, ` Europe`, ` South`, ` ocean` | continent ✓ |

**Lexical-concept check (b): PASS.** All 5 columns fire on their TRUE
concept's surface forms.

### (c) january-vs-march correlation — the false pass explained

The reported `r = 0.277` was computed on store cols 69 and 73, which are
name-sorted positions of january/march but truly hold **east** (col 69) and
**south** (col 73) — both directions-family, so a mid-range positive
correlation is expected and says nothing about January vs. March. The TRUE
january and march store columns are **col 81** and **col 84** (L8 block,
family-sorted positions of january/march within months). **This specific
correlation cannot be recomputed from local artifacts** — `g1_corpus_stats`
only retained the raw int8 code arrays for the two (mislabeled) columns it
sampled, not for arbitrary columns. Recomputing the real January-vs-March
Pearson needs a one-shot pod pass of the fixed `g1_corpus_check.py` (which
now indexes main-block columns by `main_block_concepts`, so it samples cols
81/84). Flagged as the one item requiring a (cheap, optional) pod re-touch.

### Secondary issue still open

The int8 clipping finding in §3 (severe saturation for high-scale dom
columns) is **independent of the permutation** and still stands. `quant.json`
`scale = 4·std/127` may be too tight for the true per-column distributions;
worth re-checking on a small pod pass, but it does not affect the G1 verdict.

### Bottom line

With correct labels, G1 **PASSES**: (a) 3% residual flags, all benign; (b)
all spot columns fire on their true concept's vocabulary; (c) the "false
pass" is understood (east-vs-south, not january-vs-march). The 440 GB store
is internally valid — the numbers were always right, only the names were
swapped. No rescore is required to trust the store; only the label contract
(`main_block_concepts` / `dom_block_concepts`, now in `probe_set.json`) is
needed downstream.

---

## Residual checks (2026-07-08, pod-side, CPU-only)

Two items flagged as still-open in the post-fix re-evaluation: (c)'s
"cannot be recomputed from local artifacts" note, and §3's int8-clipping
concern for the (then-unidentified) worst columns. Both closed now, using
the already-fixed, already-synced `code/g1_corpus_check.py`
(`main_block_concepts`-indexed). Raw results:
`concept_probes/stage7_oracle/out/g1_residual_checks.json`. No pod rescoring,
no code changes — CPU-only reads against the already-landed shards, run
alongside the live `expA_prod` training job and the rsync pull loop without
touching either.

**Housekeeping note:** the `g1_corpus_stats.json` sitting in `/workspace/scores`
at the start of this check (and mirrored to `out/g1_corpus_stats.json`,
timestamped after the fix) turned out to still carry the **pre-fix**
name-sorted main-block labels (e.g. col 15 shown as "january" when
`main_block_concepts[15]` is actually "east") despite the script on disk
already being the fixed version — an intermediate artifact left over from
mid-fix, not a fresh run. `code/g1_corpus_check.py` was re-run fresh on the
pod to eliminate the ambiguity (14 fully-transferred shards now available,
up from 8; 64,000,000 tokens, 50s). All numbers below use this fresh,
verified-correct run. The dom block was never affected by the label bug
(§1), so dom-block concept names throughout this report were trustworthy
all along.

### 1. True january-vs-march correlation (and other pairs)

Computed two independent ways, in agreement:

| pair | family relation | col a / col b (true) | n | Pearson r |
|---|---|---|---:|---:|
| january – march | months, within-family | 81 / 84 | 24.0M | **0.308** |
| monday – friday | weekdays, within-family | 102 / 101 | 24.0M | **0.314** |
| red – blue | color_wheel, within-family | 58 / 54 | 24.0M | **0.407** |
| january – north | months × directions, cross-family | 81 / 70 | 24.0M | **0.133** |
| january – march (full-script rerun, cross-check) | months, within-family | 81 / 84 | 64.0M | **0.308** |

The two independent january-march computations (a 24M-token standalone
script and a fresh full 64M-token run of the official
`g1_corpus_check.py`) agree to 3 decimal places (0.3081 vs 0.3077),
confirming the true-concept indexing is stable and correct. All three
within-family pairs land in a consistent mid-range positive band
(0.31–0.41) — correlated but nowhere near collinear — while the
cross-family january–north pair is markedly lower (0.13), exactly the
ordering the G1 criterion expects. This is the real signal the original
(mislabeled) `r=0.277` was accidentally imitating.

**Verdict: PASS.** True january-vs-march r=0.31, consistent (0.3081 vs
0.3077) across two independent computations, clearly within-family
correlated and well under the 0.9 collinearity ceiling; other same-family
pairs (0.31, 0.41) and the cross-family control (0.13) all behave as
expected.

### 2. int8 clipping audit (all 216 columns)

Exact clip fractions (share of stored codes at exactly ±127) from the fresh
64M-token histogram, full 216-column audit:

- **0/216 columns exceed the 1% concern threshold.** Worst overall: col 13
  (main, L6, **oceania**) at **0.908%**.
- **195/216 (90%) exceed the 0.1% "fine" threshold** — clipping in the
  0.1–0.9% band is essentially universal across the store, not isolated to
  a few columns.
- **Dom block (54 cols): 0/54 over 1%, 54/54 (100%) over 0.1%.** Median
  clip fraction **0.695%**, max **0.899%** (col 201, **spring**). All
  clipping in the dom block is one-sided (positive only, neg ≈ 0 to
  machine precision) — consistent with DoM-ablation scores being
  fundamentally one-sided (magnitude-of-presence), not symmetric.

Worst 10 columns overall (all near-tied in the 0.84–0.91% band, well under
1%):

| col | block | layer | concept | clip frac (total) |
|---:|---|---:|---|---:|
| 13 | main | 6 | oceania | 0.908% |
| 201 | dom | 8 | spring | 0.899% |
| 10 | main | 6 | asia | 0.895% |
| 67 | main | 8 | oceania | 0.883% |
| 190 | dom | 8 | oceania | 0.883% |
| 171 | dom | 8 | europe | 0.869% |
| 164 | dom | 8 | asia | 0.856% |
| 68 | main | 8 | south_america | 0.855% |
| 198 | dom | 8 | south_america | 0.855% |
| 9 | main | 6 | africa | 0.841% |

**Reading:** no column is individually alarming (all safely under 1%), but
the near-universal 0.1–0.9% clipping (195/216 columns) confirms the §3
prediction in the original report — `quant.json`'s `scale = 4·std/127`,
calibrated on the first 10M tokens, is systematically a bit tight for the
true full-corpus tails, most visibly for continents/seasons-family dom
columns. Worth a recalibration pass before a from-scratch rescore is ever
undertaken, but **not** blocking for the current store.

**Verdict: CONCERN (non-blocking).** No column crosses the 1% hard
threshold; the pervasive 0.1–0.9% band (90% of columns, including 100% of
the dom block) is a real, systematic under-calibration of `quant.json`,
not sampling noise — recommend recalibrating scale on a larger/later
sample if the store is ever regenerated, but it does not change the G1
PASS verdict.

### Exp-B impact estimate: worst dom column (spring, col 201)

The `v* = D_dom · G_dom_inv · (s_dom − t_nat_dom)` formula (`train_encoder.py`
`ProbeSet.v_star`) is linear in the dequantized dom scores, so a
per-column dequantization error δ_c propagates to `v*` via a fixed
sensitivity vector `∂v*/∂s_c = G_dom_inv[c,:] @ D_dom.T` (2304-dim). Using
the real `probe_set_arrays.npz` matrices (no pod needed for this part) for
`c = spring`:

- Quant-rounding floor (always present, no clipping): `scale/√12 = 0.345`
  raw score units.
- Clip-fraction for this column: 0.899% (one-sided, positive only).
- Modeling the excess-beyond-clip magnitude with a Mills-ratio Gaussian-tail
  approximation, **tail-probability-matched** to the observed clip rate
  (z_eff=2.37; note the *geometric* z from the calibration std would be
  4.43, at which a true Gaussian would clip only 4.7e-6 of tokens —
  **1900× less** than actually observed, confirming the true score
  distribution is far heavier-tailed than Gaussian near this boundary, so
  this estimate should be read as a conservative lower bound, not an
  upper bound): RMS excess ≈ 15.6 raw units when a token does saturate.
- Combined (floor + clip) per-token RMS dequantization error for this
  column: **1.52 raw units, ≈4.4× the pure-rounding floor.**
- Propagated through the real sensitivity vector (‖∂v*/∂s_spring‖₂ =
  0.172): v*-space L2-norm RMS error rises from **0.059 (floor only) to
  0.262 (floor+clip)** — also a 4.4× increase (the ratio is
  sensitivity-independent, since the same linear map multiplies both).
  Per-dimension (2304-way) RMS error: 0.0012 → 0.0055.
- Context: this sensitivity vector's norm is only ~0.46% of
  `D_dom[:, spring]`'s own L2 norm (37.05) — `G_dom_inv`'s whitening
  substantially damps any single dom column's contribution to `v*`,
  because the 54 dom-column probes are correlated and information about
  "spring" is spread across several correlated dimensions, not
  concentrated on one raw column.

**Bottom line:** clipping roughly quadruples the per-token quantization
noise floor for the worst dom column, but the resulting absolute v*-space
error (L2 norm ≈0.26 across 2304 dims) is modest and does not by itself
threaten Gate G3's ≥0.5 heldout-R² threshold on `v*`. It is a real,
non-negligible degradation worth fixing on a future rescore/recalibration,
not an urgent blocker for the current store.

Full numeric detail (all 216 columns' clip fractions, both correlation
runs, and the Exp-B propagation calculation with all intermediate values):
`concept_probes/stage7_oracle/out/g1_residual_checks.json`.
