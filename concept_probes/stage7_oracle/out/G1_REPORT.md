# Gate G1 — corpus-scoring sanity — **VERDICT: FAIL**

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
