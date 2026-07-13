# Stage 7-Oracle — label-permutation fix (metadata + downstream labels only)

Companion to `G1_REPORT.md` (root cause). Written 2026-07-08 during the
no-rescoring remediation. **No score-store bytes and no
`probe_set_arrays.npz` array VALUES were changed.** The 440 GB int8 store on
the pods and the live `expA_prod` training run are untouched; this fix only
(a) adds two block-order keys to `probe_set.json` and (b) makes every
downstream consumer attach names to store columns via those explicit keys.

## 1. What was actually corrupted (and what was NOT)

The bug is confined to the **main-block `W`/`b` arrays** in
`probe_set_arrays.npz`. Their rows were assembled by
`enumerate(survivors)`, and `survivors` is `(family, concept)`-sorted, while
`probe_set.json["concepts"]` — the order every downstream reader consumes
positionally — is name-sorted. So main-block row `ci` holds the probe for
`main_block_concepts[ci]` (family-sorted), not `concepts[ci]` (name-sorted).
1/54 rows coincide (`september`, at col 34); 53/54 are permuted.

The DoM block (`W_dom_abl`, `b_dom_abl`, `t_nat_dom`, `G_dom`, `G_dom_inv`)
was assembled by `enumerate(concept_names)` — name-sorted — so it is
**correct** and index-aligned with `concepts`.

### Two independent verifications of the permutation (`code`/scratch)

**(a) Exact array equality vs Stage-5 source probes.** For spot columns
{january, north, red, september, europe} at all 3 layers, the row of
`probe_set_arrays.npz["W"]` at that column index was compared to
`native_wb()` recomputed from `stage5/probes/<fam>/probes_l{L}.npz`:
- Under the **family-sorted hypothesis** (row `ci` == `main_block_concepts[ci]`):
  **exact match (`np.array_equal`) at every layer**, including `b`.
- Under the **name-sorted hypothesis** (row `ci` == `concepts[ci]`): **no
  match** except `september` (the fixed point).
- DoM rows for {january, north, red}: exact match under the **name-sorted**
  hypothesis — confirming the DoM block is correctly ordered.

**(b) G1 firing-token evidence reproduces exactly.** The reconstructed
permutation yields precisely the associations G1_REPORT documented:
`col 15 january→east`, `col 19 march→south`, `col 23 north→april`,
`col 31 red→may`, `col 9 europe→africa`, `col 13 full_moon→oceania`, and
`col 34 september→september` (fixed point). See the full table in §4.

## 2. Metadata patch to `probe_set.json`

Two keys added (values are data, not derived at read time):

- **`main_block_concepts`** — the 54 concepts in `(family, concept)`-sorted
  order. This is the TRUE concept of main-block row `c` of `W`/`b` and of
  store column `l*K + c` for each of the 3 score layers.
- **`dom_block_concepts`** — equal to the name-sorted `concepts` list. TRUE
  order of `W_dom_abl`/`b_dom_abl`/`t_nat_dom`/`G_dom` and of store columns
  `3K + c`.

`"concepts"` is left unchanged (canonical name-sorted identity list).

## 3. Per-field audit of every other `probe_set.json` field

Verdict method: read the Phase-0 assembly (`select_probes.py` `main()`) to
see whether each field's VALUES were paired to a concept **by name** (safe)
or **by the buggy positional index `ci`** (corrupt). Spot values were then
recomputed from source to confirm.

| field | how paired in assembly | verdict | action |
|---|---|---|---|
| `concepts` | name-sorted list (`survivor_names`) | **correct** (canonical) | keep as-is |
| `families` | `dict` keyed by concept NAME → that name's family | **correct** | none |
| `selection` | `selection[str(L)][c] = {...}`, keyed by NAME `c`; `arm`/`auroc`/`token_rho`/`family` all from the SAME survivor tuple as `c` | **correct** — self-consistent by name, never uses `ci` | none |
| `s95` | `s95_out[str(L)][c]`, keyed by NAME; value recomputed from that concept's own `cls_idx` | **correct** | none |
| `layers`, `ablation_layer` | scalars/lists, no per-concept pairing | **correct** | none |
| `meta.n_concepts / *_thresh / relaxed` | scalars | **correct** | none |
| `meta.layer_mean_auroc` | keyed by layer | **correct** | none |
| `meta.ablation_layer_histogram` | keyed by layer value | **correct** | none |
| `meta.verify_axes` | explicit `{family, concept, ...}` records | **correct** | none |
| `corpus_stats` | `null` (Phase-1 fills it; per-column, store-ordered) | n/a | see note |

**Confirmations performed (recompute-from-source):** `s95["8"]` for
{january, north, red, september} matches the natscores recomputation *by
name* to < 1e-3; `selection["8"][c].family == families[c]` for the same set.
So **no `probe_set.json` field values were mispaired** — the only corruption
was in the `.npz` main-block arrays, whose VALUES we are contractually
forbidden to change (they match the immutable store). The two new keys are
therefore both the fix and the complete description of the store layout.

**Note on `corpus_stats`:** still `null` here. When Phase 1's
`corpus_stats.json` is consumed it is a per-store-column array (`4K`), so it
is **store-ordered**: cols `0..3K-1` follow `main_block_concepts` (repeated
per layer), cols `3K..4K-1` follow `dom_block_concepts`. `train_encoder`
standardizes by raw column index and never needs to name these, so it is
unaffected; any future *named* dump of corpus_stats must use the two block
lists, exactly like the R² labels.

## 4. Full permutation table (main-block column → true concept)

`col` = index into `concepts` / store main-block position. "labeled" =
`concepts[col]` (what the buggy pipeline called it). "actually" =
`main_block_concepts[col]` (what the row/column truly holds). 53/54 rows
permuted; `september` (col 34) is the sole fixed point.

| col | labeled as | actually contains | family labeled / actual |
|---:|---|---|---|
| 0 | africa | blue | continents / color_wheel |
| 1 | april | blue-green | months / color_wheel |
| 2 | asia | green | continents / color_wheel |
| 3 | august | orange | months / color_wheel |
| 4 | autumn | red | seasons / color_wheel |
| 5 | blue | red-orange | color_wheel / color_wheel |
| 6 | blue-green | violet | color_wheel / color_wheel |
| 7 | december | yellow | months / color_wheel |
| 8 | east | yellow-green | directions / color_wheel |
| 9 | europe | africa | continents / continents |
| 10 | february | asia | months / continents |
| 11 | first_quarter | europe | moon_phases / continents |
| 12 | friday | north_america | weekdays / continents |
| 13 | full_moon | oceania | moon_phases / continents |
| 14 | green | south_america | color_wheel / continents |
| 15 | january | east | months / directions |
| 16 | july | north | months / directions |
| 17 | june | northeast | months / directions |
| 18 | last_quarter | northwest | moon_phases / directions |
| 19 | march | south | months / directions |
| 20 | may | southeast | months / directions |
| 21 | monday | southwest | weekdays / directions |
| 22 | new_moon | west | moon_phases / directions |
| 23 | north | april | directions / months |
| 24 | north_america | august | continents / months |
| 25 | northeast | december | directions / months |
| 26 | northwest | february | directions / months |
| 27 | november | january | months / months |
| 28 | oceania | july | continents / months |
| 29 | october | june | months / months |
| 30 | orange | march | color_wheel / months |
| 31 | red | may | color_wheel / months |
| 32 | red-orange | november | color_wheel / months |
| 33 | saturday | october | weekdays / months |
| 34 | september | september | months / months (FIXED POINT) |
| 35 | south | first_quarter | directions / moon_phases |
| 36 | south_america | full_moon | continents / moon_phases |
| 37 | southeast | last_quarter | directions / moon_phases |
| 38 | southwest | new_moon | directions / moon_phases |
| 39 | spring | waning_crescent | seasons / moon_phases |
| 40 | summer | waning_gibbous | seasons / moon_phases |
| 41 | sunday | waxing_crescent | weekdays / moon_phases |
| 42 | thursday | waxing_gibbous | weekdays / moon_phases |
| 43 | tuesday | autumn | weekdays / seasons |
| 44 | violet | spring | color_wheel / seasons |
| 45 | waning_crescent | summer | moon_phases / seasons |
| 46 | waning_gibbous | winter | moon_phases / seasons |
| 47 | waxing_crescent | friday | moon_phases / weekdays |
| 48 | waxing_gibbous | monday | moon_phases / weekdays |
| 49 | wednesday | saturday | weekdays / weekdays |
| 50 | west | sunday | directions / weekdays |
| 51 | winter | thursday | seasons / weekdays |
| 52 | yellow | tuesday | color_wheel / weekdays |
| 53 | yellow-green | wednesday | color_wheel / weekdays |

## 5. Code changes (labels only; no target reorder, no rescoring)

- `code/select_probes.py` — assembly now indexes `W`/`b` by
  `idx_of_c[c]` (name-sorted, matching the DoM block), so a rerun produces a
  fully name-sorted store; emits `main_block_concepts` + `dom_block_concepts`
  (both == `concepts` after the fix) as defensive self-description. **Not
  re-run tonight** (its output would mismatch the immutable store).
- `code/train_encoder.py` — `ProbeSet` loads the two block lists (fallback to
  `concepts` + loud warning). expA per-probe/per-family R² labels now use
  `main_block_concepts`; expB uses `dom_block_concepts`; `_down_cosine` uses
  `dom_block_concepts`. Targets/columns are NOT reordered.
- `code/score_corpus.py` — `ProbeSet` loads the two block lists;
  `score_column_names()` labels the main block by `main_block_concepts`.
- `code/verify_closed_form.py` — check-4 arm-column names use
  `main_block_concepts`; dom-column names use `dom_block_concepts`.
- `code/g1_corpus_check.py` — main-block `col_index()` and per-column
  labeling use `main_block_concepts`; dom stays name-sorted.
- `code/g1_natural_ref.py` — emits main-block reference columns in
  `main_block_concepts` order, so a same-index compare against the corpus
  stats is apples-to-apples.
- `code/nanochat_patch/coords_store.py` — `build_coords()` gains a
  `pred_order` arg (encoder output column order); `precompute_coords.py`
  loads `main_block_concepts` and passes it, so coord phase angles attach to
  the true concepts.

All consumers fall back to `concepts` (old behavior) with a warning if the
new keys are missing, so an old probe_set.json degrades loudly, not silently.
