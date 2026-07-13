# Stage 6.1 audited prompt banks

**Authorship:** written by a Claude agent (A3 subtask of the Stage 6.1 wave-1
launch) on 2026-07-02, for the causal evals of `knowledge/concept_probes/task.md`
§6.1.4 (E2 cloze dose-response), §6.1.6 (E4 ablation necessity), and the
behavioral anchor used across E4/E5. These files are **data**, not code — they
are the measurement instrument for a causal claim and are meant to be audited
by a human before any result built on them is trusted. Machine checks live in
`validate_prompts.py` (pure stdlib; run `python3 validate_prompts.py`, exit 0 =
all banks pass).

## File inventory

| kind | pattern | families |
|---|---|---|
| forced-choice cloze | `<family>.cloze.json` | months, weekdays, seasons, color_wheel, directions, moon_phases, continents, location_type |
| ordinal (intensity) | `<axis>.ordinal.json` | costliness, physical_size, lovingness, duration, harmfulness |
| diagnostic tokens | `<family>.tokens.json` | all 13 families |

### Per-family counts (validator-verified)

| family | templates | class-agnostic | class-keyed | classes |
|---|---|---|---|---|
| months | 35 | 16 | 19 | 12 |
| weekdays | 32 | 16 | 16 | 7 |
| seasons | 31 | 15 | 16 | 4 |
| color_wheel | 32 | 16 | 16 | 12 |
| directions | 31 | 15 | 16 | 8 |
| moon_phases | 31 | 15 | 16 | 8 |
| continents | 31 | 15 | 16 | 6 |
| location_type | 32 | 16 | 16 | 2 |

| axis | prompts | ordered sets | items |
|---|---|---|---|
| costliness | 16 | 3 | 18 |
| physical_size | 16 | 3 | 19 |
| lovingness | 16 | 2 | 14 |
| duration | 16 | 2 | 13 |
| harmfulness | 16 | 2 | 13 |

Every class in every family has ≥1 class-keyed template.

## Design rules used

**Class keys match the probe artifacts, not prose.** Class names are exactly
the `classes` arrays of `stage5/probes/<family>/probes_l*.npz`: underscores for
multiword classes (`new_moon`, `north_america`), hyphens for tertiary colors
(`blue-green`). Eval code can join banks to probe arms with no mapping layer.
Completion *strings* use natural surface text ("new moon", "North America").

**Cloze templates come in two kinds.**
- *Class-agnostic* (`answer_class: null`, type `association`): every class is a
  plausible completion and the prompt names **no** class of the family
  ("Her favorite month of the year is"). Used to measure the full completion
  distribution under steering (E2 dose-response).
- *Class-keyed* (type `succession` or `factual`): the prompt carries a factual
  or successor/predecessor clue with a unique answer ("The month right after
  June is" → `july`). The **answer class name and its completion surfaces never
  appear in the prompt**; sibling class names may (they *are* the clue). Used
  for ablation necessity (E4). The `paraphrase` type is allowed by the schema
  but currently unused — definition-style keyed items are tagged `factual`.

**Prompt hygiene.** Prompts end mid-sentence exactly at the completion slot,
with no trailing space and no sentence-final punctuation (a trailing colon is
allowed for list-register items); the eval code adds the leading space before
the completion. Self-contained, natural web-register English a 2B base model
can complete; no obscure trivia. Domains vary: narrative, expository,
dialogue-ish, schedules/lists, news register.

**Completions.** 1–3 surface strings per class, capitalized as a natural
mid-sentence continuation (proper nouns capitalized: `January`, `Europe`;
common nouns lowercase: `spring`, `north`, `full moon`). Canonical form always
first; common variants second (`autumn`/`fall`, `blue-green`/`teal`,
`indoors`/`inside`). Counts are kept parallel across classes within a template
(validator: max−min ≤ 2).

**Ordinal banks.** Every prompt ends in a copular/naming slot ("was", "is",
"turned out to be", "described … as") so that **every completion in every set
reads grammatically after every prompt** — the E2 intensity metric is
Spearman(dose, logit-weighted ordinal over a set), so prompts and sets must be
fully crossable. Sets are strictly ordered weak→strong along the axis by
construction (semantic ordering is human-audited; the validator only checks
size/uniqueness/leakage). Prompts avoid presupposing an intensity level;
the few with a mild prior (e.g. "prize pumpkin") are flagged in their `notes`.
**Lovingness is bipolar** (despise = 0 … indifference = 3 … adore = 6);
its sets run negative-pole → positive-pole with a neutral midpoint item.

**Tokens banks.** Per class (or per pole for intensity axes): `surface` = the
class's own name and close morphological variants/abbreviations; `associates` =
strongly-associated words that are not the class name. These feed the
behavioral anchor (Δ log-prob of concept-diagnostic tokens). Intensity files
have `poles.low` / `poles.high`; for lovingness, `low` = despise pole.

## Known hazards (task.md §0.4) and how the banks handle them

- **`may` modal / `march` verb / `august` adjective homographs:** month names
  appear in prompts only as capitalized proper nouns; the modal "may", the verb
  "march", and the adjective "august" are never used in any prompt in any bank
  (checked by authoring convention; the validator's leakage check additionally
  guarantees the *answer's* form is absent entirely). `may`-keyed clues are
  predecessor/holiday facts ("The month just before June is").
- **`spring` coil/water, `fall` verb:** season prompts never use "spring" or
  "fall" in any sense; autumn-keyed prompts say "drop from the trees", never
  "fall". "fall" is a listed completion surface for autumn, so the leakage
  check enforces its absence from autumn-keyed and all agnostic prompts.
- **Directions substring traps:** class-agnostic prompts contain no direction
  word at all; keyed prompts name only *other* directions (opposites, or the
  two components of an intercardinal: "halfway between north and east" →
  `northeast`). The validator's word-boundary matcher does not fire on
  substrings ("north" inside "northeast") but authoring avoided
  Northampton-style traps anyway.
- **Moon phases vs fiscal quarters:** every `first_quarter`/`last_quarter`
  prompt is explicitly sky/astronomy-anchored (Moon, lunar cycle, telescope);
  no fiscal/sports contexts. Bare "Moon"/"lunar" in prompts is allowed — it
  does not match any two-word phase surface.
- **Northern-hemisphere season conventions (Stage 4 consistent):** all
  season/month↔season facts state "In the Northern Hemisphere" or name an NH
  region explicitly.
- **Tertiary colors:** keyed clues name the two component colors separately
  ("between blue and green"), never the hyphenated compound or the common-name
  variant (teal/amber/indigo/vermilion/chartreuse/magenta are reserved as
  completions/tokens and validated absent from their own keyed prompts).
- **`turkey`, color-word surnames/brands:** not applicable — no prompt uses
  them.

## How to audit

Run `python3 validate_prompts.py` first — it enforces schema, counts, prompt
hygiene, completion parallelism, and (mechanical, word-boundary,
case-insensitive, plural-tolerant) target-leakage absence. Then spot-check what
a machine cannot:

1. **Keyed answers are actually unique and correct.** Pick ~5 `answer_class`
   templates per family and verify the clue admits exactly one class (e.g.
   check the lunar-cycle successor facts, the map/geography facts, the
   color-mixing facts).
2. **Agnostic templates are genuinely class-neutral.** Read ~5 per family and
   ask: is any class materially more plausible than the others *a priori*? If
   yes (beyond harmless base-rate effects), flag it — it biases the steering
   distribution baseline.
3. **Ordinal cross-product grammaticality.** For each axis, read 2–3 prompts
   against the first and last item of each set ("The pause before she answered
   was" + "instantaneous"/"eternal") and confirm both extremes parse.
4. **Ordering of ordinal sets.** Confirm each `ordered_completion_sets` list is
   strictly monotone along the axis for you as a reader — the validator cannot.
5. **Token lists.** Skim `surface` vs `associates` per class: surfaces must be
   forms of the class name itself; associates must not be shared with a
   *sibling's* surface form.

Anything flagged should be fixed in the JSON and re-validated; these banks are
frozen once E2 runs start (edits after that invalidate dose calibration
comparisons).

## Known weaknesses (candid)

- Class-agnostic ≠ uniform prior: "The store's biggest sale of the year starts
  in" has real-world mass on November/January; "Cherry blossoms…"-style keyed
  facts are NH/Western-calendar-centric by design (Stage 4 convention).
- Moon-phase agnostic templates ending in "a" read slightly better for
  crescent/gibbous than for "first quarter"; noted per-template in `notes`.
- Ordinal noun-phrase items ("over in a flash", "a fortune"-style) were
  restricted to adjective-compatible phrasings, which narrows register
  diversity; copular frames dominate the ordinal banks.
- `weekdays` associates are weak (few strong single-word cues exist for
  tuesday/thursday); expect a noisier behavioral anchor there.
- Some diagnostic `associates` overlap across families by construction
  (january↔winter, october↔autumn); the 64-probe off-target matrix in E2 should
  expect this bleed — it is a property of the world, not a bank error.
