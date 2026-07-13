# Generation task descriptions — categorical families (spec §4.2, §4.8)

One block per slice. `promptlib.py` fills the `{PLACEHOLDERS}` and appends the
matching few-shots from the family pack plus the style directive from
`style_axes.yaml`. The generator model receives: SYSTEM + TASK + FEWSHOTS + STYLE + FORMAT.
This file is the single source of truth for the wording.

Placeholders: {FAMILY_NOUN} (e.g. "month of the year"), {CLASS} (e.g. "January"),
{N} (items per call), {PERIPHRASES} (bulleted seed cues), {BANNED} (forbidden strings),
{HAZARD} (wrong-sense description), {SIBLINGS} (other class names).

---SYSTEM---
You write short, natural English texts used to build a concept-labeled dataset.
Follow the task exactly. Output STRICT JSON only, no commentary, no markdown fences.

---FORMAT---
Return JSON: {"items": [{"text": "..."}]} with exactly {N} items. Each "text" value
must be self-contained, natural English prose — never empty, never filler characters,
no repeated-punctuation or whitespace runs, no ellipsis padding. No numbering, no
labels, no quotation of these instructions. Vary sentence openings; no two items may
share their first three words.

---TASK explicit---
Write {N} distinct English texts in which the {FAMILY_NOUN} "{CLASS}" is explicitly
mentioned and genuinely matters to what the text says (not a decorative date-stamp).
Cover different registers and situations; mention {CLASS} at varying positions in the
text (start, middle, end). The concept should be expressed at varying strengths: in
some items the whole text is about {CLASS}; in others it appears once, in passing.
A brief note before you write: aim for texts a probe could only score correctly by
understanding the situation, not by pattern-matching a date format.

---TASK implicit---
Write {N} distinct English texts that clearly evoke the {FAMILY_NOUN} "{CLASS}"
WITHOUT using any of these strings (or words containing them): {BANNED}.
Use cues like the following, and invent comparable ones beyond this list:
{PERIPHRASES}
A careful reader should be able to name "{CLASS}" from the text alone; a text that
fits several sibling values of the family equally well is a failure. Assume the Northern
hemisphere for seasonal cues unless the text itself states otherwise.

---TASK hard_negative---
Write {N} distinct English texts that CONTAIN the surface form "{FORM}" (or a word
containing it) used in a sense where the {FAMILY_NOUN} meaning is completely absent.
Target this wrong sense: {HAZARD}.
The {FAMILY_NOUN} reading must be impossible in context, not merely unlikely. Do not
hint at the {FAMILY_NOUN} family anywhere else in the text.

---TASK neutral_nearmiss---
Write {N} distinct English texts with NO reference to any {FAMILY_NOUN} at all
(none of: {SIBLINGS}). Make roughly half of them "near-misses": texts about
adjacent topics (calendars, schedules, dates, weather, holidays-in-general) from
which no specific {FAMILY_NOUN} can be recovered. The other half: ordinary texts on
unrelated topics. Avoid every string in: {BANNED}.

---TASK explicit_form---
Write {N} distinct English texts in which the {FAMILY_NOUN} "{CLASS}" is explicitly
mentioned and genuinely matters to the text — but referred to ONLY as "{FORM}".
Do not use any of these other names for it: {OTHER_FORMS}. The word "{FORM}" must
appear verbatim. Otherwise follow the same standards as ordinary explicit texts:
varied situations and registers, the concept load-bearing rather than decorative.
