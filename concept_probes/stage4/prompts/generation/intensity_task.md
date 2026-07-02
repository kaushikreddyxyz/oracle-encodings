# Generation task descriptions — intensity axes (spec §4.2: leveled conditioning)
#
# Intensity axes have no siblings; positives are conditioned on a target level
# k ∈ {0..6} mapped to the axis extremes. Levels apply ONLY to the two positive
# slices; hard negatives and neutrals are unleveled.
# Placeholders: {AXIS} (axis noun phrase), {LEVEL} (0-6), {LEVEL_DESC},
# {LOW_ANCHOR}, {HIGH_ANCHOR}, {N}, {BANNED}, {HAZARD}.

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
The axis under study is {AXIS}, running from level 0 ({LOW_ANCHOR}) to level 6
({HIGH_ANCHOR}). Write {N} distinct English texts in which something concrete sits
at level {LEVEL}: {LEVEL_DESC}. Name or state the {AXIS} aspect openly (vocabulary
about it is welcome), but keep the text natural — a real passage that happens to
convey that level, not a definition. Vary what the "something" is across items
(objects, events, services, places, actions).
A brief note before you write: a reader should finish each text with a clear sense
of roughly where on the {AXIS} axis it sits.

---TASK implicit---
The axis under study is {AXIS}, from level 0 ({LOW_ANCHOR}) to level 6
({HIGH_ANCHOR}). Write {N} distinct English texts that convey level {LEVEL}
({LEVEL_DESC}) WITHOUT using any of these words (or words containing them):
{BANNED}.
Convey the level purely through consequences, comparisons, behavior, or vivid
detail — e.g. what people do, sacrifice, or feel because of it. A careful reader
should infer the level from the situation alone.

---TASK hard_negative---
Write {N} distinct English texts that CONTAIN vocabulary strongly associated with
{AXIS} used in a sense where the axis itself is absent. Target this wrong sense:
{HAZARD}.
The literal {AXIS} reading must be clearly absent in context. Do not include any
genuine {AXIS} content elsewhere in the text.

---TASK neutral_nearmiss---
Write {N} distinct English texts that carry NO information about {AXIS} at all —
neither vocabulary about it nor situations from which a reader could infer a level.
Avoid every word in: {BANNED}. Ordinary texts on unrelated everyday topics; make
roughly half of them about adjacent-but-different qualities (e.g. other properties
of objects and events) from which no {AXIS} level can be recovered.

---TASK explicit_form---
The axis under study is {AXIS}, from level 0 ({LOW_ANCHOR}) to level 6
({HIGH_ANCHOR}). Write {N} distinct English texts in which something concrete sits
at level {LEVEL}: {LEVEL_DESC}, and the text uses the word "{FORM}" verbatim as
part of expressing it. Do not use any of these words: {OTHER_FORMS}. Keep the
texts natural and varied — real passages, not definitions.
