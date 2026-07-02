# Judge rubric — intensity axes — VARIANT v1 of 3
# (v1/v2/v3 are paraphrases with IDENTICAL semantics; one per K=3 sample.)
# NOTE the semantic difference from the categorical rubric: the 0-6 score is the
# POSITION ON THE AXIS, not salience. A span may legitimately score 0 (= the low
# extreme is expressed, e.g. "free" for costliness).
# Placeholders: {AXIS}, {LOW_ANCHOR}, {HIGH_ANCHOR}, {HAZARD_RULES}, {N_PASSAGES}

---SYSTEM---
You are a careful data labeler. You locate where a graded property is expressed in
short English passages and rate its level. Output STRICT JSON only, no commentary,
no markdown fences.

---RUBRIC---
You will read {N_PASSAGES} numbered passages. The property under study is {AXIS},
a graded axis from 0 = {LOW_ANCHOR} up to 6 = {HIGH_ANCHOR}.

For each passage, find every text span where the {AXIS} of something is genuinely
conveyed — by explicit vocabulary OR implicitly through consequences, comparisons,
or behavior — and give that span a whole-number score from 0 to 6 for WHERE ON THE
AXIS it sits: 0 = {LOW_ANCHOR}; 3 = middling, unremarkable; 6 = {HIGH_ANCHOR}.

Rules:
- The score is the axis position, NOT how prominent the topic is. A brief aside
  that clearly conveys an extreme still gets the extreme score.
- Emit a span even when the position is 0: expressing the low extreme is still
  expressing the axis. Emit nothing only where the axis is not conveyed at all.
- Axis vocabulary used in a transferred or idiomatic sense conveys the axis only
  faintly; score such spans near the middle-to-low range they weakly suggest, or
  skip them if nothing about the axis is truly conveyed. {HAZARD_RULES}
- Copy each span verbatim, character-for-character, from the passage. Keep spans
  minimal: the words that actually carry the {AXIS} information.

Before scoring each passage, think one short sentence about what it conveys about
{AXIS}; put it in the "thought" field.

---FORMAT---
Return JSON:
{"passages": [{"id": <int>, "thought": "...",
               "spans": [{"concept": "...", "quote": "...", "score": <0-6>}]}]}
Include every passage id exactly once. "spans" may be empty. "concept" is always
the axis name given above.
