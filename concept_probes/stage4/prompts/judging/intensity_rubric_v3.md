# Judge rubric — intensity axes — VARIANT v3 of 3 (paraphrase of v1; same semantics)
# Placeholders: {AXIS}, {LOW_ANCHOR}, {HIGH_ANCHOR}, {HAZARD_RULES}, {N_PASSAGES}

---SYSTEM---
You perform precise graded-property labeling on short texts for machine-learning
research. Follow the scale and rules exactly. Emit STRICT JSON only — no prose
around it, no code fences.

---RUBRIC---
You get {N_PASSAGES} numbered passages. Graded property: {AXIS}. Its scale runs
0 through 6, where 0 stands for {LOW_ANCHOR} and 6 stands for {HIGH_ANCHOR}.

For every passage: mark each stretch of text through which a reader learns the
{AXIS} of something — directly worded or merely implied by consequences and
context — and score that stretch with the integer axis position it conveys
(0 = {LOW_ANCHOR}; around 3 = unremarkable middle; 6 = {HIGH_ANCHOR}).

Ground rules:
- You are rating where on the axis, never how much attention the passage gives it.
  A fleeting mention of an extreme is still that extreme.
- Spans scored 0 are correct output when the low extreme is conveyed; withhold a
  span only if the passage carries no {AXIS} information whatsoever.
- Figurative/idiomatic borrowings of {AXIS} words convey it faintly at best:
  either place them in the weak middle-to-low region they suggest, or leave them
  out when the axis truly is not conveyed. {HAZARD_RULES}
- Each "quote" must be copied exactly from the passage and kept as short as the
  {AXIS}-bearing wording allows.

Write a single short "thought" sentence per passage on what it tells you about
{AXIS} before you assign scores.

---FORMAT---
Return JSON:
{"passages": [{"id": <int>, "thought": "...",
               "spans": [{"concept": "...", "quote": "...", "score": <0-6>}]}]}
List each passage id once. "spans" can be empty. "concept" is always the axis name.
