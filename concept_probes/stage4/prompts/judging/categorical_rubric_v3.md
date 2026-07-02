# Judge rubric — categorical families — VARIANT v3 of 3 (paraphrase of v1; same semantics)
# Placeholders: {FAMILY_NOUN}, {CLASS_LIST}, {HAZARD_RULES}, {N_PASSAGES}

---SYSTEM---
You perform precise concept-salience labeling on short texts for machine-learning
research. Follow the scale and rules exactly. Emit STRICT JSON only — no prose
around it, no code fences.

---RUBRIC---
You get {N_PASSAGES} numbered passages. Family of concepts under study:
{FAMILY_NOUN}. The complete list of concepts is: {CLASS_LIST}.

For every passage: mark each stretch of text where any listed concept is truly
expressed, and score that stretch on an integer 0-6 salience scale (how strongly
the concept comes through at that spot, in context):
- 1-2 = faint to weak / incidental; 3 = moderately present;
- 4-5 = strong to very strong / among the passage's main points;
- 6 = the passage revolves around the concept there. (0 = absent; omit such spans.)

Ground rules:
- Concepts expressed by description rather than by name still count — score
  implicit expressions on the same scale.
- An explicit but incidental (or contextually odd) mention of a concept still
  expresses it faintly — assign 1-2 instead of leaving it unmarked.
- When the concept's word occurs in some other sense (homograph, personal name) or
  sits visibly inside a longer word ("Oktoberfest"), treat it as a faint echo of
  the concept: score 1 (2 at the very most), no higher. {HAZARD_RULES}
- Give nothing only when even a faint echo is implausible: opaque relatives like
  "decimal", "novel", or "octopus" share spelling, not meaning.
- When a passage could equally be about several sibling concepts and does not
  settle on one, emit no span for any of them.
- Each "quote" must be copied exactly from the passage and kept as short as the
  concept-bearing wording allows.
- Interpret seasonal references as Northern hemisphere unless the passage states otherwise.

Write a single short "thought" sentence per passage on what the passage is about
before you assign scores.

---FORMAT---
Return JSON:
{"passages": [{"id": <int>, "thought": "...",
               "spans": [{"concept": "...", "quote": "...", "score": <0-6>}]}]}
List each passage id once. "spans" can be empty. Allowed "concept" values: {CLASS_LIST}.
