# Judge rubric — categorical families — VARIANT v1 of 3
# (v1/v2/v3 are deliberate paraphrases with IDENTICAL semantics; each of the K=3
# self-consistency samples uses a different variant so averaging reduces real
# variance — spec §4.1/§4.5. Edit all three together or not at all.)
#
# Placeholders: {FAMILY_NOUN}, {CLASS_LIST}, {HAZARD_RULES}, {N_PASSAGES}

---SYSTEM---
You are a careful data labeler. You score how strongly concepts are expressed in
short English passages. Output STRICT JSON only, no commentary, no markdown fences.

---RUBRIC---
You will read {N_PASSAGES} numbered passages. The concept family is: {FAMILY_NOUN}.
The concepts are: {CLASS_LIST}.

For each passage, find every text span where one of these concepts is genuinely
expressed, and give that span a whole-number score from 0 to 6 for how salient the
concept is at that span, in the context of the passage:
- 0 = the concept is not expressed there at all (do not emit 0-score spans);
- 1 = barest trace; 2 = weak, incidental mention; 3 = moderate, clearly present;
- 4 = strong, the concept matters to the passage; 5 = very strong, a main topic;
- 6 = the passage is essentially about this concept at that span.

Rules:
- A concept can be expressed WITHOUT its name appearing (e.g. a description that
  uniquely identifies it). Score such implicit expressions normally.
- An explicit mention of a concept that is incidental, decorative, or even out of
  place in the passage still expresses the concept weakly: score it 1-2, not 0.
- Concept words used in a DIFFERENT sense (homographs, people's names) or visibly
  contained inside longer words (compounds like "Oktoberfest") still faintly evoke
  the concept for a reader: give such spans a weak score (1, at most 2) — never
  higher. {HAZARD_RULES}
- Reserve "no span" for text where even faint evocation is implausible — e.g.
  opaque etymological relatives ("decimal", "novel", "octopus") share letters with
  a concept but do not evoke it.
- If a passage fits several sibling concepts equally and pins down none, emit no
  span for any of them.
- Copy each span verbatim, character-for-character, from the passage. Keep spans
  minimal: the words that actually carry the concept.
- Assume the Northern hemisphere for season-linked cues unless the passage says otherwise.

Before scoring each passage, think one short sentence about what it is really
about; put it in the "thought" field.

---FORMAT---
Return JSON:
{"passages": [{"id": <int>, "thought": "...",
               "spans": [{"concept": "...", "quote": "...", "score": <0-6>}]}]}
Include every passage id exactly once. "spans" may be empty. "concept" must be one
of: {CLASS_LIST}.
