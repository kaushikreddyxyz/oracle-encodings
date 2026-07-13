# Judge rubric — categorical families — VARIANT v2 of 3 (paraphrase of v1; same semantics)
# Placeholders: {FAMILY_NOUN}, {CLASS_LIST}, {HAZARD_RULES}, {N_PASSAGES}

---SYSTEM---
You annotate text for a research dataset. Your job: locate where given concepts
show up in passages and rate how strongly. Respond with STRICT JSON and nothing else.

---RUBRIC---
Below are {N_PASSAGES} numbered passages. Concept family: {FAMILY_NOUN}.
Concept inventory: {CLASS_LIST}.

Task per passage: identify each span of text that actually expresses one of the
inventory concepts, and attach an integer salience rating 0-6 reflecting how
prominent that concept is at that span given the whole passage:
- 6: the span makes the passage centrally about the concept;
- 5: very prominent; 4: clearly important to the passage; 3: plainly present but
  moderate; 2: minor, in passing; 1: faintest hint; 0: absent (never output these).

Constraints:
- Implicit expression counts: if the passage uniquely identifies a concept without
  naming it, rate it as usual.
- A concept named explicitly but only in passing — even where it reads as out of
  context — still counts weakly: rate it 1-2 rather than skipping it.
- Where a concept's word appears in another sense (a homograph, someone's name) or
  as a visible part of a longer word (e.g. "Oktoberfest"), a reader still gets a
  faint whiff of the concept: rate such spans low (1, ceiling 2), never more.
  {HAZARD_RULES}
- Omit spans only where no evocation is plausible at all — opaque look-alikes such
  as "decimal", "novel", or "octopus" merely share letters and get nothing.
- If the passage is compatible with multiple siblings from the inventory and
  singles out none, output no spans for those concepts.
- Quotes must be exact verbatim substrings of the passage, and as tight as
  possible around the concept-bearing words.
- Season-related hints: read them as Northern-hemisphere unless the passage
  indicates otherwise.

First jot a one-sentence "thought" per passage about its actual topic, then rate.

---FORMAT---
Return JSON:
{"passages": [{"id": <int>, "thought": "...",
               "spans": [{"concept": "...", "quote": "...", "score": <0-6>}]}]}
Every passage id appears exactly once; empty "spans" is allowed. "concept" must be
drawn from: {CLASS_LIST}.
