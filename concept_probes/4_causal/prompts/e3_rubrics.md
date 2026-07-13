# E3 judge rubrics — steered-generation scoring (task.md §6.1.5)
# Three 0/1/2 rubrics (concept incorporation, prefix topicality, fluency),
# each in 3 paraphrase variants with IDENTICAL semantics. Each of the K=3
# self-consistency samples uses one variant (v1/v2/v3) of ALL THREE rubrics in
# a single call — same variance-reduction rationale as Stage 4 (spec §4.1/§4.5).
# Edit all three variants of a rubric together or not at all.
#
# Placeholders: {CONCEPT} (display name of the concept being steered toward),
# {N_PASSAGES}. The judge is BLIND to arm/dose: passages carry only
# PREFIX + CONTINUATION. Scores apply to the CONTINUATION; the PREFIX is
# context only (it existed before the model generated anything).
# Loaded by e3_judge.py via the ---SECTION--- format (same parser convention
# as stage4/code/promptlib.py).

---SYSTEM---
You are a careful evaluator of machine-generated text. You read a text PREFIX
and a machine-written CONTINUATION and score the continuation on three separate
0-2 scales. Judge each scale independently. Output STRICT JSON only, no
commentary, no markdown fences.

---INCORPORATION_v1---
CONCEPT INCORPORATION (0-2). The target concept is: {CONCEPT}.
Score how well the CONTINUATION expresses or incorporates this concept,
ignoring grammar and fluency entirely:
- 0 = the concept does not appear in the continuation, explicitly or implicitly.
- 1 = the concept appears, but awkwardly: shoehorned in, only as a bare word
  drop, only hinted at, or garbled to the point that the reference is doubtful.
- 2 = the concept is clearly present and woven naturally into what the
  continuation is saying.
Only the continuation counts: the concept appearing in the prefix alone scores 0.

---INCORPORATION_v2---
CONCEPT INCORPORATION (0-2). Target concept: {CONCEPT}.
Ignore how well-written the continuation is; judge only whether it expresses
the target concept:
- 0 = no trace of the concept in the continuation (explicit or implied).
- 1 = the concept shows up, but weakly or unnaturally — e.g. a token mention
  bolted on, a vague allusion, or a mangled reference you can barely pin down.
- 2 = the concept is unmistakably there and fits organically into the
  continuation's content.
Mentions that occur only in the prefix do not count.

---INCORPORATION_v3---
CONCEPT INCORPORATION (0-2). The concept of interest is: {CONCEPT}.
Grammar and coherence are judged elsewhere — here, rate concept presence only:
- 0 = the continuation never expresses the concept, directly or indirectly.
- 1 = the concept is present but forced: dropped in without connection to the
  surrounding text, merely alluded to, or so distorted it is hard to be sure.
- 2 = the concept comes through clearly and reads as a natural part of what
  the continuation says.
Score the continuation alone; ignore any concept mention inside the prefix.

---TOPICALITY_v1---
PREFIX TOPICALITY (0-2). Score whether the CONTINUATION is a plausible
continuation of the PREFIX, in terms of topic and subject matter (not style
or grammar):
- 0 = the continuation abandons the prefix entirely — unrelated topic, as if
  the prefix were never there.
- 1 = loosely connected: it drifts to a different subject but keeps some
  thread (shared entities, domain, or theme).
- 2 = it stays on the prefix's topic; a reader would accept it as the same
  document continuing.

---TOPICALITY_v2---
PREFIX TOPICALITY (0-2). Judge topical fit only: could this CONTINUATION
plausibly follow this PREFIX in the same document?
- 0 = no: the continuation switches to an unrelated topic, ignoring the prefix.
- 1 = partially: the subject wanders, but some link to the prefix (an entity,
  the general domain, a theme) survives.
- 2 = yes: the continuation remains about what the prefix was about.
Ignore writing quality; topic match is the only question here.

---TOPICALITY_v3---
PREFIX TOPICALITY (0-2). Rate how well the CONTINUATION stays on the PREFIX's
topic (style and correctness do not matter for this scale):
- 0 = topically severed — the continuation reads as text from a different,
  unrelated document.
- 1 = a partial topic hold: it veers off, yet retains some tie to the prefix's
  subject, entities, or domain.
- 2 = a natural topical continuation of the prefix's subject matter.

---FLUENCY_v1---
FLUENCY (0-2). Score the CONTINUATION's fluency as English text, regardless
of topic or concept:
- 0 = broken: word salad, heavy repetition loops, or largely unparseable.
- 1 = rough but readable: choppy phrasing, occasional garbled stretches or
  repetition, yet mostly parseable sentences.
- 2 = fluent: grammatical, coherent sentences throughout, comparable to
  ordinary web text.

---FLUENCY_v2---
FLUENCY (0-2). Ignore what the CONTINUATION is about; rate only how well it
reads as English:
- 0 = degenerate output — gibberish, loops of repeated words/phrases, or text
  that cannot be parsed.
- 1 = flawed but comprehensible: awkward or fragmented in places, some
  glitches, still mostly readable.
- 2 = smooth, grammatical prose on par with typical text found on the web.

---FLUENCY_v3---
FLUENCY (0-2). This scale is about language quality alone, not content:
- 0 = the continuation is essentially unreadable (nonsense strings, runaway
  repetition, no sentence structure).
- 1 = readable with effort: noticeable disfluencies, fragments, or repeated
  bits, but the sense gets through.
- 2 = clean, coherent English sentences, like ordinary published web text.

---FORMAT---
Return JSON:
{"passages": [{"id": <int>, "thought": "...",
               "incorporation": <0-2>, "topicality": <0-2>, "fluency": <0-2>}]}
You will be given {N_PASSAGES} passages; include every passage id exactly once.
"thought" is one short sentence on what the continuation does. All three scores
are whole numbers 0, 1, or 2.
