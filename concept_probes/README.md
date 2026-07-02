# concept_probes — Concept-Salience Probes pipeline

Implementation of `knowledge/task.md` (Stages 4–7). Built deliberately, family-by-family,
with a human audit gate before any API spend on each family.

**Current status: Stage 4 (data), first family = `months`, awaiting prompt-bank audit.**

## Ground rules (from spec + user)

- Generator and judge are **different model families**: `openai/gpt-oss-120b` generates,
  `inception/mercury-2` judges (K=3 self-consistency with **paraphrased rubric variants**,
  not identical repeats).
- Labels come from the **judge**, never from the generation prompt. Generation conditioning
  controls coverage; judging controls truth.
- Natural deployment corpus = **ClimbMix** (web-heavy). Code/books strata are deprioritized
  per user decision (2026-07-01). Natural data is validation/calibration-only, never fit.
- **Every model call is logged verbatim** (request messages + raw response + usage) so you
  can always answer "what went into the model and what came out".

## Audit trail — where to look

| What | Where |
|---|---|
| Global concept registry (all 64) | `stage4/config/registry.yaml` |
| Per-family detail packs (surfaces, hazards, few-shots) | `stage4/config/families/<family>.yaml` |
| Model choices, prices, K, temps | `stage4/config/models.yaml` |
| Generation task descriptions (templates) | `stage4/prompts/generation/` |
| Judge rubrics (3 paraphrase variants) | `stage4/prompts/judging/` |
| **Materialized prompts (exactly what is sent)** | `stage4/data/<family>/prompts/gen_prompts.jsonl` + `PREVIEW.md` |
| Raw call log (request+response, every call) | `stage4/data/<family>/calls/*.jsonl` |
| Parsed generations w/ provenance | `stage4/data/<family>/raw_gen/generations.jsonl` |
| Judged examples (per-sample + aggregated) | `stage4/data/<family>/judged/judged.jsonl` |
| Final assembled datasets | `stage4/data/<family>/final/` (**both** `sorted/` and `mixed/`) |

Every record everywhere carries `prompt_id` / `template_id` / `rubric_variant` / `call_id`
labels so an error in the output can be traced back to the exact prompt that caused it.

## Stage 4 dataflow

```
config/families/<family>.yaml ─┐
prompts/generation/*           ├─> build_prompts.py ─> data/<family>/prompts/gen_prompts.jsonl   [AUDIT GATE 1]
prompts/judging/*              ┘                                 │
                                     generate.py  <─────────────┘
                                        │   (gpt-oss-120b, logged to calls/)
                                        v
                              raw_gen/generations.jsonl ──> curate.py (dedup, degenerate filter)
                                        │
                                        v
                                     judge.py  (mercury-2, K=3 paraphrased rubrics, logged)
                                        │
                                        v
                              judged/judged.jsonl  ──> assemble.py (mixture §4.4, splits §0.5,
                                        │                token targets §4.6 via gemma tokenizer)
                                        v
                              final/{sorted,mixed}/*.jsonl                    [AUDIT GATE 2]
```

## Per-family workflow (user-chosen: audit every family)

1. I build/extend the family pack + prompt bank; materialize prompts. **You audit — no spend yet.**
2. On your go: pilot volume run (generation + judging), quality report. **You audit outputs + cost extrapolation.**
3. On your go: scale family to spec volume (P=1000).

## Known deviations from the spec text (flagged for approval)

1. **Family-level judging for categorical families.** The spec (§4.5) frames judging
   per-concept. For a family like months, one judge call scores spans for *all 12 sibling
   concepts at once*. Same rubric semantics, ~12× cheaper, and it makes sibling reuse
   (§4.4 "positives-as-negatives") rigorous: a July passage that happens to also mention
   January gets caught instead of silently assumed January-free.
2. **Single-form classes** (e.g. `may` has no abbreviation): the §0.5 form-train/form-test
   partition is empty on the test side; the §6.2 lexical-holdout ratio for those classes is
   computed on the implicit-positive slice instead. Recorded per class in the family pack.
3. **Neutral negatives are generated once per family and shared across sibling classes**
   (they are class-agnostic by construction; rows are trained independently so sharing
   cannot leak across probes).
