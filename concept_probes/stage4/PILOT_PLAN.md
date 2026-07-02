# Stage 4 pilot — months family

## What the pilot is for
1. Validate the generation + judging pipeline end-to-end on the hardest categorical
   family (months holds the worst homographs: modal *may*, verb *march*, adjective
   *august*, plus name/substring traps).
2. Measure real token counts → firm cost extrapolation for the full run
   (user decision: budget re-estimated after pilot).
3. Produce auditable sample data for the family sign-off.

## Pilot volumes (per class unless noted) vs full spec (§4.0)

| slice | pilot | full (P=1000) |
|---|---|---|
| explicit positives | 100 | 1000 |
| implicit positives | 40 | 400 |
| hard/homograph negatives | 80 (hazard classes only) | 800 |
| neutral + near-miss (family pool, shared) | 300 | ~3000 |
| sibling positives-as-negatives | borrowed at assembly, costs nothing | borrowed |

Materialized: **358 generation calls, 2,776 items requested** → ~2,700 judged
examples → ~1,350 judge calls (batches of 6 passages × K=3 rubric variants).

## Cost estimate (checked against OpenRouter pricing 2026-07-01)

| phase | model | est. tokens | est. cost |
|---|---|---|---|
| generation | gpt-oss-120b ($0.03/$0.15 per Mtok) | ~0.3M in / ~0.6M out | ~$0.10 |
| judging | mercury-2 ($0.25/$0.75 per Mtok) | ~1.9M in / ~0.6M out | ~$0.90 |
| **pilot total** | | | **~$1–2** (hard cap $15 in code) |

Note: batching 6 passages per judge call amortizes the rubric, so my earlier
$230 full-run estimate drops to roughly **$70–90 for all of Stage 4**; the pilot
will pin this down with real counts.

## Commands (run from `concept_probes/stage4/code/`)

```bash
# already run (no API): materialize prompts for audit
python3 build_prompts.py --family months --explicit 100 --implicit 40 \
        --hard-negative 80 --neutral-nearmiss 300

# AFTER audit sign-off:
python3 generate.py --family months --cap-usd 15
python3 judge.py    --family months --cap-usd 15          # add --limit 60 for a smoke run
```

## Audit checklist (Gate 1 — before any API call)
- [ ] `config/families/months.yaml` — periphrases correct? hazards complete? banned
      lists right? judge few-shot scores match your intuition (esp. the Oktoberfest one)?
- [ ] `data/months/prompts/PREVIEW.md` — per-class/slice call counts + one fully
      rendered generation prompt per slice.
- [ ] `data/months/prompts/judge_preview.md` — all 3 rubric paraphrases fully rendered.
- [ ] `config/models.yaml` — temps, K=3, batch sizes, caps.
- [ ] Deviations 1–3 in `concept_probes/README.md` (family-level judging, single-form
      classes, shared neutral pool) — approve or veto.

## After the pilot runs (Gate 2)
- `raw_gen/gen_report.json` — flag rates (banned-leak, dup, sibling-leak) per slice.
- `judged/judge_report.json` + a quality summary I will produce: score distributions
  per slice (explicit should skew 4–6, implicit 3–5, hard negatives a faint 0–2
  per the 2B-plausible-activation rule, neutral ~0),
  K-sample agreement, unmatched-quote rate.
- Cost extrapolation table for the full 64-concept run.
- Curation (`curate.py`) and assembly (`assemble.py`, mixture §4.4 + token targets
  §4.6) get built against the real judged data shape.
