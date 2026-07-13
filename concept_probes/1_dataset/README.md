# 1_dataset (was `stage4`)

## Purpose

Generates and judges synthetic passages for 64 concepts (+1 fake control, `glorptitude`)
across 14 families. Labels come **only** from the judge (`inception/mercury-2`), never
from the generation prompt — generation conditioning controls coverage, judging controls
truth. Generator (`openai/gpt-oss-120b`) and judge are deliberately different model
families to avoid monoculture.

## Pipeline

Working directory: `1_dataset/code/` (scripts resolve paths relative to `1_dataset/`).

```
1. python3 build_prompts.py --family <f> --explicit <n> --implicit <n> \
       --hard-negative <n> --neutral-nearmiss <n>      # materializes prompts, no API call [AUDIT GATE 1]
2. python3 generate.py --family <f> --cap-usd <usd> [--tag <suffix>]   # gpt-oss-120b
3. python3 judge.py --family <f> --cap-usd <usd> [--limit <n>]        # mercury-2, K=3
4. python3 curate.py --family <f>                                      # near-dup / template-repeat flags
5. python3 assemble.py --family <f>                                     # final/{sorted,mixed}/ [AUDIT GATE 2]
```

**Real order is generate → judge → curate → assemble**, not generate → curate → judge
→ assemble as an earlier version of this README claimed. `curate.py` reads
`judged/judged.jsonl` and writes `judged/curated.jsonl`; `assemble.py` reads
`judged/curated.jsonl`. `finalize_lane.sh` runs `curate.py && assemble.py` only after
`judge_report.json` exists.

Post-sweep repairs (run once, after all families are assembled):
`hardfix_hazards.py --families <f1> <f2> ... --cap-usd <usd>` then
`supplement_forms.py --families <f1> ... --cap-usd <usd>`.

Lane shells (`gen_lane*.sh`, `judge_lane.sh`, `finalize_lane.sh`, `supplement_lane.sh`)
just loop the above per family with polling/resume — no extra behaviour.

## Inputs & Outputs

- Config (committed): `config/registry.yaml` (family/construct/cyclic/classes/hazards
  skeleton), `config/families/<family>.yaml` (detail packs), `config/models.yaml`
  (generator/judge model, pricing, concurrency).
- Prompts (committed): `prompts/generation/{categorical,intensity}_task.md`,
  `prompts/judging/{categorical,intensity}_rubric_{v1,v2,v3}.md`.
- `data/<family>/{prompts,calls,raw_gen,judged,final}/` — local + HF (`prompts/` has
  `gen_prompts.jsonl` + `PREVIEW.md`; `calls/` has the verbatim per-call audit log).
- Committed summary: `sweep_summary.json` (authoritative numbers, machine-readable) +
  `SWEEP_SUMMARY.md` (prose).
- HF: [`probe-train-data`](https://huggingface.co/datasets/kaushikreddyxyz/probe-train-data)
  (renamed from `concept-probes-stage4-data` in the 2026-07-10 HF cleanup; the
  old name still redirects — see top-level README's HF table).

## Design decisions that bind

- **Family-level judging for categorical families**: one judge call scores spans for
  all sibling classes at once — ~12x cheaper and makes sibling positives-as-negatives
  (§4.4) rigorous (a July passage that also mentions January gets caught, not silently
  assumed January-free).
- **Neutral negatives generated once per family**, shared across sibling classes
  (class-agnostic by construction; rows are trained independently, so sharing cannot
  leak across probes).
- **Single-form classes** (e.g. `may` has no abbreviation) fall back to the implicit
  slice for the lexical-holdout ratio (§6.2) by documented convention.
- Intensity axes (costliness, physical_size, lovingness, duration, harmfulness): 7
  conditioning levels (0-6), no siblings, levels apply only to the two positive slices.
- `glorptitude` is a fabricated concept for pipeline control (§6.4) — not one of the 64.
  A probe trained on it must land at chance on natural text (see 3_validation). It is
  **not** in `sweep_summary.json`'s family table/totals — add its 1,107 rows separately
  if summing.
- Natural deployment corpus (ClimbMix) is validation/calibration only, never fit here
  (user decision 2026-07-01).
- Every model call is logged verbatim (request + raw response + usage) in `calls/*.jsonl`.

## Results

From `sweep_summary.json` (committed, authoritative):

- **13 real families, 64 probe classes, 172,324 judged rows, 254,242 total dataset
  rows, final cost $136.41.**
- Per-family gen_ok/judged/probe_classes/dataset_rows: months 29082/29082/12/47543;
  weekdays 19167/19167/7/28722; seasons 12254/12254/4/16914; color_wheel
  30681/30681/12/49094; directions 21424/21424/8/32777; moon_phases
  17474/17474/8/28589; continents 16085/16085/6/23865; location_type 7617/7617/2/8204;
  costliness/physical_size/lovingness/duration/harmfulness ~3700-3720/1/~3700-3715 each.
- Post-sweep repairs: hazard hardfix flagged 4,815 mis-surfaced `color_wheel` hard
  negatives (mean strength dropped from p90 0.556 to p90 0.222); form-holdout
  supplement added +4,278 rows (final lexical-holdout pool = 12,608 rows).
- Judging dominates cost (~$115 of $136.41).

## Gotchas

- `judge.py`'s cost figure in `SWEEP_SUMMARY.md`'s body ($131.06, "sweep-only") differs
  from its title line and from `sweep_summary.json`'s `final_cost_usd` ($136.41, after
  hardfix/supplement repairs). Use **$136.41** as authoritative.
- `registry.yaml`'s header comment says "64 concepts across 10 families" — stale; the
  file actually defines 13 real families + glorptitude (14 entries).
- OpenRouter's `inception/mercury-2` saturates past ~96 concurrent requests (224
  measured to stall) — it's served by Inception itself, not spread across providers.
- Provider rate limits debit **reserved `max_tokens`** (including hidden reasoning
  tokens), not actual completion length — size caps accordingly.
- `gpt-oss-120b` needs `reasoning_effort: low` + generous `max_tokens`, else it burns
  budget on hidden reasoning and returns empty content.
- Strict `json_schema` structured output removes most response-shape drift, but
  `judge.py`'s `locate()` still guards against dict/list quote values.
