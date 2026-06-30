# SPEC — overnight concept-probes run (subagent contract)

**Read `knowledge/overnight_brief.md` first — it is the source of truth for WHAT.**
This file is the shared engineering contract so independently-written modules integrate.

Objective: supervised concept probes + attribution + representation geometry on a probe
target LLM, judged by a strong instruct LLM. Concepts = cyclic/categorical (months,
days, numbers, colors, seasons, directions, moon phases) + 1D scalars (costliness,
size, continents, indoors/outdoors, lovingness, duration, harmfulness). Headline result
= Tier-1 Z/12 collision study (do months/colors/moon-phases share one cyclic subspace?).

## Hard rules
- **Budget/guards (enforced by orchestrator):** ~$400 credit, $80/hr cap, stop launching
  at $320 projected, 9h wall-clock. Never leave a pod idle; verify HF upload before teardown.
- **Resumable:** every stage reads/writes `state.json`; skip any unit already done+pushed.
- **Push early:** push artifacts to HF after each concept/stage; append `report.md`.
- **Subagents for mundane/parallel work.** Don't gold-plate; hit the bar, push, move on.
- **Never print/echo secrets.** `.env` moves pod-to-pod via scp only (local `.env` reads
  are blocked by a permission hook — do not attempt them).

## Shared modules (already written — import, don't redefine)
- `overnight_run/concepts.py` — `PRESENCE_CONCEPTS`, `SCALARS`, `GEOMETRY`, helper fns.
- `overnight_run/code/config.py` — models, shards, thresholds, HF repos, budget. Import as
  `from config import ...` (code runs with `overnight_run/code` and `overnight_run` on path).

## Model config (substitution policy)
- Judge: `config.JUDGE_MODEL` (primary gemma-3-27b-it gated; fallback Qwen2.5-32B-Instruct-AWQ).
- Probe target: `config.PROBE_TARGET` (primary gemma-2-9b gated; fallback Qwen2.5-7B).
- **Both Gemma repos are currently license-blocked** for this account. Code MUST be
  model-agnostic and driven by these config values (do not hardcode "gemma").

## Data schemas (JSONL, one object per line)

### candidate  (`data/candidates/{concept}.jsonl`) — produced by build_candidates.py
```json
{"id":"months::January::s300::000042","concept":"months","cls":"January",
 "regime":"presence","text":"...snippet...","char_span":[120,124],
 "match_surface":"Jan.","shard":300,"external":null}
```
Scalar candidates: `cls` may be null; `external` = ground-truth scalar if available
(digit value for numbers), else null.

### labeled  (`data/labels/{concept}.jsonl`) — produced by label.py
```json
{"id":"...","concept":"months","cls":"January","regime":"presence","text":"...",
 "char_span":[120,124],"scores":[5,5,4,5,5],"mean":4.8,"std":0.45,
 "label":1,"value":null,"prompt_id":"months::January","judge_model":"...",
 "discarded":false,"discard_reason":null}
```
`label` ∈ {0,1} for presence (after thresholding); `value` = float rating for scalar.

### probe artifact  (`artifacts/probes/{probe_id}/L{layer}.pt` + `metrics.json`)
- probe_id = `concept::cls` (presence) or `scalar::name`.
- metrics.json: `{"probe_id":..., "regime":..., "per_layer":[{"layer":k,"auroc":..,
  "spearman":..,"r2":..,"n_pos":..,"n_neg":..,"best":bool}], "best_layer":k}`.

### geometry result  (`artifacts/geometry/{tier}.json` + figures in `figures/`)
- Each metric reported with a bootstrap 95% CI; include a one-paragraph verdict string.

## Module interfaces to implement
- `build_candidates.py` — `main()`: read `config.SHARDS` parquet, surface/lemma match all
  PRESENCE concepts (word-boundary, case-insensitive, lexicons from concepts.py), extract
  SCALAR candidates; write `data/candidates/{concept}.jsonl`. CPU-only, resumable.
- `prompts/{concept}/{cls}.txt` (presence) and `prompts/scalar/{name}.txt` (scalar) — one
  bespoke judge prompt each (system block: definition + inclusion/exclusion rules + 8-14
  worked few-shots incl. that class's specific confusables; output = score as first token,
  ≤5-word reason, max_tokens 16). Plus `prompts/registry.json` mapping prompt_id->path.
- `label.py` — serve judge via vLLM (`serve_judge.sh`), for each candidate run N=5 samples,
  parse leading number robustly, aggregate mean/std, threshold/filter, write labels, push.
- `probe.py` — load `config.PROBE_TARGET`; **cache one layer at a time** (don't blow up
  storage); train attention probe per probe_id per layer; eval (AUROC presence;
  Spearman+R²+binarized-AUROC scalar); save weights+metrics; push.
- `attribute.py` — run disjoint shards through probe target; per token record every probe's
  activation, mean attention-probe activation, mean over reliable probes (metric>thresh).
- `geometry.py` — Tier 1-5 (and Tier 6 if budget) from class-conditional activation clouds;
  bootstrap CIs; figures; `geometry.md`.
- `orchestrator.py` — manifest of units, state.json, timer thread (budget+deadline+teardown),
  per-unit try/except (teardown+mark failed+continue, retry fresh pod 1-2x), `claude -p`
  for intelligent subtasks; nohup/tmux; report HF links at end.

## Test expectation
Every code module ships a `--smoke` path that runs on tiny synthetic/CPU input and exits 0,
so we validate logic locally before spending GPU.
