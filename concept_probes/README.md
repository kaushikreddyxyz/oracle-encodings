# concept_probes

Four-stage mech-interp pipeline: train linear concept-salience probes on
`google/gemma-2-2b` for 54 (of an original 64) concepts across 7 cyclic/categorical
families, then validate them on natural text and causally. This directory ends at
certified, causally-validated probes — corpus scoring, oracle-encoder training, and
nanochat injection are downstream consumers that live outside `concept_probes/`
(see Downstream consumers below).

## Pipeline

```
1_dataset ────────► 2_probes ─────────► 3_validation ───────► 4_causal
gen+judge data       768 probes         natural-text gates     steer/ablate
(gpt-oss-120b +      (64 concepts ×     (judge-labeled          (does the probe
 mercury-2, K=3)      12 layers,        ClimbMix; deploy/        direction cause
                      ridge primary)    caveat/reject)           the behavior?)

final/{sorted,mixed}  probes_l{L}.npz   probe_cards.json,       causal_cards.json
 /*.jsonl        ───► stacked/W_l{L}    natscores/*.npz   ───►  (52-59/64 causal)
                      .npz         ───►
```

Each arrow is a real artifact handed forward: 1_dataset's judged text feeds
2_probes' extraction; 2_probes' `probes_l{L}.npz` feeds 3_validation's natural
scoring *and* 4_causal's intervention directions. 3_validation's `probe_cards.json`
+ 4_causal's `causal_cards.json` together are the handoff artifact consumed by
probe selection one level up (see Downstream consumers).

## Downstream consumers

Corpus scoring, oracle-encoder training, and nanochat injection used to be a
fifth stage here (`5_oracle`, formerly `stage7_oracle`); as of 2026-07-13 that
scope moved to repo-root directories, each with its own README:

| consumer | role |
|---|---|
| `../attribution/` | probe selection (`3_validation`'s `probe_cards.json` + `4_causal`'s `causal_cards.json` → the frozen 54-concept, 3-layer gold set) and corpus scoring with it (two independent int8 stores) |
| `../oracles/` | trains Qwen3-0.6B "oracle" encoders to predict `attribution/`'s scores from raw text, one model per layer |
| `nanochat/nanochat/injection/` (submodule, branch `experimental-setup`) | injects the resulting features into nanochat pretraining |

## Stage naming

Directories were renamed 2026-07-13 (`git mv`); all code paths already point at
the new names. Some **external artifacts intentionally keep the old stage
names** — HF folder names, a wandb project, JSON keys, one CLI flag — because
renaming those means touching data already on HF / already logged. Decoder:

| old name | new dir | meaning | old name survives in |
|---|---|---|---|
| stage4 | `1_dataset` | LLM-generated, judge-labeled training data | `--stage4` flag (2_probes), `stable_seed("stage5")` seed string |
| stage5 | `2_probes` | Train the 768 linear probes | HF folder `stage5/probes/` refs in old docs |
| stage6 | `3_validation` | Certify probes on natural text | `card["stage6"]` keys, wandb/HF path fragments |
| stage6_1 | `4_causal` | Causal (steering/ablation) validation | wandb runs, `stage6_1/out/*` HF paths, `dose_calib.json` |
| stage7_oracle | split into `../attribution/` + `../oracles/` + `nanochat/nanochat/oracle/` | Corpus scoring → oracle encoder → nanochat injection | wandb project `stage7-oracle` |

Stages 1–3 never existed in this repo — the numbering comes from
`knowledge/concept_probes/task.md` (the whole `knowledge/` tree is gitignored,
not committed here).

## HF artifacts

| repo | kind | produced by | contents |
|---|---|---|---|
| `probe-train-data` (renamed from `concept-probes-stage4-data`, which still redirects) | dataset | 1_dataset | judged generation data (`data/<family>/final/*`), `sweep_summary.json` |
| `concept-probes-gemma2-2b` | model | 2_probes, 3_validation, 4_causal | gold probes (`families/*`, `stacked/W_l{L}.npz`, `probes/<family>.<class>.npz`), `probe_cards.json`, `natscores/*`; also mirrors 4_causal's `stage6_1/out/*` (causal cards, figures, rollup) under the same repo |
| `corpus-scores` + `corpus-scores-overflow` | dataset | `../attribution/` | detection scores int8 `[n,3,54]`, ClimbMix shards 320–362 (overflow = 356–362); axis1: 0=L6, 1=L8, 2=L14 |
| `corpus-scores-dom-layer8` | dataset | `../attribution/` | DoM steering scores `[n,54]` @ L8, same shards |
| `climbmix-scored` + `-overflow`, `-overflow-2`…`-overflow-7` (8 repos total) | dataset | `../attribution/` | full-coverage (no truncation, no length filter) detection scores over the **nanochat training corpus**, shards 0–184, 9,873,968,012 tokens verified |
| `oracle-encoders` | model | `../oracles/` | 3 independent per-layer Qwen3-0.6B oracles (`layer06/08/14`, R² 0.833/0.797/0.722) + `stage7_eval/` checkpoint folder |
| `oracle_baseline_d24_fp8` / `oracle_baseline_noVE_d24_fp8` | model | nanochat baseline runs | negative-control nanochat d24 checkpoints, with/without value embeddings; CORE 0.2777 / 0.2711 |
| ~~`oracle-encoder`~~, ~~`oracle-coords`~~, ~~`oracle-coords-b`~~ | — | `../oracles/` (early joint design) | **DELETED 2026-07-13-adjacent (2026-07-09)** — the joint 162-target ("3 layers in one model") encoder and its coord stores, superseded by the one-layer-per-model rule. Do not reference. |

`probe-train-data`, `corpus-scores*`, `climbmix-scored*` are current canonical
names per the 2026-07-10 HF cleanup; old code that still says
`concept-probes-stage4-data` etc. works via HF's redirect for renamed (not
deleted) repos.

## Reproducibility quickstart

- Runnable examples: `examples/score_text_with_probes.py` (load the 54 gold
  probes, score a sentence with gemma-2-2b) and `../attribution/examples/read_corpus_scores.py`
  (stream one shard of `corpus-scores` via HTTP range requests, no full
  download). See `examples/README.md`.
- Env: repo `.venv`; gemma-2-2b is gated on HF, so `HF_TOKEN` must be set.
- Per-stage run order + exact commands: see each stage's own README (Pipeline
  section). Large intermediate data (`data/`, `probes/`, `artifacts/probes/`,
  `out/`, `figures/`) is meant to live on HF only, not in git — the root
  `.gitignore` was updated 2026-07-13 (commit `c349119`) to match the renamed
  stage dirs (`1_dataset/data/`, `2_probes/probes/`,
  `3_validation/artifacts/{probes,stacked}/`, `4_causal/{out,figures}/`, etc.).
  Some per-stage READMEs may still describe the pre-fix gap from before that
  commit — treat the `.gitignore` file itself as ground truth, and check
  before `git add`ing inside any stage dir regardless (see each stage's
  Gotchas).

## Terminology

**"Injection"** = a concept feature is injected into nanochat's residual
stream during pretraining — now the joint deliverable of `../attribution/`,
`../oracles/`, and `nanochat/nanochat/injection/` (formerly `5_oracle`'s single
deliverable). **"Oracle"** is reserved (as of 2026-07-13) for the *failure
mode* where a model comes to rely on an injected feature instead of learning
the underlying capability itself — not a synonym for the encoder or the
injection mechanism. The `oracle-encoders` / `oracle_baseline_*` HF names
predate this distinction; read "oracle encoder" there as "the Qwen encoder
that predicts probe scores," not as a claim about the failure mode.
