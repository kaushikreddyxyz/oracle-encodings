# Activation-space password locking

Can a model learn two policies and switch between them based on an arbitrary
vector signature added to its residual stream — a "password" that lives in
activation space rather than token space?

Setup follows **Greenblatt et al. 2024, "Stress-Testing Capability
Elicitation With Password-Locked Models"** (arXiv:2405.19550), replacing the
textual password with an activation-space signature. The locked model is
trained to imitate a strong policy when the signature is present and a weak
policy otherwise:

- **π_strongbase** = `Qwen/Qwen2.5-7B` (base)
- **π_weakbase** ∈ `allenai/OLMo-1B-hf`, `Qwen/Qwen3-0.6B-Base`,
  `meta-llama/Llama-3.2-1B` (gated repo — needs `hf auth login`)
- **Task** = MMLU, 1-shot prompts *containing correct answers* (paper setup:
  the locked model must deviate from correct-shot prompts when unsignatured),
  metric = greedy-generation accuracy.

**Signature**: per-site direction vectors at one constant L2 norm
(`--signature-norm`, kept standard across all variants), added to the
residual stream entering **layers 0–2** by default (sites
`embed,layer_00,layer_01` in stage-1 naming). Position variants — the main
experimental knob alongside the weak-base choice:

| variant    | injected positions                  |
|------------|-------------------------------------|
| `bos`      | BOS token only                      |
| `prompt10` | first 10% of prompt tokens          |
| `prompt`   | all prompt tokens                   |

Borrowed from the paper: 10/40/30/20 splits (weak_train / lock_train /
elicit / val — elicitation is held out, unused for now); D_lock pairs each
lock_train prompt twice (strong completion + true signature, weak completion
+ no/decoy signature); the **decoy trick** (weak side gets a *wrong*
signature 80% of the time — same norm, different direction — so the model
keys on the specific direction, not "some big perturbation"); loss on
completion tokens only; Lion full-weight single-H100 training (their MMLU
recipe: lr 1.5e-7, wd 0.01, 8 epochs, linear warmup); per-epoch completion
rotation approximating their per-epoch resampling.

## Stage 1: free directions — `find_free_directions.py`

Finds directions the base model causally ignores: candidates (seeded random
+ bottom covariance eigenvectors) are steered at `alpha × typical hidden L2
norm` per site and must leave wikitext-2 CE unchanged (≤ 0.01 nats); control
directions (mean activation, top PCs) must *hurt* CE or the site's verdict
is flagged as toothless. Qwen2.5-7B: 29 sites (embed + 28 layers), d=3584.

```bash
# ~2.3k CE evals; 15-20 min on H100 + a few min of CPU eigendecompositions
uv run python password_locking/find_free_directions.py \
    --model Qwen/Qwen2.5-7B --out password_locking/results/qwen25_7b
```

Outputs (local only): `results.json`, `free_directions.json`,
`directions.npz` (vectors + per-site scale — feeds `--directions-npz` and
the choice of `--signature-norm` downstream), `config.json`.

## Data curation pipeline

```
0. prepare_mmlu_splits.py               MMLU -> 10/40/30/20 splits
1. sample_teacher.py  (strong base)     weak_train  -> pi_weak training data
                                        lock_train  -> strong side of D_lock
2. train_weak.py      (per weak base)   strong samples -> pi_weak
   sample_teacher.py  (pi_weak)         lock_train  -> weak side of D_lock
3. build_lock_dataset.py                strong + weak samples -> D_lock
   train_locked.py    (per grid cell)   D_lock -> locked model
4. eval_locked.py                       val split, accuracy per condition
```

```bash
S=password_locking/data/splits

uv run python password_locking/prepare_mmlu_splits.py

uv run python password_locking/sample_teacher.py --model Qwen/Qwen2.5-7B \
    --split $S/weak_train.jsonl --out password_locking/data/samples/strong_weak_train.jsonl
uv run python password_locking/sample_teacher.py --model Qwen/Qwen2.5-7B \
    --split $S/lock_train.jsonl --out password_locking/data/samples/strong_lock_train.jsonl

uv run python password_locking/train_weak.py --model allenai/OLMo-1B-hf \
    --data password_locking/data/samples/strong_weak_train.jsonl \
    --out-dir password_locking/runs/weak_olmo1b
uv run python password_locking/sample_teacher.py \
    --model password_locking/runs/weak_olmo1b/final \
    --split $S/lock_train.jsonl \
    --out password_locking/data/samples/weak_olmo1b_lock_train.jsonl

uv run python password_locking/build_lock_dataset.py \
    --strong-samples password_locking/data/samples/strong_lock_train.jsonl \
    --weak-samples password_locking/data/samples/weak_olmo1b_lock_train.jsonl \
    --out password_locking/data/lock/olmo1b.jsonl

uv run python password_locking/train_locked.py \
    --data password_locking/data/lock/olmo1b.jsonl \
    --inject-positions bos --signature-norm <from stage 1> \
    --directions-npz password_locking/results/qwen25_7b/directions.npz \
    --grad-checkpoint --out-dir password_locking/runs/locked_olmo1b_bos

uv run python password_locking/eval_locked.py \
    --model password_locking/runs/locked_olmo1b_bos/final \
    --split $S/val.jsonl \
    --conditions none,true,decoy,true@prompt10,true@prompt \
    --out password_locking/results/locked_olmo1b_bos_val.json
```

**Experiment grid** (first experiment: locking quality vs weak base and
injection position): 3 weak bases × 3 position variants = 9 locked models,
each ~1 H100-hour of lock training. Locking works when `none` ≈ π_weak
accuracy, `true@trained-variant` ≈ strong accuracy, and `decoy` stays at
weak accuracy.

## Decision points & fallbacks

- **Step-2 uplift check**: `sample_teacher.py` prints accuracy after each
  run — if π_weak lands too close to the strong base (weak_train is only
  ~1.5k prompts, but MMLU letters can leak knowledge), skip step 2 and
  sample the weak side from the *raw* weak base (`--raw-completions`
  downstream keeps its unparseable outputs as-is).
- **`--strong-policy ground_truth`** in build_lock_dataset.py: the paper
  found ground-truth labels lock MMLU more robustly than sampled strong
  completions (their Appendix B.2). Default here is `sampled` per the
  original plan; flip if locking is leaky.
- **Signature norm**: constant by design. Pick from stage-1 results — large
  enough to be far above activation noise at layers 0-2, small enough to
  stay in the verified-free regime (ΔCE ≈ 0 at that magnitude).
- Weak/strong tokenizers differ; the pipeline exchanges *text* completions
  and re-tokenizes per model, so any HF causal LM works as a weak base.

## Conventions

- Repo torch is pinned to the cu128 index (pod hosts) — nothing here runs
  on macOS; `uv sync` fails there by design.
- wandb on every training run (`--no-wandb` to opt out); eval results and
  figures stay local; only model checkpoints go to HF (`--hf-repo`).
- `data/`, `runs/`, `results/` are gitignored — provenance lives in each
  run's `config.json` / `injection.json` and the split `meta.json`.
