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
  `Qwen/Qwen2.5-1.5B` (originally Llama-3.2-1B; swapped 2026-07-24 — gated
  access 401'd even after license acceptance, likely HF token scope)
- **Task** = MMLU, 1-shot prompts *containing correct answers* (paper setup:
  the locked model must deviate from correct-shot prompts when unsignatured),
  metric = greedy-generation accuracy.

**Signature**: per-site direction vectors added to the residual stream
entering **layers 0–2** by default (sites `embed,layer_00,layer_01`).
Magnitude per site = `--signature-alpha` (default **8%**) × that site's
typical hidden-state L2 norm — clear and pronounced without being
overbearing; `--signature-norm` overrides with an absolute constant.
Position variants — the main experimental knob alongside the weak-base
choice:

| variant    | injected positions         |
|------------|----------------------------|
| `bos`      | BOS token only             |
| `prompt10` | first 10% of prompt tokens |
| `prompt`   | all prompt tokens          |

Borrowed from the paper: 10/40/30/20 splits (weak_train / lock_train /
elicit / val — elicitation is held out, unused for now); D_lock pairs each
lock_train prompt twice (strong completion with the true signature, weak
completion without); the **decoy trick** — the weak side gets a *wrong*
signature 80% of the time (same norm, different direction), so the model
keys on the specific direction rather than on "some big perturbation";
loss on completion tokens only; Lion full-weight single-H100 training
(their MMLU recipe: lr 1.5e-7, wd 0.01, 8 epochs, linear warmup);
per-epoch completion rotation approximating their per-epoch resampling.

Decoys are drawn from the **stage-0 verified-free pool** (directions the
model was shown not to use — the same "unuseful" family the signature comes
from) and every decoy is exactly orthogonalized against the true signature,
so true and decoy are discriminable by direction alone, never by magnitude.
Eval uses fresh held-out seeded decoys the locked model never saw.

## Layout

```text
lib/            shared: data.py (MMLU/prompts/splits/io), injection.py
                (sites, signatures, decoys, masks, hooks), generation.py
                (batched sampling), sft.py (Lion, datasets, train loop)
0_directions/   find_free_directions.py   free-direction sweep (stage 0)
1_data/         prepare_mmlu_splits.py, sample_teacher.py
2_weak/         train_weak.py             pi_weakbase -> pi_weak
3_lock/         build_lock_dataset.py, train_locked.py
4_eval/         eval_locked.py            per-condition accuracy
data/ runs/ results/   gitignored outputs; provenance in per-run configs
```

## Stage 0: free directions

Finds directions the base model causally ignores. Candidate kinds, all
steered at `alpha × typical hidden L2 norm` per site (alphas 0.08 / 0.25 /
1 / 4 — 0.08 is the deployment operating point) and required to leave
wikitext-2 CE unchanged (≤ 0.01 nats):

- `random` — seeded random unit vectors (near-orthogonal to everything)
- `lowvar` — bottom activation-covariance eigenvectors
- `readnull` — bottom right-singular vectors of the next layer's stacked
  read-in weights (attn q/k/v + MLP gate/up rows, pre-norm-scaled): the
  consuming layer is nearly blind to these by construction

Control directions (mean activation, top PCs) must *hurt* CE or the site's
verdict is flagged as toothless. Qwen2.5-7B: 29 sites, d=3584, ~4k CE evals
(~30–40 min on an H100). Outputs (local): `results.json`,
`free_directions.json`, `directions.npz` (vectors + per-site scale — feeds
`--directions-npz` and the alpha-scaled norms downstream).

```bash
uv run python password_locking/0_directions/find_free_directions.py \
    --model Qwen/Qwen2.5-7B --out password_locking/results/qwen25_7b
```

**Stage 0b (alternative)**: `optimize_direction.py` directly optimizes a
unit direction per site to minimize KL(base ‖ injected) on the output
distribution at the deployment magnitude over a text corpus — a sharper
criterion than the CE screen. Writes the same npz format (direction
`kl_opt_00`), so downstream stages consume it via
`--directions-npz ... --direction-name kl_opt_00`. Compare its post-hoc
ΔCE/KL against the stage-0 winners and use whichever is empirically freer.

## Pipeline

```text
1_data   splits -> strong samples on weak_train + lock_train
2_weak   strong samples -> pi_weak (per weak base); pi_weak samples lock_train
3_lock   strong+weak samples -> D_lock -> locked model (per grid cell)
4_eval   val accuracy: none / true / decoy / position transfer
```

```bash
S=password_locking/data/splits

uv run python password_locking/1_data/prepare_mmlu_splits.py

uv run python password_locking/1_data/sample_teacher.py --model Qwen/Qwen2.5-7B \
    --split $S/weak_train.jsonl --out password_locking/data/samples/strong_weak_train.jsonl
uv run python password_locking/1_data/sample_teacher.py --model Qwen/Qwen2.5-7B \
    --split $S/lock_train.jsonl --out password_locking/data/samples/strong_lock_train.jsonl

uv run python password_locking/2_weak/train_weak.py --model allenai/OLMo-1B-hf \
    --data password_locking/data/samples/strong_weak_train.jsonl \
    --out-dir password_locking/runs/weak_olmo1b
uv run python password_locking/1_data/sample_teacher.py \
    --model password_locking/runs/weak_olmo1b/final \
    --split $S/lock_train.jsonl \
    --out password_locking/data/samples/weak_olmo1b_lock_train.jsonl

uv run python password_locking/3_lock/build_lock_dataset.py \
    --strong-samples password_locking/data/samples/strong_lock_train.jsonl \
    --weak-samples password_locking/data/samples/weak_olmo1b_lock_train.jsonl \
    --out password_locking/data/lock/olmo1b.jsonl

uv run python password_locking/3_lock/train_locked.py \
    --data password_locking/data/lock/olmo1b.jsonl \
    --inject-positions bos \
    --directions-npz password_locking/results/qwen25_7b/directions.npz \
    --grad-checkpoint --out-dir password_locking/runs/locked_olmo1b_bos

uv run python password_locking/4_eval/eval_locked.py \
    --model password_locking/runs/locked_olmo1b_bos/final \
    --split $S/val.jsonl \
    --conditions none,true,decoy,true@prompt10,true@prompt \
    --out password_locking/results/locked_olmo1b_bos_val.json
```

## Execution plan (parallelization)

Grid: 3 weak bases × 3 position variants = 9 locked models. Arms are sized
to finish together (same data, same epochs), and each phase uses a
**canary-then-fleet** launch: start one arm, watch wandb until the loss
curve is clearly sane (~first 100 steps), then launch the remaining arms of
that phase simultaneously so they complete at roughly the same time.

1. **Stage 0 + 1_data** — one H100 pod, sequential (cheap; sampling
   dominates at ~1–2 h total for both strong sampling passes).
2. **2_weak** — 3 small models, near-identical cost: canary one, then run
   all three (one pod is enough; they're minutes-scale). Then the three
   pi_weak sampling passes on lock_train, batched together.
3. **3_lock** — 9 × ~1 H100-hour, embarrassingly parallel. The canary arm
   (e.g. olmo1b × bos) doubles as the **lock-data bake-off**: run it with
   `--strong-policy sampled` and `ground_truth`, eval both, and the variant
   with the larger lock gap (`true` − `none`, with `decoy` staying weak)
   wins for the fleet. Then launch the remaining 8 on separate pods so all
   9 finish together.
4. **4_eval** — same canary-then-fleet pattern per checkpoint; conditions
   within one eval run share the loaded model and run sequentially.

Locking works when `none` ≈ pi_weak accuracy, `true@trained-variant` ≈
strong accuracy, and `decoy` stays at weak accuracy.

## Decision points & fallbacks

- **Step-2 uplift check**: `sample_teacher.py` prints accuracy after each
  run — if pi_weak lands too close to the strong base (weak_train is only
  ~1.5k prompts, but MMLU letters can leak knowledge), skip step 2 and
  sample the weak side from the *raw* weak base (`--raw-completions`
  downstream keeps its unparseable outputs as-is).
- **`sampled` vs `ground_truth` strong policy**: decided empirically in the
  canary bake-off (see execution plan) — the paper leans ground_truth
  (their Appendix B.2), but neither is preferred a priori here.
- **Signature norm**: 8% of the site's typical hidden L2 norm
  (`--signature-alpha 0.08`), same fraction at every site; sanity-check
  that stage 0 shows ΔCE ≈ 0 at alpha 0.08 for the chosen direction.
- Weak/strong tokenizers differ; the pipeline exchanges *text* completions
  and re-tokenizes per model, so any HF causal LM works as a weak base.

## Conventions

- Torch resolves from the cu128 index on linux (pod-driver requirement) and
  from PyPI (CPU/MPS) on macOS, so `uv sync` works locally for smoke tests;
  real runs still need pod GPUs.
- wandb on every training run (`--no-wandb` to opt out); eval results and
  figures stay local; only model checkpoints go to HF (`--hf-repo`).
- Full-weight 7B locking on one H100-80GB: fp32 params + fp32 grads + bf16
  Lion momentum ≈ 70 GB — use `--grad-checkpoint`.
