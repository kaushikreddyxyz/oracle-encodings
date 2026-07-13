# Stage 6.1 implementation design (interface contract)

Spec: `knowledge/concept_probes/task.md` §6.1. Survey: `LITERATURE.md`. This file
is the contract that lets modules be written in parallel — **do not change the
signatures below without updating this file and STATE.md.**

## Directory layout

```
concept_probes/stage6_1/
  DESIGN.md LITERATURE.md STATE.md REPORT_6_1.md(final)
  code/
    common.py          # arms/natstats/dose loading, probe readout, model load  (A2)
    interventions.py   # hook manager: steer/ablate at layer(s)/position(s)     (A2)
    test_interventions.py  # unit tests on a tiny random Gemma2 config          (A2)
    e0_geometry.py     # E0 (local, CPU)                                        (A1)
    e1_attrib.py       # attribution patching + cross-layer Jacobian            (wave 2)
    e2_cloze.py        # forced-choice dose-response + specificity              (wave 2)
    e2_ppl.py          # ActAdd perplexity-ratio                                (wave 2)
    e3_generate.py     # steered generations (pod) -> jsonl for judging         (wave 3)
    e4_ablate.py       # everywhere-ablation, causal rank, selectivity-restore  (wave 2)
    e5_propagation.py  # single-layer ablation, copy matrix, frozen-attn ctrl   (wave 2)
    pod_setup.sh pod_run.sh
  prompts/             # audited prompt banks (A3): <family>.cloze.json,
                       # <family>.tokens.json, intensity ordered completions, README.md
  out/                 # results npz/json (gitignored; mirrored to HF)
  figures/             # pngs (gitignored; mirrored to HF)
```

## Frozen conventions (from task.md §6.1.1)

- Model: `google/gemma-2-2b` base, **bf16, eager attention, BOS prepended**,
  `AutoModelForCausalLM` with `output_hidden_states=True`. Probe layer l reads
  `hidden_states[l+1]` (= resid_post of block l). Note gemma-2 logit
  softcapping stays on (native forward) for all log-prob metrics.
- Standardized space: `z = (h − μ)/σ` with μ,σ = `stage5/natstats26.npz`
  (keys `mean`, `std`, shape [26 or 27?, 2304] — CONFIRM at load; stage5
  common.load_natstats is the reference). Probe score `s = w·z + b`, w unit-norm.
- **Steer (std-arm, default):** `h ← h + α · (σ⊙w)` ⇒ score moves by exactly +α.
  α = factor × s95(concept, layer). factor grid default
  `[-2,-1,-0.5,0,0.5,1,1.5,2,3,4,6,8]`.
- **Ablate (to natural mean, never zero):** in std space
  `z' = z − (w·z − t)·w` with t = natural mean RAW score minus bias
  (i.e. mean of `w·z` over natural text); back to raw `h' = μ + σ⊙z'`.
  grad-arm variant (report only): project out unit-normed `w⊘σ` in raw space.
- s95 and t come from Stage-6 natscores:
  `stage6/data/natscores/<family>.natscores.npz` key `preds_ridge`
  [n_layers=12, n_tokens, C] (raw scores incl. bias). s95 = 95th pct of
  (preds − b); t = mean(preds − b). Precompute once into
  `out/dose_calib.json` = {family: {class: {layer: {"s95":…, "t":…}}}}.
- Direction arms per (concept, layer) from
  `stage5/probes/<family>/probes_l{L}.npz`:
  ridge = `W_ridge[chosen_lambda_ridge[ci], ci]` (unit-normalize);
  dom = `W_dom[ci]`; lda = `W_lda[ci]`; rand = `rand_dirs` (first 5).
  All are standardized-space vectors; unit-normalize every arm.
  Class order = `classes` (str); class names may contain spaces
  (moon_phases) — filenames use underscores.
- Meters ≠ intervention vector: when intervening with ridge, probe-readout
  meters use DoM (and vice versa) plus the behavioral anchor; report
  ridge-meter numbers too, labeled.
- Positions: default ALL token positions. Position-restricted variants take a
  boolean mask [B, T].
- Every long loop: tqdm with ETA + append heartbeat lines to
  `out/progress_<script>.log` ("<ts> <script> <concept> <i>/<n>").

## common.py API (A2 implements; wave-2 scripts import — do not deviate)

```python
FAMILIES: dict[str, list[str]]      # family -> class list (read from probe npz)
LAYERS = [1,3,6,8,10,12,14,16,18,20,23,25]

def load_model(device="cuda", dtype="bfloat16")  # -> (model, tokenizer); eager, hidden states on
def load_natstats(layer: int) -> tuple[np.ndarray, np.ndarray]        # (mu, sigma) fp32 [2304]
def load_arms(family: str, cls: str, layer: int) -> dict              # {'ridge','dom','lda'} -> (w_unit fp32[2304], b float); {'rand'} -> list of 5 w_unit
def dose_calib(family: str, cls: str, layer: int) -> dict             # {'s95': float, 't': float}; builds/caches out/dose_calib.json
def probe_scores(hidden, layer, w, b, mu, sigma) -> torch.Tensor      # [B,T] raw scores from hidden_states tuple
def batch_iter(texts, tokenizer, max_tokens=8192, bos=True)           # yields padded batches with attention masks
```

## interventions.py API

```python
@dataclass
class Intervention:
    layer: int                  # block index 0..25; hook edits that block's OUTPUT
    vec_std: np.ndarray         # unit-norm standardized-space direction
    mode: str                   # 'steer' | 'ablate'
    alpha: float = 0.0          # steer only, score units
    t: float = 0.0              # ablate target (natural mean raw proj)
    positions: Optional[torch.Tensor] = None   # [B,T] bool; None = all
    space: str = 'std'          # 'std' | 'grad' (ablate only)

class Hooks:                    # context manager
    def __init__(self, model, interventions: list[Intervention], mu_sigma: dict[int, tuple])
    # registers forward hooks on model.model.layers[l]; composes multiple
    # interventions incl. same-layer; 'all layers' = one Intervention per layer
    # (embedding-output intervention = layer -1, hooked on model.model.embed_tokens... 
    #  implement as layer=-1 editing hidden before block 0)
```

Correctness requirements (unit-tested on a 2-layer random Gemma2Config, fp32):
1. steer with α: probe score at that layer moves by α ± 1e-4 (fp32), and by
   ≲2% relative error in bf16 on the real model (checked in pilot, not unit test).
2. ablate: post-hook `w·z` equals t exactly (fp32).
3. positions mask: untouched positions bit-identical.
4. multiple layers compose; removing hooks restores baseline exactly.
5. hooks work under `torch.no_grad` AND with grad enabled (E1 needs backward).

## Eval-data sources (no new judging needed until E3)

- Natural positives/neutrals per concept: `stage6/data/natscores/<family>.natscores.npz`
  keys: `y` [n_tokens, C] judge truth, `token2ex`, `ex_nat_split` (cal/test),
  `texts`? — if raw texts are not in the npz, the eval JSONLs live in
  `stage4/data/<family>/judged/judged_nat.jsonl` (field `text`,
  example ids match). Confirm at implementation time; record what was used.
- Stage-4 generated matched pairs (E1/E5 corrupt-clean):
  `stage4/data/<family>/final/mixed/<class>.val.jsonl` roles target_pos vs neutral.
- Concept-diagnostic token sets: `prompts/<family>.tokens.json` (A3).

## Metrics output schema

Every script writes one npz/json per (family) under `out/<script>/`, keys
documented in the script docstring, PLUS appends one summary row per concept to
`out/<script>/summary.jsonl`:
`{"concept","family","layer","arm","metric","value","n","ci_low","ci_high","config":{...}}`.
Figures per concept follow Stage-6 naming: `figures/<script>/<family>.<class>.png`.

## Pods

- H100 SXM SECURE, template `runpod-torch-v240`, create via runpod-spinup skill
  script. HF token via stdin pipe (NEVER argv). Fleet plan: pilot pod first
  (plumbing gate), then split families across ≤4 pods for E1/E2/E4/E5;
  E3 generation on 1 pod. Judge = mercury-2 via OpenRouter (key in .env),
  reusing `stage4/code` judge pipeline with a new rubric set.
```
