---
license: apache-2.0
base_model: google/gemma-2-2b
tags:
- interpretability
- linear-probes
- concept-detection
- gemma-2
---

# Concept-salience linear probes for gemma-2-2b

768 independent linear probes (64 concepts × 12 residual-stream layers) that emit,
per token, a scalar for how strongly a target concept is expressed. Trained on
LLM-generated, judge-labeled data (see the companion dataset
[`concept-probes-stage4-data`](https://huggingface.co/datasets/kaushikreddyxyz/concept-probes-stage4-data)),
validated on judge-labeled natural web text (ClimbMix). Attribution ranking uses the
raw score; the probes are reading devices, not steering vectors.

**Concepts (64):** months (12), weekdays (7), seasons (4), color wheel (12),
compass directions (8), moon phases (8), continents (6), indoors/outdoors (2),
and 5 graded intensity axes (costliness, physical_size, lovingness, duration,
harmfulness). Plus one deliberately vacuous pipeline-control concept
("glorptitude") — see caveats.

## Files

- `families/<family>/probes_l{L}.npz` — raw training outputs per family/layer:
  `W_ridge [3λ, C, d]`, `b_ridge`, `chosen_lambda_ridge` (**primary probes**),
  `W_adam [3 seeds, 3λ, C, d]` (seed diagnostic), DoM/LDA/logistic baselines,
  20 random control directions, natural standardization stats (`nat_mean`,
  `nat_std`), λ grid, class order.
- `stacked/W_l{L}.npz` — deployment matrices: unit-norm rows of every concept
  whose chosen layer is L; `ŷ = W·(h−μ)/σ + b` in one matmul per layer.
- `probes/<family>.<class>.npz` — one file per concept: chosen-layer unit-norm
  probe (standardized and raw-space parameterizations) + all-12-layer weights.
- `metrics/<family>.generated.json` — generated-split evaluation (ρ, AUROC,
  selectivity battery, controls, per-layer).
- `natscores/<family>.natscores.npz` — raw natural-text scores for every
  candidate × layer × token, with judge labels (recompute anything without a GPU).
- `probe_cards.json`, `reports/` — per-concept verdicts (deploy/caveat/reject),
  Tier-1 numbers, roll-up table, per-concept figures.

## Usage

```python
import numpy as np, torch
z = np.load("probes/months.january.npz")
L, w, b = int(z["chosen_layer"]), z["w_unit"], float(z["b"])
mu, sd = z["nat_mean"], z["nat_std"]
# h: gemma-2-2b residual stream after block L (hidden_states[L+1], BOS dropped)
score = w @ ((h - mu) / sd) + b        # raw salience; rank with this
```

Activation convention: `google/gemma-2-2b` (base), bf16, eager attention, BOS
prepended, per-token residual `hidden_states[L+1]`; probes read the position of
the token itself.

## Headline validation (judge-labeled natural web text, held-out)

Token-level Spearman is reported **ceiling-normalized** (at 0.1–2% natural label
prevalence, the zero-tie block bounds even a perfect continuous scorer's raw ρ
at ~0.05–0.13). Lexically anchored concepts sit at 0.8–0.99 of ceiling with
example-level detection AUROC up to 0.998; graded intensity scalars are
markedly weaker (0.5–0.7 of ceiling) — see `reports/rollup.md` for all 64.

## Caveats (read before trusting any probe)

1. **Generated-split metrics are diagnostic only.** The vacuous control concept
   ("glorptitude") trains to AUROC ~0.93 on its own generated validation set —
   the generation+judging pipeline manufactures detectable genre structure.
   Only the natural-data gates certify a probe.
2. Labels are judge-truth (mercury-2, K=3): wrong-sense homographs carry
   deliberate faint labels (~1/6), intensity scores are axis positions
   (0 = low extreme, not absence).
3. Covariate shift generated→natural is large (domain-classifier AUC 0.95–0.99);
   lexical concepts survive it, graded scalars visibly lose signal.
4. λ grid saturated at its 1e-2 edge for all classes (val-MSE flat within 0.3%,
   direction rotation ≤9°/decade — expected impact small, but noted).
5. Deployment corpus is web-only (ClimbMix); no code/books validation.

Training/validation pipeline: `concept_probes/` in the
[oracle-encodings repo](https://github.com/kaushikreddyxyz/oracle-encodings).
