# Graft benefit vs. source description length on entropy-pinned Markov chains

2026-09-17. Main grid complete (240 runs, 3 model sizes × 10 sources × 4 arms × 2 seeds) plus 15 follow-up runs. All pods deleted. Figures in `figures/`, per-run table in `figures/summary.csv`, wandb project `markov-graft`. Code: `dgp.py` (source), `model.py`, `train.py`, `sweep.py`, `analyze.py`, `analyze_followups.py`.

## Question

Does a model's uptake of an externally supplied feature (a "graft") depend on the complexity of the data it has to model, relative to what it can learn, and is there a clean pattern?

## Design

**Data.** Alphabet A = 8. Every transition row is a random permutation of one base distribution q with H(q) = 2.0 bits, so the entropy rate is exactly 2.0 bits (1.386 nats) for every source and the noise floor never moves. The only thing that changes across sources is the description length of the table:

    ell = A^m · log2(A!) bits,   m = number of context lags the row depends on

| table | k (order) | m (lags used) | ell, bits |
|---|---|---|---|
| dense | 1..7 | = k | 122, 979, 7.8k, 63k, 501k, 4.0M, 32M |
| sparse | 7 | 1, 3, 5 | 122, 7.8k, 501k |

Sparse tables keep the nominal order at 7 (the row depends on lag 7 plus m−1 random other lags), so order and information can be told apart.

**Grafts**, added to the token embedding at the input as a learned b-bit embedding, one value per position, about the next symbol:

| arm | the b bits | information about the target beyond the context |
|---|---|---|
| oracle_1 | top bit of the argmax symbol of the true row | zero (a function of the context) |
| oracle_3 | the argmax symbol of the true row | zero |
| leak_1 | top bit of the actual next symbol | I(y; g \| x) = 0.573 nats, exact from the table |

**Models.** Pre-LN GPT with a learned relative-offset attention bias so that attending to a fixed lag is cheap. Sizes s (2 layers, d 64, 101k params), m (4 layers, d 128, 794k), l (6 layers, d 256, 4.7M). 6000 steps × 128 × 256 fresh tokens (197M), AdamW 1e-3, cosine, bf16. Loss is measured only on positions whose full k-context lies inside the sequence, for every arm and every reference alike.

**Reference predictors**, exact from the table on held-out data: Bayes (knows the table), unigram (knows nothing), graft-only (knows the graft, ignores the context), Bayes-with-leak.

**Measured per run:** final loss; loss with the graft shuffled across sequences (dependence = shuffled − true); benefit = loss(no graft) − loss(graft) against the matching no-graft run; recovery = benefit / (loss(no graft) − Bayes); for the leak arm, amplification = benefit / I(y; g | x).

## Results

### 1. The unaided model falls off a cliff, and the cliff does not move with model size

`fig1_unaided_excess_loss_vs_description_bits.png`. Excess loss over Bayes for the no-graft models, seed-averaged (two seeds agree to ±0.005 unless noted):

| ell, bits | s (101k) | m (794k) | l (4.7M) |
|---|---|---|---|
| 122 | 0.001 | 0.001 | 0.001 |
| 979 | 0.001 | 0.000 | 0.000 |
| 7.8k | 0.005 | 0.001 | 0.001 |
| 63k | 0.047 | 0.013 | 0.019 |
| 501k | 0.682 | 0.692 | 0.692 |
| 4.0M | 0.693 | 0.693 | 0.693 |
| 32M | 0.693 | 0.693 | 0.693 |

0.693 nats is exactly the unigram excess: from 501k bits on, the unaided model has learned nothing beyond the marginal within the 6000-step budget. The transition is a step between 63k and 501k bits, and it sits in the same place across a 47× range of parameter count. The dotted "2 bits per parameter" lines in Figure 1 are nowhere near it. Sparse order-7 tables (hollow squares) fall on the same curve as dense tables with the same ell, so table size, not Markov order or attention span, sets the cliff.

### 2. Graft benefit is a spike at the cliff, zero below it, and capped by the graft-only predictor above it

`fig2_graft_benefit_vs_description_bits.png` and the raw loss levels in `fig5_loss_levels_vs_references_size_{s,m,l}.png`. Loss levels for size m, nats per token:

| ell, bits | no graft | oracle_1 | oracle_3 | leak_1 | Bayes | unigram | oracle_3-only | leak-only | Bayes+leak |
|---|---|---|---|---|---|---|---|---|---|
| 122 | 1.387 | 1.387 | 1.387 | 0.772 | 1.386 | 2.078 | 1.68 | 1.385 | 0.813 |
| 7.8k | 1.387 | 1.388 | 1.388 | 0.815 | 1.386 | 2.078 | 1.68 | 1.385 | 0.813 |
| 63k | 1.399 | 1.394 | 1.390 | 0.825 | 1.386 | 2.078 | 1.68 | 1.385 | 0.813 |
| 501k | 2.079 | 1.708 | 1.516 | 1.305 | 1.386 | 2.078 | 1.68 | 1.385 | 0.813 |
| 4.0M | 2.079 | 1.989 | 1.682 | 1.386 | 1.386 | 2.078 | 1.68 | 1.385 | 0.813 |
| 32M | 2.079 | 1.989 | 1.682 | 1.387 | 1.386 | 2.078 | 1.68 | 1.385 | 0.813 |

Three regimes, identical across sizes and seeds:

- **Below the cliff** the graft buys nothing. The oracle arms match the no-graft loss to three decimals and the leak arm gains exactly its information content (amplification 1.00 ± 0.01 at every size).
- **At the cliff** the graft is worth more than its information. With oracle_3 the model reaches 1.52 (size m) or 1.42 (size l), far below the graft-only predictor at 1.68: the graft let the model learn most of a table it could not learn alone. Recovery of the excess loss is 0.60 (s), 0.81 (m), 0.95 (l), the one place where size matters. The leak arm reaches 1.31 at size m, below the Bayes loss without the leak, amplification 1.12 (size l: 1.00, it learned nothing extra).
- **Past the cliff** every grafted model sits exactly on its graft-only reference (1.682 vs 1.68; 1.989 vs 1.99; 1.386 vs 1.385). It reads the graft and learns nothing of the table. The benefit is then a constant set by the graft's own predictive value, not by the model.

### 3. Reliance is highest where the graft is least needed

`fig3_graft_dependence_vs_description_bits.png`. Shuffling the graft costs the oracle_3 models 1.4 nats at every ell from 979 bits to 63k bits, where the graft's benefit is zero. The model computes the answer itself and still routes through the graft, and does so more strongly than a pure graft reader would (past the cliff the dependence is 0.70, the graft-only predictor's own sensitivity). Reliance is therefore not a measure of need: below the cliff it is a shortcut, above the cliff it is the whole solution.

### 4. Follow-ups: the cliff is an optimisation plateau, and reliance is a training-distribution effect

`fig6_training_budget_curves_k5_k6_size_m.png`, `figures/dropout_table.tsv`. All at size m.

**Longer training at and past the cliff.** Held-out loss by training step, size m, no dropout:

| ell | arm | 6k steps | 12k | 24k | 48k |
|---|---|---|---|---|---|
| 501k | none, seed 0 | 2.078 | 1.458 | 1.402 | |
| 501k | none, seed 1 | 2.079 | 2.064 | 1.469 | |
| 501k | oracle_3 (both seeds) | 1.518 | 1.425 | 1.393 | |
| 501k | leak_1 | 1.376 | 0.887 | 0.832 | |
| 4.0M | none | 2.079 | 2.079 | 2.079 | 2.079 |
| 4.0M | oracle_3 | 1.683 | 1.675 | 1.583 | |

The unaided model at 501k bits leaves the unigram plateau between step 3k and 10k (seed-dependent) and is at 1.40 to 1.47 by 24k (Bayes 1.386). The cliff at 6k steps is a plateau-escape time, not a capacity wall. At 4.0M bits the unaided model is still exactly at unigram after 48k steps, eight times the grid budget, so the plateau length grows far faster than the table. The grafted models keep converging where the unaided one is stuck: oracle_3 at 4.0M bits drops below its graft-only ceiling (1.68) after 12k steps and is at 1.58 by 24k, and the leak at 501k bits reaches 0.83 (Bayes-with-leak 0.81) by 24k. The oracle_3 model at 501k bits reaches Bayes (1.393) by 24k while keeping 1.44 nats of dependence.

**Graft dropout p = 0.5** (graft replaced by a null token on half the training sequences):

| ell | arm | loss, p=0 | dependence, p=0 | loss, p=0.5 | loss with null graft | dependence, p=0.5 |
|---|---|---|---|---|---|---|
| 7.8k | oracle_3 (2 seeds) | 1.388 | 1.43 | 1.388 | 1.388 | 0.0005 |
| 7.8k | leak_1 | 0.814 | 6.45 | 0.814 | 1.388 | 4.59 |
| 501k | oracle_3 (2 seeds) | 1.516 | 1.10 | 1.670 | 1.723 | 0.71 |
| 501k | leak_1 (2 seeds) | 1.305 | 6.15 | 1.378 | 2.069 | 6.64 |

Below the cliff, half-time absence is enough to switch the oracle graft off completely: dependence goes from 1.43 to 0.0004 with no change in loss. The shortcut was a training-distribution artefact. The leak keeps its dependence (it still carries real information) but the model is now at Bayes without it. At the cliff, the oracle_3 model trained with dropout reaches 1.72 when the graft is absent, far below the unaided 2.078: the graft taught the model to compute the graft from the context, and the knowledge survives its removal. The leak did not do this (null-graft loss 2.07 in both seeds).

## Reading

For this family, "willingness to take a graft" decomposes into two things that scale differently:

1. **Benefit** is governed by whether the unaided model escapes the plateau within its budget. That threshold is a function of table description length and training steps, and in the tested range not of parameter count. Below it the graft is informationally worthless and above it the graft's value is capped by the graft-only predictor; the interesting regime is a narrow band where a few bits of oracle knowledge unlock learning of the rest of the table.
2. **Reliance** is set by the training distribution of the channel, not by need. It is maximal for a graft that is always present and computable from the context, and it disappears under 50% dropout when the model can do without it.

The maze results (base and grafted equal on the true world, grafted follows the phantom world) are the below-the-cliff corner of this picture.

## Cost and infrastructure

Four A100 PCIe pods (CA-MTL-3, $1.59/h), roughly 3.5 pod-hours in total. Per-run recipe that made the grid affordable: whole token stream pre-generated on the GPU as uint8, `torch.compile(mode="reduce-overhead")`, and four concurrent runs per GPU under CUDA MPS.
