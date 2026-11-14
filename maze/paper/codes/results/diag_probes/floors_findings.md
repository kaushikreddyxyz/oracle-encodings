# No-model floors for the wall probes

`probe_floors.py` (CPU, ~50 s for g6+g8+g10). The positions, targets, split, ridge, MLP and metrics are the same as in `probe_nonlinear.py`. The script reproduces the sampling loop and rng call order exactly: its train/val/test row counts match `results/probe_v2/g{n}-s0-base-e4-oe.json`. Only the probe input changes: it is a hand-built feature vector with no model activations. The metric is macro AUROC on the test mazes (chance 0.5). The model numbers are linear probes from `results/probe_v2/*.json`, the same source as `ev_probes.csv`. Their "all" AUROCs match `ev_probes.csv` exactly (528 rows, max diff 0). Base is seed 0 only. Edge and ego are the mean over seeds 0-4. Every number, including bal_acc, R², acc and n_targets, is in `results/diag_probes/floors.csv`.

## Feature sets (exact contents)
- **F0**: an intercept only.
- **F1**: a one-hot of the current cell (n²).
- **F2**: F1, plus a one-hot of the previous path cell (n², zeros at the first path token), plus a one-hot of the incoming move N/E/S/W (4, zeros at the first path token).
- **F3**: F2, plus the bag of path cells visited so far, including the current cell (n²).
- **F4**: F3, plus a one-hot of the target cell (n²).
- **F5**: F4, plus the sequence position index and the number of path steps taken (2 scalars).
- **Extras** (not part of the requested ladder; they are hand-computed transforms of the same walk history):
  - **F3edge**: F2 plus a bag of traversed connections in absolute coordinates (2n², same indexing as `conn`).
  - **F3rel**: F2 plus the traversed connections placed in the 50 ego slots (kind, dr, dc).
  - **F3-mlp**: the same 512-GELU MLP as `fit_mlp`, trained on F3.
- **Subsets**:
  - ego `imm`: the S/N/E/W walls of the current cell.
  - ego `far`: the other 46 offsets.
  - allo `bnd`: walls with at least one endpoint cell on the grid boundary.
  - allo `int`: the remaining allo walls.

## Headline: no-model floor vs model AUROC

The only thing available at block-0 input (L0) is the current cell plus the position, so its floor is **F1**. For later layers the floors are **F5**, the whole requested ladder (for ego, F5 = the ladder max; for allo, F3 is the ladder max), and **best**, the maximum over all floors including the extras. "Model late" is the best of L3/L6/L12, with the layer given in brackets.

| family/subset | grid | F1 | F5 | best floor | base L0 | base late | edge L0 | edge late | ego L0 | ego late |
|---|---|---|---|---|---|---|---|---|---|---|
| ego all | 6 | 0.624 | 0.649 | 0.778 F3rel | 0.624 | 0.746 (L3) | 0.689 | 0.771 (L6) | 1.000 | 1.000 |
| ego all | 8 | 0.591 | 0.618 | 0.771 F3rel | 0.591 | 0.663 (L6) | 0.633 | 0.690 (L6) | 1.000 | 1.000 |
| ego all | 10 | 0.571 | 0.598 | 0.774 F3rel | 0.571 | 0.630 (L3) | 0.597 | 0.627 (L6) | 0.999 | 0.995 |
| ego imm (S,N,E,W) | 6 | 0.661 | 0.828 | 0.842 F3rel | 0.668 | 0.949 (L6) | 0.707 | 0.931 (L6) | 1.000 | 1.000 |
| ego imm | 8 | 0.623 | 0.811 | 0.837 F3rel | 0.622 | 0.892 (L6) | 0.647 | 0.830 (L12) | 1.000 | 1.000 |
| ego imm | 10 | 0.602 | 0.805 | 0.837 F3rel | 0.604 | 0.858 (L3) | 0.618 | 0.776 (L12) | 0.999 | 1.000 |
| ego far (46) | 6 | 0.621 | 0.634 | 0.772 F3rel | 0.621 | 0.729 (L3) | 0.688 | 0.757 (L6) | 1.000 | 1.000 |
| ego far | 8 | 0.588 | 0.601 | 0.766 F3rel | 0.588 | 0.643 (L6) | 0.632 | 0.678 (L6) | 1.000 | 0.999 |
| ego far | 10 | 0.569 | 0.580 | 0.768 F3rel | 0.569 | 0.610 (L3) | 0.596 | 0.619 (L6) | 0.999 | 0.995 |
| allo all | 6 | 0.513 | 0.576 (F3 0.581) | 0.690 F3edge | 0.512 | 0.858 (L3) | 1.000 | 1.000 | 0.685 | 0.734 (L6) |
| allo all | 8 | 0.506 | 0.548 (F3 0.557) | 0.664 F3edge | 0.506 | 0.623 (L3) | 1.000 | 1.000 | 0.594 | 0.601 (L6) |
| allo all | 10 | 0.503 | 0.537 (F3 0.545) | 0.651 F3edge | 0.503 | 0.560 (L3) | 1.000 | 1.000 | 0.554 | 0.555 (L3) |

## What the numbers say
1. **The floor sanity check passes.** F0 gives 0.500 in every family and grid.
2. **Base block-0 input is fully explained by the location prior.** Base L0 matches F1 to within 0.007 in every family, subset and grid:
   - ego: 0.624 / 0.591 / 0.571 (g6 / g8 / g10), identical to F1;
   - allo: 0.512 / 0.506 / 0.503, against F1 0.513 / 0.506 / 0.503.
3. **Along the ladder, only the incoming move matters for ego.** F1 to F2 raises ego imm from 0.66 to 0.81 at g6, from 0.62 to 0.80 at g8 and from 0.60 to 0.79 at g10: the wall behind the current cell is known to be open. The visited bag, the target cell and the position each add at most 0.01. For allo, the visited bag (F3) adds 0.04-0.06 over F2. The target cell lowers allo by about 0.01. F5 is identical to F4 to 3 decimals.
4. **Base model, later layers:**
   - **Share explained by the requested ladder.** It explains 61 / 73 / 75 % of the base model's above-chance margin on ego-all (share = (F5 − 0.5) / (model − 0.5)). On ego-imm it explains 73 / 79 / 85 %, and on allo 23 / 46 / 75 % (using F3).
   - **Remainder over the ladder.**
     - ego all: +0.10 / +0.05 / +0.03
     - ego imm: +0.12 / +0.08 / +0.05
     - allo: +0.28 / +0.07 / +0.02
   - **Against the extras.** Base ego-all and ego-far (up to 0.746 / 0.729) stay below F3rel at every grid (0.77). Base ego-imm exceeds every floor, including F3rel, by +0.11 / +0.06 / +0.02. Base allo exceeds F3edge only at g6 (0.858 vs 0.690). At g8 and g10 it is below F3edge (0.623 vs 0.664; 0.560 vs 0.651).
5. **Cross-frame readability:**
   - **Edge model, ego targets at L0.** It is above F1: +0.065 / +0.042 / +0.026 on ego-all. These are the only L0 numbers that exceed the location prior. At L0 the graft is present.
   - **Edge model, ego targets at L6.** Ego-all is above F5 by +0.12 / +0.07 / +0.03, but below F3rel at every grid.
   - **Edge model, egocentric immediate walls (best layer).** 0.931 / 0.830 / 0.776 against F5 0.828 / 0.811 / 0.805: above the floor at g6 (+0.10), at the floor at g8 (+0.02), and below it at g10 (−0.03). At g10, L6 gives 0.724, which is −0.08 below F5.
   - **Ego model, allo targets.** At L0 it reaches 0.685 / 0.594 / 0.554, against F1 0.51 / 0.51 / 0.50 (the graft is present at L0). At L6 it reaches 0.734 / 0.601 / 0.548, which is +0.15 / +0.04 / +0.00 over F3 and +0.04 / −0.06 / −0.10 relative to F3edge.
6. **The nonlinear reader on F3 is not an upper bound for walk history.** F3-mlp scores below the hand-made history transforms:
   - ego: 0.675 / 0.624 / 0.601 vs F3rel 0.778 / 0.771 / 0.774;
   - allo: 0.649 / 0.601 / 0.523 vs F3edge 0.690 / 0.664 / 0.651.
   With 16-18k training rows, the MLP does not learn the re-indexing into the ego frame.

## Conclusion
- **Block-0 base readability is entirely the location prior.** Base L0 equals F1 in every family and grid.
- **The ladder explains most of the base model's later-layer readability, except allo at g6.** The requested floors (location, incoming move, visited cells, target, position) explain 61-75 % of the ego margin, 73-85 % of the ego-imm margin, and 23 / 46 / 75 % of the allo margin (g6 / g8 / g10). The share grows with grid size because the model's own AUROC falls toward the floor.
- **The remainder over the ladder is +0.03 to +0.12 for ego and +0.02 to +0.28 for allo.** This is the part a model must compute from the adjacency list, or from a history re-encoding that these linear features lack.
- **Only a few model numbers beat all floors, extras included.**
  - Base ego-imm beats every floor: by +0.11 / +0.06 / +0.02.
  - Base allo beats them at g6 (+0.17).
  - The edge model's ego-imm and the ego model's allo beat them only at g6 (+0.09 and +0.04).
- **Everything else is at or below what walk history supports once it is laid out in the right frame.** This covers:
  - base ego-all and ego-far, and base allo at g8/g10;
  - the edge model's ego readout at every grid (all subsets at g8/g10, all and far at g6);
  - the ego model's allo readout at g8/g10.

  The cross-frame numbers can therefore not be taken as evidence of adjacency-list or graft use without this floor.
- **At L0, the edge model's ego readout and the ego model's allo readout exceed F1.** At L0 the only extra input is the graft.

## Full tables

### ego: no-model floors (linear ridge unless marked; AUROC, test mazes)

| grid | subset | F0 | F1 | F2 | F3 | F4 | F5 | F3-mlp | F3edge | F3rel |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | all | 0.500 | 0.624 | 0.644 | 0.648 | 0.649 | 0.649 | 0.675 | 0.662 | 0.778 |
| 6 | imm | 0.500 | 0.661 | 0.811 | 0.818 | 0.828 | 0.828 | 0.829 | 0.819 | 0.842 |
| 6 | far | 0.500 | 0.621 | 0.630 | 0.633 | 0.634 | 0.634 | 0.662 | 0.648 | 0.772 |
| 8 | all | 0.500 | 0.591 | 0.616 | 0.617 | 0.618 | 0.618 | 0.624 | 0.628 | 0.771 |
| 8 | imm | 0.500 | 0.623 | 0.796 | 0.803 | 0.811 | 0.811 | 0.806 | 0.802 | 0.837 |
| 8 | far | 0.500 | 0.588 | 0.600 | 0.601 | 0.601 | 0.601 | 0.609 | 0.613 | 0.766 |
| 10 | all | 0.500 | 0.571 | 0.597 | 0.598 | 0.598 | 0.598 | 0.601 | 0.604 | 0.774 |
| 10 | imm | 0.500 | 0.602 | 0.790 | 0.798 | 0.805 | 0.805 | 0.801 | 0.797 | 0.837 |
| 10 | far | 0.500 | 0.569 | 0.580 | 0.580 | 0.580 | 0.580 | 0.584 | 0.588 | 0.768 |

### ego: model linear-probe AUROC (base = seed 0; edge/ego = mean over seeds 0-4, (min-max))

| grid | subset | base L0 | base L3 | base L6 | base L12 | edge L0 | edge L3 | edge L6 | edge L12 | ego L0 | ego L3 | ego L6 | ego L12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | all | 0.624 | 0.746 | 0.741 | 0.708 | 0.689 (0.69-0.69) | 0.742 (0.71-0.75) | 0.771 (0.77-0.78) | 0.736 (0.73-0.74) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.999 (1.00-1.00) | 0.983 (0.98-0.99) |
| 6 | imm | 0.668 | 0.939 | 0.949 | 0.914 | 0.707 (0.71-0.71) | 0.809 (0.72-0.85) | 0.931 (0.91-0.94) | 0.902 (0.90-0.91) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) |
| 6 | far | 0.621 | 0.729 | 0.723 | 0.690 | 0.688 (0.69-0.69) | 0.736 (0.71-0.75) | 0.757 (0.75-0.76) | 0.721 (0.72-0.73) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.999 (1.00-1.00) | 0.981 (0.98-0.99) |
| 8 | all | 0.591 | 0.654 | 0.663 | 0.637 | 0.633 (0.63-0.63) | 0.648 (0.64-0.65) | 0.690 (0.68-0.69) | 0.666 (0.66-0.67) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.990 (0.98-0.99) | 0.939 (0.94-0.94) |
| 8 | imm | 0.622 | 0.877 | 0.892 | 0.798 | 0.647 (0.65-0.65) | 0.673 (0.66-0.69) | 0.825 (0.80-0.84) | 0.830 (0.82-0.84) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) |
| 8 | far | 0.588 | 0.634 | 0.643 | 0.624 | 0.632 (0.63-0.63) | 0.646 (0.64-0.65) | 0.678 (0.67-0.68) | 0.651 (0.65-0.65) | 1.000 (1.00-1.00) | 0.999 (1.00-1.00) | 0.989 (0.98-0.99) | 0.934 (0.93-0.94) |
| 10 | all | 0.571 | 0.630 | 0.612 | 0.597 | 0.597 (0.60-0.60) | 0.584 (0.58-0.59) | 0.627 (0.62-0.64) | 0.616 (0.61-0.62) | 0.999 (1.00-1.00) | 0.995 (0.99-1.00) | 0.972 (0.96-0.98) | 0.883 (0.86-0.90) |
| 10 | imm | 0.604 | 0.858 | 0.795 | 0.746 | 0.618 (0.61-0.62) | 0.598 (0.58-0.61) | 0.724 (0.69-0.77) | 0.776 (0.76-0.79) | 0.999 (1.00-1.00) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) |
| 10 | far | 0.569 | 0.610 | 0.596 | 0.584 | 0.596 (0.60-0.60) | 0.583 (0.57-0.59) | 0.619 (0.61-0.63) | 0.602 (0.60-0.61) | 0.999 (1.00-1.00) | 0.995 (0.99-1.00) | 0.970 (0.96-0.98) | 0.873 (0.84-0.89) |

### allo: no-model floors (linear ridge unless marked; AUROC, test mazes)

| grid | subset | F0 | F1 | F2 | F3 | F4 | F5 | F3-mlp | F3edge | F3rel |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | all | 0.500 | 0.513 | 0.521 | 0.581 | 0.576 | 0.576 | 0.649 | 0.690 | 0.534 |
| 6 | bnd | 0.500 | 0.514 | 0.522 | 0.596 | 0.588 | 0.588 | 0.639 | 0.679 | 0.536 |
| 6 | int | 0.500 | 0.510 | 0.518 | 0.559 | 0.557 | 0.557 | 0.663 | 0.705 | 0.531 |
| 8 | all | 0.500 | 0.506 | 0.510 | 0.557 | 0.548 | 0.548 | 0.601 | 0.664 | 0.515 |
| 8 | bnd | 0.500 | 0.508 | 0.513 | 0.571 | 0.559 | 0.559 | 0.596 | 0.648 | 0.517 |
| 8 | int | 0.500 | 0.504 | 0.508 | 0.546 | 0.539 | 0.539 | 0.606 | 0.678 | 0.514 |
| 10 | all | 0.500 | 0.503 | 0.505 | 0.545 | 0.537 | 0.537 | 0.523 | 0.651 | 0.507 |
| 10 | bnd | 0.500 | 0.506 | 0.509 | 0.561 | 0.552 | 0.552 | 0.530 | 0.633 | 0.510 |
| 10 | int | 0.500 | 0.501 | 0.503 | 0.535 | 0.528 | 0.528 | 0.519 | 0.662 | 0.505 |

### allo: model linear-probe AUROC (base = seed 0; edge/ego = mean over seeds 0-4, (min-max))

| grid | subset | base L0 | base L3 | base L6 | base L12 | edge L0 | edge L3 | edge L6 | edge L12 | ego L0 | ego L3 | ego L6 | ego L12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | all | 0.512 | 0.858 | 0.810 | 0.731 | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.998 (1.00-1.00) | 0.961 (0.95-0.97) | 0.685 (0.68-0.69) | 0.732 (0.73-0.74) | 0.734 (0.72-0.75) | 0.688 (0.68-0.70) |
| 6 | bnd | 0.514 | 0.858 | 0.816 | 0.742 | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.998 (0.99-1.00) | 0.959 (0.95-0.97) | 0.718 (0.72-0.72) | 0.746 (0.74-0.76) | 0.749 (0.73-0.77) | 0.704 (0.70-0.72) |
| 6 | int | 0.509 | 0.860 | 0.800 | 0.714 | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.998 (1.00-1.00) | 0.964 (0.96-0.97) | 0.637 (0.64-0.64) | 0.711 (0.70-0.72) | 0.713 (0.69-0.73) | 0.665 (0.66-0.67) |
| 8 | all | 0.506 | 0.623 | 0.623 | 0.599 | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.989 (0.98-1.00) | 0.858 (0.84-0.89) | 0.594 (0.59-0.59) | 0.600 (0.60-0.60) | 0.601 (0.59-0.61) | 0.579 (0.57-0.58) |
| 8 | bnd | 0.508 | 0.635 | 0.633 | 0.619 | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.988 (0.98-1.00) | 0.850 (0.83-0.89) | 0.602 (0.60-0.60) | 0.611 (0.61-0.61) | 0.613 (0.61-0.63) | 0.590 (0.58-0.60) |
| 8 | int | 0.503 | 0.614 | 0.613 | 0.581 | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.990 (0.98-1.00) | 0.865 (0.85-0.90) | 0.587 (0.59-0.59) | 0.591 (0.59-0.59) | 0.591 (0.58-0.60) | 0.569 (0.56-0.57) |
| 10 | all | 0.503 | 0.560 | 0.549 | 0.542 | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.983 (0.95-1.00) | 0.725 (0.69-0.75) | 0.554 (0.55-0.55) | 0.555 (0.55-0.56) | 0.548 (0.55-0.55) | 0.534 (0.53-0.54) |
| 10 | bnd | 0.505 | 0.569 | 0.564 | 0.558 | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.985 (0.96-1.00) | 0.727 (0.70-0.75) | 0.554 (0.55-0.56) | 0.558 (0.56-0.56) | 0.554 (0.55-0.56) | 0.542 (0.54-0.54) |
| 10 | int | 0.501 | 0.555 | 0.540 | 0.532 | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.982 (0.95-1.00) | 0.724 (0.69-0.75) | 0.554 (0.55-0.55) | 0.553 (0.55-0.55) | 0.545 (0.54-0.55) | 0.530 (0.53-0.53) |
