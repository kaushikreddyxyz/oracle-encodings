# Where cross-frame and base-model wall readability comes from (probe_v2, 66 models)

Source: `results/probe_v2/*.json` (grids 5-10; edge/ego seeds 0-4, base seed 0). Averages are over seeds within a grid,
then over grids with equal weight. Per-target AUROC had full length in all 3432 model x site x (family, probe) entries,
so nothing was dropped. Figure: `../../fig_ev_probe_offsets.{pdf,svg}`.

Offset groups (ego family, 50 targets): **immediate** = own walls of the current cell (S = down(0,0), N = down(-1,0),
E = right(0,0), W = right(0,-1)); **ring2** = max(|dr|,|dc|) = 2 (32 targets); **ring1-other** = the other 14.

**Prior reference (`prior_sim.py`, no transformer run).** I regenerated the same probe mazes (gen_dfs, seed 999, 4000
mazes, same rng draws; the train/val/test counts match every JSON `n_rows` exactly, e.g. g6 test 6801) and fitted the same
ridge on (i) one-hot(current cell) + one-hot(step along path), which is all an additive token + position embedding can carry
(the adjacency list has a fixed length, so position equals path step); (ii) (i) + the raw edge bits; (iii) (i) + the raw
ego bits. This was needed because the JSON stores `frac_open` only as a macro average, not per target.

## 1. Ego targets: per-offset linear AUROC, mean over grids

| model | site | immediate | ring1-other | ring2 | all 50 |
|---|---|---|---|---|---|
| prior (cell+step) | – | 0.642 | 0.600 | 0.601 | – |
| prior + edge bits | – | 0.673 | 0.647 | 0.657 | – |
| base | L0in | **0.642** | 0.600 | 0.601 | 0.604 |
| base | L3 | 0.890 | 0.719 | 0.652 | 0.690 |
| base | L6 | **0.893** | 0.722 | 0.647 | 0.688 |
| base | L9 | 0.861 | 0.702 | 0.637 | 0.673 |
| base | L12 | 0.839 | 0.688 | 0.629 | 0.662 |
| edge | L0in | **0.673** | 0.647 | 0.658 | 0.656 |
| edge | L3 | 0.737 | 0.692 | 0.687 | 0.693 |
| edge | L6 | **0.855** | 0.739 | 0.697 | 0.721 |
| edge | L9 | 0.887 | 0.734 | 0.680 | 0.712 |
| edge | L12 | 0.857 | 0.713 | 0.665 | 0.693 |
| ego (in-frame) | L0in / L6 / L12 | 1.000 / 1.000 / 1.000 | 1.000 / 0.993 / 0.960 | 1.000 / 0.986 / 0.931 | |

The four own walls behave alike (base L6 S/N/E/W 0.895/0.885/0.892/0.900; edge L6 0.856/0.858/0.853/0.855), and
RIGHT(dr,dc) equals the transpose of DOWN(dc,dr) to within 0.022 (base) / 0.011 (edge). The next tier (about 0.73-0.76 at
L6) is the walls touching the corners of the current cell and the far walls of the N/S neighbours (map in the figure).

MLP probe, same groups:

| model | site | immediate | ring1-other | ring2 |
|---|---|---|---|---|
| base | L0in | 0.641 | 0.600 | 0.600 |
| base | L3 | 0.979 | 0.835 | 0.711 |
| base | L6 | **0.9998** | 0.869 | 0.718 |
| base | L12 | 0.991 | 0.761 | 0.640 |
| edge | L0in | 0.728 | 0.729 | 0.734 |
| edge | L3 | 0.889 | 0.882 | 0.875 |
| edge | L6 | **0.990** | 0.960 | 0.913 |
| edge | L12 | 0.992 | 0.853 | 0.744 |

Per grid, L6 (linear | MLP), with the linear prior for reference:

| grid | base imm | base ring2 | edge imm | edge ring2 | prior imm | prior ring2 | prior+edge imm | prior+edge ring2 | base MLP imm | edge MLP imm | edge MLP ring2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.976 | 0.743 | 0.982 | 0.832 | 0.695 | 0.648 | 0.742 | 0.743 | 1.000 | 1.000 | 1.000 |
| 6 | 0.949 | 0.695 | 0.931 | 0.740 | 0.668 | 0.621 | 0.707 | 0.691 | 1.000 | 1.000 | 0.993 |
| 7 | 0.919 | 0.643 | 0.893 | 0.694 | 0.650 | 0.599 | 0.684 | 0.660 | 1.000 | 1.000 | 0.953 |
| 8 | 0.892 | 0.619 | 0.825 | 0.665 | 0.622 | 0.588 | 0.646 | 0.633 | 1.000 | 0.997 | 0.913 |
| 9 | 0.825 | 0.602 | 0.776 | 0.639 | 0.611 | 0.579 | 0.637 | 0.616 | 1.000 | 0.989 | 0.851 |
| 10 | 0.796 | 0.582 | 0.724 | 0.610 | 0.604 | 0.569 | 0.622 | 0.599 | 1.000 | 0.951 | 0.770 |

## 2. Allo targets (absolute walls) by location, linear

Mean per-wall AUROC per grid:

| model / site | g5 | g6 | g7 | g8 | g9 | g10 |
|---|---|---|---|---|---|---|
| prior (cell+step) | 0.517 | 0.512 | 0.508 | 0.506 | 0.503 | 0.503 |
| base L0in | 0.517 | 0.512 | 0.508 | 0.506 | 0.503 | 0.503 |
| base L6 | 0.858 | 0.810 | 0.697 | 0.623 | 0.588 | 0.549 |
| prior + ego bits | 0.755 | 0.694 | 0.647 | 0.615 | 0.590 | 0.571 |
| ego L0in | 0.755 | 0.686 | 0.627 | 0.594 | 0.572 | 0.554 |
| ego L6 | 0.853 | 0.734 | 0.650 | 0.602 | 0.569 | 0.548 |
| ego L12 | 0.801 | 0.688 | 0.630 | 0.579 | 0.552 | 0.534 |

Boundary (wall touches the outer ring of cells) vs interior vs corner-adjacent, mean over grids:

| model / site | boundary | interior | corner-adjacent |
|---|---|---|---|
| base L0in | 0.511 | 0.505 | 0.514 |
| base L6 | 0.698 | 0.677 | 0.696 |
| ego L0in | 0.646 | 0.606 | 0.669 |
| ego L6 | 0.670 | 0.646 | 0.679 |

The ego model's input readability is location-concentrated on small grids (g5 corner-adjacent 0.860 vs interior 0.671;
g6 down walls by row 0.73/0.66/0.64/0.66/0.74, edges high), which is what the ridge on raw ego bits predicts (extreme
relative offsets pin the absolute position). By L6 it is spread almost evenly: the top 10% of walls hold 12-14% of the
above-chance excess (uniform = 10%), boundary minus interior is 0.01-0.04, and on g10 the by-row / by-column profiles
are flat to within 0.02 (0.542-0.560). Per-wall correlation of ego L6 with base L6 is weak (r = 0.21-0.49 per grid),
but the grid means match closely (ego minus base: -0.005, -0.075, -0.047, -0.021, -0.019, -0.001).

## 3. Location / base-rate prior check

| comparison (per target, all grids) | Pearson r | mean diff | max abs diff |
|---|---|---|---|
| base L0in ego targets vs prior sim | 1.000 | +0.0000 | 0.0007 |
| base L0in allo walls vs prior sim | 1.000 (every grid) | +0.0000 | – |
| edge L0in ego targets vs prior + edge bits | 0.999 | +0.0002 | 0.0105 |
| ego L0in allo walls vs prior + ego bits | 0.905 (per grid 1.00, 0.99, 0.77, 0.71, 0.58, 0.37) | -0.017 | – |
| base L6 ego targets vs prior | 0.767 | +0.084 | 0.302 |

- The base model's block-0 input reproduces the cell + step prior to within 0.0007 AUROC for every target. Its per-offset
  AUROC correlates with the targets' base-rate imbalance |frac_open - 0.5| (Spearman +0.62). Offset frac_open ranges
  0.49-0.66, and absolute-wall frac_open ranges 0.41-0.87. Own walls score higher under the prior (0.642 vs 0.600)
  because the current cell sits on the solution path, so the pairing of cell and step predicts its own walls better than
  distant ones.
- The edge model's input equals prior + linear readout of raw absolute bits, and the ego model's input equals prior + raw ego bits for g5-g6.
  From g7 on, the ego model falls below that reference. g7 is also where 2(2n-1)^2 ego features (338) first exceed d_model = 256. This fits a
  lossy injection projection, but I did not test it.

## Conclusion

- **Input (L0in): (a) a location/step prior, plus linear leakage of the raw graft bits.** No computation is involved. It is
  reproduced to within 0.001 (base) and 0.01 (edge) AUROC by a ridge fitted on maze statistics alone.
- **Deeper layers, own walls of the current cell: (b) computed.** Base L6 is 0.893 linear and 0.9998 MLP, against a prior of 0.642.
  Edge L6 is 0.855 linear and 0.990 MLP, against 0.673. The base model has no graft, so this comes from the prompt, and
  the edge model does not exceed the base model on these walls (linear 0.855 vs 0.893; lower on every grid from g6 on).
  The gain falls off with distance. For base at L6 it is +0.25 on own walls, +0.12 on the rest of ring 1 and +0.05 on ring 2.
- **Ego model, absolute walls: not specific to the graft.** Past the input, its absolute-wall readability sits at the same
  per-grid level as the base model, which has no graft (0.853 vs 0.858 on g5; 0.548 vs 0.549 on g10). It is spread
  evenly over walls and falls with grid size. At L0in the only extra above base is the corner/boundary-concentrated leakage
  of raw ego bits. What the base model's own absolute-wall readability at L6 (0.69 against a 0.51 prior) is made of can't
  be settled from these JSONs.
- Caveat from the JSON alone: at a path token, the wall toward the previous path cell is open by construction, so one of
  the four own walls can be inferred from the preceding token. The saved per-target metrics do not split by "toward the
  previous cell", so how much of the 0.89 comes from this cannot be separated here.

## Anomalies (`anomalies.txt`)

- No per-target AUROC below 0.47. The pooled AUROC equals the mean of `per_target_auroc` everywhere. No non-saturated per-target
  arrays are duplicated across files. One per-target R^2 is below -0.05 (edge g9 s2 L12 ego MLP, -0.057).
- Not a data error, but notable: base-model MLP probes read the four own walls at AUROC >= 0.99 at L3-L12 (229 target x
  site cases), while the linear probe tops out at 0.89. The information is present but not fully linearly decodable, so
  linear numbers understate it.
- `frac_open` in the JSON is the macro mean over targets (it equals the maze's open fraction, e.g. 35/60 at g6), not a
  per-target value, so the base-rate check needed the regenerated mazes above.

## Files

`probe_offsets.py` (JSON -> tables), `prior_sim.py` (model-free prior reference), `plot_offsets.py` (figure);
`ego_offsets_linear.csv`, `ego_offsets_mlp.csv` (per offset x model x site x grid, plus grid mean), `ego_groups.csv`,
`allo_by_location.csv`, `allo_per_wall.csv`, `allo_concentration.csv`, `sim_ego_offsets.csv`, `sim_allo_per_wall.csv`,
`anomalies.txt`. (`floors.csv` in this folder was not written by this analysis.)
