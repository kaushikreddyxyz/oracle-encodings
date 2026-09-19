# Training-curve campaign: base vs grafted maze transformers, grid sizes 5–10, 5 seeds

Completed 2026-09-16 17:20 PT. Figures: see the current figure set below.
Data: `all_metrics.csv` (6,900 rows: one per grid × seed × condition × checkpoint × world),
`train_loss_by_ckpt.csv` (mean train loss over the window before each checkpoint), per-run raw files under
`results/<pod>/g{n}-s{s}/<run>/` (`eval/metrics.json`, `eval/per_maze.npz` with per-maze p_sp / p_valid / solved /
reached / clean for every checkpoint, `train_loss.csv` with every step, `model.final.pt`, `model.final.zanj`,
`injector.final.pt`, `config_holder.json`). Final models are also on wandb (project `maze-curves`, 93 finished runs
with a `final` artifact; the 12 crashed runs there are guard-killed duplicates and the fp32 false start).

## Design
- 6 grid sizes (5×5 … 10×10) × {base, fixed graft, trainable graft} × 5 seeds (seed = dataset seed and init seed)
  = 90 runs. Same architecture everywhere (jirpyish d256 / 12 layers / 16 heads), 2M random-DFS mazes, one epoch,
  batch 256, AdamW 6e-4, 7,812 steps. Torch-native trainer (`maze/core/train.py`), bf16 autocast, padding_mode
  right, torch.compile. 46 checkpoints per run: steps 0, 25, 50, 100, 150, 200, 300, 400, 600, 800 then every 200,
  plus the final model.
- Every checkpoint scored on 2,000 fresh mazes of its grid size (maze_dataset seed 4321; identical set and identical
  phantom edits for every model of that size). Teacher-forced along the shortest path: p(SP token) = mass on a
  shortest-path continuation (or `<PATH_END>` at the target); p(Σ valid − SP) = mass on other wall-free neighbours;
  p(Σ invalid) = the rest. Greedy rollout (KV-cache decoder, verified identical to the full forward) from the origin:
  solved = reached the target with every move contiguous and wall-free; also reached, clean, valid-move fraction,
  length ratio (in metrics.json).
- Worlds: true (all models; grafted models receive the true 2n² connection features through the channel) and
  phantom (grafted models only: 2 phantom walls on the true shortest path + 2 phantom doors where true walls were,
  target still reachable; channel carries the phantom features, teacher-forcing follows the phantom shortest path,
  scoring is against the phantom world).

## Final-checkpoint numbers (mean over 5 seeds)

| grid | model · world | p(SP) | p(Σ valid−SP) | p(Σ invalid) | solved |
|---|---|---|---|---|---|
| 5 | base · true | 0.993 | 0.007 | 0.001 | 0.970 |
| 5 | fixed · true / phantom | 0.993 / 0.926 | 0.007 / 0.062 | 0.001 / 0.012 | 0.969 / 0.708 |
| 5 | trainable · true / phantom | 0.994 / 0.931 | 0.005 / 0.057 | 0.001 / 0.012 | 0.976 / 0.715 |
| 6 | base · true | 0.971 | 0.028 | 0.001 | 0.795 |
| 6 | fixed · true / phantom | 0.975 / 0.908 | 0.024 / 0.085 | 0.001 / 0.007 | 0.838 / 0.480 |
| 6 | trainable · true / phantom | 0.974 / 0.913 | 0.025 / 0.080 | 0.001 / 0.007 | 0.828 / 0.519 |
| 7 | base · true | 0.957 | 0.042 | 0.001 | 0.660 |
| 7 | fixed · true / phantom | 0.954 / 0.891 | 0.044 / 0.102 | 0.002 / 0.006 | 0.659 / 0.288 |
| 7 | trainable · true / phantom | 0.954 / 0.899 | 0.044 / 0.094 | 0.002 / 0.007 | 0.650 / 0.345 |
| 8 | base · true | 0.952 | 0.047 | 0.001 | 0.576 |
| 8 | fixed · true / phantom | 0.950 / 0.886 | 0.046 / 0.107 | 0.003 / 0.007 | 0.580 / 0.175 |
| 8 | trainable · true / phantom | 0.947 / 0.895 | 0.049 / 0.099 | 0.004 / 0.006 | 0.568 / 0.223 |
| 9 | base · true | 0.947 | 0.051 | 0.002 | 0.480 |
| 9 | fixed · true / phantom | 0.941 / 0.882 | 0.053 / 0.109 | 0.005 / 0.009 | 0.450 / 0.109 |
| 9 | trainable · true / phantom | 0.940 / 0.891 | 0.055 / 0.102 | 0.005 / 0.007 | 0.442 / 0.138 |
| 10 | base · true | 0.943 | 0.054 | 0.003 | 0.398 |
| 10 | fixed · true / phantom | 0.934 / 0.884 | 0.058 / 0.106 | 0.008 / 0.011 | 0.364 / 0.073 |
| 10 | trainable · true / phantom | 0.918 / 0.880 | 0.062 / 0.101 | 0.019 / 0.020 | 0.322 / 0.086 |

## Final solve rate, stratified by graft-maze / prompt-maze Hamming distance (2026-09-18)

This supersedes the `solved` column of the table above for the grafted models and is what the solve-rate panel of
`fig_final_*_lines` now shows. Mean over 5 seeds of the per-seed solve rate, 3,000 mazes per size; base model from the
2,000-maze campaign set. Distance = share of the 2n(n−1) free wall features that differ (boundary features excluded).
The token-probability columns above and the token panels of the figure still use the 2-wall + 2-door phantom edit:
teacher-forced probabilities were not measured on the stratified set.

| grid | base | fixed: same maze / 0–10% / 10–30% / 30–60% / fresh DFS | trainable: same maze / 0–10% / 10–30% / 30–60% / fresh DFS | phantom edit (fixed / trainable) |
|---|---|---|---|---|
| 5 | 0.970 | 0.968 / 0.956 / 0.939 / 0.933 / 0.897 | 0.974 / 0.952 / 0.927 / 0.890 / 0.869 | 0.708 / 0.715 |
| 6 | 0.795 | 0.838 / 0.814 / 0.757 / 0.742 / 0.693 | 0.832 / 0.816 / 0.777 / 0.773 / 0.742 | 0.480 / 0.519 |
| 7 | 0.660 | 0.661 / 0.637 / 0.589 / 0.564 / 0.533 | 0.654 / 0.638 / 0.606 / 0.584 / 0.558 | 0.288 / 0.345 |
| 8 | 0.576 | 0.573 / 0.540 / 0.469 / 0.441 / 0.407 | 0.560 / 0.535 / 0.481 / 0.448 / 0.430 | 0.175 / 0.223 |
| 9 | 0.479 | 0.443 / 0.421 / 0.358 / 0.321 / 0.318 | 0.437 / 0.425 / 0.390 / 0.356 / 0.349 | 0.109 / 0.138 |
| 10 | 0.398 | 0.364 / 0.346 / 0.288 / 0.253 / 0.237 | 0.318 / 0.312 / 0.296 / 0.280 / 0.265 | 0.073 / 0.086 |

### Distance experiment: design and findings
**Question.** Does a grafted model solve the grafted maze less often the more that maze differs from the maze in the prompt?

**Metric.** Hamming distance over the 2n(n−1) free connection features (`maze_distance.py`). Encoding checked against
`maze/core`: n² down-connections then n² right-connections, ±1; the bottom-row downs and right-column rights (2n features)
are closed in every maze and are excluded. Spanning trees have n²−1 open connections, so distances are even; reported as a
fraction of 2n(n−1).

**Eval set.** Per grid 5…10: 3,000 fresh DFS mazes (seed 4321, serial) with their origin/target = token-space maze A.
For each A: level 0 (B=A), one variant B per 5%-wide distance bucket up to 60% (12 buckets), plus one independent fresh
DFS maze. Variants are genuine DFS trees of the grid (in-family with training): DFS from A's generating root that follows
A's connections with prob 1−ε, ε=0 ⇒ A, ε=1 ⇒ fresh uniform DFS. 41,989–42,000 items per grid; sets bit-identical across
the 10 pods (sha check in `results/distance/logs/set_shas.txt`). Origin/target never change; rollouts judged against B.

**Models.** The 60 final grafted checkpoints of the curves campaign (fixed / trainable W_inj × 5 seeds × 6 grids).
Each heatmap cell = 3,000 mazes × 5 seeds = 15,000 greedy rollouts (min cell 14,690); widest Wilson 95% interval 1.6 points.

**Files.** `fig_distance_heatmap_routechanged.{pdf,svg}`, `distance_cells.csv`, `distance_items.csv.gz` (2.5M rollouts, one row each),
`results/distance/{sets,out}` raw npz, `results/distance/pods.txt`. Code: `eval_distance.py` (pod), `plot_heatmap.py`, `distance_deploy_one.sh`.

**Findings.**
1. Dissimilarity costs little. 6×6 fixed: 0.84 (same maze) → 0.74 at 55–60% distance; 10×10: 0.36 → 0.25. Maze size
   dominates the heatmap, distance is second-order. Trainable W_inj degrades less at large grids (10×10 retains 0.89 vs 0.68).
2. The loss saturates by ~25–30% distance, and it is a step, not a slope: among items whose origin→target route differs
   between A and B the solve rate is flat in distance (6×6 ≈ 0.72, 8×8 ≈ 0.42, 10×10 ≈ 0.25 at every bucket). The gradual
   decline in the main heatmap is the share of route-changed items rising (35% in the first bucket → 95%).
   Length-matched, a changed route costs ~10 points at 6×6 (0.72 vs 0.82), ~14 at 8×8, ~12 at 10×10.
3. Fresh independent DFS mazes score 2–5 points below variants at the same distance.
4. The phantom-edit collapse in the curves campaign (6×6: 0.48 with only 4 features flipped) is therefore NOT a
   dissimilarity effect: a completely different DFS maze scores 0.69–0.74. What differs is that phantom mazes are not
   spanning trees (2 walls + 2 doors leave a cycle and a cut-off region) and the edits sit adversarially on the route.

Cost: 10 × A100 SXM 80GB secure at $1.59/h for ~26 min ≈ $7. All pods deleted.

## Activation steering of the base models on the standard eval set (2026-09-18)

Question: can the BASE model (no graft) be made to solve maze B while its prompt describes maze A, by editing activations,
and how do steering vectors compare with SAE features? Same 3,000-maze distance-stratified set, same route-changed cut,
30 base finals (6 sizes x 5 seeds). Figures: `fig_steering_lines_{fixed,trainable}.{pdf,svg}`; data: `steering_cells.csv`
(route-changed and all items, Wilson intervals), `steering_items.csv.gz` (one row per rollout), raw npz in
`results/distance/steer_out/<method>/`, per-model statistics + SAEs in `results/distance/steer_prep/`, tuning-set outputs in
`results/distance/steer_tune/`. Code: `eval_steering.py` (pod), `plot_steering.py`, `steer_deploy_one.sh`,
`steer_status.sh`, `steer_reap.sh`.

**Protocol.** A and B are spanning trees, so A -> B removes k connections and adds k. Each removed connection is paired with an
added one (fixed random matching) and the removed connection's adjacency-list chunk `c1 <--> c2 ;` is rewritten into the added
one at its four tokens, in resid_post of block 0, during the prompt pass only (the site found in experiments/07: the connection
is stored at the chunk's coordinate tokens, not at `<PATH_START>` or later positions, which is why experiment 04 was null).
Classes are directed connections (first cell, second cell). Class means and SAEs are fitted per model on 6,000 fresh mazes
(seed 777). Methods: steering vectors `x += mu[target] - mu[source]` per site; class-mean patch `x <- mu[target]`; SAE features
(source latents := 0, target latents := class-typical value, delta written through the decoder; ReLU SAE x8, L1); floor = no
edit; ceiling = the chunk's two coordinate tokens replaced in the prompt. Hyper-parameters were chosen on a separate tuning set
(300 mazes, seed 999, seed-0 models of 6/8/9/10): alpha 1 and block 0 for the vectors (alpha 2 and block 1 are worse); for the
SAE, l1 0.03 with the top-16 latents per class (the first config, l1 0.1 / top-4 as in experiment 07, left >50% dead latents on
the larger grids and is kept in the CSVs only; TopK and x32 dictionaries were worse).

**Route-changed solve rate, all distance bins pooled (5 seeds):**

| grid | no steering | steering vectors | SAE features | class-mean patch | prompt rewritten (ceiling) | fixed graft |
|---|---|---|---|---|---|---|
| 5 | 0.000 | 0.702 | 0.709 | 0.716 | 0.958 | 0.921 |
| 6 | 0.002 | 0.594 | 0.573 | 0.498 | 0.774 | 0.716 |
| 7 | 0.002 | 0.611 | 0.494 | 0.495 | 0.625 | 0.538 |
| 8 | 0.002 | 0.501 | 0.440 | 0.415 | 0.520 | 0.416 |
| 9 | 0.002 | 0.436 | 0.331 | 0.327 | 0.448 | 0.308 |
| 10 | 0.002 | 0.327 | 0.196 | 0.184 | 0.366 | 0.242 |

- Without steering the base model never solves the other maze (0.2%): it follows its prompt.
- Steering vectors reach the token-rewrite ceiling at 7x7-10x10 (0.61/0.63, 0.50/0.52, 0.44/0.45, 0.33/0.37) and beat the
  fixed graft there (0.54, 0.42, 0.31, 0.24); at 5x5 and 6x6 they fall short of both ceiling and graft (0.70 vs 0.96 / 0.92).
  Like the graft, they are flat in maze difference: rewriting half the maze costs no more than rewriting 5%.
- SAE features are at best equal to steering vectors, never better, and bimodal per model. Where the SAE's latents decode the
  connection at the chunk's second coordinate token (accuracy 1.00; 6 of the 10 models at 8x8 and 10x10), SAE steering matches
  the vectors and the ceiling (e.g. 10x10 seed 2: 0.363 vs 0.364 vs 0.367). Where they only decode the cell (accuracy 0.34-0.41)
  it collapses (0.06-0.19), and the vectors also dip in those models (10x10 seed 4: 0.23 vs ceiling 0.36), so those models
  store the pair less linearly at this site. Four other SAE settings did not recover it on the tuning set.
- The class-mean patch (best method of experiment 07) is worse than the vectors at every size above 5x5: replacing the whole
  vector discards the position information that adding a difference keeps.
- Accounting for model quality (`plot_steering.py --normalize`, `fig_steering_lines_*_normalized`): the grafted models are slightly
  weaker solvers than the base model at 9x9-10x10 (original maze 0.44/0.36 fixed, 0.44/0.32 trainable, vs 0.48/0.39 base), so each
  method is also reported as a share of what the SAME model solves on these items with no prompt/target mismatch. That rate is
  estimated per model and seed from its original-maze rollouts, matched on path length; for the base model it lands 0.4-1.9 points
  above the measured prompt-rewrite ceiling, so the ceiling line sits at 97-99%. Shares by size 5..10: steering vectors
  73 76 97 93 96 89; SAE features 73 73 79 82 73 53; fixed graft 96 87 84 76 73 70; trainable graft 91 91 89 80 83 89.
  The ordering against the fixed graft is unchanged; the trainable graft ties the vectors at 10x10 and trails them at 7-9.
- Caveats: one site/layer for all models; the SAE result is a lower bound on what a per-model tuned dictionary could do;
  steering assumes a removed connection to overwrite, which spanning-tree targets always provide.

Infra: pilot + 15-shard run + SAE tuning check + 15-shard SAE rerun, all A100 SXM 80GB at $1.59/h, about 17 pod-hours (~$27);
all pods deleted. A bug worth remembering: modifying `weight.grad` outside `torch.no_grad()` in the SAE trainer put Adam's
moments on the autograd graph and leaked ~2 MB/step (OOM after ~35k steps); fixed in `eval_steering.py`.

## What the curves show (see figures)
- True world: grafted ≈ base at every size on the per-token metrics; the channel costs nothing at 5–8 and a
  little at 9–10 (trainable at 10×10 keeps 2% invalid mass at the end, base 0.3%).
- Learning dynamics differ: the base model's p(SP) rises in two stages (a first rise near step 100 to a ~0.4–0.5
  plateau, then the main rise), whereas the grafted models stay near zero until step ~300 and then rise in one
  sharp step, overtaking the base by step ~600 and reaching the same asymptote. p(Σ valid − SP) shows the same
  thing: the base's transient bump comes ~3× earlier than the grafted models'.
- Phantom world: p(SP) on the grafted world reaches 0.88–0.93 at every size (the models follow the channel),
  with ≤ 2% invalid mass; the phantom-world solve rate falls with grid size much faster than the true-world one
  (0.71 → 0.08 vs 0.97 → 0.40). The distance-stratified eval shows this is not a dissimilarity effect: in-family DFS
  mazes at any distance cost far less (see the section above); the phantom mazes are not spanning trees and their
  edits sit on the route.
- Seed spread (shaded bands) is small everywhere; 5 seeds resolve the base-vs-grafted differences.

## Caveats
- All 90 runs use bf16 autocast (the legacy runs were fp32-TF32); this is consistent within the campaign.
- Three 7×7 seed-4 runs finished on two pods; the aggregator keeps the first copy (`plot_curves.py` prints which).
- Rollouts are greedy; "solved" requires the target reached with zero rule breaks.

## Infra
30 RunPod pods total (13 A100 SXM, 5 A100 PCIe, 2 H100 SXM, 2 H100 PCIe/NVL mix, 2 RTX PRO 6000, 2 L40S …),
one run per GPU at the end; ~3 h wall-clock from first launch (14:25 PT) to last result (17:18 PT); estimated pod
cost ≈ $116. Scripts: `worker.sh` / `worker1.sh` (pod), `curves_pod.sh` (setup/run/status/pull), `eval_ckpts.py`,
`plot_curves.py`, `reap.sh`, `pods.json` (manifest incl. deletion times).

## Current figure set

Only PDF and SVG are exported; PNGs and superseded layouts have been removed.

- `fig_curves_fixed_overlay.{pdf,svg}`: fixed-graft learning curves, maze sizes overlaid within each model condition.
- `fig_curves_trainable_overlay.{pdf,svg}`: corresponding trainable-graft learning curves.
- `fig_final_fixed_lines.{pdf,svg}`: fixed-graft final-checkpoint results versus maze size; seed means ± SD.
- `fig_final_trainable_lines.{pdf,svg}`: corresponding trainable-graft final-checkpoint results.
  In both, the solve-rate panel uses the distance-stratified set (same maze, 0–10%, 10–30%, 30–60%); token panels use the phantom edit.
- `fig_steering_lines_{fixed,trainable}.{pdf,svg}`: base-model activation steering on the same set, route-changed items. One panel
  per maze size, x = maze difference, lines = steering vectors, SAE features, prompt-rewrite ceiling and the graft; the
  identical-maze control is a single hollow marker per model at 0%. The no-steering floor (0%) and the class-mean patch are
  not drawn; their numbers stay in `steering_cells.csv`. (Replaces the earlier steering heatmaps.)
- `fig_steering_lines_{fixed,trainable}_normalized.{pdf,svg}`: same, y = share of the same model's length-matched no-mismatch solve rate.
- `fig_distance_heatmap_routechanged.{pdf,svg}`: solve rate by maze size × Hamming-distance bucket (5% buckets + fresh DFS
  maze), fixed and trainable, restricted to items whose origin→target route differs between the two mazes (column 0
  unrestricted). The unrestricted and relative-to-column-0 heatmaps were dropped; their numbers are in `distance_cells.csv`.

All axes are linear. Learning curves show seed means without SD bands; final results include SD error bars.
The final checkpoint is the end of the training budget, not a claim of convergence.

Regenerate from the repository root:

```sh
.venv/bin/python maze/paper/curves/plot_curves.py
.venv/bin/python maze/paper/curves/plot_final_lines.py
.venv/bin/python maze/paper/curves/plot_final_lines.py --graft trainable
.venv/bin/python maze/paper/curves/plot_heatmap.py   # distance heatmaps + distance_cells.csv + distance_items.csv.gz
.venv/bin/python maze/paper/curves/plot_steering.py [--graft trainable]   # steering line figures + steering_cells.csv + steering_items.csv.gz
```

Raw results, aggregate CSVs, and evaluation/worker scripts are retained.

## Distance heatmap display

The route-changed heatmap pools independently sampled (formerly “Fresh”) mazes into their measured 5-percentage-point distance bins before averaging per-seed solve rates. The identical-maze control remains separate at 0. The original minimum of 1,000 rollouts per displayed cell is retained; sparse bins above 60% are omitted. Raw evaluation data and distance_cells.csv retain their original grouping.
