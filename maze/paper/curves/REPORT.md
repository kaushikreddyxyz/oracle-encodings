# Training-curve campaign: base vs grafted maze transformers, grid sizes 5–10, 5 seeds

Completed 2026-09-16 17:20 PT. Figures: `figures/` (the current set is listed below). Aggregate tables: `tables/`.
Data: `tables/all_metrics.csv` (6,900 rows: one per grid × seed × condition × checkpoint × world),
`tables/train_loss_by_ckpt.csv` (mean train loss over the window before each checkpoint), per-run raw files under
`results/<pod>/g{n}-s{s}/<run>/` (`eval/metrics.json`, `eval/per_maze.npz` with per-maze p_sp / p_valid / solved /
reached / clean for every checkpoint, `train_loss.csv` with every step, `model.final.pt`, `model.final.zanj`,
`injector.final.pt`, `config_holder.json`). Final models are also on wandb (project `maze-curves`, 93 finished runs
with a `final` artifact; the 12 crashed runs there are guard-killed duplicates and the fp32 false start).

## Layout of this folder
- `figures/` — every figure, PDF + SVG only (no PNGs).
- `tables/` — every aggregate CSV/JSON the figures and the numbers below are read from.
- `results/` — raw per-run outputs; never deleted. `results/distance/` is the standard eval set.
- `infra/` — pod and orchestration scripts (only needed to rerun GPU work); they resolve `results/` and the
  `eval_*.py` code relative to their parent, so they keep working from `infra/`.
- top level — this REPORT, the `eval_*.py` scripts that run on pods, `sae_lib.py`, `maze_distance.py`,
  `build_local.py`, `select_*.py`, and the `plot_*.py` scripts that rebuild `figures/` and `tables/`.

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

**Files.** `figures/fig_distance_heatmap_routechanged.{pdf,svg}`, `tables/distance_cells.csv`, `tables/distance_items.csv.gz`
(2.5M rollouts, one row each), `results/distance/{sets,out}` raw npz, `results/distance/pods.txt`.
Code: `eval_distance.py` (pod), `plot_heatmap.py`, `infra/distance_deploy_one.sh`.

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
30 base finals (6 sizes x 5 seeds). Figures: `figures/fig_steering_lines_{fixed,trainable}_merged.{pdf,svg}` (main) and
`figures/fig_steering_lines_fixed_by_size.{pdf,svg}`; data: `tables/steering_cells.csv`
(route-changed and all items, Wilson intervals), `tables/steering_items.csv.gz` (one row per rollout), raw npz in
`results/distance/steer_out/<method>/`, per-model statistics + SAEs in `results/distance/steer_prep/`, tuning-set outputs in
`results/distance/steer_tune/`. Code: `eval_steering.py` (pod; adjacency-token steering), `eval_steering_graftsite.py` (pod; token-matched steering), `plot_steering.py`, `infra/steer_deploy_one.sh`,
`infra/steer_status.sh`, `infra/steer_reap.sh`.

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
- Caveats: one site/layer for all models; the SAE result is a lower bound on what a per-model tuned dictionary could do;
  steering assumes a removed connection to overwrite, which spanning-tree targets always provide.

Infra: pilot + 15-shard run + SAE tuning check + 15-shard SAE rerun, all A100 SXM 80GB at $1.59/h, about 17 pod-hours (~$27);
all pods deleted. A bug worth remembering: modifying `weight.grad` outside `torch.no_grad()` in the SAE trainer put Adam's
moments on the autograd graph and leaked ~2 MB/step (OOM after ~35k steps); fixed in `eval_steering.py`.

## Token-matched steering: the fair comparison with the graft (2026-09-19)

Rule (binding for every graft-vs-steering comparison from here on): a steering method may be fitted only on, and applied only to,
the token positions the graft writes to. For the `complete` graft that is every position from `<ADJLIST_END>` onward, including
every token generated during the rollout; the adjacency list is never touched. `eval_steering_graftsite.py` enforces this with one
mask, `graft_tokens()`, used for fitting and application alike (a new graft type = change that mask; a new steering method = add
a fitter and a branch in `make_steer`). The layer is free. Same standard set, 30 base finals, 3,000 mazes x 5 seeds per cell.

Methods: steering vectors per connection feature fitted on graft tokens of 6,000 seed-777 mazes (120k-207k tokens per model):
`dom` = mean(open) - mean(closed); `reg` = ridge regression of the residual on the +-1 features with token-identity and position
one-hots as covariates (the graft's own linear form, read off the base model instead of trained). Edit: x += alpha * sum_f
(B_f - A_f) V_f at resid_pre of one block or of every block. SAE features: ReLU SAE (x8, l1 0.03) trained on graft-token residuals;
per connection the 16 latents whose activation differs most between open and closed; latents shifted by v x that difference.

Tuning set (300 mazes, seed 999, seed-0 models of 6x6 and 9x9; route-changed solve rate, floor 0.003 / 0.001): every single block
0-10 at alpha 1/4/16, blocks 0-11, 1-6, 6-11 together at alpha 0.5/1/2, SAE at blocks 3 and 6 with 4/16 latents and v 1/4/16.
Best values: 6x6 reg all blocks alpha 1 = 0.014, SAE block 6 v 16 = 0.020; 9x9 nothing above 0.004. Raising the strength only
stops the model solving its PROMPT maze (e.g. 6x6 reg block 3: prompt-maze solve rate 0.79 -> 0.28 -> 0.00 at alpha 1/4/16)
without moving it to the target maze. (The pilot pod's tuning outputs and sweep logs were not pulled before deletion; these
numbers are from the session transcript.)

Full run, route-changed solve rate, all bins pooled, sizes 5..10 (%):

| method | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| no steering | 0.0 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| steering vectors, graft tokens, block 3 (alpha 4) | 2.1 | 0.9 | 0.2 | 0.2 | 0.2 | 0.2 |
| steering vectors, graft tokens, all blocks (alpha 1) | 3.1 | 1.3 | 0.5 | 0.4 | 0.3 | 0.3 |
| SAE features, graft tokens (block 6) | 2.6 | 1.5 | 0.7 | 0.4 | 0.3 | 0.3 |
| fixed graft | 92.1 | 71.6 | 53.8 | 41.6 | 30.8 | 24.2 |
| trainable graft | 88.7 | 74.7 | 56.0 | 42.8 | 34.3 | 26.2 |
| steering vectors, adjacency-list tokens (not token-matched) | 70.2 | 59.4 | 61.1 | 50.1 | 43.6 | 32.7 |

- Under the token-matched rule steering is essentially null: the only visible effect is at 5x5 for the smallest edits
  (0-5% difference: 18% vectors, 13% SAE), gone by 15% difference and absent from 7x7 up.
- Why: at the graft tokens the base model barely represents the maze. The SAE latent most associated with a connection differs
  by a median of 0.3 SD (6x6) to 0.1 SD (10x10) between mazes where it is open and closed. The base model looks each move up in
  the adjacency list by attention, so there is no world-model at those positions to steer; the grafted models were trained to
  read one there. The adjacency-token result above shows the model IS steerable where it stores the maze, which the graft
  never touches, so it is reported in the table and CSVs only, not in the figure.
- Cost: pilot + 15 shards, A100 SXM at $1.59/h, about 8 pod-hours (~$13); all pods deleted.

**Which configuration the illustrative figures draw (2026-09-21, `select_best_steering.py`, `tables/steering_selection.json/.csv`).** The
steering figures show ONE "Steering vectors" and ONE "SAE features" line. Each is a steelman: for every base model (size x seed)
the candidate with the highest route-changed solve rate on the standard set itself is taken (vectors: ridge all blocks alpha 1 |
ridge block 3 alpha 4; SAE: the five `sae_lib` kinds at block 6 | the first 2,048-latent ReLU dictionary), and its per-item
outcomes are copied to `results/distance/steer_out/gs_vec_best` / `gs_sae_best`. Selecting on the evaluation metric per model is
an optimistic upper bound for steering, which is the point: the graft is compared against steering's best case. It changes
little: vectors 3.5 / 1.3 / 0.5 / 0.4 / 0.3 / 0.3 (fair single config 3.1 / 1.3 / 0.5 / 0.4 / 0.3 / 0.3); SAE 3.0 / 1.7 / 1.0 /
0.6 / 0.5 / 0.3 (fair: BatchTopK, chosen on the tuning set, 1.9 / 1.5 / 0.6 / 0.4 / 0.4 / 0.3). Kinds picked across the 30
models: Matryoshka 8, matching pursuit 8, BatchTopK 6, ReLU 4, TopK 4, first dictionary 0; vectors: all-blocks 28, block-3 2.

## SAE kinds as steering tools: rerun with properly built dictionaries (2026-09-21, complete)

Why: the earlier SAE results used one kind (ReLU+L1, plus plain TopK in a sweep), a fixed 2,048-latent dictionary, no held-out
split and no loss-recovered check. `sae_lib.py` implements five kinds behind one interface (ReLU+L1, TopK with aux-k, BatchTopK,
Matryoshka BatchTopK, matching pursuit); dictionary size is a multiple of the expected feature count at the site (adjacency
tokens: 4n(n-1) directed connections + n^2 cells; graft tokens: 2n(n-1) + n^2); training is 10k steps x 16,384 (164M samples),
warmup + linear decay, unit-norm decoder, last 5% of mazes held out, training data 12,000 mazes (adjacency) / 30,000 mazes
(graft tokens); every dictionary gets held-out FVU, L0, dead fraction and a loss-recovered score (next-token CE on path tokens
with the reconstruction spliced in, relative to mean-ablation). Both eval scripts use it. Same standard set, 30 base finals,
3,000 mazes x 5 seeds per cell. Figure `figures/fig_sae_kinds.{pdf,svg}`; data `tables/sae_kinds_cells.csv`, `tables/sae_kinds_training.csv`, raw
`results/distance/steer_out/*_f/`, dictionaries + diagnostics `results/distance/steer_prep|steer_gs_prep/*/stats_f*`.

Pilot (tuning set: 300 mazes, seed 999, seed-0 models of 6x6 and 9x9; 30 configs = 5 kinds x {1, 1.5, 3} x F_exp x 2 sparsity
levels; `results/distance/sae_logs/sae_selection_used.json`): the 3x dictionary won for every kind; 1.5x and 1x were worse.
Selected: relu-m3-l0.1, topk-m3-k8, btopk-m3-k8, mat-m3-k8, mp-m3-k16, 16 latents per connection, strength 1 (adjacency);
graft tokens the same specs except mp-m1-k16, strength 16. (Three token-matched pilot chunks had not reported when the selection
was frozen, so that selection used 4-8 of 12 configs per kind; every score was at the floor.)

Route-changed solve rate %, mean +- SD over 5 seeds, all maze-difference bins pooled:

| adjacency-list tokens | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| prompt rewritten (ceiling) | 95.8 | 77.4 | 62.5 | 52.0 | 44.8 | 36.6 |
| steering vectors | 70.1 +- 14.8 | 59.4 +- 4.3 | 61.1 +- 2.2 | 50.1 +- 4.2 | 43.6 +- 1.5 | 32.7 +- 5.4 |
| SAE, ReLU + L1 | 70.9 +- 13.3 | 54.2 +- 12.3 | 49.8 +- 24.3 | 42.9 +- 16.4 | 32.9 +- 15.8 | 20.4 +- 14.3 |
| SAE, TopK | 71.0 +- 12.6 | 48.4 +- 21.0 | 42.1 +- 26.8 | 39.8 +- 19.0 | 26.6 +- 16.3 | 16.4 +- 16.7 |
| SAE, BatchTopK | 70.8 +- 12.1 | 45.1 +- 20.2 | 49.1 +- 25.0 | 41.0 +- 19.6 | 32.2 +- 16.6 | 14.9 +- 15.8 |
| SAE, Matryoshka | 70.0 +- 11.9 | 43.2 +- 18.4 | 42.8 +- 26.0 | 34.0 +- 18.6 | 25.2 +- 16.0 | 14.2 +- 15.4 |
| SAE, matching pursuit | 70.7 +- 13.0 | 48.4 +- 20.3 | 42.4 +- 26.6 | 34.5 +- 19.3 | 24.0 +- 16.9 | 12.7 +- 13.7 |

Token-matched (graft tokens): fixed graft 92 / 72 / 54 / 42 / 31 / 24; steering vectors 3.1 / 1.3 / 0.5 / 0.4 / 0.3 / 0.3;
every SAE kind 1.1-2.9 at 5x5, 0.9-1.7 at 6x6, 0.6-1.0 at 7x7, <= 0.6 from 8x8; floor 0.0-0.2.

- No SAE kind beats steering vectors. Per model, the best of the five kinds beats the vectors by more than 2 points in 2 of 30
  models (max +4.1) and is 7.6 points worse on average. ReLU+L1 is the best kind, matching pursuit and Matryoshka the worst at
  large sizes; at 5x5 all kinds equal the vectors.
- SAE steering is bimodal per model, which is what the large seed SDs are: it matches the vectors (within 5 points) in 22/30
  models for ReLU, 20 BatchTopK, 18 TopK, 16 matching pursuit, 14 Matryoshka, and collapses in the rest. For ReLU the split is
  explained by the dictionary: in 95% of the matching models the latents at the chunk's second coordinate token decode the
  connection (accuracy > 0.95), against 12% of the failing models. For the TopK family and matching pursuit the latents often
  decode the connection and steering still fails (42-50% of failing models), so decodability is necessary, not sufficient.
- The token-matched null does not depend on the SAE kind.
- Dictionary quality (means over 30 models): adjacency ReLU FVU .0015, L0 48, 0% dead, loss recovered 1.000; matching pursuit
  FVU .0028, L0 16, 0% dead, 1.000; TopK FVU .0096, 19% dead, .999; BatchTopK .0053, 40% dead, .997; Matryoshka .0070, 35% dead,
  .995 (aux-k did not revive them; worst single model .969). Graft tokens: FVU .005-.035, loss recovered .983-1.000. Held-out
  FVU was still falling 2-19% over the last 3k steps (the lr decay), so training is near but not at a plateau.
- The 3x-feature-count dictionaries (470-1,400 latents) are no better than the earlier fixed 2,048-latent ReLU dictionary
  (8x8/9x9/10x10: 43/33/20 now, 44/33/20 then); larger was better within the sweep, so size is not the limit either way.

Infra and cost: the first attempt stopped at 29/60 units when the RunPod balance ran out (14:11 PT); after funds were added the
remaining 31 units (split into 39) ran on 29 pods in 33 minutes with no failures. This session's `ssh`/DNS were broken, so pods
were driven through `infra/fleet_ssh.py` (paramiko), `infra/fleet_queue.py` (work queue: staged uploads, never drops a unit, pulls each
unit before the next), `infra/fleet_harvest.py`, `infra/parse_pods.py`, pods created through the RunPod MCP with a pod-side auto-delete guard.
Cost: about $95 for the first attempt (its first wave ran ~65 min per unit because two agents saturated the laptop uplink)
plus about $25 for the resume. All pods deleted.

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
cost ≈ $116. Scripts: `infra/worker.sh` / `infra/worker1.sh` (pod), `infra/curves_pod.sh` (setup/run/status/pull), `eval_ckpts.py`,
`plot_curves.py`, `infra/reap.sh`, `infra/pods.json` (manifest incl. deletion times).

## Current figure set (`figures/`, 15 PDF + SVG pairs)

Only PDF and SVG are exported; PNGs and superseded layouts have been removed. Every figure below regenerates from
`tables/` and `results/` with the script named after it, no GPU needed.

Training curves and final checkpoints
- `fig_curves_fixed_overlay.{pdf,svg}` (`plot_curves.py` → `plot_curves_overlay.py`): fixed-graft learning curves,
  maze sizes overlaid within each model condition.
- `fig_curves_trainable_overlay.{pdf,svg}`: corresponding trainable-graft learning curves.
- `fig_final_fixed_lines.{pdf,svg}`, `fig_final_trainable_lines.{pdf,svg}` (`plot_final_lines.py`): final-checkpoint
  results versus maze size; seed means ± SD. The solve-rate panel uses the distance-stratified set (same maze, 0–10%,
  10–30%, 30–60%); the token panels use the phantom edit.

Maze difference (the standard eval set)
- `fig_distance_heatmap_routechanged.{pdf,svg}` (`plot_heatmap.py`): solve rate by maze size × Hamming-distance bucket
  (5% buckets + fresh DFS maze), fixed and trainable, restricted to items whose origin→target route differs between the
  two mazes (column 0 unrestricted). The unrestricted and relative-to-column-0 heatmaps were dropped; their numbers are
  in `tables/distance_cells.csv`.

Graft vs steering (main comparison)
- `fig_steering_lines_fixed_merged.{pdf,svg}`, `fig_steering_lines_trainable_merged.{pdf,svg}`
  (`plot_steering_merged.py`): MAIN figure. Left = the single-swap local experiment, right = the global result
  (solved vs maze difference), both averaged over the six maze sizes. Replaces the earlier `fig_steering_lines_*`
  mean-over-sizes figure, which `plot_steering_summary.py --view mean` can still rebuild if needed.
- `fig_steering_local_{fixed,trainable}_by_size.{pdf,svg}` and `fig_steering_global_{fixed,trainable}_by_size.{pdf,svg}`
  (`plot_steering_size_comparison.py`): the two panels of the main figure, one panel per maze size.
- `fig_steering_lines_fixed_by_size.{pdf,svg}` (`plot_steering.py`): the global result per maze size with all drawn
  methods (ceiling, graft, token-matched vectors, SAE); the adjacency-token steering, the no-steering floor and the
  class-mean patch are not drawn, their numbers stay in `tables/steering_cells.csv`.
- `fig_local_lines_{fixed,trainable}.{pdf,svg}` (`plot_local.py`): descriptive per-maze-size version of the local
  panel, plus the log-prob gain of the new move.
- `fig_sae_kinds.{pdf,svg}` (`plot_sae_kinds.py`): five SAE kinds vs steering vectors, ceiling, graft and floor; dot
  plot per maze size, adjacency-token steering on top, token-matched steering below, bars = SD over seeds.

All axes are linear. Learning curves show seed means without SD bands; final results include SD error bars.
The final checkpoint is the end of the training budget, not a claim of convergence.

Regenerate everything from the repository root (figures land in `figures/`, tables in `tables/`):

```sh
.venv/bin/python maze/paper/curves/plot_curves.py                        # tables/all_metrics.csv + train_loss_by_ckpt.csv + both overlay figures
.venv/bin/python maze/paper/curves/plot_final_lines.py                   # + tables/final_results_fixed_summary.csv
.venv/bin/python maze/paper/curves/plot_final_lines.py --graft trainable
.venv/bin/python maze/paper/curves/plot_heatmap.py                       # distance heatmap + tables/distance_cells.csv + distance_items.csv.gz
.venv/bin/python maze/paper/curves/select_best_steering.py               # tables/steering_selection.{json,csv} + results/distance/steer_out/gs_{vec,sae}_best
.venv/bin/python maze/paper/curves/plot_steering.py                      # per-size steering figure + tables/steering_cells.csv + steering_items.csv.gz
.venv/bin/python maze/paper/curves/plot_sae_kinds.py                     # + tables/sae_kinds_cells.csv + sae_kinds_training.csv
.venv/bin/python maze/paper/curves/plot_local.py [--graft trainable]     # + tables/local_cells.csv + local_items.csv.gz + local_selection.json
.venv/bin/python maze/paper/curves/plot_steering_merged.py [--graft trainable]         # main figure
.venv/bin/python maze/paper/curves/plot_steering_size_comparison.py [--graft fixed]    # its per-size companions
```

Raw results, aggregate tables, and the evaluation / pod scripts are retained.

## Distance heatmap display

The route-changed heatmap pools independently sampled (formerly “Fresh”) mazes into their measured 5-percentage-point distance bins before averaging per-seed solve rates. The identical-maze control remains separate at 0. The original minimum of 1,000 rollouts per displayed cell is retained; sparse bins above 60% are omitted. Raw evaluation data and tables/distance_cells.csv retain their original grouping.

## Local comparison: graft vs token-matched steering at one decision point (2026-09-21)

Question: the global figure (`fig_steering_lines_*`) scores whole rollouts on mazes that differ in many connections. Here the
unit is one decision: does a single edit of the world change the single next move it should change, and by how much?
Figures (no floor drawn, shaded bands = +-1 SD over the 5 seeds, ceiling thin dashed, each panel says which way is good):
- main, `figures/fig_steering_lines_{fixed,trainable}_merged.{pdf,svg}` (`plot_steering_merged.py`), meant to replace `fig_steering_lines_*`:
  left = this experiment, right = the global result (solved vs maze difference); everything is the mean over the six maze sizes
  (equal weight) on one shared y axis. The left panel shows five outcomes, each an UNCONDITIONAL share of all single-connection
  edits (none is conditional on the one before it). Follows the edit (higher is better): picks new move = the model's top choice
  at the decision token is the move that is correct in the edited maze; uses new door = the greedy rollout from the decision cell
  goes through the added connection; solves new maze = that rollout reaches the target with every move legal in the edited maze.
  Ignores the edit (lower is better): picks old move = the top choice is the pre-edit move; crosses new wall = the rollout goes
  through the removed connection. In the local panel steering uses the locally tuned strength (best case for steering), in the
  global panel the global-figure settings. Size-averaged values, picks new / uses door / solves | picks old / crosses wall (%):
  ceiling 89 / 91 / 80 | 10 / 3; fixed graft 80 / 81 / 70 | 17 / 8; trainable graft 83 / 84 / 72 | 15 / 7;
  steering vectors 41 / 34 / 14 | 43 / 37; SAE features (best of 6 dictionaries x 4 strengths, cross-fitted) 33 / 31 / 11 | 49 / 47.
- descriptive, `figures/fig_local_lines_{fixed,trainable}.{pdf,svg}` (`plot_local.py`): the same five outcomes per maze size, plus the
  log-prob gain of the new move (nats); steering at the tuned strength.
Data: `tables/local_cells.csv` (every
method x size x item subset x metric; mean, SD and t interval over seeds, maze-clustered SE, Wilson for rates),
`tables/local_items.csv.gz` (one row per item x model, headline methods), raw npz in `results/local/out/`, eval sets in
`results/local/sets/`, pod logs in `results/local/logs/`. Code: `build_local.py` (numpy only), `eval_local.py` (pod),
`plot_local.py` (tables, descriptive figure), `plot_steering_merged.py` (main figure), `infra/local_deploy.py` / `infra/local_fleet.py` / `infra/local_fleet2.py` (fleet).

**Definition (why a swap).** The prompt shows maze A of the standard set. The edit is ONE SWAP: remove a connection e- that lies
on A's origin->target route, add a connection e+ so that B = A - e- + e+ is again a spanning tree. A spanning tree stays one only
under a swap, so this is the smallest in-family edit; in a tree a swap changes the route iff e- is on it, so every item is at
once a BLOCK (a wall invented on the route) and an OPEN (a door the new route must use), and the rewritten prompt has the same
length (one chunk's two coordinate tokens). Primary family `dfs`: swaps after which B is still a DFS tree of the lattice from
some root (checked exactly: every non-tree lattice edge joins an ancestor and a descendant; validated against brute force), so B
is in the training family both as graft input and as prompt. Only 7% (5x5) to 0.5% (10x10) of route-changing swaps qualify;
up to 3 per maze are drawn uniformly. Robustness family `tree`: 2 swaps per maze drawn uniformly from all route-changing
spanning-tree swaps. Decision cell d = last cell the routes of A and B share. The path origin..d is teacher-forced (legal and
shortest in both mazes), and because both mazes are trees exactly one next move is correct in each: the old move (toward e-) and
the new move (toward e+), always different cells. Every item is re-verified on the pod with an independent BFS
(`eval_ckpts.shortest_path`). Items per size: 5,919 / 5,465 / 5,095 / 4,961 / 5,066 / 5,079 `dfs` (from 2,260-2,445 mazes) and
~6,000 `tree`; x 5 seeds. In 50-64% of `dfs` items the wall is the very next move, in 44-63% the door is.

**Conditions.** Grafted models: prompt A, the graft carries B (2 of the 2n^2 features flipped); reference = the same model with
graft A. Base model, token-matched rule as above (edit only at positions >= `<ADJLIST_END>`, prefix and generated tokens):
steering vectors `x += alpha (V[e+] - V[e-])` and SAE features, using the per-model statistics fitted for the global figure
(`steer_gs_prep/*/stats_sae.pt`, unchanged). Strength sweep, all reported in `tables/local_cells.csv`: ridge vectors in every block at
alpha 1/2/4/8/16, block 3 at alpha 4/16, difference-of-means vectors; SAE features at block 6 with six dictionaries, the first one (ReLU x8, l1 0.03, 2026-09-19) and the five
sae_lib kinds trained for the global figure (`stats_f*.pt`: relu-m3-l0.1, topk-m3-k8, btopk-m3-k8, mat-m3-k8, mp-m1-k16), each at
strength 16/64/256/1024 (24 SAE candidates per model; the 1024 runs added 2026-09-22 bracket the optimum, which the first sweep did not
at 9x9/10x10). "Tuned" = per model,
the setting with the highest p(new move) on one half of the mazes is the one scored on the other half (2-fold, so no item is
scored by a setting it helped choose; both folds chose the same setting in 59 of 60 cases). Chosen alpha grows with size: 1-2 at
5x5/6x6, 4 at 7x7/8x8, 8 at 9x9/10x10 (16 is worse, so the sweep brackets the optimum). SAE selection over the 24 candidates (`tables/local_selection.json`): strength 64 in 43 of 60
folds, 256 in 12, 16 in 5, 1024 never; kind topk-m3-k8 in 33 folds, the first dictionary in 14, relu-m3-l0.1 in 6, mat in 5, btopk in 2, mp
never; both folds agree in 28 of 30 models. topk wins at 7x7-10x10 (30 of 40 folds), the first dictionary at 5x5/6x6 (12 of 20). An
"oracle" selection (best candidate per model on all items, no held-out half) is recorded as `sae_oracle` in `tables/local_cells.csv`: it is
within 0.3 points of the cross-fitted value on every metric, so selection noise is not what limits SAE steering. Floor = base model, no edit; ceiling = base model with e-'s chunk rewritten to e+.
The KV-cache decoder was checked against the repo's full forward for the base and grafted models (max |dlogp| 3e-5, greedy
tokens identical).

**p(new correct move) at the decision token, %, `dfs` family, mean ± SD over 5 seeds** (maze-clustered SE of the pooled
estimate is 0.2-0.5 points everywhere; the seed-to-seed spread is the larger uncertainty):

| method | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| no edit (floor) | 3.3 ± 0.7 | 11.1 ± 1.2 | 13.5 ± 0.3 | 13.0 ± 0.3 | 13.3 ± 0.4 | 13.4 ± 0.3 |
| steering vectors, global-figure setting (all blocks, alpha 1) | 31.2 ± 4.8 | 27.6 ± 3.5 | 16.8 ± 1.0 | 14.9 ± 0.4 | 15.0 ± 0.4 | 14.3 ± 0.6 |
| steering vectors, strength tuned | 36.4 ± 6.2 | 33.6 ± 9.4 | 38.1 ± 5.9 | 38.7 ± 11.5 | 41.3 ± 7.9 | 37.1 ± 7.4 |
| SAE features, best of 6 dictionaries x 4 strengths, cross-fitted | 27.3 ± 5.6 | 33.5 ± 2.5 | 35.1 ± 2.8 | 34.7 ± 5.1 | 32.0 ± 2.5 | 28.6 ± 2.9 |
| fixed graft | 88.3 ± 3.9 | 78.1 ± 3.1 | 77.0 ± 3.2 | 77.7 ± 1.2 | 76.3 ± 1.8 | 75.2 ± 0.6 |
| trainable graft | 88.7 ± 2.8 | 81.7 ± 1.6 | 79.3 ± 3.9 | 78.9 ± 3.1 | 78.4 ± 4.9 | 77.9 ± 2.6 |
| prompt rewritten (ceiling) | 94.4 ± 0.9 | 87.0 ± 1.1 | 84.7 ± 0.1 | 85.1 ± 0.3 | 85.2 ± 0.6 | 83.8 ± 0.4 |

Log-prob effect of the same edit, nats, in the edit's direction (gain of the new move / drop of the old move):

| method | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| steering vectors, global-figure setting | 4.53 / 1.43 | 3.12 / 0.86 | 1.05 / 0.10 | 0.89 / 0.05 | 0.75 / 0.05 | 0.50 / 0.03 |
| steering vectors, strength tuned | 5.38 / 3.08 | 3.82 / 1.85 | 4.23 / 2.17 | 4.21 / 1.89 | 4.47 / 2.76 | 4.10 / 2.29 |
| SAE features, best-of, cross-fitted | 4.11 / 2.43 | 3.67 / 3.34 | 3.78 / 1.91 | 4.00 / 1.45 | 3.50 / 1.22 | 3.28 / 2.14 |
| fixed graft | 7.97 / 7.03 | 6.75 / 5.68 | 6.21 / 5.34 | 6.21 / 4.96 | 5.67 / 4.57 | 5.35 / 4.12 |
| trainable graft | 8.00 / 7.06 | 6.77 / 6.01 | 6.25 / 5.76 | 6.05 / 5.37 | 5.79 / 5.10 | 5.07 / 4.58 |
| prompt rewritten (ceiling) | 8.16 / 8.07 | 7.00 / 7.12 | 6.52 / 6.70 | 6.64 / 6.67 | 6.40 / 6.60 | 6.22 / 6.24 |

Greedy rollout from the decision cell, % (uses the opened door / crosses the new wall / solves the edited maze):

| method | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| no edit (floor) | 0.4 / 98.3 / 0.4 | 2.1 / 92.2 / 1.7 | 2.9 / 89.6 / 1.8 | 3.1 / 89.1 / 1.8 | 2.7 / 88.7 / 1.4 | 3.0 / 87.8 / 1.3 |
| steering vectors, strength tuned | 34.1 / 33.4 / 15.5 | 26.7 / 46.6 / 14.3 | 33.0 / 36.7 / 16.3 | 33.5 / 43.4 / 17.8 | 40.4 / 29.6 / 11.9 | 34.9 / 34.6 / 10.0 |
| SAE features, best-of, cross-fitted | 25.8 / 47.7 / 11.8 | 34.2 / 31.5 / 9.4 | 32.4 / 46.7 / 12.1 | 32.6 / 56.0 / 14.8 | 28.7 / 57.9 / 12.9 | 31.4 / 43.6 / 4.8 |
| fixed graft | 90.6 / 4.6 / 88.9 | 81.0 / 8.8 / 76.5 | 80.1 / 8.8 / 70.9 | 80.2 / 8.6 / 67.7 | 78.5 / 8.6 / 60.8 | 77.7 / 8.5 / 54.4 |
| trainable graft | 91.1 / 4.9 / 89.5 | 85.5 / 6.0 / 80.9 | 83.0 / 7.3 / 73.9 | 82.4 / 8.1 / 69.3 | 80.9 / 7.4 / 62.9 | 81.4 / 5.7 / 53.9 |
| prompt rewritten (ceiling) | 96.8 / 1.3 / 95.4 | 91.8 / 2.4 / 86.9 | 89.5 / 2.9 / 80.8 | 89.7 / 3.2 / 77.6 | 88.8 / 2.9 / 72.8 | 87.8 / 3.0 / 66.6 |

**Findings.**
1. At a single decision the graft works almost as well as rewriting the prompt and far better than any token-matched steering.
   p(new move): graft 75-89%, ceiling 84-94%, best steering 34-41%, floor 3-13%. In nats the graft delivers 82-98% of the
   ceiling's gain of the new move (5.1-8.0 of 6.2-8.2) and 66-88% of its suppression of the old move; tuned steering vectors 55-70%
   and 26-42%. The graft's shortfall to the ceiling is 5-9 points of p(new move) at every size, so the growth of the global gap
   with maze size is compounding over a longer rollout, not a weaker local effect.
2. The cleanest stratum, where the wall is the very next move and the door is at the decision cell (1,754-2,301 items per size;
   floor 0.0-0.1%): ceiling 97-99%, fixed graft 90-97%, trainable 92-95%, tuned steering vectors 30-52%, the global-figure steering
   setting 20% at 5x5 and <= 1% from 7x7. Where the wall lies further down the route every method drops, the ceiling included
   (60-90%; graft 49-83%; floor 5-21%): the hard part there is the model's own look-ahead at a fork, not the edit channel.
3. Token-matched steering is not null locally, unlike in the global figure, but it has to be pushed hard and then breaks the model.
   With the strength tuned per model (alpha 4-8 at 7x7 and above, vs alpha 1 chosen on global solve rate) the new move reaches
   34-41% and the rollout goes through the door 27-40% of the time, but the old move keeps 33-50% of the mass, the wall is still
   crossed in 30-47% of rollouts, only 10-18% of rollouts then solve the edited maze (graft 54-89%), and the result varies
   strongly between seeds (SD 6-12 points vs 0.6-5 for the graft). At the global-figure setting the effect is confined to 5x5/6x6
   (31% / 28%) and is within 1-3 points of the floor from 7x7, which matches the global result. SAE features stay below the
   tuned vectors at every size (27-35% vs 34-41%) even when the best of six dictionaries and four strengths is taken per model:
   the best kind (topk-m3-k8) gains 2-7 points over the first dictionary at 7x7-9x9, none at 5x5/6x6, and the optimistic oracle
   selection adds nothing (see the selection paragraph).
4. Trainable >= fixed graft locally at every size (by 0.4-3.6 points of p(new move); not resolved from seed noise at any single
   size). The grafted models read the world from the graft, not the prompt: rewriting only the prompt moves them by 0.5-1.5
   nats (p(new move) 7-20%), and rewriting both adds 3-10 points over the graft alone, landing on the base model's ceiling (81-95%).
5. DFS-ness of the edited maze matters little at matched look-ahead: for arbitrary spanning-tree swaps whose wall is the next
   move, the ceiling is 95-97% and the fixed graft 82-93% (vs 97-99% / 90-96% for DFS-preserving swaps). The `tree` family's lower
   pooled numbers (ceiling 61-90%, fixed graft 49-84%) come from its larger share of far-ahead walls (wall next in 17-36% of items
   vs 50-64%).

**Caveats.** Floor p(new move) is 11-13% from 6x6 up because decision cells are forks, where the unedited base model already
errs; the nats columns and the wall-next stratum are free of this. The tuned steering lines use the local metric for tuning
(cross-fitted), the global figure used global solve rate: the two are not the same setting. The SAE strength sweep is bracketed at every size
(1024 is never selected). The five sae_lib dictionaries are the ones trained for the global figure (block 6, graft tokens), used as is. Mean nats are
dominated by items that move from ~1e-4 to ~1e-1. Rollouts are greedy and start at the decision cell. Runs were split over
A100 and H100 pods; the same model on the two GPU types differs by <= 1.5e-3 in probability and 0.02% of argmaxes. The figure
palette is the project's fixed one; it fails the dataviz validator's lightness/chroma bands (navy, grey) but passes its
colour-blind separation check, and the ceiling/floor are additionally dashed/dotted.

**Infra / cost.** 16 pods created, all deleted (checked against the account's pod list): 8 A100 SXM at $1.59/h (2 deleted after
14 min: their volume was too slow to deploy to), 4 H100 SXM at $3.49/h, 3 L40S at $1.09/h (deleted unused after 13 min), plus
the pilot. About $44 in total, over the $40 cap: all jobs had finished by about 20:15 UTC but the pods were only pulled and
deleted at 20:48 because the waiting command on the laptop hung, roughly $13 of idle time. A100 capacity ran out mid-run and the
laptop uplink fell to 0.4 MB/s for a while, which is why the run was rebalanced twice (`local_fleet.py`, `local_fleet2.py`:
holder pods run what they already hold, md5-verified; fed pods get one large base model at a time).

SAE-kinds run 2026-09-22: 60 units (30 base models x 2 halves of the candidate list) over 10 A100 pods (6 SXM, 4 PCIe, $1.59/h) through
`infra/local_queue.py` (a copy of `infra/fleet_queue.py`), 66 minutes wall-clock, about $18; all pods deleted. Decoding was made ~1.7x faster for this
run (`FAST_ATTN` in `eval_local.py`: plain matmul attention for the single query token instead of SDPA on non-contiguous cache slices;
verified against the full forward, max |dlogp| 3e-5).

Regenerate: `.venv/bin/python maze/paper/curves/build_local.py` (sets; deterministic, seed 4321), then
`.venv/bin/python maze/paper/curves/plot_local.py [--graft trainable]` (tables + descriptive figure) and
`.venv/bin/python maze/paper/curves/plot_steering_merged.py [--graft trainable]` (main figure).
