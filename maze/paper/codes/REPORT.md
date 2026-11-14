# Graft codes beyond walls: egocentric and axis-split degree grafts vs the edge graft (6x6 pilot, 2026-09-23; scale-up grids 5-10 x 5 seeds in sec. 5, 2026-09-24)

Question: does a maze transformer accept world-state grafts that are not "one direction per wall", and does it use them
the way it uses the edge graft? Sections 1-4 (pilot): 6x6, seed 0, jirpyish d256/12L, 2M random-DFS mazes x 4 epochs (31,248
steps), prompt INTACT, graft injected at block-0 input from the `<ORIGIN_END>` token onward (`start = origin_end`).
Blank-prompt smoke runs and the onion pilot are archived (`results/_smoke_blank_prompt/`, `../_archive/onion/`).

## Codes (`maze/core/graft.py` interface; `codes_ego.py`, `codes_axdeg.py`; tests in `maze/core/tests/`)
* **edge** (reference) -- one +-1 feature per connection, one random unit direction each (2n^2 = 72 features).
* **ego** -- egocentric walls: at every position the whole wall map re-indexed relative to the CURRENT cell (offsets
  dr, dc in [-5, 5] x {down, right}: 242 features, exactly 72 nonzero). The frame is the origin from `<ORIGIN_END>` and
  moves with every path token, teacher-forced or generated (dynamic code: `init_state/step` in every KV decoder). The
  same absolute wall is feature i at one step and feature j later; nothing about the graft is an absolute wall.
* **axdeg** -- axis-split degree: per cell h = number of open horizontal connections and v = vertical (0..2), value
  count-1 in {-1, 0, +1}, one direction per (cell, axis) (72 features). Complete (rows/columns are paths, so the counts
  invert triangularly); affine in the wall bits, i.e. an edge graft with tied directions (adjacent walls: cos 0.5).
* **axdeg1h** -- the same counts as a one-hot over {0, 1, 2} per (cell, axis) (216 features, 72 active): complete and
  non-affine in the wall bits (linear fit R^2 0.58).
Injected norm matched to the edge graft's. Trainer flags `--inject_code`, `--inject_start`. The evaluators were
refactored onto the code interface and verified bit-exact against the legacy path; both KV decoders are checked
against the full forward on real checkpoints (`test_graft_eval_equivalence.py`, `test_graft_decode_paths.py`).

## 1. Following the graft: standard set and one-swap set (6 pods, ~$12)
`eval_local` drives grafted models through their own code (`GraftDriver`), so the one-swap test covers the dynamic ego code.

| code | W | true solved | phantom solved | standard set, route-changed: solves graft maze / prompt maze | one-swap: p(new) / uses door / crosses wall / solves edited |
|---|---|---|---|---|---|
| edge (matched ref) | fixed | 0.981 | 0.534 | 0.725 / 0.016 | 0.77 / 0.79 / 0.12 / 0.77 |
| ego | fixed | 0.941 | 0.556 | 0.714 / 0.005 | 0.81 / 0.85 / 0.07 / 0.81 |
| ego | trainable | 0.948 | **0.610** | 0.721 / 0.004 | **0.83 / 0.87 / 0.06 / 0.83** |
| axdeg (count values) | fixed | 0.985 | 0.575 | 0.612 / 0.037 | 0.77 / 0.77 / 0.18 / 0.76 |
| axdeg1h (one-hot counts) | fixed | 0.978 | **0.087** | **0.082 / 0.425** | **0.19** / 0.14 / 0.80 / 0.14 (follows the prompt: p(new) 0.77 with the prompt rewritten) |
| axdeg1h (one-hot counts) | trainable | 0.982 | 0.227 | 0.311 / 0.092 | 0.50 / 0.49 / 0.41 / 0.47 |
| edge, curves campaign (5 seeds, 1 ep, adjlist_end) | fixed | 0.838 | 0.480 | 0.716 / 0.009 | 0.78 / 0.81 / 0.09 / 0.77 |

Probes (intact prompt): edge / axdeg / axdeg1h have linear walls at the input by construction (1.00) and keep them
(0.87-0.92 at L12); ego never has them (0.68 at input, 0.75-0.78 at `<PATH_START>` by L6 -- what the prompt supplies)
and keeps the egocentric N/E/S/W walls at 1.00 through every layer.

### Findings
1. **Ego is adopted at edge level against the prompt**: route-changed 0.71-0.72 vs edge 0.73, and it is the best follower
   on a single edit (p(new move) 0.81/0.83 vs 0.77; crosses the new wall 6-7% vs 12%). Its only cost is true-world solve
   rate (0.94-0.95 vs 0.98). The count-value code (affine = tied-direction edge graft) is adopted too (0.61).
2. **The one-hot count code is dropped in favour of the prompt** (route-changed 0.08 graft / 0.43 prompt; single edit 0.19),
   exactly like the onion -- although with the prompt blanked the same code reached 0.99. A trainable W recovers only part
   (0.31 / 0.50).
3. So the "no privilege" result holds only when the graft is the sole source. In competition with the tokens the model keeps
   the codes that are affine in the wall bits (edge, axdeg, ego = an index-permuted edge code) and prunes the ones that are
   not (onion, one-hot counts; the corner one-hot would presumably fall on this side too). Egocentric framing is not a
   handicap; non-affine feature values are.
4. A fixed 4-epoch edge graft with the prompt intact follows the graft no better than the 1-epoch campaign models (0.725 vs
   0.716 route-changed).

## 2. Inference-time ablations: is the graft used the way the edge graft is? (`eval_ablate.py`; true world, 2,000 mazes)
Conditions: graft removed everywhere / only for generated tokens / only for the prompt; scaled x0.5, x2; replaced by noise
of the same per-position norm; adjacency list blanked with the graft on (graft only) and off (floor); graft carrying a
different maze than the prompt (shuffled). Solve rate (greedy rollout) / p(invalid move) teacher-forced:

| model | full | graft off | off gen | off prompt | x0.5 | x2 | noise | graft only | floor | shuffled |
|---|---|---|---|---|---|---|---|---|---|---|
| edge fixed | 0.98 / .000 | 0.00 / .90 | 0.00 / .76 | 0.00 / .40 | 0.63 / .03 | 0.00 / .63 | 0.00 / .95 | 0.00 / .41 | 0.00 / 1.0 | 0.08 / .30 |
| ego fixed | 0.94 / .001 | 0.00 / .83 | 0.02 / .71 | 0.08 / .12 | 0.68 / .01 | 0.01 / .05 | 0.00 / .93 | 0.25 / .02 | 0.00 / .82 | 0.07 / .32 |
| ego trainable | 0.95 / .000 | 0.00 / .98 | 0.00 / .86 | 0.10 / .12 | 0.81 / .02 | 0.02 / .01 | 0.00 / .92 | 0.27 / .02 | 0.00 / .89 | 0.07 / .33 |
| axdeg fixed | 0.99 / .000 | 0.00 / .79 | 0.01 / .65 | 0.00 / .44 | 0.55 / .03 | 0.01 / .61 | 0.00 / .92 | 0.37 / .05 | 0.00 / .62 | 0.09 / .26 |
| axdeg1h fixed | 0.98 / .000 | 0.00 / .92 | 0.00 / .76 | 0.00 / .52 | 0.27 / .06 | 0.00 / .60 | 0.00 / .93 | 0.14 / .21 | 0.00 / .52 | 0.41 / .04 |
| axdeg1h trainable | 0.98 / .000 | 0.00 / .86 | 0.01 / .66 | 0.00 / .51 | 0.17 / .24 | 0.01 / .46 | 0.00 / .91 | 0.08 / .27 | 0.00 / .94 | 0.13 / .22 |
| base (no graft), 4 ep | 0.97 / .000 | - | - | - | - | - | - | 0.00 / .94 (prompt blank) | - | - |

* Every grafted model, whatever the code, collapses without its graft: 0% solved, 79-98% of the next-token mass on invalid
  moves, and replacing the graft by same-norm noise is as bad. The model does not fall back to reading the prompt; the graft
  is a trained interface, not an optional hint (the edge finding of the 2026-08 hunches, reproduced for ego and axdeg).
* Removing the graft only during generation is equally fatal; removing it only during the prompt costs the static codes
  everything but leaves ego at p(invalid) 0.12 -- the ego frame is undefined until the origin anyway, so its prompt-side
  graft carries little.
* Dose-response is the same shape everywhere (x0.5 halves the solve rate, x2 breaks it); ego is the most tolerant of x2
  on the token metric (p(invalid) 0.01-0.05 vs 0.6 for the static codes) because its per-position code is sparser.
* Graft only (prompt blanked): intact-trained models cannot run on the graft alone (0.00-0.37); the edge model least of all.
* Shuffled graft: edge, ego and axdeg models follow neither maze (0.07-0.09 solved), i.e. they combine both sources and
  break when they disagree; the one-hot count model follows the PROMPT (0.41) -- the same prompt preference it shows on the
  standard set.

## 3. Probes: are the grafted features turned into wall features downstream? (`probe_walls.py`, ridge, 4,000 mazes)
Ridge probes at block-0 input (incl. the graft) and after every block, at path tokens and at `<PATH_START>`; targets =
the 60 absolute wall bits and the 4 walls N/E/S/W of the current cell. New: the geometry of the wall readout directions,
|cos| between the probe directions of adjacent walls (sharing a cell) vs distant walls.

| model | abs walls: input -> L6 -> L12 (path tokens) | abs walls at PATH_START, L6 | local N/E/S/W: input -> L6 | adjacent-wall probe |cos|: input -> L4 -> L12 (distant pairs ~0.05) |
|---|---|---|---|---|
| edge fixed | 1.00 -> 0.99 -> 0.92 | 1.00 | 0.72 -> 0.86 | 0.08 -> 0.23 -> 0.27 |
| axdeg fixed | 1.00 -> 0.96 -> 0.87 | 1.00 | 0.72 -> 0.86 | 0.28 -> 0.26 -> 0.27 |
| axdeg1h fixed | 1.00 -> 0.91 -> 0.82 | 0.96 | 0.72 -> 0.86 | 0.22 -> 0.26 -> 0.28 |
| axdeg1h trainable | 1.00 -> 0.95 -> 0.87 | 1.00 | 0.72 -> 0.88 | 0.15 -> 0.25 -> 0.28 |
| ego fixed | 0.68 -> 0.72 -> 0.69 | 0.75 | 1.00 -> 1.00 | 0.29 -> 0.25 -> 0.32 |
| ego trainable | 0.68 -> 0.73 -> 0.68 | 0.78 | 1.00 -> 1.00 | 0.29 -> 0.24 -> 0.31 |
| base (no graft), 4 ep | 0.61 -> 0.76 -> 0.71 | **0.99** | 0.69 -> 0.88 | 0.30 -> 0.23 -> 0.28 |

* The static codes are not "converted" into edge-like wall features -- the opposite: by block 4 every model, the edge
  model included, holds the walls in the same geometry (adjacent walls correlated |cos| ~0.25, distant ~0.05). The edge
  model's orthogonal input directions are re-tied downstream to what the axdeg code injects directly.
* The ego models never build a linear absolute-wall representation: 0.68-0.75 at `<PATH_START>`, BELOW the base model's
  0.99 there -- so the ego graft does not merely fail to be translated into walls, it displaces the absolute wall map the
  base model would otherwise build from the prompt. The egocentric walls stay perfectly linear at every layer. Every
  model, base included, builds egocentric N/E/S/W features downstream (0.69-0.72 -> 0.86-0.88 by block 6), and the base
  model's own wall geometry converges to the same adjacent |cos| ~0.25 as every grafted model.

## 4. Learning dynamics (`fig_codes_curves_intact`)
Onset and asymptote of p(SP) and solve rate are the same for edge, ego and axdeg (all clear 0.9 solved by ~2 epochs, and the
4-epoch base model lands at 0.97, so the true-world numbers of the grafted models -- edge 0.98, axdeg 0.99, ego 0.94 -- are
"base-level", ego slightly below); the one-hot count models' phantom-world curves rise early and then decay as training
proceeds (the graft is being pruned).

## Summary of evidence that ego / axdeg grafts are the same kind of object as the edge graft
Same following of the graft against the prompt (sec. 1), same total dependence and same failure modes under ablation,
scaling and noise (sec. 2), same downstream wall geometry for the static codes and the same egocentric features built by all
(sec. 3), same learning curves (sec. 4). The differences: ego costs ~4 points of true-world solve rate and never re-encodes
absolute walls; the one-hot count code is the odd one out on the graft-vs-prompt axis.

Caveats: single seed, 6x6 only, one start rule (origin_end) for all codes (the curves-campaign edge reference used
adjlist_end). Linear probes only. Ego was never run under adjlist_end; vertex was scrapped.

## 5. Scale-up: allocentric (edge) vs egocentric (ego) graft, grids 5x5-10x10, 5 seeds (`figures_ev.py`, `fig_ev_*`)
Design: for every grid 5-10, seeds 0-4 x {edge, ego} fixed-W grafts, 4 epochs, prompt intact, injected from `<ORIGIN_END>`
(6x6 = the pilot's 5 seeds), plus one base model (no graft, seed 0) per grid; 9x9 and 10x10 seeds 0-2 continued to 8 epochs; probes re-run 2026-09-24 with AUROC / balanced accuracy / per-class recall (`probe_v2.sh`, 12 A100 pods, ~$7; outputs in `results/probe_v2/`)
(`worker_cont.sh`: weights warm-started, AdamW moments reset, same 2M mazes). Every model: eval_ckpts (true + phantom world,
2,000 mazes, all 40 checkpoints), eval_ablate, probe_walls, probe_nonlinear (all 13 sites), and for grafted models the
standard set (eval_distance) and the one-swap set (eval_local, ego driven through `GraftDriver`). Checkpoints in
`results/runs/`, mirrored to HF `kaushikreddyxyz/maze-graft-codes` (models only). Tables below come from `ev_tables.py`;
raw rows in `ev_summary.csv` (one per model) and `ev_probes.csv` (one per model x site x target x probe). Pods:
`fleet_ev.json`, `fleet_cont.json` (H100 SXM, $3.49/h, 57 (incl. 4 that never booted and were recreated) pods, ~$$400).

Figures, in story order:
* `fig_ev_main` -- does the model work with the other code? one-swap p(new move), route-changed graft solve, true-world solve, vs grid.
* `fig_ev_ablate` -- does it depend on the graft the same way? solve rate and p(invalid) under graft removal / noise / scaling / swap.
* `fig_ev_probes` -- does the code set the internal frame? AUROC of linear (ridge) probes for allocentric and egocentric wall targets, all layers, per model (MLP probes dropped from the figure at the user's request; their numbers stay in `ev_probes.csv`).
* `fig_ev_curves` -- how is it learned? solve rate (true and phantom world) vs epoch, grids overlaid.
* `fig_ev_collapse` -- why some 9x9 / 10x10 ego seeds solve ~0% at 4 epochs (`diag_collapse.py`).
* `fig_ev_cont` -- 9x9 / 10x10 continued to 8 epochs: delay or ceiling?

### Final metrics, 4 epochs (seed mean ± SD)

| grid | code | true solved | phantom solved | route-changed: graft solve | one-swap p(new) | crosses new wall |
|---|---|---|---|---|---|---|
| 5×5 | base | 1.00 (n=1) | - | - | - | - |
| 5×5 | edge | 1.00 ± 0.00 (n=5) | 0.50 ± 0.06 (n=5) | 0.70 ± 0.06 (n=5) | 0.68 ± 0.03 (n=5) | 0.22 ± 0.03 (n=5) |
| 5×5 | ego | 0.99 ± 0.00 (n=5) | 0.47 ± 0.03 (n=5) | 0.66 ± 0.02 (n=5) | 0.67 ± 0.01 (n=5) | 0.15 ± 0.03 (n=5) |
| 6×6 | base | 0.97 (n=1) | - | - | - | - |
| 6×6 | edge | 0.98 ± 0.00 (n=5) | 0.46 ± 0.09 (n=5) | 0.65 ± 0.08 (n=5) | 0.71 ± 0.06 (n=5) | 0.19 ± 0.05 (n=5) |
| 6×6 | ego | 0.95 ± 0.01 (n=5) | 0.54 ± 0.03 (n=5) | 0.70 ± 0.02 (n=5) | 0.80 ± 0.02 (n=5) | 0.08 ± 0.01 (n=5) |
| 7×7 | base | 0.89 (n=1) | - | - | - | - |
| 7×7 | edge | 0.91 ± 0.02 (n=5) | 0.39 ± 0.05 (n=5) | 0.62 ± 0.07 (n=5) | 0.75 ± 0.05 (n=5) | 0.13 ± 0.04 (n=5) |
| 7×7 | ego | 0.83 ± 0.02 (n=5) | 0.51 ± 0.02 (n=5) | 0.72 ± 0.02 (n=5) | 0.86 ± 0.01 (n=5) | 0.04 ± 0.01 (n=5) |
| 8×8 | base | 0.73 (n=1) | - | - | - | - |
| 8×8 | edge | 0.76 ± 0.02 (n=5) | 0.21 ± 0.02 (n=5) | 0.41 ± 0.05 (n=5) | 0.72 ± 0.03 (n=5) | 0.15 ± 0.02 (n=5) |
| 8×8 | ego | 0.73 ± 0.01 (n=5) | 0.37 ± 0.02 (n=5) | 0.62 ± 0.01 (n=5) | 0.87 ± 0.00 (n=5) | 0.03 ± 0.00 (n=5) |
| 9×9 | base | 0.62 (n=1) | - | - | - | - |
| 9×9 | edge | 0.62 ± 0.01 (n=5) | 0.14 ± 0.02 (n=5) | 0.25 ± 0.06 (n=5) | 0.67 ± 0.07 (n=5) | 0.21 ± 0.08 (n=5) |
| 9×9 | ego | 0.42 ± 0.25 (n=5) | 0.20 ± 0.11 (n=5) | 0.37 ± 0.22 (n=5) | 0.86 ± 0.02 (n=5) | 0.02 ± 0.01 (n=5) |
| 10×10 | base | 0.54 (n=1) | - | - | - | - |
| 10×10 | edge | 0.54 ± 0.02 (n=5) | 0.10 ± 0.01 (n=5) | 0.19 ± 0.04 (n=5) | 0.68 ± 0.04 (n=5) | 0.19 ± 0.06 (n=5) |
| 10×10 | ego | 0.29 ± 0.25 (n=5) | 0.12 ± 0.09 (n=5) | 0.26 ± 0.22 (n=5) | 0.85 ± 0.01 (n=5) | 0.02 ± 0.01 (n=5) |

### 4 vs 8 epochs at the largest grids (per seed: true solved / phantom solved / route-changed graft solve / p(new))

| grid | code | seed | 4 epochs | 8 epochs |
|---|---|---|---|---|
| 9×9 | edge | 0 | 0.60 / 0.13 / 0.25 / 0.67 | 0.69 / 0.13 / 0.20 / 0.60 |
| 9×9 | edge | 1 | 0.63 / 0.12 / 0.14 / 0.54 | 0.71 / 0.10 / 0.10 / 0.43 |
| 9×9 | edge | 2 | 0.61 / 0.14 / 0.34 / 0.75 | 0.69 / 0.14 / 0.34 / 0.75 |
| 9×9 | ego | 0 | 0.28 / 0.13 / 0.26 / 0.84 | 0.68 / 0.33 / 0.61 / 0.88 |
| 9×9 | ego | 1 | 0.65 / 0.31 / 0.57 / 0.88 | 0.73 / 0.34 / 0.62 / 0.89 |
| 9×9 | ego | 2 | 0.01 / 0.03 / 0.01 / 0.83 | 0.01 / 0.03 / 0.01 / 0.84 |
| 10×10 | edge | 0 | 0.51 / 0.10 / 0.21 / 0.70 | 0.58 / 0.10 / 0.22 / 0.70 |
| 10×10 | edge | 1 | 0.57 / 0.09 / 0.14 / 0.61 | 0.61 / 0.06 / 0.09 / 0.48 |
| 10×10 | edge | 2 | 0.55 / 0.09 / 0.18 / 0.67 | 0.59 / 0.07 / 0.12 / 0.58 |
| 10×10 | ego | 0 | 0.30 / 0.10 / 0.26 / 0.84 | 0.57 / 0.21 / 0.49 / 0.86 |
| 10×10 | ego | 1 | 0.01 / 0.02 / 0.01 / 0.84 | 0.01 / 0.02 / 0.01 / 0.84 |
| 10×10 | ego | 2 | 0.54 / 0.21 / 0.48 / 0.86 | 0.64 / 0.25 / 0.54 / 0.88 |

### Probes: AUROC (threshold-free, chance 0.5) of wall targets at path tokens (linear ridge / MLP), pooled over grids and seeds

| model | target | layer 0 (input) | layer 3 | layer 6 | layer 9 | layer 12 |
|---|---|---|---|---|---|---|
| base | allocentric walls | 0.51 / 0.51 | 0.70 / 0.72 | 0.69 / 0.72 | 0.66 / 0.70 | 0.64 / 0.65 |
| base | egocentric walls (±2) | 0.60 / 0.60 | 0.69 / 0.77 | 0.69 / 0.78 | 0.67 / 0.76 | 0.66 / 0.70 |
| edge | allocentric walls | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 | 0.92 / 0.95 | 0.87 / 0.89 |
| edge | egocentric walls (±2) | 0.66 / 0.73 | 0.69 / 0.88 | 0.72 / 0.93 | 0.71 / 0.91 | 0.69 / 0.79 |
| ego | allocentric walls | 0.63 / 0.85 | 0.66 / 0.85 | 0.66 / 0.83 | 0.64 / 0.79 | 0.63 / 0.69 |
| ego | egocentric walls (±2) | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 | 0.96 / 0.98 | 0.94 / 0.96 |

### Probes: R² of the ±1 targets of wall targets at path tokens (linear ridge / MLP), pooled over grids and seeds

| model | target | layer 0 (input) | layer 3 | layer 6 | layer 9 | layer 12 |
|---|---|---|---|---|---|---|
| base | allocentric walls | -0.01 / -0.01 | 0.14 / 0.19 | 0.13 / 0.19 | 0.10 / 0.15 | 0.08 / 0.09 |
| base | egocentric walls (±2) | 0.06 / 0.06 | 0.15 / 0.30 | 0.15 / 0.34 | 0.13 / 0.29 | 0.11 / 0.19 |
| edge | allocentric walls | 1.00 / 0.99 | 0.96 / 0.97 | 0.75 / 0.85 | 0.52 / 0.63 | 0.41 / 0.47 |
| edge | egocentric walls (±2) | 0.08 / 0.29 | 0.13 / 0.51 | 0.18 / 0.62 | 0.17 / 0.55 | 0.14 / 0.32 |
| ego | allocentric walls | 0.05 / 0.45 | 0.08 / 0.42 | 0.09 / 0.37 | 0.07 / 0.29 | 0.06 / 0.13 |
| ego | egocentric walls (±2) | 0.95 / 0.97 | 0.88 / 0.92 | 0.78 / 0.85 | 0.67 / 0.75 | 0.61 / 0.65 |

### Probes at the block-0 input, per grid (linear / MLP AUROC)

| grid | edge: allo | edge: ego | ego: allo | ego: ego | base: allo | base: ego |
|---|---|---|---|---|---|---|
| 5×5 | 1.00 / 1.00 | 0.74 / 1.00 | 0.75 / 1.00 | 1.00 / 1.00 | 0.52 / 0.52 | 0.65 / 0.65 |
| 6×6 | 1.00 / 1.00 | 0.69 / 0.96 | 0.69 / 1.00 | 1.00 / 1.00 | 0.51 / 0.51 | 0.62 / 0.62 |
| 7×7 | 1.00 / 1.00 | 0.66 / 0.75 | 0.63 / 0.94 | 1.00 / 1.00 | 0.51 / 0.51 | 0.60 / 0.60 |
| 8×8 | 1.00 / 1.00 | 0.63 / 0.58 | 0.59 / 0.82 | 1.00 / 1.00 | 0.51 / 0.50 | 0.59 / 0.59 |
| 9×9 | 1.00 / 1.00 | 0.62 / 0.56 | 0.57 / 0.72 | 1.00 / 1.00 | 0.50 / 0.50 | 0.58 / 0.58 |
| 10×10 | 1.00 / 1.00 | 0.60 / 0.54 | 0.55 / 0.64 | 1.00 / 1.00 | 0.50 / 0.50 | 0.57 / 0.57 |

### Inference-time ablations, true world, pooled over all 4-epoch grafted models: solved / p(invalid move)

| code | full | graft removed | graft → noise | off during generation | off during prompt | ×0.5 | ×2 | graft only (prompt blank) | other maze's graft |
|---|---|---|---|---|---|---|---|---|---|
| edge (n=30) | 0.80 / 0.00 | 0.00 / 0.92 | 0.00 / 0.96 | 0.01 / 0.80 | 0.00 / 0.73 | 0.46 / 0.05 | 0.00 / 0.85 | 0.12 / 0.29 | 0.07 / 0.32 |
| ego (n=30) | 0.70 / 0.00 | 0.00 / 0.87 | 0.00 / 0.94 | 0.01 / 0.76 | 0.04 / 0.12 | 0.54 / 0.02 | 0.01 / 0.12 | 0.11 / 0.33 | 0.05 / 0.38 |

### Findings
1. **Same use.** From 6x6 up the egocentric graft is followed *better* than the allocentric one on a single edit
   (p(new move) 0.80-0.87 vs 0.67-0.75; it crosses the edited-away wall 3-8% of the time vs 13-23%), and it solves the
   grafted maze against a disagreeing prompt at least as often at every size (route-changed 0.62 vs 0.41 at 8x8; equal
   at 5x5). Whatever frame the graft is written in, the model reads it and prefers it to the tokens.
2. **Same dependence.** Removing the graft, replacing it by same-norm noise, or switching it off during generation gives 0%
   solved and 0.77-0.96 of the next-token mass on invalid moves for both codes at every grid size; halving it halves the
   solve rate; another maze's graft is followed by neither (0.05-0.06). The one difference is scale: doubling the graft
   breaks the edge model (p(invalid) 0.84) but barely disturbs the ego model (0.11; 0.03 at 5x5 rising to 0.36 at 10x10).
3. **The code sets the frame.** Probe metric = AUROC (wall open vs closed is a binary outcome; AUROC is threshold-free,
   class-balance-invariant, chance 0.5; R² and sign accuracy are kept in `ev_probes.csv` and tell the same story). The
   edge model holds the allocentric walls linearly at AUROC 1.00 through block 6 and 0.87 at block 12, and the egocentric
   walls only through an MLP (linear 0.65-0.72, MLP 0.93 at block 6); the ego model is the mirror image (egocentric 1.00
   -> 0.95 linear; allocentric 0.63-0.66 linear, MLP 0.86 at the input falling to 0.70). The base model has neither
   (<= 0.74 linear, <= 0.79 MLP). The MLP-only readability of the *other* frame shrinks with grid size (input-layer MLP
   AUROC for the ego model's allocentric walls: 1.00 at 5x5, 0.82 at 8x8, lower at 10x10) -- the position-dependent
   re-indexing between frames outgrows a 512-unit probe. Nothing in the network re-encodes the graft into the other
   frame; the frame it is given is the frame it keeps.
4. **The cost is learning speed, not capacity.** True-world solve rate: edge = base at every size; ego = base through
   8x8 except a 1-7 point lag at 6x6-7x7. At 9x9 and 10x10 ego is bimodal at 4 epochs (five seeds: 0.66 / 0.64 / 0.48 / 0.27 / 0.00
   and 0.59 / 0.54 / 0.29 / 0.00 / 0.00; edge seeds all within 0.60-0.63 and 0.51-0.57). `diag_collapse.py` (fig_ev_collapse): the failed seeds are perfect on corridors (0.997), as good as
   the healthy ones at junctions far from the target (0.83 vs 0.83), and fail only within 3 steps of the target (0.55 vs
   0.96) -- and they do not stop on it (p(PATH_END) at the target 0.19-0.22 vs 1.00). The target token is an absolute
   coordinate; with an egocentric graft, "am I there" needs a coordinate subtraction the allocentric graft never needs,
   and that is the piece learned last (the staircase in fig_ev_curves: switch-on at ~0.3 epoch for 6x6, ~0.7 for 7x7,
   ~1.3 for 8x8, 1.3-4+ for 9x9, 2-4+ for 10x10, while the edge graft is usable within the first quarter epoch at every
   size). Loss curves do not show it: the adjacency-list tokens dominate the loss and the healthy and failed seeds differ
   by 0.01 nats.
5. **Continuation (fig_ev_cont).** Seeds 0-2 of 9x9 and 10x10 trained 4 more epochs. The two seeds that were *partly* switched on at
   epoch 4 (9x9 s0 0.28, 10x10 s0 0.30) reach edge level (0.68 and 0.57; edge 0.69-0.71 and 0.58-0.61) and the two that were
   already on improve further (0.73 and 0.64, both above every edge seed); the two seeds that were at 0.00 (9x9 s2, 10x10 s1)
   stay at 0.00 with p(SP) still on the 0.82-0.84 plateau. Edge gains 0.04-0.09 from the extra epochs. So the deficit is a
   delay for seeds that have started the transition and, within 8 epochs, a stuck solution for those that have not; 3 of the
   10 large-grid ego seeds are stuck at 4 epochs and neither of the two stuck seeds we continued recovered.
6. **Once switched on, ego is the better graft at large grids too**: 9x9 seeds at 8 epochs solve 0.68-0.73 (edge 0.69-0.71)
   and follow the graft far better (route-changed 0.61-0.62 vs 0.10-0.34; p(new) 0.88-0.89 vs 0.43-0.75).

Caveats: one base seed per grid; the MLP probe has fixed capacity (512 GELU, 4,000 mazes), so "not MLP-readable" at 10x10
is a bound on that probe, not on the network; the 10x10 seed-3 ego checkpoint sweep is truncated at step 14,400 (its
intermediate checkpoints were deleted by a concurrent worker before scoring; final checkpoint scored separately, see the
note in its metrics.json); several trainers hung in wandb shutdown after saving their final checkpoints and were killed by a
watchdog (`worker.sh`, `worker_cont.sh`) or by hand -- no run was lost.
