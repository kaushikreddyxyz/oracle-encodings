# Next-token probability mass: base vs grafted models, true vs phantom world

Generated 2026-09-16. Figure: `figures/fig_next_token.{pdf,svg}` (linear y; per category: base · true, grafted · true, grafted · phantom; no seed dots; error bars = SD over seeds; the base-model phantom-world control is NOT plotted, since the base model has no channel and the setting is ill-defined, but its numbers remain in results.json), `figures/fig_next_token_log.{pdf,svg}` (log-y variant), plus the `_s0` and `_smoke` variants. PDF + SVG only, no PNGs.
Regenerate (from this folder, no GPU): `python plot_next_token.py --results results/results.json --out fig_next_token [--log True]` → `figures/`.
The committed pairs were rendered with matplotlib 3.11.1; the repo's current venv (3.10.9) reproduces the same content but not byte-identical files.
Raw: `results/results.json` (per-model summaries incl. first-move / interior / at-target splits),
`results/recs_<model>_<world>.npy` (one row per scored position: maze, step, path_len, p_correct, p_valid),
`results/phantom_edits.json` (the edits, identical for every model).

## Protocol
- 2,000 fresh held-out 6x6 random-DFS mazes (maze_dataset seed 4321, serial generation), tokenized with the
  models' tokenizer. Same mazes and same edits for all 9 models.
- Teacher-forced along the shortest path of the world being scored; at every path position the softmax
  over the vocabulary is split into: **correct** = next cell on a shortest path to the target (at the target:
  `<PATH_END>`); **other valid** = other lattice neighbours with no wall between; **invalid** = everything else
  (non-adjacent cells, walled-off neighbours, special tokens, premature `<PATH_END>`). Mean over all positions
  (22,616 positions in the true world, 28,036 in the phantom world).
- **true world**: the maze as given; grafted models receive the true 72-feature connectivity through the channel.
- **phantom world**: 2 phantom walls placed on the true shortest path + 2 phantom doors opened where true walls
  were (target still reachable; both doors on the new shortest path where possible; all 2,000 mazes admitted such
  an edit). Grafted models receive the phantom connectivity through the channel and are teacher-forced on the
  phantom shortest path; scored against the phantom world. The base model sees only the (true) prompt, so its
  phantom-world row is the control for "cannot see the edit".
- Models: jirpyish d256/L12 trained on 2M mazes with the legacy `train_fast.py`; seed 0 = the original runs
  (baseline, complete-fixed, complete-trainable-retrain); seeds 1, 2 trained 2026-09-16 (dataset + init seed),
  wandb project maze-injection (dataset_cfg.seed = 1/2), local copies in `maze_legacy/final_models/*-s{1,2}/`.

## Results (mean ± SD over 3 seeds)

| model · world | shortest-path | other valid | invalid |
|---|---|---|---|
| base · true | 0.974 ± 0.003 | 0.025 ± 0.003 | 0.0008 ± 0.0000 |
| fixed graft · true | 0.977 ± 0.002 | 0.021 ± 0.002 | 0.0017 ± 0.0002 |
| trainable graft · true | 0.978 ± 0.001 | 0.020 ± 0.002 | 0.0014 ± 0.0001 |
| fixed graft · phantom | 0.917 ± 0.006 | 0.076 ± 0.006 | 0.0077 ± 0.0009 |
| trainable graft · phantom | 0.925 ± 0.003 | 0.067 ± 0.002 | 0.0079 ± 0.0003 |
| base · phantom (not plotted; ill-defined, kept for reference) | 0.781 ± 0.004 | 0.166 ± 0.002 | 0.0531 ± 0.0012 |

Reading: on the true world the grafted models match the base model (the channel costs nothing). On the
phantom world the grafted models follow the grafted world (0.92 on the phantom shortest path, invalid mass
under 1%), whereas the base model, which can only see the true prompt, puts 0.78 on the phantom path and
5% on tokens that are invalid in the phantom world (it walks into phantom walls / ignores phantom doors).
The remaining gap between 0.92 and 0.98 is where the grafted models still partly trust the prompt.

Cost/infra: six A100 pods for ~1 h each (one run per pod; two runs per GPU halve each other), ≈ $13.
