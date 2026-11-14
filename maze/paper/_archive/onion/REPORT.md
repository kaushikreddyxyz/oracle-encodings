# Onion (magnitude-layered) tile graft: 6x6 pilot, 2026-09-22

Question: can a maze transformer read a NONLINEAR graft, where a tile's four connections are stored as four orders of
magnitude along one direction (after arXiv 2408.10920), instead of one direction per connection?

## Code
`maze/core/onion.py`: one direction per tile, m_t = sum_k s_k 2^-k (s = +-1, k = 0..3 the tile's four sides, rotated per tile so
depth is not confounded with orientation); 16 evenly spaced levels per tile; layer k changes sign 2^(k+1)-1 times along the
direction, so only layer 0 is a threshold. Injected norm matched to the edge graft. `checker` = tiles with r+c even own
connections (every free connection stored once, at one depth); `full` = every tile (each connection stored twice, at two
depths). The feature -> residual map stays linear but many-to-one (18 / 36 directions for 60 free connections); checkpoints
store the effective 72x256 `W_inj`, so every evaluator runs unchanged. Trainer: `--inject_code onion-checker|onion-full`,
`--epochs`, `--init_from`, `--tag_suffix`. Tests: `maze/core/tests/test_onion.py`.

## Runs (6x6, seed 0, curves-campaign recipe, A100)
Round 1 = 1 epoch (2M mazes). Continuation (`-cont1`) = warm start, +3 epochs on a fresh 2M-maze set (optimizer moments
restart). Six pods, one run each; ~$16 total; all pods deleted.

| run | prompt | code | W |
|---|---|---|---|
| onion-checker fixed / trainable | intact | checker | fixed / trainable |
| onion-full fixed / trainable | intact | full | fixed / trainable |
| onion-checker-dummy fixed | blanked (`;`) | checker | fixed |
| edge-dummy fixed | blanked | edge | fixed |

Reference: the curves campaign's 6x6 base / edge-graft models (5 seeds, 1 epoch).

## Results (`tables.md`, `onion_*.csv`, `fig_onion_*.{pdf,svg}`)

Blanked prompt (graft is the only source of the maze), true-world solve rate / phantom-world solve rate:

| | 1 epoch | 4 epochs |
|---|---|---|
| edge graft | 0.956 / 0.718 | 0.990 / 0.853 |
| onion checker | 0.731 / 0.305 | **0.971** / 0.542 |
| base model with the full prompt (5 seeds, 1 epoch) | 0.795 | - |

Intact prompt, standard set (route-changed items): solves the GRAFT maze / solves the PROMPT maze:

| | 1 epoch | 4 epochs |
|---|---|---|
| edge graft (5 seeds) | 0.72 / 0.01 | - |
| onion checker fixed | 0.08 / 0.24 | 0.04 / 0.52 |
| onion checker trainable | 0.07 / 0.27 | 0.03 / 0.66 |
| onion full fixed | 0.20 / 0.06 | 0.15 / 0.19 |
| onion full trainable | 0.18 / 0.07 | 0.09 / 0.28 |

One-swap set, p(new correct move) by the onion layer of the edited connection (edge graft on the same items: 0.78, flat):

| | layer 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| onion checker fixed, 1 ep | 0.41 | 0.30 | 0.23 | 0.19 |
| onion checker fixed, 4 ep | 0.29 | 0.10 | 0.08 | 0.07 |
| onion full fixed, 1 ep | 0.61 | 0.47 | 0.33 | 0.38 |
| onion full fixed, 4 ep | 0.50 | 0.34 | 0.24 | 0.24 |

## Findings
1. The onion is learnable. With the prompt blanked, 4 epochs take the onion model to 0.97 solved (p(SP) 0.996), on par
   with the 1-epoch edge graft (0.96) and above the base model with the full prompt (0.80). All four magnitude layers are
   read; 1 epoch was not enough (0.73).
2. With the prompt intact the onion loses to the prompt, and more training makes it worse: solve rate on the graft maze
   falls (0.08 -> 0.04), solve rate on the prompt maze rises (0.24 -> 0.52), and the one-swap effect shrinks at every depth.
   The edge graft is adopted early and kept; the onion is the harder channel and is pruned as prompt reading improves.
3. Depth matters: a monotone dose-response by layer (log-prob gain 4.5 -> 1.1 nats at 1 epoch), absent for the edge graft
   on the same items. Storing each connection twice (`full`) helps at every depth. Trainable tile directions do not help.
4. Side result: the 1-epoch models of the curves campaign are far from converged (6x6 true-world solved 0.80-0.84 -> 0.98-0.99
   at 4 epochs, for every graft type).

## Rule for the next representation (user, 2026-09-22)
Test a new graft code on the BLANKED prompt first. If it fails there (loss / solve rate far below the edge graft), do not
run the intact-prompt version. Only codes that pass blanked go on to the intact-prompt comparison.

## Files
`worker.sh`, `worker_cont.sh` (pod), `onion_pod.sh` (Mac: setup/run/cont/status/pull), `pods.json`, `analyze.py` -> `tables.md`
+ csvs, `plot_onion.py` -> figures; raw results under `results/{runs,dist_out,local_out,logs}`; wandb project `maze-onion`.
