# Maze paper: where the evidence is

Start here. Think of the experiment, find its row, open the figure; the write-up column says where the protocol and numbers are.
All figures are PDF + SVG (no PNGs anywhere under `maze/paper/`). Models: jirpyish d256 / 12 layers, random-DFS mazes,
grids 5x5-10x10, 5 seeds, base / fixed graft / trainable graft = 90 final checkpoints under `curves/results/runpod-*/g{n}-s{seed}/<run>/`
(`model.final.pt`, `injector.final.pt`, `config_holder.json`; also on wandb `maze-curves`). The maze source tree (`maze/core`, trainer) was
restored from `git stash@{0}` into the working tree on 2026-09-22 (stash kept; not yet committed).

Layout: every experiment folder keeps its figures in `figures/` (PDF + SVG), its raw per-run outputs in `results/` (never deleted;
`curves/results/distance/` is the standard eval set), and its write-up in `REPORT.md`; `curves/` additionally has `tables/` for the
aggregate CSV/JSON the figures are built from and `infra/` for the pod / orchestration scripts, with the analysis scripts
(`eval_*.py`, `plot_*.py`, `select_*.py`) at its top level.

Colour code everywhere: base model `#2a7a57`, grafted model `#000c7a`, SAE `#e0a100`, edits/phantom orange, references grey.

## Experiment -> evidence

| # | Experiment (what I ran) | Question | Figure(s) | Numbers | Code | Write-up |
|---|---|---|---|---|---|---|
| 1 | Counterfactual-edit schematic | What is a graft / phantom edit? (illustration) | `fig1_counterfactual_edit/figures/fig1.{pdf,svg}` | `fig1_counterfactual_edit/fig1.json` | `fig1_counterfactual_edit/make_fig1.py` | - |
| 2 | Next-token probabilities, 3 seeds (legacy 6x6 models) | p(shortest-path token) / p(other valid) / p(invalid), base vs grafted, true vs phantom world | `fig_next_token/figures/fig_next_token.{pdf,svg}` (`_log`, `_s0` variants) | `fig_next_token/results/` | `fig_next_token/next_token_probs.py`, `plot_next_token.py` | `fig_next_token/REPORT.md` |
| 3 | Training-curve campaign (90 runs, 46 checkpoints each) | How do loss, token masses and solve rate evolve; base vs graft; sizes 5-10 | `curves/figures/fig_curves_{fixed,trainable}_overlay` | `curves/tables/all_metrics.csv`, `train_loss_by_ckpt.csv` | `curves/eval_ckpts.py`, `plot_curves.py`, `plot_curves_overlay.py`, `infra/worker*.sh`, `infra/curves_pod.sh` | `curves/REPORT.md` > Design, Final-checkpoint numbers, What the curves show |
| 4 | Final-checkpoint performance vs maze size | Where do base and grafted models end up | `curves/figures/fig_final_{fixed,trainable}_lines` | `curves/tables/final_results_*_summary.csv` | `curves/plot_final_lines.py` | `curves/REPORT.md` > Final-checkpoint numbers |
| 5 | Phantom edit judged against both mazes | When the graft is edited, does the model solve the OLD maze, or break arbitrary rules? | - | `curves/tables/phantom_vs_true_{summary,breakdown}.csv`, `curves/results/judge/` | `curves/eval_phantom_vs_true.py` | session notes only (rule breaks are mostly illegal in BOTH mazes) |
| 6 | Maze difference (Hamming) sweep = THE STANDARD EVAL SET | Does solve rate fall as the grafted maze differs more from the prompt maze | `curves/figures/fig_distance_heatmap_routechanged` | `curves/tables/distance_cells.csv`, `distance_items.csv.gz`; sets `curves/results/distance/sets/g{n}.npz` (never delete), outcomes `curves/results/distance/out/` | `curves/maze_distance.py`, `eval_distance.py`, `plot_heatmap.py` | `curves/REPORT.md` > Final solve rate, stratified... |
| 7 | Steering the BASE model at the adjacency-list tokens (not token-matched) | Is the base model steerable where it stores the maze | (not drawn; numbers only) | `curves/tables/steering_cells.csv` (methods `dom_L0_a1`, `sae_*`, `patch_L0`, `oracle`, `none`) | `curves/eval_steering.py` | `curves/REPORT.md` > Activation steering of the base models |
| 8 | Token-matched steering (fair comparison with the graft) | Steering fitted on / applied to the graft's tokens only: vectors and SAE features vs graft | MAIN: `curves/figures/fig_steering_lines_{fixed,trainable}_merged` (left: one connection edited; right: many connections edited); per size: `curves/figures/fig_steering_{global,local}_*_by_size`, `fig_steering_lines_fixed_by_size` | `curves/tables/steering_cells.csv`, `steering_items.csv.gz`; raw `curves/results/distance/steer_out/gs_*` | `curves/eval_steering_graftsite.py`, `plot_steering.py`, `plot_steering_merged.py`, `plot_steering_size_comparison.py`, `plot_steering_summary.py` | `curves/REPORT.md` > Token-matched steering |
| 9 | Best-of selection for the illustrative figures | Which vector / SAE configuration feeds the single "Steering vectors" / "SAE features" line | (feeds #8) | `curves/tables/steering_selection.json`, `.csv`; folders `curves/results/distance/steer_out/gs_vec_best`, `gs_sae_best` | `curves/select_best_steering.py` (rule: per model, best candidate on the test metric = optimistic upper bound; fair choice recorded too) | docstring of the script; `curves/REPORT.md` > Token-matched steering |
| 10 | SAE kinds (ReLU, TopK, BatchTopK, Matryoshka, matching pursuit) | Does a better-built SAE change the steering result | `curves/figures/fig_sae_kinds` | `curves/tables/sae_kinds_cells.csv`, `sae_kinds_training.csv`; dictionaries `curves/results/distance/steer_prep`, `steer_gs_prep` (`stats_f*`); pilot `curves/results/distance/sae_logs/sae_selection_used.json` | `curves/sae_lib.py`, `select_sae.py`, `infra/make_fleet_units.py`, `plot_sae_kinds.py` | `curves/REPORT.md` > SAE kinds as steering tools |
| 11 | Local comparison at one decision point (one-connection swap) | At a single decision, how well do graft vs token-matched steering follow an edit | `curves/figures/fig_local_lines_{fixed,trainable}`, `fig_steering_local_*_by_size`, left panel of #8 | `curves/tables/local_cells.csv`, `local_items.csv.gz`, `local_selection.json`; `curves/results/local/` | `curves/build_local.py`, `eval_local.py`, `plot_local.py` | `curves/REPORT.md` > Local comparison |
| 12 | Graft direction recovery | Do SAEs / probes recover the known graft directions, and do recovered directions work as edits | `direction_recovery/figures/` (+ `figures/supplementary/`; steelman rerun in `direction_recovery/steelman/`) | `direction_recovery/direction_metrics.csv`, `intervention_comparison.csv`, `alignment_distribution_values.csv`, `results/` | `direction_recovery/experiment.py`, `methods.py`, `figures.py`, `plot_alignment_*.py` | `direction_recovery/REPORT.md`, `README.md`, `ALIGNMENT_DISTRIBUTIONS.md` |
| 13 | Graft codes: allocentric edge vs egocentric (ego) graft, prompt intact, injected from ORIGIN_END, 4 ep; grids 5-10 x 5 seeds (+ base per grid, + 8-epoch continuations at 9x9/10x10); 6x6 pilot also has axdeg (h/v counts) and axdeg1h (one-hot) | Does the model use a graft written in another frame the way it uses the edge graft (follows it against the prompt, depends on it, keeps its frame internally); linear vs MLP probes of both frames at all layers; why large-grid ego seeds fail | `codes/fig_ev_main` (headline), `fig_ev_ablate`, `fig_ev_probes` (paired-maze linear encoding, `plot_pairs.py`), `fig_ev_curves`, `fig_ev_collapse`, `fig_ev_cont`; 6x6 pilot figures (incl. axdeg) archived in `_archive/codes_pilot/` | `codes/ev_summary.csv`, `ev_probes.csv`, `ev_tables.py`; pilot: `codes/tables.md`, `codes_*.csv`; `codes/results/runs/` (78 final checkpoints, mirrored to HF `kaushikreddyxyz/maze-graft-codes`), `results/diag/` | `maze/core/graft.py`, `codes_ego.py`, `codes_axdeg.py`, `codes/worker.sh`, `worker_cont.sh`, `codes_pod.sh`, `fleet_launch.sh`, `dispatch.sh`, `figures_ev.py`, `probe_nonlinear.py`, `eval_ablate.py`, `probe_walls.py`, `diag_collapse.py`, `push_hf.py` | `codes/REPORT.md` (sec. 5 = scale-up) |

Archived (smoke tests, superseded): `_archive/onion/` (magnitude-layered tile graft pilot), `codes/results/_smoke_blank_prompt/` (blank-prompt runs of every code).

Older maze experiments (pre-paper, 6x6 only; figures restored from the stash): `maze/experiments/02..07/*/figures/`, `maze_legacy/figures/`;
04 = first probe-steering null, 05 = injection variants, 07 = probes vs SAE at the adjacency tokens (their lessons are folded into #7-#8).

## Running things

Every figure regenerates locally from saved outcomes with its `plot_*.py` (no GPU): see "Regenerate" in each REPORT.
Pods and orchestration (only needed to rerun GPU work) live in `curves/infra/`: `fleet_ssh.py` (pure-Python SSH),
`fleet_queue.py` (work queue), `fleet_deploy.py`, `fleet_harvest.py`, `parse_pods.py`; older shell versions `curves_pod.sh`,
`distance_deploy_one.sh`, `steer_deploy_one.sh`, `steer_status.sh`, `steer_reap.sh`, `reap.sh`; pod-side workers `worker.sh`,
`worker1.sh`; the local (one-swap) fleet `local_deploy.py`, `local_fleet.py`, `local_fleet2.py`, `local_queue.py`,
`make_fleet_units.py`; pod manifests `curves/infra/pods.json` and `curves/results/distance/*pods*.txt`.
These scripts resolve `results/` and the `eval_*.py` code relative to `curves/`, so they run unchanged from `infra/`.
