# Steelman rerun of the direction-recovery campaign

Give the SAEs and the supervised readouts every reasonable advantage, then ask again whether they recover the graft
directions (rows of `W_inj`) and whether the recovered directions carry the A→B maze edit. Everything here is new data;
the original campaign's files one level up are untouched. Figures are written next to the originals with a `_steelman`
suffix (`../figures/*_steelman.{pdf,svg}`, `../figures/supplementary/*_steelman.{pdf,svg}`); the write-up is the
"Steelman rerun" section appended to `../REPORT.md`.

## What changed relative to the original protocol

| | original | steelman |
|---|---|---|
| models | 2 seeds × 6 grids × 2 grafts = 24 | all 5 seeds: 60 grafted finals of the curves campaign (`manifest.json`, sha256 of every checkpoint) |
| fitting mazes | 16,000 (one PATH_START token each) | 512,000 i.i.d. fresh-DFS mazes + 32,000 validation mazes per grid, exact-graph exclusion of every A/B eval graph and of the validation graphs (`common.build_pool`) |
| injection-site activations | model forward | analytic `x = const + s @ W_inj` (verified against the hooked forward pass to 0 abs. error on real prompts, `verify_summary.csv`), so the fitting set costs no model forwards |
| SAE training | 2,000 steps × 512 (1M samples), 1,024 atoms, k=16 / L1 0.1, one seed, no dead-latent handling | `maze/paper/curves/sae_lib.py`: 10,000 steps × 16,384 (164M samples), warm-up + decay, aux-k / resampling for dead latents; 5 kinds (ReLU+L1, TopK, BatchTopK, Matryoshka, matching pursuit); dictionary 1×, 1.5×, 3×, 8× the free-connection count and 1,024; k ∈ {4, 8, 16, 32, (n−1)², n²−1}, L1 ∈ {.01, .03, .1, .3}; centred and uncentred inputs; coding-aware `open` / `closed` reference points; 2 SAE seeds on the shortlist |
| probes | DoM, ridge (one λ), logistic (one C), generative OLS | DoM, ridge λ ∈ 9 values (LDA ≡ ridge direction, checked numerically), logistic λ ∈ 6 values, generative OLS, joint pseudo-inverse of the readout matrix; fitting-set sizes 1k … 512k |
| selection | label-correlation atom (fair) | recorded `fair` and `steelman` functions (`selection.py`, `selection.json`) |
| interventions | 60 test base mazes per grid, levels {0,1,4,8,12,13}, strength ∈ {.5,1,2} on 20 mazes | the FULL standard set (3,000 base mazes × 13 levels per grid), all 5 seeds, strength ∈ {.5,.75,1,1.5,2} tuned on 100 disjoint base mazes, rollout limit n²+2 (the standard-set convention) |

## Selection functions (recorded in `selection.json`)

* **fair** – may look at maze labels on validation mazes, never at `W_inj`. SAE: the configuration whose latents have the
  largest mean |validation label correlation| (among centred / uncentred inputs), one latent per connection = the atom
  with the largest |label correlation| (the original protocol). Probe: the readout with the best held-out accuracy at
  the largest fitting size (ties to the stronger regulariser).
* **steelman** – may look at `W_inj`. SAE: the best (configuration, direction family) by mean true |cosine| in the
  identifiable gauge; families are `fair` (label atom), `oracle` (best live atom per connection), `multi{2,4,16,all}`
  (decoder image of the open-minus-wall mean code on the m most label-correlated latents). Probe: the best method and
  regularisation strength by true alignment. This is an **upper bound** for the method family; it is what the minimal
  figures show, labelled as such.
* `sae_steelman_nocoding` – steelman restricted to the unsupervised variants (`ctr`, `raw`), i.e. without the
  coding-aware `open`/`closed` reference points. `probe_pinv` – joint pseudo-inverse of the ridge readout matrix; it
  needs all labels at once and is reported as a control, not as a probe direction.

## Files

Code (pod side): `sae_sweep.py`, `model_unit.py`, `run_sae.sh`, `common.py`, `selection.py`, `sae_lib.py` (copy of
`../../curves/sae_lib.py` + `b_dec_init`, faster MP encoder), `model.py`, `maze_distance.py` (copies). Local:
`prep_local.py` (inputs + manifest), `make_jobs.py` / `shortlist.py` / `make_units.py` (job and unit lists),
`fleet_queue.py` (orchestrator, adapted copy of the curves one), `collect.py` (tables), `figures_steelman.py`.

Data: `results/<tag>/` raw pod outputs (SAE metrics + per-connection cosines per configuration, probe metrics,
intervention rollouts per method), `sae_metrics.csv`, `probe_metrics.csv`, `alignment_values.csv.gz`,
`alignment_models.csv`, `alignment_summary.csv`, `intervention_models.csv`, `intervention_cells.csv`,
`intervention_summary.csv`, `graft_control_vs_distance_campaign.csv`, `verify_summary.csv`, `selection.json`,
`figure_summaries_steelman.csv`, `jobs/shortlist.json`, `pod_ids.txt`, `logs/`.
