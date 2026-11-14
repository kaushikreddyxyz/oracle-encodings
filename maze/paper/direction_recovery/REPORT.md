# Recovering known graft directions in maze transformers

The main result is that the tested SAEs reconstruct useful activation structure but do not reliably isolate the individual graft directions across maze sizes. Supervised readout directions align much better, and generative linear regression recovers the identifiable graft directions almost exactly. These are results for a fast, explicitly specified training budget, not a fully tuned ranking of SAE architectures.

**Coverage.** Six grid sizes (5–10), two model seeds, fixed and trainable grafts: 24 grafted checkpoints. Each is analyzed immediately after the layer-0 graft at PATH_START and after block 0. Additional experiments use two grid-6 baseline checkpoints, the archived counterfactual corpus, and a grid-10 SAE sensitivity study. All final outputs are under this directory; input checkpoint hashes are in `input_manifest.json`.

**The quantity being recovered.** The graft writes signed edge features through rows of `W_inj`. At PATH_START immediately after injection, the token/position embedding is constant within a grid, making the activations an exact linear mixture of edge features. Every DFS tree has exactly n²−1 open edges, so adding a common vector to all free-edge directions changes only the intercept. The primary target therefore subtracts the mean free-edge direction. Forced boundary edges are excluded. Raw-direction metrics are also saved. Post-block-0 comparisons measure alignment with the original graft directions in residual coordinates; they are not comparisons against a separately identified nonlinear transport of those directions.

**Primary direction recovery.** Each edge is associated with the SAE atom having the largest absolute validation correlation, without consulting the graft directions; multiple edges may select the same atom. Inactive atoms are excluded from best-atom geometric scores. The table gives mean absolute cosine for these label-associated directions, averaged equally across the 12 model/grid combinations per graft condition. Standard readouts retain their edge labels. Generative OLS predicts activations from edge labels, whereas the ridge/logistic probes predict edges from activations; generative OLS is a supervised writing-direction control, not a conventional readout probe.

| Method | Fixed graft | Trainable graft |
| --- | --- | --- |
| ReLU SAE | 0.425 | 0.384 |
| BatchTopK SAE | 0.369 | 0.317 |
| Matryoshka BatchTopK | 0.409 | 0.361 |
| Matching Pursuit SAE | 0.355 | 0.302 |
| Difference of means | 0.818 | 0.759 |
| Ridge readout | 0.784 | 0.893 |
| Logistic readout | 0.859 | 0.860 |
| Generative OLS (positive control) | 1.000 | 1.000 |


Unlabelled best-atom cosine, one-to-one assignment, the fraction of graft features with an atom above cosine 0.8, raw-column alignment, and per-edge scores are retained in the method JSON files and `direction_metrics.csv`. Best-of-dictionary comparisons include equally sized random dictionaries in the label-predictable activation subspace, plus ambient random dictionaries; labelled scores include shuffled-edge controls. A best atom need not be the atom whose activations identify the right edge. A discriminative readout is also a dual direction rather than a writing direction: accurate edge prediction does not by itself imply equality with a graft column.


**Reconstruction and sparsity.** A 1,024-atom dictionary is trained for 2,000 Adam steps of 512 examples, on 16,000 fitting mazes with 4,000 separate validation mazes. BatchTopK and Matryoshka use k=16, with calibrated threshold inference; Matching Pursuit unrolls 16 residual-greedy selections. ReLU uses L1=0.1. The compact implementations omit auxiliary dead-latent losses and extensive tuning. Measured sparsity differs substantially, so reconstruction alone does not support an architecture ranking.

| SAE | Explained variance | Active latents | Best-atom cosine | Label-associated cosine |
| --- | --- | --- | --- | --- |
| ReLU SAE | 0.993 | 371.2 | 0.450 | 0.405 |
| BatchTopK SAE | 0.728 | 17.0 | 0.475 | 0.343 |
| Matryoshka BatchTopK | 0.742 | 16.8 | 0.470 | 0.385 |
| Matching Pursuit SAE | 0.864 | 16.0 | 0.478 | 0.329 |


**Maze-diff interventions.** The prompt and original channel describe maze A. The intervention adds the recovered edge-direction change for B−A from ADJLIST_END onward, including decoding. Per-edge amplitudes are fitted from training activations, and global strength is selected from 0.5, 1, 2 on 20 validation base mazes. Test evaluation uses 60 disjoint base mazes per grid at distance levels 0, 1, 4, 8, 12 and independent-maze level 13, with all variants of a base maze kept together. The original graft replacement uses its trained strength (1). Rows below are paired solve-rate differences against that original graft at the injection site, in percentage points. Intervals bootstrap base mazes within each grid, jointly across the two model seeds and conditional on these checkpoints; they are not population confidence intervals over training seeds.

| Method | Fixed: Δ [95% interval] | Trainable: Δ [95% interval] |
| --- | --- | --- |
| ReLU SAE | -29.4 [-31.0, -27.8] | -31.7 [-33.5, -29.9] |
| BatchTopK SAE | -30.3 [-31.9, -28.7] | -33.4 [-35.2, -31.6] |
| Matryoshka BatchTopK | -29.9 [-31.5, -28.4] | -32.2 [-33.8, -30.3] |
| Matching Pursuit SAE | -31.6 [-33.3, -30.0] | -34.3 [-36.0, -32.5] |
| Difference of means | -13.9 [-15.1, -12.6] | -15.6 [-17.0, -14.2] |
| Ridge readout | -24.0 [-25.3, -22.6] | -10.4 [-11.7, -9.1] |
| Logistic readout | -17.5 [-18.8, -16.1] | -10.8 [-12.1, -9.6] |
| Generative OLS (positive control) | -0.0 [-0.2, +0.1] | -0.0 [-0.1, +0.1] |




Here ε is the biased-DFS generation probability, not the realized fraction of changed edges. The corpus was sampled by target Hamming-distance buckets, so the ε plot is a descriptive view of that corpus, not a controlled sweep sampled uniformly in ε. Per-pair ε and normalized edge difference are saved in `intervention_trials.csv.gz`. The rollout limit is 2n²+4 tokens, consistently applied within this experiment. The fitted injection-site generative coefficients agree with the mean-centered graft to a maximum relative error of 1.12×10⁻⁷ (`generative_control_precision.json`). Differences between generative OLS and the original graft can arise from numerical precision and strength selection; they should not be interpreted as discovery of better ground-truth directions. A win after block 0 would additionally change the intervention site.

Across both tested sites, the largest observed SAE-minus-graft difference was **-11.67 percentage points** (g10-s0-fixed, post0, Matching Pursuit SAE). This is an exploratory maximum over many comparisons, not a corrected significance claim.

The largest observed difference across **all** interpretation methods was +0.56 percentage points (g7-s0-trainable, post0, Generative OLS (positive control)). Small positive values in the generative control should be read in light of the numerical/strength caveat above.

Individual-maze wins do occur despite the aggregate losses. At the injection site, ReLU solves 55 of 8,632 test cases that the original graft misses, while missing 2,694 cases that the graft solves. The four SAEs each have 50–56 such individual wins at this site. `sae_paired_wins.csv` records wins and losses for both sites; these cases do not imply improved aggregate accuracy.

**Archived probe-steering replication.** The archived asset contains 4,522 edits over 600 tokenized mazes. We use current grid-6 baseline checkpoints, 8,000 fitting and 2,000 validation mazes, sweep probe accuracy across all 12 layers, and test layer 0 plus each model’s best logistic-probe layer. The original class-coordinate projection protocol is applied at every active position, with strength selected from 1, 4, 8 on 30 validation base mazes and evaluated on 100 held-out base mazes (750 edits). This reproduces the protocol and qualitative finding, rather than claiming numerically identical results from the historical checkpoints.


| Method at best probe layer | Still crosses blocked edge ↓ | Uses newly opened edge ↑ |
| --- | --- | --- |
| Unedited | 88.2% | 1.1% |
| ReLU SAE | 89.1% | 1.6% |
| BatchTopK SAE | 86.7% | 1.6% |
| Matryoshka BatchTopK | 88.5% | 1.4% |
| Matching Pursuit SAE | 88.7% | 1.1% |
| Difference of means | 88.7% | 1.6% |
| Ridge readout | 88.5% | 1.4% |
| Logistic readout | 88.7% | 1.4% |
| Generative OLS (positive control) | 87.6% | 1.6% |

**Sensitivity and limits.** At grid 10, a second SAE seed and increased sparsity budget (k=64, or ReLU L1=1) test whether the result is an obvious artifact of one initialization or the primary sparsity setting. Reconstruction can improve substantially without comparable improvement in directional alignment. Results are saved separately in `robustness_summary.csv`; these settings were not selected by optimizing the reported primary ground-truth cosine. Two model seeds, one dictionary width, two intervention sites and a limited training budget restrict generalization. The experiments do not establish that a fully optimized SAE, a multi-atom intervention, or another feature-distribution design cannot recover the graft.


**Verification and data separation.** The small numerical pilot checked the fast decoder against the model’s own forward pass. A separate 12-token comparison on four prompts for each of a fixed-graft, trainable-graft and baseline model exactly matched the existing reference KV decoder. Zero-strength archived interventions also matched across the two tested layers. Unit tests check the batch sparsity budget, Matching Pursuit’s monotonic reconstruction, finite gradients for all four SAEs, and the constant-sum identifiability correction. Exact connectivity auditing found overlap in the initial small-grid datasets; grids 5 and 6 were rerun with all fitting graphs deduplicated and excluded from every frozen A/B evaluation graph. Discarded outputs are isolated under `initial_overlap_runs/`. Final split and output-completeness checks are saved in `validation.json`. Each completed pod’s downloaded files are SHA-256 verified before termination.

**Artifacts and reproduction.** `experiment.py` runs the graft study; `methods.py` implements the SAEs/probes; `legacy.py` runs archived steering; `robustness.py` runs sensitivity checks; `figures.py` regenerates PNG/SVG figures; `subspace_nulls.py` computes the stronger dictionary null. `README.md` records the detailed protocol and sources. The implementation references are [dictionary_learning](https://github.com/saprmarks/dictionary_learning) for ReLU/BatchTopK/Matryoshka and [overcomplete MpSAE](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/sae/mp_sae.py) for residual-greedy Matching Pursuit.

**Compute.** 11 campaign pods have verified downloads and recorded termination. Estimated GPU runtime cost: $6.18, using the quoted hourly rates and observed rental durations; this is not a billing statement. Other concurrently running RunPod projects are outside this campaign.

## Steelman rerun (2026-09-21)

The campaign above trained one small SAE per kind on 16,000 mazes and probed 24 checkpoints; a sceptic could attribute its negative SAE result to that budget. This rerun gives every method the advantages such a reader would ask for and repeats the two questions (alignment with the graft rows; A→B edits) at the same two sites. All code, raw outputs and tables are under `steelman/` (its `README.md` lists the protocol table and every file); figures are written next to the originals with a `_steelman` suffix and the originals are unchanged.

**Protocol.** 60 grafted finals (grids 5–10 × 5 seeds × fixed/trainable; `steelman/manifest.json` with checkpoint hashes). Per grid, 512,000 i.i.d. fresh-DFS fitting mazes and 32,000 validation mazes, with every A/B graph of the frozen distance eval set and every validation graph excluded from the fitting pool by exact connectivity (5×5 has only ~2.7×10⁵ distinct trees among the 512k draws; duplicates are genuine i.i.d. draws and are kept). At PATH_START the injection-site activation is an exact linear function of the ±1 connection features, `x = const + s·W_inj`; this was checked against the model's own hooked forward pass on 512 real prompts for every model (max |error| 0, `steelman/verify_summary.csv`; the fast one-row block-0 evaluation agrees with the full forward to 1.6×10⁻⁵), so the injection-site fitting set costs no forwards and the post-block-0 set is one block pass. SAEs use `maze/paper/curves/sae_lib.py` (10,000 steps × 16,384 = 164M samples per SAE, warm-up/decay, aux-k or resampling for dead latents): 5 kinds (ReLU+L1, TopK, BatchTopK, Matryoshka, matching pursuit), dictionaries of 1×, 1.5×, 3×, 8× the free-connection count and 1,024 atoms, k ∈ {4, 8, 16, 32, (n−1)², n²−1}, L1 ∈ {.01, .03, .1, .3}, centred and uncentred inputs, plus coding-aware `open`/`closed` reference points (input shifted so the true code is the non-negative wall or door indicator; these use knowledge of the coding and are reported separately). The full grid ran on the 12 seed-0 models (117–133 SAEs each; 8× and 1,024-atom sizes trimmed to k ∈ {16, dense} on grids 5–7 after the 8–10 sweeps showed them losing at every sparsity); a per-grid shortlist (`steelman/jobs/shortlist.json`) then ran on all 60 models with two SAE seeds, and its centred members after block 0 on the seed-0/1 models. Total 3,096 SAEs (2,860 injection-site, 236 post-block-0). Probes: difference of means, ridge (9 λ; the LDA direction equals the ridge direction, verified to cosine 1 − 10⁻¹⁶), logistic (6 λ), generative OLS, and the joint pseudo-inverse of the ridge readout matrix, at fitting sizes 1k, 4k, 16k, 64k, 256k, 512k (14,400 fits). Interventions: as above (graft of A plus the recovered B−A edit from ADJLIST_END onward, decoding included) but on the FULL standard set (3,000 base mazes × 13 levels per grid, 41,937–42,000 items), all 60 models, per-connection amplitude from the generative fit, global strength ∈ {.5, .75, 1, 1.5, 2} chosen on 100 disjoint base mazes, rollout limit n²+2 (the standard-set convention; the graft control reproduces the distance campaign's per-item outcomes to 98.6–99.0% and its solve rates to 0.15 points, `steelman/graft_control_vs_distance_campaign.csv`). Metric: route-changed items solved with no rule break.

**Selection functions** (`steelman/selection.py`, choices per model in `steelman/selection.json`). *fair*: may use maze labels on validation mazes, never `W_inj` — SAE configuration with the largest mean |label correlation| of its best latents, one latent per connection by |label correlation| (the original protocol); probe by held-out readout accuracy. *steelman*: may use `W_inj` — best configuration and direction family (label atom, oracle atom, or the decoder image of the mean code on the 2/4/16/all most label-correlated latents) by true |cosine| in the identifiable gauge; probe method and λ by true alignment. The steelman numbers are **upper bounds** and are what the two minimal figures show, labelled as such; fair vs steelman and per-kind detail are in the supplementary figures and `steelman/alignment_summary.csv`.

**Alignment at the injection site** (mean |cosine| to the graft rows, identifiable gauge; equal weight per model and per connection; 30 models per graft; SD over the 5 seeds of the grid-averaged value in brackets; `frac ≥ 0.8` = share of connections above 0.8).

| Entry | Fixed | Trainable | frac ≥ 0.8 (fixed / trainable) |
| --- | --- | --- | --- |
| SAE, steelman | 0.980 [0.007] | 0.970 [0.014] | 1.00 / 0.99 |
| SAE, steelman without coding-aware variants | 0.979 [0.006] | 0.967 [0.012] | 1.00 / 0.98 |
| SAE, fair | 0.961 [0.017] | 0.941 [0.011] | 0.98 / 0.97 |
| Probe, steelman (best of ridge/logistic/DoM, best λ) | 0.867 [0.001] | 0.900 [0.003] | 0.96 / 0.97 |
| Probe, fair (best held-out accuracy) | 0.832 [0.001] | 0.843 [0.007] | 0.62 / 0.72 |
| Difference of means | 0.820 [0.000] | 0.759 [0.005] | 0.79 / 0.18 |
| Pseudo-inverse of all ridge readouts (control) | 0.999 | 1.000 | 1.00 / 1.00 |
| Generative OLS (control) | 1.000 | 1.000 | 1.00 / 1.00 |

The steelman SAE is a ReLU+L1 dictionary of 1× the free-connection count with L1 = 0.1 or 0.03 on centred activations in 48 of 60 models, and its label-selected atom in 47 of 60. Per kind (steelman / fair, both grafts pooled): ReLU 0.974 / 0.952, TopK 0.860 / 0.649, BatchTopK 0.856 / 0.649, Matryoshka 0.816 / 0.675, matching pursuit 0.773 / 0.568. Across the sweep (`steelman/sae_metrics.csv`, `figures/supplementary/sae_sweep_steelman`), alignment tracks the code density, not the dictionary size: TopK/BatchTopK with k = n²−1 (the number of open connections) reach 0.90–0.91 best-latent |cosine| and k = (n−1)² 0.79–0.83, while k = 16 gives 0.63–0.64 and k = 4 0.36–0.44 at every dictionary size; ReLU at L1 = 0.01 (L0 ≈ 160) 0.91. After block 0 the picture reverts to the original one: SAE steelman 0.743 / 0.767, probe steelman 0.728 / 0.773, generative OLS itself only 0.719 / 0.776 (the graft rows are not the right target there).

**Interventions on the full standard set** (route-changed items, injection site; mean over grids then seeds, SD over seeds in brackets; Δ = paired difference to the actual graft with a 95% bootstrap over base mazes within grid, pooled over seeds, conditional on these 60 checkpoints; 975,170 rollouts per row).

| Method | Fixed: solved | Δ vs graft | Trainable: solved | Δ vs graft |
| --- | --- | --- | --- | --- |
| Actual graft of B | 0.524 [0.017] | — | 0.537 [0.024] | — |
| Generative OLS (control) | 0.524 | +0.0 [−0.0, +0.0] | 0.537 | −0.0 [−0.0, +0.0] |
| SAE, steelman | 0.507 [0.017] | −1.6 pt [−1.7, −1.6] | 0.513 [0.032] | −2.4 pt [−2.5, −2.4] |
| SAE, fair | 0.479 [0.047] | −4.5 pt [−4.5, −4.4] | 0.483 [0.020] | −5.4 pt [−5.5, −5.3] |
| Probe, steelman | 0.325 [0.011] | −19.9 pt [−20.0, −19.8] | 0.410 [0.009] | −12.7 pt [−12.8, −12.6] |
| Probe, fair | 0.286 [0.009] | −23.7 pt [−23.9, −23.6] | 0.377 [0.009] | −16.1 pt [−16.2, −15.9] |
| Difference of means | 0.335 [0.009] | −18.9 pt | 0.331 [0.011] | −20.6 pt |
| No edit | 0.002 | −52.2 pt | 0.002 | −53.5 pt |

Per grid the SAE steelman edit is within 0–4 points of the graft at every size for both grafts (e.g. 10×10 fixed 0.234 vs 0.242; 5×5 fixed 0.879 vs 0.921), the fair SAE within 1–13 points (worst at 5×5 trainable), and the best probe 8–27 points below (`steelman/intervention_models.csv`, `figures/maze_diff_accuracy_steelman`). After block 0 every method loses: SAE steelman −11.2 / −14.7 pt, probe steelman −23.2 / −23.7 pt, and the generative direction itself −5.8 / −7.5 pt (`figures/supplementary/maze_diff_accuracy_all_post0_steelman`). The edit strength selected on the tuning mazes was 1.0 for 118 of the 120 injection-site SAE entries.

**Findings.**
1. The original SAE conclusion does not survive the steelman at the injection site. With 164M training samples, dead-latent handling and a dictionary of about the true feature count, a plain ReLU+L1 SAE recovers the graft rows to |cosine| 0.96–0.98 (≥ 0.8 for 98–100% of connections), and editing along its label-selected latents solves the edited maze within 1.6–2.4 points of the actual graft (upper bound) and 4.5–5.4 points under the fair selection. The original numbers (0.30–0.43 alignment, −29 to −34 points) were a property of the training budget and of k = 16, not of SAEs.
2. The failure mode is sparsity mismatch, as anticipated: every maze has n²−1 open connections, i.e. about half the free connections are "on" in every sample. k-sparse SAEs with k ≤ 32 cannot represent that code and learn mixtures (best-latent |cosine| 0.36–0.68 at every dictionary size); the same architectures with k = n²−1 reach 0.90–0.91. Reconstruction is not the issue: 1,024-atom ReLU dictionaries reach FVU 4×10⁻⁵ with alignment 0.87 because they split each connection across several latents.
3. The probe conclusion survives, and more data does not change it. The best per-connection readout direction (ridge/LDA or logistic at the oracle λ) plateaus at |cosine| 0.87–0.90 from 4,000 fitting mazes onward (0.878 at 1k, 0.883 at 512k) and its edits lose 13–20 points; the fair choice by held-out accuracy loses 16–24. A readout is a dual direction: the joint pseudo-inverse of all readouts, which needs every label at once, is the generative direction (|cosine| 0.999–1.000, Δ ≈ 0), so the shortfall is not estimation noise but the object being estimated.
4. Fixed and trainable grafts differ in the expected direction: trainable rows are more correlated (max |cos| between rows up to 0.56 vs 0.25), which helps the discriminative probes (0.90 vs 0.87) and hurts the fair SAE choice slightly (0.941 vs 0.961); the steelman SAE is unaffected.
5. Nothing recovers the graft after block 0. The block-0 output is no longer linear in the features, the generative direction fitted there aligns only 0.72–0.78 with the original rows and loses 6–7 points when used to edit, and SAEs/probes sit at 0.73–0.77. The block-0 numbers are comparisons against the original rows in residual coordinates, as in the original report, not against a separately identified transport of those rows.
6. The coding-aware `open`/`closed` variants (true code non-negative and exactly k-sparse) do not beat the unsupervised centred ReLU SAE (steelman without them 0.979/0.967 vs 0.980/0.970), so the result does not depend on knowing the coding.

**Caveats.** The steelman selections use `W_inj` and are upper bounds; the fair selection is what a user could do, and it favours large dictionaries (max mean label correlation) which cost 2–5 alignment points and 3–9 intervention points. SAE-seed agreement on the shortlist is moderate (mean |Δ oracle cosine| 0.10 between the two seeds over 780 pairs), so a single SAE run is not a reliable estimate of a configuration; the shortlist was chosen from seed-0 models with the steelman criterion, which is optimistic for those 12 models and neutral for the other 48. Bootstrap intervals resample base mazes conditional on the 60 checkpoints; the SD over seeds (brackets) is the relevant spread for generalising over training runs and is much wider (e.g. ±1.7 points on the graft). Trimming: 8×/1,024-atom dictionaries on grids 5–7 were run only at k ∈ {16, dense}, L1 ∈ {.03, .1}, and matching pursuit only at 1.5× with k ≤ 16 (its unrolled encoder costs k passes per step), so the MP entry is not a tuned ranking. The dense-k finding relies on knowing the code density; a user without it would find it by sweeping k against label correlation, which is what the fair criterion does (it picks the dense-k or ReLU configurations in 57 of 60 models).

**Cost and provenance.** 21 pods (RTX 4090 / 6000 Ada / PRO 4500 / L40S / 2 A100), 2,143 pod-minutes of units, estimated $41.5 at the quoted hourly rates (`steelman/pod_ids.txt`); all pods deleted at the end. One community-cloud 4090 was deleted after 5 minutes (host CUDA broken, no results used). Unit journal: `steelman/state-wave1.jsonl` (144 units, 0 failures); orchestrator logs in `steelman/logs/`.
