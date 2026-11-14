# Expanded alignment resampling

Five new fits (initialization seeds 301–305) for each of eight methods across all 24 original grafted checkpoints: grids 5–10, underlying-model seeds 0/1, fixed/trainable grafts. This adds 960 fits (480 SAE fits and 480 supervised estimators). No intervention evaluation is rerun.

Training uses bootstrap resamples of the original 16,000-maze training split; data seeds are 29301–29305. The 4,000-maze validation split is unchanged. Bootstrap data are shared across methods and checkpoints of the same grid/replicate, intentionally enabling paired comparison. These are repeated fits of existing underlying models, not new underlying-model training runs or 960 independent model samples. Generative OLS should be nearly invariant under bootstrap resampling because the observed injection is exactly linear; duplicates in that distribution are not additional independent directional discoveries.

SAE hyperparameters remain unchanged: 1,024 atoms, 2,000 Adam steps, minibatch 512, learning rate 0.001, k=16 for sparse methods, ReLU L1=0.1. The original compact implementations are reused. `expanded_methods.py` copies the original methods with only an optional logistic seed argument added (the original file hardcoded seed 0). Logistic regularization, 600 optimization steps, and ridge regularization are unchanged.

Immediate-injection activations are reconstructed as signed connectivity times the archived injector. Token and position terms at this fixed site are constant and disappear under each estimator's existing centering operation. There is no centering of the ground-truth directions used for cosine evaluation. All results compare against the original free-edge rows of the actual saved injector, including its scale.

Each fit exports two distributions: `matched` maximizes total absolute cosine using Hungarian one-to-one matching, one active direction per graft feature; `all_directions_nearest` includes every validation-active SAE atom / supervised coefficient, each associated with its nearest graft direction. Matches may repeat only for the second metric. Activity means nonzero code on at least one heldout activation, following the original analysis. Sign is ignored via absolute cosine.

`g5.npz` and `g6.npz` come from archived deduplicated training corpora; `g7.npz` through `g10.npz` use the archived original corpus. The g10 training NPZ is locally available even though not all original remote archive materials are retained. Every corpus contains 20,000 connectivity rows; all have zero exact graph overlap between their training and validation partitions. `provenance.json` records input, split, checkpoint, and code hashes. `manifest.json` records original checkpoint locations and hashes.

The validator checks downloaded archive SHA-256 against the remote checksum, checkpoint hashes, all 960 expected method/model/replicate combinations, bootstrap indices, finite cosine bounds, full one-to-one edge coverage, active IDs reconstructed on validation data, per-direction cosine recomputation, and agreement with independently recomputed optimal Hungarian scores. Final merged values are written only after validation passes.

Run orchestration owns only pods named `mzalign-*` recorded here. Completion requires verified local archives and termination of these dedicated GPUs. Other pods in the account are unrelated.

Six dedicated GPU pods were provisioned in total; five performed training. The first g6 pod never became reachable and was terminated without running fits. The original g5 pilot was reused for g6 after g5 finished, in a separate remote working directory. The g7 and g9 jobs were restarted into two workers with disjoint fit seeds to improve GPU utilization; completed fits were retained and any incomplete fit was restarted from its original seed. This changed scheduling only. `parallel_restarts.json` and the orchestration entry in `provenance.json` record the changes; termination receipts record GPU cleanup.

Completed: `validation.json` passes all 960 new fits, with maximum independent
CPU/GPU cosine discrepancy 1.25e-6. The exported CSV contains 99,200 matched
and 508,956 all-active-nearest observations. All campaign GPUs are terminated;
`resource_audit.json` confirms no owned pod remains live and estimates compute
rental at $3.76 (storage/transfer and billing rounding excluded).

Final figures are in `../figures/alignment_violins.*`, with all-active and
individual-size variants alongside them. Each method combines its original
24 fits with 120 new fits, weighting underlying models and fits equally.
