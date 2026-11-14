# 08 Seed and fade

Question: if a graft writes known linear directions into the residual stream during the first part of training and is then
taken away, does the model build its own mechanism that writes the SAME directions (so they stay usable as steer / ablate
handles at inference), or does it fall back to the base model's route and leave the directions unused?

Written 2026-09-21, before any run. Results: `REPORT.md` (2026-09-22, complete).

## Setup
Same recipe as the curves campaign (`maze/paper/curves/REPORT.md`): jirpyish d256 / 12 layers, 6x6 random-DFS mazes, 2M mazes,
one epoch, batch 256, AdamW 6e-4, 7,812 steps, bf16, torch.compile. The graft is the paper's `complete` edge graft: 72 signed
connection features through a FIXED unit-Gaussian `W_inj` (seed = run seed), added at every position from `<ADJLIST_END>` on.

Two changes:
1. **Site.** The paper's grafts sit at `blocks.0.hook_resid_pre`, where the residual is only token + position embedding: no
   computation precedes it, so a model cannot regenerate the code there. Here the graft is added at `resid_pre` of block k
   (k = 2, 3, 4, 6), so blocks < k can learn to write the code and blocks >= k keep reading it. k = 0 is run as the
   predicted-failure control.
2. **Fade.** p(step) = share of the graft removed: 0 until step 1,500 (the grafted models of the curves campaign learn to read
   the graft in a sharp rise at steps 300-600), linear to 1 at step 4,000, then 3,812 steps with no graft at all.
   `bernoulli`: each sequence loses its WHOLE graft with probability p (every forward pass is fully on or fully off, so
   nothing is off-distribution for the readers). `scale`: every graft multiplied by 1-p (pre-LN can absorb a rescale of a
   dominant component; run once to document that).

## Arms (`jobs_wave1.txt`, 40 runs)
| arm | flags | purpose |
|---|---|---|
| base | `--inject none` | reference for the autopilot solve rate; negative control for every direction metric (scored with the seed's W_inj, which it never saw) |
| permanent graft, k = 0,2,3,4,6 | `--fade none` | does a mid-layer graft train and steer as well as the layer-0 one; its `off` evaluation is the "reads but never writes" control |
| fade | `--fade bernoulli` | the claim |
| fade + frozen readers | `--freeze_readers True` | from step 1,500 nothing at or after the site trains: coping with a missing graft must happen below the site |
| fade + aux loss | `--aux_weight 1.0` | positive control: adoption is forced (MSE between the code read off the model's own residual through pinv(W_inj) and the true code, on dropped sequences). Tests whether steering/ablation work when the directions ARE the model's |
| one-offs at k=3 | `--fade scale`; `--freeze_readers True --unfreeze_at 5000`; `--inject trainable` | LN absorption; do directions survive once the readers are free again; trainable W_inj |
| fade at k=0 | | predicted failure (site not computable) |
Seed 0: everything. Seeds 1, 2: base, permanent, fade, fade+frozen, fade+aux at k = 2 and 4. All arms of a seed read the same
dataset file and the drop mask has its own RNG, so arms of a seed share their trajectory up to step 1,500 (up to GPU
nondeterminism).

## Measurements (`scripts/eval_fade.py`, standard eval set `maze/paper/curves/results/distance/sets/g6.npz`)
Validated before use: on the existing 6x6 fixed-graft seed-0 model its `on` condition reproduces the stored standard-set
outcomes item for item (2,100 / 2,100). Revised after the pre-launch audit (2026-09-21 evening, before any result was read);
the audit's numbers are quoted below.
- **Retention** (graft off; the model's own residual at the site along the teacher-forced true path, 3,000 mazes, fit on
  2,000 / test 1,000). Headline position = the ORIGIN token (first position where the code is needed); also `<PATH_START>` and
  path positions 1-3, and every layer 0-11 at the origin. Generative OLS of the residual on the +-1 connection features gives,
  per connection, the writing direction's cosine with the graft direction and its **gain** (amplitude relative to the graft's,
  zero-mean gauge). The criterion is gain, not cosine: the base model's own maze content at these positions has amplitude
  1-5x the graft's, so even a code planted at full gain 1 on top of it reads cos ~0.22 (audit), while gain is unbiased
  (1.00 / 0.50 / 0.20 recovered exactly). Also: accuracy of reading the connections through pinv(W_inj) with nothing fitted on
  the model ("zero-fit"), a fitted ridge probe, the majority rate, and the share of label-explained variance inside the graft
  span (chance level ~0.25 = 59/256, base model 0.246).
- **Control** (graft off, prompt = maze A): `delta{s}` adds s * (code(B) - code(A)) at the site on the graft's token positions;
  route-changed solve rate against B. Steering is sharply tuned to s ~ gain (audit, emulated: gain 1 -> s 1 gives 0.69, s 2
  gives 0.05; gain 0.5 -> s 0.5 gives 0.34), so s runs over {0.1, 0.25, 0.5, 1, 2} plus gain-matched s = gain x {0.5, 1, 2}.
  References: `on` of the permanent graft at the same k; token-matched steering of the base model at 6x6 = 1.3%; `delta_perm`
  (directions permuted across connections, s = 1) and the base model under the same edits = floors.
- **Necessity**: same-maze solve rate with the span of the zero-mean free directions mean-ablated at the site, PER POSITION
  (a single global mean erases position/token content and gives a random-subspace floor swinging 0.33-0.57 between draws;
  per-position means give 0.77-0.79 for three random draws and 0.78 for the graft span on the base model), vs 3 random
  subspaces of the same rank.
- **Autopilot**: `off` same-maze solve rate vs the base model at the same step and at its final step; the `off` rate at step
  1,400 (identical model for every arm of a seed) is the strength of the token route at fade start.
- Trajectory: steps 1,400 / 2,000 / 3,000 / 4,000 / 5,000 / 6,000 / 7,000 on 500 base mazes (`off`, `on`, `delta`, retention);
  final on all 3,000 with every condition. Checkpoints at those steps plus 1,500 and 4,000 are kept for re-evaluation.

## Reading the outcome (fixed before the runs)
- *Adopted*: at the final step, graft off, at the origin: gain >= 0.2 with zero-fit accuracy clearly above majority and
  in-span share above the base model's 0.25 (base model: gain 0.02-0.03, zero-fit 0.54 vs majority 0.61); some `delta`
  strength (fixed or gain-matched) gives a route-changed solve rate far above `delta_perm` and the base-model floor; ablating
  the span costs clearly more than the three random subspaces. Cosine is reported as descriptive only.
- *Reverted*: autopilot solve rate recovers but gain ~0 and every `delta` ~ floor: the model rebuilt the base route and the
  seeded directions are vestigial. Then the frozen-reader and aux arms say whether adoption is possible at all at that site
  (the aux arm is the real positive control: the frozen readers can still use the token route they had at step 1,500).
- *Broken*: autopilot solve rate stays well below the base model's at the matched step count.
Known limits: one grid size; fade schedule not tuned; fixed graft (one trainable run); the fade arms have fewer graft-free
steps than the base model has steps, so autopilot is compared against the base model's trajectory, not only its endpoint.
`aux_code_loss` reads the residual through pinv(W_inj), so it also pushes non-code content out of the 72-dim span: a cost to
autopilot in the aux arm is not evidence that adoption itself is costly. `on` for a faded model adds code(B) on top of
whatever the model writes itself (not "the graft as trained").

## Files
`scripts/train_fade.py` trainer (checkpoint layout = `maze/core/train.py`, so every existing eval reads these runs),
`scripts/fade_lib.py` schedule / forward-with-site / aux loss / freezing / site edits / KV decoder / retention,
`scripts/eval_fade.py` evaluator, `scripts/test_fade_lib.py` unit tests (8, pass locally), `scripts/pod_queue.sh` pod-side
GPU work queue, `scripts/reeval_traj.sh` (pod-side re-eval), `scripts/pull.sh` (local pull), `scripts/collect.py` (CSVs + table),
`scripts/plot_fade.py` / `show_traj.py` (figures / trajectory dump), `jobs_wave1.txt` the runs, `jobs_wave2_decorrelated.txt`
(staged, not run: W_inj decorrelated from the model init, see REPORT).
