# Seed and fade: results (2026-09-22)

40 runs, 6x6, one 8x A100 pod (00:03-03:41 UTC, ~$46), design in `README.md` (criteria fixed after the pre-launch audit,
before any result was read). Every number below is graft OFF at test time unless the column says otherwise. Eval set = the
standard 3,000-maze distance-stratified set (`maze/paper/curves/results/distance/sets/g6.npz`); "steer" = route-changed
solve rate of maze B with s * (code(B) - code(A)) added along `W_inj` at the site, best s of {0.1, 0.25, 0.5, 1, 2, gain x
{0.5, 1, 2}}; "perm" = same edit with the directions permuted across connections (s = 1); "ablate" = same-maze solve rate with
the graft span mean-ablated per position at the site vs the mean of 3 random subspaces of the same rank (59).

Figures: `fig_trajectory_s0.{pdf,svg}` (sites 2/3/4/6, all arms, 500 mazes per checkpoint), `fig_trajectory_s{1,2}` (sites
2/4). Tables: `results/final_table.txt` (every run), `results/{conditions,retention,retention_by_layer}.csv`.

## Seed means, sites k = 2 and 4, seeds 0-2 (mean +- SD over seeds)

| arm | k | autopilot solve | gain along W (origin) | steer, best s | perm floor | ablate span | ablate random |
|---|---|---|---|---|---|---|---|
| base model | 2 | 0.799 +- 0.033 | -0.006 | 0.004 | 0.001 | 0.777 | 0.775 |
| permanent graft | 2 | 0.215 +- 0.059 | 0.122 | 0.161 | 0.027 | 0.181 | 0.182 |
| fade | 2 | 0.813 +- 0.026 | 0.198 +- 0.047 | 0.046 +- 0.007 | 0.003 | 0.776 | 0.777 |
| fade, readers frozen | 2 | 0.720 +- 0.005 | 0.278 +- 0.029 | 0.183 +- 0.009 | 0.010 | **0.190** | 0.606 |
| fade + aux (diagnostic only, see note) | 2 | **0.855 +- 0.001** | **0.993** | 0.164 +- 0.053 | 0.008 | **0.446** | 0.743 |
| base model | 4 | 0.799 +- 0.033 | -0.015 | 0.003 | 0.002 | 0.771 | 0.772 |
| permanent graft | 4 | 0.438 +- 0.038 | 0.199 | 0.158 | 0.019 | 0.284 | 0.362 |
| fade | 4 | 0.820 +- 0.028 | 0.338 +- 0.004 | 0.027 +- 0.008 | 0.002 | 0.732 | 0.786 |
| fade, readers frozen | 4 | 0.762 +- 0.017 | 0.344 +- 0.014 | 0.183 +- 0.020 | 0.007 | **0.235** | 0.681 |
| fade + aux (diagnostic only, see note) | 4 | **0.846 +- 0.015** | **0.992** | **0.412 +- 0.109** | 0.013 | **0.284** | 0.703 |

Seed 0 only, k = 3 and 6, plus the one-offs (`results/final_table.txt`):

| arm | k | autopilot | gain | steer | perm | ablate | random |
|---|---|---|---|---|---|---|---|
| permanent graft (L0 reference: graft ON steers 0.696) | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| permanent | 3 | 0.490 | 0.112 | 0.185 | 0.025 | 0.324 | 0.396 |
| fade | 3 | 0.861 | 0.136 | 0.036 | 0.002 | 0.807 | 0.816 |
| fade, frozen | 3 | 0.745 | 0.306 | 0.118 | 0.007 | 0.230 | 0.669 |
| fade + aux | 3 | 0.864 | 0.985 | 0.245 | 0.011 | 0.397 | 0.770 |
| fade, scale ramp instead of Bernoulli | 3 | 0.863 | 0.174 | 0.067 | 0.003 | 0.772 | 0.823 |
| fade, trainable W_inj | 3 | 0.823 | 0.275 | 0.033 | 0.003 | 0.777 | 0.798 |
| fade at k = 0 (site not computable) | 0 | 0.758 | 0.000 | 0.005 | 0.000 | - | - |
| permanent | 6 | 0.804 | 0.292 | 0.109 | 0.006 | 0.574 | 0.735 |
| fade | 6 | 0.856 | 0.416 | 0.013 | 0.001 | 0.774 | 0.799 |
| fade, frozen | 6 | 0.810 | 0.413 | 0.138 | 0.005 | 0.269 | 0.684 |
| fade + aux | 6 | 0.856 | 0.991 | 0.323 | 0.012 | 0.292 | 0.739 |

Graft ON at the same sites (permanent-graft models, route-changed solve): k = 0 0.696, k = 2 0.417 / 0.231 / 0.418 (seeds
0-2), k = 3 0.135, k = 4 0.144 / 0.092 / 0.164, k = 6 0.011.

## Reading (against the pre-registered criteria)

1. **Plain fade = reverted, with a readable but unused residue.** Autopilot returns to the base model's level (0.81-0.86 vs
   0.77-0.83), a code IS present along the seeded directions (gain 0.14-0.44, zero-fit readout 0.58-0.60 vs majority 0.61,
   base model gain -0.01), but steering along them is at the floor (0.01-0.05; 10x the permuted floor, 10x below the graft)
   and ablating the span costs 0-5 points, same as random subspaces. The trajectory shows why: steering works transiently
   during the fade (0.18-0.24 at step 2,000, when the graft is still present 80% of the time) and decays as the token route
   takes over. Bernoulli vs scale fade and fixed vs trainable W_inj make no difference.
2. **Frozen readers = partial adoption at a cost.** Steering holds at 0.12-0.20 through the end, the span becomes strongly
   necessary (ablate 0.19-0.27 vs random 0.61-0.69), but autopilot is 5-10 points below the base and the models are fragile
   to any 59-dim ablation. (The "unfreeze at 5,000" one-off is invalid: under torch.compile the unfreeze never took effect and
   the run is bit-identical to the frozen one; fixed in `train_fade.py` afterwards, not re-run.)
3. **Aux arm: excluded from the claims (2026-09-22 decision: we never train on our own interp method; a loss on the
   graft directions makes "adoption" circular). Kept only as the diagnostic upper bound of what the evaluator can detect.** The model writes the exact seeded code itself (gain 0.99, zero-fit
   readout 1.000, in-span share 1.0), autopilot is at or above the base and the plain fade, steering along W_inj at s = 1
   gives 0.25-0.50 route-changed solves (k = 4: 0.45 / 0.29 / 0.50 across seeds; the layer-0 graft itself gives 0.70, the
   permanent k = 4 graft 0.09-0.16), and ablating the span costs 40-60 points vs 6-12 for random subspaces. Cost: the aux loss
   drives every non-code component out of the 72-dim span, and `on` (adding code(B) on top of the self-written code(A)) is
   weaker than the delta edit (0.04-0.10), as expected once the model owns the channel. The k = 2 site is the weakest for
   steering (0.16) although the code is written perfectly there: the readers above block 2 have less depth to use it.
4. **Two side findings.** (a) Mid-layer permanent grafts steer far worse than layer 0 (0.42 -> 0.01 from k = 2 to k = 6) and
   the model progressively stops needing them (k = 6 autopilot 0.09 -> 0.80 over training), because the maze is already
   computed below the site. (b) Even with a permanent graft, the blocks below the site come to write a partial copy of the
   code along W_inj (gain 0 -> 0.1-0.3), co-adapting to the readers.

So with this fade the model does not keep using the seeded directions on its own: the pre-existing token route wins, and
freezing the readers buys partial adoption at a performance cost. (The aux arm shows the evaluator would detect full
adoption if it happened, nothing more.)

**Where the code goes (per-layer retention, `results/retention_by_layer.csv`).** The seeded code is not lost; it moves
downstream. In the faded models the gain along W_inj at the origin grows with depth: fade k = 4: 0.34 at the site (block 4),
0.60 at block 5, 0.78 at blocks 7-11; fade k = 6: 0.42 at the site, 0.80 at blocks 9-11; k = 2: 0.14 -> 0.64. The base model
has 0 everywhere. So the blocks at and after the site keep writing the code (they were trained as its readers and co-adapted
to it), while the blocks below the site never fully take over the writer role; steering at the site then acts upstream of
where the code lives. Base-model probe legibility at the origin peaks at blocks 3-7 (ridge 0.755-0.763 vs 0.602 at block 0).

## Wave 3: 5x5, late sites, blanked-prompt curriculum (2026-09-22 18:28-19:43 UTC, 7 single-GPU pods, all deleted, ~$20)

Why: the user's target is a route-changed steering rate >= 0.90 "to see if it is possible at all"; at 6x6 the prompt-rewrite
ceiling is 0.77, at 5x5 it is 0.96 (layer-0 graft 0.92). Also tests injection at blocks 8/10 and a curriculum without any
interp-targeted loss: **blanked prompt** — during seeding the adjacency list is blanked on every graft-carrying sequence (the
graft is the model's only maze source, so the readers must depend on it); as the graft fades the share of blanked prompts
falls 1 -> 0, and a sequence that loses its graft always has its prompt (`train_fade.py --blank_prompt`). 15 runs, seed 0:
sites 4/6/8/10 x {permanent, fade, fade+blank}, blank+frozen readers at 4 and 8, base. Every faded model is also edited at
site+2 and site+4 (`--edit_layer`), and a **replace** edit (mean-ablate the span, then write s*code(B); `replaceA` writes the
model's OWN maze's canonical code back) was added after the additive edit kept failing. Figure `fig_trajectory_5x5_s0`,
tables `results_g5/final_table.txt`, CSVs in `results_g5/`.

| arm (5x5, graft off at test, edit at the site) | k | autopilot | gain @site / @site+4 | steer add. | replace | ablate span -> / random -> | replaceA |
|---|---|---|---|---|---|---|---|
| base | - | 0.966 | 0 / 0 | 0.001 | - | = random (0.95-0.96) | - |
| permanent (graft ON steers 0.005 / 0.005 / 0.003 / 0.003) | 4 / 6 / 8 / 10 | 0.962-0.976 | 0.46 / 1.05 (k4) | <= 0.115 | 0.006 (k10) | 0.863 / 0.945 (k4); 0.948 / 0.969 (k10) | 0.958 (k10) |
| fade | 4 / 6 / 8 / 10 | 0.965-0.982 | 0.60 / 1.28 (k4) | <= 0.016 | 0.015 (k8) | 0.933 / 0.953 (k4), ~random elsewhere | 0.954 (k8) |
| fade + blank | 4 | 0.972 | 0.50 / 1.36 | 0.062 | - | **0.767 / 0.955** | - |
| fade + blank | 6 | 0.982 | 0.53 / 1.46 | 0.061 | 0.110 | **0.832 / 0.971** | 0.912 |
| fade + blank | 8 | 0.982 | 0.58 / 1.25 | 0.070 | 0.131 | **0.802 / 0.972** | 0.909 |
| fade + blank | 10 | 0.982 | 0.72 | 0.043 | 0.081 | 0.846 / 0.973 | 0.911 |
| **fade + blank + frozen readers** | **8** | 0.976 | 0.66 / 1.01 | **0.262** | **0.256** | **0.420 / 0.965** | 0.814 |
| **fade + blank + frozen readers** | **4** | 0.937 | 0.64 / 1.34 | **0.450** | **0.380** | **0.310 / 0.900** | 0.708 |

Findings.
1. **A mid-layer graft is abandoned even when it is never faded.** At 5x5 the permanent grafts at k = 4-10 end with graft-on
   route-changed steering of 0.002-0.005 and solve 0.96 with the graft off; the trajectory shows the k = 4 graft read early
   (0.41 at step 1,400) and then dropped as the token route matures. The reversion in the fade arms is therefore not caused
   by the fade: once the maze is computable from the tokens below the site, the model discards the graft on its own. Only the
   layer-0 graft, cheaper than the token route, stays in use (paper: 0.92 at 5x5). Blocks 8-10 are the worst sites for a
   permanent graft and no better after a fade.
2. **The code migrates downstream and grows past the graft's amplitude** (gain 1.0-1.5 at site+2/+4 in every faded and
   permanent model), but editing it there does nothing either.
3. **The blanked-prompt curriculum makes the seeded subspace necessary but not steerable.** Ablating the span costs 13-19
   points (random subspaces <= 1); writing the canonical code of the model's own maze back restores most of it (0.91); but
   writing another maze's code, additively or by replacement, moves the model at most 0.13. The readers consume the subspace,
   yet the token route wins any conflict.
4. **Blank + frozen readers is the only configuration with real control and no interp loss:** at k = 4 steering 0.45
   (additive, s = 2; replace 0.38), span ablation 0.94 -> 0.31 (random 0.90), autopilot 0.937 (base 0.966); at k = 8
   steering 0.26, ablation 0.97 -> 0.42, autopilot 0.976. The trajectory holds k = 4 steering at 0.46-0.53 from step 3,000 to
   the end. Editing at site+2/+4 does nothing in these models either: the control lives at the seeded site. Steering peaks mid-fade in every blank arm (0.5-0.55 at step 3,000) and decays
   only where the readers are free to retrain.
5. **0.90 was not reached** by any configuration; the best legitimate number is ~0.5 (blank + frozen, k = 4), against a
   graft-on peak of ~0.6 in the same runs and 0.92 for the paper's layer-0 graft.

## Confound found during analysis (affects the paper's fixed grafts too)

`MazeInjector` draws `W_inj` from `torch.Generator().manual_seed(seed)` and the model init uses `torch.manual_seed(seed)`:
the two streams produce the same numbers, so `W_inj` row f equals the INITIAL embedding of token f for f < 47 (11 special +
36 coordinate tokens; verified: trained embeddings of the curves-campaign fixed-graft models keep |cos| 0.84 with their
W_inj row, 0.05 for shuffled rows). The base-model controls here are null on every metric at k >= 2 (gain 0, steer 0,
ablation = random), so the conclusions above do not rest on it, but (i) the layer-0 span-ablation numbers are artifacts
(ablating the span at L0 destroys token identity: base 0.009 vs 0.81 random) and (ii) the paper's description of the fixed
graft as random directions is wrong, and any direction-recovery statement that contrasts graft directions with embedding
directions needs re-checking. `train_fade.py --w_seed_offset 1000` decorrelates them; an 8-run check is staged in
`jobs_wave2_decorrelated.txt` (not run).

## Caveats
One grid size, one fade window (1,500-4,000 of 7,812 steps), aux weight 1.0 only, 3 seeds at k = 2/4 and one seed elsewhere.
The trajectory evals use 500 mazes (final: 3,000). Steering strengths stop at s = 2 and gain x 2. The 8 wave-1 trajectory
evals were re-run after a thread-oversubscription stall (identical code). Cost overran the $40 cap (~$46): the pod-side
auto-delete guard silently failed (no runpodctl API key on the pod) and evals ran ~2x slower than estimated.

## Reproduce
`scripts/pod_queue.sh jobs_wave1.txt` on an 8-GPU pod (datasets built per seed), `scripts/pull.sh`, then
`scripts/collect.py results`, `scripts/plot_fade.py --seed 0`, `scripts/show_traj.py results`. Unit tests: `scripts/test_fade_lib.py`.
