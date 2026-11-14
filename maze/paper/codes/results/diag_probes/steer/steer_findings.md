# Probe-direction steering: are the cross-frame wall probes causally used?

Script: `maze/paper/codes/steer_probe.py` (run on the pod). Eval set: EvalSet g{6,8}, seed 4321, 2000 fresh mazes, true world,
teacher-forced. Ridge probes as in probe_nonlinear.fit_linear (train = first 60 % of mazes, AUROC on last 25 %), all path
positions. Steering: a forward pre-hook on `model.blocks[k]` adds `alpha * u` at ONE path position (not the target) of a
test maze; `u` = unit(W[:-1, j] / sd). alpha = s * std(projection onto the ego-probe direction for D) -- one alpha per
(model, site, D), shared by every direction type (same-norm perturbations). 150 positions per (D in NSEW) x (closed / open-not-SP)
= 600 + 600 per cell; random = mean over 8 fixed random unit vectors. Hooking worked (a pre-hook on blocks[0] sees the
residual after the graft delta is added). Ego run used: `fast-inj-fixed-ego-complete` (not the trainable one).

Files: `steer_table.csv` (combined: g6 L0in/L6in from pass2 with s up to 64 and the graft direction; g6 L3in and all g8 from
pass1, s <= 16), `probe_auroc.csv`, per-pass raw per-position CSVs in `pass1_sites036_s1to16/` and `pass2_g6_bigs/`
(pass1 g6_edge L0in lacks the graft direction: bug fixed before pass2 / g8).

## Probe AUROC (4 immediate-neighbour ego walls, mean; allo = all absolute walls)
| model | L0in ego / allo | L3in ego / allo | L6in ego / allo |
|---|---|---|---|
| g6 base | 0.657 / 0.513 | 0.944 / 0.856 | 0.952 / 0.809 |
| g6 edge | 0.703 / 1.000 | 0.848 / 1.000 | 0.938 / 0.999 |
| g6 ego  | 1.000 / 0.684 | 1.000 / 0.728 | 1.000 / 0.730 |
| g8 edge | 0.663 / 1.000 | 0.679 / 1.000 | 0.803 / 0.996 |
| g8 ego  | 1.000 / 0.600 | 1.000 / 0.601 | 1.000 / 0.610 |
Base/edge L6 ego-neighbour AUROC here (0.95 / 0.94) is higher than the quoted ~0.89 / ~0.86 (those were on a different probe set /
target set); caveat: on a path, the wall toward the previous cell is always open, which is decodable from tokens alone.

## Key numbers (delta p_D at closed-wall positions; argmax -> D rate; n = 600 closed / 600 open)
Baseline p_D at closed walls: mean 6e-5 (median 6e-6); the models are near-deterministic, so steering rarely moves probability
mass until the perturbation is residual-sized.

| model site | s (alpha/|x|) | cross-frame + | cross-frame - | own-frame + | own-frame - | graft + | random |
|---|---|---|---|---|---|---|---|
| g6 base L6 (ego probe) | 16 (0.40) | 0.000 | 0.000 | -- | -- | -- | 0.000 |
| g6 base L6 | 32 (0.80) | 0.005 / 0.002 | 0.005 | -- | -- | -- | 0.005 / 0.003 |
| g6 edge L6 (ego x-frame, allo own) | 16 (0.54) | 0.003 | 0.000 | 0.001 | 0.000 | -- | 0.001 |
| g6 edge L6 | 32 (1.07) | 0.041 / 0.040 | 0.002 | 0.027 / 0.022 | 0.003 | -- | 0.014 / 0.010 |
| g6 edge L0 | 32 (0.26) | 0.047 / 0.045 | 0.044 | 0.114 / 0.100 | 0.000 | 0.247 / 0.250 | 0.027 / 0.027 |
| g8 edge L6 | 16 (0.48) | 0.007 / 0.005 | 0.002 | **0.129 / 0.125** | 0.000 | -- | 0.005 / 0.004 |
| g8 edge L3 | 16 (0.22) | 0.001 | 0.000 | **0.188 / 0.188** | 0.000 | -- | 0.003 |
| g6 ego L0 (allo x-frame, ego own) | 16 (0.48) | 0.003 / 0.002 | 0.001 | **0.348 / 0.350** | 0.000 | 0.437 / 0.442 | 0.003 / 0.002 |
| g6 ego L6 | 4 (0.29) | 0.000 | 0.000 | 0.000 (dlogp +0.47) | 0.000 | -- | 0.000 (dlogp +0.20) |
| g6 ego L6 | 16 (1.15) | 0.014 | 0.024 | 0.031 | 0.002 | -- | 0.019 |
| g8 ego L0 | 16 (0.79) | 0.109 | 0.042 | **0.356 / 0.425** | 0.011 | 0.544 / 0.568 | 0.033 |
(x / y = delta p_D / argmax->D rate.) Negative own-frame directions suppress p_D at open positions (e.g. g8 edge L3 allo-: dlogp_D
-2.2 at open positions; g6 ego L0 ego-: -0.44 nats at s=4), negative cross-frame directions do not beyond random.

## Conclusion
The cross-frame probe directions are **correlated features, not causal handles**: in the base and edge models, pushing along the
egocentric "wall D open" direction changes p(step D) no more than a random direction of the same norm at every site and scale
(and the + and - directions act alike, i.e. only the norm matters). For the ego model steered along the allocentric probe
for the same physical wall the picture is mostly null but not perfectly clean: g6 L0 and L6 = random; g8 L0 allo+ gives
+0.109 vs random 0.033 (a third of own-frame, and allo- also gives +0.042); g6 L3 is sign-INVERTED (allo- +0.171, allo+
+0.007, random +0.068) -- not the behaviour of a used "wall open" feature. Own-frame directions are causal and sign-specific where the frame lives:
the ego model's ego probe (and graft row) at block 0 moves p_D by +0.35 (argmax -> D 35 %) vs random 0.003; the edge model's
allo probe works at block 0 (g6) and strongly at blocks 3/6 for g8 (+0.13..0.19 vs random 0.003..0.005). Caveat: at block 6 in
the 6x6 models even own-frame directions are weak (the ego model's L6 own-frame effect is only ~2x random in log-prob and
nothing in probability before the perturbation is residual-sized), so single-position block-6 steering is a blunt tool there;
the cross-frame null is nonetheless consistent across all 5 models and 3 sites.
