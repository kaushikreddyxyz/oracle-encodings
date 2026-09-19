# Ground-truth features in the model's native position subspace — living report

**Started**: 2026-09-02 · **Model**: gemma-2-9b (paper-validated in App. C.2 of 2501.00070) · **Task**: 4×4 grid tracing, 16 single-token words per grid, random walks · **Hardware**: RunPod H100 SXM (`icl-gt-inject`) · **Code**: `gt_inject.py` (pod), `e1_analysis.py`, `e1_shared_subspace.py`, `plot_gt.py` (local) · **Raw**: `results_gt/` · **Figures**: `figures_gt/`

**Hypothesis under test.** The model can use *ground-truth* spectral-embedding coordinates, written into directions it natively uses, as a crutch: faster in-context learning and causal control over next-token predictions. Secondary: injected demo grids increase uptake.

Design principle: every causal claim rests on a readout where the tokens are insufficient or contradictory (override, unseen-node), not just on faster learning.

---

## E1 — Is there a native position subspace to write into? (COMPLETE)

14 long walks (1200 tokens): 3 word sets × 2 assignments × 2 walks, plus 2 extra word sets. Per run and per layer: centered node means (last 200 tokens), linear fit from spectral coords Z (Laplacian eigenvectors 2–3 of the 4×4 grid) to the node means, cross-run transfer, principal angles.

**Within a context, the model's plane is the spectral embedding.** In-sample R² of Z → node means inside the top-3 PCs: 0.73 at hs 26 (the Dirichlet-energy minimum), 0.65–0.72 for hs 24–40. Consistent with the paper's Table 2 (cos 0.95 between PCs and spectral embedding). Fig g1.

**The plane does not transfer.** Fig g1, g1b.

| top-2 PC plane, mean cos(principal angles) | hs 26 | hs 40 |
|---|---|---|
| same words, same assignment, different walk | 0.76 | 0.93 |
| same words, **different assignment** | 0.42 | 0.22 |
| different word set | 0.28 | 0.09 |
| (random 2-planes in ℝ³⁵⁸⁴) | 0.02 | 0.02 |

Cross-context R² of a map fitted on one word set and applied to another is ≤ 0 at every layer; even same-words-different-assignment is ≤ 0. The plane is reproducible across walks (it is a property of the word→position binding, not of the sample), but it is rebuilt per context. **There is no fixed, reusable 2-D position channel.** This retroactively explains why the earlier "green" injection worked (same words, same assignment: cos 0.93) and "purple" did not.

**It is not a linear function of the words' embeddings.** Only 2% of the plane's norm lies in the span of the 16 input embeddings at hs 26 (random-16-dim baseline 0.4%); 25% at hs 40. The analytic directions Eᵀz₂, Eᵀz₃ align poorly with the actual PCs (cos 0.10 at hs 26). Fig g1b left.

**But there is a shared higher-dimensional subspace at mid layers.** Pool the top-4 PCs of all runs on *other* word sets (52 dims): a new context's plane lies 64% inside it at hs 26 (random 1.5%), 47% at hs 16, 34% at hs 34, 14% at hs 40. So the model writes in-context structure into a shared ~50-dim subspace at mid depth, with a context-specific orientation inside it; late layers become word-specific. Fig g1b right.

**Consequence for the program.** "Native subspace from a different word set" (E2's original design) is expected to null out. The right single-grid oracle is the *self-calibrated* plane (E2b), and the calibration-free options to test are (i) random planes *inside* the shared subspace and (ii) the shared subspace's dominant directions. The multi-grid demos (E3) are now the decisive test of whether a *fixed externally-imposed* plane can be learned in context, since the model has no such plane of its own.

## E2 — Ground-truth coords via a map from a *different* word set (COMPLETE)

Query word set 4 (never used for calibration), 6 seeds × 1200-token walks (same walks in every condition). Features = Z·M where M is the pooled least-squares map from spectral coords to node means at hs 26, fitted on word set 0 (or set 1 for the "calB" arm). Injected at every node token by a forward hook on layer 25 (= hidden state 26).

**Tokens of context needed to reach a given accuracy (smoothed mean of 6 seeds):**

| condition | 0.50 | 0.60 | 0.65 | 0.70 |
|---|---|---|---|---|
| baseline (none) | 55 | 151 | 244 | 313 |
| foreign-set plane + GT2, β=1 / 2 / 3 | 53 / 51 / 53 | 149 / 138 / 111 | 235 / 161 / 151 | 309 / 259 / 233 |
| foreign-set plane (2nd calibration set) + GT2, β=2 | 50 | 136 | 155 | 238 |
| foreign-set plane + GT3, β=2 | 55 | 140 | 206 | 301 |
| random plane + GT2, β=2 (3 planes) | 55 | 150–152 | 238–239 | 311–312 |
| **extracted (model's own plane, top-3 PCs), β=2** | **42** | **57** | **82** | **111** |
| foreign-set plane @ hs 8 / 14 / 20 | 67 / 67 / 55 | 155 / 161 / 150 | 253 / 259 / 237 | 313 / 314 / 311 |

Reading (Fig g2): a plane fitted on a different word set is **weakly** read — 35–40% fewer tokens to reach 0.65–0.70 at β≥2, replicated with a second calibration set, dose-dependent, and layer-specific (hs 26 only; nothing at hs 8–20). Random planes are exactly baseline (3/3). The model's own plane is read strongly (3× fewer tokens). This is what E1 predicted: a foreign plane lies partly (≈64%) inside the shared mid-layer subspace, so it is partly legible; a random plane is not.

**Override** (Fig g3): features encode a permuted grid (fixed points 0–1, few shared edges). Injected-map mass rises only from the 0.15–0.18 chance level to 0.20–0.25 (β=3), while true-map mass drops ~0.05. The model follows the tokens; the foreign plane nudges but does not override.

**Unseen node** (Fig g4; X never appears in the 600-token context, then is presented with its feature; n=18): P(true neighbors of X) 0.112 → 0.137 with true coords everywhere; with X claiming Y's coords, P(neighbors of Y) 0.130 → 0.181 and P(neighbors of X) drops to 0.091; injecting only at the query token does nothing. Small but directionally correct causal effect, again consistent with weak legibility of a foreign plane.


## E2b — Ground-truth coords in the run's OWN plane; calibration-free planes (COMPLETE)

Same query walks as E2. "Self-calibrated": M fitted on one separate 1200-token calibration walk on the *same* word set and assignment (exactly the data the "extracted" reference uses), then features = Z·M — ideal spectral coordinates instead of the model's noisy node means. Calibration-free arms: analytic planes a = Eᵀz₂, b = Eᵀz₃ from the words' input embeddings or from their isolated hs-26 representations; the shared subspace's dominant 2 directions; random 2-planes *inside* the 52-dim shared subspace. All scaled to the self-calibrated norm.

| condition | 0.50 | 0.60 | 0.65 | 0.70 | Δ mass at 100 / 200 / 400 tokens (paired, 6 seeds) |
|---|---|---|---|---|---|
| baseline (none) | 55 | 151 | 244 | 313 | — |
| **GT2 in own plane, β=1 / 2 / 3** | 48 / 46 / 44 | 91 / 75 / 70 | 151 / 122 / 101 | 239 / 159 / 151 | β=2: **+0.093 / +0.093 / +0.050** (p < 2e-3, 6/6 seeds) |
| GT3 in own plane, β=2 | 46 | 75 | 129 | 205 | +0.089 / +0.083 / +0.040 |
| extracted (model's own top-3 PCs), β=2 | 42 | 57 | 82 | 111 | +0.136 / +0.128 / +0.076 |
| analytic plane from input embeddings, β=2 | 48 | 137 | 159 | 259 | +0.037 / +0.041 / +0.025 (p < 7e-3) |
| analytic plane from hs-26 token reps, β=2 | 73 | 155 | 257 | 313 | −0.008 / −0.005 / −0.008 (n.s.) |
| shared-subspace top-2 directions, β=2 | 55 | 148 | 257 | 313 | +0.012 / 0.000 / −0.018 (n.s.) |
| random plane inside shared subspace, β=2 (2 planes) | 81 / 73 | 150 | 238 | 317 / 320 | ≈ 0 / n.s.; **harmful at β=3** (0.60 reached at 197–234 vs 151) |

Reading (Fig g2b): **ground-truth coordinates are used when written into the plane the model built for that context** — 2× fewer tokens to 0.65–0.70, dose-dependent, every seed. But they are **not better than the model's own extracted vectors** (self-GT − extracted = −0.04 ± 0.03, n.s.); the model's own estimate is at least as informative as the ideal 2-D/3-D spectral embedding. No calibration-free construction substitutes: the input-embedding analytic plane gives a small but significant effect (+0.04, comparable to a foreign-set plane), the hs-26 analytic plane and the shared-subspace planes give nothing, and random planes inside the shared subspace *hurt* at β=3 — they perturb a subspace the model is using without encoding anything it can read.

**Override, own plane** (Fig g3b): with the permuted grid written into the model's own plane, true-map mass falls by 0.11–0.14 (β=2) and **0.18–0.28 (β=3)** below baseline (p ≤ 1e-3, 0/6 seeds above baseline), while injected-map mass rises 0.07 (β=2) / 0.12 (β=3) above its chance level. Features now compete seriously with tokens, although tokens still win in absolute terms (true-map ≈ 0.5 vs injected-map ≈ 0.3 at β=3).

**Unseen node, own plane** (Fig g4b; n=18): ΔP(true neighbors of X) = **+0.062 ± 0.019** (p=0.004) with true coords; with X claiming Y's coords, ΔP(neighbors of Y) = **+0.077 ± 0.015** (p=1e-4) and ΔP(neighbors of X) = −0.050 (p=8e-5). Injecting only at the query token still works (+0.034, p=7e-4) — unlike the foreign plane. This is causal control over a token the model has *no* contextual evidence about, purely from the injected coordinate.

## E3 — Multi-grid demos (COMPLETE): demos install nothing

Context = up to 3 demo grids (disjoint word sets, 500 tokens each, newline-separated) + query grid (600 tokens). The injected feature map is the foreign-set plane from E2 (the only fixed plane available across word sets — E1 showed the model has none of its own). Query-grid mass at 100 / 200 / 400 tokens:

| condition | 100 | 200 | 400 | query-injected − matched uninjected |
|---|---|---|---|---|
| no injection, 0 / 1 / 2 / 3 demos | .53 / .58 / .57 / .57 | .64 / .66 / .65 / .64 | .75 / .75 / .75 / .76 | — |
| injected, 0 / 1 / 2 / 3 demos | .57 / .61 / .60 / .60 | .70 / .70 / .69 / .68 | .78 / .77 / .76 / .76 | +0.038 / +0.030 / +0.032 / +0.030 |
| uninjected demos + injected query | .60 | .68 | .76 | vs injected demos: **+0.001 ± 0.001** |
| injected demos + uninjected query | .57 | .65 | .76 | = no-injection curve |
| geometrically scrambled demos + injected query | .60 | .68 | .76 | vs consistent demos: −0.001 |
| random subspace, 0 / 3 injected demos | .54 / .58 | .65 / .64 | .75 / .76 | vs none: +0.001 / +0.003 (n.s.) |

Reading (Fig g5, g6, g7): the query's gain is the same small foreign-plane effect (+0.03) regardless of how many injected demo grids precede it; injected demos do not change the query at all (Δ = 0.001); scrambled-geometry demos are indistinguishable from consistent ones; and a random subspace stays exactly at baseline even after 3 injected demos. Per-segment curves (g6) show no acceleration from grid 1 to grid 4 in either the injected or the uninjected condition. Uninjected demos themselves do not help the query either (−0.000). **There is no in-context acquisition of an externally imposed feature channel at this scale**, and no "learning to trust" the injections.

## Headline

1. The paper's in-context "world model" is real but **context-bound**: the 2-D plane that encodes position is rebuilt per word→position assignment (cos 0.42 across assignments of the same words, 0.28 across word sets; 0.76–0.93 across walks). It is not a linear function of the words' embeddings, but it lives inside a shared ~50-dim subspace at mid depth (64% at hs 26).
2. **Ground-truth spectral coordinates are used as a crutch — but only when written into that context's own plane.** Then: 2× faster rule-following, a strong override effect, and causal control over never-seen nodes (E2b). Written into any other plane (foreign-set, random, analytic, shared-subspace) they are at best weakly legible.
3. Whether ideal GT content beats the model's own extracted vectors is model-dependent (9b: no; 2b: yes). Either way the crutch works because the *plane* is right, not because the content is ideal.
4. **Demos do not install a channel** (E3): zero effect of 1–3 injected demo grids, for either a foreign-native or a random plane.

Practical consequence: causal control via injected coordinates requires one calibration pass on the target context (or, equivalently, the model's own node means). A transferable, calibration-free coordinate channel does not exist in gemma-2-9b at this task scale.

## Robustness

**12-seed replication of E2b on gemma-2-9b** (`results_gt_9b_s12/`, `figures_gt_9b_s12/`; fresh 12-seed baseline and extracted reference on the same walks):

| contrast (Δ mass at 100 / 200 / 400 tokens) | 12 seeds | p | seeds in direction |
|---|---|---|---|
| GT2 in own plane, β=2 − baseline | +0.098 / +0.087 / +0.042 | ≤ 2e-5 | 12/12 |
| GT2 in own plane, β=1 − baseline | +0.055 / +0.053 / +0.032 | ≤ 2e-6 | 12/12 |
| GT2 own plane − extracted | −0.028 / −0.017 / −0.014 | 0.09 / 0.29 / 0.26 | — |
| analytic input-embedding plane − baseline | +0.042 / +0.043 / +0.025 | ≤ 4e-5 | 12/12 |
| analytic hs-26 plane − baseline | −0.001 / +0.002 / −0.003 | n.s. | 6/12 |
| random plane inside shared subspace (2 planes) − baseline | ≈ 0 / −0.017 / −0.026 | n.s. / 0.005 / 0.002 | 1–4/12 |
| override β=3: true-map − baseline | −0.172 / −0.211 / −0.256 | ≤ 1e-7 | 0/12 |
| override β=3: injected-map − chance | +0.098 / +0.121 / +0.117 | ≤ 7e-8 | 12/12 |
| unseen node (n=36): true coords, ΔP(nbrs X) | +0.077 ± 0.013 | 2e-6 | — |
| unseen node: X claims Y, ΔP(nbrs Y) / ΔP(nbrs X) | +0.099 ± 0.014 / −0.054 ± 0.009 | 6e-8 / 9e-7 | — |
| unseen node: query-token-only, ΔP(nbrs X) | +0.036 ± 0.005 | 4e-8 | — |

Tokens to reach 0.65 / 0.70 (12 seeds): baseline 235 / 311, GT2-own-plane 135 / 160, extracted 93 / 149.

**gemma-2-2b replication of E1 / E2 / E2b** (`results_gt_2b/`, `figures_gt_2b/`; 26 layers, h_energy = hs 14; 6 seeds):

- *E1 replicates*: plane cos across walks 0.72–0.91, across assignments 0.22–0.62, across word sets 0.08–0.21; cross-set R² ≤ 0 at every layer. No fixed plane in 2b either.
- *Uptake* (Δ mass at 100 / 200 / 400): GT2 in own plane β=1: +0.042 / +0.045 / +0.034 (p ≤ 2e-3, 6/6); β=2: +0.034 / +0.039 / +0.008; β=3 hurts (−0.065 at 400). Foreign-set plane: 0 (set 0) / +0.03 (set 1). Random plane: 0. Random-in-shared plane: −0.04 (harmful). **Extracted vectors do not help on 2b** (−0.003 / −0.017 / −0.046), so here GT content *beats* the model's own estimate (+0.055 at 200, p=1e-3) — the opposite ordering from 9b.
- *Override* β=3, own plane: true-map **−0.46** at 400 tokens (p=2e-5), injected-map +0.11. Notably even the *foreign*-set plane overrides on 2b: true-map −0.25, injected-map +0.12 (p < 1e-4) — 2b is more susceptible to a foreign plane than 9b is.
- *Unseen node* (n=18): own plane, true coords ΔP(nbrs X) = +0.079 (p=3e-3); X claims Y: ΔP(nbrs Y) = +0.080 (p=4e-4), ΔP(nbrs X) = −0.031; query-token-only +0.035 (p=4e-4). Foreign plane: +0.061 / +0.048 (p < 1e-3). Replicates.

Cross-model summary: (1) no reusable plane — both models; (2) GT coordinates in the context's own plane give causal control (override, unseen-node) — both models, same magnitude; (3) uptake speed-up — both, weaker on 2b; (4) whether ideal GT content beats the model's own extracted vectors is model-dependent (9b: extracted ≥ GT; 2b: GT > extracted); (5) foreign/random planes give no uptake in either model, but 2b's predictions are causally movable by a foreign plane where 9b's barely are.

---

# Part II — Can a stable injection *port* be established in context? (2026-09-02, pod `icl-port`)

Code `port_inject.py` (stages p1/p2/p4/p2b/p3), `plot_port.py`; raw `results_port/`; figures `figures_port/`.

## P1 — Priming alignment: does a preceding grid align the next grid's plane? NO (COMPLETE)

Grid A (word set 1, 500 tokens) then grid B (set 4, 600 tokens) in one context, vs each alone; 6 seeds; planes = top-2 PCs of node means at hs 16/20/26/30.

| cos(principal angles), hs 26 | value |
|---|---|
| B after A vs A (same context) | 0.17 |
| B alone vs A alone (separate contexts) | 0.20 |
| B after A vs B alone | **0.92** |
| A in context vs A alone | 1.00 |
| 4 grids in one context, pairwise | 0.21–0.28 |
| same words, different placement, same context | 0.37 (0.61 at hs 16) |

A preceding grid leaves the next grid's plane exactly where it would be alone; nothing aligns. Consistently, injecting GT into B through A's in-context plane gives the same small gain as through A's separate-context plane (+0.025 over the uninjected in-context baseline, identical curves), versus +0.07 through B's own calibrated plane. Fig p1.

## P2 — Seeding at hs 26: uninformative by construction (COMPLETE, superseded by P2b)

Injecting P at hidden state 26 for the first 200–400 tokens changed early accuracy (proving the injection was applied) but left the model's later plane at hs ≤ 26 *identical* to the unseeded run (cos 1.0000): a hook at layer 25 can only reach later positions through layers 26–41, so it cannot touch the layers that build the plane. The one arm seeded at hs 16 did propagate (cos 0.96 vs unseeded) but did not align to P (0.14 vs 0.19). **The same caveat applies to E3**: demo injections at hs 26 were invisible to the query's plane-building layers. Hence P2b/P3 with early-layer injection.

## P4 — Withdrawal: the crutch substitutes for, rather than scaffolds, the model's own computation (COMPLETE)

Own-plane GT (β=2) for the first N tokens, then removed. Mass in the window after removal vs never-injected baseline: N=200: 0.666 vs 0.689 (−0.02); N=400: 0.757 vs 0.782 (−0.03); N=800: 0.796 vs 0.834 (−0.04); with the crutch kept, +0.03 to +0.06 above baseline. Removing the crutch leaves the model *worse* than if it had never had it, for a few hundred tokens. Fig p4.

## P2b — Seeding at EARLY layers: partial adoption of an overlapping plane, none for a random one (COMPLETE)

Seed plane P (β ∈ {2, 4, 8}) at hs 4 / 8 / 16 for the first 300 tokens only; measure the model's own plane at hs 26 over tokens 1000–1200 (uninjected). Δcos vs unseeded, paired over 6 seeds:

| plane | hs 4, β=2 | hs 8, β=2 | hs 8, β=4 | hs 16, β=4 | absolute best |
|---|---|---|---|---|---|
| foreign-set (overlaps shared subspace) | +0.15 (p=0.02) | **+0.21 (p=0.007)** | +0.08 (n.s.) | −0.06 (p=0.009) | 0.40 vs 0.19 unseeded |
| random inside shared subspace | +0.02 | +0.04 | 0.00 | −0.02 | 0.18 |
| random | 0.00 | 0.00 | 0.00 | +0.01 | 0.03 |

A gentle early seed of a plane that already partly lives in the shared subspace pulls the model's own plane toward it (0.19 → 0.40; "same plane" would be 0.76–0.93). Stronger seeds do worse and seeds at hs 16 push the plane *away*. A random plane is never adopted. Late injection through P after seeding gives no uptake benefit. Unseen-node after a foreign-plane seed (hs 8, β=4) with the X feature injected at hs 8: P(nbrs X) 0.112 → 0.192, but the seed alone gives 0.142 and P(nbrs Y) also rises to 0.162, so the specific effect is ~+0.03–0.05 (n=18). Fig p2b_early_seeding, p2b_unseen.

## P3 — Demos with EARLY-layer injection (the fair version of E3): still no port, but a new lead (COMPLETE)

E3 redone with the injection at hs 8 or hs 16 (β ∈ {2, 4}), so the demos' features reach the layers that build the query's plane. Query-grid mass, paired contrasts (6 seeds), windows [50–150] / [150–300] / [300–600] tokens:

| plane, layer, β | injected demos − uninjected demos (query injected in both) | query-only injection − no injection |
|---|---|---|
| foreign, hs 8, β=2 | +0.017 / +0.012 / +0.008 (p ≤ 5e-4) | −0.003 / −0.010 / −0.002 (n.s.) |
| foreign, hs 8 or 16, β=4 | ≈ 0 | **−0.07 to −0.12** (p ≤ 0.02) — harmful |
| random, hs 8, β=2 or 4 | ≈ 0 | ≈ 0 |
| random, hs 16, β=2 | +0.004 / +0.002 / +0.001 | +0.011 / +0.014 / +0.012 (p ≈ 0.01–0.07) |
| **random, hs 16, β=4** | +0.014 / +0.007 / +0.005 (p = 0.004 / 0.08 / 0.01) | **+0.026 / +0.031 / +0.040** (p = 0.05 / 0.01 / 0.004) |

Demos contribute at most +0.01–0.02 even when their features reach the plane-building layers — no port is installed by demonstration. The unexpected result is the last row: a **random** 2-plane injected at hs 16 with β=4, *without* demos or calibration, produces a small but reliable uptake gain (+0.03–0.04), whereas at hs 26 / β=2 (E2) random planes did nothing — and a *foreign* plane at the same strength hurts. Reading: a plane orthogonal to everything the model uses can be injected hard without interfering, and downstream attention then weakly exploits the consistent similarity structure; a plane that overlaps the shared subspace collides with the model's own computation at high β. Fig p3_early_demos. Follow-up: P5.

## P5 — A random plane as a calibration-free port: weak but real (COMPLETE)

GT coordinates written into a random 2-plane (two independent planes), injected at every node token at hs 12 / 16 / 20 with β ∈ {4, 8, 16}; no calibration, no demos. 6 seeds, same walks as E2.

**Uptake** (Δ mass vs baseline, tokens 200–600): hs 20 β=4 **+0.058 ± 0.008** (p=6e-4), second plane +0.032 (p=8e-3); hs 16 β=4 +0.038 (p=0.01), second plane +0.013 (n.s.); β=8 hurts (−0.02 to −0.07); β=16 destroys (−0.4 to −0.5); hs 12 never helps. A sharp optimum at β≈4, hs 16–20. Fig p5_random_port.

**Override** (permuted grid written into the random plane): at hs 20 β=4 the injected-map mass rises only +0.007 above chance (p=0.04) while true-map accuracy *rises* +0.024 (p=0.01) — a wrong map helps the model about half as much as the right one. So part of the uptake gain is nonspecific (a consistent per-token tag sharpens token tracking) and part is geometry-specific (right map +0.058 vs wrong map +0.024), but the geometry is far too weak to override the tokens. At β=8 there is a small genuine override (+0.035, p=3e-3) at the cost of overall accuracy (−0.10).

**Unseen node** (X never in context; X's coordinate written into the random plane; n=18, paired):

| layer, β | ΔP(true neighbors of X), true coords | ΔP(neighbors of Y), X claims Y's coords | specificity |
|---|---|---|---|
| hs 16, β=4 | +0.022 ± 0.006 (p=2e-3) | +0.043 ± 0.008 (p=8e-5) | ΔP on the other set ≈ 0 |
| hs 20, β=8 | +0.033 (p=0.05) | +0.073 (p=1e-3) | ≈ 0 |
| **hs 12, β=8** | **+0.065 ± 0.016 (p=6e-4)** | **+0.093 ± 0.020 (p=2e-4)** | ≈ 0 |
| (own-plane crutch, hs 26, β=2, for reference) | +0.062 / +0.077 (12 seeds) | +0.077 / +0.099 | |

This is the first calibration-free causal control we have found: a token the model has no contextual evidence about is placed where its injected random-plane coordinate says, with no demos and no knowledge of the model's own plane. At hs 12 β=8 the effect is as large as the own-plane crutch — but that setting wrecks ordinary uptake (−0.11), i.e. strong early injection lets the coordinate dominate a token that has nothing else, while interfering with tokens that do. Fig p5_unseen.

## Part II headline

1. **No in-context port can be installed by priming or demonstration.** A preceding grid leaves the next grid's plane exactly where it would be alone (P1); demos change the query by ≤ 0.02 even when their features reach the plane-building layers (P3).
2. **Seeding barely moves the model's own plane.** Only a gentle early seed of a plane that already overlaps the shared subspace pulls it partway (cos 0.19 → 0.40); random planes are never adopted (P2b). The P2 arm at hs 26 was uninformative by construction — injections at layer h reach other positions only above h — and the same caveat softens E3.
3. **The crutch is a substitute, not a scaffold:** removing an own-plane injection mid-context leaves the model 0.02–0.04 worse than never having had it (P4).
4. **A random orthogonal plane is a weak universal port** (P5): injected at hs 16–20 with β≈4 it gives a modest, replicated uptake gain and, more importantly, calibration-free causal control over never-seen tokens (+0.02 to +0.09, p ≤ 2e-3); but it cannot override token evidence, and pushing it hard enough to matter for seen tokens breaks ordinary processing. The trade-off suggests the next design: strong injection only where the token carries no other information.

---

# Part III — The QK-space port (2026-09-02, pod `icl-qk-port`)

Code `qk_port.py` (stages screen / ablate / port / universal / port2), `plot_qk.py`; raw `results_qk/`; figures `figures_qk/`. All on gemma-2-9b, word set 4 unless stated, walks shared with E2/E2b (paired).

**Idea.** The model's position *plane* is context-specific (Part I), but the *weights* that read it are fixed. If we find the heads that read position and write coordinates directly into their query/key subspace, injected coordinates produce adjacency-structured attention logits in every context — a port defined by weights, not by calibration.

## Q1 — Which heads read position? (screen, 3 seeds)

Per head, attention from node tokens (after 400 tokens) to previous same-node occurrences, neighbor occurrences, other nodes, successor-of-same (induction targets), etc., as enrichment over base rate; baseline vs own-plane injection (hs 26) vs random plane (hs 16 / 20).

- The strongest "neighbor-attending" heads at baseline (L7H1, L28H2, L15H3; enrichment 4–4.5) are the induction heads (successor-of-same enrichment 12–14): on a random walk the token after a previous occurrence of X is always a neighbor of X. Pure adjacency heads (neighbor occurrences that are not induction targets): L10H4, L0H2, L6H9, L27H2 (3.9–4.3).
- **Under the own-plane injection, nearly every head at layers 26 and 28 raises its neighbor- and same-node attention** (e.g. L26H2: same 0.05 → 0.25, neighbors 0.14 → 0.39 of attention mass). A layer-wide response, not a few special heads. Under the random plane at hs 16, a weaker diffuse response led by L16H13, L17H4, L20H8. Fig q1_screen.
- Readers chosen for the port: L26H2, L26H4, L26H5, L28H11 (own-plane responders) + L16H13, L17H4, L20H8, L16H14 (random-plane responders); injection at hs 16 so it is present at every reader's input.

## Q2 — Are those heads necessary? (ablation, 3 seeds)

| ablated group (8 heads) | no injection | own-plane | random plane |
|---|---|---|---|
| none | 0.782 | 0.790 | 0.841 |
| readers (own-plane responders) | 0.791 | 0.791 | 0.848 |
| top baseline adjacency/induction heads | **0.433** | 0.551 | 0.501 |
| top induction heads | 0.559 | 0.654 | 0.609 |
| 8 random heads | 0.809 | 0.801 | 0.863 |

Ablating the reader heads removes neither baseline performance nor the injection gains: reading is distributed. Ablating the induction/adjacency heads collapses rule-following, and then the own-plane injection recovers +0.12 — the injected geometry is an alternative route the model falls back on. Fig q2_ablate.

## Q3 — The port (6 seeds)

Construction: for each reader head, the bilinear QK form B = (W_Q D)ᵀ(W_K D) restricted to the 35 slow-RoPE dimension pairs (wavelength > 5000 tokens, so the injected similarity is position-invariant), with the RMSNorm scale D folded in. Directions = top generalized eigenvectors of the summed symmetrized forms, penalized against projection through the fast-RoPE rows (the unpenalized version leaks 40–55% of its norm into fast dims; penalized 10–20%). Coordinates = spectral coords Z₂ along those directions, scaled like the other planes. Implied added logits per reader head at β=1: self 2.7–4.4, neighbor 1.8–3.0, other −0.7 to −1.1 (×β² at strength β).

| condition (Δ mass vs baseline, tokens 200–600; p, 6 seeds) | Δ |
|---|---|
| **QK-port β=4 @hs 16** | **+0.062 ± 0.008 (p=7e-4)** |
| QK-port β=2 / β=8 | +0.013 / **−0.105** (over the softcap) |
| own-plane crutch (calibrated) | +0.048 |
| random plane @hs 16 / @hs 20, β=4 | +0.038 / +0.058 |
| unpenalized port β=4 | +0.034 (port − plain: +0.028, p=4e-4) |
| non-reader port β=4 (same recipe, 8 random heads) | +0.038 (port − nonreader: +0.024, p=0.01) |
| anti-port β=4 (most negative eigen-directions) | +0.030 (port − anti: +0.033, p=3e-4) |

The port is the best uptake condition and beats every design control, but is statistically tied with the best random plane (+0.004, n.s.): about +0.03 of any strong consistent per-token vector is nonspecific. Override through the port: injected-map +0.023 above chance (p=3e-3), true-map unchanged — it does not overpower tokens.

**Where the port separates: causal control at a non-destructive strength** (unseen node, n=18, paired):

| coordinate channel | ΔP(true nbrs of X), true coords | ΔP(nbrs of Y), X claims Y |
|---|---|---|
| own plane (calibrated), hs 26 β=2 | +0.062 (p=4e-3) | +0.077 (p=9e-5) |
| **QK-port, hs 16 β=4 (no calibration)** | **+0.054 (p=5e-4)** | **+0.079 (p=3e-6)** |
| random plane, hs 16 β=4 | +0.022 (p=2e-3) | +0.043 (p=8e-5) |

Random planes only reach this level at strengths that wreck ordinary uptake (P5: hs 12 β=8, −0.11); the port reaches it at the strength that *improves* uptake.

Mechanism check (Fig q3_reader_attention): under the port the mid-layer readers' neighbor attention rises (L17H4 1.06 → 2.03, L16H13 0.86 → 1.46) but the layer-26/28 readers barely move (0.64–0.92 → 0.71–1.11), unlike under the own-plane injection (→ 1.9–2.0). The port acts through the mid readers.

## Q5 — Universality (3 new word sets, new assignments, no recalibration; 3 seeds each)

Uptake +0.065 / +0.059 / +0.051 (random plane +0.02 each). Unseen node (n=18): ΔP(true nbrs of X) = **+0.057 (p=8e-5)**; X claims Y: ΔP(nbrs Y) = **+0.092 (p=3.5e-7)**, ΔP(nbrs X) = −0.039. A weight-defined channel that moves a never-seen token to an arbitrary grid position on word sets it was never calibrated on. Fig q5_universal.

## Q4 — Round two: adjacency kernel, late-reader port, specificity (6 seeds, word set 4)

Three new constructions from the same weights: (i) an **adjacency-kernel** port — the grid adjacency matrix has 6 positive and 6 negative eigenmodes, and the readers' summed QK form is indefinite, so mapping positive modes onto positive generalized eigen-directions and negative onto negative realizes K ≈ A exactly (self ≈ 0, neighbor > 0, other ≈ 0); (ii) a **late-reader port** built only from L26H2/H4/H5/L28H11 and injected at hs 26 (their input); (iii) the late-reader adjacency port.

| condition | Δ uptake [100–200] / [200–600] / [600–1200] | override: injected-map − chance; true-map − baseline | unseen: ΔP(nbrs X) true / ΔP(nbrs Y) claim |
|---|---|---|---|
| random plane @hs 20, β=4 | +0.053 / +0.058 / +0.049 | — | +0.012 (n.s.) / +0.026 (p=0.03) |
| spectral port @hs 16, β=4 | +0.069 / +0.062 / +0.062 | +0.023 (p=3e-3); −0.009 | +0.054 / +0.079 |
| **adjacency port @hs 16, β=6** | **+0.093 / +0.094 / +0.083** (p ≤ 1e-3) | (β=4) +0.017 (p=4e-6); +0.009 | (β=6) +0.044 (p=4e-3) / +0.080 (p=1e-4) |
| adjacency port β=8 | +0.101 / +0.086 / +0.065 — still not destructive | | |
| **late-reader port @hs 26, β=4** | +0.092 (p=4e-4) / +0.018 (n.s.) / −0.037 | **+0.191 (p=4e-5); −0.216 (p=1e-5)** | **+0.126 (p=5e-4) / +0.124 (p=2e-5)**; P(nbrs X): 0.11 → 0.24, fake set 0.05 |
| late-reader port β=3 | +0.090 / +0.036 / −0.006 | | +0.082 / +0.088 (p ≤ 2e-4) |
| late-reader adjacency port β=4 | +0.074 / +0.040 / +0.008 (all p ≤ 1e-4) | | +0.080 (p=1e-7) / +0.078 (p=4e-5) |
| own-plane crutch (calibrated), for reference | +0.101 / +0.048 / +0.008 | (β=3) +0.12; −0.28 | +0.062 / +0.077 |

- The adjacency port beats the best random plane by +0.036 (p=7e-4) and the spectral port by +0.032 (p=2e-3) on uptake: writing *adjacency* rather than *self-similarity* into the readers' QK space is what a random plane cannot do.
- The late-reader port is the causal-control champion of the whole program: the model *follows a wrong map written into it* (injected-map +0.19, the first real override by any calibration-free channel, larger than the calibrated own plane's +0.12) and places a never-seen node with ~±0.125 of probability mass, twice the own-plane effect. Its uptake profile is the own-plane's (strong early, converging later). Mechanism check: the late readers' neighbor attention goes 0.64–0.92 → 2.0–2.07 under it — the same signature the own-plane injection produces (1.9–2.0) — and 4.7 under the late adjacency port.
- Specificity: ablating the 8 reader heads removes 76% of the spectral port's gain, 58% of the adjacency port's, and 48% of the random plane's (all p ≤ 2e-3). Fig q4_port2, q4_reader_attention, q4_unseen.

## Q6 — Round three: the late port on new word sets, with the controls that could falsify it

Same walks/protocol; **main** = word set 4 (6 seeds, unseen n=18), **new** = word sets 1–3 with new assignments (9 contexts, unseen n=27). Everything at hs 26 except the calibrated own plane (hs 26, β=2). All Δ are paired vs no injection.

| channel @hs 26 | uptake Δ [100–300] main / new | override: injected-map − chance (true-map − base) main / new | unseen ΔP(true nbrs of X) main / new | unseen ΔP(nbrs Y \| X claims Y) main / new |
|---|---|---|---|---|
| own plane, calibrated per context | +0.088 / +0.062 | +0.062 (−0.13) / +0.076 (−0.15) | +0.081 / +0.086 | +0.110 / +0.108 |
| random plane, β=4 | +0.015 / +0.004 (n.s.) | +0.006 / +0.004 (n.s.) | +0.007 / +0.031 | +0.004 / +0.017 (n.s.) |
| **late-reader port, β=4 (weights only)** | +0.074 / +0.046 | **+0.191 (−0.22) / +0.176 (−0.23)**, p ≤ 4e-5 / 1e-9 | **+0.163 / +0.134**, p ≤ 5e-4 / 3e-6 | **+0.144 / +0.183**, p ≤ 7e-5 / 1e-9 |
| late-reader port, β=3 | +0.079 / +0.058 | — | +0.103 / +0.081 | +0.126 / +0.146 (p=6e-12) |
| late adjacency port, β=4 | +0.066 / +0.049 (sustained: +0.017 / 0.000 at 300–1200) | +0.091 (−0.07) / +0.094 (−0.09) | +0.069 / +0.091 | +0.133 / +0.123 |
| late **anti**-port, β=4 (negative eigen-directions) | **−0.082 / −0.078** | −0.009 / −0.009 | −0.013 / −0.004 | −0.025 / −0.021 |
| **non-reader** late port, β=4 (4 random heads at layers ≥ 26) | +0.096 / +0.061 | +0.189 / +0.168 | +0.154 / +0.155 | +0.139 / +0.179 |

Reading:
1. **The late port replicates on word sets it was never calibrated on, at full strength.** On new words, writing a wrong map into it makes the model follow that map (+0.18 above chance; true-map −0.23) — more than the calibrated own plane does (+0.08) — and it places a never-seen token with ±0.13–0.18 of probability mass, ~1.7× the calibrated crutch. Uptake acceleration matches the own plane early and converges later (the crutch profile).
2. **It is the directions, not the strength.** A random plane at the same layer and strength does nothing; the anti-port (negative self-similarity through the same heads) *hurts* uptake by 0.08 and has no causal effect.
3. **But it is not the readers.** A port built by the same recipe from four *random* heads at layers ≥ 26 is indistinguishable from the reader port on every readout. The channel is the subspace along which mid/late-layer attention heads share positive query–key self-similarity, not a property of particular heads.

**What the channel is** (`qk_planes.py`, `results_qk/planes.json`): ports built from different random head sets are *not* one plane (cos 0.03–0.45 to the reader port; two random late sets are ≈ 0.05 from everything), yet their effectiveness tracks their overlap with it: a mid-layer set with cos 0.59 matches the reader port on unseen-node (0.15 / 0.26 / 0.32 vs 0.22 / 0.28 / 0.32 across seeds; none 0.07–0.13); a late set with cos 0.23 is partial (0.12 / 0.16 / 0.22); early-layer sets (cos ≈ 0.01) give nothing and cut uptake by 0.09. So there is a family of usable directions — the QK-positive cone of mid/late heads — and the reader plane is a good representative of it, not a unique key.

## Q7 — Unseen EDGE: where the port's power ends (COMPLETE)

The walk never traverses one true edge (X,Y); both nodes appear with their other neighbors (bipartite grid ⇒ no shared neighbors, so tokens put X,Y at path-distance 3). 15 contexts × 2 directions = 30 paired items, ~43 queries each, all channels at hs 26 unless noted. Code `unseen_edge.py`, raw `results_edge/edge.json`, Fig story4_unseen_edge.

| channel | P(held-out Y \| X) | × baseline | P(seen neighbor, each) | P(dist-2 non-nbr, each) | items with Y > dist-2 |
|---|---|---|---|---|---|
| no injection | 0.006 | — | 0.41 | 0.023 | 10% |
| own plane (calibrated) | 0.011 | 1.9 | 0.39 | 0.023 | 30% |
| **QK-port β=4** | **0.019** | **3.2** | 0.38 | 0.017 | **40%** |
| QK-port β=3 | 0.011 | 1.8 | 0.40 | 0.020 | 20% |
| random plane | 0.008 | 1.4 (n.s.) | 0.41 | 0.023 | 13% |
| anti-port | 0.010 | 1.7 | 0.38 | 0.021 | 17% |
| adjacency port, edge **in** kernel | 0.011 | 1.8 (log-ratio +0.56, p=5e-3) | 0.39 | 0.024 | 30% |
| adjacency port, edge **deleted** from kernel | **0.002** | **0.4 (log-ratio −0.94, p=5e-5)** | 0.41 | 0.021 | 7% |

Reading:
1. **The model does not fill in the missing edge on its own.** Baseline P(Y|X) = 0.006 is *below* the distance-2 non-neighbors (0.023) and 70× below a seen neighbor. The in-context world model follows the walk's path-distance geometry, not the true lattice: it is retrieval-plus-smoothing, not a geometric prior that completes edges. (An answer to the paper's memorization-vs-geometry question for this task.)
2. **The port carries the specific edge, in the right direction, but cannot overpower local token evidence.** The QK-port triples P(Y|X) and moves Y above the distance-2 nodes in 40% of items (paired contrast +0.020, p=6e-4), and the surgical control is clean — the adjacency port with the edge in its kernel raises Y (+0.56 nats) while the same port with that one edge deleted suppresses it (−0.94 nats, p=5e-5). But the absolute level stays ≈ 1/20 of a seen neighbor.
3. Why this differs from the unseen-node result: at an X query the induction heads fire on X's many previous occurrences and copy its seen successors; the port's attention to Y's occurrences is one voice against that. In the unseen-node test those heads have nothing to copy, so the port dominates; in the override test the channel contradicts *every* query consistently and wins ~45% of the mass. Inserting one edge against intact local evidence is the hardest case, and the port loses it.

## Part III headline

**A calibration-free port exists, and it is defined by the attention weights.** Take the query/key matrices of a handful of mid-to-late heads, restrict to the RoPE-slow dimensions, fold in the RMSNorm scale, penalize leakage through the fast dimensions, and take the directions of largest shared positive self-similarity. Write grid coordinates along those directions into the residual stream at hs 26 and, on word sets and assignments the port never saw:

- the model **follows an injected wrong map** (+0.18 above chance, p ≤ 1e-9) — the first override by any channel that doesn't require a per-context calibration, and stronger than the calibrated one;
- a **never-seen node is placed wherever the coordinate says** (ΔP ≈ +0.13 to +0.18, p ≤ 1e-9), ~1.7× the calibrated crutch;
- learning is accelerated early (+0.05–0.09) with the own-plane's profile;
- the reader heads' neighbor-attention rises to exactly the level the model's own plane produces (0.7 → 2.0);
- and the effect is sign-specific (anti-port hurts), direction-specific (random plane null) but *not* head-specific (any mid/late heads with sufficient overlap work).

The adjacency-kernel variant (self ≈ 0, neighbor > 0) is the best learning accelerator (+0.09 sustained at hs 16, not destructive up to β=8) and the late adjacency port is the balanced choice (strong causal control *and* sustained positive uptake).

Caveats specific to Part III: one model (gemma-2-9b); the port parameters (slow-RoPE cutoff 5000 tokens, fast-penalty 1.0, k=2, β) were chosen with minimal tuning; reader selection was by attention response to injection; the ablation shows reading is distributed, consistent with the non-reader result. The QK-positive-cone characterization is qualitative (plane overlap vs effect, 11 head sets) and deserves a proper study.

# Part IV — The subspace zoo: which planes accept injected coordinates, and how they relate to the native plane and the QK port (2026-09-03, pod `icl-subspace-zoo`)

Question (user): are there other subspaces that work as well as the QK-derived port or the model's own plane — e.g. principal
components of the 16 word embeddings, or of the whole vocabulary — and are they orthogonal to, or overlapping with, the native
plane and the QK plane? Goal: a plane that works arbitrarily well and is grid-agnostic.

Setup (`subspace_zoo.py`, `zoo_round2.py`, `zoo_round3.py`, `zoo_round4.py`; figures `figures_zoo/`, raw `results_zoo/`).
Every candidate is a 2-D plane U. Node j receives Z2_j · U (spectral grid coordinates along the two directions), rescaled to the
reference norm used for the QK port, times β, added at hidden state 26 at every occurrence of node j. All candidates are run on
the same 15 contexts (word set 4 × 6 assignments; word sets 1, 2, 3 × 3 assignments), so every comparison is paired.
Readouts: faster learning (Δ rule-following mass, tokens 100–300), override (mass on the neighbors of a permuted map minus
chance), unseen-node control (Δ P(true neighbors) at a node never shown; and the fake-coordinate flip), plus the late-context
accuracy (tokens 400–1200) and the mass on the current node's own token (a self-copy side effect). Geometry per plane: cos of the
principal angles with the context's own plane and with the QK port, the "QK score" uᵀSu/λmax(S) of the plane under the
late-reader form S, fast-RoPE leakage, alignment with the 16 word embeddings, energy inside the vocabulary and activation subspaces.

## Z1 — The zoo (47 planes, β 2 and 4, `zoo1_effects.png`, `zoo2_geometry.png`, `zoo3_overlap.png`, `zoo_summary.txt`)

**Only planes with a positive QK score work; nothing built from token embeddings does.** Override at β4, 15 contexts:

| family | plane | uptake | override | unseen X | claim-Y | QK score | cos own | cos QK port |
|---|---|---|---|---|---|---|---|---|
| QK, all 256 late heads (L26–41), no reader selection | `qk_alllate` | +0.070 | **+0.197** | **+0.185** | +0.162 | 0.42 | 0.15 | 0.34 |
| QK, all 16 heads of L26 | `qk_L26all` | +0.059 | +0.197 | +0.179 | +0.185 | 0.53 | 0.10 | 0.54 |
| QK, 4 reader heads (Part III port) | `qk_late` | +0.057 | +0.182 | +0.145 | +0.167 | 0.62 | 0.10 | 1.00 |
| QK form restricted to top-1024 vocab PCs | `vocab_qk1024` | +0.077 | +0.159 | +0.124 | +0.156 | 0.35 | 0.09 | 0.57 |
| QK form restricted to top-64 activation PCs (foreign word set) | `act_qk64` | +0.070 | +0.153 | +0.123 | +0.135 | 0.21 | 0.22 | 0.13 |
| own plane, calibrated (β2 equivalent ≈ 2.9× ref norm) | `own` (β4) | +0.027 | +0.130 | +0.154 | +0.207 | 0.17 | 0.95 | 0.10 |
| foreign word set's fitted plane (E2 arm) | `foreign_set0` | −0.090 | +0.116 | +0.025 | +0.007 | 0.16 | 0.28 | 0.06 |
| top-2 activation PCs of a foreign walk | `act_pc1_2` | +0.020 | +0.089 | +0.041 | +0.035 | 0.10 | 0.21 | 0.03 |
| QK, 4 mid readers (L16–20), injected at 26 | `qk_mid4` | +0.044 | +0.072 | +0.062 | +0.112 | 0.04 | 0.03 | 0.03 |
| QK form restricted to top-256 / 64 / 16 vocab PCs | `vocab_qk256/64/16` | +0.051/−0.011/−0.024 | +0.069/+0.038/+0.017 | +0.044/+0.042/+0.022 | | 0.18/0.08/0.03 | | 0.36/0.10/0.02 |
| vocabulary PCs (pairs 1–2 … 3583–3584), centered or not | 12 planes | −0.09 … +0.01 | −0.03 … +0.02 | ≤ +0.03 | ≤ +0.03 | ≤ 0.02 | 0.03 | 0.02 |
| PCs of the 16 word embeddings (8 pairs) + QK-best plane inside their span | 10 planes | −0.07 … −0.03 | ≤ +0.012 | ≤ +0.03 | ≤ +0.03 | ≤ 0.02 | 0.03–0.06 | 0.02 |
| MLP-input singular directions, L26 | `mlp_in26` | **−0.230** | 0.00 | 0.00 | +0.01 | −0.01 | 0.01 | 0.03 |
| attention-output singular directions, L25 | `attn_out25` | −0.004 | +0.012 | +0.006 | +0.012 | 0.00 | 0.04 | 0.01 |
| random planes (2) | `rand`, `rand2` | +0.01 | +0.01 | +0.02 | +0.02 | 0.00 | 0.02 | 0.02 |
| anti-ports (bottom of the spectrum) | `qk_anti`, `qk_alllate_anti` | −0.08 / −0.12 | −0.01 / +0.03 | ≈ 0 | ≈ 0 | −0.52 / −0.20 | | |

(all working-plane effects p ≤ 1e-8; the ≈ +0.02 unseen-X floor of random planes is a nonspecific effect of adding norm at
the query node and is absent in the claim-Y readout at that size.)

Reading the table:
- **The best calibration-free plane is the top-2 plane of the QK form summed over all 256 late heads** (`qk_alllate`) — no head
  screening at all. It beats the 4-reader port on every causal readout, and the all-of-L26 form ties it.
- **Token-embedding directions do not work.** Every PC pair of the 16 word embeddings and of the whole vocabulary (from PC 1 to
  PC 3584, centered or not) gives zero override and *hurts* early learning by 0.03–0.09. Their QK score is ≈ 0 and their
  fast-RoPE leakage > 1 (the late heads read them through position-dependent dimensions, so the injected similarity oscillates
  with token distance). Restricting the QK form to the top-k vocabulary PCs recovers the port only gradually: 16 → 64 → 256 →
  1024 PCs gives override +0.02 → +0.04 → +0.07 → +0.16. The port therefore *does* live mostly inside the vocabulary's span
  (1024 of 3584 dimensions), but not in any small set of PCs of it.
- **A working plane nearly orthogonal to the QK port exists**: `act_qk64`, the QK-best plane inside the top-64 activation PCs of
  a walk on a *different* word set, works as well as the reader port (override +0.15, unseen +0.12) while its overlap with the
  port plane is cos 0.13 (random: 0.02). So the port is not one plane.
- **Across all 47 planes, causal control tracks the QK score** (Spearman ρ = 0.67, p = 3e-7, for override and for unseen-X) and,
  more weakly, the overlap with the native plane (ρ = 0.53–0.58); it is anti-correlated with fast-RoPE leakage (ρ = −0.45).
- **Overlap structure** (`zoo3_overlap.png`): two clusters. The QK cluster {4-reader port, L26 form, all-late form, vocab-restricted
  forms} has pairwise cos 0.3–0.6; the activation cluster {`act_qk64`, foreign plane, foreign activation PCs} has pairwise cos
  0.3–0.6; between the clusters cos 0.1–0.3. The model's own plane overlaps the activation cluster (0.17–0.28) more than the QK
  cluster (0.10–0.17), and every non-working plane sits at the random floor (0.02).

## Z2 — Strength sweep (`zoo4_sweep.png`; β 1–8 for the five best grid-agnostic planes + references)

**Every port has the same profile: control rises to a peak at β 4–6 and then collapses, while late-context accuracy and the
self-copy side effect degrade monotonically.** At β6 the all-L26 form reaches override +0.21 and unseen +0.28 (the largest
control seen at hs 26), but late accuracy is −0.21 and the model puts +0.30 extra mass on repeating the current node — the
isotropic kernel (self > neighbor) is what limits strength, as in Part III (the adjacency kernel fixes it, Z5). The calibrated
own plane is the earliest to break (its fitted scale is 1.45× the reference norm, so "β2" is already ≈ β3 of the ports): at
β4 its late accuracy is −0.11 and by β6 its override is gone.

## Z3 — Round 2: the cone, not the plane (`r2a_spectrum.png`, `r2b_layer.png`, `r2c_own_in_cone.png`, `r2d_dims.png`)

Eigen-decompose the all-late-heads form S (generalized, fast-RoPE penalized); 1982 of 3584 eigenvalues are positive, the top
six are 4.4–5.1.

**a. Spectrum walk.** Planes taken at eigen-ranks (1,2), (3,4), (5,6), (9,10), (17,18) — five mutually (Gn-)orthogonal planes —
*all* give full-strength control: override +0.18 to +0.22, unseen +0.16 to +0.19, early uptake +0.07 to +0.09 (the rank 17–18
plane is the best of all on uptake, +0.091). From rank 33 the effect halves (+0.10), by 65 it is a quarter, by 257 it is
+0.03, by 1025 it is gone. The bottom planes destroy learning (−0.14/−0.19) with no control. **The port is the top ≈ 20–30
directions of the late-head QK spectrum — a cone of dimension ~1 % of the residual stream, any 2-D slice of which works.**

**b. Where the native plane sits.** The energy of each context's own plane inside the top-k eigen-directions of S is 4 % (k=2),
20 % (k=8), **44 % (k=16)**, 54 % (k=32), 57 % (k=64), 67 % (k=512); a random plane has 0 %, 0 %, 0.4 %, 1 %, 2 %, 14 %. So
**about half of the model's own position plane lies inside the top-1 % QK cone**, although its overlap with any particular
2-D port plane is small (cos 0.10–0.17). This is the answer to "orthogonal or overlapping": the native plane and the QK ports
are different 2-D slices of the same low-dimensional cone. (Under the 4-reader form the native plane's energy is lower — 11 %
at k=16, 33 % at k=64 — the reader-specific form is narrower than the all-heads one.)

**c. Injection layer.** For the fixed all-late plane, control grows monotonically with the depth of injection: override 0.00
(hs 8), +0.01 (12), +0.02 (16), +0.07 (20), +0.18 (26), **+0.27 (30), +0.30 (34)**; unseen +0.02 → +0.26/+0.24. The
self-copy side effect is *smaller* at hs 34 (+0.08 vs +0.16 at 26) and early uptake is best at hs 20–30 (+0.08). Injecting at
several layers at once is worse than the best single layer (late accuracy −0.14 to −0.26). Early injection (hs 8–16) gives
small but real uptake gains (+0.04) with essentially no override — the mid-layer heads use the plane to learn faster but the
readout is not moved by it.

**d. Dimensionality.** Spreading the same coordinates over two or three orthogonal cone planes (ranks 1–2 + 3–4, or 1–6)
changes nothing relative to one plane (+0.20 vs +0.18); three coordinates on the top three directions is slightly worse.

## Z4 — Round 3: dissecting the cone (`r3a_own_split.png`, `r3b_bands_act.png`, `r3_summary.txt`)

**a. The native plane works *because of* its component inside the cone.** Split each context's fitted own-plane features
F_own into the part inside the top-k eigen-directions of S_all and the remainder (F_in + F_out = F_own, same fitted scale,
β2), 15 contexts:

| k | share of energy in cone | own plane | in-cone part only | out-of-cone part only | in-cone part rescaled to full norm |
|---|---|---|---|---|---|
| 16 | 47 % | override +0.071 / unseen +0.084 / uptake +0.073 | +0.053 / +0.064 / +0.076 | +0.027 / +0.044 / +0.022 | **+0.114 / +0.111 / +0.085** |
| 64 | 61 % | same | +0.066 / +0.079 / +0.077 | +0.009 / +0.025 / +0.015 | +0.108 / +0.111 / +0.082 |
| 256 | 67 % | same | +0.069 / +0.087 / +0.078 | +0.003 / +0.014 / +0.011 | +0.105 / +0.114 / +0.083 |

The in-cone part alone reproduces the whole plane's learning speed-up and ≈ 90 % of its causal control; the out-of-cone
part (a third to a half of the plane's energy) is causally inert. Rescaling the in-cone part to the full norm gives a plane
that is *better* than the model's own calibrated one (override +0.11 vs +0.07, unseen +0.11 vs +0.08) with no loss of late
accuracy. In other words, the calibrated crutch of Part I was a diluted port: the useful half of it is the half that lies in
the attention weights' shared positive cone, and the rest is either fit noise or a representation that the readout does not
consult.

**b. The cone is a global property of the attention weights, not of particular layers.** Top-2 planes of the QK form summed
over different head sets, all injected at hs 26 (β4): all heads of L12–25 (+0.174 override — these heads cannot even *see*
an hs-26 injection; their form still points into the same cone, cos 0.61 with the all-late plane), L26–33 (+0.221), L34–41
(+0.195), L12–41 (+0.189), **all 672 heads of the model (+0.229, the strongest at hs 26)**; unseen +0.16 to +0.19 for all.
The vocabulary-restricted forms reach the full port at 2048–3072 PCs (+0.172 / +0.187).

**c. A handful of foreign activation PCs suffice.** The QK-best plane inside the top-k activation PCs of one 1200-token walk
on a *different* word set gives override +0.121 (k=8), +0.130 (16), +0.149 (32), +0.153 (64), +0.172 (128); unseen +0.08 →
+0.16. It is not specific to that word set (word set 1: +0.166 / +0.156) nor to the form used to pick it (S_all: +0.151).
So the residual stream's own top-8 activation directions at hs 26 already contain a slice of the cone that carries 70 % of
the port's control — the cone is where the model itself puts its high-variance late-layer activity.

## Z5 — Round 4: the late-layer port (`r4a_layer_fine.png`, `r4b_strength_kernel.png`, `r4_summary.txt`)

**a. Injection depth.** The all-late plane (β4) injected at hs 28 / 30 / 32 / 34 gives override +0.30 / +0.27 / +0.30 / +0.30 and
unseen +0.30 / +0.26 / +0.26 / +0.24 — 1.6–1.8× the same plane at hs 26 and 2× the Part III port. Control collapses beyond hs 34
(+0.13 at 36, +0.04 at 38, +0.02 at 40): the readers live in layers 26–35. Planes built only from the heads that can read an
injection at layer h (all heads of layers ≥ 30 / ≥ 34) are the same plane (cos 0.99 / 0.97 with the all-late plane) and give the
same numbers. Best single-layer plane setting: **hs 32, β3** — override +0.30, unseen +0.24, early uptake +0.10, late accuracy
unchanged (+0.004), self-copy +0.07. Larger β keeps raising override (+0.35 at β8) but costs late accuracy (−0.20). The rank
3–4 and 9–10 cone planes at hs 34 work identically to rank 1–2 (+0.29 / +0.28) — the cone property holds at the late layer too.
A random plane at hs 34 is null on every readout (override −0.000, unseen +0.000).

**b. The adjacency kernel at hs 34 is a different regime.** Building the node vectors so that F S F^T reproduces the grid
adjacency (self ≈ 0, neighbor > 0, non-neighbor ≈ 0; 12 cone directions instead of 2; Part III Q4 recipe, now with the
all-late form) and injecting at hs 34:

| condition | early uptake | late accuracy (400–1200) | override | unseen X | claim-Y | self-copy |
|---|---|---|---|---|---|---|
| adjacency @26 β4 | −0.008 | −0.048 | +0.063 | +0.084 | +0.134 | +0.065 |
| adjacency @30 β4 | +0.138 | +0.078 | +0.247 | +0.185 | +0.186 | −0.011 |
| adjacency @34 β4 | +0.218 | **+0.126** | **+0.482** | **+0.312** | **+0.300** | −0.032 |
| adjacency @34 β6 | **+0.235** | **+0.133** | **+0.642** | **+0.407** | **+0.373** | −0.036 |
| plane @32 β3 (best plane) | +0.100 | +0.004 | +0.303 | +0.244 | +0.180 | +0.073 |
| 4-reader port @26 β4 (Part III) | +0.060 | −0.039 | +0.168 | +0.149 | +0.175 | +0.110 |
| own plane @26 β2 | +0.073 | +0.002 | +0.071 | +0.084 | +0.109 | +0.046 |

(all p ≤ 1e-10; 15 contexts, 45 unseen items.) With the self term removed there is no self-copy side effect, so the injection
can be made strong without the late-accuracy collapse that limited every plane port: at β6 the model puts +0.64 more mass on the
*injected* map's neighbors than chance — it follows the channel almost completely over 600 tokens of contradicting evidence —
places a never-seen node with ΔP +0.41, follows a fake coordinate with +0.37, learns 2.5× faster than any earlier channel
(+0.24 at tokens 100–300), and is *more* accurate late in context (+0.13) than the un-injected model. This is the
"arbitrarily well, grid-agnostic" port the user asked for: nothing in it depends on the word set, the assignment, or a
calibration walk — only on the attention weights of layers 26–41 and the grid's adjacency matrix.

## Z6 — Round 5: the adjacency-kernel port, layer × strength, and the unseen edge (`r5a_adj_grid.png`, `r5b_unseen_edge.png`, `r5_summary.txt`, `edge5_summary.txt`)

**a. Layer × strength grid** (adjacency kernel from the all-late form; 15 contexts; all p ≤ 1e-6):

| hs \ β | 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| 30 | ov +0.25 / un +0.19 / late +0.08 | +0.41 / +0.29 / +0.09 | +0.56 / +0.37 / +0.09 | +0.59 / +0.37 / +0.04 |
| **32** | +0.53 / +0.34 / +0.13 | **+0.71 / +0.46 / +0.16** | +0.71 / +0.44 / +0.14 | +0.66 / +0.36 / +0.09 |
| 34 | +0.48 / +0.31 / +0.13 | +0.64 / +0.41 / +0.13 | +0.64 / +0.37 / +0.11 | +0.60 / +0.28 / +0.08 |
| 36 | +0.20 / +0.16 / +0.08 | +0.28 / +0.23 / +0.09 | +0.28 / +0.20 / +0.07 | +0.25 / +0.15 / +0.04 |

(ov = override, un = unseen-X, late = Δ accuracy at tokens 400–1200; early uptake is +0.14 to +0.27 everywhere in the grid and
the self-copy side effect is ≤ 0 everywhere.) The optimum is **hs 32, β6: override +0.71, unseen +0.46, fake-coordinate
+0.44, learning speed-up +0.27, late accuracy +0.16** — the un-injected model's late accuracy is ≈ 0.75, so the port lifts it
to ≈ 0.91 while making it follow whatever map is written into it. For comparison the best plane port (hs 32, β3) reaches
+0.30 / +0.24 / +0.00, the Part III port +0.17 / +0.15 / −0.04 and the calibrated own plane +0.07 / +0.08 / +0.00. The kernel
is saturating by β6–8 and declines at β10; hs 36 is already half as effective, consistent with the readers living in
layers 26–35.

**b. The unseen edge, revisited (30 items = 15 contexts × 2 directions; `r5b_unseen_edge.png`).** Part III Q7 found that no
channel at hs 26 could make the model predict a true grid edge that the walk never traversed (P(Y|X) = 0.006, below the
dist-2 non-neighbors at 0.023). The late adjacency port can:

| condition | P(held-out Y after X) | P(a seen neighbor) | P(a dist-2 non-neighbor) | held-out − dist-2, paired Δ vs none | accuracy (tokens ≥ 300) |
|---|---|---|---|---|---|
| no injection | 0.006 | 0.407 | 0.023 | — | 0.802 |
| adjacency port @34 β6, edge **in** kernel | **0.088** | 0.440 | 0.005 | **+0.101** (p = 7e-4) | 0.921 |
| adjacency port @34 β6, edge **deleted** from kernel | 0.000 | 0.464 | 0.011 | +0.006 (p = 0.1) | 0.903 |
| adjacency port @34 β4, edge in / deleted | 0.035 / 0.000 | 0.46 / 0.46 | 0.008 / 0.015 | +0.045 / +0.002 | 0.92 / 0.89 |
| plane port @32 β3 | 0.065 | 0.351 | 0.015 | +0.068 (p = 0.006) | 0.804 |
| 4-reader port @26 β4 (Part III) | 0.012 | 0.386 | 0.017 | +0.012 | 0.761 |
| own plane @26 β2 | 0.011 | 0.393 | 0.023 | +0.005 | 0.801 |

With the edge in the kernel the never-traversed neighbor goes from 1 % to 20 % of a seen neighbor's probability (15× the
baseline, and 18× the non-neighbor level); with the same port but the edge removed from the kernel it is exactly 0 — the
channel decides whether a never-observed edge exists, while the seen edges and the overall accuracy are untouched (0.92 in
both). The Part III conclusion "the port is a minority voice that loses against intact induction evidence" was a statement
about hs 26 and β4; at hs 32–34 with the self term removed, the channel is a majority voice.

## Part IV headline

**There is no second port. Every plane that accepts injected coordinates is a slice of one low-dimensional cone — the top ≈ 20–30
eigen-directions of the attention weights' shared, RoPE-slow query–key form — and the model's own in-context position plane is
about half inside that cone, which is the half that does the work.** Token-embedding directions (PCs of the 16 words, or of the
whole vocabulary at any rank) are outside the cone and do nothing or hurt; the cone is the same whichever layers' heads define
it (L12–25, L26–33, L34–41, or all 672 heads), and the residual stream's own top-8 activation directions at hs 26 already
contain a usable slice of it. The port that results from taking this seriously — the grid adjacency written as an attention
kernel (self ≈ 0, neighbor > 0) into the cone at hs 32, β6, no calibration, no reader selection, nothing depending on the
words — makes the model follow an injected wrong map almost completely (override +0.71), place a never-seen node (+0.46),
follow a fake coordinate (+0.44), learn 2.5× faster early (+0.27), become *more* accurate late in context (+0.16), and supply
a never-traversed edge (P 0.006 → 0.088, exactly 0 when the edge is removed from the kernel). Relative to the Part III port,
that is 4× the override and 3× the unseen-node control, with the side effects (self-copy, late-accuracy loss) removed.

What the zoo rules out: any grid-agnostic subspace defined by *token* geometry (embedding PCs), any generic high-gain weight
direction (MLP-in, attention-out singular vectors), and any plane with a QK score near zero. What it leaves open: whether the
cone is the same across models of a family (only gemma-2-9b was tested here), and whether other in-context structures (rings,
trees, other graph sizes) map onto the same cone with a different kernel.

# Part V — Larger grids: 25, 64, 144, 216 nodes (2026-09-03, pod `icl-big-grid`)

Setup (`big_grid.py`, `plot_big.py`; `results_big/big.json`, `figures_big/`). Grids 4×4, 5×5, 8×8, 12×12 and 12×18 (216 nodes as a
rectangle so the two spectral coordinates stay the top pair); context 1200 / 2000 / 4000 / 7000 / 8000 tokens; words are
single-token common English words sampled fresh per context from a 19k-word pool; 6 contexts per grid, paired. Strategies: the
context's own plane (calibrated on a separate walk of the same words, hs 26, β2), a random plane (hs 32, β4), the all-late QK
plane port (hs 32, β3 fixed and β matched), the adjacency-kernel port (hs 32, β6 fixed and β matched; the kernel is the grid
adjacency truncated to its 12 largest-positive and 12 most-negative eigenmodes, so for 4×4 it is the Part IV construction
exactly). "Matched" β keeps the implied neighbor-vs-distance-2 attention-logit contrast equal to its 4×4 value.

**Baseline.** Rule-following mass in the last two thirds of context: 0.77, 0.75, 0.73, 0.40, 0.29 for 16 → 216 nodes. The low
numbers for 144 and 216 nodes are a **context-window artifact, not a learning failure**: every curve (injected or not) drops by
≈ 0.4 at exactly 4096 tokens (`big1_curves.png`), which is the sliding window of Gemma-2's local attention layers. Before the
cliff (tokens T/3 to 4096) the baseline reaches 0.66 (144 nodes) and 0.61 (216 nodes) — the large grids *are* learned; after
it (tokens 4200 to T) the baseline sits at 0.24 / 0.17. The adjacency port gives 0.78 / 0.72 before the cliff and 0.52 / 0.45
after it, i.e. it roughly doubles what survives the window. The "late" window in the table below spans both regimes.

**Results** (paired Δ vs no injection; 6 contexts; unseen items 12; `big_summary.txt`):

| nodes | strategy | early Δ | late Δ | override | unseen X | fake coord |
|---|---|---|---|---|---|---|
| 16 | own plane | +0.04 | −0.03 | +0.07 | +0.01 | +0.02 |
| | random plane | −0.02 | −0.01 | 0.00 | +0.01 | −0.02 |
| | QK plane port β3 | +0.08 | −0.01 | +0.24 | +0.19 | +0.09 |
| | adjacency port β6 | **+0.23** | **+0.11** | **+0.66** | **+0.40** | **+0.36** |
| 25 | own / random / plane β3 / adjacency β6 | +0.04 / −0.03 / +0.06 / **+0.21** | −0.03 / −0.01 / −0.03 / **+0.09** | +0.06 / 0.00 / +0.21 / **+0.64** | +0.07 / 0.00 / +0.20 / **+0.37** | +0.04 / −0.01 / +0.15 / **+0.25** |
| 64 | own / random / plane β3 / adjacency β6 | +0.03 / −0.05 / +0.03 / **+0.25** | −0.02 / −0.03 / −0.03 / **+0.12** | +0.04 / 0.00 / +0.13 / **+0.68** | +0.01 / −0.01 / +0.09 / **+0.43** | +0.01 / −0.01 / +0.04 / **+0.43** |
| 144 | own / random / plane β3 / adjacency β6 | −0.01 / −0.05 / −0.01 / **+0.22** | +0.02 / −0.08 / −0.02 / **+0.21** | +0.01 / 0.00 / +0.06 / **+0.55** | 0.00 / 0.00 / +0.01 / **+0.26** | 0.00 / 0.00 / +0.02 / **+0.28** |
| 216 | own / random / plane β3 / adjacency β6 | 0.00 / −0.06 / −0.02 / **+0.19** | +0.02 / −0.07 / −0.01 / **+0.23** | 0.00 / 0.00 / +0.04 / **+0.47** | 0.00 / 0.00 / +0.02 / **+0.16** | 0.00 / 0.00 / +0.01 / **+0.25** |

(adjacency-port effects p ≤ 3e-3 at every size, most ≤ 1e-6; matched-β adjacency is within 0.01 of fixed β6 everywhere because
the truncated kernel's neighbor contrast barely changes with N.)

Reading:
- **The adjacency-kernel port works at every grid size, and its late-accuracy gain grows with the grid**: +0.11 at 16 nodes,
  +0.21 to +0.23 at 144–216 nodes, where the baseline model has not learned the graph. Override stays at +0.47 to +0.68 and the
  unseen-node placement at +0.16 to +0.43. On the 216-node grid the injected model reaches 0.53 late accuracy vs 0.29 without.
- **The plane port decays with grid size** (override +0.24 → +0.04) and cannot be rescued by strength: its kernel gives self ≈
  neighbor ≈ distance-2 (8891 / 8443 / 7870 at 216 nodes, vs 8822 / 6060 / 2354 at 16), so matching the neighbor contrast needs
  β 7.6, which destroys accuracy (−0.33 / −0.24) without raising override (+0.03). Two coordinates cannot single out the four
  neighbors among 216 nodes; the 24-mode kernel can.
- **The own plane is weak at all sizes and null at 144–216.** The spectral-coordinate fit to the node means degrades from R² 0.32
  (16) to 0.06 (216); the top-2 PCs of the node means explain 33 % → 10 % of the variance. The model does not build a clean
  2-D spectral plane for large grids at these context lengths — consistent with the baseline not learning them — yet whatever
  plane it does build still has 50–74 % of its energy inside the top-32 QK cone at every size.
- **Random planes hurt at large N** (late −0.07 to −0.08 at 144–216) and never help.

Caveats: 6 contexts per grid; for 144 and 216 nodes the "late" window straddles the 4096-token sliding-window cliff, so its
absolute level mixes two regimes (pre-cliff numbers above are the cleaner statement); contexts longer than 4096 tokens are seen
in full only by Gemma-2's global layers; the truncated-mode kernel gives distance-3 nodes a positive entry at large N (1606 /
2302 vs neighbors 3597 / 3788 at 144 / 216), a limitation of a 24-direction kernel.

## Caveats
- Two models of one family (gemma-2-9b, gemma-2-2b), one graph (4×4); 6–12 seeds for dense curves, n=18–36 for unseen-node. Part II (port experiments) is gemma-2-9b only, 6 seeds, n=18.
- Injection at a single layer (hs 26), additive, with β ∈ {1,2,3}; multi-layer injection and larger β were not explored (β=3 already degrades late-context accuracy in some arms).
- E3 used 3 demos × 500 tokens; longer or more demos were not tried. The E3 null is at this scale.
- "Chance" for the override injected-map is the baseline scored under the permuted map (the two maps share a few edges), so injected-map gains are conservative.
- Part IV: one model (gemma-2-9b), 15 contexts (word set 4 × 6 assignments + word sets 1–3 × 3), 45 unseen-node items, 30 unseen-edge items; the cone's eigen-rank
  cutoff (≈ 20–30) was read from a coarse spectrum walk (ranks 1–2, 3–4, 5–6, 9–10, 17–18, 33–34, 65–66, …); the adjacency kernel uses 12 cone directions with
  the generalized eigen-normalization of Part III; "late accuracy +0.16" is relative to an un-injected baseline of ≈ 0.75 at tokens 400–1200; the strongest
  settings (β ≥ 8 at hs 32–34) begin to reduce late accuracy again.
