# In-context oracle injections on grid graph tracing — preliminary results

**Date**: 2026-08-12 · **Model**: gemma-2-9b (paper-validated in App. C.2 of 2501.00070; Llama-3.1-8B files are gate-blocked for this HF token at the file level, metadata passes) · **Hardware**: RunPod H100 SXM (`icl-grid-oracle`) · **Task**: 4×4 grid (16 single-token, semantically unrelated words), random-walk contexts, per paper 2501.00070v2.

Code: `grid_icl.py` (pod) + `plot_results.py` (local). Raw arrays: `results/replicate.npz`, `results/inject_fast.json`, `results/inject_phantom.json`. Figures: `figures/fig1..fig5`.

## 1. Replication — works

- Rule-following probability mass (sum of next-token prob over valid grid neighbors) climbs from ~0.2 to **0.74–0.79** over 1200 tokens of context (4 seeds). Fig 1.
- PCA of windowed mean node representations at the energy-minimizing hidden layer (h\*=40 of 42) shows the 4×4 lattice visibly assembling: distorted at 50–200 tokens, clean planar grid at 400–1200 tokens. Fig 2 — matches paper Fig 1.
- Normalized Dirichlet energy decreases with context, most strongly in late layers. Fig 3.

## 2. Injection A — extracted structure coords speed up in-context learning

Structure vectors = centered per-node mean reps at h\*, projected onto top-3 PCs, **extracted from one long (1200-token) run** and injected (forward hook, layer 39) at every node-token position of **different, short** walks. Rule-following prob at the query (8 seeds):

| condition | T=16 | T=32 | T=64 | T=128 |
|---|---|---|---|---|
| baseline (none) | 0.372 | 0.364 | 0.384 | 0.537 |
| **extracted, β=2, L39** | **0.537** | **0.559** | **0.597** | **0.648** |
| extracted, β=1, L39 | 0.480 | 0.486 | 0.539 | 0.636 |
| shuffled assignment (control), β=2, L39 | 0.240 | 0.359 | 0.445 | 0.487 |
| arbitrary random-plane coords, β=2, L39 | 0.363 | 0.363 | 0.381 | 0.567 |
| extracted, β=2, L8 (early layer) | 0.379 | 0.456 | 0.457 | 0.578 |

Dense curves (Fig 4, `results/fast_curve.json`): same 1200-token walks with/without injection, accuracy at every position. Tokens needed to reach accuracy X — baseline vs extracted β=2: 0.50: 99→29, 0.60: 185→50, 0.65: 303→67, 0.70: 476→146 (**3–4.5× context saving**). Curves converge at ~700–1000 tokens (head start, not a permanent boost). Shuffled caps at ~0.5 even at 1200 tokens; arbitrary tracks baseline exactly.

Reading (query-only table above):
- **Injected@16 tokens ≈ baseline@128 tokens** — roughly an 8× context-efficiency gain at these lengths.
- **Shuffled control hurts** at short contexts (0.24 vs 0.37 baseline) → the gain is node-assignment-specific, not a generic perturbation effect.
- **Arbitrary-subspace coordinates do nothing** → the model reads *its own* structure directions; you can't paint a world model into a random plane and have it used zero-shot (at these strengths/lengths).
- **Cross-context transfer worked**: the basis came from a different walk than the injected contexts → for a fixed word set, the in-context structure subspace is context-stable, so it's extract-once / inject-anywhere.
- Early-layer (L8) injection is much weaker than at the structure layer (L39).

## 3. Injection B — world-state edits at the query token

Long context (1200 tokens), current node *i*; *k* = far non-neighbor. Single-position hook at layer 39 (4 seeds):

| condition | P(true nbrs of i) | P(fake neighbor k) | P(nbrs of k) |
|---|---|---|---|
| null | 0.806 | **0.001** | 0.016 |
| move (replace top-2 PC coords with k's) | 0.406 | 0.072 | **0.227** (14×) |
| phantom, β=1 (add k's structure vec) | 0.626 | 0.031 | 0.131 |
| phantom, β=2 | 0.448 | **0.079** (66×) | 0.221 |

Reading (Fig 5):
- **"Move" replicates paper App. C.4 on the grid**: overwrite the query's PC coords with k's and the model predicts k's neighbors (0.016 → 0.227) at the expense of i's true neighbors.
- **"Phantom" is dose-dependent**: adding k's structure direction *on top of* i's raises P(k) 26–66× while true neighbors are partially retained (0.63 at β=1) — the model behaves as if an i–k edge exists.
- Caveat: phantom also raises P(nbrs of k), i.e. the edit partly smears position between i and k rather than purely grafting one edge. A cleaner "edge-direction" (rather than node-direction) injection is the obvious next step.

## Caveats (preliminary!)

- Single model (gemma-2-9b), single graph size (4×4), 4–8 seeds, no significance tests.
- h\*=40/42 is very late in the network; the paper used last layers for grids too, but injection-layer sweep was minimal (L8 vs L39).
- Memorization baselines in Fig 1 use the paper's iid-sampling formula, which is only approximate for random walks.
- Injection strength β and the top-k PC truncation (k=3) were barely tuned.

## Next steps

1. Sweep injection strength/layer properly; try injecting at multiple layers simultaneously.
2. Edge-level phantom: inject a *relative* direction (h̄_k − h̄_i within the PC plane) or attention-targeted edit to graft one edge without moving the node.
3. Test whether arbitrary-subspace coords become usable after a few in-context examples that "ground" them (curriculum in context).
4. Rerun on Llama-3.1-8B (needs an HF token with file-level access to meta-llama) and larger grids (transition-point shift prediction).
