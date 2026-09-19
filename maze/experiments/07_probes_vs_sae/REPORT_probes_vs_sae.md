# Probes vs injection, re-tested at the storage site; SAE-feature steering (2026-09-05)

**Question.** Hunch #4 (`04_probe_steering`) found that steering the baseline maze GPT along linear-probe directions was a
total behavioral null (blocked edge still crossed 0.913 vs floor 0.914; detour wall used 0.005), while the trained injection
channel gave 0.07–0.14 / 0.77–0.83. Spies et al. (arXiv 2412.11867, *Transformers Use Causal World Models in Maze-Solving
Tasks*) steer maze transformers by toggling per-connection SAE features at the `;` tokens of the adjacency list (resid_post of
block 0) and report that linear probes had not worked for them. The user doubts the 150× gap. This experiment retrains the
probes at the place where the world model is actually stored and re-tests, then adds SAE features.

Model: our baseline jirpy-replica (d 256, 12 layers, 16 heads, learned absolute positions), 2M DFS 6×6 mazes; the same
BFS-verified counterfactual edit set as before (600 fresh mazes, seed 1234, 4,522 edits: 1,568 detour [block a traversed
path edge + open a repair wall], 1,154 shortcut-open, 1,200 off-path block controls, 600 zero-gain open controls). Baseline
floor reproduced on the pod: blocked crossed 0.908, detour wall used 0.004, shortcut used 0.013, reach 0.975.

## 1. Why the old probe steering could not work

The old probes were fit on one activation per maze, at `<PATH_START>` (layers 0–11, best L6 at 0.891 accuracy), and the
steering overwrote the probe coordinate at every position from `<ADJLIST_END>` onward. The adjacency list itself — where the
connection information lives and where the path tokens look it up — was never touched, so at every generation step the model
could re-read the true maze from the prompt. That is a read-out site, not the storage site.

**Where the storage site is in this model** (`wm_probes.py data/acts/train`; 3,000 fresh mazes, 105k connection chunks
`A <--> B ;`; "which of the 72 connections does this token encode", 72-way, validation accuracy, chance 1/60):

| site | L0 | L1 | L2 | L3 | L4 | L6 | L9 |
|---|---|---|---|---|---|---|---|
| A token | 0.40 | 0.41 | 0.42 | 0.43 | 0.44 | 0.45 | 0.46 |
| `<-->` token | 0.47 | 0.49 | 0.50 | 0.52 | 0.53 | 0.54 | 0.54 |
| **B token** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `;` token | 0.12 | 0.14 | 0.14 | 0.14 | 0.14 | 0.15 | 0.15 |
| `<PATH_START>` (72 binary probes, mean) | 0.79 | 0.81 | 0.86 | 0.89 | 0.89 | 0.87 | 0.82 |

Unlike the paper's models, the semicolon carries almost nothing here; the connection is complete at the second coordinate
token from layer 0 on. Attention from path tokens confirms it: averaged over heads, 20–30 % of the mass goes to the A and B
coordinate tokens at every layer and 10–20 % to the semicolons, with many heads putting 40–80 % of their attention on the
A or B tokens (e.g. L2H14 B 0.53, L3H14 A 0.68, L5H0 A 0.79, L11H2 B 0.72). So the intervention site for this model is the
coordinate tokens of the chunk, and an *added* connection has to be written somewhere the readers look — we use the
semicolons (nearly empty slots) or the B tokens.

## 2. Interventions (all at prompt positions, so they persist through the rollout)

- **prompt** — token oracle: insert `A <--> B ;` before `<ADJLIST_END>` or delete the four tokens of the chunk.
- **copy_chunk** — remove only: overwrite the four residuals of f's chunk with those of another chunk of the same maze
  (activation patch; deletes f without inventing anything).
- **mean_patch** — remove: A and B residuals of f's chunk ← generic class means (mean over chunks of other connections);
  add: write the full class-mean vector of connection f at the B site into the semicolon positions (all of them, or only the
  last one) or into all B tokens.
- **dir_dim / dir_logit** — the probe-direction protocol: the coordinate along the unit diff-in-means / logistic direction
  is set to mid + α·(class mean − mid), at the same positions as mean_patch.
- **old_style** — the hunch-#4 protocol exactly (L6 `<PATH_START>` probes, from `<ADJLIST_END>` onward, α 2 and 4).
- **SAE** — ReLU SAE (expansion 8, L1 sweep) on resid_post of block 0 over all tokens; per-connection latent chosen by F1
  at the B site; add: latent set to its typical value at all semicolons / all B tokens; remove: latent zeroed at f's A and B
  tokens; also a reconstruction-patched variant (the paper's implementation).

Metrics per edit family (paired, same edits for every condition; 400 edits per family in the sweep): detour — blocked
edge still crossed (↓), repair wall used (↑), solves the edited maze optimally (↑), reached target; shortcut — opened
shortcut used (↑), solves edited (↑); off-path block — blocked edge crossed (specificity; should stay ≈ 0.04) and reach;
zero-gain open — opened wall used (should stay ≈ 0).

## 3. Results — layer 0 (first sweep; the other layers, the full set and the SAE runs are appended below as they land)

| condition (L0 resid_post) | detour crossed ↓ | detour wall used ↑ | detour solves | shortcut used ↑ | shortcut solves | off-path crossed | off-path reach | zero-gain used |
|---|---|---|---|---|---|---|---|---|
| none | 0.932 | 0.000 | 0.000 | 0.018 | 0.013 | 0.045 | 0.980 | 0.007 |
| prompt (token oracle) | **0.030** | **0.855** | **0.713** | 0.052 | 0.000 | 0.000 | **0.007** | 0.058 |
| copy_chunk (remove) | 0.297 | 0.107 | 0.065 | 0.018 | 0.013 | 0.020 | 0.973 | 0.007 |
| mean_patch, add at all `;` | 0.138 | 0.752 | 0.445 | 0.662 | 0.438 | 0.020 | 0.978 | 0.265 |
| **mean_patch, add at last `;`** | **0.080** | **0.775** | **0.595** | **0.568** | **0.497** | 0.020 | 0.978 | 0.138 |
| dir_dim α1 / α2 (add at `;`) | 0.34 / 0.30 | 0.12 / 0.12 | 0.07 / 0.08 | 0.018 | 0.013 | 0.02 | 0.98 | 0.007 |
| dir_logit α1 / α2 (add at `;`) | 0.37 / 0.29 | 0.12 / 0.14 | 0.08 / 0.08 | 0.018 | 0.013 | 0.02 | 0.98 | 0.007 |
| *reference: fixed / trainable injection (full set, earlier)* | 0.141 / 0.069 | 0.773 / 0.831 | — | 0.610 / 0.681 | — | — | — | — |

Reading (layer 0):
- **The token oracle exposes the length problem.** Deleting a chunk and inserting one (detour) works almost perfectly at
  the token level (crossed 0.03, wall used 0.86), but any edit that changes the number of chunks breaks the model: an
  off-path deletion alone sends reach from 0.98 to 0.007, and an inserted 36th chunk is simply not used (shortcut 0.05).
  This is the paper's "Stan" failure (learned positional embeddings cannot cope with more connections than trained on) —
  so, as in the paper, residual-level interventions are the only way to *add* a connection to this model.
- **Full-vector patching at the storage site steers the model about as well as the trained injection channel.** Writing
  connection f's B-site class mean into the last semicolon and blanking f's chunk gives crossed 0.08, wall used 0.78,
  shortcut used 0.57, and 50–60 % of the edited mazes solved *optimally*, with off-path specificity intact (crossed 0.02,
  reach 0.98). The zero-gain control (0.14) shows the added edge is slightly over-attractive.
- **One-dimensional probe directions are what fails.** Removing along the probe direction gets a third of the way
  (crossed 0.30–0.37, from 0.93); adding along it does nothing (shortcut 0.018 = floor). The old protocol's null is
  therefore two things at once: the wrong site *and* a one-dimensional edit of a representation that needs the full vector.

### 3b. Token oracles and the old protocol (same 1,600-edit subsample)

| condition | detour crossed ↓ | wall used ↑ | solves | shortcut used ↑ | solves | off-path crossed | off-path reach | zero-gain used |
|---|---|---|---|---|---|---|---|---|
| old protocol (hunch #4 code path: L6 `<PATH_START>` probes, from `<ADJLIST_END>` on), α2 / α4 | 0.927 / 0.925 | 0.000 | 0.000 | 0.018 | 0.013 | 0.05 | 0.97 | 0.01 |
| token oracle, length changes (insert / delete chunk) | 0.030 | 0.855 | 0.713 | 0.052 | 0.000 | 0.000 | **0.007** | 0.058 |
| token oracle, fixed length (removed chunk → duplicate of another; added chunk → replaces the last chunk) | 0.048 | 0.792 | 0.490 | **0.713** | 0.570 | 0.005 | 0.960 | 0.290 |

The old protocol reproduces its null exactly. With the sequence length held fixed, the token-level ceiling for *adding* a
connection is ≈ 0.71 (shortcut used) — so the layer-0 class-mean patch (0.57–0.66) recovers most of what the model can do
at all, and the residual gap to the injection channel (0.61–0.68) is small. (The fixed-length "add" oracle deletes the last
chunk as a side effect, which is why its detour "solves" and reach are a little lower than the length-changing oracle's.)

### 3c. SAE features (block-0 resid_post, expansion 8; connection latents identified at the B site)

| SAE | L0 (avg active latents) | rollout agreement with reconstruction patched in | connections with a single latent of F1 > 0.9 | 72-way logistic on latents |
|---|---|---|---|---|
| L1 penalty 1e-2 | 243 | 0.968 | — | — |
| L1 penalty 3e-2 | 105 | 0.954 | — | — |
| L1 penalty 1e-1 | 38 | 0.922 | 4 / 60 (median F1 0.63) | 1.00 |
| TopK k = 16 | 16 | 0.932 | **19 / 60** (median F1 0.82) | 1.00 |
| TopK k = 32 | 32 | 0.938 | 12 / 60 (median F1 0.70) | 1.00 |

At the semicolon site no SAE has any connection feature (median F1 0.06; 72-way 0.11) — consistent with the probes: this
model does not consolidate connections into the `;` token. At the B site the connection is fully recoverable from the latent
vector (logistic 1.00) but mostly *not* as one latent: the code is compositional (cell-of-B features plus attended-A
features), the "Stan" pattern of the paper.

Single-latent toggling (paper protocol: set the connection's latent to its typical value at all semicolons or all B tokens;
zero it at the connection's own chunk), best settings:

| SAE, protocol | detour crossed ↓ | wall used ↑ | shortcut used ↑ | zero-gain used |
|---|---|---|---|---|
| L1 1e-1, add at all `;`, value ×2 | 0.743 | 0.115 | 0.155 | 0.033 |
| L1 1e-1, add at all B tokens, value ×1 | 0.825 | 0.117 | 0.220 | 0.092 |
| TopK 16, add at all `;` | 0.900 | 0.013 | 0.018 | 0.007 |
| TopK 16 / 32, add at all B tokens, value ×2 | 0.740 / 0.835 | 0.203 / 0.130 | 0.242 / 0.145 | 0.10 / 0.09 |

Single SAE latents are weak handles here (at best a fifth of the class-mean patch). Multi-latent toggling (the top-4
latents of the logistic decoder per connection, i.e. the compositional code) is reported below.

### 3d. Layer dependence (same 1,600-edit subsample; `fig3_layer_sweep.png`)

| intervention layer (resid_post) | class-mean patch (add at last `;`): crossed / wall used / shortcut used | copy_chunk: crossed | probe direction (best): crossed / wall used |
|---|---|---|---|
| 0 | **0.080 / 0.775 / 0.568** | 0.297 | 0.29 / 0.14 |
| 1 | 0.142 / 0.725 / 0.550 | 0.343 | 0.36 / 0.10 |
| 2 | 0.223 / 0.645 / 0.485 | 0.412 | 0.42 / 0.08 |
| 3 | 0.280 / 0.552 / 0.412 | 0.485 | 0.51 / 0.08 |
| 6 | 0.833 / 0.080 / 0.080 | 0.860 | 0.88 / 0.01 |

Control decays smoothly with depth and is gone by layer 6 — the readers that matter sit in layers 1–5 (consistent with the
attention survey: strong A/B-token heads at L1–L5, and the published finding that this architecture's world model is
decodable at layer 2). This is the third ingredient of the old null: layers 5–7 (and 3–9) were the only ones ever steered.

## 4. Full edit set (all 4,522 edits; directly comparable with the injection numbers)

| condition | detour crossed ↓ | wall used ↑ | detour solves | shortcut used ↑ | shortcut solves | off-path crossed | off-path reach | zero-gain used |
|---|---|---|---|---|---|---|---|---|
| baseline (no intervention) | 0.908 | 0.004 | 0.003 | 0.013 | 0.006 | 0.058 | 0.977 | 0.007 |
| old protocol (L6 `<PATH_START>` probe, α4) | 0.903 | 0.005 | 0.003 | 0.013 | 0.006 | 0.054 | 0.973 | 0.008 |
| probe direction at the chunk (diff-in-means, L0, α2) | 0.313 | 0.112 | 0.065 | 0.015 | 0.006 | 0.013 | 0.979 | 0.007 |
| copy another chunk over f's chunk (L0) | 0.300 | 0.117 | 0.071 | 0.013 | 0.006 | 0.012 | 0.983 | 0.007 |
| **class-mean patch, add at last `;` (L0)** | **0.098** | **0.746** | **0.577** | **0.572** | **0.489** | 0.017 | 0.978 | 0.140 |
| class-mean patch, add at all `;` (L0) | 0.143 | 0.747 | 0.452 | 0.644 | 0.406 | 0.017 | 0.978 | 0.248 |
| token oracle, fixed length | 0.045 | 0.763 | 0.503 | 0.708 | 0.567 | 0.005 | 0.975 | 0.298 |
| token oracle, length changes | 0.034 | 0.854 | 0.700 | 0.039 | 0.001 | 0.000 | 0.007 | 0.065 |
| *fixed injection (channel edit, earlier)* | 0.141 | 0.773 | — | 0.610 | — | — | — | — |
| *trainable injection (channel edit, earlier)* | 0.069 | 0.831 | — | 0.681 | — | — | — | — |

**Bottom line.** The 150× gap was an artifact of how the probes were used, not evidence that the baseline's world model
cannot be steered. Steering the *baseline* with nothing but activation statistics from 3,000 fresh mazes — no training of
the model, no channel — reaches the fixed injection channel on every control metric (crossed 0.10 vs 0.14; repair wall
used 0.75 vs 0.77; shortcut used 0.57 vs 0.61) and sits ≈ 0.07–0.09 below the trainable channel. What it took: (i) the
right site — the coordinate tokens of the adjacency chunk, which is where this model stores each connection and where
its path-token heads look (not `<PATH_START>`, not the semicolons the paper's models use); (ii) the right layer — 0–1
(control is gone by layer 6); (iii) the full class-mean vector rather than a single probe direction — a 1-D edit along
the best direction removes a connection only a third of the time and cannot add one at all.

Two caveats keep this from being "probes beat injection": the residual-level *add* is still a full-vector write (a
72-way class mean, i.e. 72 vectors per layer, learned from data), and it comes with a mild side effect (zero-gain walls
used 0.14 vs 0.007 — the written connection is slightly too attractive, as it also is for the token-level oracle, 0.30).

### 4b. A caution about "flooding": directions can be forced to act, but not faithfully

Writing a connection's probe direction (or class mean) into *every* B token of the list ("B_all") does make the model use
the new edge — shortcut used 0.68–0.74 at L0, as high as the token oracle — but it is not a world-model edit: the same
irrelevant edge is used 0.42–0.56 of the time in the zero-gain control, reach drops to 0.62–0.85, and only 5–17 % of the
edited mazes are solved optimally. The connection has simply been made attractive everywhere. The single-slot write (last
semicolon) is the faithful version: the same "used" rates, but 0.14 on the zero-gain control and 49–58 % optimal solutions.
When comparing methods, the paper-style efficacy (optimal solution of the edited maze) and the zero-gain control are the
numbers that separate steering from flooding; the figure's representative rows are chosen by that criterion.

## 5. SAE features, compositional code (top-4 latents of the logistic decoder per connection; 1,600-edit subsample)

Toggling the connection's *set* of latents (value ×2 = twice the typical activation; add at the last `;` or all `;`;
remove = zero them at the connection's own A and B tokens):

| SAE (block 0) | latents toggled | detour crossed ↓ | wall used ↑ | detour solves | shortcut used ↑ | shortcut solves | zero-gain used |
|---|---|---|---|---|---|---|---|
| L1 penalty 1e-1 (L0 = 38) | 1 (best F1) | 0.743 | 0.115 | 0.085 | 0.155 | 0.133 | 0.033 |
| L1 penalty 1e-1 | 2 | 0.175 | 0.620 | 0.477 | 0.410 | 0.365 | 0.075 |
| **L1 penalty 1e-1** | **4** | **0.128** | **0.677** | **0.515** | **0.435** | **0.390** | **0.077** |
| TopK 16 | 4 | 0.360 | 0.410 | 0.312 | 0.207 | 0.182 | 0.043 |
| TopK 32 | 4 | 0.647 | 0.182 | 0.145 | 0.120 | 0.090 | 0.022 |
| *class-mean patch (same subsample)* | full vector | 0.080 | 0.775 | 0.595 | 0.568 | 0.497 | 0.138 |

With four latents the SAE edit is a faithful world-model edit — 50 % of the detour mazes and 39 % of the shortcut mazes are
solved *optimally* under the edited maze, with a smaller zero-gain side effect than the class-mean patch (0.08 vs 0.14) —
at about 85 % of the class-mean patch's strength on the "used" metrics. Exactly the paper's picture for the learned-position
model: the connection is a *compositional* code (several latents), single-latent toggles do little, and the
compositional toggle steers. Writing the latents into every B token instead ("flooding") again raises "used" to 0.75 but
with zero-gain 0.58 and 5 % optimal solutions — not an edit.

## 6. Conclusions

1. **The old "injection beats probes by 150×" result is not evidence about probes.** It was a null produced by
   probing the read-out site (`<PATH_START>`), steering the wrong layers (5–9), and overwriting one direction. All three
   were necessary for the null: the same hunch-#4 code reproduces 0.90 crossed / 0.005 used on the full set today.
2. **The baseline's world model is steerable without any training, to the level of the fixed injection channel**, using
   activation statistics only: at layer 0, blank the connection's chunk with generic class means and write the connection's
   class-mean vector into one semicolon slot — 0.10 / 0.75 / 0.57 on the full 4,522-edit set vs 0.14 / 0.77 / 0.61 for the
   fixed channel and 0.07 / 0.83 / 0.68 for the trainable one, with off-path specificity intact and 49–58 % of the edited
   mazes solved optimally. Control decays with depth and is gone by layer 6.
3. **What still separates the channel from the probes is dimensionality, not site.** A single probe direction removes a
   connection only a third of the time and cannot add one without flooding every chunk (which over-steers). The channel's
   72 trained directions and the class-mean patch's 72 × d vectors are both *multi-dimensional* handles.
4. **SAE features behave as in the paper's learned-position model**: no per-connection semicolon feature (this model never
   consolidates into `;`), a compositional code at the coordinate tokens, single-latent toggles weak, and a 4-latent toggle
   that reaches ~85 % of the class-mean patch with the cleanest specificity profile of all residual-level methods.
5. **The token-level ceiling is lower than the residual-level one for adds**: this learned-position model cannot take an
   inserted 36th chunk (0.05 used) or a deleted chunk (reach 0.007) — the paper's "Stan" failure — while both residual-level
   methods add a connection at fixed length.

Artifacts: `scripts/wm_probes.py` (data/acts/train/steer; `--only`, `--full`), `scripts/wm_sae.py` (train/features/steer;
L1 or TopK; `--nlat`), `scripts/run*.sh` (pod drivers), `results/` (all JSON: `steer_L0/L1/L236/oracle/full.json`,
`sae/steer_L0_<sae>_B[_n2|_n4].json`, `probe_acc.json`, run logs), `figures/` (`fig1_probe_accuracy.png`,
`fig2_steering_comparison.png`, `fig3_layer_sweep.png`, `summary.txt`), `REPORT_probes_vs_sae.md`. Pod deleted after
pulling; models never left wandb/the stash.
