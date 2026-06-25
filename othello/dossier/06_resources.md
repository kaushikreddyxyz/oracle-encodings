# Resources — annotated by purpose

Each entry says what the resource **is for**, so you reach for the right one. Local copies of
the papers are in `papers/` (see `papers/README.md` if they need fetching).

---

## Primary sources (the canonical facts)

- **Nanda blog post — "Actually, Othello-GPT Has A Linear Emergent World Representation"**
  `papers/nanda_blog_othello.md` (saved locally) — also at
  https://www.neelnanda.io/mechanistic-interpretability/othello
  → **The densest single source.** The linear/mine-theirs result, probe setup, exact
  hyperparams, intervention method, the dimension budget, the blank/unembed entanglement, and
  a long "future work" section full of open problems. **This is the file to grep first** for
  most conceptual/method questions.

- **Nanda, Lee & Wattenberg (2023), "Emergent Linear Representations in World Models of
  Self-Supervised Sequence Models"** — arXiv **2309.00941** (BlackboxNLP/EMNLP).
  → The peer-reviewed version of the linear-representation result. Cite this for the formal
  claim. Companion probe/intervention code: Andrew Lee's `mech_int_othelloGPT` (below).

- **Li, Hopkins, Bau, Viégas, Pfister & Wattenberg (2022/2023), "Emergent World
  Representations: Exploring a Sequence Model Trained on a Synthetic Task"** — arXiv
  **2210.13382** (ICLR 2023 Oral).
  → The original Othello-GPT. Establishes the emergent world model and the (non-linear-probe)
  evidence. Source of the model, dataset, and training code. Gradient blog summary:
  https://thegradient.pub/othello/

---

## Code / tooling (by role)

- **`likenneth/othello_world`** — https://github.com/likenneth/othello_world
  → **The data + board engine.** Use it as the source of truth for: the synthetic game
  generator (uniform-random legal moves), the board-state engine (legal-move enumeration,
  board updates), and the original training code (minGPT-based). The mech-interp folder
  contains Nanda's probing scripts (`mechanistic_interpretability/tl_probing_v1.py`,
  `tl_initial_exploration.py`). **Lift the data/board utilities from here**; the training loop
  itself is notebook-based and built for an 8-GPU default (not a hard floor). Datasets
  (synthetic + championship) and pretrained checkpoints are linked from its README (Google
  Drive). MIT licensed.

- **TransformerLens** — https://github.com/TransformerLensOrg/TransformerLens (formerly
  `neelnanda-io/TransformerLens`)
  → **The analysis toolkit.** Activation caching, hooks, residual-stream surgery — what you
  want for probing, geometry, and interventions. **Mind the v3.0 break** (see `05_gotchas.md`
  G1 and `02_facts_and_config.md`). Othello demo notebook:
  `demos/Othello_GPT.ipynb` (Colab badge in-repo). Pretrained Othello weights on HuggingFace:
  **`NeelNanda/Othello-GPT-Transformer-Lens`**.

- **`ajyl/mech_int_othelloGPT`** (Andrew Lee) — https://github.com/ajyl/mech_int_othelloGPT
  → Probe/intervention code accompanying the 2309.00941 paper. Cross-reference for the formal
  version of the linear-probe and intervention experiments. **[UNVERIFIED current state]** —
  confirm it still builds against current deps.

---

## Adjacent / optional (came up as related, not core Othello)

- **Chess-GPT (Karvonen, 2024)** — linearly decodable chess board state *and* a steerable
  linear "skill"/Elo direction (a *global scalar* feature, contrast with Othello's *local
  categorical* cell features). Relevant if a global-scalar feature geometry is of interest.
  **[UNVERIFIED exact arXiv id]** — search "Karvonen Chess-GPT linear board".

- **Nanda et al. (2023), grokking / "Progress Measures for Grokking via Mechanistic
  Interpretability"** — arXiv **2301.05217**. The modular-addition Fourier-feature work.
  Conceptual sibling: linear features (Fourier directions there ↔ mine/theirs here) in clean
  algorithmic transformers. Useful framing for feature-direction work.

- **Toy Models of Superposition (Elhage et al., 2022)** — transformer-circuits.pub.
  Background for the superposition discussion in `03_concepts.md` §7. **Note:** it's an
  autoencoder studying packing, *not* a task-learning transformer — relevant as conceptual
  background, not as an Othello-style substrate.

---

## The bridge to the broader program (orientation, not instruction)

This Othello work sits in a line of "controllable linear-feature" substrates: modular
arithmetic (Fourier directions) → Othello-GPT (mine/theirs cell directions) → Chess-GPT
(board + skill). The common thread is *features represented as directions in a clean,
trainable-from-scratch transformer, with externally-checkable ground truth*. When something
in a task is underspecified, the sensible default is usually whatever keeps the linear
feature **clean, isolated, and causally checkable** — that's the property this whole lineage
is built around. (This is context for judgment, not a task definition; tasks live in the
per-run brief.)
