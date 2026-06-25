# Executive Summary — Othello-GPT

## The one-paragraph version

**Othello-GPT** is a GPT-2-style transformer trained by Li et al. (2022) to predict
*legal* next moves in the board game Othello, given a sequence of prior moves. It was
never told the rules or shown the board — only move sequences. Yet it spontaneously
learned to compute the **full board state** internally: an *emergent world model*. Li et
al. could recover the board from the residual stream with a **non-linear** probe (a 2-layer
MLP) but **not** a linear one, which suggested the representation was fundamentally
non-linear. Neel Nanda (2023) showed this conclusion was an artifact of the **basis**: if
you probe for "this cell is **mine** vs **theirs**" (relative to whoever is about to move)
rather than "black vs white" (absolute color), the board state **is** linearly decodable,
to high accuracy, by a simple linear probe — and you can **causally intervene** on it by
editing a single direction in the residual stream. This is one of the cleanest known
examples of an emergent, linearly-represented, causally-validated world model in a trained
transformer.

## Why anyone cares

- **Linear Representation Hypothesis evidence.** It's strong support for the idea that
  models represent features as *directions* in activation space — the foundational
  assumption that makes mechanistic interpretability tractable. Li's result briefly looked
  like a counterexample; Nanda's correction turned it into confirming evidence.
- **A "transformer circuit laboratory."** Othello-GPT is algorithmic enough to have a
  ground truth (you can compute the true board state externally and check the model against
  it), yet complex enough to exhibit real phenomena (superposition questions, modular
  circuits, non-trivial neuron behavior). It sits between toy models (modular addition) and
  real LLMs.
- **A controllable substrate for feature work.** Because the board state is a known,
  finite, externally-checkable set of features that the model *must* compute, it's an
  unusually clean place to study how features are represented, and whether/how they can be
  manipulated.

## The key facts at a glance (full detail in `02_facts_and_config.md`)

- **Model:** 8-layer GPT-2 architecture, d_model = 512, 8 heads, context length 60,
  vocab size 61. ~25M parameters. Trained on *synthetic* games (uniformly-random legal
  moves). Trained **with** attention + residual dropout (a quirk inherited from the
  minGPT/GPT-2 lineage — see gotchas; a from-scratch retrain should usually drop this).
- **Task:** input is the first 59 moves (`[0:-1]`), target is moves `[1:]` — standard
  next-move prediction. The objective is *legal* moves, not *good* moves.
- **The representation:** by ~layer 4, the board state is fully computed. It occupies
  ~124 of the 512 residual dimensions: ~64 "my-colour" directions (one per non-center cell)
  + ~60 "blank" directions (centers are never blank). Roughly 2 directions per cell.
- **The probe:** ~60 parallel 3-way (blank/black/white) logistic regressions reading one
  shared residual-stream vector. Linear. Reaches high-90s % accuracy on middle moves in the
  mine/theirs frame.
- **The crucial trick:** train/interpret the probe in the **mine/theirs** (player-relative)
  frame, which means accounting for the fact that the meaning flips every move (parity).
  Without this, a linear probe sees a sign-flipping target and fails — this is exactly why
  Li et al.'s linear probe looked broken.

## The mental model to carry

The model is "thinking" in terms of **my pieces vs their pieces**, not black vs white,
because *legality is invariant under flipping whose turn it is and flipping every piece's
color*. The task privileges the player-relative frame, so gradient descent builds the
representation in that frame. This is the single most important conceptual key to the whole
result, and the recurring lesson is Nanda's: the model is an *alien* whose natural concepts
aren't the ones a human would reach for first — find the model's frame and the structure
becomes simple and linear.

## What this dossier deliberately does NOT contain

No experimental design, no task scoping, no success thresholds-as-targets, no "do X then Y."
Those belong in the per-run brief. This is a knowledge base you draw on while working, not
a set of instructions. Where the literature states an empirical number (e.g. "linear probe
reaches high-90s on middle moves"), that appears here as a *fact / known-good anchor*, not
as a target you are being told to hit.
