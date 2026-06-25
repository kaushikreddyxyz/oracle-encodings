# Actually, Othello-GPT Has A Linear Emergent World Representation

**Author:** Neel Nanda · **Published:** Mar 28 (2023)
**Source:** https://www.neelnanda.io/mechanistic-interpretability/othello

> Local clean copy for the dossier. Site navigation chrome removed; substantive content
> preserved and lightly reformatted. Quotes of exact numbers/hyperparameters are faithful to
> the original. For figures, see the live post.

---

## Overview

- Context: a recent paper (Li et al., *Emergent World Representations*) trained a model to
  play legal moves in Othello by predicting the next move, and found it had spontaneously
  learned to compute the full board state — an **emergent world representation**.
  - This could be recovered by non-linear probes but *not* linear probes.
  - The representation could be causally intervened on to predictably change model outputs,
    so it's telling us something real.
- Nanda finds there's actually a **linear** representation of the board state — but rather
  than "this cell is black," it represents "this cell has **my** colour," since the model
  plays both black and white moves.
- One can causally intervene with the linear probe, and the model makes legal moves in the
  new board.
- This is evidence for the **linear representation hypothesis**: that models in general
  compute features and represent them linearly, as directions in space.

## Background (the two pieces of original evidence)

Li et al. trained Othello-GPT on random games (uniformly random legal next move) to predict
the next move. Headline: it learns to compute board state despite never being given it.

Two pieces of evidence in the original:
1. **Probes:** linear probes did *not* recover board state (error rate **20.4%**), while the
   simplest non-linear probe (a two-layer MLP with one hidden ReLU layer) worked extremely
   well (error rate **1.7%**). The non-linear probes did *not* work on a randomly initialised
   network and worked better on some layers than others — suggesting they learned something
   real.
2. **Causal interventions:** by using gradient descent so the probe *thinks* the residual
   stream represents a new board state, the model then makes legal moves in the *new* board —
   even moves illegal in the old board, even for board states unreachable by legal play.

Nanda's intuition: if you've really understood a model's internals, you should be able to
manipulate it with *simple* techniques (linear probes, linear interventions). The fact that
only non-linear probes worked, yet causal intervention succeeded, was genuinely ambiguous —
it could mean a real non-linear representation, or a linear representation of *simpler*
features from which the probe computes board state.

## The finding

Starting from activation patching and neuron inspection, Nanda noticed neurons firing every
other move with different parity each game (e.g. **neuron 1393 in layer 5** seemed to learn
`(D1==white) AND (E2==black)` on odd moves and the flip on even moves). Generalising: the
model learns a linear representation, but in terms of **my vs their** colour, not black vs
white. This makes sense from the model's perspective — it plays both colours, and valid
moves for black become valid for white if you flip every piece.

> If you train a linear probe on just odd / just even moves (i.e. with black / white to
> play) it gets near-perfect accuracy, and transfers reasonably to the other moves if you
> flip its output.

Speculation: the original non-linear probe just learned to compute
`XOR("I am playing white", "this square has my colour")` to recover absolute colour.
Without the insight to flip every other representation, this is a **pathological example for
linear probes — the representation flips positive to negative every time, so it's impossible
to recover the true linear structure**.

Causal intervention: simply **negating the coordinate in the probe direction** for a square
(residual stream after layer 4, no further intervention) just worked.

## Probing — technical setup (the exact recipe)

- Model: the **synthetic** model from the paper. An **8-layer GPT-2**, trained on a synthetic
  dataset to predict the next move. Games are length 60; it receives the first 59 moves
  (`[0:-1]`) and predicts the final 59 (`[1:]`). Trained with **attention dropout and
  residual dropout**.
- **Vocab size 61** — one per square (1–60), minus the four centre squares (filled at start,
  unplayable), plus a special token (0) for passing.
- Trained the probe on **four million synthetic games** (far fewer would suffice).
- Trained **separate probes on even, odd, and all moves**.
- Trained only on moves **`[5:-5]`** because the model does weirder things on early/late
  moves (e.g. the residual stream on the first move has **~20× the norm** of every other).
- Trained to minimise cross-entropy for predicting empty/black/white, using **AdamW** with
  **lr=1e-4, weight_decay=1e-2, eps=1e-8, betas=(0.9, 0.99)**.
- Trained on the residual stream **after layer 6** — `get_act_name("resid_post", 6)`. (In
  hindsight, board state is fully computed by layer 4.)

For each square the probe has 3 directions (blank/black/white). Convert to two directions:
- `my_probe = black_dir - white_dir` (for black to play)
- `blank_probe = blank_dir - 0.5*black_dir - 0.5*white_dir` (the 0.5 isn't principled but
  worked)
- Discard the third dimension (softmax is translation-invariant); normalise to unit vectors
  (norm only affects probe confidence, not accuracy). Done for the black-to-play probe.

## Results

- The probe works great at layer 6. Odd (black to play) transfers fairly well zero-shot to
  even (white to play) by swapping mine/theirs (worse on corners). Accuracy taken over 100
  games (~5000 moves), scored on the middle band.
- Flipping either probe transfers well to the other side; odd and even probes are nearly
  negations of each other.
- Transfers zero-shot to other layers — great at layer 4 too, worse at layer 3 / layer 7.

## Intervening

- Take the residual stream after layer 4 (or 3), take the coordinate projecting onto
  `my_probe`, negate it and multiply by hyperparameter `scale` (0–16).
- First experiment (layer 4, scale 1) worked well. `scale` matters. Some cases worked better
  intervening at layer 3 — evidence processing spreads across adjacent layers.
- Multi-cell edits (flip F5 *and* F6) kinda work but are weaker/jankier.
- Edits don't perfectly recover logit magnitudes. Partly because the model was trained with
  dropout → built-in redundancy → editing is messy and unpredictable.

## Key takeaways (selected)

- **Models learn decomposable, linear representations** — moderate evidence for the linear
  representation hypothesis.
- **Mech interp == alien neuroscience:** the model was alien (think my/their, not
  black/white); once you find its frame, it becomes interpretable. (Cf. modular addition
  making sense in Fourier terms.)
- **Probing is surprisingly legit** here — naive logistic regression worked, even with
  imbalanced class labels.
- **Dropout ⇒ redundancy:** Othello-GPT was trained with attention + residual dropout
  (inherited from minGPT/GPT-2), giving backup-circuit-style redundancy. Nanda's
  recommendation for future work: **train a new model from scratch with no dropout — it "will
  make your life much easier."** He speculates the model is bigger than needed and things
  might be cleaner with fewer layers and a wider (or narrower) stream.
- **Over-parametrization caveat:** "Othello-GPT is likely over-parametrised for good
  performance on this task while language models are always under-parametrised." Plus there's
  a ground-truth solution — so generalising to LLMs is speculative.

## Dimension budget (modular-circuits section)

> By about **layer 4**, of the 512 residual dimensions, we have **64 directions** for which
> cell has "my colour" and **60 directions** for which cells are blank (the 4 centre cells are
> never blank). Taking up **2 dimensions per square consumes 128 of 512** residual dimensions
> — a major investment (~25%).

### The blank/unembed entanglement (a confusing case study)

Looking at top layer-4 neurons in the blank-probe basis initially looked like clean
"single cell is blank" neurons — but that's surprising, since blankness is easy (a cell is
blank iff never played). What's really going on: **the blank probe is highly correlated with
the unembed** — a cell can be legal only if blank, so a high end-of-model logit ⇒ probably
blank. The late-layer (layer 6) probe absorbs this. **Cosine sim ~0.8–0.9** to the unembed.
Two readings: contamination from probing late, or *intentional* alignment because the model
uses the is_blank subspace to contribute to the relevant unembed.

## Selected open problems / future work (abbreviated)

- Finding the *right* probe directions robustly (cosine 0.7 to "true" may still work;
  separating genuine overlap from superposition interference; ideas: high weight decay, SGD,
  amnesiac probing, orthogonalised iterative probes, using neurons as a privileged-ish basis).
- How is the **blank** world model computed? (Should be a single attention head per cell;
  what's the efficient version?) What's a principled blank-vs-filled direction?
- How is the **my vs their** world model computed? (Where the real meat is.)
- Superposition tests (binary features, residual-stream vs neuron superposition; best done in
  a **no-dropout, possibly smaller** model with stronger superposition incentive).
- Memory management / signal boosting in the residual stream; privileged-basis questions;
  whether heads/neurons are the right units; modular vs integrated circuits.

## Cleaning-up suggestions Nanda lists (relevant to a from-scratch redo)

- Train the probe on both colours at once (predict my/their, flip state every other move).
- Re-check whether cutting first/last 5 moves matters.
- Cells are correlated (corner needs a filled neighbour) so probes may be non-orthogonal for
  boring reasons — does constraining orthogonality help?
- What's the right probe layer? What's the most principled blank vs mine/theirs split?
- **Retrain with no dropout** (someone was working on this) — narrower/wider stream, fewer
  layers may be cleaner.

---

*Local copy ends. See the live post for figures, the full research-process narrative, and the
extensive open-problems brainstorm.*
