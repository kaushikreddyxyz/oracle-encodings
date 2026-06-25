# Concepts — the "why" behind the facts

Conceptual explanations for the things in `02_facts_and_config.md`. Read this when you need
to *understand* something rather than look up a value.

---

## 1. The mine/theirs reframing (the central insight)

**Claim:** The board state is represented linearly, but in a **player-relative** basis
("does this cell hold *my* piece or *their* piece"), not an **absolute-color** basis
("black or white").

**Why the task forces this:** Othello-GPT only ever has to predict **legal** moves. Legality
is invariant under the symmetry "swap whose turn it is AND flip every piece's color" — a
position with black to move and its color-flipped twin with white to move have the *same*
set of legal moves, relative to the mover. So the variable that actually matters for the
task is mine-vs-theirs, and gradient descent builds the representation in the frame the task
privileges. Absolute color is *not* needed and is *not* what comes out linear.

**Consequence:** Black-to-play and white-to-play use the **same** physical direction in the
residual stream, with the **sign flipped**. "My piece" points one way; on the opponent's
move the very same cells are now "their" pieces, so the projection flips.

**The lesson (Nanda frames it as "mech interp == alien neuroscience"):** the model's natural
concepts are not a human's. Find the model's frame and the representation collapses from
"mysteriously non-linear" to "obviously linear." (Directly analogous to modular addition
making sense only once you think in Fourier terms.)

---

## 2. The parity sign-flip (why Li's linear probe "failed")

Because the mine/theirs direction's *meaning* inverts every move, a linear probe trained on
the **absolute black/white** labels across all moves is trying to fit a target whose sign
flips every step. Nanda: without flipping every other representation, this is a
**pathological case for linear probes — the representation flips positive to negative every
time, so the true linear structure is unrecoverable.** That's exactly what produced Li et
al.'s ~20.4% linear-probe error and the false conclusion of non-linearity.

**The fix:** condition on move parity. Train separate probes on odd vs even moves (i.e. on
"black to play" vs "white to play"), or equivalently train one probe on all moves after
relabeling board state into the mine/theirs frame (flip the labels every other move). In
the parity-correct frame the linear probe gets near-perfect accuracy, and the odd and even
probes turn out to be near-negations of each other — confirming it's one direction with a
flipping sign.

**Why the non-linear MLP probe "worked" for Li et al.:** Nanda's hypothesis — the MLP just
learned to compute `XOR("I am playing white", "this cell is my colour")` to recover absolute
color. The non-linearity was doing the parity bookkeeping that the basis change makes
unnecessary.

---

## 3. What counts as a genuinely linear feature

Important distinction to keep crisp (it matters for any feature-direction work):

- Saying a feature "lives in a linear subspace" is nearly **vacuous** — everything in an
  embedding space already lives in R^d.
- The meaningful claim is **semantic**: a feature is linearly represented if there's a
  direction such that *projecting onto it recovers the feature*, and ideally such that
  *moving along it causally changes the feature* in downstream computation. The mine/theirs
  result qualifies because both hold: linear decodability AND causal interventability (edit
  one direction → model plays as if the cell flipped).
- **Decodable ≠ used.** A probe can read out a feature the model computes only incidentally.
  The causal intervention is what upgrades "the probe found a correlate" to "this direction
  is part of the model's actual computation." Always keep the decode/use distinction in mind.

---

## 4. The per-cell geometry: ~2 directions, not 1

Each playable cell gets a **3-way** state (blank / black / white), which after discarding
the softmax-redundant dimension is **2 meaningful directions**:

1. **mine-vs-theirs axis** — a clean *signed scalar*. Project, read the sign: + = mine,
   − = theirs. This is the famous, genuinely-binary, task-load-bearing, causally-validated
   direction. When people say "the mine/theirs direction" (singular), this is it.
2. **blank-vs-filled axis** — occupied or empty. This is the *second* contrast, the one the
   "mine/theirs" slogan quietly drops. It is **messier** (see below).

**Cardinality intuition:** a truly binary feature needs 1 direction (sign = class). A
k-way categorical feature needs at most k−1 dims (3 classes → a 2-D plane, because argmax is
invariant to a shared offset). The cell is ternary, hence 2 dims; but the *task-relevant*
contrast (mine vs theirs) collapses to the clean 1-D binary case, which is why the result
feels like "a single direction" despite the underlying variable being ternary.

**Why "blank" is not a clean isolated feature:** Nanda found the blank direction is highly
correlated with the **unembedding** (cosine sim ~0.8–0.9). A cell can be legal only if it's
blank, so "high logit for cell X" ⇒ "X is blank"; the probe (trained at layer 6, late)
partly picks up this unembed alignment. Two readings: (a) it's contamination from probing
late, or (b) the model *intentionally* aligns is_blank with the unembed because it knows the
causal link. Either way: **prefer the mine/theirs axis as your clean 1-D object; treat blank
as entangled.**

---

## 5. "Different direction per cell" — address vs. format

Each cell has its own direction (that's why the board eats ~124 dims, not 2). But "different"
splits into two claims:

- **Definitely true — distinct address.** Cells must be distinguishable, so F5's axis is a
  different vector from F6's. This is *required*; if two cells shared a direction their states
  would be confounded. (The specific *orientation* is arbitrary/gauge — see below — but the
  *distinctness* is forced.)
- **Open question — shared format vs bespoke.** Are the 60 cells "one shared mine/theirs
  template placed at 60 different (near-orthogonal) addresses," or genuinely bespoke,
  neighbor-coupled representations? Evidence is mixed: the mine/theirs feature-type is
  regular across the board (suggests shared format), but corners behave differently (worse
  probe accuracy) and causally/correlationally linked cells have non-orthogonal directions
  (suggests coupling). A direct diagnostic: compute the pairwise cosine-similarity matrix of
  the 60 mine/theirs directions. All-orthogonal ⇒ "shared format, distinct address."
  Structured overlap tracking board adjacency ⇒ coupled regime.

---

## 6. Arbitrary vs. forced (gauge freedom)

A recurring analytical frame worth applying deliberately:

- **Arbitrary (gauge):** *which* absolute direction in R^512 a given cell's axis points. Re-run
  training with a new seed → a rotated/relabeled set of directions. Nothing in the task
  privileges a particular orientation; the embedding/unembedding co-adapt to whatever frame
  emerges. (Caveat: LayerNorm and standard-basis-aligned ops introduce *mild* basis
  preferences, so orientation may be slightly less free than pure gauge symmetry implies —
  the "privileged basis" question.)
- **Forced (non-arbitrary):** *that* the feature is mine/theirs rather than black/white;
  *that* there are ~2 dims/cell with one clean binary axis; the relative geometry imposed by
  Othello's causal structure (a corner can't be filled unless a neighbor is). Re-running
  training gives different *vectors* but the same *semantic* structure.

So the directions are arbitrary in **orientation** but non-arbitrary in **structure and
semantics**. The distinction is the crux of any direction-manipulation work: the orientation
is a free degree of freedom the task left unspecified, but the directions are co-adapted with
the embed/unembed, so they can't be rotated in isolation — downstream readers expect them
where training put them.

---

## 7. Over- vs. under-parametrization and superposition

Nanda's framing: Othello-GPT is likely **over-parametrized** for its task (bounded, finite:
60 cells, fixed rules), while real LMs are **always under-parametrized** (effectively
unbounded feature demand).

- Under-parametrization is the driver of **superposition** — more features than dims forces
  packing multiple features into overlapping, non-orthogonal directions, accepting
  interference.
- Over-parametrization (capacity slack) lets the model afford near-orthogonal, clean,
  monosemantic directions → tidy probes, surgical interventions. Some of Othello-GPT's
  cleanliness is an *artifact of capacity slack*, and may not transfer to the
  under-parametrized regime real models live in.
- **Open question this raises:** does a result obtained under capacity slack hold under
  superposition pressure? A natural knob at this scale is the capacity-to-task ratio (vary
  model width / depth) — moving from over- to under-parametrized turns the caveat into an
  independent variable.

Note: even Othello-GPT *may* use residual-stream superposition (512 dims vs 8×2048 neurons),
and the board state alone consuming ~25% of the stream is a hint there's pressure. Whether it
genuinely superposes is itself unsettled.

---

## 8. Why a transformer finds this hard (and why that's interesting)

Computing board state is harder than it looks: a single move can flip pieces far away along
any line, a piece can be re-flipped many times, and **transformers can't recurse**. A human
would compute state at move n from state at move n−1; the transformer must compute the state
at *every* position *in parallel*, in a fixed number of layers, with attention as the *only*
cross-position mixer (the MLP can't move information across positions). This is part of why
the model is a good "laboratory": the *way* it solves a non-trivially-parallel algorithmic
problem may teach general lessons about what's natural to represent in a transformer.
