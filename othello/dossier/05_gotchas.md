# Gotchas — pre-flight checklist

These are verified failure modes: things that produce **plausible-looking but wrong results
with no error message**. Read before writing pipeline or probe code. Each entry: the trap,
why it bites, and the guard.

---

## G1. TransformerLens v3.0 numerics landmine ⚠️ (highest severity)

- **Trap:** Loading the model with the post-3.0 default ("bridge" / raw-HF numerics) instead
  of the legacy `HookedTransformer` numerics (LayerNorm folded + weights centered).
- **Why it bites:** Every published probe direction, the "board computed by layer 4" finding,
  and the residual-stream geometry are in the **LN-folded, weight-centered** frame. The new
  default puts you in a *different numeric frame* → your geometry and accuracy silently
  disagree with the literature. No error is raised.
- **Guard:** Pin a pre-3.0 TransformerLens, OR use the compatibility path
  (`bridge.enable_compatibility_mode()` / legacy `HookedTransformer`). **Record the exact
  installed version.** Sanity-check by reproducing the high-90s middle-move probe accuracy
  *before* building anything on top. (Full detail in `02_facts_and_config.md`.)

---

## G2. The parity sign-flip (the classic Othello trap)

- **Trap:** Training/interpreting a probe on **absolute black/white** labels across all moves.
- **Why it bites:** The mine/theirs direction's meaning flips every move; an absolute-frame
  target sign-flips every step, which is pathological for a linear probe. This is the exact
  bug that made Li et al.'s linear probe report ~20.4% error and conclude "non-linear."
- **Guard:** Work in the **mine/theirs** frame — condition on move parity (separate odd/even
  probes, or relabel state to player-relative every other move). Confirm which parity is
  "black to play" in the data; don't assume. Expect odd and even probes to be near-negations.

---

## G3. Center-cell exclusion off-by-one

- **Trap:** Treating the board as a clean 64-cell array and indexing it with vocab tokens, or
  mixing 60-length and 64-length arrays.
- **Why it bites:** The 4 center squares are excluded from the vocab (60 playable + pass), but
  the board engine works over 64. Converting vocab/playable indices ↔ flat 0–63 is **not** a
  simple offset — it's center-aware.
- **Guard:** Use the engine's mapping, never a hand-rolled offset. Tag arrays with their
  length (60 vs 64). When scoring probes, decide explicitly how centers are handled (they're
  never blank, never playable).

---

## G4. Pass-token shift

- **Trap:** Forgetting token `0` = pass, so cell tokens are `1`–`60`, shifted by one vs a
  0-based 60-cell index.
- **Why it bites:** Every cell is off by one; the board "looks plausible" but is rotated/
  shifted, giving mediocre-but-not-zero probe accuracy that's easy to miss.
- **Guard:** Centralize token↔index conversion in one tested helper. Remember passes are rare
  in random games but *do* occur — don't crash on them, but they sit outside per-cell board
  structure (the pass token has no cell state).

---

## G5. Frame confusion (absolute vs mine/theirs)

- **Trap:** Comparing a probe output in one frame against ground truth in the other, or
  mixing frames mid-pipeline.
- **Why it bites:** Accuracy lands near chance (~33% for 3-way, ~50% for the binary contrast)
  for a reason that looks like "the probe didn't learn," when really it's a frame mismatch.
- **Guard:** Tag every state array with its frame. Convert deliberately and once. A near-chance
  result is almost always a convention bug, not a modeling failure — check frame/parity/
  indexing before concluding anything about the model.

---

## G6. Wrong layer / assuming a clean per-layer switchover

- **Trap:** Assuming board state appears exactly at one layer, or probing a layer where it's
  not yet clean.
- **Why it bites:** Board state is fully computed by ~layer 4 and *used* by 5–6; Nanda probed
  layer 6 but intervened at layer 4. Processing "spreads across adjacent layers when it can
  get away with it" — there isn't a crisp switchover, and it may differ per cell.
- **Guard:** Probe a middle layer (6 is the reference; 4 also works well; 3 and 7 are worse).
  For interventions, layer 4 (sometimes 3) is the reference. Don't assume a single clean
  boundary layer.

---

## G7. The "blank" direction is entangled with the unembed

- **Trap:** Treating the blank/filled direction as a clean isolated feature, or using it as
  your 1-D injection/analysis target.
- **Why it bites:** blank-probe ↔ unembed cosine sim is ~0.8–0.9 (a cell is legal only if
  blank, so high-logit ⇒ blank, and the late-layer probe absorbs this). Whether this is
  contamination or intentional alignment is unsettled.
- **Guard:** Prefer the **mine/theirs** axis as the clean, genuinely-binary,
  causally-validated 1-D object. Treat blank as messy/entangled; don't build a "clean single
  direction" story on it.

---

## G8. Dropout-induced redundancy (in the original model)

- **Trap:** Expecting one-dimension-per-feature surgical behavior, or perfect intervention
  recovery, from the *original* pretrained model.
- **Why it bites:** The original was trained **with attention + residual dropout**, which
  incentivizes backup circuits and redundant representations. Interventions don't perfectly
  recover logit magnitudes; patching/editing is messier and "a bit suspect."
- **Guard:** Know whether your model has dropout. A from-scratch **no-dropout** retrain is
  cleaner and the standard recommendation for intervention/representation work (Nanda
  explicitly: no-dropout "will make your life much easier"). Whether *you* retrain is a brief
  decision, not this dossier's call — but the *fact* of dropout's effect is why results differ.

---

## G9. Early/late-move weirdness

- **Trap:** Training/scoring the probe on all 60 moves including the first and last few.
- **Why it bites:** Early/late moves behave anomalously — e.g. the **first-move residual
  stream has ~20× the norm** of others. Including them degrades/contaminates results.
- **Guard:** Use the middle band (Nanda used moves `[5:-5]`). Report accuracy on the middle
  band. (Whether to *also* characterize early/late behavior is an open question, not a bug.)

---

## G10. Synthetic vs championship model mix-up

- **Trap:** Loading the championship (real-human-games) model when you want the synthetic one
  (or vice versa).
- **Why it bites:** They're different models with different distributions; the championship
  model has a measurably *worse* world model. The linear result is established on
  **synthetic**. Mixing them up confounds comparisons.
- **Guard:** Confirm which checkpoint you loaded. Default to synthetic for representation
  work unless you have a specific reason.

---

## G11. Decodable ≠ used

- **Trap:** Concluding a direction is part of the model's computation because a probe decodes
  it.
- **Why it bites:** Probes can read out incidental correlates the model doesn't causally use.
- **Guard:** Validate causally (intervention changes behavior as predicted) before claiming a
  direction is functional, not just present. This is *the* methodological lesson of the whole
  Othello line.

---

## Quick pre-flight (tick before trusting any result)

- [ ] Installed TransformerLens version recorded; numerics in legacy/compat frame (G1)
- [ ] Reproduced the known-good probe accuracy (high-90s, middle moves) before building on top
- [ ] Probe/board in the **mine/theirs** frame with parity handled (G2, G5)
- [ ] Token↔index↔board conversions centralized and unit-tested against a fixture (G3, G4)
- [ ] Correct layer for probing vs intervening (G6)
- [ ] Using mine/theirs (not blank) as the clean 1-D object (G7)
- [ ] Know whether the model has dropout (G8)
- [ ] Scoring on the middle band of moves (G9)
- [ ] Correct checkpoint: synthetic vs championship (G10)
- [ ] Causal validation before claiming a direction is "used" (G11)
