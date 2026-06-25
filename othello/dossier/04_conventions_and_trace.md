# Conventions & Worked Trace

The highest-frequency source of silent bugs in Othello-GPT work is translating between the
several coordinate/indexing systems. This file lays them out and gives a procedure for
building a verified ground-truth trace you can use as a test fixture.

> **Honesty note on verification:** the *existence* and *roles* of these coordinate systems
> are verified from the sources. Some exact mappings (e.g. the precise int↔(row,col) formula,
> the exact orientation of row/col) depend on the implementation in Li et al.'s
> `othello_world` code and **must be confirmed against that code** before you rely on them —
> these are flagged **[CONFIRM IN CODE]**. Do not hardcode an unconfirmed mapping; derive it
> from the engine and check it against the worked-trace procedure below.

---

## The coordinate/indexing systems in play

There are (at least) four representations of "a square," and bugs come from conflating them:

1. **Board notation** — human-readable, like `A1`–`H8` (column letter A–H, row 1–8) or
   `(row, col)` with row,col ∈ 0–7. The 8×8 grid has 64 cells. **[CONFIRM IN CODE]** which
   convention (letter-number vs row-col, 0- vs 1-indexed, and whether row 0 is top or bottom)
   the `othello_world` engine uses.

2. **Flat board index 0–63** — the 8×8 grid flattened, typically `idx = row * 8 + col`.
   Used by the board-state engine. **[CONFIRM IN CODE]** the exact flattening.

3. **The 60-cell "playable" index** — the 4 center cells (the starting 2×2) are removed,
   leaving 60. Probes and board-state arrays are often indexed over these 60 (or sometimes
   over 64 with centers masked). **You must know whether a given array is length 60 or 64.**

4. **Model vocab token 0–60** — token `0` = pass; tokens `1`–`60` = the 60 playable cells.
   This is **shifted by one** relative to a 0-based 60-cell index, AND skips the centers
   relative to the 0–63 flat index. Two off-by-ones live here at once.

**The two off-by-one traps, explicitly:**
- **Center exclusion:** vocab/playable indices skip the 4 center squares; flat 0–63 does
  not. Converting between them is *not* a simple add/subtract — you need the engine's
  center-aware mapping. **[CONFIRM IN CODE]**
- **Pass token shift:** vocab token `t` for a cell corresponds to playable-index `t−1`
  (because token 0 is pass). Off-by-one if you forget pass occupies slot 0.

**The mine/theirs "frame" is a fifth axis of convention** (not spatial but semantic): board
state can be stored as {empty, black, white} (absolute) or {empty, mine, theirs}
(player-relative). The conversion is parity-dependent: on black-to-play moves mine=black; on
white-to-play moves mine=white. **Always know which frame an array is in.** (See
`03_concepts.md` §1–2.)

---

## Recommended discipline (prevents most bugs)

- **Never hardcode a mapping you haven't derived from the engine.** Use Li's
  `othello_world` board-state code as the single source of truth for: token → move,
  move → board update, board → {empty,black,white} array. Wrap conversions in named helper
  functions and unit-test each one against the worked trace below.
- **Tag every state array with its frame and length** (in a variable name or comment):
  e.g. `board_abs_64`, `board_mine_60`. Most "the probe accuracy is mysteriously ~50%" bugs
  are a frame or center-exclusion mismatch.
- **Centers:** confirm whether a given pipeline represents the board over 60 or 64 cells,
  and if 64, that centers are masked/ignored in probe scoring (centers are never blank and
  never playable).
- **Parity:** when computing mine/theirs targets, flip absolute→relative based on move index
  parity. Confirm which parity is "black to play" in the dataset **[CONFIRM IN CODE]** — do
  not assume even=black; verify it.

---

## Building a verified ground-truth trace (your test fixture)

A single correct end-to-end trace is the most valuable debugging asset you can have. Build
it **from the engine** (don't transcribe one from memory — including mine; any literal board
below would be **[UNVERIFIED]**). Procedure:

1. Generate or take one synthetic game = a sequence of ~60 move tokens.
2. Run Li's board-state engine forward over the moves, recording after each move:
   - the absolute board array ({empty, black, white} over 64 or 60 cells),
   - whose turn it is (parity),
   - the derived mine/theirs array.
3. Pick a representative middle move (e.g. move 20). Freeze: the token sequence so far, the
   absolute board, the parity, the mine/theirs board, and the set of legal next moves.
4. Independently, run the model on the same prefix, grab `resid_post` at the chosen layer,
   apply the probe, and confirm the probe's predicted board matches the engine's
   ground-truth board (in the correct frame).
5. **Save this trace to disk** (token seq + boards + parity + legal moves + probe output) as
   a fixture. Every future pipeline change can be regression-tested against it: if the
   pipeline reproduces the fixture, your conversions are correct; if not, you have a
   localized, immediately-caught bug instead of a silent late one.

**What "correct" looks like at the checkpoints:**
- Engine legal-move set should match the model's high-logit cells on an *unedited* game
  (the model plays legal moves with high accuracy).
- Probe-predicted board should match engine board to high-90s % on middle moves (the
  known-good anchor from `02`). A result near ~50% or ~33% signals a frame/parity/indexing
  bug, not a modeling problem.

---

## Quick reference: the conversions you'll write

| From → To                          | Watch out for                                  |
|------------------------------------|------------------------------------------------|
| vocab token → playable index       | subtract 1 (token 0 = pass)                     |
| playable index → flat 0–63         | center-aware; **[CONFIRM IN CODE]**             |
| flat 0–63 → (row,col)              | `row=idx//8, col=idx%8` **[CONFIRM orientation]**|
| absolute {b,w} → mine/theirs        | parity-dependent flip; confirm which parity=black|
| board array length                 | is it 60 (centers excluded) or 64 (masked)?     |

Treat this table as the checklist for any function that touches squares.
