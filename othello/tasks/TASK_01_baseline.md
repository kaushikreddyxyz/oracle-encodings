# TASK 01 — Baseline: recover mine/theirs cell-state directions in a fresh no-dropout Othello-GPT

## What this is

Your run brief for the first experiment. It sits alongside the **Othello-GPT dossier**
(`00_MANIFEST.md` and the numbered files). This brief tells you *what to do*; the dossier is
your *reference* — pull the relevant dossier file when you need a fact, concept, convention,
or failure-mode (the manifest routes you). Read `00_MANIFEST.md` first for orientation, then
come back here.

> Decisions in this brief are settled. Do not re-litigate them (e.g. don't switch to a
> smaller model, don't add TransformerLens, don't swap the data source) unless you hit a hard
> blocker, in which case document it and proceed per the Failure Handling section.

> **Branch:** do all experiment and code work on the **`othello` branch** you are on — commit
> your code, configs, and run artifacts there. The **one exception** is the Almanac report:
> per the `almanac` skill, the report file goes on **`main`** even though the rest of your
> work stays on `othello` (Stage 5 spells out how). So: code → `othello`; report → `main`.

---

## Objective

Produce a **convincing demonstration of linear mine/theirs cell-state feature directions** in
an Othello-GPT that **you train from scratch**. The minimum deliverable is a linear probe
that decodes per-cell mine/theirs state to high accuracy; stronger evidence (geometry, causal
intervention) is welcomed if the floor is met and budget remains.

The point of training your own model (rather than loading a pretrained one) is to have a
**clean, dropout-free baseline you fully control** — this is the foundation for later
feature-injection experiments, so cleanliness and trustworthiness matter more than speed.

---

## Settled decisions (the spec)

- **Model:** fresh **8-layer, d_model=512, 8-head** GPT-2-style model (the faithful Li et al.
  config). **dropout = 0** everywhere (attention, residual, embedding). ~25M params.
  - Rationale for faithful size: a known architecture makes failure *attributable*. If clean
    directions don't emerge at the established config, that's a real finding — not a "did I
    undersize it?" ambiguity. Smaller models are a deliberate *later* experiment, not a
    first-run shortcut.
- **Trainer:** **nanoGPT** (Karpathy). Keep its model definition + training loop; **replace
  its data layer** with Othello tokens. Othello is **not text** — do **not** use nanoGPT's
  BPE/text path. Set `dropout=0.0` in config. Expose **post-block residual activations via a
  forward hook** so the probe can read `resid_post` at a chosen layer (design this in from the
  start; don't retrofit).
- **No TransformerLens.** You own the model code, so activation caching and interventions are
  native forward hooks (~a few lines). This also makes dossier gotcha **G1** (the TL v3.0
  numerics landmine) **not apply** to your code — it's only relevant if you *read* others'
  TL-based code for reference.
- **Data:** use Li et al.'s **published synthetic dataset** (uniform-random legal-move games)
  from the `othello_world` repo — do **not** write your own game generator (it's the
  highest-bug-density component and they've already published a verified one). Use their
  **board-state engine** from the same repo to compute ground-truth board state for probe
  targets. Vendor these in; don't rewrite them.
  - The dataset is distributed via a Google Drive link (see `06_resources.md`), which can be
    flaky for automated download. **Confirm you can pull the data as an early step.** If the
    download genuinely fails, fall back to their generator code — but prefer the published
    data.
- **Probe:** linear (multinomial logistic regression), per-cell, in the **mine/theirs
  (player-relative) frame**, trained on `resid_post`. Use Nanda's setup in
  `02_facts_and_config.md` as the known-good reference (layer 6, moves `[5:-5]`, the
  hyperparams listed there) — adapt as needed, but those are your anchors.
- **Training "done" condition:** train until **held-out legal-move prediction accuracy
  reaches ~99%+** (illegal-move rate near zero), then checkpoint and stop. This is the *task
  metric* and the *only* thing you need to monitor during training. (No probe checks mid-
  training — the probe comes after, kept sequential so "is the model competent" and "are its
  directions clean" stay unconfounded.)

---

## Stages (gated; do them in order)

### Stage 0 — Setup
- Vendor in Li et al.'s data + board engine. Get nanoGPT. Pin and record exact versions of
  everything (torch, nanoGPT commit, etc.) in the run log.
- **Early check:** confirm the synthetic dataset actually downloads. If not, fall back to the
  generator and note it.

### Stage 1 — Data pipeline + GROUND-TRUTH GATE  ⛔ hard gate
- Build the path: published games → tokens → (via Li's engine) ground-truth board state →
  mine/theirs target arrays. Centralize and **unit-test every coordinate/index/frame
  conversion** (see `04_conventions_and_trace.md` and gotchas **G3, G4, G5**).
- **Build the verified trace fixture** per the procedure in `04`: run the engine on ≥1 game,
  freeze (token seq, absolute board, parity, mine/theirs board, legal moves) at a middle move,
  save to disk. Confirm your pipeline reproduces it exactly.
- **GATE: do not start training until the fixture reproduces and conversion unit-tests pass.**
  This gate exists because the worst outcome is training for hours on top of a board-state
  bug. Most "the probe doesn't work" failures are really bugs caught here.

### Stage 2 — Train the fresh no-dropout model
- Train the 8L/512 dropout-0 model on the synthetic data via the nanoGPT loop.
- Monitor **held-out legal-move accuracy** only. Train to **~99%+**, then checkpoint and stop.
- If you reach your own budget limits before hitting the target, do **not** silently stop —
  treat it as underperformance and enter Failure Handling (it likely indicates a config/data
  issue worth diagnosing, not just "needs more time").

### Stage 3 — Probe for mine/theirs directions (THE DELIVERABLE)
- Train the linear per-cell probe on `resid_post` (forward hook) in the mine/theirs frame,
  scoring on the middle band of moves. Handle parity correctly (**G2, G5** — this is *the*
  classic Othello trap).
- **Success floor:** **high-90s % per-cell accuracy on middle moves.** This is "convincingly
  found the directions," minimum bar. Compare against the known-good anchor in `02`.
- If accuracy is high but you want a quick confidence read, check odd↔even probes are near-
  negations (expected if the frame is right).

### Stage 4 — Evidence escalation (only if Stage 3 floor met AND budget remains)
More evidence is better; each tier is independently valuable and independently reportable. Do
as many as budget allows, in order:
- **Tier A — Geometry:** compute the **60×60 cosine-similarity matrix** of the mine/theirs
  directions. Report whether they're ~orthogonal ("shared format, distinct address") or show
  structured overlap tracking board adjacency ("coupled regime"). See `03_concepts.md` §5 —
  this directly informs later injection difficulty, so it's high-value.
- **Tier B — Causal:** single-cell intervention via native forward hook — negate/scale the
  coordinate along one cell's mine/theirs direction at ~layer 4 and confirm the model's legal-
  move logits change as if that cell flipped (see `02` intervention reference, `05` G11). This
  upgrades "decodable" to "used."

---

## Stage 5 — Write and publish the Almanac report (required; the public deliverable)

This run isn't done until it's **written up and published** as an Almanac report. The report
is a *separate deliverable* from the run artifacts above: the artifacts are working outputs in
`runs/`; the **report** is a single sober Markdown file in `reports/`, pushed to `main`, that
writes up the result for people who weren't there.

**Use the `almanac` skill — it is the authority for this stage.** Before writing, **read
`reference/house-style.md`** (synced into that skill); it is the binding authoring contract
for organization, voice, callouts, math, and figure syntax. If anything below appears to
conflict with the house style, **the house style wins.** What follows is only the
task-specific framing.

- **It's a different register from `SUMMARY.md`.** Drop all voice. Sober, factual, scientific
  prose for a reader who wasn't here — no persona, no project-internal shorthand, no jokes.
- **File + figures:** one Markdown file at `reports/othello-mine-theirs-baseline.md` (match an
  existing reports dir/slug convention if the repo already has one). Put figures in the same
  directory, referenced by relative path with real alt text. **Check `reports/` first** for an
  existing report on this topic — if one exists, update it rather than adding a near-duplicate.
- **Structure (per house style):** open with the question and the headline result, close with
  a `conclusion` callout. Plain GFM + math + directive callouts only — no HTML/JSX/MDX.
- **Figures to include:** the per-cell mine/theirs probe-accuracy board heatmap (the core
  result); and, if Stage 4 ran, the 60×60 cosine-similarity matrix (Tier A) and/or the
  before/after intervention logit boards (Tier B). If any figure is SVG, verify it's
  well-formed XML (`xmllint --noout <file>.svg`) before committing — a parse check on your own
  output, not a render preview.
- **What to report:** the question (can a fresh no-dropout Othello-GPT be shown to linearly
  represent mine/theirs cell state), the model + setup in brief, the headline numbers (legal-
  move accuracy; probe accuracy on middle moves), the geometry/causal evidence if obtained, and
  honest caveats (e.g. corner-cell degradation, anything from the diagnostic trace). Report
  numbers plainly; don't oversell.
- **A negative/failed result is still a valid report.** If the floor genuinely wasn't reached
  after bounded remediation, write that up soberly too — what was attempted, what was observed,
  what it suggests. Clean negative results belong on the Almanac.
- **Publish it: commit AND push to `main`** as the final step, via the project's deploy key,
  **without pausing to ask** — publishing is part of the task and is pre-authorized by the run.
  If your experiment code lives on a working branch, the *report* still goes on `main` anyway
  (from your `othello` branch: check out `main`, add the report + figures, commit, push `main`,
  then return to `othello`). A report left only on a feature branch never publishes. (This instruction is consistent with
  the failure-handling policy: "document and stop" for *code/experiment* rabbit-holes never
  means "skip publishing" — the write-up of whatever happened is always published.)

> Note: the `almanac` skill's push-without-pausing instruction governs this stage. It does not
> conflict with this task's general caution elsewhere — pushing a report to `main` is the
> expected, routine completion of the work, not a separate action needing a green light.

---

## Progress check-ins (every ~30 min)

This is a long unattended run, so emit a **heartbeat roughly every 30 minutes** while working,
so the run can be monitored and a stalled or diverging run caught early. Append each check-in
to a dedicated **`runs/task01_<timestamp>/progress.md`** (timestamped entries, newest at the
bottom) — a file is the reliable channel for an unattended run; if your orchestration also
allows surfacing it elsewhere, do that too, but always write it to disk.

Each check-in should be a few lines covering, as applicable to the current stage:
- **Timestamp** and **current stage** (e.g. "Stage 2 — training").
- **Primary metric right now:** during training, current held-out **legal-move accuracy** and
  step/epoch; during probing, current probe accuracy; etc.
- **Trajectory + ETA:** is the metric still improving, and a rough **expected time to the
  stage's target** (e.g. "~88% at step X, climbing ~+2%/30min, est. ~99% in ~1.5h"). An honest
  "uncertain" is fine — a rough number beats none.
- **Health flags:** anything off — loss plateaued/diverged, throughput far below expectation
  (recall the model is tiny; if it's slow, suspect the data pipeline, not the GPU), errors
  recovered from, gate status.
- **Next:** what you're about to do.

Keep them short and factual (working-log register, like `SUMMARY.md` — not the sober report
voice). The goal is a glanceable trail of "is this run healthy and roughly on schedule," not
prose. If a check-in would reveal the run is stuck or off-target, that's exactly when the
**Failure handling** path should already be engaging — don't just keep logging a flatlined
metric for hours.

---

## Failure handling (applies at every stage)

If a stage fails or underperforms (e.g. legal-move target not reached; probe accuracy below
the floor; fixture won't reproduce):

1. **Diagnose against the dossier first.** Most failures are known issues, not novel ones.
   Walk `05_gotchas.md` — especially parity/frame (G2, G5), indexing/off-by-ones (G3, G4),
   wrong layer (G6), early/late moves (G9). A near-chance probe result (~33% or ~50%) is
   almost always a convention bug, not a modeling failure.
2. **Form explicit hypotheses and attempt bounded remediation.** Reason carefully about the
   likely cause, try the most probable fix, re-test. Tweak config, layer, or frame handling as
   the diagnosis indicates. **Log every hypothesis and what it changed** — the morning-readable
   reasoning trace matters as much as the fix.
3. **Don't rabbit-hole.** Try the well-motivated fixes; if remediation is exhausted without
   resolution, write a clean failure report (what failed, what you tried, what you suspect,
   what you'd try next) and stop rather than flailing.

A clean "here's exactly where it broke and why I think so" is a good outcome. A confusing pile
of half-tried changes is not.

---

## What to write to disk

Write everything to a clear run directory (e.g. `runs/task01_<timestamp>/`). Produce:

- **`SUMMARY.md`** — the morning-readable top-level **working** summary (this is a run
  artifact, *not* the Almanac report — see Stage 5). Lead with: did the floor get met? Final
  legal-move accuracy, final probe accuracy, which evidence tiers were reached, and the
  one-paragraph bottom line. Then any failures/caveats. Write it for a human checking the run;
  it can be informal. (The polished, voiceless, public write-up is the separate Almanac report.)
- **`run_log.md`** (or `.jsonl`) — chronological log: versions pinned, decisions made,
  gates passed, training curve milestones, and **the full diagnose/remediation reasoning trace**
  if anything went sideways.
- **`progress.md`** — the ~30-min heartbeat trail (see Progress check-ins). Timestamped
  entries; the at-a-glance health/ETA record for someone checking in mid-run.
- **`model/`** — the trained model checkpoint (+ the exact config used) and the training
  loss/accuracy curve (csv or plot).
- **`probe/`** — the trained probe weights, AND the extracted **mine/theirs direction vectors**
  (one per cell, the actual deliverable), plus per-cell accuracy (a board-shaped heatmap is
  ideal). Save directions in a plain, reload-friendly format (e.g. `.npy`/`.pt` with a
  documented shape and the frame/convention noted).
- **`fixture/`** — the verified ground-truth trace from Stage 1 (so future runs can regression-
  test the pipeline against it).
- **Stage 4 outputs (if reached):** the **60×60 cosine-sim matrix** (array + plotted heatmap)
  for Tier A; for Tier B, before/after logit boards for the intervened game(s) and a note on
  whether the flip behaved as predicted.
- **`environment.md`** — exact versions / commits / how to reproduce the run, so the result is
  re-runnable.

Label every saved board/direction array with its **frame** (absolute vs mine/theirs) and
**length/indexing** (60 vs 64, token vs cell index) — per `04`, unlabeled arrays are the main
source of silent downstream confusion.

---

## Definition of done

In all cases, the run is complete only once the result is **written up as an Almanac report
and pushed to `main`** (Stage 5) — alongside the run artifacts in `runs/`.

- **Floor (required):** trained no-dropout 8L/512 model at ~99%+ legal-move accuracy; linear
  mine/theirs probe at high-90s middle-move accuracy; directions + checkpoint + probe + working
  summary written to `runs/`; **report published to `main`**.
- **Strong (if budget allows):** + the geometry matrix (Tier A) and/or a working single-cell
  causal intervention (Tier B), included as figures in the report.
- **Acceptable failure:** a clean, well-reasoned account localizing where it broke and why,
  with the diagnostic trace — if the floor genuinely couldn't be reached after bounded
  remediation. **This still gets written up soberly and published to the Almanac** — a clean
  negative result is a valid report.
