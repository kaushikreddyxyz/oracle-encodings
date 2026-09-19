# In-context binding of an injected residual-stream channel

Code: `bind_icl.py` (stages `main`, `sig`), figures `plot_bind.py`, `plot_bind_models.py`; raw `results_bind_{2b,9b,qwen7b}/`,
figs `figures_bind_{2b,9b,qwen7b}/`. Runs on pod `icl-bind` (A100 80GB). 2026-09-03.

## Task

A frozen model reads a many-shot prompt of lines ` item: label`. The items are unique arbitrary words, the two
labels are arbitrary words, and the label of each line is a deterministic function of a hidden bit that the text
never contains. The bit is written into the residual stream of that line, at the item token or at the `:` readout
token, as a signed vector along a chosen direction. The sign convention, plus means label A or plus means label B,
is fixed for a whole prompt and counterbalanced across prompts. We ask whether the model comes to predict the label
from the injection with the sign set by the convention, which a pure steering vector cannot do. The ceiling is the
same task with the bit written as a marker word, and the floor is the same prompt with no injection.

## Design

- 128 examples per prompt, 12 seeds x 2 conventions = 24 prompts per condition; readout at every `:` in one causal
  forward pass, so each prompt yields the full shot curve. Late bin = shots 64-127.
- Directions: `rand` random unit vector; `actpc` random unit vector inside the top-64 activation PCs at the
  injection layer (readable but meaningless); `native` the model's own marker direction from the ceiling prompts
  (difference of means at the same layer/position); `unembed` W_U[A]-W_U[B] (a pure steering vector).
- Scale: beta x median centered residual norm at that layer and position. beta in {0.25, 0.5, 1, 2}.
  For reference, the native marker direction's own half-gap in the ceiling prompts is ~0.1-0.25 x that norm,
  so beta = 0.25 is the model's natural amplitude for its own marker representation.
- Layers: hidden states at ~L/6, L/3, L/2, 2L/3. Positions: item token or `:` token.
- Metrics: counterbalanced accuracy p(correct label); decomposition of the logit difference into a learned
  component (sign x convention interaction, flips with convention) and a steering component (fixed sign);
  health = fraction of readouts whose top-1 token is one of the two labels (1.00 everywhere below).

## Results: main grid

### gemma-2-2b (figures_bind_2b/fig1_curves_{item,sep}.png, fig2_late_heatmap.png, fig3_decomposition.png)

Floor 0.506, ceiling 0.999 (late-bin counterbalanced accuracy).

| arm | best condition | late acc | learned (logit) | steering (logit) |
|---|---|---|---|---|
| token (ceiling) | - | 0.999 | 19.8 ± 0.7 | -0.1 ± 0.3 |
| rand | hs8, item, beta 2 | 0.998 | 17.7 ± 0.7 | 0.2 ± 0.2 |
| actpc | hs4, item, beta 2 | 1.000 | 21.0 ± 0.7 | 0.3 ± 0.4 |
| native | hs13, sep, beta 2 | 0.999 | 16.9 ± 0.8 | -0.2 ± 0.5 |
| unembed | hs4, item, beta 2 | 0.990 | 15.2 ± 0.5 | 4.9 ± 0.4 |

- A fully random direction is bound to an arbitrary label at ceiling level when injected at the item token in
  early layers (hs4/hs8) at beta 2; the shot curve tracks the word-marker ceiling. The learned component equals
  the ceiling's and the steering component is zero.
- Dose: rand at hs8/item gives 0.57 (beta 0.5), 0.90 (beta 1), 0.998 (beta 2). At the natural amplitude
  (beta 0.25) nothing is read, for any direction including the model's own marker direction (<= 0.61).
- Position x layer: item-token injection works only in early layers (hs13/17 item: <= 0.68 for every arm);
  `:`-token injection works in mid/late layers (rand hs13 beta 2: 0.77; actpc 0.97; native 1.00). Consistent
  with item-token injections being carried by previous-token/induction heads and `:`-token injections being read
  directly at the readout position.
- Readability sets the dose: actpc (random inside the activation PCs) reaches 0.99-1.00 at beta 1 where rand
  gives 0.71-0.90, and 0.68-0.80 at beta 0.5 where rand gives 0.52-0.57.
- The pure-steering control is exposed by the decomposition: at the `:` token, hs17 beta 2, its steering
  component (8.9) exceeds its learned component (6.7) and counterbalanced accuracy is 0.63. At the item token
  it behaves like any other direction (learned 15.2) plus a steering offset (4.9).

### gemma-2-9b (figures_bind_9b/)

Floor 0.504, ceiling 1.000.

| arm | best condition | late acc | learned (logit) | steering (logit) |
|---|---|---|---|---|
| token (ceiling) | - | 1.000 | 24.1 ± 0.7 | 0.0 ± 0.4 |
| rand | hs7, item, beta 2 | 0.970 | 9.7 ± 0.6 | -0.3 ± 0.1 |
| actpc | hs7, item, beta 2 | 1.000 | 23.1 ± 0.7 | -0.4 ± 0.5 |
| native | hs7, item, beta 2 | 0.999 | 15.7 ± 0.3 | -0.2 ± 0.2 |
| unembed | hs7, item, beta 2 | 0.983 | 11.6 ± 0.3 | 3.3 ± 0.3 |

Same structure: random direction at the item token in early layers at beta 2 gives 0.97 (hs7 and hs14), 0.56-0.63
at beta 1, chance at beta <= 0.5; actpc reaches 1.00 already at beta 1 and 0.74 at beta 0.5; `:`-token
injection works in mid layers (rand hs21 beta 2: 0.84; actpc hs14/21: 0.97; native hs28: 0.95).

### Qwen2.5-7B (figures_bind_qwen7b/)

Floor 0.504, ceiling 1.000. Same structure, more sensitive: random direction at the item token, hs4/hs9, gives
0.997-1.00 at beta 1, 0.87-0.90 at beta 0.5 (learned 5.8-6.8 logits, steering ~0); actpc reaches 0.99-1.00 at
beta 0.5 and 0.71-0.77 at beta 0.25; `:`-token injection works at hs14/18 (rand 0.85-0.90 at beta 2, native
0.98-1.00). The unembedding control at the item token: learned 15.7, steering 1.9.

Cross-model summary: `figures_bind_models.png` (top: best-condition shot curves per arm vs floor/ceiling;
bottom: learned vs steering decomposition with per-seed points).

## Results: signatures (figures_bind_{2b,9b}/fig4_signatures.png, sig_table.txt)

Numbers are p(correct label) under the current convention in shot windows 0-32 / 32-48 / 48-64 / 64-80 / 80-96 / 96-128.

**Delayed onset** (injection absent and labels random for the first 32 shots). Before onset every arm sits at 0.50;
after onset the random-direction channel is picked up with a lag comparable to the word-marker ceiling:
gemma-2-2b rand hs8/item beta 2: 0.50 / 0.63 / 0.88 / 0.97 / 0.99 / 1.00 (ceiling 0.79 / 0.98 / 0.99 / 1.00 ...);
gemma-2-9b rand hs7/item beta 2: 0.50 / 0.57 / 0.67 / 0.80 / 0.87 / 0.91 (actpc: 0.78 / 0.97 / 0.99 / 1.00).

**Convention flip at shot 64.** Accuracy under the new convention collapses at the flip and is relearned at the
same pace as the word marker: 2b rand: 0.80 / 0.99 / 1.00 | 0.21 / 0.48 / 0.71 (ceiling 0.82 / 0.98 / 0.99 |
0.26 / 0.47 / 0.66); 9b rand: 0.73 / 0.93 / 0.95 | 0.29 / 0.49 / 0.66 (ceiling ... | 0.29 / 0.54 / 0.70).
A steering vector cannot track a flip; a fixed-sign component would show as a floor below 0.5 that never recovers.

**XOR with a text feature** (label = injected bit XOR item-is-an-animal). The model combines the channel with the
text: 2b rand hs8/item beta 2 reaches 0.90 in shots 96-128, above the word-marker ceiling (0.79); 9b rand 0.74 vs
ceiling 0.77. (`:`-token injections do XOR poorly on both models, <= 0.62.)

**Qwen2.5-7B signatures** (figures_bind_qwen7b/fig4_signatures.png): onset rand hs4/item beta 1: 0.50 / 0.82 / 0.96 /
0.99 / 0.99 / 0.99 (ceiling 0.90 / 1.00 ...); flip: 0.88 / 0.99 / 1.00 | 0.52 / 0.78 / 0.90 (ceiling 0.90 / 1.00 / 1.00 |
0.34 / 0.67 / 0.82), i.e. relearned faster than the word marker. XOR: rand beta 1 at hs4/item fails (0.58) while
beta 0.5 reaches 0.70 and the native `:`-token direction 0.75 (ceiling 0.75): a large early-layer injection at the item
token overwrites the item's own category feature, so the dose that maximises binding of the bit alone is too high
for composing it with text at the same position.

## Results: 512-shot prompts, low dose (figures_bind_2b/fig5_long512.png)

gemma-2-2b, 12 seeds x 2 conventions, windows 0-16 / 16-32 / 32-64 / 64-128 / 128-256 / 256-384 / 384-512.
Low-dose channels keep being picked up as context grows rather than saturating:

| condition | 64-128 | 128-256 | 256-384 | 384-512 |
|---|---|---|---|---|
| rand hs8 item beta 1 | 0.94 | 0.98 | 0.98 | 0.99 |
| rand hs8 item beta 0.5 | 0.61 | 0.65 | 0.68 | 0.70 |
| actpc hs8 item beta 0.5 | 0.84 | 0.92 | 0.95 | 0.96 |
| native hs13 sep beta 0.5 | 0.83 | 0.88 | 0.91 | 0.92 |
| actpc hs13 sep beta 1 | 0.94 | 0.96 | 0.98 | 0.98 |
| rand hs13 item beta 1 | 0.53 | 0.53 | 0.54 | 0.54 |

Items for 512-shot prompts are drawn from a vocabulary-derived single-token word pool (the curated pool has 191 words).

## Interpretation

- The strong hypothesis holds in this minimal form on three frozen models from two families: an injected bit along
  a direction the model has never been trained to read is bound in context to an arbitrary label, the binding
  flips with the prompt's convention, tracks delayed onset and mid-context flips, and composes with a text feature.
  The steering component is zero for every non-steering direction, and the design exposes the steering vector.
- What it costs: amplitude and placement. Random directions need ~1-2x the centered residual norm within 128 shots
  and are read only where the ICL circuitry can see them: early layers at the item token (carried by previous-token
  and induction heads), mid/late layers at the readout token. Readable-but-meaningless directions inside the
  activation PC subspace need 2-4x less amplitude; with 4x more context the required amplitude drops further.
- Mechanistically this looks like induction over a pseudo-token: at high amplitude the injected sign becomes the
  dominant content of the item position, and the standard copy circuitry binds it to the following label. That is
  still in-context use of an arbitrary channel, but it also says the "port" is the existing ICL machinery, not a
  new reader, which is consistent with the Gemma grid-walk findings (no fixed plane; readable cone set by weights).

## Files

- `results_bind_{2b,9b,qwen7b}/main.json` (+ `main_ref.npz`), `results_bind_{2b,9b}/sig.json`, `results_bind_2b_long/main.json`
- `figures_bind_{2b,9b,qwen7b}/fig1_curves_{item,sep}.png, fig2_late_heatmap.png, fig3_decomposition.png, main_table.md`
- `figures_bind_{2b,9b}/fig4_signatures.png, sig_table.txt`; `figures_bind_2b/fig5_long512.png`; `figures_bind_models.png`
