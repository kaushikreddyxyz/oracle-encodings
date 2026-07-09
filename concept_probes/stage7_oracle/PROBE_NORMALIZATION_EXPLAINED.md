# Probe normalization, explained plainly

(2026-07-09; corrects a confusing earlier explanation)

## 1. "Is the natural text not ClimbMix?"

It IS ClimbMix. "Natural" in the stage-5/6 docs just means "real web text, as
opposed to our synthetic training sentences". Concretely
(`stage6/data/natural/MINING_REPORT.md`):

- activation mean/std sample = ClimbMix **shard 310** (5,000 docs)
- probe evaluation pools = ClimbMix **shards 311–316**
- these are disjoint from nanochat training (shards 0–190) and from the
  score-store shards (320–362)

So the probes are already calibrated to ClimbMix, end to end. Nothing needs
re-normalizing for ClimbMix work.

## 2. The two normalizations ("input side" / "output side", said better)

**Step 1 — before the probe (activation standardization).**
Each gemma activation vector h (2304 numbers) is rescaled per dimension:
`(h − mean) / std`, where mean/std were measured once from gemma running over
the 5,000 ClimbMix docs of shard 310. The probe weights were trained on top of
this rescaled input, so this step is part of the probe's definition. Never
refit it — changing it changes what the probe computes.

**Step 1 is PER-LAYER.** mean/std are computed and stored separately for each
layer (`nat_mean`/`nat_std`, shape [2304], one set per layer 6/8/14;
`natstats.py`). This matters — residual-stream norms grow with depth, so a
layer-14 activation standardized with layer-6 stats would be badly scaled.
Within a layer the stats are shared across all concepts (they are a property
of the activation space, not the concept); across layers they are never
shared. Each gold-probe layer file embeds its own `nat_mean`/`nat_std`.

**Step 2 — after the probe (score standardization).**
Each probe outputs one number per token, on its own private scale (see §4 for
why the scales differ). To make scores comparable across probes and storable,
each probe's score is standardized using that probe's own mean and standard
deviation measured over 10.4M ClimbMix tokens (shard 320):
`(score − mean) / std`. The score store and the oracle encoder both live in
these units.

Only step 2 would ever be redone, and only if you deployed the probes on a
different corpus (not ClimbMix) and needed thresholds there.

## 3. "Why 4σ/127 — is σ a standard deviation?"

Yes. σ is that probe's score standard deviation on ClimbMix (from step 2).
Scores are stored as int8 — integers from −127 to +127 — to keep the 2.0B-token
store small. The mapping puts ±4 standard deviations around the mean onto that
integer range:

    stored_int8 = round( (score − mean) / scale )      scale = 4σ/127

    decode:      score = stored_int8 * scale + mean

`mean` and `scale` are stored per probe in `quant.json` (`zero` = mean,
`scale` = 4σ/127); raw per-probe mean/std are also in `corpus_stats.json`.
The decode is exact up to one quantization step.

### The int8 does NOT store integer z-scores (common misread — read this)

The step is `4σ/127 ≈ 0.0315σ`, NOT `σ`. So the integer and the z-score are
related by a factor of ~32, and resolution is ~0.03σ, not 1σ:

    z-score      stored_int8
    ------------------------
      0            0
     +1          ~+32
     +2          ~+64
     +4         +127   (clip point)

A token at 1.5σ is stored as ~48, not "1 or 2". The bulk of the data (within
±2σ) occupies the middle ~half of the range (~−64…+64) = **128 distinct
levels for the common region**. Quantization error is ~scale/√12 ≈ 0.009σ RMS
— far below the ~0.15σ prediction-noise floor, so int8 costs effectively
nothing here. If the scale were `σ` (true integer z-scores) you'd have ~5
usable levels and it WOULD be a bad codec; the `4σ` is exactly the choice that
avoids that, spending the bits on the dense ±2σ region and clipping only the
rare tail beyond ±4σ.

### Zero-centering is both a codec AND principled — and it deletes nothing

- **Codec:** without per-column centering, a probe whose scores sit at
  +37 ± 2 would clip or waste the range encoding a constant offset.
- **Principled:** a probe's raw mean carries no concept info — it's an
  artifact of that probe's weight-norm, bias convention (ridge has a bias,
  difference-of-means has 0), and the fact that ~all corpus tokens are
  negatives. The signal is *deviation from corpus-typical*. You would
  standardize before any multi-probe training/comparison even with unlimited
  storage (otherwise an MSE loss over mixed-scale columns is dominated by the
  large-scale probes).
- **Nothing is discarded:** the per-probe mean is SAVED (`quant.json` `zero`)
  and added back on decode. If a corpus's baseline level is real signal, it
  lives in that stored number — recover it by using the stored mean, not by
  assuming the true mean is 0. Centering only removes ONE global constant per
  probe; all per-token / per-document / per-domain baseline *structure* is in
  the deviations and is fully preserved. A global constant cannot distinguish
  one token from another and is absorbed by any downstream model's bias terms.

### z-score comparability ≠ percentile comparability

After step 2, decoded value `z` means "z standard deviations above this
probe's corpus mean" for EVERY probe — so equal z = equal number of SDs out
(this is what lets the encoder train with one MSE across all columns). But
probe score distributions are heavy-tailed and right-skewed (rare positives,
dense negative lump), and the skew differs by concept, so **equal z is NOT
equal percentile**: z=10 might be p99.9 for one probe and p99.99 for another.
z-scores equalize the first two moments (mean, variance), not skew/tails.
When you need true cross-probe *rank* equivalence (steering doses,
thresholds), use **s95** (95th-percentile score per probe, shipped in the
gold-probe and score-store metadata) — it is explicitly rank-based. Step 2 =
comparable scale; s95 = comparable rank.

## 4. "Arm" — plain translation

Where docs say "arm", read **probe training method**. Every concept has probes
trained three ways on the same data:

- **ridge** — ridge-regression probe
- **DoM** — difference of class means (average activation on concept tokens
  minus average on non-concept tokens)
- **LDA** — linear discriminant analysis

Their weight vectors have different lengths, so their raw scores sit on
different scales — that's why step 2 above exists.

- `gold_probes/probe_set_mixed_detection_l6_l8_l14.npz`: for each
  (concept, layer), the best DETECTOR of the three methods (chosen by AUROC).
- `gold_probes/probe_set_dom_steering_l6_l8_l14.npz`: difference-of-means only,
  because it won the steering/erasure comparisons in stage 6.1.

## 5. s95 (separate from all the above)

s95 is not a normalization of the probes — it's a **dose unit for steering**:
the 95th-percentile score a probe reaches on evaluation text that actually
contains its concept. "Steer at 1–2× s95" means "push the score to roughly its
strong-natural-occurrence level". It is a project convention, not a field
standard (the field has no single standard; percentile-of-max scaling like
this is common in SAE work, and steering papers often just sweep raw
coefficients).

## 6. Does step-2 normalization clash with nanochat's residual-stream units?

First, what gets injected: not the 54 probe scores directly, but a
**14-number summary per token** (called "coords" in the code). The six cyclic
concept families (months, weekdays, seasons, moon phases, color wheel,
compass directions) are each encoded as a position on a circle — 2 numbers
per family, like a clock hand — and continents get 2 numbers from PCA:
6×2 + 2 = 14. Below, `coords` means this 14-number vector.

The units question: no clash — by construction the probe/score units never
reach nanochat. The injection site (`nanochat_patch/gpt_inject.diff`, after
block 7) does, per token:

    rms_x = RMS of nanochat's own residual vector at this token
    rms_z = RMS of zc = P @ coords, the 14-number summary mapped into
            nanochat's 1536-dim space (P: fixed orthonormal 1536x14)
    x = x + beta * (rms_x / rms_z) * zc          # beta = 0.064

Dividing by `rms_z` erases whatever units the coords arrived in; multiplying
by `rms_x` re-expresses the vector in nanochat's units at that exact token.
So the injected vector is always exactly 6.4% of the residual's own RMS —
whether nanochat's activations are small at init or grow 10x during training,
the injection tracks them. That is what makes it "in distribution" in
magnitude: it is defined relative to the model's own live statistics, never
in gemma/probe units.

What survives from the probe pipeline is only the DIRECTION of the 14-number
summary (the pattern across its 14 entries), not its overall size — any
nonzero summary is renormalized to full beta amplitude. Two consequences:

- Step-2 standardization is actually what makes that direction meaningful:
  all 14 dims are on unit variance over ClimbMix, so no dim dominates the
  direction by a units artifact.
- All-zero coords (doc missing from the store, or no concept signal) give
  exactly zero injection — a true no-op, no noise (the loader deliberately
  skips adding noise to zeros, since renormalization would blow pure noise
  up to full amplitude).

Remaining in-distribution safeguards: the 14-dim target subspace is a fixed
random orthonormal basis (statistically indistinguishable from any other
direction at init); coords carry deterministic per-doc noise (sigma = 0.15)
so the channel isn't a noiseless oracle; beta was set to 0.064 = 0.05/sqrt(0.61)
so the TRUE-signal component is ~5% of residual RMS after oracle fidelity;
and gate G4 kills the run if loss diverges >5% bpb from baseline by 2k steps.
