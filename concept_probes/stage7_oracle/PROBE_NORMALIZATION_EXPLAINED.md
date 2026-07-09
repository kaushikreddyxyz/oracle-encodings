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

    stored_int8 = round( (score − mean) / (4σ/127) )

So one int8 step = 4σ/127 ≈ 0.03σ of resolution, and scores beyond ±4σ clip
to ±127. That's all "(4σ/127)" is: the size of one integer step.

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

No — by construction the probe/score units never reach nanochat. The
injection site (`nanochat_patch/gpt_inject.diff`, after block 7) does, per
token:

    rms_x = RMS of nanochat's own residual vector at this token
    rms_z = RMS of the projected coord vector zc = P @ coords   (P: fixed
            orthonormal 1536x14)
    x = x + beta * (rms_x / rms_z) * zc          # beta = 0.064

Dividing by `rms_z` erases whatever units the coords arrived in; multiplying
by `rms_x` re-expresses the vector in nanochat's units at that exact token.
So the injected vector is always exactly 6.4% of the residual's own RMS —
whether nanochat's activations are small at init or grow 10x during training,
the injection tracks them. That is what makes it "in distribution" in
magnitude: it is defined relative to the model's own live statistics, never
in gemma/probe units.

What survives from the probe pipeline is only the DIRECTION of the 14-dim
coord vector (the pattern across dims), not its overall size — any nonzero
coord vector is renormalized to full beta amplitude. Two consequences:

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
