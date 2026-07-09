# Probe normalization, explained plainly

*(2026-07-09; corrects a confusing earlier explanation)*

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
