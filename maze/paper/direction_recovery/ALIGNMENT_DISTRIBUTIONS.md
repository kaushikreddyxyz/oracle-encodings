# Alignment distributions

The final sweep added 960 independently validated fits. Combined with the
original fits, each method has 144 fits across 24 underlying checkpoints and
14,880 one-to-one matched cosine observations. All six maze sizes receive
equal weight.

## Retained figures

The [main violin plot](figures/alignment_violins.svg) compares original-graft
absolute cosine after optimal one-to-one matching. Unmatched SAE latents are
excluded. Dots mark weighted empirical means; inner bars span the middle 50%.
The right column gives the fraction reaching cosine 0.8.

The [all-active-directions view](figures/supplementary/alignment_violins_all_directions.svg)
compares each active latent or probe with its nearest graft direction,
allowing repeated matches. It answers a different question from unique
feature coverage. Only the violin figures are retained, as PDF and SVG.
Superseded plotting iterations and their dedicated scripts and summaries
have been deleted.

Reproduce the final plots with:

```sh
.venv/bin/python maze/paper/direction_recovery/plot_alignment_violins.py
```

The shared helpers in `plot_alignment_distributions.py` reconstruct the
original cosine samples from saved checkpoints if their CSV is absent.

## Weighting and interpretation

Each model receives equal total weight, each fit within a model equal weight,
and each direction within a fit equal weight. Thus larger mazes do not
dominate merely because they contain more edges. No ground-truth mean
centering is applied.

Violins use a Gaussian bandwidth of 0.015 cosine units, reflection at the
0 and 1 boundaries, and trimming to each method's observed range. Widths
are scaled separately to reveal distribution shape. Means, quantiles, and
fractions are computed from the unsmoothed weighted samples. These are
alignment distributions, not confidence intervals.

Extra fits vary initialization and bootstrap the existing training corpora;
they do not represent independently trained additional maze transformers.
Directions and repeated fits from a checkpoint are dependent.

## Data and validation

`alignment_distribution_values.csv` holds the original samples and
`expanded_alignment/alignment_distribution_values.csv` the added fits.
The matching indices and all-active nearest matches are retained. Original
reconstruction checks are in `alignment_distribution_validation.json`, and
expanded checks in `expanded_alignment/validation.json`. The latter verifies
active latent IDs, full matching coverage, bootstrap seeds, cosine values,
and the archived checkpoint and download hashes.

All six campaign pods were terminated. Receipts and estimated compute costs
are recorded in `expanded_alignment/resource_audit.json`.
