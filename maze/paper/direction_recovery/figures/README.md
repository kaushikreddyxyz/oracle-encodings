# Alignment violin figures

Only the final violin figures are retained.

- **Optimal one-to-one matching:** [PDF](alignment_violins.pdf) · [SVG](alignment_violins.svg).
- **Every active learned direction, matched to its closest graft feature:** [PDF](supplementary/alignment_violins_all_directions.pdf) · [SVG](supplementary/alignment_violins_all_directions.svg).

Both use original-graft absolute cosine, 144 fits per method, and equal model
and fit weights across maze sizes 5–10. Dots mark means, inner bars show the
middle 50%, and violin widths are scaled separately per method.

Reproduce from the repository root:

```sh
.venv/bin/python maze/paper/direction_recovery/plot_alignment_violins.py
```
