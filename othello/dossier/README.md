# Papers — fetch instructions

The two core papers are on arXiv. They were **not** bundled into this directory because the
build environment's network allowlist blocked `arxiv.org`. Fetch them in one step from an
environment with network access:

```bash
cd "$(dirname "$0")"   # this papers/ directory

# Li et al. 2022 — Emergent World Representations (original Othello-GPT)
curl -L -o li_et_al_2210.13382.pdf "https://arxiv.org/pdf/2210.13382"

# Nanda, Lee & Wattenberg 2023 — Emergent Linear Representations
curl -L -o nanda_lee_wattenberg_2309.00941.pdf "https://arxiv.org/pdf/2309.00941"

# (optional) Nanda et al. 2023 — Progress Measures for Grokking
curl -L -o nanda_grokking_2301.05217.pdf "https://arxiv.org/pdf/2301.05217"
```

Abstract / HTML landing pages (if you want metadata or can't fetch PDFs):
- Li et al.: https://arxiv.org/abs/2210.13382
- Nanda/Lee/Wattenberg: https://arxiv.org/abs/2309.00941
- Grokking: https://arxiv.org/abs/2301.05217

## What's already captured without the PDFs

The single densest source — **Nanda's blog post** — is saved locally as
`nanda_blog_othello.md` in this directory (it is not on arXiv). The load-bearing facts from
both papers are already distilled into `../02_facts_and_config.md`, `../03_concepts.md`, and
`../05_gotchas.md`, so the dossier is usable even before the PDFs are fetched. Pull the PDFs
when you need figures, tables, or exact wording from the papers themselves.
