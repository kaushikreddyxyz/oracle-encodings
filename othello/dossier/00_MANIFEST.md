# Othello-GPT Dossier — Manifest

**What this is:** A self-contained background knowledge base on Othello-GPT and its
linear "mine/theirs" world representation. Pure reference material — it contains
*facts about the model and the literature*, not experimental decisions, task
definitions, or success targets (those live in your per-run brief).

**How to use it:** Don't read all files into context at once. Use the routing table
below to open the one file that answers your current question. Each file is short and
independently loadable.

---

## Routing table — which file answers which question

| If you need to know…                                                                 | Open                       |
|--------------------------------------------------------------------------------------|----------------------------|
| The high-level picture: what Othello-GPT is, what the result is, why it matters       | `01_executive_summary.md`  |
| A specific number/config: model dims, vocab, probe hyperparams, layer, move range, loading API, library version pins | `02_facts_and_config.md`   |
| A concept explained: mine/theirs, the parity flip, what "linear feature" means, the per-cell geometry, superposition implications | `03_concepts.md`           |
| Coordinate systems and conventions: token ↔ board-index ↔ (row,col) ↔ notation, the mine/theirs frame, a worked ground-truth trace | `04_conventions_and_trace.md` |
| Things that silently break your work — read before writing any pipeline code          | `05_gotchas.md`            |
| Where to find papers, notebooks, repos, and what each is *for*                         | `06_resources.md`          |
| The papers themselves                                                                  | `papers/` (+ `papers/README.md` for fetch instructions) |

---

## One-paragraph recap of each file

- **01_executive_summary.md** — The whole story in one read: Li et al. trained a GPT to
  predict legal Othello moves; it learned an emergent board representation; they found it
  *non*-linearly decodable. Nanda showed it's actually *linear* in a player-relative
  ("mine/theirs") basis, and causally interventable. Why this is a clean substrate.

- **02_facts_and_config.md** — The lookup table. Every load-bearing number: 8 layers,
  d_model 512, 8 heads, ctx 60, vocab 61; probe = 60 parallel 3-way logistic regressions
  on resid_post; Nanda's probe hyperparams; the model-loading API and the **critical
  TransformerLens v3.0 version landmine**. Go here for "what's the value of X."

- **03_concepts.md** — The "why" behind the facts. The mine/theirs reframing and why the
  task forces it; the parity sign-flip; what counts as a genuinely linear feature; the
  ~2-directions-per-cell geometry (mine/theirs axis + blank axis); over- vs.
  under-parametrization and superposition implications.

- **04_conventions_and_trace.md** — The highest-frequency source of silent bugs:
  coordinate and indexing conventions, and ONE fully worked ground-truth trace
  (game string → tokens → board state → probe target) usable as a test fixture.

- **05_gotchas.md** — A pre-flight checklist of verified failure modes. Read it before
  building a data pipeline or a probe. Each item is a thing that produces plausible-looking
  but wrong results with no error message.

- **06_resources.md** — Annotated, by purpose: which repo is the data/board engine, which
  notebook is the analysis toolkit, which papers establish which claim.

---

## Provenance note

Facts in this dossier were verified against primary sources (Nanda's blog post; the
Li et al. paper abstract/repo; TransformerLens docs and the Othello demo notebook
metadata) as of June 2026. Where a fact could not be verified to primary source, it is
flagged inline as **[UNVERIFIED]**. The TransformerLens ecosystem is actively changing
(v3.0 was a breaking change) — treat API specifics in `02` as "check against the installed
version," and see `05_gotchas.md`.
