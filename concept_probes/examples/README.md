# concept_probes/examples

Minimal, runnable demo for the pipeline's key published probe artifact.
Deps: `numpy`, `huggingface_hub`, `transformers`, `torch`.
`google/gemma-2-2b` is gated on HF — set `HF_TOKEN`.

## `score_text_with_probes.py`

Loads the 54 gold detection probes (trained in `2_probes`, certified in
`3_validation`/`4_causal`, frozen for the corpus attribution scan) from
[concept-probes-gemma2-2b](https://huggingface.co/kaushikreddyxyz/concept-probes-gemma2-2b)
and scores each token of a sentence with gemma-2-2b (eager attention,
`hidden_states[layer+1]`, BOS dropped). ~5 GB model download; runs on CPU.

## Moved

`read_corpus_scores.py` (reading the pre-scored ClimbMix corpus-score store)
now lives at `attribution/examples/read_corpus_scores.py` — corpus-score
store demos belong with the attribution pipeline (repo-root `attribution/`).
