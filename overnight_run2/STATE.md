# STATE — concept-probes-v2 rectification run

Start 2026-06-30T03:22Z · 9h deadline 2026-06-30T12:22Z · budget $650 / $80-hr / stop-launch $550.
Authoritative: `knowledge/run_contract.md` + `probe_rectification_handoff.md` + the two specs.
Resume rule: on startup read this + state.json; skip any unit marked `done` with an HF path.

Models: gen=`qwen/qwen-2.5-72b-instruct`, judge=`meta-llama/llama-3.3-70b-instruct` (distinct, >25B). Probe target=`google/gemma-2-9b`.
Layers (12): [1,2,6,10,15,19,24,28,33,37,41,42]. Shards: 0–185 + 6542. Storage: sparse firings + gated-tail mean/std.

## Phase A — datasets (16 concept files → 57 rows), 4k pos/class
- [ ] month (12) · [ ] weekday (7) · [ ] color_hue (12) · [ ] moon_phase (8) · [ ] season (4) · [ ] compass (4)
- [ ] costliness · [ ] size · [ ] europe · [ ] america · [ ] africa · [ ] indoors · [ ] outdoors · [ ] lovingness · [ ] harmfulness · [ ] duration
- push each to HF dataset repo `concept-probes-v2-datasets` as it finishes.

## Phase B — probes (57 rows × 12 layers)
- [ ] extract gemma-2-9b activations @ 12 layers · [ ] fit rows · [ ] summary.json (test + held-out-vocab + lexical_gap) · [ ] push weights → `concept-probes-v2-weights`

## Phase C — attribution (shards 0–185 + 6542)
- [ ] per-shard sparse firings + gated-tail stats · [ ] manifest.json · push each shard → `concept-probes-v2-attribution`

## Log
- 03:22Z preflight PASS (HF/RunPod/OpenRouter auth ok; gemma tokenizer + ClimbMix shards + Qwen JSON span verified). Phase A needs no GPU.
