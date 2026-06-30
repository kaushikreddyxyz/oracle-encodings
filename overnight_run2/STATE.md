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
- 04:3x Phase A pipeline built+validated (season+costliness gates pass). Phase B (train_probes.py) + Phase C (attribute_corpus.py) built+self-tested. Committed f11878a, 2b809f2.
- ~04:44 INCIDENT: OpenRouter account had $0 credits (402). User added $100 (now ~$90). n=3 batching unsupported by providers (tested). Resolved.
- ~05:10 Per-call latency ~20s on 70B models => 120-conc pace ~5-6h. Bumped to 240 concurrency.
- ~05:12 SCALE CUT: user wanted results by 9AM (08:00Z) => Phase A reduced 4000->**600 pos/value** @ 240 conc (~30min). GPU runner subagent a6ce5cb4158507932 spun **H100 izeqdzii2z4owj** (216.243.220.230:16919, $3.29/hr) for B+C.
- ~05:25 DEADLINE RELAXED ("take your time"); user asleep. New target: **>=30% shards (>=56 of 187) for Phase C**. Ceilings: 12:00Z wall / $550 spend / $80hr. Plan: keep A@600, B on the H100, **Phase C across 3-4 parallel H100 pods** to hit >=56 shards (0-185 prioritized).
- 05:43Z **Phase A COMPLETE**: 16/16 ALL_PASS, datasets pushed to hf `concept-probes-v2-datasets` (verified 16 jsonl). ~31min wall, ~$4 OR. Runner resumed -> running **Phase B** on H100 izeqdzii2z4owj.
- NEXT: on runner "phase B done" -> fan out 2-3 H100 pods for **Phase C** shards 16-63+ (target >=56 = 30%); runner does 0-15. Ceilings 12:00Z / $550 / $80hr.
- ~06:55Z **Phase B COMPLETE** on hf `concept-probes-v2-weights` (12 npz + summary.json). **47/47 binary rows test AUROC>=0.8, 42/47 >=0.9, median 0.954**; scalars Spearman 0.61-0.80; lexical_gap median 0.198. Skip-rate 0%. CAVEAT: month-name + color_hue rows most lexical (gap up to 0.45 — color_hue::Yellow 0.449, month::May 0.447) -> treat their attribution with caution.
- ~07:10Z **Phase C OOM on shard 0** (padding blowup: MAX_BATCH_DOCS=64 -> 64x1024 tok/fwd + all-43-layer hidden states ~70GB). FIX (no code change): **MAX_BATCH_DOCS=16 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True** (peak ~28GB, keep eager). Runner + 3 helpers relaunched 0-15/16-31/32-47/48-63 with fix. First shards ~07:26-07:30. 4 H100 pods running (izeqdzii2z4owj/s4e2ychl7xz7vb/dbx39sk3oa65bl/fq0u8qvpqfx4vp).
