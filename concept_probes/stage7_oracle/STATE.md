# Stage 7-Oracle live progress

(newest first)

- 2026-07-08: `code/train_encoder.py` DONE (Phase 2 Exp A + Phase 3 Exp B,
  all 3 modes). Smoke test `code/test_train_encoder.py`: 11/11 PASS (tiny
  random Qwen2Config encoder, real gemma-2-2b + Qwen3-0.6B-Base tokenizers,
  synthetic score-store fixture matching the DESIGN.md schema exactly —
  verified against the REAL `out/probe_set.json`/`probe_set_arrays.npz`
  Phase-0 produced too, K=54/layers=[6,8,14]/ablation_layer=8 load and
  `v* = D·G_inv·(s-t)` compute cleanly). Picks up the real `align.py` (now
  present; `_align_fallback.py` kept as a dead-but-harmless fallback, API-
  compatible, unused since `align.py` imports first). Qwen3-0.6B-Base is
  NOT gated — tokenizer+config load fine offline from cache.
  **Cross-checked the gemma tokenization convention against the real
  score_corpus.py (written by a parallel agent) and fixed a mismatch**:
  score_corpus.py uses `tok(text, add_special_tokens=False)["input_ids"]`
  sliced (not tokenizer-truncated) to `MAX_DOC_TOKENS=2048` — BOS is never
  in the stored ids (manually prepended + its hidden row dropped around the
  forward pass). train_encoder.py's `process_doc` was updated to match
  exactly (was previously assuming `add_special_tokens=True`); the smoke
  test's fixture builder was updated the same way and re-verified green.
  `MIN_DOC_TOKENS=64` already matched. The token-id reproduction hard-assert
  (first `--assert-first-n-docs` train docs) will still hard-fail loudly if
  any further drift exists (e.g. a different gemma tokenizer revision via
  `--gemma-model`), by design.

- 2026-07-08: Phase 0 DONE, gate **G0 = GO**. `code/select_probes.py` (CPU,
  local, $0) scored all 64 concepts x 12 gemma layers x 4 arms
  (ridge/dom/lda/logistic) on the natural TEST half (example-level max-pool
  AUROC, label ymax>=0.34; token Spearman for audit). Chosen layers
  **[6, 8, 14]** (mean best-arm AUROC 0.9629/0.9645/0.9601; data-driven —
  differs from SPEC's speculative {8,12,16} guess but satisfies the
  causal-band + non-adjacent-spread tie-break rules; all layer scores were
  within ~0.005 of each other, i.e. nearly flat, since probe quality is
  uniformly high). **K=54 survivors** (AUROC>=0.90 at all 3 layers
  simultaneously) out of 64 — no relax fallback needed (>=20 threshold).
  Dropped: all 5 intensity scalars (costliness, duration, harmfulness,
  lovingness, physical_size — expected per SPEC), both location_type
  concepts (indoors, outdoors — new, family fully eliminated), and 3
  color_wheel blends (blue-violet, red-violet, yellow-orange). Note:
  blue-violet, yellow-orange and outdoors were Stage-6 "deploy" tier but
  fail Phase-0's stricter all-3-layers-simultaneously bar — not a
  regression, just a stricter joint constraint. **Ablation layer = 8**
  (mode of causal_cards `e5_salient_layer_corrected` over the 54 survivors;
  histogram {6:1, 8:22, 10:4, 12:17, 14:4, 16:4, 18:2} — matches SPEC's
  expected L8-or-L12). Axis-convention verification (natscores/probes W,b,
  lambda, class-index, layer-index reading) cross-checked against
  stage6_1's independently-computed `dose_calib.json` for 2 concepts:
  **exact match, 0.00e+00 relative error** on both t and s95 (no raw gemma
  hidden states are available locally to fully re-derive from scratch —
  this is the strongest available loop closure; see
  `select_probes.py::verify_axes` docstring for the reasoning). Outputs:
  `out/probe_set.json`, `out/probe_set_arrays.npz`, `out/selection_table.md`
  (full 64-concept x 3-layer x 4-arm audit table). Not committed to git.

- ~2:15 AM: align.py DONE (42/42 tests pass, real gemma+qwen tokenizers).
  Measured gemma→qwen crossing rate 7.08% < 10% threshold → prefix-state
  mode confirmed as primary bridge, no boundary fallback needed. Other 4
  agents (Phase 0, bench pod, score_corpus, train_encoder) still running.

- 2026-07-08 ~2:00 AM: post-compaction instance online. User Q&A before
  sleep resolved two decisions, spec updated: (1) marginal G2 (R² 0.4–0.6)
  → LAUNCH nanochat anyway with caveat; (2) BOTH baselines (VE and no-VE)
  already exist on HF — train NO baseline tonight, one injected no-VE run
  only, matched exactly to the existing no-VE baseline's recipe/token
  budget (no shortening; if unaffordable, hold launch for user).
  Beginning execution: Phase 0 (sonnet subagent) + throughput/parity
  benchmark pod.

- 2026-07-08 1:50 AM: SPEC.md written by pre-compaction instance (amendments
  to the user's proposal documented inline). Nothing executed yet. Next
  instance: read SPEC.md top-to-bottom, then start Phase 0 + the throughput
  benchmark immediately. Budget ≤$400 target / $550 hard; delegate to
  sonnet-model subagents aggressively; user wakes 11 AM.
