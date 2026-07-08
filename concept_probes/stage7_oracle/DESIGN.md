# Stage 7-Oracle DESIGN — frozen interfaces for parallel implementation

Companion to SPEC.md (the contract). This file freezes data formats and APIs
so independently-written modules compose. Written by the orchestrator
2026-07-08 ~2:05 AM. Change only via the orchestrator.

## File layout

```
concept_probes/stage7_oracle/
  SPEC.md  DESIGN.md  STATE.md
  code/
    select_probes.py      # Phase 0 (CPU, local)
    align.py              # tokenizer bridge (both modes) + tests
    score_corpus.py       # Phase 1 pod scoring
    bench_gemma.py        # sdpa-vs-eager throughput + parity
    train_encoder.py      # Exp A (+ Exp B via --mode)
    verify_closed_form.py # Exp B v* live-teacher check
    pod_setup.sh          # adapted from stage6_1
  out/
    probe_set.json  probe_set_arrays.npz  selection_table.md
    bench/  corpus_stats.json  ...
```

Score memmaps live on pod NVMe under `/workspace/scores/` (+ HF archival).

## probe_set.json (Phase 0 output)

```json
{
  "layers": [8, 12, 16],            // the 3 chosen gemma layers (probe layer l reads hidden_states[l+1])
  "ablation_layer": 12,             // Exp-B layer (causal-salient consensus)
  "concepts": ["january", "..."],   // K names, ORDER IS CANONICAL everywhere
  "families": {"january": "months", "...": "..."},
  "selection": { "<layer>": { "<concept>": {"arm": "ridge", "auroc": 0.97, "token_rho": 0.55} } },
  "s95":   { "<layer>": { "<concept>": 3.1 } },   // 95th-pct natural score, chosen arm
  "corpus_stats": null              // filled by Phase 1: per-column mean/std on scoring corpus
}
```

`probe_set_arrays.npz` (companion, all float32):
- `W` [3, K, 2304] — chosen-arm direction per (layer, concept), in
  STANDARDIZED space (i.e. score = w·((h−nat_mean)/nat_std)+b), NOT unit-norm
  — keep the arm's native scale so scores match natscores conventions.
- `b` [3, K]
- `nat_mean`, `nat_std` [3, 2304] — per chosen layer (natscores stats)
- `W_dom_abl` [K, 2304] — DoM directions at ablation_layer (std space)
- `b_dom_abl` [K], `t_nat_dom` [K] — natural-pool mean DoM score (ablation target)
- `G_dom` [K, K] — Gram matrix of the RAW-space dom directions
  d_c = nat_std ⊙ W_dom_abl[c] (see SPEC Phase 3); also `G_dom_inv`.
- `layer_index` [3] — same as json layers, for safety.

## Score store (Phase 1 output, per shard)

Docs are ClimbMix shards ≥320, each doc TRUNCATED to first 2048 gemma tokens
(single window, no chunking — keeps alignment trivial). Skip docs < 64 gemma
tokens.

Per ClimbMix shard `<sid>` (one pair of files + one index row per doc):
- `tokens_<sid>.npy` — int32 [N_total] gemma token ids, docs concatenated.
- `scores_<sid>.npy` — int8 [N_total, 4K] where columns are
  `[layer0 concepts 0..K-1, layer1 ..., layer2 ..., dom@ablation_layer 0..K-1]`
  (4K columns total; if ablation_layer's chosen arm IS dom for a concept the
  dom column is still stored — uniform layout beats cleverness).
- `docs_<sid>.jsonl` — one row per doc:
  `{"doc": <index-within-shard>, "start": <token offset in memmap>, "n": <n_tokens>}`
  Raw text is NOT stored — re-derived at training time from the ClimbMix
  shard by doc index (tokenization is deterministic; encoder side re-reads
  text, re-tokenizes gemma to recover char offsets).

Quantization: per column, `int8 = clip(round((score - zero)/scale), -127, 127)`
with `zero = mean`, `scale = 4*std/127`, calibrated on the first 10M tokens;
stored in `quant.json` `{ "zero": [4K], "scale": [4K] }` (shared fleet-wide —
pod A calibrates, others download). Streaming true mean/std per column also
accumulated per shard → merged into `corpus_stats.json`.

## align.py API

```python
gemma_to_qwen_map(text: str, gemma_offsets, qwen_offsets, mode="prefix") -> np.ndarray  # [Tg] int
```
For each gemma token t: index of the last qwen token whose char span ends
<= end_char(t); -1 if none yet (drop those positions from the loss).
`offsets` = list of (start, end) from HF fast tokenizers
(`return_offsets_mapping=True`). Also:
```python
crossing_rate(text, gemma_offsets, qwen_offsets) -> float  # frac of gemma tokens with no exactly-matching qwen end boundary
```
Mode "boundary" (fallback): per gemma token, re-tokenize its substring with
qwen and mean-pool — returns list of qwen-subtoken index lists instead.
Same module must work qwen→nanochat later (it's tokenizer-agnostic: takes
offset maps only).

## Exp A/B training data flow (train_encoder.py)

Batch unit = doc. For each doc: load text from ClimbMix shard (doc index from
docs_<sid>.jsonl) → qwen tokenize (max 3072 tokens; if qwen tokenization of
the gemma-truncated char span exceeds that, truncate BOTH to the shorter char
prefix) → forward Qwen3-0.6B-Base (bf16) → hidden [Tq, 1024] → gather at
`gemma_to_qwen_map` indices → head → loss vs dequantized scores
standardized by corpus_stats.

Heads:
- Exp A: `up = nn.Linear(1024, 3K)`; loss MSE on standardized scores (3K cols).
- Exp B: target `v* = D · G_inv · (s_dom − t_nat_dom)` computed on the fly
  from the dom columns (dequantized, RAW score units, D = raw-space dom
  dirs); variant (i) predict `y = G_inv (s−t)` [K] with fixed decoder D;
  variant (ii) `up(1024→K)` then learnable `down(K→2304)`, loss on v* itself.

Report: per-probe R² train/heldout (heldout = distinct shards), median +
per-family medians. Early stop: heldout median R² Δ < 0.005 over last 20% of
steps.

## Pod conventions (from stage6_1, binding)

- H100 SXM SECURE, template `runpod-torch-v240`, disk ≥ 200GB.
- Direct SSH; port from `runpodctl pod get` (creation-time port can be stale).
- Tokens (HF/GH) via stdin pipe ONLY — never argv. See stage6_1/code/pod_setup.sh.
- hf CLI: ONE `--include` pattern per call.
- Long scripts: tqdm + heartbeat file (`/workspace/hb_<name>.txt` touched
  every 60s with step + throughput) so the orchestrator can monitor cheaply.
- gemma-2-2b is GATED: pod needs the HF token before `from_pretrained`.
- Monitors run from macOS zsh: no bash arrays; parse with awk.
