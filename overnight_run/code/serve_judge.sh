#!/usr/bin/env bash
# serve_judge.sh — launch a vLLM OpenAI-compatible server for the Step-1 judge.
#
# Talks the OpenAI /v1/chat/completions API on $PORT so label.py's VLLMJudgeClient
# (httpx -> http://localhost:8000/v1) works unchanged. Sized for ONE 80GB card
# (A100/H100). DO NOT run locally (no GPU; gemma is gated). Run this ON THE POD.
#
# Model notes:
#   * google/gemma-3-27b-it  : the primary judge. bf16 weights ~54GB -> fits one
#     80GB card with room for KV cache. REQUIRES a recent vLLM (gemma3 arch support
#     landed in vLLM >=0.8.x; `pip install -U vllm` on the pod). It is license-gated,
#     so `huggingface-cli login` (or HF_TOKEN env) with an accepted license first.
#   * Qwen/Qwen2.5-32B-Instruct-AWQ : non-gated fallback. The "-AWQ" suffix is
#     auto-detected below and adds `--quantization awq` (4-bit, faster, ~20GB).
#
# Override anything via env: JUDGE_MODEL, PORT, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION,
# MAX_NUM_SEQS, DTYPE. Extra raw vLLM flags can be appended as "$@".
set -euo pipefail

JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-3-27b-it}"   # keep in sync with config.JUDGE_PRIMARY
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"                # judge prompts are short; 4k is ample
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"                   # high concurrency: many tiny requests
DTYPE="${DTYPE:-bfloat16}"

# AWQ auto-detection: if the repo id ends in -AWQ / -awq, serve it quantized.
QUANT_ARGS=()
case "$JUDGE_MODEL" in
  *-AWQ|*-awq) QUANT_ARGS=(--quantization awq) ;;
esac

echo "[serve_judge] model=$JUDGE_MODEL port=$PORT max_len=$MAX_MODEL_LEN dtype=$DTYPE quant=${QUANT_ARGS[*]:-none}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$JUDGE_MODEL" \
  --served-model-name "$JUDGE_MODEL" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  "${QUANT_ARGS[@]}" \
  "$@"
