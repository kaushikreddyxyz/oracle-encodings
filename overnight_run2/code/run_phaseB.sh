#!/usr/bin/env bash
# ============================================================================
# Phase B pod runbook — train & evaluate concept probes on frozen gemma-2-9b.
#
# Assumes: a fresh RunPod GPU pod with the oracle-encodings repo checked out and
# the project venv present. Pulls the Phase-A datasets from the HF dataset repo,
# runs train_probes.py, and pushes the probe weights + summary.json to the HF
# model (weights) repo.
#
# REQUIRES (env):
#   HF_TOKEN  — a Hugging Face token with (a) read access to the GATED
#               google/gemma-2-9b model, and (b) write access to the weights repo.
#               huggingface_hub reads HF_TOKEN automatically. NEVER printed.
#
# OPTIONAL (env):
#   PY                 — python interpreter (default: repo .venv, else python3)
#   DATASETS_REPO      — HF dataset repo id (default from state.json)
#   WEIGHTS_REPO       — HF model repo id   (default from state.json)
#   DEVICE             — cuda | cpu (default cuda)
#   EPOCHS LR BATCH_SIZE STORE_DTYPE MAX_PRE_PER_SEQ KEEP_CACHE
#
# Usage:   HF_TOKEN=hf_xxx bash run_phaseB.sh
# ============================================================================
set -euo pipefail

# --- GPU hygiene: cap CPU thread pools (handoff §5) -------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# --- locate paths -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # .../overnight_run2/code
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                          # .../overnight_run2
REPO_ROOT="$(cd "${RUN_DIR}/.." && pwd)"                           # oracle-encodings

if [[ -z "${PY:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then PY="${REPO_ROOT}/.venv/bin/python3";
  else PY="python3"; fi
fi
echo "[phaseB] python: ${PY}"
"${PY}" --version

# --- config (defaults match overnight_run2/state.json) ----------------------
DATASETS_REPO="${DATASETS_REPO:-kaushikreddyxyz/concept-probes-v2-datasets}"
WEIGHTS_REPO="${WEIGHTS_REPO:-kaushikreddyxyz/concept-probes-v2-weights}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-5e-3}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
STORE_DTYPE="${STORE_DTYPE:-float16}"
MAX_PRE_PER_SEQ="${MAX_PRE_PER_SEQ:-}"     # empty => keep all pre-span negatives
KEEP_CACHE="${KEEP_CACHE:-}"               # set to 1 to keep per-layer activation memmaps

DATA_DIR="${RUN_DIR}/data_pulled"
OUT_DIR="${RUN_DIR}/phaseB_out"
mkdir -p "${DATA_DIR}" "${OUT_DIR}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[phaseB] ERROR: HF_TOKEN is not set (need gemma-2-9b read + weights-repo write)." >&2
  exit 2
fi

# --- 1) pull Phase-A datasets from the HF dataset repo ----------------------
echo "[phaseB] pulling datasets from ${DATASETS_REPO} ..."
"${PY}" - "$DATASETS_REPO" "$DATA_DIR" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest,
                         allow_patterns=["*.jsonl"])
print("[phaseB] datasets downloaded to", path)
PYEOF

# flatten: train_probes globs <data-dir>/*.jsonl
find "${DATA_DIR}" -name '*.jsonl' -not -path "${DATA_DIR}/*.jsonl" -exec mv -t "${DATA_DIR}" {} + 2>/dev/null || true
N_FILES="$(find "${DATA_DIR}" -maxdepth 2 -name '*.jsonl' | wc -l | tr -d ' ')"
echo "[phaseB] found ${N_FILES} *.jsonl dataset files"
if [[ "${N_FILES}" -eq 0 ]]; then echo "[phaseB] ERROR: no datasets pulled" >&2; exit 3; fi

# point at the dir that actually holds the jsonl files
GLOB_DIR="$(dirname "$(find "${DATA_DIR}" -name '*.jsonl' | head -1)")"

# --- 2) train + evaluate ----------------------------------------------------
echo "[phaseB] training probes (device=${DEVICE}, epochs=${EPOCHS}, bs=${BATCH_SIZE}) ..."
EXTRA=()
[[ -n "${MAX_PRE_PER_SEQ}" ]] && EXTRA+=(--max-pre-per-seq "${MAX_PRE_PER_SEQ}")
[[ -n "${KEEP_CACHE}" ]] && EXTRA+=(--keep-cache)

"${PY}" "${SCRIPT_DIR}/train_probes.py" \
  --data-dir "${GLOB_DIR}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --store-dtype "${STORE_DTYPE}" \
  "${EXTRA[@]}"

if [[ ! -f "${OUT_DIR}/summary.json" ]]; then
  echo "[phaseB] ERROR: summary.json not produced" >&2; exit 4
fi
echo "[phaseB] training complete. summary.json + weights/ in ${OUT_DIR}"

# --- 3) push weights + summary.json to the HF weights repo ------------------
echo "[phaseB] uploading weights + summary.json to ${WEIGHTS_REPO} ..."
"${PY}" - "$WEIGHTS_REPO" "$OUT_DIR" <<'PYEOF'
import sys, os
from huggingface_hub import HfApi
repo, out_dir = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo_id=repo, repo_type="model", exist_ok=True, private=False)
# upload summary.json and the per-layer weight npz files (+ labels.npz if present)
api.upload_folder(
    folder_path=out_dir, repo_id=repo, repo_type="model",
    allow_patterns=["summary.json", "weights/*.npz", "labels.npz"],
    commit_message="Phase B: concept-probe weights + summary.json",
)
print("[phaseB] upload complete ->", repo)
PYEOF

echo "[phaseB] DONE."
