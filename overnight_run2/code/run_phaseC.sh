#!/usr/bin/env bash
# ============================================================================
# Phase C pod runbook — corpus attribution over ClimbMix shards 0-185 + val 6542.
#
# Sweeps the 12-layer concept probes over each shard, per token, stores SPARSE
# firings + gated-tail stats, and uploads to the HF attribution dataset repo
# PER SHARD (then purges local files). Fully resumable: an already-uploaded shard
# is skipped, so re-running and crash-recovery are free.
#
# PARALLEL ACROSS PODS: pass a disjoint SHARD_RANGE per pod (the script just forwards
# it to attribute_corpus.py --shards). Each pod uploads to the same repo; per-shard
# filenames + per-shard manifest fragments avoid collisions. The global manifest.json
# is written ONCE (WRITE_MANIFEST=1, run that on exactly one pod / the first pod).
#
#   Example: split 187 shards across 4 pods (~47 each), pod 1 also writes the manifest:
#     pod1:  WRITE_MANIFEST=1 SHARD_RANGE="0-46"       HF_TOKEN=hf_xxx bash run_phaseC.sh
#     pod2:                   SHARD_RANGE="47-92"      HF_TOKEN=hf_xxx bash run_phaseC.sh
#     pod3:                   SHARD_RANGE="93-139"     HF_TOKEN=hf_xxx bash run_phaseC.sh
#     pod4:                   SHARD_RANGE="140-185,6542" HF_TOKEN=hf_xxx bash run_phaseC.sh
#
# REQUIRES (env):
#   HF_TOKEN  — HF token with (a) READ on gated google/gemma-2-9b, (b) READ on the
#               corpus dataset karpathy/climbmix-400b-shuffle, (c) WRITE on the
#               weights + attribution repos. huggingface_hub reads it automatically.
#               NEVER printed.
#
# OPTIONAL (env):
#   PY               python interpreter (default: repo .venv, else python3)
#   WEIGHTS_REPO     HF model repo with probe weights+summary (default from state.json)
#   ATTR_REPO        HF dataset repo to upload attribution to (default from state.json)
#   SHARD_RANGE      shard spec (default "0-185,6542" = all 187). Slice it per pod.
#   DEVICE           cuda | cpu (default cuda)
#   MAX_SEQ          truncate length (default 1024)
#   BATCH_TOKENS     per-forward token budget (default 16384)
#   MAX_BATCH_DOCS   per-forward doc cap (default 64)
#   SCALAR_THRESHOLD firing threshold on sigmoid magnitude for scalar rows (default 0.5)
#   WRITE_MANIFEST   set to 1 to (re)write the top-level manifest.json (run on ONE pod)
#   KEEP_LOCAL       set to 1 to keep local shard outputs (debug; default purge)
#
# Usage:  HF_TOKEN=hf_xxx SHARD_RANGE="0-46" WRITE_MANIFEST=1 bash run_phaseC.sh
# ============================================================================
set -euo pipefail

# --- GPU hygiene: cap CPU thread pools (handoff §5) -------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTHONUNBUFFERED=1

# --- locate paths -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # .../overnight_run2/code
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                          # .../overnight_run2
REPO_ROOT="$(cd "${RUN_DIR}/.." && pwd)"                           # oracle-encodings

if [[ -z "${PY:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then PY="${REPO_ROOT}/.venv/bin/python3";
  else PY="python3"; fi
fi
echo "[phaseC] python: ${PY}"
"${PY}" --version

# --- config (defaults match overnight_run2/state.json) ----------------------
WEIGHTS_REPO="${WEIGHTS_REPO:-kaushikreddyxyz/concept-probes-v2-weights}"
ATTR_REPO="${ATTR_REPO:-kaushikreddyxyz/concept-probes-v2-attribution}"
SHARD_RANGE="${SHARD_RANGE:-0-185,6542}"
DEVICE="${DEVICE:-cuda}"
MAX_SEQ="${MAX_SEQ:-1024}"
BATCH_TOKENS="${BATCH_TOKENS:-16384}"
MAX_BATCH_DOCS="${MAX_BATCH_DOCS:-64}"
SCALAR_THRESHOLD="${SCALAR_THRESHOLD:-0.5}"
GATE_MODE="${GATE_MODE:-relative}"
TOP_FRAC="${TOP_FRAC:-0.002}"
CALIB_SHARD="${CALIB_SHARD:-48}"
CALIB_TOKENS="${CALIB_TOKENS:-1000000}"
ATTN="${ATTN:-eager}"
WRITE_MANIFEST="${WRITE_MANIFEST:-}"
KEEP_LOCAL="${KEEP_LOCAL:-}"

WEIGHTS_DIR="${RUN_DIR}/phaseC_weights"
WORK_DIR="${RUN_DIR}/phaseC_work"
mkdir -p "${WEIGHTS_DIR}" "${WORK_DIR}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[phaseC] ERROR: HF_TOKEN is not set (need gemma read + corpus read + repo write)." >&2
  exit 2
fi

# --- 1) pull probe weights + summary.json from the HF weights repo ----------
echo "[phaseC] pulling probe weights from ${WEIGHTS_REPO} ..."
"${PY}" - "$WEIGHTS_REPO" "$WEIGHTS_DIR" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(repo_id=repo, repo_type="model", local_dir=dest,
                         allow_patterns=["summary.json", "weights/*.npz"])
print("[phaseC] weights downloaded to", path)
PYEOF

if ! ls "${WEIGHTS_DIR}"/weights/layer_*.npz >/dev/null 2>&1; then
  echo "[phaseC] ERROR: no probe weights (weights/layer_*.npz) pulled" >&2; exit 3
fi
echo "[phaseC] probe weights present: $(ls "${WEIGHTS_DIR}"/weights/layer_*.npz | wc -l | tr -d ' ') layers"

# --- 2) run the per-shard attribution sweep ---------------------------------
echo "[phaseC] sweeping shards [${SHARD_RANGE}] (device=${DEVICE}, max_seq=${MAX_SEQ}) ..."
EXTRA=()
[[ -n "${WRITE_MANIFEST}" ]] && EXTRA+=(--write-global-manifest)
[[ -n "${KEEP_LOCAL}" ]] && EXTRA+=(--keep-local)

"${PY}" "${SCRIPT_DIR}/attribute_corpus.py" \
  --shards "${SHARD_RANGE}" \
  --weights-dir "${WEIGHTS_DIR}" \
  --device "${DEVICE}" \
  --work-dir "${WORK_DIR}" \
  --max-seq "${MAX_SEQ}" \
  --batch-tokens "${BATCH_TOKENS}" \
  --max-batch-docs "${MAX_BATCH_DOCS}" \
  --scalar-threshold "${SCALAR_THRESHOLD}" \
  --gate-mode "${GATE_MODE}" \
  --top-frac "${TOP_FRAC}" \
  --calib-shard "${CALIB_SHARD}" \
  --calib-tokens "${CALIB_TOKENS}" \
  --attn "${ATTN}" \
  "${EXTRA[@]}"
rc="${PIPESTATUS[0]}"   # guard rc=0 masked by a pipe (handoff §5)

if [[ "${rc}" -ne 0 ]]; then
  echo "[phaseC] ERROR: attribute_corpus.py exited ${rc}" >&2; exit "${rc}"
fi
echo "[phaseC] DONE for range [${SHARD_RANGE}] -> ${ATTR_REPO}"
