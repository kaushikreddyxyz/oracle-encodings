#!/bin/bash
# Stage 6.1 per-pod driver: run the experiment scripts sequentially over this
# pod's families, with per-script logs, exit-code capture, and optional HF
# upload of out/ after every script.
#
# Usage (heartbeat-safe — survives ssh disconnects):
#   cd /workspace/oracle-encodings/concept_probes/4_causal/code
#   nohup env FAMILIES="months,seasons" SCRIPTS="e1,e2_cloze,e2_ppl,e4,e5" \
#       UPLOAD=1 bash pod_run.sh > /workspace/pod_run.out 2>&1 &
#
# Env:
#   FAMILIES   (required) comma/space list of families for this pod
#   SCRIPTS    default "e1,e2_cloze,e2_ppl,e4,e5"; also accepts e3 (=generate)
#   OUT        default <4_causal>/out (repo-relative, where common.py caches
#              dose_calib.json — keep the default unless you know better)
#   DEVICE     default cuda
#   ARGS_<s>   extra args for script <s>, e.g. ARGS_e2_cloze="--limit 3"
#              ARGS_e1="--classes january,harmfulness,europe" (pilot)
#   UPLOAD=1   after each script (and at the end) push $OUT to
#              hf.co/kaushikreddyxyz/concept-probes-gemma2-2b under stage6_1/out
#              (token: $HF_TOKEN env, else /workspace/.hf_token from pod_setup)
#   DRYRUN=1   print the commands instead of running them
#
# Exit code: 0 iff every requested script exited 0. Per-script status in
# $OUT/logs/status.tsv, per-script logs in $OUT/logs/<script>.log.
set -o pipefail   # deliberately NOT -e: one failing script must not kill the run
cd "$(dirname "$0")"

DRYRUN="${DRYRUN:-0}"
SCRIPTS="${SCRIPTS:-e1,e2_cloze,e2_ppl,e4,e5}"
SCRIPTS="$(echo "$SCRIPTS" | tr ',' ' ')"
OUT="${OUT:-$(cd .. && pwd)/out}"
DEVICE="${DEVICE:-cuda}"
WREPO="kaushikreddyxyz/concept-probes-gemma2-2b"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ -z "$FAMILIES" ]; then echo "FATAL: FAMILIES env is required"; exit 1; fi
FAMILIES_CSV="$(echo "$FAMILIES" | tr ' ' ',')"

if [ -z "$HF_TOKEN" ] && [ -f /workspace/.hf_token ]; then
  HF_TOKEN="$(cat /workspace/.hf_token)"; export HF_TOKEN
fi
if [ "${UPLOAD:-0}" = "1" ] && [ -z "$HF_TOKEN" ] && [ "$DRYRUN" != "1" ]; then
  echo "FATAL: UPLOAD=1 but no HF_TOKEN (env or /workspace/.hf_token)"; exit 1
fi

mkdir -p "$OUT/logs"
STATUS="$OUT/logs/status.tsv"

upload() {
  [ "${UPLOAD:-0}" = "1" ] || return 0
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN: hf upload $WREPO $OUT stage6_1/out --commit-message '...'"
    return 0
  fi
  hf upload "$WREPO" "$OUT" "stage6_1/out" \
    --commit-message "stage6_1 pod results ($FAMILIES_CSV) after $1" \
    > /dev/null 2>&1 && echo "upload after $1: ok" \
    || echo "upload after $1: FAILED (results still on pod at $OUT)"
}

FAIL=0
for S in $SCRIPTS; do
  # NOTE on --out semantics: the wave-2 scripts (e1/e2_*/e4/e5) take their
  # SCRIPT-SPECIFIC dir (heartbeats go to its parent); e3_generate takes the
  # PARENT out dir (it writes <out>/e3 and reads <out>/e2_cloze/selection.json).
  case "$S" in
    e1)       PY=e1_attrib.py;      OUTARG="$OUT/e1" ;;
    e2_cloze) PY=e2_cloze.py;       OUTARG="$OUT/e2_cloze" ;;
    e2_ppl)   PY=e2_ppl.py;         OUTARG="$OUT/e2_ppl" ;;
    e3)       PY=e3_generate.py;    OUTARG="$OUT" ;;
    e4)       PY=e4_ablate.py;      OUTARG="$OUT/e4" ;;
    e5)       PY=e5_propagation.py; OUTARG="$OUT/e5" ;;
    *) echo "SKIP unknown script key '$S'"; continue ;;
  esac
  if [ ! -f "$PY" ]; then
    echo "SKIP $S: $PY not present (git pull the repo?)"
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$S" "missing" "-" >> "$STATUS"
    FAIL=1
    continue
  fi
  # per-script extras via ARGS_<key> ($S is from the fixed case list above)
  eval "EXTRA=\${ARGS_$S:-}"
  LOG="$OUT/logs/$S.log"
  echo "=== [$S] python $PY --families $FAMILIES_CSV --out $OUTARG --device $DEVICE $EXTRA (log: $LOG) ==="
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN: python $PY --families $FAMILIES_CSV --out $OUTARG --device $DEVICE $EXTRA"
    RC=0
  else
    # shellcheck disable=SC2086
    python "$PY" --families "$FAMILIES_CSV" --out "$OUTARG" --device "$DEVICE" \
      $EXTRA >> "$LOG" 2>&1
    RC=$?
  fi
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$S" "rc=$RC" \
    "$FAMILIES_CSV" >> "$STATUS"
  if [ "$RC" -ne 0 ]; then
    echo "=== [$S] FAILED rc=$RC — tail of log: ==="
    [ -f "$LOG" ] && tail -20 "$LOG"
    FAIL=1
  else
    echo "=== [$S] OK ==="
  fi
  upload "$S"
done

upload "final"
echo "pod_run done: families=$FAMILIES_CSV scripts=$SCRIPTS fail=$FAIL"
[ -f "$STATUS" ] && column -t "$STATUS" 2>/dev/null || cat "$STATUS" 2>/dev/null
exit "$FAIL"
