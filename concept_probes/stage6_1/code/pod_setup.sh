#!/bin/bash
# Stage 6.1 pod setup — idempotent. Run ON the pod (template runpod-torch-v240).
#
# SECRETS COME FROM STDIN (never argv, never logged):
#   line 1: HF token (required; gemma-2-2b is gated + authorizes uploads)
#   line 2: GitHub token (optional; needed only if the repo is not yet cloned)
# Orchestrator invocation (see FLEET.md):
#   scp pod_setup.sh <pod>:/workspace/
#   { cat ~/.cache/huggingface/token; gh auth token; } | \
#       ssh <pod> 'bash /workspace/pod_setup.sh'
#
# Env knobs (all optional):
#   WORKROOT   default /workspace
#   REPO_DIR   default $WORKROOT/oracle-encodings
#   FAMILIES   space/comma list; default = all 14 stage-4 families
#   SKIP_MODEL=1  skip the gemma snapshot download
#   DRYRUN=1   print every action instead of executing it
set -eo pipefail

DRYRUN="${DRYRUN:-0}"
WORKROOT="${WORKROOT:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKROOT/oracle-encodings}"
STAGING="$WORKROOT/hf_staging"
GH_REPO="${GH_REPO:-github.com/kaushikreddyxyz/oracle-encodings.git}"
WREPO="kaushikreddyxyz/concept-probes-gemma2-2b"
DREPO="kaushikreddyxyz/concept-probes-stage4-data"
FAMILIES="${FAMILIES:-color_wheel continents costliness directions duration glorptitude harmfulness location_type lovingness months moon_phases physical_size seasons weekdays}"
FAMILIES="$(echo "$FAMILIES" | tr ',' ' ')"

run() { if [ "$DRYRUN" = "1" ]; then echo "DRYRUN: $*"; else "$@"; fi; }

# ---------------------------------------------------------------- secrets
# (read even under DRYRUN so the pipe contract is identical; tokens are
#  never echoed — DRYRUN lines below print commands that only reference env)
IFS= read -r HF_TOKEN || true
IFS= read -r GH_TOKEN || true
if [ -z "$HF_TOKEN" ]; then echo "FATAL: no HF token on stdin line 1"; exit 1; fi
export HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=1

# ------------------------------------------------------------ python deps
# torch ships with the template; transformers 5.x is what the harness expects
run pip install -q --upgrade "transformers>=5,<6" huggingface_hub hf_transfer \
    numpy tqdm pyyaml
run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('gpu:', torch.cuda.get_device_name(0)); import transformers; print('transformers', transformers.__version__)"
if [ "$DRYRUN" != "1" ]; then
  hf auth whoami >/dev/null 2>&1 && echo "hf auth: ok (env token)" \
    || { echo "FATAL: HF token rejected"; exit 1; }
  # persist for later shells / pod_run.sh (pod-local, 0600)
  umask 077
  printf '%s\n' "$HF_TOKEN" > "$WORKROOT/.hf_token"
fi

# ------------------------------------------------------------- repo clone
# Token-over-stdin + one-shot credential helper (runpod-spinup skill pattern):
# the token lives only in env; never in argv, the URL, or shell history.
if [ -d "$REPO_DIR/.git" ]; then
  echo "repo already cloned: $REPO_DIR"
  if [ -n "$GH_TOKEN" ]; then
    export GH_TOKEN
    run git config --global credential.helper \
      '!f() { echo username=x-access-token; echo password=$GH_TOKEN; }; f'
    run git -C "$REPO_DIR" pull --ff-only
    run git config --global --unset credential.helper
  fi
else
  if [ -z "$GH_TOKEN" ]; then
    echo "FATAL: $REPO_DIR missing and no GitHub token on stdin line 2"; exit 1
  fi
  export GH_TOKEN
  run git config --global credential.helper \
    '!f() { echo username=x-access-token; echo password=$GH_TOKEN; }; f'
  run git clone --depth 1 "https://$GH_REPO" "$REPO_DIR"
  run git config --global --unset credential.helper
fi
unset GH_TOKEN

CP="$REPO_DIR/concept_probes"

# ------------------------------------------------------- model snapshot
if [ "${SKIP_MODEL:-0}" != "1" ]; then
  echo "== download google/gemma-2-2b snapshot =="
  run hf download google/gemma-2-2b --exclude "*.gguf" --quiet
fi

# ------------------------------------- probes + natscores (HF model repo)
# HF layout: families/<fam>/probes_l*.npz, natscores/<fam>.natscores.npz,
# stage6_1/inputs/random_pool.jsonl (uploaded by FLEET.md step 0).
# Repo-relative targets are exactly what stage6_1/code/common.py resolves:
#   concept_probes/stage5/probes/<fam>/  and
#   concept_probes/stage6/data/natscores/ .
# (natstats26.npz + stage6/artifacts/probe_cards.json are git-tracked and
#  arrive with the clone.)
for FAM in $FAMILIES; do
  echo "== [$FAM] probes =="
  run hf download "$WREPO" --include "families/$FAM/*" \
    --local-dir "$STAGING" --quiet
  run mkdir -p "$CP/stage5/probes/$FAM"
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN: cp -a $STAGING/families/$FAM/. $CP/stage5/probes/$FAM/"
  else
    cp -a "$STAGING/families/$FAM/." "$CP/stage5/probes/$FAM/"
  fi
done
echo "== natscores + stage6_1 inputs =="
# NOTE: one --include pattern per call — multiple patterns after --include get
# parsed as explicit positional filenames by the hf CLI (found the hard way).
run hf download "$WREPO" --include "natscores/*" --local-dir "$STAGING" --quiet
run hf download "$WREPO" --include "stage6_1/inputs/*" --local-dir "$STAGING" --quiet
run mkdir -p "$CP/stage6/data/natscores" "$CP/stage6/data/natural"
if [ "$DRYRUN" = "1" ]; then
  echo "DRYRUN: cp -a $STAGING/natscores/. $CP/stage6/data/natscores/"
  echo "DRYRUN: cp stage6_1/inputs/random_pool.jsonl into stage6/data/natural/"
else
  [ -d "$STAGING/natscores" ] && cp -a "$STAGING/natscores/." "$CP/stage6/data/natscores/"
  if [ -f "$STAGING/stage6_1/inputs/random_pool.jsonl" ]; then
    cp "$STAGING/stage6_1/inputs/random_pool.jsonl" \
       "$CP/stage6/data/natural/random_pool.jsonl"
  else
    echo "WARNING: stage6_1/inputs/random_pool.jsonl not on HF yet — E3 will"
    echo "         fall back to judged_nat neutrals. Run FLEET.md step 0."
  fi
  # tokenized natural eval jsonls — primary positives source for E2/E4/E5
  if [ -d "$STAGING/stage6_1/inputs/eval" ]; then
    mkdir -p "$CP/stage6/data/natural/eval"
    cp -a "$STAGING/stage6_1/inputs/eval/." "$CP/stage6/data/natural/eval/"
  else
    echo "WARNING: stage6_1/inputs/eval/ not on HF — E2/E4/E5 positives missing."
  fi
fi

# ------------------------------ stage-4 judged/final jsonls (dataset repo)
# Dataset paths start with data/<fam>/..., so --local-dir at stage4/ lands
# them at concept_probes/stage4/data/<fam>/{judged,final}/ — the exact
# relative paths the harness reads.
for FAM in $FAMILIES; do
  echo "== [$FAM] stage4 judged/final =="
  run hf download "$DREPO" --repo-type dataset \
    --include "data/$FAM/final/*" --local-dir "$CP/stage4" --quiet
  run hf download "$DREPO" --repo-type dataset \
    --include "data/$FAM/judged/*" --local-dir "$CP/stage4" --quiet
done

# ------------------------------------------------------------ verification
if [ "$DRYRUN" != "1" ]; then
  echo "== verify harness resolution =="
  REPO_DIR="$REPO_DIR" python - <<'PY'
import os, sys
repo = os.environ["REPO_DIR"]
sys.path.insert(0, os.path.join(repo, "concept_probes/stage6_1/code"))
import common
assert common.NATSTATS_PATH.exists(), f"missing {common.NATSTATS_PATH}"
fams = common.FAMILIES
print(f"FAMILIES discovered from probes: {len(fams)} -> {sorted(fams)}")
missing_nat, missing_jn = [], []
for f in fams:
    if not (common.NATSCORES_DIR / f"{f}.natscores.npz").exists():
        missing_nat.append(f)
    if not (common.CP_DIR / "stage4" / "data" / f / "judged" /
            "judged_nat.jsonl").exists():
        missing_jn.append(f)
print("natscores missing:", missing_nat or "none",
      "(glorptitude expected: nonsense control, no natural scoring)")
print("judged_nat.jsonl missing:", missing_jn or "none",
      "(if unexpected, run FLEET.md step 0 uploads)")
pool = common.CP_DIR / "stage6" / "data" / "natural" / "random_pool.jsonl"
print("random_pool.jsonl:", "present" if pool.exists() else "MISSING")
ev = common.CP_DIR / "stage6" / "data" / "natural" / "eval"
n_ev = len(list(ev.glob("*.jsonl"))) if ev.exists() else 0
print(f"natural/eval jsonls: {n_ev} (expect 13)")
PY
fi
echo "pod_setup: DONE (repo=$REPO_DIR)"
