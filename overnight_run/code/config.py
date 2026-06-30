"""
config.py — central configuration for the overnight concept-probes run.

Everything model/path/threshold/budget related lives here so the orchestrator,
labeling, probing, and geometry stages stay consistent. Override via env vars where
noted (the orchestrator sets these per-pod).
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent          # overnight_run/
CODE = ROOT / "code"
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
for _d in (DATA, ARTIFACTS, FIGURES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Models  (substitution policy — the brief blesses "good enough" substitutes)
# --------------------------------------------------------------------------- #
# JUDGE (Step 1 labeling). Primary is gated; fallback is a non-gated strong instruct
# model so labeling can proceed autonomously if the Gemma license stays unaccepted.
JUDGE_PRIMARY  = "google/gemma-3-27b-it"            # gated=manual -> needs license accept
JUDGE_FALLBACK = "Qwen/Qwen2.5-32B-Instruct-AWQ"   # non-gated, AWQ for vLLM throughput
JUDGE_MODEL    = os.environ.get("JUDGE_MODEL", JUDGE_PRIMARY)

# PROBE TARGET (Steps 2-4). gemma-2-9b chosen for Gemma Scope SAEs (Tier 6).
# NOT silently substituted: losing gemma-2-9b loses the SAE cross-check. If the user
# never accepts the license, PROBE_TARGET_FALLBACK is a non-gated option of last resort.
PROBE_TARGET_PRIMARY  = "google/gemma-2-9b"
PROBE_TARGET_FALLBACK = "Qwen/Qwen2.5-7B"          # non-gated; no Gemma Scope -> Tier 6 skipped
PROBE_TARGET = os.environ.get("PROBE_TARGET", PROBE_TARGET_PRIMARY)

SAE_REPO = "google/gemma-scope-9b-pt-res"          # non-gated, accessible

# --------------------------------------------------------------------------- #
# Data / shards  (DISJOINT from nanochat training: it consumed ~0-183, val=6542)
# --------------------------------------------------------------------------- #
DATASET = "karpathy/climbmix-400b-shuffle"
SHARDS = list(range(300, 310))                      # shard_00300 .. shard_00309
MAX_DOCS_PER_SHARD = int(os.environ.get("MAX_DOCS_PER_SHARD", "20000"))
SNIPPET_TOKENS = 64                                 # window around a match for context

# --------------------------------------------------------------------------- #
# Labeling (Step 1)
# --------------------------------------------------------------------------- #
N_SAMPLES = 5                                       # judge samples per example
JUDGE_TEMPERATURE = 0.8
JUDGE_MAX_TOKENS = 16
# presence filtering: mean>=3.5 -> positive ; <=0.5 -> negative ; middle -> discard
PRESENCE_POS_THRESH = 3.5
PRESENCE_NEG_THRESH = 0.5
SCALAR_MAX_SEED_STD = 1.25                          # discard high inter-seed variance
TARGET_BALANCE = 0.5                                # ~50/50 pos/neg
MAX_CANDIDATES_PER_CLASS = int(os.environ.get("MAX_CANDIDATES_PER_CLASS", "400"))

# --------------------------------------------------------------------------- #
# Probes (Step 2)
# --------------------------------------------------------------------------- #
PROBE_TYPE = "attention"                            # sequence-level attention probe
LAYER_STRIDE = int(os.environ.get("LAYER_STRIDE", "1"))   # sweep every Nth layer
PROBE_EPOCHS = 60
PROBE_LR = 1e-3
PROBE_BATCH = 64
RELIABLE_METRIC_THRESH = 0.9                        # "reliable" probe = AUROC/R2 > this

# --------------------------------------------------------------------------- #
# HF push targets  (push artifacts immediately; verify before any teardown)
# --------------------------------------------------------------------------- #
HF_USER = "kaushikreddyxyz"
HF_DATASET_REPO = f"{HF_USER}/concept-probes-overnight"     # labels + report + geometry
HF_MODEL_REPO   = f"{HF_USER}/concept-probes-weights"       # probe weights
HF_PRIVATE = False                                          # public per prior pattern

# --------------------------------------------------------------------------- #
# Budget / guards (HARD)  — orchestrator enforces; these are the source of truth
# --------------------------------------------------------------------------- #
BUDGET_STOP_LAUNCH_USD = 320.0                     # stop launching new GPU work
HOURLY_CAP_USD = 80.0                              # RunPod account cap (cannot change)
DEADLINE_HOURS = 9.0                               # wall-clock from orchestrator start
RUNPOD_SSH_KEY = os.path.expanduser("~/.ssh/runpod")

def summary():
    return {
        "judge": JUDGE_MODEL, "probe_target": PROBE_TARGET, "sae": SAE_REPO,
        "shards": f"{SHARDS[0]}-{SHARDS[-1]}", "n_samples": N_SAMPLES,
        "hf_dataset": HF_DATASET_REPO, "hf_model": HF_MODEL_REPO,
        "budget_stop": BUDGET_STOP_LAUNCH_USD, "deadline_h": DEADLINE_HOURS,
    }

if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
