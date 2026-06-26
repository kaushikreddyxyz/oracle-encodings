# server_setup

Provisioning + GPU-sharing scripts for oracle-encodings runs on a fresh pod
(RunPod H100). Container disk resets on Stop→Start, so these are meant to be
re-run per container; they're all idempotent.

## Scripts

| script | what it does | when |
| --- | --- | --- |
| `bootstrap.sh` | tmux + Claude CLI + uv, `uv sync` the venv, load `.env`, log into wandb + HF, check SSH-agent forwarding | once per fresh container |
| `mps_start.sh` | start the CUDA MPS daemon (daemon-first, optional per-client SM cap) | before launching multi-process GPU runs |
| `mps_check.sh` | confirm MPS is up **and** funneling 2 concurrent workers through one server | after `mps_start.sh` |

## Typical flow

```bash
# 1. provision the container (outer repo). For nanochat: REPO_DIR=nanochat UV_SYNC_ARGS="--extra gpu"
bash server_setup/bootstrap.sh

# 2. work inside tmux so runs survive SSH drops
tmux new -s run
source .venv/bin/activate

# 3. bring up GPU sharing, then verify it
bash server_setup/mps_start.sh            # MPS_THREAD_PCT=50 to cap each client to 50% SMs
bash server_setup/mps_check.sh            # expect: MPS CHECK: PASS

# 4. launch your N training workers (they inherit MPS automatically); each can:
#    source /tmp/mps_env.sh
```

## Why MPS

Training is launch-bound (~85% per-launch overhead at our depths), so packing
several training processes onto one GPU is the speedup lever. MPS runs their
kernels concurrently through a single `nvidia-cuda-mps-server` instead of
time-slicing. Two guardrails matter:

- **daemon-first** — start `mps_start.sh` before any worker, or the first worker
  takes an exclusive context and the rest serialize.
- **worker-cap** — MPS shares but does **not** partition GPU memory; cap worker
  count / batch size so the total fits. An OOM under MPS can wedge the context.

Stop MPS with `echo quit | nvidia-cuda-mps-control`. Pod-side runbook (if present):
`/workspace/mps/MPS.md`.

## Tokens

`bootstrap.sh` reads `HF_TOKEN` and `WANDB_TOKEN` from `.env` (gitignored; copy
from `.env.example`). It mirrors `WANDB_TOKEN` → `WANDB_API_KEY`, matching how
`nanochat/nanochat/common.py` loads them.
