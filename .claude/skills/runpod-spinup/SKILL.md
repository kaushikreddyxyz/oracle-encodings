---
name: runpod-spinup
description: Sets up a remote GPU (or CPU) pod on RunPod using the runpodctl CLI.
---

# RunPod Pod Spinup Skill

## Overview

This skill covers creating GPU (and CPU-only) pods on RunPod using the `runpodctl` CLI. The CLI is installed at `~/.local/bin/runpodctl` (on PATH). Auth is already configured — API key and SSH key live in `~/.runpod/`.

## IMPORTANT: Cost Check

**Before creating any GPU pod, you MUST tell the user the hourly cost and get explicit confirmation before proceeding.** Pull live prices with `runpodctl gpu list` (no flag) or check the table below.

### GPU Pricing (last verified: 2026-05-18, secure cloud, 1× GPU)

| GPU | VRAM | $/hr (observed) | $/day |
|-----|------|----------------|-------|
| **RTX 4090** | 24 GB | $0.34 | $8.16 |
| **RTX 5090** | 32 GB | $0.69 | $16.56 |
| **L40S** | 48 GB | $0.79 | $18.96 |
| **A100 SXM4 40 GB** | 40 GB | $1.00 | $24.00 |
| **A100 PCIe 40 GB** | 40 GB | $1.19 | $28.56 |
| **A100 SXM4 80 GB** | 80 GB | $1.39 | $33.36 |
| **H100 PCIe** | 80 GB | $1.99 | $47.76 |
| **H100 NVL** | 94 GB | $2.59 | $62.16 |
| **H100 SXM5** | 80 GB | $2.69 | $64.56 |
| **H200 SXM** | 141 GB | $3.59 | $86.16 |
| **B200 SXM** | 192 GB | $4.99 | $119.76 |
| **B300 SXM6** | 262 GB | $6.94 | $166.56 |

Only the RTX 4090 figure is observed from a real create call; the rest are typical RunPod prices and should be verified. Community cloud is usually cheaper but availability is patchier. Storage is billed separately for stopped pods.

To get live prices for a specific GPU, the cheapest route is `runpodctl pod create --dry-run` is **not** supported — instead query the GraphQL API or just attempt creation and read `costPerHr` in the response.

## Account Details

- **User:** jonathan@arcadiaimpact.org
- **API key:** stored in `~/.runpod/config.toml` (configured via `runpodctl config --apiKey ...`)
- **SSH key:** `~/.runpod/ssh/runpodctl-ssh-key{,.pub}` — auto-generated and registered with RunPod during config

## Available GPU IDs (use with `--gpu-id`)

From `runpodctl gpu list`:
- `NVIDIA GeForce RTX 4090`
- `NVIDIA A100 80GB PCIe`
- `NVIDIA A100-SXM4-80GB`
- `NVIDIA H100 80GB HBM3`
- `NVIDIA H200`
- `NVIDIA B200`
- `NVIDIA B300 SXM6 AC`

Run `runpodctl gpu list` for the current full list and availability/stock.

## Cloud Types

- `SECURE` — RunPod-operated datacenters, more reliable, slightly pricier (default)
- `COMMUNITY` — third-party hosts, cheaper, sometimes resource-constrained ("This machine does not have the resources" errors → fall back to SECURE)

## Common Templates

Use `runpodctl template search <query>` to find templates. Verified working:
- `runpod-torch-v21` — PyTorch 2.1, CUDA 11.8, Ubuntu 22.04 (the default that worked in testing)

You can also bypass templates with `--image <docker-image>`.

## Creating a Pod — Use the Script

**ALWAYS use `create-pod.sh` in this skill directory.** Do NOT call `runpodctl pod create` manually unless the script doesn't cover the case.

### Usage

```bash
/path/to/create-pod.sh <name> <gpu-id> [cloud-type] [template-id] [gpu-count]
```

Defaults: cloud-type=`SECURE`, template-id=`runpod-torch-v21`, gpu-count=`1`.

### Examples

**RTX 4090 (cheap test, ~$0.69/hr):**
```bash
./create-pod.sh test "NVIDIA GeForce RTX 4090"
```

**H100 SXM (community cloud, 1 GPU):**
```bash
./create-pod.sh h100-work "NVIDIA H100 80GB HBM3" COMMUNITY
```

**8× A100 secure cloud cluster:**
```bash
./create-pod.sh a100-cluster "NVIDIA A100-SXM4-80GB" SECURE runpod-torch-v21 8
```

### After Creation

The script prints the pod ID, GPU, hourly cost, and SSH command, **and appends a `runpod-<name>` alias to `~/.ssh/config`**. With the `Host runpod-*` defaults block in that file (User=root, IdentityFile=runpodctl-ssh-key, `ForwardAgent yes`), this means:

- `ssh runpod-<name>` just works — no flags needed.
- Agent forwarding is on, so `git push`/`git clone` of private GitHub repos from the pod uses the local ssh-agent (no token piping, no deploy keys). Requires the local user's GitHub-registered key to be loaded in ssh-agent (`ssh-add -l` should show it).
- `cleanup-pod.sh` removes the alias on delete.

The local ssh-agent is the systemd user service `ssh-agent.service` with socket at `$XDG_RUNTIME_DIR/openssh_agent`. `~/.bashrc` exports `SSH_AUTH_SOCK` to that path. Non-interactive shells need to export it explicitly: `export SSH_AUTH_SOCK="/run/user/$(id -u)/openssh_agent"`.

**Prefer direct SSH over `runpodctl ssh connect`.** The `runpodctl ssh connect` wrapper has been observed to hang or silently fail to return output for minutes after pod creation, even when the pod is fully booted. Direct SSH using the IP/port from `runpodctl pod get <pod-id>` works as soon as sshd is up (~10–30s after creation).

Get the direct SSH command:
```bash
runpodctl pod get <pod-id> | jq -r .ssh.ssh_command
# → ssh -i /home/.../runpodctl-ssh-key root@<ip> -p <port>
```

Then:
```bash
ssh -o StrictHostKeyChecking=no -i /home/jonathandbostock/.runpod/ssh/runpodctl-ssh-key root@<ip> -p <port> '<command>'
```

**Polling for sshd to come up** — use direct SSH, not the runpodctl wrapper:
```bash
until ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i /home/jonathandbostock/.runpod/ssh/runpodctl-ssh-key root@<ip> -p <port> 'echo ready' 2>/dev/null | grep -q ready; do sleep 5; done
```

RunPod also auto-exposes Jupyter on `8888/http` for the default template.

### Cloning private GitHub repos onto the pod

The default pod has no GitHub credentials and `gh` is not installed. Do **not** embed `gh auth token` into the clone URL — the auto-mode classifier blocks this (token leaks into argv/history). Instead, pipe the token over stdin and use a one-shot credential helper that reads it from the environment:

```bash
gh auth token | ssh -o StrictHostKeyChecking=no -i ~/.runpod/ssh/runpodctl-ssh-key root@<ip> -p <port> '
  read -r GH_TOKEN; export GH_TOKEN
  git config --global credential.helper "!f() { echo username=x-access-token; echo password=$GH_TOKEN; }; f"
  cd /workspace && git clone https://github.com/OWNER/REPO.git
  git config --global --unset credential.helper
'
```

The token never appears in argv, the URL, or shell history. The credential helper is removed immediately after the clone.

SSH-agent forwarding (`ssh -A`) is a viable alternative *only if* `~/.ssh/id_ed25519` (or similar) is registered with GitHub. Verify first with `ssh -T git@github.com` — if it returns "Permission denied (publickey)", fall back to the token-over-stdin approach.

## Cleanup

**Use `cleanup-pod.sh`:**

```bash
./cleanup-pod.sh <pod-id>
```

To merely stop (pause GPU billing, storage still charged):
```bash
runpodctl pod stop <pod-id>
runpodctl pod start <pod-id>
```

## Useful Commands

```bash
runpodctl pod list                 # all pods
runpodctl pod get <pod-id>         # details
runpodctl me                       # account balance + spend
runpodctl gpu list                 # GPU types + availability
runpodctl template search <q>      # find templates
runpodctl ssh connect <pod-id>     # ssh in
runpodctl billing                  # billing history
```

## Troubleshooting

- **"This machine does not have the resources to deploy your pod"** — community cloud host is full. Retry with `SECURE` or try again later.
- **"There are no longer any instances available with the requested specifications"** — that GPU type is currently out of stock across the chosen cloud. Check `runpodctl gpu list` for `available: true` GPUs and try the next-best option. In practice RTX 4090, A5000, and the RTX PRO 4500 are often unavailable during peak hours; A100/H100/H200 SXM and RTX 2000 Ada have been more reliable fallbacks.
- **SSH connect fails right after creation** — pod still booting; wait 10–30s. Use direct SSH (see "After Creation"), not `runpodctl ssh connect` (which can hang).
- **`runpodctl ssh connect` appears to hang** — known issue; switch to direct SSH using the IP/port from `runpodctl pod get`.
- **SSH refused for 5+ minutes (pod state: RUNNING)** — you almost certainly created the pod with `--image <image>` *without* a `--template-id`. The official `runpod/pytorch:*` images do not auto-start sshd by themselves; the template provides the start script that runs `service ssh start`. Fix: delete the pod and recreate via `create-pod.sh` (or pass `--template-id runpod-torch-v240` / similar). The deprecated `runpodctl create pods` (note plural) command is especially prone to this — always use the new `runpodctl pod create` (singular) via the script.
- **`No space left on device` mid-download or training-step crash with truncated weights** — the container disk filled. The script defaults to 20GB which barely fits a venv (~5GB); HF model snapshots eat the rest fast. A Qwen-14B snapshot is ~28GB, Llama-3.1-8B is ~16GB, Gemma-3-12B is ~24GB. Pass the 6th positional arg to `create-pod.sh` to size up: `./create-pod.sh name "NVIDIA H200" SECURE runpod-torch-v240 1 200` for a 200GB disk. Errors that signal this: `OSError: No space left on device (os error 28)` and `huggingface_hub` `Background writer channel closed` (the xet downloader's polite version of the same).
- **Need a non-default port** — pass `--ports "8080/http,22/tcp"` to `runpodctl pod create` directly (script doesn't expose this yet).
- All `runpodctl` commands default to JSON output (`-o json`); use `-o yaml` for yaml.

## Stock-availability fallback strategy

GPU availability changes minute-to-minute. When the user requests an unavailable GPU and hasn't specified a strict tier:
1. Try the requested GPU on SECURE.
2. Try the requested GPU on COMMUNITY.
3. Run `runpodctl gpu list` and pick the cheapest GPU with `available: true` and adequate VRAM for the task.
4. Tell the user what you fell back to and the new cost before proceeding further.
