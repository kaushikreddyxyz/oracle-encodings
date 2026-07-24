---
name: runpod-spinup
description: Companion to the `runpod` MCP server — cost-confirmation policy, local SSH alias wiring, and GitHub-clone-on-pod workaround for RunPod pods. Pod/GPU/pricing actions themselves go through the MCP server, not this skill.
---

# RunPod pod lifecycle (MCP-first)

Pod, GPU-type, pricing, and account operations go through the **`runpod` MCP
server** (`claude mcp add runpod --scope user -e RUNPOD_API_KEY=... -- npx -y
@runpod/mcp-server@latest`, wraps the RunPod REST API) and the
**`runpod-docs` MCP server** (`https://docs.runpod.io/mcp`, no auth) for
documentation lookups. Inspect their tool schemas at call time instead of
trusting hardcoded tool names/params here — the MCP server evolves
independently of this skill, and anything about it written here would go
stale.

This skill covers only what the MCP server can't: local machine setup, a
couple of binding policies, and troubleshooting knowledge for API-level
errors that recur no matter which client triggers them.

## Cost-confirmation policy (binding)

**Before creating any GPU pod, get the live hourly cost from the MCP
server's pricing/GPU-list tool and get the user's explicit confirmation.**
Never quote a remembered or cached number — RunPod GPU pricing moves with
market supply, so any figure written into this file would be wrong within
weeks. If the MCP server is unreachable, fall back to `runpodctl gpu list`
(CLI still installed locally) rather than guessing.

## After creating a pod: wire up SSH

The MCP create-pod tool returns connection info (IP, SSH port) but has no
access to this machine's `~/.ssh/config` — that's a local step:

```bash
./register-ssh-alias.sh <name> <ip> <port>
```

This adds a `Host runpod-<name>` block matching the convention already used
for every pod on this machine:

```
Host runpod-<name>
  HostName <ip>
  Port <port>
  User root
  IdentityFile ~/.ssh/runpod
  AddKeysToAgent yes
  UseKeychain yes
  ForwardAgent yes
  StrictHostKeyChecking no
```

(`~/.ssh/runpod` is the real key in use on this machine — not any
`runpodctl`-auto-generated key some older notes reference.) `ssh
runpod-<name>` then works immediately, with agent forwarding on.

Poll for sshd readiness with direct SSH, not an MCP/CLI "connect" wrapper
(both have been observed to return before sshd is actually up, or hang):

```bash
until ssh -o ConnectTimeout=5 runpod-<name> 'echo ready' 2>/dev/null | grep -q ready; do sleep 5; done
```

## Before/after deleting a pod: remove the alias

```bash
./deregister-ssh-alias.sh <name>
```

Skipping this leaves a stale alias pointing at an IP RunPod may reassign to
someone else's pod later.

## Cloning private GitHub repos onto a fresh pod

A brand-new pod has no GitHub credentials and `gh` isn't installed. Agent
forwarding (above) covers `git push`/`git clone` for the pod's own git
operations — verify the local key is loaded first with `ssh -T
git@github.com`. Also re-run this project's git-identity setup script (e.g.
`scripts/runpod_setup.sh` in oracle-encodings) once per fresh container: the
container disk resets on Stop→Start, so with no identity set, VS Code's
GitHub integration fills the gap with whatever account it's signed into and
misattributes commits.

If agent forwarding isn't available for some reason, fall back to piping a
token over stdin — never embed it in the clone URL or argv, where it can
leak into logs/history:

```bash
gh auth token | ssh runpod-<name> '
  read -r GH_TOKEN; export GH_TOKEN
  git config --global credential.helper "!f() { echo username=x-access-token; echo password=$GH_TOKEN; }; f"
  cd /workspace && git clone https://github.com/OWNER/REPO.git
  git config --global --unset credential.helper
'
```

## Minimizing per-pod setup time

For iterative work, attach a **network volume** (create-network-volume MCP
tool; pick the datacenter first — the volume pins pods to it) and mount it
at `/workspace`. `scripts/runpod_setup.sh` points `UV_CACHE_DIR` and
`HF_HOME` at `/workspace/.cache/`, and repos + their `.venv`s live on
`/workspace` by convention — so torch (~4GB), the venv, and model weights
(Qwen-7B ~15GB) download on the FIRST pod only; every later pod's
`uv sync` and model load resolve from the volume in seconds. Ephemeral
volume-less pods still work; they just pay full downloads each time.

## Troubleshooting (API-level, client-independent)

- **"This machine does not have the resources to deploy your pod"** —
  community-cloud host is full. Retry `SECURE`, or query the MCP GPU-list
  tool live for another GPU with `available: true`.
- **Requested GPU out of stock everywhere** — check live availability via
  the MCP GPU-list tool rather than assuming; A100/H100/H200 SXM tend to be
  more reliable than RTX 4090/A5000 during peak hours.
- **SSH refused for 5+ minutes on a RUNNING pod** — almost always means the
  pod was created from a raw image with no template attached, so nothing
  started sshd. Recreate with a template that includes RunPod's start
  script (search live via MCP — don't trust a specific template ID written
  here, template availability changes).
- **`No space left on device` mid-download, or a training step crashes with
  truncated weights** — container disk filled. Size it generously up front
  via the MCP create-pod tool's disk-size parameter: a Llama-3.1-8B
  snapshot is ~16GB, Qwen-14B ~28GB, Gemma-3-12B ~24GB. (After
  `runpod_setup.sh` runs, the HF cache sits under /workspace — size the
  volume or container disk accordingly.)

## Fallback

`runpodctl` CLI is still installed locally (`~/.local/bin/runpodctl` or
Homebrew) as a manual fallback if the MCP server is ever unreachable, but it
should not be the default path anymore.
