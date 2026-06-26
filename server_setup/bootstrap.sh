#!/usr/bin/env bash
# bootstrap.sh — provision a fresh pod/container for oracle-encodings runs.
# =============================================================================
# Does, in order:
#   1. System packages: tmux (long runs must survive SSH drops) + curl/git/ca-certs
#   2. Claude Code CLI       (curl installer)
#   3. uv                    (curl installer)
#   4. uv sync + venv        (in the target repo; activated for the steps below)
#   5. .env -> environment   (HF_TOKEN, WANDB_TOKEN) + wandb login + huggingface login
#   6. Validate SSH agent forwarding (so git push uses YOUR forwarded key)
#
# Idempotent: re-running skips anything already installed/authenticated.
# Run ON THE POD from the repo root:  bash server_setup/bootstrap.sh
#
# KNOBS (env-overridable)
#   REPO_DIR=<repo root>      # which repo to uv-sync (default: parent of this script)
#   UV_SYNC_ARGS=""           # extra `uv sync` args, e.g. "--extra gpu" for nanochat
#   ENV_FILE=<REPO_DIR>/.env  # which .env to load tokens from
#   CHECK_GITHUB=0            # 1 = also probe `ssh -T git@github.com`
#   RUN_GIT_IDENTITY=0        # 1 = also run scripts/runpod_setup.sh (git identity)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:-}"
CHECK_GITHUB="${CHECK_GITHUB:-0}"
RUN_GIT_IDENTITY="${RUN_GIT_IDENTITY:-0}"

banner () { echo ""; echo "=== $* ==="; }

# --- privilege escalation + a portable package installer --------------------
SUDO=""
if [ "$(id -u)" != "0" ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi

# pkg_install pkg...  — install via whichever Linux package manager is present.
# Returns the installer's exit status (127 if no known package manager found).
pkg_install () {
    if command -v apt-get >/dev/null 2>&1; then
        $SUDO apt-get update -y -qq || true
        $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$@"
    elif command -v dnf >/dev/null 2>&1; then
        $SUDO dnf install -y -q "$@"
    elif command -v yum >/dev/null 2>&1; then
        $SUDO yum install -y -q "$@"
    elif command -v apk >/dev/null 2>&1; then
        $SUDO apk add --no-cache "$@"
    elif command -v pacman >/dev/null 2>&1; then
        $SUDO pacman -Sy --noconfirm "$@"
    elif command -v zypper >/dev/null 2>&1; then
        $SUDO zypper install -y "$@"
    else
        return 127
    fi
}

# ----------------------------------------------------------------------------
banner "1/6  System packages — tmux (REQUIRED) + curl, git, ca-certificates"
if command -v tmux >/dev/null 2>&1; then
    echo "  >> tmux already present: $(tmux -V)"
else
    echo "  >> installing tmux (+ curl git ca-certificates)..."
    pkg_install tmux curl git ca-certificates || true
    command -v tmux >/dev/null 2>&1 || pkg_install tmux || true   # retry tmux alone if the bundle failed
fi
# INVARIANT: long runs live inside tmux, so treat it as a hard requirement —
# stop here if it still isn't available rather than failing mid-run later.
if ! command -v tmux >/dev/null 2>&1; then
    echo "  ✗ ERROR: tmux is not installed and could not be installed automatically."
    echo "    Install it by hand (e.g. 'apt-get install -y tmux') and re-run bootstrap.sh."
    exit 1
fi
echo "  ✓ tmux ready: $(command -v tmux) ($(tmux -V))"

# ----------------------------------------------------------------------------
banner "2/6  Claude Code CLI"
export PATH="$HOME/.local/bin:$PATH"
if command -v claude >/dev/null 2>&1; then
    echo "  >> claude already installed: $(command -v claude)"
else
    # Native installer -> ~/.local/bin/claude. (Fallback: npm i -g @anthropic-ai/claude-code)
    curl -fsSL https://claude.ai/install.sh | bash
    command -v claude >/dev/null 2>&1 && echo "  >> installed: $(command -v claude)" \
        || echo "  !! claude not on PATH yet — add \$HOME/.local/bin to PATH."
fi

# ----------------------------------------------------------------------------
banner "3/6  uv"
if command -v uv >/dev/null 2>&1; then
    echo "  >> uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# uv installs to ~/.local/bin and writes an env shim; make it usable in THIS shell.
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH after install."; exit 1; }

# ----------------------------------------------------------------------------
banner "4/6  uv sync + venv  ($REPO_DIR)"
cd "$REPO_DIR"
[ -d ".venv" ] || uv venv
if [ -n "$UV_SYNC_ARGS" ]; then
    # shellcheck disable=SC2086
    uv sync $UV_SYNC_ARGS
else
    uv sync
fi
# Activate so the wandb/hf CLIs below come from the project venv.
# shellcheck disable=SC1091
source "$REPO_DIR/.venv/bin/activate"
echo "  >> venv active: $(command -v python)  ($(python --version 2>&1))"

# ----------------------------------------------------------------------------
banner "5/6  .env -> environment, wandb + huggingface login"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "  >> loaded $ENV_FILE"
    # Mirror the WANDB_TOKEN name to what wandb authenticates with (matches common.py).
    if [ -n "${WANDB_TOKEN:-}" ] && [ -z "${WANDB_API_KEY:-}" ]; then
        export WANDB_API_KEY="$WANDB_TOKEN"
    fi
else
    echo "  !! $ENV_FILE not found — copy .env.example to .env and fill in HF_TOKEN / WANDB_TOKEN."
fi

# wandb
if [ -n "${WANDB_API_KEY:-}" ]; then
    if wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1; then
        echo "  >> wandb: logged in ($(wandb whoami 2>/dev/null | head -1 || echo authenticated))"
    else
        echo "  !! wandb login failed — check the key in $ENV_FILE."
    fi
else
    echo "  >> wandb: no WANDB_TOKEN/WANDB_API_KEY found — skipping login."
fi

# huggingface (prefer the new `hf auth login`, fall back to huggingface-cli)
if [ -n "${HF_TOKEN:-}" ]; then
    if command -v hf >/dev/null 2>&1 && hf auth login --token "$HF_TOKEN" >/dev/null 2>&1; then
        echo "  >> huggingface: logged in ($(hf auth whoami 2>/dev/null | head -1 || echo authenticated))"
    elif command -v huggingface-cli >/dev/null 2>&1 && huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1; then
        echo "  >> huggingface: logged in (huggingface-cli)"
    else
        echo "  !! huggingface login failed — check HF_TOKEN. (HF_TOKEN is also read natively from the env.)"
    fi
else
    echo "  >> huggingface: no HF_TOKEN found — skipping login."
fi

# ----------------------------------------------------------------------------
banner "6/6  SSH agent forwarding"
if [ -z "${SSH_AUTH_SOCK:-}" ]; then
    echo "  ✗ SSH_AUTH_SOCK is unset — agent NOT forwarded."
    echo "    Reconnect with: ssh -A <pod>   (or set ForwardAgent yes in ~/.ssh/config)"
else
    set +e
    keys=$(ssh-add -l 2>&1); rc=$?
    set -e
    case "$rc" in
        0) echo "  ✓ agent forwarded; identities:"; echo "$keys" | sed 's/^/      /' ;;
        1) echo "  ✗ agent reachable but has NO identities — run 'ssh-add' on your LAPTOP, then reconnect." ;;
        *) echo "  ✗ cannot reach the agent at \$SSH_AUTH_SOCK ($keys)." ;;
    esac
    if [ "$CHECK_GITHUB" = "1" ] && [ "$rc" = "0" ]; then
        # GitHub returns exit 1 for `ssh -T` even on success; the message is the signal.
        gh=$(ssh -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 || true)
        echo "  >> github: $(echo "$gh" | head -1)"
    fi
fi

# ----------------------------------------------------------------------------
if [ "$RUN_GIT_IDENTITY" = "1" ] && [ -f "$REPO_DIR/scripts/runpod_setup.sh" ]; then
    banner "extra  git identity (scripts/runpod_setup.sh)"
    bash "$REPO_DIR/scripts/runpod_setup.sh" || true
fi

banner "BOOTSTRAP COMPLETE"
echo "  repo:   $REPO_DIR"
echo "  next:   tmux new -s run   then activate:  source $REPO_DIR/.venv/bin/activate"
echo "  MPS:    bash server_setup/mps_start.sh  &&  bash server_setup/mps_check.sh"
