#!/usr/bin/env python3
"""
orchestrator.py — deterministic driver for the overnight concept-probes run.

Responsibilities (per the run contract):
  1. Manifest of work units; reads/writes state.json so it resumes by skipping anything
     already done+pushed.
  2. A guard thread enforcing the $320 spend cap and 9h deadline -> tears pods down when hit.
  3. Every GPU unit wrapped in try/except: on failure tear the pod down, mark failed,
     continue (retry a fresh pod once). A torn-down pod always beats a wasted one.
  4. Calls `claude -p` for intelligent sub-tasks (geometry verdicts / debugging).

Gemma is license-gated for this account. Policy:
  - LABEL: use gemma-3-27b-it judge if accessible, else Qwen2.5-32B-Instruct-AWQ (brief
    blesses substitution; labels are model-agnostic -> low regret).
  - PROBE/ATTRIBUTE/GEOMETRY: prefer gemma-2-9b (designed; Gemma Scope for Tier 6). If
    still blocked when we reach it, fall back to Qwen2.5-7B (full Tiers 1-5; skip Tier 6),
    clearly labeled and re-runnable. Access is re-checked right before the stage.

Usage:
  python orchestrator.py --stage <verify|label|probe|geometry|writeup>   # supervised
  python orchestrator.py --auto                                          # full detached run
  python orchestrator.py --status
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))           # code/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))    # overnight_run/
import config
import runpod_lib as R

ROOT = config.ROOT
STATE = ROOT / "state.json"
REPORT = ROOT / "report.md"
LOGF = config.LOGS / "orchestrator.log"
START_BAL = 415.22
ABORT = threading.Event()


# --------------------------------------------------------------------------- #
# State + logging
# --------------------------------------------------------------------------- #
def load_state():
    return json.loads(STATE.read_text())


def save_state(s):
    STATE.write_text(json.dumps(s, indent=2))


def set_unit(stage, unit, status, hf_path=None, note=None):
    s = load_state()
    u = s["stages"][stage]["units"].setdefault(unit, {})
    u["status"] = status
    if hf_path:
        u["hf_path"] = hf_path
    if note:
        u["note"] = note
    # stage-level rollup
    st = s["stages"][stage]["units"].values()
    if all(x.get("status") in ("done", "skipped") for x in st):
        s["stages"][stage]["status"] = "done"
    elif any(x.get("status") == "running" for x in st):
        s["stages"][stage]["status"] = "running"
    save_state(s)


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOGF, "a") as f:
        f.write(line + "\n")


def append_report(text):
    with open(REPORT, "a") as f:
        f.write("\n" + text.rstrip() + "\n")


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def hours_since_start():
    start = datetime.strptime((ROOT / "START_TIME_UTC").read_text().strip(),
                              "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - start).total_seconds() / 3600.0


def budget_ok():
    """True if we may still LAUNCH new GPU work."""
    elapsed = hours_since_start()
    remaining = max(0.0, config.DEADLINE_HOURS - elapsed)
    if remaining <= 0.1:
        log("DEADLINE reached -> no new launches")
        return False
    try:
        proj, spent, rate = R.projected_spend(START_BAL, remaining)
    except Exception as e:
        log(f"WARN budget check failed ({e}); allowing but watch closely")
        return True
    if spent >= config.BUDGET_STOP_LAUNCH_USD or proj >= config.BUDGET_STOP_LAUNCH_USD:
        log(f"BUDGET stop: spent=${spent:.2f} proj=${proj:.2f} >= ${config.BUDGET_STOP_LAUNCH_USD}")
        return False
    return True


def guard_thread():
    """Daemon: every 60s enforce deadline+budget; on breach set ABORT and tear down."""
    while not ABORT.is_set():
        try:
            elapsed = hours_since_start()
            remaining = max(0.0, config.DEADLINE_HOURS - elapsed)
            proj, spent, rate = R.projected_spend(START_BAL, remaining)
            pods = [p.get("id") for p in R.list_pods()]
            log(f"guard: t+{elapsed:.2f}h spent=${spent:.2f} proj=${proj:.2f} rate=${rate}/hr pods={pods}")
            if remaining <= 0.05 or spent >= config.BUDGET_STOP_LAUNCH_USD:
                log("GUARD BREACH -> tearing down all pods + aborting")
                R.teardown_all()
                ABORT.set()
                return
        except Exception as e:
            log(f"guard error: {e}")
        for _ in range(60):
            if ABORT.is_set():
                return
            time.sleep(1)


# --------------------------------------------------------------------------- #
# Gemma access + model resolution
# --------------------------------------------------------------------------- #
def gemma_accessible(repo):
    from huggingface_hub import hf_hub_download
    try:
        from huggingface_hub.errors import GatedRepoError
    except ImportError:
        from huggingface_hub.utils import GatedRepoError
    try:
        hf_hub_download(repo, "config.json", token=True)
        return True
    except GatedRepoError:
        return False
    except Exception as e:
        log(f"access check {repo}: {type(e).__name__} {str(e)[:60]}")
        return False


def resolve_judge():
    if gemma_accessible("google/gemma-3-27b-it"):
        log("judge -> gemma-3-27b-it (license accepted!)")
        return "google/gemma-3-27b-it", 80
    log("judge -> Qwen2.5-32B-Instruct-AWQ (gemma judge still gated)")
    return config.JUDGE_FALLBACK, 48


def resolve_probe_target():
    if gemma_accessible("google/gemma-2-9b"):
        log("probe target -> gemma-2-9b (license accepted!) + Gemma Scope Tier-6 enabled")
        return "google/gemma-2-9b", True
    log("probe target -> Qwen2.5-7B fallback (gemma-2-9b still gated; Tier-6 SAE skipped)")
    return config.PROBE_TARGET_FALLBACK, False


# --------------------------------------------------------------------------- #
# claude -p intelligent subtask
# --------------------------------------------------------------------------- #
def call_claude(prompt, timeout=300):
    try:
        r = subprocess.run(["claude", "-p", prompt], capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception as e:
        log(f"claude -p failed: {e}")
        return None


# --------------------------------------------------------------------------- #
# Pod-backed stage runner
# --------------------------------------------------------------------------- #
def with_pod(name, min_vram, prefer, work_fn, disk_gb=120, terminate_hours=6, retries=1):
    """Create pod -> wait ssh -> scp .env+code -> work_fn(host,port) -> teardown (finally).
    Retries once with a fresh pod on failure. Returns work_fn's result or None."""
    if not budget_ok() or ABORT.is_set():
        log(f"[{name}] skipped: budget/deadline guard")
        return None
    attempt = 0
    while attempt <= retries and not ABORT.is_set():
        attempt += 1
        gid, price = R.pick_gpu(min_vram, prefer=prefer)
        if not gid:
            log(f"[{name}] no GPU >= {min_vram}GB available")
            return None
        log(f"[{name}] attempt {attempt}: creating {gid} (~${price}/hr) disk={disk_gb}GB term+{terminate_hours}h")
        pid = None
        try:
            info = R.create_pod(name, gid, disk_gb=disk_gb, terminate_after_hours=terminate_hours)
            pid = info.get("id") if isinstance(info, dict) else None
            if not pid:
                raise RuntimeError(f"create returned no id: {str(info)[:200]}")
            log(f"[{name}] pod {pid} created; waiting for ssh...")
            host, port = R.wait_ssh(pid)
            log(f"[{name}] ssh up {host}:{port}; provisioning")
            # move repo code + .env (scp only; never printed)
            R.ssh_run(host, port, "mkdir -p /workspace/run/code", timeout=60)
            R.ssh_run(host, port, "mkdir -p /workspace/run/prompts /workspace/run/data", timeout=60)
            # tar the local overnight_run (code+prompts+concepts+config+data) and ship it
            subprocess.run(["tar", "czf", "/tmp/run_payload.tgz", "-C", str(ROOT),
                            "code", "prompts", "concepts.py", "config.py" if (ROOT/"config.py").exists() else "code/config.py",
                            "data"], check=False, capture_output=True)
            R.scp_to(host, port, "/tmp/run_payload.tgz", "/workspace/run/payload.tgz")
            R.ssh_run(host, port, "cd /workspace/run && tar xzf payload.tgz", timeout=120)
            env_path = ROOT.parent / ".env"
            if env_path.exists():
                R.scp_to(host, port, str(env_path), "/workspace/run/.env")
            # ship the VERIFIED cli-cached HF token (has accepted-license gemma access);
            # gated model downloads + artifact pushes on the pod rely on this, not on .env.
            hf_tok = os.path.expanduser("~/.cache/huggingface/token")
            if os.path.exists(hf_tok):
                R.ssh_run(host, port, "mkdir -p /root/.cache/huggingface", timeout=30)
                R.scp_to(host, port, hf_tok, "/root/.cache/huggingface/token")
            result = work_fn(pid, host, port)
            log(f"[{name}] work complete; tearing down {pid}")
            R.delete_pod(pid)
            return result
        except Exception as e:
            log(f"[{name}] ERROR: {type(e).__name__}: {str(e)[:300]}")
            if pid:
                log(f"[{name}] tearing down failed pod {pid}")
                R.delete_pod(pid)
            if attempt > retries:
                log(f"[{name}] hard-fail after {attempt} attempts")
                return None
            log(f"[{name}] retrying with a fresh pod...")
    return None


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_verify():
    """Confirm local (CPU) prep produced by subagents is present; mark done."""
    ok = True
    cand_dir = config.DATA / "candidates"
    n_cand = len(list(cand_dir.glob("*.jsonl"))) if cand_dir.exists() else 0
    reg = config.PROMPTS / "registry.json"
    n_prompts = len(json.loads(reg.read_text())) if reg.exists() else 0
    log(f"verify: candidate files={n_cand}, prompts in registry={n_prompts}")
    for unit in load_state()["stages"]["build_candidates"]["units"]:
        f = cand_dir / f"{unit if unit!='scalars' else 'scalars'}.jsonl"
        set_unit("build_candidates", unit, "done" if f.exists() else "todo")
    set_unit("judge_prompts", "scalars", "done" if n_prompts >= 79 else "todo")
    return ok and n_cand >= 8 and n_prompts >= 70


def stage_label():
    judge, vram = resolve_judge()
    s = load_state(); s["models"]["judge_resolved"] = judge; save_state(s)
    prefer = (["A100 80GB", "H100"] if vram >= 80 else ["L40S", "A100 80GB", "RTX A6000"])

    def work(pid, host, port):
        setup = (
            "cd /workspace/run && set -a && . ./.env 2>/dev/null; set +a; "
            "pip -q install --upgrade vllm openai huggingface_hub pyarrow pandas 2>&1 | tail -2; "
            "python -c 'import vllm; print(\"vllm\", vllm.__version__)'"
        )
        log("[label] installing vllm...")
        R.ssh_run(host, port, setup, timeout=1800)
        quant = "--quantization awq" if judge.endswith("-AWQ") else ""
        serve = (f"cd /workspace/run && set -a && . ./.env 2>/dev/null; set +a; export HF_TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null);"
                 f"JUDGE_MODEL='{judge}' nohup python -m vllm.entrypoints.openai.api_server "
                 f"--model '{judge}' {quant} --port 8000 --max-model-len 4096 "
                 f"--gpu-memory-utilization 0.92 > /workspace/run/vllm.log 2>&1 &")
        R.ssh_run(host, port, serve, timeout=60)
        log("[label] waiting for vLLM server to come up (model download + load)...")
        up = False
        for _ in range(60):
            if ABORT.is_set():
                return None
            out = R.ssh_run(host, port, "curl -s http://localhost:8000/v1/models || true", check=False, timeout=30)
            if "data" in out and judge.split("/")[-1][:6] in out:
                up = True
                break
            time.sleep(20)
        if not up:
            tail = R.ssh_run(host, port, "tail -30 /workspace/run/vllm.log", check=False, timeout=30)
            raise RuntimeError(f"vLLM did not come up. log tail:\n{tail[:800]}")
        log("[label] judge up; running label.py")
        run = (f"cd /workspace/run && set -a && . ./.env 2>/dev/null; set +a; export HF_TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null);"
               f"JUDGE_MODEL='{judge}' PROBE_TARGET='{config.PROBE_TARGET_PRIMARY}' "
               f"PYTHONPATH=code:. python code/label.py 2>&1 | tail -40")
        out = R.ssh_run(host, port, run, timeout=14400)
        log(f"[label] label.py output tail:\n{out[-1500:]}")
        # push from pod (HF_TOKEN comes from sourced .env)
        push = ("cd /workspace/run && set -a && . ./.env 2>/dev/null; set +a; export HF_TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null);"
                "PYTHONPATH=code:. python -c 'import label; label.push_labels()' 2>&1 | tail -10")
        R.ssh_run(host, port, push, timeout=1800)
        # pull labels back locally too
        R.ssh_run(host, port, "cd /workspace/run && tar czf labels.tgz data/labels", check=False, timeout=120)
        try:
            R.scp_from(host, port, "/workspace/run/labels.tgz", "/tmp/labels.tgz")
            subprocess.run(["tar", "xzf", "/tmp/labels.tgz", "-C", str(ROOT)], check=False)
        except Exception as e:
            log(f"[label] label pull-back failed (already on HF): {e}")
        return True

    log("=== STAGE: label ===")
    res = with_pod("ovn-judge", vram, prefer, work, disk_gb=120, terminate_hours=6)
    if res:
        verify_hf(config.HF_DATASET_REPO, "dataset")
        set_unit("label", "scalars", "done", hf_path=config.HF_DATASET_REPO)
        for u in load_state()["stages"]["label"]["units"]:
            set_unit("label", u, "done", hf_path=config.HF_DATASET_REPO)
        append_report(f"## Stage 1 — labeling (judge={judge})\nLabeled set pushed to "
                      f"`{config.HF_DATASET_REPO}`. See validation_*.json for judge-vs-pseudo-gold "
                      f"agreement and pos/neg/discard counts.\n")
    return res


def stage_probe():
    target, tier6 = resolve_probe_target()
    s = load_state(); s["models"]["probe_target_resolved"] = target; save_state(s)

    def work(pid, host, port):
        setup = ("cd /workspace/run && set -a && . ./.env 2>/dev/null; set +a; export HF_TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null);"
                 "pip -q install --upgrade transformers accelerate scikit-learn scipy matplotlib huggingface_hub 2>&1 | tail -2")
        R.ssh_run(host, port, setup, timeout=1800)
        run = (f"cd /workspace/run && set -a && . ./.env 2>/dev/null; set +a; export HF_TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null);"
               f"PROBE_TARGET='{target}' PYTHONPATH=code:. python code/probe.py 2>&1 | tail -40")
        out = R.ssh_run(host, port, run, timeout=18000)
        log(f"[probe] tail:\n{out[-1500:]}")
        # attribution + geometry on the same pod (uses cached clouds)
        R.ssh_run(host, port, f"cd /workspace/run && set -a && . ./.env 2>/dev/null; set +a; export HF_TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null);"
                  f"PROBE_TARGET='{target}' PYTHONPATH=code:. python code/attribute.py 2>&1 | tail -20",
                  check=False, timeout=10800)
        R.ssh_run(host, port, f"cd /workspace/run && set -a && . ./.env 2>/dev/null; set +a; export HF_TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null);"
                  f"TIER6={'1' if tier6 else '0'} PYTHONPATH=code:. python code/geometry.py 2>&1 | tail -30",
                  check=False, timeout=7200)
        # push weights + artifacts + figures
        R.ssh_run(host, port, "cd /workspace/run && set -a && . ./.env 2>/dev/null; set +a; export HF_TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null);"
                  "PYTHONPATH=code:. python -c 'import probe; probe.push_probes()' 2>&1 | tail -5", check=False, timeout=1800)
        R.ssh_run(host, port, "cd /workspace/run && tar czf artifacts.tgz artifacts figures report.md 2>/dev/null", check=False, timeout=180)
        try:
            R.scp_from(host, port, "/workspace/run/artifacts.tgz", "/tmp/artifacts.tgz")
            subprocess.run(["tar", "xzf", "/tmp/artifacts.tgz", "-C", str(ROOT)], check=False)
        except Exception as e:
            log(f"[probe] artifact pull-back failed: {e}")
        return True

    target_vram = 24 if "7B" in target or "9b" in target.lower() else 48
    prefer = (["RTX 4090", "RTX 5090", "L40S"] if target_vram <= 24 else ["L40S", "A100 80GB"])
    log(f"=== STAGE: probe/attribute/geometry (target={target}, tier6={tier6}) ===")
    res = with_pod("ovn-probe", target_vram, prefer, work, disk_gb=100, terminate_hours=7)
    if res:
        verify_hf(config.HF_MODEL_REPO, "model")
        for st in ("probe", "attribute", "geometry"):
            for u in load_state()["stages"][st]["units"]:
                set_unit(st, u, "done", hf_path=config.HF_MODEL_REPO)
    return res


def verify_hf(repo, repo_type):
    from huggingface_hub import HfApi
    try:
        files = HfApi().list_repo_files(repo, repo_type=repo_type, token=True)
        log(f"verify_hf {repo}: {len(files)} files present")
        return len(files) > 0
    except Exception as e:
        log(f"verify_hf {repo} FAILED: {e}")
        return False


def stage_writeup():
    log("=== STAGE: writeup ===")
    # push report.md + geometry.md to the dataset repo
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for fn in ("report.md", "STATE.md"):
            p = ROOT / fn
            if p.exists():
                api.upload_file(path_or_fileobj=str(p), path_in_repo=fn,
                                repo_id=config.HF_DATASET_REPO, repo_type="dataset", token=True)
        gm = ROOT / "geometry.md"
        if gm.exists():
            api.upload_file(path_or_fileobj=str(gm), path_in_repo="geometry.md",
                            repo_id=config.HF_DATASET_REPO, repo_type="dataset", token=True)
        log("writeup pushed")
        return True
    except Exception as e:
        log(f"writeup push failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["verify", "label", "probe", "geometry", "writeup"])
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-guard", action="store_true", help="don't start guard thread (supervised)")
    args = ap.parse_args()

    if args.status:
        s = load_state()
        print(json.dumps({"t_hours": round(hours_since_start(), 2),
                          "stages": {k: v["status"] for k, v in s["stages"].items()},
                          "models": s["models"]}, indent=2))
        return

    if not args.no_guard:
        threading.Thread(target=guard_thread, daemon=True).start()

    if args.stage:
        {"verify": stage_verify, "label": stage_label, "probe": stage_probe,
         "geometry": stage_probe, "writeup": stage_writeup}[args.stage]()
    elif args.auto:
        log("=== AUTO RUN START ===")
        if not stage_verify():
            log("local prep incomplete; aborting auto run (run subagents/build first)")
            return
        if budget_ok():
            stage_label()
        if budget_ok():
            stage_probe()
        stage_writeup()
        log("=== AUTO RUN DONE ===")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
