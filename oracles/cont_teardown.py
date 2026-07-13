#!/usr/bin/env python3
"""Stage-7 continuation pod: final upload + verify + self-terminate.

Runs AFTER train_oracle_perlayer.py exits (cont_supervise.sh sequences it).
Clean stop (metrics.jsonl ends with a {"final": true, ...} record) ->
upload best_stripped.pt / metrics.jsonl / best.pt (WITH optimizer state, as
best_full.pt — the overnight run stripped it and made exact resume
impossible; never again) / train.log to <HF_SUBDIR>/ on HF_REPO, byte-verify
each, write <HF_SUBDIR>/DONE.json, then podTerminate via GraphQL.

Sad path (trainer crashed, upload mismatch, or upload hangs past the alarm):
write /workspace/HOLD_REASON.txt, upload it to <HF_SUBDIR>/HOLD_REASON.txt,
and DO NOT terminate — the pod stays up as evidence.

Env (from /workspace/.env via cont_supervise.sh): LAYER, HF_SUBDIR, HF_REPO,
POD_ID. Secrets: /workspace/.hf_token (exported by supervisor), /workspace/.rp_key.
"""
import json
import os
import signal
import time
import urllib.request

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

WS = "/workspace"
LAYER = int(os.environ["LAYER"])
SUB = os.environ["HF_SUBDIR"]
REPO = os.environ.get("HF_REPO", "kaushikreddyxyz/oracle-encoders")
POD_ID = os.environ.get("POD_ID") or os.environ.get("RUNPOD_POD_ID", "")
METRICS = f"{WS}/run/metrics.jsonl"
HOLD = f"{WS}/HOLD_REASON.txt"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [teardown-L{LAYER}] {msg}"
    print(line, flush=True)


class UploadTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise UploadTimeout()


def with_timeout(fn, seconds):
    """SIGALRM watchdog: HF uploads have hung silently before (b346961)."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)


def rp_key():
    return open(f"{WS}/.rp_key").read().strip()


def terminate_self():
    q = 'mutation{podTerminate(input:{podId:"%s"})}' % POD_ID
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={rp_key()}",
        data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json",
                 # Cloudflare 403s the default Python-urllib User-Agent.
                 "User-Agent": "curl/8.4.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        log(f"podTerminate resp: {r.read().decode()[:200]}")


def final_record():
    try:
        lines = [json.loads(l) for l in open(METRICS) if l.strip()]
    except Exception:
        return None
    for l in reversed(lines):
        if l.get("final"):
            return l
    return None


def main():
    from huggingface_hub import HfApi
    api = HfApi()

    fin = final_record()
    if fin is None:
        reason = "trainer did NOT write a final record (crash?) — holding pod for inspection"
        log(f"HOLD: {reason}")
        with open(HOLD, "w") as f:
            f.write(reason + "\n")
        try:
            api.upload_file(path_or_fileobj=HOLD, path_in_repo=f"{SUB}/HOLD_REASON.txt",
                            repo_id=REPO, repo_type="model")
        except Exception as e:  # noqa: BLE001
            log(f"HOLD upload failed: {e!r}")
        return
    log(f"clean stop: {fin.get('stop_reason')} step={fin.get('step')} "
        f"tokens={fin.get('train_tokens')} best_r2={fin.get('best_median_r2')}")

    def remote_size(path):
        try:
            for x in api.get_paths_info(REPO, paths=[path], repo_type="model"):
                if getattr(x, "path", None) == path:
                    return getattr(x, "size", None)
        except Exception:  # noqa: BLE001
            return None
        return None

    problems = []
    uploads = [
        (f"{WS}/run/best_stripped.pt", f"{SUB}/best_stripped.pt", True),
        (f"{WS}/run/metrics.jsonl", f"{SUB}/metrics.jsonl", True),
        (f"{WS}/run/best.pt", f"{SUB}/best_full.pt", True),
        (f"{WS}/train.log", f"{SUB}/logs/train.log", False),
        (f"{WS}/prefetch.log", f"{SUB}/logs/prefetch.log", False),
    ]
    for local, remote, required in uploads:
        if not os.path.exists(local):
            if required:
                problems.append(f"missing local {local}")
            continue
        lsize = os.path.getsize(local)
        ok = False
        for attempt in (1, 2):
            if remote_size(remote) == lsize:
                ok = True
                break
            try:
                with_timeout(lambda: api.upload_file(
                    path_or_fileobj=local, path_in_repo=remote,
                    repo_id=REPO, repo_type="model",
                    commit_message=f"cont teardown: {remote}"), 5400)
                ok = remote_size(remote) == lsize
                if ok:
                    break
            except Exception as e:  # noqa: BLE001
                log(f"upload attempt {attempt} {remote}: {e!r}")
        log(f"{remote}: local={lsize} ok={ok}")
        if required and not ok:
            problems.append(f"{remote} not verified on repo")

    if problems:
        reason = "upload verification problems: " + "; ".join(problems)
        log(f"HOLD: {reason}")
        with open(HOLD, "w") as f:
            f.write(reason + "\n")
        try:
            api.upload_file(path_or_fileobj=HOLD, path_in_repo=f"{SUB}/HOLD_REASON.txt",
                            repo_id=REPO, repo_type="model")
        except Exception as e:  # noqa: BLE001
            log(f"HOLD upload failed: {e!r}")
        return

    done = {"layer": LAYER, "subdir": SUB, "pod_id": POD_ID,
            "stop_reason": fin.get("stop_reason"), "step": fin.get("step"),
            "train_tokens": fin.get("train_tokens"),
            "best_median_r2": fin.get("best_median_r2"),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    dp = f"{WS}/DONE.json"
    with open(dp, "w") as f:
        json.dump(done, f, indent=1)
    with_timeout(lambda: api.upload_file(path_or_fileobj=dp, path_in_repo=f"{SUB}/DONE.json",
                                         repo_id=REPO, repo_type="model",
                                         commit_message=f"cont teardown: {SUB} DONE"), 600)
    log("all artifacts verified + DONE.json up. Self-terminating in 60s.")
    time.sleep(60)
    terminate_self()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        log(f"FATAL {type(e).__name__}: {e} — NOT terminating")
        with open(HOLD, "a") as f:
            f.write(f"teardown FATAL: {e!r}\n")
