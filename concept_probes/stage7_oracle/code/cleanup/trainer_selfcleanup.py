#!/usr/bin/env python3
"""Stage-7 oracle trainer pod: self-cleanup + self-terminate.

Waits (foreground poll, 120s) for THREE conditions, then finalizes uploads and
self-terminates via RunPod GraphQL podTerminate:

  (a) expB-learn done (metrics.jsonl step >= 1282) and its HF/wandb watcher has
      fired (expB_learn_watch.log shows "HF upload finished"); we also directly
      ensure expB-learn/best.pt is on the encoder repo in case the watcher
      failed.
  (b) archival driver state = 43/43 shards (320..362) uploaded AND
      final_corpus_stats_uploaded == true.
  (c) coords HF repo (kaushikreddyxyz/stage7-oracle-coords) has all 6
      done_podN.json markers (trainer's role as rsync source is then over).

Sad-path rule: anything unexpected (a training/driver process died before its
work finished, or an upload size mismatch) -> DO NOT terminate; write the
reason to /workspace/HOLD_REASON.txt AND upload it to the coords HF repo so
it's visible from anywhere.

Env: POD_ID, RP_KEY_FILE (chmod600), DRY_RUN ("1" = verify terminate path
only), ENCODER_REPO, COORDS_REPO.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

POD_ID = os.environ.get("POD_ID", "0te256xap9vakv")
RP_KEY_FILE = os.environ.get("RP_KEY_FILE", "/workspace/.rp_key")
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
ENCODER_REPO = os.environ.get("ENCODER_REPO", "kaushikreddyxyz/stage7-oracle-encoder")
COORDS_REPO = os.environ.get("COORDS_REPO", "kaushikreddyxyz/stage7-oracle-coords")
POLL_SECS = int(os.environ.get("POLL_SECS", "120"))

WS = "/workspace"
METRICS = f"{WS}/expB_learn/metrics.jsonl"
WATCH_LOG = f"{WS}/expB_learn_watch.log"
UP_STATE = f"{WS}/hf_upload_state.json"
HOLD = f"{WS}/HOLD_REASON.txt"
LOG = f"{WS}/trainer_cleanup.log"
FINAL_STEP = 1282
SCORE_SHARDS = list(range(320, 363))  # 320..362 == 43 shards


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [trainer] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# --------------------------- RunPod GraphQL ---------------------------------- #
def rp_key():
    return open(RP_KEY_FILE).read().strip()


def graphql(query):
    url = f"https://api.runpod.io/graphql?api_key={rp_key()}"
    data = json.dumps({"query": query}).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def terminate_self(dry_run):
    mutation = 'mutation{podTerminate(input:{podId:"%s"})}' % POD_ID
    desc = graphql('query{pod(input:{podId:"%s"}){id name desiredStatus}}' % POD_ID)
    log(f"self-describe: {json.dumps(desc.get('data', desc))[:200]}")
    if dry_run:
        log(f"DRY_RUN: would send -> {mutation}  (NOT executed)")
        return
    log(f"terminating self -> {mutation}")
    log(f"podTerminate resp: {json.dumps(graphql(mutation))[:200]}")


# --------------------------- condition probes -------------------------------- #
def pgrep(pat):
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0


def expb_step():
    try:
        last = [l for l in open(METRICS) if l.strip()][-1]
        return int(json.loads(last).get("step", 0))
    except Exception:
        return -1


def watcher_fired():
    try:
        return "HF upload finished" in open(WATCH_LOG).read()
    except Exception:
        return False


def driver_state():
    try:
        st = json.load(open(UP_STATE))
        up = set(st.get("uploaded_shards", []))
        ok = set(SCORE_SHARDS) <= up and st.get("final_corpus_stats_uploaded")
        return bool(ok), up
    except Exception:
        return False, set()


def coords_markers_ready(api):
    try:
        files = set(api.list_repo_files(COORDS_REPO, repo_type="dataset"))
        return all(f"done_pod{i}.json" in files for i in range(1, 7))
    except Exception as e:
        log(f"coords marker check err: {type(e).__name__} {str(e)[:100]}")
        return False


def hold(reason, api):
    log(f"HOLD: {reason}")
    with open(HOLD, "w") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} trainer self-cleanup HOLD\n\n{reason}\n")
    try:
        api.upload_file(path_or_fileobj=HOLD, path_in_repo="HOLD_REASON_trainer.txt",
                        repo_id=COORDS_REPO, repo_type="dataset",
                        commit_message="trainer: HOLD reason (not terminating)")
        log("uploaded HOLD_REASON to coords repo")
    except Exception as e:
        log(f"failed to upload HOLD reason: {e}")


def ensure_on_repo(api, local, repo_path, repo, repo_type="model"):
    """Upload local->repo_path if remote size != local size. Returns ok bool."""
    if not os.path.exists(local):
        return None
    lsize = os.path.getsize(local)

    def remote_size():
        try:
            for x in api.get_paths_info(repo, paths=[repo_path], repo_type=repo_type):
                if getattr(x, "path", None) == repo_path:
                    return getattr(x, "size", None)
        except Exception:
            return None
        return None

    if remote_size() == lsize:
        log(f"already on {repo}: {repo_path}")
        return True
    api.upload_file(path_or_fileobj=local, path_in_repo=repo_path, repo_id=repo,
                    repo_type=repo_type, commit_message=f"trainer cleanup: {repo_path}")
    ok = remote_size() == lsize
    log(f"uploaded {repo_path} ok={ok}")
    return ok


def main():
    from huggingface_hub import HfApi
    api = HfApi()
    log(f"start. dry_run={DRY_RUN}")

    while True:
        step = expb_step()
        a_ready = step >= FINAL_STEP
        a_failed = (not a_ready) and (step >= 0) and (not pgrep("train_encoder.py"))

        b_ready, up = driver_state()
        b_failed = (not b_ready) and (not pgrep("hf_upload_driver.py"))

        c_ready = coords_markers_ready(api)

        log(f"a: step={step} ready={a_ready} watcher_fired={watcher_fired()} | "
            f"b: {len(up)}/43 ready={b_ready} | c: coords_markers={c_ready}")

        if a_failed:
            return hold(f"expB-learn stalled at step {step} < {FINAL_STEP} and "
                        f"train_encoder.py is not running.", api)
        if b_failed:
            return hold(f"archival driver incomplete ({len(up)}/43, "
                        f"final_corpus_stats missing) and hf_upload_driver.py "
                        f"is not running.", api)
        if a_ready and b_ready and c_ready:
            break
        time.sleep(POLL_SECS)

    # --------- finalize uploads ---------
    problems = []
    # expA_prod best.pt is expected at encoder-repo root best.pt (verify).
    r = ensure_on_repo(api, f"{WS}/expA_prod/best.pt", "best.pt", ENCODER_REPO)
    if r is False:
        problems.append("expA_prod best.pt size mismatch")
    # expB-learn best.pt (watcher may have failed) -> ensure present.
    r = ensure_on_repo(api, f"{WS}/expB_learn/best.pt", "expB-learn/best.pt", ENCODER_REPO)
    if r is False:
        problems.append("expB-learn best.pt size mismatch")
    # optional artifacts + logs.
    ensure_on_repo(api, f"{WS}/g2_retention.json", "fullft-prod/g2_retention.json", ENCODER_REPO)
    for lg in ("expB_learn.log", "g2_run.log", "hf_push_stage7.log", "expB_learn_watch.log"):
        ensure_on_repo(api, f"{WS}/{lg}", f"logs/{lg}", ENCODER_REPO)

    if problems:
        return hold("upload verification problems: " + "; ".join(problems), api)

    log("all conditions met and artifacts verified. Self-terminating.")
    terminate_self(DRY_RUN)
    log("done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL {type(e).__name__}: {e}")
        sys.exit(1)
