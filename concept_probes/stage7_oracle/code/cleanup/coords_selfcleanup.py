#!/usr/bin/env python3
"""Stage-7 oracle coords: per-pod self-cleanup + self-terminate.

Deployed identically to all 6 coords pods (nohup). Each pod:
  1. Polls until the `precompute_coords.py --mode sweep` process exits AND
     every assigned shard has its shard_NNNNN.done marker.
     Sad path (process dead but shards incomplete): still upload what exists,
     tag the state file "INCOMPLETE", and DO NOT terminate (leave evidence).
  2. Uploads its shard files (coords_/index_/meta_/.done) — plus coord_fit.npz
     and coord_fit.json if this is the fit-source pod (coords1) — to the public
     HF dataset repo, verifying each file's remote byte size against local.
  3. Writes done_podN.json (shard list + sizes + script sha) INTO the repo.
  4. SELF-TERMINATES via RunPod GraphQL podTerminate (own pod id) on the happy
     path only. RUNPOD_API_KEY is read from a root-only file (chmod 600).

Config via env:
  POD_NUM        1..6 (coords1..coords6); POD_INDEX = POD_NUM-1
  POD_ID         RunPod pod id (for self-terminate)
  IS_FIT_SOURCE  "1" on coords1 -> also upload coord_fit.{npz,json}
  RP_KEY_FILE    path to chmod-600 file holding the RunPod API key
  DRY_RUN        "1" -> verify terminate path (self-describe query + print the
                 mutation string) but DO NOT actually terminate. Default "0".
  REPO           HF dataset repo id (default kaushikreddyxyz/stage7-oracle-coords)
  COORDS_DIR     default /workspace/coords
  POLL_SECS      default 120
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

REPO = os.environ.get("REPO", "kaushikreddyxyz/stage7-oracle-coords")
COORDS_DIR = os.environ.get("COORDS_DIR", "/workspace/coords")
SHARDS_DIR = os.path.join(COORDS_DIR, "shards")
POD_NUM = int(os.environ["POD_NUM"])
POD_INDEX = POD_NUM - 1
POD_ID = os.environ["POD_ID"]
IS_FIT_SOURCE = os.environ.get("IS_FIT_SOURCE", "0") == "1"
RP_KEY_FILE = os.environ.get("RP_KEY_FILE", "/workspace/.rp_key")
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
POLL_SECS = int(os.environ.get("POLL_SECS", "120"))
STATE_PATH = os.path.join(COORDS_DIR, "coords_cleanup_state.json")
LOG = os.path.join(COORDS_DIR, f"coords_cleanup_pod{POD_NUM}.log")

N_SHARDS = 191  # shards 0..190
ASSIGNED = [s for s in range(N_SHARDS) if s % 6 == POD_INDEX]


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [pod{POD_NUM}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def script_sha():
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def sweep_alive():
    r = subprocess.run(["pgrep", "-f", "precompute_coords.py --mode sweep"],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""


def done_shards():
    return [s for s in ASSIGNED
            if os.path.exists(os.path.join(SHARDS_DIR, f"shard_{s:05d}.done"))]


def write_state(status, uploaded, mismatches, terminated):
    st = {
        "pod_num": POD_NUM, "pod_index": POD_INDEX, "pod_id": POD_ID,
        "status": status, "assigned_shards": ASSIGNED,
        "done_shards": done_shards(), "uploaded_shards": uploaded,
        "size_mismatches": mismatches, "terminated": terminated,
        "is_fit_source": IS_FIT_SOURCE, "script_sha256": script_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(STATE_PATH, "w") as f:
        json.dump(st, f, indent=2)
    return st


# --------------------------------------------------------------------------- #
# RunPod GraphQL
# --------------------------------------------------------------------------- #
def rp_key():
    return open(RP_KEY_FILE).read().strip()


def graphql(query):
    url = f"https://api.runpod.io/graphql?api_key={rp_key()}"
    data = json.dumps({"query": query}).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json",
                                          # RunPod API sits behind Cloudflare,
                                          # which 403s the default Python-urllib
                                          # User-Agent. Use a curl-like UA.
                                          "User-Agent": "curl/8.4.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def self_describe():
    q = ('query{pod(input:{podId:"%s"}){id name desiredStatus '
         'runtime{uptimeInSeconds}}}' % POD_ID)
    return graphql(q)


def terminate_self(dry_run):
    mutation = 'mutation{podTerminate(input:{podId:"%s"})}' % POD_ID
    desc = self_describe()  # verifies auth + pod exists
    log(f"self-describe: {json.dumps(desc.get('data', desc))[:200]}")
    if dry_run:
        log(f"DRY_RUN: would send -> {mutation}  (NOT executed)")
        return
    log(f"terminating self -> {mutation}")
    resp = graphql(mutation)
    log(f"podTerminate resp: {json.dumps(resp)[:200]}")


# --------------------------------------------------------------------------- #
# HF upload + verify
# --------------------------------------------------------------------------- #
def upload_and_verify(api, local_path, repo_path):
    """Upload one file, then verify remote size == local size. Idempotent:
    skips re-upload if remote already matches. Returns (repo_path, size, ok)."""
    from huggingface_hub import HfApi  # noqa
    lsize = os.path.getsize(local_path)

    def remote_size():
        try:
            info = api.get_paths_info(REPO, paths=[repo_path], repo_type="dataset")
            for x in info:
                if getattr(x, "path", None) == repo_path:
                    return getattr(x, "size", None)
        except Exception:
            return None
        return None

    if remote_size() == lsize:
        return repo_path, lsize, True
    api.upload_file(path_or_fileobj=local_path, path_in_repo=repo_path,
                    repo_id=REPO, repo_type="dataset",
                    commit_message=f"pod{POD_NUM}: add {repo_path}")
    ok = remote_size() == lsize
    return repo_path, lsize, ok


def main():
    log(f"start. assigned={len(ASSIGNED)} shards "
        f"[{ASSIGNED[0]}..{ASSIGNED[-1]}] fit_source={IS_FIT_SOURCE} "
        f"dry_run={DRY_RUN}")
    write_state("WAITING", [], [], False)

    # ---- poll: wait for sweep process to exit (happy path needs BOTH exit
    #      AND all-assigned-done; sad path = exit but incomplete) ----
    while sweep_alive():
        d = done_shards()
        log(f"sweep alive; done {len(d)}/{len(ASSIGNED)}")
        time.sleep(POLL_SECS)

    d = set(done_shards())
    complete = d >= set(ASSIGNED)
    status = "COMPLETE" if complete else "INCOMPLETE"
    log(f"sweep exited. done {len(d)}/{len(ASSIGNED)} -> {status}")

    # ---- upload every shard that has a .done marker ----
    from huggingface_hub import HfApi
    api = HfApi()
    uploaded, mismatches, sizes = [], [], {}
    for s in sorted(d):
        for prefix, ext in (("coords", "int8"), ("index", "npy"),
                            ("meta", "json"), ("shard", "done")):
            fn = f"{prefix}_{s:05d}.{ext}"
            lp = os.path.join(SHARDS_DIR, fn)
            if not os.path.exists(lp):
                log(f"MISSING local {fn}")
                continue
            rp, sz, ok = upload_and_verify(api, lp, f"shards/{fn}")
            sizes[fn] = sz
            if not ok:
                mismatches.append(rp)
                log(f"SIZE MISMATCH {rp}")
        uploaded.append(s)
        log(f"uploaded shard {s} ({len([k for k in sizes if str(s).zfill(5) in k])} files)")

    # ---- fit source also uploads coord_fit.{npz,json} ----
    if IS_FIT_SOURCE:
        for fn in ("coord_fit.npz", "coord_fit.json"):
            lp = os.path.join(COORDS_DIR, fn)
            if os.path.exists(lp):
                rp, sz, ok = upload_and_verify(api, lp, fn)
                sizes[fn] = sz
                if not ok:
                    mismatches.append(rp)
                log(f"fit file {fn} uploaded ok={ok}")
            else:
                log(f"WARN fit file missing: {lp}")

    # ---- per-pod completion manifest into the repo ----
    marker = {
        "pod_num": POD_NUM, "pod_index": POD_INDEX, "pod_id": POD_ID,
        "status": status, "assigned_shards": ASSIGNED,
        "uploaded_shards": uploaded, "file_sizes": sizes,
        "size_mismatches": mismatches, "is_fit_source": IS_FIT_SOURCE,
        "script_sha256": script_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    mpath = os.path.join(COORDS_DIR, f"done_pod{POD_NUM}.json")
    with open(mpath, "w") as f:
        json.dump(marker, f, indent=2)
    api.upload_file(path_or_fileobj=mpath, path_in_repo=f"done_pod{POD_NUM}.json",
                    repo_id=REPO, repo_type="dataset",
                    commit_message=f"pod{POD_NUM}: completion marker ({status})")
    log(f"uploaded done_pod{POD_NUM}.json ({status})")

    healthy = complete and not mismatches
    write_state(status, uploaded, mismatches, terminated=False)

    if not healthy:
        log(f"NOT healthy (complete={complete}, mismatches={len(mismatches)}). "
            f"Leaving pod ALIVE as evidence. No termination.")
        return

    # ---- happy path: self-terminate ----
    terminate_self(DRY_RUN)
    if not DRY_RUN:
        write_state(status, uploaded, mismatches, terminated=True)
    log("done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL {type(e).__name__}: {e}")
        try:
            write_state("ERROR", [], [str(e)[:200]], False)
        except Exception:
            pass
        sys.exit(1)
