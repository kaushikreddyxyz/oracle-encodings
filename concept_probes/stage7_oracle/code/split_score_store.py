#!/usr/bin/env python3
"""Split the joint 216-column ClimbMix score store into four per-layer repos.

For each shard sid in 320..362:
  scores_<sid>.npy [n,216] int8  ->  four [n,54] slices, one per block:
    [0:54]    layer-6 detection    -> concept-probes-corpus-scores-layer6
    [54:108]  layer-8 detection    -> concept-probes-corpus-scores-layer8
    [108:162] layer-14 detection   -> concept-probes-corpus-scores-layer14
    [162:216] layer-8 diff-of-means-> concept-probes-corpus-scores-dom-layer8
  tokens_<sid>.npy and docs_<sid>.jsonl are copied verbatim into each repo so
  every repo is self-contained.

Invariants (audited 2026-07-09):
  - idempotent/resumable: a shard is skipped IFF all four repos hold its
    scores/tokens/docs at exact expected byte sizes, checked BEFORE any
    download (sizes come from the source repo's own file metadata);
  - every upload byte-verified against the local file;
  - never deletes anything from the source repos;
  - ANY unexpected exception -> /workspace/HOLD_REASON.txt + exit(1), pod
    left alive as evidence; transient network errors retried with backoff;
  - self-terminate (RunPod GraphQL, User-Agent curl/8.4.0 for Cloudflare)
    only after a full 43/43 sweep verification, and the GraphQL response is
    checked for an "errors" key.

Env: POD_ID (required unless NO_TERMINATE=1), RP_KEY_FILE=/workspace/.rp_key,
HF_TOKEN or logged-in hub, SHARD_MAP=/workspace/shard_repo_map.json.
"""
import os

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")     # hub >= 1.0 (xet)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")   # hub 0.x fallback; BEFORE hub import
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import io
import json
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

U = "kaushikreddyxyz"
SRC_PRIMARY = f"{U}/concept-probes-corpus-scores"
SRC_SECOND = f"{U}/concept-probes-corpus-scores-2"
BLOCKS = [
    ("layer6", 0, 54),
    ("layer8", 54, 108),
    ("layer14", 108, 162),
    ("dom-layer8", 162, 216),
]
DEST = {name: f"{U}/concept-probes-corpus-scores-{name}" for name, _, _ in BLOCKS}
SIDS = list(range(320, 363))
WS = Path(os.environ.get("WORKDIR", "/workspace/split"))
HOLD = Path("/workspace/HOLD_REASON.txt")
LOG = Path("/workspace/split.log")
MAP_PATH = Path(os.environ.get("SHARD_MAP", "/workspace/shard_repo_map.json"))
RETRIES = 5


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [split] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def hold(reason):
    HOLD.write_text(reason)
    log(f"HOLD: {reason[:2000]}")
    sys.exit(1)


def with_retries(desc, fn):
    for attempt in range(RETRIES):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - transient network errors
            if attempt == RETRIES - 1:
                raise
            wait = 15 * (2 ** attempt)
            log(f"retry {attempt+1}/{RETRIES} after error in {desc}: {e!r}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def npy_int8_size(shape):
    """Exact np.save() on-disk size for an int8 C-order array of `shape`."""
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buf, {"descr": "|i1", "fortran_order": False, "shape": shape})
    n = 1
    for s in shape:
        n *= s
    return buf.tell() + n


def npy_int8_rows(total_size, n_cols):
    """Invert npy_int8_size for a 2-D int8 array with n_cols columns."""
    n = (total_size - 128) // n_cols          # header is 128B for these shapes
    if npy_int8_size((n, n_cols)) != total_size:
        return None
    return n


def remote_sizes(api, rid, paths):
    infos = with_retries(f"get_paths_info({rid})",
                         lambda: api.get_paths_info(rid, paths, repo_type="dataset"))
    got = {i.path: (i.lfs.size if i.lfs else i.size) for i in infos}
    return {p: got.get(p) for p in paths}


def validate_startup(api, shard_map):
    if HOLD.exists():
        hold(f"pre-existing HOLD file — clear it before rerunning:\n{HOLD.read_text()}")
    if os.environ.get("NO_TERMINATE") != "1":
        if not os.environ.get("POD_ID"):
            hold("POD_ID env not set (and NO_TERMINATE != 1)")
        key_file = Path(os.environ.get("RP_KEY_FILE", "/workspace/.rp_key"))
        if not key_file.exists() or not key_file.read_text().strip():
            hold(f"RunPod key file missing/empty: {key_file}")
    for sid in SIDS:
        entry = shard_map.get(str(sid))
        if (not isinstance(entry, dict)
                or set(entry) != {"scores", "tokens", "docs"}
                or not all(v in (SRC_PRIMARY, SRC_SECOND) for v in entry.values())):
            hold(f"shard map invalid for sid {sid}: {entry!r}")
    for name in DEST:
        with_retries("create_repo", lambda rid=DEST[name]: api.create_repo(
            rid, repo_type="dataset", exist_ok=True))


def process(api, shard_map):
    for sid in SIDS:
        tag = f"{sid:05d}"
        names = {
            "scores": f"scores_{tag}.npy",
            "tokens": f"tokens_{tag}.npy",
            "docs": f"docs_{tag}.jsonl",
        }
        src = shard_map[str(sid)]

        # -------- source metadata first: enables download-free skip --------
        src_sizes = {}
        for kind, fname in names.items():
            s = remote_sizes(api, src[kind], [fname])[fname]
            if s is None:
                hold(f"shard {sid}: {fname} missing from source {src[kind]}")
            src_sizes[kind] = s
        n = npy_int8_rows(src_sizes["scores"], 216)
        if n is None:
            hold(f"shard {sid}: source scores size {src_sizes['scores']} is not a valid [n,216] int8 npy")
        exp_slice = npy_int8_size((n, 54))

        def shard_done():
            for name, _, _ in BLOCKS:
                rs = remote_sizes(api, DEST[name], list(names.values()))
                if (rs[names["scores"]] != exp_slice
                        or rs[names["tokens"]] != src_sizes["tokens"]
                        or rs[names["docs"]] != src_sizes["docs"]):
                    return False
            return True

        if shard_done():
            log(f"shard {sid}: complete in all four repos — skip (no download)")
            continue

        # -------- download originals --------
        local = {}
        for kind, fname in names.items():
            local[kind] = Path(with_retries(f"download {fname}", lambda k=kind, f=fname: hf_hub_download(
                src[k], f, repo_type="dataset", local_dir=str(WS / "src"))))
        if local["scores"].stat().st_size != src_sizes["scores"]:
            hold(f"shard {sid}: downloaded scores size mismatch")
        scores = np.load(local["scores"], mmap_mode="r")
        if scores.dtype != np.int8 or scores.shape != (n, 216):
            hold(f"shard {sid}: unexpected scores array {scores.dtype} {scores.shape}")

        # -------- slice + upload per block --------
        for name, a, b in BLOCKS:
            out = WS / f"scores_{tag}.npy"    # same filename in every repo
            np.save(out, np.ascontiguousarray(scores[:, a:b]))
            if out.stat().st_size != exp_slice:
                hold(f"shard {sid} -> {name}: local slice size {out.stat().st_size} != expected {exp_slice}")
            ops = [
                CommitOperationAdd(names["scores"], str(out)),
                CommitOperationAdd(names["tokens"], str(local["tokens"])),
                CommitOperationAdd(names["docs"], str(local["docs"])),
            ]
            with_retries(f"commit shard {sid} -> {name}", lambda o=ops, r=DEST[name]: api.create_commit(
                repo_id=r, repo_type="dataset", operations=o,
                commit_message=f"shard {sid}: columns slice + tokens + docs"))
            rs = remote_sizes(api, DEST[name], list(names.values()))
            if (rs[names["scores"]] != exp_slice
                    or rs[names["tokens"]] != src_sizes["tokens"]
                    or rs[names["docs"]] != src_sizes["docs"]):
                hold(f"shard {sid} -> {name}: post-upload size mismatch {rs}")
            out.unlink()
            log(f"shard {sid} -> {name}: OK ({n} tokens, {exp_slice/1e6:.0f}MB)")

        del scores
        for p in local.values():
            p.unlink(missing_ok=True)

    # -------- final sweep: exact-size verification of every file --------
    for sid in SIDS:
        tag = f"{sid:05d}"
        src = shard_map[str(sid)]
        src_scores = remote_sizes(api, src["scores"], [f"scores_{tag}.npy"])[f"scores_{tag}.npy"]
        n = npy_int8_rows(src_scores, 216)
        exp_slice = npy_int8_size((n, 54))
        for name, _, _ in BLOCKS:
            got = remote_sizes(api, DEST[name], [f"scores_{tag}.npy"])[f"scores_{tag}.npy"]
            if got != exp_slice:
                hold(f"final sweep: {name} shard {sid} size {got} != {exp_slice}")
    log("ALL DONE — four per-layer repos complete, 43/43 shards exact-size verified")


def terminate_self():
    if os.environ.get("NO_TERMINATE") == "1":
        log("NO_TERMINATE=1 — leaving pod alive")
        return
    pod_id = os.environ["POD_ID"]
    key = Path(os.environ.get("RP_KEY_FILE", "/workspace/.rp_key")).read_text().strip()
    body = json.dumps({"query": 'mutation{podTerminate(input:{podId:"%s"})}' % pod_id}).encode()
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={key}", data=body,
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.4.0"})
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    if resp.get("errors"):
        hold(f"podTerminate returned errors: {resp['errors']}")
    log(f"pod {pod_id} terminate acknowledged: {resp}")


def main():
    api = HfApi()
    WS.mkdir(parents=True, exist_ok=True)
    shard_map = json.load(open(MAP_PATH))
    validate_startup(api, shard_map)
    process(api, shard_map)
    terminate_self()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        hold(f"unhandled exception:\n{traceback.format_exc()}")
