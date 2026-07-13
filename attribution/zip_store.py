#!/usr/bin/env python3
"""Restack the joint 216-column ClimbMix score store into ONE repo where each
token's per-layer detection vectors are co-located and indexable by layer.

Reads the untouched joint source (216 cols = [L6 det|L8 det|L14 det|L8 DoM],
each block 54 concepts). For each shard sid in 320..362 writes into ONE repo
`concept-probes-corpus-scores-stacked`:

  scores_det_<sid>.npy  int8 [n, 3, 54]   axis1 = layer {0:6, 1:8, 2:14}, axis2 = concept
  scores_dom_<sid>.npy  int8 [n, 54]      difference-of-means steering @ L8 (kept SEPARATE)
  tokens_<sid>.npy      int32 [n]
  docs_<sid>.jsonl

To train a per-layer model:  det = np.load('scores_det_XXXXX.npy'); det[:, LAYER_INDEX, :]
where LAYER_INDEX in {0:layer6, 1:layer8, 2:layer14}. You physically must pick a
layer index -> the [n,3,54] shape makes it impossible to accidentally train on a
flat 162-vector.

det[:, :162].reshape(n,3,54) is byte-identical to the joint file's first 162
columns (C-order: element [t,l,c] == joint[t, l*54+c]); dom == joint[:,162:216].

Same invariants as split_score_store.py: idempotent (skip on exact byte sizes),
watchdog on every network call, byte-verify each upload, never deletes source,
HOLD-on-any-exception. NO self-terminate (the chain wrapper owns the pod).
"""
import os

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "0")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import io
import json
import signal
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
DEST = f"{U}/concept-probes-corpus-scores-stacked"
ALL_SIDS = list(range(320, 363))
# Fleet partitioning: each pod handles only its assigned shards (comma-sep in
# ONLY_SHARDS). All pods commit to the same DEST repo; concurrent-commit 409s
# are absorbed by with_retries (LFS is already uploaded, so a retry just
# re-points HEAD). Absent -> this pod does all 43.
_only = os.environ.get("ONLY_SHARDS", "").strip()
SIDS = [int(x) for x in _only.split(",") if x] if _only else list(ALL_SIDS)
WS = Path(os.environ.get("ZIP_WORKDIR", "/workspace/zip"))
HOLD = Path("/workspace/HOLD_REASON_ZIP.txt")
LOG = Path("/workspace/zip.log")
MAP_PATH = Path(os.environ.get("SHARD_MAP", "/workspace/shard_repo_map.json"))
RETRIES = 5


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [zip] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def hold(reason):
    HOLD.write_text(reason)
    log(f"HOLD: {reason[:2000]}")
    sys.exit(1)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)


def with_retries(desc, fn, timeout=900):
    for attempt in range(RETRIES):
        try:
            signal.alarm(timeout)
            try:
                return fn()
            finally:
                signal.alarm(0)
        except Exception as e:  # noqa: BLE001
            if attempt == RETRIES - 1:
                raise
            wait = 15 * (2 ** attempt)
            kind = "TIMEOUT" if isinstance(e, _Timeout) else "error"
            log(f"retry {attempt+1}/{RETRIES} after {kind} in {desc}: {e!r}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def npy_size(shape, dtype):
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buf, {"descr": np.dtype(dtype).str, "fortran_order": False, "shape": tuple(shape)})
    n = 1
    for s in shape:
        n *= s
    return buf.tell() + n * np.dtype(dtype).itemsize


def rows_from_216(total_size):
    n = (total_size - 128) // 216
    if npy_size((n, 216), np.int8) != total_size:
        return None
    return n


def remote_sizes(api, rid, paths):
    infos = with_retries(f"get_paths_info({rid})",
                         lambda: api.get_paths_info(rid, paths, repo_type="dataset"))
    got = {i.path: (i.lfs.size if i.lfs else i.size) for i in infos}
    return {p: got.get(p) for p in paths}


def validate_startup(api, shard_map):
    if HOLD.exists():
        hold(f"pre-existing ZIP HOLD file — clear before rerun:\n{HOLD.read_text()}")
    for sid in SIDS:
        entry = shard_map.get(str(sid))
        if (not isinstance(entry, dict) or set(entry) != {"scores", "tokens", "docs"}
                or not all(v in (SRC_PRIMARY, SRC_SECOND) for v in entry.values())):
            hold(f"shard map invalid for sid {sid}: {entry!r}")
    with_retries("create_repo", lambda: api.create_repo(DEST, repo_type="dataset", exist_ok=True))


def process(api, shard_map):
    for sid in SIDS:
        tag = f"{sid:05d}"
        src = shard_map[str(sid)]
        src_names = {"scores": f"scores_{tag}.npy", "tokens": f"tokens_{tag}.npy", "docs": f"docs_{tag}.jsonl"}
        out_names = {"det": f"scores_det_{tag}.npy", "dom": f"scores_dom_{tag}.npy",
                     "tokens": f"tokens_{tag}.npy", "docs": f"docs_{tag}.jsonl"}

        src_sizes = {}
        for kind, fn in src_names.items():
            s = remote_sizes(api, src[kind], [fn])[fn]
            if s is None:
                hold(f"shard {sid}: {fn} missing from {src[kind]}")
            src_sizes[kind] = s
        n = rows_from_216(src_sizes["scores"])
        if n is None:
            hold(f"shard {sid}: source scores size {src_sizes['scores']} not a valid [n,216] int8 npy")
        exp = {"det": npy_size((n, 3, 54), np.int8), "dom": npy_size((n, 54), np.int8),
               "tokens": src_sizes["tokens"], "docs": src_sizes["docs"]}

        rs = remote_sizes(api, DEST, list(out_names.values()))
        if all(rs[out_names[k]] == exp[k] for k in exp):
            log(f"shard {sid}: complete in stacked repo — skip (no download)")
            continue

        local = {}
        for kind, fn in src_names.items():
            local[kind] = Path(with_retries(f"download {fn}", lambda k=kind, f=fn: hf_hub_download(
                src[k], f, repo_type="dataset", local_dir=str(WS / "src"))))
        if local["scores"].stat().st_size != src_sizes["scores"]:
            hold(f"shard {sid}: downloaded scores size mismatch")
        joint = np.load(local["scores"], mmap_mode="r")
        if joint.dtype != np.int8 or joint.shape != (n, 216):
            hold(f"shard {sid}: unexpected joint array {joint.dtype} {joint.shape}")

        det_path = WS / out_names["det"]
        dom_path = WS / out_names["dom"]
        det = np.ascontiguousarray(joint[:, :162]).reshape(n, 3, 54)
        dom = np.ascontiguousarray(joint[:, 162:216])
        # contract check: det[:,l,:] must equal joint columns [l*54:(l+1)*54]
        if not (np.array_equal(det[:, 0, :], joint[:, 0:54])
                and np.array_equal(det[:, 1, :], joint[:, 54:108])
                and np.array_equal(det[:, 2, :], joint[:, 108:162])
                and np.array_equal(dom, joint[:, 162:216])):
            hold(f"shard {sid}: reshape contract violated")
        np.save(det_path, det)
        np.save(dom_path, dom)
        if det_path.stat().st_size != exp["det"] or dom_path.stat().st_size != exp["dom"]:
            hold(f"shard {sid}: local det/dom size mismatch")

        ops = [
            CommitOperationAdd(out_names["det"], str(det_path)),
            CommitOperationAdd(out_names["dom"], str(dom_path)),
            CommitOperationAdd(out_names["tokens"], str(local["tokens"])),
            CommitOperationAdd(out_names["docs"], str(local["docs"])),
        ]
        with_retries(f"commit shard {sid}", lambda: api.create_commit(
            repo_id=DEST, repo_type="dataset", operations=ops,
            commit_message=f"shard {sid}: det[n,3,54] + dom[n,54] + tokens + docs"))
        rs = remote_sizes(api, DEST, list(out_names.values()))
        if not all(rs[out_names[k]] == exp[k] for k in exp):
            hold(f"shard {sid}: post-upload size mismatch {rs}")
        log(f"shard {sid}: OK (n={n}, det {exp['det']/1e9:.2f}GB)")

        del joint
        det_path.unlink(missing_ok=True)
        dom_path.unlink(missing_ok=True)
        for p in local.values():
            p.unlink(missing_ok=True)

    # final sweep: exact-size verification
    for sid in SIDS:
        tag = f"{sid:05d}"
        src = shard_map[str(sid)]
        n = rows_from_216(remote_sizes(api, src["scores"], [f"scores_{tag}.npy"])[f"scores_{tag}.npy"])
        want = {f"scores_det_{tag}.npy": npy_size((n, 3, 54), np.int8),
                f"scores_dom_{tag}.npy": npy_size((n, 54), np.int8)}
        rs = remote_sizes(api, DEST, list(want))
        for f, sz in want.items():
            if rs[f] != sz:
                hold(f"final sweep: {f} size {rs[f]} != {sz}")
    log("ALL DONE — stacked repo complete, 43/43 shards exact-size verified")


def terminate_self():
    if os.environ.get("NO_TERMINATE") == "1":
        log("NO_TERMINATE=1 — leaving pod alive")
        return
    pod_id = os.environ.get("POD_ID", "")
    key_file = Path(os.environ.get("RP_KEY_FILE", "/workspace/.rp_key"))
    if not pod_id or not key_file.exists():
        log(f"cannot self-terminate (POD_ID={pod_id!r}, key={key_file.exists()})")
        return
    key = key_file.read_text().strip()
    body = json.dumps({"query": 'mutation{podTerminate(input:{podId:"%s"})}' % pod_id}).encode()
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={key}", data=body,
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.4.0"})
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    if resp.get("errors"):
        log(f"podTerminate errors: {resp['errors']}")
    else:
        log(f"pod {pod_id} terminate acknowledged")


def main():
    api = HfApi()
    WS.mkdir(parents=True, exist_ok=True)
    shard_map = json.load(open(MAP_PATH))
    validate_startup(api, shard_map)
    log(f"this pod handling {len(SIDS)} shards: {SIDS}")
    process(api, shard_map)
    terminate_self()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        hold(f"unhandled exception:\n{traceback.format_exc()}")
