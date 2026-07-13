#!/usr/bin/env python3
"""Complete concept-probes-corpus-scores-dom-layer8 to 43/43 shards.

For each shard sid in ONLY_SHARDS (default: the 14 missing, 349..362), slice
the DoM steering block (columns 162:216 of the joint 216-col source) and
upload scores_<sid>.npy int8 [n, 54] + tokens_<sid>.npy + docs_<sid>.jsonl,
matching the layout of the 29 shards already in the repo. Also seeds
dom_quant.json / dom_corpus_stats.json / README.md (with YAML frontmatter).

Same harness as stack_corpus_scores.py: idempotent exact-size skip, SIGALRM
watchdog, chunked memmap slice (RAM-safe), sha256 verification of every
upload against HF's stored LFS hash, HOLD-on-any-exception, never deletes
anything remote.

Env: HF_TOKEN, SHARD_MAP, WORKDIR, LOG_PATH, HOLD_PATH, ONLY_SHARDS,
MIN_FREE_GB (default 15), SEED_ONLY=1 (metadata only).
"""
import os

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "0")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import hashlib
import io
import json
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

U = "kaushikreddyxyz"
SRC_PRIMARY = f"{U}/concept-probes-corpus-scores"
SRC_SECOND = f"{U}/concept-probes-corpus-scores-2"
SEED_META = f"{U}/concept-probes-corpus-scores-stacked"
DEST = f"{U}/concept-probes-corpus-scores-dom-layer8"
DEFAULT_MISSING = list(range(349, 363))
_only = os.environ.get("ONLY_SHARDS", "").strip()
SIDS = [int(x) for x in _only.split(",") if x] if _only else list(DEFAULT_MISSING)
WS = Path(os.environ.get("WORKDIR", "/workspace/wsdom"))
HOLD = Path(os.environ.get("HOLD_PATH", str(WS / "HOLD_REASON_DOM.txt")))
LOG = Path(os.environ.get("LOG_PATH", str(WS / "dom.log")))
MAP_PATH = Path(os.environ.get("SHARD_MAP", "/workspace/shard_repo_map.json"))
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "15"))
CHUNK_ROWS = 1_000_000
RETRIES = 6


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dom] {msg}"
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
            wait = min(15 * (2 ** attempt), 480)
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
    return n if npy_size((n, 216), np.int8) == total_size else None


def sha256_file(path, bufsize=32 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def paths_info(api, rid, paths):
    infos = with_retries(f"get_paths_info({rid})",
                         lambda: api.get_paths_info(rid, paths, repo_type="dataset"))
    out = {}
    for i in infos:
        out[i.path] = ((i.lfs.size if i.lfs else i.size),
                       (i.lfs.sha256 if i.lfs else None))
    return {p: out.get(p, (None, None)) for p in paths}


def seed_metadata(api):
    meta_dir = WS / "seed_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    src_meta = {}
    for f in ("dom_quant.json", "dom_corpus_stats.json", "columns.json"):
        p = with_retries(f"download {f}", lambda f=f: hf_hub_download(
            SEED_META, f, repo_type="dataset", local_dir=str(meta_dir)))
        src_meta[f] = json.load(open(p))
    cols = src_meta["columns.json"]
    columns = {
        "concepts": cols["concepts"], "families": cols["families"],
        "layer": 8, "method": "difference-of-means (steering)",
        "scores_shape": "[n_tokens, 54]",
        "axes": {"0": "token position within shard; aligns 1:1 with tokens_<sid>.npy / docs_<sid>.jsonl",
                 "1": "concept index into concepts[] (54 entries)"},
    }
    readme = """---
pretty_name: Concept-Probe DoM Steering Scores (gemma-2-2b, layer 8)
tags:
- interpretability
- probing
- gemma-2
size_categories:
- 1B<n<10B
---
# concept-probes-corpus-scores-dom-layer8

Difference-of-means (DoM) STEERING scores for ClimbMix, gemma-2-2b **layer 8**.
Kept separate from the detection score store
[corpus-scores](https://huggingface.co/datasets/kaushikreddyxyz/corpus-scores)
([overflow](https://huggingface.co/datasets/kaushikreddyxyz/corpus-scores-overflow)).

## Files (per shard `<sid>`, ClimbMix shards 320-362, 43 total)
- `scores_<sid>.npy` — int8 `[n_tokens, 54]`; **axis 0** token, **axis 1** concept
  (index into `columns.json` `concepts[]`)
- `tokens_<sid>.npy` — int32 `[n_tokens]` gemma tokenizer ids
- `docs_<sid>.jsonl` — per-document metadata (token spans)

## Provenance
All-DoM steering probes from
[concept-probes-gemma2-2b](https://huggingface.co/kaushikreddyxyz/concept-probes-gemma2-2b)
`gold_probes/layer08` applied to gemma-2-2b residual activations
(~46.6M tokens/shard, ~2.0B total). Scores standardized per probe over
ClimbMix, then int8-quantized (clipped at ±4σ). Dequantize with
`dom_quant.json`: `score = int8 * scale[c] + zero[c]`.
"""
    payloads = {
        "README.md": readme.encode(),
        "columns.json": json.dumps(columns, indent=1).encode(),
        "dom_quant.json": json.dumps(src_meta["dom_quant.json"], indent=1).encode(),
        "dom_corpus_stats.json": json.dumps(src_meta["dom_corpus_stats.json"], indent=1).encode(),
    }
    ops = [CommitOperationAdd(k, io.BytesIO(v)) for k, v in payloads.items()]
    with_retries("seed dom metadata", lambda: api.create_commit(
        repo_id=DEST, repo_type="dataset", operations=ops,
        commit_message="seed: README (yaml frontmatter) + columns/quant/stats"))
    log("seeded dom metadata")


def process(api, shard_map):
    for sid in SIDS:
        tag = f"{sid:05d}"
        src = shard_map[str(sid)]
        src_names = {"scores": f"scores_{tag}.npy", "tokens": f"tokens_{tag}.npy",
                     "docs": f"docs_{tag}.jsonl"}
        src_sizes = {}
        for kind, fn in src_names.items():
            s = paths_info(api, src[kind], [fn])[fn][0]
            if s is None:
                hold(f"shard {sid}: {fn} missing from {src[kind]}")
            src_sizes[kind] = s
        n = rows_from_216(src_sizes["scores"])
        if n is None:
            hold(f"shard {sid}: bad source scores size {src_sizes['scores']}")
            return
        exp = {f"scores_{tag}.npy": npy_size((n, 54), np.int8),
               f"tokens_{tag}.npy": src_sizes["tokens"],
               f"docs_{tag}.jsonl": src_sizes["docs"]}

        rs = paths_info(api, DEST, list(exp))
        if all(rs[f][0] == sz for f, sz in exp.items()):
            log(f"shard {sid}: complete in dom repo — skip")
            continue

        if shutil.disk_usage(WS).free / 1e9 < MIN_FREE_GB:
            hold(f"shard {sid}: below {MIN_FREE_GB}GB free")

        local = {}
        for kind, fn in src_names.items():
            local[kind] = Path(with_retries(f"download {fn}", lambda k=kind, f=fn: hf_hub_download(
                src[k], f, repo_type="dataset", local_dir=str(WS / "src")), timeout=1800))
        joint = np.load(local["scores"], mmap_mode="r")
        if joint.dtype != np.int8 or joint.shape != (n, 216):
            hold(f"shard {sid}: unexpected joint {joint.dtype} {joint.shape}")

        out_path = WS / f"scores_{tag}.npy"
        out = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.int8, shape=(n, 54))
        for a in range(0, n, CHUNK_ROWS):
            b = min(a + CHUNK_ROWS, n)
            blk = np.asarray(joint[a:b, 162:216])
            out[a:b] = blk
            if not np.array_equal(out[a:b], blk):
                hold(f"shard {sid}: slice contract violated at rows {a}:{b}")
        out.flush()
        del out, joint
        if out_path.stat().st_size != exp[f"scores_{tag}.npy"]:
            hold(f"shard {sid}: local dom size mismatch")

        uploads = {f"scores_{tag}.npy": out_path,
                   f"tokens_{tag}.npy": local["tokens"],
                   f"docs_{tag}.jsonl": local["docs"]}
        local_sha = {fn: sha256_file(p) for fn, p in uploads.items()}
        ops = [CommitOperationAdd(fn, str(p)) for fn, p in uploads.items()]
        with_retries(f"commit shard {sid}", lambda: api.create_commit(
            repo_id=DEST, repo_type="dataset", operations=ops,
            commit_message=f"shard {sid}: dom[n,54] + tokens + docs"), timeout=2400)

        rs = paths_info(api, DEST, list(exp))
        for fn, sz in exp.items():
            got_sz, got_sha = rs[fn]
            if got_sz != sz:
                hold(f"shard {sid}: post-upload size mismatch {fn}")
            if got_sha is not None and got_sha != local_sha[fn]:
                hold(f"shard {sid}: post-upload sha mismatch {fn}")
        log(f"shard {sid}: OK (n={n:,}, {exp[f'scores_{tag}.npy']/1e9:.2f}GB, sha verified)")
        out_path.unlink(missing_ok=True)
        for p in local.values():
            p.unlink(missing_ok=True)
    log(f"dom pass complete for {len(SIDS)} shards")


def main():
    api = HfApi()
    WS.mkdir(parents=True, exist_ok=True)
    if HOLD.exists():
        hold(f"pre-existing HOLD — clear before rerun:\n{HOLD.read_text()}")
    shard_map = json.load(open(MAP_PATH))
    seed_metadata(api)
    if os.environ.get("SEED_ONLY") == "1":
        return
    log(f"handling {len(SIDS)} shards: {SIDS}")
    process(api, shard_map)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        hold(f"unhandled exception:\n{traceback.format_exc()}")
