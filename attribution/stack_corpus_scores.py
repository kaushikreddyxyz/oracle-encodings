#!/usr/bin/env python3
"""Build the consolidated `corpus-scores` repo: one [n, 3, 54] int8 array per
ClimbMix shard, co-locating the layer-6/8/14 detection scores per token.

Reads the untouched joint source (216 cols = [L6 det | L8 det | L14 det |
L8 DoM], 54 concepts each). For each shard sid in 320..362 writes:

  scores_<sid>.npy  int8 [n, 3, 54]  axis0=token, axis1=layer {0:6, 1:8, 2:14},
                                     axis2=concept (order in columns.json)
  tokens_<sid>.npy  int32 [n]        copied verbatim from source
  docs_<sid>.jsonl                   copied verbatim from source

DoM steering scores (source cols 162:216) are EXCLUDED by design.

Destination is kaushikreddyxyz/corpus-scores until the cumulative expected
size reaches CAP_BYTES, then kaushikreddyxyz/corpus-scores-overflow. The
assignment is computed from source file sizes in ascending sid order, so it
is deterministic across resumes; it is uploaded as assignment.json.

Built for a disk-constrained laptop:
  - strictly ONE shard on disk at a time; all local files deleted per shard;
  - free-disk guard before each shard (MIN_FREE_GB, default 25);
  - RAM-safe: chunked memmap copy, never materializes a full [n,162] slice.

Because the source repos are slated for deletion once this repo is verified,
every upload is content-verified: local sha256 must equal the LFS sha256 that
HF reports back for the committed file (not just byte size).

Same harness invariants as zip_store.py: idempotent (exact-size skip before
any download), SIGALRM watchdog on every network call, HOLD-on-any-exception,
never deletes anything remote.

Env: HF_TOKEN, SHARD_MAP (json path), WORKDIR, LOG_PATH, HOLD_PATH,
CAP_BYTES (default 280e9), MIN_FREE_GB (default 25), ONLY_SHARDS (optional
comma-sep subset), SEED_ONLY=1 (create repos + metadata, then exit).
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
SEED_META = f"{U}/concept-probes-corpus-scores-stacked"   # det_quant/columns/stats live here
DEST_MAIN = f"{U}/corpus-scores"
DEST_OVF = f"{U}/corpus-scores-overflow"
ALL_SIDS = list(range(320, 363))
_only = os.environ.get("ONLY_SHARDS", "").strip()
SIDS = [int(x) for x in _only.split(",") if x] if _only else list(ALL_SIDS)
WS = Path(os.environ.get("WORKDIR", "/tmp/stack_ws"))
HOLD = Path(os.environ.get("HOLD_PATH", str(WS / "HOLD_REASON_STACK.txt")))
LOG = Path(os.environ.get("LOG_PATH", str(WS / "stack.log")))
MAP_PATH = Path(os.environ.get("SHARD_MAP", "shard_repo_map.json"))
CAP_BYTES = int(float(os.environ.get("CAP_BYTES", "280e9")))
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "25"))
CHUNK_ROWS = 1_000_000                # 216MB materialized per chunk, RAM-safe
RETRIES = 6


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [stack] {msg}"
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
    """Retry with backoff; SIGALRM watchdog turns silent uploader hangs into
    retryable timeouts (main-thread only — this script is single-threaded)."""
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


def source_manifest(api, shard_map):
    """One batched metadata call per source repo -> {sid: {kind: size}}."""
    from collections import defaultdict
    byrepo = defaultdict(list)
    for sid in ALL_SIDS:
        tag = f"{sid:05d}"
        e = shard_map[str(sid)]
        for kind, fn in (("scores", f"scores_{tag}.npy"),
                         ("tokens", f"tokens_{tag}.npy"),
                         ("docs", f"docs_{tag}.jsonl")):
            byrepo[e[kind]].append((sid, kind, fn))
    sizes = {sid: {} for sid in ALL_SIDS}
    for repo, items in byrepo.items():
        got = paths_info(api, repo, [fn for _, _, fn in items])
        for sid, kind, fn in items:
            sz = got[fn][0]
            if sz is None:
                hold(f"source manifest: {fn} missing from {repo}")
            sizes[sid][kind] = sz
    return sizes


def plan_assignment(src_sizes):
    """Deterministic dest per shard: fill DEST_MAIN in sid order up to CAP_BYTES."""
    assign, cum = {}, 0
    for sid in ALL_SIDS:
        n = rows_from_216(src_sizes[sid]["scores"])
        if n is None:
            hold(f"shard {sid}: source scores size {src_sizes[sid]['scores']} not a valid [n,216] int8 npy")
        total = (npy_size((n, 3, 54), np.int8)
                 + src_sizes[sid]["tokens"] + src_sizes[sid]["docs"])
        if cum + total <= CAP_BYTES:
            assign[sid] = DEST_MAIN
            cum += total
        else:
            assign[sid] = DEST_OVF
    return assign


def seed_metadata(api, assign):
    """Create both repos; upload columns/quant/stats/assignment + READMEs."""
    for rid in (DEST_MAIN, DEST_OVF):
        with_retries(f"create_repo({rid})",
                     lambda r=rid: api.create_repo(r, repo_type="dataset", exist_ok=True))
    meta_dir = WS / "seed_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    src_meta = {}
    for f in ("columns.json", "det_quant.json", "det_corpus_stats.json"):
        p = with_retries(f"download {f}", lambda f=f: hf_hub_download(
            SEED_META, f, repo_type="dataset", local_dir=str(meta_dir)))
        src_meta[f] = json.load(open(p))

    cols = src_meta["columns.json"]
    columns = {
        "concepts": cols["concepts"],
        "families": cols["families"],
        "layers": [6, 8, 14],
        "scores_shape": "[n_tokens, 3, 54]",
        "axes": {
            "0": "token position within shard; aligns 1:1 with tokens_<sid>.npy and doc spans in docs_<sid>.jsonl",
            "1": "layer index: 0 -> layer 6, 1 -> layer 8, 2 -> layer 14",
            "2": "concept index into concepts[] (54 entries)",
        },
    }
    q = src_meta["det_quant.json"]
    quant = {
        "zero": q["zero"], "scale": q["scale"],
        "axes": "zero/scale are [layer_idx][concept_idx]; layer_idx 0->L6, 1->L8, 2->L14",
        "dequant": "score = int8_value * scale[l][c] + zero[l][c]  (int8 clipped at +/-4 sigma)",
    }
    ovf_sids = sorted(s for s, r in assign.items() if r == DEST_OVF)
    main_sids = sorted(s for s, r in assign.items() if r == DEST_MAIN)
    assignment = {"cap_bytes": CAP_BYTES,
                  "main": {str(s): DEST_MAIN for s in main_sids},
                  "overflow": {str(s): DEST_OVF for s in ovf_sids}}

    def readme(is_ovf):
        rng = (f"{min(ovf_sids)}-{max(ovf_sids)}" if ovf_sids else "none") if is_ovf \
              else f"{min(main_sids)}-{max(main_sids)}"
        other = (f"Main repo: [{DEST_MAIN}](https://huggingface.co/datasets/{DEST_MAIN})."
                 if is_ovf else
                 (f"Overflow (shards {min(ovf_sids)}-{max(ovf_sids)}): "
                  f"[{DEST_OVF}](https://huggingface.co/datasets/{DEST_OVF})." if ovf_sids else ""))
        return f"""# corpus-scores{"-overflow" if is_ovf else ""}

Concept-probe detection scores for ClimbMix, layers 6/8/14 co-located per token.
{"Overflow continuation — same format as the main repo. " if is_ovf else ""}Shards here: {rng}. {other}

## Files (per shard `<sid>`, ClimbMix shards 320-362)
- `scores_<sid>.npy` — int8 `[n_tokens, 3, 54]`
  - **axis 0**: token (aligns 1:1 with `tokens_<sid>.npy` / `docs_<sid>.jsonl`)
  - **axis 1**: layer — **0 = layer 6, 1 = layer 8, 2 = layer 14**
  - **axis 2**: concept — index into `columns.json` `concepts[]` (54 concepts, 7 families)
- `tokens_<sid>.npy` — int32 `[n_tokens]` gemma tokenizer ids
- `docs_<sid>.jsonl` — per-document metadata (token spans into the shard)

## Provenance
Gold detection probes (best of ridge/DoM/LDA per concept-layer) from
[concept-probes-gemma2-2b](https://huggingface.co/kaushikreddyxyz/concept-probes-gemma2-2b)
`gold_probes/layer06|08|14`, applied to gemma-2-2b residual activations on
ClimbMix shards 320-362 (~46.6M tokens/shard, ~2.0B total). Scores are
standardized per probe over ClimbMix, then int8-quantized (clipped at ±4σ);
dequantize with `quant.json`: `score = int8 * scale[l][c] + zero[l][c]`.
DoM steering scores are NOT in this repo.

## Usage — pick exactly ONE layer per model
```python
import numpy as np
det = np.load("scores_00320.npy")   # [n, 3, 54]
l8 = det[:, 1, :]                   # layer 8 scores, [n, 54]
```
"""

    ops_by_repo = {DEST_MAIN: [], DEST_OVF: []}
    for rid, is_ovf in ((DEST_MAIN, False), (DEST_OVF, True)):
        payloads = {
            "README.md": readme(is_ovf).encode(),
            "columns.json": json.dumps(columns, indent=1).encode(),
            "quant.json": json.dumps(quant, indent=1).encode(),
            "corpus_stats.json": json.dumps(src_meta["det_corpus_stats.json"], indent=1).encode(),
            "assignment.json": json.dumps(assignment, indent=1).encode(),
        }
        for name, data in payloads.items():
            ops_by_repo[rid].append(CommitOperationAdd(name, io.BytesIO(data)))
    for rid, ops in ops_by_repo.items():
        with_retries(f"seed metadata -> {rid}", lambda r=rid, o=ops: api.create_commit(
            repo_id=r, repo_type="dataset", operations=o,
            commit_message="seed: README + columns/quant/stats/assignment"))
    log(f"seeded metadata; main={len(main_sids)} shards, overflow={len(ovf_sids)} shards {ovf_sids}")


def free_gb():
    return shutil.disk_usage(WS).free / 1e9


def process(api, shard_map, src_sizes, assign):
    for sid in SIDS:
        tag = f"{sid:05d}"
        dest = assign[sid]
        src = shard_map[str(sid)]
        n = rows_from_216(src_sizes[sid]["scores"])
        if n is None:
            hold(f"shard {sid}: invalid source scores size")
            return
        exp = {f"scores_{tag}.npy": npy_size((n, 3, 54), np.int8),
               f"tokens_{tag}.npy": src_sizes[sid]["tokens"],
               f"docs_{tag}.jsonl": src_sizes[sid]["docs"]}

        rs = paths_info(api, dest, list(exp))
        if all(rs[f][0] == sz for f, sz in exp.items()):
            log(f"shard {sid}: complete in {dest.split('/')[-1]} — skip")
            continue

        if free_gb() < MIN_FREE_GB:
            hold(f"shard {sid}: only {free_gb():.1f}GB free (< {MIN_FREE_GB}GB) — clear disk and rerun")

        # ---- download source (one shard on disk at a time) ----
        local = {}
        for kind, fn in (("scores", f"scores_{tag}.npy"), ("tokens", f"tokens_{tag}.npy"),
                         ("docs", f"docs_{tag}.jsonl")):
            local[kind] = Path(with_retries(f"download {fn}", lambda k=kind, f=fn: hf_hub_download(
                src[k], f, repo_type="dataset", local_dir=str(WS / "src")), timeout=1800))
        if local["scores"].stat().st_size != src_sizes[sid]["scores"]:
            hold(f"shard {sid}: downloaded scores size mismatch")
        joint = np.load(local["scores"], mmap_mode="r")
        if joint.dtype != np.int8 or joint.shape != (n, 216):
            hold(f"shard {sid}: unexpected joint array {joint.dtype} {joint.shape}")

        # ---- chunked restack [n,216] -> [n,3,54] (RAM-safe) + contract check ----
        out_path = WS / f"scores_{tag}.npy"
        out = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.int8, shape=(n, 3, 54))
        for a in range(0, n, CHUNK_ROWS):
            b = min(a + CHUNK_ROWS, n)
            blk = np.asarray(joint[a:b, :162])
            out[a:b] = blk.reshape(b - a, 3, 54)
            if not (np.array_equal(out[a:b, 0, :], blk[:, 0:54])
                    and np.array_equal(out[a:b, 1, :], blk[:, 54:108])
                    and np.array_equal(out[a:b, 2, :], blk[:, 108:162])):
                hold(f"shard {sid}: reshape contract violated at rows {a}:{b}")
        out.flush()
        del out, joint
        if out_path.stat().st_size != exp[f"scores_{tag}.npy"]:
            hold(f"shard {sid}: local stacked size mismatch")

        # local sha256s BEFORE upload; compared to HF's LFS sha256 after commit
        uploads = {f"scores_{tag}.npy": out_path,
                   f"tokens_{tag}.npy": local["tokens"],
                   f"docs_{tag}.jsonl": local["docs"]}
        local_sha = {fn: sha256_file(p) for fn, p in uploads.items()}

        ops = [CommitOperationAdd(fn, str(p)) for fn, p in uploads.items()]
        with_retries(f"commit shard {sid} -> {dest}", lambda: api.create_commit(
            repo_id=dest, repo_type="dataset", operations=ops,
            commit_message=f"shard {sid}: scores[n,3,54] + tokens + docs"), timeout=2400)

        rs = paths_info(api, dest, list(exp))
        for fn, sz in exp.items():
            got_sz, got_sha = rs[fn]
            if got_sz != sz:
                hold(f"shard {sid}: post-upload size mismatch {fn}: {got_sz} != {sz}")
            if got_sha is not None and got_sha != local_sha[fn]:
                hold(f"shard {sid}: post-upload sha256 mismatch {fn}")
        log(f"shard {sid}: OK -> {dest.split('/')[-1]} (n={n:,}, "
            f"{exp[f'scores_{tag}.npy']/1e9:.2f}GB, sha verified)")

        out_path.unlink(missing_ok=True)
        for p in local.values():
            p.unlink(missing_ok=True)

    # ---- final sweep over ALL 43: exact sizes in assigned dest ----
    bad = []
    for sid in ALL_SIDS:
        tag = f"{sid:05d}"
        n = rows_from_216(src_sizes[sid]["scores"])
        if n is None:
            hold(f"final sweep: shard {sid} invalid source scores size")
            return
        exp = {f"scores_{tag}.npy": npy_size((n, 3, 54), np.int8),
               f"tokens_{tag}.npy": src_sizes[sid]["tokens"],
               f"docs_{tag}.jsonl": src_sizes[sid]["docs"]}
        rs = paths_info(api, assign[sid], list(exp))
        for f, sz in exp.items():
            if rs[f][0] != sz:
                bad.append((sid, f, rs[f][0], sz))
    if bad:
        log(f"final sweep: {len(bad)} files missing/mismatched (other worker may still be running): {bad[:5]}")
    else:
        log("ALL DONE — corpus-scores(+overflow) complete, 43/43 shards exact-size verified")


def main():
    api = HfApi()
    WS.mkdir(parents=True, exist_ok=True)
    if HOLD.exists():
        hold(f"pre-existing HOLD — clear before rerun:\n{HOLD.read_text()}")
    shard_map = json.load(open(MAP_PATH))
    for sid in ALL_SIDS:
        e = shard_map.get(str(sid))
        if (not isinstance(e, dict) or set(e) != {"scores", "tokens", "docs"}
                or not all(v in (SRC_PRIMARY, SRC_SECOND) for v in e.values())):
            hold(f"shard map invalid for sid {sid}: {e!r}")
    src_sizes = source_manifest(api, shard_map)
    assign = plan_assignment(src_sizes)
    n_ovf = sum(1 for r in assign.values() if r == DEST_OVF)
    log(f"assignment: {43 - n_ovf} shards -> {DEST_MAIN}, {n_ovf} -> {DEST_OVF}")
    if os.environ.get("SEED_ONLY") == "1":
        seed_metadata(api, assign)
        return
    log(f"this run handling {len(SIDS)} shards: {SIDS}")
    process(api, shard_map, src_sizes, assign)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        hold(f"unhandled exception:\n{traceback.format_exc()}")
