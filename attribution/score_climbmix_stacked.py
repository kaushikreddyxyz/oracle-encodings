#!/usr/bin/env python3
"""Score ClimbMix-shuffle shards (nanochat training data, shards 0-184) with
the frozen gold probe set — FULL TOKEN COVERAGE, detection-only, stacked.

SPEC (2026-07-10, supersedes the corpus-scores conventions in two ways):
  * EVERY token is annotated. No min-doc-length filter; no 2048 truncation.
    Docs longer than 2048 gemma tokens are scored in consecutive
    NON-OVERLAPPING 2048-token windows (each window forwarded exactly like a
    score_corpus.py doc: BOS prepended, BOS row dropped, eager attention,
    bf16, probes + frozen per-layer nat_mean/nat_std standardization).
    Window-boundary context truncation is an accepted artifact. Empty docs
    get a docs row with n=0. Every parquet row appears in docs_<sid>.jsonl,
    so sum(n) == total gemma token count of the shard.
  * Detection only — no DoM outputs (the dom part of ScoreHead's [B,T,216]
    is computed and discarded; cols 162:216 never stored).

Per shard sid:
  scores_<sid>.npy  int8 [n, 3, 54]  axis1: 0=L6, 1=L8, 2=L14; axis2=concept
                    (store order == corpus-scores columns.json)
  tokens_<sid>.npy  int32 [n]  full BOS-free gemma ids, docs in parquet order
  docs_<sid>.jsonl  {doc, start, n}  n = FULL doc token count (0 allowed)

Quantization: FROZEN quant216.json (corpus-scores lineage, calibrated once on
shard 320) — cols 0:162 only. Never recalibrated.

Repo assignment (fixed by COUNT, 25 shards/repo — full-coverage shards are
~10-10.8GB): climbmix-scored (0-24), -overflow (25-49), -overflow-2 (50-74),
... -overflow-7 (175-184).

Invariants: idempotent (HF exact-size skip), SIGALRM watchdog + retries on
network calls, OOM -> halve batch and retry shard, sha256 of every upload
verified against HF's LFS sha, HOLD on any unexpected error,
/workspace/DONE_SCORE.txt only after all assigned shards verified (self-
terminate watchdog keys on it).

Env: HF_TOKEN, ONLY_SHARDS, BATCH_SIZE (default 32), VALIDATE_FIRST=1
(coverage + distribution report after the first shard -> /workspace/validation.json).
"""
import os

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "0")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import hashlib
import io
import json
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_corpus as sc  # noqa: E402
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download  # noqa: E402

U = "kaushikreddyxyz"
K = 54
DET_COLS = 3 * K
DET_PER_REPO = 25
DET_REPOS = [f"{U}/climbmix-scored"] + [f"{U}/climbmix-scored-overflow" + ("" if i == 1 else f"-{i}")
                                          for i in range(1, 8)]
ALL_SIDS = list(range(0, 185))
WINDOW = sc.MAX_DOC_TOKENS  # 2048 — per-forward window, NOT a truncation
DOC_BUFFER = 1024

WS = Path(os.environ.get("SCORE_WORKDIR", "/workspace/scores"))
HOLD = Path("/workspace/HOLD_SCORE.txt")
DONE = Path("/workspace/DONE_SCORE.txt")
LOG = Path("/workspace/score.log")
HB = "/workspace/hb_score.txt"
QUANT216 = os.environ.get("QUANT216", "/workspace/meta/quant216.json")
PROBE_DIR = os.environ.get("PROBE_DIR", "/workspace/meta")
RETRIES = 6
_only = os.environ.get("ONLY_SHARDS", "").strip()
SIDS = [int(x) for x in _only.split(",") if x] if _only else list(ALL_SIDS)
BATCH0 = int(os.environ.get("BATCH_SIZE", "32"))


def det_repo(sid):
    return DET_REPOS[sid // DET_PER_REPO]


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [attrib] {msg}"
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
            log(f"retry {attempt+1}/{RETRIES} after {kind} in {desc}: {e!r}; sleep {wait}s")
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
        out[i.path] = ((i.lfs.size if i.lfs else i.size), (i.lfs.sha256 if i.lfs else None))
    return {p: out.get(p, (None, None)) for p in paths}


def shard_on_hf(api, sid):
    tag = f"{sid:05d}"
    rs = paths_info(api, det_repo(sid),
                    [f"scores_{tag}.npy", f"tokens_{tag}.npy", f"docs_{tag}.jsonl"])
    tok_sz = rs[f"tokens_{tag}.npy"][0]
    det_sz = rs[f"scores_{tag}.npy"][0]
    if None in (tok_sz, det_sz) or rs[f"docs_{tag}.jsonl"][0] is None:
        return False
    n = (tok_sz - 128) // 4
    return (npy_size((n,), np.int32) == tok_sz
            and det_sz == npy_size((n, 3, K), np.int8))


def upload_verified(api, rid, files):
    shas = {rp: sha256_file(p) for rp, p in files.items()}
    sizes = {rp: Path(p).stat().st_size for rp, p in files.items()}
    ops = [CommitOperationAdd(rp, str(p)) for rp, p in files.items()]
    with_retries(f"commit -> {rid}", lambda: api.create_commit(
        repo_id=rid, repo_type="dataset", operations=ops,
        commit_message=f"attrib: {', '.join(sorted(files))}"), timeout=2400)
    rs = paths_info(api, rid, list(files))
    for rp in files:
        got_sz, got_sha = rs[rp]
        if got_sz != sizes[rp]:
            hold(f"{rid}/{rp}: post-upload size {got_sz} != {sizes[rp]}")
        if got_sha is not None and got_sha != shas[rp]:
            hold(f"{rid}/{rp}: post-upload sha mismatch")


# ------------------------------------------------------------ full coverage
def iter_all_docs(shard):
    """Yield (doc_index, text) for EVERY parquet row, including empty texts
    (as ""), in parquet order. (sc.iter_shard_docs skips falsy texts — this
    spec requires every row to appear in docs_<sid>.jsonl.)"""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(sc._shard_path(shard))
    col = sc._detect_text_column(pf)
    n = 0
    for batch in pf.iter_batches(batch_size=512, columns=[col]):
        for v in batch.column(0):
            yield n, (v.as_py() or "")
            n += 1


def process_shard_full(sid, probe, head, tok, model, device, zero_det, scale_det,
                        batch_size, hb):
    """Full-coverage scorer: every row, every token; long docs in consecutive
    2048-token windows; det cols only. Writes tokens/scores/docs atomically."""
    import torch  # noqa: F401
    tag = f"{sid:05d}"
    paths = {"tokens": WS / f"tokens_{tag}.npy", "scores": WS / f"scores_{tag}.npy",
             "docs": WS / f"docs_{tag}.jsonl"}
    tmp_tokens = WS / f"tokens_{tag}.tmp.npy"
    tmp_scores = WS / f"scores_{tag}.tmp.npy"
    tmp_docs = WS / f"docs_{tag}.jsonl.tmp"

    token_chunks, score_chunks = [], []
    docs_f = open(tmp_docs, "w")
    offset = docs_done = tokens_done = 0
    t0 = time.time()
    total_rows = sc.shard_num_rows(sid)
    # pending docs in parquet order; each: [doc_idx, full_ids, n_windows, {widx: scores}]
    pending = []
    buf = []  # window work items: (pending_slot, widx, ids_window)

    def emit_ready():
        """Flush completed docs from the FRONT of `pending` (parquet order)."""
        nonlocal offset, docs_done, tokens_done
        while pending and len(pending[0][3]) == pending[0][2]:
            doc_idx, ids, n_w, scored = pending.pop(0)
            n = len(ids)
            if n:
                token_chunks.append(np.asarray(ids, dtype=np.int32))
                parts = [scored[w] for w in range(n_w)]
                score_chunks.append(np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0])
            docs_f.write(json.dumps({"doc": doc_idx, "start": offset, "n": n}) + "\n")
            offset += n
            docs_done += 1
            tokens_done += n

    def flush_buf():
        if not buf:
            return
        buf.sort(key=lambda it: len(it[2]))
        for sub_items in sc.chunked(buf, batch_size):
            sub = [(slot[0], ids_w) for slot, _w, ids_w in sub_items]  # (doc_idx, ids)
            raw, lens = sc.forward_score_batch(model, head, sub, tok.bos_token_id,
                                                tok.pad_token_id or 0, device)
            raw_np = raw[:, :, :DET_COLS].float().cpu().numpy()
            for i, (slot, widx, _ids) in enumerate(sub_items):
                nrows = lens[i]
                q = sc.quantize(raw_np[i, :nrows, :], zero_det, scale_det)
                slot[3][widx] = q
            del raw, raw_np
        buf.clear()
        emit_ready()
        hb.maybe_write({"shard": sid, "docs_done": docs_done, "tokens_done": tokens_done,
                         "tok_per_s": tokens_done / max(time.time() - t0, 1e-9),
                         "pct_docs": round(100 * docs_done / max(total_rows, 1), 1)})

    log(f"shard {sid}: {total_rows} rows (full coverage)")
    for doc_idx, text in iter_all_docs(sid):
        ids = tok(text, add_special_tokens=False)["input_ids"] if text else []
        n_w = max(1, (len(ids) + WINDOW - 1) // WINDOW) if ids else 0
        slot = [doc_idx, ids, n_w, {}]
        pending.append(slot)
        for w in range(n_w):
            buf.append((slot, w, ids[w * WINDOW:(w + 1) * WINDOW]))
        if not ids:
            emit_ready()
        if len(buf) >= DOC_BUFFER:
            flush_buf()
    flush_buf()
    emit_ready()
    if pending:
        hold(f"shard {sid}: {len(pending)} docs left unemitted (window bookkeeping bug)")
    docs_f.close()

    tokens_arr = np.concatenate(token_chunks) if token_chunks else np.zeros(0, dtype=np.int32)
    scores_arr = (np.concatenate(score_chunks, axis=0) if score_chunks
                  else np.zeros((0, DET_COLS), dtype=np.int8))
    if scores_arr.shape[0] != tokens_arr.shape[0]:
        hold(f"shard {sid}: rows mismatch tokens={tokens_arr.shape} scores={scores_arr.shape}")
    n = tokens_arr.shape[0]
    np.save(tmp_tokens, tokens_arr.astype(np.int32))
    np.save(tmp_scores, np.ascontiguousarray(scores_arr.astype(np.int8)).reshape(n, 3, K))
    os.replace(tmp_tokens, paths["tokens"])
    os.replace(tmp_scores, paths["scores"])
    os.replace(tmp_docs, paths["docs"])
    log(f"shard {sid}: scored {docs_done} docs, {tokens_done} tokens "
        f"({tokens_done / max(time.time() - t0, 1e-9):.0f} tok/s)")
    return paths, n


def validate_first(api, sid, paths, n, tok):
    """Coverage + window-boundary reproduction + distribution vs shard 320."""
    try:
        rep = {"shard": sid, "n_tokens": int(n), "spec": "full-coverage-det-only-2026-07-10"}
        # (a) coverage: sum(n) over docs == full re-tokenized count over ALL rows
        docs = [json.loads(x) for x in open(paths["docs"]) if x.strip()]
        rep["n_docs"] = len(docs)
        rep["sum_n"] = int(sum(d["n"] for d in docs))
        total, checked_rows = 0, 0
        for _idx, text in iter_all_docs(sid):
            total += len(tok(text, add_special_tokens=False)["input_ids"]) if text else 0
            checked_rows += 1
        rep["retokenized_total"] = int(total)
        rep["rows_in_parquet"] = checked_rows
        rep["coverage_ok"] = (rep["sum_n"] == total == n and len(docs) == checked_rows)
        # (a2) token-id reproduction across window boundaries for 3 longest docs
        tokens = np.load(paths["tokens"], mmap_mode="r")
        texts = dict(iter_all_docs(sid))
        long3 = sorted(docs, key=lambda d: -d["n"])[:3]
        ok_long = []
        for d in long3:
            ids = tok(texts[d["doc"]], add_special_tokens=False)["input_ids"]
            ok_long.append(bool(np.array_equal(np.asarray(ids, np.int32),
                                                np.asarray(tokens[d["start"]:d["start"] + d["n"]]))))
        rep["longdoc_token_reproduction"] = ok_long
        rep["longest_doc_tokens"] = int(long3[0]["n"]) if long3 else 0
        # (b) distributions: first-window vs continuation tokens, vs shard 320 ref
        det = np.load(paths["scores"], mmap_mode="r")
        fw_idx, cont_idx = [], []
        for d in docs:
            if d["n"] == 0:
                continue
            fw_idx.append((d["start"], d["start"] + min(d["n"], WINDOW)))
            if d["n"] > WINDOW:
                cont_idx.append((d["start"] + WINDOW, d["start"] + d["n"]))

        def sample(ranges, cap=2_000_000):
            outs, got = [], 0
            for a, b in ranges:
                if got >= cap:
                    break
                take = min(b - a, cap - got)
                outs.append(np.asarray(det[a:a + take], dtype=np.float64))
                got += take
            return np.concatenate(outs, axis=0) if outs else np.zeros((0, 3, K))
        fw, ct = sample(fw_idx), sample(cont_idx)
        rep["firstwin_int8_mean_per_layer"] = np.mean(fw, axis=(0, 2)).round(3).tolist()
        rep["firstwin_int8_std_per_layer"] = np.std(fw, axis=(0, 2)).round(3).tolist()
        if ct.shape[0]:
            rep["cont_int8_mean_per_layer"] = np.mean(ct, axis=(0, 2)).round(3).tolist()
            rep["cont_int8_std_per_layer"] = np.std(ct, axis=(0, 2)).round(3).tolist()
            rep["n_cont_tokens_sampled"] = int(ct.shape[0])
        ref_p = hf_hub_download(f"{U}/corpus-scores", "scores_00320.npy",
                                repo_type="dataset", local_dir=str(WS / "ref"))
        ref = np.load(ref_p, mmap_mode="r")
        rsub = np.asarray(ref[::max(1, ref.shape[0] // 2_000_000)][:2_000_000], dtype=np.float64)
        rep["ref320_int8_mean_per_layer"] = np.mean(rsub, axis=(0, 2)).round(3).tolist()
        rep["ref320_int8_std_per_layer"] = np.std(rsub, axis=(0, 2)).round(3).tolist()
        rep["fw_vs_ref_percol_meandiff_p90"] = float(np.percentile(
            np.abs(fw.reshape(-1, DET_COLS).mean(axis=0)
                   - rsub.reshape(-1, DET_COLS).mean(axis=0)), 90)) if fw.shape[0] else None
        Path("/workspace/validation.json").write_text(json.dumps(rep, indent=1))
        log(f"validation.json: coverage_ok={rep['coverage_ok']} longdoc={ok_long}")
        Path(ref_p).unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        log(f"validate_first FAILED (non-fatal to scoring): {e!r}")
        Path("/workspace/validation.json").write_text(json.dumps({"error": repr(e)}))


def main():
    import torch
    api = HfApi()
    WS.mkdir(parents=True, exist_ok=True)
    if HOLD.exists():
        hold(f"pre-existing HOLD — clear before rerun:\n{HOLD.read_text()}")
    q = sc.load_quant(QUANT216)
    if q["zero"].shape != (4 * K,):
        hold(f"quant216 shape {q['zero'].shape}, expected (216,)")
    zero_det, scale_det = q["zero"][:DET_COLS], q["scale"][:DET_COLS]
    probe = sc.ProbeSet(PROBE_DIR)
    if probe.layers != [6, 8, 14] or probe.K != K:
        hold(f"probe set mismatch: layers={probe.layers} K={probe.K}")
    device = torch.device("cuda")
    tok, model = sc.load_model_and_tok(sc.GEMMA_MODEL_DEFAULT, "eager", device, None)
    head = sc.ScoreHead(probe, device)
    hb = sc.Heartbeat(HB)
    log(f"model loaded (eager, bf16); FULL-COVERAGE det-only; {len(SIDS)} shards: {SIDS}")

    bs = BATCH0
    first_done = False
    for sid in SIDS:
        tag = f"{sid:05d}"
        if shard_on_hf(api, sid):
            log(f"shard {sid}: complete on HF — skip")
            continue
        t0 = time.time()
        while True:
            try:
                paths, n = process_shard_full(sid, probe, head, tok, model, device,
                                               zero_det, scale_det, bs, hb)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs <= 4:
                    hold(f"shard {sid}: OOM at batch size {bs}")
                bs //= 2
                log(f"shard {sid}: CUDA OOM -> retry at batch size {bs}")
        if not first_done and os.environ.get("VALIDATE_FIRST") == "1":
            validate_first(api, sid, paths, n, tok)
        upload_verified(api, det_repo(sid), {f"scores_{tag}.npy": paths["scores"],
                                              f"tokens_{tag}.npy": paths["tokens"],
                                              f"docs_{tag}.jsonl": paths["docs"]})
        log(f"shard {sid}: OK -> {det_repo(sid).split('/')[-1]} (n={n:,}, "
            f"{npy_size((n, 3, K), np.int8)/1e9:.2f}GB, {(time.time()-t0)/60:.1f} min)")
        first_done = True
        for f in paths.values():
            Path(f).unlink(missing_ok=True)
        for pq_f in Path("/workspace/hf_cache").rglob(f"shard_{tag}.parquet"):
            pq_f.unlink(missing_ok=True)

    bad = [sid for sid in SIDS if not shard_on_hf(api, sid)]
    if bad:
        hold(f"final sweep: shards missing on HF: {bad}")
    DONE.write_text(json.dumps({"shards": SIDS, "t": time.strftime("%F %T")}))
    log(f"ALL DONE — {len(SIDS)} shards verified on HF; DONE marker written")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        hold(f"unhandled exception:\n{traceback.format_exc()}")
