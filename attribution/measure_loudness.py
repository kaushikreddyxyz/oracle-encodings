#!/usr/bin/env python3
"""Donor loudness — how loud the 54 gold concepts are NATIVELY in the donor
model (gemma-2-2b), in the SAME units as the nanochat injection gate.

The injection site adds ``x + gate·rms(x).detach()·z/rms(z)`` — gate = ‖Δx‖/‖x‖,
a dimensionless fraction of the local residual-stream norm. Probe z-scores are
standardized readouts (units of per-concept corpus σ); this module converts them
to stream-relative loudness so a donor-matched gate can be chosen on principled
grounds.

Pipeline (frozen; verified against attribution/score_climbmix_stacked.py +
score_corpus.py::ScoreHead — DO NOT change without re-deriving the identity):
  step-1: x_std = (x − nat_mean_L) / nat_std_L        (per-layer standardization)
  probe:  raw_c = ⟨x_std, w_c⟩ + b_c
  step-2: z_c   = (raw_c − mu2_c) / std2_c            (corpus standardization)
mu2/std2 come from the store's corpus_stats.json; std2_c ≈ quant.scale·127/4.

For layer L ∈ {6,8,14}, concept c (probe_set.json main_block_concepts ORDER ==
the int8 store's axis-2 order — NEVER the name-sorted `concepts` key), x = raw
residual hidden state at L for a non-BOS token, d_model = 2304:
  1. v_c = w_c ⊘ nat_std_L   (raw-space ridge read direction);  u_c = v_c/‖v_c‖
  2. κ_c = std2_c / ‖v_c‖     (raw-space norm per 1 z-σ)                  [analytic]
  3. λ_c = κ_c / median‖x‖    (fraction of the residual stream per 1σ)    [measure]
  4. ℓ_c = |⟨x,u_c⟩ − m_c| / ‖x‖,  m_c = sample-mean ⟨x,u_c⟩             [measure]
  5. IDENTITY (self-audit): (⟨x,u_c⟩ − m_c) = (z_c − z̄_c)·κ_c EXACTLY (affine
     pipeline). median rel-err of ℓ_c vs |z_c−z̄_c|·κ_c/‖x‖ must be < 1e-3, else
     the pipeline was misread — STOP, do not upload.
  6. DoM: probe_set_dom_steering_l6_l8_l14.npz W_dom is a STANDARDIZED-space read
     direction (== probe_set_arrays W_dom_abl at L8; applied to standardized
     activations in ScoreHead), stored in name-sorted `concepts` order. We
     reindex it to main_block order and convert to the raw-space read direction
     exactly like the ridge (v_dom = W_dom ⊘ nat_std), then normalize — honoring
     the "raw-space direction" intent. ℓ_dom_c as in (4); purely empirical.
  7. Subspace total: per (layer, kind), QR-orthonormalize the 54 u's → Q;
     ℓ_tot = ‖Qᵀ(x − x̄)‖/‖x‖ — the whole-packet analogue of the gate.
  8. residual_norm: mean + p05/p25/p50/p75/p95/p99 of ‖x‖ per layer.
  9. "active" tokens: recomputed z_c ≥ 2.

Subcommands: analytic (CPU, $0 — κ only) | measure (gemma over sampled store
windows — fills residual_norm + empirical fields + gates) | upload (push
loudness.json to store repo roots, one commit each, ONLY after gates pass).

Two store variants share this code (different sampling conventions, both feed the
EXACT stored token ids so the recomputed scores reproduce the store):
  climbmix-scored(+overflow) — full-coverage 2048-token tiling, shards 0-184
  corpus-scores(+overflow)   — eval store, ≥64-token / 2048-truncation, 320-362
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = Path(__file__).resolve().parent / "out"
GEMMA_MODEL = "google/gemma-2-2b"
LAYERS = [6, 8, 14]
D_MODEL = 2304
WINDOW = 2048  # per-forward window; matches the scoring convention (NOT a truncation)
QUANTS = ("p05", "p25", "p50", "p75", "p95", "p99")

# All eight climbmix store repos (loudness.json is a property of gemma+probes+
# corpus, identical across the sharded repos). First overflow has NO numeric
# suffix; there is no "-overflow-1".
CLIMBMIX_REPOS = ["kaushikreddyxyz/climbmix-scored"] + [
    "kaushikreddyxyz/climbmix-scored-overflow" + ("" if i == 1 else f"-{i}") for i in range(1, 8)]
CORPUS_REPOS = ["kaushikreddyxyz/corpus-scores", "kaushikreddyxyz/corpus-scores-overflow"]


# --------------------------------------------------------------------------
# Probe geometry (CPU, $0) — raw-space directions, κ. Frozen probe definition.
# --------------------------------------------------------------------------
class ProbeGeom:
    def __init__(self, out_dir: Path = OUT_DIR):
        meta = json.load(open(out_dir / "probe_set.json"))
        self.concepts = list(meta["main_block_concepts"])  # store axis-2 order
        self.K = len(self.concepts)
        assert list(meta["layers"]) == LAYERS, meta["layers"]
        arr = np.load(out_dir / "probe_set_arrays.npz")
        W = np.asarray(arr["W"], np.float64)                # [3,K,D] std-space ridge
        self.nat_std = np.asarray(arr["nat_std"], np.float64)   # [3,D]
        self.nat_mean = np.asarray(arr["nat_mean"], np.float64)  # [3,D]
        self.b = np.asarray(arr["b"], np.float64)           # [3,K]
        self.W = W
        assert W.shape == (3, self.K, D_MODEL)

        # DoM steering npz: W_dom [3,K,D] in name-sorted `concepts` order == the
        # std-space read direction used by ScoreHead. Reindex to main_block order
        # and convert to the raw-space read direction (⊘ nat_std) like the ridge.
        dom = np.load(out_dir / "probe_set_dom_steering_l6_l8_l14.npz")
        name_sorted = list(meta["concepts"])               # W_dom row order
        perm = [name_sorted.index(c) for c in self.concepts]  # name -> main_block
        Wdom = np.asarray(dom["W_dom"], np.float64)[:, perm, :]  # [3,K,D] main order
        dom_nat_std = np.asarray(dom["nat_std"], np.float64)    # [3,D]

        # Per-layer raw-space directions + norms.
        self.V_ridge, self.U_ridge, self.vnorm_ridge = self._dirs(W, self.nat_std)
        self.V_dom, self.U_dom, self.vnorm_dom = self._dirs(Wdom, dom_nat_std)

    @staticmethod
    def _dirs(Wstd, nat_std):
        V = Wstd / nat_std[:, None, :]                     # [3,K,D] raw-space
        vnorm = np.linalg.norm(V, axis=2)                  # [3,K]
        U = V / vnorm[:, :, None]                           # unit
        return V, U, vnorm

    def kappa(self, std2):
        """std2 [3,K] (corpus σ) -> κ [3,K] = std2 / ‖v_ridge‖."""
        return np.asarray(std2, np.float64) / self.vnorm_ridge


# --------------------------------------------------------------------------
# Store metadata + ranged .npy readers (no full-shard download).
# --------------------------------------------------------------------------
def _repo_or_dir_json(loc: str, name: str) -> dict:
    if os.path.isdir(loc):
        return json.load(open(os.path.join(loc, name)))
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(loc, name, repo_type="dataset")))


def load_store_meta(repo: str) -> dict:
    cols = _repo_or_dir_json(repo, "columns.json")
    quant = _repo_or_dir_json(repo, "quant.json")
    stats = _repo_or_dir_json(repo, "corpus_stats.json")
    return {"columns": cols, "quant": quant, "corpus_stats": stats}


def _http_range(url: str, start: int, end: int) -> bytes:
    import requests
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(10, 600))
    r.raise_for_status()
    data = r.content
    assert len(data) == end - start + 1, f"server ignored Range ({len(data)} B) {url}"
    return data


def _npy_header(url: str):
    head = _http_range(url, 0, 11)
    assert head[:6] == b"\x93NUMPY", f"not npy: {url}"
    if head[6] == 1:
        hlen, off = int.from_bytes(head[8:10], "little"), 10
    else:
        hlen, off = int.from_bytes(head[8:12], "little"), 12
    hdr = ast.literal_eval(_http_range(url, off, off + hlen - 1).decode("latin1"))
    assert not hdr["fortran_order"], "need C order"
    return np.dtype(hdr["descr"]), hdr["shape"], off + hlen


def read_npy_rows(url: str, r0: int, r1: int):
    """Rows [r0, r1) of a remote C-order .npy via one Range request -> (arr, total_rows)."""
    dtype, shape, data_off = _npy_header(url)
    total = shape[0]
    r1 = min(r1, total)
    row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
    buf = _http_range(url, data_off + r0 * row_bytes, data_off + r1 * row_bytes - 1)
    return np.frombuffer(buf, dtype=dtype).reshape((r1 - r0,) + tuple(shape[1:])), total


def _shard_url(repo: str, name: str):
    from huggingface_hub import hf_hub_url
    return hf_hub_url(repo, name, repo_type="dataset")


def load_shard_head(repo: str, sid: int, rows: int):
    """First ``rows`` rows of tokens + int8 scores for shard sid + its doc spans.
    Local dir: mmap. HF repo: ranged HTTP. Returns (tokens[int32], scores[int8
    n,3,54], docs[list], total_rows)."""
    tag = f"{sid:05d}"
    docs = []
    docs_path = os.path.join(repo, f"docs_{tag}.jsonl") if os.path.isdir(repo) else None
    if docs_path is None:
        from huggingface_hub import hf_hub_download
        docs_path = hf_hub_download(repo, f"docs_{tag}.jsonl", repo_type="dataset")
    with open(docs_path) as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    if os.path.isdir(repo):
        tok = np.load(os.path.join(repo, f"tokens_{tag}.npy"), mmap_mode="r")
        sco = np.load(os.path.join(repo, f"scores_{tag}.npy"), mmap_mode="r")
        total = tok.shape[0]
        rows = min(rows, total)
        return np.asarray(tok[:rows]), np.asarray(sco[:rows]), docs, total
    tok, total = read_npy_rows(_shard_url(repo, f"tokens_{tag}.npy"), 0, rows)
    sco, tot2 = read_npy_rows(_shard_url(repo, f"scores_{tag}.npy"), 0, rows)
    assert total == tot2, (total, tot2)
    return tok.copy(), sco.copy(), docs, total


# --------------------------------------------------------------------------
# Streaming statistics (rank-free correlation helpers, no scipy dependency).
# --------------------------------------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties averaged) — for Spearman via Pearson-on-ranks."""
    a = np.asarray(a)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), np.float64)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i + 1
        while j < n and sa[j] == sa[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0  # 1-based average rank
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rankdata(a), _rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def quantile_dict(x: np.ndarray, qs=QUANTS) -> dict:
    pct = [int(q[1:]) for q in qs]
    v = np.percentile(np.asarray(x, np.float64), pct)
    return {q: float(val) for q, val in zip(qs, v)}


# --------------------------------------------------------------------------
# gemma forward — raw residual hidden states at LAYERS for BOS-free windows.
# --------------------------------------------------------------------------
def load_gemma(device: str):
    import torch
    # This machine's torchvision is ABI-broken against torch 2.10 (its C++ ops
    # don't register); transformers' lazy loader otherwise poisons the Gemma2Model
    # import. torchvision is irrelevant here — force it "unavailable" before import.
    import transformers.utils.import_utils as _iu
    _iu.is_torchvision_available = lambda: False
    if hasattr(_iu, "_torchvision_available"):
        _iu._torchvision_available = False
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEMMA_MODEL)
    dtype = torch.float16 if device == "mps" else (torch.bfloat16 if device == "cuda" else torch.float32)
    # eager is MANDATORY: sdpa silently drops gemma-2's logit softcapping.
    model = AutoModel.from_pretrained(GEMMA_MODEL, dtype=dtype, attn_implementation="eager")
    model.eval().to(device)
    return tok, model


def forward_windows(model, tok, windows, device):
    """windows: list[list[int]] (BOS-free ids, each ≤ WINDOW). Returns list of
    per-window {L: x[n,D] float32} — BOS row dropped, matching the scoring
    convention. Padded batch, eager attention."""
    import torch
    lens = [len(w) for w in windows]
    Lmax = max(lens) + 1
    bos = tok.bos_token_id
    pad = tok.pad_token_id or 0
    ids = torch.full((len(windows), Lmax), pad, dtype=torch.long)
    attn = torch.zeros((len(windows), Lmax), dtype=torch.long)
    for i, w in enumerate(windows):
        ids[i, 0] = bos
        if w:
            ids[i, 1:1 + len(w)] = torch.tensor(w, dtype=torch.long)
        attn[i, :1 + len(w)] = 1
    with torch.no_grad():
        out = model(input_ids=ids.to(device), attention_mask=attn.to(device),
                    output_hidden_states=True)
    res = []
    for i, n in enumerate(lens):
        d = {}
        for L in LAYERS:
            h = out.hidden_states[L + 1][i, 1:1 + n, :].float().cpu().numpy()  # drop BOS row
            d[L] = h
        res.append(d)
    del out
    return res


# --------------------------------------------------------------------------
# Measurement accumulator.
# --------------------------------------------------------------------------
class Accum:
    """Per-token components needed for quantiles, kept in CPU numpy (fp32).
    Streaming would need two gemma passes for the sample means; storing the
    projections is cheaper and the token budget is bounded."""

    def __init__(self, geom: ProbeGeom):
        self.geom = geom
        self.buf = {L: {"xnorm": [], "ridge_comp": [], "dom_comp": [],
                        "z": [], "stored_z": [], "ridge_Q": [], "dom_Q": []}
                    for L in LAYERS}
        # QR of the 54 unit read directions -> orthonormal basis of the subspace.
        self.Q_ridge = {L: np.linalg.qr(geom.U_ridge[i].T)[0] for i, L in enumerate(LAYERS)}
        self.Q_dom = {L: np.linalg.qr(geom.U_dom[i].T)[0] for i, L in enumerate(LAYERS)}
        self.n_tokens = 0

    def add(self, x_by_layer, stored_z_by_layer):
        g = self.geom
        for i, L in enumerate(LAYERS):
            x = np.asarray(x_by_layer[L], np.float64)          # [n,D]
            b = self.buf[L]
            xnorm = np.linalg.norm(x, axis=1)                   # [n]
            xstd = (x - g.nat_mean[i]) / g.nat_std[i]
            raw = xstd @ g.W[i].T + g.b[i]                      # [n,K]
            b["xnorm"].append(xnorm.astype(np.float32))
            b["ridge_comp"].append((x @ g.U_ridge[i].T).astype(np.float32))
            b["dom_comp"].append((x @ g.U_dom[i].T).astype(np.float32))
            b["z"].append(raw)                                  # standardized below (needs mu2/std2)
            b["stored_z"].append(np.asarray(stored_z_by_layer[L], np.float32))
            b["ridge_Q"].append((x @ self.Q_ridge[L]).astype(np.float32))
            b["dom_Q"].append((x @ self.Q_dom[L]).astype(np.float32))
        self.n_tokens += len(x_by_layer[LAYERS[0]])

    def finalize(self, corpus_stats):
        """Turn buffers into arrays; standardize recomputed raw -> z with the
        store's mu2/std2 (same as the stored scores)."""
        mu2 = np.asarray(corpus_stats["mean"], np.float64)   # [3,K]
        std2 = np.asarray(corpus_stats["std"], np.float64)   # [3,K]
        arr = {}
        for i, L in enumerate(LAYERS):
            b = self.buf[L]
            raw = np.concatenate(b["z"], axis=0)              # [N,K] raw scores
            z = (raw - mu2[i]) / std2[i]
            arr[L] = {
                "xnorm": np.concatenate(b["xnorm"]),
                "ridge_comp": np.concatenate(b["ridge_comp"], axis=0),
                "dom_comp": np.concatenate(b["dom_comp"], axis=0),
                "z": z,
                "stored_z": np.concatenate(b["stored_z"], axis=0),
                "ridge_Q": np.concatenate(b["ridge_Q"], axis=0),
                "dom_Q": np.concatenate(b["dom_Q"], axis=0),
            }
        return arr


def loudness_stats(comp: np.ndarray, xnorm: np.ndarray, z: np.ndarray,
                   active_thresh=2.0):
    """comp [N,K]=⟨x,u_c⟩, xnorm [N], z [N,K] recomputed. Returns per-concept
    all-token + active-token loudness quantiles. m_c = sample-mean ⟨x,u_c⟩."""
    m = comp.mean(axis=0)                                     # [K]
    ell = np.abs(comp - m) / xnorm[:, None]                   # [N,K]
    K = comp.shape[1]
    all_p = {"p50": [], "p95": [], "p99": []}
    act_p = {"p50": [], "p95": [], "n_active": []}
    for c in range(K):
        col = ell[:, c]
        all_p["p50"].append(float(np.percentile(col, 50)))
        all_p["p95"].append(float(np.percentile(col, 95)))
        all_p["p99"].append(float(np.percentile(col, 99)))
        mask = z[:, c] >= active_thresh
        n_act = int(mask.sum())
        act_p["n_active"].append(n_act)
        if n_act:
            act_p["p50"].append(float(np.percentile(col[mask], 50)))
            act_p["p95"].append(float(np.percentile(col[mask], 95)))
        else:
            act_p["p50"].append(0.0)
            act_p["p95"].append(0.0)
    return m, all_p, act_p


def subspace_total(Qproj: np.ndarray, xnorm: np.ndarray):
    """Qproj [N,54] = Qᵀx. ℓ_tot = ‖Qᵀ(x−x̄)‖/‖x‖."""
    centered = Qproj - Qproj.mean(axis=0)
    ell = np.linalg.norm(centered, axis=1) / xnorm
    return {q: float(np.percentile(ell, int(q[1:]))) for q in ("p50", "p90", "p95", "p99")}


# --------------------------------------------------------------------------
# Schema (v1) writer / merger.
# --------------------------------------------------------------------------
def _empty_schema(geom: ProbeGeom, model=GEMMA_MODEL):
    per_layer_null = {str(L): None for L in LAYERS}
    return {
        "version": 1, "model": model, "d_model": D_MODEL, "layers": LAYERS,
        "concepts": list(geom.concepts),
        "corpus": None,
        "residual_norm": dict(per_layer_null),
        "ridge": {"kappa": dict(per_layer_null), "loudness_per_sigma": dict(per_layer_null),
                  "all_loudness": dict(per_layer_null), "active_loudness": dict(per_layer_null)},
        "dom": {"all_loudness": dict(per_layer_null), "active_loudness": dict(per_layer_null)},
        "subspace_total": {"ridge": dict(per_layer_null), "dom": dict(per_layer_null)},
        "crosscheck": {"identity_median_rel_err": None, "z_spearman_min": None,
                       "z_median_abs_diff_max": None},
        "provenance": {"created": None, "device": None, "dtype": None},
    }


def load_or_init(path: Path, geom: ProbeGeom) -> dict:
    if path.exists():
        return json.load(open(path))
    return _empty_schema(geom)


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# analytic subcommand
# --------------------------------------------------------------------------
def cmd_analytic(args):
    geom = ProbeGeom()
    meta = load_store_meta(args.store)
    cols = meta["columns"]["concepts"]
    assert cols == geom.concepts, "store columns.json order != main_block_concepts (permutation!)"
    std2 = np.asarray(meta["corpus_stats"]["std"], np.float64)   # [3,K]
    kappa = geom.kappa(std2)                                     # [3,K]

    # cross-check: std2_c ≈ local quant.scale·127/4 (calibration vs full-corpus σ).
    ql = json.load(open(OUT_DIR / "quant.json"))
    qscale = np.asarray(ql["scale"], np.float64)[:3 * geom.K].reshape(3, geom.K)
    ratio = std2 / (qscale * 127.0 / 4.0)

    out = load_or_init(Path(args.out), geom)
    for i, L in enumerate(LAYERS):
        out["ridge"]["kappa"][str(L)] = kappa[i].tolist()
    out["crosscheck"]["std2_vs_quant_ratio_median"] = float(np.median(ratio))
    out["crosscheck"]["std2_vs_quant_ratio_range"] = [float(ratio.min()), float(ratio.max())]
    out["provenance"]["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out["provenance"]["analytic_store"] = args.store
    write_json(Path(args.out), out)
    print(f"[analytic] κ written for L{LAYERS} -> {args.out}")
    for i, L in enumerate(LAYERS):
        print(f"  L{L}: κ median={np.median(kappa[i]):.4f} "
              f"range=[{kappa[i].min():.4f},{kappa[i].max():.4f}]")
    print(f"  std2 vs quant·127/4: median ratio {np.median(ratio):.4f} "
          f"range [{ratio.min():.4f},{ratio.max():.4f}]")


# --------------------------------------------------------------------------
# measure subcommand
# --------------------------------------------------------------------------
def _sample_docs(repo, sid, rows, max_docs, rng):
    """Return list of (token_ids[list], stored_int8[n,3,54]) for docs fully
    inside the first ``rows`` rows of shard sid, sampled up to max_docs."""
    tok, sco, docs, total = load_shard_head(repo, sid, rows)
    got = [d for d in docs if d["n"] > 0 and d["start"] + d["n"] <= tok.shape[0]]
    idx = rng.permutation(len(got))[:max_docs]
    out = []
    for j in idx:
        d = got[int(j)]
        s, n = d["start"], d["n"]
        out.append((np.asarray(tok[s:s + n], np.int64).tolist(),
                    np.asarray(sco[s:s + n], np.int8)))
    return out, total


def cmd_measure(args):
    import torch
    geom = ProbeGeom()
    meta = load_store_meta(args.store)
    assert meta["columns"]["concepts"] == geom.concepts, "store column order != main_block (permutation!)"
    corpus_stats = meta["corpus_stats"]
    quant = meta["quant"]
    q_zero = np.asarray(quant["zero"], np.float64)   # [3,K]
    q_scale = np.asarray(quant["scale"], np.float64)  # [3,K]
    mu2 = np.asarray(corpus_stats["mean"], np.float64)
    std2 = np.asarray(corpus_stats["std"], np.float64)

    shards = [int(s) for s in args.shards.split(",") if s.strip()]
    per_shard = max(1, args.n_docs // len(shards))
    rng = np.random.default_rng(args.seed)

    print(f"[measure] store={args.store} shards={shards} device={args.device} "
          f"target n_docs={args.n_docs} max_tokens={args.max_tokens}")
    tok, model = load_gemma(args.device)
    acc = Accum(geom)
    n_docs_done = 0
    shards_used = []
    t0 = time.time()
    for sid in shards:
        if acc.n_tokens >= args.max_tokens:
            break
        docs, total = _sample_docs(args.store, sid, args.rows_per_shard, per_shard, rng)
        shards_used.append(sid)
        print(f"[measure] shard {sid}: {len(docs)} docs sampled (shard has {total:,} rows)")
        # window each doc into ≤WINDOW chunks; batch windows across docs.
        win_items = []  # (doc_local_id, widx, ids)
        doc_map = {}    # doc_local_id -> {"stored": int8[n,3,54], "n": n, "windows": {}}
        for k, (ids, stored) in enumerate(docs):
            n = len(ids)
            nw = (n + WINDOW - 1) // WINDOW
            doc_map[k] = {"stored": stored, "n": n, "nw": nw, "x": {}}
            for w in range(nw):
                win_items.append((k, w, ids[w * WINDOW:(w + 1) * WINDOW]))
        win_items.sort(key=lambda it: len(it[2]))
        bs = args.batch
        for i in range(0, len(win_items), bs):
            batch = win_items[i:i + bs]
            try:
                outs = forward_windows(model, tok, [b[2] for b in batch], args.device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    raise SystemExit(f"[measure] OOM at batch {bs}; rerun with --batch {bs // 2}")
                raise
            for (k, w, _ids), xo in zip(batch, outs):
                doc_map[k]["x"][w] = xo
        # reassemble per-doc token order, add to accumulator with stored z.
        for k, dm in doc_map.items():
            if acc.n_tokens >= args.max_tokens:
                break
            x_by_layer = {L: np.concatenate([dm["x"][w][L] for w in range(dm["nw"])], axis=0)
                          for L in LAYERS}
            stored = dm["stored"]                            # int8 [n,3,54]
            # dequant + standardize per layer/concept: stored[:, li, c]*scale[li,c]+zero[li,c]
            stored_z = {}
            for li, L in enumerate(LAYERS):
                r = stored[:, li, :].astype(np.float64) * q_scale[li] + q_zero[li]
                stored_z[L] = (r - mu2[li]) / std2[li]
            acc.add(x_by_layer, stored_z)
            n_docs_done += 1
        print(f"[measure]  running: {n_docs_done} docs, {acc.n_tokens:,} tokens, "
              f"{acc.n_tokens / max(time.time() - t0, 1e-9):.0f} tok/s")

    dur = time.time() - t0
    print(f"[measure] forward done: {n_docs_done} docs, {acc.n_tokens:,} tokens in {dur/60:.1f} min")
    arr = acc.finalize(corpus_stats)

    # ---- gates + aggregation ----
    kappa = geom.kappa(std2)                                 # [3,K]
    out = load_or_init(Path(args.out), geom)
    for i, L in enumerate(LAYERS):
        out["ridge"]["kappa"][str(L)] = kappa[i].tolist()

    identity_errs = []
    z_spearmans = []
    z_absdiffs = []
    for i, L in enumerate(LAYERS):
        a = arr[L]
        xnorm = a["xnorm"].astype(np.float64)
        # residual-norm distribution
        rn = quantile_dict(xnorm)
        rn["mean"] = float(xnorm.mean())
        out["residual_norm"][str(L)] = rn
        # ridge loudness + λ
        m_ridge, all_r, act_r = loudness_stats(a["ridge_comp"], xnorm, a["z"])
        out["ridge"]["all_loudness"][str(L)] = all_r
        out["ridge"]["active_loudness"][str(L)] = act_r
        out["ridge"]["loudness_per_sigma"][str(L)] = (kappa[i] / rn["p50"]).tolist()
        # dom loudness
        _, all_d, act_d = loudness_stats(a["dom_comp"], xnorm, a["z"])
        out["dom"]["all_loudness"][str(L)] = all_d
        out["dom"]["active_loudness"][str(L)] = act_d
        # subspace totals
        out["subspace_total"]["ridge"][str(L)] = subspace_total(a["ridge_Q"], xnorm)
        out["subspace_total"]["dom"][str(L)] = subspace_total(a["dom_Q"], xnorm)
        # IDENTITY gate (5): |comp - m| vs |z - z̄|·κ  (both /‖x‖ cancels)
        comp = a["ridge_comp"].astype(np.float64)
        z = a["z"]
        lhs = np.abs(comp - comp.mean(axis=0))
        rhs = np.abs(z - z.mean(axis=0)) * kappa[i]
        denom = np.maximum(np.abs(lhs), 1e-9)
        rel = np.abs(lhs - rhs) / denom
        identity_errs.append(float(np.median(rel)))
        # z cross-check vs stored int8 z (per column)
        zc = a["z"]; zs = a["stored_z"].astype(np.float64)
        for c in range(geom.K):
            z_spearmans.append(spearman(zc[:, c], zs[:, c]))
            z_absdiffs.append(float(np.median(np.abs(zc[:, c] - zs[:, c]))))

    id_med = float(np.median(identity_errs))
    sp_min = float(np.min(z_spearmans))
    ad_max = float(np.max(z_absdiffs))
    out["crosscheck"]["identity_median_rel_err"] = id_med
    out["crosscheck"]["z_spearman_min"] = sp_min
    out["crosscheck"]["z_median_abs_diff_max"] = ad_max
    out["corpus"] = {"dataset": args.store, "shards_sampled": shards_used,
                     "n_docs": n_docs_done, "n_tokens": int(acc.n_tokens),
                     "variant": args.corpus_name}
    out["provenance"] = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "device": args.device,
                         "dtype": "fp16" if args.device == "mps" else
                                  ("bf16" if args.device == "cuda" else "fp32"),
                         "measure_store": args.store, "corpus_name": args.corpus_name}
    write_json(Path(args.out), out)

    print("\n===== GATES =====")
    print(f"identity median rel-err: {id_med:.2e}  (gate < 1e-3)")
    print(f"z crosscheck: spearman min {sp_min:.4f} (need >0.99; <0.95 hard fail), "
          f"median|Δz| max {ad_max:.4f} (need <0.15)")
    ok_identity = id_med < 1e-3
    ok_z = sp_min > 0.99 and ad_max < 0.15
    hard_fail = sp_min < 0.95
    print("\n===== HEADLINE =====")
    for i, L in enumerate(LAYERS):
        lam = kappa[i] / out["residual_norm"][str(L)]["p50"]
        st = out["subspace_total"]["ridge"][str(L)]
        std_dom = out["subspace_total"]["dom"][str(L)]
        print(f"  L{L}: ‖x‖ p50={out['residual_norm'][str(L)]['p50']:.2f}  "
              f"λ median={np.median(lam):.4f} range=[{lam.min():.4f},{lam.max():.4f}]  "
              f"ℓ_tot ridge p50={st['p50']:.4f}/p95={st['p95']:.4f}  "
              f"dom p50={std_dom['p50']:.4f}")
    if hard_fail:
        print("\n*** HARD FAIL: z spearman < 0.95 (permutation-scale). DO NOT UPLOAD. ***")
        sys.exit(2)
    if not ok_identity:
        print("\n*** IDENTITY GATE FAILED: pipeline misread. DO NOT UPLOAD. ***")
        sys.exit(2)
    if not ok_z:
        print("\n*** Z CROSSCHECK soft-failed (spearman<=0.99 or |Δz|>=0.15). Review before upload. ***")
        sys.exit(3)
    print("\nGATES PASSED — loudness.json ready for upload.")


# --------------------------------------------------------------------------
# upload subcommand
# --------------------------------------------------------------------------
def cmd_upload(args):
    from huggingface_hub import HfApi, CommitOperationAdd
    path = Path(args.file)
    obj = json.load(open(path))
    # refuse to upload an ungated / analytic-only file.
    cc = obj.get("crosscheck", {})
    if cc.get("identity_median_rel_err") is None:
        sys.exit("[upload] file has no measurement/gates (analytic-only); refusing.")
    if cc["identity_median_rel_err"] >= 1e-3:
        sys.exit(f"[upload] identity gate not passed ({cc['identity_median_rel_err']}); refusing.")
    if cc.get("z_spearman_min", 0) <= 0.99 or cc.get("z_median_abs_diff_max", 1) >= 0.15:
        sys.exit(f"[upload] z crosscheck not passed (spearman {cc.get('z_spearman_min')}, "
                 f"|Δz| {cc.get('z_median_abs_diff_max')}); refusing.")
    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    api = HfApi()
    for rid in repos:
        api.create_commit(repo_id=rid, repo_type="dataset",
                          operations=[CommitOperationAdd("loudness.json", str(path))],
                          commit_message=f"loudness: donor loudness v1 ({obj['corpus']['variant']})")
        print(f"[upload] {path.name} -> {rid}/loudness.json OK")
    print(f"[upload] pushed to {len(repos)} repo(s).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analytic", help="CPU, $0: κ_c + std2/quant cross-check")
    a.add_argument("--store", default="kaushikreddyxyz/climbmix-scored")
    a.add_argument("--out", default=str(OUT_DIR / "loudness.json"))
    a.set_defaults(fn=cmd_analytic)

    m = sub.add_parser("measure", help="gemma over sampled store windows -> empirical loudness + gates")
    m.add_argument("--store", default="kaushikreddyxyz/climbmix-scored")
    m.add_argument("--corpus-name", default="climbmix", help="variant label in schema (climbmix|corpus-scores)")
    m.add_argument("--shards", default="2,12,22", help="well-separated shard ids in the store's primary repo")
    m.add_argument("--n-docs", type=int, default=500)
    m.add_argument("--max-tokens", type=int, default=800_000)
    m.add_argument("--rows-per-shard", type=int, default=500_000, help="rows ranged-read per shard")
    m.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    m.add_argument("--batch", type=int, default=4, help="windows per gemma forward (×2048 tokens)")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", default=str(OUT_DIR / "loudness.json"))
    m.set_defaults(fn=cmd_measure)

    u = sub.add_parser("upload", help="push loudness.json to store repo roots (gated)")
    u.add_argument("--file", default=str(OUT_DIR / "loudness.json"))
    u.add_argument("--repos", default=",".join(CLIMBMIX_REPOS),
                   help="comma-separated HF dataset repo ids")
    u.set_defaults(fn=cmd_upload)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
