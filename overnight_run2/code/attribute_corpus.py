#!/usr/bin/env python3
"""Phase C — corpus attribution: sweep the 12-layer concept probes over the
ClimbMix training shards (0-185 + val 6542), per token, and store SPARSE firings
+ gated-tail statistics, uploading to HF per shard.

Authoritative specs:
  knowledge/probe_rectification_handoff.md  §3 "Phase C" (THE spec)
  knowledge/probe_training_spec.md          §6 (downstream sweep)
  overnight_run2/code/train_probes.py       (probe-weight artifact format consumed here)

WHAT THIS DOES  (per shard, fully resumable, disk-bounded to <=1-2 shards)
  1. Load google/gemma-2-9b FROZEN (inference_mode, eager attn, on `device`) and the 12
     per-layer probe artifacts (weights/layer_<L>.npz + summary.json) downloaded from the
     HF weights repo (kaushikreddyxyz/concept-probes-v2-weights).
  2. For each shard:
       a. resume: skip if <shard>.firings.parquet already exists on the HF attribution repo.
       b. download shard_XXXXX.parquet from karpathy/climbmix-400b-shuffle (HF dataset).
       c. iterate text rows, tokenize with the gemma fast tokenizer (truncate to max_seq),
          batch docs by a token budget, ONE forward per batch, extract the 12 layers'
          hidden_states.
       d. per token, per layer: standardize h_t with that layer's stored mean/std, compute
          s = sigmoid(w_c . h_std + b_c) for all 57 rows (one matmul).
       e. SPARSE store: emit a firing row only where s > the per-(row,layer) threshold
          (binary: the val-calibrated Youden threshold stored per layer in summary.json;
           scalar: a configurable constant since no calibrated threshold exists — see
           ThresholdPolicy). Columns: (shard, doc_id, token_pos, concept_id, layer, score).
          Per-token fields (token_id, ||h|| at each layer) are stored ONCE in a side table.
       f. GATED-TAIL stats: for every gated-out (s <= threshold) cell, accumulate
          mean+std+count per (shard, concept_id, layer) with a streaming Welford accumulator
          (never keeps the values). Tiny.
       g. write <shard>.firings.parquet + <shard>.tokens.parquet + <shard>.tailstats.parquet,
          upload all to the HF attribution repo, write a per-shard manifest fragment, then
          DELETE the local raw shard + outputs.

GPU hygiene (handoff §5): inference_mode, GPU-resident apply, device passed,
OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=8 set at import.

SELF-TEST (no GPU, no gemma, no real probes):  python attribute_corpus.py --selftest
  Injects SYNTHETIC per-token activations + a SYNTHETIC probe artifact (known W,b,mean,std +
  thresholds) and runs the apply->threshold->sparse-write->tailstats->parquet path on 3 fake
  shards, asserting: (a) only above-threshold cells land in firings; (b) gated-tail mean/std
  match a direct numpy computation to <1e-6 (Welford); (c) parquet round-trips with the right
  schema/dtypes; (d) the per-token side table dedups (token/||h|| stored once, not per row);
  (e) resume skips an already-uploaded shard; plus a torch-vs-numpy apply cross-check so the
  GPU apply formula is validated on CPU.
"""
import os
import sys
import json
import glob
import time
import argparse
import logging

# Thread caps MUST be set before numpy / torch import (GPU-hygiene, handoff §5).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------
# Constants (from state.json / specs)
# ----------------------------------------------------------------------------
MODEL_NAME = "google/gemma-2-9b"
SELECTED_LAYERS = [1, 2, 6, 10, 15, 19, 24, 28, 33, 37, 41, 42]  # index into hidden_states[0..42]
D_MODEL = 3584
N_ROWS = 57
CORPUS_REPO = "karpathy/climbmix-400b-shuffle"
WEIGHTS_REPO = "kaushikreddyxyz/concept-probes-v2-weights"
ATTR_REPO = "kaushikreddyxyz/concept-probes-v2-attribution"

# HF repo layout (attribution dataset repo)
DIR_FIRINGS = "firings"
DIR_TOKENS = "tokens"
DIR_TAIL = "tailstats"
DIR_MANIFEST = "manifests"

log = logging.getLogger("phaseC")


# ============================================================================
# Stable sigmoid (numpy)
# ============================================================================
def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def logit_np(p, eps=1e-7):
    """Inverse sigmoid: map a probability threshold in (0,1) to a LOGIT threshold.
    Gating runs in logit space (sigmoid saturates), so the absolute sigmoid thresholds from
    summary.json are converted here once at load."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# ============================================================================
# THRESHOLD POLICY
# ----------------------------------------------------------------------------
# Resolve a per-(row, layer) decision threshold on the sigmoid SCORE s.
#
#   binary rows : train_probes calibrates a Youden-J threshold ON THE VAL SPLIT for
#                 EVERY layer (not just best_layer) and stores it at
#                 summary["rows"][rid]["by_layer"][str(L)]["test"]["threshold"].
#                 => use that per-(row,layer) val-calibrated threshold directly.
#                 fallback chain if a particular (row,layer) is missing it:
#                   (1) the row's best_layer threshold,
#                   (2) `default_binary` constant.
#   scalar rows : the probe emits sigmoid(logit) ~ magnitude in [0,1], trained with
#                 soft-target BCE; NO calibrated threshold exists in the artifacts.
#                 => fire when magnitude > `scalar` constant (default 0.5 = "concept at
#                    least half-present"). Recorded in the manifest; flagged as a decision.
# ============================================================================
class ThresholdPolicy:
    def __init__(self, scalar=0.5, default_binary=0.5, binary_floor=None):
        self.scalar = float(scalar)
        self.default_binary = float(default_binary)
        # SELECTIVITY FLOOR for binary firings. The Youden-J val thresholds in
        # summary.json sit near the classification operating point (sigmoid~0.5),
        # which fires on a huge fraction of raw corpus tokens (~42% of cells) -
        # the OPPOSITE of the spec's "only nontrivial firings stored" intent.
        # Enforce a minimum sigmoid score so the sparse store stays sparse.
        # 0.0 => disabled (original behaviour). Set via BINARY_FIRING_FLOOR env.
        if binary_floor is None:
            bf = os.environ.get("BINARY_FIRING_FLOOR", "").strip()
            binary_floor = float(bf) if bf else 0.0
        self.binary_floor = float(binary_floor)

    def resolve(self, summary, row_ids, kinds, layers):
        """Return thr[L] -> float array[n_rows] and a provenance dict (per row,layer)."""
        rows = summary.get("rows", {}) if summary else {}
        thr = {L: np.empty(len(row_ids), dtype=np.float64) for L in layers}
        prov = {}
        for ci, rid in enumerate(row_ids):
            kind = kinds[ci]
            ent = rows.get(rid, {})
            by_layer = ent.get("by_layer", {})
            best_layer = ent.get("best_layer", None)
            # best-layer threshold (binary) used as a fallback
            best_thr = None
            if best_layer is not None:
                bl = by_layer.get(str(best_layer), {})
                t = bl.get("test")
                if t is not None and t.get("threshold") is not None:
                    best_thr = float(t["threshold"])
            for L in layers:
                if kind == "scalar":
                    # absolute thresholds are sigmoid-space -> store in logit space (gate runs on logits)
                    thr[L][ci] = float(logit_np(self.scalar))
                    prov[(rid, L)] = "scalar_const_logit"
                    continue
                src = "per_layer_val"
                v = None
                bl = by_layer.get(str(L), {})
                t = bl.get("test")
                if t is not None and t.get("threshold") is not None:
                    v = float(t["threshold"])
                if v is None and best_thr is not None:
                    v, src = best_thr, "best_layer_val"
                if v is None:
                    v, src = self.default_binary, "default_const"
                if self.binary_floor > 0.0 and v < self.binary_floor:
                    v, src = self.binary_floor, src + "+floor%.3f" % self.binary_floor
                thr[L][ci] = float(logit_np(v))      # sigmoid threshold -> logit space
                prov[(rid, L)] = src + "_logit"
        return thr, prov


# ============================================================================
# PROBE BUNDLE  (consumes the train_probes .npz + summary.json artifacts)
# ============================================================================
class ProbeBundle:
    """Loaded per-layer probe weights + resolved per-(row,layer) thresholds."""

    def __init__(self, layers, row_ids, kinds, W, b, mean, std, thr, thr_prov):
        self.layers = list(layers)            # sorted layer ids actually present
        self.row_ids = list(row_ids)          # canonical row id strings (len n_rows)
        self.kinds = list(kinds)              # "binary"|"scalar" per row
        self.n_rows = len(row_ids)
        self.W = W                            # dict L -> [n_rows, d] float32
        self.b = b                            # dict L -> [n_rows] float32
        self.mean = mean                      # dict L -> [d] float32
        self.std = std                        # dict L -> [d] float32
        self.thr = thr                        # dict L -> [n_rows] float64
        self.thr_prov = thr_prov              # (rid,L) -> provenance str

    @staticmethod
    def _kind_of(rid):
        return "scalar" if str(rid).startswith("scalar::") else "binary"

    @classmethod
    def load(cls, weights_dir, policy=None, summary_path=None):
        policy = policy or ThresholdPolicy()
        # locate the per-layer npz (weights/ subdir or flat)
        cands = sorted(glob.glob(os.path.join(weights_dir, "weights", "layer_*.npz")))
        if not cands:
            cands = sorted(glob.glob(os.path.join(weights_dir, "layer_*.npz")))
        if not cands:
            raise FileNotFoundError(f"no layer_*.npz under {weights_dir} (or its weights/ subdir)")
        if summary_path is None:
            for c in (os.path.join(weights_dir, "summary.json"),
                      os.path.join(os.path.dirname(cands[0]), "..", "summary.json")):
                if os.path.exists(c):
                    summary_path = c
                    break
        summary = json.load(open(summary_path)) if summary_path and os.path.exists(summary_path) else None

        W, b, mean, std = {}, {}, {}, {}
        row_ids = None
        layers = []
        for path in cands:
            z = np.load(path, allow_pickle=True)
            L = int(z["selected_layer"])
            layers.append(L)
            W[L] = np.asarray(z["W"], dtype=np.float32)
            b[L] = np.asarray(z["b"], dtype=np.float32)
            mean[L] = np.asarray(z["mean"], dtype=np.float32)
            std[L] = np.asarray(z["std"], dtype=np.float32)
            rid = [str(x) for x in z["row_ids"].tolist()]
            if row_ids is None:
                row_ids = rid
            elif row_ids != rid:
                raise ValueError(f"row_ids mismatch across layers ({path})")
        layers = sorted(layers)
        kinds = [cls._kind_of(r) for r in row_ids]
        thr, thr_prov = policy.resolve(summary, row_ids, kinds, layers)
        log.info("loaded probe bundle: %d layers %s, %d rows", len(layers), layers, len(row_ids))
        return cls(layers, row_ids, kinds, W, b, mean, std, thr, thr_prov)


# ============================================================================
# SCORE COMPUTATION  (numpy reference; torch mirror for the GPU path)
# ----------------------------------------------------------------------------
# Both compute, per layer L:  z = ((h - mean_L)/std_L) @ W_L^T + b_L   (the probe LOGIT)
# and the raw residual norm ||h||_2. We return the LOGIT, not sigmoid(z): the corpus probes
# saturate (sigmoid -> exactly 1.0 in float for large z), which DESTROYS the ranking in the
# tail and makes a high percentile gate land at 1.0 (firing nothing). The logit keeps the full
# ranking, so gating/calibration/tail-stats all run in logit space; downstream gets the logit
# as the firing "score". The self-test asserts numpy/torch agree, validating the GPU formula.
# ============================================================================
def scores_numpy(H_by_layer, bundle):
    """H_by_layer: dict L -> [T, d] (raw residuals). Returns
       Z: dict L -> [T, n_rows] probe LOGITS (float64),
       norms: [T, n_layers] float (raw ||h|| per layer, column order = bundle.layers)."""
    layers = bundle.layers
    T = H_by_layer[layers[0]].shape[0]
    S = {}
    norms = np.empty((T, len(layers)), dtype=np.float64)
    for j, L in enumerate(layers):
        H = np.asarray(H_by_layer[L], dtype=np.float64)
        norms[:, j] = np.sqrt((H * H).sum(1))
        Hstd = (H - bundle.mean[L].astype(np.float64)) / bundle.std[L].astype(np.float64)
        logits = Hstd @ bundle.W[L].astype(np.float64).T + bundle.b[L].astype(np.float64)
        S[L] = logits
    return S, norms


def scores_torch(H_by_layer, bundle, device="cpu"):
    """torch mirror of scores_numpy (used on the GPU during the real sweep).
    H_by_layer: dict L -> torch tensor [T, d] on `device`. Returns numpy LOGITS Z / norms."""
    import torch
    layers = bundle.layers
    T = H_by_layer[layers[0]].shape[0]
    norms = torch.empty((T, len(layers)), dtype=torch.float32, device=device)
    S = {}
    for j, L in enumerate(layers):
        H = H_by_layer[L].to(torch.float32)
        norms[:, j] = torch.linalg.vector_norm(H, dim=1)
        mean = torch.as_tensor(bundle.mean[L], device=device, dtype=torch.float32)
        std = torch.as_tensor(bundle.std[L], device=device, dtype=torch.float32)
        Wt = torch.as_tensor(bundle.W[L], device=device, dtype=torch.float32)
        bt = torch.as_tensor(bundle.b[L], device=device, dtype=torch.float32)
        Hstd = (H - mean) / std
        logits = Hstd @ Wt.t() + bt
        S[L] = logits.to(torch.float64).cpu().numpy()
    return S, norms.to(torch.float64).cpu().numpy()


# ============================================================================
# WELFORD (streaming mean/std of gated-out scores, per (row, layer))
# ----------------------------------------------------------------------------
# Chan's parallel update merges per-batch stats into the running accumulator. Vectorized
# over the n_rows for a single layer. Matches numpy's population std (ddof=0).
# ============================================================================
class GatedTail:
    def __init__(self, layers, n_rows):
        self.layers = list(layers)
        self.n_rows = n_rows
        self.count = {L: np.zeros(n_rows, dtype=np.float64) for L in layers}
        self.mean = {L: np.zeros(n_rows, dtype=np.float64) for L in layers}
        self.M2 = {L: np.zeros(n_rows, dtype=np.float64) for L in layers}
        self.fired = {L: np.zeros(n_rows, dtype=np.int64) for L in layers}
        self.total = {L: np.zeros(n_rows, dtype=np.int64) for L in layers}

    def update(self, L, S, gated_mask):
        """S: [T, n_rows] scores; gated_mask: [T, n_rows] bool (True = gated-out, include)."""
        S = np.asarray(S, dtype=np.float64)
        g = np.asarray(gated_mask)
        T = S.shape[0]
        self.total[L] += T
        self.fired[L] += (~g).sum(0).astype(np.int64)
        nb = g.sum(0).astype(np.float64)                       # [n_rows] gated count this batch
        sum_b = np.where(g, S, 0.0).sum(0)                     # [n_rows]
        safe_nb = np.where(nb > 0, nb, 1.0)
        mean_b = sum_b / safe_nb
        diff = np.where(g, S - mean_b[None, :], 0.0)           # (x-mean_b) on gated, 0 elsewhere
        M2_b = (diff * diff).sum(0)                            # [n_rows] stable batch M2
        # Chan merge (per row)
        old_n = self.count[L]
        new_n = old_n + nb
        delta = mean_b - self.mean[L]
        safe_new = np.where(new_n > 0, new_n, 1.0)
        self.mean[L] = np.where(new_n > 0, self.mean[L] + delta * (nb / safe_new), self.mean[L])
        self.M2[L] = self.M2[L] + M2_b + (delta * delta) * (old_n * nb / safe_new)
        self.count[L] = new_n

    def finalize(self, bundle, shard):
        """Return list of dict rows for the tailstats table."""
        rows = []
        for L in self.layers:
            cnt = self.count[L]
            var = np.where(cnt > 0, self.M2[L] / np.where(cnt > 0, cnt, 1.0), 0.0)
            std = np.sqrt(np.maximum(var, 0.0))
            for ci in range(self.n_rows):
                rows.append({
                    "shard": np.int32(shard),
                    "concept_id": np.int16(ci),
                    "layer": np.int8(L),
                    "kind": bundle.kinds[ci],
                    "threshold": np.float32(bundle.thr[L][ci]),
                    "total_count": np.int64(self.total[L][ci]),
                    "fired_count": np.int64(self.fired[L][ci]),
                    "gated_count": np.int64(int(cnt[ci])),
                    "gated_mean": np.float64(self.mean[L][ci] if cnt[ci] > 0 else 0.0),
                    "gated_std": np.float64(std[ci] if cnt[ci] > 0 else 0.0),
                })
        return rows


# ============================================================================
# PARQUET SCHEMAS + WRITERS
# ============================================================================
FIRINGS_SCHEMA = pa.schema([
    ("shard", pa.int32()),
    ("doc_id", pa.int32()),
    ("token_pos", pa.int32()),
    ("concept_id", pa.int16()),
    ("layer", pa.int8()),
    ("score", pa.float16()),
])
TAIL_SCHEMA = pa.schema([
    ("shard", pa.int32()),
    ("concept_id", pa.int16()),
    ("layer", pa.int8()),
    ("kind", pa.string()),
    ("threshold", pa.float32()),
    ("total_count", pa.int64()),
    ("fired_count", pa.int64()),
    ("gated_count", pa.int64()),
    ("gated_mean", pa.float64()),
    ("gated_std", pa.float64()),
])


def tokens_schema(layers):
    fields = [("doc_id", pa.int32()), ("token_pos", pa.int32()), ("token_id", pa.int32())]
    fields += [(f"h_norm_L{L}", pa.float16()) for L in layers]
    return pa.schema(fields)


def _firings_writer(path):
    return pq.ParquetWriter(
        path, FIRINGS_SCHEMA, compression="zstd", compression_level=5,
        use_dictionary=["shard", "concept_id", "layer"],
        sorting_columns=[
            pq.SortingColumn(FIRINGS_SCHEMA.get_field_index("doc_id")),
            pq.SortingColumn(FIRINGS_SCHEMA.get_field_index("token_pos")),
            pq.SortingColumn(FIRINGS_SCHEMA.get_field_index("layer")),
            pq.SortingColumn(FIRINGS_SCHEMA.get_field_index("concept_id")),
        ],
    )


def _tokens_writer(path, schema):
    return pq.ParquetWriter(path, schema, compression="zstd", compression_level=5)


# ============================================================================
# CORE: gate a batch's scores -> firing rows + fired-token rows + tail update
# ----------------------------------------------------------------------------
# This is the heart of the pipeline and the path the self-test validates.
# ============================================================================
def gate_batch(S, norms, bundle, tail, shard, doc_ids, token_pos, token_ids):
    """S: dict L -> [T, n_rows] scores. norms: [T, n_layers]. doc_ids/token_pos/token_ids: [T].
    Returns (firings_arrow_batch | None, tokens_arrow_batch | None). Updates `tail` in place."""
    layers = bundle.layers
    T = len(doc_ids)
    fired_any = np.zeros(T, dtype=bool)

    f_doc, f_pos, f_cid, f_layer, f_score = [], [], [], [], []
    for L in layers:
        s = np.asarray(S[L], dtype=np.float64)            # [T, n_rows]
        thr = bundle.thr[L][None, :]                      # [1, n_rows]
        fired = s > thr                                    # strict >
        gated = ~fired
        tail.update(L, s, gated)
        if fired.any():
            ti, ci = np.nonzero(fired)                     # token-idx, concept-idx
            f_doc.append(doc_ids[ti])
            f_pos.append(token_pos[ti])
            f_cid.append(ci.astype(np.int16))
            f_layer.append(np.full(ti.shape[0], L, dtype=np.int8))
            f_score.append(s[ti, ci].astype(np.float16))
            fired_any[ti] = True

    firings_batch = None
    if f_doc:
        doc = np.concatenate(f_doc).astype(np.int32)
        pos = np.concatenate(f_pos).astype(np.int32)
        cid = np.concatenate(f_cid)
        lay = np.concatenate(f_layer)
        sco = np.concatenate(f_score)
        # sort to match declared sorting_columns (doc, pos, layer, concept) -> better compression
        order = np.lexsort((cid, lay, pos, doc))
        firings_batch = pa.record_batch([
            pa.array(np.full(doc.shape[0], shard, dtype=np.int32)),
            pa.array(doc[order]), pa.array(pos[order]),
            pa.array(cid[order]), pa.array(lay[order]),
            pa.array(sco[order]),
        ], schema=FIRINGS_SCHEMA)

    # per-token side table: ONE row per fired token (dedup of token_id + ||h||-by-layer)
    tokens_batch = None
    if fired_any.any():
        sel = np.nonzero(fired_any)[0]
        arrs = [pa.array(doc_ids[sel].astype(np.int32)),
                pa.array(token_pos[sel].astype(np.int32)),
                pa.array(token_ids[sel].astype(np.int32))]
        for j in range(len(layers)):
            arrs.append(pa.array(norms[sel, j].astype(np.float16)))
        tokens_batch = pa.record_batch(arrs, schema=tokens_schema(layers))
    return firings_batch, tokens_batch


# ============================================================================
# SHARD WRITER  (streams batches -> parquet, bounds RAM to one batch)
# ============================================================================
class ShardWriter:
    def __init__(self, out_dir, shard, bundle):
        self.shard = shard
        self.bundle = bundle
        self.tok_schema = tokens_schema(bundle.layers)
        self.firings_path = os.path.join(out_dir, f"shard_{shard:05d}.firings.parquet")
        self.tokens_path = os.path.join(out_dir, f"shard_{shard:05d}.tokens.parquet")
        self.tail_path = os.path.join(out_dir, f"shard_{shard:05d}.tailstats.parquet")
        self.fw = _firings_writer(self.firings_path)
        self.tw = _tokens_writer(self.tokens_path, self.tok_schema)
        self.tail = GatedTail(bundle.layers, bundle.n_rows)
        self.n_firings = 0
        self.n_tokens_fired = 0
        self.n_tokens_total = 0
        self.n_docs = 0

    def add_batch(self, S, norms, doc_ids, token_pos, token_ids):
        self.n_tokens_total += len(doc_ids)
        fb, tb = gate_batch(S, norms, self.bundle, self.tail, self.shard,
                            doc_ids, token_pos, token_ids)
        if fb is not None:
            self.fw.write_batch(fb)
            self.n_firings += fb.num_rows
        if tb is not None:
            self.tw.write_batch(tb)
            self.n_tokens_fired += tb.num_rows

    def close(self):
        self.fw.close()
        self.tw.close()
        tail_rows = self.tail.finalize(self.bundle, self.shard)
        tail_tbl = pa.Table.from_pylist(tail_rows, schema=TAIL_SCHEMA)
        pq.write_table(tail_tbl, self.tail_path, compression="zstd")
        return {
            "firings": self.firings_path, "tokens": self.tokens_path, "tailstats": self.tail_path,
            "n_firings": int(self.n_firings), "n_tokens_fired": int(self.n_tokens_fired),
            "n_tokens_total": int(self.n_tokens_total), "n_docs": int(self.n_docs),
        }


# ============================================================================
# GEMMA EXTRACTION + shard processing  (real path; not exercised by selftest)
# ============================================================================
def _detect_text_column(table):
    for cand in ("text", "content", "raw_content", "document"):
        if cand in table.column_names:
            return cand
    # first string-typed column
    for name in table.column_names:
        if pa.types.is_string(table.schema.field(name).type) or \
           pa.types.is_large_string(table.schema.field(name).type):
            return name
    raise ValueError(f"no text column found among {table.column_names}")


def _resolve_shard_filename(api, shard, template):
    """Find the actual parquet filename for `shard` in the corpus repo.
    Tries the template first; else lists the repo and matches by zero-padded number."""
    fn = template.format(shard)
    try:
        if api.file_exists(repo_id=CORPUS_REPO, filename=fn, repo_type="dataset"):
            return fn
    except Exception:
        pass
    files = api.list_repo_files(repo_id=CORPUS_REPO, repo_type="dataset")
    pqs = [f for f in files if f.endswith(".parquet")]
    for pad in (5, 4, 6, 0):
        key = (f"{shard:0{pad}d}" if pad else str(shard))
        for f in pqs:
            base = os.path.basename(f)
            if "shard" in base and key in base:
                return f
    raise FileNotFoundError(f"could not resolve a parquet filename for shard {shard} in {CORPUS_REPO}")


def hf_shard_done(api, shard, repo=ATTR_REPO):
    fn = f"{DIR_FIRINGS}/shard_{shard:05d}.firings.parquet"
    try:
        return api.file_exists(repo_id=repo, filename=fn, repo_type="dataset")
    except Exception:
        try:
            return fn in set(api.list_repo_files(repo_id=repo, repo_type="dataset"))
        except Exception:
            return False


def _build_plan(texts, tokenizer, max_seq):
    """Tokenize docs -> ordered list of (doc_id, input_ids), skipping empties. Deterministic."""
    plan = []
    for doc_id, txt in enumerate(texts):
        if not txt:
            continue
        ids = tokenizer(txt, add_special_tokens=True, truncation=True,
                        max_length=max_seq)["input_ids"]
        if len(ids) == 0:
            continue
        plan.append((doc_id, ids))
    return plan


def _iter_score_batches(plan, model, bundle, device, batch_tokens=16384, max_batch_docs=64):
    """Yield (S, norms, doc_ids, token_pos, token_ids) per forward batch. ONE forward each:
    pad-batch by a token budget, run gemma (AutoModel -> no lm_head/logits, saves ~33GB at big
    batches), gather the 12 layers' UNPADDED token activations on-GPU, apply the 57 probe rows.
    Shared by attribution AND relative-gate calibration so both see identical activations."""
    import torch
    layers = bundle.layers
    nseq = len(plan)
    i = 0
    with torch.inference_mode():
        while i < nseq:
            batch = [plan[i]]
            tok = len(plan[i][1])
            i += 1
            while i < nseq and len(batch) < max_batch_docs and tok + len(plan[i][1]) <= batch_tokens:
                batch.append(plan[i])
                tok += len(plan[i][1])
                i += 1
            maxlen = max(len(ids) for _, ids in batch)
            B = len(batch)
            input_ids = np.zeros((B, maxlen), dtype=np.int64)
            attn = np.zeros((B, maxlen), dtype=np.int64)
            for bi, (_, ids) in enumerate(batch):
                input_ids[bi, :len(ids)] = ids
                attn[bi, :len(ids)] = 1
            ii = torch.tensor(input_ids, device=device)
            am = torch.tensor(attn, device=device)
            out = model(input_ids=ii, attention_mask=am, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states  # tuple len 43 (embeddings + 42 layers); AutoModel == CausalLM here

            # gather the unpadded tokens of this batch into flat [T, d] per layer (on GPU)
            doc_ids_l, pos_l, tokid_l, sel_b, sel_t = [], [], [], [], []
            for bi, (doc_id, ids) in enumerate(batch):
                Lb = len(ids)
                doc_ids_l.append(np.full(Lb, doc_id, dtype=np.int64))
                pos_l.append(np.arange(Lb, dtype=np.int64))
                tokid_l.append(np.asarray(ids, dtype=np.int64))
                sel_b.append(np.full(Lb, bi, dtype=np.int64))
                sel_t.append(np.arange(Lb, dtype=np.int64))
            doc_ids = np.concatenate(doc_ids_l)
            token_pos = np.concatenate(pos_l)
            token_ids = np.concatenate(tokid_l)
            gb = torch.tensor(np.concatenate(sel_b), device=device)
            gt = torch.tensor(np.concatenate(sel_t), device=device)
            H_by_layer = {L: hs[L][gb, gt, :] for L in layers}     # [T, d] on GPU
            S, norms = scores_torch(H_by_layer, bundle, device=device)
            yield S, norms, doc_ids, token_pos, token_ids


def calibrate_relative_gate(calib_plan, model, bundle, device, top_frac=0.007,
                            batch_tokens=16384, max_batch_docs=64, max_tokens=1_000_000):
    """RELATIVE GATE calibration (correctness fix). The probes RANK well (AUROC 0.93-0.98) but
    their absolute sigmoid scale is miscalibrated on corpus (standardization train/serve gap) ->
    ~40% of (token,row,layer) cells exceed ANY fixed sigmoid threshold. Fix: per-(row,layer)
    tau = the (1-top_frac) score quantile measured on a CORPUS sample, so only the top `top_frac`
    of corpus tokens fire per (row,layer) -> sparse, and uses RANKING not absolute scale.
    Deterministic (ordered sample, fixed shard) => identical tau on every fleet pod.
    Returns (tau: dict L->[n_rows] float64, n_calib_tokens)."""
    layers = bundle.layers
    acc = {L: [] for L in layers}
    n_tok = 0
    t0 = time.time()
    for S, _norms, doc_ids, _p, _t in _iter_score_batches(
            calib_plan, model, bundle, device,
            batch_tokens=batch_tokens, max_batch_docs=max_batch_docs):
        for L in layers:
            acc[L].append(np.asarray(S[L], dtype=np.float32))
        n_tok += int(doc_ids.shape[0])
        if n_tok >= max_tokens:
            break
    q = 1.0 - float(top_frac)
    tau = {}
    for L in layers:
        A = np.concatenate(acc[L], axis=0)            # [N, n_rows]
        tau[L] = np.quantile(A, q, axis=0).astype(np.float64)
    log.info("relative gate calibrated on %d tokens in %.1fs: top_frac=%.4f quantile=%.5f",
             n_tok, time.time() - t0, top_frac, q)
    return tau, n_tok


def extract_and_attribute(shard, parquet_path, bundle, model, tokenizer, device,
                          writer, max_seq=1024, batch_tokens=16384, max_batch_docs=64):
    """Stream the shard's docs through gemma, extract the 12 layers, apply probes, write."""
    table = pq.read_table(parquet_path)
    col = _detect_text_column(table)
    texts = table.column(col).to_pylist()
    plan = _build_plan(texts, tokenizer, max_seq)
    writer.n_docs = len(plan)
    log.info("shard %d: %d docs -> forward (max_seq=%d max_batch_docs=%d batch_tokens=%d)",
             shard, len(plan), max_seq, max_batch_docs, batch_tokens)
    t0 = time.time()
    done_tok = 0
    for S, norms, doc_ids, token_pos, token_ids in _iter_score_batches(
            plan, model, bundle, device, batch_tokens=batch_tokens, max_batch_docs=max_batch_docs):
        writer.add_batch(S, norms, doc_ids, token_pos, token_ids)
        done_tok += int(doc_ids.shape[0])
        if (time.time() - t0) > 30 and done_tok > 0:
            log.info("shard %d: %d tokens, %.0f tok/s, %d firings",
                     shard, done_tok, done_tok / (time.time() - t0), writer.n_firings)
    log.info("shard %d: forward done in %.1fs (%d tokens, %d firings)",
             shard, time.time() - t0, done_tok, writer.n_firings)


def upload_shard_outputs(api, shard, paths, repo=ATTR_REPO, manifest_frag=None):
    """Upload firings/tokens/tailstats (+ a per-shard manifest fragment) to the HF repo."""
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, private=False)
    pairs = [
        (paths["firings"], f"{DIR_FIRINGS}/shard_{shard:05d}.firings.parquet"),
        (paths["tokens"], f"{DIR_TOKENS}/shard_{shard:05d}.tokens.parquet"),
        (paths["tailstats"], f"{DIR_TAIL}/shard_{shard:05d}.tailstats.parquet"),
    ]
    ops = []
    from huggingface_hub import CommitOperationAdd
    for local, remote in pairs:
        ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local))
    if manifest_frag is not None:
        frag_bytes = json.dumps(manifest_frag, indent=2).encode()
        ops.append(CommitOperationAdd(
            path_in_repo=f"{DIR_MANIFEST}/shard_{shard:05d}.json", path_or_fileobj=frag_bytes))
    # Retry on commit-race conflicts (409/412) + rate limits (429) with exp backoff.
    # Multiple pods write disjoint files to this same dataset repo, so the git revision
    # underneath create_commit can race even though our filenames never collide.
    delays = [5, 15, 45, 90, 180]
    for attempt in range(len(delays) + 1):
        try:
            api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                              commit_message=f"Phase C: shard {shard} firings+tokens+tailstats")
            return
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            msg = str(e).lower()
            retryable = (status in (409, 412, 429, 500, 502, 503, 504)
                         or "conflict" in msg or "412" in msg or "429" in msg
                         or "too many requests" in msg or "rate limit" in msg)
            if not retryable or attempt >= len(delays):
                raise
            d = delays[attempt]
            log.warning("shard %d upload conflict (%s); retry %d/%d in %ds",
                        shard, status or msg[:60], attempt + 1, len(delays), d)
            time.sleep(d)


def build_global_manifest(bundle, policy, shard_list, max_seq):
    """The top-level manifest.json: config, the 12 layers, per-(row,layer) thresholds,
    schema, and the planned shard list. Written once (avoids the multi-pod race on a shared
    file); per-shard fragments record token counts; 'shards done' = list the repo."""
    thresholds = {rid: {str(L): float(bundle.thr[L][ci]) for L in bundle.layers}
                  for ci, rid in enumerate(bundle.row_ids)}
    thr_prov = {f"{rid}@{L}": bundle.thr_prov[(rid, L)] for ci, rid in enumerate(bundle.row_ids)
                for L in bundle.layers}
    return {
        "model": MODEL_NAME,
        "corpus": CORPUS_REPO,
        "selected_layers": bundle.layers,
        "n_layers": len(bundle.layers),
        "n_rows": bundle.n_rows,
        "row_ids": bundle.row_ids,
        "row_kinds": bundle.kinds,
        "shard_list": shard_list,
        "n_shards": len(shard_list),
        "max_seq": max_seq,
        "gate": getattr(bundle, "gate_meta", {"mode": "absolute"}),
        "threshold_policy": {
            "active": getattr(bundle, "gate_meta", {}).get("mode", "absolute"),
            "relative": "per-(row,layer) tau = the (1-top_frac) score quantile on a fixed corpus "
                        "sample (the per-(row,layer) tau is stored in 'thresholds'); fire iff "
                        "score>tau. Ranking-based, sidesteps the absolute-scale miscalibration. "
                        "Deterministic on a fixed shard -> identical across fleet pods. DEFAULT.",
            "binary_absolute": "per-(row,layer) val Youden-J from summary.json; fallback best_layer, "
                               "then default constant (used only when gate.mode=='absolute').",
            "binary_default": policy.default_binary,
            "scalar_absolute": "constant sigmoid magnitude (used only when gate.mode=='absolute')",
            "scalar_const": policy.scalar,
        },
        "thresholds": thresholds,
        "threshold_provenance": thr_prov,
        "schema": {
            "firings": "(shard:int32, doc_id:int32, token_pos:int32, concept_id:int16, "
                       "layer:int8, score:float16)  -- one row per (token,row,layer) with "
                       "LOGIT>threshold. score = probe LOGIT (NOT sigmoid: sigmoid saturates at "
                       "1.0, losing tail ranking). concept_id indexes row_ids[0..n_rows). zstd, "
                       "dictionary(shard,concept_id,layer).",
            "tokens": "(doc_id:int32, token_pos:int32, token_id:int32, h_norm_L<L>:float16 x12)"
                      "  -- ONE row per fired token (dedup of token_id + raw ||h|| at each layer).",
            "tailstats": "(shard, concept_id, layer, kind, threshold, total_count, fired_count, "
                         "gated_count, gated_mean, gated_std)  -- Welford mean/std (population, "
                         "ddof=0) of the gated-out (logit<=threshold) probe LOGITS per (concept,layer); "
                         "threshold is a logit.",
        },
    }


def process_shard(shard, bundle, model, tokenizer, device, api, work_dir, policy,
                  max_seq=1024, batch_tokens=16384, max_batch_docs=64,
                  shard_filename_template="shard_{:05d}.parquet", keep_local=False,
                  skip_upload=False):
    """Full per-shard pipeline: resume-check -> download -> forward -> write -> upload -> purge."""
    if not skip_upload and hf_shard_done(api, shard):
        log.info("shard %d already on HF -> skip", shard)
        return {"shard": shard, "status": "skipped"}

    from huggingface_hub import hf_hub_download
    os.makedirs(work_dir, exist_ok=True)
    fn = _resolve_shard_filename(api, shard, shard_filename_template)
    log.info("shard %d: downloading %s ...", shard, fn)
    local_pq = hf_hub_download(repo_id=CORPUS_REPO, filename=fn, repo_type="dataset",
                               local_dir=os.path.join(work_dir, "corpus"))

    writer = ShardWriter(work_dir, shard, bundle)
    try:
        extract_and_attribute(shard, local_pq, bundle, model, tokenizer, device, writer,
                              max_seq=max_seq, batch_tokens=batch_tokens,
                              max_batch_docs=max_batch_docs)
    finally:
        info = writer.close()
    info["shard"] = shard
    frag = {"shard": shard, "corpus_file": fn, "n_docs": info["n_docs"],
            "n_tokens_total": info["n_tokens_total"], "n_tokens_fired": info["n_tokens_fired"],
            "n_firings": info["n_firings"], "max_seq": max_seq}
    log.info("shard %d: %s", shard, json.dumps(frag))

    if not skip_upload:
        log.info("shard %d: uploading outputs to %s ...", shard, ATTR_REPO)
        upload_shard_outputs(api, shard, info, manifest_frag=frag)

    if not keep_local:
        for p in (info["firings"], info["tokens"], info["tailstats"], local_pq):
            try:
                os.remove(p)
            except OSError:
                pass
    info["status"] = "done"
    return info


# ============================================================================
# SHARD-LIST PARSING  (range spec -> ordered list; supports the 6542 val shard)
# ============================================================================
def parse_shard_spec(spec):
    """'0-185,6542' or '0-92' or '5' or '0-9,6542' -> ordered de-duplicated list of ints."""
    out = []
    seen = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            for v in range(int(a), int(b) + 1):
                if v not in seen:
                    seen.add(v); out.append(v)
        else:
            v = int(part)
            if v not in seen:
                seen.add(v); out.append(v)
    return out


# ============================================================================
# REAL RUN
# ============================================================================
def run(shards_spec, weights_dir, device="cuda", work_dir=None, max_seq=1024,
        batch_tokens=16384, max_batch_docs=64, scalar_threshold=0.5,
        default_binary_threshold=0.5, shard_filename_template="shard_{:05d}.parquet",
        write_global_manifest=False, keep_local=False, skip_upload=False,
        gate_mode="relative", top_frac=0.002, calib_shard=48, calib_tokens=1_000_000,
        attn="eager"):
    # AutoModel (not AutoModelForCausalLM): drops the 256k-vocab lm_head -> skips materializing
    # the [B,seq,256000] logits tensor (~33GB at big batches) -> bigger batches -> ~2-3x faster.
    # Proven byte-identical to CausalLM on all 12 probed layers (helper-1; re-verified here).
    from transformers import AutoModel, AutoTokenizer
    from huggingface_hub import HfApi, hf_hub_download
    import torch

    shard_list = parse_shard_spec(shards_spec)
    work_dir = work_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "phaseC_work")
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    policy = ThresholdPolicy(scalar=scalar_threshold, default_binary=default_binary_threshold)
    bundle = ProbeBundle.load(weights_dir, policy=policy)
    api = HfApi()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    assert tok.is_fast
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    log.info("loading %s (AutoModel, frozen, attn=%s) on %s ...", MODEL_NAME, attn, device)
    model = AutoModel.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, output_hidden_states=True, attn_implementation=attn)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- RELATIVE GATE (correctness): calibrate per-(row,layer) tau on a corpus sample --------
    # Absolute sigmoid thresholds are miscalibrated on corpus (-> ~40% dense firings, breaks the
    # sparse store). Replace them with a per-(row,layer) percentile keeping only the top `top_frac`
    # of corpus tokens. Calibrate ONCE on a fixed shard (deterministic -> fleet-consistent tau).
    bundle.gate_meta = {"mode": gate_mode, "attn": attn, "max_batch_docs": int(max_batch_docs),
                        "batch_tokens": int(batch_tokens)}
    if gate_mode == "relative":
        fn = _resolve_shard_filename(api, calib_shard, shard_filename_template)
        log.info("calibrating relative gate on corpus shard %d (%s), <=%d tokens ...",
                 calib_shard, fn, calib_tokens)
        calib_pq = hf_hub_download(repo_id=CORPUS_REPO, filename=fn, repo_type="dataset",
                                   local_dir=os.path.join(work_dir, "corpus"))
        ctab = pq.read_table(calib_pq)
        calib_plan = _build_plan(ctab.column(_detect_text_column(ctab)).to_pylist(), tok, max_seq)
        tau, n_calib = calibrate_relative_gate(
            calib_plan, model, bundle, device, top_frac=top_frac,
            batch_tokens=batch_tokens, max_batch_docs=max_batch_docs, max_tokens=calib_tokens)
        for L in bundle.layers:
            bundle.thr[L] = tau[L]
            for rid in bundle.row_ids:
                bundle.thr_prov[(rid, L)] = "relative_corpus_q%.5f" % (1.0 - top_frac)
        bundle.gate_meta.update({"top_frac": float(top_frac), "calib_shard": int(calib_shard),
                                 "calib_tokens": int(n_calib), "quantile": 1.0 - float(top_frac),
                                 "score_space": "logit",
                                 "note": "per-(row,layer) corpus percentile of the probe LOGIT "
                                         "(sigmoid saturates); ranking-based, sidesteps the "
                                         "absolute-scale miscalibration. thresholds=logit."})

    if write_global_manifest and not skip_upload:
        man = build_global_manifest(bundle, policy, shard_list, max_seq)
        api.create_repo(repo_id=ATTR_REPO, repo_type="dataset", exist_ok=True, private=False)
        api.upload_file(path_or_fileobj=json.dumps(man, indent=2).encode(),
                        path_in_repo="manifest.json", repo_id=ATTR_REPO, repo_type="dataset",
                        commit_message="Phase C: global manifest")
        log.info("wrote global manifest.json (%d shards, %d layers)",
                 len(shard_list), len(bundle.layers))

    results = []
    for shard in shard_list:
        try:
            r = process_shard(shard, bundle, model, tok, device, api, work_dir, policy,
                              max_seq=max_seq, batch_tokens=batch_tokens,
                              max_batch_docs=max_batch_docs,
                              shard_filename_template=shard_filename_template,
                              keep_local=keep_local, skip_upload=skip_upload)
            results.append(r)
        except Exception as e:
            log.exception("shard %d FAILED: %s", shard, e)
            results.append({"shard": shard, "status": "error", "error": str(e)})
    n_done = sum(1 for r in results if r.get("status") == "done")
    n_skip = sum(1 for r in results if r.get("status") == "skipped")
    log.info("run complete: %d done, %d skipped, %d total", n_done, n_skip, len(shard_list))
    return results


# ============================================================================
# SELF-TEST  (no GPU, no gemma, no real probes)
# ============================================================================
def _make_synth_probe(tmp_dir, layers=(1, 2), n_rows=4, d=16, seed=0):
    """Write a synthetic probe artifact (weights/layer_<L>.npz + summary.json) with KNOWN
    W,b,mean,std and per-(row,layer) thresholds. Rows: 3 binary + 1 scalar."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(tmp_dir, "weights"), exist_ok=True)
    row_ids = ["demo::A", "demo::B", "demo::C", "scalar::mag"][:n_rows]
    summary = {"model": "synthetic", "n_layers": len(layers), "selected_layers": list(layers),
               "n_rows": n_rows, "rows": {}}
    mean = {L: rng.standard_normal(d).astype(np.float32) for L in layers}
    std = {L: (np.abs(rng.standard_normal(d)).astype(np.float32) + 0.5) for L in layers}
    W = {L: (rng.standard_normal((n_rows, d)).astype(np.float32) * 0.3) for L in layers}
    b = {L: (rng.standard_normal(n_rows).astype(np.float32) * 0.1) for L in layers}
    for L in layers:
        np.savez(os.path.join(tmp_dir, "weights", f"layer_{L}.npz"),
                 W=W[L], b=b[L], mean=mean[L], std=std[L],
                 pos_weight=np.ones(n_rows, np.float32), selected_layer=np.int64(L),
                 row_ids=np.array(row_ids))
    # known per-(row,layer) thresholds in the summary (binary rows only)
    thr_table = {}
    for ci, rid in enumerate(row_ids):
        kind = "scalar" if rid.startswith("scalar::") else "binary"
        ent = {"family": "scalar" if kind == "scalar" else "cyclic", "kind": kind,
               "best_layer": layers[0], "by_layer": {}}
        for L in layers:
            if kind == "binary":
                thr = float(0.4 + 0.1 * ci + 0.05 * (L == layers[1]))
                ent["by_layer"][str(L)] = {"val": {"auroc": 0.9},
                                           "test": {"auroc": 0.9, "ap": 0.5, "n_pos": 10,
                                                    "n_neg": 90, "threshold": thr},
                                           "heldout_vocab": {"auroc": 0.8}}
                thr_table[(rid, L)] = thr
            else:
                ent["by_layer"][str(L)] = {"val": {"spearman": 0.5},
                                           "test": {"spearman": 0.5, "r2": 0.2, "bin_auroc": 0.8}}
        summary["rows"][rid] = ent
    json.dump(summary, open(os.path.join(tmp_dir, "summary.json"), "w"), indent=2)
    return row_ids, thr_table


def selftest():
    import tempfile
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    print("=== Phase C self-test (synthetic, CPU, no gemma) ===")
    rng = np.random.default_rng(7)
    d = 16
    layers = [1, 2]
    n_rows = 4
    scalar_thr = 0.5

    tmp = tempfile.mkdtemp(prefix="phaseC_selftest_")
    probe_dir = os.path.join(tmp, "probe")
    os.makedirs(probe_dir, exist_ok=True)
    row_ids, thr_table = _make_synth_probe(probe_dir, layers=tuple(layers), n_rows=n_rows, d=d)
    policy = ThresholdPolicy(scalar=scalar_thr, default_binary=0.5)
    bundle = ProbeBundle.load(probe_dir, policy=policy)

    # threshold resolution: binary rows read the per-(row,layer) val threshold; scalar = const.
    # Thresholds are stored in LOGIT space (gating runs on logits), so compare to logit(sigmoid_thr).
    ok_thr = True
    for ci, rid in enumerate(row_ids):
        for L in layers:
            got = bundle.thr[L][ci]
            if rid.startswith("scalar::"):
                ok_thr &= abs(got - float(logit_np(scalar_thr))) < 1e-6
            else:
                ok_thr &= abs(got - float(logit_np(thr_table[(rid, L)]))) < 1e-6
    print(f"(thr) per-(row,layer) thresholds resolved (logit space) from summary.json: "
          f"{'OK' if ok_thr else 'FAIL'}")

    # ---- (cross-check) torch apply == numpy apply on the same activations -----
    Hxc = {L: rng.standard_normal((23, d)).astype(np.float32) * 2.0 for L in layers}
    Sn, Nn = scores_numpy(Hxc, bundle)
    try:
        import torch
        Hxc_t = {L: torch.as_tensor(Hxc[L]) for L in layers}
        St, Nt = scores_torch(Hxc_t, bundle, device="cpu")
        xc_ok = all(np.allclose(Sn[L], St[L], atol=1e-5) for L in layers) and \
            np.allclose(Nn, Nt, atol=1e-3)
        print(f"(apply) torch GPU-path apply matches numpy reference to <1e-5: "
              f"{'OK' if xc_ok else 'FAIL'}")
    except Exception as e:
        xc_ok = True
        print(f"(apply) torch cross-check skipped: {e}")

    # ---- build 3 fake shards of synthetic activations + run the full path ------
    # We mirror gate_batch with a direct numpy reference and compare firings & tailstats.
    work = os.path.join(tmp, "work")
    os.makedirs(work, exist_ok=True)
    all_checks = []
    for shard in range(3):
        n_docs = 4 + shard
        # per shard, build docs of random length; per layer random H -> deterministic scores
        doc_ids_all, pos_all, tokid_all = [], [], []
        H_layers = {L: [] for L in layers}
        for doc_id in range(n_docs):
            Lb = int(rng.integers(3, 9))
            doc_ids_all.append(np.full(Lb, doc_id, dtype=np.int64))
            pos_all.append(np.arange(Lb, dtype=np.int64))
            tokid_all.append(rng.integers(0, 50000, Lb).astype(np.int64))
            for L in layers:
                H_layers[L].append(rng.standard_normal((Lb, d)).astype(np.float32) * 2.5)
        doc_ids = np.concatenate(doc_ids_all)
        token_pos = np.concatenate(pos_all)
        token_ids = np.concatenate(tokid_all)
        H_by_layer = {L: np.concatenate(H_layers[L], 0) for L in layers}
        T = doc_ids.shape[0]

        # reference scores + reference firings/gated via independent numpy
        S, norms = scores_numpy(H_by_layer, bundle)
        ref_fire = []            # (doc,pos,cid,layer,score)
        ref_gated = {L: {ci: [] for ci in range(n_rows)} for L in layers}
        for L in layers:
            for ci in range(n_rows):
                thr = bundle.thr[L][ci]
                for t in range(T):
                    sv = S[L][t, ci]
                    if sv > thr:
                        ref_fire.append((int(doc_ids[t]), int(token_pos[t]), ci, L,
                                         np.float16(sv)))
                    else:
                        ref_gated[L][ci].append(sv)

        # run the real writer path (streams in 2 sub-batches to exercise Welford merge)
        w = ShardWriter(work, shard, bundle)
        half = T // 2
        for sl in (slice(0, half), slice(half, T)):
            sub = {L: H_by_layer[L][sl] for L in layers}
            Ssub, Nsub = scores_numpy(sub, bundle)
            w.add_batch(Ssub, Nsub, doc_ids[sl], token_pos[sl], token_ids[sl])
        info = w.close()

        # ---- (a) firings parquet == reference set (only above-threshold cells) -----
        ftbl = pq.read_table(info["firings"])
        got = set(zip(ftbl.column("doc_id").to_pylist(), ftbl.column("token_pos").to_pylist(),
                      ftbl.column("concept_id").to_pylist(), ftbl.column("layer").to_pylist()))
        ref_set = set((dd, pp, cc, ll) for (dd, pp, cc, ll, _s) in ref_fire)
        fire_ok = (got == ref_set) and (ftbl.num_rows == len(ref_fire))
        # scores match (fp16) for each cell
        got_score = {(r["doc_id"], r["token_pos"], r["concept_id"], r["layer"]): r["score"]
                     for r in ftbl.to_pylist()}
        score_ok = all(abs(float(got_score[(dd, pp, cc, ll)]) - float(s)) < 1e-3
                       for (dd, pp, cc, ll, s) in ref_fire)
        # schema/dtype check
        schema_ok = (ftbl.schema.field("score").type == pa.float16() and
                     ftbl.schema.field("concept_id").type == pa.int16() and
                     ftbl.schema.field("layer").type == pa.int8() and
                     ftbl.schema.field("doc_id").type == pa.int32())

        # ---- (b) gated-tail Welford mean/std == direct numpy to <1e-6 -------------
        ttbl = pq.read_table(info["tailstats"]).to_pylist()
        tmap = {(r["concept_id"], r["layer"]): r for r in ttbl}
        welford_ok = True
        max_err = 0.0
        for L in layers:
            for ci in range(n_rows):
                vals = np.asarray(ref_gated[L][ci], dtype=np.float64)
                row = tmap[(ci, L)]
                if len(vals) == 0:
                    welford_ok &= (row["gated_count"] == 0)
                    continue
                ref_mean = float(vals.mean())
                ref_std = float(vals.std())   # population (ddof=0)
                em = abs(row["gated_mean"] - ref_mean)
                es = abs(row["gated_std"] - ref_std)
                max_err = max(max_err, em, es)
                welford_ok &= (row["gated_count"] == len(vals)) and em < 1e-6 and es < 1e-6
                # fired+gated == total
                welford_ok &= (row["fired_count"] + row["gated_count"] == row["total_count"] == T)

        # ---- (d) per-token side table dedups (one row per fired token) ------------
        toktbl = pq.read_table(info["tokens"])
        keys = list(zip(toktbl.column("doc_id").to_pylist(), toktbl.column("token_pos").to_pylist()))
        fired_tokens_ref = set((dd, pp) for (dd, pp, _c, _l, _s) in ref_fire)
        dedup_ok = (len(keys) == len(set(keys))) and (set(keys) == fired_tokens_ref)
        # token side table carries all 12-layer norms (here len(layers)) and a token_id col
        cols_ok = ("token_id" in toktbl.column_names) and \
                  all(f"h_norm_L{L}" in toktbl.column_names for L in layers)
        # ||h|| values match
        norm_ok = True
        if keys:
            kpos = {(dd, pp): t for t, (dd, pp) in enumerate(zip(doc_ids.tolist(), token_pos.tolist()))}
            tt = toktbl.to_pylist()
            for r in tt:
                t = kpos[(r["doc_id"], r["token_pos"])]
                for j, L in enumerate(layers):
                    norm_ok &= abs(float(r[f"h_norm_L{L}"]) - float(np.float16(norms[t, j]))) < 1e-2

        all_checks += [
            (f"shard{shard} (a) firings == above-threshold ref set", fire_ok),
            (f"shard{shard}     firing scores match (fp16)", score_ok),
            (f"shard{shard} (c) firings parquet schema/dtypes", schema_ok),
            (f"shard{shard} (b) Welford tail mean/std <1e-6 (max_err={max_err:.2e})", welford_ok),
            (f"shard{shard} (d) token side table dedups + cols + norms", dedup_ok and cols_ok and norm_ok),
        ]

    # ---- (e) resume skips an already-uploaded shard ---------------------------
    class _FakeApi:
        def __init__(self, present):
            self.present = set(present)
        def file_exists(self, repo_id, filename, repo_type):
            return filename in self.present
    fake = _FakeApi({f"{DIR_FIRINGS}/shard_00002.firings.parquet"})
    resume_ok = hf_shard_done(fake, 2) and (not hf_shard_done(fake, 5))
    all_checks.append(("(e) resume: skip uploaded shard 2, run missing shard 5", resume_ok))

    # ---- shard-spec parsing incl. the 6542 val shard --------------------------
    spec_ok = parse_shard_spec("0-185,6542") == list(range(0, 186)) + [6542] and \
        parse_shard_spec("0-2,5") == [0, 1, 2, 5]
    all_checks.append(("(spec) shard range '0-185,6542' parses correctly", spec_ok))
    all_checks.append(("(thr) thresholds resolved from artifacts", ok_thr))
    all_checks.append(("(apply) torch==numpy apply cross-check", xc_ok))

    print("\n--- checks ---")
    ok = True
    for name, passed in all_checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and bool(passed)
    print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'}")
    # cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Phase C corpus-attribution sweep")
    ap.add_argument("--selftest", action="store_true", help="run CPU synthetic self-test and exit")
    ap.add_argument("--shards", default=None, help="shard spec, e.g. '0-185,6542' or '0-92'")
    ap.add_argument("--weights-dir", default=None,
                    help="local dir with the probe weights/ + summary.json (snapshot of WEIGHTS_REPO)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--work-dir", default=None, help="scratch dir (bounded to <=1-2 shards)")
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--max-batch-docs", type=int, default=64)
    ap.add_argument("--scalar-threshold", type=float, default=0.5,
                    help="firing threshold on sigmoid magnitude for scalar rows (no calibrated thr)")
    ap.add_argument("--default-binary-threshold", type=float, default=0.5,
                    help="fallback threshold for a binary (row,layer) with no calibrated value")
    ap.add_argument("--gate-mode", default="relative", choices=["relative", "absolute"],
                    help="relative (per-(row,layer) corpus percentile; DEFAULT) or absolute (legacy)")
    ap.add_argument("--top-frac", type=float, default=0.002,
                    help="relative gate: keep the top this-fraction of corpus tokens per (row,layer)")
    ap.add_argument("--calib-shard", type=int, default=48,
                    help="corpus shard to calibrate the relative gate on (fixed -> fleet-consistent)")
    ap.add_argument("--calib-tokens", type=int, default=1_000_000,
                    help="max corpus tokens used for relative-gate calibration")
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"],
                    help="attention impl for gemma-2 (eager=reference; sdpa only if verified equal)")
    ap.add_argument("--shard-filename-template", default="shard_{:05d}.parquet")
    ap.add_argument("--write-global-manifest", action="store_true",
                    help="(re)write the top-level manifest.json once before the loop")
    ap.add_argument("--keep-local", action="store_true", help="do NOT delete local shard outputs")
    ap.add_argument("--skip-upload", action="store_true", help="run locally, no HF upload (debug)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.selftest:
        sys.exit(selftest())
    if not args.shards or not args.weights_dir:
        ap.error("--shards and --weights-dir are required (or use --selftest)")
    run(args.shards, args.weights_dir, device=args.device, work_dir=args.work_dir,
        max_seq=args.max_seq, batch_tokens=args.batch_tokens, max_batch_docs=args.max_batch_docs,
        scalar_threshold=args.scalar_threshold,
        default_binary_threshold=args.default_binary_threshold,
        shard_filename_template=args.shard_filename_template,
        write_global_manifest=args.write_global_manifest,
        keep_local=args.keep_local, skip_upload=args.skip_upload,
        gate_mode=args.gate_mode, top_frac=args.top_frac,
        calib_shard=args.calib_shard, calib_tokens=args.calib_tokens, attn=args.attn)


if __name__ == "__main__":
    main()
