#!/usr/bin/env python3
"""Stage 7-Oracle Phase 2/3 — encoder training (Exp A + Exp B).

SPEC.md Phase 2 ("Exp A": probe-prediction encoder) and Phase 3 ("Exp B":
ablation-repair dual). DESIGN.md sections "Score store", "align.py API",
"Exp A/B training data flow", "Pod conventions" are the frozen contract this
file implements.

Architecture (one stack for A and B, DESIGN.md/SPEC.md Phase 2):
    Qwen encoder (causal, hidden H) --[gather at gemma_to_qwen_map indices]-->
    linear `up` (H -> n_targets) --[Exp B only]--> linear `down` (K -> 2304)

Modes (--mode):
  expA        up: H -> 3K, MSE vs corpus-standardized dequantized scores.
  expB-fixed  up: H -> K,  predicts y = G_inv (s_dom - t_nat_dom); decoder D
              is FIXED (closed-form, not trained); R² also reported after
              decoding to v* = D @ y (SPEC Phase 3 variant (i), "literally
              Exp A in different units").
  expB-learn  up: H -> K, down: K -> 2304 (learnable, no bias); loss is MSE
              on v* directly (variant (ii)); logs per-concept
              cosine(down column, D column) each eval (free interpretability
              check per DESIGN.md).

Data flow (DESIGN.md "Exp A/B training data flow"): batch unit = doc. Per
doc: recover raw text from the ClimbMix shard by doc index (docs_<sid>.jsonl
+ the Stage-6 shard loader, see `stage6/code/mine_natural.py` /
`nat_common.py`) -> re-tokenize with gemma (offsets) -> assert token-id
reproduction against tokens_<sid>.npy for the first N docs (hard fail
otherwise: it means the Phase-1 scoring convention drifted from what this
script assumes) -> qwen-tokenize the same char span (truncate both to the
shorter char prefix if qwen exceeds --max-qwen-tokens) -> forward Qwen (bf16)
-> gather hidden states at gemma_to_qwen_map(mode="prefix") indices, dropping
-1 positions -> head -> loss vs dequantized/standardized scores.

Smoke test: test_train_encoder.py (tiny random Qwen2Config model, tiny
synthetic score store, real small tokenizer pair).
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pyarrow.parquet as pq
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
STAGE6_CODE = HERE.parent.parent / "stage6" / "code"
if str(STAGE6_CODE) not in sys.path:
    sys.path.insert(0, str(STAGE6_CODE))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import nat_common as nc  # Stage-6 ClimbMix shard loader (hub-download fallback only)
except ImportError:
    nc = None

# align.py may not exist yet (DESIGN.md's align.py is written by a parallel
# agent per SPEC.md's timeline); code against the frozen API, fall back to a
# local minimal-correct implementation if the module is absent.
try:
    from align import gemma_to_qwen_map, crossing_rate  # noqa: F401
    ALIGN_SOURCE = "align"
except ImportError:
    from _align_fallback import gemma_to_qwen_map, crossing_rate  # noqa: F401
    ALIGN_SOURCE = "_align_fallback"

from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

QWEN_PRIMARY = "Qwen/Qwen3-0.6B-Base"
QWEN_FALLBACK = "Qwen/Qwen2.5-0.5B"
GEMMA_TOKENIZER_DEFAULT = "google/gemma-2-2b"
D_MODEL_GEMMA = 2304  # gemma-2-2b residual width (Exp B v* dimension)

SCRIPT = "train_encoder"


# ============================================================== heartbeat
def heartbeat(path, **fields):
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), **fields}) + "\n")
    except OSError:
        pass


# ========================================================== score store IO
def load_quant(scores_dir):
    with open(os.path.join(scores_dir, "quant.json")) as f:
        q = json.load(f)
    return np.array(q["zero"], dtype=np.float32), np.array(q["scale"], dtype=np.float32)


def load_corpus_stats(scores_dir, zero, scale):
    """corpus_stats.json (Phase-1 streaming true mean/std) if present, else
    derive approximate stats from the quantization calibration (DESIGN.md:
    zero=mean, scale=4*std/127) and warn -- this is what the task spec calls
    for explicitly ("if absent, standardize with quant.json zero/scale-
    derived stats and warn")."""
    p = os.path.join(scores_dir, "corpus_stats.json")
    if os.path.exists(p):
        with open(p) as f:
            cs = json.load(f)
        return np.array(cs["mean"], dtype=np.float32), np.array(cs["std"], dtype=np.float32), True
    warnings.warn(
        f"{p} not found; standardizing with quant.json zero/scale-derived "
        f"stats (mean=zero, std=scale*127/4). This is an approximation of "
        f"the true corpus mean/std."
    )
    std = np.maximum(scale * 127.0 / 4.0, 1e-6)
    return zero.copy(), std, False


def dequantize(int8_arr, zero, scale):
    """int8 [..., C] -> float32 raw score units. quant convention (DESIGN.md):
    int8 = clip(round((score - zero) / scale), -127, 127)."""
    return int8_arr.astype(np.float32) * scale + zero


class ProbeSet:
    """Loads probe_set.json + probe_set_arrays.npz (DESIGN.md schema) from a
    directory and derives the Exp-B closed-form pieces (SPEC.md Phase 3)."""

    def __init__(self, probe_set_dir):
        with open(os.path.join(probe_set_dir, "probe_set.json")) as f:
            self.meta = json.load(f)
        npz = np.load(os.path.join(probe_set_dir, "probe_set_arrays.npz"))
        self.W = npz["W"].astype(np.float32)                # [3, K, 2304]
        self.b = npz["b"].astype(np.float32)                 # [3, K]
        self.nat_mean = npz["nat_mean"].astype(np.float32)   # [3, 2304]
        self.nat_std = npz["nat_std"].astype(np.float32)     # [3, 2304]
        self.W_dom_abl = npz["W_dom_abl"].astype(np.float32)  # [K, 2304]
        self.b_dom_abl = npz["b_dom_abl"].astype(np.float32)  # [K]
        self.t_nat_dom = npz["t_nat_dom"].astype(np.float32)  # [K]
        self.G_dom = npz["G_dom"].astype(np.float32)          # [K, K]
        self.G_dom_inv = npz["G_dom_inv"].astype(np.float32)  # [K, K]
        self.layer_index = npz["layer_index"]                 # [3]

        self.layers = list(self.meta["layers"])
        self.ablation_layer = int(self.meta["ablation_layer"])
        self.concepts = list(self.meta["concepts"])
        self.K = len(self.concepts)
        self.families = self.meta["families"]

        if self.ablation_layer not in list(self.layer_index):
            raise ValueError(
                f"ablation_layer={self.ablation_layer} not among probe "
                f"layer_index={list(self.layer_index)}; DESIGN.md's D_dom "
                f"construction (d_c = nat_std ⊙ W_dom_abl[c]) needs the "
                f"ablation layer's natural-pool std, which is only stored "
                f"for the 3 probe layers."
            )
        abl_idx = int(np.where(self.layer_index == self.ablation_layer)[0][0])
        nat_std_abl = self.nat_std[abl_idx]  # [2304]
        # d_c = nat_std ⊙ w_c^dom  (SPEC.md Phase 3); D_dom columns = d_c.
        self.D_dom = (nat_std_abl[None, :] * self.W_dom_abl).T.astype(np.float32)  # [2304, K]

    def n_score_cols(self):
        return 4 * self.K  # [layer0 K, layer1 K, layer2 K, dom@ablation K]

    def v_star(self, s_dom_raw):
        """s_dom_raw: [..., K] raw (dequantized) dom scores at ablation_layer.
        Returns (y [..., K], v [..., 2304])."""
        y = (s_dom_raw - self.t_nat_dom) @ self.G_dom_inv  # G_dom_inv symmetric
        v = y @ self.D_dom.T
        return y, v


# ============================================================== shard text
def find_local_shard(climbmix_dir, sid):
    p = os.path.join(climbmix_dir, f"shard_{sid:05d}.parquet")
    return p if os.path.exists(p) else None


def load_shard_texts(climbmix_dir, sid, wanted_doc_idxs):
    """Recover raw doc text by doc-index-within-shard, mirroring the Stage-6
    shard loader (stage6/code/mine_natural.py + nat_common.py): local
    parquet file first (pod-local raw shards under --climbmix-dir), else
    fall back to nat_common's HF-hub loader (same shard-file naming/schema)
    if available. Single sequential pass, in parquet row order (deterministic,
    matches how docs_<sid>.jsonl doc indices were assigned)."""
    wanted = set(wanted_doc_idxs)
    out = {}
    local = find_local_shard(climbmix_dir, sid)
    if local is not None:
        pf = pq.ParquetFile(local)
        col = nc.detect_text_column(pf) if nc is not None else _detect_text_column(pf)
        n = 0
        for batch in pf.iter_batches(batch_size=512, columns=[col]):
            for v in batch.column(0):
                if n in wanted:
                    out[n] = v.as_py()
                n += 1
                if len(out) == len(wanted):
                    return out
        return out
    if nc is None:
        raise FileNotFoundError(
            f"shard_{sid:05d}.parquet not found under {climbmix_dir} and "
            f"nat_common (hub-download fallback) is not importable."
        )
    max_idx = max(wanted)
    for idx, text in nc.iter_shard_docs(sid, max_docs=max_idx + 1):
        if idx in wanted:
            out[idx] = text
            if len(out) == len(wanted):
                break
    return out


def _detect_text_column(pf):
    import pyarrow as pa
    sch = pf.schema_arrow
    for cand in ("text", "content", "raw_content", "document", "body"):
        if cand in sch.names and (pa.types.is_string(sch.field(cand).type)
                                   or pa.types.is_large_string(sch.field(cand).type)):
            return cand
    for name in sch.names:
        t = sch.field(name).type
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            return name
    raise RuntimeError(f"no string column found; schema={sch.names}")


class ShardStore:
    """Per-shard tokens_<sid>.npy / scores_<sid>.npy memmaps + docs_<sid>.jsonl."""

    def __init__(self, scores_dir, sid):
        self.sid = sid
        self.tokens = np.load(os.path.join(scores_dir, f"tokens_{sid:05d}.npy"), mmap_mode="r")
        self.scores = np.load(os.path.join(scores_dir, f"scores_{sid:05d}.npy"), mmap_mode="r")
        self.docs = []
        with open(os.path.join(scores_dir, f"docs_{sid:05d}.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.docs.append(json.loads(line))


def iter_docs(shards, scores_dir, climbmix_dir, loop):
    """Yields dicts: {sid, doc_idx, text, gemma_ids (int32 np array, stored),
    scores_raw_i8 (int8 [n, 4K], stored slice)}. `loop`=True cycles forever
    (training stream); False = single pass (eval)."""
    while True:
        for sid in shards:
            store = ShardStore(scores_dir, sid)
            doc_idxs = [d["doc"] for d in store.docs]
            texts = load_shard_texts(climbmix_dir, sid, doc_idxs)
            for d in store.docs:
                text = texts.get(d["doc"])
                if text is None:
                    continue
                start, n = d["start"], d["n"]
                yield {
                    "sid": sid,
                    "doc_idx": d["doc"],
                    "text": text,
                    "gemma_ids": np.array(store.tokens[start:start + n]),
                    "scores_raw_i8": np.array(store.scores[start:start + n, :]),
                }
        if not loop:
            return


# =========================================================== doc processing
def process_doc(doc, gemma_tok, qwen_tok, max_gemma_tokens, max_qwen_tokens,
                 min_gemma_tokens, assert_tokens=False):
    """Re-tokenize gemma (offsets), verify against stored ids if requested,
    qwen-tokenize the same char span (truncate both to shorter char prefix
    if qwen exceeds max_qwen_tokens), align, gather target rows.

    Returns None if the doc should be skipped (too short / degenerate), else
    a dict {qwen_ids: List[int], map_idx: np.ndarray[int] (into qwen_ids,
    already filtered to valid i.e. no -1), scores_raw_i8: np.ndarray[int8]
    [n_valid, 4K] filtered to the same valid gemma positions}.
    """
    text = doc["text"]
    # Tokenizer convention MUST match score_corpus.py (Phase 1) exactly:
    # tok(text, add_special_tokens=False)["input_ids"][:MAX_DOC_TOKENS] --
    # BOS is NEVER obtained from the tokenizer call; score_corpus.py
    # manually prepends BOS before the forward pass and drops that hidden
    # row before storing, so stored tokens_<sid>.npy positions are the raw
    # BOS-free ids, truncated by plain python slicing (not tokenizer
    # truncation=True/max_length=, to match byte-for-byte).
    enc_g = gemma_tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids_g = np.array(enc_g["input_ids"][:max_gemma_tokens], dtype=np.int32)
    offsets_g = enc_g["offset_mapping"][:max_gemma_tokens]

    stored = doc["gemma_ids"].astype(np.int32)
    if assert_tokens:
        if ids_g.shape != stored.shape or not np.array_equal(ids_g, stored):
            n_show = min(len(ids_g), len(stored), 20)
            raise RuntimeError(
                "TOKEN-ID REPRODUCTION FAILED (convention drift) for "
                f"shard={doc['sid']} doc={doc['doc_idx']}: re-tokenizing "
                f"the recovered raw text with the gemma tokenizer "
                f"(add_special_tokens=False, sliced to the first "
                f"{max_gemma_tokens} ids -- score_corpus.py's convention, "
                f"BOS is added/dropped around the forward pass, never "
                f"present in the stored ids) does NOT reproduce the stored "
                f"tokens_<sid>.npy slice.\n"
                f"  stored  (n={len(stored)}): {stored[:n_show].tolist()}...\n"
                f"  retok   (n={len(ids_g)}):  {ids_g[:n_show].tolist()}...\n"
                f"This means Phase-1 scoring used a different tokenization "
                f"convention (e.g. add_special_tokens, different truncation, "
                f"different gemma tokenizer revision) than this script "
                f"assumes -- fix the convention here or in score_corpus.py "
                f"before proceeding; do NOT silently ignore."
            )

    if len(stored) < min_gemma_tokens:
        return None

    nonzero_ends = [e for (_, e) in offsets_g if e > 0]
    if not nonzero_ends:
        return None
    gemma_char_end = max(nonzero_ends)
    char_prefix = text[:gemma_char_end]

    enc_q = qwen_tok(char_prefix, add_special_tokens=True, return_offsets_mapping=True)
    ids_q = enc_q["input_ids"]
    offsets_q = enc_q["offset_mapping"]

    if len(ids_q) > max_qwen_tokens:
        # DESIGN.md: "truncate BOTH to the shorter char prefix"
        new_char_end = offsets_q[max_qwen_tokens - 1][1]
        char_prefix = text[:new_char_end]
        enc_g = gemma_tok(char_prefix, add_special_tokens=False, return_offsets_mapping=True)
        ids_g = np.array(enc_g["input_ids"][:max_gemma_tokens], dtype=np.int32)
        offsets_g = enc_g["offset_mapping"][:max_gemma_tokens]
        enc_q = qwen_tok(char_prefix, add_special_tokens=True, return_offsets_mapping=True)
        ids_q = enc_q["input_ids"][:max_qwen_tokens]
        offsets_q = enc_q["offset_mapping"][:max_qwen_tokens]
        # gemma tokens are stored-fixed; keep only the prefix of stored rows
        # whose char span now fits inside the (possibly shrunk) prefix.
        n_keep = len(ids_g)
        stored = stored[:n_keep]
        scores_raw_i8 = doc["scores_raw_i8"][:n_keep]
        gemma_offsets_used = offsets_g[:n_keep]
    else:
        scores_raw_i8 = doc["scores_raw_i8"]
        gemma_offsets_used = offsets_g

    n_g = min(len(stored), len(gemma_offsets_used))
    if n_g < min_gemma_tokens or len(ids_q) == 0:
        return None
    gemma_offsets_used = gemma_offsets_used[:n_g]
    scores_raw_i8 = scores_raw_i8[:n_g]

    map_idx = gemma_to_qwen_map(char_prefix, gemma_offsets_used, offsets_q, mode="prefix")
    valid = map_idx >= 0
    if not valid.any():
        return None

    return {
        "qwen_ids": ids_q,
        "map_idx": map_idx[valid].astype(np.int64),
        "scores_raw_i8": scores_raw_i8[valid],
    }


# =================================================================== model
def load_encoder(model_name, dtype, device):
    """AutoModel.from_pretrained with the Qwen3->Qwen2.5 fallback (SPEC risk
    register #2). Returns (model, tokenizer, actual_model_name)."""
    names_to_try = [model_name]
    if model_name == QWEN_PRIMARY:
        names_to_try.append(QWEN_FALLBACK)
    last_err = None
    for name in names_to_try:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            model = AutoModel.from_pretrained(name, dtype=dtype)
            model.to(device)
            return model, tok, name
        except Exception as e:  # noqa: BLE001 -- deliberately broad, we try the fallback next
            last_err = e
            continue
    raise RuntimeError(f"could not load any of {names_to_try}: {last_err}")


def load_gemma_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)


class EncoderHead(nn.Module):
    def __init__(self, hidden_size, K, mode):
        super().__init__()
        self.mode = mode
        if mode == "expA":
            self.up = nn.Linear(hidden_size, 3 * K)
            self.down = None
        elif mode == "expB-fixed":
            self.up = nn.Linear(hidden_size, K)
            self.down = None
        elif mode == "expB-learn":
            self.up = nn.Linear(hidden_size, K)
            self.down = nn.Linear(K, D_MODEL_GEMMA, bias=False)
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def forward(self, h):
        y = self.up(h)
        v = self.down(y) if self.down is not None else None
        return y, v


# ================================================================ batching
def collate_batch(docs, qwen_tok, device):
    """docs: list of process_doc() outputs. Right-pads qwen ids, forwards a
    single batch through the encoder is done by the caller; this just builds
    the padded input tensors + per-doc valid-position bookkeeping."""
    lens = [len(d["qwen_ids"]) for d in docs]
    max_len = max(lens)
    pad_id = qwen_tok.pad_token_id
    input_ids = torch.full((len(docs), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(docs), max_len), dtype=torch.long)
    for i, d in enumerate(docs):
        L = lens[i]
        input_ids[i, :L] = torch.tensor(d["qwen_ids"], dtype=torch.long)
        attn_mask[i, :L] = 1
    return input_ids.to(device), attn_mask.to(device)


def gather_targets(docs, batch_row_offset_hidden, hidden):
    """hidden: [B, Tq_max, H] encoder output. For each doc, gather rows at
    its map_idx (positions within that doc's own qwen sequence -- valid
    because padding is on the right and causal attention never looks
    forward, so padded tail positions don't leak into real positions).
    Returns concatenated (features [N,H] tensor, scores_raw_i8 [N,4K] np)."""
    feats = []
    scores = []
    for i, d in enumerate(docs):
        idx = torch.as_tensor(d["map_idx"], device=hidden.device, dtype=torch.long)
        feats.append(hidden[i, idx, :])
        scores.append(d["scores_raw_i8"])
    return torch.cat(feats, dim=0), np.concatenate(scores, axis=0)


# =============================================================== R^2 accum
class R2Accumulator:
    """Streaming per-column R^2: R2 = 1 - SSE/SST, SST = sum((y-mean)^2) =
    sum(y^2) - n*mean^2, computed from running sums (no need to hold all
    eval data in memory -- DESIGN.md's "streaming, fixed ~20M-token
    subsample" eval)."""

    def __init__(self, n_cols):
        self.n_cols = n_cols
        self.n = 0
        self.sum_y = np.zeros(n_cols, dtype=np.float64)
        self.sum_y2 = np.zeros(n_cols, dtype=np.float64)
        self.sum_err2 = np.zeros(n_cols, dtype=np.float64)

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        self.n += y_true.shape[0]
        self.sum_y += y_true.sum(axis=0)
        self.sum_y2 += (y_true ** 2).sum(axis=0)
        self.sum_err2 += ((y_true - y_pred) ** 2).sum(axis=0)

    def r2(self):
        if self.n == 0:
            return np.zeros(self.n_cols)
        mean = self.sum_y / self.n
        sst = self.sum_y2 - self.n * mean ** 2
        sst = np.maximum(sst, 1e-8)
        r2 = 1.0 - self.sum_err2 / sst
        return r2

    def r2_overall(self):
        """Single scalar R^2 pooling all columns (used for Exp-B v* space)."""
        if self.n == 0:
            return 0.0
        sst = self.sum_y2.sum() - (self.sum_y.sum() ** 2) / (self.n * self.n_cols)
        sst = max(sst, 1e-8)
        return float(1.0 - self.sum_err2.sum() / sst)


# ============================================================ standardize
def standardize(raw, mean, std):
    return (raw - mean) / np.maximum(std, 1e-6)


# =============================================================== eval loop
@torch.no_grad()
def run_eval(shards, scores_dir, climbmix_dir, gemma_tok, qwen_tok, model, head,
             ps, mode, corpus_mean, corpus_std, quant_zero, quant_scale,
             bsz_docs, max_gemma_tokens, max_qwen_tokens, min_gemma_tokens,
             eval_tokens, device):
    K = ps.K
    n_targets = 3 * K if mode == "expA" else K
    acc = R2Accumulator(n_targets)
    acc_v = R2Accumulator(D_MODEL_GEMMA) if mode.startswith("expB") else None

    buf = []
    n_tokens_seen = 0
    stream = iter_docs(shards, scores_dir, climbmix_dir, loop=False)

    def flush():
        nonlocal buf
        if not buf:
            return 0
        input_ids, attn_mask = collate_batch(buf, qwen_tok, device)
        out = model(input_ids=input_ids, attention_mask=attn_mask)
        hidden = out.last_hidden_state
        feats, scores_raw_i8 = gather_targets(buf, None, hidden)
        y_pred, v_pred = head(feats)
        n_new = _accumulate(mode, ps, scores_raw_i8, y_pred, v_pred,
                             corpus_mean, corpus_std, quant_zero, quant_scale,
                             acc, acc_v)
        buf = []
        return n_new

    for doc in stream:
        if n_tokens_seen >= eval_tokens:
            break
        pd = process_doc(doc, gemma_tok, qwen_tok, max_gemma_tokens,
                          max_qwen_tokens, min_gemma_tokens, assert_tokens=False)
        if pd is None:
            continue
        buf.append(pd)
        if len(buf) >= bsz_docs:
            n_tokens_seen += flush()
    n_tokens_seen += flush()

    result = {"n_tokens": n_tokens_seen}
    if mode == "expA":
        r2 = acc.r2()
        names = [f"{ps.concepts[c]}@L{ps.layers[l]}" for l in range(3) for c in range(K)]
        result["per_probe_r2"] = dict(zip(names, r2.tolist()))
        result["median_r2"] = float(np.median(r2))
        fam_groups = {}
        for l in range(3):
            for c in range(K):
                fam = ps.families[ps.concepts[c]]
                fam_groups.setdefault(fam, []).append(r2[l * K + c])
        result["per_family_median_r2"] = {f: float(np.median(v)) for f, v in fam_groups.items()}
        result["primary_metric"] = result["median_r2"]
    else:
        r2 = acc.r2()
        names = ps.concepts
        result["per_probe_r2"] = dict(zip(names, r2.tolist()))
        result["median_r2"] = float(np.median(r2))
        fam_groups = {}
        for c in range(K):
            fam = ps.families[ps.concepts[c]]
            fam_groups.setdefault(fam, []).append(r2[c])
        result["per_family_median_r2"] = {f: float(np.median(v)) for f, v in fam_groups.items()}
        result["v_star_r2"] = acc_v.r2_overall()
        result["v_star_per_dim_r2_median"] = float(np.median(acc_v.r2()))
        result["primary_metric"] = result["v_star_r2"]
    return result


def _accumulate(mode, ps, scores_raw_i8, y_pred, v_pred, corpus_mean, corpus_std,
                 quant_zero, quant_scale, acc, acc_v):
    raw = dequantize(scores_raw_i8, quant_zero, quant_scale)  # [N,4K]
    if mode == "expA":
        target = standardize(raw[:, :3 * ps.K], corpus_mean[:3 * ps.K], corpus_std[:3 * ps.K])
        acc.update(target, y_pred.float().cpu().numpy())
    else:
        s_dom = raw[:, 3 * ps.K:4 * ps.K]
        y_true, v_true = ps.v_star(s_dom)
        acc.update(y_true, y_pred.float().cpu().numpy())
        if mode == "expB-fixed":
            v_pred_np = (y_pred.float().cpu().numpy()) @ ps.D_dom.T
        else:
            v_pred_np = v_pred.float().cpu().numpy()
        acc_v.update(v_true, v_pred_np)
    return raw.shape[0]


# =================================================================== loss
def compute_loss(mode, ps, scores_raw_i8, y_pred, v_pred, corpus_mean, corpus_std,
                  quant_zero, quant_scale, device):
    raw = dequantize(scores_raw_i8, quant_zero, quant_scale)
    if mode == "expA":
        target = standardize(raw[:, :3 * ps.K], corpus_mean[:3 * ps.K], corpus_std[:3 * ps.K])
        target_t = torch.as_tensor(target, dtype=y_pred.dtype, device=device)
        return ((y_pred - target_t) ** 2).mean()
    s_dom = raw[:, 3 * ps.K:4 * ps.K]
    y_true, v_true = ps.v_star(s_dom)
    if mode == "expB-fixed":
        y_true_t = torch.as_tensor(y_true, dtype=y_pred.dtype, device=device)
        return ((y_pred - y_true_t) ** 2).mean()
    v_true_t = torch.as_tensor(v_true, dtype=v_pred.dtype, device=device)
    return ((v_pred - v_true_t) ** 2).mean()


# ============================================================= checkpoints
def save_checkpoint(path, step, head, model, full_ft, optimizer, args, ps, model_name, hidden_size):
    state = {
        "step": step,
        "head_state": head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
        "mode": args.mode,
        "model_name": model_name,
        "hidden_size": hidden_size,
        "K": ps.K,
        "concepts": ps.concepts,
    }
    if full_ft:
        state["encoder_state"] = model.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, head, model, optimizer, full_ft, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    head.load_state_dict(state["head_state"])
    if full_ft and "encoder_state" in state:
        model.load_state_dict(state["encoder_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    return state["step"]


# ==================================================================== cli
def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True, help="dir with tokens_<sid>.npy/scores_<sid>.npy/docs_<sid>.jsonl/quant.json/[corpus_stats.json]")
    p.add_argument("--climbmix-dir", required=True, help="dir with raw ClimbMix shard parquet files (for text recovery)")
    p.add_argument("--probe-set", required=True, help="dir with probe_set.json + probe_set_arrays.npz")
    p.add_argument("--mode", required=True, choices=["expA", "expB-fixed", "expB-learn"])
    ft = p.add_mutually_exclusive_group()
    ft.add_argument("--freeze-encoder", action="store_true", default=True, dest="freeze_encoder")
    ft.add_argument("--full-ft", action="store_false", dest="freeze_encoder")
    p.add_argument("--train-shards", required=True, help="comma-separated shard ids")
    p.add_argument("--val-shards", required=True, help="comma-separated shard ids")
    p.add_argument("--model", default=QWEN_PRIMARY)
    p.add_argument("--gemma-model", default=GEMMA_TOKENIZER_DEFAULT,
                    help="tokenizer used to re-derive gemma offsets / verify token-id reproduction (not listed in the task's arg table; needed for the DESIGN.md alignment step -- documented ambiguity)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--bsz-docs", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-tokens", type=int, default=20_000_000)
    p.add_argument("--max-gemma-tokens", type=int, default=2048)
    p.add_argument("--max-qwen-tokens", type=int, default=3072)
    p.add_argument("--min-gemma-tokens", type=int, default=64)
    p.add_argument("--assert-first-n-docs", type=int, default=100)
    p.add_argument("--early-stop-r2-delta", type=float, default=0.005)
    p.add_argument("--early-stop-window-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", default=None)
    p.add_argument("--heartbeat-path", default="/workspace/hb_train.txt")
    p.add_argument("--heartbeat-interval", type=float, default=60.0)
    p.add_argument("--out", required=True)
    return p


# =================================================================== main
def run_training(args, encoder_and_tok=None):
    """encoder_and_tok: optional (model, qwen_tok, model_name) override, used
    by the smoke test to inject a tiny random model instead of downloading
    Qwen3-0.6B-Base."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics.jsonl")

    ps = ProbeSet(args.probe_set)
    zero, scale = load_quant(args.scores)
    corpus_mean, corpus_std, corpus_stats_real = load_corpus_stats(args.scores, zero, scale)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    if encoder_and_tok is not None:
        model, qwen_tok, model_name = encoder_and_tok
        model.to(args.device)
    else:
        model, qwen_tok, model_name = load_encoder(args.model, dtype, args.device)
    gemma_tok = load_gemma_tokenizer(args.gemma_model)

    hidden_size = model.config.hidden_size
    head = EncoderHead(hidden_size, ps.K, args.mode).to(args.device)
    if dtype == torch.bfloat16:
        head = head.to(dtype)

    if args.freeze_encoder:
        for prm in model.parameters():
            prm.requires_grad_(False)
        model.eval()
        trainable = list(head.parameters())
    else:
        trainable = list(model.parameters()) + list(head.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, args.max_steps // 20)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=args.max_steps)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, head, model, optimizer, not args.freeze_encoder, args.device)
        print(f"[{SCRIPT}] resumed from {args.resume} at step {start_step}")

    train_shards = [int(x) for x in args.train_shards.split(",") if x != ""]
    val_shards = [int(x) for x in args.val_shards.split(",") if x != ""]

    train_stream = iter_docs(train_shards, args.scores, args.climbmix_dir, loop=True)

    best_metric = -1e9
    best_path = os.path.join(args.out, "best.pt")
    last_path = os.path.join(args.out, "last.pt")
    eval_history = []  # (step, primary_metric)

    n_docs_asserted = 0
    buf = []
    step = start_step
    last_hb = 0.0
    t0 = time.time()
    tokens_since_hb = 0
    last_loss = float("nan")

    pbar = tqdm(total=args.max_steps, initial=start_step, desc=f"train[{args.mode}]")
    stop = False
    while step < args.max_steps and not stop:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        n_accum_tokens = 0
        for _ in range(args.grad_accum):
            buf = []
            while len(buf) < args.bsz_docs:
                doc = next(train_stream)
                assert_this = n_docs_asserted < args.assert_first_n_docs
                pd = process_doc(doc, gemma_tok, qwen_tok, args.max_gemma_tokens,
                                  args.max_qwen_tokens, args.min_gemma_tokens,
                                  assert_tokens=assert_this)
                if assert_this:
                    n_docs_asserted += 1
                if pd is None:
                    continue
                buf.append(pd)
            input_ids, attn_mask = collate_batch(buf, qwen_tok, args.device)
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else _nullcontext()
            enc_ctx = torch.no_grad() if args.freeze_encoder else _nullcontext()
            with enc_ctx:
                with ctx:
                    out = model(input_ids=input_ids, attention_mask=attn_mask)
                    hidden = out.last_hidden_state
            if args.freeze_encoder:
                hidden = hidden.detach()
            feats, scores_raw_i8 = gather_targets(buf, None, hidden)
            feats = feats.float()
            y_pred, v_pred = head(feats)
            loss = compute_loss(args.mode, ps, scores_raw_i8, y_pred, v_pred,
                                 corpus_mean, corpus_std, zero, scale, args.device)
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            n_accum_tokens += scores_raw_i8.shape[0]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        last_loss = accum_loss / args.grad_accum
        tokens_since_hb += n_accum_tokens
        pbar.update(1)
        pbar.set_postfix(loss=f"{last_loss:.4f}")

        now = time.time()
        if now - last_hb >= args.heartbeat_interval:
            tok_s = tokens_since_hb / max(now - last_hb, 1e-6)
            heartbeat(args.heartbeat_path, step=step, loss=last_loss,
                      median_r2=(eval_history[-1][1] if eval_history else None),
                      tok_per_s=tok_s)
            last_hb = now
            tokens_since_hb = 0

        if step % args.eval_every == 0 or step == args.max_steps:
            model.eval()
            eval_res = run_eval(val_shards, args.scores, args.climbmix_dir, gemma_tok, qwen_tok,
                                 model, head, ps, args.mode, corpus_mean, corpus_std, zero, scale,
                                 args.bsz_docs, args.max_gemma_tokens, args.max_qwen_tokens,
                                 args.min_gemma_tokens, args.eval_tokens, args.device)
            if not args.freeze_encoder:
                model.train()
            metric = eval_res["primary_metric"]
            eval_history.append((step, metric))
            log_entry = {"step": step, "loss": last_loss, **eval_res, "align_source": ALIGN_SOURCE,
                         "corpus_stats_real": corpus_stats_real}
            if args.mode == "expB-learn":
                log_entry["down_cosine"] = _down_cosine(head, ps)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"[{SCRIPT}] step={step} loss={last_loss:.4f} primary_metric={metric:.4f}")

            if metric > best_metric:
                best_metric = metric
                save_checkpoint(best_path, step, head, model, not args.freeze_encoder,
                                 optimizer, args, ps, model_name, hidden_size)
            save_checkpoint(last_path, step, head, model, not args.freeze_encoder,
                             optimizer, args, ps, model_name, hidden_size)

            # early stop: heldout primary-metric improvement < delta over the
            # trailing `early_stop_window_frac` of max_steps (DESIGN.md).
            window = args.early_stop_window_frac * args.max_steps
            if step >= window:
                ref_step = step - window
                ref_metric = None
                for s, m in eval_history:
                    if s <= ref_step:
                        ref_metric = m
                    else:
                        break
                if ref_metric is not None and (metric - ref_metric) < args.early_stop_r2_delta:
                    print(f"[{SCRIPT}] early stop at step={step}: Δmetric={metric - ref_metric:.4f} < {args.early_stop_r2_delta}")
                    stop = True
    pbar.close()
    save_checkpoint(last_path, step, head, model, not args.freeze_encoder, optimizer, args, ps, model_name, hidden_size)
    heartbeat(args.heartbeat_path, step=step, loss=last_loss,
              median_r2=(eval_history[-1][1] if eval_history else None), tok_per_s=0, done=True)
    return {"final_step": step, "eval_history": eval_history, "best_metric": best_metric}


def _down_cosine(head, ps):
    if head.down is None:
        return {}
    W = head.down.weight.detach().float().cpu().numpy()  # [2304, K]
    D = ps.D_dom  # [2304, K]
    out = {}
    for c, name in enumerate(ps.concepts):
        a, b = W[:, c], D[:, c]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        out[name] = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0
    return out


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main():
    args = build_argparser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
