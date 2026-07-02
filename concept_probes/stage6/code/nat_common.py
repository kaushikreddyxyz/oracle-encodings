"""Shared helpers for the Stage 6 natural deployment split (§0.5, §6).

Frozen decisions (see stage5/PLAN.md):
  - corpus: karpathy/climbmix-400b-shuffle, shards >= 310 only
    (nanochat consumed ~0-183; overnight runs 300-309)
  - seed 20260702 everywhere; nat_split ("cal"/"test") by md5(example_id) parity
"""
import hashlib
import re

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET = "karpathy/climbmix-400b-shuffle"
SEED = 20260702

# characters counted as "English-ish" for the doc-level filter
_DOC_ALLOWED = re.compile(r"[A-Za-z \t.,;:'\"!?()\-\n]")
# alpha+space for the window-level filter
_WIN_ALLOWED = re.compile(r"[A-Za-z ]")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def doc_english_ratio(text):
    if not text:
        return 0.0
    return len(_DOC_ALLOWED.findall(text)) / len(text)


def window_alpha_ratio(text):
    if not text:
        return 0.0
    return len(_WIN_ALLOWED.findall(text)) / len(text)


def sentence_spans(text):
    """[(start, end)] covering text, split naively on [.!?]+whitespace."""
    spans, prev = [], 0
    for m in _SENT_SPLIT.finditer(text):
        spans.append((prev, m.start()))
        prev = m.end()
    if prev < len(text):
        spans.append((prev, len(text)))
    return [(s, e) for s, e in spans if e > s]


def nat_split_of(example_id):
    return "cal" if int(hashlib.md5(example_id.encode()).hexdigest(), 16) % 2 == 0 else "test"


def detect_text_column(pf):
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


def shard_file(shard):
    return hf_hub_download(DATASET, f"shard_{shard:05d}.parquet",
                           repo_type="dataset", token=True)


def iter_shard_docs(shard, max_docs=None):
    """Yield (doc_index, text) from one shard, in parquet order (deterministic)."""
    pf = pq.ParquetFile(shard_file(shard))
    col = detect_text_column(pf)
    n = 0
    for batch in pf.iter_batches(batch_size=512, columns=[col]):
        for v in batch.column(0):
            if max_docs is not None and n >= max_docs:
                return
            t = v.as_py()
            if t:
                yield n, t
            n += 1


def shard_num_rows(shard):
    return pq.ParquetFile(shard_file(shard)).metadata.num_rows


# --- stable 8-gram shingles / Jaccard near-dup (port of stage4 curate.py with a
#     process-independent hash) ---
NGRAM = 8
JACCARD_T = 0.6


def shingles(text):
    toks = re.findall(r"[a-z0-9']+", text.lower())
    if len(toks) < NGRAM:
        return {hashlib.md5(" ".join(toks).encode()).hexdigest()[:16]}
    return {hashlib.md5(" ".join(toks[i:i + NGRAM]).encode()).hexdigest()[:16]
            for i in range(len(toks) - NGRAM + 1)}


def is_near_dup(sh, kept_shingle_sets):
    for other in kept_shingle_sets:
        inter = len(sh & other)
        if inter and inter / len(sh | other) >= JACCARD_T:
            return True
    return False
