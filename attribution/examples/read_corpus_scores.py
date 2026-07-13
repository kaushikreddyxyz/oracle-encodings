"""Read the pre-scored ClimbMix concept-score store without an 8 GB download.

The attribution corpus scan (formerly concept_probes/5_oracle) applied the 54
gold detection probes (see concept_probes/examples/score_text_with_probes.py)
to ClimbMix and published one file set per shard:

  scores_<sid>.npy  int8 [n_tokens, 3, 54]
                    axis 0: token (aligns 1:1 with tokens/docs)
                    axis 1: layer — 0 -> L6, 1 -> L8, 2 -> L14
                    axis 2: concept — index into columns.json concepts[]
  tokens_<sid>.npy  int32 [n_tokens]  BOS-free gemma-2-2b token ids
  docs_<sid>.jsonl  {"doc", "start", "n"} spans into the two arrays

Repos (HF datasets, public, identical format + metadata files):
  kaushikreddyxyz/corpus-scores            shards 320-355
  kaushikreddyxyz/corpus-scores-overflow   shards 356-362
  kaushikreddyxyz/climbmix-scored(+-overflow-N)  shards 0-184

Scores are int8-quantized: raw = int8 * scale[l][c] + zero[l][c]
(quant.json), CLIPPED at +/-127 ~= +/-4 sigma — extreme firings saturate.
Standardize raw scores to z with corpus_stats.json mean/std [3][54].

A full shard is ~7.5 GB, so by default this script reads only the npy
header plus the first --max-rows rows via HTTP Range requests (~81 MB at
the 500k default; the npy header is tiny and rows are contiguous, 162 B
each for scores). --max-rows 0 downloads the full shard instead.

Demo: dequantize + standardize one (concept, layer) column, print the
top-k scoring tokens with +/-N tokens of decoded context. Decoding uses
the gemma-2-2b tokenizer, which is GATED on HF (set HF_TOKEN).

Expected runtime: ~1 min at defaults (download-bound). Deps: numpy,
requests, huggingface_hub, transformers (tokenizer only — no torch model).

Usage:
  HF_TOKEN=... python read_corpus_scores.py
  python read_corpus_scores.py --concept africa --layer 14 --max-rows 2000000
"""
from __future__ import annotations

import argparse
import ast
import json

import numpy as np
import requests
from huggingface_hub import hf_hub_download, hf_hub_url

DEFAULT_REPO = "kaushikreddyxyz/corpus-scores"
GEMMA_MODEL = "google/gemma-2-2b"


# --------------------------------------------------------------------------
# Ranged .npy reads — fetch the header + first N rows of a huge HF-hosted
# array without downloading the file. Works because npy data is row-major
# and contiguous after a small self-describing header.
# --------------------------------------------------------------------------

def http_range(url: str, start: int, end: int) -> bytes:
    """GET bytes [start, end] (inclusive). Public repos only: requests drops
    the Authorization header on the redirect to the CDN host anyway."""
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                     timeout=(10, 300))
    r.raise_for_status()
    data = r.content
    assert len(data) == end - start + 1, \
        f"server ignored Range header ({len(data)} bytes for {url})"
    return data


def npy_header_ranged(url: str) -> tuple[np.dtype, tuple[int, ...], int]:
    """Parse a remote .npy header -> (dtype, shape, data_offset)."""
    head = http_range(url, 0, 11)
    assert head[:6] == b"\x93NUMPY", f"not an npy file: {url}"
    major = head[6]
    if major == 1:  # header length is uint16 (v1.0) or uint32 (v2/3)
        hlen, off = int.from_bytes(head[8:10], "little"), 10
    else:
        hlen, off = int.from_bytes(head[8:12], "little"), 12
    header = ast.literal_eval(http_range(url, off, off + hlen - 1).decode("latin1"))
    assert not header["fortran_order"], "ranged row reads need C order"
    return np.dtype(header["descr"]), header["shape"], off + hlen


def read_npy_rows_ranged(url: str, n_rows: int) -> tuple[np.ndarray, int]:
    """Fetch the first n_rows rows of a remote .npy -> (array, total_rows)."""
    dtype, shape, data_off = npy_header_ranged(url)
    n_rows = min(n_rows, shape[0])
    row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
    buf = http_range(url, data_off, data_off + n_rows * row_bytes - 1)
    return np.frombuffer(buf, dtype=dtype).reshape((n_rows,) + shape[1:]), shape[0]


# --------------------------------------------------------------------------
# Store loading
# --------------------------------------------------------------------------

def load_metadata(repo: str, shard: int) -> dict:
    """Small files via hf_hub_download: columns/quant/stats + the doc spans."""
    meta = {}
    for name in ("columns.json", "quant.json", "corpus_stats.json"):
        with open(hf_hub_download(repo, name, repo_type="dataset")) as f:
            meta[name.split(".")[0]] = json.load(f)
    docs = []
    with open(hf_hub_download(repo, f"docs_{shard:05d}.jsonl",
                              repo_type="dataset")) as f:
        for line in f:
            docs.append(json.loads(line))
    meta["docs"] = docs
    return meta


def load_shard_arrays(repo: str, shard: int, max_rows: int):
    """Return (scores int8 [n,3,54], tokens int32 [n], total_rows)."""
    names = (f"scores_{shard:05d}.npy", f"tokens_{shard:05d}.npy")
    if max_rows <= 0:  # full shard: ~7.5 GB scores + ~185 MB tokens
        scores = np.load(hf_hub_download(repo, names[0], repo_type="dataset"),
                         mmap_mode="r")
        tokens = np.load(hf_hub_download(repo, names[1], repo_type="dataset"),
                         mmap_mode="r")
        return scores, tokens, scores.shape[0]
    scores, total = read_npy_rows_ranged(
        hf_hub_url(repo, names[0], repo_type="dataset"), max_rows)
    tokens, total_t = read_npy_rows_ranged(
        hf_hub_url(repo, names[1], repo_type="dataset"), max_rows)
    assert total == total_t, "scores/tokens row counts disagree"
    return scores, tokens, total


# --------------------------------------------------------------------------
# Demo: top-k tokens for one concept, with decoded context
# --------------------------------------------------------------------------

def top_hits_nonoverlapping(z: np.ndarray, k: int, min_gap: int) -> list[int]:
    """Greedy top-k positions, no two within min_gap (dedupe hot passages)."""
    hits: list[int] = []
    for pos in np.argsort(z)[::-1]:
        if all(abs(int(pos) - h) > min_gap for h in hits):
            hits.append(int(pos))
            if len(hits) == k:
                break
    return hits


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="HF dataset repo holding the shard")
    ap.add_argument("--shard", type=int, default=320,
                    help="shard id (320-355 in corpus-scores)")
    ap.add_argument("--max-rows", type=int, default=500_000,
                    help="rows to fetch via HTTP Range (~162 B/row); "
                         "0 = download the full ~7.5 GB shard")
    ap.add_argument("--concept", default="winter",
                    help="concept name from columns.json")
    ap.add_argument("--layer", type=int, default=8, choices=[6, 8, 14],
                    help="which of the 3 stored probe layers to read")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--context", type=int, default=10,
                    help="tokens of context to decode on each side")
    args = ap.parse_args()

    meta = load_metadata(args.repo, args.shard)
    concepts = meta["columns"]["concepts"]
    li = meta["columns"]["layers"].index(args.layer)   # axis-1 index
    ci = concepts.index(args.concept)                  # axis-2 index
    family = meta["columns"]["families"][ci]

    scores, tokens, total = load_shard_arrays(args.repo, args.shard, args.max_rows)
    n = scores.shape[0]
    print(f"[store] shard {args.shard:05d}: {total:,} tokens; "
          f"read {n:,} rows ({scores.nbytes / 1e6:.0f} MB of scores)")
    print(f"[store] '{args.concept}' (family {family}) at L{args.layer} "
          f"-> scores[:, {li}, {ci}]")

    # int8 -> raw probe score -> corpus z-score. quant zero/scale and
    # corpus_stats mean/std are both [layer][concept] (same axes as the store).
    q8 = scores[:n, li, ci]
    raw = q8.astype(np.float32) * meta["quant"]["scale"][li][ci] \
        + meta["quant"]["zero"][li][ci]
    z = (raw - meta["corpus_stats"]["mean"][li][ci]) \
        / meta["corpus_stats"]["std"][li][ci]

    # +/-4 sigma clip caveat: int8 saturates at +/-127, so the very strongest
    # firings are floored to ~4 sigma — top-of-list z values are lower bounds.
    saturated = np.mean(np.abs(q8) == 127)
    print(f"[store] z: mean {z.mean():+.3f}, p99 {np.percentile(z, 99):+.2f}, "
          f"max {z.max():+.2f}; {saturated:.4%} of this column saturated")

    # docs.jsonl spans -> position -> document (context must not cross docs)
    starts = np.array([d["start"] for d in meta["docs"]])
    from transformers import AutoTokenizer  # tokenizer only, no model
    tok = AutoTokenizer.from_pretrained(GEMMA_MODEL)

    print(f"\ntop-{args.top_k} '{args.concept}' tokens in the first {n:,} rows:")
    for pos in top_hits_nonoverlapping(z, args.top_k, args.context):
        d = meta["docs"][int(np.searchsorted(starts, pos, "right")) - 1]
        lo = max(d["start"], pos - args.context)
        hi = min(d["start"] + d["n"], pos + args.context + 1, n)
        left = tok.decode(tokens[lo:pos])
        piece = tok.decode([int(tokens[pos])])
        right = tok.decode(tokens[pos + 1:hi])
        ctx = f"{left}[[{piece}]]{right}".replace("\n", " ")
        print(f"  z={z[pos]:+6.2f}  pos={pos:>9,}  doc={d['doc']:>6}  ...{ctx}...")


if __name__ == "__main__":
    main()
