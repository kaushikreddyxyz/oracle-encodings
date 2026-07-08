"""Precompute per-document oracle coords over the nanochat pretraining corpus.

Long pole of Phase 4 -- START EARLY (SPEC: by ~4 AM to be worth it vs inline).

For every document in the nanochat ClimbMix shards (karpathy/climbmix-400b-shuffle,
the SAME parquet 'text' the dataloader reads), produce an (n_nanochat_tokens, r)
standardized coord array and append it to an int8 memmap keyed by doc-content
hash. The nanochat model later rides these through best-fit packing (see
coord_dataloader.py).

Bridge per token (align.py, prefix mode -- the tokenizer-agnostic module already
validated at 7.08% crossing gemma->qwen):
    nanochat-tokenize doc  -> char offsets (reconstructed from tiktoken token
                              bytes; nanochat's RustBPE/tiktoken has no
                              return_offsets_mapping, so accumulate byte lengths
                              -> byte spans -> char spans)
    qwen-tokenize doc      -> char offsets (HF fast tokenizer)
    nanochat_tok t         -> last qwen token whose char span ends <= end(t)
    gather Qwen hidden[that idx] -> encoder head -> preds[3K] -> take the
    gemma-layer-8 block (K cols) -> build_coords -> (n, r)

Run (per pod, shard range split across the fleet):
    python precompute_coords.py --encoder-ckpt <expA.pt> --probe-set <probe_set.json> \
        --shards 0-190 --out /workspace/coords --r-check 14
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # stage7 code/
from align import get_offsets, gemma_to_qwen_map            # noqa: E402
from coords_store import build_coords, make_orthonormal_P, doc_hash, CYCLIC_ORDER  # noqa: E402


def nanochat_char_offsets(tok, ids):
    """Reconstruct (start,end) CHAR spans for tiktoken/RustBPE ids by accumulating
    per-token decoded byte lengths and converting byte spans -> char spans."""
    enc = tok.enc if hasattr(tok, "enc") else tok  # tiktoken.Encoding
    byte_lens = [len(enc.decode_single_token_bytes(i)) for i in ids]
    b = np.concatenate([[0], np.cumsum(byte_lens)])          # byte offset per token boundary
    fb = enc.decode(ids).encode("utf-8")
    # byte offset -> char offset: a utf-8 continuation byte (0b10xxxxxx) does not
    # start a new char, so the char index increments only on lead bytes.
    char_at = [0]
    ci = 0
    for byte in fb:
        if (byte & 0xC0) != 0x80:
            ci += 1
        char_at.append(ci)
    char_at = np.asarray(char_at)
    return [(int(char_at[b[i]]), int(char_at[b[i + 1]])) for i in range(len(ids))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-ckpt", required=True)
    ap.add_argument("--probe-set", required=True)
    ap.add_argument("--shards", required=True, help="e.g. 0-190")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer8-block", type=int, default=8, help="gemma layer whose K predicted cols build the coords")
    ap.add_argument("--n-embd", type=int, default=1536)
    ap.add_argument("--noise-none", action="store_true", help="do NOT bake noise (added fresh at train time -- default)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ps = json.load(open(args.probe_set))
    concepts, families = ps["concepts"], ps["families"]
    # The encoder head's per-layer K output columns are in the score store's
    # MAIN-block order == main_block_concepts (see out/PERMUTATION_FIX.md), NOT
    # `concepts` (name-sorted). build_coords MUST index the encoder preds by
    # this order so each family's phase angles attach to the TRUE concept.
    # Fall back to `concepts` (with a warning) only if the key is absent.
    pred_order = ps.get("main_block_concepts")
    if pred_order is None:
        print("WARNING: probe_set.json has no 'main_block_concepts'; using "
              "name-sorted 'concepts' as the encoder pred column order. If the "
              "encoder was trained on the pre-fix (family-sorted) score store, "
              "this attaches coord phase angles to the WRONG concepts.",
              file=sys.stderr)
        pred_order = concepts
    layers = ps["layers"]
    block = layers.index(args.layer8_block)  # which of the 3 layer-blocks in preds[3K]
    K = len(concepts)

    # fixed P + save once (seed pinned)
    P = make_orthonormal_P(args.n_embd, r=None or 14, seed=1337)  # r resolved after first build
    # --- load encoder (frozen Exp-A) + qwen + nanochat tokenizers ---
    #   encoder = load_expA(args.encoder_ckpt)  # Qwen3-0.6B-Base + up head, eval, bf16
    #   qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")   # fast
    #   nano_tok = RustBPETokenizer.from_directory($NANOCHAT_BASE_DIR/tokenizer)
    #   pca = {"continents": np.load(.../continents_pca.npy)}  # saved family-PCA-2D
    #   corpus_stats = ps["corpus_stats"]  # per-col mean/std to standardize preds
    #
    # for each doc text in the shard range (nanochat.dataset.parquets_iter_batched):
    #     n_ids = nano_tok.encode(text)                       # no BOS here (BOS row added by loader)
    #     nano_off = nanochat_char_offsets(nano_tok, n_ids)
    #     q_ids, q_off = get_offsets(qwen_tok, text)
    #     amap = gemma_to_qwen_map(text, nano_off, q_off, mode="prefix")  # (len(n_ids),)
    #     H = encoder.qwen(torch.tensor([q_ids])).hidden      # (Tq, 1024)
    #     preds = encoder.head(H)                             # (Tq, 3K)  standardized
    #     preds = standardize(preds, corpus_stats)
    #     pt = preds[amap.clamp(min=0)]                       # (n, 3K); amap==-1 -> row0 masked
    #     pt[amap < 0] = 0.0
    #     kcols = pt[:, block*K:(block+1)*K]                  # gemma-layer-8 K cols
    #     # pred_order = main_block_concepts: encoder-output column order (store MAIN block)
    #     z, legend = build_coords(kcols.numpy(), concepts, families, pca=pca, pred_order=pred_order)  # (n, r)
    #     if not args.noise_none: z += ...                    # (default: leave noise to train time)
    #     append z (int8, scale=4*std/127) to coords.int8; record (doc_hash(text), off, n)
    #
    # write index.npy (hash,off,n), meta.json {r, scale, layer8_block, legend,
    #   families, class_order=CYCLIC_ORDER, encoder_ckpt, P_path}, P.npy
    raise SystemExit("SKELETON: wire encoder/tokenizers/pca per the commented block "
                     "once the Exp-A checkpoint exists; structure + bridge are final.")


if __name__ == "__main__":
    main()
