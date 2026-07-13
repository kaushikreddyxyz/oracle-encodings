"""Cache gemma-2-2b residual-stream activations for a family's unique examples.

One forward pass per unique example (BOS prepended, dropped from the cache so
cache row positions == Stage-4 token_ids positions). Writes one fp16 memmap per
layer: <out>/acts_l{L}.npy of shape [total_tokens, 2304], plus index.json
mapping example_id -> [offset, n_tokens].

Usage:
  python extract.py --family months --stage4 <1_dataset/data> --out <cache/months> \
      [--layers 1,3,...] [--natural <passages.jsonl>]

--natural mode caches text passages instead (tokenized here, truncated to
--max-seq), writing the same layout keyed by example_id.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import FamilyData

DEF_LAYERS = "1,3,6,8,10,12,14,16,18,20,23,25"


def load_model(model_name: str):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                      attn_implementation="eager")
    model.eval().cuda()
    return tok, model


def iter_batches(items, batch_padded_tokens: int):
    """items: list of (key, token_ids) sorted by length; yield padded batches."""
    batch = []
    maxlen = 0
    for key, ids in items:
        newmax = max(maxlen, len(ids))
        if batch and newmax * (len(batch) + 1) > batch_padded_tokens:
            yield batch
            batch, maxlen = [], 0
            newmax = len(ids)
        batch.append((key, ids))
        maxlen = newmax
    if batch:
        yield batch


@torch.no_grad()
def run(items, layers, out_dir: Path, model_name: str, batch_padded_tokens: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    tok, model = load_model(model_name)
    bos = tok.bos_token_id
    d = model.config.hidden_size

    order = sorted(items, key=lambda kv: len(kv[1]))
    offsets, off = {}, 0
    for key, ids in sorted(items, key=lambda kv: kv[0]):
        offsets[key] = [off, len(ids)]
        off += len(ids)
    total = off
    memmaps = {L: np.lib.format.open_memmap(out_dir / f"acts_l{L}.npy", mode="w+",
                                            dtype=np.float16, shape=(total, d))
               for L in layers}

    t0, done_tok, done_ex = time.time(), 0, 0
    for batch in iter_batches(order, batch_padded_tokens):
        lens = [len(ids) for _, ids in batch]
        L_max = max(lens) + 1
        input_ids = torch.full((len(batch), L_max), tok.pad_token_id or 0, dtype=torch.long)
        attn = torch.zeros((len(batch), L_max), dtype=torch.long)
        for i, (_, ids) in enumerate(batch):
            input_ids[i, 0] = bos
            input_ids[i, 1:1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[i, :1 + len(ids)] = 1
        out = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(),
                    output_hidden_states=True)
        # hidden_states[l+1] == post-block-l residual; drop the BOS row
        for L in layers:
            hs = out.hidden_states[L + 1].to(torch.float16).cpu().numpy()
            for i, (key, ids) in enumerate(batch):
                o, n = offsets[key]
                memmaps[L][o:o + n] = hs[i, 1:1 + n]
        del out
        done_tok += sum(lens); done_ex += len(batch)
        if done_ex % 2000 < len(batch):
            print(f"[extract] {done_ex}/{len(items)} examples, "
                  f"{done_tok / max(time.time() - t0, 1e-9):.0f} tok/s", flush=True)

    for mm in memmaps.values():
        mm.flush()
    with open(out_dir / "index.json", "w") as f:
        json.dump({"offsets": offsets, "layers": list(layers), "total_tokens": total,
                   "model": model_name, "prepend_bos": True}, f)
    print(f"[extract] done: {total} tokens x {len(layers)} layers -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family")
    ap.add_argument("--stage4", default="concept_probes/1_dataset/data")
    ap.add_argument("--natural", help="jsonl of {example_id|doc_id, text} passages")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=DEF_LAYERS)
    ap.add_argument("--classes", help="comma subset of the family's classes (pilot)")
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--batch-padded-tokens", type=int, default=32768)
    ap.add_argument("--max-seq", type=int, default=512)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    if args.natural:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        items = []
        with open(args.natural) as f:
            for line in f:
                r = json.loads(line)
                key = r.get("example_id") or r["doc_id"]
                ids = tok(r["text"], add_special_tokens=False)["input_ids"][: args.max_seq]
                if ids:
                    items.append((key, ids))
        print(f"[extract] natural mode: {len(items)} passages")
    else:
        fam = FamilyData(Path(args.stage4) / args.family / "final",
                         args.classes.split(",") if args.classes else None)
        items = [(eid, fam.tokens[eid]) for eid in fam.example_ids]
        print(f"[extract] family {args.family}: {len(items)} unique examples, "
              f"{fam.total_tokens} tokens")
    run(items, layers, Path(args.out), args.model, args.batch_padded_tokens)


if __name__ == "__main__":
    main()
