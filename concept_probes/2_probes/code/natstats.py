"""Per-layer standardization stats (§0.6) from the NATURAL deployment sample.

Streams the standardization_sample.jsonl passages through gemma-2-2b and
accumulates float64 mean/std per layer over all real (non-pad, non-BOS) tokens.

  python natstats.py --passages .../standardization_sample.jsonl \
      --layers 0..25 --out natstats.npz [--max-tokens 3000000]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from extract import load_model, iter_batches


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=",".join(map(str, range(26))))
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=3_000_000)
    ap.add_argument("--batch-padded-tokens", type=int, default=32768)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")] if "," in args.layers else list(range(26))

    tok, model = load_model(args.model)
    d = model.config.hidden_size
    items = []
    n_tok = 0
    with open(args.passages) as f:
        for line in f:
            r = json.loads(line)
            ids = tok(r["text"], add_special_tokens=False)["input_ids"][: args.max_seq]
            if len(ids) >= 8:
                items.append((r.get("doc_id") or r.get("example_id"), ids))
                n_tok += len(ids)
            if n_tok >= args.max_tokens:
                break
    print(f"[natstats] {len(items)} passages, {n_tok} tokens")

    s1 = {L: torch.zeros(d, dtype=torch.float64, device="cuda") for L in layers}
    s2 = {L: torch.zeros(d, dtype=torch.float64, device="cuda") for L in layers}
    count = 0
    bos = tok.bos_token_id
    for batch in iter_batches(sorted(items, key=lambda kv: len(kv[1])), args.batch_padded_tokens):
        L_max = max(len(ids) for _, ids in batch) + 1
        input_ids = torch.full((len(batch), L_max), tok.pad_token_id or 0, dtype=torch.long)
        attn = torch.zeros((len(batch), L_max), dtype=torch.long)
        for i, (_, ids) in enumerate(batch):
            input_ids[i, 0] = bos
            input_ids[i, 1:1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[i, :1 + len(ids)] = 1
        out = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(),
                    output_hidden_states=True)
        real = attn.cuda().bool()
        real[:, 0] = False                      # exclude BOS
        for L in layers:
            h = out.hidden_states[L + 1][real].to(torch.float64)   # [n_real, d]
            s1[L] += h.sum(0)
            s2[L] += (h * h).sum(0)
        count += int(real.sum())
        del out

    save = {}
    for L in layers:
        mean = (s1[L] / count).cpu().numpy()
        var = (s2[L] / count).cpu().numpy() - mean ** 2
        save[f"mean_{L}"] = mean.astype(np.float32)
        save[f"std_{L}"] = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    np.savez(args.out, **save, n_tokens=np.int64(count))
    print(f"[natstats] wrote {args.out} over {count} tokens")


if __name__ == "__main__":
    main()
