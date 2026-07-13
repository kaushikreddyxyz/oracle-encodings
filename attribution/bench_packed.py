#!/usr/bin/env python3
"""Follow-up check: real corpus scoring will concatenate docs into the flat
tokens_<sid>.npy memmap anyway (DESIGN.md). This measures throughput when we
feed PACKED, no-padding 2048-token rows (attention_mask=None -> pure causal
fast path) instead of one-doc-per-row-with-padding-mask, to see whether the
explicit padding attention_mask (used in bench_gemma.py, per SPEC's literal
"padded batches" wording) is suppressing SDPA's flash/mem-efficient kernel.
"""
import json, time, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEQ_LEN = 2048
WARMUP, MEASURE = 3, 5
BATCH_SIZES = [16, 32, 64, 96, 128]

def log(*a): print(*a, flush=True)

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
pf = pq.ParquetFile("/root/.cache/huggingface/hub/datasets--karpathy--climbmix-400b-shuffle/snapshots/915333b4f8b8684f39aeaafea600fea6f43fb703/shard_00320.parquet")
ids_flat = []
for batch in pf.iter_batches(batch_size=512, columns=["text"]):
    for v in batch.column(0):
        t = v.as_py()
        if not t:
            continue
        ids_flat.extend(tok(t, add_special_tokens=True)["input_ids"])
        if len(ids_flat) > (max(BATCH_SIZES) * (WARMUP+MEASURE) + 4) * SEQ_LEN:
            break
    if len(ids_flat) > (max(BATCH_SIZES) * (WARMUP+MEASURE) + 4) * SEQ_LEN:
        break
log(f"[data] packed {len(ids_flat)} raw tokens available")
ids_flat = np.array(ids_flat, dtype=np.int64)
n_rows_avail = len(ids_flat) // SEQ_LEN
rows = ids_flat[: n_rows_avail * SEQ_LEN].reshape(n_rows_avail, SEQ_LEN)
log(f"[data] {n_rows_avail} packed rows of {SEQ_LEN} (100% non-pad)")

results = {}
for attn_impl in ["sdpa", "eager"]:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b", dtype=torch.bfloat16, attn_implementation=attn_impl
        ).to("cuda").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b", torch_dtype=torch.bfloat16, attn_implementation=attn_impl
        ).to("cuda").eval()
    log(f"[model] {attn_impl} resolved={model.config._attn_implementation}")
    results[attn_impl] = {}
    for B in BATCH_SIZES:
        need = B * (WARMUP + MEASURE)
        if need > n_rows_avail:
            results[attn_impl][str(B)] = dict(status="skip_insufficient_data")
            continue
        try:
            torch.cuda.empty_cache(); torch.cuda.synchronize()
            batch_rows = [rows[i*B:(i+1)*B] for i in range(WARMUP+MEASURE)]
            for s in range(WARMUP):
                x = torch.from_numpy(batch_rows[s]).to("cuda")
                with torch.no_grad():
                    out = model(input_ids=x, attention_mask=None,
                                output_hidden_states=True, use_cache=False)
                del out
            torch.cuda.synchronize()
            t0 = time.time()
            for s in range(WARMUP, WARMUP+MEASURE):
                x = torch.from_numpy(batch_rows[s]).to("cuda")
                with torch.no_grad():
                    out = model(input_ids=x, attention_mask=None,
                                output_hidden_states=True, use_cache=False)
                del out
            torch.cuda.synchronize()
            dt = time.time() - t0
            tot = B * MEASURE * SEQ_LEN
            tps = tot / dt
            results[attn_impl][str(B)] = dict(status="ok", tok_per_s=tps, elapsed_s=dt, total_tokens=tot)
            log(f"[packed] attn={attn_impl} B={B}: {tps:,.0f} tok/s")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results[attn_impl][str(B)] = dict(status="oom")
            log(f"[packed] attn={attn_impl} B={B}: OOM")
    del model
    torch.cuda.empty_cache()

with open("/workspace/bench_packed.json", "w") as f:
    json.dump(results, f, indent=2)
log("DONE")
