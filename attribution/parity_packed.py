#!/usr/bin/env python3
"""Re-run the sdpa-vs-eager probe-score parity check under the RECOMMENDED
packed (attention_mask=None) config, since sdpa only engages its flash/
mem-efficient kernel when no explicit attention_mask is passed -- the padded
parity check in bench_gemma.py measured a different numerical path (sdpa
"math" backend, triggered by the padding mask) than what would actually run
in production (packed, causal-only). Softcapping is dropped by sdpa either
way (structural: sdpa_attention_forward never receives the softcap kwarg),
but kernel-level accumulation order differs between math/mem-efficient/flash,
so this needs its own measurement.
"""
import json, os, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace")
from bench_gemma import load_probes, probe_scores, PROBE_LAYERS

SEQ_LEN = 2048
N_ROWS = 50  # ~100k tokens, matches the padded parity budget

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
pf = pq.ParquetFile("/root/.cache/huggingface/hub/datasets--karpathy--climbmix-400b-shuffle/snapshots/915333b4f8b8684f39aeaafea600fea6f43fb703/shard_00320.parquet")
ids_flat = []
for batch in pf.iter_batches(batch_size=512, columns=["text"]):
    for v in batch.column(0):
        t = v.as_py()
        if not t:
            continue
        ids_flat.extend(tok(t, add_special_tokens=True)["input_ids"])
        if len(ids_flat) >= (N_ROWS + 2) * SEQ_LEN:
            break
    if len(ids_flat) >= (N_ROWS + 2) * SEQ_LEN:
        break
ids_flat = np.array(ids_flat[:N_ROWS * SEQ_LEN], dtype=np.int64)
rows = ids_flat.reshape(N_ROWS, SEQ_LEN)
print(f"[data] {N_ROWS} packed rows x {SEQ_LEN} = {rows.size} tokens, 100% non-pad", flush=True)

probes = load_probes("/workspace/probes/months")

cache = {}
for attn_impl in ["sdpa", "eager"]:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b", dtype=torch.bfloat16, attn_implementation=attn_impl
        ).to("cuda").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b", torch_dtype=torch.bfloat16, attn_implementation=attn_impl
        ).to("cuda").eval()
    print(f"[model] {attn_impl} resolved={model.config._attn_implementation}", flush=True)
    scores = {L: {c: [] for c in probes[L]["classes"]} for L in probes}
    B = 10
    for s in range(0, N_ROWS, B):
        x = torch.from_numpy(rows[s:s + B]).to("cuda")
        with torch.no_grad():
            out = model(input_ids=x, attention_mask=None,
                        output_hidden_states=True, use_cache=False)
        for L in probes:
            h = out.hidden_states[L + 1].float()
            sc = probe_scores(h, probes[L], "cuda")  # [B,T,C]
            sc_np = sc.reshape(-1, sc.shape[-1]).cpu().numpy()
            for ci, cls in enumerate(probes[L]["classes"]):
                scores[L][cls].append(sc_np[:, ci])
        del out
    for L in scores:
        for cls in scores[L]:
            scores[L][cls] = np.concatenate(scores[L][cls])
    cache[attn_impl] = scores
    del model
    torch.cuda.empty_cache()

per_probe = {}
worst = 0.0
for L in probes:
    for cls in probes[L]["classes"]:
        a = cache["sdpa"][L][cls]
        b = cache["eager"][L][cls]
        delta = np.abs(a - b)
        std = float(np.std(a))
        p50 = float(np.percentile(delta, 50))
        p99 = float(np.percentile(delta, 99))
        ratio = p99 / std if std > 1e-8 else float("inf")
        worst = max(worst, ratio)
        per_probe[f"L{L}/{cls}"] = dict(p50=p50, p99=p99, std=std, ratio=ratio, accept=bool(ratio < 0.05))

accept_all = all(v["accept"] for v in per_probe.values())
print(f"[parity-packed] accept_all={accept_all} worst_p99_over_std={worst:.4f}", flush=True)
for k, v in sorted(per_probe.items(), key=lambda kv: -kv[1]["ratio"])[:8]:
    print(f"    {k}: p50={v['p50']:.4f} p99={v['p99']:.4f} std={v['std']:.4f} ratio={v['ratio']:.4f} accept={v['accept']}", flush=True)

with open("/workspace/parity_packed.json", "w") as f:
    json.dump(dict(accept_all=accept_all, worst_p99_over_std=worst, per_probe=per_probe,
                    n_rows=N_ROWS, seq_len=SEQ_LEN), f, indent=2)
print("DONE", flush=True)
