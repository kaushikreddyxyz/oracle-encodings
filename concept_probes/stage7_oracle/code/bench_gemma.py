#!/usr/bin/env python3
"""Stage 7-Oracle Phase 1 gate: gemma-2-2b throughput (sdpa vs eager) + probe-score
parity between attn implementations, on REAL ClimbMix docs.

SPEC.md Phase 1 amended bullet:
  (a) benchmark gemma-2-2b forward-only bf16 at seq 2048 / large batch with
      attn_implementation="sdpa" vs "eager" on one H100
  (b) parity check: probe scores under sdpa vs eager on ~100k tokens; accept
      sdpa if per-probe score deltas << probe noise (|Delta| p99 < 0.05*std)

Design notes (this implementation):
  - "seq 2048, padded batches" = ONE real ClimbMix doc per row, right-padded
    (or truncated) to exactly 2048 gemma tokens; attention_mask marks real
    tokens. tok/s is computed over non-pad tokens only. This is the literal,
    honest (if pessimistic) number: ClimbMix docs average ~550-600 gemma
    tokens, so a large fraction of a 2048-wide row is padding. That padding
    fraction itself is a first-class finding (see report) -- if it dominates,
    the real Phase-1 pipeline should PACK docs instead of padding them, and
    projected tok/s should be scaled accordingly.
  - parity set = 50 docs x 2048 padded positions ~= 100k position budget
    (~50 * mean_doc_len actual non-pad tokens), same doc order/tokenization
    fed through both attn implementations for an apples-to-apples diff.

Usage (on the pod):
  python bench_gemma.py --shard /root/.cache/.../shard_00320.parquet \
      --probes-dir /workspace/probes/months --model google/gemma-2-2b \
      --out /workspace/bench_out.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEQ_LEN = 2048
PROBE_LAYERS = [8, 12, 16]          # probe layer l reads hidden_states[l+1]
BATCH_SIZES = [8, 16, 32, 64]
WARMUP_STEPS = 3
MEASURE_STEPS = 5
MIN_DOC_TOKENS = 64
PARITY_N_DOCS = 50

HB_PATH = "/workspace/hb_bench_gemma.txt"


def hb(msg):
    try:
        with open(HB_PATH, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass


def log(*a):
    print(*a, flush=True)
    hb(" ".join(str(x) for x in a))


# --------------------------------------------------------------------------- data
def load_shard_docs(path, tokenizer, min_docs, min_tokens=MIN_DOC_TOKENS):
    """[input_ids] (each <=SEQ_LEN, truncated), real ClimbMix text, in shard order."""
    pf = pq.ParquetFile(path)
    docs = []
    for batch in pf.iter_batches(batch_size=256, columns=["text"]):
        for v in batch.column(0):
            t = v.as_py()
            if not t:
                continue
            ids = tokenizer(t, add_special_tokens=True)["input_ids"]
            if len(ids) < min_tokens:
                continue
            docs.append(ids[:SEQ_LEN])
            if len(docs) >= min_docs:
                log(f"[data] collected {len(docs)} docs from {os.path.basename(path)}")
                return docs
    return docs


def make_batch(docs, idx_list, pad_id, device):
    B = len(idx_list)
    input_ids = torch.full((B, SEQ_LEN), pad_id, dtype=torch.long)
    attn = torch.zeros((B, SEQ_LEN), dtype=torch.long)
    for i, di in enumerate(idx_list):
        ids = docs[di]
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attn[i, :n] = 1
    return input_ids.to(device), attn.to(device)


# ------------------------------------------------------------------------- probes
def load_probes(probes_dir, layers=PROBE_LAYERS):
    """probes_l{L}.npz has W_ridge [n_lam,C,d], b_ridge [n_lam,C],
    chosen_lambda_ridge [C] (per-class best lambda index -- resolves the
    extra leading axis, per train.py:250-264). score = W.((h-mu)/sd)+b."""
    probes = {}
    for L in layers:
        d = np.load(os.path.join(probes_dir, f"probes_l{L}.npz"))
        classes = [c.decode() if isinstance(c, bytes) else str(c) for c in d["classes"]]
        chosen = d["chosen_lambda_ridge"]
        W_ridge, b_ridge = d["W_ridge"], d["b_ridge"]
        C = len(classes)
        W = np.stack([W_ridge[chosen[c], c] for c in range(C)]).astype(np.float32)  # [C,d]
        b = np.array([b_ridge[chosen[c], c] for c in range(C)], dtype=np.float32)   # [C]
        probes[L] = dict(
            classes=classes,
            W=torch.from_numpy(W),
            b=torch.from_numpy(b),
            mean=torch.from_numpy(d["nat_mean"].astype(np.float32)),
            std=torch.from_numpy(d["nat_std"].astype(np.float32)),
        )
        log(f"[probes] layer {L}: {C} classes ({', '.join(classes[:4])}...)")
    return probes


def probe_scores(hidden, probe, device):
    # hidden: [B,T,d] float32 on `device`
    mean = probe["mean"].to(device)
    std = probe["std"].to(device)
    W = probe["W"].to(device)
    b = probe["b"].to(device)
    z = (hidden - mean) / std
    return z @ W.T + b  # [B,T,C]


# ---------------------------------------------------------------------- throughput
def load_model(model_name, attn_impl, device):
    log(f"[model] loading {model_name} attn_implementation={attn_impl} bf16")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, attn_implementation=attn_impl
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation=attn_impl
        )
    model.to(device).eval()
    # report what the model actually resolved to (gemma2 silently falls back
    # to eager for sdpa when attn_logit_softcapping is set, in some versions)
    real_impl = getattr(model.config, "_attn_implementation", "?")
    log(f"[model] resolved _attn_implementation={real_impl} "
        f"(requested {attn_impl}); attn_logit_softcapping="
        f"{getattr(model.config, 'attn_logit_softcapping', None)}")
    return model, real_impl


def bench_throughput(model, docs, pad_id, device, attn_impl, hs_flag, batch_sizes):
    results = {}
    for B in batch_sizes:
        try:
            n_needed = B * (WARMUP_STEPS + MEASURE_STEPS)
            idxs = [i % len(docs) for i in range(n_needed)]
            batches = [idxs[i * B:(i + 1) * B] for i in range(WARMUP_STEPS + MEASURE_STEPS)]
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            for s in range(WARMUP_STEPS):
                input_ids, attn = make_batch(docs, batches[s], pad_id, device)
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn,
                                output_hidden_states=hs_flag, use_cache=False)
                del out
            torch.cuda.synchronize()
            t0 = time.time()
            total_tok = 0
            for s in range(WARMUP_STEPS, WARMUP_STEPS + MEASURE_STEPS):
                input_ids, attn = make_batch(docs, batches[s], pad_id, device)
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn,
                                output_hidden_states=hs_flag, use_cache=False)
                total_tok += int(attn.sum().item())
                del out
            torch.cuda.synchronize()
            dt = time.time() - t0
            toks_per_s = total_tok / dt
            pad_frac = 1.0 - total_tok / (B * MEASURE_STEPS * SEQ_LEN)
            results[str(B)] = dict(status="ok", tok_per_s=toks_per_s, elapsed_s=dt,
                                    total_nonpad_tokens=total_tok, pad_frac=pad_frac)
            log(f"[throughput] attn={attn_impl} hs={hs_flag} B={B}: "
                f"{toks_per_s:,.0f} tok/s (pad_frac={pad_frac:.2f})")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results[str(B)] = dict(status="oom")
            log(f"[throughput] attn={attn_impl} hs={hs_flag} B={B}: OOM")
        except Exception as e:
            torch.cuda.empty_cache()
            results[str(B)] = dict(status="error", error=str(e))
            log(f"[throughput] attn={attn_impl} hs={hs_flag} B={B}: ERROR {e}")
    return results


# -------------------------------------------------------------------------- parity
def run_parity_pass(model, docs, pad_id, device, probes, batch_size=10):
    """Returns {layer: {class: np.ndarray [n_nonpad_tokens]}} scores, concatenated
    across the parity doc set, non-pad positions only, doc order preserved."""
    n = len(docs)
    out_scores = {L: {c: [] for c in probes[L]["classes"]} for L in probes}
    for s in range(0, n, batch_size):
        idx_list = list(range(s, min(s + batch_size, n)))
        input_ids, attn = make_batch(docs, idx_list, pad_id, device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=True, use_cache=False)
        mask = attn.bool()  # [B,T]
        for L in probes:
            h = out.hidden_states[L + 1].float()  # [B,T,d]
            sc = probe_scores(h, probes[L], device)  # [B,T,C]
            sc_masked = sc[mask]  # [n_nonpad_in_batch, C]
            sc_np = sc_masked.cpu().numpy()
            for ci, cls in enumerate(probes[L]["classes"]):
                out_scores[L][cls].append(sc_np[:, ci])
        del out
    for L in out_scores:
        for cls in out_scores[L]:
            out_scores[L][cls] = np.concatenate(out_scores[L][cls])
    return out_scores


def compare_parity(scores_a, scores_b, probes):
    """scores_a/scores_b: {layer:{class:[N]}} aligned position-for-position."""
    per_probe = {}
    worst_ratio = 0.0
    for L in probes:
        for cls in probes[L]["classes"]:
            a = scores_a[L][cls]
            b = scores_b[L][cls]
            assert a.shape == b.shape, (L, cls, a.shape, b.shape)
            delta = np.abs(a - b)
            std = float(np.std(a))  # std of the "reference" (sdpa) score distribution
            p50 = float(np.percentile(delta, 50))
            p99 = float(np.percentile(delta, 99))
            ratio = p99 / std if std > 1e-8 else float("inf")
            worst_ratio = max(worst_ratio, ratio)
            per_probe[f"L{L}/{cls}"] = dict(p50=p50, p99=p99, std=std, p99_over_std=ratio,
                                             accept=bool(ratio < 0.05))
    accept_all = all(v["accept"] for v in per_probe.values())
    return dict(per_probe=per_probe, accept_all=accept_all, worst_p99_over_std=worst_ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--probes-dir", required=True)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--out", default="/workspace/bench_out.json")
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--batch-sizes", default=",".join(str(b) for b in BATCH_SIZES))
    args = ap.parse_args()

    open(HB_PATH, "a").close()
    device = "cuda"
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    log("=== Stage7 bench_gemma.py starting ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    log(f"[tok] pad_token_id={pad_id}")

    n_docs_needed = max(BATCH_SIZES) * (WARMUP_STEPS + MEASURE_STEPS)
    docs_all = load_shard_docs(args.shard, tokenizer, min_docs=max(n_docs_needed, 500))
    doc_lens = [len(d) for d in docs_all]
    log(f"[data] {len(docs_all)} docs, mean_len={np.mean(doc_lens):.1f} "
        f"median_len={np.median(doc_lens):.1f} max_len={max(doc_lens)}")

    parity_docs = docs_all[:PARITY_N_DOCS]
    parity_nonpad = sum(len(d) for d in parity_docs)
    log(f"[parity] {len(parity_docs)} docs, {parity_nonpad} non-pad tokens "
        f"(of {len(parity_docs) * SEQ_LEN} padded positions)")

    probes = load_probes(args.probes_dir)

    results = dict(
        meta=dict(model=args.model, seq_len=SEQ_LEN, batch_sizes=batch_sizes,
                   warmup_steps=WARMUP_STEPS, measure_steps=MEASURE_STEPS,
                   shard=os.path.basename(args.shard), n_docs_loaded=len(docs_all),
                   doc_len_mean=float(np.mean(doc_lens)),
                   doc_len_median=float(np.median(doc_lens)),
                   parity_n_docs=len(parity_docs), parity_nonpad_tokens=parity_nonpad,
                   torch_version=torch.__version__,
                   gpu=torch.cuda.get_device_name(0)),
        throughput={}, parity={}, hs_overhead={},
    )

    parity_score_cache = {}  # attn_impl -> {layer:{class:[N]}}

    for attn_impl in ["sdpa", "eager"]:
        model, real_impl = load_model(args.model, attn_impl, device)
        results["throughput"].setdefault(attn_impl, {})["resolved_impl"] = real_impl

        if not args.skip_throughput:
            log(f"--- throughput sweep: attn={attn_impl} (hidden_states=True) ---")
            tp = bench_throughput(model, docs_all, pad_id, device, attn_impl,
                                   hs_flag=True, batch_sizes=batch_sizes)
            results["throughput"][attn_impl]["hs_true"] = tp

            # (c) hidden_states True vs False at the best OK batch size
            ok_bs = [int(b) for b, r in tp.items() if r.get("status") == "ok"]
            if ok_bs:
                best_b = max(ok_bs)
                log(f"--- hs True vs False overhead at B={best_b}, attn={attn_impl} ---")
                tp_no_hs = bench_throughput(model, docs_all, pad_id, device, attn_impl,
                                             hs_flag=False, batch_sizes=[best_b])
                results["hs_overhead"][attn_impl] = dict(
                    best_batch=best_b,
                    hs_true=tp.get(str(best_b)),
                    hs_false=tp_no_hs.get(str(best_b)),
                )

        if not args.skip_parity:
            log(f"--- parity pass: attn={attn_impl} on {len(parity_docs)} docs ---")
            parity_score_cache[attn_impl] = run_parity_pass(
                model, parity_docs, pad_id, device, probes)

        del model
        torch.cuda.empty_cache()

    if not args.skip_parity and "sdpa" in parity_score_cache and "eager" in parity_score_cache:
        log("--- comparing parity: sdpa vs eager ---")
        cmp = compare_parity(parity_score_cache["sdpa"], parity_score_cache["eager"], probes)
        results["parity"] = cmp
        log(f"[parity] accept_all={cmp['accept_all']} "
            f"worst_p99_over_std={cmp['worst_p99_over_std']:.4f}")
        for k, v in sorted(cmp["per_probe"].items(), key=lambda kv: -kv[1]["p99_over_std"])[:8]:
            log(f"    {k}: p50={v['p50']:.4f} p99={v['p99']:.4f} std={v['std']:.4f} "
                f"ratio={v['p99_over_std']:.4f} accept={v['accept']}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"=== DONE, wrote {args.out} ===")


if __name__ == "__main__":
    main()
