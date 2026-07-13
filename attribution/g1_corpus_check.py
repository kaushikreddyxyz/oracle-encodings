"""G1 PART 2 (pod-side, CPU/numpy only): corpus-side score stats from the
Phase-1 int8 score store, over the fully-present shards under /workspace/scores.

- exact quantiles per column via a 255-bin histogram of the int8 codes
  (dequantized through the affine quant map at the end) -- avoids ever
  holding a giant float array.
- streaming mean/std per column (also from the int8 histogram, exact).
- top-100 firing token ids for 5 spot-check concepts' layer-8 columns.
- january-vs-march (layer-8) Pearson correlation over the sampled tokens.

Memory: chunked reads (memmap slices), never loads a whole shard. CPU-only.
Does NOT touch the GPU, the rsync pull loop, or any training files.
"""
import json
import os
import sys
import time
from collections import Counter
import numpy as np

SCORES_DIR = "/workspace/scores"
PROBE_SET_PATH = "/workspace/stage7/probe_set.json"
QUANT_PATH = "/workspace/scores/quant.json"
OUT_PATH = "/workspace/scores/g1_corpus_stats.json"
GEMMA_TOK_PATH = "/workspace/models/gemma-2-2b-tokenizer"

N_COLS = 216
TARGET_TOTAL_TOKENS = 64_000_000  # >= 50M required
BLOCKS_PER_SHARD = 4  # spread offsets to avoid doc-order bias
TOPK_FINAL = 100
TOPK_CANDIDATE_PER_CHUNK = 400  # generous margin before final trim

SPOT_CHECK_CONCEPTS = ["january", "red", "north", "full_moon", "europe"]
SPOT_CHECK_LAYER = 8

QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
QNAMES = ["p1", "p5", "p25", "p50", "p75", "p95", "p99"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_complete_shards():
    """A shard is 'complete' if non-hidden tokens_/scores_/docs_ files exist
    and scores_<sid>.npy's byte size is an exact multiple of N_COLS matching
    tokens_<sid>.npy's element count (i.e. rsync fully landed it, not the
    in-progress dotfile temp)."""
    files = os.listdir(SCORES_DIR)
    sids = sorted({f[len("tokens_"):-4] for f in files
                   if f.startswith("tokens_") and f.endswith(".npy")})
    complete = []
    for sid in sids:
        tok_p = os.path.join(SCORES_DIR, f"tokens_{sid}.npy")
        sc_p = os.path.join(SCORES_DIR, f"scores_{sid}.npy")
        doc_p = os.path.join(SCORES_DIR, f"docs_{sid}.jsonl")
        if not (os.path.exists(tok_p) and os.path.exists(sc_p) and os.path.exists(doc_p)):
            continue
        # .npy header is 128 bytes typically for these dumps; compute n rows
        # robustly via np.load mmap header parse instead of assuming offset.
        try:
            tok_arr = np.load(tok_p, mmap_mode="r")
            sc_arr = np.load(sc_p, mmap_mode="r")
        except Exception as e:
            log(f"shard {sid}: failed to mmap ({e}), skipping")
            continue
        if sc_arr.ndim != 2 or sc_arr.shape[1] != N_COLS:
            log(f"shard {sid}: unexpected scores shape {sc_arr.shape}, skipping")
            continue
        if sc_arr.shape[0] != tok_arr.shape[0]:
            log(f"shard {sid}: row mismatch tokens={tok_arr.shape[0]} scores={sc_arr.shape[0]}, "
                f"likely still transferring, skipping")
            continue
        complete.append((sid, tok_arr.shape[0]))
    return complete


def main():
    t0 = time.time()
    ps = json.loads(open(PROBE_SET_PATH).read())
    concepts = ps["concepts"]
    layers = ps["layers"]  # [6,8,14]
    ablation_layer = ps["ablation_layer"]
    K = len(concepts)
    assert 4 * K == N_COLS, (K, N_COLS)
    # Block-order contract (out/PERMUTATION_FIX.md): the score store's MAIN
    # block columns (l*K+ci) are in main_block_concepts order (family-sorted
    # in the current immutable store), NOT `concepts` (name-sorted). The DOM
    # block (3K+ci) is in concepts order. Using `concepts` for the main block
    # here was the exact indexing half of the permutation bug that made the
    # spot checks read the WRONG concept's column. Fall back to `concepts` if
    # the key is absent (pre-fix probe_set.json), which reproduces the bug --
    # so warn.
    main_block_concepts = ps.get("main_block_concepts")
    if main_block_concepts is None:
        print("WARNING: probe_set.json has no 'main_block_concepts'; main-block "
              "columns will be indexed by name-sorted 'concepts' -- this is the "
              "permutation bug. Regenerate probe_set.json with the fix.",
              file=sys.stderr)
        main_block_concepts = concepts
    main_idx = {c: i for i, c in enumerate(main_block_concepts)}

    quant = json.loads(open(QUANT_PATH).read())
    zero = np.array(quant["zero"], dtype=np.float64)
    scale = np.array(quant["scale"], dtype=np.float64)
    assert len(zero) == N_COLS and len(scale) == N_COLS

    def col_index(concept, layer, dom=False):
        if dom:
            return 3 * K + concepts.index(concept)      # dom block: name-sorted
        li = layers.index(layer)
        return li * K + main_idx[concept]               # main block: main_block order

    spot_cols = {c: col_index(c, SPOT_CHECK_LAYER) for c in SPOT_CHECK_CONCEPTS}
    jan_col = col_index("january", SPOT_CHECK_LAYER)
    mar_col = col_index("march", SPOT_CHECK_LAYER)
    log(f"spot check cols (layer {SPOT_CHECK_LAYER}): {spot_cols}, jan={jan_col} mar={mar_col}")

    shards = find_complete_shards()
    log(f"complete shards: {shards}")
    if not shards:
        print("NO COMPLETE SHARDS FOUND -- aborting", file=sys.stderr)
        sys.exit(1)

    per_shard_target = max(1, TARGET_TOTAL_TOKENS // len(shards))
    per_block = max(1, per_shard_target // BLOCKS_PER_SHARD)
    log(f"target ~{TARGET_TOTAL_TOKENS} tokens total, "
        f"~{per_shard_target}/shard, {BLOCKS_PER_SHARD} blocks/shard, ~{per_block}/block")

    # histogram accumulator: [N_COLS, 255] int64, codes shifted to 0..254
    hist = np.zeros((N_COLS, 255), dtype=np.int64)
    n_sampled_total = 0

    # top-K candidates per spot column: list of (score_int8, token_id)
    # (kept as a fallback / for reference -- see also saturated_counts below,
    # which is the more meaningful signal once we saw int8 clipping is severe)
    top_candidates = {c: [] for c in spot_cols}
    # frequency count of token ids AMONG SATURATED (code==+127, i.e. clipped
    # to the int8 ceiling) tokens per spot column -- since int8 quantization
    # means "top-100 by argpartition" is largely an arbitrary sample of ties
    # at the clip value, frequency-among-saturated is the honest way to see
    # which tokens dominate the extreme-score bucket.
    saturated_counts = {c: Counter() for c in spot_cols}
    saturated_totals = {c: 0 for c in spot_cols}

    # for jan/mar correlation: accumulate raw int8 arrays (small: sample size only)
    jan_codes_all = []
    mar_codes_all = []

    for sid, n_rows in shards:
        tok_p = os.path.join(SCORES_DIR, f"tokens_{sid}.npy")
        sc_p = os.path.join(SCORES_DIR, f"scores_{sid}.npy")
        tok_arr = np.load(tok_p, mmap_mode="r")
        sc_arr = np.load(sc_p, mmap_mode="r")

        block_size = min(per_block, n_rows // BLOCKS_PER_SHARD) if n_rows >= BLOCKS_PER_SHARD else n_rows
        offsets_frac = [0.0, 0.25, 0.5, 0.75][:BLOCKS_PER_SHARD]
        shard_sampled = 0
        for frac in offsets_frac:
            start = int(frac * n_rows)
            end = min(start + block_size, n_rows)
            if end <= start:
                continue
            chunk_scores = np.asarray(sc_arr[start:end])  # [n, 216] int8, materialize
            chunk_tokens = np.asarray(tok_arr[start:end])  # [n] int32
            n = chunk_scores.shape[0]
            shard_sampled += n
            n_sampled_total += n

            # --- histogram accumulation (exact); int32 throughout to keep
            # the transient flat-index array small (well under the "no big
            # wholesale arrays" budget). ---
            idx = chunk_scores.astype(np.int32) + 127  # 0..254
            col_offset = (np.arange(N_COLS, dtype=np.int32) * 255)[None, :]
            flat_idx = (idx + col_offset).ravel()
            counts = np.bincount(flat_idx, minlength=N_COLS * 255)
            hist += counts.reshape(N_COLS, 255)

            # --- top-K candidates for spot-check columns (raw argpartition,
            # kept for reference even though it will mostly sample ties) ---
            for concept, col in spot_cols.items():
                colvals = chunk_scores[:, col]
                kk = min(TOPK_CANDIDATE_PER_CHUNK, n)
                cand_idx = np.argpartition(-colvals, kk - 1)[:kk]
                for i in cand_idx:
                    top_candidates[concept].append((int(colvals[i]), int(chunk_tokens[i])))
                # trim periodically to bound memory
                if len(top_candidates[concept]) > 5000:
                    top_candidates[concept].sort(key=lambda x: -x[0])
                    top_candidates[concept] = top_candidates[concept][:TOPK_FINAL * 3]
                # --- saturated (code==+127) token-id frequency ---
                sat_mask = colvals == 127
                if sat_mask.any():
                    sat_ids = chunk_tokens[sat_mask]
                    saturated_totals[concept] += int(sat_mask.sum())
                    saturated_counts[concept].update(sat_ids.tolist())

            # --- jan/mar codes for correlation ---
            jan_codes_all.append(chunk_scores[:, jan_col].copy())
            mar_codes_all.append(chunk_scores[:, mar_col].copy())

        log(f"shard {sid}: sampled {shard_sampled} tokens (n_rows={n_rows}), "
            f"running total {n_sampled_total}, elapsed {time.time()-t0:.0f}s")

    log(f"TOTAL sampled tokens: {n_sampled_total}")

    # ---- finalize quantiles from exact histogram ----
    cum = np.cumsum(hist, axis=1)  # [N_COLS, 255]
    totals = cum[:, -1]  # per-column total count (== n_sampled_total, sanity)
    columns_out = []
    for col in range(N_COLS):
        block = "dom" if col >= 3 * K else "main"
        if block == "main":
            li = col // K
            layer = layers[li]
            concept = main_block_concepts[col % K]   # true concept in this store col
        else:
            layer = ablation_layer
            concept = concepts[col - 3 * K]          # dom block: name-sorted
        tot = int(totals[col])
        qvals = {}
        for qname, qfrac in zip(QNAMES, QUANTILES):
            target = qfrac * tot
            code = int(np.searchsorted(cum[col], target, side="left"))
            code = min(code, 254)
            qvals[qname] = float(zero[col] + scale[col] * (code - 127))
        # mean/std from histogram (exact, in code space then affine-mapped)
        codes = np.arange(255) - 127
        p = hist[col].astype(np.float64) / max(tot, 1)
        code_mean = float((codes * p).sum())
        code_var = float(((codes - code_mean) ** 2 * p).sum())
        mean = zero[col] + scale[col] * code_mean
        std = scale[col] * (code_var ** 0.5)
        clip_pos = float(hist[col, 254] / max(tot, 1))  # code == +127
        clip_neg = float(hist[col, 0] / max(tot, 1))    # code == -127
        columns_out.append({
            "col": col, "block": block, "layer": layer, "concept": concept,
            **qvals, "mean": mean, "std": std, "n": tot,
            "clip_frac_pos127": clip_pos, "clip_frac_neg127": clip_neg,
        })

    # ---- top-100 firing tokens per spot concept (raw argpartition sample;
    # noted in the JSON that this is largely arbitrary among clip-value ties) ----
    top_tokens_out = {}
    for concept, cands in top_candidates.items():
        cands.sort(key=lambda x: -x[0])
        top = cands[:TOPK_FINAL]
        top_tokens_out[concept] = [{"score_int8": s, "token_id": tid} for s, tid in top]

    # ---- most-frequent token ids AMONG SATURATED (code==+127) tokens ----
    saturated_top_out = {}
    for concept, counter in saturated_counts.items():
        most_common = counter.most_common(TOPK_FINAL)
        saturated_top_out[concept] = {
            "n_saturated_sampled": saturated_totals[concept],
            "frac_saturated_of_sample": saturated_totals[concept] / max(n_sampled_total, 1),
            "top_by_frequency": [{"token_id": tid, "count": cnt} for tid, cnt in most_common],
        }

    # ---- decode top tokens with gemma tokenizer ----
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(GEMMA_TOK_PATH)
        for concept, lst in top_tokens_out.items():
            for entry in lst:
                entry["token_str"] = tok.decode([entry["token_id"]])
        for concept, d in saturated_top_out.items():
            for entry in d["top_by_frequency"]:
                entry["token_str"] = tok.decode([entry["token_id"]])
    except Exception as e:
        log(f"tokenizer decode failed: {e}")

    # ---- jan/mar correlation ----
    jan_codes = np.concatenate(jan_codes_all).astype(np.float64)
    mar_codes = np.concatenate(mar_codes_all).astype(np.float64)
    jan_vals = zero[jan_col] + scale[jan_col] * jan_codes
    mar_vals = zero[mar_col] + scale[mar_col] * mar_codes
    corr = float(np.corrcoef(jan_vals, mar_vals)[0, 1])
    log(f"january-march L8 pearson corr: {corr:.4f} (n={len(jan_vals)})")

    out = {
        "shards_used": [sid for sid, _ in shards],
        "n_sampled_total": n_sampled_total,
        "quantile_names": QNAMES,
        "columns": columns_out,
        "spot_check": {
            "layer": SPOT_CHECK_LAYER,
            "top_tokens": top_tokens_out,
            "saturated_top_by_frequency": saturated_top_out,
        },
        "january_march_corr_L8": corr,
        "elapsed_sec": time.time() - t0,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
