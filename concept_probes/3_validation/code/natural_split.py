"""Freeze the natural deployment split (§0.5): standardization sample + random pool.

Writes data/natural/standardization_sample.jsonl
       data/natural/random_pool.jsonl

standardization_sample: 5,000 random English-ish docs from shard 310, truncated to
their first 2,000 chars — used ONLY for §0.6 per-layer activation mean/std.

random_pool: 6,000 random docs from shards 311-312 (disjoint from the
standardization sample by construction), each reduced to ONE random contiguous
window of 1-3 sentences (target 200-450 chars) — the concept-absent side of the
§6.2 natural eval pool. nat_split cal/test by md5 parity.
"""
import json
import os
import random

import nat_common as nc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "natural")

STD_SHARD = 310
STD_N = 5000
POOL_SHARDS = [311, 312]
POOL_N_PER_SHARD = 3000
OVERSAMPLE = 4  # sampled candidate indices per kept doc, to survive filters


def sample_indices(shard, n_target, rng):
    n_rows = nc.shard_num_rows(shard)
    n_pick = min(n_rows, n_target * OVERSAMPLE)
    return set(rng.sample(range(n_rows), n_pick)), n_rows


def build_standardization(rng):
    picked, n_rows = sample_indices(STD_SHARD, STD_N, rng)
    out, kept = [], 0
    for i, text in nc.iter_shard_docs(STD_SHARD):
        if kept >= STD_N:
            break
        if i not in picked:
            continue
        if nc.doc_english_ratio(text) < 0.6:
            continue
        out.append({"doc_id": f"s{STD_SHARD:05d}_d{i:06d}", "shard": STD_SHARD,
                    "text": text[:2000]})
        kept += 1
    print(f"[std] shard {STD_SHARD}: rows={n_rows} kept={kept}/{STD_N}")
    return out


def random_window(text, rng):
    """One random contiguous 1-3 sentence window targeting 200-450 chars, or None."""
    sents = nc.sentence_spans(text)
    if not sents:
        return None
    for _ in range(5):
        i = rng.randrange(len(sents))
        s, e = sents[i]
        k = i
        while e - s < 200 and k + 1 < len(sents) and k - i < 2:
            k += 1
            e = sents[k][1]
        win = text[s:e].strip()
        if len(win) > 450:
            cut = win[:450]
            sp = cut.rfind(" ")
            win = cut[:sp] if sp > 120 else cut
        if len(win) >= 120 and nc.window_alpha_ratio(win) >= 0.6:
            return win
    return None


def build_pool(rng):
    out = []
    for shard in POOL_SHARDS:
        picked, n_rows = sample_indices(shard, POOL_N_PER_SHARD, rng)
        kept = 0
        for i, text in nc.iter_shard_docs(shard):
            if kept >= POOL_N_PER_SHARD:
                break
            if i not in picked:
                continue
            if nc.doc_english_ratio(text) < 0.6:
                continue
            win = random_window(text, rng)
            if win is None:
                continue
            doc_id = f"s{shard:05d}_d{i:06d}"
            ex_id = f"nat_r_{doc_id}"
            out.append({"example_id": ex_id, "doc_id": doc_id, "shard": shard,
                        "text": win, "nat_split": nc.nat_split_of(ex_id)})
            kept += 1
        print(f"[pool] shard {shard}: rows={n_rows} kept={kept}/{POOL_N_PER_SHARD}")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(nc.SEED)
    std = build_standardization(rng)
    pool = build_pool(rng)
    with open(os.path.join(OUT, "standardization_sample.jsonl"), "w") as f:
        for r in std:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "random_pool.jsonl"), "w") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_cal = sum(1 for r in pool if r["nat_split"] == "cal")
    print(f"[done] std={len(std)} pool={len(pool)} (cal={n_cal} test={len(pool)-n_cal})")


if __name__ == "__main__":
    main()
