"""Curation pass (§4.7) over judged examples: near-duplicate detection and
template-repetition caps, on top of the validity flags set at generation time.

Reads  data/<family>/judged/judged.jsonl
Writes data/<family>/judged/curated.jsonl   (same records + curate_flag field;
                                             nothing is silently dropped)
       data/<family>/judged/curate_report.json

curate_flag values:
  near_dup      — >= 0.6 Jaccard overlap of 8-gram shingles with an earlier kept
                  example (first occurrence wins)
  opening_rep   — identical leading 4 words already used by >= 25 kept examples
                  of the same (class, slice) (template monoculture cap)
"""
import argparse
import collections
import json
import os
import re

import promptlib

STAGE4 = promptlib.STAGE4
NGRAM = 8
JACCARD_T = 0.6
OPENING_CAP = 25


def shingles(text):
    toks = re.findall(r"[a-z0-9']+", text.lower())
    if len(toks) < NGRAM:
        return {hash(" ".join(toks))}
    return {hash(" ".join(toks[i:i + NGRAM])) for i in range(len(toks) - NGRAM + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    args = ap.parse_args()
    base = os.path.join(STAGE4, "data", args.family, "judged")
    recs = [json.loads(l) for l in open(os.path.join(base, "judged.jsonl"))]

    index = collections.defaultdict(list)     # shingle-hash -> [kept example idx]
    kept_shingles = {}
    openings = collections.Counter()
    n_dup = n_rep = 0

    for i, r in enumerate(recs):
        r["curate_flag"] = None
        if r.get("flag") is not None:
            continue  # already excluded at generation; skip expensive checks
        sh = shingles(r["text"])
        # candidate earlier examples sharing any shingle
        cand = collections.Counter()
        for h in sh:
            for j in index[h]:
                cand[j] += 1
        is_dup = False
        for j, common in cand.most_common(5):
            union = len(sh | kept_shingles[j])
            if union and common / union >= JACCARD_T:
                is_dup = True
                break
        if is_dup:
            r["curate_flag"] = "near_dup"
            n_dup += 1
            continue
        okey = (r["class"], r["slice"],
                " ".join(re.findall(r"[a-z']+", r["text"].lower())[:4]))
        openings[okey] += 1
        if openings[okey] > OPENING_CAP:
            r["curate_flag"] = "opening_rep"
            n_rep += 1
            continue
        kept_shingles[i] = sh
        for h in sh:
            index[h].append(i)

    with open(os.path.join(base, "curated.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(1 for r in recs if r.get("flag") is None and r["curate_flag"] is None)
    by = collections.Counter((r["class"], r["slice"]) for r in recs
                             if r.get("flag") is None and r["curate_flag"] is None)
    report = {"family": args.family, "records": len(recs), "kept": ok,
              "near_dup": n_dup, "opening_rep": n_rep,
              "kept_by_class_slice": {f"{c}/{s}": n for (c, s), n in sorted(by.items())}}
    with open(os.path.join(base, "curate_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("family", "records", "kept", "near_dup", "opening_rep")}))


if __name__ == "__main__":
    main()
