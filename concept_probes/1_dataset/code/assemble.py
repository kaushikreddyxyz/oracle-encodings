"""Assemble per-probe training datasets from curated+judged examples.

Implements §4.4 (training mixture incl. sibling positives-as-negatives),
§0.5 (train/val split + lexical form holdout), §4.6 (token-level targets via
the gemma tokenizer offset mapping).

Reads  data/<family>/judged/curated.jsonl
Writes data/<family>/final/sorted/<class>.jsonl      (grouped by role — audit view)
       data/<family>/final/mixed/<class>.train.jsonl (shuffled)
       data/<family>/final/mixed/<class>.val.jsonl
       data/<family>/final/mixed/<class>.form_holdout.jsonl  (§6.2 eval only)
       data/<family>/final/assembly_report.json

Row schema (one row = one example labeled FOR one probe class):
  probe_class, role (target_pos|implicit_pos|hard_neg|sibling_neg|neutral),
  source_class, slice, level, text, split,
  target_spans [[start,end,strength] for the probe class],
  token_ids, token_targets_sparse [[tok_idx, strength]], n_tokens,
  provenance {prompt_id, gen_call_id, judge_call_ids, template_id}

Targets: token target = max aggregated-span strength overlapping the token
(§4.6: in-span tokens inherit the span strength; out-of-span tokens are 0).
The mixture is materialized per probe row; sharing examples across sibling
probes is by-design (§4.0: sibling slots are borrowed, not re-generated).
"""
import argparse
import collections
import json
import os
import random
import re

import promptlib

STAGE4 = promptlib.STAGE4
SIBLING_N = 800
NEUTRAL_N = 1000
TRAIN_FRAC = 0.85

_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        for name in ("google/gemma-2-2b", "google/gemma-2-9b"):
            try:
                _tok = AutoTokenizer.from_pretrained(name)
                break
            except Exception:
                continue
        if _tok is None:
            raise RuntimeError("no gemma tokenizer available (HF auth?)")
    return _tok


def token_targets(text, spans):
    """spans: [[s,e,strength]]. Returns (token_ids, sparse_targets, n_tokens)."""
    enc = tokenizer()(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    sparse = []
    for ti, (a, b) in enumerate(enc["offset_mapping"]):
        if a == b:
            continue
        best = 0.0
        for s, e, v in spans:
            if a < e and b > s:
                best = max(best, v)
        if best > 0:
            sparse.append([ti, round(best, 4)])
    return ids, sparse, len(ids)


def probe_spans(rec, probe_class):
    return [[sp["char_span"][0], sp["char_span"][1], sp["strength"]]
            for sp in rec.get("aggregated_spans", [])
            if sp["concept"] == probe_class]


def form_regexes(cls_cfg):
    def rx(forms):
        return [re.compile(r"(?<!\w)" + re.escape(f.rstrip(".")) + r"(?!\w)", re.I)
                for f in forms]
    return rx(cls_cfg["form_train"]), rx(cls_cfg["form_test"])


def make_row(rec, probe_class, role, split):
    spans = probe_spans(rec, probe_class)
    ids, sparse, n = token_targets(rec["text"], spans)
    return {"probe_class": probe_class, "role": role,
            "source_class": rec["class"], "slice": rec["slice"],
            "level": rec.get("level"), "example_id": rec["example_id"],
            "text": rec["text"], "split": split, "target_spans": spans,
            "token_ids": ids, "token_targets_sparse": sparse, "n_tokens": n,
            "provenance": {"prompt_id": rec["prompt_id"],
                           "gen_call_id": rec["gen_call_id"],
                           "judge_call_ids": rec.get("judge_call_ids"),
                           "template_id": rec["template_id"]}}


def split_of(rng):
    return "train" if rng.random() < TRAIN_FRAC else "val"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    args = ap.parse_args()
    cfg = promptlib.load_all(args.family)
    pack = cfg["pack"]
    base = os.path.join(STAGE4, "data", args.family)
    recs = [json.loads(l) for l in open(os.path.join(base, "judged", "curated.jsonl"))]
    ok = [r for r in recs if r.get("flag") is None and r.get("curate_flag") is None]
    rng = random.Random(cfg["models"]["runtime"]["seed"])

    by_cs = collections.defaultdict(list)
    for r in ok:
        by_cs[(r["class"], r["slice"])].append(r)
    neutral_pool = by_cs.get(("_family", "neutral_nearmiss"), [])
    classes = list(pack["classes"])
    intensity = cfg["construct"] == "intensity"

    os.makedirs(os.path.join(base, "final", "sorted"), exist_ok=True)
    os.makedirs(os.path.join(base, "final", "mixed"), exist_ok=True)
    report = {"family": args.family, "classes": {}}

    for cls in classes:
        rows = []
        ftr_rx, fte_rx = form_regexes(pack["classes"][cls])

        def add(rec, role, holdout_check=False):
            split = split_of(rng)
            if holdout_check:
                has_te = any(rx.search(rec["text"]) for rx in fte_rx)
                has_tr = any(rx.search(rec["text"]) for rx in ftr_rx)
                if has_te and not has_tr:
                    split = "form_holdout"
            rows.append(make_row(rec, cls, role, split))

        for r in by_cs.get((cls, "explicit"), []):
            add(r, "target_pos", holdout_check=True)
        for r in by_cs.get((cls, "explicit_form"), []):
            add(r, "target_pos", holdout_check=True)  # §6.2 supplement slice
        for r in by_cs.get((cls, "implicit"), []):
            add(r, "implicit_pos")
        for r in by_cs.get((cls, "hard_negative"), []):
            add(r, "hard_neg", holdout_check=True)
        if not intensity:
            sibs = [c for c in classes if c != cls]
            per_sib = max(1, SIBLING_N // max(1, len(sibs)))
            for sc in sibs:
                pool = by_cs.get((sc, "explicit"), [])
                for r in rng.sample(pool, min(per_sib, len(pool))):
                    add(r, "sibling_neg")
        for r in rng.sample(neutral_pool, min(NEUTRAL_N, len(neutral_pool))):
            add(r, "neutral")

        srt = sorted(rows, key=lambda x: (x["role"], x["example_id"]))
        safe = cls.replace(" ", "_")
        with open(os.path.join(base, "final", "sorted", f"{safe}.jsonl"), "w") as f:
            for row in srt:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        mixed = list(rows)
        rng.shuffle(mixed)
        outs = {}
        for sp in ("train", "val", "form_holdout"):
            outs[sp] = open(os.path.join(base, "final", "mixed",
                                         f"{safe}.{sp}.jsonl"), "w")
        for row in mixed:
            outs[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
        for fh in outs.values():
            fh.close()

        rc = collections.Counter(r["role"] for r in rows)
        sc = collections.Counter(r["split"] for r in rows)
        nz = sum(1 for r in rows if r["token_targets_sparse"])
        report["classes"][cls] = {
            "rows": len(rows), "by_role": dict(rc), "by_split": dict(sc),
            "rows_with_nonzero_targets": nz}

    with open(os.path.join(base, "final", "assembly_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({c: v["rows"] for c, v in report["classes"].items()}, indent=1))


if __name__ == "__main__":
    main()
