"""Quality gates (probe_dataset_spec.md §10). Operates on a produced JSONL.

Gates:
  G1 class balance      per label_value & per negative_type
  G2 span-mapping audit 50 random positives: span tokens' offsets overlap a claimed char span,
                        and decoded span tokens visibly cover the claimed substring
  G3 token-target integrity   EVERY record: recompute labels from text; assert stored ==
                        recomputed (len==len==n_tokens; loss_mask 0 exactly on post; span target
                        == value; pre target == 'absent')
  G4 minimal-pair diff  per mp group: token-level difflib diff localizes to the recorded span tokens
  G5 heldout-vocab leak banned trigger truly absent in every in_vocabulary==False record (+ skipped list)
  G6 judge agreement    reported from build stats (votes not stored per-record)

Any G2/G3/G4 failure is a HALT-AND-FIX (offset/mask bug corrupts training silently).
"""
import json
import random
import re

import concept_configs as cc
import labeling as L

random.seed(7)


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def g1_balance(recs):
    pos = [r for r in recs if r["polarity"] == "positive" and r["split"] != "heldout_vocab"]
    by_val, by_split, by_gen = {}, {}, {}
    for r in pos:
        by_val[r["label_value"]] = by_val.get(r["label_value"], 0) + 1
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
        by_gen[r["generator"]] = by_gen.get(r["generator"], 0) + 1
    neg = [r for r in recs if r["polarity"] == "negative"]
    by_negtype = {}
    for r in neg:
        by_negtype[r["negative_type"]] = by_negtype.get(r["negative_type"], 0) + 1
    ho = [r for r in recs if r["split"] == "heldout_vocab"]
    by_ho = {}
    for r in ho:
        by_ho[r["label_value"]] = by_ho.get(r["label_value"], 0) + 1
    return {"positives_by_value": by_val, "positives_by_split": by_split,
            "positives_by_generator": by_gen, "negatives_by_type": by_negtype,
            "heldout_by_value": by_ho, "n_pos": len(pos), "n_neg": len(neg), "n_heldout": len(ho)}


def g2_span_audit(recs, k=50):
    pos = [r for r in recs if r["polarity"] == "positive" and r.get("concept_span_char")]
    sample = random.sample(pos, min(k, len(pos)))
    ok = 0
    fails = []
    for r in sample:
        _ids, offsets = L.encode_with_offsets(r["text"])
        spans = r["concept_span_char"]
        good = True
        for ti in r["concept_span_tokens"]:
            if ti >= len(offsets):
                good = False
                break
            a, b = offsets[ti]
            if not any(a < e and b > s for (s, e) in spans):
                good = False
                break
        # every claimed char span must be covered by at least one span token
        covered = all(any(ti < len(offsets) and offsets[ti][0] < e and offsets[ti][1] > s
                          for ti in r["concept_span_tokens"]) for (s, e) in spans)
        if good and covered and r["concept_span_tokens"]:
            ok += 1
        else:
            fails.append({"text": r["text"][:80], "spans": spans, "tokens": r["concept_span_tokens"]})
    return {"checked": len(sample), "passed": ok, "failures": fails[:5]}


def g3_integrity(recs):
    bad_len, bad_mask, bad_span_target, bad_pre, n = 0, 0, 0, 0, 0
    examples = []
    for r in recs:
        n += 1
        spans = r["concept_span_char"] if r["polarity"] == "positive" else None
        tgt = None
        if r["polarity"] == "positive":
            tgt = r["label_index"] if r["family"] == "cyclic" else r["label_value"]
        ids, tt, lm, span_idx = L.build_token_labels(r["text"], spans, tgt)
        nt = len(ids)
        if not (len(r["token_targets"]) == len(r["loss_mask"]) == nt == r.get("n_tokens", nt)):
            bad_len += 1
            if len(examples) < 5:
                examples.append(("len", r["text"][:60]))
            continue
        # mask 0 exactly on post-span
        post_idx = {i for i, t in enumerate(tt) if t["region"] == "post"}
        stored_zero = {i for i, m in enumerate(r["loss_mask"]) if m == 0}
        if post_idx != stored_zero:
            bad_mask += 1
            if len(examples) < 5:
                examples.append(("mask", r["text"][:60]))
        # span tokens carry the value; pre carry absent
        for i, t in enumerate(r["token_targets"]):
            if t["region"] == "span" and t["target"] != (r["label_index"] if r["family"] == "cyclic"
                                                         else r["label_value"]):
                bad_span_target += 1
                break
        for i, t in enumerate(r["token_targets"]):
            if t["region"] == "pre" and t["target"] != "absent":
                bad_pre += 1
                break
    return {"n": n, "bad_len": bad_len, "bad_mask": bad_mask,
            "bad_span_target": bad_span_target, "bad_pre_target": bad_pre, "examples": examples}


def g4_minimal_pairs(recs):
    groups = {}
    for r in recs:
        mp = r.get("minimal_pair_id")
        if mp:
            groups.setdefault(mp, []).append(r)
    checked = consistent = equal = 0
    fails = []
    for mp, grp in groups.items():
        if len(grp) < 2:
            continue
        base = grp[0]
        base_ids, _ = L.encode_with_offsets(base["text"])
        for other in grp[1:]:
            other_ids, _ = L.encode_with_offsets(other["text"])
            a_ch, b_ch = L.token_diff_indices(base_ids, other_ids)
            checked += 1
            base_span = set(base["concept_span_tokens"] or [])
            other_span = set(other["concept_span_tokens"] or [])
            da, db = set(a_ch), set(b_ch)
            # the diff must localize to the slot: changed tokens equal (or contained within) span tokens
            eq = (da == base_span and db == other_span)
            cons = (da <= base_span or base_span <= da) and (db <= other_span or other_span <= db)
            if eq:
                equal += 1
            if cons:
                consistent += 1
            elif len(fails) < 5:
                fails.append({"mp": mp, "base": base["text"][:50], "other": other["text"][:50],
                              "diff_b": sorted(db), "span_b": sorted(other_span)})
    return {"pairs_checked": checked, "exact": equal, "consistent": consistent, "failures": fails}


def g5_heldout_leak(recs):
    ho = [r for r in recs if r["split"] == "heldout_vocab"]
    leaks = []
    for r in ho:
        cfg = cc.get(r["concept"])
        banned = []
        for v in cfg.get("values", []):
            if v["name"] == r["label_value"]:
                banned = v["banned"]
        for b in banned:
            if re.search(r"\b" + re.escape(b) + r"\w*", r["text"], re.IGNORECASE):
                leaks.append({"value": r["label_value"], "banned": b, "text": r["text"][:80]})
                break
    return {"n_heldout": len(ho), "leaks": len(leaks), "examples": leaks[:5]}


def run_all(path, build_stats=None):
    recs = load(path)
    report = {
        "path": path, "n_records": len(recs),
        "G1_balance": g1_balance(recs),
        "G2_span_audit": g2_span_audit(recs),
        "G3_integrity": g3_integrity(recs),
        "G4_minimal_pairs": g4_minimal_pairs(recs),
        "G5_heldout_leak": g5_heldout_leak(recs),
    }
    g2, g3, g4, g5 = report["G2_span_audit"], report["G3_integrity"], report["G4_minimal_pairs"], report["G5_heldout_leak"]
    report["PASS"] = {
        "G2_span_audit": g2["passed"] == g2["checked"] and g2["checked"] > 0,
        "G3_integrity": g3["bad_len"] == 0 and g3["bad_mask"] == 0 and g3["bad_span_target"] == 0 and g3["bad_pre_target"] == 0,
        "G4_minimal_pairs": g4["pairs_checked"] == 0 or g4["consistent"] == g4["pairs_checked"],
        "G5_heldout_leak": g5["leaks"] == 0,
    }
    if build_stats:
        report["G6_judge"] = {
            "judge_pos_agree_mean_votes_of_3": build_stats.get("judge_pos_agree_mean"),
            "kept_positives": build_stats.get("kept_positives"),
            "kept_heldout": build_stats.get("kept_heldout"),
            "kept_negatives": build_stats.get("kept_negatives"),
            "heldout_skipped_values": build_stats.get("heldout_skipped_values"),
            "or_calls": (build_stats.get("or_stats") or {}).get("calls"),
            "or_cost_usd": (build_stats.get("or_stats") or {}).get("cost_usd"),
        }
    report["ALL_PASS"] = all(report["PASS"].values())
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--build-stats", default=None)
    args = ap.parse_args()
    bs = json.load(open(args.build_stats)) if args.build_stats else None
    print(json.dumps(run_all(args.path, bs), indent=2))
