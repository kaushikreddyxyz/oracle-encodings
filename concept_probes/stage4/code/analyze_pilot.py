"""Gate-2 quality report for a family pilot: judged-score behavior per slice,
K-sample agreement, and full-run cost extrapolation.

Reads  data/<family>/raw_gen/gen_report.json
       data/<family>/judged/judged.jsonl + judge_report.json
Writes data/<family>/reports/pilot_report.md  (+ prints a JSON summary)

Slice expectations (post 2B-plausible-activation rule, 2026-07-01):
  explicit        target-class strength skews high (>=0.5 typical)
  implicit        moderate-high, and MUST NOT be zero-heavy (generator failed) —
                  strength here is judged without any trigger token present
  hard_negative   faint: <=0.33 (score <=2); higher = judge crediting wrong sense
  neutral_nearmiss  ~0 for all classes
"""
import argparse
import collections
import json
import os
import statistics

import promptlib

STAGE4 = promptlib.STAGE4


def cls_strength(rec, cls):
    """Max aggregated strength for a class over the example's spans."""
    return max((s["strength"] for s in rec["aggregated_spans"]
                if s["concept"] == cls), default=0.0)


def sample_max_score(sample_spans, cls):
    return max((float(s.get("score") or 0) for s in sample_spans
                if (s.get("concept") or "").lower() == cls), default=0.0)


def quantiles(xs):
    if not xs:
        return None
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "mean": round(statistics.mean(xs), 3),
            "p10": round(q(.10), 3), "p50": round(q(.50), 3),
            "p90": round(q(.90), 3), "frac_zero": round(sum(x == 0 for x in xs) / len(xs), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    args = ap.parse_args()
    base = os.path.join(STAGE4, "data", args.family)
    recs = [json.loads(l) for l in open(os.path.join(base, "judged", "judged.jsonl"))]
    gen_rep = json.load(open(os.path.join(base, "raw_gen", "gen_report.json")))
    jud_rep = json.load(open(os.path.join(base, "judged", "judge_report.json")))

    strengths = collections.defaultdict(list)   # slice -> target-class strengths
    off_target = []                             # neutral: max strength any class
    agreement = []                              # per-example pairwise |diff| of sample scores
    per_class = collections.defaultdict(lambda: collections.defaultdict(list))

    for r in recs:
        cls = r["class"]
        if r["slice"] == "neutral_nearmiss":
            off_target.append(max((s["strength"] for s in r["aggregated_spans"]),
                                  default=0.0))
        else:
            v = cls_strength(r, cls)
            strengths[r["slice"]].append(v)
            per_class[cls][r["slice"]].append(v)
            sm = [sample_max_score(r["judge_samples"][k], cls)
                  for k in ("v1", "v2", "v3")]
            agreement.extend(abs(a - b) for i, a in enumerate(sm)
                             for b in sm[i + 1:])

    # cumulative cost from the audit logs (per-run reports only cover the last
    # resume pass; the logs are ground truth across all passes)
    def log_cost(name):
        p = os.path.join(base, "calls", name)
        return sum(float((json.loads(l).get("usage") or {}).get("cost") or 0)
                   for l in open(p)) if os.path.exists(p) else 0.0
    gen_cost, judge_cost = log_cost("gen.jsonl"), log_cost("judge.jsonl")

    summary = {
        "family": args.family,
        "examples_judged": len(recs),
        "strength_by_slice": {sl: quantiles(v) for sl, v in sorted(strengths.items())},
        "neutral_any_class_strength": quantiles(off_target),
        "judge_sample_agreement_mean_abs_diff_0to6": (
            round(statistics.mean(agreement), 3) if agreement else None),
        "unmatched_quotes": jud_rep["unmatched_quotes"],
        "bad_judge_calls": jud_rep["bad_judge_calls"],
        "gen_cost_usd_cumulative": round(gen_cost, 3),
        "judge_cost_usd_cumulative": round(judge_cost, 3),
    }

    # cost extrapolation: full spec volumes vs pilot volumes, scaled by measured cost/item
    pilot_items = len(recs)
    cost_per_item = (gen_cost + judge_cost) / max(1, pilot_items)
    reg = promptlib._load_yaml("config/registry.yaml")
    vols = reg["volumes"]
    catp = sum(1 for f, c in reg["families"].items()
               if c["construct"] in ("categorical", "presence")
               for _ in c["classes"])
    intensity_axes = sum(1 for c in reg["families"].values() if c["construct"] == "intensity")
    # produced_per_class already excludes the borrowed sibling slice; the shared
    # family neutral pool makes this a slight overestimate (conservative).
    full_items = catp * vols["categorical_presence"]["produced_per_class"] \
        + intensity_axes * vols["intensity"]["produced_per_axis"]
    summary["cost_per_judged_item_usd"] = round(cost_per_item, 6)
    summary["full_run_items_approx"] = full_items
    summary["full_run_cost_extrapolated_usd"] = round(full_items * cost_per_item, 2)

    os.makedirs(os.path.join(base, "reports"), exist_ok=True)
    lines = [f"# Pilot report — {args.family}", "",
             "```json", json.dumps(summary, indent=2), "```", "",
             "## Per-class target-class strength (mean [n])", "",
             "| class | " + " | ".join(sorted(strengths)) + " |",
             "|---|" + "---|" * len(strengths)]
    for cls in sorted(per_class):
        row = [cls]
        for sl in sorted(strengths):
            xs = per_class[cls].get(sl, [])
            row.append(f"{statistics.mean(xs):.2f} [{len(xs)}]" if xs else "—")
        lines.append("| " + " | ".join(row) + " |")
    with open(os.path.join(base, "reports", "pilot_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
