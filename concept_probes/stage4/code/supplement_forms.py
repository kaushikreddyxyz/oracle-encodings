"""Form-holdout supplement (§6.2): for classes whose assembled form_holdout split
is starved (< MIN_HOLDOUT rows), generate explicit positives conditioned to use
ONLY a held-out surface form, judge them normally, merge into the family's
judged pool, and re-run curation + assembly.

Runs per family AFTER its main assembly exists. Idempotent: skips classes
already at quota and skips merging duplicate example_ids.

Artifacts (all tagged _forms, same audit guarantees as the main run):
  prompts/gen_prompts_forms.jsonl   calls/gen_forms.jsonl   calls/judge_forms.jsonl
  raw_gen/generations_forms.jsonl   judged/judged_forms.jsonl
Merged records are identifiable in judged/judged.jsonl by slice == 'explicit_form'.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys

import promptlib
import generate as gen_mod
import judge as judge_mod

STAGE4 = promptlib.STAGE4
MIN_HOLDOUT = 50
PER_CLASS = 120


def starved_classes(family):
    rep_path = os.path.join(STAGE4, "data", family, "final", "assembly_report.json")
    rep = json.load(open(rep_path))
    out = []
    for cls, v in rep["classes"].items():
        if v["by_split"].get("form_holdout", 0) < MIN_HOLDOUT:
            out.append(cls)
    return out


def run_family(family, cap_usd):
    base = os.path.join(STAGE4, "data", family)
    cfg = promptlib.load_all(family)
    have_ft = [c for c, cc in cfg["pack"]["classes"].items() if cc["form_test"]]
    starved = [c for c in starved_classes(family) if c in have_ft]
    if not starved:
        print(f"{family}: no starved classes, skipping")
        return
    skip = [c for c in cfg["pack"]["classes"] if c not in starved]
    recs = promptlib.build_form_prompts(family, PER_CLASS, skip_classes=skip)
    print(f"{family}: supplementing {starved} with {len(recs)} calls")
    with open(os.path.join(base, "prompts", "gen_prompts_forms.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    asyncio.run(gen_mod.run(family, cap_usd, tag="forms"))
    asyncio.run(judge_mod.run(family, cap_usd, tag="forms"))

    # merge (idempotent by example_id) then re-curate + re-assemble
    main_path = os.path.join(base, "judged", "judged.jsonl")
    existing = {json.loads(l)["example_id"] for l in open(main_path)}
    added = 0
    with open(main_path, "a") as f:
        for l in open(os.path.join(base, "judged", "judged_forms.jsonl")):
            r = json.loads(l)
            if r["example_id"] in existing:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            added += 1
    print(f"{family}: merged {added} form-holdout examples")
    code = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, os.path.join(code, "curate.py"),
                    "--family", family], check=True)
    subprocess.run([sys.executable, os.path.join(code, "assemble.py"),
                    "--family", family], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--cap-usd", type=float, default=10.0)
    args = ap.parse_args()
    for fam in args.families:
        rep = os.path.join(STAGE4, "data", fam, "final", "assembly_report.json")
        if not os.path.exists(rep):
            print(f"{fam}: not yet assembled, skipping (rerun later)")
            continue
        run_family(fam, args.cap_usd)
