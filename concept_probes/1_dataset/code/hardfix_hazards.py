"""Repair hard-negative streams whose hazard lives on a SYNONYM surface form.

Root cause (found in color_wheel): the hard_negative template forced the class
display name ("Blue-Green") into texts targeting a synonym's wrong sense (teal
the duck), producing nonsense that judges scored as concept-present. Pack
hazards for such cases are now dicts {form, sense}; the template uses {FORM}.

This script, per family:
 1. flags the old mis-surfaced hard_negative records (flag=hard_neg_wrong_surface;
    kept in judged.jsonl for audit, excluded downstream),
 2. regenerates those hazard streams with the corrected form (tag=hardfix),
 3. judges, merges, re-curates, re-assembles.
Idempotent: skips if no dict hazards or if hardfix examples already merged.
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


def run_family(family, cap_usd, hard_per_class):
    cfg = promptlib.load_all(family)
    pack = cfg["pack"]
    dict_senses = {}   # class -> set of sense strings that were mis-surfaced
    for cls, cc in pack["classes"].items():
        senses = {hz["sense"] for hz in (cc.get("hazards") or [])
                  if isinstance(hz, dict)
                  and hz["form"].lower() != promptlib._display(cls).lower()}
        if senses:
            dict_senses[cls] = senses
    if not dict_senses:
        print(f"{family}: no synonym-form hazards, skipping")
        return

    base = os.path.join(STAGE4, "data", family)
    jp = os.path.join(base, "judged", "judged.jsonl")
    recs = [json.loads(l) for l in open(jp)]
    n_flag = 0
    for r in recs:
        if (r["slice"] == "hard_negative" and r.get("flag") is None
                and r["class"] in dict_senses and r.get("hazard") in dict_senses[r["class"]]):
            r["flag"] = "hard_neg_wrong_surface"
            n_flag += 1
    with open(jp, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{family}: flagged {n_flag} mis-surfaced hard negatives")

    # regenerate only the affected streams
    all_recs = promptlib.build_gen_prompts(
        family, {"explicit": 0, "implicit": 0,
                 "hard_negative": hard_per_class, "neutral_nearmiss": 0},
        cfg["models"]["runtime"]["seed"] + 7)
    fix = [r for r in all_recs if r["slice"] == "hard_negative"
           and r["class"] in dict_senses and r.get("hazard") in dict_senses[r["class"]]]
    if not fix:
        print(f"{family}: nothing to regenerate")
        return
    with open(os.path.join(base, "prompts", "gen_prompts_hardfix.jsonl"), "w") as f:
        for r in fix:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{family}: regenerating {len(fix)} hard-negative calls with corrected forms")
    asyncio.run(gen_mod.run(family, cap_usd, tag="hardfix"))
    asyncio.run(judge_mod.run(family, cap_usd, tag="hardfix"))

    existing = {json.loads(l)["example_id"] for l in open(jp)}
    added = 0
    with open(jp, "a") as f:
        for l in open(os.path.join(base, "judged", "judged_hardfix.jsonl")):
            r = json.loads(l)
            if r["example_id"] in existing:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            added += 1
    print(f"{family}: merged {added} corrected hard negatives")
    code = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, os.path.join(code, "curate.py"),
                    "--family", family], check=True)
    subprocess.run([sys.executable, os.path.join(code, "assemble.py"),
                    "--family", family], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--cap-usd", type=float, default=10.0)
    ap.add_argument("--hard-per-class", type=int, default=800)
    args = ap.parse_args()
    for fam in args.families:
        run_family(fam, args.cap_usd, args.hard_per_class)
