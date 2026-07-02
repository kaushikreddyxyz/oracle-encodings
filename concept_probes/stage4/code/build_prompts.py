"""Materialize the exact generation prompts for a family run — the AUDIT GATE 1
artifact. Writes:
  data/<family>/prompts/gen_prompts.jsonl   — every prompt verbatim (machine input)
  data/<family>/prompts/PREVIEW.md          — human-readable render: one full prompt
                                              per (slice) + per-class/slice call counts
  data/<family>/prompts/judge_preview.md    — one fully-rendered judge prompt per
                                              rubric variant, on the few-shot passages

No API calls are made here.
"""
import argparse
import collections
import json
import os

import promptlib

STAGE4 = promptlib.STAGE4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--explicit", type=int, required=True)
    ap.add_argument("--implicit", type=int, required=True)
    ap.add_argument("--hard-negative", type=int, required=True)
    ap.add_argument("--neutral-nearmiss", type=int, required=True,
                    help="family-level shared pool size")
    args = ap.parse_args()

    cfg = promptlib.load_all(args.family)
    seed = cfg["models"]["runtime"]["seed"]
    volumes = {"explicit": args.explicit, "implicit": args.implicit,
               "hard_negative": args.hard_negative,
               "neutral_nearmiss": args.neutral_nearmiss}
    recs = promptlib.build_gen_prompts(args.family, volumes, seed)

    out_dir = os.path.join(STAGE4, "data", args.family, "prompts")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gen_prompts.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- human preview -----------------------------------------------------
    counts = collections.Counter((r["class"], r["slice"]) for r in recs)
    items = collections.Counter()
    for r in recs:
        items[(r["class"], r["slice"])] += r["n_items"]
    lines = [f"# Materialized generation prompts — {args.family}",
             f"\nTotal calls: {len(recs)} | volumes/class: {volumes}\n",
             "| class | slice | calls | items requested |", "|---|---|---|---|"]
    for (cls, sl), n in sorted(counts.items()):
        lines.append(f"| {cls} | {sl} | {n} | {items[(cls, sl)]} |")
    lines.append("\n---\n\nOne fully-rendered prompt per slice "
                 "(all others differ only in class fills / style line / item count):\n")
    seen = set()
    for r in recs:
        if r["slice"] in seen:
            continue
        seen.add(r["slice"])
        lines.append(f"\n## slice={r['slice']}  class={r['class']}  "
                     f"template={r['template_id']}  prompt_id={r['prompt_id']}\n")
        for m in r["messages"]:
            lines.append(f"**[{m['role']}]**\n\n```\n{m['content']}\n```\n")
    with open(os.path.join(out_dir, "PREVIEW.md"), "w") as f:
        f.write("\n".join(lines))

    # ---- judge preview (rendered on the family's few-shot passages) --------
    demo = [{"id": i, "text": s["text"]}
            for i, s in enumerate(cfg["pack"]["judge_fewshots"][:4])]
    jl = [f"# Rendered judge prompts — {args.family} (demo passages, all 3 variants)\n"]
    for v in ("v1", "v2", "v3"):
        msgs = promptlib.build_judge_prompt(args.family, demo, v, cfg)
        jl.append(f"\n## variant {v}\n")
        for m in msgs:
            jl.append(f"**[{m['role']}]**\n\n```\n{m['content']}\n```\n")
    with open(os.path.join(out_dir, "judge_preview.md"), "w") as f:
        f.write("\n".join(jl))

    print(json.dumps({"calls": len(recs),
                      "items_requested": sum(r["n_items"] for r in recs),
                      "out_dir": out_dir}, indent=2))


if __name__ == "__main__":
    main()
