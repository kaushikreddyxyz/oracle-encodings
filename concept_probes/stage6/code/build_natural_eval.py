"""Stage 6: judged natural pool -> token-level eval rows (runs on the mac, CPU).

Reads stage4/data/<family>/judged/judged_nat.jsonl, tokenizes with the gemma-2
tokenizer (add_special_tokens=False — same convention as Stage 4), paints each
class's aggregated char spans onto tokens (max over overlapping chars), and
writes stage6/data/natural/eval/<family>.jsonl with one row per example:

  {example_id, text, token_ids, n_tokens, nat_split, slice, cls_mined, surface,
   targets: {class: [[tok_idx, strength], ...]}}

Every example carries targets for ALL family classes (family-level judging), so
the scorer builds a [tokens, classes] target matrix directly.

  python build_natural_eval.py --families months,weekdays,...
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE6 = HERE.parent
STAGE4 = STAGE6.parent / "stage4"
OUT = STAGE6 / "data" / "natural" / "eval"

FAMILIES = ["months", "weekdays", "seasons", "color_wheel", "directions",
            "moon_phases", "continents", "location_type",
            "costliness", "physical_size", "lovingness", "duration", "harmfulness"]


def family_class_names(fam: str) -> list[str]:
    import yaml
    pack = yaml.safe_load(open(STAGE4 / "config" / "families" / f"{fam}.yaml"))
    return sorted(c.lower() for c in pack["classes"])


def paint(spans, offsets, n_tok):
    """spans: [{concept, char_span, strength}] for ONE concept; -> sparse tok targets."""
    out = {}
    for sp in spans:
        cs, ce = sp["char_span"]
        s = float(sp["strength"])
        if s <= 0:
            continue
        for ti in range(n_tok):
            ts, te = offsets[ti]
            if ts < ce and te > cs:            # token overlaps span
                out[ti] = max(out.get(ti, 0.0), s)
    return sorted([list(kv) for kv in out.items()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--model", default="google/gemma-2-2b")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    OUT.mkdir(parents=True, exist_ok=True)

    for fam in args.families.split(","):
        src = STAGE4 / "data" / fam / "judged" / "judged_nat.jsonl"
        if not src.exists():
            print(f"{fam}: judged_nat.jsonl missing — skip")
            continue
        classes = family_class_names(fam)
        n_rows = 0
        with open(src) as f, open(OUT / f"{fam}.jsonl", "w") as g:
            for line in f:
                r = json.loads(line)
                enc = tok(r["text"], add_special_tokens=False,
                          return_offsets_mapping=True)
                ids, offs = enc["input_ids"], enc["offset_mapping"]
                if not ids or len(ids) > 512:
                    continue
                by_cls = {}
                for sp in r.get("aggregated_spans") or []:
                    by_cls.setdefault(sp["concept"], []).append(sp)
                targets = {c: paint(by_cls.get(c, []), offs, len(ids))
                           for c in classes}
                g.write(json.dumps({
                    "example_id": r["example_id"], "text": r["text"],
                    "token_ids": ids, "n_tokens": len(ids),
                    "nat_split": r.get("nat_split"), "slice": r.get("slice"),
                    "cls_mined": r.get("cls"), "surface": r.get("surface"),
                    "targets": targets}) + "\n")
                n_rows += 1
        print(f"{fam}: {n_rows} eval rows ({len(classes)} classes) -> {OUT}/{fam}.jsonl")


if __name__ == "__main__":
    main()
