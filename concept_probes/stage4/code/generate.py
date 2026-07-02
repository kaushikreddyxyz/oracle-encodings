"""Run the materialized generation prompts through the generator model, with
quota top-up: flagged/degenerate/short-fall items trigger resampling rounds per
(class, slice) until the requested item count is met (or topup_rounds exhausted).

Reads  data/<family>/prompts/gen_prompts.jsonl  (audited before this runs)
Writes data/<family>/calls/gen.jsonl            (full audit: request+response per call)
       data/<family>/raw_gen/generations.jsonl  (one record per generated item,
                                                 with provenance + validity flags)
       data/<family>/raw_gen/gen_report.json    (counts, flag rates, cost)

Degeneracy defenses, layered:
  prompt     — FORMAT block forbids filler/empty/repetition (audited text)
  sampling   — temperature 0.9 + strict json_schema constrained decoding
  filter     — flags below; flagged records are kept for audit, excluded downstream
  resample   — top-up rounds regenerate any (class, slice) below quota

Validity flags (records kept, never silently dropped):
  dup_exact               — normalized text already seen (first occurrence wins)
  banned_leak             — implicit item contains a banned surface form
  missing_surface         — hard-negative item lacks the class surface form
  sibling_leak            — neutral item mentions a sibling surface form
  too_short/too_long      — outside 20..900 chars
  degenerate_repetitive   — zlib compression ratio < 0.35 (repetition collapse)
  degenerate_lowalpha     — < 55% alphabetic+space chars (filler/punctuation soup)
  degenerate_filler       — whitespace runs / ellipsis padding / same-char runs
"""
import argparse
import asyncio
import collections
import json
import os
import re
import zlib

import promptlib
from or_client import ORClient, _safe_json

STAGE4 = promptlib.STAGE4

GEN_SCHEMA = {"type": "object", "additionalProperties": False,
              "required": ["items"],
              "properties": {"items": {"type": "array", "items": {
                  "type": "object", "additionalProperties": False,
                  "required": ["text"],
                  "properties": {"text": {"type": "string"}}}}}}


def _norm(t):
    return re.sub(r"\W+", " ", t.lower()).strip()


def _degenerate(text):
    if len(text) >= 60 and len(zlib.compress(text.encode())) / len(text) < 0.35:
        return "degenerate_repetitive"
    if sum(c.isalpha() or c == " " for c in text) / len(text) < 0.55:
        return "degenerate_lowalpha"
    if re.search(r"\s{5,}", text) or text.count("...") >= 3 \
            or re.search(r"(.)\1{7,}", text):
        return "degenerate_filler"
    return None


def flag(text, rec, pack):
    cls_cfgs = pack["classes"]
    if len(text) < 20:
        return "too_short"
    if len(text) > 900:
        return "too_long"
    deg = _degenerate(text)
    if deg:
        return deg
    if rec["slice"] == "implicit":
        for rx in cls_cfgs[rec["class"]].get("banned_in_implicit", []):
            if re.search(rx, text, re.I):
                return "banned_leak"
    if rec["slice"] == "hard_negative":
        if rec.get("hazard_form"):
            if rec["hazard_form"].lower() not in text.lower():
                return "missing_surface"
        else:
            surfs = cls_cfgs[rec["class"]]["form_train"] + cls_cfgs[rec["class"]]["form_test"]
            if not any(s.lower() in text.lower() for s in surfs):
                return "missing_surface"
    if rec["slice"] == "explicit_form":
        if rec.get("form", "").lower() not in text.lower():
            return "missing_surface"
        for s in cls_cfgs[rec["class"]]["form_train"]:
            if re.search(r"(?<!\w)" + re.escape(s.rstrip(".")) + r"(?!\w)", text, re.I):
                return "banned_leak"
    if rec["slice"] == "neutral_nearmiss":
        for cc in cls_cfgs.values():
            for s in cc["form_train"] + cc["form_test"]:
                if re.search(r"\b" + re.escape(s.lower().rstrip(".")) + r"\b",
                             text.lower()):
                    return "sibling_leak"
    return None


def parse_call(p, call_id, obj, pack, seen, results):
    items = (obj or {}).get("items") or []
    for j, it in enumerate(items):
        t = it.get("text") if isinstance(it, dict) else it
        text = t.strip() if isinstance(t, str) else ""
        if not text:
            continue
        fl = flag(text, p, pack)
        if fl is None and _norm(text) in seen:
            fl = "dup_exact"
        if fl is None:
            seen.add(_norm(text))
        results.append({
            "example_id": f"{p['prompt_id']}_{j:02d}",
            "family": p["family"], "class": p["class"], "slice": p["slice"],
            "hazard": p.get("hazard"), "level": p.get("level"),
            "template_id": p["template_id"],
            "prompt_id": p["prompt_id"], "gen_call_id": call_id,
            "text": text, "flag": fl,
        })


async def run(family, cap_usd, tag=""):
    sfx = f"_{tag}" if tag else ""
    cfg = promptlib.load_all(family)
    gen = cfg["models"]["generator"]
    rt = cfg["models"]["runtime"]
    base = os.path.join(STAGE4, "data", family)
    prompts = [json.loads(l) for l in
               open(os.path.join(base, "prompts", f"gen_prompts{sfx}.jsonl"))]
    os.makedirs(os.path.join(base, "raw_gen"), exist_ok=True)

    # quota key includes level so intensity top-ups preserve per-level balance
    quota = collections.Counter()
    by_key = collections.defaultdict(list)
    for p in prompts:
        key = (p["class"], p["slice"], p.get("level"))
        quota[key] += p["n_items"]
        by_key[key].append(p)

    # resume: reuse successful calls already in the audit log (never re-buy)
    cached = {}
    log_path = os.path.join(base, "calls", f"gen{sfx}.jsonl")
    if os.path.exists(log_path):
        for l in open(log_path):
            rec = json.loads(l)
            pid = (rec.get("meta") or {}).get("prompt_id")
            obj = _safe_json(rec["raw_response"]) if rec.get("raw_response") else None
            if pid and obj and obj.get("items"):
                cached[pid] = (rec["call_id"], obj)

    extra = ({"reasoning": {"effort": gen["reasoning_effort"]}}
             if gen.get("reasoning_effort") else None)
    seen, results = set(), []
    per_call = gen["items_per_call"]
    topup_rounds = int(gen.get("topup_rounds", 0))

    async with ORClient(log_path,
                        concurrency=rt["concurrency"], max_retries=rt["max_retries"],
                        timeout=rt["timeout_s"], cost_cap_usd=cap_usd) as client:
        async def one(p):
            if p["prompt_id"] in cached:
                call_id, obj = cached[p["prompt_id"]]
                return p, call_id, obj
            call_id, obj = await client.chat_json(
                gen["slug"], p["messages"], gen["temperature"], gen["max_tokens"],
                meta={"prompt_id": p["prompt_id"], "class": p["class"],
                      "slice": p["slice"]}, extra_body=extra,
                response_schema=GEN_SCHEMA)
            return p, call_id, obj

        todo = prompts
        for rnd in range(topup_rounds + 1):
            n_cached = sum(1 for p in todo if p["prompt_id"] in cached)
            print(f"round {rnd}: {len(todo)} calls ({n_cached} cached)", flush=True)
            done = await asyncio.gather(*[one(p) for p in todo])
            for p, call_id, obj in done:
                parse_call(p, call_id, obj, cfg["pack"], seen, results)
            ok = collections.Counter((r["class"], r["slice"], r.get("level"))
                                     for r in results if r["flag"] is None)
            shortfall = {k: quota[k] - ok[k] for k in quota if quota[k] > ok[k]}
            if not shortfall or rnd == topup_rounds:
                if shortfall:
                    print(f"quota shortfall after {rnd} top-ups: "
                          f"{ {'/'.join(str(x) for x in k if x is not None): v for k, v in shortfall.items()} }",
                          flush=True)
                break
            todo = []
            for key, missing in shortfall.items():
                src = by_key[key]
                i = 0
                while missing > 0:
                    tmpl = src[i % len(src)]
                    n = min(per_call, missing)
                    clone = dict(tmpl)
                    clone["prompt_id"] = f"{tmpl['prompt_id']}-r{rnd + 1}n{i:02d}"
                    clone["n_items"] = n  # messages still ask for the original N;
                    # over-delivery is fine — quota is enforced on parsed items
                    todo.append(clone)
                    missing -= per_call
                    i += 1
        stats = client.stats()

    with open(os.path.join(base, "raw_gen", f"generations{sfx}.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    okr = [r for r in results if r["flag"] is None]
    by = collections.Counter((r["class"], r["slice"]) for r in okr)
    by_lvl = collections.Counter((r["class"], r["slice"], r.get("level")) for r in okr)
    flags = collections.Counter(r["flag"] for r in results if r["flag"])
    report = {"family": family, "items_total": len(results), "items_ok": len(okr),
              "flags": dict(flags),
              "quota_met": all(by_lvl[k] >= quota[k] for k in quota),
              "quota_shortfall": {"/".join(str(x) for x in k if x is not None):
                                  quota[k] - by_lvl[k]
                                  for k in quota if by_lvl[k] < quota[k]},
              "ok_by_class_slice": {f"{c}/{s}": n for (c, s), n in sorted(by.items())},
              "or_stats": stats}
    with open(os.path.join(base, "raw_gen", f"gen_report{sfx}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--cap-usd", type=float, default=30.0)
    ap.add_argument("--tag", default="", help="suffix for prompts/outputs (e.g. forms)")
    args = ap.parse_args()
    asyncio.run(run(args.family, args.cap_usd, args.tag))
