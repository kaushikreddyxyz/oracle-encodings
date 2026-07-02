"""Judge generated passages with mercury-2: K=3 self-consistency, one paraphrased
rubric variant per sample (v1/v2/v3), family-level span scoring on the 0-6 scale.

Reads  data/<family>/raw_gen/generations.jsonl  (flag==null records only)
Writes data/<family>/calls/judge.jsonl          (full audit per call)
       data/<family>/judged/judged.jsonl        (per example: 3 raw samples +
                                                 char-level aggregated spans)
       data/<family>/judged/judge_report.json

Aggregation (per example, per concept): each sample's spans paint their 0-6 score
onto character positions (score/6). Per-char target = mean over the K samples
(absent = 0). Maximal nonzero char runs become aggregated spans; span strength =
mean per-char value inside the run. Quote-to-offset mapping requires an exact
substring match (case-sensitive first, then insensitive); unmatched quotes are
kept in the raw sample but excluded from aggregation and counted in the report.
"""
import argparse
import asyncio
import collections
import json
import os

import promptlib
from or_client import ORClient

STAGE4 = promptlib.STAGE4
VARIANTS = ("v1", "v2", "v3")


def judge_schema(class_names):
    """Strict structured-output schema: constrained decoding removes format drift
    (array-wrapped objects, capitalization drift in concept names, etc.)."""
    return {"type": "object", "additionalProperties": False,
            "required": ["passages"],
            "properties": {"passages": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["id", "thought", "spans"],
                "properties": {
                    "id": {"type": "integer"},
                    "thought": {"type": "string"},
                    "spans": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["concept", "quote", "score"],
                        "properties": {
                            "concept": {"type": "string", "enum": list(class_names)},
                            "quote": {"type": "string"},
                            "score": {"type": "integer", "minimum": 0, "maximum": 6},
                        }}}}}}}}


def locate(text, quote):
    if quote is None or isinstance(quote, (dict, list)):
        return None
    quote = str(quote)
    if not quote:
        return None
    i = text.find(quote)
    if i < 0:
        i = text.lower().find(quote.lower())
    return None if i < 0 else (i, i + len(quote))


def aggregate(text, samples, classes, k):
    """samples: list of span lists [{concept, quote, score}]. Returns per-concept
    aggregated spans [{concept, char_span, strength}] using char-level mean over k."""
    per = {}
    for spans in samples:
        painted = {}
        for sp in spans:
            c = sp.get("concept", "").strip().lower()
            if c not in classes:
                continue
            loc = sp.get("_loc")
            if not loc:
                continue
            try:
                sc = max(0.0, min(6.0, float(sp.get("score", 0)))) / 6.0
            except (TypeError, ValueError):
                continue
            arr = painted.setdefault(c, [0.0] * len(text))
            for i in range(loc[0], min(loc[1], len(text))):
                arr[i] = max(arr[i], sc)
        for c, arr in painted.items():
            acc = per.setdefault(c, [0.0] * len(text))
            for i, v in enumerate(arr):
                acc[i] += v
    out = []
    for c, acc in per.items():
        vals = [v / k for v in acc]
        i = 0
        while i < len(vals):
            if vals[i] > 0:
                j = i
                while j < len(vals) and vals[j] > 0:
                    j += 1
                seg = vals[i:j]
                out.append({"concept": c, "char_span": [i, j],
                            "strength": round(sum(seg) / len(seg), 4)})
                i = j
            else:
                i += 1
    return out


async def run(family, cap_usd, limit=None, tag=""):
    sfx = f"_{tag}" if tag else ""
    cfg = promptlib.load_all(family)
    jd = cfg["models"]["judge"]
    rt = cfg["models"]["runtime"]
    classes = {c.lower() for c in cfg["pack"]["classes"]}
    base = os.path.join(STAGE4, "data", family)
    gens = [json.loads(l) for l in
            open(os.path.join(base, "raw_gen", f"generations{sfx}.jsonl"))]
    todo = [g for g in gens if g["flag"] is None][: (limit or None)]
    os.makedirs(os.path.join(base, "judged"), exist_ok=True)

    chunk = jd["passages_per_call"]
    batches = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]

    # resume: reuse successful judge calls from ANY provider's audit log (keyed by
    # variant + exact example-id batch; batching is deterministic per input file)
    import glob as _glob
    from or_client import _safe_json
    cached = {}
    for lp in _glob.glob(os.path.join(base, "calls", f"judge{sfx}*.jsonl")):
        for l in open(lp):
            rec = json.loads(l)
            m = rec.get("meta") or {}
            obj = _safe_json(rec["raw_response"]) if rec.get("raw_response") else None
            if m.get("variant") and obj and obj.get("passages"):
                cached[(m["variant"], tuple(m.get("example_ids") or ()))] = \
                    (rec["call_id"], obj)

    providers = jd.get("providers") or [{
        "name": "openrouter", "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY", "model": jd["slug"],
        "concurrency": rt["concurrency"]}]
    clients = []
    for pv in providers:
        if pv.get("cap_usd") == "auto" and pv["name"] == "openrouter":
            # spend down the live remaining balance, leaving a small buffer
            import urllib.request
            from dotenv import load_dotenv
            load_dotenv(os.path.join(promptlib.STAGE4, "..", "..", ".env"))
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/credits",
                headers={"Authorization": f"Bearer {os.environ[pv['key_env']]}"})
            d = json.load(urllib.request.urlopen(req, timeout=15))["data"]
            pv = dict(pv)
            pv["cap_usd"] = max(0.0, d["total_credits"] - d["total_usage"] - 0.75)
            print(f"openrouter auto-cap: ${pv['cap_usd']:.2f} remaining budget")
        lp = os.path.join(base, "calls", f"judge{sfx}.{pv['name']}.jsonl")
        cl = ORClient(
            lp, concurrency=pv.get("concurrency", rt["concurrency"]),
            max_retries=rt["max_retries"], timeout=rt["timeout_s"],
            cost_cap_usd=pv.get("cap_usd", cap_usd), endpoint=pv["endpoint"],
            key_env=pv["key_env"], pricing=pv.get("pricing_per_mtok"),
            provider=pv["name"], limits=pv.get("limits"))
        cl.conc = pv.get("concurrency", rt["concurrency"])
        clients.append((pv, cl))

    schema = judge_schema(sorted(classes))
    try:
        for _, cl in clients:
            await cl.__aenter__()

        async def one(batch, variant, start_idx):
            key = (variant, tuple(g["example_id"] for g in batch))
            if key in cached:
                call_id, obj = cached[key]
                return call_id, variant, obj
            passages = [{"id": i, "text": g["text"]} for i, g in enumerate(batch)]
            msgs = promptlib.build_judge_prompt(family, passages, variant, cfg)
            meta = {"variant": variant, "n_passages": len(batch),
                    "example_ids": [g["example_id"] for g in batch]}
            last = (None, None)
            # ADAPTIVE provider choice: start with whichever provider has the
            # largest free-capacity fraction right now (work-sharing), fall back
            # to the others on failure/cap. A fixed round-robin pins resumed
            # leftover work to the slower provider and starves the fast one.
            order = [c for _, c in sorted(
                clients, key=lambda pc: pc[1].sem._value / pc[1].conc,
                reverse=True)]
            _ = start_idx  # kept for cache-key stability of the task signature
            for cl in order:
                model, mt = pv_params(cl)
                try:
                    call_id, obj = await cl.chat_json(
                        model, msgs, jd["temperature"], mt,
                        meta=meta, response_schema=schema)
                except RuntimeError:   # provider cost cap hit — try the next one
                    continue
                if obj and obj.get("passages"):
                    return call_id, variant, obj
                last = (call_id, obj)
            return last[0], variant, last[1]

        def pv_params(cl):
            return next((pv["model"], pv.get("max_tokens", jd["max_tokens"]))
                        for pv, c in clients if c is cl)

        tasks = [one(b, v, i * len(VARIANTS) + vi)
                 for i, b in enumerate(batches) for vi, v in enumerate(VARIANTS)]
        raw = await asyncio.gather(*tasks)
        stats = {"calls": sum(c.calls for _, c in clients),
                 "cost_usd": round(sum(c.cost for _, c in clients), 5),
                 "prompt_tokens": sum(c.prompt_tokens for _, c in clients),
                 "completion_tokens": sum(c.completion_tokens for _, c in clients),
                 "errors": sum(c.errors for _, c in clients),
                 "per_provider": {pv["name"]: c.stats() for pv, c in clients}}
    finally:
        for _, cl in clients:
            await cl.__aexit__()

    # regroup: batch index -> {variant: (call_id, parsed)}
    per_batch = collections.defaultdict(dict)
    flat = [(bi, v) for bi in range(len(batches)) for v in VARIANTS]
    for (bi, v), (call_id, variant, obj) in zip(flat, raw):
        per_batch[bi][variant] = (call_id, obj)

    n_unmatched = n_badcall = 0
    out_path = os.path.join(base, "judged", f"judged{sfx}.jsonl")
    kept = 0
    with open(out_path, "w") as f:
        for bi, batch in enumerate(batches):
            parsed = {}   # variant -> {passage_id: spans}
            call_ids = {}
            for v in VARIANTS:
                call_id, obj = per_batch[bi][v]
                call_ids[v] = call_id
                m = {}
                for p in (obj or {}).get("passages") or []:
                    if not isinstance(p, dict):
                        continue
                    try:
                        m[int(p.get("id"))] = p.get("spans") or []
                    except (TypeError, ValueError):
                        continue
                if not m:
                    n_badcall += 1
                parsed[v] = m
            for i, g in enumerate(batch):
                samples = []
                raw_samples = {}
                for v in VARIANTS:
                    spans = parsed[v].get(i, [])
                    for sp in spans:
                        if isinstance(sp, dict):
                            sp["_loc"] = locate(g["text"], sp.get("quote"))
                            if sp["_loc"] is None and sp.get("quote"):
                                n_unmatched += 1
                    samples.append([sp for sp in spans if isinstance(sp, dict)])
                    raw_samples[v] = [{k: sp.get(k) for k in ("concept", "quote", "score")}
                                      for sp in spans if isinstance(sp, dict)]
                agg = aggregate(g["text"], samples, classes, k=len(VARIANTS))
                rec = dict(g)
                rec.update({"judge_call_ids": call_ids, "judge_samples": raw_samples,
                            "aggregated_spans": agg})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    report = {"family": family, "examples_judged": kept,
              "judge_calls": stats["calls"], "bad_judge_calls": n_badcall,
              "unmatched_quotes": n_unmatched, "or_stats": stats}
    with open(os.path.join(base, "judged", f"judge_report{sfx}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--cap-usd", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only the first N ok generations (smoke test)")
    ap.add_argument("--tag", default="", help="suffix matching generate.py --tag")
    args = ap.parse_args()
    asyncio.run(run(args.family, args.cap_usd, args.limit, args.tag))
