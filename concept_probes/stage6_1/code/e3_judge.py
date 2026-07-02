"""Stage 6.1 E3 (judging half): score steered generations with mercury-2,
reusing the Stage-4 judging stack (task.md §6.1.5).

REUSE: the OpenRouter client is IMPORTED from stage4/code/or_client.py
(ORClient: async concurrency, retries/backoff, sliding-window rate limits,
full audit trail, cost cap, _safe_json). The Stage-4 judge.py resume trick —
reuse any successful call found in the audit logs, keyed by (rubric variant,
exact batch of example keys) — is mirrored here with E3 keys. judge.py itself
is NOT imported: its prompt build (promptlib family YAML + span rubric) and
its char-span aggregation are Stage-4-specific; E3 has a different rubric
contract (3 x 0/1/2 subscores per generation, no spans).

Judging protocol (frozen in prompts/e3_rubrics.md):
- model inception/mercury-2 via OpenRouter (OPENROUTER_API_KEY in repo .env);
- K=3 self-consistency: each batch is judged 3 times, once per paraphrased
  rubric-variant v1/v2/v3, all three rubrics (incorporation / prefix
  topicality / fluency, each 0/1/2) in one call, strict JSON schema
  (constrained decoding — same anti-drift device as Stage 4);
- the judge is blind to arm/dose/layer: it sees only PREFIX + CONTINUATION;
- batches of --passages-per-call (default 6) generations, all sharing one
  concept (the rubric names the concept).

Outputs under <out>/e3/:
- calls/judge_e3.openrouter.jsonl   full per-call audit (or_client format)
- judged_e3.jsonl                   generation row + judge samples + means
- e3_scores.json                    per-concept aggregation:
    per (arm, factor): mean subscores (0-2) and overall = harmonic mean of
    the three subscore means normalized to [0,1] — computed separately on the
    'selection' and 'heldout' splits; best factor per arm is argmax of the
    SELECTION-split overall; the reported number is the HELDOUT-split overall
    at that factor, plus its delta vs the alpha=0 baseline heldout overall
    (deltas-vs-baseline per §6.1.5: base models are disfluent at temp 1.0).

--smoke: MockClient (no aiohttp, no API, no .env) returning deterministic
scores; exercises batching, caching, parsing, aggregation end to end.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import json
import re
import statistics
import sys
import time
import zlib
from pathlib import Path

from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parent
STAGE_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(STAGE_DIR.parent / "stage4" / "code"))

RUBRICS_PATH = STAGE_DIR / "prompts" / "e3_rubrics.md"
VARIANTS = ("v1", "v2", "v3")
JUDGE_MODEL = "inception/mercury-2"
TEMPERATURE = 0.4          # >0 so the K samples actually vary (Stage-4 value)
MAX_TOKENS = 3000
SUBS = ("incorporation", "topicality", "fluency")


def row_key(r: dict) -> str:
    return f'{r["concept"]}|{r["layer"]}|{r["arm"]}|{r["factor"]}|{r["prefix_id"]}'


# ------------------------------------------------------------------ prompts
def load_sections(path: Path) -> dict:
    """---NAME--- sectioned file, same convention/parser as Stage-4
    promptlib._load_sections (copied: importing promptlib would drag in the
    family-YAML machinery E3 does not use)."""
    parts = re.split(r"^---([A-Za-z0-9_ ]+)---\s*$", path.read_text(), flags=re.M)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        out[parts[i].strip()] = parts[i + 1].strip()
    return out


def judge_schema() -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": ["passages"],
            "properties": {"passages": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["id", "thought", *SUBS],
                "properties": {
                    "id": {"type": "integer"},
                    "thought": {"type": "string"},
                    **{s: {"type": "integer", "minimum": 0, "maximum": 2}
                       for s in SUBS}}}}}}


def build_messages(sections: dict, concept: str, batch: list[dict],
                   variant: str) -> list[dict]:
    fills = {"CONCEPT": concept.replace("_", " "),
             "N_PASSAGES": str(len(batch))}
    rubric = "\n\n".join(
        sections[f"{name}_{variant}"]
        for name in ("INCORPORATION", "TOPICALITY", "FLUENCY"))
    fmt = sections["FORMAT"]
    for k, v in fills.items():
        rubric = rubric.replace("{" + k + "}", v)
        fmt = fmt.replace("{" + k + "}", v)
    body = "\n\n".join(
        f'PASSAGE {i}\nPREFIX: {r["prefix"]}\nCONTINUATION: {r["continuation"]}'
        for i, r in enumerate(batch))
    return [{"role": "system", "content": sections["SYSTEM"]},
            {"role": "user", "content":
             rubric + "\n\n## Passages\n" + body + "\n\n" + fmt}]


# ------------------------------------------------------------------- mock
class MockClient:
    """--smoke stand-in for ORClient: deterministic scores, no network."""

    def __init__(self):
        self.calls = 0
        self.cost = 0.0
        self.prompt_tokens = self.completion_tokens = self.errors = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def chat_json(self, model, messages, temperature, max_tokens,
                        meta=None, response_schema=None, extra_body=None):
        self.calls += 1
        keys = (meta or {}).get("row_keys") or ()
        variant = (meta or {}).get("variant", "v1")
        passages = [{"id": i, "thought": "mock",
                     **{s: zlib.crc32(f"{k}|{variant}|{s}".encode()) % 3
                        for s in SUBS}}
                    for i, k in enumerate(keys)]
        return f"mock{self.calls:06d}", {"passages": passages}

    def stats(self):
        return {"calls": self.calls, "cost_usd": 0.0, "errors": 0}


# ------------------------------------------------------------------ scoring
def hmean01(vals: list[float]) -> float:
    """Harmonic mean of subscore means normalized to [0,1]; 0 if any is 0."""
    xs = [v / 2.0 for v in vals]
    if any(x <= 0 for x in xs):
        return 0.0
    return len(xs) / sum(1.0 / x for x in xs)


def summarize(rows: list[dict]) -> dict | None:
    """Mean subscores (0-2) + overall harmonic mean over judged rows."""
    rows = [r for r in rows if r["judge"]["n_valid"] > 0]
    if not rows:
        return None
    means = {s: statistics.fmean(r["judge"]["mean"][s] for r in rows)
             for s in SUBS}
    return {**{s: round(means[s], 4) for s in SUBS},
            "overall": round(hmean01([means[s] for s in SUBS]), 4),
            "n": len(rows)}


def aggregate(rows: list[dict]) -> dict:
    by_concept: dict = collections.defaultdict(list)
    for r in rows:
        by_concept[r["concept"]].append(r)
    out = {}
    for concept, rs in sorted(by_concept.items()):
        fam, layer = rs[0]["family"], rs[0]["layer"]
        cells: dict = collections.defaultdict(list)
        for r in rs:
            cells[(r["arm"], r["factor"], r["split"])].append(r)
        base_sel = summarize(cells.get(("baseline", 0.0, "selection"), []))
        base_held = summarize(cells.get(("baseline", 0.0, "heldout"), []))
        arms = {}
        for arm in sorted({r["arm"] for r in rs} - {"baseline"}):
            factors = sorted({r["factor"] for r in rs if r["arm"] == arm})
            per_sel = {str(f): summarize(cells.get((arm, f, "selection"), []))
                       for f in factors}
            scored = [(f, per_sel[str(f)]["overall"]) for f in factors
                      if per_sel[str(f)]]
            if not scored:
                continue
            best = max(scored, key=lambda t: t[1])[0]
            held = summarize(cells.get((arm, best, "heldout"), []))
            arms[arm] = {
                "factors": factors, "per_factor_selection": per_sel,
                "best_factor": best,
                "selection_at_best": per_sel[str(best)],
                "heldout_at_best": held,
                "delta_overall_vs_baseline_heldout":
                    (round(held["overall"] - base_held["overall"], 4)
                     if held and base_held else None)}
        out[concept] = {"family": fam, "layer": layer,
                        "baseline": {"selection": base_sel,
                                     "heldout": base_held},
                        "arms": arms}
    return out


# --------------------------------------------------------------------- run
async def run(args) -> None:
    out_dir = Path(args.out)
    e3_dir = out_dir / "e3"
    gen_files = sorted(glob.glob(str(e3_dir / "generations_*.jsonl")))
    if not gen_files:
        sys.exit(f"FATAL: no generations_*.jsonl under {e3_dir}")
    rows = []
    fam_filter = set(args.families.split(",")) if args.families else None
    cls_filter = ({c.replace(" ", "_") for c in args.classes.split(",")}
                  if args.classes else None)
    for gf in gen_files:
        for l in open(gf):
            r = json.loads(l)
            if fam_filter and r["family"] not in fam_filter:
                continue
            if cls_filter and r["concept"] not in cls_filter:
                continue
            rows.append(r)
    # deterministic order => deterministic batches => resumable cache keys
    rows.sort(key=lambda r: (r["family"], r["concept"], r["layer"], r["arm"],
                             r["factor"], r["prefix_id"]))
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} generations from {len(gen_files)} files")

    sections = load_sections(RUBRICS_PATH)
    by_concept: dict = collections.defaultdict(list)
    for r in rows:
        by_concept[r["concept"]].append(r)
    batches = []
    for concept in sorted(by_concept):
        cr = by_concept[concept]
        batches += [(concept, cr[i:i + args.passages_per_call])
                    for i in range(0, len(cr), args.passages_per_call)]

    # resume cache: reuse successful calls from prior audit logs
    # (Stage-4 judge.py mechanic, keyed on variant + exact row-key batch)
    cached = {}
    if not args.smoke:
        from or_client import _safe_json
        for lp in glob.glob(str(e3_dir / "calls" / "judge_e3*.jsonl")):
            for l in open(lp):
                rec = json.loads(l)
                m = rec.get("meta") or {}
                obj = (_safe_json(rec["raw_response"])
                       if rec.get("raw_response") else None)
                if m.get("variant") and obj and obj.get("passages"):
                    cached[(m["variant"], tuple(m.get("row_keys") or ()))] = \
                        (rec["call_id"], obj)
        if cached:
            print(f"resume: {len(cached)} cached judge calls")

    if args.smoke:
        client = MockClient()
    else:
        from or_client import ORClient
        client = ORClient(
            str(e3_dir / "calls" / "judge_e3.openrouter.jsonl"),
            concurrency=args.concurrency, max_retries=6, timeout=180,
            cost_cap_usd=args.cap_usd)
    schema = judge_schema()
    capped: list[str] = []

    async def one(bi: int, concept: str, batch: list[dict], variant: str):
        keys = tuple(row_key(r) for r in batch)
        if (variant, keys) in cached:
            return bi, variant, cached[(variant, keys)][1]
        msgs = build_messages(sections, concept, batch, variant)
        meta = {"variant": variant, "concept": concept,
                "row_keys": list(keys)}
        try:
            _, obj = await client.chat_json(
                JUDGE_MODEL, msgs, TEMPERATURE, MAX_TOKENS,
                meta=meta, response_schema=schema)
        except RuntimeError as e:      # cost cap — record and stop this task
            if not capped:
                capped.append(str(e))
                print(f"COST CAP: {e} (remaining tasks return unjudged)")
            return bi, variant, None
        return bi, variant, obj

    results: dict = collections.defaultdict(dict)
    async with client:
        tasks = [asyncio.ensure_future(one(bi, c, b, v))
                 for bi, (c, b) in enumerate(batches) for v in VARIANTS]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                        desc="judge calls"):
            bi, variant, obj = await fut
            results[bi][variant] = obj
    stats = client.stats()
    print(f"judge stats: {stats}")

    # ------------------------------------------------ per-row samples/means
    n_bad = 0
    judged_path = e3_dir / "judged_e3.jsonl"
    judged_rows = []
    with judged_path.open("w") as f:
        for bi, (concept, batch) in enumerate(batches):
            per_variant = {}
            for v in VARIANTS:
                obj = results[bi].get(v)
                m = {}
                for p in (obj or {}).get("passages") or []:
                    if isinstance(p, dict):
                        try:
                            m[int(p["id"])] = {
                                s: max(0, min(2, int(p[s]))) for s in SUBS}
                        except (KeyError, TypeError, ValueError):
                            continue
                if not m:
                    n_bad += 1
                per_variant[v] = m
            for i, r in enumerate(batch):
                samples = {v: per_variant[v].get(i) for v in VARIANTS}
                valid = [s for s in samples.values() if s]
                mean = ({s: round(statistics.fmean(x[s] for x in valid), 4)
                         for s in SUBS} if valid else None)
                rec = dict(r)
                rec["judge"] = {"samples": samples, "mean": mean,
                                "n_valid": len(valid)}
                judged_rows.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    scores = {
        "config": {"model": JUDGE_MODEL, "k": len(VARIANTS),
                   "temperature": TEMPERATURE,
                   "rubrics": str(RUBRICS_PATH.relative_to(STAGE_DIR)),
                   "passages_per_call": args.passages_per_call,
                   "smoke": bool(args.smoke),
                   "generated_files": [Path(g).name for g in gen_files],
                   "judge_stats": stats, "bad_variant_batches": n_bad,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "concepts": aggregate(judged_rows)}
    (e3_dir / "e3_scores.json").write_text(json.dumps(scores, indent=1))
    print(f"wrote {judged_path} ({len(judged_rows)} rows) and "
          f"{e3_dir / 'e3_scores.json'} ({len(scores['concepts'])} concepts)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", default=None, help="csv filter")
    ap.add_argument("--classes", default=None, help="csv filter")
    ap.add_argument("--out", default=str(STAGE_DIR / "out"),
                    help="stage6_1 out dir holding e3/generations_*.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only the first N generations (smoke/pilot)")
    ap.add_argument("--cap-usd", type=float, default=20.0)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--passages-per-call", type=int, default=6)
    ap.add_argument("--smoke", action="store_true",
                    help="mock judge, no API calls")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
