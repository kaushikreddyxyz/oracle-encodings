"""
build_candidates.py — Stage 1a of the overnight concept-probes run.

Reads ClimbMix parquet shards (config.SHARDS, disjoint from nanochat training),
surface/lemma-matches every PRESENCE concept (word-boundary, case-insensitive,
lexicons straight out of concepts.py) and seeds SCALAR candidate pools, then writes
one JSONL file per concept under `data/candidates/` using the SPEC `candidate` schema:

    {"id":"months::January::s300::000042","concept":"months","cls":"January",
     "regime":"presence","text":"...snippet...","char_span":[120,124],
     "match_surface":"Jan.","shard":300,"external":null}

CPU-only. No GPU/RunPod. Resumable at concept granularity (a non-empty
`data/candidates/{concept}.jsonl` is skipped). Single streaming pass over the shards;
download stops early once every active class has hit MAX_CANDIDATES_PER_CLASS.

Usage:
    python build_candidates.py            # full build over config.SHARDS
    python build_candidates.py --smoke    # one shard, tiny caps, scratch dir
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Path wiring: code/ and overnight_run/ on sys.path, then import shared modules
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent          # .../overnight_run/code
ROOT = HERE.parent                               # .../overnight_run
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config        # noqa: E402
import concepts      # noqa: E402

import pyarrow.parquet as pq      # noqa: E402
import pyarrow as pa              # noqa: E402
from huggingface_hub import hf_hub_download   # noqa: E402

# --------------------------------------------------------------------------- #
# Tunables (a few that are not in config.py)
# --------------------------------------------------------------------------- #
SNIPPET_HALF_WORDS = config.SNIPPET_TOKENS // 2   # ~32 words either side of the match
MAX_SNIPPET_CHARS = 1400                           # clamp pathological no-whitespace docs
PER_DOC_PER_CLASS_LIMIT = 3                        # diversity: cap matches per class per doc
MAX_NUMBER_DIGITS = 9                              # >9 integer digits => likely an ID, skip

# --------------------------------------------------------------------------- #
# SCALAR seed lexicons (keyword-seeded sampling so the judge has gradable text).
# external stays null for these — the judge rates them later.
# --------------------------------------------------------------------------- #
SCALAR_SEEDS = {
    "costliness": [
        "price", "prices", "priced", "cost", "costs", "costly", "expensive",
        "cheap", "cheaper", "cheapest", "affordable", "afford", "pricey", "pricy",
        "bargain", "fortune", "free", "priceless", "lavish", "luxury", "luxurious",
        "dollar", "dollars", "inexpensive", "pennies", "fee", "fees",
    ],
    "physical_size": [
        "tiny", "huge", "enormous", "microscopic", "gigantic", "small", "large",
        "vast", "miniature", "massive", "colossal", "immense", "minuscule", "big",
        "little", "giant", "towering", "dwarf", "sprawling", "compact",
    ],
    "lovingness": [
        "love", "loved", "loves", "loving", "adore", "adored", "cherish",
        "cherished", "despise", "despised", "loathe", "loathed", "hate", "hated",
        "affection", "devoted", "beloved", "treasure", "detest", "adoration",
    ],
    "duration": [
        "instant", "instantaneous", "moment", "momentary", "brief", "briefly",
        "fleeting", "forever", "eternal", "eternity", "millennia", "millennium",
        "centuries", "ages", "perpetual", "everlasting", "transient", "ephemeral",
        "prolonged", "lasting", "split-second",
    ],
    "harmfulness": [
        "harmless", "benign", "dangerous", "deadly", "lethal", "catastrophic",
        "safe", "hazardous", "toxic", "harmful", "fatal", "destructive",
        "perilous", "innocuous", "devastating", "ruinous", "noxious",
    ],
    "indoors": [
        "indoor", "indoors", "inside", "interior", "hallway", "basement",
        "bedroom", "kitchen", "office", "lobby", "corridor", "cellar",
        "living room", "indoor arena",
    ],
    "outdoors": [
        "outdoor", "outdoors", "outside", "wilderness", "open-air", "meadow",
        "forest", "mountainside", "prairie", "campsite", "trail", "backyard",
        "open field", "wild",
    ],
    "europe": [
        "Europe", "European", "France", "Germany", "Italy", "Spain", "Portugal",
        "Greece", "Norway", "Sweden", "Finland", "Denmark", "Netherlands",
        "Belgium", "Austria", "Switzerland", "Poland", "Hungary", "Ireland",
        "Scotland", "England", "Britain", "Romania", "Paris", "Berlin", "Rome",
        "Madrid", "Lisbon", "Athens", "Vienna", "Amsterdam", "Prague", "Warsaw",
        "Budapest", "London", "Dublin", "Stockholm", "Oslo", "Copenhagen",
    ],
    "america": [
        "America", "American", "United States", "USA", "Canada", "Mexico",
        "Brazil", "Argentina", "Chile", "Peru", "Colombia", "Venezuela",
        "Bolivia", "Ecuador", "Uruguay", "New York", "Los Angeles", "Chicago",
        "Toronto", "Houston", "Boston", "San Francisco", "Mexico City",
        "Buenos Aires", "Rio de Janeiro", "Lima", "Bogota", "Quebec",
    ],
    "africa": [
        "Africa", "African", "Nigeria", "Egypt", "Kenya", "Ethiopia", "Ghana",
        "Morocco", "Algeria", "Tunisia", "Tanzania", "Uganda", "Senegal",
        "Sudan", "Somalia", "Zimbabwe", "Zambia", "Angola", "Cameroon", "Cairo",
        "Lagos", "Nairobi", "Johannesburg", "Cape Town", "Accra", "Casablanca",
        "Addis Ababa", "Dakar", "Khartoum", "South Africa",
    ],
}
# Extra non-alnum patterns appended per scalar (e.g. dollar amounts for costliness).
SCALAR_EXTRA_PATTERNS = {
    "costliness": [r"\$\s?\d[\d,]*(?:\.\d+)?"],
}

# --------------------------------------------------------------------------- #
# Regex builders
# --------------------------------------------------------------------------- #
def _form_to_subpattern(form: str) -> str:
    """Turn one lexicon surface form into an anchored regex sub-pattern.

    - internal spaces/hyphens become a flexible [-\\s]+ separator
      ("red-orange" and "red orange" -> one pattern),
    - a leading \\b only if the form starts with an alnum char,
    - a trailing \\b only if the form ends with an alnum char
      (so "jan." matches "Jan. 2019" — a trailing \\b after '.' would fail).
    """
    form = form.strip()
    parts = re.split(r"[-\s]+", form)
    body = r"[-\s]+".join(re.escape(p) for p in parts if p)
    left = r"\b" if form[:1].isalnum() else ""
    right = r"\b" if form[-1:].isalnum() else ""
    return left + body + right


def compile_lexicon(forms) -> re.Pattern:
    """Combined case-insensitive regex over a class lexicon (longest form first)."""
    ordered = sorted({f.strip() for f in forms if f and f.strip()}, key=len, reverse=True)
    body = "|".join(_form_to_subpattern(f) for f in ordered)
    return re.compile("(?:" + body + ")", re.IGNORECASE)


# Shared digit matcher (integers + decimals) for numbers100 + the "numbers" scalar.
DIGIT_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
WORD_RE = re.compile(r"\S+")


# --------------------------------------------------------------------------- #
# Snippet extraction (char-exact: slice the doc directly so char_span is trivially
# correct, with a clamp for pathological no-whitespace docs)
# --------------------------------------------------------------------------- #
def make_snippet(doc, word_starts, word_ends, ms, me):
    if word_starts:
        wi = bisect.bisect_right(word_starts, ms) - 1
        if wi < 0:
            wi = 0
        lo = max(0, wi - SNIPPET_HALF_WORDS)
        hi = min(len(word_starts) - 1, wi + SNIPPET_HALF_WORDS)
        snip_s, snip_e = word_starts[lo], word_ends[hi]
    else:
        snip_s, snip_e = 0, len(doc)
    if snip_e - snip_s > MAX_SNIPPET_CHARS:
        snip_s = max(snip_s, ms - MAX_SNIPPET_CHARS // 2)
        snip_e = min(snip_e, me + MAX_SNIPPET_CHARS // 2)
    snip_s = min(snip_s, ms)
    snip_e = max(snip_e, me)
    return doc[snip_s:snip_e], ms - snip_s, me - snip_s


# --------------------------------------------------------------------------- #
# Candidate build
# --------------------------------------------------------------------------- #
def detect_text_column(pf: "pq.ParquetFile") -> str:
    sch = pf.schema_arrow
    names = list(sch.names)
    for cand in ("text", "content", "raw_content", "document", "body"):
        if cand in names and (pa.types.is_string(sch.field(cand).type)
                              or pa.types.is_large_string(sch.field(cand).type)):
            return cand
    for name in names:
        t = sch.field(name).type
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            return name
    raise RuntimeError(f"no string column found; schema names={names}")


def iter_shard_docs(shard: int, max_docs: int):
    """Download (cached) one shard and yield its text docs, up to max_docs."""
    path = hf_hub_download(
        config.DATASET, f"shard_{shard:05d}.parquet",
        repo_type="dataset", token=True,
    )
    pf = pq.ParquetFile(path)
    text_col = detect_text_column(pf)
    n = 0
    for batch in pf.iter_batches(batch_size=512, columns=[text_col]):
        col = batch.column(0)
        for v in col:
            if n >= max_docs:
                return
            t = v.as_py()
            if t:
                yield t
            n += 1


def build(out_dir: Path, shards, cap: int, max_docs: int, smoke: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- which concepts still need building (resumability) ----
    presence_files = {c: out_dir / f"{c}.jsonl" for c in concepts.PRESENCE_CONCEPTS}
    scalar_file = out_dir / "scalars.jsonl"

    def done(fp: Path) -> bool:
        return fp.exists() and fp.stat().st_size > 0

    todo_presence = []
    for c, fp in presence_files.items():
        if done(fp):
            print(f"[skip] {c}.jsonl already non-empty")
        else:
            todo_presence.append(c)
    todo_scalars = not done(scalar_file)
    if not todo_scalars:
        print("[skip] scalars.jsonl already non-empty")

    if not todo_presence and not todo_scalars:
        print("[build] nothing to do — all candidate files present. Summarising.")
        summarize(out_dir, cap)
        return

    # ---- matchers ----
    # presence per-class regexes (everything except numbers100, which is value-based)
    class_matchers = []   # (concept, cls, compiled_regex)
    for c in todo_presence:
        if c == "numbers100":
            continue
        for cls, spec in concepts.PRESENCE_CONCEPTS[c]["classes"].items():
            lex = spec.get("lexicon") or []
            if lex:
                class_matchers.append((c, cls, compile_lexicon(lex)))

    numbers100_on = "numbers100" in todo_presence
    numbers100_buckets = list(concepts.PRESENCE_CONCEPTS["numbers100"]["classes"].keys()) \
        if numbers100_on else []

    # scalar matchers
    scalar_seed_matchers = []   # (scalar_name, compiled_regex)
    numbers_scalar_on = False
    if todo_scalars:
        for sname in concepts.SCALARS:
            if sname == "numbers":
                numbers_scalar_on = True
                continue
            forms = list(SCALAR_SEEDS.get(sname, []))
            patterns = [_form_to_subpattern(f) for f in
                        sorted({f.strip() for f in forms if f.strip()}, key=len, reverse=True)]
            patterns += SCALAR_EXTRA_PATTERNS.get(sname, [])
            if patterns:
                scalar_seed_matchers.append(
                    (sname, re.compile("(?:" + "|".join(patterns) + ")", re.IGNORECASE)))

    # ---- output handles (.tmp; atomic-renamed on success) ----
    tmp_suffix = ".tmp"
    handles = {}
    for c in todo_presence:
        handles[c] = open(presence_files[c].with_suffix(".jsonl" + tmp_suffix), "w", encoding="utf-8")
    if todo_scalars:
        handles["__scalars__"] = open(scalar_file.with_suffix(".jsonl" + tmp_suffix), "w", encoding="utf-8")

    counts = defaultdict(int)        # (concept, cls) -> count   (cls=None for scalars)
    counters = defaultdict(int)      # outfile-key -> running id counter

    def emit(outkey, fh, concept, cls, regime, surface, ms, me, doc, ws, we, shard, external):
        snippet, cs, ce = make_snippet(doc, ws, we, ms, me)
        cnum = counters[outkey]
        counters[outkey] += 1
        if cls is not None:
            cid = f"{concept}::{cls}::s{shard}::{cnum:06d}"
        else:
            cid = f"{concept}::s{shard}::{cnum:06d}"
        rec = {
            "id": cid, "concept": concept, "cls": cls, "regime": regime,
            "text": snippet, "char_span": [cs, ce], "match_surface": surface,
            "shard": shard, "external": external,
        }
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[(concept, cls)] += 1

    def all_full() -> bool:
        for c, cls, _ in class_matchers:
            if counts[(c, cls)] < cap:
                return False
        if numbers100_on:
            for b in numbers100_buckets:
                if counts[("numbers100", b)] < cap:
                    return False
        if numbers_scalar_on and counts[("numbers", None)] < cap:
            return False
        for sname, _ in scalar_seed_matchers:
            if counts[(sname, None)] < cap:
                return False
        return True

    t0 = time.time()
    docs_seen = 0
    try:
        for shard in shards:
            print(f"[shard] downloading + scanning shard_{shard:05d} "
                  f"(elapsed {time.time()-t0:.0f}s, docs {docs_seen})")
            try:
                doc_iter = iter_shard_docs(shard, max_docs)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] shard {shard} failed to load: {e!r} — skipping")
                continue

            for doc in doc_iter:
                docs_seen += 1
                ws = we = None  # lazy word spans for this doc
                per_doc = defaultdict(int)

                # --- presence per-class lexicon matches ---
                for c, cls, rgx in class_matchers:
                    if counts[(c, cls)] >= cap:
                        continue
                    fh = handles[c]
                    for m in rgx.finditer(doc):
                        if per_doc[(c, cls)] >= PER_DOC_PER_CLASS_LIMIT:
                            break
                        if counts[(c, cls)] >= cap:
                            break
                        if ws is None:
                            wm = [(w.start(), w.end()) for w in WORD_RE.finditer(doc)]
                            ws = [a for a, _ in wm]
                            we = [b for _, b in wm]
                        per_doc[(c, cls)] += 1
                        emit(c, fh, c, cls, "presence", m.group(0),
                             m.start(), m.end(), doc, ws, we, shard, None)

                # --- numbers (digit scan shared by numbers100 + numbers scalar) ---
                need_digits = (numbers100_on and any(
                    counts[("numbers100", b)] < cap for b in numbers100_buckets)) or \
                    (numbers_scalar_on and counts[("numbers", None)] < cap)
                if need_digits:
                    for m in DIGIT_RE.finditer(doc):
                        tok = m.group(0)
                        is_int = "." not in tok
                        intpart = tok.split(".")[0]
                        if len(intpart) > MAX_NUMBER_DIGITS:
                            continue  # likely an ID / phone / hash
                        try:
                            val = round(float(tok))
                        except ValueError:
                            continue
                        if ws is None:
                            wm = [(w.start(), w.end()) for w in WORD_RE.finditer(doc)]
                            ws = [a for a, _ in wm]
                            we = [b for _, b in wm]
                        # numbers100 bucket (integer tokens 0..99 only)
                        if numbers100_on and is_int and 0 <= val <= 99:
                            lo = (val // 10) * 10
                            b = f"{lo}-{lo+9}"
                            if counts[("numbers100", b)] < cap and \
                                    per_doc[("numbers100", b)] < PER_DOC_PER_CLASS_LIMIT:
                                per_doc[("numbers100", b)] += 1
                                emit("numbers100", handles["numbers100"], "numbers100", b,
                                     "presence", tok, m.start(), m.end(), doc, ws, we, shard, val)
                        # numbers scalar (any magnitude; external = rounded value)
                        if numbers_scalar_on and counts[("numbers", None)] < cap and \
                                per_doc[("numbers", None)] < PER_DOC_PER_CLASS_LIMIT:
                            per_doc[("numbers", None)] += 1
                            emit("__scalars__", handles["__scalars__"], "numbers", None,
                                 "scalar", tok, m.start(), m.end(), doc, ws, we, shard, val)

                # --- scalar seed matches ---
                for sname, rgx in scalar_seed_matchers:
                    if counts[(sname, None)] >= cap:
                        continue
                    for m in rgx.finditer(doc):
                        if per_doc[(sname, None)] >= PER_DOC_PER_CLASS_LIMIT:
                            break
                        if counts[(sname, None)] >= cap:
                            break
                        if ws is None:
                            wm = [(w.start(), w.end()) for w in WORD_RE.finditer(doc)]
                            ws = [a for a, _ in wm]
                            we = [b for _, b in wm]
                        per_doc[(sname, None)] += 1
                        emit("__scalars__", handles["__scalars__"], sname, None,
                             "scalar", m.group(0), m.start(), m.end(), doc, ws, we, shard, None)

            if all_full():
                print(f"[shard] all active classes at quota after shard {shard} — stopping early")
                break
    finally:
        for fh in handles.values():
            fh.close()

    # ---- atomic rename tmp -> final ----
    for c in todo_presence:
        tmp = presence_files[c].with_suffix(".jsonl" + tmp_suffix)
        if tmp.exists():
            os.replace(tmp, presence_files[c])
    if todo_scalars:
        tmp = scalar_file.with_suffix(".jsonl" + tmp_suffix)
        if tmp.exists():
            os.replace(tmp, scalar_file)

    print(f"[build] done: {docs_seen} docs scanned in {time.time()-t0:.0f}s")
    summarize(out_dir, cap)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _count_file(fp: Path):
    """Return {cls_or_concept: count} for a candidate file."""
    out = defaultdict(int)
    if not fp.exists():
        return out
    with open(fp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("cls") if obj.get("cls") is not None else obj.get("concept")
            out[key] += 1
    return out


def summarize(out_dir: Path, cap: int):
    print("\n" + "=" * 64)
    print(f"CANDIDATE SUMMARY   (cap = {cap} per class)   dir={out_dir}")
    print("=" * 64)
    total = 0
    under = []

    for c, spec in concepts.PRESENCE_CONCEPTS.items():
        fp = out_dir / f"{c}.jsonl"
        per = _count_file(fp)
        ctotal = sum(per.values())
        total += ctotal
        status = "" if fp.exists() else "  (MISSING)"
        print(f"\n[{c}] regime=presence  total={ctotal}{status}")
        for cls in spec["classes"]:
            n = per.get(cls, 0)
            flag = "  <-- UNDER QUOTA" if n < cap else ""
            if n < cap:
                under.append(f"{c}::{cls} ({n}/{cap})")
            print(f"    {cls:<16} {n:>6}{flag}")

    # scalars (all in one file; group by concept)
    sfp = out_dir / "scalars.jsonl"
    print(f"\n[scalars] regime=scalar  file={'present' if sfp.exists() else 'MISSING'}")
    sper = defaultdict(int)
    sext = defaultdict(int)
    if sfp.exists():
        with open(sfp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sper[obj.get("concept")] += 1
                if obj.get("external") is not None:
                    sext[obj.get("concept")] += 1
    for sname in concepts.SCALARS:
        n = sper.get(sname, 0)
        total += n
        flag = "  <-- UNDER QUOTA" if n < cap else ""
        if n < cap:
            under.append(f"scalar::{sname} ({n}/{cap})")
        ext = f"  (external set on {sext.get(sname,0)})" if sext.get(sname) else ""
        print(f"    {sname:<16} {n:>6}{flag}{ext}")

    print("\n" + "-" * 64)
    print(f"TOTAL candidates: {total}")
    if under:
        print(f"UNDER-QUOTA classes ({len(under)}):")
        for u in under:
            print(f"    - {u}")
    else:
        print("All classes at quota.")
    print("=" * 64)
    return total, under


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Build candidate example pools from ClimbMix.")
    ap.add_argument("--smoke", action="store_true",
                    help="one shard, tiny caps, scratch output dir (logic validation).")
    args = ap.parse_args()

    if args.smoke:
        out_dir = config.DATA / "candidates_smoke"
        shards = config.SHARDS[:1]
        cap = 5
        max_docs = 3000
        print(f"[smoke] shards={shards} cap={cap} max_docs={max_docs} out={out_dir}")
        build(out_dir, shards, cap, max_docs, smoke=True)
    else:
        out_dir = config.DATA / "candidates"
        shards = list(config.SHARDS)
        cap = config.MAX_CANDIDATES_PER_CLASS
        max_docs = config.MAX_DOCS_PER_SHARD
        print(f"[full] shards={shards} cap={cap} max_docs={max_docs} out={out_dir}")
        build(out_dir, shards, cap, max_docs, smoke=False)


if __name__ == "__main__":
    main()
