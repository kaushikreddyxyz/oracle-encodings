"""Lexically mine natural concept candidates from ClimbMix shards 311-316 (§6.2).

For the 8 non-intensity families, surfaces = form_train + form_test from the
Stage 4 packs. Matches are windowed (match sentence ± 1, 250-500 chars) and the
judge later assigns truth — lexical hits include wrong senses (modal "May",
"North" inside "North America") by design; those become natural hard negatives.

The 5 intensity families are NOT mined (their form lists are generic adjectives
that would match everything); they are validated on the shared random pool.

Case policy:
  - proper-noun families (months, weekdays, continents): match Title-case or
    ALL-CAPS variants only.
  - other families: case-insensitive, EXCEPT all-caps abbreviation surfaces of
    length <= 3 ("NE", "SW"), which must match exact-case.
  - multi-word / hyphenated surfaces: tokens joined by [-–\\s]+ (so "blue-green"
    also matches "blue green").

Writes data/natural/mined/<family>.jsonl and prints per-class counts.
"""
import collections
import json
import os
import re

import yaml

import nat_common as nc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "natural", "mined")
FAM_DIR = os.path.join(HERE, "..", "..", "1_dataset", "config", "families")

SHARDS = list(range(311, 317))
MAX_DOCS_PER_SHARD = 150_000
PER_CLASS_CAP = 120
PER_DOC_PER_CLASS_CAP = 2
WIN_MIN, WIN_MAX = 250, 500

PROPER_FAMILIES = {"months", "weekdays", "continents"}
SKIP_FAMILIES = {"costliness", "physical_size", "lovingness", "duration", "harmfulness",
                 "glorptitude"}  # intensity axes + the §6.4 nonsense-control pack

_B_L = r"(?<![A-Za-z0-9])"
_B_R = r"(?![A-Za-z0-9])"


def surface_regex(surface):
    toks = re.split(r"[-\s]+", surface.strip())
    return r"[\-–\s]+".join(re.escape(t) for t in toks)


def load_family_specs():
    """{family: [(compiled_regex, {normalized_surface: cls})]}"""
    specs = {}
    for fp in sorted(os.listdir(FAM_DIR)):
        if not fp.endswith(".yaml"):
            continue
        pack = yaml.safe_load(open(os.path.join(FAM_DIR, fp)))
        fam = pack["family"]
        if fam in SKIP_FAMILIES:
            continue
        ci_parts, cs_parts, surf2cls = [], [], {}
        for cls, cfg in pack["classes"].items():
            for surface in (cfg.get("form_train") or []) + (cfg.get("form_test") or []):
                key = re.sub(r"[-–\s]+", " ", surface.lower())
                if surf2cls.get(key, cls) != cls:
                    raise RuntimeError(f"{fam}: surface {surface!r} maps to both "
                                       f"{surf2cls[key]} and {cls}")
                surf2cls[key] = cls
                pat = surface_regex(surface)
                if fam in PROPER_FAMILIES:
                    up = surface_regex(surface.upper())
                    cs_parts.append(f"{pat}|{up}" if up != pat else pat)
                elif len(surface) <= 3 and surface.isupper():
                    cs_parts.append(pat)
                else:
                    ci_parts.append(pat)
        regexes = []
        # longest-first so alternation prefers full phrases
        for parts, flags in ((cs_parts, 0), (ci_parts, re.IGNORECASE)):
            if parts:
                parts = sorted(set(parts), key=len, reverse=True)
                regexes.append(re.compile(_B_L + "(?:" + "|".join(parts) + ")" + _B_R,
                                          flags))
        specs[fam] = (regexes, surf2cls)
    return specs


def resolve_overlaps(matches):
    """Keep longest match at each position; drop matches inside a kept one."""
    matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))
    kept, last_end = [], -1
    for ms, me, cls, surf in matches:
        if ms >= last_end:
            kept.append((ms, me, cls, surf))
            last_end = me
    return kept


def window_around(text, sents, ms, me):
    """Match sentence ± 1, extended to >= WIN_MIN, trimmed to <= WIN_MAX chars."""
    idx = next((k for k, (s, e) in enumerate(sents) if s <= ms < e), None)
    if idx is None:
        return None
    lo, hi = max(0, idx - 1), min(len(sents) - 1, idx + 1)
    while sents[hi][1] - sents[lo][0] < WIN_MIN and (lo > 0 or hi < len(sents) - 1):
        if lo > 0:
            lo -= 1
        if sents[hi][1] - sents[lo][0] < WIN_MIN and hi < len(sents) - 1:
            hi += 1
    ws, we = sents[lo][0], sents[hi][1]
    if we - ws > WIN_MAX:  # trim around the match, snapping to word-ish edges
        ws = max(ws, ms - (WIN_MAX // 2))
        we = min(we, ws + WIN_MAX)
        ws = min(ws, ms)
        we = max(we, me)
        if ws > 0 and not text[ws - 1].isspace():
            nxt = text.find(" ", ws, ms)
            if 0 <= nxt < ms:
                ws = nxt + 1
        if we < len(text) and not text[we].isspace():
            prv = text.rfind(" ", me, we)
            if prv > me:
                we = prv
    win = text[ws:we].strip()
    off = text.index(win, ws) if win else ws
    if len(win) < 200 or not (ms >= off and me <= off + len(win)):
        return None
    if nc.window_alpha_ratio(win) < 0.6:
        return None
    return win, ms - off, me - off


def main():
    os.makedirs(OUT, exist_ok=True)
    specs = load_family_specs()
    kept = {f: collections.defaultdict(list) for f in specs}       # fam -> cls -> recs
    kept_sh = {f: collections.defaultdict(list) for f in specs}    # shingle sets
    n_dup = collections.Counter()

    def family_full(fam):
        _, surf2cls = specs[fam]
        return all(len(kept[fam][c]) >= PER_CLASS_CAP for c in set(surf2cls.values()))

    for shard in SHARDS:
        active = [f for f in specs if not family_full(f)]
        if not active:
            break
        print(f"[shard {shard}] scanning (active families: {len(active)})", flush=True)
        n_docs = 0
        for i, text in nc.iter_shard_docs(shard, MAX_DOCS_PER_SHARD):
            n_docs += 1
            if n_docs % 25_000 == 0:
                total = sum(len(v) for f in kept for v in kept[f].values())
                print(f"  ..{n_docs} docs, {total} kept", flush=True)
                active = [f for f in active if not family_full(f)]
                if not active:
                    break
            if nc.doc_english_ratio(text) < 0.6:
                continue
            sents = None
            doc_id = f"s{shard:05d}_d{i:06d}"
            per_doc = collections.Counter()
            for fam in active:
                regexes, surf2cls = specs[fam]
                matches = []
                for rx in regexes:
                    for m in rx.finditer(text):
                        key = re.sub(r"[-–\s]+", " ", m.group(0).lower())
                        cls = surf2cls.get(key)
                        if cls is not None:
                            matches.append((m.start(), m.end(), cls, m.group(0)))
                for ms, me, cls, surf in resolve_overlaps(matches):
                    if len(kept[fam][cls]) >= PER_CLASS_CAP:
                        continue
                    if per_doc[(fam, cls)] >= PER_DOC_PER_CLASS_CAP:
                        continue
                    if sents is None:
                        sents = nc.sentence_spans(text)
                    w = window_around(text, sents, ms, me)
                    if w is None:
                        continue
                    win, cs, ce = w
                    sh = nc.shingles(win)
                    if nc.is_near_dup(sh, kept_sh[fam][cls]):
                        n_dup[(fam, cls)] += 1
                        continue
                    k = len(kept[fam][cls])
                    ex_id = f"nat_m_{fam}_{cls.replace(' ', '_')}_{doc_id}_{k}"
                    kept[fam][cls].append({
                        "example_id": ex_id, "family": fam, "cls": cls,
                        "surface": surf, "match_char_span": [cs, ce],
                        "shard": shard, "doc_id": doc_id, "text": win,
                        "nat_split": nc.nat_split_of(ex_id)})
                    kept_sh[fam][cls].append(sh)
                    per_doc[(fam, cls)] += 1

    for fam, (_, surf2cls) in specs.items():
        with open(os.path.join(OUT, f"{fam}.jsonl"), "w") as f:
            for cls in sorted(set(surf2cls.values())):
                for r in kept[fam][cls]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts = {c: len(kept[fam][c]) for c in sorted(set(surf2cls.values()))}
        print(f"[{fam}] {counts}")
    total_dup = sum(n_dup.values())
    print(f"[done] near-dups skipped: {total_dup}")


if __name__ == "__main__":
    main()
