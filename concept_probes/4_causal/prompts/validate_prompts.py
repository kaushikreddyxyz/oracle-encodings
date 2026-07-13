#!/usr/bin/env python3
"""Machine validation of the Stage 6.1 audited prompt banks (pure stdlib).

Run:  python3 validate_prompts.py            (from anywhere; validates its own dir)

Checks per file type
--------------------
*.cloze.json
  - schema: family/classes/templates keys; template fields id/type/prompt/
    completions/answer_class/notes; family matches filename; unique ids with
    family prefix; type in {succession, association, factual, paraphrase}.
  - prompt hygiene: non-empty; no leading/trailing whitespace (must end
    mid-sentence right at the completion slot); no trailing '.', '!', '?';
    no double spaces.
  - completions: keys == classes exactly; 1-3 non-empty strings per class;
    no leading/trailing whitespace (eval code adds the leading space);
    unique within class; parallelism: per template max-min completion count
    across classes <= 2 and every completion <= 5 words.
  - leakage (case-insensitive, word-boundary, hyphen/space/underscore-flexible,
    optional plural 's'):
      answer_class == null  -> NO class name and NO completion surface of ANY
                               class may appear in the prompt;
      answer_class set      -> the answer class name and ITS completion
                               surfaces must be absent (siblings allowed).
  - counts: 30-50 templates, >=15 class-agnostic, >=15 class-keyed
    (errors); every class covered by >=1 keyed template (warning).

*.ordinal.json
  - schema: axis/prompts/ordered_completion_sets; axis matches filename;
    unique prompt/set ids.
  - prompt hygiene as above; 15-25 prompts (error).
  - sets: 2-4 per axis; 5-7 items each; unique, non-empty, trimmed items.
  - leakage: no completion item of any set may appear in any prompt
    (an intensity word in the prompt would presuppose the answer).

*.tokens.json
  - schema: categorical -> classes{cls: {surface, associates}};
    intensity -> poles{low,high: {surface, associates}}.
  - per class/pole: >=1 surface; 5-15 total strings; trimmed, non-empty,
    unique within the class.

Ordering inside ordinal completion sets is semantic (weak -> strong along the
axis) and is asserted by construction/human audit; a machine cannot verify it.

Exit status 0 iff no ERRORs (warnings allowed).
"""
import json
import os
import re
import sys
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
CLOZE_TYPES = {"succession", "association", "factual", "paraphrase"}

errors = []    # (file, msg)
warnings = []  # (file, msg)


def err(f, msg):
    errors.append((f, msg))


def warn(f, msg):
    warnings.append((f, msg))


def term_pattern(term):
    """Word-boundary regex for a term; multiword/hyphenated terms match with
    any of space/hyphen/underscore between the words; optional plural 's'."""
    words = [w for w in re.split(r"[\s\-_]+", term.strip()) if w]
    body = r"[\s\-_]+".join(re.escape(w) for w in words)
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"s?(?![A-Za-z0-9])", re.IGNORECASE)


def leak_terms(cls, completions):
    """Terms whose presence in a prompt counts as leakage for class `cls`."""
    return {cls.replace("_", " ")} | set(completions.get(cls, []))


def check_prompt_hygiene(f, pid, prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        err(f, f"{pid}: empty prompt")
        return
    if prompt != prompt.strip():
        err(f, f"{pid}: leading/trailing whitespace in prompt")
    if prompt.rstrip().endswith((".", "!", "?")):
        err(f, f"{pid}: prompt ends with sentence-final punctuation (must end mid-sentence at the slot)")
    if "  " in prompt:
        err(f, f"{pid}: double space in prompt")


def check_completion_string(f, pid, cls, s):
    if not isinstance(s, str) or not s.strip():
        err(f, f"{pid}/{cls}: empty completion string")
        return
    if s != s.strip():
        err(f, f"{pid}/{cls}: completion {s!r} has leading/trailing whitespace (eval adds the leading space)")
    if len(s.split()) > 5:
        err(f, f"{pid}/{cls}: completion {s!r} longer than 5 words")


def validate_cloze(path):
    f = os.path.basename(path)
    data = json.load(open(path))
    family = data.get("family")
    if f != f"{family}.cloze.json":
        err(f, f"family field {family!r} does not match filename")
    classes = data.get("classes", [])
    if len(set(classes)) != len(classes) or not classes:
        err(f, "classes empty or not unique")
    templates = data.get("templates", [])

    seen_ids = set()
    n_agnostic = n_keyed = 0
    keyed_coverage = {c: 0 for c in classes}

    for t in templates:
        pid = t.get("id", "<no-id>")
        for key in ("id", "type", "prompt", "completions", "answer_class", "notes"):
            if key not in t:
                err(f, f"{pid}: missing field {key!r}")
        if pid in seen_ids:
            err(f, f"duplicate id {pid}")
        seen_ids.add(pid)
        if not pid.startswith(family + "_"):
            err(f, f"{pid}: id does not start with {family}_")
        if t.get("type") not in CLOZE_TYPES:
            err(f, f"{pid}: bad type {t.get('type')!r}")

        prompt = t.get("prompt", "")
        check_prompt_hygiene(f, pid, prompt)

        comp = t.get("completions", {})
        if set(comp) != set(classes):
            err(f, f"{pid}: completion keys != classes (missing {set(classes)-set(comp)}, extra {set(comp)-set(classes)})")
        counts = []
        for cls, lst in comp.items():
            if not isinstance(lst, list) or not (1 <= len(lst) <= 3):
                err(f, f"{pid}/{cls}: needs 1-3 completions, got {lst!r}")
                continue
            if len(set(lst)) != len(lst):
                err(f, f"{pid}/{cls}: duplicate completions")
            for s in lst:
                check_completion_string(f, pid, cls, s)
            counts.append(len(lst))
        if counts and max(counts) - min(counts) > 2:
            err(f, f"{pid}: completion counts not parallel across classes (min {min(counts)}, max {max(counts)})")

        ans = t.get("answer_class")
        if ans is None:
            n_agnostic += 1
            forbidden = set()
            for c in classes:
                forbidden |= leak_terms(c, comp)
        else:
            if ans not in classes:
                err(f, f"{pid}: answer_class {ans!r} not in classes")
                continue
            n_keyed += 1
            keyed_coverage[ans] = keyed_coverage.get(ans, 0) + 1
            forbidden = leak_terms(ans, comp)
        for term in sorted(forbidden):
            if term_pattern(term).search(prompt):
                err(f, f"{pid}: leakage — {term!r} appears in prompt (answer_class={ans})")

    if not (30 <= len(templates) <= 50):
        err(f, f"{len(templates)} templates (need 30-50)")
    if n_agnostic < 15:
        err(f, f"only {n_agnostic} class-agnostic templates (need >=15)")
    if n_keyed < 15:
        err(f, f"only {n_keyed} class-keyed templates (need >=15)")
    for c, n in keyed_coverage.items():
        if n == 0:
            warn(f, f"class {c!r} has no class-keyed template")
    return f"{len(templates)} templates ({n_agnostic} agnostic / {n_keyed} keyed), {len(classes)} classes"


def validate_ordinal(path):
    f = os.path.basename(path)
    data = json.load(open(path))
    axis = data.get("axis")
    if f != f"{axis}.ordinal.json":
        err(f, f"axis field {axis!r} does not match filename")
    prompts = data.get("prompts", [])
    sets = data.get("ordered_completion_sets", [])

    seen = set()
    for p in prompts:
        pid = p.get("id", "<no-id>")
        if pid in seen:
            err(f, f"duplicate prompt id {pid}")
        seen.add(pid)
        if "notes" not in p:
            err(f, f"{pid}: missing notes field")
        check_prompt_hygiene(f, pid, p.get("prompt", ""))
    if not (15 <= len(prompts) <= 25):
        err(f, f"{len(prompts)} prompts (need 15-25)")

    if not (2 <= len(sets) <= 4):
        err(f, f"{len(sets)} ordered completion sets (need 2-4)")
    all_items = []
    sids = set()
    for s in sets:
        sid = s.get("id", "<no-id>")
        if sid in sids:
            err(f, f"duplicate set id {sid}")
        sids.add(sid)
        items = s.get("completions", [])
        if not (5 <= len(items) <= 7):
            err(f, f"set {sid}: {len(items)} items (need 5-7)")
        if len(set(items)) != len(items):
            err(f, f"set {sid}: duplicate items")
        for it in items:
            check_completion_string(f, sid, "set", it)
            all_items.append(it)
    # leakage: no completion item may appear in any prompt
    for p in prompts:
        for it in all_items:
            if term_pattern(it).search(p.get("prompt", "")):
                err(f, f"{p.get('id')}: leakage — set item {it!r} appears in prompt")
    return f"{len(prompts)} prompts, {len(sets)} ordered sets ({sum(len(s['completions']) for s in sets)} items)"


def validate_tokens(path):
    f = os.path.basename(path)
    data = json.load(open(path))
    family = data.get("family")
    if f != f"{family}.tokens.json":
        err(f, f"family field {family!r} does not match filename")
    ftype = data.get("type")
    if ftype == "categorical":
        groups = data.get("classes", {})
    elif ftype == "intensity":
        groups = data.get("poles", {})
        if set(groups) != {"low", "high"}:
            err(f, f"intensity poles must be exactly low/high, got {sorted(groups)}")
    else:
        err(f, f"bad type {ftype!r} (categorical|intensity)")
        return "invalid"
    total = 0
    for name, g in groups.items():
        surf, assoc = g.get("surface", []), g.get("associates", [])
        if not surf:
            err(f, f"{name}: no surface forms")
        toks = list(surf) + list(assoc)
        if not (5 <= len(toks) <= 15):
            err(f, f"{name}: {len(toks)} diagnostic tokens (need 5-15)")
        if len(set(t.lower() for t in toks)) != len(toks):
            err(f, f"{name}: duplicate tokens (case-insensitive)")
        for t in toks:
            if not isinstance(t, str) or not t.strip() or t != t.strip():
                err(f, f"{name}: bad token {t!r}")
        total += len(toks)
    return f"{len(groups)} {'classes' if ftype == 'categorical' else 'poles'}, {total} tokens"


def main():
    reports = []
    for path in sorted(glob(os.path.join(HERE, "*.cloze.json"))):
        reports.append((os.path.basename(path), validate_cloze(path)))
    for path in sorted(glob(os.path.join(HERE, "*.ordinal.json"))):
        reports.append((os.path.basename(path), validate_ordinal(path)))
    for path in sorted(glob(os.path.join(HERE, "*.tokens.json"))):
        reports.append((os.path.basename(path), validate_tokens(path)))

    print("=== Stage 6.1 prompt-bank validation ===")
    for fname, rep in reports:
        flagged = [m for ff, m in errors if ff == fname]
        status = "FAIL" if flagged else "ok"
        print(f"[{status:4s}] {fname:32s} {rep}")
    for fname, msg in warnings:
        print(f"WARNING {fname}: {msg}")
    for fname, msg in errors:
        print(f"ERROR   {fname}: {msg}")
    print(f"--- {len(reports)} files, {len(errors)} errors, {len(warnings)} warnings ---")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
