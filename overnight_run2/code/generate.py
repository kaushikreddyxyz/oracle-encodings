"""Phase-A concept-probe dataset pipeline (resumable, concurrent, OpenRouter + local tokenizer).

Per concept -> one spec-compliant JSONL (schema EXACTLY per probe_dataset_spec.md §8):
  - 60% templates (typed-frame fill + minimal pairs by slot-swap) / 40% LLM free-gen (verbatim span)
  - 3 negative types: hard (confusable surface form w/o the concept), family_disjoint
    (another concept's text), neutral (topic-free filler); mix 40/35/25
  - held-out-vocab eval split: unique periphrasis, canonical trigger BANNED, in_vocabulary=false
  - per-token labels via gemma offset overlap (labeling.py); minimal-pair difflib cross-check
  - judge filter: distinct model, self-consistency 3, keep >=2/3 (all free-gen + heldout + hard negs; sample templates/fillers)
  - splits 70/15/15 stratified, no minimal-pair/frame leakage; heldout-vocab is a SEPARATE split

Resolved ambiguities (see report): hard negatives are the per-concept CONFUSABLES (Wed/wed, Sun/sun,
fall-the-verb, Easter!=East ...) -- true concept-absent surface traps that defeat the lexical shortcut;
sibling-value positives additionally serve as in-family hard negatives inside the trainer. A `split`
field is added (additive metadata the trainer needs) alongside `in_vocabulary`.
"""
import asyncio
import json
import os
import random
import re

import concept_configs as cc
import labeling as L
from or_client import ORClient

random.seed(1234)

GEN_MODEL = "qwen/qwen-2.5-72b-instruct"
JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"
GEN_TEMP = 0.9
JUDGE_TEMP = 0.2

DECOR_PRE = ["", "", "Honestly, ", "You know, ", "By the way, ", "Frankly, ",
             "As it happens, ", "To be fair, ", "In my experience, ", "Believe it or not, "]
DECOR_SUF = ["", "", " I think.", ", more or less.", ", as usual.", " these days.",
             ", at least for me.", ", or so they say."]

# noun-phrase-friendly frames for held-out periphrasis ("the coldest season of the year", etc.)
# -- avoid frames like "happens every {X}" / "on {X}" that need a bare noun and read ungrammatically.
HELDOUT_FRAMES = [
    "I really enjoy {X}.", "My favorite time of year is {X}.", "We always travel during {X}.",
    "She looks forward to {X} every year.", "Nothing beats a long walk in {X}.",
    "The town comes alive in {X}.", "Business picks up in {X}.", "He proposed to her in {X}.",
    "The harvest festival is held in {X}.", "Tourists love visiting in {X}.",
    "I was born in {X}.", "Everything feels different in {X}.",
]

NEUTRAL_BANK = [
    "The committee reviewed the quarterly budget on schedule.",
    "She replaced the batteries in the smoke detector.",
    "He alphabetized the folders in the filing cabinet.",
    "The printer ran out of toner during the report.",
    "They repaved the parking lot behind the office.",
    "A new coat of paint brightened the stairwell.",
    "The software update finished installing overnight.",
    "He tightened the loose hinge on the cabinet door.",
    "The librarian reshelved the returned paperbacks.",
    "She balanced the spreadsheet before the call.",
    "The plumber fixed the slow drain in the sink.",
    "We rearranged the chairs for the workshop.",
    "The clerk stamped each form and filed it away.",
    "He swapped the worn tires on the delivery van.",
    "The intern organized the supply closet neatly.",
    "A gentle hum came from the server room.",
]


# --------------------------------------------------------------------------- frames

def _decorate(base_frame):
    """Yield (text_template_with_{X}, prefix_len_before_slot_offset_helper) decorated variants.

    We never use str.find for the concept span: we know the slot's char offset by construction.
    Returns a list of (full_template, slot_char_start_fn) where slot_char_start_fn(value)->[s,e].
    """
    out = []
    pre_slot, post_slot = base_frame.split("{X}")
    for p in DECOR_PRE:
        for s in DECOR_SUF:
            head = pre_slot
            if p and head and head[0] != "I":  # avoid mid-sentence Capital after a prefix
                head = head[0].lower() + head[1:]
            full_pre = p + head
            post = post_slot
            if s and post.endswith("."):  # avoid doubled terminal punctuation ("spring. these days.")
                post = post[:-1]
            full_post = post + s
            out.append((full_pre, full_post))
    return out


def make_template_positives(concept, n_groups, surface_pick="cycle"):
    """Build minimal-pair groups. Each group = a fixed decorated frame + surface variant, filled
    across ALL cyclic values (one record per value). Returns list of groups; each group is a list
    of dicts {value_idx, value_name, text, span}."""
    cfg = cc.get(concept)
    values = cfg["values"]
    frames = cfg["frames"]
    combos = []  # (full_pre, full_post)
    for f in frames:
        combos.extend(_decorate(f))
    random.shuffle(combos)
    groups = []
    gi = 0
    ci = 0
    while len(groups) < n_groups:
        if ci >= len(combos):
            ci = 0  # exhausted; reshuffle for more decorator coverage
            random.shuffle(combos)
        full_pre, full_post = combos[ci]
        ci += 1
        # pick a surface variant index for this group (varied across groups for lexical breadth)
        svar = gi
        recs = []
        for vidx, v in enumerate(values):
            surfs = v["surface"]
            surf = surfs[svar % len(surfs)]
            text = full_pre + surf + full_post
            s = len(full_pre)
            recs.append({"value_idx": vidx, "value_name": v["name"],
                         "text": text, "span": [s, s + len(surf)]})
        groups.append({"mp_id": f"mp_{concept}_{gi:05d}", "records": recs})
        gi += 1
    return groups


def make_scalar_template_positives(concept, n_items):
    """Scalar templates: wrap anchor phrases in frames. Each item carries its anchored [0,1] value
    as the per-token span target. Span = the inserted anchor phrase (known by construction)."""
    cfg = cc.get(concept)
    frames = cfg["frames"]
    anchors = cfg["anchors"]
    items = []
    i = 0
    while len(items) < n_items:
        val, phrase = anchors[i % len(anchors)]
        pre_slot, post_slot = frames[i % len(frames)].split("{X}")
        for fp, ps in [_decorate(frames[i % len(frames)])[i % 80]]:
            text = fp + phrase + ps
            s = len(fp)
            items.append({"value": round(float(val), 4), "text": text, "span": [s, s + len(phrase)]})
        i += 1
    return items[:n_items]


# --------------------------------------------------------------------------- free-gen

async def freegen_positives(client, concept, value_idx, n_target, batch=6, max_calls=None):
    """LLM free-gen: text that entails the value WITHOUT naming it; returns verbatim span.
    Rejects any item whose span is not an exact substring. Returns list of {text, span:[s,e]}."""
    cfg = cc.get(concept)
    out, seen = [], set()
    if cfg["family"] == "cyclic":
        v = cfg["values"][value_idx]
        target_desc = f'the {concept} value "{v["name"]}"'
        extra = cfg["freegen_hint"]
    else:
        # scalar value_idx encodes the target magnitude bucket
        val, anch = value_idx
        target_desc = (f'a {concept} magnitude of about {val:.2f} on a 0..1 scale '
                       f'(rubric: {cfg["rubric"]}; e.g. like "{anch}")')
        extra = cfg["freegen_hint"]
    sys = ("You generate concept-probe training data. Output STRICT JSON only.")
    prompt = (
        f'Write {batch} short, naturalistic, DISTINCT English texts (1-2 sentences each) that entail '
        f'{target_desc}, {extra}. The texts must NOT merely state the value as a bare label. '
        f'For EACH text, also return "span": the EXACT verbatim substring of that text that most '
        f'carries the concept (copy it character-for-character from the text). '
        f'Return JSON: {{"items":[{{"text":"...","span":"..."}}]}}'
    )
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": prompt}]
    rounds = 0
    while len(out) < n_target and rounds < 6:
        remaining = n_target - len(out)
        n_calls = -(-remaining // batch) + 1  # ceil + 1 spare; client semaphore bounds real concurrency
        results = await asyncio.gather(*[
            client.chat_json(GEN_MODEL, msgs, temperature=GEN_TEMP, max_tokens=1600)
            for _ in range(n_calls)])
        rounds += 1
        for obj in results:
            if not obj or "items" not in obj:
                continue
            for it in obj["items"]:
                text = (it.get("text") or "").strip()
                span_str = (it.get("span") or "").strip()
                if not text or not span_str or text.lower() in seen:
                    continue
                span = L.find_span_char(text, span_str)  # exact-substring or None
                if span is None:
                    continue
                seen.add(text.lower())
                out.append({"text": text, "span": span})
    return out[:n_target]


async def freegen_hard_negatives(client, concept, word, n_target, batch=6, max_calls=None):
    """Confusable hard negatives: use the surface form `word` in a sense where the concept is ABSENT."""
    cfg = cc.get(concept)
    _ = cfg, max_calls
    out, seen = [], set()
    prompt = (
        f'Write {batch} short, DISTINCT English sentences that contain the word "{word}" used in a '
        f'sense that has NOTHING to do with the {concept} concept (a homonym, name, idiom, or '
        f'unrelated meaning). The {concept} meaning must be clearly absent. '
        f'Return JSON: {{"items":[{{"text":"..."}}]}}'
    )
    rounds = 0
    while len(out) < n_target and rounds < 6:
        remaining = n_target - len(out)
        n_calls = -(-remaining // batch) + 1
        results = await asyncio.gather(*[
            client.chat_json(GEN_MODEL, [{"role": "user", "content": prompt}],
                             temperature=GEN_TEMP, max_tokens=1200) for _ in range(n_calls)])
        rounds += 1
        for obj in results:
            if not obj or "items" not in obj:
                continue
            for it in obj["items"]:
                text = (it.get("text") or "").strip()
                if not text or text.lower() in seen or word.lower() not in text.lower():
                    continue
                seen.add(text.lower())
                out.append(text)
    return out[:n_target]


# --------------------------------------------------------------------------- held-out vocab

def _banned_re(banned):
    return [re.compile(r"\b" + re.escape(b) + r"\w*", re.IGNORECASE) for b in banned]


def make_heldout_vocab(concept, per_value):
    """Periphrasis-based eval items with the canonical trigger banned. Returns (items, skipped).
    items: {value_idx, value_name, text, span, banned}. skipped: list of value names with no unique form."""
    cfg = cc.get(concept)
    items, skipped = [], []
    if cfg["family"] != "cyclic":
        return items, skipped  # held-out vocab defined for cyclic value-pinning
    frames = HELDOUT_FRAMES
    for vidx, v in enumerate(cfg["values"]):
        periphr = v.get("periphrasis") or []
        if not periphr:
            skipped.append(v["name"])
            continue
        bre = _banned_re(v["banned"])
        made = 0
        i = 0
        guard = 0
        while made < per_value and guard < per_value * 20:
            guard += 1
            phrase = periphr[i % len(periphr)]
            fp, ps = _decorate(frames[i % len(frames)])[(i * 7) % 80]
            text = fp + phrase + ps
            i += 1
            if any(rx.search(text) for rx in bre):  # leak guard
                continue
            s = len(fp)
            items.append({"value_idx": vidx, "value_name": v["name"], "text": text,
                          "span": [s, s + len(phrase)], "banned": v["banned"]})
            made += 1
    return items, skipped


# --------------------------------------------------------------------------- family-disjoint

def make_family_disjoint(concept, n):
    """Sentences drawn from OTHER concepts' text -> concept-absent for this concept's family."""
    others = [k for k in cc.CONCEPTS if k != concept]
    pool = []
    for oc in others:
        ocfg = cc.get(oc)
        frames = ocfg["frames"]
        if ocfg["family"] == "cyclic":
            for v in ocfg["values"]:
                surf = v["surface"][0]
                pre, post = frames[len(pool) % len(frames)].split("{X}")
                pool.append(pre + surf + post)
        else:
            for val, phrase in ocfg["anchors"]:
                pre, post = frames[len(pool) % len(frames)].split("{X}")
                pool.append(pre + phrase + post)
    random.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    return [pool[i % len(pool)] for i in range(n)]


def make_neutral(n):
    out = []
    i = 0
    while len(out) < n:
        base = NEUTRAL_BANK[i % len(NEUTRAL_BANK)]
        fp, ps = DECOR_PRE[i % len(DECOR_PRE)], DECOR_SUF[i % len(DECOR_SUF)]
        b = base
        if fp and b[0] != "I":
            b = b[0].lower() + b[1:]
        out.append((fp + b + ps).strip())
        i += 1
    return out[:n]


# --------------------------------------------------------------------------- judging

async def judge_batch(client, concept, items, expect_entail, check_span=False):
    """items: list of {"text", "claim"[, "span"]}; claim is the value name (positives) or None (neg).
    Self-consistency 3 calls; returns list of n_yes (0..3) per item. yes = 'text entails the claimed
    value' (and, if check_span, the quoted span is the locus); for negatives, yes = 'concept present'."""
    if not items:
        return []
    lines = []
    for i, it in enumerate(items):
        if expect_entail and check_span:
            lines.append(f'{i}: text="{it["text"]}"  ||  claimed_value="{it["claim"]}"  ||  span="{it.get("span","")}"')
        elif expect_entail:
            lines.append(f'{i}: text="{it["text"]}"  ||  claimed_value="{it["claim"]}"')
        else:
            lines.append(f'{i}: text="{it["text"]}"')
    body = "\n".join(lines)
    if expect_entail and check_span:
        task = (f'For each numbered item decide BOTH: (a) does the text ENTAIL that the {concept} is exactly '
                f'the claimed_value (soft/associative hints do NOT count -- it must genuinely pin that value), '
                f'AND (b) is the quoted span the main part of the text that carries that {concept} cue? '
                f'Answer entails=true only if BOTH hold. '
                f'Return JSON {{"verdicts":[{{"id":<int>,"entails":<true|false>}}]}}.')
    elif expect_entail:
        task = (f'For each numbered item decide: does the text ENTAIL that the {concept} is exactly the '
                f'claimed_value? (Soft/associative hints do NOT count -- it must genuinely pin that value.) '
                f'Return JSON {{"verdicts":[{{"id":<int>,"entails":<true|false>}}]}}.')
    else:
        task = (f'For each numbered item decide: does the text clearly entail ANY specific {concept} value '
                f'(i.e. is the {concept} concept actually present)? Return JSON '
                f'{{"verdicts":[{{"id":<int>,"entails":<true|false>}}]}}.')
    prompt = task + "\n\n" + body
    _ = check_span  # already folded into task/lines above
    counts = [0] * len(items)
    for _ in range(3):
        obj = await client.chat_json(JUDGE_MODEL, [{"role": "user", "content": prompt}],
                                     temperature=JUDGE_TEMP, max_tokens=1400)
        if not obj or "verdicts" not in obj:
            continue
        for vd in obj["verdicts"]:
            try:
                idx = int(vd["id"])
            except Exception:
                continue
            if 0 <= idx < len(items) and bool(vd.get("entails")):
                counts[idx] += 1
    return counts


async def judge_all(client, concept, items, expect_entail, chunk=10, check_span=False):
    """Returns list of n_yes parallel to items."""
    out = [0] * len(items)
    tasks = []
    spans = []
    for st in range(0, len(items), chunk):
        sub = items[st:st + chunk]
        spans.append((st, len(sub)))
        tasks.append(judge_batch(client, concept, sub, expect_entail, check_span=check_span))
    results = await asyncio.gather(*tasks)
    for (st, ln), counts in zip(spans, results):
        for j in range(ln):
            out[st + j] = counts[j] if j < len(counts) else 0
    return out


# --- identification-style judge (robust for periphrasis: "which value does this describe?") ---

def _accept_set(concept, value_name):
    cfg = cc.get(concept)
    for v in cfg.get("values", []):
        if v["name"] == value_name:
            acc = {value_name.lower()}
            for sf in v["surface"]:
                acc.add(sf.lower().replace("the ", "").strip())
            return acc
    return {value_name.lower()}


async def judge_identify_batch(client, concept, items, allowed, check_span):
    """items: {text, claim[, span]}. The judge NAMES the best-fit value (or 'none'). Returns
    (value_match_counts, span_ok_counts) over 3 self-consistency calls. A periphrastic-but-unique
    description (e.g. 'the hottest season' -> Summer) is accepted; an associative one
    ('the depths of winter' -> ambiguous across Dec/Jan/Feb) yields 'none' or a mismatch."""
    if not items:
        return [], []
    lines = []
    for i, it in enumerate(items):
        if check_span:
            lines.append(f'{i}: text="{it["text"]}"  ||  span="{it.get("span","")}"')
        else:
            lines.append(f'{i}: text="{it["text"]}"')
    allowed_str = ", ".join(f'"{a}"' for a in allowed) + ', or "none"'
    if check_span:
        task = (f'For each numbered item: (1) which single {concept} value does the text MOST clearly and '
                f'unambiguously describe? Choose exactly one of: {allowed_str}. If it fits several values '
                f'equally or none clearly, answer "none". (2) Does the quoted span contain the main cue for '
                f'that value? Return JSON {{"verdicts":[{{"id":<int>,"value":"<one choice>","span_ok":<true|false>}}]}}.')
    else:
        task = (f'For each numbered item: which single {concept} value does the text MOST clearly and '
                f'unambiguously describe? Choose exactly one of: {allowed_str}. If it fits several values '
                f'equally or none clearly, answer "none". '
                f'Return JSON {{"verdicts":[{{"id":<int>,"value":"<one choice>"}}]}}.')
    prompt = task + "\n\n" + "\n".join(lines)
    match = [0] * len(items)
    span_ok = [0] * len(items)
    accs = [_accept_set(concept, it["claim"]) for it in items]
    for _ in range(3):
        obj = await client.chat_json(JUDGE_MODEL, [{"role": "user", "content": prompt}],
                                     temperature=JUDGE_TEMP, max_tokens=1400)
        if not obj or "verdicts" not in obj:
            continue
        for vd in obj["verdicts"]:
            try:
                idx = int(vd["id"])
            except Exception:
                continue
            if not (0 <= idx < len(items)):
                continue
            jv = str(vd.get("value", "")).lower().strip()
            if jv and any(a in jv for a in accs[idx]):
                match[idx] += 1
            if check_span and bool(vd.get("span_ok")):
                span_ok[idx] += 1
    return match, span_ok


async def judge_identify_all(client, concept, items, allowed, check_span, chunk=10):
    match = [0] * len(items)
    span_ok = [0] * len(items)
    tasks, spans = [], []
    for st in range(0, len(items), chunk):
        sub = items[st:st + chunk]
        spans.append((st, len(sub)))
        tasks.append(judge_identify_batch(client, concept, sub, allowed, check_span))
    results = await asyncio.gather(*tasks)
    for (st, ln), (m, so) in zip(spans, results):
        for j in range(ln):
            match[st + j] = m[j] if j < len(m) else 0
            span_ok[st + j] = so[j] if j < len(so) else 0
    return match, span_ok


# --------------------------------------------------------------------------- record assembly

def build_record(concept, family, n, text, label_value, label_index, polarity, negative_type,
                 spans, generator, in_vocabulary, mp_id, split, judge_verified, span_target):
    """span_target: label_index (cyclic) or float (scalar); used as per-token span target."""
    ids, token_targets, loss_mask, span_tokens = L.build_token_labels(
        text, spans if polarity == "positive" else None, span_target)
    rec = {
        "concept": concept, "family": family, "n": n, "text": text,
        "label_value": label_value, "label_index": label_index,
        "polarity": polarity, "negative_type": negative_type,
        "concept_span_char": spans if polarity == "positive" else None,
        "concept_span_tokens": span_tokens if polarity == "positive" else None,
        "token_targets": token_targets, "loss_mask": loss_mask,
        "n_tokens": len(ids), "generator": generator, "in_vocabulary": in_vocabulary,
        "minimal_pair_id": mp_id, "split": split, "judge_verified": judge_verified,
    }
    return rec


def assign_splits(groups_or_items, ratios=(0.70, 0.15, 0.15)):
    """Shuffle and split a list of opaque units into train/val/test."""
    units = list(groups_or_items)
    random.shuffle(units)
    n = len(units)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    tr = units[:n_tr]
    va = units[n_tr:n_tr + n_va]
    te = units[n_tr + n_va:]
    return tr, va, te


# --------------------------------------------------------------------------- main per-concept

async def build_concept(concept, pos_per_value, out_path, neg_ratio=0.6, heldout_per_value=20,
                        concurrency=64, template_frac=0.6, judge_sample_frac=0.10,
                        freegen_overgen=1.8):
    cfg = cc.get(concept)
    family, n = cfg["family"], cfg["n"]
    records = []
    stats = {"concept": concept}

    async with ORClient(concurrency=concurrency) as client:
        # ---------------- POSITIVES ----------------
        n_templ = int(round(pos_per_value * template_frac))
        n_free = pos_per_value - n_templ

        positives = []  # (text, label_value, label_index, span, generator, mp_id, span_target, value_key)

        if family == "cyclic":
            groups = make_template_positives(concept, n_templ)
            tg_tr, tg_va, tg_te = assign_splits(groups)
            split_of = {}
            for g in tg_tr:
                split_of[g["mp_id"]] = "train"
            for g in tg_va:
                split_of[g["mp_id"]] = "val"
            for g in tg_te:
                split_of[g["mp_id"]] = "test"
            for g in groups:
                for r in g["records"]:
                    positives.append(dict(text=r["text"], label_value=r["value_name"],
                                          label_index=r["value_idx"], span=[r["span"]],
                                          generator="template", mp_id=g["mp_id"],
                                          span_target=r["value_idx"], split=split_of[g["mp_id"]],
                                          judged=False, expect=True))
            # free-gen per value (over-generate so the post-judge count lands near n_free)
            fg_target = int(round(n_free * freegen_overgen))
            fg_tasks = [freegen_positives(client, concept, vidx, fg_target)
                        for vidx in range(len(cfg["values"]))]
            fg_results = await asyncio.gather(*fg_tasks)
            for vidx, fglist in enumerate(fg_results):
                vname = cfg["values"][vidx]["name"]
                tr, va, te = assign_splits(fglist)
                for bucket, sp in [(tr, "train"), (va, "val"), (te, "test")]:
                    for it in bucket:
                        positives.append(dict(text=it["text"], label_value=vname, label_index=vidx,
                                              span=[it["span"]], generator="free_gen", mp_id=None,
                                              span_target=vidx, split=sp, judged=True, expect=True))
        else:  # scalar
            sc_templ = make_scalar_template_positives(concept, n_templ)
            tr, va, te = assign_splits(sc_templ)
            for bucket, sp in [(tr, "train"), (va, "val"), (te, "test")]:
                for it in bucket:
                    positives.append(dict(text=it["text"], label_value=it["value"], label_index=None,
                                          span=[it["span"]], generator="template", mp_id=None,
                                          span_target=it["value"], split=sp, judged=False, expect=True))
            # free-gen across the anchor magnitudes
            anchors = cfg["anchors"]
            per_anchor = max(1, n_free // len(anchors))
            fg_tasks = [freegen_positives(client, concept, (val, anch), per_anchor)
                        for (val, anch) in anchors]
            fg_results = await asyncio.gather(*fg_tasks)
            for (val, anch), fglist in zip(anchors, fg_results):
                tr, va, te = assign_splits(fglist)
                for bucket, sp in [(tr, "train"), (va, "val"), (te, "test")]:
                    for it in bucket:
                        positives.append(dict(text=it["text"], label_value=round(float(val), 4),
                                              label_index=None, span=[it["span"]], generator="free_gen",
                                              mp_id=None, span_target=round(float(val), 4), split=sp,
                                              judged=True, expect=True))

        n_pos = len(positives)

        # ---------------- NEGATIVES ----------------
        n_neg = int(round(n_pos * neg_ratio))
        n_hard = int(round(n_neg * 0.40))
        n_fam = int(round(n_neg * 0.35))
        n_neu = n_neg - n_hard - n_fam

        # hard = confusables (seed from config + free-gen to fill)
        conf = cfg.get("confusables", [])
        hard_texts = []
        for c in conf:
            for s in c["sentences"]:
                hard_texts.append(s)
        seed_hard = list(hard_texts)
        if conf and len(hard_texts) < n_hard:
            per_word = (n_hard - len(hard_texts)) // len(conf) + 2
            fg_tasks = [freegen_hard_negatives(client, concept, c["word"], per_word) for c in conf]
            fg_more = await asyncio.gather(*fg_tasks)
            for lst in fg_more:
                hard_texts.extend(lst)
        random.shuffle(hard_texts)
        hard_texts = hard_texts[:n_hard]

        fam_texts = make_family_disjoint(concept, n_fam)
        neu_texts = make_neutral(n_neu)

        negatives = []
        for t in hard_texts:
            negatives.append(dict(text=t, negative_type="hard", judged=True, expect=False))
        for t in fam_texts:
            negatives.append(dict(text=t, negative_type="family_disjoint", judged=False, expect=False))
        for t in neu_texts:
            negatives.append(dict(text=t, negative_type="neutral", judged=False, expect=False))
        # split negatives 70/15/15 per type
        for nt in ("hard", "family_disjoint", "neutral"):
            sub = [x for x in negatives if x["negative_type"] == nt]
            tr, va, te = assign_splits(sub)
            for bucket, sp in [(tr, "train"), (va, "val"), (te, "test")]:
                for x in bucket:
                    x["split"] = sp

        # ---------------- HELD-OUT VOCAB ----------------
        ho_items, ho_skipped = make_heldout_vocab(concept, heldout_per_value)
        for it in ho_items:
            it["split"] = "heldout_vocab"
            it["judged"] = True
            it["expect"] = True
        stats["heldout_skipped_values"] = ho_skipped

        # ---------------- JUDGE ----------------
        # positives: judge all free-gen + a sample of templates; heldout: judge all
        judge_pos = [p for p in positives if p["generator"] == "free_gen"]
        sample_templ = [p for p in positives if p["generator"] == "template"]
        random.shuffle(sample_templ)
        sample_templ = sample_templ[:int(len(sample_templ) * judge_sample_frac)]
        judge_pos_all = judge_pos + sample_templ
        pos_items = [{"text": p["text"], "claim": p["label_value"],
                      "span": p["text"][p["span"][0][0]:p["span"][0][1]]} for p in judge_pos_all]
        ho_judge_items = [{"text": it["text"], "claim": it["value_name"]} for it in ho_items]

        neg_hard = [x for x in negatives if x["negative_type"] == "hard"]
        sample_softneg = [x for x in negatives if x["negative_type"] != "hard"]
        random.shuffle(sample_softneg)
        sample_softneg = sample_softneg[:int(len(sample_softneg) * judge_sample_frac)]
        neg_judge_all = neg_hard + sample_softneg
        neg_items = [{"text": x["text"]} for x in neg_judge_all]

        KEEP = 2  # >=2/3
        agree_num = []
        if family == "cyclic":
            allowed = [v["name"] for v in cfg["values"]]
            (pos_match, pos_span), (ho_match, _hs), neg_counts = await asyncio.gather(
                judge_identify_all(client, concept, pos_items, allowed, check_span=True),
                judge_identify_all(client, concept, ho_judge_items, allowed, check_span=False),
                judge_all(client, concept, neg_items, expect_entail=False),
            )
            for p, m, so in zip(judge_pos_all, pos_match, pos_span):
                p["_judged"] = True
                p["_keep"] = (m >= KEEP and so >= KEEP)
                agree_num.append(m)
            for it, m in zip(ho_items, ho_match):
                it["_keep"] = (m >= KEEP)
                agree_num.append(m)
        else:  # scalar: identification doesn't apply -> present-check (concept expressed) + span trust
            pos_present, neg_counts = await asyncio.gather(
                judge_all(client, concept, pos_items, expect_entail=False),
                judge_all(client, concept, neg_items, expect_entail=False),
            )
            for p, c in zip(judge_pos_all, pos_present):
                p["_judged"] = True
                p["_keep"] = (c >= KEEP)
                agree_num.append(c)
        for x, c in zip(neg_judge_all, neg_counts):
            x["_njudge"] = c  # for negatives, c = #judges saying concept PRESENT (bad)

        # ---------------- FILTER + ASSEMBLE ----------------
        kept_pos = kept_ho = kept_neg = 0
        # enforce ~template_frac:freegen ratio + value balance by capping kept free_gen to n_free
        # per value, split proportionally 70/15/15 so splits stay stratified.
        tr_c = int(round(n_free * 0.70))
        va_c = int(round(n_free * 0.15))
        cap_by_split = {"train": tr_c, "val": va_c, "test": max(0, n_free - tr_c - va_c)}
        fg_seen = {}
        # positives
        for p in positives:
            if p.get("_judged"):
                if not p.get("_keep"):
                    continue  # drop failed-judge positive
                jv = True
            else:
                jv = None  # clean-by-construction template not in the judged sample
            if family == "cyclic" and p["generator"] == "free_gen":
                key = (p["label_value"], p["split"])
                fg_seen[key] = fg_seen.get(key, 0) + 1
                if fg_seen[key] > cap_by_split.get(p["split"], 0):
                    continue  # trim over-generated free_gen to keep the 60/40 ratio
            rec = build_record(
                concept, family, n, p["text"], p["label_value"], p["label_index"], "positive",
                None, p["span"], p["generator"], True, p["mp_id"], p["split"],
                jv, p["span_target"])
            records.append(rec)
            kept_pos += 1
        # held-out vocab
        for it in ho_items:
            if not it.get("_keep"):
                continue
            rec = build_record(
                concept, family, n, it["text"], it["value_name"], it["value_idx"], "positive",
                None, [it["span"]], "template", False, None, "heldout_vocab", True, it["value_idx"])
            records.append(rec)
            kept_ho += 1
        # negatives
        for x in negatives:
            njudge = x.get("_njudge")
            if njudge is not None and njudge >= KEEP:
                continue  # judge says the concept is actually present -> not a clean negative
            label_idx_none = None
            rec = build_record(
                concept, family, n, x["text"], None, None, "negative", x["negative_type"],
                None, "template" if x["negative_type"] != "hard" else "template", True, None,
                x["split"], (njudge is not None and njudge < KEEP) or None, None)
            records.append(rec)
            kept_neg += 1

    # ---------------- WRITE ----------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    stats.update({
        "n_records": len(records), "kept_positives": kept_pos, "kept_heldout": kept_ho,
        "kept_negatives": kept_neg, "or_stats": client.stats(),
        "judge_pos_agree_mean": (sum(agree_num) / len(agree_num)) if agree_num else None,
    })
    return stats


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--pos-per-value", type=int, default=4000)
    ap.add_argument("--neg-ratio", type=float, default=0.6)
    ap.add_argument("--heldout-per-value", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", f"{args.concept}.jsonl")
    t0 = time.time()
    st = asyncio.run(build_concept(
        args.concept, args.pos_per_value, out, neg_ratio=args.neg_ratio,
        heldout_per_value=args.heldout_per_value, concurrency=args.concurrency))
    st["wall_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(st, indent=2))
