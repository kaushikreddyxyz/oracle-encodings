"""Prompt assembly for Stage 4 generation and judging.

Everything the models see is assembled here from the audited sources:
  config/registry.yaml                       (family construct: categorical/presence/intensity)
  config/families/<family>.yaml              (classes, periphrases, hazards, few-shots)
  prompts/generation/{categorical,intensity}_task.md
  prompts/generation/style_axes.yaml         (register/length cycling)
  prompts/judging/{categorical,intensity}_rubric_v{1,2,3}.md

Presence families reuse the categorical machinery (the salience scale reads as
centrality); intensity families use leveled conditioning and the axis-position
rubric. No prompt text is invented in code — code only fills placeholders and
cycles styles. Each materialized prompt carries a prompt_id and template_id so
outputs trace back to their exact inputs.
"""
import hashlib
import json
import os
import random
import re

import yaml

STAGE4 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_yaml(rel):
    with open(os.path.join(STAGE4, rel)) as f:
        return yaml.safe_load(f)


def _load_sections(rel):
    """Parse a prompts/*.md file into {section_name: text} using ---NAME--- markers."""
    with open(os.path.join(STAGE4, rel)) as f:
        raw = f.read()
    parts = re.split(r"^---([A-Za-z_ ]+)---\s*$", raw, flags=re.M)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        out[parts[i].strip()] = parts[i + 1].strip()
    return out


def load_all(family):
    registry = _load_yaml("config/registry.yaml")
    construct = registry["families"][family]["construct"]
    kind = "intensity" if construct == "intensity" else "categorical"
    return {
        "models": _load_yaml("config/models.yaml"),
        "registry": registry,
        "construct": construct,
        "pack": _load_yaml(f"config/families/{family}.yaml"),
        "styles": _load_yaml("prompts/generation/style_axes.yaml"),
        "gen_task": _load_sections(f"prompts/generation/{kind}_task.md"),
        "rubrics": {v: _load_sections(f"prompts/judging/{kind}_rubric_{v}.md")
                    for v in ("v1", "v2", "v3")},
    }


# --------------------------------------------------------------------- helpers

def _style_cycle(styles, seed):
    """Deterministic shuffled cycle over the weighted register x length product."""
    combos = []
    for r in styles["registers"]:
        for l in styles["lengths"]:
            combos.extend([(r, l)] * (r["weight"] * l["weight"]))
    rng = random.Random(seed)
    rng.shuffle(combos)
    i = 0
    while True:
        yield combos[i % len(combos)]
        i += 1


def _fewshot_block(shots):
    lines = ["Examples of the kind of text wanted (do NOT copy them):"]
    for s in shots:
        lines.append(f'- "{s["text"]}"  ({s["note"]})')
    return "\n".join(lines)


def _surfaces(cls_cfg):
    return cls_cfg["form_train"] + cls_cfg["form_test"]


def _hz_parts(hz, default_form):
    """Hazard entries are either a string (wrong sense of the class's own name)
    or {form, sense} (wrong sense living on another surface, e.g. teal for
    blue-green). Returns (form, sense_description)."""
    if isinstance(hz, dict):
        return hz["form"], hz["sense"]
    return default_form, hz


def _display(name):
    return name.title() if name[0].islower() else name


# ----------------------------------------------------------- generation prompts

def build_gen_prompts(family, volumes, seed):
    """Materialize generation prompts.
    categorical/presence volumes: {explicit, implicit, hard_negative} per class
      + {neutral_nearmiss} family-level.
    intensity volumes: {explicit, implicit} PER LEVEL (7 levels)
      + {hard_negative, neutral_nearmiss} unleveled.
    Returns a list of prompt records."""
    cfg = load_all(family)
    pack, styles, task = cfg["pack"], cfg["styles"], cfg["gen_task"]
    per_call = cfg["models"]["generator"]["items_per_call"]
    fam_noun = pack["concept_noun"]
    cyc = _style_cycle(styles, seed)
    shots = pack["generation_fewshots"]
    records = []

    def emit(cls, slice_name, task_key, fills, n_items, hazard=None, level=None):
        reg, length = next(cyc)
        body = task[task_key]
        for k, v in fills.items():
            body = body.replace("{" + k + "}", str(v))
        body = body.replace("{N}", str(n_items)).replace("{FAMILY_NOUN}", fam_noun)
        fmt = task["FORMAT"].replace("{N}", str(n_items))
        user = (body + "\n\n" + _fewshot_block(shots[slice_name]) + "\n\n"
                + reg["directive"] + " " + length["directive"] + "\n\n" + fmt)
        messages = [{"role": "system", "content": task["SYSTEM"]},
                    {"role": "user", "content": user}]
        template_id = f"{reg['id']}|{length['id']}"
        pid_src = f"{family}|{cls}|{slice_name}|{level}|{template_id}|{len(records)}"
        rec = {
            "prompt_id": "g_" + hashlib.sha1(pid_src.encode()).hexdigest()[:10],
            "family": family, "class": cls, "slice": slice_name, "hazard": hazard,
            "template_id": template_id, "n_items": n_items, "messages": messages,
        }
        if level is not None:
            rec["level"] = level
        records.append(rec)

    def calls_for(total):
        out, left = [], total
        while left > 0:
            out.append(min(per_call, left))
            left -= per_call
        return out

    class_cfgs = pack["classes"]

    if cfg["construct"] == "intensity":
        axis = next(iter(class_cfgs))
        cc = class_cfgs[axis]
        ax = pack["axis"]
        base = {"AXIS": fam_noun, "LOW_ANCHOR": ax["low_anchor"],
                "HIGH_ANCHOR": ax["high_anchor"]}
        banned = ", ".join(sorted(set(_surfaces(cc))))
        for level in range(7):
            lev = {**base, "LEVEL": level, "LEVEL_DESC": ax["levels"][level]}
            for n in calls_for(volumes["explicit"]):
                emit(axis, "explicit", "TASK explicit", lev, n, level=level)
            for n in calls_for(volumes["implicit"]):
                emit(axis, "implicit", "TASK implicit",
                     {**lev, "BANNED": banned}, n, level=level)
        hazards = cc.get("hazards") or []
        if hazards and volumes["hard_negative"] > 0:
            per_h = max(1, volumes["hard_negative"] // len(hazards))
            for hz in hazards:
                for n in calls_for(per_h):
                    emit(axis, "hard_negative", "TASK hard_negative",
                         {**base, "HAZARD": hz}, n, hazard=hz)
        for n in calls_for(volumes["neutral_nearmiss"]):
            emit("_family", "neutral_nearmiss", "TASK neutral_nearmiss",
                 {**base, "BANNED": banned}, n)
        return records

    # categorical / presence
    all_names = [_display(c) for c in class_cfgs]
    for cls, cc in class_cfgs.items():
        disp = _display(cls)
        for n in calls_for(volumes["explicit"]):
            emit(cls, "explicit", "TASK explicit", {"CLASS": disp}, n)
        peri = "\n".join(f"- {p}" for p in cc["periphrases"])
        banned = ", ".join(sorted(set(_surfaces(cc))))
        for n in calls_for(volumes["implicit"]):
            emit(cls, "implicit", "TASK implicit",
                 {"CLASS": disp, "PERIPHRASES": peri, "BANNED": banned}, n)
        hazards = cc.get("hazards") or []
        if hazards and volumes["hard_negative"] > 0:
            per_h = max(1, volumes["hard_negative"] // len(hazards))
            for hz in hazards:
                form, sense = _hz_parts(hz, disp)
                for n in calls_for(per_h):
                    emit(cls, "hard_negative", "TASK hard_negative",
                         {"CLASS": disp, "FORM": form, "HAZARD": sense}, n,
                         hazard=sense)
                    records[-1]["hazard_form"] = form

    sib = ", ".join(all_names)
    banned_all = ", ".join(sorted({s for c in class_cfgs.values() for s in _surfaces(c)}))
    for n in calls_for(volumes["neutral_nearmiss"]):
        emit("_family", "neutral_nearmiss", "TASK neutral_nearmiss",
             {"SIBLINGS": sib, "BANNED": banned_all}, n)
    return records


# --------------------------------------------------------------- judge prompts

def hazard_rules(pack):
    rules = []
    for cls, cc in pack["classes"].items():
        for hz in cc.get("hazards") or []:
            form, sense = _hz_parts(hz, _display(cls))
            rules.append(f'For "{_display(cls)}": {sense} — the target sense is'
                         f' absent there; at most a faint trace (score 1, never above 2).')
    return " ".join(rules) if rules else "No sense hazards are known for this family."


def _judge_fewshot_block(pack):
    lines = ["Worked examples (study the reasoning, then apply it):"]
    for s in pack["judge_fewshots"]:
        spans = s["verdict"]["spans"]
        js = json.dumps({"spans": [
            {"concept": sp["concept"], "quote": sp["quote"], "score": sp["score"]}
            for sp in spans]}, ensure_ascii=False)
        lines.append(f'Passage: "{s["text"]}"\nCorrect spans: {js}   // {s["note"]}')
    return "\n".join(lines)


def build_judge_prompt(family, passages, variant, cfg=None):
    """passages: [{'id': int, 'text': str}]. Returns messages for one judge call."""
    cfg = cfg or load_all(family)
    pack = cfg["pack"]
    rub = cfg["rubrics"][variant]
    if cfg["construct"] == "intensity":
        ax = pack["axis"]
        fills = {"AXIS": pack["concept_noun"], "LOW_ANCHOR": ax["low_anchor"],
                 "HIGH_ANCHOR": ax["high_anchor"],
                 "HAZARD_RULES": hazard_rules(pack),
                 "N_PASSAGES": str(len(passages))}
    else:
        class_list = ", ".join(_display(c) for c in pack["classes"])
        fills = {"FAMILY_NOUN": pack["concept_noun"], "CLASS_LIST": class_list,
                 "HAZARD_RULES": hazard_rules(pack),
                 "N_PASSAGES": str(len(passages))}
    rubric, fmt = rub["RUBRIC"], rub["FORMAT"]
    for k, v in fills.items():
        rubric = rubric.replace("{" + k + "}", v)
        fmt = fmt.replace("{" + k + "}", v)
    body = "\n".join(f'{p["id"]}: {p["text"]}' for p in passages)
    user = (rubric + "\n\n" + _judge_fewshot_block(pack)
            + "\n\n## Passages\n" + body + "\n\n" + fmt)
    return [{"role": "system", "content": rub["SYSTEM"]},
            {"role": "user", "content": user}]


# ------------------------------------------------ form-holdout supplement (§6.2)

def build_form_prompts(family, per_class, skip_classes=()):
    """Explicit positives conditioned to use ONLY a held-out (form_test) surface.
    Fills the §6.2 lexical holdout for classes where free generation never uses
    the held-out form (e.g. autumn -> 'fall'). Returns prompt records with
    slice='explicit_form' and a 'form' field."""
    cfg = load_all(family)
    pack, styles, task = cfg["pack"], cfg["styles"], cfg["gen_task"]
    per_call = cfg["models"]["generator"]["items_per_call"]
    fam_noun = pack["concept_noun"]
    cyc = _style_cycle(styles, cfg["models"]["runtime"]["seed"] + 1)
    shots = pack["generation_fewshots"]
    intensity = cfg["construct"] == "intensity"
    records = []
    for cls, cc in pack["classes"].items():
        if cls in skip_classes or not cc["form_test"]:
            continue
        forms = cc["form_test"]
        per_form = max(1, per_class // len(forms))
        for fi, form in enumerate(forms):
            other = [s for s in cc["form_train"] + forms if s != form]
            left = per_form
            lvl_i = 0
            while left > 0:
                n = min(per_call, left)
                left -= n
                reg, length = next(cyc)
                fills = {"CLASS": _display(cls), "FORM": form,
                         "OTHER_FORMS": ", ".join(other)}
                level = None
                if intensity:
                    ax = pack["axis"]
                    level = [1, 2, 4, 5, 6, 3][lvl_i % 6]  # skip 0 (form words rarely fit 'free/tiny' text)
                    lvl_i += 1
                    fills.update({"AXIS": fam_noun, "LOW_ANCHOR": ax["low_anchor"],
                                  "HIGH_ANCHOR": ax["high_anchor"], "LEVEL": level,
                                  "LEVEL_DESC": ax["levels"][level]})
                body = task["TASK explicit_form"]
                for k, v in fills.items():
                    body = body.replace("{" + k + "}", str(v))
                body = body.replace("{N}", str(n)).replace("{FAMILY_NOUN}", fam_noun)
                fmt = task["FORMAT"].replace("{N}", str(n))
                user = (body + "\n\n" + _fewshot_block(shots["explicit"]) + "\n\n"
                        + reg["directive"] + " " + length["directive"] + "\n\n" + fmt)
                pid_src = f"{family}|{cls}|explicit_form|{form}|{level}|{len(records)}"
                rec = {"prompt_id": "f_" + hashlib.sha1(pid_src.encode()).hexdigest()[:10],
                       "family": family, "class": cls, "slice": "explicit_form",
                       "hazard": None, "form": form,
                       "template_id": f"{reg['id']}|{length['id']}",
                       "n_items": n,
                       "messages": [{"role": "system", "content": task["SYSTEM"]},
                                    {"role": "user", "content": user}]}
                if level is not None:
                    rec["level"] = level
                records.append(rec)
    return records
