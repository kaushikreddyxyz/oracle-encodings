#!/usr/bin/env python3
"""Validate that every concept prompt_id has a non-empty, class-specific prompt file."""
import json
import os
import re
import sys

OVR = "/Users/kaushikreddy/Projects/oracle-encoding-project/oracle-encodings/overnight_run"
sys.path.insert(0, OVR)
import concepts  # noqa: E402

reg_path = os.path.join(OVR, "prompts", "registry.json")
with open(reg_path) as f:
    registry = json.load(f)

expected = concepts.presence_probe_ids() + concepts.scalar_probe_ids()
problems = []

# 1. every expected id present in registry
missing = [pid for pid in expected if pid not in registry]
if missing:
    problems.append(f"MISSING from registry: {missing}")

# 2. no stray ids in registry
stray = [pid for pid in registry if pid not in expected]
if stray:
    problems.append(f"STRAY ids in registry: {stray}")

# 3. each file exists, non-empty, mentions the class, has output instruction + few-shots
def class_of(pid):
    if pid.startswith("scalar::"):
        return pid.split("::", 1)[1]
    return pid.split("::", 1)[1]

for pid, rel in registry.items():
    path = os.path.join(OVR, rel)
    if not os.path.isfile(path):
        problems.append(f"{pid}: file not found {rel}")
        continue
    text = open(path, encoding="utf-8").read()
    if len(text.strip()) < 200:
        problems.append(f"{pid}: file too short ({len(text)} chars)")
    cls = class_of(pid)
    # mentions the class: literal token, sanitized form, space<->underscore, or singular
    hay = text.lower()
    variants = {cls.lower(), re.sub(r"\s+", "_", cls).lower(),
                cls.replace("_", " ").lower(), cls.lower().rstrip("s")}
    mentions = any(v in hay for v in variants)
    if not mentions and not pid.startswith("scalar::"):
        cname = pid.split("::", 1)[0]
        lex = concepts.PRESENCE_CONCEPTS[cname]["classes"][cls].get("lexicon", [])
        mentions = any(l.lower() in hay for l in lex)
    if not mentions:
        problems.append(f"{pid}: does not mention class '{cls}'")
    # output instruction present
    if "first token" not in hay:
        problems.append(f"{pid}: missing front-loaded output instruction")
    # at least 6 worked few-shots (lines beginning with a digit score)
    shot_lines = [ln for ln in text.splitlines() if re.match(r'^\d\s+"', ln)]
    if len(shot_lines) < 6:
        problems.append(f"{pid}: only {len(shot_lines)} few-shots (<6)")

# 4. count balance: presence files have a 0 and a 5 example (positives + negatives)
for pid, rel in registry.items():
    if pid.startswith("scalar::"):
        continue
    text = open(os.path.join(OVR, rel), encoding="utf-8").read()
    scores = [int(m.group(1)) for m in re.finditer(r'^(\d)\s+"', text, re.M)]
    if 5 not in scores:
        problems.append(f"{pid}: no positive (score 5) example")
    if 0 not in scores:
        problems.append(f"{pid}: no negative (score 0) example")

print(f"registry ids: {len(registry)}   expected: {len(expected)}")
print(f"presence: {len(concepts.presence_probe_ids())}  scalar: {len(concepts.scalar_probe_ids())}")
if problems:
    print(f"\nFAIL — {len(problems)} problem(s):")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("\nALL CHECKS PASSED")
