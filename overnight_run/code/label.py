"""
label.py — Step 1 of the overnight concept-probes run: turn surface-matched
candidates into supervised probe labels using a strong instruct LLM as judge.

Pipeline (per concept):
  candidates jsonl  ->  bespoke system prompt (registry)  ->  N=5 judge samples
  (show-then-ask, temp>0, genuine sampling)  ->  robust leading-number parse
  ->  aggregate mean/std  ->  per-regime threshold/filter  ->  ~50/50 balance
  (presence)  ->  write labels jsonl + validation json.

The judge is abstracted behind `JudgeClient.score(system, user, n) -> list[str]`:
  * VLLMJudgeClient — OpenAI-compatible (httpx -> vLLM /v1/chat/completions), used
    on the pod (serve_judge.sh).
  * MockJudgeClient — offline; derives plausible scores from the candidate so the
    full pipeline (parse/aggregate/threshold/balance/validate/schema) is testable
    with no GPU. Used by `--smoke`.

Resumable: a concept whose labels file already exists is skipped.

Usage:
  python label.py --smoke                 # offline self-test, exits 0
  python label.py                         # all concepts, real vLLM judge
  python label.py --concepts months days  # subset
  python label.py --push                  # also upload labels+candidates to HF
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Optional

# overnight_run/code and overnight_run on path (per SPEC) so imports resolve when
# run directly from anywhere.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))            # overnight_run/code
sys.path.insert(0, str(_HERE.parent))     # overnight_run

import config            # noqa: E402
import concepts          # noqa: E402

# --------------------------------------------------------------------------- #
# Tunables not in config (Step-1 internal)
# --------------------------------------------------------------------------- #
MIN_VALID_SAMPLES = 3          # of N=5, require a majority of parseable samples
# regex: first signed int/float anywhere in the string (front-loaded output).
_NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\.\d+|\d+)")

# Default data dirs (overridable -> used so --smoke runs in a scratch dir).
CANDIDATES_DIR = config.DATA / "candidates"
LABELS_DIR = config.DATA / "labels"


# =========================================================================== #
# Judge clients
# =========================================================================== #
class JudgeClient(ABC):
    """Score one (system, user) example. Returns N raw, unparsed completion strings.

    `meta` is the candidate dict — VLLM ignores it; the mock uses it to fabricate
    plausible offline scores. The canonical interface is score(system, user, n).
    """

    @abstractmethod
    def score(self, system: str, user: str, n: int, meta: Optional[dict] = None) -> list[str]:
        ...


class VLLMJudgeClient(JudgeClient):
    """OpenAI-compatible client (httpx) for a vLLM server (serve_judge.sh)."""

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = config.JUDGE_MODEL,
                 temperature: float = config.JUDGE_TEMPERATURE,
                 max_tokens: int = config.JUDGE_MAX_TOKENS,
                 api_key: str = "EMPTY", timeout: float = 120.0):
        import httpx  # local import: not needed for --smoke
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.Client(
            timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}
        )

    def _request(self, system: str, user: str, n: int, seed: Optional[int] = None) -> list[str]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "n": n,                              # genuine independent samples
            "temperature": self.temperature,     # >0 -> real sampling, not greedy copies
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        r = self._client.post(self.url, json=body)
        r.raise_for_status()
        choices = r.json().get("choices", [])
        return [c.get("message", {}).get("content", "") or "" for c in choices]

    def score(self, system: str, user: str, n: int, meta: Optional[dict] = None) -> list[str]:
        # Ask for n in one shot; if the server returns fewer, top up with
        # seeded single-sample requests so we always end with n genuine samples.
        outs = self._request(system, user, n)
        seed = 0
        while len(outs) < n:
            seed += 1
            outs.extend(self._request(system, user, 1, seed=seed))
        return outs[:n]


class MockJudgeClient(JudgeClient):
    """Offline judge: plausible scores derived from the candidate (`meta`).

    Control hooks (smoke only): meta["_mock_scores"] = exact per-sample scores
    (None -> a malformed/unparseable sample). Otherwise a heuristic base score is
    jittered to produce genuine N-sample spread.
    """

    def score(self, system: str, user: str, n: int, meta: Optional[dict] = None) -> list[str]:
        meta = meta or {}
        if "_mock_scores" in meta:
            vals = meta["_mock_scores"]
            return [("garbage" if vals[i % len(vals)] is None
                     else f"{vals[i % len(vals)]} mock") for i in range(n)]
        base = self._heuristic_base(meta, user)
        rng = Random(hash((user, base)) & 0xFFFFFFFF)
        out = []
        for _ in range(n):
            v = max(0, min(5, round(base + rng.uniform(-0.5, 0.5))))
            out.append(f"{v} mock")
        return out

    @staticmethod
    def _heuristic_base(meta: dict, user: str) -> float:
        if meta.get("regime") == "scalar":
            ext = meta.get("external")
            return float(ext) if ext is not None else 2.5
        # presence: surface-matched -> likely positive; else likely negative.
        return 4.5 if meta.get("match_surface") else 0.5


# =========================================================================== #
# Prompts / registry
# =========================================================================== #
def prompt_id_for(cand: dict) -> str:
    if cand.get("regime") == "presence":
        return f"{cand['concept']}::{cand['cls']}"
    return f"scalar::{cand['concept']}"


def load_registry(prompts_dir: Path) -> Optional[dict]:
    reg = prompts_dir / "registry.json"
    if not reg.exists():
        return None
    return json.loads(reg.read_text())


def _resolve_prompt_path(raw: str, prompts_dir: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (prompts_dir, prompts_dir.parent):  # prompts/ or overnight_run/
        if (base / p).exists():
            return base / p
    return prompts_dir / p


def _generic_system(cand: dict) -> str:
    """Fallback system prompt when the registry/bespoke prompt is missing.

    REAL runs require the registry (prompts are the whole ballgame). This exists
    only so --smoke works before the prompts subagent has finished.
    """
    concept, cls = cand["concept"], cand.get("cls")
    if cand.get("regime") == "presence":
        return (
            f"You judge whether the word/concept '{cls}' (from '{concept}') is "
            f"LITERALLY, TEXTUALLY present in the text (abbreviations/misspellings/"
            f"casing count; mere associations or proper-name uses do NOT). "
            f"Output a single integer 0-5 as the FIRST token (5=clearly present in "
            f"the right sense, 0=absent/wrong sense), then an optional <=5-word reason."
        )
    return (
        f"You rate the text on the scalar '{concept}' from 0 (low pole) to 5 "
        f"(high pole), grounded in the text. Output the number FIRST, then an "
        f"optional <=5-word reason."
    )


def get_system_prompt(cand: dict, registry: Optional[dict], prompts_dir: Path,
                      require: bool) -> str:
    pid = prompt_id_for(cand)
    if registry and pid in registry:
        val = registry[pid]
        raw = val["path"] if isinstance(val, dict) else val
        return _resolve_prompt_path(raw, prompts_dir).read_text()
    if require:
        raise FileNotFoundError(
            f"prompt for '{pid}' missing from registry {prompts_dir/'registry.json'} "
            f"(real runs REQUIRE bespoke prompts; only --smoke may fall back)."
        )
    return _generic_system(cand)


# =========================================================================== #
# Parse + aggregate
# =========================================================================== #
def parse_score(raw: Optional[str]) -> Optional[float]:
    """First number in [0,5]; out-of-range or unparseable -> None (never guess)."""
    if not raw:
        return None
    m = _NUM_RE.search(raw)
    if not m:
        return None
    try:
        v = float(m.group())
    except ValueError:
        return None
    if v < 0.0 or v > 5.0:
        return None
    return int(v) if v.is_integer() else v


def aggregate(raws: list[str]) -> dict:
    parsed = [parse_score(r) for r in raws]
    valid = [v for v in parsed if v is not None]
    n_malformed = len(parsed) - len(valid)
    if len(valid) < MIN_VALID_SAMPLES:
        return {"scores": valid, "n_valid": len(valid), "n_malformed": n_malformed,
                "mean": None, "std": None, "ok": False}
    mean = round(statistics.fmean(valid), 4)
    std = round(statistics.stdev(valid), 4) if len(valid) >= 2 else 0.0
    return {"scores": valid, "n_valid": len(valid), "n_malformed": n_malformed,
            "mean": mean, "std": std, "ok": True}


# =========================================================================== #
# Token-position helper (char_span -> probe-target token indices)
# =========================================================================== #
_TOK_CACHE: dict = {}


def get_tokenizer(model: str = config.PROBE_TARGET, offline: bool = False):
    """Try to load the probe-target tokenizer; return None if unavailable
    (gated/offline). probe.py can resolve char_span -> tokens later either way."""
    if model in _TOK_CACHE:
        return _TOK_CACHE[model]
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, local_files_only=offline)
    except Exception:
        tok = None
    _TOK_CACHE[model] = tok
    return tok


def char_span_to_tokens(text: str, char_span, tokenizer) -> Optional[list[int]]:
    """Token indices overlapping the matched char span (span-local — NOT broadcast
    onto every token). Needs a fast tokenizer (offset_mapping)."""
    if tokenizer is None or not char_span:
        return None
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
    except Exception:
        return None
    s, e = char_span
    return [i for i, (a, b) in enumerate(offsets) if a < e and b > s and b > a]


# =========================================================================== #
# Per-candidate labeling
# =========================================================================== #
def label_candidate(cand: dict, judge: JudgeClient, system: str,
                    tokenizer=None) -> dict:
    raws = judge.score(system, cand["text"], config.N_SAMPLES, meta=cand)
    agg = aggregate(raws)
    regime = cand.get("regime")
    rec = {
        "id": cand["id"], "concept": cand["concept"], "cls": cand.get("cls"),
        "regime": regime, "text": cand["text"], "char_span": cand.get("char_span"),
        "match_surface": cand.get("match_surface"), "shard": cand.get("shard"),
        "external": cand.get("external"),
        "scores": agg["scores"], "n_valid": agg["n_valid"],
        "n_malformed": agg["n_malformed"], "mean": agg["mean"], "std": agg["std"],
        "label": None, "value": None,
        "token_positions": char_span_to_tokens(cand["text"], cand.get("char_span"), tokenizer),
        "prompt_id": prompt_id_for(cand), "judge_model": config.JUDGE_MODEL,
        "discarded": False, "discard_reason": None,
    }

    if not agg["ok"]:
        rec.update(discarded=True, discard_reason="too_few_valid_samples")
        return rec

    if regime == "presence":
        m = agg["mean"]
        if m >= config.PRESENCE_POS_THRESH:
            rec["label"] = 1
        elif m <= config.PRESENCE_NEG_THRESH:
            rec["label"] = 0
        else:
            rec.update(discarded=True, discard_reason="ambiguous_mean")
    else:  # scalar
        ext = cand.get("external")
        rec["value"] = float(ext) if ext is not None else agg["mean"]
        if ext is None and agg["std"] > config.SCALAR_MAX_SEED_STD:
            # No ground truth + judge can't agree with itself -> drop.
            rec.update(discarded=True, discard_reason="high_seed_variance")
        # external rows are kept (authoritative); judge mean/std stay as confidence.
    return rec


# =========================================================================== #
# Balance (presence: ~50/50 pos/neg per class, by capping the majority)
# =========================================================================== #
def balance_presence(records: list[dict]) -> dict:
    """Mark majority-side overflow as discarded(reason=balance_cap) per class so
    each class probe trains ~50/50. label stays set (validation still sees it)."""
    by_cls = defaultdict(lambda: {1: [], 0: []})
    for r in records:
        if r["discarded"] or r["label"] is None:
            continue
        by_cls[r["cls"]][r["label"]].append(r)

    capped = 0
    for cls, sides in by_cls.items():
        pos, neg = sides[1], sides[0]
        if not pos or not neg:
            continue  # nothing to balance against; leave as-is (note imbalance)
        keep = min(len(pos), len(neg))
        # keep the most confident: positives by highest mean, negatives by lowest.
        pos.sort(key=lambda r: r["mean"], reverse=True)
        neg.sort(key=lambda r: r["mean"])
        for overflow in (pos[keep:] + neg[keep:]):
            overflow.update(discarded=True, discard_reason="balance_cap")
            capped += 1
    return {"capped": capped}


# =========================================================================== #
# Validation
# =========================================================================== #
def _pearson(xs, ys) -> Optional[float]:
    if len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return round(num / (dx * dy), 4) if dx and dy else None


def build_validation(concept: str, regime: str, records: list[dict]) -> dict:
    reasons = Counter(r["discard_reason"] for r in records if r["discarded"])
    kept = [r for r in records if not r["discarded"]]
    out = {
        "concept": concept, "regime": regime, "judge_model": config.JUDGE_MODEL,
        "n_candidates": len(records), "n_kept": len(kept),
        "n_discarded": sum(1 for r in records if r["discarded"]),
        "discard_reasons": dict(reasons),
    }
    if regime == "presence":
        # judge-vs-pseudo-gold: pseudo-gold = surface-match==positive assumption.
        graded = [r for r in records if r["label"] is not None]
        agree = sum(1 for r in graded
                    if r["label"] == (1 if r.get("match_surface") else 0))
        out.update(
            n_pos=sum(1 for r in kept if r["label"] == 1),
            n_neg=sum(1 for r in kept if r["label"] == 0),
            pseudo_gold_n=len(graded),
            pseudo_gold_agreement=round(agree / len(graded), 4) if graded else None,
        )
    else:  # scalar
        seed_stds = [r["std"] for r in kept if r["std"] is not None]
        ext_pairs = [(r["external"], r["mean"]) for r in kept
                     if r.get("external") is not None and r["mean"] is not None]
        out.update(
            n_with_external=len(ext_pairs),
            mean_seed_std=round(statistics.fmean(seed_stds), 4) if seed_stds else None,
            external_vs_judge_pearson=_pearson([float(a) for a, _ in ext_pairs],
                                               [b for _, b in ext_pairs]),
        )
    return out


# =========================================================================== #
# Concept driver
# =========================================================================== #
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def label_concept(concept: str, judge: JudgeClient, *, candidates_dir: Path,
                  labels_dir: Path, prompts_dir: Path, registry: Optional[dict],
                  require_prompts: bool, tokenizer=None) -> Optional[dict]:
    cand_path = candidates_dir / f"{concept}.jsonl"
    out_path = labels_dir / f"{concept}.jsonl"
    val_path = labels_dir / f"validation_{concept}.json"

    if out_path.exists():                      # RESUMABLE
        print(f"[label] {concept}: labels exist -> skip")
        return None
    if not cand_path.exists():
        print(f"[label] {concept}: no candidates ({cand_path}) -> skip")
        return None

    cands = read_jsonl(cand_path)
    regime = cands[0].get("regime") if cands else "presence"
    # Prompt loading single-threaded (populates any cache safely), then fan out the
    # network-bound judge calls across threads — vLLM continuous-batches them. This
    # is the throughput lever for a 28k-candidate run; aggregation/filter is unchanged.
    sys_prompts = [get_system_prompt(c, registry, prompts_dir, require_prompts) for c in cands]
    workers = int(os.environ.get("LABEL_CONCURRENCY", "48"))

    def _do(pair):
        c, sp = pair
        return label_candidate(c, judge, sp, tokenizer=tokenizer)

    if workers > 1 and len(cands) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            records = list(ex.map(_do, list(zip(cands, sys_prompts))))
    else:
        records = [_do(p) for p in zip(cands, sys_prompts)]
    if regime == "presence":
        balance_presence(records)

    validation = build_validation(concept, regime, records)
    write_jsonl(out_path, records)
    val_path.write_text(json.dumps(validation, indent=2))
    print(f"[label] {concept}: {len(records)} candidates -> {out_path.name} "
          f"(kept={validation['n_kept']} discarded={validation['n_discarded']})")
    return validation


# =========================================================================== #
# HF push
# =========================================================================== #
def push_labels(repo: str = config.HF_DATASET_REPO, *, labels_dir: Path = LABELS_DIR,
                candidates_dir: Path = CANDIDATES_DIR):
    """Upload labels/ + candidates/ to the HF dataset repo (cached-login token)."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=config.HF_PRIVATE, exist_ok=True)
    if labels_dir.exists():
        api.upload_folder(folder_path=str(labels_dir), path_in_repo="labels",
                          repo_id=repo, repo_type="dataset")
    if candidates_dir.exists():
        api.upload_folder(folder_path=str(candidates_dir), path_in_repo="candidates",
                          repo_id=repo, repo_type="dataset")
    print(f"[label] pushed labels+candidates -> hf.co/datasets/{repo}")


# =========================================================================== #
# main / smoke
# =========================================================================== #
def discover_concepts(candidates_dir: Path) -> list[str]:
    return sorted(p.stem for p in candidates_dir.glob("*.jsonl"))


def run(concepts_list, judge, *, candidates_dir, labels_dir, prompts_dir,
        require_prompts, tokenizer=None):
    registry = load_registry(prompts_dir)
    if registry is None and require_prompts:
        raise FileNotFoundError(f"registry missing: {prompts_dir/'registry.json'}")
    for c in concepts_list:
        label_concept(c, judge, candidates_dir=candidates_dir, labels_dir=labels_dir,
                      prompts_dir=prompts_dir, registry=registry,
                      require_prompts=require_prompts, tokenizer=tokenizer)


def _smoke():
    """Full pipeline offline with MockJudgeClient on synthetic candidates.

    Exercises: parse, malformed-discard, aggregate, presence threshold
    (pos/neg/ambiguous), per-class balance cap, scalar value/external/high-variance,
    validation (agreement + counts), schema, resumability. Exits 0 on success.
    """
    tmp = Path(tempfile.mkdtemp(prefix="label_smoke_"))
    cdir, ldir = tmp / "candidates", tmp / "labels"
    cdir.mkdir(parents=True)
    pdir = tmp / "prompts"  # intentionally absent -> generic-prompt fallback path

    def pc(cid, cls, text, span, surface, scores, regime="presence", concept="months",
           external=None):
        return {"id": cid, "concept": concept, "cls": cls, "regime": regime,
                "text": text, "char_span": span, "match_surface": surface,
                "shard": 300, "external": external, "_mock_scores": scores}

    # presence "months": 4 positives + 2 negatives for January (-> balance caps
    # positives to 2), one ambiguous (discard), one malformed (discard), plus a
    # May class with 1 pos / 0 neg (left imbalanced, not nuked).
    months = [
        pc("m1", "January", "My birthday is in January.", [16, 23], "January", [5, 5, 4, 5, 5]),
        pc("m2", "January", "We met back in Jan. 2019.", [14, 18], "Jan.", [5, 4, 5, 5, 4]),
        pc("m3", "January", "A cold Janurary morning.", [7, 15], "Janurary", [4, 4, 5, 4, 4]),
        pc("m4", "January", "See you in January again.", [10, 17], "January", [5, 5, 5, 4, 5]),
        pc("m5", "January", "January Jones starred in it.", [0, 7], "January", [1, 0, 0, 0, 1]),
        pc("m6", "January", "The janitor mopped the hall.", [4, 11], "jan", [0, 0, 0, 1, 0]),
        pc("m7", "January", "Unclear mention maybe january?", [21, 28], "january", [3, 2, 3, 2, 3]),
        pc("m8", "January", "Totally broken sample row.", [0, 7], "Jan", [None, None, None, None, None]),
        pc("may1", "May", "See you in May.", [11, 14], "May", [5, 5, 4, 5, 5]),
    ]
    write_jsonl(cdir / "months.jsonl", [{k: v for k, v in r.items()} for r in months])

    # scalar "costliness": external-backed rows (kept, authoritative) + a no-external
    # high-variance row (discarded) + a clean no-external row.
    cost = [
        pc("c1", None, "It was dirt cheap, basically free.", [7, 17], None, [0, 1, 0, 0, 1],
           regime="scalar", concept="costliness", external=0.5),
        pc("c2", None, "An utterly priceless masterpiece.", [10, 19], None, [5, 5, 4, 5, 5],
           regime="scalar", concept="costliness", external=4.8),
        pc("c3", None, "Moderately priced, nothing wild.", [0, 10], None, [2, 3, 2, 3, 2],
           regime="scalar", concept="costliness", external=None),
        pc("c4", None, "Ambiguous cost signals everywhere.", [0, 9], None, [0, 5, 0, 5, 2],
           regime="scalar", concept="costliness", external=None),
    ]
    write_jsonl(cdir / "costliness.jsonl", cost)

    judge = MockJudgeClient()
    run(["months", "costliness"], judge, candidates_dir=cdir, labels_dir=ldir,
        prompts_dir=pdir, require_prompts=False, tokenizer=None)

    # ---- assertions ----
    REQUIRED = {"id", "concept", "cls", "regime", "text", "char_span", "scores",
                "mean", "std", "label", "value", "prompt_id", "judge_model",
                "discarded", "discard_reason", "token_positions", "external"}
    mrecs = read_jsonl(ldir / "months.jsonl")
    crecs = read_jsonl(ldir / "costliness.jsonl")
    for r in mrecs + crecs:
        assert REQUIRED <= set(r), f"missing keys: {REQUIRED - set(r)}"
        assert "_mock_scores" not in r, "private mock key leaked into labels"

    by = {r["id"]: r for r in mrecs}
    assert by["m1"]["label"] == 1 and by["m1"]["mean"] >= config.PRESENCE_POS_THRESH
    assert by["m5"]["label"] == 0, "name-sense should threshold to negative"
    assert by["m7"]["discarded"] and by["m7"]["discard_reason"] == "ambiguous_mean"
    assert by["m8"]["discarded"] and by["m8"]["discard_reason"] == "too_few_valid_samples"
    assert by["m8"]["n_malformed"] == 5 and by["m8"]["mean"] is None
    assert by["m1"]["prompt_id"] == "months::January"

    # balance: January had 4 pos / 2 neg -> capped to 2 pos / 2 neg.
    jan_kept_pos = [r for r in mrecs if r["cls"] == "January"
                    and r["label"] == 1 and not r["discarded"]]
    jan_kept_neg = [r for r in mrecs if r["cls"] == "January"
                    and r["label"] == 0 and not r["discarded"]]
    assert len(jan_kept_pos) == 2 and len(jan_kept_neg) == 2, \
        (len(jan_kept_pos), len(jan_kept_neg))
    assert any(r["discard_reason"] == "balance_cap" for r in mrecs)

    # scalar: external authoritative; high-variance no-external discarded.
    cby = {r["id"]: r for r in crecs}
    assert cby["c2"]["value"] == 4.8 and cby["c2"]["label"] is None
    assert cby["c3"]["value"] == cby["c3"]["mean"]
    assert cby["c4"]["discarded"] and cby["c4"]["discard_reason"] == "high_seed_variance"

    # validation files
    mval = json.loads((ldir / "validation_months.json").read_text())
    cval = json.loads((ldir / "validation_costliness.json").read_text())
    assert mval["pseudo_gold_agreement"] is not None
    assert {"n_pos", "n_neg", "n_discarded", "discard_reasons"} <= set(mval)
    assert cval["n_with_external"] == 2 and "mean_seed_std" in cval

    # resumability: re-run must skip (mtime unchanged)
    before = (ldir / "months.jsonl").stat().st_mtime_ns
    run(["months"], judge, candidates_dir=cdir, labels_dir=ldir, prompts_dir=pdir,
        require_prompts=False)
    assert (ldir / "months.jsonl").stat().st_mtime_ns == before, "resume skip failed"

    print("\n[smoke] PASS")
    print(f"[smoke] tmp dir: {tmp}")
    print(f"[smoke] months : kept={mval['n_kept']} pos={mval['n_pos']} "
          f"neg={mval['n_neg']} discarded={mval['n_discarded']} "
          f"agreement={mval['pseudo_gold_agreement']} reasons={mval['discard_reasons']}")
    print(f"[smoke] cost   : kept={cval['n_kept']} discarded={cval['n_discarded']} "
          f"external={cval['n_with_external']} "
          f"ext_vs_judge_r={cval['external_vs_judge_pearson']} "
          f"mean_seed_std={cval['mean_seed_std']}")
    print(f"[smoke] sample label row: {json.dumps(by['m1'])}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Step-1 judge labeling")
    ap.add_argument("--smoke", action="store_true", help="offline self-test, exit 0")
    ap.add_argument("--concepts", nargs="*", default=None,
                    help="subset of concepts (default: all candidate files)")
    ap.add_argument("--push", action="store_true", help="upload to HF after labeling")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    args = ap.parse_args()

    if args.smoke:
        return _smoke()

    judge = VLLMJudgeClient(base_url=args.base_url)
    targets = args.concepts or discover_concepts(CANDIDATES_DIR)
    if not targets:
        print(f"[label] no candidate files in {CANDIDATES_DIR}; run build_candidates.py first")
        return 1
    run(targets, judge, candidates_dir=CANDIDATES_DIR, labels_dir=LABELS_DIR,
        prompts_dir=config.PROMPTS, require_prompts=True, tokenizer=get_tokenizer())
    if args.push:
        push_labels()
    return 0


if __name__ == "__main__":
    sys.exit(main())
