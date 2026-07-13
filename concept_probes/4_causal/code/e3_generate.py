"""Stage 6.1 E3 (generation half): steered generations for judge scoring
(task.md §6.1.5, AxBench-adapted for a base model).

Per concept (all 64 with probe cards; glorptitude has no natscores => no dose
calibration => skipped with a warning):
- 10 neutral ClimbMix prefixes sampled deterministically from the Stage-6
  natural random pool (3_validation/data/natural/random_pool.jsonl; fields
  example_id/doc_id/shard/text/nat_split — confirmed 2026-07-02). Fallback if
  the pool file is missing on this machine: neutral rows (no aggregated_spans)
  of 1_dataset/data/<family>/judged/judged_nat.jsonl. Each prefix is truncated to
  --prefix-tokens (~40) tokens; prefixes containing the concept's surface form
  (word-boundary, case-insensitive) are excluded when that filter leaves
  enough candidates (recorded per row as prefix_filtered).
- Split: the first 5 sampled prefixes are 'selection' (dose selection), the
  last 5 'heldout' (reporting) — recorded in every output row.
- Arms: ridge, dom, rand (= rand_dirs[0]) + the mandatory alpha=0 'baseline'.
- Doses: --factors, default = {selected factor} ∪ {one lower dose}: selected
  is read from <out>/e2_cloze/selection.json when present (tolerated schemas:
  {family: {class: {...}}} or {concept: {...}}, with factor under
  'factor'/'selected_factor'/'best_factor' and optional 'layer'), else 2.0;
  the lower dose is 1.0 (or selected/2 when selected <= 1.0).
- Layer: the concept's chosen layer from 3_validation/artifacts/probe_cards.json
  (selection.json layer, then --layers, override it in that priority order).
- alpha = factor * s95(concept, layer) via common.dose_calib.
- Generation: max_new_tokens 128, do_sample, temperature 1.0, top_p=1.0,
  top_k=0, repetition_penalty=1.0 (pure sampling, no repetition penalty),
  left-padded batches of all prefixes at once (capped by --batch-tokens).
  Hooks steer at ALL positions (DESIGN default), including each newly
  generated token under the KV cache.

Output: <out>/e3/generations_<family>.jsonl, one row per generation:
  {concept, family, layer, arm, factor, alpha, prefix_id, split, prefix,
   continuation, seed, prefix_source, prefix_filtered, n_prefix_tokens}
Resume-safe: existing (concept, layer, arm, factor, prefix_id) rows are kept
and skipped. Heartbeats append to <out>/progress_e3_generate.log.

--smoke: 2-layer random Gemma2 (d=64, CPU fp32), a deterministic fake
tokenizer, synthetic arms/calibration/prefix pool — exercises the full loop
(hooks, batching, resume, output schema) with no downloads and no GPU.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
import common                                        # noqa: E402
from interventions import Hooks, Intervention        # noqa: E402

CARDS_PATH = common.CP_DIR / "3_validation" / "artifacts" / "probe_cards.json"
POOL_PATH = common.CP_DIR / "3_validation" / "data" / "natural" / "random_pool.jsonl"
STAGE4_DATA = common.CP_DIR / "1_dataset" / "data"

ARMS_DEFAULT = ["ridge", "dom", "rand"]


def canon(name: str) -> str:
    return str(name).replace(" ", "_")


def stable_seed(base: int, *parts) -> int:
    return (base + zlib.crc32("|".join(str(p) for p in parts).encode())) % (2**31)


# --------------------------------------------------------------- inputs
def load_cards() -> dict:
    """{(family, concept): card} from probe_cards.json (concepts canonized)."""
    cards = json.loads(CARDS_PATH.read_text())
    return {(c["family"], canon(c["concept"])): c for c in cards}


def load_selection(out_dir: Path) -> dict:
    """Tolerant reader for <out>/e2_cloze/selection.json (written by the
    concurrent e2_cloze agent; schema not frozen yet — see docstring)."""
    p = Path(out_dir) / "e2_cloze" / "selection.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text())
    except Exception as e:                                    # noqa: BLE001
        print(f"WARNING: could not parse {p}: {e} — using default factors")
        return {}
    sel: dict = {}

    def put(concept, d):
        if not isinstance(d, dict):
            return
        f = next((d[k] for k in ("factor", "selected_factor", "best_factor")
                  if isinstance(d.get(k), (int, float))), None)
        L = next((d[k] for k in ("layer", "selected_layer", "best_layer")
                  if isinstance(d.get(k), int)), None)
        if f is not None or L is not None:
            sel[canon(concept)] = {"factor": f, "layer": L}

    for k, v in obj.items():
        if k in common.FAMILIES and isinstance(v, dict):
            for cls, d in v.items():
                put(cls, d)
        else:
            put(k, v)
    return sel


def load_pool(family: str) -> tuple[list[dict], str]:
    """Neutral prefix candidates: the ClimbMix random pool, else neutral rows
    of the family's judged_nat.jsonl (no aggregated spans)."""
    if POOL_PATH.exists():
        rows = [json.loads(l) for l in POOL_PATH.open()]
        return ([{"prefix_id": r["example_id"], "text": r["text"]} for r in rows],
                "random_pool")
    p = STAGE4_DATA / family / "judged" / "judged_nat.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"neither {POOL_PATH} nor {p} exists — no neutral prefix source")
    rows = []
    for l in p.open():
        r = json.loads(l)
        if not r.get("aggregated_spans"):
            rows.append({"prefix_id": r["example_id"], "text": r["text"]})
    return rows, "judged_nat_neutral"


def sample_prefixes(pool: list[dict], concept: str, tok, n: int,
                    prefix_tokens: int, seed: int) -> list[dict]:
    """Deterministic per-concept sample of n prefixes, truncated to
    prefix_tokens tokens. Surface-form filter is applied only when it leaves
    >= 5n candidates (e.g. 'may' as a modal would otherwise gut the pool)."""
    surface = concept.replace("_", " ")
    pat = re.compile(r"\b" + re.escape(surface) + r"\b", re.IGNORECASE)
    ok_len = []
    for r in pool:
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= min(24, prefix_tokens):
            ok_len.append((r, ids))
    filtered = [(r, ids) for r, ids in ok_len if not pat.search(r["text"])]
    use, was_filtered = (filtered, True) if len(filtered) >= 5 * n else (ok_len, False)
    if len(use) < n:
        raise RuntimeError(f"{concept}: only {len(use)} prefix candidates (< {n})")
    rng = random.Random(f"{seed}|prefix|{concept}")
    picks = rng.sample(range(len(use)), n)
    out = []
    for j, k in enumerate(picks):
        r, ids = use[k]
        ids = ids[:prefix_tokens]
        out.append({
            "prefix_id": r["prefix_id"],
            "prefix": tok.decode(ids, skip_special_tokens=True)
            if hasattr(tok, "decode") else r["text"],
            "ids": ids,
            "split": "selection" if j < n - n // 2 else "heldout",
            "prefix_filtered": was_filtered,
        })
    return out


# ----------------------------------------------------------- generation
def generate_batch(model, tok, ids_list, max_new: int, temperature: float,
                   seed: int, device: str) -> list[str]:
    """Left-padded sampled generation; returns decoded continuations."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    B, L = len(ids_list), max(len(x) for x in ids_list)
    input_ids = torch.full((B, L), pad, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        input_ids[i, L - len(ids):] = torch.tensor(ids, dtype=torch.long)
        attn[i, L - len(ids):] = 1
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids.to(device), attention_mask=attn.to(device),
            max_new_tokens=max_new, do_sample=True, temperature=temperature,
            top_p=1.0, top_k=0, repetition_penalty=1.0, pad_token_id=pad)
    return [tok.decode(out[i, L:].tolist(), skip_special_tokens=True)
            for i in range(B)]


# -------------------------------------------------------------- smoke bits
class SmokeTokenizer:
    """Deterministic word->id fake tokenizer (ids 3..127), pad=0 bos=2."""
    pad_token_id = 0
    bos_token_id = 2
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        ids = [3 + zlib.crc32(w.encode()) % 125 for w in text.split()]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{int(i)}" for i in ids if int(i) > 2)


def smoke_model(d: int = 64):
    from transformers import Gemma2Config, Gemma2ForCausalLM
    cfg = Gemma2Config(
        vocab_size=128, hidden_size=d, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=256, attn_implementation="eager")
    torch.manual_seed(0)
    model = Gemma2ForCausalLM(cfg)
    model.eval()
    return model


def smoke_context(args):
    """Provider dict mirroring the real one, entirely synthetic."""
    d = 64
    rng = np.random.default_rng(7)
    model = smoke_model(d)
    tok = SmokeTokenizer()
    pool = [{"prefix_id": f"smk{i:03d}",
             "text": " ".join(f"w{rng.integers(0, 40)}" for _ in range(60))}
            for i in range(64)]
    mu = rng.normal(size=d).astype(np.float32)
    sd = (0.5 + rng.random(d)).astype(np.float32)

    def arms(fam, cls, layer):
        w = rng.normal(size=d).astype(np.float32)
        w /= np.linalg.norm(w)
        r = rng.normal(size=d).astype(np.float32)
        return {"ridge": (w, 0.0), "dom": (w, 0.0),
                "rand": [r / np.linalg.norm(r)]}

    return {
        "model": model, "tok": tok, "device": "cpu",
        "families": {"smoke_fam": ["alpha", "beta"]},
        "card_layer": lambda fam, cls: 0,
        "arms": arms,
        "s95": lambda fam, cls, layer: 1.0,
        "natstats": lambda layer: (mu, sd),
        "pool": lambda fam: (pool, "smoke_pool"),
    }


def real_context(args):
    cards = load_cards()
    model, tok = common.load_model(args.device)
    model.config.output_hidden_states = False   # generation: no HS capture
    fams = {}
    for fam in args.families:
        classes = [c for c in common.FAMILIES[fam]
                   if (fam, c) in cards or args.layers]
        missing = [c for c in common.FAMILIES[fam] if (fam, c) not in cards]
        if missing and not args.layers:
            print(f"WARNING: {fam}: no probe card for {missing} — skipped "
                  "(pass --layers to force)")
        if classes:
            fams[fam] = classes
    return {
        "model": model, "tok": tok, "device": args.device,
        "families": fams,
        "card_layer": lambda fam, cls: cards[(fam, cls)]["layer"],
        "arms": common.load_arms,
        "s95": lambda fam, cls, layer:
            common.dose_calib(fam, cls, layer)["s95"],
        "natstats": lambda layer: common.load_natstats(layer),
        "pool": load_pool,
    }


# ------------------------------------------------------------------ main
def existing_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        for l in path.open():
            try:
                r = json.loads(l)
                keys.add((r["concept"], r["layer"], r["arm"],
                          round(float(r["factor"]), 6), r["prefix_id"]))
            except Exception:                                 # noqa: BLE001
                continue
    return keys


def concept_factors(concept: str, selection: dict, args) -> list[float]:
    if args.factors:
        return sorted(set(args.factors))
    selected = (selection.get(canon(concept), {}) or {}).get("factor")
    if selected is None:
        selected = 2.0
    lower = 1.0 if selected > 1.0 else round(selected / 2, 4)
    return sorted({round(float(selected), 6), round(float(lower), 6)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", default=None,
                    help="csv; default = all families with probe cards")
    ap.add_argument("--classes", default=None, help="csv filter on class names")
    ap.add_argument("--layers", default=None,
                    help="csv override; default = each concept's probe-card layer")
    ap.add_argument("--factors", default=None,
                    help="csv override; default = e2 selected (else 2.0) + one lower (1.0)")
    ap.add_argument("--arms", default=",".join(ARMS_DEFAULT))
    ap.add_argument("--out", default=str(common.OUT_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N concepts")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--prefix-tokens", type=int, default=40)
    ap.add_argument("--n-prefixes", type=int, default=10)
    args = ap.parse_args()
    args.factors = ([float(x) for x in args.factors.split(",")]
                    if args.factors else None)
    args.families = (args.families.split(",") if args.families
                     else sorted(common.FAMILIES))
    args.arms = [a for a in args.arms.split(",") if a]
    args.layers = ([int(x) for x in args.layers.split(",")]
                   if args.layers else None)
    if args.smoke:
        args.device = "cpu"
        args.max_new_tokens = min(args.max_new_tokens, 8)
        args.n_prefixes = min(args.n_prefixes, 4)
        args.prefix_tokens = min(args.prefix_tokens, 8)

    out_dir = Path(args.out)
    e3_dir = out_dir / "e3"
    e3_dir.mkdir(parents=True, exist_ok=True)
    hb_path = out_dir / "progress_e3_generate.log"
    selection = load_selection(out_dir)
    if selection:
        print(f"e2_cloze selection.json: {len(selection)} concepts")

    ctx = smoke_context(args) if args.smoke else real_context(args)
    model, tok, device = ctx["model"], ctx["tok"], ctx["device"]

    todo = [(fam, cls) for fam in sorted(ctx["families"])
            for cls in ctx["families"][fam]
            if not args.classes or canon(cls) in
            {canon(c) for c in args.classes.split(",")}]
    if args.limit:
        todo = todo[:args.limit]

    n_rows = 0
    for i, (fam, cls) in enumerate(tqdm(todo, desc="e3 concepts")):
        with hb_path.open("a") as hb:
            hb.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                     f"e3_generate {cls} {i + 1}/{len(todo)}\n")
        sel_layer = (selection.get(canon(cls), {}) or {}).get("layer")
        layers = args.layers or \
            ([sel_layer] if sel_layer is not None else None) or \
            [ctx["card_layer"](fam, cls)]
        try:
            pool, pool_src = ctx["pool"](fam)
            prefixes = sample_prefixes(pool, canon(cls), tok, args.n_prefixes,
                                       args.prefix_tokens, args.seed)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"WARNING: {fam}/{cls}: {e} — skipped")
            continue
        factors = concept_factors(cls, selection, args)
        out_path = e3_dir / f"generations_{fam}.jsonl"
        done = existing_keys(out_path)

        for layer in layers:
            try:
                arms_d = ctx["arms"](fam, cls, layer)
                s95 = ctx["s95"](fam, cls, layer)
                mu, sd = ctx["natstats"](layer)
            except FileNotFoundError as e:
                print(f"WARNING: {fam}/{cls} L{layer}: {e} — skipped "
                      "(glorptitude has no natscores => no dose calibration)")
                continue
            configs = [("baseline", 0.0)] + \
                [(a, f) for a in args.arms for f in factors]
            with out_path.open("a") as f:
                for arm, factor in configs:
                    pend = [p for p in prefixes
                            if (canon(cls), layer, arm, round(factor, 6),
                                p["prefix_id"]) not in done]
                    if not pend:
                        continue
                    alpha = 0.0 if arm == "baseline" else factor * s95
                    seed_cfg = stable_seed(args.seed, cls, layer, arm, factor)
                    if arm == "baseline":
                        cm = contextlib.nullcontext()
                    else:
                        w = (arms_d[arm][0] if arm != "rand"
                             else arms_d["rand"][0])
                        cm = Hooks(model, [Intervention(
                            layer=layer, vec_std=w, mode="steer",
                            alpha=alpha)], {layer: (mu, sd)})
                    b_cap = max(1, args.batch_tokens //
                                (args.prefix_tokens + args.max_new_tokens))
                    with cm:
                        conts: list[str] = []
                        for j in range(0, len(pend), b_cap):
                            chunk = pend[j:j + b_cap]
                            ids_list = [[tok.bos_token_id] + p["ids"]
                                        for p in chunk]
                            conts += generate_batch(
                                model, tok, ids_list, args.max_new_tokens,
                                args.temperature, seed_cfg + j, device)
                    for p, cont in zip(pend, conts):
                        f.write(json.dumps({
                            "concept": canon(cls), "family": fam,
                            "layer": layer, "arm": arm,
                            "factor": round(float(factor), 6),
                            "alpha": round(float(alpha), 6),
                            "prefix_id": p["prefix_id"], "split": p["split"],
                            "prefix": p["prefix"], "continuation": cont,
                            "seed": seed_cfg, "prefix_source": pool_src,
                            "prefix_filtered": p["prefix_filtered"],
                            "n_prefix_tokens": len(p["ids"]),
                        }, ensure_ascii=False) + "\n")
                        n_rows += 1
    print(f"e3_generate done: {n_rows} new rows -> {e3_dir}")


if __name__ == "__main__":
    main()
