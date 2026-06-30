"""
run_geometry.py — Step 4 GLUE: build real class-conditional activation clouds from
the probe-target model and run the validated geometry tiers in `geometry.py`.

`geometry.py` owns ALL the Tier 1-5 math but operates on clouds PASSED IN; it does not
touch a model. This module is the builder + driver:

  1. Load the probe target + tokenizer (probe.load_probe_target) and the labeled set
     (probe.load_labeled_examples).
  2. Per-example representation = the residual-stream vector at the **matched token**
     (where the class word / rated token appears). We tokenize each example.text with
     return_offsets_mapping=True, map raw["char_span"] -> the token(s) whose offset
     span overlaps it (same overlap rule as label.char_span_to_tokens), and take the
     mean of hidden_states[layer] over those tokens. If a span can't be resolved (no
     char_span, truncated out, slow tokenizer) we fall back to mean-pool over real
     (non-pad) tokens.
  3. Sweep layers ONE AT A TIME (range(1, n_layers+1, config.LAYER_STRIDE)): for each
     layer run a single batched forward, pull each example's matched-token vector,
     assemble presence_clouds + scalar_clouds (+ a moon-illumination bridge), call
     geometry.run_all(layer=L) -> per-layer tier{n}_L{L}.json + figures, free the reps.
  4. Pick a headline layer (best mean Tier-1 planarity, else ~65% depth) and re-run the
     tiers on it with layer=None to write the CANONICAL tier{1..5}.json + figures, then
     assemble geometry.md (leads with Tier-1).
  5. push_geometry() uploads artifacts/geometry/ + figures/ + geometry.md to the HF
     dataset repo. Not called during --smoke.

Model extraction lives behind extract_matched_token_reps(model, tokenizer, examples,
layer) -> {example_id: vec(d)}; --smoke injects synthetic reps via reps_fn (the same
cache_fn trick probe.py uses), so no model / GPU / .env is touched.

Run with the project venv:
    .venv/bin/python3 overnight_run/code/run_geometry.py --smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Path wiring: code/ (config, geometry, probe) and overnight_run/ (concepts)
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent          # overnight_run/code
_ROOT = _HERE.parent                             # overnight_run
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import config            # noqa: E402
import concepts          # noqa: E402
import geometry          # noqa: E402  (Tier 1-5 math — called, never reimplemented)
import probe             # noqa: E402  (load_probe_target / load_labeled_examples / LabeledExample)

LabeledExample = probe.LabeledExample
N_BOOT = geometry.N_BOOT


# =========================================================================== #
# 1. char_span -> token mapping (pure; mirrors label.char_span_to_tokens overlap)
# =========================================================================== #
def match_token_indices(offsets, char_span, valid=None) -> List[int]:
    """Indices of tokens whose (a,b) char offsets OVERLAP char_span=[s,e].

    `offsets` is an offset_mapping (list of (a,b)). `valid[i]` (optional) gates real
    tokens (drops padding). The `b > a` term drops special / pad tokens, which carry
    the (0,0) span. Overlap rule matches label.char_span_to_tokens: a < e and b > s.
    """
    if not char_span:
        return []
    s, e = int(char_span[0]), int(char_span[1])
    out: List[int] = []
    for i, ab in enumerate(offsets):
        a, b = int(ab[0]), int(ab[1])
        if valid is not None and not valid[i]:
            continue
        if b > a and a < e and b > s:
            out.append(i)
    return out


# =========================================================================== #
# 2. Model extraction — matched-token reps for one layer (one forward pass)
# =========================================================================== #
def extract_matched_token_reps(
    model,
    tokenizer,
    examples: List[LabeledExample],
    layer: int,
    batch_size: int = 8,
    max_length: int = 256,
) -> Dict[str, np.ndarray]:
    """One batched forward over `examples`; return {example_id: vec(d)} at `layer`.

    Per example: the mean of hidden_states[layer] over the tokens overlapping
    raw["char_span"] (the matched class word / rated token). Fallback = mean-pool over
    real tokens when the span can't be resolved. `layer` indexes hidden_states
    (0=embeddings, 1..N=blocks). Tiny output (one vector/example) -> caller frees it
    before the next layer; the (B,T,d) activations live only inside the batch loop.
    """
    import torch

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token  # gemma has no pad token by default

    device = next(model.parameters()).device
    reps: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            texts = [ex.text for ex in batch]
            offsets = None
            try:
                enc = tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length,
                                return_offsets_mapping=True)
                offsets = enc.pop("offset_mapping")          # (B, T, 2) tensor
            except Exception:  # slow tokenizer / no offsets -> mean-pool fallback
                enc = tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            H = out.hidden_states[layer]                     # (B, T, d)
            amask = enc["attention_mask"].bool()
            for j, ex in enumerate(batch):
                mask_j = amask[j]
                idxs: List[int] = []
                if offsets is not None:
                    span = (ex.raw or {}).get("char_span")
                    idxs = match_token_indices(offsets[j].tolist(), span,
                                               valid=mask_j.tolist())
                if idxs:
                    vec = H[j][idxs].float().mean(dim=0)
                else:
                    real = H[j][mask_j]
                    vec = real.float().mean(dim=0) if real.shape[0] else H[j].float().mean(dim=0)
                reps[ex.id] = vec.cpu().numpy()
            del out, H
    return reps


# =========================================================================== #
# 3. Example pool -> class-conditional clouds (the geometry.py input contract)
# =========================================================================== #
MIN_SCALAR = 2          # need >=2 points for a regression cloud


def assemble_clouds(
    examples: List[LabeledExample], reps: Dict[str, np.ndarray]
) -> Tuple[dict, dict, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Build the exact shapes geometry.py expects, from POSITIVES (presence) + targets:

      presence_clouds = {concept: {cls: ndarray (n, d)}}     (label == 1 only)
      scalar_clouds   = {scalar_name: (X (n, d), y (n,))}    (y = scalar_target())
      moon_bridge     = (X, y) with y = MOON_PHASES illum    (cyclic -> scalar bridge)
    """
    pres: Dict[str, Dict[str, list]] = {}
    scal: Dict[str, Tuple[list, list]] = {}
    moon_X: list = []
    moon_y: list = []
    moon_illum = {c: v.get("illum") for c, v in concepts.MOON_PHASES["classes"].items()}

    for ex in examples:
        v = reps.get(ex.id)
        if v is None:
            continue
        if ex.regime == "presence":
            if ex.cls is None or ex.label != 1:        # positives attach to matched token
                continue
            pres.setdefault(ex.concept, {}).setdefault(ex.cls, []).append(v)
            if ex.concept == "moon_phases" and moon_illum.get(ex.cls) is not None:
                moon_X.append(v)
                moon_y.append(moon_illum[ex.cls])
        else:
            t = ex.scalar_target()
            if t is None:
                continue
            xy = scal.setdefault(ex.concept, ([], []))
            xy[0].append(v)
            xy[1].append(float(t))

    presence_clouds = {c: {k: np.stack(a) for k, a in d.items()} for c, d in pres.items()}
    scalar_clouds = {k: (np.stack(X), np.asarray(y, float))
                     for k, (X, y) in scal.items() if len(y) >= MIN_SCALAR}
    moon_bridge = ((np.stack(moon_X), np.asarray(moon_y, float))
                   if len(moon_y) >= MIN_SCALAR else None)
    return presence_clouds, scalar_clouds, moon_bridge


# =========================================================================== #
# 4. geometry.md (leads with Tier 1) — reuses geometry.py's verdicts
# =========================================================================== #
_TITLES = {
    "tier1": "Tier 1 — Z/12 collision study (headline)",
    "tier2": "Tier 2 — Harmonic nesting",
    "tier3": "Tier 3 — Abstract magnitude axis",
    "tier4": "Tier 4 — Recovered world map",
    "tier5": "Tier 5 — Antipodal / opponent structure",
}


def write_geometry_md(results: dict, headline_layer=None, n_layers=None, path=None) -> str:
    path = Path(path) if path else (geometry.GEODIR / "geometry.md")
    L = ["# Representation geometry (Step 4) — verdicts\n"]
    L.append(
        f"Probe target: `{config.PROBE_TARGET}`. Headline layer: **L{headline_layer}**"
        + (f" of {n_layers}" if n_layers else "")
        + ". Per-example feature = residual stream at the matched token. "
        "Every angle/cosine/spacing carries a bootstrap 95% CI. Full depth sweep in "
        "`artifacts/geometry/tier*_L*.json` and `figures/*_L*.png`.\n"
    )
    for t in ("tier1", "tier2", "tier3", "tier4", "tier5"):
        if t not in results:
            continue
        r = results[t]
        L.append(f"## {_TITLES[t]}")
        L.append(r["verdict"])
        if r.get("figure"):
            L.append(f"\n_figure_: `{r['figure']}`")
        if t == "tier1":
            L.append(
                "\n_Note_: `moon_phases` is **Z/8** (there is no canonical 12-phase set); "
                "its positive cloud may be thin, which widens its CIs."
            )
        L.append("")
    path.write_text("\n".join(L))
    return str(path)


# =========================================================================== #
# 5. Driver — layer sweep (per-layer) + headline (canonical)
# =========================================================================== #
def _tier1_mean_planarity(res: dict) -> Optional[float]:
    if "tier1" not in res:
        return None
    m = res["tier1"]["metrics"]
    pl = [m[k]["point"] for k in m if k.endswith("/planarity")]
    return float(np.mean(pl)) if pl else 0.0


def run_geometry(
    examples: List[LabeledExample],
    layers: List[int],
    reps_fn: Optional[Callable[[int], Dict[str, np.ndarray]]] = None,
    model=None,
    tokenizer=None,
    n_layers: Optional[int] = None,
    n_boot: int = N_BOOT,
    save_fig: bool = True,
    push: bool = False,
    verbose: bool = True,
) -> Tuple[dict, Optional[int], dict]:
    """Sweep `layers` (one layer's reps at a time), run geometry.run_all per layer, then
    re-run the headline layer with layer=None for canonical artifacts + geometry.md.

    `reps_fn(layer) -> {example_id: vec(d)}` lets --smoke inject synthetic reps; if None,
    reps come from extract_matched_token_reps on the real model.
    """
    if reps_fn is None:
        if model is None or tokenizer is None:
            raise ValueError("Need model+tokenizer (or reps_fn) to extract activations.")
        reps_fn = lambda ly: extract_matched_token_reps(model, tokenizer, examples, ly)  # noqa: E731

    nl = n_layers or (max(layers) if layers else 1)
    depth_target = max(1, round(0.65 * nl))

    best = {"score": -1e18, "layer": None, "reps": None}
    per_layer: Dict[int, dict] = {}

    for ly in layers:
        reps = reps_fn(ly)
        pres, scal, moon = assemble_clouds(examples, reps)
        if not pres and not scal:
            if verbose:
                print(f"[layer {ly}] no clouds assembled (no resolved reps); skip")
            del reps
            continue
        res = geometry.run_all(presence_clouds=pres, scalar_clouds=scal,
                               moon_bridge=moon, layer=ly, n_boot=n_boot, save_fig=save_fig)
        per_layer[ly] = res

        plan = _tier1_mean_planarity(res)
        score = plan if plan is not None else -abs(ly - depth_target)
        if score > best["score"]:
            best = {"score": score, "layer": ly, "reps": reps}
        if reps is not best["reps"]:
            del reps                              # free everything but the headline
        if verbose:
            tiers = ",".join(sorted(res))
            extra = f"tier1_planarity={plan:.3f}" if plan is not None else f"depth_dist={abs(ly-depth_target)}"
            print(f"[layer {ly}] tiers={{{tiers}}} {extra}")

    if best["reps"] is None:
        if verbose:
            print("[run_geometry] no usable layer; nothing written")
        return {}, None, per_layer

    # ---- canonical artifacts from the headline layer (layer=None -> no suffix) ----
    pres, scal, moon = assemble_clouds(examples, best["reps"])
    canon = geometry.run_all(presence_clouds=pres, scalar_clouds=scal, moon_bridge=moon,
                             layer=None, n_boot=n_boot, save_fig=save_fig)
    md = write_geometry_md(canon, headline_layer=best["layer"], n_layers=n_layers)
    if verbose:
        print(f"[run_geometry] headline layer L{best['layer']} (score={best['score']:.3f}); "
              f"canonical tiers={sorted(canon)}; geometry.md -> {md}")
    if push:
        push_geometry()
    return canon, best["layer"], per_layer


# =========================================================================== #
# 6. HF push
# =========================================================================== #
def push_geometry(repo_id: Optional[str] = None) -> str:
    """Upload artifacts/geometry/ (incl. geometry.md) + figures/ to the HF dataset repo.
    Uses a cached login token. Not called during --smoke."""
    from huggingface_hub import HfApi, upload_folder

    repo_id = repo_id or config.HF_DATASET_REPO
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=config.HF_PRIVATE, exist_ok=True)
    upload_folder(folder_path=str(geometry.GEODIR), repo_id=repo_id, repo_type="dataset",
                  path_in_repo="geometry", commit_message="geometry artifacts + geometry.md")
    if geometry.FIGDIR.exists():
        upload_folder(folder_path=str(geometry.FIGDIR), repo_id=repo_id, repo_type="dataset",
                      path_in_repo="figures", commit_message="geometry figures")
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"[push] uploaded geometry -> {url}")
    return url


# =========================================================================== #
# 7. Real entrypoint
# =========================================================================== #
def main_real(args) -> int:
    import torch
    torch.set_num_threads(8)      # host has 252 cores; tiny CPU tensor ops thrash otherwise
    examples = probe.load_labeled_examples()
    if not examples:
        print("No labeled examples in data/labels/. Run label.py first.")
        return 1
    model, tok = probe.load_probe_target()
    n_layers = model.config.num_hidden_layers
    layers = list(range(1, n_layers + 1, config.LAYER_STRIDE))
    print(f"[run_geometry] {len(examples)} examples; layers={layers} (n_layers={n_layers})")
    run_geometry(examples, layers, model=model, tokenizer=tok,
                 n_layers=n_layers, push=args.push)
    return 0


# =========================================================================== #
# 8. Smoke — synthetic examples + injected reps, NO model. Plant known geometry.
# =========================================================================== #
def _smoke_data(seed: int = 0, d: int = 48, npc: int = 12, nsc: int = 80, noise: float = 0.03):
    """Labeled examples + {id: rep} with PLANTED structure: months/colors/moon (and the
    two Z/4 cycles) coplanar in one (u,v) plane (color phase +40, others 0); the
    magnitude scalars share axis g (numbers linear, cost/size/duration log); indoors=-h,
    outdoors=+(-h), lovingness=p _|_ harmfulness=q. Mirrors geometry._synthesize's plant
    so the GLUE (example -> cloud) is what's under test."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, 10)))
    u, v, g, h, p, q = (Q[:, k] for k in range(6))
    examples: List[LabeledExample] = []
    reps: Dict[str, np.ndarray] = {}

    def add_cycle(concept: str, phase_deg: float):
        names = list(concepts.PRESENCE_CONCEPTS[concept]["classes"])
        n = len(names)
        for ci, cls in enumerate(names):
            th = 2 * np.pi * ci / n + np.radians(phase_deg)
            base = np.cos(th) * u + np.sin(th) * v
            for k in range(npc):
                eid = f"{concept}::{cls}::{k}"
                examples.append(LabeledExample(
                    id=eid, concept=concept, cls=cls, regime="presence",
                    text=cls, label=1, raw={"char_span": [0, len(cls)]}))
                reps[eid] = base + noise * rng.standard_normal(d)

    add_cycle("months", 0.0)
    add_cycle("color_wheel", 40.0)
    add_cycle("moon_phases", 0.0)
    add_cycle("seasons", 0.0)
    add_cycle("directions", 0.0)

    def add_scalar(name: str, yvals, fy):
        for k, y in enumerate(yvals):
            eid = f"scalar::{name}::{k}"
            examples.append(LabeledExample(
                id=eid, concept=name, cls=None, regime="scalar",
                text="x", value=float(y), raw={}))
            reps[eid] = fy(float(y)) + noise * rng.standard_normal(d)

    add_scalar("numbers", rng.uniform(1, 10, nsc), lambda y: y * g)              # linear
    add_scalar("costliness", rng.uniform(1, 1000, nsc), lambda y: np.log(y) * g)  # log
    add_scalar("physical_size", rng.uniform(1, 100, nsc), lambda y: np.log(y) * g)
    add_scalar("duration", rng.uniform(1, 1e6, nsc), lambda y: np.log(y) * g)
    A = 5.0
    add_scalar("indoors", rng.uniform(0, 1, nsc), lambda y: A * y * h)
    add_scalar("outdoors", rng.uniform(0, 1, nsc), lambda y: A * y * (-h))
    add_scalar("lovingness", rng.uniform(-1, 1, nsc), lambda y: A * y * p)
    add_scalar("harmfulness", rng.uniform(-1, 1, nsc), lambda y: A * y * q)
    return examples, reps


def main_smoke() -> int:
    import shutil
    import tempfile

    print("=== run_geometry.py --smoke (synthetic, no model / GPU / .env) ===")
    fails: List[str] = []

    def check(name, cond, got):
        if not cond:
            fails.append(name)
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {got}")

    # ---- (a) char_span -> token mapping (the extraction crux) ----
    # text "the january sky" with a leading special token: offsets [(0,0)=BOS, the, january, sky]
    offs = [(0, 0), (0, 3), (4, 11), (12, 15)]
    valid = [True, True, True, True]
    check("map: single matched token (january)",
          match_token_indices(offs, [4, 11], valid) == [2], match_token_indices(offs, [4, 11], valid))
    check("map: partial overlap still hits token",
          match_token_indices(offs, [5, 8], valid) == [2], match_token_indices(offs, [5, 8], valid))
    check("map: special/(0,0) token never matched",
          match_token_indices(offs, [0, 0], valid) == [], match_token_indices(offs, [0, 0], valid))
    check("map: out-of-range span -> []",
          match_token_indices(offs, [40, 50], valid) == [], match_token_indices(offs, [40, 50], valid))
    check("map: padding gated out by valid mask",
          match_token_indices(offs, [4, 11], [True, True, False, True]) == [],
          match_token_indices(offs, [4, 11], [True, True, False, True]))

    # ---- (b) isolate artifacts to a temp dir (never clobber real outputs) ----
    tmp = Path(tempfile.mkdtemp(prefix="geom_smoke_"))
    geometry.GEODIR = tmp / "geometry"
    geometry.FIGDIR = tmp / "figures"
    geometry.GEODIR.mkdir(parents=True, exist_ok=True)
    geometry.FIGDIR.mkdir(parents=True, exist_ok=True)

    try:
        examples, reps = _smoke_data()
        n_pres = sum(1 for e in examples if e.regime == "presence")
        n_scal = sum(1 for e in examples if e.regime == "scalar")
        print(f"  synthetic: {len(examples)} examples ({n_pres} presence pos, {n_scal} scalar)")

        # sanity: cloud assembly recovers the planted classes
        pres, scal, moon = assemble_clouds(examples, reps)
        check("assemble: months has 12 class clouds", len(pres.get("months", {})) == 12,
              len(pres.get("months", {})))
        check("assemble: scalar clouds are (X,y) with matching n",
              all(X.shape[0] == y.shape[0] for X, y in scal.values()) and "numbers" in scal,
              {k: scal[k][0].shape for k in scal})
        check("assemble: moon illum bridge built", moon is not None and moon[0].shape[0] > 0,
              None if moon is None else moon[0].shape)

        # full driver over 2 fake layers; reps injected (no model)
        layers = [4, 8]
        canon, hl, per = run_geometry(examples, layers, reps_fn=lambda ly: reps,
                                      n_layers=12, n_boot=150)

        # ---- (c) Tier-1 planted structure recovered (the headline assertion) ----
        m = canon["tier1"]["metrics"]
        for c in ("months", "color_wheel", "moon_phases"):
            check(f"T1 {c} planar", m[f"{c}/planarity"]["point"] > 0.95,
                  f"{m[f'{c}/planarity']['point']:.3f}")
        for a, b in (("months", "color_wheel"), ("months", "moon_phases")):
            t2 = m[f"{a}|{b}/theta2"]["point"]
            check(f"T1 principal angle {a}|{b} ~ 0", t2 < 6.0, f"theta2={t2:.2f}deg")
        pc = m["months|color_wheel/phase_deg"]["point"]
        check("T1 month->color phase ~ 40", min(abs(pc - 40), abs(pc + 40)) < 8.0, f"{pc:.1f}deg")
        z4 = m["seasons|directions/theta2"]["point"]
        check("T1 Z/4 seasons|directions ~ 0", z4 < 6.0, f"theta2={z4:.2f}deg")

        # ---- (d) every applicable tier ran (real geometry.py functions) ----
        check("driver ran tiers 1/2/3/5", all(t in canon for t in ("tier1", "tier2", "tier3", "tier5")),
              sorted(canon))
        check("headline layer chosen", hl in layers, hl)

        # ---- (e) artifacts: per-layer + canonical JSON + figures + geometry.md ----
        for ly in layers:
            check(f"per-layer tier1_L{ly}.json", (geometry.GEODIR / f"tier1_L{ly}.json").exists(),
                  str(geometry.GEODIR / f"tier1_L{ly}.json"))
        check("canonical tier1.json", (geometry.GEODIR / "tier1.json").exists(),
              str(geometry.GEODIR / "tier1.json"))
        figs = list(geometry.FIGDIR.glob("tier1_cycles*.png"))
        check("tier1 figures written (per-layer + canonical)", len(figs) >= 2, [f.name for f in figs])
        check("geometry.md written", (geometry.GEODIR / "geometry.md").exists(),
              str(geometry.GEODIR / "geometry.md"))

        print("\n--- headline verdict (Tier 1) ---")
        print(canon["tier1"]["verdict"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\ncleaned up smoke artifacts under {tmp}")

    if fails:
        print(f"\n=== SMOKE FAIL ({len(fails)}): {fails} ===")
        return 1
    print("\n=== SMOKE PASS: planted Tier-1 geometry recovered; per-layer + canonical "
          "artifacts written; real geometry.py tiers called ===")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Step-4 geometry driver (cloud builder + tier runner)")
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic CPU smoke (no real model) and exit")
    ap.add_argument("--push", action="store_true",
                    help="push artifacts/geometry/ + figures/ to HF after running")
    args = ap.parse_args()
    if args.smoke:
        return main_smoke()
    return main_real(args)


if __name__ == "__main__":
    sys.exit(main())
