"""
attribute.py — Step 3 of the overnight concept-probes run: DATA ATTRIBUTION.

Pull the trained attention probes (Step 2), run a sample of the DISJOINT shards
through the probe-target LLM, and for EVERY token position record how every probe
"fires" on it, alongside the raw residual-stream norm so a genuine feature hit can
be told apart from a merely high-norm token.

Design contract (see SPEC.md / overnight_brief.md, "Step 3 — Attribute data"):
  * Load ALL trained probes from artifacts/probes/ (weights + metrics): each probe's
    best_layer, primary metric, and `reliable` flag (metric > RELIABLE_METRIC_THRESH).
    Per concept identify the highest-AUROC/R^2 "high performer", but KEEP ALL probes.
  * Evaluate each probe at ITS OWN best_layer. Read activations ONE LAYER AT A TIME
    (the storage-safe pattern probe.py uses) over the set of distinct best_layers.
  * For each (snippet, token) at the probe's layer, record for every probe:
        s_t = query · h_t   (pre-softmax ALIGNMENT score; the raw localization signal)
        a_t = softmax_t(s)  (normalized ATTENTION weight; localization distribution)
        pooled_score        (snippet/sequence-level pooled score for that probe)
    For attention probes the per-token "attribution" is the attention weight / alignment
    score (LOCALIZATION), NOT a calibrated per-token class probability (per the brief).
  * Also record, per (snippet, token):
        ||h_t||             (L2 norm of the RAW residual hidden state at token t, layer L)
        mean_a / mean_s across ALL probes ("across concepts")
        mean_a / mean_s across RELIABLE probes only (+ how many are reliable)
    Norms are layer-specific, so ||h_t|| is captured at every layer actually read.
  * Output (artifacts/attribution/):
        attribution.parquet     long: one row per (snippet, token, probe)
        token_aggregates.parquet one row per (snippet, token): cross-probe means + norms
        summary.json            per-probe + global stats
    Falls back to .jsonl if parquet write fails (documented in summary).
  * push_attribution() uploads artifacts/attribution/ to config.HF_DATASET_REPO
    (repo_type=dataset, path_in_repo="attribution"). NOT called during --smoke.
  * --smoke plants a few fake probes + synthetic hidden states (no real model), runs
    the FULL attribution path, and asserts every row carries BOTH a per-token probe
    activation AND a finite, positive ||h_t|| norm. Cleans up its temp artifacts.

Gemma tokenization differs from nanochat's BPE; per the brief we ignore that here and
just note it in summary.json (token strings are whatever config.PROBE_TARGET emits).

Run with the project venv:
    .venv/bin/python3 overnight_run/code/attribute.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Path setup: make both `config` (code/) and `concepts` (overnight_run/) importable
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent          # overnight_run/code
_ROOT = _HERE.parent                             # overnight_run
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import config  # noqa: E402
import concepts  # noqa: E402
# Reuse the Step-2 probe arch + loaders — do NOT reimplement the model load or probe.
import probe  # noqa: E402
from probe import AttentionProbe, LabeledExample  # noqa: E402

PROBES_DIR = config.ARTIFACTS / "probes"
ATTR_DIR = config.ARTIFACTS / "attribution"

# How many corpus snippets to attribute (env knob; default ~300 per the brief).
ATTR_MAX_SNIPPETS = int(os.environ.get("ATTR_MAX_SNIPPETS", "300"))
# Max tokens per snippet fed to the model (keeps the per-token table tractable).
ATTR_MAX_LENGTH = int(os.environ.get("ATTR_MAX_LENGTH", str(config.SNIPPET_TOKENS)))
# Rounding for compact float storage.
ND = int(os.environ.get("ATTR_ROUND", "6"))


def _r(x) -> float:
    """Round to ND digits as a plain float (compact + json/parquet friendly)."""
    return round(float(x), ND)


# =========================================================================== #
# 1. Load trained probes (weights + metrics) at their best_layer
# =========================================================================== #
@dataclass
class ProbeHandle:
    probe_id: str
    regime: str                 # "presence" | "scalar"
    concept: str                # grouping key for high-performer selection
    best_layer: int
    metric: Optional[float]     # AUROC (presence) / R^2 (scalar) at best_layer
    reliable: bool              # metric > config.RELIABLE_METRIC_THRESH
    high_performer: bool        # top metric within its concept group
    module: AttentionProbe
    mean: torch.Tensor          # (d,) feature standardization (from training)
    std: torch.Tensor           # (d,)


def _concept_group(probe_id: str, regime: str) -> str:
    """Grouping for 'high performer per concept'. Presence: the concept name
    (e.g. 'months::January' -> 'months', so the 12 class-probes compete). Scalar:
    each scalar is its own concept (e.g. 'scalar::numbers')."""
    if regime == "presence":
        return probe_id.split("::", 1)[0]
    return probe_id  # scalar::name — already its own group


def _read_index(probes_dir: Path) -> Dict[str, str]:
    idx_path = probes_dir / "_index.json"
    if idx_path.exists():
        try:
            return json.loads(idx_path.read_text())
        except json.JSONDecodeError:
            pass
    # Fallback: scan subdirs that carry a metrics.json, key by its probe_id.
    out: Dict[str, str] = {}
    for sub in sorted(probes_dir.glob("*/")):
        mp = sub / "metrics.json"
        if mp.exists():
            try:
                pid = json.loads(mp.read_text()).get("probe_id")
            except json.JSONDecodeError:
                pid = None
            if pid:
                out[pid] = sub.name
    return out


def load_probes(probes_dir: Path = PROBES_DIR) -> List[ProbeHandle]:
    """Load every trained probe at its best_layer. Skips probes with no best_layer
    (all layers skipped) or a missing best-layer weight file. Marks the per-concept
    high performer (max AUROC/R^2) but keeps ALL probes (per the brief)."""
    index = _read_index(probes_dir)
    handles: List[ProbeHandle] = []
    for probe_id, dirname in index.items():
        pdir = probes_dir / dirname
        mp = pdir / "metrics.json"
        if not mp.exists():
            continue
        try:
            doc = json.loads(mp.read_text())
        except json.JSONDecodeError:
            continue
        regime = doc.get("regime", "presence")
        best_layer = doc.get("best_layer")
        if best_layer is None:
            continue
        wp = pdir / f"L{best_layer}.pt"
        if not wp.exists():
            continue

        # primary metric at best layer: AUROC (presence) / R^2 (scalar) per brief.
        best_entry = next((e for e in doc.get("per_layer", [])
                           if e.get("layer") == best_layer), {})
        metric = (best_entry.get("auroc") if regime == "presence"
                  else best_entry.get("r2"))
        reliable = bool(doc.get("reliable", False))

        ckpt = torch.load(wp, map_location="cpu", weights_only=False)
        meta = ckpt["meta"]
        d_model = int(meta["d_model"])
        mod = AttentionProbe(d_model, regime=regime)
        mod.load_state_dict(ckpt["state_dict"])
        mod.eval()
        mean = torch.tensor(meta.get("feat_mean", [0.0] * d_model), dtype=torch.float32)
        std = torch.tensor(meta.get("feat_std", [1.0] * d_model), dtype=torch.float32)
        std = std.clamp_min(1e-6)

        handles.append(ProbeHandle(
            probe_id=probe_id, regime=regime,
            concept=_concept_group(probe_id, regime),
            best_layer=int(best_layer), metric=metric, reliable=reliable,
            high_performer=False, module=mod, mean=mean, std=std,
        ))

    # Mark the per-concept high performer (highest metric in each concept group).
    by_group: Dict[str, List[ProbeHandle]] = {}
    for h in handles:
        by_group.setdefault(h.concept, []).append(h)
    for group in by_group.values():
        scored = [h for h in group if h.metric is not None]
        if scored:
            max(scored, key=lambda h: h.metric).high_performer = True
    return handles


# =========================================================================== #
# 2. Per-snippet, per-probe attribution math (replicates AttentionProbe.forward
#    for a single full-mask sequence, while ALSO exposing the pre-softmax score).
# =========================================================================== #
@torch.no_grad()
def _attribute_sequence(h: ProbeHandle, H: torch.Tensor
                        ) -> Tuple[np.ndarray, np.ndarray, float]:
    """H: (T, d) RAW hidden states for one snippet at probe `h`'s best_layer.
    Returns (s_t, a_t, pooled_score):
        s_t  (T,)  pre-softmax alignment score query·h_t (on standardized features)
        a_t  (T,)  attention weight (masked-softmax over tokens) — localization
        pooled_score  float  sequence-level pooled readout (sigmoid prob for
                      presence, raw scalar for scalar)."""
    Hn = (H - h.mean) / h.std                       # standardize as in training
    s = Hn @ h.module.query                          # (T,)
    a = torch.softmax(s, dim=0)                       # (T,) full mask -> plain softmax
    pooled = (a.unsqueeze(-1) * Hn).sum(dim=0)        # (d,)
    out = h.module.head(pooled).squeeze(-1)           # scalar logit/value
    pooled_score = (torch.sigmoid(out).item() if h.regime == "presence"
                    else out.item())
    return s.numpy(), a.numpy(), float(pooled_score)


# =========================================================================== #
# 3. Table writers (parquet, jsonl fallback)
# =========================================================================== #
def _write_table(rows: List[dict], stem: Path) -> Tuple[Path, str]:
    """Write rows as parquet; fall back to jsonl. Returns (path, format)."""
    if not rows:
        # still emit an empty jsonl so downstream globbing is predictable
        p = stem.with_suffix(".jsonl")
        p.write_text("")
        return p, "jsonl(empty)"
    try:
        import pandas as pd  # noqa: WPS433
        p = stem.with_suffix(".parquet")
        pd.DataFrame(rows).to_parquet(p, index=False)
        return p, "parquet"
    except Exception as e:  # noqa: BLE001  (pyarrow/pandas missing or quirk)
        p = stem.with_suffix(".jsonl")
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return p, f"jsonl(parquet_failed:{type(e).__name__})"


# =========================================================================== #
# 4. Core attribution driver (layer-outer, probe-inner = storage-safe)
# =========================================================================== #
def run_attribution(
    snippets: List[LabeledExample],
    probes: List[ProbeHandle],
    *,
    token_map: Dict[str, List[str]],
    model=None,
    tokenizer=None,
    cache_fn: Optional[Callable[[int], Dict[str, torch.Tensor]]] = None,
    out_dir: Path = ATTR_DIR,
    verbose: bool = True,
) -> dict:
    """Run attribution for every probe at its best_layer, reading activations ONE
    LAYER AT A TIME. `cache_fn(layer) -> {snippet_id: (T,d)}` injects synthetic
    activations for --smoke; otherwise activations come from the real model via
    probe.extract_layer_activations. token_map gives the token strings per snippet
    (aligned to the activation rows). Writes attribution + aggregate tables + summary.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not probes:
        raise ValueError("no probes loaded — run probe.py (Step 2) first")

    layers = sorted({h.best_layer for h in probes})
    probes_by_layer: Dict[int, List[ProbeHandle]] = {}
    for h in probes:
        probes_by_layer.setdefault(h.best_layer, []).append(h)

    long_rows: List[dict] = []
    # per-(snippet, token) accumulators across layers/probes
    agg: Dict[Tuple[str, int], dict] = {}
    norm_sum: Dict[int, float] = {ly: 0.0 for ly in layers}
    norm_cnt: Dict[int, int] = {ly: 0 for ly in layers}

    def _agg(sid: str, tidx: int, tok: str) -> dict:
        key = (sid, tidx)
        e = agg.get(key)
        if e is None:
            e = {"snippet_id": sid, "token_idx": tidx, "token": tok,
                 "a_all": [], "s_all": [], "a_rel": [], "s_rel": [],
                 "norms": {}}
            agg[key] = e
        return e

    for layer in layers:
        if cache_fn is not None:
            cache = cache_fn(layer)
        else:
            if model is None or tokenizer is None:
                raise ValueError("need model+tokenizer (or cache_fn) for activations")
            _dev = next(model.parameters()).device.type   # move inputs onto model device
            cache = probe.extract_layer_activations(model, tokenizer, snippets, layer, device=_dev)

        # (a) raw residual norm ||h_t|| per token at THIS layer (probe-independent)
        for sid, H in cache.items():
            toks = token_map.get(sid, [])
            norms = H.norm(dim=-1)                       # (T,) raw L2 norm
            T = H.shape[0]
            for tidx in range(T):
                tok = toks[tidx] if tidx < len(toks) else ""
                e = _agg(sid, tidx, tok)
                nv = float(norms[tidx])
                e["norms"][str(layer)] = _r(nv)
                norm_sum[layer] += nv
                norm_cnt[layer] += 1

        # (b) per-probe attribution at this layer
        for h in probes_by_layer[layer]:
            for sid, H in cache.items():
                toks = token_map.get(sid, [])
                s, a, pooled = _attribute_sequence(h, H)
                for tidx in range(H.shape[0]):
                    tok = toks[tidx] if tidx < len(toks) else ""
                    nv = float(H[tidx].norm())
                    long_rows.append({
                        "snippet_id": sid, "token_idx": tidx, "token": tok,
                        "layer": layer, "h_norm": _r(nv),
                        "probe_id": h.probe_id, "concept": h.concept,
                        "regime": h.regime,
                        "s_t": _r(s[tidx]), "a_t": _r(a[tidx]),
                        "pooled_score": _r(pooled),
                        "is_high_performer": bool(h.high_performer),
                        "is_reliable": bool(h.reliable),
                    })
                    e = _agg(sid, tidx, tok)
                    e["a_all"].append(float(a[tidx]))
                    e["s_all"].append(float(s[tidx]))
                    if h.reliable:
                        e["a_rel"].append(float(a[tidx]))
                        e["s_rel"].append(float(s[tidx]))

        del cache  # FREE the layer cache before advancing (storage/memory safety)
        if verbose:
            print(f"[layer {layer}] probes={len(probes_by_layer[layer])} "
                  f"snippets={len(token_map)} rows={len(long_rows)}")

    n_reliable = sum(1 for h in probes if h.reliable)

    # ---- per-(snippet, token) aggregate rows ----
    agg_rows: List[dict] = []
    for e in agg.values():
        n_all = len(e["a_all"])
        n_rel = len(e["a_rel"])
        agg_rows.append({
            "snippet_id": e["snippet_id"], "token_idx": e["token_idx"],
            "token": e["token"],
            "mean_a_concepts": _r(np.mean(e["a_all"])) if n_all else None,
            "mean_s_concepts": _r(np.mean(e["s_all"])) if n_all else None,
            "mean_a_reliable": _r(np.mean(e["a_rel"])) if n_rel else None,
            "mean_s_reliable": _r(np.mean(e["s_rel"])) if n_rel else None,
            "n_concept_probes": n_all,
            "n_reliable_probes": n_rel,
            # ||h_t|| at every layer actually read for this token (layer -> norm)
            "norms_by_layer": json.dumps(e["norms"]),
        })

    attr_path, attr_fmt = _write_table(long_rows, out_dir / "attribution")
    agg_path, agg_fmt = _write_table(agg_rows, out_dir / "token_aggregates")

    # ---- summary.json ----
    summary = {
        "probe_target": config.PROBE_TARGET,
        "reliable_metric_thresh": config.RELIABLE_METRIC_THRESH,
        "n_snippets": len(token_map),
        "n_tokens": len(agg),
        "n_token_probe_rows": len(long_rows),
        "n_probes": len(probes),
        "n_reliable_probes": int(n_reliable),
        "n_high_performers": sum(1 for h in probes if h.high_performer),
        "layers_read": layers,
        "mean_norm_per_layer": {
            str(ly): _r(norm_sum[ly] / norm_cnt[ly]) if norm_cnt[ly] else None
            for ly in layers
        },
        "attribution_file": attr_path.name, "attribution_format": attr_fmt,
        "aggregates_file": agg_path.name, "aggregates_format": agg_fmt,
        "attention_attribution_note": (
            "Per-token a_t/s_t are LOCALIZATION (attention weight / pre-softmax "
            "alignment score), NOT calibrated per-token class probabilities."
        ),
        "tokenization_note": (
            f"Token strings are {config.PROBE_TARGET}'s tokenizer output, which "
            f"differs from nanochat's BPE; not reconciled here (per brief)."
        ),
        "probes": [
            {"probe_id": h.probe_id, "concept": h.concept, "regime": h.regime,
             "best_layer": h.best_layer, "metric": h.metric,
             "reliable": h.reliable, "high_performer": h.high_performer}
            for h in sorted(probes, key=lambda x: x.probe_id)
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"[done] wrote {attr_path.name} ({attr_fmt}), {agg_path.name} "
              f"({agg_fmt}), summary.json -> {out_dir}")
    return summary


# =========================================================================== #
# 5. Real snippet loading (disjoint shards) — mirrors build_candidates loader
# =========================================================================== #
def load_attribution_snippets(
    max_snippets: int = ATTR_MAX_SNIPPETS,
    shards: Optional[List[int]] = None,
) -> List[LabeledExample]:
    """Sample raw documents from the DISJOINT shards (config.SHARDS) as attribution
    snippets. Reuses build_candidates.iter_shard_docs (cached HF download). Each doc
    head becomes one snippet; the tokenizer later truncates to ATTR_MAX_LENGTH. Not
    exercised by --smoke (requires network + the gated/large model later)."""
    import build_candidates as bc  # lazy: pulls in pyarrow + hf_hub_download

    shards = shards or list(config.SHARDS)
    max_chars = getattr(bc, "MAX_SNIPPET_CHARS", 1400)
    out: List[LabeledExample] = []
    for shard in shards:
        for doc in bc.iter_shard_docs(shard, max_docs=config.MAX_DOCS_PER_SHARD):
            if len(out) >= max_snippets:
                return out
            text = doc[:max_chars].strip()
            if not text:
                continue
            out.append(LabeledExample(
                id=f"attr::s{shard}::{len(out):06d}", concept="_attr",
                cls=None, regime="presence", text=text))
    return out


def _token_strings(tokenizer, snippets: List[LabeledExample],
                   max_length: int = ATTR_MAX_LENGTH) -> Dict[str, List[str]]:
    """Per-snippet token strings aligned to extract_layer_activations' rows. Both
    tokenize the same text with the same truncation/special tokens and keep the real
    (non-pad) tokens in order, so converting these ids gives the aligned strings."""
    tmap: Dict[str, List[str]] = {}
    for ex in snippets:
        ids = tokenizer(ex.text, truncation=True, max_length=max_length)["input_ids"]
        tmap[ex.id] = tokenizer.convert_ids_to_tokens(ids)
    return tmap


# =========================================================================== #
# 6. HF push
# =========================================================================== #
def push_attribution(repo_id: Optional[str] = None, out_dir: Path = ATTR_DIR) -> str:
    """Upload artifacts/attribution/ to config.HF_DATASET_REPO (dataset repo,
    path_in_repo='attribution') with a cached-login token. NOT called in --smoke."""
    from huggingface_hub import HfApi, upload_folder

    repo_id = repo_id or config.HF_DATASET_REPO
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset",
                    private=config.HF_PRIVATE, exist_ok=True)
    upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset",
                  path_in_repo="attribution",
                  commit_message="Step-3 data attribution (per-token probe activations + norms)")
    url = f"https://huggingface.co/datasets/{repo_id}/tree/main/attribution"
    print(f"[push] uploaded attribution -> {url}")
    return url


# =========================================================================== #
# 7. Real entrypoint
# =========================================================================== #
def main_real(args) -> int:
    torch.set_num_threads(8)      # host has 252 cores; tiny CPU tensor ops thrash otherwise
    probes = load_probes()
    if not probes:
        print("No trained probes in artifacts/probes/. Run probe.py (Step 2) first.")
        return 1
    print(f"loaded {len(probes)} probes across layers "
          f"{sorted({h.best_layer for h in probes})}")
    model, tok = probe.load_probe_target()
    snippets = load_attribution_snippets(max_snippets=args.max_snippets)
    print(f"attributing {len(snippets)} snippets from shards {config.SHARDS}")
    token_map = _token_strings(tok, snippets)
    run_attribution(snippets, probes, token_map=token_map, model=model, tokenizer=tok)
    if args.push:
        push_attribution()
    return 0


# =========================================================================== #
# 8. Smoke — plant fake probes + synthetic activations, NO real model
# =========================================================================== #
def _plant_fake_probe(probes_dir: Path, probe_id: str, regime: str,
                      d: int, layer: int, metric: float, seed: int = 0):
    """Write a probe artifact (L{layer}.pt + metrics.json + _index.json entry) in the
    exact layout probe.save_probe produces, using the real AttentionProbe class."""
    torch.manual_seed(seed)
    dirname = probe.sanitize_probe_id(probe_id)
    pdir = probes_dir / dirname
    pdir.mkdir(parents=True, exist_ok=True)

    mod = AttentionProbe(d, regime=regime)
    meta = {"probe_id": probe_id, "regime": regime, "d_model": d,
            "feat_mean": [0.0] * d, "feat_std": [1.0] * d,
            "y_mean": 0.0, "y_std": 1.0}
    metric_key = "auroc" if regime == "presence" else "r2"
    layer_metrics = {"layer": layer, "auroc": None, "spearman": None, "r2": None,
                     "bin_auroc": None, "best": True}
    layer_metrics[metric_key] = metric
    if regime == "scalar":
        layer_metrics["spearman"] = metric
    torch.save({"state_dict": mod.state_dict(), "meta": meta,
                "layer_metrics": layer_metrics}, pdir / f"L{layer}.pt")

    reliable = metric > config.RELIABLE_METRIC_THRESH
    metrics = {"probe_id": probe_id, "regime": regime, "d_model": d,
               "per_layer": [layer_metrics], "best_layer": layer,
               "best_metric": metric, "reliable": reliable}
    (pdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    idx_path = probes_dir / "_index.json"
    idx = {}
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
    idx[probe_id] = dirname
    idx_path.write_text(json.dumps(idx, indent=2, sort_keys=True))


def main_smoke() -> int:
    import shutil
    import tempfile

    print("=== attribute.py --smoke (synthetic, CPU, no real model) ===")
    d = 64
    tmp = Path(tempfile.mkdtemp(prefix="attr_smoke_"))
    probes_dir = tmp / "probes"
    out_dir = tmp / "attribution"
    probes_dir.mkdir(parents=True, exist_ok=True)

    # ---- plant fake probes: 2 presence (same concept -> high-performer contest) +
    #      1 scalar; two distinct best_layers (2,3) to exercise layer-at-a-time. ----
    _plant_fake_probe(probes_dir, "months::January", "presence", d, layer=2, metric=0.95, seed=1)
    _plant_fake_probe(probes_dir, "months::February", "presence", d, layer=3, metric=0.70, seed=2)
    _plant_fake_probe(probes_dir, "scalar::numbers", "scalar", d, layer=2, metric=0.92, seed=3)

    probes = load_probes(probes_dir)
    assert len(probes) == 3, [p.probe_id for p in probes]
    hp = {p.probe_id: p.high_performer for p in probes}
    assert hp["months::January"] and not hp["months::February"], hp
    assert hp["scalar::numbers"], hp
    rel = {p.probe_id: p.reliable for p in probes}
    assert rel["months::January"] and rel["scalar::numbers"] and not rel["months::February"], rel
    print(f"loaded probes: high_performer={hp} reliable={rel}")

    # ---- synthetic snippets + per-layer activation cache (T<=16, d=64) ----
    rng = np.random.default_rng(0)
    sids = [f"s{i}" for i in range(5)]
    Tmap = {sid: int(rng.integers(4, 17)) for sid in sids}
    token_map = {sid: [f"t{j}" for j in range(Tmap[sid])] for sid in sids}
    base = {sid: rng.standard_normal((Tmap[sid], d)).astype(np.float32) for sid in sids}

    def cache_fn(layer: int) -> Dict[str, torch.Tensor]:
        # deeper layer -> scaled norms, so mean_norm_per_layer differs per layer.
        scale = 1.0 + 0.5 * layer
        return {sid: torch.from_numpy(base[sid] * scale) for sid in sids}

    summary = run_attribution(snippets=[], probes=probes, token_map=token_map,
                              cache_fn=cache_fn, out_dir=out_dir)

    # ---- verify outputs ----
    import pandas as pd
    attr_path = out_dir / summary["attribution_file"]
    agg_path = out_dir / summary["aggregates_file"]
    assert attr_path.exists() and agg_path.exists(), (attr_path, agg_path)
    assert (out_dir / "summary.json").exists()

    df = (pd.read_parquet(attr_path) if attr_path.suffix == ".parquet"
          else pd.read_json(attr_path, lines=True))
    cols = set(df.columns)
    need = {"snippet_id", "token_idx", "token", "layer", "h_norm", "probe_id",
            "regime", "s_t", "a_t", "pooled_score", "is_high_performer", "is_reliable"}
    assert need <= cols, f"missing cols: {need - cols}"

    # BOTH per-token probe activation AND ||h_t|| present, finite, norm > 0
    assert np.isfinite(df["s_t"]).all() and np.isfinite(df["a_t"]).all(), "non-finite activation"
    assert np.isfinite(df["h_norm"]).all(), "non-finite norm"
    assert (df["h_norm"] > 0).all(), "norm not strictly positive"
    assert ((df["a_t"] >= 0) & (df["a_t"] <= 1)).all(), "a_t out of [0,1]"

    # attention weights per (snippet, probe) sum to ~1 (localization distribution)
    s = df.groupby(["snippet_id", "probe_id"])["a_t"].sum()
    assert np.allclose(s.values, 1.0, atol=1e-3), f"a_t doesn't sum to 1: {s.min()},{s.max()}"

    # expected row count: sum over probes of that probe's snippet token counts
    exp_rows = sum(Tmap[sid] for sid in sids) * len(probes)
    assert len(df) == exp_rows, (len(df), exp_rows)

    # aggregates: cross-probe means + per-layer norms present
    ag = (pd.read_parquet(agg_path) if agg_path.suffix == ".parquet"
          else pd.read_json(agg_path, lines=True))
    assert {"mean_a_concepts", "mean_s_concepts", "mean_a_reliable",
            "n_reliable_probes", "norms_by_layer"} <= set(ag.columns)
    assert (ag["n_reliable_probes"] == 2).all(), "reliable probe count wrong"
    n_tokens = sum(Tmap.values())
    assert len(ag) == n_tokens, (len(ag), n_tokens)
    # each token has norms recorded at BOTH layers read (2 and 3)
    nbl = json.loads(ag["norms_by_layer"].iloc[0])
    assert set(nbl.keys()) == {"2", "3"}, nbl
    assert all(v > 0 for v in nbl.values()), nbl

    # summary sanity
    assert summary["n_reliable_probes"] == 2
    assert summary["n_high_performers"] == 2  # January + numbers
    assert set(summary["mean_norm_per_layer"].keys()) == {"2", "3"}
    assert summary["mean_norm_per_layer"]["3"] > summary["mean_norm_per_layer"]["2"]

    print("\n--- SMOKE CHECKS ---")
    print(f"attribution rows           = {len(df)}  (format={summary['attribution_format']})")
    print(f"aggregate rows (tokens)    = {len(ag)}")
    print(f"probes loaded              = {summary['n_probes']} "
          f"(reliable={summary['n_reliable_probes']}, high_perf={summary['n_high_performers']})")
    print(f"layers read                = {summary['layers_read']}")
    print(f"mean ||h_t|| per layer     = {summary['mean_norm_per_layer']}")
    print(f"h_norm range               = [{df['h_norm'].min():.3f}, {df['h_norm'].max():.3f}]  (all > 0, finite)")
    print(f"s_t range                  = [{df['s_t'].min():.3f}, {df['s_t'].max():.3f}]  (finite)")
    print(f"a_t per (snippet,probe) sums to 1: OK")
    print("BOTH per-token probe activation (s_t, a_t) AND ||h_t|| norm recorded: OK")

    shutil.rmtree(tmp, ignore_errors=True)
    print("cleaned up temp artifacts")
    print("=== SMOKE PASS ===")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Step-3 data attribution")
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic CPU smoke test (no real model) and exit")
    ap.add_argument("--push", action="store_true",
                    help="push artifacts/attribution/ to HF after attributing")
    ap.add_argument("--max-snippets", type=int, default=ATTR_MAX_SNIPPETS,
                    dest="max_snippets", help="number of corpus snippets to attribute")
    args = ap.parse_args()
    if args.smoke:
        return main_smoke()
    return main_real(args)


if __name__ == "__main__":
    sys.exit(main())
