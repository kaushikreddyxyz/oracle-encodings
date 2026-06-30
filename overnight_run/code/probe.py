"""
probe.py — Step 2 of the overnight concept-probes run: train sequence-level
ATTENTION probes on the residual stream of the probe-target LLM.

Design contract (see SPEC.md / overnight_brief.md):
  * One probe per probe_id:
        presence:  "concept::cls"   -> binary  (AUROC)
        scalar:    "scalar::name"   -> scalar   (Spearman, R^2, binarized-extremes AUROC)
  * Sweep layers (every config.LAYER_STRIDE-th layer of the probe target).
  * STORAGE-SAFE: cache ONE LAYER AT A TIME for the whole labeled set, fit every
    probe for that layer, write weights, free the cache, advance. Gemma-2-9b is
    ~42 layers x d~=3584 -> caching all-layer x all-token would blow the volume.
  * Resumable: skip a (probe_id, layer) whose weight file + metrics entry exist.
  * Artifacts:  artifacts/probes/{probe_dir}/L{layer}.pt   (state_dict + meta)
                artifacts/probes/{probe_dir}/metrics.json  (per_layer + best_layer)
                artifacts/probes/_index.json               (probe_id <-> dir map)
  * push_probes() uploads artifacts/probes/ to config.HF_MODEL_REPO.
  * load_probe_target() loads config.PROBE_TARGET (bf16, device_map=auto,
    output_hidden_states) with a clear gated-model error message.
  * --smoke runs the FULL train/eval/save path on a planted synthetic dataset for
    2 fake layers x 2 probe_ids (one presence, one scalar) with NO real model, and
    asserts presence AUROC > 0.8 and scalar R^2 > 0.5.

Run with the project venv:
    .venv/bin/python3 overnight_run/code/probe.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
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
import torch.nn as nn
import torch.nn.functional as F

import config  # noqa: E402
import concepts  # noqa: E402

PROBES_DIR = config.ARTIFACTS / "probes"
INDEX_PATH = PROBES_DIR / "_index.json"

# Minimum examples required to even attempt a probe (need both classes for presence).
MIN_EXAMPLES = 8
MIN_PER_CLASS = 3


# =========================================================================== #
# 1. Attention probe module
# =========================================================================== #
class AttentionProbe(nn.Module):
    """Sequence-level attention probe.

    A learned query vector scores each token's hidden state; a (masked) softmax
    over tokens gives an attention distribution; the attention-weighted sum pools
    the sequence to one vector; a linear head emits a single logit (presence) or
    scalar (regression). Exposes BOTH the pooled score AND the attention
    distribution (the latter is the Step-3 localization signal).
    """

    def __init__(self, d_model: int, regime: str = "presence"):
        super().__init__()
        self.d_model = d_model
        self.regime = regime
        # learned query vector over tokens. Scores are NOT divided by sqrt(d): with
        # standardized features the raw dot product is already O(1)-scaled, and the
        # query is free to grow its norm to SHARPEN attention onto the signal token
        # (a fixed 1/sqrt(d) factor would pin attention near-uniform -> poor
        # localization, which kills scalar readout).
        self.query = nn.Parameter(torch.randn(d_model) * (1.0 / math.sqrt(d_model)))
        # linear readout head on the pooled vector
        self.head = nn.Linear(d_model, 1)

    def forward(
        self, H: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """H: (B, T, d) padded hidden states. mask: (B, T) bool, True = real token.
        Returns (out, attn): out (B,) pooled logit/scalar, attn (B, T) over tokens."""
        scores = H @ self.query                                  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)                     # (B, T)
        attn = torch.nan_to_num(attn, nan=0.0)                   # all-pad row safety
        pooled = (attn.unsqueeze(-1) * H).sum(dim=1)             # (B, d)
        out = self.head(pooled).squeeze(-1)                      # (B,)
        return out, attn

    @torch.no_grad()
    def score_sequence(self, H: torch.Tensor) -> Tuple[float, np.ndarray]:
        """Single (T, d) sequence -> (pooled score, attention weights over tokens).
        Used by attribute.py for localization. Score is sigmoid prob for presence."""
        self.eval()
        Hb = H.unsqueeze(0)
        mask = torch.ones(Hb.shape[:2], dtype=torch.bool, device=H.device)
        out, attn = self.forward(Hb, mask)
        val = out.item()
        if self.regime == "presence":
            val = float(torch.sigmoid(out).item())
        return val, attn.squeeze(0).cpu().numpy()


# =========================================================================== #
# 2. Labeled-data loading & per-probe dataset assembly
# =========================================================================== #
@dataclass
class LabeledExample:
    id: str
    concept: str
    cls: Optional[str]
    regime: str
    text: str = ""
    label: Optional[int] = None        # presence 0/1
    value: Optional[float] = None      # scalar rating
    external: Optional[float] = None   # ground-truth scalar where available
    discarded: bool = False
    raw: dict = field(default_factory=dict)

    def scalar_target(self) -> Optional[float]:
        """Prefer the external scalar (brief: 'prefer the external scalar where one
        exists'); else the judge mean rating."""
        if self.external is not None:
            return float(self.external)
        if self.value is not None:
            return float(self.value)
        return None


def load_labeled_examples(labels_dir: Optional[Path] = None) -> List[LabeledExample]:
    """Read every data/labels/{concept}.jsonl into a flat list.

    Keeps every row carrying a usable supervised target, EVEN if flagged discarded for
    *balance* reasons. The labeling stage marks surplus positives discarded to force a
    50/50 set, but those positives are exactly what rich geometry clouds need (and probe
    training balances via cross-class negatives + a per-probe negative cap). Only
    genuinely unlabeled discards (ambiguous_mean / high_variance / too_few_valid ->
    label/value None) are dropped. Presence positives are capped per (concept,cls) to
    bound activation-cache compute; presence negatives (scarce) are all kept; scalars
    capped per concept. Tune via MAX_POS_PER_CLASS / MAX_PER_SCALAR env.
    """
    import os as _os
    labels_dir = labels_dir or (config.DATA / "labels")
    cap_pos = int(_os.environ.get("MAX_POS_PER_CLASS", "250"))
    cap_scalar = int(_os.environ.get("MAX_PER_SCALAR", "600"))
    out: List[LabeledExample] = []
    pos_count: Dict[tuple, int] = {}
    scal_count: Dict[str, int] = {}
    if not labels_dir.exists():
        return out
    for fp in sorted(labels_dir.glob("*.jsonl")):
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                regime = o.get("regime", "presence")
                if regime == "presence":
                    if o.get("label") not in (0, 1):
                        continue
                    if int(o["label"]) == 1:                       # cap positives/class
                        k = (o["concept"], o.get("cls"))
                        if pos_count.get(k, 0) >= cap_pos:
                            continue
                        pos_count[k] = pos_count.get(k, 0) + 1
                else:
                    if o.get("value") is None and o.get("external") is None:
                        continue
                    c = o["concept"]
                    if scal_count.get(c, 0) >= cap_scalar:
                        continue
                    scal_count[c] = scal_count.get(c, 0) + 1
                out.append(
                    LabeledExample(
                        id=o["id"], concept=o["concept"], cls=o.get("cls"),
                        regime=regime, text=o.get("text", ""), label=o.get("label"),
                        value=o.get("value"), external=o.get("external"),
                        discarded=bool(o.get("discarded", False)), raw=o,
                    )
                )
    return out


def examples_for_probe(
    probe_id: str, all_examples: List[LabeledExample]
) -> Tuple[str, List[str], np.ndarray]:
    """Assemble a probe's training set from the flat labeled pool.

    Presence "concept::cls": positive = same concept & cls & label==1;
        negative = same concept but a *different* class is present, or this class's
        wrong-sense rejects (label==0). Within-concept negatives keep the task hard
        and the balance near 50/50 (the labeling stage already targets that).
    Scalar "scalar::name": every example of that concept with a usable target.

    Returns (regime, [example_id...], y) aligned by index.
    """
    regime, name = _parse_probe_id(probe_id)
    ids: List[str] = []
    ys: List[float] = []

    if regime == "presence":
        concept, cls = name.split("::", 1) if "::" in name else (name, None)
        for ex in all_examples:
            if ex.concept != concept or ex.regime != "presence":
                continue
            if ex.cls == cls:
                if ex.label is None:
                    continue
                y = 1 if int(ex.label) == 1 else 0
            else:
                # another class of the same concept that IS present -> negative for `cls`
                if ex.label is None or int(ex.label) != 1:
                    continue
                y = 0
            ids.append(ex.id)
            ys.append(float(y))
    else:  # scalar
        for ex in all_examples:
            if ex.concept != name or ex.regime != "scalar":
                continue
            t = ex.scalar_target()
            if t is None:
                continue
            ids.append(ex.id)
            ys.append(float(t))

    ys_arr = np.asarray(ys, dtype=np.float64)
    if regime == "presence" and len(ids) > 0:
        pos = [i for i in range(len(ys_arr)) if ys_arr[i] == 1.0]
        neg = [i for i in range(len(ys_arr)) if ys_arr[i] == 0.0]
        cap = max(1, len(pos)) * 4                  # avoid extreme cross-class imbalance
        if len(neg) > cap:
            rng = np.random.default_rng(abs(hash(probe_id)) % (2**32))
            neg = sorted(int(j) for j in rng.choice(neg, size=cap, replace=False))
            keep = sorted(pos + neg)
            ids = [ids[i] for i in keep]
            ys_arr = ys_arr[keep]
    return regime, ids, ys_arr


def _parse_probe_id(probe_id: str) -> Tuple[str, str]:
    """('months::January') -> ('presence','months::January');
    ('scalar::numbers') -> ('scalar','numbers')."""
    if probe_id.startswith("scalar::"):
        return "scalar", probe_id[len("scalar::"):]
    return "presence", probe_id


def sanitize_probe_id(probe_id: str) -> str:
    """Filesystem-safe directory name; keep a real mapping in _index.json."""
    s = probe_id.replace("::", "__")
    s = s.replace(" ", "_").replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^A-Za-z0-9_.\-]", "_", s)
    return s


def _register_probe_dir(probe_id: str) -> Path:
    pdir = PROBES_DIR / sanitize_probe_id(probe_id)
    pdir.mkdir(parents=True, exist_ok=True)
    index: Dict[str, str] = {}
    if INDEX_PATH.exists():
        try:
            index = json.loads(INDEX_PATH.read_text())
        except json.JSONDecodeError:
            index = {}
    if index.get(probe_id) != pdir.name:
        index[probe_id] = pdir.name
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True))
    return pdir


# =========================================================================== #
# 3. Activation extraction (one layer at a time)
# =========================================================================== #
def load_probe_target(model_name: Optional[str] = None):
    """Load config.PROBE_TARGET for activation extraction. bf16, device_map=auto,
    output_hidden_states. A gated/missing model raises a CLEAR RuntimeError instead
    of an opaque stack trace. NEVER hit during --smoke."""
    model_name = model_name or config.PROBE_TARGET
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            output_hidden_states=True,
        )
        model.eval()
        return model, tok
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Could not load probe target '{model_name}'. If it is license-gated "
            f"(e.g. google/gemma-2-9b), accept the license on HF and `huggingface-cli "
            f"login`, or set PROBE_TARGET=<fallback>. Original error: {type(e).__name__}: {e}"
        ) from e


@torch.no_grad()
def extract_layer_activations(
    model,
    tokenizer,
    examples: List[LabeledExample],
    layer: int,
    device: Optional[str] = None,
    batch_size: int = 64,
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """Run the probe target once and return {example_id: (T, d) hidden states} for a
    SINGLE layer. `layer` indexes hidden_states (0 = embeddings, 1..N = blocks), so
    a depth sweep is range(1, num_hidden_layers+1, config.LAYER_STRIDE). Caller is
    expected to free the returned dict before requesting the next layer.
    """
    cache: Dict[str, torch.Tensor] = {}
    with torch.inference_mode():                    # pure inference: no autograd graph
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            enc = tokenizer(
                [ex.text for ex in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            if device is not None:
                enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer]              # (B, T, d)
            attn_mask = enc["attention_mask"]          # (B, T)
            for j, ex in enumerate(batch):
                keep = attn_mask[j].bool()
                cache[ex.id] = hs[j][keep].float().cpu()
    return cache


# =========================================================================== #
# 4. Training / eval for one probe at one layer
# =========================================================================== #
def _collate(batch: List[Tuple[torch.Tensor, float]], device: str):
    Hs = [b[0] for b in batch]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)
    T = max(h.shape[0] for h in Hs)
    d = Hs[0].shape[1]
    B = len(Hs)
    H_pad = torch.zeros(B, T, d, dtype=torch.float32, device=device)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for k, h in enumerate(Hs):
        t = h.shape[0]
        H_pad[k, :t] = h.to(device)
        mask[k, :t] = True
    return H_pad, mask, ys


def _feature_stats(Hs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    allcat = torch.cat(Hs, dim=0)                 # (sum_T, d)
    mean = allcat.mean(dim=0)
    std = allcat.std(dim=0).clamp_min(1e-6)
    return mean, std


def train_probe_on_layer(
    probe_id: str,
    regime: str,
    Hs: List[torch.Tensor],
    y: np.ndarray,
    device: str = "cpu",
    seed: int = 0,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> Tuple[Optional[AttentionProbe], dict, dict]:
    """Train one attention probe on one layer's cached activations.
    Returns (model_or_None, meta, metrics). model is None if the probe was skipped
    (too few examples / single class); metrics carries a 'skipped' reason then."""
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from scipy.stats import spearmanr

    epochs = epochs or config.PROBE_EPOCHS
    lr = lr or config.PROBE_LR
    batch_size = batch_size or config.PROBE_BATCH
    d_model = Hs[0].shape[1]
    n = len(Hs)
    idx = np.arange(n)

    base = {"layer": None, "auroc": None, "spearman": None, "r2": None,
            "bin_auroc": None, "n_pos": None, "n_neg": None,
            "n_train": None, "n_val": None, "best": False}

    # ---- guards ----
    if n < MIN_EXAMPLES:
        return None, {}, {**base, "skipped": f"too_few_examples({n})"}

    if regime == "presence":
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        if pos < MIN_PER_CLASS or neg < MIN_PER_CLASS:
            return None, {}, {**base, "n_pos": pos, "n_neg": neg,
                              "skipped": f"class_imbalance(pos={pos},neg={neg})"}
        tr, va = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    else:
        tr, va = train_test_split(idx, test_size=0.2, random_state=seed)

    # ---- stack each split into ONE GPU-resident padded tensor (the speed fix) ----
    # The old per-batch _collate rebuilt tensors + did per-example CPU->GPU copies
    # every step (GPU ~0% util, ~500s/probe). Pre-stack once; train by slicing on GPU.
    def _stack(indices):
        # Move each cached (T,d) tensor to GPU FIRST, then pad there. Padding on CPU is
        # pathologically slow on this host (252 threads -> PyTorch thrashes the tiny
        # per-example copies: ~69s/probe). GPU pad is ~0.15s/probe. (measured)
        subs = [Hs[i].to(device, non_blocking=True) for i in indices]
        H = nn.utils.rnn.pad_sequence(subs, batch_first=True)        # (n, T, d) on GPU
        lengths = torch.tensor([h.shape[0] for h in subs], device=device)
        M = torch.arange(H.shape[1], device=device)[None, :] < lengths[:, None]  # (n,T) bool
        return H, M

    Htr, Mtr = _stack(tr)
    Hva, Mva = _stack(va)

    # feature standardization on GPU (fit on train tokens only; stored in meta)
    flat = Htr[Mtr]                                  # (sum_T_tr, d)
    mean = flat.mean(0)
    std = flat.std(0).clamp_min(1e-6)
    del flat
    # keep pad rows at 0 (match original H_pad semantics; avoid attention contamination)
    Htr = ((Htr - mean) / std).masked_fill(~Mtr.unsqueeze(-1), 0.0)
    Hva = ((Hva - mean) / std).masked_fill(~Mva.unsqueeze(-1), 0.0)

    # scalar target standardization (stable MSE; invert for metrics)
    y_mean, y_std = 0.0, 1.0
    if regime == "scalar":
        y_mean = float(y[tr].mean())
        y_std = float(y[tr].std()) or 1.0
    ytr = torch.tensor(y[tr], dtype=torch.float32, device=device)
    if regime == "scalar":
        ytr = (ytr - y_mean) / y_std

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AttentionProbe(d_model, regime=regime).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = (nn.BCEWithLogitsLoss() if regime == "presence" else nn.MSELoss())

    model.train()
    n_tr = Htr.shape[0]
    bs = batch_size                                   # same dynamics as before; fast now
                                                      # because data is GPU-resident (no
                                                      # per-batch CPU collate per step)
    for _ep in range(epochs):
        perm = torch.from_numpy(np.random.permutation(n_tr)).to(device)  # match orig RNG
        for s in range(0, n_tr, bs):
            bidx = perm[s : s + bs]
            out, _ = model(Htr[bidx], Mtr[bidx])
            loss = loss_fn(out, ytr[bidx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ---- eval on val ----
    model.eval()
    with torch.no_grad():
        out, _ = model(Hva, Mva)
        out = out.cpu().numpy()
    del Htr, Mtr, Hva, Mva, ytr
    y_val = y[va]

    metrics = {**base}
    if regime == "presence":
        probs = 1.0 / (1.0 + np.exp(-out))
        metrics["auroc"] = float(roc_auc_score(y_val, probs))
        metrics["n_pos"] = int((y == 1).sum())
        metrics["n_neg"] = int((y == 0).sum())
    else:
        preds = out * y_std + y_mean
        sr = spearmanr(preds, y_val).correlation
        metrics["spearman"] = float(sr) if sr == sr else None  # nan guard
        metrics["r2"] = float(r2_score(y_val, preds))
        metrics["bin_auroc"] = _binarized_extremes_auroc(y_val, preds)
        metrics["n_pos"] = metrics["n_neg"] = None
    metrics["n_train"] = int(len(tr))
    metrics["n_val"] = int(len(va))

    meta = {
        "probe_id": probe_id,
        "regime": regime,
        "d_model": int(d_model),
        "feat_mean": mean.cpu().numpy().tolist(),
        "feat_std": std.cpu().numpy().tolist(),
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return model, meta, metrics


def _binarized_extremes_auroc(y_true: np.ndarray, preds: np.ndarray) -> Optional[float]:
    """Top-quantile vs bottom-quantile AUROC so scalar probes are comparable to
    presence probes. Returns None if extremes are degenerate."""
    from sklearn.metrics import roc_auc_score

    if len(y_true) < 6:
        return None
    lo, hi = np.quantile(y_true, 0.3), np.quantile(y_true, 0.7)
    if not (hi > lo):
        return None
    sel = (y_true <= lo) | (y_true >= hi)
    bin_lab = (y_true[sel] >= hi).astype(int)
    if bin_lab.min() == bin_lab.max():
        return None
    return float(roc_auc_score(bin_lab, preds[sel]))


# =========================================================================== #
# 5. Artifact I/O (resumable) + best-layer selection
# =========================================================================== #
def _metrics_path(probe_id: str) -> Path:
    return PROBES_DIR / sanitize_probe_id(probe_id) / "metrics.json"


def _weight_path(probe_id: str, layer: int) -> Path:
    return PROBES_DIR / sanitize_probe_id(probe_id) / f"L{layer}.pt"


def _primary_metric(regime: str, m: dict) -> Optional[float]:
    return m.get("auroc") if regime == "presence" else m.get("spearman")


def _reliable(regime: str, m: dict) -> bool:
    """brief: reliable = AUROC/R2 > RELIABLE_METRIC_THRESH (R2 for scalars)."""
    v = m.get("auroc") if regime == "presence" else m.get("r2")
    return v is not None and v > config.RELIABLE_METRIC_THRESH


def already_done(probe_id: str, layer: int) -> bool:
    """Resumable skip: weight file exists AND metrics.json has this layer."""
    if not _weight_path(probe_id, layer).exists():
        return False
    mp = _metrics_path(probe_id)
    if not mp.exists():
        return False
    try:
        m = json.loads(mp.read_text())
    except json.JSONDecodeError:
        return False
    return any(e.get("layer") == layer for e in m.get("per_layer", []))


def save_probe(probe_id: str, regime: str, layer: int,
               model: AttentionProbe, meta: dict, layer_metrics: dict):
    pdir = _register_probe_dir(probe_id)
    layer_metrics = {**layer_metrics, "layer": int(layer)}
    torch.save({"state_dict": model.state_dict(), "meta": meta,
                "layer_metrics": layer_metrics}, _weight_path(probe_id, layer))
    _update_metrics(probe_id, regime, meta.get("d_model"), layer_metrics)


def record_skipped(probe_id: str, regime: str, layer: int, layer_metrics: dict):
    """Persist a layer entry even when the probe was skipped, so resume doesn't retry."""
    _register_probe_dir(probe_id)
    _update_metrics(probe_id, regime, layer_metrics.get("d_model"),
                    {**layer_metrics, "layer": int(layer)})


def _update_metrics(probe_id: str, regime: str, d_model, layer_metrics: dict):
    mp = _metrics_path(probe_id)
    doc = {"probe_id": probe_id, "regime": regime, "d_model": d_model,
           "per_layer": [], "best_layer": None, "best_metric": None, "reliable": False}
    if mp.exists():
        try:
            doc = json.loads(mp.read_text())
        except json.JSONDecodeError:
            pass
    doc["probe_id"], doc["regime"] = probe_id, regime
    if d_model is not None:
        doc["d_model"] = d_model
    per = [e for e in doc.get("per_layer", []) if e.get("layer") != layer_metrics["layer"]]
    per.append(layer_metrics)
    per.sort(key=lambda e: e["layer"])
    # recompute best over layers that actually trained
    best_layer, best_val = None, -np.inf
    for e in per:
        e["best"] = False
        v = _primary_metric(regime, e)
        if v is not None and v > best_val:
            best_val, best_layer = v, e["layer"]
    if best_layer is not None:
        for e in per:
            if e["layer"] == best_layer:
                e["best"] = True
                doc["reliable"] = _reliable(regime, e)
    doc["per_layer"] = per
    doc["best_layer"] = best_layer
    doc["best_metric"] = None if best_layer is None else float(best_val)
    mp.write_text(json.dumps(doc, indent=2))


# =========================================================================== #
# 6. Orchestration driver (layer-outer, probe-inner = storage-safe)
# =========================================================================== #
def run_training(
    examples: List[LabeledExample],
    layers: List[int],
    probe_ids: Optional[List[str]] = None,
    cache_fn: Optional[Callable[[int], Dict[str, torch.Tensor]]] = None,
    model=None,
    tokenizer=None,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Train every probe across every layer, caching ONE layer at a time.

    `cache_fn(layer) -> {example_id: (T,d) tensor}` lets the smoke path inject
    synthetic activations. If None, activations come from extract_layer_activations
    on the real model. The cache is freed before advancing to the next layer.
    """
    if probe_ids is None:
        probe_ids = concepts.all_probe_ids()

    # Pre-assemble each probe's (example_ids, y) once (independent of layer).
    assembled: Dict[str, Tuple[str, List[str], np.ndarray]] = {}
    for pid in probe_ids:
        assembled[pid] = examples_for_probe(pid, examples)

    summary = {"layers": list(layers), "n_probes": len(probe_ids),
               "trained": 0, "skipped": 0, "resumed": 0}

    for layer in layers:
        # which probes still need this layer?
        todo = [pid for pid in probe_ids if not already_done(pid, layer)]
        if not todo:
            summary["resumed"] += len(probe_ids)
            if verbose:
                print(f"[layer {layer}] all {len(probe_ids)} probes already done; skip")
            continue

        if cache_fn is not None:
            cache = cache_fn(layer)
        else:
            if model is None or tokenizer is None:
                raise ValueError("Need model+tokenizer (or cache_fn) to extract activations.")
            cache = extract_layer_activations(model, tokenizer, examples, layer, device=device)

        for pid in probe_ids:
            if already_done(pid, layer):
                summary["resumed"] += 1
                continue
            regime, ex_ids, y = assembled[pid]
            Hs = [cache[i] for i in ex_ids if i in cache]
            if len(Hs) != len(ex_ids):
                # some ids missing from cache (truncated/empty) -> realign y
                keep = [k for k, i in enumerate(ex_ids) if i in cache]
                y = y[keep]
            model_, meta, m = train_probe_on_layer(pid, regime, Hs, y,
                                                   device=device, seed=layer)
            if model_ is None:
                record_skipped(pid, regime, layer, m)
                summary["skipped"] += 1
                if verbose:
                    print(f"[layer {layer}] {pid}: SKIP ({m.get('skipped')})")
            else:
                save_probe(pid, regime, layer, model_, meta, m)
                summary["trained"] += 1
                if verbose:
                    pm = _primary_metric(regime, m)
                    extra = (f"auroc={m['auroc']:.3f}" if regime == "presence"
                             else f"spearman={m['spearman']:.3f} r2={m['r2']:.3f} "
                                  f"bin_auroc={m['bin_auroc']}")
                    print(f"[layer {layer}] {pid}: {extra}")

        # FREE the layer cache before advancing (storage/memory safety)
        del cache
        if device != "cpu":
            torch.cuda.empty_cache()

    if verbose:
        print(f"[done] {summary}")
    return summary


# =========================================================================== #
# 7. HF push helper
# =========================================================================== #
def push_probes(repo_id: Optional[str] = None) -> str:
    """Upload artifacts/probes/ to config.HF_MODEL_REPO using a cached login token.
    Not called during --smoke."""
    from huggingface_hub import HfApi, upload_folder

    repo_id = repo_id or config.HF_MODEL_REPO
    api = HfApi()
    api.create_repo(repo_id, repo_type="model",
                    private=config.HF_PRIVATE, exist_ok=True)
    upload_folder(folder_path=str(PROBES_DIR), repo_id=repo_id,
                  repo_type="model", commit_message="probe weights + metrics")
    url = f"https://huggingface.co/{repo_id}"
    print(f"[push] uploaded probes -> {url}")
    return url


# =========================================================================== #
# 8. Real entrypoint
# =========================================================================== #
def main_real(args):
    torch.set_num_threads(8)      # host has 252 cores; tiny CPU tensor ops thrash otherwise
    examples = load_labeled_examples()
    if not examples:
        print("No labeled examples in data/labels/. Run label.py first.")
        return 1
    model, tok = load_probe_target()
    n_layers = model.config.num_hidden_layers
    layers = list(range(1, n_layers + 1, config.LAYER_STRIDE))
    device = next(model.parameters()).device.type
    run_training(examples, layers, model=model, tokenizer=tok, device=device)
    if args.push:
        push_probes()
    return 0


# =========================================================================== #
# 9. Smoke test — planted synthetic data, NO real model
# =========================================================================== #
def _make_synthetic(seed: int = 0, n: int = 200, d: int = 64, layers=(4, 8)):
    """Build a synthetic labeled set + per-layer activation cache with a PLANTED
    linear signal. A 'concept token' carries a constant marker (so the attention
    probe can localize it) plus, along an orthogonal direction, the target value
    (1 for presence positives, the rating for scalar). Deeper fake layer = stronger
    signal, so best_layer should resolve to the stronger one.
    """
    rng = np.random.default_rng(seed)
    marker = rng.standard_normal(d); marker /= np.linalg.norm(marker)
    w_val = rng.standard_normal(d); w_val /= np.linalg.norm(w_val)
    M, Vscale = 6.0, 12.0
    gain = {ly: 0.9 + 0.25 * k for k, ly in enumerate(layers)}  # 0.9, 1.15, ...

    examples: List[LabeledExample] = []
    spec = {}  # id -> (T, pos_idx, bears_token, value)

    # presence probe: synthetic_presence::Pos  (~50/50)
    for i in range(n):
        T = int(rng.integers(6, 24))
        label = int(i % 2 == 0)
        pos_idx = int(rng.integers(0, T))
        examples.append(LabeledExample(
            id=f"pres_{i}", concept="synthetic_presence", cls="Pos",
            regime="presence", text="", label=label, value=None))
        spec[f"pres_{i}"] = (T, pos_idx, bool(label), 1.0)

    # scalar probe: scalar::synthetic_scalar  (value in [0,1])
    for i in range(n):
        T = int(rng.integers(6, 24))
        v = float(rng.random())
        pos_idx = int(rng.integers(0, T))
        examples.append(LabeledExample(
            id=f"scal_{i}", concept="synthetic_scalar", cls=None,
            regime="scalar", text="", value=v, external=None))
        spec[f"scal_{i}"] = (T, pos_idx, True, v)

    caches: Dict[int, Dict[str, torch.Tensor]] = {}
    for ly in layers:
        g = gain[ly]
        cache: Dict[str, torch.Tensor] = {}
        for ex in examples:
            T, pos_idx, bears, val = spec[ex.id]
            H = rng.standard_normal((T, d)).astype(np.float32)
            if bears:
                H[pos_idx] += (M * g) * marker + (val * Vscale * g) * w_val
            cache[ex.id] = torch.from_numpy(H)
        caches[ly] = cache
    return examples, caches, list(layers)


def main_smoke():
    print("=== probe.py --smoke (synthetic, CPU, no real model) ===")
    layers = (4, 8)
    examples, caches, layer_list = _make_synthetic(layers=layers)
    probe_ids = ["synthetic_presence::Pos", "scalar::synthetic_scalar"]

    # sanity: assembly produces sane sets
    rg, ids, y = examples_for_probe("synthetic_presence::Pos", examples)
    assert rg == "presence" and len(ids) == 200, (rg, len(ids))
    assert set(np.unique(y).tolist()) == {0.0, 1.0}, np.unique(y)
    rg, ids, y = examples_for_probe("scalar::synthetic_scalar", examples)
    assert rg == "scalar" and len(ids) == 200, (rg, len(ids))

    run_training(examples, layer_list, probe_ids=probe_ids,
                 cache_fn=lambda ly: caches[ly], device="cpu")

    # ---- verify artifacts + metrics ----
    ok = True
    best = {}
    for pid in probe_ids:
        mp = _metrics_path(pid)
        assert mp.exists(), f"missing metrics.json for {pid}"
        doc = json.loads(mp.read_text())
        assert len(doc["per_layer"]) == len(layer_list), doc["per_layer"]
        for ly in layer_list:
            wp = _weight_path(pid, ly)
            assert wp.exists(), f"missing weights {wp}"
            assert already_done(pid, ly), f"resume check failed for {pid} L{ly}"
        best[pid] = doc
        print(f"  {pid}: best_layer={doc['best_layer']} "
              f"best_metric={doc['best_metric']:.3f} reliable={doc['reliable']}")

    # ---- assert planted signal recovered ----
    pres = best["synthetic_presence::Pos"]
    pres_auroc = max(e["auroc"] for e in pres["per_layer"])
    assert pres_auroc > 0.8, f"presence AUROC too low: {pres_auroc:.3f}"

    scal = best["scalar::synthetic_scalar"]
    scal_r2 = max(e["r2"] for e in scal["per_layer"])
    scal_sp = max(e["spearman"] for e in scal["per_layer"])
    scal_binau = max(e["bin_auroc"] for e in scal["per_layer"])
    assert scal_r2 > 0.5, f"scalar R^2 too low: {scal_r2:.3f}"

    # index + load-back round trip of one probe
    idx = json.loads(INDEX_PATH.read_text())
    assert idx["scalar::synthetic_scalar"] == "scalar__synthetic_scalar", idx
    ckpt = torch.load(_weight_path("scalar::synthetic_scalar", layers[-1]),
                      weights_only=False)
    m = AttentionProbe(ckpt["meta"]["d_model"], regime="scalar")
    m.load_state_dict(ckpt["state_dict"])
    H0 = caches[layers[-1]]["scal_0"]
    sc, attn = m.score_sequence((H0 - torch.tensor(ckpt["meta"]["feat_mean"]))
                                / torch.tensor(ckpt["meta"]["feat_std"]))
    assert attn.ndim == 1 and abs(attn.sum() - 1.0) < 1e-4, attn.sum()

    print("\n--- SMOKE METRICS (synthetic) ---")
    print(f"presence  best AUROC      = {pres_auroc:.3f}   (threshold > 0.80)")
    print(f"scalar    best R^2        = {scal_r2:.3f}   (threshold > 0.50)")
    print(f"scalar    best Spearman   = {scal_sp:.3f}")
    print(f"scalar    best bin-AUROC  = {scal_binau:.3f}")
    print(f"attention dist sums to 1, localized; load-back OK")
    print(f"artifacts under: {PROBES_DIR}")

    # Clean up synthetic artifacts so they never leak into the real run / HF push.
    import shutil
    for pid in probe_ids:
        shutil.rmtree(PROBES_DIR / sanitize_probe_id(pid), ignore_errors=True)
    if INDEX_PATH.exists():
        idx = json.loads(INDEX_PATH.read_text())
        for pid in probe_ids:
            idx.pop(pid, None)
        INDEX_PATH.write_text(json.dumps(idx, indent=2, sort_keys=True))
    print("cleaned up synthetic artifacts")
    print("=== SMOKE PASS ===" if ok else "=== SMOKE FAIL ===")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Step-2 attention-probe training")
    ap.add_argument("--smoke", action="store_true",
                    help="run synthetic CPU smoke test (no real model) and exit")
    ap.add_argument("--push", action="store_true",
                    help="push artifacts/probes/ to HF after training")
    args = ap.parse_args()
    if args.smoke:
        return main_smoke()
    return main_real(args)


if __name__ == "__main__":
    sys.exit(main())
