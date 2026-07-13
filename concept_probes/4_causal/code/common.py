"""Stage 6.1 shared plumbing: model/natstats/arm loading, dose calibration,
probe readout, batching. Implements the DESIGN.md "common.py API".

Confirmed data shapes (loaded from disk 2026-07-02; downstream scripts can
rely on these):
- 2_probes/natstats26.npz: keys ``mean_{l}``/``std_{l}`` for l = 0..25, each
  fp32 [2304], plus scalar ``n_tokens`` (1,654,040). l is the PROBE layer
  (block index); mean_l/std_l are stats of block-l output = hidden_states[l+1].
- 2_probes/probes/<family>/probes_l{L}.npz: ``classes`` [C] str,
  ``W_ridge`` [3, C, 2304] fp32 (NOT unit-norm; ||W||~0.08),
  ``b_ridge`` [3, C], ``chosen_lambda_ridge`` [C] int (index into the 3
  lambdas), ``W_dom`` [C, 2304] (||W||~13-16), ``W_lda`` [C, 2304] (~4),
  ``rand_dirs`` [20, 2304] (unit-norm), plus adam/logistic candidates.
- 3_validation/data/natscores/<family>.natscores.npz: ``preds_ridge``
  [12, n_tokens, C] fp32 = z @ W_ridge_chosen.T + b_ridge_chosen (raw
  non-unit W, bias INCLUDED — verified in 3_validation/code/score_natural.py
  proj()), ``layers`` [12] giving the row order, ``classes`` [C],
  ``y`` [n_tokens, C], ``token2ex``, ``ex_nat_split`` etc.

DEVIATIONS from DESIGN.md (documented per contract):
1. Score units. DESIGN assumed (preds_ridge - b) is the unit-w probe score;
   in reality preds used the raw non-unit W_ridge. dose_calib therefore
   computes s95/t on (preds_ridge - b_ridge) / ||W_ridge_chosen||, which IS
   the unit-w standardized score w_unit.z. Correspondingly load_arms returns
   the ridge bias as b_ridge / ||W_ridge_chosen|| so that
   s = w_unit.z + b_unit is the trained probe's output rescaled by the
   positive constant 1/||W|| (rank/threshold-order preserving, and in the
   same units as s95/t and the steering alpha).
2. glorptitude has probes but NO natscores file (it was the Stage-6 nonsense
   control, never scored on natural text). It appears in FAMILIES, but
   dose_calib raises FileNotFoundError for it with a clear message.
3. Class-name canonicalization: stage5 probes npz classes use underscores
   ('first_quarter') while stage6 natscores classes use spaces ('first
   quarter'). All stage6_1 APIs canonicalize to the UNDERSCORE form
   (FAMILIES values, load_arms cls, dose_calib json keys); inputs with
   spaces are normalized on the way in.
4. load_model sets model.config.output_hidden_states = True so hidden states
   are on by default ("hidden states on" per DESIGN); under transformers 5.x
   this config default is read at call time, and any forward can still
   override with output_hidden_states=False (recommended for E3 generation).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
STAGE_DIR = CODE_DIR.parent                          # concept_probes/4_causal
CP_DIR = STAGE_DIR.parent                            # concept_probes
PROBES_DIR = CP_DIR / "2_probes" / "probes"
NATSTATS_PATH = CP_DIR / "2_probes" / "natstats26.npz"
NATSCORES_DIR = CP_DIR / "3_validation" / "data" / "natscores"
OUT_DIR = STAGE_DIR / "out"

MODEL_NAME = "google/gemma-2-2b"
LAYERS = [1, 3, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25]
N_RAND_ARMS = 5


def _discover_families() -> dict[str, list[str]]:
    """family -> class list, read from each family's probes_l{first}.npz.

    Class names are the npz ``classes`` strings and may contain spaces
    (moon_phases); file/dir names use underscores.
    """
    fams: dict[str, list[str]] = {}
    if not PROBES_DIR.is_dir():
        return fams
    for d in sorted(PROBES_DIR.iterdir()):
        f = d / f"probes_l{LAYERS[0]}.npz"
        if d.is_dir() and f.exists():
            with np.load(f) as z:
                fams[d.name] = [str(c).replace(" ", "_") for c in z["classes"]]
    return fams


FAMILIES: dict[str, list[str]] = _discover_families()


def load_model(device: str = "cuda", dtype: str = "bfloat16"):
    """(model, tokenizer): AutoModelForCausalLM gemma-2-2b, eager attention,
    hidden states on by default (config.output_hidden_states=True; per-call
    override allowed). dtype is a torch dtype name ("bfloat16", "float32", …)
    or a torch.dtype — pass "float32" + device="cpu" for local tests.
    Gemma-2 logit softcapping stays on (native forward)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    td = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=td, attn_implementation="eager")
    model.config.output_hidden_states = True
    model.eval().to(device)
    return model, tok


def load_natstats(layer: int) -> tuple[np.ndarray, np.ndarray]:
    """(mu, sigma) fp32 [2304] for probe layer ``layer`` (block output stats;
    keys mean_{layer}/std_{layer} of natstats26.npz, layer in 0..25). There
    are NO stats for the embedding stream (layer -1) — callers doing
    embedding-level (layer=-1) interventions must supply their own."""
    with np.load(NATSTATS_PATH) as z:
        return (z[f"mean_{layer}"].astype(np.float32),
                z[f"std_{layer}"].astype(np.float32))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("zero-norm direction")
    return (np.asarray(v, dtype=np.float32) / n).astype(np.float32)


def load_arms(family: str, cls: str, layer: int) -> dict:
    """Direction arms for one (concept, layer), all unit-norm fp32 [2304] in
    STANDARDIZED space: {'ridge': (w, b), 'dom': (w, 0.0), 'lda': (w, 0.0),
    'rand': [5 unit vectors]}.

    ridge: W_ridge[chosen_lambda_ridge[ci], ci] unit-normalized; its bias is
    b_ridge[chosen_lambda_ridge[ci], ci] / ||W_ridge[...]|| so (w, b) is the
    trained affine probe rescaled by 1/||W|| (deviation #1 in module
    docstring — keeps the bias in the same unit-w score units as dose_calib).
    """
    with np.load(PROBES_DIR / family / f"probes_l{layer}.npz") as z:
        classes = [str(c).replace(" ", "_") for c in z["classes"]]
        ci = classes.index(cls.replace(" ", "_"))
        li = int(z["chosen_lambda_ridge"][ci])
        W = z["W_ridge"][li, ci]
        nrm = float(np.linalg.norm(W))
        return {
            "ridge": (_unit(W), float(z["b_ridge"][li, ci]) / nrm),
            "dom": (_unit(z["W_dom"][ci]), 0.0),
            "lda": (_unit(z["W_lda"][ci]), 0.0),
            "rand": [_unit(z["rand_dirs"][k]) for k in range(N_RAND_ARMS)],
        }


def _build_family_calib(family: str) -> dict:
    """{class: {str(layer): {'s95','t'}}} from the family's natscores +
    probes. s95 = 95th percentile and t = mean of the natural-text unit-w
    standardized ridge score (preds_ridge - b_ridge) / ||W_ridge_chosen||
    over ALL natural-pool tokens (see module docstring, deviation #1)."""
    path = NATSCORES_DIR / f"{family}.natscores.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — no Stage-6 natscores for family '{family}' "
            "(expected for glorptitude, the nonsense control); dose "
            "calibration is undefined for it.")
    with np.load(path) as nat:
        # natscores class names may contain spaces (moon_phases: 'first
        # quarter'); probes/FAMILIES use underscores ('first_quarter').
        # Canonical stage6_1 names = the probes/FAMILIES underscore form.
        nat_classes = [str(c).replace(" ", "_") for c in nat["classes"]]
        nat_layers = [int(x) for x in nat["layers"]]
        preds = nat["preds_ridge"]                       # [12, T, C] incl bias
    out: dict = {c: {} for c in nat_classes}
    for li, L in enumerate(nat_layers):
        with np.load(PROBES_DIR / family / f"probes_l{L}.npz") as z:
            pr_classes = [str(c).replace(" ", "_") for c in z["classes"]]
            for c in nat_classes:
                ci_p, ci_n = pr_classes.index(c), nat_classes.index(c)
                lam = int(z["chosen_lambda_ridge"][ci_p])
                nrm = float(np.linalg.norm(z["W_ridge"][lam, ci_p]))
                s = (preds[li, :, ci_n] - float(z["b_ridge"][lam, ci_p])) / nrm
                out[c][str(L)] = {"s95": float(np.percentile(s, 95)),
                                  "t": float(s.mean())}
    return out


def dose_calib(family: str, cls: str, layer: int) -> dict:
    """{'s95': float, 't': float} for (family, cls, layer); builds/extends the
    cache out/dose_calib.json = {family: {class: {layer: {...}}}} on first
    use per family. Raises FileNotFoundError for glorptitude (no natscores)."""
    path = OUT_DIR / "dose_calib.json"
    calib = json.loads(path.read_text()) if path.exists() else {}
    if family not in calib:
        calib[family] = _build_family_calib(family)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(calib, indent=1, sort_keys=True))
        tmp.replace(path)
    return dict(calib[family][cls.replace(" ", "_")][str(layer)])


def probe_scores(hidden, layer: int, w, b: float, mu, sigma) -> torch.Tensor:
    """Raw probe scores [B, T] fp32 from a forward's hidden_states tuple:
    s = w . (h - mu)/sigma + b with h = hidden[layer + 1] (resid_post of
    block ``layer``; layer=-1 reads hidden[0], the embedding stream). All
    math in fp32. NOTE (transformers >= 5): hidden[-1] is tied to the
    post-final-RMSNorm stream, so layer 25 reads THROUGH the final norm —
    same convention Stage-5 probes were trained with, but the exact-alpha
    steering identity does not hold at layer 25 (see interventions.py)."""
    h = hidden[layer + 1]
    dev = h.device
    w_t = torch.as_tensor(np.asarray(w), dtype=torch.float32, device=dev)
    mu_t = torch.as_tensor(np.asarray(mu), dtype=torch.float32, device=dev)
    sd_t = torch.as_tensor(np.asarray(sigma), dtype=torch.float32, device=dev)
    z = (h.to(torch.float32) - mu_t) / sd_t
    return z @ w_t + float(b)


def batch_iter(texts, tokenizer, max_tokens: int = 8192, bos: bool = True):
    """Yield (indices, input_ids [B, Lmax], attention_mask [B, Lmax]) batches.

    Tokenizes with add_special_tokens=False, prepends BOS when bos=True
    (Stage-5 convention), sorts by length and greedily packs so that
    B * Lmax <= max_tokens (a batch always holds at least one text). Padding
    uses tokenizer.pad_token_id (fallback 0) with attention_mask=0; indices
    map rows back into the input ``texts`` order."""
    enc = []
    for i, t in enumerate(texts):
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if bos:
            ids = [tokenizer.bos_token_id] + ids
        enc.append((i, ids))
    enc.sort(key=lambda kv: len(kv[1]))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    batch: list[tuple[int, list[int]]] = []
    maxlen = 0
    def emit(b, m):
        idx = [i for i, _ in b]
        ids = torch.full((len(b), m), pad, dtype=torch.long)
        attn = torch.zeros((len(b), m), dtype=torch.long)
        for r, (_, seq) in enumerate(b):
            ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[r, :len(seq)] = 1
        return idx, ids, attn

    for i, ids in enc:
        newmax = max(maxlen, len(ids))
        if batch and newmax * (len(batch) + 1) > max_tokens:
            yield emit(batch, maxlen)
            batch, maxlen = [], 0
        batch.append((i, ids))
        maxlen = max(maxlen, len(ids))
    if batch:
        yield emit(batch, maxlen)
