"""Stage 6.1 E5 — causally-salient layer & copy-vs-recompute (task.md §6.1.7).

Per concept (family.class), on judge-confirmed natural positive texts:

  (a) Erasure-propagation: for each ablation layer l in the probed LAYERS, one
      clean forward + one ablated forward (Intervention mode='ablate', ridge
      arm, t = natural-mean score from dose_calib). Read ALL probed-layer
      probe scores at their layers l' with BOTH meters (ridge = same-as-
      intervention, labeled; dom = the non-identical meter, primary) plus the
      behavioral anchor (mean log-prob of the class's diagnostic tokens at
      next-token positions, prompts/<family>.tokens.json). Output: score-
      deficit curves deficit[l, l', meter, posset].
  (b) Copy matrix: from the two hidden_states tuples alone (no extra hooks),
      decompose the readout-layer delta into the identity path vs downstream
      block recomputation. With h_i = hidden_states[i] (i = 0 embeddings,
      i = k+1 block-k output, i = n_layers+... see L25 note), the residual
      stream is a literal telescoping sum, so for ablation at l and readout
      at l' >= l:
          Δh_{l'+1} = Δh_{l+1}  (identity path = the edit as it stands at l)
                    + Σ_{s=l+2..l'+1} Δ(h_s − h_{s−1})   (downstream blocks)
      Project every part onto each readout meter's standardized direction
      (w_{l'} ⊘ σ_{l'}): identity share = num/den with
          num = w·(Δh_{l+1}/σ_{l'}),  den = w·(Δh_{l'+1}/σ_{l'}).
      ESTIMATOR (documented per spec): per example, num and den are SUMMED
      over the selected positions; C[l,l'] = weighted median over examples of
      the per-example ratio Σnum/Σden with weights |Σden| (weighting by the
      total deficit magnitude avoids divide-by-noise examples). Also stored:
      pooled ratio-of-sums C_pooled = Σ_ex num / Σ_ex den, the raw per-example
      numerator/denominator sums, and per-block downstream projection sums
      (down_by_block). Two position sets: 'all' (all non-BOS real tokens) and
      'concept' (tokens whose judged span strength >= --pos-threshold, from
      the jsonl token targets; +1 shift for the prepended BOS).
      EXACTNESS: identity + Σ per-block downstream = total, exactly, on the
      cached (fp16-rounded) states; the smoke test asserts the residual is
      < 1e-3 relative.
  (c) --frozen (top-2 salient layers per concept): rerun the ablated pass
      with every downstream block's ATTENTION-MODULE OUTPUT (self_attn output
      tensor, i.e. before the post-attn norm/residual add) replayed from the
      clean run. Freezing blocks l+1..25 at once gives the frozen-(l+1..l')
      readout at EVERY l' simultaneously (readout at l' only depends on
      blocks <= l'). NOTE this is a STRONGER freeze than freezing attention
      patterns (it freezes value/output recomputation too); documented
      deviation from McGrath-style pattern freezing. Recovery with vs without
      freezing = self-repair magnitude attributable to attention
      recomputation. Salience = |behavioral-anchor deficit| from (a);
      fallback (no tokens.json): mean |dom-meter deficit| over l' > l.
  (d) Ablate-from-l-onward vs ablate-at-l-only: onward composes one ablate
      Intervention at every PROBED layer >= l (arms/dose exist only at probed
      layers — documented interpretation of "layers l..25").
  (e) --patch ("distributional patching", documented deviation from true
      clean/corrupt activation interchange): matched positive vs neutral
      texts of the same family. Denoise-lite: on NEUTRAL text, ablate at l
      with t = mean positive-text clean score at l (concept positions).
      Erase: on POSITIVE text, ablate at l with t = mean neutral-text clean
      score at l. Metric: normalized diagnostic-logprob-margin recovery
      m = (LD_patched − LD_neg) / (LD_pos − LD_neg). This moves the
      probe-direction component to the other distribution's MEAN rather than
      interchanging per-pair values — no cross-run pairing is needed.

Data sources (verified on disk 2026-07-02):
- PRIMARY: 3_validation/data/natural/eval/<family>.jsonl — the Stage-6 tokenized
  derivative of 1_dataset/data/<family>/judged/judged_nat.jsonl (fields verified
  by reading lines: example_id, text, token_ids [gemma-2 tokenizer,
  add_special_tokens=False, <=512 tokens], targets = {class: [[tok_idx,
  strength], ...]} painted from the judge `aggregated_spans`). Positive for
  class c = max strength over c's target tokens >= --pos-threshold (default
  0.34, the §6.1.4 judge-confirmed bar). Class keys are canonicalized to
  underscores (moon_phases uses spaces in the jsonl).
- FALLBACK (natural positives < 50; affects color_wheel x4, directions x2,
  moon_phases x5): 1_dataset/data/<family>/final/mixed/<class>.val.jsonl rows
  with role == 'target_pos' (fields verified: text, token_ids,
  token_targets_sparse). Recorded in outputs as source='stage4_val'.
- Neutral texts (for --patch): natural eval rows with NO class reaching the
  threshold.
- Diagnostic tokens: prompts/<family>.tokens.json (A3 bank, schema verified:
  categorical {"classes": {cls: {"surface": [...], "associates": [...]}}},
  intensity {"poles": {"low"/"high": {"surface": ...}}}). We use SURFACE
  strings only (high pole for intensity axes); see load_diag_ids for the
  single-token rule and the multiword first-token fallback. If the file is
  missing, anchor metrics are NaN and --patch is skipped with a warning —
  smoke does not require it.

L25 note (transformers 5.x): hidden_states[26] is the POST-final-RMSNorm
stream (tied to last_hidden_state); block-25's raw output never appears in
the tuple. So (i) the layer-25 probe meter reads through the final norm,
(ii) copy-matrix entries with readout l'=25 include the final norm inside
the "block 25" downstream step (still numerically exact but the identity-
path interpretation is impure), and (iii) the ablation row l=25 has no
downstream readouts. All layer-25 rows are annotated (npz meta
`l25_post_norm`, summary config `"l25_post_norm": true`).

Forward count per concept (n_b = #example batches, n_nb = #neutral batches,
nL = #ablation layers): stage a+b+d = n_b * (1 + nL + nL); --frozen adds
n_b * (1 + 2); --patch adds n_nb * (1 + nL) + n_b * nL. With defaults
(nL = 12, ~100 examples of ~100-200 tokens => n_b ~ 2-4 at
--batch-tokens 8192) that is ~60-100 core forwards per concept.

Peak-memory math (GPU, bf16 model, B_tok = --batch-tokens, d = 2304, 27
hidden states, gemma vocab V = 256128):
  model weights                      ~5.2 GB
  logits [B_tok, V] bf16             2 * B_tok * V      = 2.1 GB @ 8192
  clean hidden cache, fp16           27 * B_tok * d * 2 = 1.0 GB @ 8192
  ablated hidden cache, fp16         same               = 1.0 GB @ 8192
  live HF output tuple (bf16)        same (transient)   = 1.0 GB @ 8192
  fp32 delta temporaries             ~3 * B_tok * d * 4 = 0.23 GB @ 8192
  anchor logsumexp chunk (fp32)      512 * V * 4        = 0.5 GB
  attn replay cache (--frozen)       26 * B_tok * d * 2 = 1.0 GB @ 8192
  => ~12 GB @ 8192 (H100-comfortable). Cached states are stored fp16;
  ALL projections/scores are computed in fp32. Default --batch-tokens 4096
  (~8 GB) is conservative; use 8192-16384 on H100.

Outputs (per DESIGN §"Metrics output schema"): out/e5/<family>.npz with keys
"<class>__<iv_arm>__<name>" (see KEY_DOC below / npz key '__doc__'), one
summary row per (concept, metric) appended to out/e5/summary.jsonl, heartbeat
lines to out/progress_e5_propagation.log. Nothing is committed.

CLI:
  python e5_propagation.py --families months --classes january \
      --n-examples 100 --batch-tokens 8192 --device cuda --frozen --patch
  python e5_propagation.py --smoke          # tiny random Gemma2, CPU, asserts
  python e5_propagation.py --dry-run        # real data, plan + counts only

Deviations (beyond those flagged above): (1) batching packs the PRE-TOKENIZED
token_ids from the source jsonl directly (same sort-by-length greedy packing
as common.batch_iter, BOS prepended) instead of re-tokenizing text through
batch_iter — avoids any retokenization drift vs the judged span indices;
(2) intervention arm 'dom' (optional via --arms) has no natscores-based t, so
t/t95 are computed from the clean pass over this concept's positive examples
(extra clean prepass) and recorded; default --arms ridge needs no prepass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from common import (CP_DIR, FAMILIES, LAYERS, STAGE_DIR, dose_calib,  # noqa: E402
                    load_arms, load_model, load_natstats, probe_scores)
from interventions import Hooks, Intervention                         # noqa: E402

NATEVAL_DIR = CP_DIR / "3_validation" / "data" / "natural" / "eval"
STAGE4_DIR = CP_DIR / "1_dataset"
PROMPTS_DIR = STAGE_DIR / "prompts"

METERS = ("ridge", "dom")           # meter order (axis 'meter' everywhere)
POSSETS = ("all", "concept")        # position-set order (axis 'posset')
MIN_NATURAL = 50                    # below this, fall back to stage4 val

KEY_DOC = """npz keys, per '<class>__<iv_arm>__' prefix (axes: L = ablation
layer index into 'layers'; R = readout layer index; M = meter [ridge, dom];
P = posset [all, concept]; N = example index):
  deficit        [L,R,M,P]  mean over examples of per-example mean
                            (score_ablated - score_clean) at readout R
  deficit_ex     [N,L,R,M,P] per-example deficits (CI material)
  onward_deficit / onward_deficit_ex   same, ablate-from-l-onward
  clean_score    [R,M,P]    mean clean probe score
  C              [L,R,M,P]  identity share, weighted median over examples of
                            per-example (num_sum/den_sum), weights |den_sum|;
                            NaN for R<L
  C_pooled       [L,R,M,P]  pooled ratio of sums
  C_num_ex/C_den_ex [N,L,R,M,P] raw per-example numerator/denominator sums
  down_by_block  [L,S,R,M,P] pooled per-block downstream projection sums,
                            S = hidden-tuple step index (block s-1; the last
                            step includes the final RMSNorm — L25 note)
  copy_resid_relmax scalar  max relative telescoping residual (QC; smoke
                            asserts < 1e-3)
  anchor_clean_ex [N], anchor_abl_ex [L,N], anchor_onward_ex [L,N]
                            behavioral anchor per example (NaN if no
                            tokens.json)
  salience       [L]        salience score per ablation layer
  frozen_layers  [2], frozen_deficit [2,R,M,P], frozen_deficit_ex
                 [2,N,R,M,P], frozen_anchor_ex [2,N]   (--frozen)
  patch_m_denoise/patch_m_erase [L], patch_t_pos/patch_t_neu [L],
  patch_LD_pos_ex/patch_LD_neg_ex [N...], patch_LD_denoise_ex [L,Nn],
  patch_LD_erase_ex [L,N]                                (--patch)
  t [L], n_examples, source, layers [L], meters, possets, l25_post_norm
"""


# --------------------------------------------------------------------- utils
def canon(name: str) -> str:
    return str(name).replace(" ", "_")


def heartbeat(log: Path, msg: str):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} e5_propagation {msg}\n")


def masked_mean_per_ex(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x [B,T] fp32, mask [B,T] bool -> [B] mean over masked positions,
    NaN where a row has no masked positions."""
    m = mask.to(x.dtype)
    c = m.sum(1)
    s = (x * m).sum(1)
    return torch.where(c > 0, s / c.clamp(min=1), torch.full_like(s, np.nan))


def weighted_median(vals: np.ndarray, wts: np.ndarray) -> float:
    ok = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    if not ok.any():
        return float("nan")
    v, w = vals[ok], wts[ok]
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    return float(v[min(np.searchsorted(cw, 0.5 * cw[-1]), len(v) - 1)])


def boot_ci(num: np.ndarray, den: np.ndarray, n_boot=1000, seed=0):
    """95% CI of pooled ratio sum(num)/sum(den) by bootstrap over examples."""
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    if len(num) < 3 or abs(den.sum()) < 1e-12:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].sum(1) / np.where(np.abs(den[idx].sum(1)) < 1e-12, np.nan,
                                   den[idx].sum(1))
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return (float(lo), float(hi))


def mean_ci(x: np.ndarray):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(x.mean())
    if len(x) < 3:
        return m, float("nan"), float("nan"), len(x)
    sem = float(x.std(ddof=1) / np.sqrt(len(x)))
    return m, m - 1.96 * sem, m + 1.96 * sem, len(x)


# --------------------------------------------------------------- data loading
_EVAL_CACHE: dict[str, list[dict]] = {}


def load_eval_rows(family: str) -> list[dict]:
    """stage6 natural eval rows with target class keys canonicalized."""
    if family not in _EVAL_CACHE:
        path = NATEVAL_DIR / f"{family}.jsonl"
        rows = []
        if path.exists():
            for line in open(path):
                r = json.loads(line)
                r["targets"] = {canon(k): v for k, v in r["targets"].items()}
                rows.append(r)
        _EVAL_CACHE[family] = rows
    return _EVAL_CACHE[family]


def _spanned(tv: list, thr: float) -> tuple[float, list[int]]:
    """token targets [[idx, strength],...] -> (max strength, concept idxs)."""
    if not tv:
        return 0.0, []
    mx = max(s for _, s in tv)
    cpos = [int(i) for i, s in tv if s >= thr] or \
        [int(i) for i, s in tv if s > 0]
    return mx, cpos


def select_positives(family: str, cls: str, n: int, thr: float,
                     seed: int) -> tuple[list[dict], str]:
    """-> (examples, source). example = {example_id, ids (no BOS), cpos}.
    Natural (stage6 eval jsonl, max span strength >= thr); if < MIN_NATURAL,
    fall back entirely to stage4 <cls>.val.jsonl role=target_pos."""
    nat = []
    for r in load_eval_rows(family):
        mx, cpos = _spanned(r["targets"].get(cls) or [], thr)
        if mx >= thr:
            nat.append(dict(example_id=r["example_id"], ids=r["token_ids"],
                            cpos=cpos))
    source = "natural"
    rows = nat
    if len(nat) < MIN_NATURAL:
        source = "stage4_val"
        rows = []
        path = STAGE4_DIR / "data" / family / "final" / "mixed" / \
            f"{cls}.val.jsonl"
        if path.exists():
            for line in open(path):
                r = json.loads(line)
                if r.get("role") != "target_pos":
                    continue
                _, cpos = _spanned(r.get("token_targets_sparse") or [], thr)
                rows.append(dict(example_id=r["example_id"],
                                 ids=r["token_ids"], cpos=cpos))
        if not rows:                      # last resort: the thin natural set
            source, rows = "natural_thin", nat
    rows.sort(key=lambda r: r["example_id"])
    rng = np.random.default_rng(
        seed + zlib.crc32(f"{family}.{cls}".encode()) % (2**31))
    if len(rows) > n:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), n,
                                                   replace=False))]
    return rows, source


def select_neutrals(family: str, n: int, thr: float, seed: int) -> list[dict]:
    """Natural eval rows where NO class reaches thr (family-neutral)."""
    rows = []
    for r in load_eval_rows(family):
        if all(_spanned(tv, thr)[0] < thr for tv in r["targets"].values()):
            rows.append(dict(example_id=r["example_id"], ids=r["token_ids"],
                             cpos=[]))
    rows.sort(key=lambda r: r["example_id"])
    rng = np.random.default_rng(
        seed + zlib.crc32(f"{family}.__neutral__".encode()) % (2**31))
    if len(rows) > n:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), n,
                                                   replace=False))]
    return rows


def load_diag_ids(family: str, cls: str, tok) -> Optional[list[int]]:
    """prompts/<family>.tokens.json -> diagnostic token ids (or None).

    A3 bank schema (verified 2026-07-02): categorical families have
    {"classes": {cls: {"surface": [...], "associates": [...]}}}; intensity
    axes have {"poles": {"low"/"high": {"surface": ...}}}. Diagnostic set =
    SURFACE strings only (associates are looser and may bleed across
    concepts); for intensity axes the HIGH pole is used (concept-expressed
    direction; note lovingness low = despise pole). Each surface is
    tokenized with and without a leading space and single-token encodings
    are kept; if NO surface is single-token (multiword classes, e.g.
    moon_phases "new moon"), we fall back to the FIRST token of the
    leading-space variant of each surface — weaker, documented limitation
    (deficits still difference out the clean baseline)."""
    path = PROMPTS_DIR / f"{family}.tokens.json"
    if not path.exists() or tok is None:
        return None
    j = json.loads(path.read_text())
    if "classes" in j:
        entries = (j["classes"].get(cls)
                   or j["classes"].get(cls.replace("_", " ")))
    elif "poles" in j:
        entries = j["poles"].get("high")
    else:
        entries = None
    if entries is None:
        return None
    surfaces = [str(s) for s in (entries.get("surface") or [])] \
        if isinstance(entries, dict) else [str(s) for s in entries]
    ids: set[int] = set()
    for s in surfaces:
        for v in (s, " " + s):
            e = tok(v, add_special_tokens=False)["input_ids"]
            if len(e) == 1:
                ids.add(int(e[0]))
    if not ids:                              # multiword fallback: first token
        for s in surfaces:
            e = tok(" " + s, add_special_tokens=False)["input_ids"]
            if e:
                ids.add(int(e[0]))
    return sorted(ids) or None


# ------------------------------------------------------------------ batching
@dataclass
class Batch:
    ex_idx: list[int]
    ids: torch.Tensor        # [B, L] long
    attn: torch.Tensor       # [B, L] long
    cmask: torch.Tensor      # [B, L] bool (concept positions, BOS-shifted)


def pack_batches(examples: list[dict], bos_id: int, pad_id: int,
                 max_tokens: int) -> list[Batch]:
    """Pre-tokenized analog of common.batch_iter: BOS prepended, sorted by
    length, greedy pack with B*Lmax <= max_tokens; examples longer than
    max_tokens-1 are truncated (cpos filtered)."""
    enc = []
    for i, ex in enumerate(examples):
        ids = list(ex["ids"])[: max_tokens - 1]
        cpos = [p + 1 for p in ex["cpos"] if p + 1 < len(ids) + 1]
        enc.append((i, [bos_id] + ids, cpos))
    enc.sort(key=lambda t: len(t[1]))

    out: list[Batch] = []

    def emit(group):
        m = max(len(s) for _, s, _ in group)
        ids = torch.full((len(group), m), pad_id, dtype=torch.long)
        attn = torch.zeros((len(group), m), dtype=torch.long)
        cm = torch.zeros((len(group), m), dtype=torch.bool)
        idx = []
        for r, (i, seq, cpos) in enumerate(group):
            ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[r, :len(seq)] = 1
            if cpos:
                cm[r, torch.tensor(cpos, dtype=torch.long)] = True
            idx.append(i)
        out.append(Batch(idx, ids, attn, cm))

    group: list = []
    maxlen = 0
    for i, seq, cpos in enc:
        newmax = max(maxlen, len(seq))
        if group and newmax * (len(group) + 1) > max_tokens:
            emit(group)
            group, maxlen = [], 0
        group.append((i, seq, cpos))
        maxlen = max(maxlen, len(seq))
    if group:
        emit(group)
    return out


# ----------------------------------------------------------- meters / anchor
@dataclass
class ConceptCtx:
    family: str
    cls: str
    iv_arm: str
    layers: list[int]
    arms: dict            # L -> {'ridge': (w, b), 'dom': (w, 0.0)}
    natstats: dict        # L -> (mu, sigma) fp32 [d]
    t: dict               # L -> ablation target (natural mean, iv arm units)
    examples: list[dict]
    source: str
    neutrals: list[dict] = field(default_factory=list)
    diag_ids: Optional[list[int]] = None
    n_layers_model: int = 26


def meter_matrix(ctx: ConceptCtx, device) -> torch.Tensor:
    """[nL*len(METERS), d] rows = w_{l',arm} / sigma_{l'} (fp32), row order
    j*len(METERS)+m for readout j, meter m — the standardized readout
    directions the copy-matrix parts are projected onto."""
    rows = []
    for L in ctx.layers:
        _, sd = ctx.natstats[L]
        for arm in METERS:
            w, _ = ctx.arms[L][arm]
            rows.append(np.asarray(w, np.float32) / np.asarray(sd, np.float32))
    return torch.tensor(np.stack(rows), dtype=torch.float32, device=device)


def meter_scores(hidden, ctx: ConceptCtx) -> dict:
    """{(L, arm): [B,T] fp32} probe scores via common.probe_scores."""
    s = {}
    for L in ctx.layers:
        mu, sd = ctx.natstats[L]
        for arm in METERS:
            w, b = ctx.arms[L][arm]
            s[(L, arm)] = probe_scores(hidden, L, w, b, mu, sd)
    return s


def anchor_per_example(logits: torch.Tensor, attn: torch.Tensor,
                       diag_ids: Optional[list[int]],
                       chunk: int = 512) -> torch.Tensor:
    """[B] mean over next-token positions of log P(diagnostic-token set)
    = logsumexp(diag logits) - logsumexp(all logits), fp32, chunked so the
    fp32 logsumexp never materializes more than `chunk` x vocab."""
    B, T, V = logits.shape
    if diag_ids is None or T < 2:
        return torch.full((B,), np.nan)
    valid = (attn[:, :-1] * attn[:, 1:]).bool()      # predict a real token
    flat = logits[:, :-1, :].reshape(-1, V)
    idx = torch.as_tensor(diag_ids, device=logits.device)
    vals = torch.empty(flat.shape[0], dtype=torch.float32,
                       device=logits.device)
    for i in range(0, flat.shape[0], chunk):
        f = flat[i:i + chunk].to(torch.float32)
        vals[i:i + chunk] = (torch.logsumexp(f[:, idx], -1)
                             - torch.logsumexp(f, -1))
    return masked_mean_per_ex(vals.view(B, T - 1), valid).cpu()


# ------------------------------------------------------------ attention hooks
class AttnCapture:
    """Capture clean self_attn OUTPUTS (handles tuple and plain-tensor module
    returns). Registered WITHOUT prepend (appended), so captured values are
    post-any-prepended-intervention — per harness convention."""

    def __init__(self, model, block_ids: list[int]):
        self.base = model.model if hasattr(model, "model") else model
        self.block_ids = block_ids
        self.store: dict[int, torch.Tensor] = {}
        self._handles = []

    def _mk(self, k):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            self.store[k] = hs.detach()
        return hook

    def __enter__(self):
        for k in self.block_ids:
            self._handles.append(
                self.base.layers[k].self_attn.register_forward_hook(
                    self._mk(k)))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


class AttnReplay:
    """Replace self_attn outputs of the given blocks with stored clean values
    (a stronger freeze than freezing attention patterns — see module doc)."""

    def __init__(self, model, store: dict[int, torch.Tensor]):
        self.base = model.model if hasattr(model, "model") else model
        self.store = store
        self._handles = []

    def _mk(self, k):
        def hook(module, args, output):
            clean = self.store[k]
            if isinstance(output, tuple):
                return (clean.to(output[0].dtype),) + tuple(output[1:])
            return clean.to(output.dtype)
        return hook

    def __enter__(self):
        for k in self.store:
            self._handles.append(
                self.base.layers[k].self_attn.register_forward_hook(
                    self._mk(k)))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


# ----------------------------------------------------------- per-concept run
def _fwd(model, batch: Batch, device):
    with torch.no_grad():
        return model(batch.ids.to(device),
                     attention_mask=batch.attn.to(device),
                     output_hidden_states=True, use_cache=False)


def _possets(batch: Batch, device):
    valid = batch.attn.bool().to(device)
    valid[:, 0] = False                      # exclude BOS from score possets
    return {"all": valid, "concept": batch.cmask.to(device) & valid}


def run_concept(model, ctx: ConceptCtx, opts, log: Path) -> dict:
    device = opts.device
    nL = len(ctx.layers)
    nM, nP = len(METERS), len(POSSETS)
    n = len(ctx.examples)
    layers = ctx.layers
    max_read_hid = max(layers) + 1           # largest hidden index we touch

    bos = opts.bos_id
    pad = opts.pad_id
    batches = pack_batches(ctx.examples, bos, pad, opts.batch_tokens)
    Mproj = meter_matrix(ctx, device)        # [nL*nM, d]

    res = dict(
        deficit_ex=np.full((n, nL, nL, nM, nP), np.nan, np.float32),
        onward_deficit_ex=np.full((n, nL, nL, nM, nP), np.nan, np.float32),
        clean_ex=np.full((n, nL, nM, nP), np.nan, np.float32),
        C_num_ex=np.full((n, nL, nL, nM, nP), np.nan, np.float32),
        C_den_ex=np.full((n, nL, nL, nM, nP), np.nan, np.float32),
        down_by_block=np.zeros((nL, max_read_hid + 1, nL, nM, nP),
                               np.float32),
        anchor_clean_ex=np.full(n, np.nan, np.float32),
        anchor_abl_ex=np.full((nL, n), np.nan, np.float32),
        anchor_onward_ex=np.full((nL, n), np.nan, np.float32),
        copy_resid_relmax=0.0,
    )

    mu_sigma_all = {L: ctx.natstats[L] for L in layers}
    iv_w = {L: ctx.arms[L][ctx.iv_arm][0] for L in layers}

    pbar = tqdm(total=len(batches) * (1 + 2 * nL),
                desc=f"E5 {ctx.family}.{ctx.cls}", leave=False)
    for bi, batch in enumerate(batches):
        masks = _possets(batch, device)
        rows = batch.ex_idx

        # ---- clean forward -------------------------------------------------
        out_c = _fwd(model, batch, device)
        clean16 = [h.to(torch.float16) for h in out_c.hidden_states]
        s_clean = meter_scores(out_c.hidden_states, ctx)
        a_clean = anchor_per_example(out_c.logits, batch.attn.to(device),
                                     ctx.diag_ids)
        res["anchor_clean_ex"][rows] = a_clean.numpy()
        for j, L in enumerate(layers):
            for m, arm in enumerate(METERS):
                for p, ps in enumerate(POSSETS):
                    v = masked_mean_per_ex(s_clean[(L, arm)], masks[ps])
                    res["clean_ex"][rows, j, m, p] = v.cpu().numpy()
        del out_c
        pbar.update(1)

        # ---- per-ablation-layer runs: propagation + copy matrix -----------
        for li, l in enumerate(layers):
            iv = Intervention(l, iv_w[l], "ablate", t=ctx.t[l])
            with Hooks(model, [iv], {l: ctx.natstats[l]}):
                out_a = _fwd(model, batch, device)
            abl16 = [h.to(torch.float16) for h in out_a.hidden_states]
            s_abl = meter_scores(out_a.hidden_states, ctx)
            a_abl = anchor_per_example(out_a.logits, batch.attn.to(device),
                                       ctx.diag_ids)
            res["anchor_abl_ex"][li, rows] = a_abl.numpy()
            for j, L in enumerate(layers):
                for m, arm in enumerate(METERS):
                    d = s_abl[(L, arm)] - s_clean[(L, arm)]
                    for p, ps in enumerate(POSSETS):
                        v = masked_mean_per_ex(d, masks[ps])
                        res["deficit_ex"][rows, li, j, m, p] = v.cpu().numpy()
            del out_a

            # copy matrix from the two cached hidden tuples (exact telescope)
            d_edit = (abl16[l + 1].float() - clean16[l + 1].float())
            P_edit = d_edit @ Mproj.T                          # [B,T,nL*nM]
            del d_edit
            P_tot = torch.zeros_like(P_edit)
            for j, L in enumerate(layers):
                if L < l:
                    continue
                cols = slice(j * nM, j * nM + nM)
                d_tot = (abl16[L + 1].float() - clean16[L + 1].float())
                P_tot[..., cols] = d_tot @ Mproj[cols].T
                del d_tot
            # per-block downstream cumulative + snapshots at probed readouts
            R = torch.zeros_like(P_edit)
            resid_max = res["copy_resid_relmax"]
            snap_at = {L + 1: j for j, L in enumerate(layers) if L > l}
            for s in range(l + 2, max_read_hid + 1):
                Ds = ((abl16[s].float() - abl16[s - 1].float())
                      - (clean16[s].float() - clean16[s - 1].float()))
                Ps = Ds @ Mproj.T
                del Ds
                R = R + Ps
                for p, ps in enumerate(POSSETS):
                    mm = masks[ps].unsqueeze(-1).to(Ps.dtype)
                    res["down_by_block"][li, s - 1, :, :, p] += (
                        (Ps * mm).sum((0, 1)).reshape(nL, nM).cpu().numpy())
                del Ps
                if s in snap_at:                    # exactness residual (QC)
                    j = snap_at[s]
                    cols = slice(j * nM, j * nM + nM)
                    resid = (P_edit[..., cols] + R[..., cols]
                             - P_tot[..., cols])
                    scale = P_tot[..., cols].abs().max().item()
                    resid_max = max(resid_max,
                                    resid.abs().max().item()
                                    / (scale + 1e-9))
            res["copy_resid_relmax"] = resid_max
            del R
            # per-example numerator/denominator sums over possets
            for j, L in enumerate(layers):
                if L < l:
                    continue
                cols = slice(j * nM, j * nM + nM)
                for p, ps in enumerate(POSSETS):
                    mm = masks[ps].unsqueeze(-1).to(P_edit.dtype)
                    num = (P_edit[..., cols] * mm).sum(1)      # [B, nM]
                    den = (P_tot[..., cols] * mm).sum(1)
                    res["C_num_ex"][rows, li, j, :, p] = num.cpu().numpy()
                    res["C_den_ex"][rows, li, j, :, p] = den.cpu().numpy()
            del P_edit, P_tot, abl16
            pbar.update(1)

        # ---- ablate-from-l-onward ------------------------------------------
        for li, l in enumerate(layers):
            ivs = [Intervention(L, iv_w[L], "ablate", t=ctx.t[L])
                   for L in layers if L >= l]
            with Hooks(model, ivs, mu_sigma_all):
                out_o = _fwd(model, batch, device)
            s_on = meter_scores(out_o.hidden_states, ctx)
            a_on = anchor_per_example(out_o.logits, batch.attn.to(device),
                                      ctx.diag_ids)
            res["anchor_onward_ex"][li, rows] = a_on.numpy()
            for j, L in enumerate(layers):
                for m, arm in enumerate(METERS):
                    d = s_on[(L, arm)] - s_clean[(L, arm)]
                    for p, ps in enumerate(POSSETS):
                        v = masked_mean_per_ex(d, masks[ps])
                        res["onward_deficit_ex"][rows, li, j, m, p] = \
                            v.cpu().numpy()
            del out_o
            pbar.update(1)

        del clean16, s_clean
        heartbeat(log, f"{ctx.family}.{ctx.cls} [{ctx.iv_arm}] "
                       f"batch {bi + 1}/{len(batches)}")
    pbar.close()

    # ------------------------------------------------- summaries of stage a/b
    res["deficit"] = np.nanmean(res["deficit_ex"], axis=0)
    res["onward_deficit"] = np.nanmean(res["onward_deficit_ex"], axis=0)
    res["clean_score"] = np.nanmean(res["clean_ex"], axis=0)

    C = np.full((nL, nL, nM, nP), np.nan, np.float32)
    Cp = np.full((nL, nL, nM, nP), np.nan, np.float32)
    for li in range(nL):
        for j in range(nL):
            if layers[j] < layers[li]:
                continue
            for m in range(nM):
                for p in range(nP):
                    num = res["C_num_ex"][:, li, j, m, p]
                    den = res["C_den_ex"][:, li, j, m, p]
                    ok = np.isfinite(num) & np.isfinite(den)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.where(np.abs(den) > 1e-12, num / den,
                                         np.nan)
                    C[li, j, m, p] = weighted_median(ratio[ok],
                                                     np.abs(den[ok]))
                    dsum = den[ok].sum()
                    if abs(dsum) > 1e-12:
                        Cp[li, j, m, p] = num[ok].sum() / dsum
    res["C"], res["C_pooled"] = C, Cp

    # ---------------------------------------------------------- salience
    a_def = np.nanmean(res["anchor_abl_ex"]
                       - res["anchor_clean_ex"][None, :], axis=1)   # [nL]
    if np.isfinite(a_def).any():
        salience = np.abs(a_def)
        crit = "abs_anchor_deficit"
    else:
        m_dom = METERS.index("dom")
        sal = np.zeros(nL)
        for li in range(nL):
            later = [j for j in range(nL) if layers[j] > layers[li]]
            sal[li] = (np.abs(res["deficit"][li, later, m_dom, 0]).mean()
                       if later else np.abs(res["deficit"][li, li, m_dom, 0]))
        salience, crit = sal, "mean_abs_dom_deficit_later_layers"
    res["salience"], res["salience_criterion"] = salience, crit
    res["anchor_deficit"] = a_def

    # ------------------------------------------------------- frozen control
    if opts.frozen:
        top2 = list(np.argsort(-np.nan_to_num(salience))[:2])
        res["frozen_layers"] = np.array([layers[i] for i in top2])
        res["frozen_deficit_ex"] = np.full((2, n, nL, nM, nP), np.nan,
                                           np.float32)
        res["frozen_anchor_ex"] = np.full((2, n), np.nan, np.float32)
        nblocks = ctx.n_layers_model
        for batch in tqdm(batches, desc="frozen", leave=False):
            masks = _possets(batch, device)
            rows = batch.ex_idx
            with AttnCapture(model, list(range(min(layers[i] for i in top2)
                                               + 1, nblocks))) as cap:
                out_c = _fwd(model, batch, device)
            s_clean = meter_scores(out_c.hidden_states, ctx)
            del out_c
            for r2, li in enumerate(top2):
                l = layers[li]
                store = {k: v for k, v in cap.store.items() if k > l}
                iv = Intervention(l, iv_w[l], "ablate", t=ctx.t[l])
                with Hooks(model, [iv], {l: ctx.natstats[l]}), \
                        AttnReplay(model, store):
                    out_f = _fwd(model, batch, device)
                a_f = anchor_per_example(out_f.logits,
                                         batch.attn.to(device), ctx.diag_ids)
                res["frozen_anchor_ex"][r2, rows] = a_f.numpy()
                s_f = meter_scores(out_f.hidden_states, ctx)
                for j, L in enumerate(layers):
                    for m, arm in enumerate(METERS):
                        d = s_f[(L, arm)] - s_clean[(L, arm)]
                        for p, ps in enumerate(POSSETS):
                            v = masked_mean_per_ex(d, masks[ps])
                            res["frozen_deficit_ex"][r2, rows, j, m, p] = \
                                v.cpu().numpy()
                del out_f
            heartbeat(log, f"{ctx.family}.{ctx.cls} [frozen] batch done")
        res["frozen_deficit"] = np.nanmean(res["frozen_deficit_ex"], axis=1)

    # ---------------------------------------------------------------- patch
    if opts.patch:
        if ctx.diag_ids is None:
            print(f"  [patch] {ctx.family}.{ctx.cls}: no diagnostic tokens — "
                  "skipping (distributional patching needs the logprob "
                  "margin)")
        elif not ctx.neutrals:
            print(f"  [patch] {ctx.family}.{ctx.cls}: no neutral texts — "
                  "skipping")
        else:
            _run_patch(model, ctx, opts, res, batches, log)

    return res


def _run_patch(model, ctx: ConceptCtx, opts, res: dict,
               pos_batches: list[Batch], log: Path):
    """(e) distributional patching — see module docstring."""
    device = opts.device
    nL = len(ctx.layers)
    layers = ctx.layers
    nn = len(ctx.neutrals)
    n = len(ctx.examples)
    m_ridge = METERS.index(ctx.iv_arm if ctx.iv_arm in METERS else "ridge")
    p_con, p_all = POSSETS.index("concept"), POSSETS.index("all")
    iv_w = {L: ctx.arms[L][ctx.iv_arm][0] for L in layers}

    # t_pos[L]: mean clean iv-arm score on positive CONCEPT positions
    # (fallback: all positions) — from stage (a) clean pass
    t_pos = np.zeros(nL, np.float32)
    for j in range(nL):
        v = res["clean_ex"][:, j, m_ridge, p_con]
        if not np.isfinite(v).any():
            v = res["clean_ex"][:, j, m_ridge, p_all]
        t_pos[j] = np.nanmean(v)

    neu_batches = pack_batches(ctx.neutrals, opts.bos_id, opts.pad_id,
                               opts.batch_tokens)
    LD_neg = np.full(nn, np.nan, np.float32)
    t_neu_sum = np.zeros(nL)
    t_neu_cnt = np.zeros(nL)
    clean_neu_scores = []                      # (batch rows, per-L means)
    for batch in tqdm(neu_batches, desc="patch/neutral-clean", leave=False):
        masks = _possets(batch, device)
        out = _fwd(model, batch, device)
        LD_neg[batch.ex_idx] = anchor_per_example(
            out.logits, batch.attn.to(device), ctx.diag_ids).numpy()
        s = meter_scores(out.hidden_states, ctx)
        for j, L in enumerate(layers):
            v = masked_mean_per_ex(s[(L, METERS[m_ridge])],
                                   masks["all"]).cpu().numpy()
            ok = np.isfinite(v)
            t_neu_sum[j] += v[ok].sum()
            t_neu_cnt[j] += ok.sum()
        del out
    t_neu = (t_neu_sum / np.maximum(t_neu_cnt, 1)).astype(np.float32)

    LD_den = np.full((nL, nn), np.nan, np.float32)   # denoise-lite (neutral)
    LD_er = np.full((nL, n), np.nan, np.float32)     # erase (positive)
    for j, L in enumerate(tqdm(layers, desc="patch/interventions",
                               leave=False)):
        iv_d = Intervention(L, iv_w[L], "ablate", t=float(t_pos[j]))
        for batch in neu_batches:
            with Hooks(model, [iv_d], {L: ctx.natstats[L]}):
                out = _fwd(model, batch, device)
            LD_den[j, batch.ex_idx] = anchor_per_example(
                out.logits, batch.attn.to(device), ctx.diag_ids).numpy()
            del out
        iv_e = Intervention(L, iv_w[L], "ablate", t=float(t_neu[j]))
        for batch in pos_batches:
            with Hooks(model, [iv_e], {L: ctx.natstats[L]}):
                out = _fwd(model, batch, device)
            LD_er[j, batch.ex_idx] = anchor_per_example(
                out.logits, batch.attn.to(device), ctx.diag_ids).numpy()
            del out
        heartbeat(log, f"{ctx.family}.{ctx.cls} [patch] layer {L}")

    LD_pos = res["anchor_clean_ex"]
    span = np.nanmean(LD_pos) - np.nanmean(LD_neg)
    if abs(span) < 1e-9 or not np.isfinite(span):
        span = np.nan
    res.update(
        patch_t_pos=t_pos, patch_t_neu=t_neu,
        patch_LD_pos_ex=LD_pos.copy(), patch_LD_neg_ex=LD_neg,
        patch_LD_denoise_ex=LD_den, patch_LD_erase_ex=LD_er,
        patch_m_denoise=((np.nanmean(LD_den, 1) - np.nanmean(LD_neg)) / span
                         ).astype(np.float32),
        patch_m_erase=((np.nanmean(LD_er, 1) - np.nanmean(LD_neg)) / span
                       ).astype(np.float32),
    )


# ------------------------------------------------------------------- outputs
def save_family(out_dir: Path, family: str, payload: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{family}.npz"
    tmp = out_dir / f".{family}.tmp.npz"    # keep .npz: savez appends it
    np.savez(tmp, **payload)
    tmp.replace(path)


def concept_payload(ctx: ConceptCtx, res: dict) -> dict:
    pre = f"{ctx.cls}__{ctx.iv_arm}__"
    keep = ["deficit", "deficit_ex", "onward_deficit", "onward_deficit_ex",
            "clean_score", "C", "C_pooled", "C_num_ex", "C_den_ex",
            "down_by_block", "copy_resid_relmax", "anchor_clean_ex",
            "anchor_abl_ex", "anchor_onward_ex", "salience",
            "anchor_deficit", "frozen_layers", "frozen_deficit",
            "frozen_deficit_ex", "frozen_anchor_ex", "patch_m_denoise",
            "patch_m_erase", "patch_t_pos", "patch_t_neu", "patch_LD_pos_ex",
            "patch_LD_neg_ex", "patch_LD_denoise_ex", "patch_LD_erase_ex"]
    out = {pre + k: np.asarray(res[k]) for k in keep if k in res}
    out[pre + "layers"] = np.array(ctx.layers)
    out[pre + "t"] = np.array([ctx.t[L] for L in ctx.layers], np.float32)
    out[pre + "n_examples"] = np.array(len(ctx.examples))
    out[pre + "source"] = np.array(ctx.source)
    out[pre + "salience_criterion"] = np.array(
        res.get("salience_criterion", ""))
    out[pre + "l25_post_norm"] = np.array(25 in ctx.layers)
    return out


def summary_rows(ctx: ConceptCtx, res: dict, opts) -> list[dict]:
    layers = ctx.layers
    nL = len(layers)
    m_dom = METERS.index("dom")
    p_all = POSSETS.index("all")
    sal = res["salience"]
    li = int(np.nanargmax(np.nan_to_num(sal, nan=-np.inf))) \
        if np.isfinite(sal).any() else nL // 2
    l_sal = layers[li]
    base_cfg = dict(iv_arm=ctx.iv_arm, source=ctx.source,
                    n_examples=len(ctx.examples),
                    pos_threshold=opts.pos_threshold,
                    batch_tokens=opts.batch_tokens,
                    salience_criterion=res.get("salience_criterion"),
                    l25_post_norm=bool(25 in layers),
                    estimator=("C = weighted median over examples of "
                               "per-example sum-ratio, weights |den sum|"))

    def row(metric, layer, value, n=len(ctx.examples), lo=None, hi=None,
            extra=None):
        cfg = dict(base_cfg)
        if extra:
            cfg.update(extra)
        return {"concept": ctx.cls, "family": ctx.family, "layer": layer,
                "arm": ctx.iv_arm, "metric": metric,
                "value": None if value is None or not np.isfinite(value)
                else float(value),
                "n": int(n), "ci_low": lo, "ci_high": hi, "config": cfg}

    rows = [row("salient_layer", l_sal, float(l_sal))]

    a = res["anchor_abl_ex"][li] - res["anchor_clean_ex"]
    m, lo, hi, k = mean_ci(a)
    rows.append(row("anchor_deficit_salient", l_sal, m, k, lo, hi))
    g = res["anchor_onward_ex"][li] - res["anchor_abl_ex"][li]
    m, lo, hi, k = mean_ci(g)
    rows.append(row("self_repair_gap_anchor", l_sal, m, k, lo, hi,
                    {"def": "onward_anchor - single_layer_anchor"}))

    later = [j for j in range(nL) if layers[j] > l_sal]
    if later:
        jn = later[0]
        num = res["C_num_ex"][:, li, jn, m_dom, p_all]
        den = res["C_den_ex"][:, li, jn, m_dom, p_all]
        lo, hi = boot_ci(num, den)
        rows.append(row("copy_identity_share_next", l_sal,
                        res["C"][li, jn, m_dom, p_all],
                        np.isfinite(num).sum(), lo, hi,
                        {"readout_layer": layers[jn], "meter": "dom",
                         "posset": "all"}))
        rows.append(row("copy_identity_share_mean_later", l_sal,
                        np.nanmean(res["C"][li, later, m_dom, p_all]),
                        extra={"meter": "dom", "posset": "all"}))
    if "frozen_deficit" in res:
        for r2, L in enumerate(res["frozen_layers"]):
            lif = layers.index(int(L))
            lat = [j for j in range(nL) if layers[j] > L]
            if not lat:
                continue
            un = np.abs(res["deficit"][lif, lat, m_dom, p_all]).mean()
            fr = np.abs(res["frozen_deficit"][r2, lat, m_dom, p_all]).mean()
            rows.append(row("attn_self_repair_dom", int(L), fr - un,
                            extra={"def": "|frozen deficit| - |deficit|, "
                                          "mean over later probed layers"}))
    if "patch_m_denoise" in res:
        md, me = res["patch_m_denoise"], res["patch_m_erase"]
        if np.isfinite(md).any():
            jw = int(np.nanargmax(md))
            rows.append(row("write_layer_denoise", layers[jw],
                            float(layers[jw]),
                            extra={"m_denoise": float(md[jw])}))
        if np.isfinite(me).any():
            hit = [j for j in range(nL) if np.isfinite(me[j])
                   and me[j] <= 0.5]
            if hit:
                rows.append(row("read_window_last_erase", layers[hit[-1]],
                                float(layers[hit[-1]]),
                                extra={"criterion": "m_erase <= 0.5"}))
    return rows


# ---------------------------------------------------------------- real setup
def build_ctx(family: str, cls: str, iv_arm: str, layers: list[int],
              opts, tok) -> Optional[ConceptCtx]:
    natstats = {L: load_natstats(L) for L in layers}
    arms = {L: load_arms(family, cls, L) for L in layers}
    try:
        calib = {L: dose_calib(family, cls, L) for L in layers}
    except FileNotFoundError as e:
        print(f"skip {family}.{cls}: {e}")
        return None
    examples, source = select_positives(family, cls, opts.n_examples,
                                        opts.pos_threshold, opts.seed)
    if not examples:
        print(f"skip {family}.{cls}: no positive examples")
        return None
    ctx = ConceptCtx(
        family=family, cls=cls, iv_arm=iv_arm, layers=layers,
        arms={L: {a: arms[L][a] for a in METERS} for L in layers},
        natstats=natstats,
        t={L: calib[L]["t"] for L in layers},
        examples=examples, source=source,
        neutrals=(select_neutrals(family, opts.n_examples,
                                  opts.pos_threshold, opts.seed)
                  if opts.patch else []),
        diag_ids=load_diag_ids(family, cls, tok),
    )
    if iv_arm != "ridge":
        # natscores t is ridge-based; other arms get t from the clean pass
        # over this concept's positives (documented deviation #2) — set later
        # by a prepass in main(); mark with NaN so it is visible if unset.
        ctx.t = {L: float("nan") for L in layers}
    return ctx


def arm_t_prepass(model, ctx: ConceptCtx, opts):
    """Clean prepass to set ctx.t for non-ridge intervention arms: t[L] =
    mean clean iv-arm score over all natural positive positions."""
    device = opts.device
    batches = pack_batches(ctx.examples, opts.bos_id, opts.pad_id,
                           opts.batch_tokens)
    sums = {L: 0.0 for L in ctx.layers}
    cnts = {L: 0 for L in ctx.layers}
    for batch in tqdm(batches, desc=f"t-prepass[{ctx.iv_arm}]", leave=False):
        masks = _possets(batch, device)
        out = _fwd(model, batch, device)
        s = meter_scores(out.hidden_states, ctx)
        for L in ctx.layers:
            v = masked_mean_per_ex(s[(L, ctx.iv_arm)],
                                   masks["all"]).cpu().numpy()
            ok = np.isfinite(v)
            sums[L] += v[ok].sum()
            cnts[L] += ok.sum()
        del out
    ctx.t = {L: float(sums[L] / max(cnts[L], 1)) for L in ctx.layers}


# -------------------------------------------------------------------- smoke
def run_smoke(opts) -> int:
    """Tiny random Gemma2, synthetic arms/natstats/texts, full pipeline incl.
    --frozen/--patch; asserts (1) the copy-matrix telescoping is exact
    (< 1e-3 relative), (2) ridge deficit at (l, l'=l) equals t - clean mean
    (fp32, layers below the post-norm-tied last layer), (3) C[l,l] == 1."""
    from transformers import Gemma2Config, Gemma2ForCausalLM
    torch.manual_seed(0)
    D, NLAY, VOC = 64, 6, 128
    cfg = Gemma2Config(vocab_size=VOC, hidden_size=D, intermediate_size=128,
                       num_hidden_layers=NLAY, num_attention_heads=2,
                       num_key_value_heads=1, head_dim=16,
                       max_position_embeddings=64,
                       attn_implementation="eager")
    model = Gemma2ForCausalLM(cfg).eval()
    layers = list(range(NLAY))                 # last layer = post-norm analog
    rng = np.random.default_rng(7)

    def uvec():
        v = rng.normal(size=D).astype(np.float32)
        return v / np.linalg.norm(v)

    natstats = {L: (rng.normal(size=D).astype(np.float32),
                    (0.5 + rng.random(D)).astype(np.float32))
                for L in layers}
    arms = {L: {"ridge": (uvec(), 0.1), "dom": (uvec(), 0.0)}
            for L in layers}

    def mk_examples(k, tag):
        exs = []
        for i in range(k):
            ln = int(rng.integers(8, 16))
            ids = rng.integers(3, VOC, size=ln).tolist()
            cpos = sorted(rng.choice(ln, size=max(1, ln // 4),
                                     replace=False).tolist())
            exs.append(dict(example_id=f"{tag}{i:02d}", ids=ids, cpos=cpos))
        return exs

    ctx = ConceptCtx(
        family="smoke", cls="smokeclass", iv_arm="ridge", layers=layers,
        arms=arms, natstats=natstats, t={L: 0.123 for L in layers},
        examples=mk_examples(6, "pos"), source="synthetic",
        neutrals=mk_examples(4, "neu"), diag_ids=[3, 5, 7],
        n_layers_model=NLAY)

    opts.device = "cpu"
    opts.bos_id = cfg.bos_token_id if cfg.bos_token_id is not None else 2
    opts.pad_id = 0
    opts.frozen = True
    opts.patch = True
    out_dir = Path(opts.out) if opts.out else STAGE_DIR / "out" / "e5_smoke"
    log = out_dir.parent / "progress_e5_propagation.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    res = run_concept(model, ctx, opts, log)
    nL = len(layers)
    fails = []

    # 1) exactness of the copy decomposition
    r = res["copy_resid_relmax"]
    print(f"[smoke] copy telescoping max relative residual: {r:.2e}")
    if not (r < 1e-3):
        fails.append(f"copy residual {r} >= 1e-3")

    # 2) ridge deficit at (l, l) == t - clean_mean (exact ablation), and
    # 3) C[l,l] == 1 — both only strictly at layers whose probe reads the raw
    #    block output (all but the last, post-norm-tied layer)
    m_r, p_a = METERS.index("ridge"), POSSETS.index("all")
    for li, l in enumerate(layers[:-1]):
        # ablate sets w.z = t exactly; the ridge METER reads w.z + b
        want = ctx.t[l] + ctx.arms[l]["ridge"][1] \
            - res["clean_score"][li, m_r, p_a]
        got = res["deficit"][li, li, m_r, p_a]
        if abs(got - want) > 5e-3:
            fails.append(f"deficit({l},{l}) {got:.5f} != t+b-clean "
                         f"{want:.5f}")
        c_ll = res["C"][li, li, m_r, p_a]
        if abs(c_ll - 1.0) > 5e-3:
            fails.append(f"C({l},{l}) = {c_ll:.5f} != 1")

    # 4) shapes + pipeline completeness
    checks = {
        "deficit": (nL, nL, 2, 2), "C": (nL, nL, 2, 2),
        "onward_deficit": (nL, nL, 2, 2),
        "frozen_deficit": (2, nL, 2, 2),
        "patch_m_denoise": (nL,), "patch_m_erase": (nL,),
    }
    for k, shape in checks.items():
        if k not in res:
            fails.append(f"missing result {k}")
        elif tuple(np.asarray(res[k]).shape) != shape:
            fails.append(f"{k} shape {np.asarray(res[k]).shape} != {shape}")
    if not np.isfinite(res["anchor_clean_ex"]).all():
        fails.append("anchor has non-finite entries despite diag ids")

    payload = {"__doc__": np.array(KEY_DOC), "meters": np.array(METERS),
               "possets": np.array(POSSETS)}
    payload.update(concept_payload(ctx, res))
    save_family(out_dir, "smoke", payload)
    with open(out_dir / "summary.jsonl", "a") as f:
        for rw in summary_rows(ctx, res, opts):
            f.write(json.dumps(rw) + "\n")

    if fails:
        print("[smoke] FAIL:")
        for m in fails:
            print("  -", m)
        return 1
    print(f"[smoke] PASS — outputs in {out_dir} "
          f"(resid {r:.2e}; deficit/C identities hold; frozen+patch ran)")
    return 0


# ------------------------------------------------------------------- dry run
def run_dry(concepts, layers, opts) -> int:
    nL = len(layers)
    tot_f = tot_tok = 0
    try:
        from transformers import AutoTokenizer
        from common import MODEL_NAME
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:                                    # noqa: BLE001
        print(f"(tokenizer unavailable: {e} — diag-token counts skipped)")
        tok = None
    print(f"E5 dry-run: {len(concepts)} concepts, layers {layers}, "
          f"n-examples {opts.n_examples}, batch-tokens {opts.batch_tokens}, "
          f"arms {opts.arms}")
    for family, cls in concepts:
        try:
            for L in layers:
                load_arms(family, cls, L)
                dose_calib(family, cls, L)
        except FileNotFoundError as e:
            print(f"  {family}.{cls}: SKIP ({e})")
            continue
        exs, source = select_positives(family, cls, opts.n_examples,
                                       opts.pos_threshold, opts.seed)
        ntok = sum(len(e["ids"]) + 1 for e in exs)
        nb = len(pack_batches(exs, 2, 0, opts.batch_tokens))
        fw = nb * (1 + 2 * nL)
        line = (f"  {family}.{cls}: {len(exs)} ex [{source}], {ntok} tok, "
                f"{nb} batches -> {fw} fwd (a/b/d)")
        if opts.frozen:
            fw += nb * 3
            line += f" +{nb * 3} frozen"
        if opts.patch:
            neu = select_neutrals(family, opts.n_examples,
                                  opts.pos_threshold, opts.seed)
            nnb = len(pack_batches(neu, 2, 0, opts.batch_tokens))
            fw += nnb * (1 + nL) + nb * nL
            line += f" +{nnb * (1 + nL) + nb * nL} patch"
        tok_json = PROMPTS_DIR / f"{family}.tokens.json"
        if not tok_json.exists():
            line += "  [no tokens.json: anchor=NaN, patch skipped]"
        elif tok is not None:
            d = load_diag_ids(family, cls, tok)
            line += (f"  [diag ids: {len(d)}]" if d
                     else "  [diag ids: NONE -> anchor=NaN, patch skipped]")
        for arm in opts.arms.split(","):
            tot_f += fw + (nb if arm != "ridge" else 0)
        tot_tok += ntok
        print(line)
    print(f"TOTAL: ~{tot_f} forwards over all intervention arms, "
          f"{tot_tok} positive-example tokens. No forwards were run.")
    return 0


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="E5 — erasure propagation, copy matrix, frozen-attn "
                    "control, write-layer localization (task.md §6.1.7)")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--classes", default="",
                    help="csv of class names (canonical underscores); "
                         "empty = all classes of the selected families")
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--arms", default="ridge",
                    help="csv of INTERVENTION arms (meters are always "
                         "ridge+dom)")
    ap.add_argument("--n-examples", type=int, default=100)
    ap.add_argument("--pos-threshold", type=float, default=0.34)
    ap.add_argument("--out", default=str(STAGE_DIR / "out" / "e5"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default=None,
                    help="default bfloat16 on cuda, float32 on cpu")
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0,
                    help="max #concepts (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--frozen", action="store_true",
                    help="run the frozen-attention control (top-2 salient "
                         "layers per concept)")
    ap.add_argument("--patch", action="store_true",
                    help="run distributional denoise/erase write-layer "
                         "localization")
    ap.add_argument("--skip", default="",
                    help="csv of families or family.class to skip")
    ap.add_argument("--resume", action="store_true",
                    help="skip concepts already present in the family npz")
    opts = ap.parse_args()

    if opts.smoke:
        return run_smoke(opts)

    layers = [int(x) for x in opts.layers.split(",")]
    bad = [L for L in layers if L not in LAYERS]
    if bad:
        ap.error(f"--layers must be probed layers {LAYERS}; got {bad}")
    fams = [f for f in opts.families.split(",") if f]
    want_cls = {canon(c) for c in opts.classes.split(",") if c}
    skip = {s for s in opts.skip.split(",") if s}
    concepts = []
    for fam in fams:
        if fam in skip or fam not in FAMILIES:
            continue
        for cls in FAMILIES[fam]:
            if want_cls and cls not in want_cls:
                continue
            if f"{fam}.{cls}" in skip:
                continue
            concepts.append((fam, cls))
    if opts.limit:
        concepts = concepts[:opts.limit]

    if opts.dry_run:
        return run_dry(concepts, layers, opts)

    out_dir = Path(opts.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir.parent / "progress_e5_propagation.log"
    dtype = opts.dtype or ("bfloat16" if opts.device.startswith("cuda")
                           else "float32")
    model, tok = load_model(device=opts.device, dtype=dtype)
    opts.bos_id = tok.bos_token_id
    opts.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    heartbeat(log, f"START {len(concepts)} concepts arms={opts.arms} "
                   f"layers={layers} frozen={opts.frozen} patch={opts.patch}")
    fam_payload: dict[str, dict] = {}
    done = 0
    for ci, (family, cls) in enumerate(concepts):
        if family not in fam_payload:
            path = out_dir / f"{family}.npz"
            fam_payload[family] = (
                dict(np.load(path, allow_pickle=False))
                if path.exists() and opts.resume else {})
            fam_payload[family].update({
                "__doc__": np.array(KEY_DOC), "meters": np.array(METERS),
                "possets": np.array(POSSETS)})
        for arm in opts.arms.split(","):
            key = f"{cls}__{arm}__C"
            if opts.resume and key in fam_payload[family]:
                print(f"resume-skip {family}.{cls} [{arm}]")
                continue
            ctx = build_ctx(family, cls, arm, layers, opts, tok)
            if ctx is None:
                continue
            if arm != "ridge":
                arm_t_prepass(model, ctx, opts)
            heartbeat(log, f"{family}.{cls} [{arm}] {ci + 1}/{len(concepts)}"
                           f" n={len(ctx.examples)} src={ctx.source}")
            res = run_concept(model, ctx, opts, log)
            fam_payload[family].update(concept_payload(ctx, res))
            save_family(out_dir, family, fam_payload[family])
            with open(out_dir / "summary.jsonl", "a") as f:
                for rw in summary_rows(ctx, res, opts):
                    f.write(json.dumps(rw) + "\n")
            done += 1
    heartbeat(log, f"DONE {done} concept-arm runs")
    print(f"E5 done: {done} concept-arm runs -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
