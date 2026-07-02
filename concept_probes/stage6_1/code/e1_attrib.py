"""Stage 6.1 E1 — attribution-patching screening + cross-layer influence
Jacobian (task.md §6.1.3; DESIGN.md wave 2).

Per concept, on --n-pairs matched (clean=concept-positive, corrupt=matched
neutral) pairs:

1. Behavioral attribution map (Nanda attribution patching):
     Δmetric ≈ Σ_{l,pos} (h_clean_l − h_corrupt_l) ⊙ ∂metric/∂h_l |_corrupt
   metric = diagnostic-token log-prob margin at the final position (from
   prompts/<family>.tokens.json when present; fallback anchor = log-prob the
   corrupt run assigns to the CLEAN run's top-1 final token — used by --smoke
   and whenever the tokens file is absent; recorded per concept as
   ``anchor``). One corrupt forward WITH grad (26 block outputs captured +
   grad'd via torch.autograd.grad), h_clean from the clean forward (values
   detached to fp16 after the Jacobian backwards). Attribution accumulated in
   fp32 on CPU, binned per layer into 3 position bins:
   [concept-span tokens, other real tokens, final token] (final wins ties).

2. Cross-layer influence Jacobian J[l,l'] (12×12 probed layers, upper
   triangle incl. diagonal meaningful): pure Jacobian on the CLEAN run —
   metric_{l'} = per-row mean over real positions of the unit-w ridge score
   at l' (probe_scores convention: reads hidden_states[l'+1]; l'=25 therefore
   reads THROUGH the final RMSNorm, harness fact). One backward per l'
   (12 per batch, retain_graph=True except the last), reusing the clean
   forward graph. Row entry per pair:
     J[l,l'] = Σ_{real pos} (σ_l ⊙ w_l) · ∂metric_{l'}/∂h_l
   i.e. the first-order change of the mean-l'-score under a unit std-arm
   steer (h_l += σ_l⊙w_l at all positions). This makes J[l,l] = 1 exactly for
   l < 25 (sanity-checked in --smoke); J[l,l'] = 0 for l > l' (causality).
   Both sides use the RIDGE arm (task formula); E1 is a screening estimate,
   the meter≠vector rule binds the confirmatory E2/E4/E5 runs.

Data / pairing (documented per deliverable):
- Primary pool: stage4/data/<family>/judged/judged_nat.jsonl.
  positive(example, class) := max strength of aggregated_spans whose concept
  (canonicalized to underscores) == class is >= 0.34 (the Stage-6 ymax bar).
  This span-based rule is required because intensity families (costliness,
  duration, …) have cls=null on ALL natural rows — positives are only
  identifiable via spans — and it also recovers cross-class span hits.
  neutral(example) := max strength over ALL aggregated_spans < 0.10
  (family-neutral; cls anything, in practice the natural_random slice).
- Fallback (used when natural positives < --min-natural, default 50; e.g.
  most moon_phases classes, glorptitude which has no judged_nat.jsonl):
  stage4/data/<family>/final/mixed/<class>.val.jsonl — role target_pos as
  clean, role neutral (with max target_span strength < 0.34) as corrupt;
  concept spans = target_spans char triplets. Source recorded per concept.
- Concept token positions: judged char spans (aggregated_spans /
  target_spans) mapped through tokenizer offset mapping (+1 for BOS); ALL
  target-concept spans mark positions (no strength filter on the mask —
  the 0.34 bar gates example inclusion, not span tokens). token_targets_sparse
  exists only in the generated val files, so char spans + offsets are the one
  code path that covers both pools.
- Pairing: pairs matched by token length. Positives are shuffled
  (seeded per concept); for each positive of length Tp the shortest UNUSED
  neutral with length >= Tp is taken (bisect), else the longest remaining
  one. Both sequences are truncated to T = min(Tp, Tn) (BOS kept), so clean
  and corrupt in a pair are exactly the same length — attribution needs
  positionwise clean−corrupt differences. A pair is kept only if >=1
  concept-span token survives at position < T−1. Neutrals are used without
  replacement within a concept (pool resets per concept).

Outputs (--out, default stage6_1/out/e1):
- <family>.npz: classes [C], probe_layers [12], bins,
  attrib [C,P,26,3] fp32 (per-pair per-bin SUMS, NaN-padded over P),
  attrib_total [C,P,26] (all-position sum = first-order full-layer-patch
  estimate), bin_counts [C,P,3], J [C,P,12,12], dmetric [C,P]
  (metric_clean − metric_corrupt, the exact full-patch target), n_pairs [C],
  candidate_layer [C] (argmax_l of mean concept-bin attribution, signed),
  candidate_layer_abs [C] (argmax |·|), chosen_layer [C] (Stage-6.0 card,
  −1 if absent e.g. glorptitude), disagree [C] (nearest probed layer to
  candidate != chosen; −1 unknown), source/anchor [C] str.
- <family>.pairs.json: per class the (clean_id, corrupt_id, T, n_concept)
  provenance list.
- summary.jsonl: per concept one behavioral row (metric
  e1_behav_attrib_concept at the candidate layer, normal-approx CI over
  pairs) + one Jacobian row (metric e1_J_diag_mean, config carries the max
  upper-off-diagonal entry).
- progress heartbeats -> <out>/../progress_e1_attrib.log.

Modes:
- --smoke: tiny random Gemma2 (test_interventions fixture config), synthetic
  pairs (corrupt = clean with k token substitutions; substituted positions =
  the "concept span"), fp32 CPU. Validates the attribution estimate against
  REAL activation patches: (a) per-(layer,position) single-position patches
  vs attribution[l,t] (Pearson r reported — the headline check), (b)
  full-layer all-position patch, whose true Δ equals metric_clean −
  metric_corrupt at every layer (residual-stream replacement), vs
  attrib_total. Also checks J diag = 1 (l < last), J lower triangle = 0, and
  prints wall-clock per phase + the retain-graph-vs-separate-forwards
  comparison.
- --dry-run: CPU; loads tokenizer, probe cards, arms, natstats, pools; builds
  all pairs; writes <out>/dryrun_plan.json with per-concept counts and
  forward/backward totals. No model weights, no forwards.

Usage:
  python e1_attrib.py --smoke
  python e1_attrib.py --dry-run
  python e1_attrib.py --families months,weekdays --n-pairs 100 \
      --device cuda --batch-tokens 4096
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                  # noqa: E402
from common import LAYERS, load_arms, load_natstats, probe_scores  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
STAGE_DIR = CODE_DIR.parent
CP_DIR = STAGE_DIR.parent
STAGE4_DATA = CP_DIR / "stage4" / "data"
PROMPTS_DIR = STAGE_DIR / "prompts"
PROBE_CARDS = CP_DIR / "stage6" / "artifacts" / "probe_cards.json"

POS_STRENGTH = 0.34          # example-level confirmed-positive bar (task.md)
NEU_STRENGTH = 0.10          # family-neutral bar on max span strength
BINS = ("concept", "other", "final")
SCRIPT = "e1_attrib"


def canon(s) -> str:
    return str(s).replace(" ", "_")


def heartbeat(log: Path, msg: str):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {SCRIPT} {msg}\n")


# =========================================================== data + pairing
def load_natural_pool(family: str):
    """judged_nat.jsonl -> list of {'id','text','spans':{cls:[(a,b,st)]},
    'max_strength': float} or None if the file is missing (glorptitude)."""
    path = STAGE4_DATA / family / "judged" / "judged_nat.jsonl"
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            spans: dict[str, list] = {}
            mx = 0.0
            for sp in r.get("aggregated_spans") or []:
                c = canon(sp["concept"])
                a, b = sp["char_span"]
                st = float(sp["strength"])
                spans.setdefault(c, []).append((int(a), int(b), st))
                mx = max(mx, st)
            rows.append({"id": r["example_id"], "text": r["text"],
                         "spans": spans, "max_strength": mx})
    return rows


def natural_pos_neu(rows, cls: str):
    pos = [r for r in rows
           if max((st for *_, st in r["spans"].get(cls, [])), default=0.0)
           >= POS_STRENGTH]
    neu = [r for r in rows if r["max_strength"] < NEU_STRENGTH]
    return pos, neu


def load_val_pool(family: str, cls: str):
    """final/mixed/<cls>.val.jsonl -> (positives, neutrals) in pool format
    (roles target_pos / neutral; spans from char target_spans)."""
    path = STAGE4_DATA / family / "final" / "mixed" / f"{cls}.val.jsonl"
    if not path.exists():
        return [], []
    pos, neu = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            spans = [(int(a), int(b), float(st))
                     for a, b, st in (r.get("target_spans") or [])]
            mx = max((st for *_, st in spans), default=0.0)
            item = {"id": r["example_id"], "text": r["text"],
                    "spans": {cls: spans}, "max_strength": mx}
            if r["role"] == "target_pos":
                pos.append(item)
            elif r["role"] == "neutral" and mx < POS_STRENGTH:
                neu.append(item)
    return pos, neu


def tokenize_with_mask(tok, text: str, spans):
    """-> (ids incl BOS, concept bool mask same length). Token i (0-based in
    the no-BOS encoding, +1 after BOS) is a concept token iff its char span
    overlaps any judged span [a, b)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [tok.bos_token_id] + list(enc["input_ids"])
    mask = [False] * len(ids)
    if spans:
        for i, (s, e) in enumerate(enc["offset_mapping"]):
            if e <= s:
                continue
            for a, b, _ in spans:
                if s < b and e > a:
                    mask[i + 1] = True
                    break
    return ids, mask


@dataclass
class Pair:
    clean_ids: list[int]        # length T, BOS first
    corrupt_ids: list[int]      # length T
    concept_mask: list[bool]    # length T (clean-text spans)
    clean_id: str
    corrupt_id: str
    T: int


def build_pairs(tok, pos_items, neu_items, cls: str, n_pairs: int, seed: int):
    """Length-matched pairing (see module docstring). Returns (pairs,
    n_skipped) — skipped = positives whose concept span did not survive
    truncation with any remaining neutral."""
    rng = np.random.default_rng(seed)
    pos_tok = []
    for it in rng.permutation(len(pos_items)):
        it = pos_items[int(it)]
        ids, mask = tokenize_with_mask(tok, it["text"], it["spans"].get(cls))
        if any(mask):
            pos_tok.append((it, ids, mask))
    neu_tok = []
    for it in neu_items:
        ids, _ = tokenize_with_mask(tok, it["text"], None)
        neu_tok.append((it, ids))
    neu_tok.sort(key=lambda x: len(x[1]))
    neu_lens = [len(ids) for _, ids in neu_tok]

    pairs, skipped = [], 0
    for it, ids, mask in pos_tok:
        if len(pairs) >= n_pairs:
            break
        if not neu_tok:
            break
        k = bisect.bisect_left(neu_lens, len(ids))
        k = k if k < len(neu_tok) else len(neu_tok) - 1   # else longest left
        nit, nids = neu_tok.pop(k)
        neu_lens.pop(k)
        T = min(len(ids), len(nids))
        if T < 3 or not any(mask[:T - 1]):
            skipped += 1                # span truncated away; neutral consumed
            continue
        pairs.append(Pair(ids[:T], nids[:T], mask[:T],
                          it["id"], nit["id"], T))
    return pairs, skipped


def concept_pairs(tok, family: str, cls: str, n_pairs: int, min_natural: int,
                  seed_base: int):
    """-> (pairs, meta) choosing natural vs generated source per the <50 rule."""
    seed = seed_base + zlib.crc32(f"{family}.{cls}".encode())
    nat = load_natural_pool.cache.get(family) if hasattr(
        load_natural_pool, "cache") else None
    if nat is None:
        nat = load_natural_pool(family)
        if not hasattr(load_natural_pool, "cache"):
            load_natural_pool.cache = {}
        load_natural_pool.cache[family] = nat
    n_pos_nat = 0
    if nat is not None:
        pos, neu = natural_pos_neu(nat, cls)
        n_pos_nat = len(pos)
    if nat is None or n_pos_nat < min_natural:
        pos, neu = load_val_pool(family, cls)
        source = "generated_val"
    else:
        source = "natural_judged"
    pairs, skipped = build_pairs(tok, pos, neu, cls, n_pairs, seed)
    meta = {"source": source, "n_pos_pool": len(pos), "n_neu_pool": len(neu),
            "n_pos_natural": n_pos_nat, "n_pairs": len(pairs),
            "n_skipped": skipped}
    return pairs, meta


# ==================================================== anchors (behavioral)
def _first_token_ids(tok, strings) -> set[int]:
    """Diagnostic token ids for a set of words. Multi-word strings are
    DROPPED ('New Year' -> first token 'New' is generically ambiguous).
    The space-prefixed form contributes its first token (the natural
    mid-sentence continuation); the raw form is added only when the whole
    word is a single token — otherwise its first token is a fragment
    ('ex' of 'expensive') that would poison the margin."""
    ids: set[int] = set()
    for s in strings or []:
        s = str(s).strip()
        if not s or " " in s:
            continue
        t = tok(s, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            ids.add(int(t[0]))
        t = tok(" " + s, add_special_tokens=False)["input_ids"]
        if t:
            ids.add(int(t[0]))
    return ids


def load_anchor_ids(tok, family: str, classes: list[str]):
    """prompts/<family>.tokens.json (A3 bank schema, see prompts/README.md)
    -> {class: {'target': [ids], 'opposite': [ids] | None}} or None.

    - categorical: ``classes.<cls>.{surface, associates}``; target = first
      tokens of surface + single-word associates; opposite = None (siblings
      supply the margin's negative side).
    - intensity: ``poles.{low, high}``; the family's single class gets
      target = high-pole tokens, opposite = low-pole tokens (margin =
      high-vs-low pole; for lovingness low = despise pole per the README).
    """
    path = PROMPTS_DIR / f"{family}.tokens.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    out: dict[str, dict] = {}
    if isinstance(raw.get("poles"), dict):
        hi = raw["poles"].get("high", {})
        lo = raw["poles"].get("low", {})
        tgt = _first_token_ids(tok, list(hi.get("surface", []))
                               + list(hi.get("associates", [])))
        opp = _first_token_ids(tok, list(lo.get("surface", []))
                               + list(lo.get("associates", [])))
        cls = classes[0] if len(classes) == 1 else canon(family)
        if tgt:
            out[cls] = {"target": sorted(tgt),
                        "opposite": sorted(opp - tgt) or None}
    elif isinstance(raw.get("classes"), dict):
        for c, v in raw["classes"].items():
            cc = canon(c)
            if cc not in classes or not isinstance(v, dict):
                continue
            ids = _first_token_ids(tok, list(v.get("surface", []))
                                   + list(v.get("associates", [])))
            if ids:
                out[cc] = {"target": sorted(ids), "opposite": None}
    return out or None


def behavioral_metric(final_logits: torch.Tensor, target_ids, sibling_ids,
                      top1: torch.Tensor) -> torch.Tensor:
    """Per-row metric [B] (fp32, differentiable) from final-position logits.
    tokens.json margin: logsumexp(target) − logsumexp(siblings) (target-only
    if no siblings). Fallback (target_ids None): logprob of the clean run's
    top-1 token."""
    lp = torch.log_softmax(final_logits.to(torch.float32), dim=-1)
    if target_ids is None:
        return lp[torch.arange(lp.shape[0], device=lp.device), top1]
    m = torch.logsumexp(lp[:, target_ids], dim=-1)
    if sibling_ids:
        m = m - torch.logsumexp(lp[:, sibling_ids], dim=-1)
    return m


# ====================================================== model-side plumbing
class BlockCaptures:
    """Forward hooks on model.model.layers[l] capturing each block's raw
    output hidden state (tuple-or-tensor safe, harness fact). Returning None
    leaves the flowing tensor untouched, so captures ARE the graph tensors
    passed downstream — valid ``inputs`` for torch.autograd.grad."""

    def __init__(self, model, layers):
        self.base = model.model if hasattr(model, "model") else model
        self.layers = list(layers)
        self.h: dict[int, torch.Tensor] = {}
        self._handles = []

    def _make(self, l):
        def hook(module, args, output):
            self.h[l] = output[0] if isinstance(output, tuple) else output
        return hook

    def __enter__(self):
        for l in self.layers:
            self._handles.append(
                self.base.layers[l].register_forward_hook(self._make(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


class PatchLayer:
    """Smoke-validation hook: replace block-l output at masked positions with
    a stored tensor (true activation patch)."""

    def __init__(self, model, layer: int, h_new: torch.Tensor,
                 mask: torch.Tensor):
        self.base = model.model if hasattr(model, "model") else model
        self.layer, self.h_new, self.mask = layer, h_new, mask
        self._handle = None

    def _hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        new = torch.where(self.mask.to(hs.device).unsqueeze(-1),
                          self.h_new.to(hs.dtype), hs)
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new

    def __enter__(self):
        self._handle = self.base.layers[self.layer].register_forward_hook(
            self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        return False


def collate(pairs: list[Pair], pad_id: int, device):
    B, L = len(pairs), max(p.T for p in pairs)
    ids_c = torch.full((B, L), pad_id, dtype=torch.long)
    ids_x = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    cmask = torch.zeros((B, L), dtype=torch.bool)
    finals = torch.tensor([p.T - 1 for p in pairs], dtype=torch.long)
    for r, p in enumerate(pairs):
        ids_c[r, :p.T] = torch.tensor(p.clean_ids)
        ids_x[r, :p.T] = torch.tensor(p.corrupt_ids)
        attn[r, :p.T] = 1
        cmask[r, :p.T] = torch.tensor(p.concept_mask)
    return (ids_c.to(device), ids_x.to(device), attn.to(device),
            cmask.to(device), finals.to(device))


def batched(pairs: list[Pair], batch_tokens: int):
    """Sort by length, greedily pack so B * Lmax <= batch_tokens (>=1 pair)."""
    order = sorted(range(len(pairs)), key=lambda i: pairs[i].T)
    batch, out = [], []
    for i in order:
        if batch and pairs[i].T * (len(batch) + 1) > batch_tokens:
            out.append(batch)
            batch = []
        batch.append(i)
    if batch:
        out.append(batch)
    return out


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


# ============================================================ core per-batch
def run_concept(model, pairs: list[Pair], probe_arms: dict, natstats: dict,
                anchor: tuple, device, batch_tokens: int, probe_layers: list,
                n_blocks: int, pad_id: int, timers: dict,
                debug_store: list | None = None, progress_cb=None):
    """probe_arms: {L: (w fp32 np [d], b float)} ridge unit-norm;
    natstats: {L: (mu, sd) fp32 np [d]}; anchor = (target_ids, sibling_ids)
    or (None, None) for the clean-top1 fallback.
    Returns dict of per-pair arrays: attrib [P,26,3], attrib_total [P,26],
    bin_counts [P,3], J [P,12,12], dmetric [P]."""
    P = len(pairs)
    nL, nP = n_blocks, len(probe_layers)
    attrib = np.full((P, nL, len(BINS)), np.nan, np.float32)
    attrib_total = np.full((P, nL), np.nan, np.float32)
    bin_counts = np.zeros((P, len(BINS)), np.int32)
    J = np.full((P, nP, nP), np.nan, np.float32)
    dmetric = np.full(P, np.nan, np.float32)
    target_ids, sibling_ids = anchor

    # per-probe-layer fp32 device constants: steer write vector sd⊙w
    steer_vec = {
        L: torch.as_tensor(natstats[L][1] * probe_arms[L][0],
                           dtype=torch.float32, device=device)
        for L in probe_layers}

    for batch_idx in batched(pairs, batch_tokens):
        bpairs = [pairs[i] for i in batch_idx]
        ids_c, ids_x, attn, cmask, finals = collate(bpairs, pad_id, device)
        B, L = ids_c.shape
        rows = torch.arange(B, device=device)
        attn_f = attn.to(torch.float32)
        denom = attn_f.sum(1).clamp(min=1.0)

        # ---------------- clean forward (grad) + 12 Jacobian backwards ----
        _sync(device); t0 = time.perf_counter()
        with BlockCaptures(model, range(n_blocks)) as cap:
            out_clean = model(input_ids=ids_c, attention_mask=attn,
                              output_hidden_states=True, use_cache=False)
            h_clean = dict(cap.h)
        _sync(device); timers["clean_fwd"] += time.perf_counter() - t0
        timers["n_batches"] += 1
        timers["n_tokens"] += int(attn.sum())

        logits_clean = out_clean.logits[rows, finals]
        top1 = logits_clean.argmax(-1).detach()
        with torch.no_grad():
            m_clean = behavioral_metric(logits_clean.detach(), target_ids,
                                        sibling_ids, top1)

        grad_inputs = [h_clean[l] for l in probe_layers]
        Jb = torch.zeros((B, nP, nP), dtype=torch.float32)
        _sync(device); t0 = time.perf_counter()
        for i, lp in enumerate(probe_layers):
            w, b = probe_arms[lp]
            mu, sd = natstats[lp]
            s = probe_scores(out_clean.hidden_states, lp, w, b, mu, sd)
            metric = ((s * attn_f).sum(1) / denom).sum()
            grads = torch.autograd.grad(
                metric, grad_inputs, retain_graph=(i < nP - 1),
                allow_unused=True)
            for j, l in enumerate(probe_layers):
                g = grads[j]
                if g is None:
                    continue
                proj = (g.to(torch.float32) * steer_vec[l]).sum(-1)   # [B,L]
                Jb[:, j, i] = ((proj * attn_f).sum(1)).cpu()
        _sync(device); timers["j_bwd"] += time.perf_counter() - t0

        h_clean_vals = {l: h_clean[l].detach().to(torch.float16)
                        for l in range(n_blocks)}
        del out_clean, logits_clean, h_clean, grad_inputs, s, metric, grads

        # ------------- corrupt forward (grad) + behavioral backward -------
        _sync(device); t0 = time.perf_counter()
        with BlockCaptures(model, range(n_blocks)) as cap:
            out_corr = model(input_ids=ids_x, attention_mask=attn,
                             output_hidden_states=False, use_cache=False)
            h_corr = dict(cap.h)
        _sync(device); timers["corr_fwd"] += time.perf_counter() - t0

        m_corr = behavioral_metric(out_corr.logits[rows, finals], target_ids,
                                   sibling_ids, top1)
        _sync(device); t0 = time.perf_counter()
        grads = torch.autograd.grad(m_corr.sum(),
                                    [h_corr[l] for l in range(n_blocks)])
        _sync(device); timers["behav_bwd"] += time.perf_counter() - t0

        fin_mask = torch.zeros_like(cmask)
        fin_mask[rows, finals] = True
        con = cmask & attn.bool() & ~fin_mask
        oth = attn.bool() & ~cmask & ~fin_mask
        masks = {"concept": con.to(torch.float32),
                 "other": oth.to(torch.float32),
                 "final": fin_mask.to(torch.float32)}

        attr_pos = None
        if debug_store is not None:
            attr_pos = torch.zeros((B, n_blocks, L), dtype=torch.float32)
        ab = np.zeros((B, nL, len(BINS)), np.float32)
        at = np.zeros((B, nL), np.float32)
        for l in range(n_blocks):
            diff = (h_clean_vals[l].to(torch.float32)
                    - h_corr[l].detach().to(torch.float32))
            a = (diff * grads[l].to(torch.float32)).sum(-1)          # [B,L]
            for bi, name in enumerate(BINS):
                ab[:, l, bi] = (a * masks[name]).sum(1).cpu().numpy()
            at[:, l] = (a * attn_f).sum(1).cpu().numpy()
            if attr_pos is not None:
                attr_pos[:, l] = a.cpu()

        dm = (m_clean - m_corr.detach()).cpu().numpy()
        for r, pi in enumerate(batch_idx):
            attrib[pi] = ab[r]
            attrib_total[pi] = at[r]
            J[pi] = Jb[r].numpy()
            dmetric[pi] = dm[r]
            for bi, name in enumerate(BINS):
                bin_counts[pi, bi] = int(masks[name][r].sum())
        if debug_store is not None:
            debug_store.append({
                "batch_idx": batch_idx, "ids_x": ids_x.cpu(),
                "attn": attn.cpu(), "finals": finals.cpu(),
                "top1": top1.cpu(), "m_corr": m_corr.detach().cpu(),
                "m_clean": m_clean.cpu(), "attr_pos": attr_pos,
                "h_clean_vals": {l: v.cpu() for l, v in h_clean_vals.items()},
            })
        del out_corr, h_corr, grads, h_clean_vals
        if progress_cb:
            progress_cb(len(batch_idx))

    return {"attrib": attrib, "attrib_total": attrib_total,
            "bin_counts": bin_counts, "J": J, "dmetric": dmetric}


# ============================================================= orchestration
def select_concepts(args) -> list[tuple[str, str]]:
    fams = ([f.strip() for f in args.families.split(",")] if args.families
            else sorted(common.FAMILIES))
    cls_filter = ({canon(c) for c in args.classes.split(",")}
                  if args.classes else None)
    out = []
    for f in fams:
        if f not in common.FAMILIES:
            raise SystemExit(f"unknown family {f!r}; have "
                             f"{sorted(common.FAMILIES)}")
        for c in common.FAMILIES[f]:
            if cls_filter is None or c in cls_filter:
                out.append((f, c))
    if args.limit:
        out = out[:args.limit]
    return out


def chosen_layers() -> dict[tuple[str, str], int]:
    cards = json.loads(Path(PROBE_CARDS).read_text())
    return {(c["family"], canon(c["concept"])): int(c["layer"])
            for c in cards}


def nearest_probed(layer: int) -> int:
    return min(LAYERS, key=lambda L: (abs(L - layer), L))


def summarize_concept(family, cls, res, meta, anchor_name, cand_cfg):
    """-> (per-concept scalars dict, summary rows list)."""
    n = meta["n_pairs"]
    if n == 0:
        row = {"concept": cls, "family": family, "layer": -1, "arm": "none",
               "metric": "e1_behav_attrib_concept", "value": None, "n": 0,
               "ci_low": None, "ci_high": None,
               "config": {**meta, "anchor": anchor_name, **cand_cfg,
                          "note": "no valid pairs"}}
        return {"candidate_layer": -1, "candidate_layer_abs": -1,
                "disagree": -1}, [row]
    con = res["attrib"][:, :, 0]                       # [P, 26]
    mean_l = np.nanmean(con, axis=0)
    cand = int(np.nanargmax(mean_l))
    cand_abs = int(np.nanargmax(np.abs(mean_l)))
    chosen = cand_cfg["chosen_layer"]
    disagree = (-1 if chosen < 0 else int(nearest_probed(cand) != chosen))
    v = con[:, cand]
    se = float(np.nanstd(v, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    Jm = np.nanmean(res["J"], axis=0)                  # [12, 12]
    iu = np.triu_indices(len(LAYERS), 1)
    diag = float(np.nanmean(np.diag(Jm)[:-1]))         # excl through-norm L25
    offmax = float(np.nanmax(Jm[iu])) if iu[0].size else float("nan")
    cfg = {**meta, "anchor": anchor_name, **cand_cfg,
           "candidate_layer_abs": cand_abs, "disagree": disagree,
           "nearest_probed": nearest_probed(cand),
           "pos_strength": POS_STRENGTH, "truncation": "pair-min-length",
           "dmetric_mean": float(np.nanmean(res["dmetric"]))}
    rows = [
        {"concept": cls, "family": family, "layer": cand, "arm": "none",
         "metric": "e1_behav_attrib_concept", "value": float(np.nanmean(v)),
         "n": n, "ci_low": float(np.nanmean(v) - 1.96 * se),
         "ci_high": float(np.nanmean(v) + 1.96 * se), "config": cfg},
        {"concept": cls, "family": family, "layer": chosen, "arm": "ridge",
         "metric": "e1_J_diag_mean", "value": diag, "n": n,
         "ci_low": None, "ci_high": None,
         "config": {"J_upper_offdiag_max": offmax, "anchor": anchor_name,
                    "meter": "ridge-both-sides (screening)"}},
    ]
    return {"candidate_layer": cand, "candidate_layer_abs": cand_abs,
            "disagree": disagree}, rows


def write_family_npz(out_dir: Path, family: str, classes, per_cls, n_blocks):
    P = max((r["meta"]["n_pairs"] for r in per_cls), default=0)
    P = max(P, 1)
    C = len(classes)
    nP = len(LAYERS)

    def pad(key, shape_tail, dtype=np.float32):
        arr = np.full((C, P) + shape_tail, np.nan, dtype)
        for ci, r in enumerate(per_cls):
            n = r["meta"]["n_pairs"]
            if n and r["res"] is not None:
                arr[ci, :n] = r["res"][key][:n]
        return arr

    np.savez(
        out_dir / f"{family}.npz",
        classes=np.array(classes), probe_layers=np.array(LAYERS),
        bins=np.array(BINS), n_layers=n_blocks,
        attrib=pad("attrib", (n_blocks, len(BINS))),
        attrib_total=pad("attrib_total", (n_blocks,)),
        bin_counts=pad("bin_counts", (len(BINS),)),
        J=pad("J", (nP, nP)),
        dmetric=pad("dmetric", ()),
        n_pairs=np.array([r["meta"]["n_pairs"] for r in per_cls]),
        candidate_layer=np.array([r["scal"]["candidate_layer"]
                                  for r in per_cls]),
        candidate_layer_abs=np.array([r["scal"]["candidate_layer_abs"]
                                      for r in per_cls]),
        chosen_layer=np.array([r["chosen"] for r in per_cls]),
        disagree=np.array([r["scal"]["disagree"] for r in per_cls]),
        source=np.array([r["meta"]["source"] for r in per_cls]),
        anchor=np.array([r["anchor"] for r in per_cls]))
    pairs_json = {r["cls"]: [{"clean": p.clean_id, "corrupt": p.corrupt_id,
                              "T": p.T,
                              "n_concept": int(sum(p.concept_mask))}
                             for p in r["pairs"]] for r in per_cls}
    (out_dir / f"{family}.pairs.json").write_text(
        json.dumps(pairs_json, indent=1))


def run_real(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir.parent / f"progress_{SCRIPT}.log"
    concepts = select_concepts(args)
    cards = chosen_layers()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL_NAME)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    # ------------------------------------------------ build all pairs first
    plan = []
    for family, cls in tqdm(concepts, desc="pairing"):
        pairs, meta = concept_pairs(tok, family, cls, args.n_pairs,
                                    args.min_natural, args.seed)
        plan.append({"family": family, "cls": cls, "pairs": pairs,
                     "meta": meta})
        heartbeat(log, f"paired {family}.{cls} n={meta['n_pairs']} "
                       f"src={meta['source']}")

    n_batches = sum(len(batched(p["pairs"], args.batch_tokens))
                    for p in plan if p["pairs"])
    est = {"concepts": len(plan), "batches": n_batches,
           "forwards": 2 * n_batches,
           "backwards": (len(LAYERS) + 1) * n_batches,
           "pair_tokens": sum(pr.T for p in plan for pr in p["pairs"])}
    print(f"plan: {json.dumps(est)}")

    if args.dry_run:
        anchors = {f: load_anchor_ids(tok, f, common.FAMILIES[f])
                   for f in {p["family"] for p in plan}}
        detail = [{k: v for k, v in p.items() if k != "pairs"}
                  | {"chosen_layer": cards.get((p["family"], p["cls"]), -1),
                     "n_batches": len(batched(p["pairs"], args.batch_tokens)),
                     "median_T": (float(np.median([pr.T for pr in p["pairs"]]))
                                  if p["pairs"] else None),
                     "anchor": ("tokens_json"
                                if (anchors[p["family"]] or {}).get(p["cls"])
                                else "clean_top1"),
                     "n_anchor_tokens": len(((anchors[p["family"]] or {})
                                             .get(p["cls"]) or {})
                                            .get("target", []))}
                  for p in plan]
        # verify arms + natstats load for every planned concept
        for p in tqdm(plan, desc="dry-run: arms/natstats"):
            for L in LAYERS:
                load_arms(p["family"], p["cls"], L)
                load_natstats(L)
        (out_dir / "dryrun_plan.json").write_text(
            json.dumps({"est": est, "concepts": detail}, indent=1))
        print(f"dry-run OK -> {out_dir / 'dryrun_plan.json'}")
        heartbeat(log, f"dry-run OK ({len(plan)} concepts, "
                       f"{est['batches']} batches)")
        return

    device = args.device
    dtype = "bfloat16" if device.startswith("cuda") else "float32"
    model, _ = common.load_model(device=device, dtype=dtype)
    for prm in model.parameters():       # grads flow to activations only;
        prm.requires_grad_(True)         # autograd.grad never fills .grad
    n_blocks = model.config.num_hidden_layers
    natstats = {L: load_natstats(L) for L in LAYERS}

    timers = {"clean_fwd": 0.0, "j_bwd": 0.0, "corr_fwd": 0.0,
              "behav_bwd": 0.0, "n_batches": 0, "n_tokens": 0}
    summary_path = out_dir / "summary.jsonl"
    by_family: dict[str, list] = {}
    pbar = tqdm(plan, desc="concepts")
    for i, p in enumerate(pbar):
        family, cls, pairs, meta = p["family"], p["cls"], p["pairs"], p["meta"]
        pbar.set_postfix_str(f"{family}.{cls}")
        chosen = cards.get((family, cls), -1)
        anchor_ids = load_anchor_ids(tok, family, common.FAMILIES[family])
        if anchor_ids and cls in anchor_ids:
            target = anchor_ids[cls]["target"]
            sib = anchor_ids[cls]["opposite"]
            if sib is None:                     # categorical: sibling classes
                sib = sorted(set(t for c, v in anchor_ids.items()
                                 if c != cls for t in v["target"])
                             - set(target))
            anchor, anchor_name = (target, sib or None), "tokens_json"
        else:
            anchor, anchor_name = (None, None), "clean_top1"

        res = None
        if pairs:
            probe_arms = {L: load_arms(family, cls, L)["ridge"]
                          for L in LAYERS}
            res = run_concept(model, pairs, probe_arms, natstats, anchor,
                              device, args.batch_tokens, LAYERS, n_blocks,
                              pad_id, timers)
        scal, rows = summarize_concept(
            family, cls, res if res else
            {"attrib": np.zeros((0, n_blocks, 3)), "J": np.zeros((0, 12, 12)),
             "dmetric": np.zeros(0)},
            meta, anchor_name, {"chosen_layer": chosen})
        with open(summary_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        by_family.setdefault(family, []).append(
            {"cls": cls, "res": res, "meta": meta, "scal": scal,
             "chosen": chosen, "anchor": anchor_name, "pairs": pairs})
        heartbeat(log, f"{family}.{cls} {i + 1}/{len(plan)} "
                       f"cand={scal['candidate_layer']} chosen={chosen}")
        if len(by_family[family]) == len(common.FAMILIES[family]) or \
                i + 1 == len(plan) or plan[i + 1]["family"] != family:
            write_family_npz(out_dir, family,
                             [r["cls"] for r in by_family[family]],
                             by_family[family], n_blocks)

    tot = sum(v for k, v in timers.items() if k.endswith(("fwd", "bwd")))
    print(f"timing: {json.dumps({k: round(v, 2) for k, v in timers.items()})}"
          f" total_model_s={tot:.1f} "
          f"tok/s={timers['n_tokens'] / max(tot, 1e-9):.0f}")
    heartbeat(log, "DONE")


# ==================================================================== smoke
def _tiny_model():
    from transformers import Gemma2Config, Gemma2ForCausalLM
    cfg = Gemma2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=64, attn_implementation="eager")
    torch.manual_seed(0)
    model = Gemma2ForCausalLM(cfg)
    model.eval()
    return model


def run_smoke(args):
    print("=== E1 smoke: tiny random Gemma2, synthetic pairs, fp32 CPU ===")
    D, T, n_sub = 64, 14, 4
    n_pairs = min(args.n_pairs, 16)
    probe_layers = [0, 1]
    model = _tiny_model()
    rng = np.random.default_rng(3)
    natstats = {}
    probe_arms = {}
    for L in probe_layers:
        mu = rng.normal(size=D).astype(np.float32)
        sd = (0.5 + rng.random(D)).astype(np.float32)
        w = rng.normal(size=D).astype(np.float32)
        w /= np.linalg.norm(w)
        natstats[L] = (mu, sd)
        probe_arms[L] = (w, 0.1)

    pairs = []
    for i in range(n_pairs):
        clean = [2] + [int(x) for x in rng.integers(3, 128, T - 1)]
        corrupt = list(clean)
        subs = sorted(rng.choice(np.arange(1, T - 1), n_sub, replace=False))
        for s in subs:
            corrupt[s] = int(rng.integers(3, 128))
        mask = [t in subs for t in range(T)]
        pairs.append(Pair(clean, corrupt, mask, f"c{i}", f"x{i}", T))

    timers = {"clean_fwd": 0.0, "j_bwd": 0.0, "corr_fwd": 0.0,
              "behav_bwd": 0.0, "n_batches": 0, "n_tokens": 0}
    store: list = []
    res = run_concept(model, pairs, probe_arms, natstats, (None, None),
                      "cpu", batch_tokens=8 * T, probe_layers=probe_layers,
                      n_blocks=2, pad_id=0, timers=timers, debug_store=store)

    # -------- J sanity: diag(l<last)=1 exactly, lower triangle = 0 --------
    Jm = np.nanmean(res["J"], axis=0)
    print(f"J mean over pairs:\n{np.array2string(Jm, precision=4)}")
    assert abs(Jm[0, 0] - 1.0) < 1e-4, f"J[0,0]={Jm[0, 0]} != 1"
    assert abs(Jm[1, 0]) < 1e-6, f"J lower-tri nonzero: {Jm[1, 0]}"
    print(f"J checks: diag[l<last]=1 OK (err {abs(Jm[0, 0] - 1):.1e}); "
          f"lower-tri=0 OK; J[1,1] (through final norm) = {Jm[1, 1]:.4f} "
          f"(!=1 expected); J[0,1] cross-layer = {Jm[0, 1]:.4f}")

    # -------- validation vs REAL activation patches -----------------------
    # (a) single-(layer,position) patches vs attribution[l, t]
    est, act = [], []
    n_val = min(6, n_pairs)
    with torch.no_grad():
        for st in store:
            for r, pi in enumerate(st["batch_idx"]):
                if pi >= n_val:
                    continue
                ids_x = st["ids_x"][r:r + 1]
                attn = st["attn"][r:r + 1]
                fin = int(st["finals"][r])
                top1 = st["top1"][r:r + 1]
                base = float(st["m_corr"][r])
                for l in (0, 1):
                    h_new = st["h_clean_vals"][l][r:r + 1].to(torch.float32)
                    for t in range(1, fin + 1):
                        m = torch.zeros(1, ids_x.shape[1], dtype=torch.bool)
                        m[0, t] = True
                        with PatchLayer(model, l, h_new, m):
                            out = model(input_ids=ids_x, attention_mask=attn,
                                        use_cache=False)
                        mp = behavioral_metric(out.logits[:, fin], None,
                                               None, top1)
                        act.append(float(mp) - base)
                        est.append(float(st["attr_pos"][r, l, t]))
    est, act = np.array(est), np.array(act)
    r_pos = float(np.corrcoef(est, act)[0, 1])
    print(f"(a) single-position patches: n={est.size}, Pearson r = {r_pos:.4f}"
          f" (attribution est vs actual delta; first-order error expected)")

    # (b) full-layer patch: true delta = m_clean - m_corr at EVERY layer
    # (residual stream fully replaced) vs attrib_total per layer
    errs = []
    with torch.no_grad():
        for st in store:
            for r, pi in enumerate(st["batch_idx"]):
                if pi >= n_val:
                    continue
                ids_x = st["ids_x"][r:r + 1]
                attn = st["attn"][r:r + 1]
                fin = int(st["finals"][r])
                top1 = st["top1"][r:r + 1]
                dm_true = float(st["m_clean"][r]) - float(st["m_corr"][r])
                for l in (0, 1):
                    h_new = st["h_clean_vals"][l][r:r + 1].to(torch.float32)
                    with PatchLayer(model, l, h_new, attn.bool()):
                        out = model(input_ids=ids_x, attention_mask=attn,
                                    use_cache=False)
                    mp = behavioral_metric(out.logits[:, fin], None, None,
                                           top1)
                    d_act = float(mp) - float(st["m_corr"][r])
                    assert abs(d_act - dm_true) < 1e-3, \
                        "full patch != clean-corrupt delta (plumbing bug)"
                    errs.append((res["attrib_total"][pi, l], d_act))
    e = np.array(errs)
    r_full = float(np.corrcoef(e[:, 0], e[:, 1])[0, 1])
    print(f"(b) full-layer patches: n={len(e)}, true Δ == m_clean−m_corr "
          f"verified; corr(attrib_total, Δ) = {r_full:.4f}, "
          f"median |est−Δ|/|Δ| = "
          f"{np.median(np.abs(e[:, 0] - e[:, 1]) / np.abs(e[:, 1])):.3f}")

    # -------- timing + strategy comparison ---------------------------------
    per_b = {k: timers[k] / max(timers["n_batches"], 1)
             for k in ("clean_fwd", "j_bwd", "corr_fwd", "behav_bwd")}
    print(f"per-batch wall-clock (tiny model): "
          f"{ {k: round(v * 1e3, 2) for k, v in per_b.items()} } ms; "
          f"12-probe-backwards strategy: retain_graph multi-backward cost = "
          f"j_bwd; the alternative (separate fwd+bwd per l') would add "
          f"~{len(probe_layers)}x clean_fwd = "
          f"{len(probe_layers) * per_b['clean_fwd'] * 1e3:.2f} ms/batch "
          f"on top — retain_graph is cheaper whenever memory allows.")
    ok = r_pos > 0.8
    print(f"SMOKE {'PASS' if ok else 'MARGINAL'}: attribution-vs-patch "
          f"r={r_pos:.4f} (threshold 0.8), J identities exact.")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default=None,
                    help="comma list (default: all discovered families)")
    ap.add_argument("--classes", default=None,
                    help="comma list of class names (underscore canonical)")
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--min-natural", type=int, default=50,
                    help="natural-positive count below which the generated "
                         "val fallback is used")
    ap.add_argument("--out", default=str(STAGE_DIR / "out" / "e1"))
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-tokens", type=int, default=4096,
                    help="B*Lmax packing budget per batch (26 retained "
                         "grads x [B,T,2304] is the memory driver)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of concepts (quick real tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random Gemma2 + synthetic pairs; validates "
                         "attribution against real activation patches")
    ap.add_argument("--dry-run", action="store_true",
                    help="load data/probes/tokenizer, build pairs, write "
                         "plan; no model, no forwards")
    args = ap.parse_args()
    if args.smoke:
        sys.exit(run_smoke(args))
    run_real(args)


if __name__ == "__main__":
    main()
