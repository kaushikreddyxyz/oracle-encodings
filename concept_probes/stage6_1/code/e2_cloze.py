"""Stage 6.1 E2a — forced-choice cloze dose-response + specificity (task.md
Section 6.1.4; DESIGN.md conventions). Steers one (concept, layer) at a time
with Intervention(mode='steer', alpha=factor*s95) at ALL positions and scores
completion log-probs of audited prompt-bank continuations.

Prompt banks (authored in parallel; NOT edited here):
  prompts/<family>.cloze.json   = {"family","classes","templates":[{"id","type",
      "prompt","completions":{class:[strings]},"answer_class",null|class,"notes"}]}
  prompts/<family>.ordinal.json = {"axis","prompts":[{"id","prompt"}],
      "ordered_completion_sets":[{"id","completions":[low..high]}]}
Completions continue the prompt; this script adds the leading space.

Metrics (per steered concept x layer x arm x factor; sum-of-token-logprobs is
the summary variant, first-token also stored):
- cloze: d(template) = [mean logP(target-class completions) - mean over sibling
  classes of their mean logP] steered minus the same at alpha=0.
  Per-template dose-response slope = OLS over factors in [-2, 2] (wider factors
  stored but excluded from the slope); anti-steerable fraction = fraction of
  templates with slope < 0; suppression symmetry = slope over factors <= 0.
- ordinal (intensity axes): E[rank] = softmax over the set's completion
  sum-logprobs dot normalized ranks in [0,1]; summary = Spearman(factor,
  mean E[rank] over prompts x sets) + per-(prompt,set) OLS slopes over [-2,2].
- Arms: ridge, dom, rand (first 5 rand_dirs; per-dir stored, rand summary rows
  use the per-template mean over dirs).

Specificity: at factor = --specificity-factor (default 2, documented choice per
task assignment) steer each concept at its probe-card layer (fallback 12) with
ridge AND dom arms, run a FIXED shared 16-prompt subset (seeded from all bank
prompts, sorted by (family, kind, id), rng seed 61), and record all-64-probe
readout means (ridge meter AND dom meter, at each readout concept's own card
layer, BOS excluded) -> out/e2_cloze/offtarget_<family>.npz rows of the future
64x64 matrix. Sibling-completion deltas at that factor are already in the main
npz.

Outputs (per family):
  out/e2_cloze/<family>.npz  keys:
    classes [C] steered concepts, bank_classes [Cb], layers [L], dirs [D]
      (ridge,dom,rand0..rand4 subset per --arms), factors [F] (includes 0),
    template_ids [T], answer_class [T],
    lp0_sum/lp0_first [R] baseline per-row completion logprobs,
    row_kind/row_a/row_b/row_c [R] row metadata (kind 0=cloze template/clsidx/
      compidx; kind 1=ordinal promptidx/setidx/rank),
    lp_sum/lp_first [C,L,D,Fnz,R] steered logprobs (Fnz = nonzero factors,
      factors_nz key; factor 0 slice == lp0),
    d_sum/d_first [C,L,D,F,T] cloze target-vs-sibling deltas,
    slope_sum/slope_first [C,L,D,T], slope_neg_sum [C,L,D,T],
    anti_frac_sum [C,L,D],
    ord_prompt_ids [P], ord_set_ids [S], erank_sum [C,L,D,F,P,S],
    ord_spearman_sum [C,L,D], ord_slope_sum [C,L,D,P,S],
    s95 [C,L] dose scale used.
  out/e2_cloze/offtarget_<family>.npz keys: steered [nS], steer_family,
    steer_layer [nS], steer_arms [2], readout_concepts [64],
    readout_families [64], readout_layers [64], meters [2] (ridge,dom),
    factor, base_mean [64,2], steered_mean [nS,2,64,2], delta [nS,2,64,2],
    subset_prompts [<=16].
  out/e2_cloze/summary.jsonl — one row per (concept, layer, arm, metric):
    {"concept","family","layer","arm","metric","value","n","ci_low","ci_high",
     "config":{...}} with metrics cloze_slope / anti_steerable_frac /
    suppression_slope / ordinal_spearman / ordinal_slope. Bootstrap CIs over
    templates (or prompt x set cells); rand rows aggregate the 5 dirs.

Layer-25 caveat: the L25 probe meter reads hidden_states[26] (post-final-
RMSNorm), so the exact-alpha steering identity holds for layers -1..24 only;
L25 rows are still measured and carry config.layer25_note.

--smoke: tiny random Gemma2 (test_interventions fixture pattern) + mock
word-hash tokenizer + synthetic arms/natstats/banks/cards; CPU fp32;
end-to-end through npz/summary/specificity. --dry-run: loads REAL arms,
dose_calib, banks (if present), prints planned work + estimated forwards,
no model forwards.

  python e2_cloze.py --families months --layers 6,12 --device cuda
  python e2_cloze.py --smoke
  python e2_cloze.py --dry-run --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                          # noqa: E402
from common import (FAMILIES, LAYERS, N_RAND_ARMS, dose_calib,  # noqa: E402
                    load_arms, load_natstats)
from interventions import Hooks, Intervention          # noqa: E402

STAGE_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = STAGE_DIR / "prompts"
CARDS_PATH = STAGE_DIR.parent / "stage6" / "artifacts" / "probe_cards.json"
DEFAULT_FACTORS = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6, 8]
FIT_RANGE = (-2.0, 2.0)
SUBSET_SEED, SUBSET_N = 61, 16
SCRIPT = "e2_cloze"


def canon(s: str) -> str:
    return str(s).replace(" ", "_")


def heartbeat(log: Path, msg: str):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {SCRIPT} {msg}\n")


def _jsonsafe(x):
    if isinstance(x, dict):
        return {k: _jsonsafe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonsafe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def append_summary(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(_jsonsafe(r)) + "\n")


def boot_ci(vals, seed=0, n=1000):
    """(mean, lo, hi) bootstrap CI over finite entries; nan-safe."""
    v = np.asarray(vals, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5))


def ols_slope(x, y):
    """OLS slope of y on x over finite pairs; nan if <2 points or x constant."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2 or np.ptp(x) == 0:
        return float("nan")
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


# --------------------------------------------------------------- smoke fixtures
class MockTok:
    """Deterministic word-hash tokenizer for the tiny (vocab 128) smoke model.
    Implements the surface used here: __call__ -> {'input_ids': [...]},
    bos_token_id, pad_token_id."""
    bos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        ids = [2 + (zlib.crc32(w.encode()) % 126) for w in text.split()]
        return {"input_ids": ids or [3]}


def tiny_model(d=64):
    """test_interventions.py fixture pattern: 2-layer random Gemma2, fp32."""
    from transformers import Gemma2Config, Gemma2ForCausalLM
    cfg = Gemma2Config(
        vocab_size=128, hidden_size=d, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=64, attn_implementation="eager")
    torch.manual_seed(0)
    model = Gemma2ForCausalLM(cfg)
    model.config.output_hidden_states = True
    model.eval()
    return model


SMOKE_CLOZE = {
    "family": "famA", "classes": ["alpha", "beta"],
    "templates": [
        {"id": "t1", "type": "cloze", "prompt": "the sky over the bay turned",
         "completions": {"alpha": ["bright crimson", "warm amber"],
                         "beta": ["cold gray"]}, "answer_class": None},
        {"id": "t2", "type": "cloze", "prompt": "she opened the box and found",
         "completions": {"alpha": ["a glowing ember"],
                         "beta": ["a chunk of ice", "frozen dew"]},
         "answer_class": "beta"},
        {"id": "t3", "type": "cloze", "prompt": "the report described it as",
         "completions": {"alpha": ["fiery"], "beta": ["icy"]},
         "answer_class": None}]}
SMOKE_ORD = {
    "axis": "axisB",
    "prompts": [{"id": "p1", "prompt": "overall the reviewer found the meal"},
                {"id": "p2", "prompt": "the inspector said the bridge was"}],
    "ordered_completion_sets": [
        {"id": "s1", "completions": ["awful", "fine", "superb"]}]}


class Ctx:
    """Data-access indirection so --smoke swaps every real dependency."""

    def __init__(self, args):
        self.args = args
        self.smoke = args.smoke
        if self.smoke:
            d = 64
            rng = np.random.default_rng(7)
            self._mu = rng.normal(size=d).astype(np.float32)
            self._sd = (0.5 + rng.random(d)).astype(np.float32)
            self.families = {"famA": ["alpha", "beta"], "axisB": ["axisb"]}
            self._dirs = {}
            for c in ["alpha", "beta", "axisb"]:
                g = np.random.default_rng(abs(zlib.crc32(c.encode())))
                self._dirs[c] = {
                    "ridge": (self._unit(g.normal(size=d)), 0.1),
                    "dom": (self._unit(g.normal(size=d)), 0.0),
                    "rand": [self._unit(g.normal(size=d)) for _ in range(5)]}
            self.cards = [{"concept": "alpha", "family": "famA", "layer": 0},
                          {"concept": "beta", "family": "famA", "layer": 0},
                          {"concept": "axisb", "family": "axisB", "layer": 0}]
        else:
            self.families = {f: [canon(c) for c in cs]
                             for f, cs in FAMILIES.items()}
            self.cards = [{"concept": canon(c["concept"]),
                           "family": c["family"], "layer": int(c["layer"])}
                          for c in json.load(open(CARDS_PATH))]
        self.card_layer = {(c["family"], c["concept"]): c["layer"]
                           for c in self.cards}

    @staticmethod
    def _unit(v):
        v = np.asarray(v, dtype=np.float32)
        return v / np.linalg.norm(v)

    def natstats(self, layer):
        return (self._mu, self._sd) if self.smoke else load_natstats(layer)

    def arms(self, family, cls, layer):
        return self._dirs[cls] if self.smoke else load_arms(family, cls, layer)

    def s95(self, family, cls, layer):
        if self.smoke:
            return 1.0
        return float(dose_calib(family, cls, layer)["s95"])

    def banks(self, family):
        """(cloze_bank | None, ordinal_bank | None) for a family."""
        if self.smoke:
            return (SMOKE_CLOZE if family == "famA" else None,
                    SMOKE_ORD if family == "axisB" else None)
        cz, od = None, None
        p = PROMPTS_DIR / f"{family}.cloze.json"
        if p.exists():
            cz = json.load(open(p))
            cz["classes"] = [canon(c) for c in cz["classes"]]
            for t in cz["templates"]:
                t["completions"] = {canon(k): v
                                    for k, v in t["completions"].items()}
                if t.get("answer_class"):
                    t["answer_class"] = canon(t["answer_class"])
        p = PROMPTS_DIR / f"{family}.ordinal.json"
        if p.exists():
            od = json.load(open(p))
        return cz, od

    def model_tok(self):
        if self.smoke:
            return tiny_model(), MockTok()
        return common.load_model(device=self.args.device,
                                 dtype=self.args.dtype)


# ------------------------------------------------------------------ row build
def build_rows(cloze, ordinal, tok, limit=None):
    """Tokenized rows for one family's bank(s).

    Returns dict with: seqs (list[list[int]]), plen/clen [R], kind [R]
    (0 cloze / 1 ordinal), a/b/c [R] (cloze: template/clsidx/compidx;
    ordinal: promptidx/setidx/rank), template meta, ordinal meta."""
    bos = tok.bos_token_id
    seqs, plen, clen, kind, ra, rb, rc = [], [], [], [], [], [], []
    template_ids, answer_class, bank_classes = [], [], []
    ord_prompt_ids, ord_set_ids, ord_set_sizes = [], [], []
    prompt_texts = []          # (sortkey, prompt) for the shared subset

    def add(pids, cids, k, a, b, c):
        seqs.append(pids + cids); plen.append(len(pids))
        clen.append(len(cids)); kind.append(k)
        ra.append(a); rb.append(b); rc.append(c)

    if cloze is not None:
        bank_classes = list(cloze["classes"])
        templates = cloze["templates"][:limit] if limit else cloze["templates"]
        for ti, t in enumerate(templates):
            template_ids.append(str(t["id"]))
            answer_class.append(t.get("answer_class") or "")
            prompt_texts.append(((cloze["family"], "cloze", str(t["id"])),
                                 t["prompt"]))
            pids = [bos] + tok(t["prompt"], add_special_tokens=False)["input_ids"]
            for cls, comps in t["completions"].items():
                if cls not in bank_classes:
                    continue
                for si, s in enumerate(comps):
                    cids = tok(" " + s, add_special_tokens=False)["input_ids"]
                    add(pids, cids, 0, ti, bank_classes.index(cls), si)
    if ordinal is not None:
        prompts = ordinal["prompts"][:limit] if limit else ordinal["prompts"]
        for pi, p in enumerate(prompts):
            ord_prompt_ids.append(str(p["id"]))
            prompt_texts.append(((ordinal["axis"], "ordinal", str(p["id"])),
                                 p["prompt"]))
            pids = [bos] + tok(p["prompt"], add_special_tokens=False)["input_ids"]
            for si, st in enumerate(ordinal["ordered_completion_sets"]):
                if pi == 0:
                    ord_set_ids.append(str(st["id"]))
                    ord_set_sizes.append(len(st["completions"]))
                for ki, s in enumerate(st["completions"]):
                    cids = tok(" " + s, add_special_tokens=False)["input_ids"]
                    add(pids, cids, 1, pi, si, ki)
    return dict(seqs=seqs, plen=np.array(plen), clen=np.array(clen),
                kind=np.array(kind), a=np.array(ra), b=np.array(rb),
                c=np.array(rc), template_ids=template_ids,
                answer_class=answer_class, bank_classes=bank_classes,
                ord_prompt_ids=ord_prompt_ids, ord_set_ids=ord_set_ids,
                ord_set_sizes=ord_set_sizes, prompt_texts=prompt_texts)


def pack(seqs, batch_tokens, pad_id):
    """Greedy length-sorted packing, B*Lmax <= batch_tokens (>=1 row/batch).
    Yields (row_indices, ids [B,L], attn [B,L])."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    groups, cur, maxlen = [], [], 0
    for i in order:
        m = max(maxlen, len(seqs[i]))
        if cur and m * (len(cur) + 1) > batch_tokens:
            groups.append(cur); cur, maxlen = [], 0
        cur.append(i); maxlen = max(maxlen, len(seqs[i]))
    if cur:
        groups.append(cur)
    out = []
    for idxs in groups:
        L = max(len(seqs[i]) for i in idxs)
        ids = torch.full((len(idxs), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(idxs), L), dtype=torch.long)
        for r, i in enumerate(idxs):
            ids[r, :len(seqs[i])] = torch.tensor(seqs[i], dtype=torch.long)
            attn[r, :len(seqs[i])] = 1
        out.append((idxs, ids, attn))
    return out


def token_logprobs(logits, ids):
    """fp32 [B, L-1] logprob of ids[:,1:], chunked so the fp32 upcast of the
    [chunk, L, V] logits stays bounded (~256M floats)."""
    B, L, V = logits.shape
    out = torch.empty(B, L - 1, dtype=torch.float32, device=logits.device)
    chunk = max(1, int(2 ** 28 / max((L - 1) * V, 1)))
    for b0 in range(0, B, chunk):
        lg = logits[b0:b0 + chunk, :-1].float()
        tgt = ids[b0:b0 + chunk, 1:].unsqueeze(-1)
        out[b0:b0 + chunk] = (lg.gather(-1, tgt).squeeze(-1)
                              - torch.logsumexp(lg, dim=-1))
    return out


@torch.no_grad()
def eval_batches(model, batches, plen, clen, device):
    """(sum_lp [R], first_lp [R]) of completion tokens under the CURRENT model
    state (hooks, if any, already registered by the caller)."""
    R = len(plen)
    s = np.full(R, np.nan, dtype=np.float32)
    f = np.full(R, np.nan, dtype=np.float32)
    for idxs, ids, attn in batches:
        out = model(ids.to(device), attention_mask=attn.to(device),
                    output_hidden_states=False)
        lp = token_logprobs(out.logits, ids.to(device)).cpu().numpy()
        for r, i in enumerate(idxs):
            p, c = int(plen[i]), int(clen[i])
            span = lp[r, p - 1:p - 1 + c]
            s[i] = span.sum(); f[i] = span[0]
    return s, f


# ------------------------------------------------------------------- metrics
def cloze_metrics(rows, factors, lp, lp0):
    """lp [C,L,D,F,R] (F includes factor 0), lp0 [R] -> dict of metric arrays.
    Per-template target-vs-sibling deltas + slopes."""
    kind, a, b = rows["kind"], rows["a"], rows["b"]
    Cb = len(rows["bank_classes"])
    T = len(rows["template_ids"])
    C, L, D, F, _ = lp.shape
    fac = np.asarray(factors, float)
    if T == 0 or Cb == 0:
        z = np.full((C, L, D, F, 0), np.nan, np.float32)
        return dict(d=z, slope=np.full((C, L, D, 0), np.nan, np.float32),
                    slope_neg=np.full((C, L, D, 0), np.nan, np.float32),
                    anti=np.full((C, L, D), np.nan, np.float32))
    # per (template, bank class) mean lp: M [C,L,D,F,T,Cb]
    M = np.full((C, L, D, F, T, Cb), np.nan, dtype=np.float32)
    M0 = np.full((T, Cb), np.nan, dtype=np.float32)
    for t in range(T):
        for cb in range(Cb):
            m = (kind == 0) & (a == t) & (b == cb)
            if m.any():
                M[..., t, cb] = lp[..., m].mean(axis=-1)
                M0[t, cb] = lp0[m].mean()
    d = np.full((C, L, D, F, T), np.nan, dtype=np.float32)
    for ci in range(C):
        cb = rows["steer_to_bank"][ci]
        if cb is None:
            continue
        sib = [j for j in range(Cb) if j != cb]
        with np.errstate(invalid="ignore"):
            tgt = M[ci, ..., cb]                         # [L,D,F,T]
            sb = np.nanmean(M[ci][..., sib], axis=-1)    # [L,D,F,T]
            tgt0, sb0 = M0[:, cb], np.nanmean(M0[:, sib], axis=-1)
        d[ci] = (tgt - sb) - (tgt0 - sb0)[None, None, None, :]
    fit = (fac >= FIT_RANGE[0]) & (fac <= FIT_RANGE[1])
    neg = fac <= 0
    slope = np.full((C, L, D, T), np.nan, dtype=np.float32)
    slope_neg = np.full((C, L, D, T), np.nan, dtype=np.float32)
    for idx in np.ndindex(C, L, D):
        for t in range(T):
            slope[idx + (t,)] = ols_slope(fac[fit], d[idx + (slice(None), t)][fit])
            slope_neg[idx + (t,)] = ols_slope(fac[neg], d[idx + (slice(None), t)][neg])
    with np.errstate(invalid="ignore"):
        fin = np.isfinite(slope)
        anti = np.where(fin.sum(-1) > 0,
                        (slope < 0).sum(-1) / np.maximum(fin.sum(-1), 1),
                        np.nan).astype(np.float32)
    return dict(d=d, slope=slope, slope_neg=slope_neg, anti=anti)


def ordinal_metrics(rows, factors, lp, lp0):
    """E[rank] per (prompt, set) + Spearman/slopes. lp [C,L,D,F,R]."""
    from scipy.stats import spearmanr
    kind, a, b, c = rows["kind"], rows["a"], rows["b"], rows["c"]
    P, S = len(rows["ord_prompt_ids"]), len(rows["ord_set_ids"])
    C, L, D, F, _ = lp.shape
    fac = np.asarray(factors, float)
    erank = np.full((C, L, D, F, P, S), np.nan, dtype=np.float32)
    for p in range(P):
        for s in range(S):
            m = (kind == 1) & (a == p) & (b == s)
            if not m.any():
                continue
            order = np.argsort(c[m])
            K = order.size
            rk = np.arange(K) / max(K - 1, 1)
            v = lp[..., m][..., order]                    # [C,L,D,F,K]
            v = v - v.max(axis=-1, keepdims=True)
            w = np.exp(v); w /= w.sum(axis=-1, keepdims=True)
            erank[..., p, s] = (w * rk).sum(axis=-1)
    sp = np.full((C, L, D), np.nan, dtype=np.float32)
    slopes = np.full((C, L, D, P, S), np.nan, dtype=np.float32)
    if P == 0 or S == 0:
        return dict(erank=erank, spearman=sp, slopes=slopes)
    fit = (fac >= FIT_RANGE[0]) & (fac <= FIT_RANGE[1])
    for idx in np.ndindex(C, L, D):
        mean_er = np.nanmean(erank[idx].reshape(F, -1), axis=1)
        if np.isfinite(mean_er).sum() >= 3:
            res = spearmanr(fac[np.isfinite(mean_er)],
                            mean_er[np.isfinite(mean_er)])
            sp[idx] = float(getattr(res, "statistic",
                                    getattr(res, "correlation", res[0])))
        for p in range(P):
            for s in range(S):
                slopes[idx + (p, s)] = ols_slope(fac[fit], erank[idx][fit, p, s])
    return dict(erank=erank, spearman=sp, slopes=slopes)


# ---------------------------------------------------------------- specificity
def shared_subset(ctx, tok):
    """Fixed shared prompt subset: all bank prompts across ALL families,
    sorted by (family, kind, id), seeded choice of SUBSET_N."""
    pool = []
    for fam in sorted(ctx.families):
        cz, od = ctx.banks(fam)
        pool += build_rows(cz, od, tok)["prompt_texts"] if (cz or od) else []
    pool.sort(key=lambda kv: kv[0])
    texts = [p for _, p in pool]
    if len(texts) > SUBSET_N:
        rng = np.random.default_rng(SUBSET_SEED)
        keep = sorted(rng.choice(len(texts), SUBSET_N, replace=False))
        texts = [texts[i] for i in keep]
    return texts


@torch.no_grad()
def readout_means(model, ids, attn, registry, device):
    """[nReadout, 2] mean probe score (meters ridge, dom) over non-BOS real
    tokens of the CURRENT (possibly hooked) forward."""
    out = model(ids.to(device), attention_mask=attn.to(device),
                output_hidden_states=True)
    mask = attn.to(device).bool()
    mask[:, 0] = False                       # exclude BOS (probe convention)
    res = np.zeros((len(registry), 2), dtype=np.float32)
    for i, r in enumerate(registry):
        for mi, meter in enumerate(("ridge", "dom")):
            w, bb = r[meter]
            s = common.probe_scores(out.hidden_states, r["layer"], w, bb,
                                    r["mu"], r["sigma"])
            res[i, mi] = float(s[mask].mean())
    return res


def run_specificity(ctx, model, tok, fam, concepts, subset_texts, out_dir,
                    factor, batch_tokens, device, log):
    if not subset_texts:
        print(f"[{SCRIPT}] {fam}: no bank prompts anywhere -> skip specificity")
        return
    registry = []
    for card in ctx.cards:
        arms = ctx.arms(card["family"], card["concept"], card["layer"])
        mu, sd = ctx.natstats(card["layer"])
        registry.append(dict(concept=card["concept"], family=card["family"],
                             layer=card["layer"], ridge=arms["ridge"],
                             dom=(arms["dom"][0], 0.0), mu=mu, sigma=sd))
    bos = tok.bos_token_id
    seqs = [[bos] + tok(t, add_special_tokens=False)["input_ids"]
            for t in subset_texts]
    (_, ids, attn), = pack(seqs, max(batch_tokens,
                                     max(len(s) for s in seqs) * len(seqs)),
                           tok.pad_token_id if tok.pad_token_id is not None else 0)
    base = readout_means(model, ids, attn, registry, device)
    steer_arms = ["ridge", "dom"]
    nS = len(concepts)
    steered = np.full((nS, 2, len(registry), 2), np.nan, dtype=np.float32)
    layers_used = np.zeros(nS, dtype=np.int64)
    for si, cls in enumerate(concepts):
        L = ctx.card_layer.get((fam, cls), 12)
        layers_used[si] = L
        alpha = factor * ctx.s95(fam, cls, L)
        arms = ctx.arms(fam, cls, L)
        mu, sd = ctx.natstats(L)
        for ai, arm in enumerate(steer_arms):
            iv = Intervention(L, arms[arm][0], "steer", alpha=alpha)
            with Hooks(model, [iv], {L: (mu, sd)}):
                steered[si, ai] = readout_means(model, ids, attn, registry,
                                                device)
        heartbeat(log, f"{fam}.{cls} specificity {si + 1}/{nS}")
    np.savez_compressed(
        out_dir / f"offtarget_{fam}.npz",
        steered=np.array(concepts), steer_family=fam,
        steer_layer=layers_used, steer_arms=np.array(steer_arms),
        readout_concepts=np.array([r["concept"] for r in registry]),
        readout_families=np.array([r["family"] for r in registry]),
        readout_layers=np.array([r["layer"] for r in registry]),
        meters=np.array(["ridge", "dom"]), factor=float(factor),
        base_mean=base, steered_mean=steered, delta=steered - base[None, None],
        subset_prompts=np.array(subset_texts))
    print(f"[{SCRIPT}] wrote {out_dir / f'offtarget_{fam}.npz'}")


# ------------------------------------------------------------------ main run
def dir_list(arms_sel):
    names = []
    for a in arms_sel:
        if a == "rand":
            names += [f"rand{k}" for k in range(N_RAND_ARMS)]
        else:
            names.append(a)
    return names


def get_dir(arms, name):
    if name.startswith("rand"):
        k = int(name[4:])
        return arms["rand"][min(k, len(arms["rand"]) - 1)]
    return arms[name][0]


def plan_family(ctx, fam, tok, args):
    """(rows, batches, concepts, n_forwards) — shared by run and dry-run.
    tok may be None (dry-run without tokenizer -> word-count estimate)."""
    cz, od = ctx.banks(fam)
    if cz is None and od is None:
        return None
    if tok is not None:
        rows = build_rows(cz, od, tok, args.limit)
        batches = pack(rows["seqs"], args.batch_tokens,
                       tok.pad_token_id if tok.pad_token_id is not None else 0)
    else:
        # count-only proxy (word-hash tokenizer approximates token counts)
        rows = build_rows(cz, od, MockTok(), args.limit)
        batches = pack(rows["seqs"], args.batch_tokens, 0)
    concepts = [c for c in ctx.families[fam]
                if not args.classes or c in args.classes]
    if rows["bank_classes"]:
        rows["steer_to_bank"] = [
            rows["bank_classes"].index(c) if c in rows["bank_classes"] else None
            for c in concepts]
    else:
        rows["steer_to_bank"] = [None] * len(concepts)
    n_dirs = len(dir_list(args.arms))
    n_nz = sum(1 for f in args.factors if f != 0)
    nf = len(batches) * (1 + len(concepts) * len(args.layers) * n_dirs * n_nz)
    return rows, batches, concepts, nf


def run_family(ctx, model, tok, fam, rows, batches, concepts, args, out_dir,
               log, device):
    factors = args.factors
    nz = [f for f in factors if f != 0]
    f0i = factors.index(0) if 0 in factors else None
    dirs = dir_list(args.arms)
    C, L, D, F = len(concepts), len(args.layers), len(dirs), len(factors)
    R = len(rows["seqs"])
    lp_sum = np.full((C, L, D, F, R), np.nan, dtype=np.float32)
    lp_first = np.full((C, L, D, F, R), np.nan, dtype=np.float32)
    s95_mat = np.full((C, L), np.nan, dtype=np.float32)

    lp0_s, lp0_f = eval_batches(model, batches, rows["plen"], rows["clen"],
                                device)
    if f0i is not None:
        lp_sum[:, :, :, f0i] = lp0_s
        lp_first[:, :, :, f0i] = lp0_f

    total = C * L
    pbar = tqdm(total=total * D * len(nz), desc=f"{SCRIPT} {fam}")
    step = 0
    for ci, cls in enumerate(concepts):
        for li, layer in enumerate(args.layers):
            s95 = ctx.s95(fam, cls, layer)
            s95_mat[ci, li] = s95
            arms = ctx.arms(fam, cls, layer)
            mu, sd = ctx.natstats(layer)
            for di, dname in enumerate(dirs):
                w = get_dir(arms, dname)
                for f in nz:
                    fi = factors.index(f)
                    iv = Intervention(layer, w, "steer", alpha=f * s95)
                    with Hooks(model, [iv], {layer: (mu, sd)}):
                        s, fr = eval_batches(model, batches, rows["plen"],
                                             rows["clen"], device)
                    lp_sum[ci, li, di, fi] = s
                    lp_first[ci, li, di, fi] = fr
                    pbar.update(1)
            step += 1
            heartbeat(log, f"{fam}.{cls} L{layer} {step}/{total}")
    pbar.close()

    cm_s = cloze_metrics(rows, factors, lp_sum, lp0_s)
    cm_f = cloze_metrics(rows, factors, lp_first, lp0_f)
    om = ordinal_metrics(rows, factors, lp_sum, lp0_s)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{fam}.npz",
        classes=np.array(concepts), bank_classes=np.array(rows["bank_classes"]),
        layers=np.array(args.layers), dirs=np.array(dirs),
        factors=np.array(factors, dtype=np.float32),
        factors_nz=np.array(nz, dtype=np.float32),
        template_ids=np.array(rows["template_ids"]),
        answer_class=np.array(rows["answer_class"]),
        row_kind=rows["kind"], row_a=rows["a"], row_b=rows["b"],
        row_c=rows["c"], lp0_sum=lp0_s, lp0_first=lp0_f,
        lp_sum=lp_sum, lp_first=lp_first,
        d_sum=cm_s["d"], d_first=cm_f["d"],
        slope_sum=cm_s["slope"], slope_first=cm_f["slope"],
        slope_neg_sum=cm_s["slope_neg"], anti_frac_sum=cm_s["anti"],
        ord_prompt_ids=np.array(rows["ord_prompt_ids"]),
        ord_set_ids=np.array(rows["ord_set_ids"]),
        erank_sum=om["erank"], ord_spearman_sum=om["spearman"],
        ord_slope_sum=om["slopes"], s95=s95_mat)
    print(f"[{SCRIPT}] wrote {out_dir / f'{fam}.npz'}")

    # ------------------------------------------------ summary.jsonl rows
    srows = []
    arm_groups = {a: ([a] if a != "rand" else
                      [d for d in dirs if d.startswith("rand")])
                  for a in args.arms}
    T = len(rows["template_ids"])
    PS = len(rows["ord_prompt_ids"]) * len(rows["ord_set_ids"])
    for ci, cls in enumerate(concepts):
        for li, layer in enumerate(args.layers):
            cfg = {"factors": factors, "fit_range": list(FIT_RANGE),
                   "s95": float(s95_mat[ci, li]), "variant": "sum_logprob",
                   "bank_classes": rows["bank_classes"]}
            if layer == 25:
                cfg["layer25_note"] = ("meter reads post-final-RMSNorm "
                                       "hidden_states[26]; exact-alpha "
                                       "identity does not hold at L25")
            for arm, members in arm_groups.items():
                dsel = [dirs.index(m) for m in members]
                base = dict(concept=cls, family=fam, layer=layer, arm=arm,
                            config=cfg)
                if T and rows["steer_to_bank"][ci] is not None:
                    sl = np.nanmean(cm_s["slope"][ci, li, dsel], axis=0)
                    m, lo, hi = boot_ci(sl, seed=ci * 997 + li)
                    srows.append({**base, "metric": "cloze_slope", "value": m,
                                  "n": T, "ci_low": lo, "ci_high": hi})
                    fin = np.isfinite(sl)
                    m2, lo2, hi2 = boot_ci((sl < 0)[fin].astype(float),
                                           seed=ci * 991 + li)
                    srows.append({**base, "metric": "anti_steerable_frac",
                                  "value": m2, "n": int(fin.sum()),
                                  "ci_low": lo2, "ci_high": hi2})
                    sn = np.nanmean(cm_s["slope_neg"][ci, li, dsel], axis=0)
                    m3, lo3, hi3 = boot_ci(sn, seed=ci * 983 + li)
                    srows.append({**base, "metric": "suppression_slope",
                                  "value": m3, "n": T, "ci_low": lo3,
                                  "ci_high": hi3})
                if PS:
                    sp = float(np.nanmean(om["spearman"][ci, li, dsel]))
                    srows.append({**base, "metric": "ordinal_spearman",
                                  "value": sp, "n": len(factors),
                                  "ci_low": float("nan"),
                                  "ci_high": float("nan")})
                    osl = np.nanmean(om["slopes"][ci, li, dsel], axis=0)
                    m4, lo4, hi4 = boot_ci(osl.ravel(), seed=ci * 977 + li)
                    srows.append({**base, "metric": "ordinal_slope",
                                  "value": m4, "n": PS, "ci_low": lo4,
                                  "ci_high": hi4})
    append_summary(out_dir / "summary.jsonl", srows)
    return srows


def dry_run(ctx, fams, args):
    print(f"[{SCRIPT}] DRY RUN — validating data plumbing, no model forwards")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(common.MODEL_NAME)
        print("  tokenizer: real gemma-2-2b tokenizer loaded")
    except Exception as e:                                   # noqa: BLE001
        tok = None
        print(f"  tokenizer unavailable ({e!r}) -> word-count token estimate")
    total_fwd, planned = 0, []
    for fam in fams:
        try:
            ctx.s95(fam, ctx.families[fam][0], args.layers[0])
            calib = "ok"
        except FileNotFoundError as e:
            print(f"  {fam}: dose_calib unavailable -> SKIP ({e})")
            continue
        n_arm_checks = 0
        for cls in ctx.families[fam]:
            for L in args.layers:
                arms = ctx.arms(fam, cls, L)
                assert arms["ridge"][0].ndim == 1
                ctx.s95(fam, cls, L)
                n_arm_checks += 1
        plan = plan_family(ctx, fam, tok, args)
        if plan is None:
            n_dirs = len(dir_list(args.arms))
            n_nz = sum(1 for f in args.factors if f != 0)
            C = len([c for c in ctx.families[fam]
                     if not args.classes or c in args.classes])
            assumed_T, assumed_rows = 40, 40 * max(len(ctx.families[fam]), 3) * 2
            est_tok = assumed_rows * 28
            n_batch = max(1, int(np.ceil(est_tok / args.batch_tokens)))
            nf = n_batch * (1 + C * len(args.layers) * n_dirs * n_nz)
            print(f"  {fam}: bank MISSING (prompts/ authored in parallel); "
                  f"calib {calib}, arms+s95 loaded x{n_arm_checks}; ASSUMED "
                  f"{assumed_T} templates -> ~{nf} forwards")
            total_fwd += nf
            continue
        rows, batches, concepts, nf = plan
        ntok = sum(len(s) for s in rows["seqs"])
        print(f"  {fam}: calib {calib}, arms+s95 loaded x{n_arm_checks}; "
              f"bank rows={len(rows['seqs'])} ({ntok} tok, "
              f"{len(batches)} batches), concepts={len(concepts)}, "
              f"forwards={nf}")
        total_fwd += nf
        planned.append(fam)
    n_spec = sum(len(ctx.families[f]) for f in fams
                 if f in ctx.families) * 2 + 1
    print(f"  specificity: ~{n_spec} extra 16-prompt forwards "
          f"(2 steer arms x concepts + baselines)")
    print(f"[{SCRIPT}] TOTAL estimated forwards: {total_fwd + n_spec} "
          f"(grid: layers={args.layers}, factors={args.factors}, "
          f"arms={args.arms})")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", default="",
                    help="comma list; default = all families with a bank")
    ap.add_argument("--classes", default="", help="comma filter (canonical)")
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--factors", default=",".join(map(str, DEFAULT_FACTORS)))
    ap.add_argument("--arms", default="ridge,dom,rand")
    ap.add_argument("--out", default=str(STAGE_DIR / "out" / SCRIPT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0,
                    help="max templates/prompts per family (0 = all)")
    ap.add_argument("--specificity-factor", type=float, default=2.0)
    ap.add_argument("--skip-specificity", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    args.layers = [int(x) for x in args.layers.split(",") if x != ""]
    args.factors = [float(x) for x in args.factors.split(",") if x != ""]
    args.factors = [int(f) if f == int(f) else f for f in args.factors]
    args.arms = [a for a in args.arms.split(",") if a]
    args.classes = [canon(c) for c in args.classes.split(",") if c]
    args.limit = args.limit or None
    if args.smoke:
        args.device, args.dtype = "cpu", "float32"
        if args.layers == LAYERS:
            args.layers = [0]           # exact-alpha layer on the 2-layer model
        if [float(f) for f in args.factors] == [float(f) for f in DEFAULT_FACTORS]:
            args.factors = [-2, -1, 0, 1, 2]
        args.out = str(STAGE_DIR / "out" / f"{SCRIPT}_smoke")
    return args


def main(argv=None):
    args = parse_args(argv)
    ctx = Ctx(args)
    if args.families:
        fams = [f for f in args.families.split(",") if f]
    else:
        fams = [f for f in sorted(ctx.families)
                if ctx.banks(f) != (None, None)] or sorted(ctx.families)
    fams = [f for f in fams if f != "glorptitude"] or fams
    if "glorptitude" in (args.families or ""):
        print(f"[{SCRIPT}] glorptitude skipped: no natscores -> no dose "
              "calibration (nonsense control)")
    if args.dry_run:
        dry_run(ctx, fams, args)
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir.parent / f"progress_{SCRIPT}.log"
    model, tok = ctx.model_tok()
    if not ctx.smoke:
        model.to(args.device)

    plans, total_fwd = {}, 0
    for fam in fams:
        plan = plan_family(ctx, fam, tok, args)
        if plan is None:
            print(f"[{SCRIPT}] {fam}: no bank file under {PROMPTS_DIR} -> skip")
            continue
        try:
            ctx.s95(fam, ctx.families[fam][0], args.layers[0])
        except FileNotFoundError as e:
            print(f"[{SCRIPT}] {fam}: {e} -> skip")
            continue
        plans[fam] = plan
        total_fwd += plan[3]
    n_spec = 0 if args.skip_specificity else \
        sum(len(p[2]) for p in plans.values()) * 2 + len(plans)
    print(f"[{SCRIPT}] planned: {len(plans)} families, estimated "
          f"{total_fwd} grid forwards + {n_spec} specificity forwards")
    if not plans:
        print(f"[{SCRIPT}] nothing to do (no banks/calib)")
        return 1

    subset = None if args.skip_specificity else shared_subset(ctx, tok)
    for fam, (rows, batches, concepts, _) in plans.items():
        run_family(ctx, model, tok, fam, rows, batches, concepts, args,
                   out_dir, log, args.device)
        if not args.skip_specificity:
            run_specificity(ctx, model, tok, fam, concepts, subset, out_dir,
                            args.specificity_factor, args.batch_tokens,
                            args.device, log)
    heartbeat(log, "DONE")
    print(f"[{SCRIPT}] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
