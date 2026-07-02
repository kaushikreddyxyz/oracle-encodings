"""E4 — ablation necessity + effective causal rank + selectivity-restore.

Spec: knowledge/concept_probes/task.md §6.1.6, conventions §6.1.1, contract
DESIGN.md. Uses the frozen harness (common.py + interventions.py) — this file
never modifies them.

Parts
  (a) Everywhere-ablation: Intervention(mode='ablate', t = natural-mean proj)
      at ALL probed layers simultaneously (default LAYERS; --all-blocks = all
      26 block outputs 0..25, direction/t borrowed from the NEAREST probed
      layer, natstats native per block — documented approximation).
      Arms: ridge, dom, rand0..rand4 (5 random dirs, separately), other0..2
      (3 fixed-seed other-family concept ridge directions, each ablated to ITS
      OWN natural mean). Metrics on held-out (nat_split='test') data:
        * diag_lp_delta   Δ mean logprob of concept-diagnostic next-tokens on
                          natural positives (positions where the diagnostic
                          token is the NEXT token).
        * bpt_delta       Δ mean logprob of ALL next tokens on positives, in
                          bits/token (Belrose-style damage; negative = worse).
        * cloze_acc_delta / cloze_target_lp_delta on class-keyed cloze
                          templates (answer_class == concept).
        * kl_neutral      do-no-harm guard: mean KL(p_clean ‖ p_ablated),
                          nats/token, on a fixed shared off-concept neutral
                          pool (other families' natural_random examples with
                          zero judged spans).
  (b) Effective causal rank (multiclass families months, weekdays, seasons,
      color_wheel, directions, moon_phases, continents): per layer, the
      family concept-subspace = top-k right-singular directions of the
      stacked UNIT-normalized DoM rows [C_fam, 2304] (standardized space);
      k = 1..C_fam−1, ablated at all intervened layers. Implemented as k
      COMPOSED rank-1 std-space ablations with orthonormal directions —
      sequential rank-1 projection with orthonormal dirs IS the exact rank-k
      projection (v_j·z unchanged by prior removals along v_i ⟂ v_j); this is
      verified numerically in --smoke (max err vs direct projection < 1e-4).
      Per-direction restore targets t_j are exact by linearity:
      t_j = E[v_j·z] = Σ_c coef_jc E[m_c·z] with coef from the SVD
      (v_j = Σ_c U[c,j]/S[j] · m_c) resp. QR (v_j = Σ_c Rinv[c,j] · a_c) and
      E[m_c·z] from natscores preds_dom / preds_rand means. Control: matched
      rank-k QR-orthonormalized random subspace (shared rand_dirs).
      Metrics: family-wide class-keyed cloze accuracy + family diagnostic
      next-token logprob on family positives, vs k.
  (c) Selectivity-restore: everywhere-ablate concept c (ridge arm), add back
      steer α·(σ⊙DoM_c) at the concept's Stage-6 chosen layer
      (stage6/artifacts/probe_cards.json), α = {0.5, 1, 2} × s95(chosen
      layer); recovery fraction of the cloze-accuracy deficit
      (acc_restore − acc_ablate) / (acc_clean − acc_ablate).

Data paths (verified on disk 2026-07-02)
  * Natural positives / neutrals: stage6/data/natural/eval/<family>.jsonl
    (built by stage6/code/build_natural_eval.py FROM
    stage4/data/<family>/judged/judged_nat.jsonl — same texts, judge spans
    pre-painted onto gemma tokens; fields text, nat_split, slice,
    targets {class: [[tok_idx, strength]]}). Positive for class c: nat_split
    =='test' AND max span strength for c ≥ 0.34 (task.md §6.1.4 ymax rule);
    if < MIN_POS examples, fall back to strength > 0 (flagged in config).
    Neutral pool: slice=='natural_random', ALL classes' spans empty,
    nat_split=='test', drawn from OTHER families' files; one fixed-seed
    shuffle shared fleet-wide, per-family pool = first --neutral-n after
    excluding the concept's own family (pools nearly identical across
    families by construction), dedup by text.
  * Ablation targets t: ridge → common.dose_calib (unit-w score units);
    dom → mean(natscores preds_dom)/‖W_dom‖ per (layer, class);
    rand → mean(natscores preds_rand) per (layer, dir) — both cached to
    out/e4/aux_t.<family>.npz. natscores verified: preds_dom = z@W_dom.T
    (no bias, non-unit W), preds_rand = z@rand_dirs.T (unit dirs, no bias)
    — stage6/code/score_natural.py proj().
  * Prompt banks (authored in parallel; loaders degrade gracefully if absent):
    prompts/<family>.tokens.json = {"classes": {class: {"surface": [...],
    "associates": [...]}}} (delivered A3 schema; bare {class: {...}} also
    accepted) → diagnostic token ids = FIRST token of each string, tokenized
    both bare and with a leading space (documented choice: multi-token
    surfaces contribute their first piece).
    prompts/<family>.cloze.json = {"templates": [{"prompt": str,
    "completions": {class: [surface, ...]}, "answer_class": str|null}, ...]}
    (delivered A3 schema; single-string completions and bare lists also
    accepted). E4 keeps only class-KEYED templates (answer_class != null —
    the class-agnostic half is E2's). A leading space is prepended to each
    completion (per prompts/README.md), each surface is scored by summed
    completion-token logprob and a class scores its BEST (max) surface.
    Ordinal-axis families (costliness, physical_size, lovingness, duration,
    harmfulness) ship .ordinal.json instead of .cloze.json → their cloze and
    restore metrics are skipped with a message (diag/bpt/KL still run).

Decisions / deviations (documented per contract)
  * Layer −1 (embedding stream) is SKIPPED everywhere: natstats has no stats
    for it and probes never read there; --all-blocks covers blocks 0..25
    (natstats mean_0..std_25). Recorded in every summary row's config.
  * With --all-blocks, non-probed blocks reuse the nearest probed layer's
    direction and t (ties → lower layer) with the block's OWN natstats μ,σ —
    an approximation (E0: adjacent-layer cosine ≈ 0.6), flagged in config.
  * summary.jsonl "layer" field is the string "all_probed" or "all_blocks26"
    for multi-layer interventions (schema says one layer; config carries the
    exact list). Restore rows use the concept's chosen (restore) layer.
  * glorptitude is excluded (no natscores ⇒ dose_calib raises; it is the
    nonsense control — cannot be ablated to a natural mean).
  * KL guard uses forward batches capped at KL_BATCH_TOKENS to bound the
    [N, 256k] logit tensors held for clean-vs-ablated comparison.

Outputs (out dir default out/e4/, smoke default out/e4_smoke/)
  <family>.ablate.npz : classes [C], arms [A], blocks [L],
      diag_lp_delta/bpt_delta/kl_neutral/cloze_acc_delta/cloze_target_lp_delta
      [C, A] (+ *_lo/_hi bootstrap CI95 where defined), cloze_acc_clean [C],
      n_pos [C], n_diag_ex [C, A? no — C], n_cloze [C], n_neutral [C],
      pos_fallback [C] (1 = strength>0 fallback used).
  <family>.rank.npz : ks [K], bases ['concept','random'],
      cloze_acc [K, 2], cloze_target_lp [K, 2], diag_lp_delta [K, 2],
      cloze_acc_clean, diag_n, singvals [n_probed_layers, C].
  <family>.restore.npz : classes [C], factors [3], chosen_layer [C],
      cloze_acc_clean [C], cloze_acc_ablate [C], cloze_acc_restore [C, 3],
      recovery_frac [C, 3], n_cloze [C].
  out/e4/summary.jsonl : one row per (concept, arm, metric) per DESIGN schema.
  out/progress_e4_ablate.log : heartbeat lines.

Usage
  python e4_ablate.py --smoke                      # tiny-Gemma2, synthetic, CPU
  python e4_ablate.py --dry-run                    # real data, plan + forward est., no model
  python e4_ablate.py --families months --device cuda
  python e4_ablate.py --all-blocks --skip rank --skip restore
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import (CP_DIR, LAYERS, OUT_DIR, STAGE_DIR,  # noqa: E402
                    batch_iter, dose_calib, load_arms, load_natstats)
from interventions import Hooks, Intervention  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kw):
        return x

PROMPTS_DIR = STAGE_DIR / "prompts"
EVAL_DIR = CP_DIR / "stage6" / "data" / "natural" / "eval"
NATSCORES_DIR = CP_DIR / "stage6" / "data" / "natscores"
PROBES_DIR = CP_DIR / "stage5" / "probes"
CARDS_PATH = CP_DIR / "stage6" / "artifacts" / "probe_cards.json"

RANK_FAMILIES = ["months", "weekdays", "seasons", "color_wheel",
                 "directions", "moon_phases", "continents"]
POS_STRENGTH = 0.34          # task.md §6.1.4 judge-positive threshold (ymax)
MIN_POS = 5                  # below this, fall back to strength > 0
N_OTHER = 3                  # other-concept specificity arms
N_RAND = 5                   # random-direction arms (separately)
RESTORE_FACTORS = (0.5, 1.0, 2.0)
KL_BATCH_TOKENS = 4096       # cap: [N, vocab] clean+ablated logits held at once
LOGP_CHUNK = 2048            # rows per fp32 log_softmax chunk
POOL_SEED = 20260702         # fixed shared neutral-pool / positives shuffle
BOOT_N = 1000
LN2 = math.log(2.0)


def canon(s: str) -> str:
    return str(s).strip().replace(" ", "_")


def heartbeat(log: Path, msg: str):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} e4_ablate {msg}\n")


def bootstrap_ci(vals: np.ndarray, seed: int = 0, n: int = BOOT_N):
    """Percentile bootstrap CI95 of the mean. Returns (lo, hi) or (nan, nan)."""
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n, vals.size))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


# ============================================================ prompt banks

def load_tokens_bank(path: Path, tok) -> dict[str, set[int]] | None:
    """{class: set of diagnostic token ids}. First token of each surface/
    associate string, tokenized bare AND with leading space."""
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    if "classes" in raw and isinstance(raw["classes"], dict):
        raw = raw["classes"]                       # delivered A3 schema
    out: dict[str, set[int]] = {}
    for cls, d in raw.items():
        if not isinstance(d, dict):
            continue                               # metadata keys (family, type)
        strings = list(d.get("surface", [])) + list(d.get("associates", []))
        ids: set[int] = set()
        for s in strings:
            for variant in (s, " " + s.lstrip()):
                enc = tok(variant, add_special_tokens=False)["input_ids"]
                if enc:
                    ids.add(int(enc[0]))
        out[canon(cls)] = ids
    return out


def load_cloze_bank(path: Path, classes: list[str]) -> list[dict] | None:
    """Normalized CLASS-KEYED templates: {'prompt': str, 'completions':
    {key: [' surface', ...]}, 'answer_class': str}. Tolerant to schema
    variants (see docstring); class-agnostic templates (answer_class null)
    are dropped — every E4 metric is class-keyed."""
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("templates", raw.get("cloze", []))
    out = []
    for t in raw:
        prompt = t.get("prompt") or t.get("template") or t.get("text")
        ans = t.get("answer_class")
        if not prompt or not ans:
            continue
        for marker in ("_____", "____", "___", "{blank}", "{answer}", "[BLANK]"):
            if marker in prompt:
                prompt = prompt.split(marker)[0]
                break
        prompt = prompt.rstrip()
        comps = t.get("completions") or t.get("options")
        if comps is None:
            continue
        if isinstance(comps, list):
            comps = dict(zip(classes, comps))       # aligned to class order
        norm = {}
        for k, v in comps.items():
            surfs = v if isinstance(v, list) else [v]
            norm[canon(k)] = [s if s.startswith((" ", "\n")) else " " + s
                              for s in surfs if s]
        ans = canon(ans)
        if ans not in norm or len(norm) < 2:
            continue
        out.append({"prompt": prompt, "completions": norm,
                    "answer_class": ans})
    return out or None


# ============================================================ forward math

def _next_token_logprobs(logits: torch.Tensor, ids: torch.Tensor,
                         attn: torch.Tensor):
    """(lp [B,T-1] fp32, valid [B,T-1] bool): logprob of the ACTUAL next
    token at each position; fp32 log_softmax in row chunks."""
    B, T, V = logits.shape
    dev = logits.device
    tgt = ids[:, 1:].to(dev)
    valid = (attn[:, 1:].bool() & attn[:, :-1].bool()).to(dev)
    flat = logits[:, :-1, :].reshape(-1, V)
    ft = tgt.reshape(-1)
    outs = []
    for s in range(0, flat.shape[0], LOGP_CHUNK):
        sl = flat[s:s + LOGP_CHUNK].to(torch.float32)
        lse = torch.logsumexp(sl, dim=-1)
        outs.append(sl.gather(1, ft[s:s + LOGP_CHUNK, None]).squeeze(1) - lse)
    return torch.cat(outs).reshape(B, T - 1), valid


def _kl_rows(clean_flat: torch.Tensor, abl_flat: torch.Tensor) -> torch.Tensor:
    """KL(p_clean ‖ p_ablated) per row, fp32 chunked. Inputs [N, V] any dtype."""
    outs = []
    for s in range(0, clean_flat.shape[0], LOGP_CHUNK):
        c = clean_flat[s:s + LOGP_CHUNK].to(torch.float32)
        a = abl_flat[s:s + LOGP_CHUNK].to(torch.float32)
        lc = c - torch.logsumexp(c, dim=-1, keepdim=True)
        la = a - torch.logsumexp(a, dim=-1, keepdim=True)
        outs.append((lc.exp() * (lc - la)).sum(dim=-1))
    return torch.cat(outs) if outs else torch.zeros(0)


def _forward(model, ids, attn, device):
    return model(input_ids=ids.to(device), attention_mask=attn.to(device),
                 output_hidden_states=False, use_cache=False).logits


def positives_lp(model, tok, device, texts, diag_ids, arm_ivs: dict,
                 mu_sigma, batch_tokens, desc=""):
    """Per arm (+ 'clean'): per-example mean all-token logprob and mean
    diagnostic-next-token logprob on ``texts``.

    Returns {name: {'lp_mean': [n_ex], 'diag_mean': [n_ex] (nan if no diag
    position in that example)}}; diag_ids None → diag_mean all-nan."""
    n_ex = len(texts)
    names = ["clean"] + list(arm_ivs)
    acc = {nm: {"lp": np.zeros(n_ex), "n": np.zeros(n_ex),
                "dlp": np.zeros(n_ex), "dn": np.zeros(n_ex)} for nm in names}
    diag_t = (torch.tensor(sorted(diag_ids), dtype=torch.long)
              if diag_ids else None)
    batches = list(batch_iter(texts, tok, max_tokens=batch_tokens))
    for idx, ids, attn in tqdm(batches, desc=f"pos {desc}", leave=False):
        tgt = ids[:, 1:]
        dm = (torch.isin(tgt, diag_t) if diag_t is not None
              else torch.zeros_like(tgt, dtype=torch.bool))
        for nm in names:
            cm = (Hooks(model, arm_ivs[nm], mu_sigma) if nm != "clean"
                  else nullcontext())
            with torch.inference_mode(), cm:
                logits = _forward(model, ids, attn, device)
            lp, valid = _next_token_logprobs(logits, ids, attn)
            lp, valid = lp.cpu(), valid.cpu()
            dmask = dm & valid
            for r, ex in enumerate(idx):
                a = acc[nm]
                a["lp"][ex] += float(lp[r][valid[r]].sum())
                a["n"][ex] += int(valid[r].sum())
                a["dlp"][ex] += float(lp[r][dmask[r]].sum())
                a["dn"][ex] += int(dmask[r].sum())
            del logits
    out = {}
    for nm in names:
        a = acc[nm]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[nm] = {"lp_mean": np.where(a["n"] > 0, a["lp"] / a["n"], np.nan),
                       "diag_mean": np.where(a["dn"] > 0, a["dlp"] / a["dn"],
                                             np.nan)}
    return out


def neutral_kl(model, tok, device, texts, arm_ivs: dict, mu_sigma,
               batch_tokens, desc=""):
    """{arm: per-example mean KL(clean‖ablated) [n_ex]} on neutral texts."""
    n_ex = len(texts)
    acc = {nm: {"kl": np.zeros(n_ex), "n": np.zeros(n_ex)} for nm in arm_ivs}
    bt = min(batch_tokens, KL_BATCH_TOKENS)
    batches = list(batch_iter(texts, tok, max_tokens=bt))
    for idx, ids, attn in tqdm(batches, desc=f"kl {desc}", leave=False):
        with torch.inference_mode():
            clean = _forward(model, ids, attn, device)
        valid = (attn[:, 1:].bool() & attn[:, :-1].bool()).to(clean.device)
        clean_flat = clean[:, :-1, :][valid]                    # [N, V]
        exmap = torch.tensor(idx)[:, None].expand(-1, valid.shape[1])
        exmap = exmap[valid.cpu()].numpy()
        del clean
        for nm, ivs in arm_ivs.items():
            with torch.inference_mode(), Hooks(model, ivs, mu_sigma):
                abl = _forward(model, ids, attn, device)
            kl = _kl_rows(clean_flat, abl[:, :-1, :][valid]).cpu().numpy()
            del abl
            np.add.at(acc[nm]["kl"], exmap, kl)
            np.add.at(acc[nm]["n"], exmap, 1)
        del clean_flat
    out = {}
    for nm in arm_ivs:
        a = acc[nm]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[nm] = np.where(a["n"] > 0, a["kl"] / a["n"], np.nan)
    return out


# ------------------------------------------------------------------- cloze

def _pack_rows(rows_ids: list[list[int]], pad: int, max_tokens: int):
    order = sorted(range(len(rows_ids)), key=lambda i: len(rows_ids[i]))
    batch: list[int] = []
    maxlen = 0
    def emit():
        m = max(len(rows_ids[i]) for i in batch)
        ids = torch.full((len(batch), m), pad, dtype=torch.long)
        attn = torch.zeros((len(batch), m), dtype=torch.long)
        for r, i in enumerate(batch):
            seq = rows_ids[i]
            ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[r, :len(seq)] = 1
        return list(batch), ids, attn
    for i in order:
        nm = max(maxlen, len(rows_ids[i]))
        if batch and nm * (len(batch) + 1) > max_tokens:
            yield emit()
            batch, maxlen = [], 0
        batch.append(i)
        maxlen = max(maxlen, len(rows_ids[i]))
    if batch:
        yield emit()


def cloze_lp(model, tok, device, bank: list[dict], arm_ivs: dict, mu_sigma,
             batch_tokens, desc="", clean_map: dict | None = None):
    """{name: {(template_i, key): summed completion logprob}} for
    name in ['clean'] + arms; a class key with several surface strings
    scores its BEST (max-logprob) surface. Row = BOS + prompt + completion.
    ``clean_map``: precomputed clean scores (arm-independent, per family) —
    skips the clean pass and returns it under 'clean' unchanged."""
    rows, meta = [], []                       # ids, (ti, key, comp_start, comp_len)
    bos, pad = tok.bos_token_id, (tok.pad_token_id or 0)
    for ti, t in enumerate(bank):
        p_ids = tok(t["prompt"], add_special_tokens=False)["input_ids"]
        for key, surfs in t["completions"].items():
            for comp in (surfs if isinstance(surfs, list) else [surfs]):
                c_ids = tok(comp, add_special_tokens=False)["input_ids"]
                if not c_ids:
                    continue
                rows.append([bos] + p_ids + c_ids)
                meta.append((ti, key, 1 + len(p_ids), len(c_ids)))
    names = ((["clean"] if clean_map is None else []) + list(arm_ivs))
    out = {nm: {} for nm in names}
    if clean_map is not None:
        out["clean"] = clean_map
    for ridx, ids, attn in tqdm(list(_pack_rows(rows, pad, batch_tokens)),
                                desc=f"cloze {desc}", leave=False):
        for nm in names:
            cm = (Hooks(model, arm_ivs[nm], mu_sigma) if nm != "clean"
                  else nullcontext())
            with torch.inference_mode(), cm:
                logits = _forward(model, ids, attn, device)
            lp, _ = _next_token_logprobs(logits, ids, attn)
            lp = lp.cpu()
            for r, i in enumerate(ridx):
                ti, key, cs, cl = meta[i]
                v = float(lp[r, cs - 1:cs - 1 + cl].sum())
                prev = out[nm].get((ti, key))
                out[nm][(ti, key)] = v if prev is None else max(prev, v)
            del logits
    return out


def cloze_metrics(lp_map: dict, bank: list[dict], target_cls: str | None):
    """(acc, mean target lp, n, correct list) over templates with
    answer_class == target_cls (or any non-null if target_cls is None)."""
    correct, tlps = [], []
    for ti, t in enumerate(bank):
        ans = t["answer_class"]
        if ans is None or (target_cls is not None and ans != target_cls):
            continue
        scores = {k: lp_map.get((ti, k)) for k in t["completions"]}
        scores = {k: v for k, v in scores.items() if v is not None}
        if ans not in scores or len(scores) < 2:
            continue
        best = max(scores, key=scores.get)
        correct.append(1.0 if best == ans else 0.0)
        tlps.append(scores[ans])
    n = len(correct)
    return ((float(np.mean(correct)) if n else float("nan")),
            (float(np.mean(tlps)) if n else float("nan")), n,
            np.array(correct))


# ============================================================ real context

class RealCtx:
    """Loads everything from repo data. tokenizer/model are None until
    load_model() is called (dry-run never calls it)."""

    def __init__(self, args):
        self.args = args
        want = (args.families.split(",") if args.families else
                sorted(common.FAMILIES))
        self.families: dict[str, list[str]] = {}
        self.skipped: list[str] = []
        for f in want:
            if f not in common.FAMILIES:
                raise SystemExit(f"unknown family {f!r}; have "
                                 f"{sorted(common.FAMILIES)}")
            if not (NATSCORES_DIR / f"{f}.natscores.npz").exists():
                self.skipped.append(f)      # glorptitude: no natural-mean t
                continue
            cs = [canon(c) for c in common.FAMILIES[f]]
            if args.classes:
                keep = {canon(c) for c in args.classes.split(",")}
                cs = [c for c in cs if c in keep]
            if cs:
                self.families[f] = cs
        probed = ([int(x) for x in args.layers.split(",")] if args.layers
                  else list(LAYERS))
        self.probed = probed
        self.blocks = list(range(26)) if args.all_blocks else list(probed)
        self.probed_src = {L: min(probed, key=lambda p: (abs(p - L), p))
                           for L in self.blocks}
        self.mu_sigma = {L: load_natstats(L) for L in self.blocks}
        self.rank_families = [f for f in self.families if f in RANK_FAMILIES
                              and len(self.families[f]) >= 3]
        self.model = self.tok = None
        self.device = args.device
        self._eval_rows: dict[str, list] = {}
        self._aux: dict[str, dict] = {}
        self._arms_cache: dict = {}
        self._cards = None
        self._neutral_order = None
        self._tokens_bank: dict = {}
        self._cloze_bank: dict = {}

    def load_model(self):
        self.model, self.tok = common.load_model(
            device=self.args.device, dtype=self.args.dtype)

    # ------------------------------------------------------------- probes
    def arms(self, fam, cls, layer):
        k = (fam, cls, layer)
        if k not in self._arms_cache:
            self._arms_cache[k] = load_arms(fam, cls, layer)
        return self._arms_cache[k]

    def calib(self, fam, cls, layer):
        return dose_calib(fam, cls, layer)

    def _aux_t(self, fam):
        """{'layers': [12], 'dom_t': [12, C], 'rand_t': [12, n_rand]} —
        natural-mean projections for unit dom / rand dirs (cached npz)."""
        if fam in self._aux:
            return self._aux[fam]
        cache = Path(self.args.out) / f"aux_t.{fam}.npz"
        if cache.exists():
            z = np.load(cache)
            self._aux[fam] = {k: z[k] for k in z.files}
            return self._aux[fam]
        with np.load(NATSCORES_DIR / f"{fam}.natscores.npz") as z:
            layers = [int(x) for x in z["layers"]]
            nat_classes = [canon(c) for c in z["classes"]]
            dom_t = z["preds_dom"].mean(axis=1)          # [12, C] non-unit W
            rand_t = z["preds_rand"].mean(axis=2)        # [12, n_rand]
        for li, L in enumerate(layers):
            with np.load(PROBES_DIR / fam / f"probes_l{L}.npz") as p:
                p_classes = [canon(c) for c in p["classes"]]
                norms = np.linalg.norm(p["W_dom"], axis=1)
            order = [p_classes.index(c) for c in nat_classes]
            dom_t[li] /= norms[order]
        aux = {"layers": np.array(layers), "classes": np.array(nat_classes),
               "dom_t": dom_t.astype(np.float32),
               "rand_t": rand_t.astype(np.float32)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **aux)
        self._aux[fam] = aux
        return aux

    def dom_t(self, fam, layer):
        a = self._aux_t(fam)
        li = list(a["layers"]).index(layer)
        classes = [str(c) for c in a["classes"]]
        return {c: float(a["dom_t"][li, ci]) for ci, c in enumerate(classes)}

    def rand_t(self, fam, layer):
        a = self._aux_t(fam)
        return a["rand_t"][list(a["layers"]).index(layer)]

    def dom_stack(self, fam, layer):
        """(M [C, d] unit-norm DoM rows in self.families[fam] class order,
        t [C] natural-mean projections)."""
        rows, ts = [], []
        dt = self.dom_t(fam, layer)
        for c in self.families[fam]:
            rows.append(self.arms(fam, c, layer)["dom"][0])
            ts.append(dt[c])
        return np.stack(rows), np.array(ts, dtype=np.float64)

    def rand_set(self, fam, layer):
        c0 = self.families[fam][0]
        R = np.stack(self.arms(fam, c0, layer)["rand"])
        # load_arms exposes N_RAND_ARMS dirs; need up to C-1 (<=11) for QR —
        # pull the full saved set directly.
        with np.load(PROBES_DIR / fam / f"probes_l{layer}.npz") as z:
            R = z["rand_dirs"].astype(np.float32)
        return R, self.rand_t(fam, layer).astype(np.float64)

    def other_concepts(self, fam, cls):
        pool = [(f, c) for f in sorted(self.all_scored_families())
                if f != fam for c in self.family_classes_full(f)]
        rng = np.random.default_rng(stable_seed("e4-other", fam, cls))
        pick = rng.choice(len(pool), size=N_OTHER, replace=False)
        return [pool[i] for i in pick]

    def all_scored_families(self):
        return [f for f in common.FAMILIES
                if (NATSCORES_DIR / f"{f}.natscores.npz").exists()]

    def family_classes_full(self, fam):
        return [canon(c) for c in common.FAMILIES[fam]]

    def chosen_layer(self, fam, cls):
        if self._cards is None:
            self._cards = {(c["family"], canon(c["concept"])): int(c["layer"])
                           for c in json.loads(CARDS_PATH.read_text())}
        return self._cards[(fam, canon(cls))]

    # --------------------------------------------------------------- data
    def _rows(self, fam):
        if fam not in self._eval_rows:
            path = EVAL_DIR / f"{fam}.jsonl"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing — run stage6/code/build_natural_eval.py")
            self._eval_rows[fam] = [json.loads(l) for l in open(path)]
        return self._eval_rows[fam]

    @staticmethod
    def _strength(row, cls):
        for c, spans in row["targets"].items():
            if canon(c) == cls:
                return max((s for _, s in spans), default=0.0)
        return 0.0

    def positives(self, fam, cls):
        """(texts, fallback_used). Held-out test positives, fixed shuffle,
        capped at --limit."""
        rows = [r for r in self._rows(fam) if r.get("nat_split") == "test"]
        pos = [r for r in rows if self._strength(r, cls) >= POS_STRENGTH]
        fallback = False
        if len(pos) < MIN_POS:
            pos = [r for r in rows if self._strength(r, cls) > 0]
            fallback = True
        pos.sort(key=lambda r: r["example_id"])
        rng = np.random.default_rng(POOL_SEED)
        rng.shuffle(pos)
        return [r["text"] for r in pos[:self.args.limit]], fallback

    def family_positives(self, fam):
        rows = [r for r in self._rows(fam) if r.get("nat_split") == "test"
                and any(self._strength(r, c) >= POS_STRENGTH
                        for c in self.families[fam])]
        rows.sort(key=lambda r: r["example_id"])
        rng = np.random.default_rng(POOL_SEED)
        rng.shuffle(rows)
        return [r["text"] for r in rows[:self.args.limit]]

    def _neutral_all(self):
        """Fixed-seed shuffled [(family, text)] of every family's clean
        natural_random test examples, deduped by text."""
        if self._neutral_order is None:
            items, seen = [], set()
            for f in self.all_scored_families():
                try:
                    rows = self._rows(f)
                except FileNotFoundError:
                    continue
                for r in rows:
                    if (r.get("slice") == "natural_random"
                            and r.get("nat_split") == "test"
                            and all(not v for v in r["targets"].values())
                            and r["text"] not in seen):
                        seen.add(r["text"])
                        items.append((f, r["text"]))
            items.sort(key=lambda ft: ft[1])
            rng = np.random.default_rng(POOL_SEED)
            rng.shuffle(items)
            self._neutral_order = items
        return self._neutral_order

    def neutrals(self, fam):
        return [t for f, t in self._neutral_all()
                if f != fam][:self.args.neutral_n]

    # -------------------------------------------------------------- banks
    def diag_ids(self, fam, cls):
        if fam not in self._tokens_bank:
            self._tokens_bank[fam] = load_tokens_bank(
                PROMPTS_DIR / f"{fam}.tokens.json", self.tok)
        bank = self._tokens_bank[fam]
        return bank.get(cls) if bank else None

    def family_diag_ids(self, fam):
        ids = set()
        for c in self.families[fam]:
            d = self.diag_ids(fam, c)
            if d:
                ids |= d
        return ids or None

    def cloze(self, fam):
        if fam not in self._cloze_bank:
            self._cloze_bank[fam] = load_cloze_bank(
                PROMPTS_DIR / f"{fam}.cloze.json", self.families[fam])
        return self._cloze_bank[fam]


# =========================================================== smoke context

class ToyTokenizer:
    """Deterministic char-level tokenizer for the tiny random Gemma2
    (vocab 128): id = 2 + ord(char) % 120; bos=1, pad=0."""
    bos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": [2 + (ord(ch) % 120) for ch in text]}


def _tiny_model(d=64):
    from transformers import Gemma2Config, Gemma2ForCausalLM
    cfg = Gemma2Config(
        vocab_size=128, hidden_size=d, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=512,
        attn_implementation="eager")
    torch.manual_seed(0)
    model = Gemma2ForCausalLM(cfg)
    model.eval()
    return model


class SmokeCtx:
    """Synthetic everything on a 2-layer random Gemma2 (fp32, CPU): two
    families (famA 4 classes → rank part, famB 1 class), synthetic
    arms/calib/aux-t, char-level prompt banks and text pools."""

    D = 64

    def __init__(self, args):
        self.args = args
        self.model = _tiny_model(self.D)
        self.tok = ToyTokenizer()
        self.device = "cpu"
        self.families = {"famA": ["a", "b", "c", "d"], "famB": ["x"]}
        self.skipped = []
        self.probed = [0, 1]
        self.blocks = [0, 1]
        self.probed_src = {0: 0, 1: 1}
        self.rank_families = ["famA"]
        rng = np.random.default_rng(7)
        self.mu_sigma = {l: (rng.normal(size=self.D).astype(np.float32),
                             (0.5 + rng.random(self.D)).astype(np.float32))
                         for l in self.blocks}
        self._dirs: dict = {}
        self._rand = np.linalg.qr(rng.normal(size=(self.D, 8)))[0].T \
            .astype(np.float32)                       # 8 orthonormal-ish dirs
        self._rng = rng
        self._diag_char = {"a": "A", "b": "B", "c": "C", "d": "D", "x": "X"}

    def load_model(self):
        pass

    def _dir(self, key):
        if key not in self._dirs:
            v = self._rng.normal(size=self.D).astype(np.float32)
            self._dirs[key] = v / np.linalg.norm(v)
        return self._dirs[key]

    def arms(self, fam, cls, layer):
        return {"ridge": (self._dir((fam, cls, layer, "ridge")), 0.0),
                "dom": (self._dir((fam, cls, layer, "dom")), 0.0),
                "lda": (self._dir((fam, cls, layer, "lda")), 0.0),
                "rand": [self._rand[k] for k in range(N_RAND)]}

    def calib(self, fam, cls, layer):
        return {"s95": 1.2, "t": 0.1}

    def dom_t(self, fam, layer):
        return {c: 0.05 * (i + 1) for i, c in enumerate(self.families[fam])}

    def rand_t(self, fam, layer):
        return np.linspace(-0.1, 0.1, self._rand.shape[0])

    def dom_stack(self, fam, layer):
        dt = self.dom_t(fam, layer)
        return (np.stack([self.arms(fam, c, layer)["dom"][0]
                          for c in self.families[fam]]),
                np.array([dt[c] for c in self.families[fam]]))

    def rand_set(self, fam, layer):
        return self._rand, np.asarray(self.rand_t(fam, layer), dtype=np.float64)

    def other_concepts(self, fam, cls):
        pool = [(f, c) for f in self.families if f != fam
                for c in self.families[f]] * N_OTHER
        return pool[:N_OTHER]

    def all_scored_families(self):
        return list(self.families)

    def family_classes_full(self, fam):
        return self.families[fam]

    def chosen_layer(self, fam, cls):
        return 0

    def positives(self, fam, cls):
        ch = self._diag_char[cls]
        rng = np.random.default_rng(stable_seed("smoke-pos", fam, cls))
        texts = ["".join(rng.choice(list("qwerty "), size=20)) + f" {ch} tail"
                 + ch for _ in range(6)]
        return texts, False

    def family_positives(self, fam):
        out = []
        for c in self.families[fam]:
            out += self.positives(fam, c)[0][:2]
        return out

    def neutrals(self, fam):
        rng = np.random.default_rng(stable_seed("smoke-neu", fam))
        return ["".join(rng.choice(list("zxcvbnm, "), size=24))
                for _ in range(5)]

    def diag_ids(self, fam, cls):
        return {self.tok(self._diag_char[cls])["input_ids"][0]}

    def family_diag_ids(self, fam):
        ids = set()
        for c in self.families[fam]:
            ids |= self.diag_ids(fam, c)
        return ids

    def cloze(self, fam):
        bank = []
        for i, c in enumerate(self.families[fam]):
            # delivered A3 schema: list of surfaces per class key
            comps = {k: [" " + self._diag_char[k], " " + self._diag_char[k]
                         + "z"] for k in self.families[fam]}
            if len(comps) < 2:                       # binary family: distractor
                comps["not_" + c] = [" Z"]
            bank.append({"prompt": f"probe {i} says", "completions": comps,
                         "answer_class": c})
        return bank


# ======================================================= intervention build

def build_arm_ivs(ctx, fam, cls, arm_names, space):
    """{arm_name: [Intervention per intervened block]} for everywhere-ablation.
    Non-probed blocks (--all-blocks) borrow direction/t from the nearest
    probed layer (ctx.probed_src)."""
    others = None
    out = {}
    for arm in arm_names:
        ivs = []
        for L in ctx.blocks:
            src = ctx.probed_src[L]
            if arm == "ridge":
                w = ctx.arms(fam, cls, src)["ridge"][0]
                t = ctx.calib(fam, cls, src)["t"]
            elif arm == "dom":
                w = ctx.arms(fam, cls, src)["dom"][0]
                t = ctx.dom_t(fam, src)[cls]
            elif arm.startswith("rand"):
                k = int(arm[4:])
                w = ctx.arms(fam, cls, src)["rand"][k]
                t = float(ctx.rand_t(fam, src)[k])
            elif arm.startswith("other"):
                if others is None:
                    others = ctx.other_concepts(fam, cls)
                ofam, ocls = others[int(arm[5:])]
                w = ctx.arms(ofam, ocls, src)["ridge"][0]
                t = ctx.calib(ofam, ocls, src)["t"]
            else:
                raise ValueError(f"unknown arm {arm!r}")
            ivs.append(Intervention(L, w, "ablate", t=float(t), space=space))
        out[arm] = ivs
    return out, others


def subspace_dirs(ctx, fam, layer, k, basis):
    """(V [k, d] orthonormal, t [k]) for the rank-k subspace at one layer.
    basis 'concept': top-k right-singular dirs of the unit-DoM stack, t by
    exact linearity through the SVD; 'random': QR-orthonormalized first k
    shared random dirs, t through R^{-1}. Also returns singular values for
    'concept'."""
    if basis == "concept":
        M, t_m = ctx.dom_stack(fam, layer)               # [C,d], [C]
        U, S, Vt = np.linalg.svd(M.astype(np.float64), full_matrices=False)
        V = Vt[:k]
        coef = (U[:, :k] / S[:k]).T                      # [k, C]: v_j = Σ coef m_c
        t = coef @ t_m
        return V.astype(np.float32), t, S
    R, t_r = ctx.rand_set(fam, layer)
    A = R[:k].astype(np.float64)                         # [k, d]
    Q, Rm = np.linalg.qr(A.T)                            # A.T = Q Rm
    Rinv = np.linalg.inv(Rm)
    t = Rinv.T @ t_r[:k]                                 # v_j = Σ Rinv[c,j] a_c
    return Q.T.astype(np.float32), t, None


def build_rank_ivs(ctx, fam, k, basis):
    ivs, sv = [], None
    for L in ctx.blocks:
        src = ctx.probed_src[L]
        V, t, S = subspace_dirs(ctx, fam, src, k, basis)
        if S is not None:
            sv = S
        for j in range(k):
            ivs.append(Intervention(L, V[j], "ablate", t=float(t[j])))
    return ivs, sv


# ============================================================== summary io

def append_summary(out_dir: Path, rows: list[dict]):
    with open(out_dir / "summary.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def base_config(ctx, args):
    return {"blocks": list(ctx.blocks), "probed": list(ctx.probed),
            "all_blocks": bool(args.all_blocks), "space": args.space,
            "pos_strength": POS_STRENGTH, "limit": args.limit,
            "neutral_n": args.neutral_n, "layer_minus1": "skipped(no natstats)",
            "nonprobed_blocks": "nearest-probed direction+t"
            if args.all_blocks else "n/a",
            "data": "stage6/data/natural/eval/<fam>.jsonl (from stage4 "
                    "judged_nat.jsonl)"}


# ================================================================= runners

def expand_arm_names(spec: str) -> list[str]:
    out = []
    for a in spec.split(","):
        a = a.strip()
        if a == "rand":
            out += [f"rand{k}" for k in range(N_RAND)]
        elif a == "other":
            out += [f"other{j}" for j in range(N_OTHER)]
        elif a:
            out.append(a)
    return out


def run_ablate(ctx, args, out_dir: Path, log: Path):
    layer_label = "all_blocks26" if args.all_blocks else "all_probed"
    arm_names = expand_arm_names(args.arms)
    cfg0 = base_config(ctx, args)
    for fam, classes in ctx.families.items():
        bank = ctx.cloze(fam)
        cloze_clean_map = None
        A = len(arm_names)
        C = len(classes)
        res = {m: np.full((C, A), np.nan) for m in
               ("diag_lp_delta", "diag_lo", "diag_hi", "bpt_delta", "bpt_lo",
                "bpt_hi", "kl_neutral", "kl_lo", "kl_hi", "cloze_acc_delta",
                "cloze_target_lp_delta")}
        cloze_acc_clean = np.full(C, np.nan)
        n_pos = np.zeros(C, int)
        n_cloze = np.zeros(C, int)
        n_neutral = np.zeros(C, int)
        pos_fallback = np.zeros(C, int)
        rows = []
        neutral = ctx.neutrals(fam)
        for ci, cls in enumerate(tqdm(classes, desc=f"ablate {fam}")):
            heartbeat(log, f"ablate {fam}.{cls} {ci + 1}/{C}")
            arm_ivs, others = build_arm_ivs(ctx, fam, cls, arm_names,
                                            args.space)
            cfg = dict(cfg0)
            if others:
                cfg["other_concepts"] = [f"{f}.{c}" for f, c in others]
            pos, fb = ctx.positives(fam, cls)
            pos_fallback[ci] = int(fb)
            n_pos[ci] = len(pos)
            diag = ctx.diag_ids(fam, cls)
            if diag is None:
                print(f"  [{fam}.{cls}] tokens.json missing/empty — "
                      "diag_lp_delta skipped")
            # --- positives: diag + bits-per-token
            if pos:
                lp = positives_lp(ctx.model, ctx.tok, ctx.device, pos, diag,
                                  arm_ivs, ctx.mu_sigma, args.batch_tokens,
                                  desc=f"{fam}.{cls}")
                for ai, arm in enumerate(arm_names):
                    d_all = lp[arm]["lp_mean"] - lp["clean"]["lp_mean"]
                    res["bpt_delta"][ci, ai] = np.nanmean(d_all) / LN2
                    lo, hi = bootstrap_ci(d_all / LN2, seed=args.seed)
                    res["bpt_lo"][ci, ai], res["bpt_hi"][ci, ai] = lo, hi
                    if diag:
                        d_dg = lp[arm]["diag_mean"] - lp["clean"]["diag_mean"]
                        res["diag_lp_delta"][ci, ai] = np.nanmean(d_dg)
                        lo, hi = bootstrap_ci(d_dg, seed=args.seed)
                        res["diag_lo"][ci, ai] = lo
                        res["diag_hi"][ci, ai] = hi
            # --- neutral KL guard
            if neutral:
                n_neutral[ci] = len(neutral)
                kl = neutral_kl(ctx.model, ctx.tok, ctx.device, neutral,
                                arm_ivs, ctx.mu_sigma, args.batch_tokens,
                                desc=f"{fam}.{cls}")
                for ai, arm in enumerate(arm_names):
                    res["kl_neutral"][ci, ai] = np.nanmean(kl[arm])
                    lo, hi = bootstrap_ci(kl[arm], seed=args.seed)
                    res["kl_lo"][ci, ai], res["kl_hi"][ci, ai] = lo, hi
            # --- cloze
            if bank:
                lpm = cloze_lp(ctx.model, ctx.tok, ctx.device, bank, arm_ivs,
                               ctx.mu_sigma, args.batch_tokens,
                               desc=f"{fam}.{cls}", clean_map=cloze_clean_map)
                if cloze_clean_map is None:
                    cloze_clean_map = lpm["clean"]
                acc0, tlp0, n0, _ = cloze_metrics(cloze_clean_map, bank, cls)
                cloze_acc_clean[ci], n_cloze[ci] = acc0, n0
                for ai, arm in enumerate(arm_names):
                    acc, tlp, n, _ = cloze_metrics(lpm[arm], bank, cls)
                    res["cloze_acc_delta"][ci, ai] = acc - acc0
                    res["cloze_target_lp_delta"][ci, ai] = tlp - tlp0
            elif ci == 0:
                print(f"  [{fam}] cloze.json missing — cloze metrics skipped")
            for ai, arm in enumerate(arm_names):
                for metric, key, n, lo_k, hi_k in (
                        ("diag_lp_delta", "diag_lp_delta", n_pos[ci],
                         "diag_lo", "diag_hi"),
                        ("bpt_delta_bits", "bpt_delta", n_pos[ci],
                         "bpt_lo", "bpt_hi"),
                        ("kl_neutral_nats", "kl_neutral", n_neutral[ci],
                         "kl_lo", "kl_hi"),
                        ("cloze_acc_delta", "cloze_acc_delta", n_cloze[ci],
                         None, None),
                        ("cloze_target_lp_delta", "cloze_target_lp_delta",
                         n_cloze[ci], None, None)):
                    v = res[key][ci, ai]
                    if not np.isfinite(v):
                        continue
                    rows.append({
                        "concept": cls, "family": fam, "layer": layer_label,
                        "arm": arm, "metric": metric, "value": float(v),
                        "n": int(n),
                        "ci_low": (float(res[lo_k][ci, ai]) if lo_k else None),
                        "ci_high": (float(res[hi_k][ci, ai]) if hi_k else None),
                        "config": {**cfg, "pos_fallback": int(pos_fallback[ci])},
                    })
        np.savez(out_dir / f"{fam}.ablate.npz",
                 classes=np.array(classes), arms=np.array(arm_names),
                 blocks=np.array(ctx.blocks),
                 cloze_acc_clean=cloze_acc_clean, n_pos=n_pos,
                 n_cloze=n_cloze, n_neutral=n_neutral,
                 pos_fallback=pos_fallback, **res)
        append_summary(out_dir, rows)
        heartbeat(log, f"ablate {fam} DONE")


def run_rank(ctx, args, out_dir: Path, log: Path):
    layer_label = "all_blocks26" if args.all_blocks else "all_probed"
    cfg0 = base_config(ctx, args)
    for fam in ctx.rank_families:
        classes = ctx.families[fam]
        C = len(classes)
        bank = ctx.cloze(fam)
        fdiag = ctx.family_diag_ids(fam)
        if bank is None and fdiag is None:
            print(f"  [rank {fam}] no cloze bank and no tokens bank — skipped")
            continue
        pos = ctx.family_positives(fam)
        ks = list(range(1, C))
        bases = ["concept", "random"]
        cloze_acc = np.full((len(ks), 2), np.nan)
        cloze_tlp = np.full((len(ks), 2), np.nan)
        diag_d = np.full((len(ks), 2), np.nan)
        diag_lo = np.full((len(ks), 2), np.nan)
        diag_hi = np.full((len(ks), 2), np.nan)
        singvals = np.stack([subspace_dirs(ctx, fam, L, 1, "concept")[2]
                             for L in ctx.probed])
        arm_ivs = {}
        for kidx, k in enumerate(ks):
            for basis in bases:
                arm_ivs[f"{basis}_k{k}"] = build_rank_ivs(ctx, fam, k, basis)[0]
        heartbeat(log, f"rank {fam}: {len(arm_ivs)} subspace arms")
        # cloze
        acc_clean = tlp_clean = float("nan")
        n_keyed = 0
        if bank:
            lpm = cloze_lp(ctx.model, ctx.tok, ctx.device, bank, arm_ivs,
                           ctx.mu_sigma, args.batch_tokens, desc=f"rank {fam}")
            acc_clean, tlp_clean, n_keyed, _ = cloze_metrics(
                lpm["clean"], bank, None)
            for kidx, k in enumerate(ks):
                for bi, basis in enumerate(bases):
                    acc, tlp, _, _ = cloze_metrics(lpm[f"{basis}_k{k}"],
                                                   bank, None)
                    cloze_acc[kidx, bi] = acc
                    cloze_tlp[kidx, bi] = tlp
        # family diag logprob on family positives
        n_dpos = 0
        if fdiag and pos:
            n_dpos = len(pos)
            lp = positives_lp(ctx.model, ctx.tok, ctx.device, pos, fdiag,
                              arm_ivs, ctx.mu_sigma, args.batch_tokens,
                              desc=f"rank {fam}")
            for kidx, k in enumerate(ks):
                for bi, basis in enumerate(bases):
                    d = (lp[f"{basis}_k{k}"]["diag_mean"]
                         - lp["clean"]["diag_mean"])
                    diag_d[kidx, bi] = np.nanmean(d)
                    diag_lo[kidx, bi], diag_hi[kidx, bi] = bootstrap_ci(
                        d, seed=args.seed)
        np.savez(out_dir / f"{fam}.rank.npz",
                 ks=np.array(ks), bases=np.array(bases),
                 classes=np.array(classes), cloze_acc=cloze_acc,
                 cloze_target_lp=cloze_tlp, cloze_acc_clean=acc_clean,
                 cloze_target_lp_clean=tlp_clean, n_cloze=n_keyed,
                 diag_lp_delta=diag_d, diag_lo=diag_lo, diag_hi=diag_hi,
                 n_pos=n_dpos, singvals=singvals,
                 singvals_layers=np.array(ctx.probed))
        rows = []
        for kidx, k in enumerate(ks):
            for bi, basis in enumerate(bases):
                arm = "concept_subspace" if basis == "concept" \
                    else "random_subspace"
                if np.isfinite(cloze_acc[kidx, bi]):
                    rows.append({"concept": f"{fam}(family)", "family": fam,
                                 "layer": layer_label, "arm": arm,
                                 "metric": "cloze_acc_delta",
                                 "value": float(cloze_acc[kidx, bi]
                                                - acc_clean),
                                 "n": int(n_keyed), "ci_low": None,
                                 "ci_high": None, "config": {**cfg0, "k": k}})
                if np.isfinite(diag_d[kidx, bi]):
                    rows.append({"concept": f"{fam}(family)", "family": fam,
                                 "layer": layer_label, "arm": arm,
                                 "metric": "diag_lp_delta",
                                 "value": float(diag_d[kidx, bi]),
                                 "n": int(n_dpos),
                                 "ci_low": float(diag_lo[kidx, bi]),
                                 "ci_high": float(diag_hi[kidx, bi]),
                                 "config": {**cfg0, "k": k}})
        append_summary(out_dir, rows)
        heartbeat(log, f"rank {fam} DONE")


def run_restore(ctx, args, out_dir: Path, log: Path):
    cfg0 = base_config(ctx, args)
    for fam, classes in ctx.families.items():
        bank = ctx.cloze(fam)
        if bank is None:
            print(f"  [restore {fam}] cloze.json missing — skipped "
                  "(metric is cloze recovery)")
            continue
        C = len(classes)
        F = len(RESTORE_FACTORS)
        acc_clean = np.full(C, np.nan)
        acc_abl = np.full(C, np.nan)
        acc_rest = np.full((C, F), np.nan)
        recov = np.full((C, F), np.nan)
        chosen = np.zeros(C, int)
        n_cloze = np.zeros(C, int)
        rows = []
        clean_map = None
        for ci, cls in enumerate(tqdm(classes, desc=f"restore {fam}")):
            heartbeat(log, f"restore {fam}.{cls} {ci + 1}/{C}")
            Lr = ctx.chosen_layer(fam, cls)
            chosen[ci] = Lr
            if Lr not in ctx.blocks:
                print(f"  [restore {fam}.{cls}] chosen layer {Lr} not in "
                      "intervened blocks — restore steer added anyway")
            base_ivs = build_arm_ivs(ctx, fam, cls, ["ridge"],
                                     args.space)[0]["ridge"]
            s95 = ctx.calib(fam, cls, Lr)["s95"]
            dom_w = ctx.arms(fam, cls, Lr)["dom"][0]
            mu_sigma = dict(ctx.mu_sigma)
            if Lr not in mu_sigma:
                mu_sigma[Lr] = load_natstats(Lr)
            arm_ivs = {"ablate": base_ivs}
            for f in RESTORE_FACTORS:
                # list order matters: ablations first, steer LAST at Lr
                arm_ivs[f"restore_{f}"] = base_ivs + [Intervention(
                    Lr, dom_w, "steer", alpha=float(f * s95))]
            lpm = cloze_lp(ctx.model, ctx.tok, ctx.device, bank, arm_ivs,
                           mu_sigma, args.batch_tokens, desc=f"{fam}.{cls}",
                           clean_map=clean_map)
            clean_map = lpm["clean"]
            a0, _, n0, _ = cloze_metrics(lpm["clean"], bank, cls)
            aa, _, _, _ = cloze_metrics(lpm["ablate"], bank, cls)
            acc_clean[ci], acc_abl[ci], n_cloze[ci] = a0, aa, n0
            for fi, f in enumerate(RESTORE_FACTORS):
                ar, _, _, _ = cloze_metrics(lpm[f"restore_{f}"], bank, cls)
                acc_rest[ci, fi] = ar
                deficit = a0 - aa
                recov[ci, fi] = ((ar - aa) / deficit if deficit > 0
                                 else float("nan"))
                rows.append({"concept": cls, "family": fam, "layer": int(Lr),
                             "arm": "ridge_ablate+dom_restore",
                             "metric": "cloze_recovery_frac",
                             "value": float(recov[ci, fi]), "n": int(n0),
                             "ci_low": None, "ci_high": None,
                             "config": {**cfg0, "alpha_factor": f,
                                        "s95": float(s95),
                                        "acc_clean": float(a0),
                                        "acc_ablate": float(aa),
                                        "acc_restore": float(ar)}})
        np.savez(out_dir / f"{fam}.restore.npz",
                 classes=np.array(classes),
                 factors=np.array(RESTORE_FACTORS), chosen_layer=chosen,
                 cloze_acc_clean=acc_clean, cloze_acc_ablate=acc_abl,
                 cloze_acc_restore=acc_rest, recovery_frac=recov,
                 n_cloze=n_cloze)
        append_summary(out_dir, rows)
        heartbeat(log, f"restore {fam} DONE")


# ===================================================== rank exactness check

def rank_exactness_check(ctx, k=3, layer=0, tol=1e-4):
    """Verify composed rank-1 std ablations with orthonormal dirs == direct
    rank-k projection (z' = z − Q(Qᵀz − t)). Run on real hidden states of the
    smoke model; raises AssertionError on failure."""
    fam = ctx.rank_families[0]
    V, t, _ = subspace_dirs(ctx, fam, layer, k, "concept")
    ortho_err = float(np.abs(V @ V.T - np.eye(k)).max())
    assert ortho_err < 1e-5, f"subspace dirs not orthonormal ({ortho_err})"
    ivs = [Intervention(layer, V[j], "ablate", t=float(t[j]))
           for j in range(k)]
    ids = torch.tensor([[1] + [5, 9, 13, 21, 34, 55]])
    with torch.inference_mode():
        clean = ctx.model(ids, output_hidden_states=True)
        with Hooks(ctx.model, ivs, ctx.mu_sigma):
            abl = ctx.model(ids, output_hidden_states=True)
    mu, sd = ctx.mu_sigma[layer]
    mu_t, sd_t = torch.tensor(mu), torch.tensor(sd)
    h = clean.hidden_states[layer + 1].to(torch.float32)
    z = (h - mu_t) / sd_t
    Q = torch.tensor(V, dtype=torch.float32)             # [k, d]
    tt = torch.tensor(t, dtype=torch.float32)
    z_exp = z - torch.einsum("btk,kd->btd", (z @ Q.T - tt), Q)
    h_exp = mu_t + sd_t * z_exp
    err = (abl.hidden_states[layer + 1].to(torch.float32) - h_exp) \
        .abs().max().item()
    assert err < tol, f"rank-{k} composition != direct projection (err {err})"
    return ortho_err, err


# ================================================================== dry run

def dry_run(ctx, args):
    print("=" * 72)
    print("E4 DRY RUN — no model forwards. Plan + forward estimate.")
    print(f"families: {list(ctx.families)}  (skipped, no natscores: "
          f"{ctx.skipped})")
    print(f"intervened blocks: {ctx.blocks}"
          + (" [all-blocks: non-probed borrow nearest probed dir+t]"
             if args.all_blocks else ""))
    print("layer -1 (embedding): SKIPPED — natstats has no embedding-stream "
          "stats (documented decision)")
    arm_names = expand_arm_names(args.arms)
    print(f"arms: {arm_names}")
    est_tok = lambda texts: sum(len(t) // 4 + 2 for t in texts)  # noqa: E731
    print("(token counts estimated as chars/4 — no tokenizer loaded)")
    total_calls = total_tokens = 0
    missing_banks = []
    for fam, classes in ctx.families.items():
        bank = ctx.cloze(fam)
        tokens_path = PROMPTS_DIR / f"{fam}.tokens.json"
        has_tokens = tokens_path.exists()
        if bank is None:
            missing_banks.append(f"{fam}.cloze.json")
        if not has_tokens:
            missing_banks.append(f"{fam}.tokens.json")
        try:
            neutral = ctx.neutrals(fam)
        except FileNotFoundError as e:
            print(f"  {fam}: eval pool missing ({e}) — cannot plan")
            continue
        cloze_rows_tok = 0
        if bank:
            for t in bank:                # one row per (template, surface)
                pt = len(t["prompt"]) // 4 + 2
                for surfs in t["completions"].values():
                    cloze_rows_tok += sum(pt + len(s) // 4 + 1 for s in surfs)
        fam_calls = fam_tok = 0
        for cls in classes:
            try:
                pos, fb = ctx.positives(fam, cls)
            except FileNotFoundError as e:
                print(f"  {fam}.{cls}: {e}")
                continue
            pt = est_tok(pos)
            nt = est_tok(neutral)
            n_arm = len(arm_names)
            b_pos = max(1, math.ceil(pt / args.batch_tokens))
            b_neu = max(1, math.ceil(nt / min(args.batch_tokens,
                                              KL_BATCH_TOKENS)))
            b_clz = max(1, math.ceil(cloze_rows_tok / args.batch_tokens)) \
                if bank else 0
            calls = (1 + n_arm) * (b_pos + b_neu) + (1 + n_arm) * b_clz
            toks = (1 + n_arm) * (pt + nt) + (1 + n_arm) * cloze_rows_tok
            fam_calls += calls
            fam_tok += toks
            flag = " [FALLBACK strength>0]" if fb else ""
            print(f"  {fam}.{cls}: pos={len(pos)}{flag} neutral={len(neutral)}"
                  f" cloze={'yes' if bank else 'NO'}"
                  f" diag_tokens={'yes' if has_tokens else 'NO'}"
                  f" -> ~{calls} calls, ~{toks:,} tok")
        # rank part
        if fam in ctx.rank_families and not args.skip_rank:
            C = len(classes)
            n_sub = 2 * (C - 1)
            fpos_tok = est_tok(ctx.family_positives(fam))
            b_pos = max(1, math.ceil(fpos_tok / args.batch_tokens))
            b_clz = max(1, math.ceil(cloze_rows_tok / args.batch_tokens)) \
                if bank else 0
            calls = (1 + n_sub) * (b_pos + b_clz)
            toks = (1 + n_sub) * (fpos_tok + cloze_rows_tok)
            fam_calls += calls
            fam_tok += toks
            print(f"  {fam} RANK: k=1..{C - 1} x 2 bases -> ~{calls} calls, "
                  f"~{toks:,} tok")
        # restore part
        if not args.skip_restore and bank:
            calls = len(classes) * (1 + 1 + len(RESTORE_FACTORS)) * max(
                1, math.ceil(cloze_rows_tok / args.batch_tokens))
            toks = len(classes) * (2 + len(RESTORE_FACTORS)) * cloze_rows_tok
            fam_calls += calls
            fam_tok += toks
            print(f"  {fam} RESTORE: {len(classes)} concepts x "
                  f"{{clean,ablate,3 alphas}} -> ~{calls} calls")
        total_calls += fam_calls
        total_tokens += fam_tok
        print(f"  {fam} TOTAL ~{fam_calls} model calls, ~{fam_tok:,} tokens")
    print("-" * 72)
    print(f"GRAND TOTAL ~{total_calls:,} model calls, ~{total_tokens:,} "
          "forward tokens")
    print(f"@ ~25k tok/s (gemma-2-2b bf16 eager, H100): "
          f"~{total_tokens / 25000 / 60:.0f} min forward time")
    if missing_banks:
        print("\nMISSING PROMPT BANKS (being authored in parallel; the "
              "affected metrics are skipped gracefully at run time):")
        for m in sorted(set(missing_banks)):
            print(f"  prompts/{m}")
    print("=" * 72)


# ===================================================================== main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", default=None,
                    help="comma list (default: all with natscores)")
    ap.add_argument("--classes", default=None, help="comma list filter")
    ap.add_argument("--layers", default=None,
                    help="comma list of probed layers to intervene at "
                         "(default: all 12 probed layers)")
    ap.add_argument("--all-blocks", action="store_true",
                    help="intervene at all 26 block outputs (0..25); "
                         "non-probed blocks borrow nearest probed dir+t")
    ap.add_argument("--arms", default="ridge,dom,rand,other",
                    help="rand -> rand0..4, other -> other0..2")
    ap.add_argument("--space", default="std", choices=["std", "grad"])
    ap.add_argument("--out", default=None,
                    help="output dir (default out/e4, smoke out/e4_smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=80,
                    help="max positives per concept / family")
    ap.add_argument("--neutral-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", action="append", default=[],
                    choices=["rank", "restore"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random Gemma2 + synthetic data, CPU")
    ap.add_argument("--dry-run", action="store_true",
                    help="real data: plan + forward estimate, no model")
    args = ap.parse_args()
    args.skip_rank = "rank" in args.skip
    args.skip_restore = "restore" in args.skip

    if args.smoke:
        args.device, args.dtype = "cpu", "float32"
        args.batch_tokens = min(args.batch_tokens, 512)
        args.limit, args.neutral_n = min(args.limit, 8), min(args.neutral_n, 5)
        out_dir = Path(args.out) if args.out else OUT_DIR / "e4_smoke"
    else:
        out_dir = Path(args.out) if args.out else OUT_DIR / "e4"
    args.out = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / "progress_e4_ablate.log"

    if args.smoke:
        # fresh summary so the end-of-smoke assertions see only this run
        (out_dir / "summary.jsonl").unlink(missing_ok=True)
        ctx = SmokeCtx(args)
        ortho_err, comp_err = rank_exactness_check(ctx)
        print(f"[smoke] rank-k composition exactness: orthonormality err "
              f"{ortho_err:.2e}, composed-vs-direct-projection err "
              f"{comp_err:.2e} (tol 1e-4) — PASS")
        run_ablate(ctx, args, out_dir, log)
        if not args.skip_rank:
            run_rank(ctx, args, out_dir, log)
        if not args.skip_restore:
            run_restore(ctx, args, out_dir, log)
        # sanity: outputs exist and are finite where expected
        for f in ("famA.ablate.npz", "famB.ablate.npz", "famA.rank.npz",
                  "famA.restore.npz"):
            p = out_dir / f
            assert p.exists(), f"smoke output missing: {p}"
            z = np.load(p, allow_pickle=True)
            assert len(z.files) > 3
        n_rows = sum(1 for _ in open(out_dir / "summary.jsonl"))
        print(f"[smoke] outputs OK ({n_rows} summary rows) — SMOKE PASS")
        return

    ctx = RealCtx(args)
    if ctx.skipped:
        print(f"skipping (no natscores → no natural-mean t): {ctx.skipped}")
    if args.dry_run:
        dry_run(ctx, args)
        return

    heartbeat(log, f"START families={list(ctx.families)} blocks="
                   f"{'0..25' if args.all_blocks else ctx.blocks} "
                   f"arms={args.arms} skip={args.skip}")
    ctx.load_model()
    run_ablate(ctx, args, out_dir, log)
    if not args.skip_rank:
        run_rank(ctx, args, out_dir, log)
    if not args.skip_restore:
        run_restore(ctx, args, out_dir, log)
    heartbeat(log, "DONE")
    print(f"E4 complete -> {out_dir}")


if __name__ == "__main__":
    main()
