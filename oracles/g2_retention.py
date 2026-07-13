#!/usr/bin/env python3
"""Stage 7-Oracle Phase 2 — Gate G2 REAL-data retention check.

SPEC.md Phase 2 Gate G2, the "Additional REAL-data check" sentence:

    "run the encoder over the Stage-6 natural eval texts and compute
     example-AUROC of its predictions against judge labels — encoder should
     retain >=90% of the gemma probes' AUROC."

This script scores the distilled Exp-A encoder's ability to DETECT concepts on
the Stage-6 *natural* evaluation texts (real, judge-labeled data — NOT the
ClimbMix distillation corpus) and compares its per-concept example-level AUROC
against the gemma probes' own AUROC on exactly the same examples/labels.

--------------------------------------------------------------------------
Ground truth (where the labels come from)
--------------------------------------------------------------------------
Stage-6 natural eval pool, per family:
  <eval-data>/eval/<fam>.jsonl        one row per example:
        {example_id, text, token_ids(gemma), n_tokens, nat_split(cal|test),
         targets: {concept: [[tok_idx, score], ...]}}
  <eval-data>/natscores/<fam>.natscores.npz  (built by stage6/score_natural.py;
        rows 1:1 and IN THE SAME ORDER as the jsonl — verified at load):
        y [T, C], token2ex [T], ex_nat_split [n_ex], classes [C] (SPACE form),
        layers [12], preds_{ridge,dom,lda,logistic} [12, T, C].

Example-level judge label for (example, concept) = (ymax >= YMAX_THRESH) where
ymax = max over the example's tokens of y[:, concept] (max-pool). This is byte-
identical to the labelling in stage7 select_probes.py `score_family`. We
evaluate on the natural-pool TEST split only (ex_nat_split == "test"),
mirroring Phase-0 discipline so the comparison is like-with-like.

--------------------------------------------------------------------------
Reference AUROC method (chosen)
--------------------------------------------------------------------------
We RECOMPUTE the gemma-probe example-AUROC ourselves from natscores.npz, using
the identical example set (TEST split), identical labels (ymax >= 0.34) and
identical max-pool method (max over token2ex) as select_probes.py — so the
encoder and the gemma reference are scored on EXACTLY the same examples and
labels. (probe_set.json also STORES a selection AUROC per (layer, concept); we
load it purely as a cross-check and report the max abs diff, but the gate uses
the recomputed value.) Rationale: the task requires the reference and the
encoder to share the same example set; recomputing guarantees it.

--------------------------------------------------------------------------
Encoder detection score (design decisions)
--------------------------------------------------------------------------
The encoder reads raw text directly (Qwen tokenization; gemma offsets / the
align.py bridge are NOT needed here). It was trained to predict per-GEMMA-TOKEN
standardized probe scores at Qwen prefix states. For example-level DETECTION we:
  * qwen-tokenize the raw eval text (add_special_tokens=True, capped at
    --max-qwen-tokens; eval texts are short so no truncation in practice),
  * forward the frozen encoder -> hidden [Tq, H] -> head -> preds [Tq, 3K],
  * MAX-POOL the head's per-position predictions over ALL real (non-pad) qwen
    positions of the text -> one [3K] score vector per example.
Max-pool is the natural "does this concept appear anywhere in the text"
detector and mirrors how the gemma reference max-pools per-token scores.
Standardization is IRRELEVANT to AUROC (it is a per-column affine map and
AUROC is rank-based), so we use raw head outputs directly and do not need
corpus_stats.json.

Head column l*K+c holds concept `main_block_concepts[c]` at layer `layers[l]`
(see out/PERMUTATION_FIX.md) — we attach names via main_block_concepts, NOT the
name-sorted `concepts` list.

LAYER BLOCK: the Phase-0 gemma AUROC is per-layer. We compute the encoder AUROC
per layer block (all 3 of layers=[6,8,14]) and report all three; the per-concept
headline "gates on the best" = the layer block where the ENCODER's AUROC is
highest, and the retention denominator is the gemma AUROC at that SAME layer
(like-with-like at one layer, no cross-layer denominator cherry-picking). Per-
layer medians are also reported for transparency.

--------------------------------------------------------------------------
Retention ratio
--------------------------------------------------------------------------
Per concept (at its best-encoder layer):
  raw ratio           =  enc_auroc / gemma_auroc
  chance-corrected    = (enc_auroc - 0.5) / (gemma_auroc - 0.5)
Both are reported. The chance-corrected form is the primary gate metric (a
0.95-AUROC probe "retained" at raw 0.90 is only 0.79 chance-corrected). PASS
(SPEC "retain >=90%") is evaluated as MEDIAN chance-corrected ratio >= 0.90.
If gemma_auroc - 0.5 < MIN_DENOM the chance-corrected ratio is undefined for
that concept and it is excluded from the aggregate (recorded in the json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_encoder as te  # ProbeSet, EncoderHead, load_encoder, D_MODEL_GEMMA

YMAX_THRESH = 0.34   # example-level positive-label threshold (matches select_probes.py)
MIN_DENOM = 0.05     # gemma_auroc-0.5 below this -> chance-corrected ratio undefined
RETAIN_BAR = 0.90    # SPEC "retain >=90%"


# ============================================================== metrics
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC via rank-sum — identical formula to
    select_probes.py / stage6 gates.py `auroc()`."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    n_p, n_n = len(pos), len(neg)
    return float((r[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


# ============================================================== ground truth
def _u2s(name: str) -> str:
    """probe_set concept (underscore/hyphen) -> natscores class (space) form."""
    return name.replace("_", " ")


def load_family_truth(eval_data_dir, fam):
    """Returns (rows, z) where rows = eval jsonl (list, ex order) and z =
    natscores npz dict OR None. The jsonl is the single source of truth for
    texts, the nat_split, and the judge labels; natscores.npz (large, ~hundreds
    of MB/family) is loaded ONLY if present, to RECOMPUTE the gemma reference
    AUROC. When absent (lightweight staging), the gemma reference falls back to
    probe_set.json's stored selection AUROC — equivalent here because the
    example set + labels are identical (verified: max|Δ|=0). Asserts 1:1
    example-order alignment when natscores is present."""
    jsonl_candidates = [
        os.path.join(eval_data_dir, "eval", f"{fam}.jsonl"),
        os.path.join(eval_data_dir, "natural", "eval", f"{fam}.jsonl"),
    ]
    jsonl = next((p for p in jsonl_candidates if os.path.exists(p)), None)
    if jsonl is None:
        raise FileNotFoundError(f"{fam}: no eval jsonl at any of {jsonl_candidates}")
    rows = [json.loads(l) for l in open(jsonl)]
    npz = os.path.join(eval_data_dir, "natscores", f"{fam}.natscores.npz")
    z = None
    if os.path.exists(npz):
        z = dict(np.load(npz, allow_pickle=True))
        exid = [str(x) for x in z["ex_example_id"]]
        if len(exid) != len(rows):
            raise RuntimeError(f"{fam}: natscores has {len(exid)} examples but jsonl has {len(rows)}")
        for i in range(min(20, len(rows))):
            if rows[i]["example_id"] != exid[i]:
                raise RuntimeError(
                    f"{fam}: example-order misalignment at row {i}: "
                    f"jsonl={rows[i]['example_id']} natscores={exid[i]}")
    return rows, z


def class_index(z, concept):
    classes = [str(c) for c in z["classes"]]
    name = _u2s(concept)
    if name in classes:
        return classes.index(name)
    if concept in classes:
        return classes.index(concept)
    raise KeyError(f"concept {concept!r} (space form {name!r}) not in classes {classes}")


def test_split_ids(rows):
    return np.array([i for i, r in enumerate(rows) if r.get("nat_split") == "test"], dtype=int)


def jsonl_labels(rows, concept):
    """Example-level judge label per example for `concept`: (ymax >= YMAX_THRESH)
    where ymax = max over the example's token target scores (tokens with
    ti < n_tokens; untargeted tokens contribute 0). Byte-identical to the
    natscores-y max-pool label used in select_probes.py (score_natural.py fills
    y[token, ci] from targets[c] for ti < n)."""
    key = _u2s(concept)
    labels = np.zeros(len(rows), dtype=int)
    for i, r in enumerate(rows):
        n = r.get("n_tokens", 10 ** 9)
        ts = r.get("targets", {}).get(key)
        if ts is None and key != concept:
            ts = r.get("targets", {}).get(concept)
        ymax = 0.0
        if ts:
            for ti, s in ts:
                if ti < n and s > ymax:
                    ymax = s
        labels[i] = int(ymax >= YMAX_THRESH)
    return labels


def gemma_auroc_at_layer(z, ci, layer, arm, labels, test_ex_ids):
    """Recompute the gemma probe's example-AUROC exactly like select_probes.py:
    max-pool preds_{arm}[layer, :, ci] over token2ex, TEST split, with the
    provided (jsonl-derived, identical) labels."""
    layers = [int(x) for x in z["layers"]]
    li = layers.index(layer)
    token2ex = z["token2ex"]
    n_ex = len(labels)
    p = z[f"preds_{arm}"][li, :, ci].astype(np.float64)
    pmax = np.full(n_ex, -np.inf)
    np.maximum.at(pmax, token2ex, p)
    pmax_t = pmax[test_ex_ids]
    lab_t = labels[test_ex_ids]
    return auroc(pmax_t[lab_t == 1], pmax_t[lab_t == 0])


# ============================================================== encoder
def load_encoder_and_head(ckpt_path, probe_set_dir, device, encoder_and_tok=None):
    """Load best.pt -> (model, qwen_tok, head, K, mode). encoder_and_tok
    (model, qwen_tok, name) overrides the (real) Qwen download for the smoke
    test; the HEAD is always rebuilt from the checkpoint's head_state."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mode = ckpt["mode"]
    if mode != "expA":
        raise ValueError(
            f"G2 detection expects an Exp-A encoder (up: H->3K); checkpoint "
            f"mode={mode!r}. Exp-B predicts dom/v* targets, not the 3-layer "
            f"concept scores this gate reads. Point --encoder-ckpt at the "
            f"expA best.pt.")
    K = ckpt["K"]
    hidden_size = ckpt["hidden_size"]
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    if encoder_and_tok is not None:
        model, qwen_tok, model_name = encoder_and_tok
        model.to(device)
    else:
        model, qwen_tok, model_name = te.load_encoder(ckpt["model_name"], dtype, device)
        if "encoder_state" in ckpt:  # full fine-tuned encoder
            model.load_state_dict(ckpt["encoder_state"])
    head = te.EncoderHead(hidden_size, K, mode).to(device)
    head.load_state_dict(ckpt["head_state"])
    if dtype == torch.bfloat16:
        head = head.to(dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in head.parameters():
        p.requires_grad_(False)
    model.eval()
    head.eval()
    return model, qwen_tok, head, K, mode, model_name


@torch.no_grad()
def encode_texts(texts, model, qwen_tok, head, K, device, max_qwen_tokens, bsz):
    """Returns enc_scores [n_texts, 3K] float32: per text, max-pool of the
    head's per-position predictions over all real qwen positions."""
    n = len(texts)
    out = np.full((n, 3 * K), -np.inf, dtype=np.float32)
    cuda = str(device).startswith("cuda")
    for s in tqdm(range(0, n, bsz), desc="encode", leave=False):
        batch = texts[s:s + bsz]
        enc = qwen_tok(batch, add_special_tokens=True, truncation=True,
                       max_length=max_qwen_tokens, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if cuda else _null()
        with ctx:
            hidden = model(input_ids=input_ids, attention_mask=attn).last_hidden_state
            y_pred, _ = head(hidden)  # [B, T, 3K]
        y_pred = y_pred.float()
        attn_b = attn.bool()
        for i in range(len(batch)):
            m = attn_b[i]
            if m.any():
                vals = y_pred[i][m].max(dim=0).values  # [3K]
                out[s + i] = vals.cpu().numpy()
    return out


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ============================================================== main gate
def run_g2(args, encoder_and_tok=None):
    t0 = time.time()
    ps = te.ProbeSet(args.probe_set)
    K = ps.K
    layers = ps.layers                      # [6, 8, 14]
    mbc = ps.main_block_concepts            # store/head main-block order
    families = ps.families
    selection = ps.meta.get("selection", {})

    model, qwen_tok, head, K_ck, mode, model_name = load_encoder_and_head(
        args.encoder_ckpt, args.probe_set, args.device, encoder_and_tok)
    if K_ck != K:
        raise ValueError(f"checkpoint K={K_ck} != probe_set K={K}")

    # group concepts by family (so we load each family's data once)
    fam_to_concepts = {}
    for c in ps.concepts:
        fam_to_concepts.setdefault(families[c], []).append(c)

    col_of = {c: i for i, c in enumerate(mbc)}   # concept -> main-block column

    per_concept = {}        # concept -> record
    per_family_n_test = {}
    xcheck_diffs = []       # |recomputed gemma auroc - probe_set stored auroc|
    ref_sources = set()     # {"natscores_recompute", "probe_set_stored"}

    for fam in sorted(fam_to_concepts):
        rows, z = load_family_truth(args.eval_data, fam)
        # test examples (shared across all concepts of this family)
        test_ex_ids = test_split_ids(rows)
        per_family_n_test[fam] = int(len(test_ex_ids))
        test_texts = [rows[i]["text"] for i in test_ex_ids]
        enc_scores = encode_texts(test_texts, model, qwen_tok, head, K,
                                  args.device, args.max_qwen_tokens, args.bsz)  # [n_test, 3K]

        for c in fam_to_concepts[fam]:
            col = col_of[c]
            labels = jsonl_labels(rows, c)              # identical to natscores-y max-pool label
            lab_t = labels[test_ex_ids]
            n_pos, n_neg = int((lab_t == 1).sum()), int((lab_t == 0).sum())
            ci = class_index(z, c) if z is not None else None

            layer_rows = []
            for l, L in enumerate(layers):
                arm = selection.get(str(L), {}).get(c, {}).get("arm")
                if arm is None:
                    # no selection metadata for this (layer, concept): can't
                    # pick the reference arm -> skip this layer for this concept
                    continue
                stored = selection.get(str(L), {}).get(c, {}).get("auroc")
                if z is not None:
                    g_auroc = gemma_auroc_at_layer(z, ci, L, arm, labels, test_ex_ids)
                    ref_sources.add("natscores_recompute")
                    if stored is not None and np.isfinite(g_auroc):
                        xcheck_diffs.append(abs(g_auroc - float(stored)))
                else:
                    g_auroc = float(stored) if stored is not None else float("nan")
                    ref_sources.add("probe_set_stored")
                enc_col = enc_scores[:, l * K + col]
                e_auroc = auroc(enc_col[lab_t == 1], enc_col[lab_t == 0])
                raw = (e_auroc / g_auroc) if (np.isfinite(e_auroc) and np.isfinite(g_auroc) and g_auroc > 0) else float("nan")
                denom = g_auroc - 0.5
                # cc requires BOTH a usable denominator AND a finite encoder
                # AUROC (a NaN enc_auroc must yield cc=None, not float('nan'),
                # or it silently poisons the np.median gate aggregate).
                cc = ((e_auroc - 0.5) / denom) if (np.isfinite(e_auroc) and np.isfinite(denom) and denom >= MIN_DENOM) else None
                layer_rows.append(dict(layer=int(L), arm=arm,
                                       enc_auroc=_f(e_auroc), gemma_auroc=_f(g_auroc),
                                       gemma_auroc_stored=(float(stored) if stored is not None else None),
                                       raw_ratio=_f(raw),
                                       cc_ratio=(float(cc) if cc is not None else None)))
            if not layer_rows:
                continue
            # best-encoder layer (gate on the best); denominator = gemma auroc
            # at that SAME layer.
            valid = [r for r in layer_rows if r["enc_auroc"] is not None]
            best = max(valid, key=lambda r: r["enc_auroc"]) if valid else layer_rows[0]
            per_concept[c] = dict(
                family=fam, n_pos=n_pos, n_neg=n_neg,
                best_layer=best["layer"], best_arm=best["arm"],
                enc_auroc=best["enc_auroc"], gemma_auroc=best["gemma_auroc"],
                raw_ratio=best["raw_ratio"], cc_ratio=best["cc_ratio"],
                per_layer=layer_rows)

    # ---------------------------------------------------------------- verdict
    cc_vals = [r["cc_ratio"] for r in per_concept.values() if r["cc_ratio"] is not None]
    raw_vals = [r["raw_ratio"] for r in per_concept.values() if r["raw_ratio"] is not None]
    excluded_cc = [c for c, r in per_concept.items() if r["cc_ratio"] is None]

    def med(v):
        return float(np.median(v)) if v else float("nan")

    def share_ge(v, bar):
        return float(np.mean([x >= bar for x in v])) if v else float("nan")

    # per-family medians
    per_family = {}
    for fam in sorted(fam_to_concepts):
        cs = [c for c in per_concept if per_concept[c]["family"] == fam]
        fc = [per_concept[c]["cc_ratio"] for c in cs if per_concept[c]["cc_ratio"] is not None]
        fr = [per_concept[c]["raw_ratio"] for c in cs if per_concept[c]["raw_ratio"] is not None]
        per_family[fam] = dict(
            n_concepts=len(cs), n_test_examples=per_family_n_test.get(fam),
            median_cc_ratio=med(fc), median_raw_ratio=med(fr),
            median_enc_auroc=med([per_concept[c]["enc_auroc"] for c in cs if per_concept[c]["enc_auroc"] is not None]),
            median_gemma_auroc=med([per_concept[c]["gemma_auroc"] for c in cs if per_concept[c]["gemma_auroc"] is not None]))

    # per-layer medians (across concepts, at each fixed layer)
    per_layer = {}
    for l, L in enumerate(layers):
        cc_l, raw_l, enc_l, gem_l = [], [], [], []
        for r in per_concept.values():
            lr = next((x for x in r["per_layer"] if x["layer"] == L), None)
            if lr is None:
                continue
            if lr["cc_ratio"] is not None:
                cc_l.append(lr["cc_ratio"])
            if lr["raw_ratio"] is not None and np.isfinite(lr["raw_ratio"]):
                raw_l.append(lr["raw_ratio"])
            if lr["enc_auroc"] is not None:
                enc_l.append(lr["enc_auroc"])
            if lr["gemma_auroc"] is not None:
                gem_l.append(lr["gemma_auroc"])
        per_layer[str(L)] = dict(median_cc_ratio=med(cc_l), median_raw_ratio=med(raw_l),
                                 median_enc_auroc=med(enc_l), median_gemma_auroc=med(gem_l),
                                 n_concepts=len(cc_l))

    median_cc = med(cc_vals)
    # SECONDARY (conservative) variant: fixed layer for BOTH encoder and gemma
    # (no per-concept best-of-3 selection). The primary metric picks each
    # concept's best encoder layer on the SAME test data it gates on, so it
    # carries a small systematic max-of-3 selection inflation (AUROC SE at
    # median n_pos=55 is ~0.02-0.03/layer; correlated across layers, net
    # ~+0.01-0.02 in cc units). If the primary median sits within ~0.02 of the
    # bar, read it alongside these fixed-layer numbers before calling GO.
    fixed_layer_cc = {str(L): per_layer[str(L)]["median_cc_ratio"] for L in layers}
    verdict = {
        "pass_gate": bool(np.isfinite(median_cc) and median_cc >= RETAIN_BAR),
        "retain_bar": RETAIN_BAR,
        "gate_metric": "median chance-corrected retention ratio (best-encoder layer per concept)",
        "median_cc_ratio": median_cc,
        "median_raw_ratio": med(raw_vals),
        "share_cc_ge_0.9": share_ge(cc_vals, 0.9),
        "share_raw_ge_0.9": share_ge(raw_vals, 0.9),
        "n_concepts_scored": len(per_concept),
        "n_concepts_cc_defined": len(cc_vals),
        "n_concepts_cc_excluded": len(excluded_cc),
        "cc_excluded_concepts": excluded_cc,
        "secondary_median_cc_ratio_fixed_layer": fixed_layer_cc,
        "secondary_note": ("conservative variant: median cc ratio at a single "
                           "fixed layer for both encoder and gemma; immune to "
                           "best-of-3 layer-selection inflation of the primary "
                           "metric (~+0.01-0.02 cc units)"),
    }

    result = dict(
        gate="G2_natural_retention",
        encoder_ckpt=os.path.abspath(args.encoder_ckpt),
        probe_set=os.path.abspath(args.probe_set),
        eval_data=os.path.abspath(args.eval_data),
        model_name=model_name, mode=mode, K=K, layers=[int(x) for x in layers],
        ymax_thresh=YMAX_THRESH, min_denom=MIN_DENOM,
        reference_source=sorted(ref_sources),
        reference_auroc_method=(
            "labels + test split + texts all from eval/<fam>.jsonl (single "
            "source of truth; ymax>=0.34 max-pool over token targets). gemma "
            "reference AUROC = recomputed from natscores.npz on the IDENTICAL "
            "examples/labels (max-pool over token2ex) where natscores is present; "
            "else probe_set.json stored selection AUROC (equivalent: same example "
            "set + labels, verified max|Δ|=0). See reference_source."),
        encoder_score_method=("max-pool of head per-position predictions over all "
                              "real qwen positions; AUROC is rank-based so "
                              "standardization is not applied"),
        gemma_stored_vs_recomputed_max_absdiff=(float(np.max(xcheck_diffs)) if xcheck_diffs else None),
        per_family_n_test_examples=per_family_n_test,
        verdict=verdict,
        per_family=per_family,
        per_layer=per_layer,
        per_concept=per_concept,
        seconds=round(time.time() - t0, 1),
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 68)
    print("GATE G2 — natural-data AUROC retention")
    print("=" * 68)
    print(f"concepts scored:        {len(per_concept)}  "
          f"(cc-defined {len(cc_vals)}, excluded {len(excluded_cc)})")
    print(f"median chance-corr ratio: {median_cc:.4f}   (raw {med(raw_vals):.4f})")
    print(f"share cc >= 0.9:          {share_ge(cc_vals, 0.9):.3f}   "
          f"(raw {share_ge(raw_vals, 0.9):.3f})")
    if xcheck_diffs:
        print(f"gemma stored-vs-recomputed max |Δ|: {np.max(xcheck_diffs):.2e}")
    print("per-family median cc ratio:")
    for fam in sorted(per_family):
        print(f"  {fam:12s} n={per_family[fam]['n_concepts']:2d} "
              f"cc={per_family[fam]['median_cc_ratio']:.4f} "
              f"raw={per_family[fam]['median_raw_ratio']:.4f} "
              f"enc_auroc={per_family[fam]['median_enc_auroc']:.4f} "
              f"gemma_auroc={per_family[fam]['median_gemma_auroc']:.4f}")
    print("per-layer median cc ratio:")
    for L in layers:
        pl = per_layer[str(L)]
        print(f"  L{L:<2d} cc={pl['median_cc_ratio']:.4f} raw={pl['median_raw_ratio']:.4f} "
              f"enc={pl['median_enc_auroc']:.4f} gemma={pl['median_gemma_auroc']:.4f}")
    print("secondary (fixed-layer, conservative) median cc ratio: "
          + "  ".join(f"L{L}={fixed_layer_cc[str(L)]:.4f}" for L in layers))
    print(f"\nVERDICT: {'PASS' if verdict['pass_gate'] else 'FAIL'} "
          f"(median cc ratio {median_cc:.4f} vs bar {RETAIN_BAR}; "
          f"conservative fixed-layer medians "
          + ", ".join(f"L{L}={fixed_layer_cc[str(L)]:.4f}" for L in layers) + ")")
    print(f"wrote {args.out}  ({result['seconds']}s)")
    return result


def _f(x):
    return float(x) if (x is not None and np.isfinite(x)) else None


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder-ckpt", required=True, help="Exp-A encoder checkpoint (best.pt)")
    p.add_argument("--probe-set", required=True, help="dir with probe_set.json + probe_set_arrays.npz")
    p.add_argument("--eval-data", required=True,
                   help="dir with eval/<fam>.jsonl and natscores/<fam>.natscores.npz "
                        "(Stage-6 data layout)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-qwen-tokens", type=int, default=1024)
    p.add_argument("--bsz", type=int, default=16)
    p.add_argument("--out", required=True, help="output json path")
    return p


def main():
    args = build_argparser().parse_args()
    run_g2(args)


if __name__ == "__main__":
    main()
