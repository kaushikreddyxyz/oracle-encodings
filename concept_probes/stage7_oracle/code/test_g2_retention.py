#!/usr/bin/env python3
"""Smoke test for g2_retention.py (Stage 7-Oracle Gate G2 natural-retention).

Plain assertion-script style (repo convention, no pytest). Builds a tiny
synthetic fixture that mirrors the Stage-6 natural-eval ground truth:
  * eval/<fam>.jsonl  (example_id, text, token_ids, nat_split, targets)
  * natscores/<fam>.natscores.npz  (classes, layers, y, token2ex,
      ex_nat_split, ex_example_id, preds_{ridge,dom,lda,logistic})
  * probe_set.json / probe_set_arrays.npz (te.ProbeSet-loadable)
  * a best.pt checkpoint holding a tiny EncoderHead(expA) state, loaded on top
    of a tiny RANDOM Qwen2Model injected via encoder_and_tok (so no Qwen
    download / no gemma model is needed).

Asserts: script runs end-to-end, per-concept AUROCs are finite, verdict +
per-family + per-layer tables are populated, retention ratios computed, and the
output json is written. Also a unit check that gemma AUROC recomputation
matches select_probes' method on a hand-built signal, and that the reference
recompute equals the probe_set stored auroc (cross-check path).

Run: python test_g2_retention.py   (exits 0 iff all checks pass)
"""
import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_encoder as te            # noqa: E402
import g2_retention as g2             # noqa: E402
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model  # noqa: E402

QWEN_MODEL = "Qwen/Qwen3-0.6B-Base"
RESULTS = []

PARAS = [
    "In January the northern markets were volatile across the continent of Africa.",
    "Blue-green algae blooms were reported across three lakes in Asia this autumn.",
    "The committee convened in December to review the budget before spring arrives.",
    "North America and Europe share long trade routes studied every single Wednesday.",
    "The orchestra performed a rare symphony in April drawing a record summer crowd.",
    "Engineers finished the eastern bridge inspection in March just before winter set in.",
    "On Saturday the red and orange banners lined the southern harbor near the market.",
    "Researchers in Oceania published new findings about deep ocean currents in June.",
]


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, None))
        print(f"[PASS] {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, f"{e}\n{traceback.format_exc()}"))
        print(f"[FAIL] {name}: {e}")


# ============================================================ fixture builders
def build_probe_set(pdir, concepts, fam, layers, ablation_layer, selection, rng):
    os.makedirs(pdir, exist_ok=True)
    K = len(concepts)
    D = te.D_MODEL_GEMMA
    W = rng.normal(0, 0.05, size=(3, K, D)).astype(np.float32)
    b = rng.normal(0, 0.1, size=(3, K)).astype(np.float32)
    nat_mean = rng.normal(0, 1, size=(3, D)).astype(np.float32)
    nat_std = rng.uniform(0.5, 1.5, size=(3, D)).astype(np.float32)
    W_dom = rng.normal(0, 0.05, size=(K, D)).astype(np.float32)
    G = (W_dom @ W_dom.T + 1e-3 * np.eye(K)).astype(np.float32)
    meta = {
        "layers": layers, "ablation_layer": ablation_layer,
        "concepts": concepts, "families": {c: fam for c in concepts},
        "main_block_concepts": concepts, "dom_block_concepts": concepts,
        "selection": selection, "s95": {}, "corpus_stats": None,
        "meta": {"fixture": True},
    }
    with open(os.path.join(pdir, "probe_set.json"), "w") as f:
        json.dump(meta, f)
    np.savez(os.path.join(pdir, "probe_set_arrays.npz"),
             W=W, b=b, nat_mean=nat_mean, nat_std=nat_std, W_dom_abl=W_dom,
             b_dom_abl=rng.normal(0, 0.1, size=(K,)).astype(np.float32),
             t_nat_dom=rng.normal(0, 0.5, size=(K,)).astype(np.float32),
             G_dom=G, G_dom_inv=np.linalg.inv(G).astype(np.float32),
             layer_index=np.array(layers, dtype=np.int64))
    return K


def build_eval_data(root, fam, concepts, layers, gemma_tok, rng, n_ex=40):
    """Synthetic eval jsonl + natscores npz with a planted signal: each example
    is assigned one 'true' concept; that concept's preds fire high on the
    example's tokens (so gemma AUROC is high) and its judge target y is set."""
    eval_dir = os.path.join(root, "eval")
    ns_dir = os.path.join(root, "natscores")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(ns_dir, exist_ok=True)
    C = len(concepts)

    rows = []
    tok_offsets = []   # (start, n) per example
    all_tokens = 0
    true_cls = []
    for i in range(n_ex):
        text = PARAS[i % len(PARAS)] + f" (case {i})"
        ids = gemma_tok(text, add_special_tokens=False)["input_ids"]
        n = len(ids)
        ci = i % C
        true_cls.append(ci)
        # judge targets: label tokens in the second half for the true concept
        tgt = {c: [] for c in concepts}
        for ti in range(n // 2, n):
            tgt[concepts[ci]].append([ti, 1.0])
        rows.append({
            "example_id": f"ex_{fam}_{i:04d}", "text": text, "token_ids": ids,
            # split by (i//C) so each concept (ci = i%C) gets BOTH test and cal
            # examples regardless of C's parity (decouples label from split).
            "n_tokens": n, "nat_split": ("test" if (i // C) % 2 == 0 else "cal"),
            "slice": "natural_mined", "cls_mined": concepts[ci],
            "surface": concepts[ci], "targets": tgt,
        })
        tok_offsets.append((all_tokens, n))
        all_tokens += n

    with open(os.path.join(eval_dir, f"{fam}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    T = all_tokens
    y = np.zeros((T, C), dtype=np.float32)
    token2ex = np.zeros(T, dtype=np.int32)
    for i, (off, n) in enumerate(tok_offsets):
        token2ex[off:off + n] = i
        ci = true_cls[i]
        y[off + n // 2: off + n, ci] = 1.0

    # preds: base noise + strong positive on the true concept's second-half
    # tokens, so the max-pooled gemma AUROC separates positives from negatives.
    arms = ["ridge", "dom", "lda", "logistic"]
    preds = {a: rng.normal(0, 0.3, size=(len(layers), T, C)).astype(np.float32) for a in arms}
    for i, (off, n) in enumerate(tok_offsets):
        ci = true_cls[i]
        for a in arms:
            preds[a][:, off + n // 2: off + n, ci] += 5.0

    ex_split = np.array([r["nat_split"] for r in rows])
    ex_id = np.array([r["example_id"] for r in rows])
    np.savez_compressed(
        os.path.join(ns_dir, f"{fam}.natscores.npz"),
        classes=np.array([c.replace("_", " ") for c in concepts]),
        layers=np.array(layers), y=y, token2ex=token2ex,
        ex_nat_split=ex_split, ex_example_id=ex_id,
        **{f"preds_{a}": v for a, v in preds.items()})
    return root


def build_ckpt(path, K, hidden, concepts, rng):
    torch.manual_seed(0)
    head = te.EncoderHead(hidden, K, "expA")
    torch.save({
        "step": 100, "head_state": head.state_dict(), "mode": "expA",
        "model_name": "tiny-qwen2-random", "hidden_size": hidden, "K": K,
        "concepts": concepts, "args": {}, "optimizer_state": {},
    }, path)


def build_tiny_model(qwen_tok, hidden=32, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(qwen_tok), hidden_size=hidden, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, pad_token_id=qwen_tok.pad_token_id)
    m = Qwen2Model(cfg)
    m.eval()
    return m


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ==================================================================== tests
def test_auroc_matches_manual():
    pos = np.array([3.0, 4.0, 5.0])
    neg = np.array([0.0, 1.0, 2.0])
    assert abs(g2.auroc(pos, neg) - 1.0) < 1e-9
    assert abs(g2.auroc(neg, pos) - 0.0) < 1e-9


def test_end_to_end(workdir, gemma_tok, qwen_tok):
    concepts = ["april", "asia", "north", "red"]
    fam = "famA"
    layers = [6, 8, 14]
    ablation_layer = 8
    rng = np.random.default_rng(0)
    # selection: give each (layer, concept) an arm (all "ridge" here)
    selection = {str(L): {c: {"arm": "ridge", "auroc": None} for c in concepts} for L in layers}

    pdir = os.path.join(workdir, "probe_set")
    K = build_probe_set(pdir, concepts, fam, layers, ablation_layer, selection, rng)

    edir = os.path.join(workdir, "evaldata")
    build_eval_data(edir, fam, concepts, layers, gemma_tok, rng, n_ex=40)

    hidden = 32
    ckpt = os.path.join(workdir, "best.pt")
    build_ckpt(ckpt, K, hidden, concepts, rng)

    out = os.path.join(workdir, "g2.json")
    args = Args(encoder_ckpt=ckpt, probe_set=pdir, eval_data=edir,
                device="cpu", max_qwen_tokens=128, bsz=8, out=out)
    model = build_tiny_model(qwen_tok, hidden)
    res = g2.run_g2(args, encoder_and_tok=(model, qwen_tok, "tiny-qwen2-random"))

    # json written
    assert os.path.exists(out)
    loaded = json.load(open(out))

    # every concept scored, AUROCs finite
    assert len(res["per_concept"]) == len(concepts), res["per_concept"].keys()
    for c, r in res["per_concept"].items():
        assert r["enc_auroc"] is not None and np.isfinite(r["enc_auroc"]), (c, r)
        assert r["gemma_auroc"] is not None and np.isfinite(r["gemma_auroc"]), (c, r)
        # planted signal -> gemma should detect strongly
        assert r["gemma_auroc"] > 0.9, (c, r["gemma_auroc"])
        assert len(r["per_layer"]) == len(layers)

    # verdict / tables populated
    v = res["verdict"]
    for key in ("pass_gate", "median_cc_ratio", "median_raw_ratio",
                "share_cc_ge_0.9", "share_raw_ge_0.9"):
        assert key in v, key
    assert set(res["per_family"].keys()) == {fam}
    assert res["per_family"][fam]["n_test_examples"] == 20  # 40 ex, even->test
    assert set(res["per_layer"].keys()) == {str(L) for L in layers}

    # reference recompute cross-check field present (stored auroc was None here,
    # so no diffs accumulated -> None is acceptable)
    assert "gemma_stored_vs_recomputed_max_absdiff" in res

    # retention ratios present and finite where cc defined
    for c, r in res["per_concept"].items():
        assert r["raw_ratio"] is not None
        # cc may be None only if gemma<=0.55; our planted gemma>0.9 so defined
        assert r["cc_ratio"] is not None, (c, r)


def test_stored_auroc_crosscheck(workdir, gemma_tok, qwen_tok):
    """When probe_set stores a selection auroc, the recomputed reference must
    match it exactly (same method, same examples) -> max |Δ| ~ 0."""
    concepts = ["april", "asia"]
    fam = "famB"
    layers = [6, 8, 14]
    rng = np.random.default_rng(1)
    root = os.path.join(workdir, "xc")
    build_eval_data(root, fam, concepts, layers, gemma_tok, rng, n_ex=24)
    # compute the true recomputed auroc, then store THAT in selection so the
    # crosscheck diff is ~0.
    rows, z = g2.load_family_truth(root, fam)
    test_ids = g2.test_split_ids(rows)
    sel = {str(L): {} for L in layers}
    for L in layers:
        for c in concepts:
            ci = g2.class_index(z, c)
            labels = g2.jsonl_labels(rows, c)
            a = g2.gemma_auroc_at_layer(z, ci, L, "ridge", labels, test_ids)
            sel[str(L)][c] = {"arm": "ridge", "auroc": float(a)}
    pdir = os.path.join(workdir, "xc_probe")
    K = build_probe_set(pdir, concepts, fam, layers, 8, sel, rng)
    ckpt = os.path.join(workdir, "xc_best.pt")
    build_ckpt(ckpt, K, 32, concepts, rng)
    out = os.path.join(workdir, "xc.json")
    args = Args(encoder_ckpt=ckpt, probe_set=pdir, eval_data=root,
                device="cpu", max_qwen_tokens=128, bsz=8, out=out)
    model = build_tiny_model(qwen_tok, 32)
    res = g2.run_g2(args, encoder_and_tok=(model, qwen_tok, "tiny"))
    d = res["gemma_stored_vs_recomputed_max_absdiff"]
    assert d is not None and d < 1e-9, d
    assert res["reference_source"] == ["natscores_recompute"], res["reference_source"]


def test_jsonl_only_fallback(workdir, gemma_tok, qwen_tok):
    """No natscores.npz staged -> gemma reference must fall back to probe_set
    stored selection AUROC, and the run must still produce the same gemma
    AUROCs as the natscores recompute (they are equal by construction)."""
    concepts = ["april", "asia"]
    fam = "famC"
    layers = [6, 8, 14]
    rng = np.random.default_rng(2)
    root = os.path.join(workdir, "jo")
    build_eval_data(root, fam, concepts, layers, gemma_tok, rng, n_ex=24)
    # store the recomputed auroc, then DELETE natscores so only the jsonl remains
    rows, z = g2.load_family_truth(root, fam)
    test_ids = g2.test_split_ids(rows)
    sel = {str(L): {} for L in layers}
    recomputed = {}
    for L in layers:
        for c in concepts:
            ci = g2.class_index(z, c)
            labels = g2.jsonl_labels(rows, c)
            a = g2.gemma_auroc_at_layer(z, ci, L, "ridge", labels, test_ids)
            sel[str(L)][c] = {"arm": "ridge", "auroc": float(a)}
            recomputed[(L, c)] = float(a)
    os.remove(os.path.join(root, "natscores", f"{fam}.natscores.npz"))

    pdir = os.path.join(workdir, "jo_probe")
    K = build_probe_set(pdir, concepts, fam, layers, 8, sel, rng)
    ckpt = os.path.join(workdir, "jo_best.pt")
    build_ckpt(ckpt, K, 32, concepts, rng)
    out = os.path.join(workdir, "jo.json")
    args = Args(encoder_ckpt=ckpt, probe_set=pdir, eval_data=root,
                device="cpu", max_qwen_tokens=128, bsz=8, out=out)
    model = build_tiny_model(qwen_tok, 32)
    res = g2.run_g2(args, encoder_and_tok=(model, qwen_tok, "tiny"))
    assert res["reference_source"] == ["probe_set_stored"], res["reference_source"]
    # gemma auroc used in the fallback must equal the stored (=recomputed) value
    for c, r in res["per_concept"].items():
        for lr in r["per_layer"]:
            assert abs(lr["gemma_auroc"] - recomputed[(lr["layer"], c)]) < 1e-9, (c, lr)


# ======================================================================= main
def main():
    workdir = tempfile.mkdtemp(prefix="stage7_g2_smoke_")
    print(f"[fixture root] {workdir}")
    try:
        gemma_tok = te.load_gemma_tokenizer("google/gemma-2-2b")
        qwen_tok = AutoTokenizer.from_pretrained(QWEN_MODEL)
        if qwen_tok.pad_token_id is None:
            qwen_tok.pad_token = qwen_tok.eos_token

        check("auroc_matches_manual", test_auroc_matches_manual)
        check("end_to_end_runs_and_writes_json",
              lambda: test_end_to_end(workdir, gemma_tok, qwen_tok))
        check("stored_auroc_crosscheck_matches_recompute",
              lambda: test_stored_auroc_crosscheck(workdir, gemma_tok, qwen_tok))
        check("jsonl_only_fallback_uses_stored_auroc",
              lambda: test_jsonl_only_fallback(workdir, gemma_tok, qwen_tok))
    finally:
        n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
        print("\n===== SUMMARY =====")
        for name, ok, err in RESULTS:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
            if not ok:
                print("    " + err.replace("\n", "\n    "))
        print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
        if n_fail == 0:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"[fixture kept for debugging: {workdir}]")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
