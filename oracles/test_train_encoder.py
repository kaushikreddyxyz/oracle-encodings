#!/usr/bin/env python3
"""Smoke test for train_encoder.py (Stage 7-Oracle Phase 2/3).

Plain assertion-script style (matches this repo's --smoke convention, no
pytest). Builds a tiny synthetic fixture (fake ClimbMix shard parquet, fake
tokens_<sid>.npy/scores_<sid>.npy/docs_<sid>.jsonl/quant.json/
corpus_stats.json, fake probe_set.json/probe_set_arrays.npz) using REAL
small tokenizers (gemma-2-2b's tokenizer as the "gemma" side -- confirmed
available offline / not gated in this environment; Qwen3-0.6B-Base's
tokenizer as the target -- also confirmed available offline) so the
char-offset alignment machinery is genuinely exercised, and a tiny
*randomly initialized* Qwen2Config model standing in for the encoder (fast
on CPU; real Qwen3-0.6B-Base is loaded only for the "does it load" check,
never forwarded here).

Run: python test_train_encoder.py
Exits 0 iff every check passes; prints PASS/FAIL per check.
"""
import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# _align_fallback.py (imported directly by one test) lives in repo_root/attribution
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "attribution"))

import train_encoder as te  # noqa: E402
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, None))
        print(f"[PASS] {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, f"{e}\n{traceback.format_exc()}"))
        print(f"[FAIL] {name}: {e}")


# ============================================================== fixture data
GEMMA_MODEL = "google/gemma-2-2b"
QWEN_MODEL = "Qwen/Qwen3-0.6B-Base"

PARAGRAPHS = [
    "The quick brown fox jumps over the lazy dog near the old stone bridge "
    "every single morning before the sun has fully risen above the hills.",
    "In January the markets were volatile, but by August analysts expected "
    "a rebound driven by strong earnings across the continent of Africa.",
    "Blue-green algae blooms were reported across three lakes in Asia this "
    "autumn, prompting officials to issue a public health advisory notice.",
    "The committee will convene on December 3rd to review the annual budget "
    "and discuss whether to relocate the regional headquarters next spring.",
    "A 2024 study of 15,832 participants found that daily exercise reduced "
    "the risk of cardiovascular disease by roughly twelve point four percent.",
    "North America and South America share a long, mountainous border region "
    "known for its unusually high biodiversity and frequent seismic activity.",
    "The orchestra performed a rare nineteenth century symphony in April, "
    "drawing a record crowd despite the unseasonably cold weather outside.",
    "Researchers in Europe published new findings about deep ocean currents "
    "and their long-term effect on global weather patterns this century.",
    "The bakery on Maple Street sells fresh sourdough every Wednesday and "
    "Saturday, and the line often stretches around the block by mid morning.",
    "Engineers completed the bridge inspection in March, noting minor rust "
    "on the eastern support beams that will require repair before winter.",
]


def make_docs(n, rng):
    docs = []
    for i in range(n):
        k = rng.integers(2, 4)
        idxs = rng.choice(len(PARAGRAPHS), size=k, replace=False)
        docs.append(" ".join(PARAGRAPHS[j] for j in idxs))
    return docs


def write_shard_parquet(path, texts):
    tbl = pa.table({"text": pa.array(texts, type=pa.large_string())})
    pq.write_table(tbl, path)


def build_probe_set(probe_set_dir, K, families, layers, ablation_layer, rng, seed_gram=0):
    os.makedirs(probe_set_dir, exist_ok=True)
    concepts = [f"c{ i}" for i in range(K)]
    fam_map = {concepts[i]: families[i % len(families)] for i in range(K)}
    D_MODEL = te.D_MODEL_GEMMA
    W = rng.normal(0, 0.05, size=(3, K, D_MODEL)).astype(np.float32)
    b = rng.normal(0, 0.1, size=(3, K)).astype(np.float32)
    nat_mean = rng.normal(0, 1, size=(3, D_MODEL)).astype(np.float32)
    nat_std = rng.uniform(0.5, 1.5, size=(3, D_MODEL)).astype(np.float32)
    W_dom_abl = rng.normal(0, 0.05, size=(K, D_MODEL)).astype(np.float32)
    abl_idx = layers.index(ablation_layer)
    nat_std_abl = nat_std[abl_idx]
    # STANDARDIZED-space Gram (matches select_probes.py post-fix; the raw
    # Gram of σ⊙w was the pre-4:15AM bug — see DESIGN.md).
    G_dom = (W_dom_abl @ W_dom_abl.T + 1e-3 * np.eye(K)).astype(np.float32)
    G_dom_inv = np.linalg.inv(G_dom).astype(np.float32)
    b_dom_abl = rng.normal(0, 0.1, size=(K,)).astype(np.float32)
    t_nat_dom = rng.normal(0, 0.5, size=(K,)).astype(np.float32)

    meta = {
        "layers": layers,
        "ablation_layer": ablation_layer,
        "concepts": concepts,
        "families": fam_map,
        "selection": {},
        "s95": {},
        "corpus_stats": None,
        "meta": {"fixture": True},
    }
    with open(os.path.join(probe_set_dir, "probe_set.json"), "w") as f:
        json.dump(meta, f)
    np.savez(os.path.join(probe_set_dir, "probe_set_arrays.npz"),
              W=W, b=b, nat_mean=nat_mean, nat_std=nat_std, W_dom_abl=W_dom_abl,
              b_dom_abl=b_dom_abl, t_nat_dom=t_nat_dom, G_dom=G_dom, G_dom_inv=G_dom_inv,
              layer_index=np.array(layers, dtype=np.int64))
    return concepts, fam_map


def build_score_shard(scores_dir, climbmix_dir, sid, texts, gemma_tok, K,
                       max_gemma_tokens, min_gemma_tokens, rng):
    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(climbmix_dir, exist_ok=True)
    write_shard_parquet(os.path.join(climbmix_dir, f"shard_{sid:05d}.parquet"), texts)

    all_ids = []
    docs_rows = []
    all_scores_raw = []
    offset = 0
    for i, text in enumerate(texts):
        # Must match score_corpus.py's real convention exactly: BOS-free,
        # add_special_tokens=False, sliced (not tokenizer-truncated) to
        # max_gemma_tokens -- see train_encoder.py::process_doc docstring.
        enc = gemma_tok(text, add_special_tokens=False)
        ids = enc["input_ids"][:max_gemma_tokens]
        if len(ids) < min_gemma_tokens:
            continue
        n = len(ids)
        # synthetic but structured (not pure noise) per-column scores: a
        # smooth function of token id + column index, so a head has SOME
        # learnable signal to fit within a 20-step smoke test.
        toks = np.array(ids, dtype=np.float32)
        cols = np.arange(4 * K, dtype=np.float32)
        raw = (np.sin(0.01 * toks[:, None] + cols[None, :]) * 3.0
               + 0.05 * rng.normal(size=(n, 4 * K))).astype(np.float32)
        all_ids.append(np.array(ids, dtype=np.int32))
        all_scores_raw.append(raw)
        docs_rows.append({"doc": i, "start": offset, "n": n})
        offset += n

    tokens_arr = np.concatenate(all_ids) if all_ids else np.zeros((0,), dtype=np.int32)
    scores_raw = np.concatenate(all_scores_raw, axis=0) if all_scores_raw else np.zeros((0, 4 * K), dtype=np.float32)
    return tokens_arr, scores_raw, docs_rows


def write_quant_and_scores(scores_dir, sid, tokens_arr, scores_raw, docs_rows, quant=None):
    np.save(os.path.join(scores_dir, f"tokens_{sid:05d}.npy"), tokens_arr)
    if quant is None:
        mean = scores_raw.mean(axis=0)
        std = scores_raw.std(axis=0) + 1e-6
        zero = mean
        scale = 4 * std / 127.0
        quant = (zero, scale)
    zero, scale = quant
    q = np.clip(np.round((scores_raw - zero[None, :]) / scale[None, :]), -127, 127).astype(np.int8)
    np.save(os.path.join(scores_dir, f"scores_{sid:05d}.npy"), q)
    with open(os.path.join(scores_dir, f"docs_{sid:05d}.jsonl"), "w") as f:
        for d in docs_rows:
            f.write(json.dumps(d) + "\n")
    return quant


def build_fixture(root, gemma_tok, K=6, max_gemma_tokens=64, min_gemma_tokens=16):
    rng = np.random.default_rng(0)
    scores_dir = os.path.join(root, "scores")
    climbmix_dir = os.path.join(root, "climbmix")
    probe_set_dir = os.path.join(root, "probe_set")

    train_sid, val_sid = 320, 321
    train_texts = make_docs(24, rng)
    val_texts = make_docs(12, rng)

    tokens_t, scores_t, docs_t = build_score_shard(
        scores_dir, climbmix_dir, train_sid, train_texts, gemma_tok, K,
        max_gemma_tokens, min_gemma_tokens, rng)
    tokens_v, scores_v, docs_v = build_score_shard(
        scores_dir, climbmix_dir, val_sid, val_texts, gemma_tok, K,
        max_gemma_tokens, min_gemma_tokens, rng)

    all_scores = np.concatenate([scores_t, scores_v], axis=0)
    mean = all_scores.mean(axis=0)
    std = all_scores.std(axis=0) + 1e-6
    zero, scale = mean, 4 * std / 127.0

    write_quant_and_scores(scores_dir, train_sid, tokens_t, scores_t, docs_t, quant=(zero, scale))
    write_quant_and_scores(scores_dir, val_sid, tokens_v, scores_v, docs_v, quant=(zero, scale))
    with open(os.path.join(scores_dir, "quant.json"), "w") as f:
        json.dump({"zero": zero.tolist(), "scale": scale.tolist()}, f)
    with open(os.path.join(scores_dir, "corpus_stats.json"), "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)

    layers = [2, 4, 6]
    ablation_layer = 4
    families = ["famA", "famB"]
    build_probe_set(probe_set_dir, K, families, layers, ablation_layer, rng)

    return {
        "scores_dir": scores_dir, "climbmix_dir": climbmix_dir,
        "probe_set_dir": probe_set_dir, "train_sid": train_sid, "val_sid": val_sid,
        "K": K,
    }


def build_tiny_model(qwen_tok, hidden_size=32, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=len(qwen_tok), hidden_size=hidden_size, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, pad_token_id=qwen_tok.pad_token_id,
    )
    model = Qwen2Model(cfg)
    model.eval()
    return model, qwen_tok, "tiny-qwen2-random"


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_args(fx, out_dir, mode, freeze_encoder, max_steps=20, eval_every=5,
              gemma_model=GEMMA_MODEL, resume=None, encoder_from=None):
    return Args(
        scores=fx["scores_dir"], climbmix_dir=fx["climbmix_dir"],
        probe_set=fx["probe_set_dir"], mode=mode, freeze_encoder=freeze_encoder,
        train_shards=str(fx["train_sid"]), val_shards=str(fx["val_sid"]),
        model=QWEN_MODEL, gemma_model=gemma_model, lr=5e-3, max_steps=max_steps,
        bsz_docs=2, grad_accum=1, warmup_steps=2, eval_every=eval_every,
        eval_tokens=5000, max_gemma_tokens=64, max_qwen_tokens=96,
        min_gemma_tokens=16, assert_first_n_docs=15, early_stop_r2_delta=0.005,
        early_stop_window_frac=0.2, seed=0, device="cpu", resume=resume,
        encoder_from=encoder_from,
        heartbeat_path=os.path.join(out_dir, "hb_train.txt"), heartbeat_interval=1000.0,
        out=out_dir,
    )


# ==================================================================== tests
def test_qwen3_tokenizer_loads():
    tok = AutoTokenizer.from_pretrained(QWEN_MODEL)
    assert tok.is_fast
    enc = tok("hello world", return_offsets_mapping=True)
    assert len(enc["input_ids"]) > 0


def test_align_fallback_prefix():
    from _align_fallback import gemma_to_qwen_map, crossing_rate
    # gemma tokens end at chars 3, 7, 10; qwen tokens end at chars 3, 6, 10
    gemma_offsets = [(0, 3), (3, 7), (7, 10)]
    qwen_offsets = [(0, 3), (3, 6), (6, 10)]
    m = gemma_to_qwen_map("abcdefghij", gemma_offsets, qwen_offsets, mode="prefix")
    # gemma tok0 end=3 -> qwen tok0 (end=3) is the last with end<=3 -> idx 0
    # gemma tok1 end=7 -> qwen ends <=7: 3,6 -> last idx 1
    # gemma tok2 end=10 -> qwen ends <=10: 3,6,10 -> last idx 2
    assert m.tolist() == [0, 1, 2], m.tolist()

    # no matching qwen boundary at all (all qwen ends > every gemma end)
    m2 = gemma_to_qwen_map("ab", [(0, 1)], [(0, 5)], mode="prefix")
    assert m2.tolist() == [-1]

    cr = crossing_rate("abcdefghij", gemma_offsets, qwen_offsets)
    # gemma ends {3,7,10}; qwen ends {3,6,10}; 7 has no exact match -> 1/3
    assert abs(cr - (1 / 3)) < 1e-9, cr


def test_r2_accumulator():
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(1)
    y = rng.normal(size=(200, 4))
    pred = y + rng.normal(scale=0.3, size=(200, 4))
    acc = te.R2Accumulator(4)
    for i in range(0, 200, 20):
        acc.update(y[i:i + 20], pred[i:i + 20])
    got = acc.r2()
    want = r2_score(y, pred, multioutput="raw_values")
    assert np.allclose(got, want, atol=1e-6), (got, want)


def test_corpus_stats_fallback(tmp_root):
    scores_dir = os.path.join(tmp_root, "cs_fallback")
    os.makedirs(scores_dir, exist_ok=True)
    zero = np.array([1.0, 2.0], dtype=np.float32)
    scale = np.array([0.1, 0.2], dtype=np.float32)
    with open(os.path.join(scores_dir, "quant.json"), "w") as f:
        json.dump({"zero": zero.tolist(), "scale": scale.tolist()}, f)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mean, std, real = te.load_corpus_stats(scores_dir, zero, scale)
        assert not real
        assert len(w) == 1 and "corpus_stats.json not found" in str(w[0].message)
    assert np.allclose(mean, zero)
    assert np.allclose(std, scale * 127.0 / 4.0)


def test_process_doc_truncation(gemma_tok, qwen_tok):
    text = " ".join(PARAGRAPHS)  # long text
    enc = gemma_tok(text, add_special_tokens=False)
    ids = np.array(enc["input_ids"][:200], dtype=np.int32)
    K = 3
    doc = {"sid": 999, "doc_idx": 0, "text": text, "gemma_ids": ids,
           "scores_raw_i8": np.zeros((len(ids), 4 * K), dtype=np.int8)}
    out = te.process_doc(doc, gemma_tok, qwen_tok, max_gemma_tokens=200,
                          max_qwen_tokens=5, min_gemma_tokens=4, assert_tokens=True)
    assert out is not None
    assert len(out["qwen_ids"]) <= 5, len(out["qwen_ids"])
    assert out["map_idx"].max() < len(out["qwen_ids"])
    assert out["scores_raw_i8"].shape[0] == out["map_idx"].shape[0]


def test_token_assertion_hard_fail(gemma_tok, qwen_tok):
    text = PARAGRAPHS[0]
    enc = gemma_tok(text, add_special_tokens=False)
    ids = np.array(enc["input_ids"][:64], dtype=np.int32)
    corrupted = ids.copy()
    corrupted[min(3, len(corrupted) - 1)] += 12345  # deliberately wrong
    K = 2
    doc_bad = {"sid": 1, "doc_idx": 0, "text": text, "gemma_ids": corrupted,
               "scores_raw_i8": np.zeros((len(corrupted), 4 * K), dtype=np.int8)}
    raised = False
    try:
        te.process_doc(doc_bad, gemma_tok, qwen_tok, max_gemma_tokens=64,
                        max_qwen_tokens=96, min_gemma_tokens=4, assert_tokens=True)
    except RuntimeError as e:
        raised = True
        assert "TOKEN-ID REPRODUCTION FAILED" in str(e)
    assert raised, "expected RuntimeError on corrupted stored tokens"

    doc_good = {"sid": 1, "doc_idx": 0, "text": text, "gemma_ids": ids,
                "scores_raw_i8": np.zeros((len(ids), 4 * K), dtype=np.int8)}
    out = te.process_doc(doc_good, gemma_tok, qwen_tok, max_gemma_tokens=64,
                          max_qwen_tokens=96, min_gemma_tokens=4, assert_tokens=True)
    assert out is not None


def _run_mode(fx, out_dir, mode, freeze, gemma_tok, qwen_tok):
    model, tok, name = build_tiny_model(qwen_tok)
    args = make_args(fx, out_dir, mode, freeze)
    result = te.run_training(args, encoder_and_tok=(model, tok, name))
    return args, result


def test_mode_expA(fx, gemma_tok, qwen_tok, workdir):
    out_dir = os.path.join(workdir, "out_expA")
    args, result = _run_mode(fx, out_dir, "expA", True, gemma_tok, qwen_tok)
    _assert_common(out_dir, result, "expA")


def test_mode_expB_fixed(fx, gemma_tok, qwen_tok, workdir):
    out_dir = os.path.join(workdir, "out_expB_fixed")
    args, result = _run_mode(fx, out_dir, "expB-fixed", True, gemma_tok, qwen_tok)
    _assert_common(out_dir, result, "expB-fixed")


def test_mode_expB_learn(fx, gemma_tok, qwen_tok, workdir):
    out_dir = os.path.join(workdir, "out_expB_learn")
    args, result = _run_mode(fx, out_dir, "expB-learn", True, gemma_tok, qwen_tok)
    _assert_common(out_dir, result, "expB-learn")
    metrics = _read_metrics(out_dir)
    assert "down_cosine" in metrics[-1], metrics[-1]
    assert len(metrics[-1]["down_cosine"]) == fx["K"]


def test_mode_full_ft(fx, gemma_tok, qwen_tok, workdir):
    out_dir = os.path.join(workdir, "out_full_ft")
    model, tok, name = build_tiny_model(qwen_tok)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    args = make_args(fx, out_dir, "expA", False, max_steps=6, eval_every=3)
    result = te.run_training(args, encoder_and_tok=(model, tok, name))
    after = model.state_dict()
    changed = any(not torch.allclose(before[k], after[k]) for k in before)
    assert changed, "full-ft: encoder weights should have changed after training"
    ckpt = torch.load(os.path.join(out_dir, "last.pt"), map_location="cpu", weights_only=False)
    assert "encoder_state" in ckpt, "full-ft checkpoint must save encoder_state"
    _assert_common(out_dir, result, "expA")


def _read_metrics(out_dir):
    path = os.path.join(out_dir, "metrics.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _assert_common(out_dir, result, mode):
    metrics = _read_metrics(out_dir)
    assert len(metrics) >= 2, f"expected >=2 eval points, got {len(metrics)}"
    losses = [m["loss"] for m in metrics]
    assert losses[-1] < losses[0], f"{mode}: loss did not decrease: {losses}"
    for m in metrics:
        assert "primary_metric" in m
        assert np.isfinite(m["primary_metric"])
    assert os.path.exists(os.path.join(out_dir, "best.pt"))
    assert os.path.exists(os.path.join(out_dir, "last.pt"))
    assert os.path.exists(os.path.join(out_dir, "hb_train.txt"))
    with open(os.path.join(out_dir, "hb_train.txt")) as f:
        hb = json.loads(f.readline())
    assert "step" in hb and "loss" in hb


def test_checkpoint_resume(fx, gemma_tok, qwen_tok, workdir):
    out_dir = os.path.join(workdir, "out_resume")
    model, tok, name = build_tiny_model(qwen_tok, seed=1)
    args1 = make_args(fx, out_dir, "expA", True, max_steps=8, eval_every=4)
    r1 = te.run_training(args1, encoder_and_tok=(model, tok, name))
    step1 = r1["final_step"]
    ckpt_path = os.path.join(out_dir, "last.pt")
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert saved["step"] == step1

    # fresh head with a DIFFERENT random init to prove load_checkpoint restores it
    head_fresh = te.EncoderHead(32, fx["K"], "expA")
    head_loaded = te.EncoderHead(32, fx["K"], "expA")
    model2, tok2, name2 = build_tiny_model(qwen_tok, seed=7)
    opt = torch.optim.AdamW(head_loaded.parameters(), lr=1e-3)
    before_diff = not torch.allclose(head_fresh.up.weight, head_loaded.up.weight)
    step_loaded = te.load_checkpoint(ckpt_path, head_loaded, model2, opt, full_ft=False)
    assert step_loaded == step1
    assert torch.allclose(head_loaded.up.weight, saved["head_state"]["up.weight"]), \
        "resume did not restore head weights"

    # full integration resume: continue training past step1
    model3, tok3, name3 = build_tiny_model(qwen_tok, seed=1)  # fresh instance, state irrelevant since frozen
    args2 = make_args(fx, out_dir, "expA", True, max_steps=step1 + 6, eval_every=3, resume=ckpt_path)
    r2 = te.run_training(args2, encoder_and_tok=(model3, tok3, name3))
    assert r2["final_step"] > step1, (r2["final_step"], step1)
    assert r2["final_step"] <= step1 + 6


def test_encoder_from(fx, gemma_tok, qwen_tok, workdir):
    """--encoder-from loads a full-FT checkpoint's encoder_state (strict) into
    the base model before a FROZEN Exp-B run (SPEC Phase 3 encoder-reuse path).
    """
    # 1) make a full-FT expA checkpoint whose encoder weights differ from init.
    src_dir = os.path.join(workdir, "out_encfrom_src")
    model_src, tok, name = build_tiny_model(qwen_tok, seed=3)
    args_src = make_args(fx, src_dir, "expA", False, max_steps=6, eval_every=3)
    te.run_training(args_src, encoder_and_tok=(model_src, tok, name))
    ckpt_path = os.path.join(src_dir, "last.pt")
    saved_enc = torch.load(ckpt_path, map_location="cpu", weights_only=False)["encoder_state"]

    # 2) fresh, DIFFERENTLY-seeded encoder; frozen expB-fixed run with --encoder-from.
    dst_dir = os.path.join(workdir, "out_encfrom_dst")
    model_dst, tok2, name2 = build_tiny_model(qwen_tok, seed=99)
    before = {k: v.clone() for k, v in model_dst.state_dict().items()}
    # sanity: the two inits genuinely differ
    assert any(not torch.allclose(before[k], saved_enc[k]) for k in before), \
        "test setup: seeds should differ so the load is observable"
    args_dst = make_args(fx, dst_dir, "expB-fixed", True, max_steps=6, eval_every=3,
                         encoder_from=ckpt_path)
    result = te.run_training(args_dst, encoder_and_tok=(model_dst, tok2, name2))
    after = model_dst.state_dict()
    # 3) frozen encoder must now EQUAL the loaded checkpoint (unchanged by training).
    for k in saved_enc:
        assert torch.allclose(after[k].cpu(), saved_enc[k].cpu(), atol=1e-5), \
            f"--encoder-from did not load param {k}"
    _assert_common(dst_dir, result, "expB-fixed")


# ======================================================================= main
def main():
    workdir = tempfile.mkdtemp(prefix="stage7_train_encoder_smoke_")
    print(f"[fixture root] {workdir}")
    try:
        gemma_tok = te.load_gemma_tokenizer(GEMMA_MODEL)
        qwen_tok = AutoTokenizer.from_pretrained(QWEN_MODEL)
        if qwen_tok.pad_token_id is None:
            qwen_tok.pad_token = qwen_tok.eos_token

        check("qwen3_tokenizer_loads", test_qwen3_tokenizer_loads)
        check("align_fallback_prefix_mode", test_align_fallback_prefix)
        check("r2_accumulator_matches_sklearn", test_r2_accumulator)
        check("corpus_stats_fallback_warns", lambda: test_corpus_stats_fallback(workdir))
        check("process_doc_truncation_path", lambda: test_process_doc_truncation(gemma_tok, qwen_tok))
        check("token_id_assertion_hard_fails_on_drift", lambda: test_token_assertion_hard_fail(gemma_tok, qwen_tok))

        fixdir = os.path.join(workdir, "fixture")
        fx = build_fixture(fixdir, gemma_tok, K=6)
        print(f"[fixture] K={fx['K']} train_sid={fx['train_sid']} val_sid={fx['val_sid']}")

        check("mode_expA_20steps", lambda: test_mode_expA(fx, gemma_tok, qwen_tok, workdir))
        check("mode_expB_fixed_20steps", lambda: test_mode_expB_fixed(fx, gemma_tok, qwen_tok, workdir))
        check("mode_expB_learn_20steps", lambda: test_mode_expB_learn(fx, gemma_tok, qwen_tok, workdir))
        check("mode_full_ft_encoder_updates", lambda: test_mode_full_ft(fx, gemma_tok, qwen_tok, workdir))
        check("checkpoint_resume", lambda: test_checkpoint_resume(fx, gemma_tok, qwen_tok, workdir))
        check("encoder_from_loads_strict", lambda: test_encoder_from(fx, gemma_tok, qwen_tok, workdir))
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
