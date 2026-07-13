"""End-to-end smoke test for score_corpus.py (CPU-only, no real gemma-2-2b
weights, no network access to ClimbMix -- uses a tiny random Gemma2Model and
a local fake parquet shard via the STAGE7_SHARD_DIR test seam).

Run:
  python concept_probes/stage7_oracle/code/test_score_corpus.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_fixture  # noqa: E402
import score_corpus as sc  # noqa: E402

GEMMA_MODEL = "google/gemma-2-2b"


def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(GEMMA_MODEL)


def test_parse_shards():
    assert sc.parse_shards("320-322") == [320, 321, 322]
    assert sc.parse_shards("320-322,330") == [320, 321, 322, 330]
    assert sc.parse_shards("5,3,1") == [1, 3, 5]
    print("[test] parse_shards OK")


def test_quantize_roundtrip():
    rng = np.random.default_rng(0)
    n_cols = 12
    mean = rng.normal(size=n_cols)
    std = rng.uniform(0.3, 2.0, size=n_cols)
    quant = sc.compute_quant(mean, std)
    zero = np.asarray(quant["zero"])
    scale = np.asarray(quant["scale"])

    raw = mean[None, :] + rng.normal(size=(500, n_cols)) * std[None, :]
    # clip to the representable range so we're testing round-trip, not saturation
    raw = np.clip(raw, zero - 126 * scale, zero + 126 * scale)

    q = sc.quantize(raw, zero, scale)
    assert q.dtype == np.int8
    assert q.min() >= -127 and q.max() <= 127
    deq = sc.dequantize(q, zero, scale)
    err = np.abs(deq - raw)
    # round-to-nearest quantization error is bounded by half a bin width
    assert np.all(err <= scale / 2 + 1e-5), f"max err {err.max()} vs scale/2 {(scale/2).max()}"
    # and strictly less than a full bin (the smoke-test's literal requirement)
    assert np.all(err < scale + 1e-9)
    print("[test] quantize/dequantize round-trip OK "
          f"(max err {err.max():.6g}, min scale {scale.min():.6g})")


def test_running_stats_merge_matches_batch():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(4000, 5)) * 3 + 1
    # single accumulator
    single = sc.RunningStats(5)
    single.update_batch(data)
    # two accumulators combined
    a = sc.RunningStats(5)
    a.update_batch(data[:1500])
    b = sc.RunningStats(5)
    b.update_batch(data[1500:])
    combined = a.combine(b)
    assert combined.count == single.count == 4000
    np.testing.assert_allclose(combined.mean, single.mean, atol=1e-8)
    np.testing.assert_allclose(combined.std(), single.std(), atol=1e-6)
    np.testing.assert_allclose(combined.mean, data.mean(axis=0), atol=1e-8)
    np.testing.assert_allclose(combined.std(), data.std(axis=0), atol=1e-6)
    print("[test] RunningStats parallel-merge matches single-pass OK")


def test_ablation_layer_outside_chosen_layers():
    """select_probes.py (Phase 0, written independently) can pick an
    ablation_layer that is NOT one of the 3 chosen probe `layers` (its vote
    is a mode over causal cards, decoupled from the AUROC-based layer
    choice). score_corpus.ProbeSet must either use an optional
    nat_mean_abl/nat_std_abl pair in that case, or fail loudly -- never
    silently mis-normalize. Exercise both branches."""
    tmp_root = tempfile.mkdtemp(prefix="stage7_abl_test_")
    try:
        # (a) ablation_layer outside `layers`, WITH the optional arrays -> OK
        ok_dir = os.path.join(tmp_root, "ok")
        make_fixture.build_probe_set(ok_dir, ablation_layer=5, include_abl_arrays=True)
        probe = sc.ProbeSet(ok_dir)
        assert probe.ablation_layer == 5
        assert probe.abl_idx is None
        assert probe.nat_mean_abl.shape == (make_fixture.TINY_HIDDEN,)
        assert set(probe.needed_hidden_layers()) == {1, 2, 3, 5}

        head = sc.ScoreHead(probe, torch.device("cpu"))
        hs = {l: torch.randn(2, 7, make_fixture.TINY_HIDDEN) for l in [1, 2, 3, 5]}
        raw = head.score(hs)
        assert raw.shape == (2, 7, 4 * len(make_fixture.CONCEPTS))
        assert torch.isfinite(raw).all()
        print("[test] ablation_layer outside `layers` WITH nat_mean_abl/nat_std_abl -> OK")

        # (b) ablation_layer outside `layers`, WITHOUT the optional arrays -> loud failure
        bad_dir = os.path.join(tmp_root, "bad")
        make_fixture.build_probe_set(bad_dir, ablation_layer=5, include_abl_arrays=False)
        try:
            sc.ProbeSet(bad_dir)
            raise RuntimeError("expected ProbeSet(...) to raise AssertionError")
        except AssertionError as e:
            assert "nat_mean_abl" in str(e)
            print("[test] ablation_layer outside `layers` WITHOUT arrays -> loud failure OK")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_end_to_end():
    tmp_root = tempfile.mkdtemp(prefix="stage7_score_corpus_test_")
    try:
        tok = _get_tokenizer()
        fixture = make_fixture.build_all(tmp_root, tok.vocab_size)
        out_dir = os.path.join(tmp_root, "out")

        os.environ["STAGE7_SHARD_DIR"] = fixture["shard_dir"]

        # ground truth: tokenize the fixture texts exactly like score_corpus
        # does (add_special_tokens=False, truncate to 2048), to predict which
        # docs should be kept/skipped and what n_tokens each doc should have.
        expected_docs = {}
        for i, text in enumerate(fixture["texts"]):
            ids = tok(text, add_special_tokens=False)["input_ids"][:sc.MAX_DOC_TOKENS]
            if len(ids) >= sc.MIN_DOC_TOKENS:
                expected_docs[i] = ids
        n_short_skipped = len(fixture["texts"]) - len(expected_docs)
        assert n_short_skipped == 4, f"expected 4 short docs skipped, got {n_short_skipped}"
        long_doc_idx = len(fixture["texts"]) - 1
        assert long_doc_idx in expected_docs
        assert len(expected_docs[long_doc_idx]) == sc.MAX_DOC_TOKENS, (
            "long fixture doc should truncate to exactly MAX_DOC_TOKENS")

        common_args = [
            "--probe-set", fixture["probe_dir"],
            "--shards", "999",
            "--out", out_dir,
            "--batch-size", "4",
            "--attn", "eager",
            "--calib-tokens", "200",
            "--quant-json", os.path.join(out_dir, "quant.json"),
            "--device", "cpu",
            "--model", GEMMA_MODEL,
            "--heartbeat", os.path.join(out_dir, "hb.txt"),
            "--tiny-model-config", fixture["tiny_model_config"],
        ]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sc.main(common_args)
        print(buf.getvalue())

        paths = sc.shard_output_paths(__import__("pathlib").Path(out_dir), 999)
        assert paths["tokens"].exists()
        assert paths["scores"].exists()
        assert paths["docs"].exists()
        assert paths["stats"].exists()
        assert os.path.exists(os.path.join(out_dir, "quant.json"))
        assert os.path.exists(os.path.join(out_dir, "hb.txt"))

        tokens = np.load(paths["tokens"])
        scores = np.load(paths["scores"])
        assert tokens.dtype == np.int32, tokens.dtype
        assert scores.dtype == np.int8, scores.dtype

        K = 3
        assert scores.shape[1] == 4 * K, scores.shape

        with open(paths["docs"]) as f:
            docs = [json.loads(l) for l in f]
        assert len(docs) == len(expected_docs), (len(docs), len(expected_docs))

        # doc index arithmetic: start/n bookkeeping must exactly tile the memmaps
        n_sum = sum(d["n"] for d in docs)
        assert n_sum == len(tokens) == scores.shape[0], (n_sum, len(tokens), scores.shape[0])
        by_doc = {d["doc"]: d for d in docs}
        assert set(by_doc.keys()) == set(expected_docs.keys())
        for doc_idx, ids in expected_docs.items():
            d = by_doc[doc_idx]
            assert d["n"] == len(ids), (doc_idx, d["n"], len(ids))
            got = tokens[d["start"]:d["start"] + d["n"]]
            np.testing.assert_array_equal(got, np.asarray(ids, dtype=np.int32),
                                           err_msg=f"doc {doc_idx} token ids mismatch")
        print(f"[test] doc index arithmetic OK ({len(docs)} docs, {n_sum} tokens)")

        with open(paths["stats"]) as f:
            stats = json.load(f)
        assert stats["n_tokens"] == n_sum
        assert len(stats["mean"]) == 4 * K

        with open(os.path.join(out_dir, "quant.json")) as f:
            quant = json.load(f)
        assert len(quant["zero"]) == len(quant["scale"]) == 4 * K
        assert all(s > 0 for s in quant["scale"])
        print("[test] shard output shapes/dtypes + quant.json OK")

        # --- resume-skip: rerun, outputs must be untouched and a skip logged ---
        mtimes_before = {k: p.stat().st_mtime_ns for k, p in paths.items()}
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            sc.main(common_args)
        out2 = buf2.getvalue()
        assert "already done, skipping" in out2, out2
        mtimes_after = {k: p.stat().st_mtime_ns for k, p in paths.items()}
        assert mtimes_before == mtimes_after, "resume should not rewrite existing shard outputs"
        print("[test] resume-skip OK")

        # --- merge-stats ---
        buf3 = io.StringIO()
        with contextlib.redirect_stdout(buf3):
            sc.main(["--out", out_dir, "--merge-stats"])
        corpus_stats_path = os.path.join(out_dir, "corpus_stats.json")
        assert os.path.exists(corpus_stats_path)
        with open(corpus_stats_path) as f:
            corpus_stats = json.load(f)
        assert corpus_stats["n_tokens"] == n_sum
        assert len(corpus_stats["mean"]) == len(corpus_stats["std"]) == 4 * K
        print("[test] --merge-stats OK")

        print("[test] end-to-end smoke test PASSED")
    finally:
        os.environ.pop("STAGE7_SHARD_DIR", None)
        shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    test_parse_shards()
    test_quantize_roundtrip()
    test_running_stats_merge_matches_batch()
    test_ablation_layer_outside_chosen_layers()
    test_end_to_end()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
