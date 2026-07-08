"""End-to-end smoke test for verify_closed_form.py (CPU-only, tiny random
Gemma2Model, no real gemma-2-2b weights, no network access to ClimbMix).

Reuses make_fixture.py's idioms (tiny Gemma2Config, fake probe_set,
fake ClimbMix shard) and RUNS score_corpus.py on the fixture to produce the
scored shard store, so the Phase-1 store conventions this script depends on
(BOS handling, quantization, docs.jsonl bookkeeping) cannot drift out of
sync with score_corpus.py's own smoke test (test_score_corpus.py).

FIXTURE GRAM HISTORY: make_fixture.build_probe_set() originally computed
G_dom from the RAW-space Gram of d_c = nat_std_abl * w_c -- the exact bug
DESIGN.md documents as fixed in select_probes.py ~4:15 AM ("G_dom must be
the STANDARDIZED-space Gram W_dom_abl @ W_dom_abl.T"). make_fixture.py has
since been FIXED upstream (commit 29015df) to the standardized-space
convention; test_gram_consistency_check() below asserts both that the fixed
fixture passes VerifyProbeSet.gram_consistency_check() and that a
deliberately raw-space-corrupted G_dom_inv is still caught.
`fix_gram_to_standardized_space()` is retained (it now merely swaps
make_fixture's 1e-3-regularized inverse for an exact pinv, keeping check 1's
1e-3 tolerance sharp on the tiny fixture).

Run:
  python concept_probes/stage7_oracle/code/test_verify_closed_form.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_fixture  # noqa: E402
import score_corpus as sc  # noqa: E402
import verify_closed_form as vcf  # noqa: E402

GEMMA_MODEL = "google/gemma-2-2b"


def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(GEMMA_MODEL)


def fix_gram_to_standardized_space(probe_set_dir: str) -> None:
    """Recompute G_dom/G_dom_inv as the exact (pinv) STANDARDIZED-space Gram
    of W_dom_abl. make_fixture.py already uses the standardized-space Gram
    (fixed upstream); this now only replaces its 1e-3-regularized inverse
    with an exact pseudo-inverse so check 1's 1e-3 tolerance stays sharp.
    See module docstring."""
    p = Path(probe_set_dir) / "probe_set_arrays.npz"
    d = dict(np.load(p))
    W_dom_abl = d["W_dom_abl"].astype(np.float32)
    G_dom = (W_dom_abl @ W_dom_abl.T).astype(np.float32)
    G_dom_inv = np.linalg.pinv(G_dom).astype(np.float32)
    d["G_dom"] = G_dom
    d["G_dom_inv"] = G_dom_inv
    np.savez(p, **d)


def test_gram_consistency_check():
    """make_fixture.py's build_probe_set() was FIXED upstream (commit
    29015df "fixture Gram fix"): it now builds the standardized-space Gram
    W_dom_abl @ W_dom_abl.T, so (a) the fixture must pass
    gram_consistency_check as-is, and (b) a probe_set whose G_dom_inv is
    deliberately corrupted back to the RAW-space Gram inverse (the pre-4:15AM
    bug DESIGN.md documents) must still be CAUGHT by
    gram_consistency_check. (This test replaces the earlier
    test_gram_bug_is_real canary, which asserted the fixture bug was still
    present -- per its own docstring, it became obsolete when make_fixture
    was fixed.)"""
    tmp_root = tempfile.mkdtemp(prefix="stage7_gram_bug_test_")
    try:
        probe_dir = os.path.join(tmp_root, "probe_set")
        make_fixture.build_probe_set(probe_dir)
        ps = vcf.VerifyProbeSet(probe_dir)
        gc = ps.gram_consistency_check()
        assert gc["consistent"], (
            f"make_fixture.py's G_dom is supposed to be standardized-space "
            f"(fixed upstream) but gram_consistency_check flags it: {gc}")
        print(f"[test] fixed make_fixture.py Gram passes gram_consistency_check "
              f"(rel dist={gc['rel_frobenius_dist_G_dom_inv_vs_standardized_space']:.4g})")

        # (b) regression guard: corrupt G_dom_inv to the raw-space convention
        # and confirm it is caught.
        p = Path(probe_dir) / "probe_set_arrays.npz"
        d = dict(np.load(p))
        W_dom_abl = d["W_dom_abl"].astype(np.float32)
        nat_std = d["nat_std"].astype(np.float32)
        layers = list(d["layer_index"])
        with open(Path(probe_dir) / "probe_set.json") as f:
            abl_layer = json.load(f)["ablation_layer"]
        if abl_layer in layers:
            nat_std_abl = nat_std[layers.index(abl_layer)]
        else:
            nat_std_abl = d["nat_std_abl"].astype(np.float32)
        D_raw = nat_std_abl[None, :] * W_dom_abl
        G_raw = (D_raw @ D_raw.T).astype(np.float32)
        d["G_dom"] = G_raw
        d["G_dom_inv"] = np.linalg.pinv(G_raw).astype(np.float32)
        np.savez(p, **d)
        ps_bad = vcf.VerifyProbeSet(probe_dir)
        gc_bad = ps_bad.gram_consistency_check()
        assert not gc_bad["consistent"], (
            "gram_consistency_check failed to catch a raw-space Gram inverse "
            f"(the exact historical bug): {gc_bad}")
        print(f"[test] raw-space Gram corruption correctly caught "
              f"(rel dist={gc_bad['rel_frobenius_dist_G_dom_inv_vs_standardized_space']:.4g})")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_end_to_end():
    tmp_root = tempfile.mkdtemp(prefix="stage7_verify_closed_form_test_")
    try:
        tok = _get_tokenizer()
        fixture = make_fixture.build_all(tmp_root, tok.vocab_size)
        fix_gram_to_standardized_space(fixture["probe_dir"])

        scores_dir = os.path.join(tmp_root, "scores")
        os.environ["STAGE7_SHARD_DIR"] = fixture["shard_dir"]

        # --- Phase 1: produce the scored shard store by actually running
        # score_corpus.py (not reimplementing it) so conventions can't drift.
        score_args = [
            "--probe-set", fixture["probe_dir"],
            "--shards", "999",
            "--out", scores_dir,
            "--batch-size", "4",
            "--attn", "eager",
            "--calib-tokens", "200",
            "--quant-json", os.path.join(scores_dir, "quant.json"),
            "--device", "cpu",
            "--model", GEMMA_MODEL,
            "--heartbeat", os.path.join(scores_dir, "hb_score.txt"),
            "--tiny-model-config", fixture["tiny_model_config"],
        ]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sc.main(score_args)
        print(buf.getvalue())

        n_sum = int(np.load(os.path.join(scores_dir, "scores_00999.npy")).shape[0])
        assert n_sum > 0, "fixture produced zero scored tokens"

        # --- run the script under test ---
        report_path = os.path.join(tmp_root, "verify_report.json")
        verify_args = [
            "--probe-set", fixture["probe_dir"],
            "--scores", scores_dir,
            "--shard", "999",
            "--n-tokens", str(n_sum),  # exercise the whole tiny shard
            "--batch-size", "4",
            "--attn", "eager",
            "--device", "cpu",
            "--model", GEMMA_MODEL,
            "--heartbeat", os.path.join(scores_dir, "hb_verify.txt"),
            "--out", report_path,
            "--tiny-model-config", fixture["tiny_model_config"],
        ]
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            report = vcf.main(verify_args)
        print(buf2.getvalue())

        assert os.path.exists(report_path)
        with open(report_path) as f:
            report_disk = json.load(f)
        assert report_disk == report or report_disk["n_tokens"] == report["n_tokens"], \
            "report written to disk should match the returned report"

        # gram consistency should now be OK (we patched it)
        gc = report["gram_consistency_check"]
        assert gc["consistent"], f"gram_consistency_check should pass after patching: {gc}"
        print(f"[test] gram_consistency_check OK after patch (rel dist="
              f"{gc['rel_frobenius_dist_G_dom_inv_vs_standardized_space']:.4g})")

        # --- check 1: score restoration ---
        c1 = report["check1_score_restoration"]
        print(f"[test] check1 max_ratio={c1['max_ratio']:.6g} (tol {c1['tol']:.0e})")
        assert c1["pass"], f"check1 (score restoration) should PASS on the smoke fixture: {c1}"

        # --- check 2: closed-form identity (float) ---
        c2 = report["check2_closed_form_identity_float"]
        print(f"[test] check2 p50={c2['p50']:.6g} p99={c2['p99']:.6g} (tol {c2['tol']:.0e})")
        assert c2["pass"], f"check2 (closed-form identity) should PASS on the smoke fixture: {c2}"

        # --- check 3: quant path -- sane numbers, not necessarily tight on
        # tiny random data + a 200-token calibration, but must be finite and
        # roughly small (int8 floor), and the report's own gate gives a
        # PASS/FAIL signal we sanity-check is present and boolean.
        c3 = report["check3_quant_path"]
        print(f"[test] check3 p50={c3['p50']:.6g} p99={c3['p99']:.6g}")
        assert np.isfinite(c3["p50"]) and np.isfinite(c3["p99"])
        assert 0.0 <= c3["p50"] < 5.0, f"check3 p50 implausibly large: {c3}"
        assert isinstance(c3["pass"], bool)

        # --- check 4: storage audit -- finite, no NaNs, correlations in range ---
        c4 = report["check4_storage_audit"]
        dom_r = np.asarray(c4["dom_columns"]["pearson_r"])
        arm_r = np.asarray(c4["arm_columns"]["pearson_r"])
        assert np.all(np.isfinite(dom_r)), dom_r
        assert np.all(np.isfinite(arm_r)), arm_r
        assert np.all(dom_r >= -1.0001) and np.all(dom_r <= 1.0001), dom_r
        assert np.all(arm_r >= -1.0001) and np.all(arm_r <= 1.0001), arm_r
        assert c4["dom_columns"]["median_pearson_r"] > 0.5, (
            f"dom columns should correlate reasonably well between stored/live given "
            f"int8 quantization only: {c4['dom_columns']}")
        print(f"[test] check4 dom median_r={c4['dom_columns']['median_pearson_r']:.4g} "
              f"arm median_r={c4['arm_columns']['median_pearson_r']:.4g}")

        print("[test] end-to-end smoke test PASSED")
    finally:
        os.environ.pop("STAGE7_SHARD_DIR", None)
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_token_id_assertion_fires_on_drift():
    """If the stored tokens_<sid>.npy disagrees with what re-tokenizing the
    recovered text produces, verify_closed_form.py must hard-fail (never
    silently mis-align positions)."""
    tmp_root = tempfile.mkdtemp(prefix="stage7_verify_drift_test_")
    try:
        tok = _get_tokenizer()
        fixture = make_fixture.build_all(tmp_root, tok.vocab_size)
        fix_gram_to_standardized_space(fixture["probe_dir"])
        scores_dir = os.path.join(tmp_root, "scores")
        os.environ["STAGE7_SHARD_DIR"] = fixture["shard_dir"]

        score_args = [
            "--probe-set", fixture["probe_dir"], "--shards", "999", "--out", scores_dir,
            "--batch-size", "4", "--attn", "eager", "--calib-tokens", "200",
            "--quant-json", os.path.join(scores_dir, "quant.json"), "--device", "cpu",
            "--model", GEMMA_MODEL, "--heartbeat", os.path.join(scores_dir, "hb_score.txt"),
            "--tiny-model-config", fixture["tiny_model_config"],
        ]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sc.main(score_args)

        # corrupt one stored token id so re-tokenization can't reproduce it
        tokens_path = os.path.join(scores_dir, "tokens_00999.npy")
        toks = np.load(tokens_path)
        toks = toks.copy()
        toks[0] = (toks[0] + 12345) % 250000
        np.save(tokens_path, toks)

        verify_args = [
            "--probe-set", fixture["probe_dir"], "--scores", scores_dir, "--shard", "999",
            "--n-tokens", "100000", "--batch-size", "4", "--attn", "eager", "--device", "cpu",
            "--model", GEMMA_MODEL, "--heartbeat", os.path.join(scores_dir, "hb_verify.txt"),
            "--out", os.path.join(tmp_root, "verify_report.json"),
            "--tiny-model-config", fixture["tiny_model_config"],
        ]
        raised = False
        try:
            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                vcf.main(verify_args)
        except RuntimeError as e:
            raised = True
            assert "TOKEN-ID REPRODUCTION FAILED" in str(e), str(e)
        assert raised, "expected a RuntimeError on token-id drift, got none"
        print("[test] token-id drift correctly raises RuntimeError")
    finally:
        os.environ.pop("STAGE7_SHARD_DIR", None)
        shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    test_gram_consistency_check()
    test_end_to_end()
    test_token_id_assertion_fires_on_drift()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
