"""Stage 7-Oracle Phase 3 (Exp B) -- mandatory pre-training verification of
the closed-form ablation-repair target.

SPEC.md Phase 3 "Coactivation note (user concern, weakly held) + mandatory
verification": the closed form v* = D_raw . G_dom_inv . (s - t_nat_dom) is
exact BY CONSTRUCTION only if the ablation is implemented as the joint
affine projection onto span{w_dom_c} in STANDARDIZED space (DESIGN.md's
corrected G_dom = W_dom_abl @ W_dom_abl.T note). This script recomputes that
projection live against real gemma-2-2b forwards on a scored shard and
checks four things (see DESIGN.md / SPEC.md for the exact prose):

  1. SCORE RESTORATION -- after the joint ablation projection, recomputed
     dom scores must equal t_nat_dom (the "it's implemented as a joint
     affine projection" guarantee). max |s(h')-t| / std(s) < 1e-3.
  2. CLOSED-FORM IDENTITY (float) -- v* (from live, full-precision dom
     scores) must equal (h - h') to high precision; this catches bugs in
     THIS SCRIPT's / the pipeline's matrix orientations, independent of
     quantization. p50/p99 relative error < 1e-3, computed in float32.
  3. QUANT PATH -- v* recomputed from the STORED int8 dom columns (the
     thing Exp-B training actually consumes) vs the true live (h - h').
     Expected few-% error from the int8 floor; PASS gate: median < 5%.
  4. STORAGE AUDIT -- stored (int8-dequantized) scores vs live float scores,
     per-column Pearson r and RMS-err/quant-scale, for the dom columns and
     for the arm-score (layer0/1/2) columns -- validates the whole Phase-1
     store end-to-end, not just the dom slice.

Frozen math (from the task spec, matches DESIGN.md's corrected G_dom note
and train_encoder.py's ProbeSet.v_star -- agreement with train_encoder's
implementation is asserted at runtime by
VerifyProbeSet._crosscheck_train_encoder_v_star, reported as
"v_star_crosscheck_vs_train_encoder"):
    h_std   = (h - nat_mean_abl) / nat_std_abl
    s       = W_dom_abl @ h_std + b_dom_abl                      # [K]
    y       = G_dom_inv @ (s - t_nat_dom)                        # [K]
    h_std'  = h_std - W_dom_abl.T @ y                             # [D]
    h'      = nat_mean_abl + nat_std_abl * h_std'                 # [D] (raw ablated residual)
    v*      = D_raw @ y,   D_raw = (nat_std_abl[None,:] * W_dom_abl).T   # [D,K] @ [K] -> [D]

Conventions this script MUST match (ported from score_corpus.py, reused
directly by importing it rather than re-implementing -- see the module
import below): tok(text, add_special_tokens=False)["input_ids"], BOS
manually written into input_ids[:,0] right before the forward pass and its
hidden_states row dropped before anything downstream sees it (so stored
positions line up 1:1 with the raw BOS-free token ids), bf16 model dtype on
CUDA (fp32 on CPU, for the smoke test), attention implementation is
CONFIGURABLE (--attn) and should be set to whatever attn_impl actually
produced the shard being verified (score_corpus.py's own CLI default is
"sdpa" post-parity-check per SPEC.md's throughput amendment; this script
defaults to "sdpa" to match that, but the CPU smoke test passes --attn eager
to mirror test_score_corpus.py's own convention, since that is what its tiny
CPU fixture is exercised with).

ClimbMix doc-text recovery: identical to score_corpus.py's iter_shard_docs
(imported, not reimplemented) -- local STAGE7_SHARD_DIR test seam or HF hub
download, deterministic parquet order. Per the task requirement, token ids
recovered by re-tokenizing that text are asserted equal to the stored
tokens_<sid>.npy slice for EVERY doc processed (not just a prefix) -- a
convention drift here would silently invalidate every check below, so it is
a hard failure, never a warning.

Usage (pod, after Phase 1 scoring has produced /workspace/scores/scores_<sid>.npy):
  python verify_closed_form.py \
      --probe-set /workspace/stage7_oracle/out \
      --scores /workspace/scores \
      --shard 320 \
      --n-tokens 1000000 \
      --out /workspace/scores/verify_closed_form_report.json

Smoke test: test_verify_closed_form.py (tiny random Gemma2Model, tiny
synthetic probe_set with a CORRECTED G_dom -- see that file's docstring for
an important fixture caveat found while writing this script).
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
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# train_encoder.py (optional v_star cross-check below) lives in repo_root/oracles
# locally; on pods everything is staged flat, so this insert is a no-op there.
_ORACLES_DIR = HERE.parent / "oracles"
if _ORACLES_DIR.is_dir() and str(_ORACLES_DIR) not in sys.path:
    sys.path.insert(0, str(_ORACLES_DIR))

import score_corpus as sc  # noqa: E402  -- reused directly, not reimplemented

SCRIPT = "verify_closed_form"


# --------------------------------------------------------------------------
# ProbeSet extension: adds the Exp-B closed-form pieces (G_dom_inv,
# t_nat_dom, D_raw) on top of score_corpus.ProbeSet (which already loads
# everything needed to compute LIVE dom/arm scores: W, b, nat_mean, nat_std,
# W_dom, b_dom, nat_mean_abl, nat_std_abl).
# --------------------------------------------------------------------------

class VerifyProbeSet(sc.ProbeSet):
    def __init__(self, probe_set_dir: str):
        super().__init__(probe_set_dir)
        arrs = np.load(Path(probe_set_dir) / "probe_set_arrays.npz")
        if "G_dom_inv" not in arrs.files or "t_nat_dom" not in arrs.files:
            raise AssertionError(
                "probe_set_arrays.npz is missing G_dom_inv/t_nat_dom -- these are "
                "required for the Exp-B closed-form verification (DESIGN.md "
                "probe_set_arrays.npz doc); score_corpus.py's own ProbeSet does not "
                "load them since Phase-1 scoring never needs them."
            )
        self.G_dom_inv = np.asarray(arrs["G_dom_inv"], dtype=np.float32)  # [K,K]
        self.t_nat_dom = np.asarray(arrs["t_nat_dom"], dtype=np.float32)  # [K]
        assert self.G_dom_inv.shape == (self.K, self.K), self.G_dom_inv.shape
        assert self.t_nat_dom.shape == (self.K,), self.t_nat_dom.shape
        # D_raw[c] = nat_std_abl * W_dom_abl[c]  (raw-space dom direction rows,
        # DESIGN.md/SPEC.md's D_raw; train_encoder.ProbeSet's D_dom == D_raw.T).
        self.D_raw = (self.nat_std_abl[None, :] * self.W_dom).astype(np.float32)  # [K,D]
        self.v_star_crosscheck = self._crosscheck_train_encoder_v_star(probe_set_dir)

    def _crosscheck_train_encoder_v_star(self, probe_set_dir: str) -> dict:
        """The verdict of this script gates Exp-B training, and Exp-B's
        targets are computed by train_encoder.ProbeSet.v_star -- a formula
        this script re-implements independently. If the two ever diverge
        (e.g. a transposed D, a re-ordered t), this script could PASS while
        training consumes wrong targets. So: evaluate BOTH implementations
        on random dom-score vectors and hard-fail on mismatch. Import/
        instantiation failure (train_encoder.py not deployed alongside, or
        an incompatible fixture) degrades to a loud warning, never silently.
        """
        try:
            from train_encoder import ProbeSet as _TrainProbeSet
            te_ps = _TrainProbeSet(probe_set_dir)
        except Exception as e:  # noqa: BLE001 -- optional cross-check, warn loudly
            print(
                f"[{SCRIPT}] WARNING: could not instantiate train_encoder.ProbeSet "
                f"for the v_star cross-check ({type(e).__name__}: {e}). The "
                f"closed-form checks below still run, but agreement with the "
                f"formula Exp-B training actually consumes is NOT being verified.",
                file=sys.stderr,
            )
            return {"ran": False, "error": f"{type(e).__name__}: {e}"}
        rng = np.random.default_rng(0)
        s = (self.t_nat_dom[None, :]
             + rng.normal(scale=10.0, size=(64, self.K))).astype(np.float32)
        y_te, v_te = te_ps.v_star(s)
        y_here = (s - self.t_nat_dom) @ self.G_dom_inv
        v_here = y_here @ self.D_raw
        scale_y = max(float(np.abs(y_here).max()), 1e-12)
        scale_v = max(float(np.abs(v_here).max()), 1e-12)
        rel_y = float(np.abs(np.asarray(y_te) - y_here).max()) / scale_y
        rel_v = float(np.abs(np.asarray(v_te) - v_here).max()) / scale_v
        ok = rel_y < 1e-5 and rel_v < 1e-5
        if not ok:
            raise AssertionError(
                f"train_encoder.ProbeSet.v_star DISAGREES with verify_closed_form's "
                f"own v* math on the same probe_set (max rel diff y={rel_y:.3g}, "
                f"v={rel_v:.3g}). Exp-B training targets would not be the quantity "
                f"this script verifies -- fix the divergence before trusting either."
            )
        return {"ran": True, "max_rel_diff_y": rel_y, "max_rel_diff_v": rel_v, "pass": ok}

    def gram_consistency_check(self) -> dict:
        """Defensive audit, independent of the checks below: the CORRECT
        Gram (DESIGN.md, select_probes.py) is the STANDARDIZED-space Gram
        W_dom_abl @ W_dom_abl.T -- NOT the raw-space Gram of
        d_c = nat_std_abl * w_c (DESIGN.md: "Original raw-Gram spec was a
        bug, corrected ~4:15 AM before Exp B ran."). Recompute the correct
        inverse from W_dom_abl directly and compare to the G_dom_inv this
        probe_set actually ships, so a regression of that exact bug (e.g. in
        a stale fixture or a future Phase-0 rewrite) is caught loudly here
        instead of silently producing a self-consistent-looking but wrong
        report."""
        G_std = self.W_dom @ self.W_dom.T  # [K,K], standardized-space, correct
        G_std_inv = np.linalg.pinv(G_std).astype(np.float32)
        num = float(np.linalg.norm(self.G_dom_inv - G_std_inv))
        den = max(float(np.linalg.norm(G_std_inv)), 1e-12)
        rel = num / den
        ok = rel < 1e-2  # loose: pinv vs a possibly-regularized inverse can differ slightly
        if not ok:
            print(
                f"[{SCRIPT}] WARNING: probe_set_arrays.npz's G_dom_inv does NOT match "
                f"inv(W_dom_abl @ W_dom_abl.T) (the standardized-space Gram inverse, "
                f"DESIGN.md's corrected convention). relative Frobenius distance="
                f"{rel:.4g}. This usually means G_dom was built from the RAW-space Gram "
                f"(nat_std_abl*W) . (nat_std_abl*W)^T instead -- the exact bug DESIGN.md "
                f"documents as fixed in select_probes.py ~4:15 AM. Checks 1/2 below will "
                f"likely FAIL as a result; the bug (if this fires) is in the probe_set "
                f"data, not in this script.",
                file=sys.stderr,
            )
        return {"rel_frobenius_dist_G_dom_inv_vs_standardized_space": rel, "consistent": ok}


# --------------------------------------------------------------------------
# Per-device tensor cache (mirrors score_corpus.ScoreHead's pattern)
# --------------------------------------------------------------------------

class ProbeTensors:
    def __init__(self, ps: VerifyProbeSet, device: torch.device):
        self.K = ps.K
        self.nat_mean_abl = torch.from_numpy(ps.nat_mean_abl).to(device=device, dtype=torch.float32)
        self.nat_std_abl = torch.from_numpy(ps.nat_std_abl).to(device=device, dtype=torch.float32)
        self.W_dom = torch.from_numpy(ps.W_dom).to(device=device, dtype=torch.float32)      # [K,D]
        self.b_dom = torch.from_numpy(ps.b_dom).to(device=device, dtype=torch.float32)      # [K]
        self.t_nat_dom = torch.from_numpy(ps.t_nat_dom).to(device=device, dtype=torch.float32)  # [K]
        self.G_dom_inv = torch.from_numpy(ps.G_dom_inv).to(device=device, dtype=torch.float32)  # [K,K]
        self.D_raw = torch.from_numpy(ps.D_raw).to(device=device, dtype=torch.float32)      # [K,D]


@torch.no_grad()
def compute_doc_quantities(h: torch.Tensor, s_all: torch.Tensor, pt: ProbeTensors) -> dict:
    """h: [n,D] float32 raw ablation-layer residual (BOS row already dropped).
    s_all: [n,4K] float32 -- exactly score_corpus.ScoreHead's output for this
    slice (so s_all[:, 3K:4K] is the live dom score, matching production's
    column layout 1:1). All computation here is float32 (task requirement:
    "fp16-tolerant: compute in float32")."""
    K = pt.K
    s_dom_live = s_all[:, 3 * K:4 * K]                                # [n,K]
    h_std = (h - pt.nat_mean_abl) / pt.nat_std_abl                    # [n,D]
    y = (s_dom_live - pt.t_nat_dom) @ pt.G_dom_inv                    # [n,K]  (G_dom_inv symmetric)
    h_std_p = h_std - y @ pt.W_dom                                    # [n,D]  U@y, U = W_dom_abl.T
    h_p = pt.nat_mean_abl + pt.nat_std_abl * h_std_p                  # [n,D]  raw ablated residual h'
    v_star_live = y @ pt.D_raw                                        # [n,D]
    diff_live = h - h_p                                               # [n,D]  actual (clean - ablated)

    # check 1: recompute dom scores ON h' (a genuine re-derivation, not an
    # algebraic simplification -- this is the "implemented as joint affine
    # projection" guarantee the SPEC calls for).
    h_std_p2 = (h_p - pt.nat_mean_abl) / pt.nat_std_abl
    s_prime = h_std_p2 @ pt.W_dom.T + pt.b_dom                        # [n,K]
    check1_abs_diff = (s_prime - pt.t_nat_dom).abs()                  # [n,K]

    # check 2: closed-form identity in float.
    num2 = torch.linalg.vector_norm(v_star_live - diff_live, dim=-1)
    den2 = torch.linalg.vector_norm(diff_live, dim=-1).clamp_min(1e-12)
    relerr2 = num2 / den2                                             # [n]

    return {
        "s_dom_live": s_dom_live,
        "check1_abs_diff": check1_abs_diff,
        "relerr2": relerr2,
        "diff_live": diff_live,
    }


# --------------------------------------------------------------------------
# Online accumulators (memory bounded regardless of --n-tokens)
# --------------------------------------------------------------------------

class PearsonAcc:
    """Streaming per-column Pearson r + RMS error between two [n,n_cols]
    streams, accumulated from running sums (Chan-style single pass; fine at
    float64 for the token counts this script handles)."""

    def __init__(self, n_cols: int):
        self.n = 0
        self.sx = np.zeros(n_cols, dtype=np.float64)
        self.sy = np.zeros(n_cols, dtype=np.float64)
        self.sxx = np.zeros(n_cols, dtype=np.float64)
        self.syy = np.zeros(n_cols, dtype=np.float64)
        self.sxy = np.zeros(n_cols, dtype=np.float64)
        self.sse = np.zeros(n_cols, dtype=np.float64)

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.shape[0] == 0:
            return
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        self.n += x.shape[0]
        self.sx += x.sum(axis=0)
        self.sy += y.sum(axis=0)
        self.sxx += (x * x).sum(axis=0)
        self.syy += (y * y).sum(axis=0)
        self.sxy += (x * y).sum(axis=0)
        self.sse += ((x - y) ** 2).sum(axis=0)

    def pearson_r(self) -> np.ndarray:
        n = max(self.n, 1)
        mx, my = self.sx / n, self.sy / n
        cov = self.sxy / n - mx * my
        varx = np.maximum(self.sxx / n - mx ** 2, 1e-20)
        vary = np.maximum(self.syy / n - my ** 2, 1e-20)
        return cov / np.sqrt(varx * vary)

    def rmse(self) -> np.ndarray:
        return np.sqrt(self.sse / max(self.n, 1))


class RunningMax:
    def __init__(self, n_cols: int):
        self.val = np.full(n_cols, -np.inf, dtype=np.float64)

    def update(self, batch_max: np.ndarray) -> None:
        self.val = np.maximum(self.val, batch_max)


# --------------------------------------------------------------------------
# Forward pass (mirrors score_corpus.forward_score_batch, additionally
# returns the ablation-layer raw hidden state)
# --------------------------------------------------------------------------

@torch.no_grad()
def forward_verify_batch(model, head: "sc.ScoreHead", sub, bos_id, pad_id, device):
    """sub: list of (doc_idx, ids). Returns (h_abl [B,Tmax,D] float32,
    raw_scores [B,Tmax,4K] float32, lens)."""
    input_ids, attn, lens = sc.make_padded_batch(sub, bos_id, pad_id, device)
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
    hs_by_layer = {}
    for l in head.probe.needed_hidden_layers():
        hs_by_layer[l] = out.hidden_states[l + 1][:, 1:, :]
    del out
    raw_scores = head.score(hs_by_layer)  # [B,T,4K] float32
    h_abl = hs_by_layer[head.probe.ablation_layer].to(torch.float32)
    del hs_by_layer
    return h_abl, raw_scores, lens


# --------------------------------------------------------------------------
# Main verification loop
# --------------------------------------------------------------------------

def load_docs_jsonl(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["doc"]] = (d["start"], d["n"])
    return out


def run_verify(args) -> dict:
    device = torch.device(args.device)
    ps = VerifyProbeSet(args.probe_set)
    gram_check = ps.gram_consistency_check()

    scores_dir = Path(args.scores)
    quant_path = args.quant_json or str(scores_dir / "quant.json")
    quant = sc.load_quant(quant_path)  # dict: zero [4K], scale [4K] (float32 arrays)
    K = ps.K
    dom_zero, dom_scale = quant["zero"][3 * K:4 * K], quant["scale"][3 * K:4 * K]
    arm_zero, arm_scale = quant["zero"][:3 * K], quant["scale"][:3 * K]

    tok, model = sc.load_model_and_tok(args.model, args.attn, device, args.tiny_model_config)
    head = sc.ScoreHead(ps, device)
    pt = ProbeTensors(ps, device)
    hb = sc.Heartbeat(args.heartbeat)

    sid = args.shard
    paths = sc.shard_output_paths(scores_dir, sid)
    for p in (paths["tokens"], paths["scores"], paths["docs"]):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found -- run score_corpus.py on shard {sid} first")
    tokens_mmap = np.load(paths["tokens"], mmap_mode="r")
    scores_mmap = np.load(paths["scores"], mmap_mode="r")
    docs_by_idx = load_docs_jsonl(paths["docs"])

    # Accumulators
    check1_max = RunningMax(K)
    check1_std_acc = sc.RunningStats(K)  # std of live s_dom over the sample
    relerr2_list: list[np.ndarray] = []
    relerr3_list: list[np.ndarray] = []
    dom_acc = PearsonAcc(K)
    arm_acc = PearsonAcc(3 * K)

    n_tokens_target = args.n_tokens
    n_tokens_done = 0
    n_docs_done = 0
    n_docs_asserted = 0
    t0 = time.time()

    buf: list[tuple[int, list[int]]] = []
    doc_meta: dict[int, tuple[int, int]] = {}  # doc_idx -> (start, n), only for buffered docs

    pbar = tqdm(total=n_tokens_target, desc="verify_closed_form", unit="tok")

    def flush():
        nonlocal n_tokens_done, n_docs_done
        if not buf:
            return
        buf.sort(key=lambda kv: len(kv[1]))
        for sub in sc.chunked(buf, args.batch_size):
            h_abl, raw_scores, lens = forward_verify_batch(
                model, head, sub, tok.bos_token_id, tok.pad_token_id or 0, device)
            for i, (doc_idx, ids) in enumerate(sub):
                n = lens[i]
                start, n_stored = doc_meta[doc_idx]
                assert n == n_stored, (
                    f"doc {doc_idx}: recomputed length {n} != stored docs.jsonl n {n_stored} "
                    f"(should be impossible after the token-id assertion above)")
                h = h_abl[i, :n, :]                      # [n,D] float32
                s_all = raw_scores[i, :n, :]              # [n,4K] float32

                q = compute_doc_quantities(h, s_all, pt)
                s_dom_live_np = q["s_dom_live"].cpu().numpy()
                check1_abs_diff_np = q["check1_abs_diff"].cpu().numpy()
                relerr2_np = q["relerr2"].cpu().numpy()
                diff_live_np = q["diff_live"].cpu().numpy()

                check1_max.update(check1_abs_diff_np.max(axis=0))
                check1_std_acc.update_batch(s_dom_live_np)
                relerr2_list.append(relerr2_np.astype(np.float32))

                # --- stored (quantized) path ---
                stored_i8 = np.asarray(scores_mmap[start:start + n, :])  # [n,4K] int8
                s_dom_stored = sc.dequantize(stored_i8[:, 3 * K:4 * K], dom_zero, dom_scale)
                s_arm_stored = sc.dequantize(stored_i8[:, :3 * K], arm_zero, arm_scale)
                s_arm_live_np = s_all[:, :3 * K].cpu().numpy()

                y_q = (s_dom_stored - ps.t_nat_dom) @ ps.G_dom_inv     # [n,K]
                v_star_q = y_q @ ps.D_raw                              # [n,D]
                num3 = np.linalg.norm(v_star_q - diff_live_np, axis=-1)
                den3 = np.maximum(np.linalg.norm(diff_live_np, axis=-1), 1e-12)
                relerr3_list.append((num3 / den3).astype(np.float32))

                dom_acc.update(s_dom_stored, s_dom_live_np)
                arm_acc.update(s_arm_stored, s_arm_live_np)

                n_tokens_done += n
                n_docs_done += 1
            pbar.update(sum(lens))
            hb.maybe_write({
                "shard": sid, "n_tokens_done": n_tokens_done, "n_docs_done": n_docs_done,
                "tok_per_s": n_tokens_done / max(time.time() - t0, 1e-9),
            })
        buf.clear()
        doc_meta.clear()

    for doc_idx, text in sc.iter_shard_docs(sid, args.max_docs):
        if n_tokens_done >= n_tokens_target:
            break
        if doc_idx not in docs_by_idx:
            continue  # score_corpus.py filtered this doc (too short)
        start, n_stored = docs_by_idx[doc_idx]

        ids = sc.tokenize_docs(tok, [text])[0]  # add_special_tokens=False, sliced to MAX_DOC_TOKENS
        stored_ids = np.asarray(tokens_mmap[start:start + n_stored], dtype=np.int32)
        if n_docs_asserted < args.assert_all_docs_cap:
            if len(ids) != n_stored or not np.array_equal(np.asarray(ids, dtype=np.int32), stored_ids):
                n_show = min(len(ids), len(stored_ids), 20)
                raise RuntimeError(
                    "TOKEN-ID REPRODUCTION FAILED (convention drift) for "
                    f"shard={sid} doc={doc_idx}: re-tokenizing the recovered raw text "
                    f"(add_special_tokens=False, sliced to score_corpus.MAX_DOC_TOKENS) does "
                    f"NOT reproduce the stored tokens_{sid:05d}.npy slice.\n"
                    f"  stored (n={len(stored_ids)}): {stored_ids[:n_show].tolist()}...\n"
                    f"  retok  (n={len(ids)}):  {list(ids[:n_show])}...\n"
                    f"This means the tokenization convention has drifted from score_corpus.py's "
                    f"-- every check in this report would be comparing mismatched positions. "
                    f"Fix the convention before trusting any result here."
                )
            n_docs_asserted += 1

        buf.append((doc_idx, ids))
        doc_meta[doc_idx] = (start, n_stored)
        if len(buf) >= sc.DOC_BUFFER:
            flush()
    flush()
    pbar.close()
    hb.maybe_write({"shard": sid, "n_tokens_done": n_tokens_done, "n_docs_done": n_docs_done,
                     "tok_per_s": n_tokens_done / max(time.time() - t0, 1e-9)}, force=True)

    if n_tokens_done == 0:
        raise RuntimeError(f"shard {sid}: no tokens processed (docs_by_idx empty or all skipped?)")
    if n_tokens_done < n_tokens_target:
        print(f"[{SCRIPT}] WARNING: shard {sid} exhausted at {n_tokens_done} tokens "
              f"(< requested --n-tokens {n_tokens_target}); reporting on what was available.",
              file=sys.stderr)

    # ---- assemble report ----
    std_s = np.maximum(check1_std_acc.std(), 1e-12)
    check1_ratio = check1_max.val / std_s
    relerr2 = np.concatenate(relerr2_list) if relerr2_list else np.zeros(0, dtype=np.float32)
    relerr3 = np.concatenate(relerr3_list) if relerr3_list else np.zeros(0, dtype=np.float32)

    def pct(a, q):
        return float(np.percentile(a, q)) if a.size else float("nan")

    check1 = {
        "per_concept_max_abs_diff": check1_max.val.tolist(),
        "per_concept_std_s": std_s.tolist(),
        "per_concept_ratio": check1_ratio.tolist(),
        "max_ratio": float(np.max(check1_ratio)),
        "tol": args.score_restore_tol,
        "pass": bool(np.max(check1_ratio) < args.score_restore_tol),
    }
    check2 = {
        "p50": pct(relerr2, 50), "p99": pct(relerr2, 99),
        "tol": args.identity_tol,
        "pass": bool(pct(relerr2, 99) < args.identity_tol),
    }
    check3 = {
        "p50": pct(relerr3, 50), "p99": pct(relerr3, 99),
        "tol_median": args.quant_median_tol,
        "pass": bool(pct(relerr3, 50) < args.quant_median_tol),
    }
    dom_r = dom_acc.pearson_r()
    dom_rms_over_scale = dom_acc.rmse() / np.maximum(dom_scale, 1e-12)
    arm_r = arm_acc.pearson_r()
    arm_rms_over_scale = arm_acc.rmse() / np.maximum(arm_scale, 1e-12)
    check4 = {
        "dom_columns": {
            # DOM block was always correctly name-sorted; label explicitly.
            "concepts": ps.dom_block_concepts,
            "pearson_r": dom_r.tolist(),
            "rms_err_over_scale": dom_rms_over_scale.tolist(),
            "median_pearson_r": float(np.median(dom_r)),
            "median_rms_over_scale": float(np.median(dom_rms_over_scale)),
        },
        "arm_columns": {
            # MAIN-block arm columns follow main_block_concepts (store/W order),
            # NOT `concepts` -- see out/PERMUTATION_FIX.md. The per-column r
            # values are self-consistent (stored vs live both use W's order);
            # only the NAME attached needs the correct block order.
            "names": [f"L{l}:{c}" for l in ps.layers for c in ps.main_block_concepts],
            "pearson_r": arm_r.tolist(),
            "rms_err_over_scale": arm_rms_over_scale.tolist(),
            "median_pearson_r": float(np.median(arm_r)),
            "median_rms_over_scale": float(np.median(arm_rms_over_scale)),
        },
    }

    report = {
        "shard": sid,
        "n_tokens": n_tokens_done,
        "n_docs": n_docs_done,
        "ablation_layer": ps.ablation_layer,
        "K": K,
        "gram_consistency_check": gram_check,
        "v_star_crosscheck_vs_train_encoder": ps.v_star_crosscheck,
        "check1_score_restoration": check1,
        "check2_closed_form_identity_float": check2,
        "check3_quant_path": check3,
        "check4_storage_audit": check4,
        "overall_pass": bool(check1["pass"] and check2["pass"] and check3["pass"]),
        "config": {
            "probe_set": args.probe_set, "scores": args.scores, "shard": sid,
            "n_tokens_target": n_tokens_target, "batch_size": args.batch_size,
            "attn": args.attn, "model": args.model, "device": args.device,
        },
    }
    return report


def print_summary(report: dict) -> None:
    c1, c2, c3 = (report["check1_score_restoration"], report["check2_closed_form_identity_float"],
                  report["check3_quant_path"])
    c4 = report["check4_storage_audit"]
    print(f"[{SCRIPT}] shard={report['shard']} n_tokens={report['n_tokens']} "
          f"n_docs={report['n_docs']} ablation_layer={report['ablation_layer']} K={report['K']}")
    gc = report["gram_consistency_check"]
    print(f"[{SCRIPT}] Gram consistency (G_dom_inv vs standardized-space, expect ~0): "
          f"{gc['rel_frobenius_dist_G_dom_inv_vs_standardized_space']:.4g} "
          f"({'OK' if gc['consistent'] else 'MISMATCH -- see warning above'})")
    xc = report["v_star_crosscheck_vs_train_encoder"]
    if xc.get("ran"):
        print(f"[{SCRIPT}] v* cross-check vs train_encoder.ProbeSet.v_star: "
              f"max rel diff y={xc['max_rel_diff_y']:.3g} v={xc['max_rel_diff_v']:.3g} -> OK")
    else:
        print(f"[{SCRIPT}] v* cross-check vs train_encoder: SKIPPED ({xc.get('error')})")
    print(f"[{SCRIPT}] CHECK 1 score restoration: max|s(h')-t|/std(s) = {c1['max_ratio']:.4g} "
          f"(tol {c1['tol']:.0e}) -> {'PASS' if c1['pass'] else 'FAIL'}")
    print(f"[{SCRIPT}] CHECK 2 closed-form identity (float): p50={c2['p50']:.4g} p99={c2['p99']:.4g} "
          f"(tol {c2['tol']:.0e}) -> {'PASS' if c2['pass'] else 'FAIL'}")
    print(f"[{SCRIPT}] CHECK 3 quant path: p50={c3['p50']:.4g} p99={c3['p99']:.4g} "
          f"(median tol {c3['tol_median']:.0%}) -> {'PASS' if c3['pass'] else 'FAIL'}")
    print(f"[{SCRIPT}] CHECK 4 storage audit: dom median r={c4['dom_columns']['median_pearson_r']:.4g} "
          f"rms/scale={c4['dom_columns']['median_rms_over_scale']:.4g}; "
          f"arm median r={c4['arm_columns']['median_pearson_r']:.4g} "
          f"rms/scale={c4['arm_columns']['median_rms_over_scale']:.4g}")
    print(f"[{SCRIPT}] OVERALL: {'PASS' if report['overall_pass'] else 'FAIL'}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-set", required=True, help="dir with probe_set.json + probe_set_arrays.npz")
    ap.add_argument("--scores", required=True, help="dir with tokens_<sid>.npy/scores_<sid>.npy/docs_<sid>.jsonl/quant.json")
    ap.add_argument("--shard", type=int, required=True, help="single scored shard id to verify")
    ap.add_argument("--n-tokens", type=int, default=1_000_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--quant-json", default=None, help="default: <scores>/quant.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-docs", type=int, default=None, help="safety cap on docs scanned from the shard")
    ap.add_argument("--assert-all-docs-cap", type=int, default=10 ** 9,
                     help="assert recovered token ids == stored ids for up to this many docs "
                          "(default effectively unlimited -- every processed doc is checked)")
    ap.add_argument("--model", default=sc.GEMMA_MODEL_DEFAULT)
    ap.add_argument("--heartbeat", default="/workspace/hb_verify.txt")
    ap.add_argument("--score-restore-tol", type=float, default=1e-3)
    ap.add_argument("--identity-tol", type=float, default=1e-3)
    ap.add_argument("--quant-median-tol", type=float, default=0.05)
    ap.add_argument("--out", required=True, help="JSON report output path")
    ap.add_argument("--tiny-model-config", default=None, help=argparse.SUPPRESS)  # test seam
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    report = run_verify(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, out_path)
    print_summary(report)
    print(f"[{SCRIPT}] report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
