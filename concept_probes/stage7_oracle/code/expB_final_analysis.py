#!/usr/bin/env python3
"""Stage-7 Exp-B final analysis bundle (G3).

Computes, on the FINAL best checkpoints (expB_learn/best.pt, expB_fixed/best.pt),
over val shards 353/354 (~5M-token sample):
  1. Final v* R2 (aggregate + per-dim median) for both variants.
  2. Principal angles between span(learned down cols)[2304x54] and span(true
     D_raw=(nat_std(+)W_dom).T)[2304x54], + matched-random control.
  3. cos(v_hat, v*) per token: mean/median/p10/p90 both variants, overall and
     top-decile-of-||v*|| slice; + magnitude ratio ||v_hat||/||v*|| same slices.
  4. Residual-level estimate cos(h_abl+v_hat, h_clean) ~ 1/sqrt(1+E||e||^2/E||h||^2).

Shares ONE frozen encoder (expA_prod) forward pass across both heads.
"""
import json
import os
import sys
import numpy as np
import torch

STAGE7 = "/workspace/stage7"
sys.path.insert(0, STAGE7)

import train_encoder as te  # noqa: E402

SCORES = "/workspace/scores"
CLIMBMIX = "/workspace/climbmix"
PROBESET = "/workspace/stage7"
VAL_SHARDS = [353, 354]
EVAL_TOKENS = 5_000_000
BSZ = 64
MAXG, MAXQ, MING = 2048, 3072, 64
CK_LEARN = "/workspace/expB_learn/best.pt"
CK_FIXED = "/workspace/expB_fixed/best.pt"
OUT = "/workspace/expB_final_analysis.json"
DEV = "cuda"


def princ_angles(A, B):
    """cos of principal angles between colspace(A) and colspace(B). A,B [n,k]."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return s


def summ_cos(s):
    return {"median": float(np.median(s)), "min": float(np.min(s)),
            "max": float(np.max(s)), "count_ge_0.7": int(np.sum(s >= 0.7)),
            "n": int(s.shape[0])}


def slice_stats(x):
    return {"mean": float(np.mean(x)), "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "n": int(x.shape[0])}


def main():
    ps = te.ProbeSet(PROBESET)
    K = ps.K
    zero, scale = te.load_quant(SCORES)
    D_dom = ps.D_dom.astype(np.float32)              # [2304, K]
    D_dom_t = torch.tensor(D_dom, device=DEV, dtype=torch.float32)

    # ---- E||h||^2 at ablation layer (layer 8) from nat_mean/nat_std ----
    abl_idx = int(np.where(ps.layer_index == ps.ablation_layer)[0][0])
    nat_mean_abl = ps.nat_mean[abl_idx].astype(np.float64)
    nat_std_abl = ps.nat_std[abl_idx].astype(np.float64)
    E_h2 = float(np.sum(nat_mean_abl ** 2 + nat_std_abl ** 2))

    # ---- load shared frozen encoder (expA_prod) ----
    dtype = torch.bfloat16
    model, qwen_tok, model_name = te.load_encoder("Qwen/Qwen3-0.6B-Base", dtype, DEV)
    ck_ref = torch.load(CK_LEARN, map_location=DEV, weights_only=False)
    enc_from = ck_ref["args"]["encoder_from"]
    enc_ck = torch.load(enc_from, map_location=DEV, weights_only=False)
    model.load_state_dict(enc_ck["encoder_state"], strict=True)
    model.to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    del enc_ck
    torch.cuda.empty_cache()
    hidden_size = model.config.hidden_size
    gemma_tok = te.load_gemma_tokenizer("google/gemma-2-2b")

    # ---- heads ----
    head_learn = te.EncoderHead(hidden_size, K, "expB-learn").to(DEV).to(dtype)
    head_learn.load_state_dict(torch.load(CK_LEARN, map_location=DEV, weights_only=False)["head_state"])
    head_learn.eval()
    ck_fixed = torch.load(CK_FIXED, map_location=DEV, weights_only=False)
    head_fixed = te.EncoderHead(hidden_size, K, "expB-fixed").to(DEV).to(dtype)
    head_fixed.load_state_dict(ck_fixed["head_state"])
    head_fixed.eval()
    step_learn = torch.load(CK_LEARN, map_location=DEV, weights_only=False)["step"]

    # ---- part 2: principal angles (needs only weights) ----
    down_W = head_learn.down.weight.detach().float().cpu().numpy()  # [2304, K]
    s_learn = princ_angles(down_W, D_dom)
    rng = np.random.default_rng(0)
    rand_summ = []
    s_rand_rep = None
    for i in range(5):
        R = rng.standard_normal((D_dom.shape[0], K)).astype(np.float32)
        sr = princ_angles(R, D_dom)
        if i == 0:
            s_rand_rep = sr
        rand_summ.append(summ_cos(sr))
    rand_avg = {k: float(np.mean([d[k] for d in rand_summ])) for k in ("median", "min", "max", "count_ge_0.7")}

    # ---- streaming eval: R2 + per-token scalars ----
    accL = te.R2Accumulator(te.D_MODEL_GEMMA)
    accF = te.R2Accumulator(te.D_MODEL_GEMMA)
    nvs, cosL, cosF, rL, rF, e2L, e2F = ([] for _ in range(7))

    buf = []
    n_tok = 0
    stream = te.iter_docs(VAL_SHARDS, SCORES, CLIMBMIX, loop=False)

    @torch.no_grad()
    def flush(buf):
        input_ids, attn_mask = te.collate_batch(buf, qwen_tok, DEV)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        feats, scores_i8 = te.gather_targets(buf, None, hidden)
        raw = te.dequantize(scores_i8, zero, scale)
        s_dom = raw[:, 3 * K:4 * K]
        _, v_true = ps.v_star(s_dom)                       # [N,2304] np f32
        _, vL = head_learn(feats)                          # [N,2304] bf16
        yF, _ = head_fixed(feats)
        vL = vL.float()
        vF = (yF.float() @ D_dom_t.T)                      # [N,2304]
        vtrue_t = torch.tensor(v_true, device=DEV, dtype=torch.float32)
        # R2
        accL.update(v_true, vL.cpu().numpy())
        accF.update(v_true, vF.cpu().numpy())
        # per-token scalars
        nvt = torch.linalg.norm(vtrue_t, dim=1)            # ||v*||
        for vh, cosl, rl, e2l in ((vL, cosL, rL, e2L), (vF, cosF, rF, e2F)):
            nvh = torch.linalg.norm(vh, dim=1)
            dot = (vh * vtrue_t).sum(dim=1)
            cos = dot / (nvh * nvt + 1e-12)
            cosl.append(cos.cpu().numpy())
            rl.append((nvh / (nvt + 1e-12)).cpu().numpy())
            e2l.append(((vh - vtrue_t) ** 2).sum(dim=1).cpu().numpy())
        nvs.append(nvt.cpu().numpy())
        return scores_i8.shape[0]

    for doc in stream:
        if n_tok >= EVAL_TOKENS:
            break
        pd = te.process_doc(doc, gemma_tok, qwen_tok, MAXG, MAXQ, MING, assert_tokens=False)
        if pd is None:
            continue
        buf.append(pd)
        if len(buf) >= BSZ:
            n_tok += flush(buf)
            buf = []
    if buf and n_tok < EVAL_TOKENS:
        n_tok += flush(buf)

    nvs = np.concatenate(nvs)
    top_thr = np.percentile(nvs, 90)
    top = nvs >= top_thr

    def variant_block(acc, cosl, rl, e2l):
        cos = np.concatenate(cosl)
        r = np.concatenate(rl)
        e2 = np.concatenate(e2l)
        E_e2 = float(np.mean(e2))
        cos_resid = 1.0 / np.sqrt(1.0 + E_e2 / E_h2)
        return {
            "v_star_r2_aggregate": acc.r2_overall(),
            "v_star_per_dim_r2_median": float(np.median(acc.r2())),
            "cos_vhat_vstar": {"overall": slice_stats(cos), "top_decile_vstar_norm": slice_stats(cos[top])},
            "mag_ratio_vhat_over_vstar": {"overall": slice_stats(r), "top_decile_vstar_norm": slice_stats(r[top])},
            "E_err2": E_e2,
            "residual_cos_estimate": float(cos_resid),
        }

    out = {
        "meta": {
            "n_tokens": int(n_tok), "val_shards": VAL_SHARDS, "K": K,
            "ablation_layer": ps.ablation_layer,
            "expB_learn_step": int(step_learn),
            "E_h2_layer8": E_h2,
            "top_decile_vstar_norm_threshold": float(top_thr),
            "residual_formula": "cos(h_abl+vhat,h_clean) ~ 1/sqrt(1+E||vhat-v*||^2/E||h||^2), assuming h_clean=h_abl+v* (repair by construction) and error e=vhat-v* uncorrelated with h_clean; E||h||^2=sum_d(nat_mean^2+nat_std^2) at layer 8.",
        },
        "subspace_recovery": {
            "learned_vs_Draw": summ_cos(s_learn),
            "random_control_vs_Draw_avg5": rand_avg,
            "random_control_rep": summ_cos(s_rand_rep),
            "note": "principal-angle cosines between span(down cols) and span(D_raw=(nat_std(+)W_dom).T); random control = 54 gaussian 2304-d vectors.",
        },
        "expB_learn": variant_block(accL, cosL, rL, e2L),
        "expB_fixed": variant_block(accF, cosF, rF, e2F),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
