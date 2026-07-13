#!/usr/bin/env python3
"""Oracle-prediction × probe-activation association for the PER-LAYER oracles.

Three panels (L6, L8, L14) — one per corrected-design oracle (Qwen3-0.6B full
FT + MLP head 1024->4096->54, each trained on exactly ONE layer's 54 probe
scores). Each panel is the 54×54 Spearman matrix between the oracle's
per-example max-pooled prediction for concept i (rows) and the gemma probe's
max-pooled activation for concept j (cols) at the SAME layer, on the Stage-6
natural-eval TEST split — computed exactly like fig_oracle_probe_assoc.py
(cell (i,j) on family(j)'s eval texts; probe reference uses the selection arm
from probe_set.json).

Usage:
  python3 fig_oracle_perlayer_assoc.py --ckpt-dir <dir with layer06/best_stripped.pt,
      layer08/..., layer14/...> --probe-set ../attribution/out \
      --eval-data ../concept_probes/3_validation/data \
      --out out/figures/oracle_perlayer_assoc.png    # run from oracles/
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_encoder as te
from g2_retention import class_index, load_family_truth, test_split_ids
from train_oracle_perlayer import LAYER_TO_IDX, OracleMLPHead  # noqa: F401

QWEN_HUB = "Qwen/Qwen3-0.6B-Base"
LAYERS = [6, 8, 14]


def load_oracle(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ck["mode"] == "perlayer-expA" and ck["K"] == 54, ck.get("mode")
    name = ck["model_name"] if not str(ck["model_name"]).startswith("/") else QWEN_HUB
    model, qwen_tok, _ = te.load_encoder(name, torch.float32, device)
    model.load_state_dict({k: v.float() for k, v in ck["encoder_state"].items()})
    head = OracleMLPHead(ck["hidden_size"], ck.get("head_up", 4096), ck["K"]).to(device)
    head.load_state_dict({k: v.float() for k, v in ck["head_state"].items()})
    model.eval()
    head.eval()
    for p in list(model.parameters()) + list(head.parameters()):
        p.requires_grad_(False)
    return model, head, qwen_tok, ck


@torch.no_grad()
def encode_texts(texts, model, head, qwen_tok, device, max_tokens=1024, bsz=8):
    """Per-example max-pooled 54-dim oracle predictions (checkpoint concept
    order). Same text handling as fig_oracle_probe_assoc.encode_family."""
    n = len(texts)
    out = np.zeros((n, 54), np.float32)
    order = np.argsort([len(t) for t in texts])
    for s in tqdm(range(0, n, bsz), desc="encode", leave=False):
        idxs = order[s:s + bsz]
        batch = [texts[i] for i in idxs]
        enc = qwen_tok(batch, add_special_tokens=True, truncation=True,
                       max_length=max_tokens, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        mask = attn.bool()
        h = model(input_ids=ids, attention_mask=attn).last_hidden_state
        y = head(h)                                     # [B, T, 54]
        for bi, i in enumerate(idxs):
            out[i] = y[bi][mask[bi]].max(0).values.float().cpu().numpy()
    return out


def probe_pmax(z, ci, layer, arm, test_ex_ids, n_ex):
    layers = [int(x) for x in z["layers"]]
    li = layers.index(layer)
    p = z[f"preds_{arm}"][li, :, ci].astype(np.float64)
    pmax = np.full(n_ex, -np.inf)
    np.maximum.at(pmax, z["token2ex"], p)
    return pmax[test_ex_ids]


def spearman_matrix(A, B):
    def rz(X):
        R = np.apply_along_axis(rankdata, 0, X)
        R -= R.mean(0, keepdims=True)
        sd = R.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        return R / sd
    n = A.shape[0]
    return (rz(A).T @ rz(B)) / n


def plot_panels(panels, disp, fam_of, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = len(disp)
    fam_breaks = [i for i in range(1, K) if fam_of[i] != fam_of[i - 1]]
    labels = [f"{fam_of[i]}.{c}" if (i == 0 or fam_of[i] != fam_of[i - 1]) else c
              for i, c in enumerate(disp)]

    fig, axes = plt.subplots(1, 4, figsize=(34, 9.5),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.06]})
    im = None
    for (title, M), ax in zip(panels, axes[:3]):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        diag = float(np.nanmedian(np.diag(M)))
        off = float(np.nanmedian(M[~np.eye(K, dtype=bool)]))
        ax.set_title(f"{title}\nmedian diag {diag:.3f} | off-diag {off:.3f}", fontsize=12)
        for b in fam_breaks:
            ax.axhline(b - .5, color="gray", lw=0.4)
            ax.axvline(b - .5, color="gray", lw=0.4)
        ax.set_xticks(range(K))
        ax.set_xticklabels(labels, rotation=90, fontsize=4.2)
        ax.set_yticks(range(K))
        ax.set_yticklabels(labels, fontsize=4.2)
        ax.set_xlabel("gemma probe activation (max-pool, test split)", fontsize=9)
        ax.set_ylabel("oracle prediction (max-pool)", fontsize=9)
    cb = fig.colorbar(im, cax=axes[3])
    cb.set_label("Spearman corr (oracle prediction vs probe activation)")
    fig.suptitle("per-layer oracle prediction × probe activation — Spearman association, "
                 "Stage-6 natural eval TEST split (rows=oracle concept, cols=probe)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    print(f"wrote {out_path}")


def _print_stats(panels, K):
    for t, M in panels:
        d = np.diag(M)
        print(f"{t:28s} diag median {np.nanmedian(d):.4f} "
              f"min {np.nanmin(d):.4f} | off-diag median "
              f"{np.nanmedian(M[~np.eye(K, dtype=bool)]):.4f} "
              f"p95 |off| {np.nanpercentile(np.abs(M[~np.eye(K, dtype=bool)]), 95):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--probe-set", required=True)
    ap.add_argument("--eval-data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--replot-from")
    args = ap.parse_args()

    ps = te.ProbeSet(args.probe_set)
    K = ps.K
    fams = ps.families
    disp = sorted(ps.concepts, key=lambda c: (fams[c], c))
    fam_of = [fams[c] for c in disp]

    if args.replot_from:
        z = np.load(args.replot_from, allow_pickle=True)
        panels = [(f"per-layer oracle @ L{L}", z[f"L{L}"]) for L in LAYERS]
        plot_panels(panels, disp, fam_of, args.out)
        _print_stats(panels, K)
        return
    if not (args.ckpt_dir and args.eval_data):
        ap.error("--ckpt-dir and --eval-data required unless --replot-from")

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")
    selection = ps.meta.get("selection", {})

    fam_concepts = {}
    for c in disp:
        fam_concepts.setdefault(fams[c], []).append(c)

    # ---- probe truth (load each family's natscores once; all 3 layers) ----
    fam_texts, fam_probes = {}, {}
    for fam in sorted(fam_concepts):
        rows, z = load_family_truth(args.eval_data, fam)
        if z is None:
            raise RuntimeError(f"{fam}: natscores npz required")
        tid = test_split_ids(rows)
        fam_texts[fam] = [rows[i]["text"] for i in tid]
        probes = {}
        for c in fam_concepts[fam]:
            ci = class_index(z, c)
            for L in LAYERS:
                arm = selection.get(str(L), {}).get(c, {}).get("arm")
                if arm is not None:
                    probes[(L, c)] = probe_pmax(z, ci, L, arm, tid, len(rows))
        fam_probes[fam] = probes
        print(f"{fam}: {len(fam_texts[fam])} test texts, {len(probes)} probe refs")
        del z
        gc.collect()

    # ---- one oracle at a time (memory: one fp32 0.6B model resident) ----
    panels = []
    for L in LAYERS:
        ckpt = os.path.join(args.ckpt_dir, f"layer{L:02d}", "best_stripped.pt")
        model, head, qwen_tok, ck = load_oracle(ckpt, device)
        ck_concepts = list(ck["concepts"])
        if ck_concepts != ps.main_block_concepts:
            raise RuntimeError(f"L{L}: checkpoint concept order != probe_set main_block_concepts")
        col_of = {c: ck_concepts.index(c) for c in disp}
        M = np.full((K, K), np.nan)
        for fam in sorted(fam_concepts):
            preds = encode_texts(fam_texts[fam], model, head, qwen_tok, device, bsz=args.bsz)
            ocols = preds[:, [col_of[c] for c in disp]]                  # [n, 54]
            pj = [c for c in fam_concepts[fam] if (L, c) in fam_probes[fam]]
            if not pj:
                continue
            P = np.stack([fam_probes[fam][(L, c)] for c in pj], 1)
            S = spearman_matrix(ocols, P)
            for k, c in enumerate(pj):
                M[:, disp.index(c)] = S[:, k]
        panels.append((f"per-layer oracle @ L{L}", M))
        del model, head
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    plot_panels(panels, disp, fam_of, args.out)
    npz_out = str(Path(args.out).with_suffix("")) + "_matrices.npz"
    np.savez_compressed(npz_out, concepts=np.array(disp),
                        **{f"L{L}": M for (t, M), L in zip(panels, LAYERS)})
    print(f"wrote {npz_out}")
    _print_stats(panels, K)


if __name__ == "__main__":
    main()
