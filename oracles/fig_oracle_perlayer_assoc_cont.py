#!/usr/bin/env python3
"""Oracle×probe Spearman association — ORIGINAL vs CONTINUATION (cont1)
checkpoints for L6 and L8, one 54×54 panel each (4 panels + colorbar).

Same computation and style as fig_oracle_perlayer_assoc.py (which this
imports): per-example max-pooled oracle predictions vs same-layer gemma probe
max-pooled activations on the Stage-6 natural-eval TEST split. The ORIGINAL
panels are re-plotted from that script's saved matrices npz (no recompute);
the cont1 panels are computed fresh from the continuation checkpoints.

Usage:
  python3 fig_oracle_perlayer_assoc_cont.py \
      --ckpt-l6 <layer06 cont1 best_stripped.pt> \
      --ckpt-l8 <layer08 cont1 best_stripped.pt> \
      --orig-npz out/figures/oracle_perlayer_assoc_matrices.npz \
      --probe-set ../attribution/out --eval-data ../concept_probes/3_validation/data \
      --out out/figures/oracle_perlayer_assoc_cont1.png    # run from oracles/
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_encoder as te
from fig_oracle_perlayer_assoc import (encode_texts, load_oracle, probe_pmax,
                                       spearman_matrix, _print_stats)
from g2_retention import class_index, load_family_truth, test_split_ids

LAYERS = [6, 8]


def plot_panels_n(panels, disp, fam_of, out_path, suptitle):
    """fig_oracle_perlayer_assoc.plot_panels generalized to N panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = len(disp)
    n = len(panels)
    fam_breaks = [i for i in range(1, K) if fam_of[i] != fam_of[i - 1]]
    labels = [f"{fam_of[i]}.{c}" if (i == 0 or fam_of[i] != fam_of[i - 1]) else c
              for i, c in enumerate(disp)]

    fig, axes = plt.subplots(1, n + 1, figsize=(11.2 * n + 1.5, 9.5),
                             gridspec_kw={"width_ratios": [1] * n + [0.06]})
    im = None
    for (title, M), ax in zip(panels, axes[:n]):
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
    cb = fig.colorbar(im, cax=axes[n])
    cb.set_label("Spearman corr (oracle prediction vs probe activation)")
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    print(f"wrote {out_path}", flush=True)


def compute_matrix(ckpt_path, layer, ps, disp, fam_concepts, fam_texts,
                   fam_probes, device, bsz):
    model, head, qwen_tok, ck = load_oracle(ckpt_path, device)
    ck_concepts = list(ck["concepts"])
    if ck_concepts != ps.main_block_concepts:
        raise RuntimeError(f"L{layer}: checkpoint concept order != probe_set main_block_concepts")
    assert ck["layer"] == layer, (ck["layer"], layer)
    col_of = {c: ck_concepts.index(c) for c in disp}
    K = len(disp)
    M = np.full((K, K), np.nan)
    for fam in sorted(fam_concepts):
        preds = encode_texts(fam_texts[fam], model, head, qwen_tok, device, bsz=bsz)
        ocols = preds[:, [col_of[c] for c in disp]]
        pj = [c for c in fam_concepts[fam] if (layer, c) in fam_probes[fam]]
        if not pj:
            continue
        P = np.stack([fam_probes[fam][(layer, c)] for c in pj], 1)
        S = spearman_matrix(ocols, P)
        for k, c in enumerate(pj):
            M[:, disp.index(c)] = S[:, k]
    tokens_m = ck.get("train_tokens", float("nan")) / 1e6
    del model, head
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return M, tokens_m, ck.get("step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-l6", required=True)
    ap.add_argument("--ckpt-l8", required=True)
    ap.add_argument("--orig-npz", required=True)
    ap.add_argument("--probe-set", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bsz", type=int, default=8)
    args = ap.parse_args()

    ps = te.ProbeSet(args.probe_set)
    fams = ps.families
    disp = sorted(ps.concepts, key=lambda c: (fams[c], c))
    fam_of = [fams[c] for c in disp]
    K = ps.K

    z0 = np.load(args.orig_npz, allow_pickle=True)
    if list(z0["concepts"]) != disp:
        raise RuntimeError("orig npz concept order mismatch")
    orig = {L: z0[f"L{L}"] for L in LAYERS}

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}", flush=True)
    selection = ps.meta.get("selection", {})

    fam_concepts = {}
    for c in disp:
        fam_concepts.setdefault(fams[c], []).append(c)

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
        print(f"{fam}: {len(fam_texts[fam])} test texts, {len(probes)} probe refs", flush=True)
        del z
        gc.collect()

    panels = []
    cont_mats = {}
    # original tokens-seen, from the overnight runs' best checkpoints
    orig_tokens = {6: 690, 8: 701}
    for L, ckpt in ((6, args.ckpt_l6), (8, args.ckpt_l8)):
        panels.append((f"L{L} original ({orig_tokens[L]}M tokens)", orig[L]))
        M, tok_m, step = compute_matrix(ckpt, L, ps, disp, fam_concepts,
                                        fam_texts, fam_probes, device, args.bsz)
        cont_mats[L] = M
        panels.append((f"L{L} cont1 ({tok_m:.0f}M tokens, step {step})", M))
        print(f"L{L} cont1 panel done", flush=True)

    plot_panels_n(panels, disp, fam_of, args.out,
                  "per-layer oracle prediction × probe activation — original vs cont1 "
                  "(Spearman, Stage-6 natural eval TEST split; rows=oracle concept, cols=probe)")
    npz_out = str(Path(args.out).with_suffix("")) + "_matrices.npz"
    np.savez_compressed(npz_out, concepts=np.array(disp),
                        **{f"L{L}_cont1": cont_mats[L] for L in LAYERS})
    print(f"wrote {npz_out}", flush=True)
    _print_stats(panels, K)


if __name__ == "__main__":
    main()
