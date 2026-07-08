#!/usr/bin/env python3
"""Oracle-prediction × probe-activation association matrix (fleet_offtarget-style).

For every oracle kind and layer, a 54×54 matrix of Spearman correlations
between the oracle's per-example prediction for concept i (rows) and the gemma
probe's activation for concept j (cols), on the Stage-6 natural eval texts
(TEST split), max-pooled to example level exactly like g2_retention.py.

Cell (i, j) is computed on family(j)'s eval texts: the oracle predicts ALL 54
concepts on any text, while gemma probe activations only exist within-family
(natscores are scored per family), so column j fixes the example set.
Diagonal = on-target fidelity; off-diagonal (within a family band) = leakage.

Panels:
  expA full-FT   @ L6, L8, L14   (head col l*K+c, main_block_concepts order)
  expA frozen    @ L6, L8, L14   (head-only baseline on base Qwen)
  expB-learn     @ dom(L8)       (v̂ = down(y);      ŝ_c = W_dom[c]·(v̂/σ_nat))
  expB-fixed     @ dom(L8)       (v̂ = y @ D_dom.T;  same readout)
The expB readout ŝ is basis-independent (decoded to raw activation space and
read with the dom probes), so the learn head's rotation non-identifiability
does not affect it. Probe reference per (layer, concept) uses the selection
arm from probe_set.json (same as g2); dom columns use preds_dom @ L8.

Usage:
  python3 fig_oracle_probe_assoc.py --ckpt-dir <dir with best.pt, frozen-baseline/,
      expB-{fixed,learn}/> --probe-set ../out --eval-data ../../stage6/data \
      --out ../out/figures/oracle_probe_assoc.png
"""
from __future__ import annotations

import argparse
import json
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
from g2_retention import load_family_truth, test_split_ids, class_index

QWEN_HUB = "Qwen/Qwen3-0.6B-Base"
DOM_LAYER = 8


# ------------------------------------------------------------------ encoders
def _resolve_model_name(name):
    # checkpoints store pod-local paths like /workspace/models/Qwen3-0.6B-Base
    return QWEN_HUB if name.startswith("/") else name


def load_all(ckpt_dir, device):
    """Returns (model_ft, model_base, qwen_tok, heads) where heads =
    {kind: EncoderHead}. expB heads ride the full-FT encoder (their ckpts
    record encoder_from=expA_prod/best.pt); frozen rides base Qwen."""
    ck_ft = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location="cpu", weights_only=False)
    ck_fr = torch.load(os.path.join(ckpt_dir, "frozen-baseline/best.pt"), map_location="cpu", weights_only=False)
    ck_bf = torch.load(os.path.join(ckpt_dir, "expB-fixed/best.pt"), map_location="cpu", weights_only=False)
    ck_bl = torch.load(os.path.join(ckpt_dir, "expB-learn/best.pt"), map_location="cpu", weights_only=False)
    assert "encoder_state" in ck_ft and ck_ft["mode"] == "expA"
    assert ck_bf["args"]["encoder_from"].endswith("expA_prod/best.pt")
    assert ck_bl["args"]["encoder_from"].endswith("expA_prod/best.pt")

    name = _resolve_model_name(ck_ft["model_name"])
    model_base, qwen_tok, _ = te.load_encoder(name, torch.float32, device)
    model_ft, _, _ = te.load_encoder(name, torch.float32, device)
    model_ft.load_state_dict(ck_ft["encoder_state"])
    for m in (model_base, model_ft):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    heads = {}
    for kind, ck in [("fullFT", ck_ft), ("frozen", ck_fr),
                     ("expB-fixed", ck_bf), ("expB-learn", ck_bl)]:
        h = te.EncoderHead(ck["hidden_size"], ck["K"], ck["mode"]).to(device)
        h.load_state_dict(ck["head_state"])
        h.eval()
        for p in h.parameters():
            p.requires_grad_(False)
        heads[kind] = h
    return model_ft, model_base, qwen_tok, heads


@torch.no_grad()
def encode_family(texts, model_ft, model_base, qwen_tok, heads, ps, device,
                  max_tokens=1024, bsz=16):
    """Two forward passes over the family's texts; per-example max-pooled
    predictions for every oracle kind. Returns dict of [n, cols] arrays."""
    K = ps.K
    n = len(texts)
    # dom readout pieces (raw-space v̂ -> per-concept dom score contribution)
    abl_idx = int(np.where(ps.layer_index == ps.ablation_layer)[0][0])
    W_over_std = torch.tensor(ps.W_dom_abl / ps.nat_std[abl_idx][None, :],
                              device=device)                      # [K, 2304]
    D_dom_T = torch.tensor(ps.D_dom.T, device=device)             # [K, 2304]

    out = {"fullFT": np.zeros((n, 3 * K), np.float32),
           "frozen": np.zeros((n, 3 * K), np.float32),
           "expB-fixed": np.zeros((n, K), np.float32),
           "expB-learn": np.zeros((n, K), np.float32)}
    # sort by length for padding efficiency, restore order at the end
    order = np.argsort([len(t) for t in texts])
    for s in tqdm(range(0, n, bsz), desc="encode", leave=False):
        idxs = order[s:s + bsz]
        batch = [texts[i] for i in idxs]
        enc = qwen_tok(batch, add_special_tokens=True, truncation=True,
                       max_length=max_tokens, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        mask = attn.bool()
        h_ft = model_ft(input_ids=ids, attention_mask=attn).last_hidden_state
        h_base = model_base(input_ids=ids, attention_mask=attn).last_hidden_state

        yA, _ = heads["fullFT"](h_ft)          # [B, T, 3K]
        yF, _ = heads["frozen"](h_base)        # [B, T, 3K]
        y_bf, _ = heads["expB-fixed"](h_ft)    # [B, T, K]
        y_bl, v_bl = heads["expB-learn"](h_ft)  # v_bl [B, T, 2304]
        v_bf = y_bf @ D_dom_T                  # [B, T, 2304]  (v̂ = y @ D_dom.T)
        s_bf = v_bf @ W_over_std.T             # [B, T, K]
        s_bl = v_bl @ W_over_std.T
        for bi, i in enumerate(idxs):
            m = mask[bi]
            out["fullFT"][i] = yA[bi][m].max(0).values.float().cpu().numpy()
            out["frozen"][i] = yF[bi][m].max(0).values.float().cpu().numpy()
            out["expB-fixed"][i] = s_bf[bi][m].max(0).values.float().cpu().numpy()
            out["expB-learn"][i] = s_bl[bi][m].max(0).values.float().cpu().numpy()
    return out


# ------------------------------------------------------------------ probes
def probe_pmax(z, ci, layer, arm, test_ex_ids, n_ex):
    layers = [int(x) for x in z["layers"]]
    li = layers.index(layer)
    p = z[f"preds_{arm}"][li, :, ci].astype(np.float64)
    pmax = np.full(n_ex, -np.inf)
    np.maximum.at(pmax, z["token2ex"], p)
    return pmax[test_ex_ids]


# ------------------------------------------------------------------ spearman
def spearman_matrix(A, B):
    """A [n, p], B [n, q] -> [p, q] Spearman correlations (rank -> Pearson)."""
    def rz(X):
        R = np.apply_along_axis(rankdata, 0, X)
        R -= R.mean(0, keepdims=True)
        sd = R.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        return R / sd
    n = A.shape[0]
    return (rz(A).T @ rz(B)) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--probe-set", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bsz", type=int, default=16)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    ps = te.ProbeSet(args.probe_set)
    K, layers = ps.K, [int(x) for x in ps.layers]
    selection = ps.meta.get("selection", {})
    mbc = ps.main_block_concepts     # expA head column order
    dbc = ps.dom_block_concepts      # expB ŝ column order (W_dom_abl row order)
    fams = ps.families

    # display order = (family, concept) sorted, matching fleet_offtarget rows
    disp = sorted(ps.concepts, key=lambda c: (fams[c], c))
    fam_of = [fams[c] for c in disp]
    col_main = {c: mbc.index(c) for c in disp}
    col_dom = {c: dbc.index(c) for c in disp}

    model_ft, model_base, qwen_tok, heads = load_all(args.ckpt_dir, device)

    fam_concepts = {}
    for c in disp:
        fam_concepts.setdefault(fams[c], []).append(c)

    # ---------------- per family: oracle preds + probe pmax on TEST split
    fam_data = {}
    for fam in sorted(fam_concepts):
        rows, z = load_family_truth(args.eval_data, fam)
        if z is None:
            raise RuntimeError(f"{fam}: natscores npz required for probe activations")
        tid = test_split_ids(rows)
        texts = [rows[i]["text"] for i in tid]
        print(f"{fam}: {len(texts)} test texts")
        enc = encode_family(texts, model_ft, model_base, qwen_tok, heads, ps,
                            device, bsz=args.bsz)
        probes = {}
        for c in fam_concepts[fam]:
            ci = class_index(z, c)
            for L in layers:
                arm = selection.get(str(L), {}).get(c, {}).get("arm")
                if arm is None:
                    continue
                probes[(L, c)] = probe_pmax(z, ci, L, arm, tid, len(rows))
            probes[("dom", c)] = probe_pmax(z, ci, DOM_LAYER, "dom", tid, len(rows))
        fam_data[fam] = dict(enc=enc, probes=probes)

    # ---------------- association matrices
    panels = []   # (title, M [54 rows oracle, 54 cols probe])
    for kind in ["fullFT", "frozen"]:
        for l, L in enumerate(layers):
            M = np.full((K, K), np.nan)
            for fam in sorted(fam_concepts):
                fd = fam_data[fam]
                ocols = fd["enc"][kind][:, [l * K + col_main[c] for c in disp]]  # [n, 54]
                pj = [c for c in fam_concepts[fam] if (L, c) in fd["probes"]]
                if not pj:
                    continue
                P = np.stack([fd["probes"][(L, c)] for c in pj], 1)
                S = spearman_matrix(ocols, P)                                    # [54, |pj|]
                for k, c in enumerate(pj):
                    M[:, disp.index(c)] = S[:, k]
            panels.append((f"expA {kind} @ L{L}", M))
    for kind in ["expB-learn", "expB-fixed"]:
        M = np.full((K, K), np.nan)
        for fam in sorted(fam_concepts):
            fd = fam_data[fam]
            ocols = fd["enc"][kind][:, [col_dom[c] for c in disp]]
            P = np.stack([fd["probes"][("dom", c)] for c in fam_concepts[fam]], 1)
            S = spearman_matrix(ocols, P)
            for k, c in enumerate(fam_concepts[fam]):
                M[:, disp.index(c)] = S[:, k]
        panels.append((f"{kind} @ dom(L{DOM_LAYER})", M))

    # ---------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fam_breaks = [i for i in range(1, K) if fam_of[i] != fam_of[i - 1]]
    labels = [f"{fam_of[i]}.{c}" if (i == 0 or fam_of[i] != fam_of[i - 1]) else c
              for i, c in enumerate(disp)]

    fig, axes = plt.subplots(3, 3, figsize=(26, 27))
    slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    for (title, M), (r, cc) in zip(panels, slots):
        ax = axes[r][cc]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        diag = float(np.nanmedian(np.diag(M)))
        off = float(np.nanmedian(M[~np.eye(K, dtype=bool)]))
        ax.set_title(f"{title}\nmedian diag {diag:.3f} | off-diag {off:.3f}", fontsize=11)
        for i in range(K):
            ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                   edgecolor="lime", lw=0.9))
        for b in fam_breaks:
            ax.axhline(b - .5, color="gray", lw=0.4)
            ax.axvline(b - .5, color="gray", lw=0.4)
        ax.set_xticks(range(K))
        ax.set_xticklabels(labels, rotation=90, fontsize=3.6)
        ax.set_yticks(range(K))
        ax.set_yticklabels(labels, fontsize=3.6)
        ax.set_xlabel("gemma probe activation (max-pool, test split)", fontsize=8)
        ax.set_ylabel("oracle prediction (max-pool)", fontsize=8)
    axes[2][2].axis("off")
    cb = fig.colorbar(im, ax=axes[2][2], fraction=0.6, aspect=12)
    cb.set_label("Spearman corr (oracle prediction vs probe activation)")
    fig.suptitle("oracle prediction × probe activation — Spearman association, "
                 "Stage-6 natural eval TEST split (rows=oracle concept, cols=probe; "
                 "green=on-target)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"wrote {args.out}")

    # numeric companion
    npz_out = str(Path(args.out).with_suffix("")) + "_matrices.npz"
    np.savez_compressed(npz_out,
                        concepts=np.array(disp),
                        **{t.replace(" ", "_").replace("(", "").replace(")", ""): M
                           for t, M in panels})
    print(f"wrote {npz_out}")
    for t, M in panels:
        d = np.diag(M)
        print(f"{t:24s} diag median {np.nanmedian(d):.4f} "
              f"min {np.nanmin(d):.4f} | off-diag median "
              f"{np.nanmedian(M[~np.eye(K, dtype=bool)]):.4f} "
              f"p95 |off| {np.nanpercentile(np.abs(M[~np.eye(K, dtype=bool)]), 95):.4f}")


if __name__ == "__main__":
    main()
