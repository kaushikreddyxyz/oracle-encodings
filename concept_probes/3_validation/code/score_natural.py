"""Stage 6: score the natural eval pool with the trained probes (runs on a pod).

Extracts gemma-2-2b activations for the family's natural eval rows (token_ids
already in the file), projects every candidate probe + the shared random
directions, and computes a light covariate-shift AUC (generated-cache vs
natural activations, logistic, per layer) while both caches are present.

Writes <out>/<family>.natscores.npz:
  y [T, C], preds_adam [L, T, C], preds_ridge/dom/lda/logistic [L, T, C],
  preds_rand [L, 20, T], token2ex [T], plus per-example metadata arrays
  (example_id, nat_split, slice, cls_mined) and covshift_auc [L].

  python score_natural.py --family months --eval <eval/months.jsonl> \
      --probes <probes/months> --natstats natstats.npz --gen-cache <cache/months> \
      --layers 1,3,... --out <natscores dir>
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from extract import run as extract_run
from common import load_natstats, stable_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--natstats", required=True)
    ap.add_argument("--gen-cache", help="generated-pool cache dir for covshift AUC")
    ap.add_argument("--cache", required=True, help="natural acts cache dir (created)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="1,3,6,8,10,12,14,16,18,20,23,25")
    ap.add_argument("--model", default="google/gemma-2-2b")
    args = ap.parse_args()
    device = "cuda"
    layers = [int(x) for x in args.layers.split(",")]

    rows = [json.loads(l) for l in open(args.eval)]
    classes = sorted(rows[0]["targets"].keys())
    C = len(classes)

    cache = Path(args.cache)
    if not (cache / "index.json").exists():
        items = [(r["example_id"], r["token_ids"]) for r in rows]
        extract_run(items, layers, cache, args.model, 32768)
    index = json.load(open(cache / "index.json"))["offsets"]

    T = sum(n for _, n in index.values())
    y = np.zeros((T, C), dtype=np.float32)
    token2ex = np.zeros(T, dtype=np.int32)
    ex_meta = {"example_id": [], "nat_split": [], "slice": [], "cls_mined": []}
    for ri, r in enumerate(rows):
        off, n = index[r["example_id"]]
        token2ex[off:off + n] = ri
        for ci, c in enumerate(classes):
            for ti, s in r["targets"][c]:
                if ti < n:
                    y[off + ti, ci] = s
        for k in ex_meta:
            ex_meta[k].append(str(r.get(k.replace("example_id", "example_id"))
                                  if k == "example_id" else r.get(k)))

    z0 = np.load(Path(args.probes) / f"probes_l{layers[0]}.npz")
    S = z0["seeds"].size
    cand_names = ["adam", "ridge", "dom", "lda", "logistic"]
    preds = {k: np.zeros((len(layers), T, C), dtype=np.float32) for k in cand_names}
    preds_rand = np.zeros((len(layers), z0["rand_dirs"].shape[0], T), dtype=np.float32)
    covshift = np.full(len(layers), np.nan, dtype=np.float32)

    for li, L in enumerate(layers):
        acts = torch.from_numpy(
            np.load(cache / f"acts_l{L}.npy", mmap_mode="r")[:]).to(device)
        mu_np, sd_np = load_natstats(Path(args.natstats), L)
        mu = torch.from_numpy(mu_np).to(device)
        sd = torch.from_numpy(sd_np).to(device)
        z = np.load(Path(args.probes) / f"probes_l{L}.npz")

        def proj(W, b):
            w_t = torch.from_numpy(np.ascontiguousarray(W)).to(device).float()
            outs = []
            for s in range(0, T, 500_000):
                h = (acts[s:s + 500_000].float() - mu) / sd
                outs.append((h @ w_t.T).cpu().numpy())
            return np.concatenate(outs) + b

        # adam: mean over seeds at each seed's chosen λ
        acc = np.zeros((T, C), dtype=np.float32)
        for s in range(S):
            Wl = np.stack([z["W_adam"][s, z["chosen_lambda_idx"][s, ci], ci]
                           for ci in range(C)])
            bl = np.array([z["b_adam"][s, z["chosen_lambda_idx"][s, ci], ci]
                           for ci in range(C)])
            acc += proj(Wl, bl)
        preds["adam"][li] = acc / S
        if "chosen_lambda_ridge" in z:
            li_r = [int(z["chosen_lambda_ridge"][ci]) for ci in range(C)]
        else:
            li_r = [int(np.bincount(z["chosen_lambda_idx"][:, ci]).argmax()) for ci in range(C)]
        preds["ridge"][li] = proj(np.stack([z["W_ridge"][li_r[ci], ci] for ci in range(C)]),
                                  np.array([z["b_ridge"][li_r[ci], ci] for ci in range(C)]))
        preds["dom"][li] = proj(z["W_dom"], np.zeros(C))
        preds["lda"][li] = proj(z["W_lda"], np.zeros(C))
        preds["logistic"][li] = proj(z["W_logistic"], z["b_logistic"])
        preds_rand[li] = proj(z["rand_dirs"], np.zeros(z["rand_dirs"].shape[0])).T

        # covariate-shift AUC (Tier 2): generated vs natural activations
        if args.gen_cache and (Path(args.gen_cache) / f"acts_l{L}.npy").exists():
            gen = np.load(Path(args.gen_cache) / f"acts_l{L}.npy", mmap_mode="r")
            rng = np.random.default_rng(stable_seed("covshift", L))
            gi = rng.choice(gen.shape[0], min(20000, gen.shape[0]), replace=False)
            ni = rng.choice(T, min(20000, T), replace=False)
            X = np.concatenate([gen[np.sort(gi)].astype(np.float32),
                                acts[torch.from_numpy(np.sort(ni)).to(device)]
                                .cpu().numpy().astype(np.float32)])
            X = (X - mu_np) / sd_np
            lab = np.concatenate([np.ones(len(gi)), np.zeros(len(ni))])
            sh = rng.permutation(len(X))
            X, lab = X[sh], lab[sh]
            n_tr = int(0.7 * len(X))
            w = torch.zeros(X.shape[1], device=device, requires_grad=True)
            b = torch.zeros(1, device=device, requires_grad=True)
            opt = torch.optim.Adam([w, b], lr=0.05)
            Xt = torch.from_numpy(X[:n_tr]).to(device)
            yt = torch.from_numpy(lab[:n_tr]).to(device).float()
            for _ in range(100):
                zt = Xt @ w + b
                loss = torch.nn.functional.binary_cross_entropy_with_logits(zt, yt) \
                    + 1e-3 * (w * w).sum()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                sc = (torch.from_numpy(X[n_tr:]).to(device) @ w + b).cpu().numpy()
            from scipy.stats import rankdata
            lb = lab[n_tr:]
            r = rankdata(sc)
            n_p = int(lb.sum())
            covshift[li] = float((r[lb == 1].sum() - n_p * (n_p + 1) / 2)
                                 / (n_p * (lb == 0).sum()))
        del acts
        torch.cuda.empty_cache()
        print(f"[score_natural] layer {L} done (covshift={covshift[li]:.3f})", flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Path(args.out) / f"{args.family}.natscores.npz",
        classes=np.array(classes), layers=np.array(layers), y=y,
        token2ex=token2ex, covshift_auc=covshift, preds_rand=preds_rand,
        **{f"preds_{k}": v for k, v in preds.items()},
        **{f"ex_{k}": np.array(v) for k, v in ex_meta.items()})
    print(f"[score_natural] wrote {args.out}/{args.family}.natscores.npz")


if __name__ == "__main__":
    main()
