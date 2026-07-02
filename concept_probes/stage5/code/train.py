"""Train all probes of one family at the probed layers.

Per (family, layer): every class's probe trains simultaneously as stacked
independent rows over the shared unique-token activation cache — one gathered
minibatch of tokens updates every (seed is a separate loop; λ × class rows are
stacked). Objective per row: buffer-masked MSE + λ‖w‖² (§5.2), inputs
standardized with NATURAL-split stats. Closed-form ridge (the exact minimizer)
is solved alongside as verification, plus the §3 baselines.

  python train.py --family months --cache <cache/months> --stage4 <stage4/data> \
      --natstats natstats.npz --layers 1,3,... --out <out/months> [--read-shift 0]
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import FamilyData, load_natstats, stable_seed

LAMBDAS = [1e-4, 1e-3, 1e-2]
SEEDS = [0, 1, 2]


def build_layer_tensors(fam: FamilyData, B: int, read_shift: int, device):
    """Dense [T, C] target/mask matrices for train, and gathered val arrays."""
    T, C = fam.total_tokens, len(fam.classes)
    y_tr = torch.zeros(T, C, dtype=torch.float16)
    m_tr = torch.zeros(T, C, dtype=torch.float16)
    val = {}
    for ci, cls in enumerate(fam.classes):
        idx, y, m, ex, eids, roles = fam.class_split_arrays(cls, "train", B, read_shift)
        y_tr[torch.from_numpy(idx), ci] = torch.from_numpy(y).half()
        m_tr[torch.from_numpy(idx), ci] = torch.from_numpy(m).half()
        vidx, vy, vm, vex, veids, vroles = fam.class_split_arrays(cls, "val", B, read_shift)
        val[cls] = dict(idx=torch.from_numpy(vidx), y=torch.from_numpy(vy),
                        m=torch.from_numpy(vm), ex=vex, eids=veids, roles=vroles)
    return y_tr.to(device), m_tr.to(device), val


def val_mse_rows(acts, mu, sd, W, b, val, classes, n_lam):
    """Masked val MSE per (λ, class) row for one seed's W [R, d], b [R]."""
    out = torch.zeros(n_lam, len(classes), device=W.device)
    for ci, cls in enumerate(classes):
        v = val[cls]
        if v["idx"].numel() == 0:
            continue
        h = acts[v["idx"].to(W.device)].float()
        h = (h - mu) / sd
        rows = torch.arange(n_lam, device=W.device) * len(classes) + ci
        pred = h @ W[rows].T + b[rows]                       # [V, n_lam]
        y = v["y"].to(W.device).unsqueeze(1)
        m = v["m"].to(W.device).unsqueeze(1)
        out[:, ci] = ((m * (pred - y) ** 2).sum(0) / m.sum(0).clamp(min=1)).flatten()
    return out


def adam_fit(acts, mu, sd, y_tr, m_tr, val, classes, lambdas, seed, args, device):
    """One seed: stacked rows R = len(lambdas) x C. Returns best-by-val W, b, stats."""
    T, C = y_tr.shape
    n_lam, R, d = len(lambdas), len(lambdas) * C, acts.shape[1]
    g = torch.Generator(device="cpu").manual_seed(stable_seed("stage5", seed))
    W = (torch.randn(R, d, generator=g) * 0.01).to(device).requires_grad_(True)
    b = torch.zeros(R, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=args.lr)
    # Adam on this convex objective oscillates at constant lr and never reaches
    # the ridge optimum (pilot: Δρ_val ≈ 0.15); a full cosine decay closes the
    # gap exactly (verified vs closed form: val MSE matches to 1e-5). So: run
    # the whole schedule, no early stopping; best-by-val snapshot as safety.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)
    lam = torch.tensor(lambdas, device=device).repeat_interleave(C)   # [R]
    n_c = m_tr.sum(0).float().clamp(min=1)                            # [C]

    best_mse = torch.full((n_lam, C), float("inf"), device=device)
    best_W = torch.zeros(R, d, device=device)
    best_b = torch.zeros(R, device=device)
    epochs_run = 0
    for epoch in range(args.max_epochs):
        perm = torch.randperm(T, generator=g)
        for s in range(0, T, args.batch_tokens):
            idx = perm[s:s + args.batch_tokens].to(device)
            h = (acts[idx].float() - mu) / sd
            pred = h @ W.T + b                                        # [B, R]
            y = y_tr[idx].float().repeat(1, n_lam)
            m = m_tr[idx].float().repeat(1, n_lam)
            data = ((m * (pred - y) ** 2).sum(0) / n_c.repeat(n_lam))
            # scale the L2 term by the batch fraction so that summed over an
            # epoch it contributes λ‖w‖² exactly once (matches the ridge form)
            reg = (lam * (W * W).sum(1)).sum() * (idx.numel() / T)
            loss = data.sum() + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        epochs_run = epoch + 1
        if (epoch + 1) % args.val_every == 0 or epoch >= args.max_epochs - 3:
            with torch.no_grad():
                mse = val_mse_rows(acts, mu, sd, W, b, val, classes, n_lam)
                improved = mse < best_mse
                if improved.any():
                    rows = improved.flatten()
                    best_W[rows] = W.detach()[rows]
                    best_b[rows] = b.detach()[rows]
                    best_mse = torch.minimum(best_mse, mse)
    return (best_W.view(n_lam, C, d).cpu(), best_b.view(n_lam, C).cpu(),
            best_mse.cpu(), epochs_run)


def ridge_fit(acts, mu, sd, fam, B, read_shift, lambdas, device):
    """Exact minimizer per (λ, class): centered ridge on the masked train tokens.
    This is the PRIMARY fit — Adam on the same convex objective systematically
    undershoots it (pilot rounds 1–3) and is kept only as a seed diagnostic.
    Also selects λ per class by masked val MSE."""
    C, d = len(fam.classes), acts.shape[1]
    W = torch.zeros(len(lambdas), C, d)
    bias = torch.zeros(len(lambdas), C)
    val_mse = torch.full((len(lambdas), C), float("nan"))
    chosen = torch.zeros(C, dtype=torch.int64)
    stats = {}
    for ci, cls in enumerate(fam.classes):
        idx, y, m, *_ = fam.class_split_arrays(cls, "train", B, read_shift)
        keep = m > 0
        idx_t = torch.from_numpy(idx[keep]).to(device)
        y_t = torch.from_numpy(y[keep]).to(device).float()
        n = idx_t.numel()
        if n < 10:
            continue
        A = torch.zeros(d, d, device=device)
        hsum = torch.zeros(d, device=device)
        hy = torch.zeros(d, device=device)
        for s in range(0, n, 400_000):
            h = (acts[idx_t[s:s + 400_000]].float() - mu) / sd
            A += h.T @ h
            hsum += h.sum(0)
            hy += h.T @ y_t[s:s + 400_000]
        hbar, ybar = hsum / n, y_t.mean()
        A -= n * torch.outer(hbar, hbar)
        c = hy - n * hbar * ybar
        vidx, vy, vm, *_ = fam.class_split_arrays(cls, "val", B, read_shift)
        vkeep = vm > 0
        vh = (acts[torch.from_numpy(vidx[vkeep]).to(device)].float() - mu) / sd
        vy_t = torch.from_numpy(vy[vkeep]).to(device).float()
        for li, lam in enumerate(lambdas):
            w = torch.linalg.solve(A + lam * n * torch.eye(d, device=device), c)
            b = ybar - w @ hbar
            W[li, ci] = w.cpu()
            bias[li, ci] = b.cpu()
            if vy_t.numel():
                val_mse[li, ci] = float(((vh @ w + b - vy_t) ** 2).mean())
        del vh
        chosen[ci] = int(val_mse[:, ci].nan_to_num(float("inf")).argmin())
        stats[cls] = dict(n_tokens=int(n), pos_frac=float((y_t >= 0.5).float().mean()))
    return W, bias, val_mse, chosen, stats


def baselines_fit(acts, mu, sd, fam, B, read_shift, args, device):
    """DoM / shrinkage-LDA / logistic per class; shared random directions."""
    C, d = len(fam.classes), acts.shape[1]
    W_dom = torch.zeros(C, d); W_lda = torch.zeros(C, d)
    W_log = torch.zeros(C, d); b_log = torch.zeros(C)
    for ci, cls in enumerate(fam.classes):
        idx, y, m, *_ = fam.class_split_arrays(cls, "train", B, read_shift)
        keep = m > 0
        idx_t = torch.from_numpy(idx[keep]).to(device)
        lab = torch.from_numpy((y[keep] >= args.binarize_at)).to(device)
        h = (acts[idx_t].float() - mu) / sd
        pos, neg = h[lab], h[~lab]
        if len(pos) < 5 or len(neg) < 5:
            continue
        mu_p, mu_n = pos.mean(0), neg.mean(0)
        W_dom[ci] = (mu_p - mu_n).cpu()
        # pooled within-class covariance + shrinkage toward scaled identity
        Xp, Xn = pos - mu_p, neg - mu_n
        cov = (Xp.T @ Xp + Xn.T @ Xn) / max(len(h) - 2, 1)
        gam = args.lda_shrinkage
        cov = (1 - gam) * cov + gam * (torch.trace(cov) / d) * torch.eye(d, device=device)
        W_lda[ci] = torch.linalg.solve(cov, mu_p - mu_n).cpu()
        # logistic (balanced)
        w = torch.zeros(d, device=device, requires_grad=True)
        bb = torch.zeros(1, device=device, requires_grad=True)
        opt = torch.optim.Adam([w, bb], lr=0.05)
        wgt = torch.where(lab, len(h) / (2 * len(pos)), len(h) / (2 * len(neg))).float()
        target = lab.float()
        for _ in range(args.logistic_epochs):
            z = h @ w + bb
            loss = (wgt * torch.nn.functional.binary_cross_entropy_with_logits(
                z, target, reduction="none")).mean() + 1e-4 * (w * w).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        W_log[ci] = w.detach().cpu(); b_log[ci] = bb.detach().cpu()
    g = torch.Generator().manual_seed(stable_seed("rand_dirs"))
    rand = torch.randn(args.n_random_directions, d, generator=g)
    rand = rand / rand.norm(dim=1, keepdim=True)
    return W_dom, W_lda, W_log, b_log, rand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--stage4", default="concept_probes/stage4/data")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--natstats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="1,3,6,8,10,12,14,16,18,20,23,25")
    ap.add_argument("--classes", help="comma subset (pilot)")
    ap.add_argument("--read-shift", type=int, default=0)
    ap.add_argument("--buffer", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-epochs", type=int, default=250)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=20)   # unused; kept for CLI compat
    # small batches matter: with ~250k-token classes, 131k-token batches gave
    # Adam only ~200 steps before early stop and it undershot the exact ridge
    # optimum by Δρ≈0.15 (pilot). 16k tokens ≈ 16+ steps/epoch converges.
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--binarize-at", type=float, default=0.5)
    ap.add_argument("--lda-shrinkage", type=float, default=0.1)
    ap.add_argument("--logistic-epochs", type=int, default=150)
    ap.add_argument("--n-random-directions", type=int, default=20)
    args = ap.parse_args()
    device = "cuda"
    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    classes = args.classes.split(",") if args.classes else None
    fam = FamilyData(Path(args.stage4) / args.family / "final", classes)
    print(f"[train] {args.family}: classes={fam.classes} tokens={fam.total_tokens}")
    y_tr, m_tr, val = build_layer_tensors(fam, args.buffer, args.read_shift, device)

    for L in layers:
        t0 = time.time()
        acts = torch.from_numpy(
            np.load(Path(args.cache) / f"acts_l{L}.npy", mmap_mode="r")[:]).to(device)
        mu_np, sd_np = load_natstats(Path(args.natstats), L)
        mu = torch.from_numpy(mu_np).to(device)
        sd = torch.from_numpy(sd_np).to(device)

        W_seeds, b_seeds, mse_seeds, epochs = [], [], [], []
        for seed in SEEDS:
            Wb, bb, mse, ep = adam_fit(acts, mu, sd, y_tr, m_tr, val, fam.classes,
                                       LAMBDAS, seed, args, device)
            W_seeds.append(Wb); b_seeds.append(bb); mse_seeds.append(mse); epochs.append(ep)
        W_adam = torch.stack(W_seeds)          # [S, Λ, C, d]
        b_adam = torch.stack(b_seeds)
        val_mse = torch.stack(mse_seeds)       # [S, Λ, C]
        chosen_lam = val_mse.argmin(dim=1)     # [S, C]

        W_ridge, b_ridge, ridge_val_mse, ridge_lam, rstats = ridge_fit(
            acts, mu, sd, fam, args.buffer, args.read_shift, LAMBDAS, device)
        W_dom, W_lda, W_log, b_log, rand = baselines_fit(acts, mu, sd, fam, args.buffer,
                                                         args.read_shift, args, device)
        np.savez(out / f"probes_l{L}.npz",
                 classes=np.array(fam.classes), lambdas=np.array(LAMBDAS),
                 seeds=np.array(SEEDS), read_shift=args.read_shift,
                 W_adam=W_adam.numpy(), b_adam=b_adam.numpy(),
                 val_mse=val_mse.numpy(), chosen_lambda_idx=chosen_lam.numpy(),
                 W_ridge=W_ridge.numpy(), b_ridge=b_ridge.numpy(),
                 ridge_val_mse=ridge_val_mse.numpy(),
                 chosen_lambda_ridge=ridge_lam.numpy(),
                 W_dom=W_dom.numpy(), W_lda=W_lda.numpy(),
                 W_logistic=W_log.numpy(), b_logistic=b_log.numpy(),
                 rand_dirs=rand.numpy(), nat_mean=mu_np, nat_std=sd_np,
                 epochs=np.array(epochs))
        with open(out / f"trainstats_l{L}.json", "w") as f:
            json.dump({"layer": L, "epochs": epochs, "ridge": rstats,
                       "sec": round(time.time() - t0, 1)}, f, indent=1)
        del acts
        torch.cuda.empty_cache()
        print(f"[train] layer {L} done in {time.time() - t0:.0f}s "
              f"(epochs={epochs})", flush=True)


if __name__ == "__main__":
    main()
