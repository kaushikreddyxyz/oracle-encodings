#!/usr/bin/env python3
"""Sparse autoencoders for the maze-steering experiments: five kinds behind one interface.

  relu   ReLU encoder + L1 penalty (decoder-norm weighted), dead latents resampled
  topk   per-token TopK (Gao et al. 2024) + aux-k loss for dead latents
  btopk  BatchTopK (Bussmann et al. 2024): top k*B activations over the batch; inference uses the learned threshold
  mat    Matryoshka BatchTopK (Bussmann et al. 2025): nested prefixes of the dictionary must each reconstruct
  mp     matching-pursuit SAE (Costa et al. 2025): tied unit-norm dictionary, k greedy residual steps, non-negative codes

Interface used by the steering code: `encode(x) -> a [.., n_lat]`, `W_dec [n_lat, d]`, `b_dec [d]`, `decode(a)`.
A steering edit is always written as x + (a_new - a) @ W_dec, so the SAE's reconstruction error is left untouched.

Spec strings: `<kind>-m<mult>-k<k>` or `relu-m<mult>-l<l1>`; n_lat = round(mult * F_exp) where F_exp is the number of
features the caller expects at that site (connections + cell identities). `mult` may also be `n<absolute>`.
Training (kernel-launch bound at these sizes, so large batches: 10k steps x 16,384 = 164M samples): inputs scaled to mean
norm sqrt(d), Adam, 500-step warmup, linear decay over the last 20%, unit-norm decoder
with the gradient projected; last 5% of rows held out (rows are ordered by maze, so held-out = unseen mazes).
"""
import math
import torch
import torch.nn.functional as F


def parse_spec(spec, f_exp):
    parts = spec.split("-"); kind = parts[0]; kw = {p[0]: p[1:] for p in parts[1:]}
    m = kw.get("m", "1.5"); n_lat = int(m[1:]) if m.startswith("n") else int(round(float(m) * f_exp))
    return kind, max(8, n_lat), int(kw.get("k", "16")), float(kw.get("l", "0.03"))


class BaseSAE(torch.nn.Module):
    def __init__(self, d, n_lat, kind, k=16):
        super().__init__()
        self.d, self.n_lat, self.kind, self.k = d, n_lat, kind, k
        W = torch.randn(n_lat, d); W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(W.clone())
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        if kind != "mp":
            self.W_enc = torch.nn.Parameter(W.clone().T * 0.5)      # [d, n_lat]
            self.b_enc = torch.nn.Parameter(torch.zeros(n_lat))
        self.register_buffer("threshold", torch.tensor(0.0))        # btopk / mat inference threshold

    def pre(self, x):
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, a):
        return a @ self.W_dec + self.b_dec

    def _mp(self, x):
        r = x - self.b_dec; a = torch.zeros(x.shape[0], self.n_lat, device=x.device, dtype=x.dtype)
        ar = torch.arange(x.shape[0], device=x.device)
        for _ in range(self.k):
            s = r @ self.W_dec.T; v, j = s.max(dim=1); v = v.clamp(min=0)
            a = a.index_put((ar, j), v, accumulate=True); r = r - v[:, None] * self.W_dec[j]
        return a

    def encode(self, x, train=False):
        shape = x.shape; x = x.reshape(-1, shape[-1])
        if self.kind == "mp":
            a = self._mp(x)
        else:
            a = self.pre(x)
            if self.kind == "topk":
                v, i = a.topk(min(self.k, self.n_lat), dim=-1); a = torch.zeros_like(a).scatter_(-1, i, v)
            elif self.kind in ("btopk", "mat"):
                if train:
                    kk = min(self.k * a.shape[0], a.numel()); thr = a.flatten().topk(kk).values[-1]
                    a = torch.where(a >= thr, a, torch.zeros_like(a))
                else:
                    a = torch.where(a > self.threshold, a, torch.zeros_like(a))
        return a.reshape(*shape[:-1], self.n_lat)

    def forward(self, x, train=False):
        a = self.encode(x, train); return self.decode(a), a


def build(sd, device):
    sae = BaseSAE(sd["d"], sd["n_lat"], sd["kind"], sd["k"]).to(device); sae.load_state_dict(sd["state"]); sae.eval(); return sae


def train(X, spec, f_exp, device, steps=10000, batch=16384, lr=1e-3, seed=0, holdout=0.05, log_every=1000):
    """X [T, d] (half, on device), rows ordered by maze. Returns a state dict (for `build`) incl. scale, held-out metrics, curve."""
    kind, n_lat, k, l1 = parse_spec(spec, f_exp)
    torch.manual_seed(seed); d = X.shape[1]
    n_tr = int(len(X) * (1 - holdout)); Xtr, Xho = X[:n_tr], X[n_tr:]
    scale = float(Xtr[:200000].float().norm(dim=1).mean() / math.sqrt(d))
    sae = BaseSAE(d, n_lat, kind, k).to(device)
    with torch.no_grad(): sae.b_dec.copy_(Xtr[:200000].float().mean(0) / scale)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = lambda s: min(1.0, (s + 1) / 500) * min(1.0, (steps - s) / (0.2 * steps))
    prefixes = sorted({max(4, n_lat // 8), max(4, n_lat // 4), max(4, n_lat // 2), n_lat}) if kind == "mat" else [n_lat]
    fired = torch.zeros(n_lat, device=device); since = torch.zeros(n_lat, device=device); curve = []
    thr_ema = None

    def heldout():
        with torch.no_grad():
            tot = 0.0; var = 0.0; l0 = 0.0; act = torch.zeros(n_lat, device=device); mu = Xho.float().mean(0) / scale
            for lo in range(0, len(Xho), 32768):
                xb = Xho[lo: lo + 32768].float() / scale; rec, a = sae(xb)
                tot += float(((rec - xb) ** 2).sum()); var += float(((xb - mu) ** 2).sum()); l0 += float((a > 0).float().sum()); act += (a > 0).float().sum(0)
            return dict(fvu=tot / var, L0=l0 / len(Xho), dead_frac=float((act == 0).float().mean()))

    for step in range(steps):
        for g in opt.param_groups: g["lr"] = lr * sched(step)
        xb = Xtr[torch.randint(0, n_tr, (batch,), device=device)].float() / scale
        a = sae.encode(xb, train=True)
        if kind == "mat":
            loss = sum(((a[:, :m] @ sae.W_dec[:m] + sae.b_dec - xb) ** 2).sum(1).mean() for m in prefixes) / len(prefixes)
        else:
            loss = ((sae.decode(a) - xb) ** 2).sum(1).mean()
        rec_loss = float(loss.detach())
        if kind == "relu":
            loss = loss + l1 * min(1.0, step / (0.05 * steps)) * (a * sae.W_dec.norm(dim=1)[None]).sum(1).mean()
        elif kind in ("topk", "btopk", "mat"):
            dead = since > 600                                       # aux-k: dead latents explain the residual
            if bool(dead.any()):
                resid = (xb - sae.decode(a)).detach(); pre = sae.pre(xb) * dead[None].float()
                kk = min(max(1, n_lat // 2), int(dead.sum())); v, i = pre.topk(kk, dim=-1)
                aux = ((torch.zeros_like(pre).scatter_(-1, i, v) @ sae.W_dec - resid) ** 2).sum(1).mean()
                loss = loss + aux / 32
        opt.zero_grad(); loss.backward()
        with torch.no_grad():       # NB: touching .grad outside no_grad leaks the graph through Adam's moments
            sae.W_dec.grad -= (sae.W_dec.grad * sae.W_dec).sum(1, keepdim=True) * sae.W_dec
        opt.step()
        with torch.no_grad():
            sae.W_dec.data /= sae.W_dec.data.norm(dim=1, keepdim=True)
            on = (a > 0).any(0); since = torch.where(on, torch.zeros_like(since), since + 1); fired += on.float()
            if kind in ("btopk", "mat"):
                pos = a[a > 0]; t = pos.min() if len(pos) else torch.tensor(0.0, device=device)
                thr_ema = t if thr_ema is None else 0.99 * thr_ema + 0.01 * t; sae.threshold.copy_(thr_ema)
            if kind in ("relu", "mp") and step % 800 == 799 and step < 0.5 * steps:
                deadi = (since > 600).nonzero().flatten()
                if len(deadi):
                    xs = Xtr[torch.randint(0, n_tr, (16384,), device=device)].float() / scale; err = ((sae(xs)[0] - xs) ** 2).sum(1)
                    v = xs[torch.multinomial(err / err.sum(), len(deadi), replacement=True)] - sae.b_dec; v = v / v.norm(dim=1, keepdim=True)
                    sae.W_dec.data[deadi] = v
                    if kind == "relu": sae.W_enc.data[:, deadi] = 0.2 * v.T; sae.b_enc.data[deadi] = 0
                    since[deadi] = 0
        if step % log_every == log_every - 1 or step == steps - 1:
            sae.eval(); h = heldout(); curve.append(dict(step=step + 1, train_mse=rec_loss, **h))
    sae.eval()
    return dict(state={k_: v.detach().cpu() for k_, v in sae.state_dict().items()}, d=d, n_lat=n_lat, kind=kind, k=k, l1=l1, spec=spec, scale=scale,
                f_exp=f_exp, n_train_rows=n_tr, heldout=curve[-1], curve=curve,
                converged_ratio=curve[-1]["fvu"] / curve[max(0, len(curve) - 4)]["fvu"])


@torch.no_grad()
def loss_recovered(model, block_fn, toks, loss_mask, site_mask, layer, sae, scale, batch=250):
    """Next-token CE on `loss_mask` positions with resid_pre of block `layer` at `site_mask` replaced by (a) nothing,
    (b) the SAE reconstruction (no error term), (c) the site mean. recovered = (CE_mean - CE_sae) / (CE_mean - CE_clean)."""
    H, dh = model.cfg.n_heads, model.cfg.d_head; dev = toks.device
    def run(fn):
        tot = 0.0; n = 0
        for lo in range(0, len(toks), batch):
            tb = toks[lo: lo + batch]; sm = site_mask[lo: lo + batch]; lm = loss_mask[lo: lo + batch]
            resid = model.embed(tb) + model.pos_embed(torch.arange(tb.shape[1], device=dev))[None]
            for i, blk in enumerate(model.blocks):
                if i == layer and fn is not None: resid[sm] = fn(resid[sm])
                resid, _, _ = block_fn(blk, resid, H, dh)
            logits = model.unembed(model.ln_final(resid))[:, :-1]; tgt = tb[:, 1:]; m = lm[:, :-1]
            tot += float(F.cross_entropy(logits[m].float(), tgt[m], reduction="sum")); n += int(m.sum())
        return tot / n
    # site mean from the data itself
    acc = None; cnt = 0
    for lo in range(0, len(toks), batch):
        tb = toks[lo: lo + batch]; sm = site_mask[lo: lo + batch]
        resid = model.embed(tb) + model.pos_embed(torch.arange(tb.shape[1], device=dev))[None]
        for i, blk in enumerate(model.blocks):
            if i == layer: break
            resid, _, _ = block_fn(blk, resid, H, dh)
        x = resid[sm]; acc = x.sum(0) if acc is None else acc + x.sum(0); cnt += len(x)
    mean = acc / cnt
    clean = run(None); rec = run(lambda x: sae.decode(sae.encode(x / scale)) * scale); abl = run(lambda x: mean[None].expand_as(x))
    return dict(ce_clean=clean, ce_sae=rec, ce_mean_ablation=abl, loss_recovered=(abl - rec) / max(1e-9, abl - clean))


if __name__ == "__main__":      # CPU smoke test on synthetic sparse data
    torch.manual_seed(0); dev = torch.device("cpu"); d, nf = 32, 40
    D = torch.randn(nf, d); D /= D.norm(dim=1, keepdim=True)
    A = (torch.rand(20000, nf) < 0.08).float() * torch.rand(20000, nf); X = (A @ D).half()
    for spec in ("relu-m1.5-l0.03", "topk-m1.5-k4", "btopk-m1.5-k4", "mat-m1.5-k4", "mp-m1.5-k4"):
        sd = train(X, spec, nf, dev, steps=600, batch=512, log_every=200); sae = build(sd, dev)
        a = sae.encode(X[:64].float() / sd["scale"]); assert a.shape == (64, sd["n_lat"]) and (a >= 0).all()
        print(spec, {k: round(v, 3) for k, v in sd["heldout"].items()}, "n_lat", sd["n_lat"])
