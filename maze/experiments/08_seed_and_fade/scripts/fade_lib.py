"""Seed-and-fade: shared pieces for the trainer and the evaluator.

Pure torch + maze.core.model only, so everything here imports (and is unit
tested) without maze_dataset / maze_transformer.

Training side
  drop_prob        the fade schedule p(it): share of the graft that is removed
  FadeForward      MazeGPT.forward, block by block, that also returns the
                   residual at the graft site (resid_pre of block `layer`, BEFORE
                   the graft is added) and the per-token loss
  aux_code_loss    positive-control loss: on sequences whose graft was dropped,
                   the model's own residual must carry the code the graft would
                   have written (read through pinv(W_inj))
  freeze_readers   stop training everything at / after the graft site

Eval side
  SiteEditor       what happens at the graft site at inference: off / on / delta /
                   ablate (mean-ablation inside a subspace)
  greedy_kv_edit   maze/paper/curves/eval_ckpts.greedy_kv with a SiteEditor
  retention        do the model's own activations carry the graft's code along
                   the graft's directions (generative OLS cosine + gain, zero-fit
                   pinv(W) probe vs a freshly fitted probe)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from maze.core.model import lm_cross_entropy_loss

FADES = ("none", "bernoulli", "scale")


# --------------------------------------------------------------------- training
def blank_prob(it: int, fade_start: int, fade_end: int) -> float:
    """Blanked-prompt curriculum: share of graft-KEEPING sequences whose adjacency list
    is blanked (so the graft is their only maze source): 1 before fade_start, linear
    to 0 at fade_end, 0 after. Sequences that lose their graft always keep the prompt."""
    if it < fade_start:
        return 1.0
    if it >= fade_end:
        return 0.0
    return 1.0 - (it - fade_start) / float(fade_end - fade_start)


def blank_adjacency(tokens, end_pos, rows, semicolon_id: int):
    """Return a copy of `tokens` with positions 1..end_pos-1 of the selected rows set
    to `;` (maze.core.data.dummy_adjacency, vectorised over a batch)."""
    out = tokens.clone()
    pos = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
    m = rows[:, None] & (pos >= 1) & (pos < end_pos[:, None])
    out[m] = semicolon_id
    return out


def drop_prob(it: int, fade: str, fade_start: int, fade_end: int) -> float:
    """Share of the graft removed at step `it`: 0 before fade_start, 1 from
    fade_end on, linear in between. `fade="none"` is a permanent graft."""
    assert fade in FADES, f"unknown fade: {fade}"
    if fade == "none" or it < fade_start:
        return 0.0
    if it >= fade_end:
        return 1.0
    return (it - fade_start) / float(fade_end - fade_start)


class FadeForward(nn.Module):
    """Same maths as `MazeGPT.forward(..., return_type="loss", loss_per_token=True)`
    (checked in test_fade_lib.py), plus the site residual."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, attention_mask, delta, layer: int, padding_mode: str,
                want_site: bool):
        m = self.model
        attn_mask, empty_rows = m._attn_masks(tokens, attention_mask, padding_mode)
        resid = m.embed(tokens) + m._positions(tokens, attention_mask)
        site = None
        for i, block in enumerate(m.blocks):
            if i == layer:
                if want_site:
                    site = resid
                if delta is not None:
                    resid = resid + delta.to(resid.dtype)
            resid = block(resid, attn_mask, empty_rows)
        logits = m.unembed(m.ln_final(resid))
        nll = lm_cross_entropy_loss(logits, tokens, attention_mask, per_token=True)
        return nll, site


def split_loss(nll, attention_mask, keep):
    """-> (loss, loss_kept, loss_dropped); `loss` is exactly the trainer's scalar
    (sum over real next-token positions / their count). The two parts are the same
    ratio restricted to sequences that kept / lost the graft (nan if none)."""
    ntm = (attention_mask[:, :-1] & attention_mask[:, 1:]).to(nll.dtype)
    seq_sum, seq_n = nll.sum(1), ntm.sum(1)
    loss = seq_sum.sum() / seq_n.sum()
    k = keep.to(nll.dtype)
    nan = torch.full((), float("nan"), device=nll.device, dtype=nll.dtype)
    kept = (seq_sum * k).sum() / (seq_n * k).sum() if bool(k.sum() > 0) else nan
    drop = (seq_sum * (1 - k)).sum() / (seq_n * (1 - k)).sum() if bool((1 - k).sum() > 0) else nan
    return loss, kept.detach(), drop.detach()


def aux_code_loss(site, feats, W, scale: float, dropped):
    """MSE between the code read off the model's own residual, (x @ pinv(W)) / scale,
    and the true +-1 code, over the graft positions of the dropped sequences.
    `feats` is the UNdropped feature tensor (0 outside the graft's positions)."""
    sel = (feats.abs().sum(-1) > 0) & dropped[:, None]
    if not bool(sel.any()):
        return site.new_zeros((), dtype=torch.float32)
    with torch.no_grad():
        pinv = torch.linalg.pinv(W.detach().float())        # [d, F]
    fhat = (site[sel].float() @ pinv) / scale
    return ((fhat - feats[sel].float()) ** 2).mean()


def freeze_readers(model, layer: int, frozen: bool = True) -> int:
    """(Un)freeze every parameter at or after the graft site: blocks[layer:],
    ln_final, unembed. AdamW skips parameters without a gradient, so frozen
    weights do not decay either. Returns the number of parameters touched."""
    n = 0
    mods = list(model.blocks[layer:]) + [model.ln_final, model.unembed]
    for mod in mods:
        for p in mod.parameters():
            p.requires_grad_(not frozen)
            n += p.numel()
    return n


# ------------------------------------------------------------------------- eval
def free_mask(n: int) -> np.ndarray:
    f = np.arange(n * n)
    m = np.zeros(2 * n * n, dtype=bool)
    m[: n * n] = (f // n) < n - 1
    m[n * n:] = (f % n) < n - 1
    return m


def span_projector(W_free: torch.Tensor) -> torch.Tensor:
    """Projector onto the span of the zero-mean free graft directions
    (rank 2n(n-1) - 1): the identifiable part of the code."""
    Wc = (W_free - W_free.mean(0, keepdim=True)).float()
    U, S, Vh = torch.linalg.svd(Wc, full_matrices=False)
    Q = Vh[S > S.max() * 1e-5]                               # [r, d]
    return Q.T @ Q


def random_projector(rank: int, d: int, seed: int, device) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, rank, generator=g))
    return (Q @ Q.T).to(device)


EDIT_MODES = ("off", "on", "delta", "ablate", "replace")


class SiteEditor:
    """Edit of the residual at the graft site for one batch.

    off     nothing (the faded model on autopilot)
    on      + scale * code(B)                      the graft as trained
    delta   + strength * scale * (code(B) - code(A))   steer along the seeded
            directions; A is the maze the model read from its prompt
    ablate  x - (x - mu[t]) P                      mean-ablate a subspace; mu is
            PER POSITION ([P + max_new, d]) so the position/token content of the
            subspace is kept (a single global mean erases it and the random-
            subspace floor then swings +-0.2 between draws)
    replace x - (x - mu[t]) P + strength * scale * code(B)   overwrite the span:
            remove whatever the model wrote there and write code(B) instead
    Only positions >= <ADJLIST_END> are touched, generated tokens included
    (the graft's token positions; see the token-matching rule)."""

    def __init__(self, mode, W=None, scale=1.0, conn_a=None, conn_b=None,
                 strength=1.0, P=None, mu=None):
        assert mode in EDIT_MODES, f"unknown edit mode: {mode}"
        self.mode, self.P, self.mu = mode, P, mu
        self.row = None
        if mode in ("on", "delta", "replace"):
            sb = conn_b.to(W.dtype) * 2.0 - 1.0
            if mode == "on":
                self.row = scale * (sb @ W)
            elif mode == "replace":
                self.row = strength * scale * (sb @ W)
            else:
                sa = conn_a.to(W.dtype) * 2.0 - 1.0
                self.row = strength * scale * ((sb - sa) @ W)

    def prompt(self, resid, active):                         # [B, P, d], [B, P, 1]
        if self.mode == "off":
            return resid
        if self.mode in ("ablate", "replace"):
            P = resid.shape[1]
            out = resid - active * ((resid - self.mu[None, :P]) @ self.P)
            return out if self.mode == "ablate" else out + active * self.row[:, None, :]
        return resid + active * self.row[:, None, :]

    def step(self, resid, t):                                # [B, d], position t
        if self.mode == "off":
            return resid
        if self.mode in ("ablate", "replace"):
            out = resid - (resid - self.mu[min(t, len(self.mu) - 1)]) @ self.P
            return out if self.mode == "ablate" else out + self.row
        return resid + self.row


@torch.no_grad()
def greedy_kv_edit(model, editor: SiteEditor, layer: int, prompt, end_pos, max_new: int,
                   path_end_id: int, pad_id: int):
    """`eval_ckpts.greedy_kv` with the additive graft replaced by a SiteEditor.
    `prompt` is [B, P], unpadded. Returns [B, max_new] ids (pad after <PATH_END>)."""
    dev = prompt.device
    B, P = prompt.shape
    H, dh = model.cfg.n_heads, model.cfg.d_head
    pos = torch.arange(P + max_new, device=dev)
    resid = model.embed(prompt) + model.pos_embed(pos[:P])[None]
    active = (pos[None, :P] >= end_pos[:, None]).to(resid.dtype)[:, :, None]
    caches = []
    for i, blk in enumerate(model.blocks):
        if i == layer:
            resid = editor.prompt(resid, active)
        x = blk.ln1(resid)
        q, k, v = blk.attn.qkv(x).split(blk.attn.d_inner, dim=-1)
        q = q.view(B, P, H, dh).transpose(1, 2)
        k = k.view(B, P, H, dh).transpose(1, 2)
        v = v.view(B, P, H, dh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=blk.attn.scale)
        resid = resid + blk.attn.out(y.transpose(1, 2).reshape(B, P, -1))
        resid = resid + blk.mlp(blk.ln2(resid))
        K = torch.empty(B, H, P + max_new, dh, device=dev, dtype=k.dtype); K[:, :, :P] = k
        V = torch.empty(B, H, P + max_new, dh, device=dev, dtype=v.dtype); V[:, :, :P] = v
        caches.append((K, V))
    cur = model.unembed(model.ln_final(resid[:, -1])).argmax(-1)
    out = torch.full((B, max_new), pad_id, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for step in range(max_new):
        out[:, step] = torch.where(done, torch.full_like(cur, pad_id), cur)
        done = done | (cur == path_end_id)
        if bool(done.all()) or step == max_new - 1:
            break
        t = P + step
        resid = model.embed(cur) + model.pos_embed(pos[t])[None]
        for i, blk in enumerate(model.blocks):
            if i == layer:
                resid = editor.step(resid, t)                # t >= P > end_pos
            x = blk.ln1(resid)
            q, k, v = blk.attn.qkv(x).split(blk.attn.d_inner, dim=-1)
            K, V = caches[i]
            K[:, :, t] = k.view(B, H, dh)
            V[:, :, t] = v.view(B, H, dh)
            y = F.scaled_dot_product_attention(q.view(B, H, 1, dh), K[:, :, : t + 1],
                                               V[:, :, : t + 1], scale=blk.attn.scale)
            resid = resid + blk.attn.out(y.reshape(B, -1))
            resid = resid + blk.mlp(blk.ln2(resid))
        cur = model.unembed(model.ln_final(resid)).argmax(-1)
    return out


@torch.no_grad()
def site_residual(model, layer: int, tokens):
    """The model's OWN residual at the graft site (resid_pre of block `layer`,
    no graft added). [B, T, d]. Right-padded batches are fine: with the causal
    mask every real position only sees real keys (padded rows are garbage)."""
    if layer == 0:
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        return model.embed(tokens) + model.pos_embed(pos)[None]
    return model(tokens, stop_at_layer=layer, padding_mode="right")


@torch.no_grad()
def retention(X_fit, F_fit, X_test, F_test, W_free, scale: float, ridge: float = 1e-2) -> dict:
    """X: site residuals [N, d] at ONE position; F: +-1 free features [N, n_free].

    cos_gen / gain   generative OLS X ~ F (the graft's own linear form): per
                     connection, cosine between the fitted writing direction and
                     the graft direction, both in the zero-mean gauge, and the
                     amplitude along it relative to the graft's (1 = the model
                     writes the code at full graft strength, 0 = not at all)
    acc_zero_fit     sign of the code read through pinv(W): no fitting on the model
    acc_probe        ridge readout fitted on X_fit: what ANY linear probe can get
    acc_majority     predicting every connection's more common state
    """
    X_fit, X_test = X_fit.double(), X_test.double()
    F_fit, F_test = F_fit.double(), F_test.double()
    W = W_free.double() * scale
    Wc = W - W.mean(0, keepdim=True)
    xm, fm = X_fit.mean(0, keepdim=True), F_fit.mean(0, keepdim=True)
    Xc, Fc = X_fit - xm, F_fit - fm
    # generative: Xc ~ Fc @ Bg. F has a constant row sum, so Fc is rank n_free-1:
    # take the minimum-norm solution, which IS the zero-mean gauge.
    Bg = torch.linalg.pinv(Fc) @ Xc                           # [n_free, d]
    Bg = Bg - Bg.mean(0, keepdim=True)
    cos = F.cosine_similarity(Bg, Wc, dim=-1)
    gain = (Bg * Wc).sum(-1) / (Wc * Wc).sum(-1)
    # zero-fit readout through the graft's own directions
    fhat = (X_test - xm) @ torch.linalg.pinv(W) + fm
    acc_zero = ((fhat > 0) == (F_test > 0)).double().mean()
    # fitted ridge readout
    d = Xc.shape[1]
    A = torch.linalg.solve(Xc.T @ Xc + ridge * len(Xc) * torch.eye(d, dtype=Xc.dtype, device=Xc.device),
                           Xc.T @ Fc)
    fprobe = (X_test - xm) @ A + fm
    acc_probe = ((fprobe > 0) == (F_test > 0)).double().mean()
    p_open = (F_fit > 0).double().mean(0)
    acc_major = ((F_test > 0).double() * (p_open > 0.5) + (F_test <= 0).double() * (p_open <= 0.5)).mean()
    # share of the label-explained variance of X that lies in the graft's span
    P = span_projector(W_free).double()
    expl = Fc @ (torch.linalg.pinv(Fc) @ Xc)
    in_span = ((expl @ P) ** 2).sum() / (expl ** 2).sum().clamp_min(1e-30)
    return dict(cos_gen=float(cos.mean()), cos_gen_median=float(cos.median()),
                gain=float(gain.mean()), acc_zero_fit=float(acc_zero), acc_probe=float(acc_probe),
                acc_majority=float(acc_major), explained_in_span=float(in_span))
