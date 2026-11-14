#!/usr/bin/env python3
"""Unit tests for fade_lib (pure torch; run with pytest or directly)."""
import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import torch

from fade_lib import (FadeForward, blank_adjacency, blank_prob, SiteEditor, aux_code_loss, drop_prob, free_mask, freeze_readers,
                      greedy_kv_edit, random_projector, retention, site_residual, span_projector,
                      split_loss)
from maze.core.model import MazeGPT, MazeGPTConfig


def tiny(seed=0):
    cfg = MazeGPTConfig(d_model=32, d_head=8, n_heads=4, n_layers=4, d_mlp=64, d_vocab=20,
                        d_vocab_out=20, n_ctx=40, attn_scale=8 ** 0.5)
    return MazeGPT(cfg).init_weights_(seed).eval()


def batch(B=6, T=17, seed=0):
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(0, 20, (B, T), generator=g)
    lens = torch.randint(8, T + 1, (B,), generator=g); lens[0] = T
    attn = (torch.arange(T)[None] < lens[:, None]).long()
    return toks, attn


def test_drop_prob():
    assert drop_prob(0, "none", 10, 20) == 0 and drop_prob(10**6, "none", 10, 20) == 0
    assert drop_prob(9, "bernoulli", 10, 20) == 0 and drop_prob(10, "bernoulli", 10, 20) == 0
    assert abs(drop_prob(15, "scale", 10, 20) - 0.5) < 1e-12
    assert drop_prob(20, "bernoulli", 10, 20) == 1 and drop_prob(99, "bernoulli", 10, 20) == 1


def test_forward_parity():
    m = tiny(); toks, attn = batch()
    delta = torch.randn(6, 17, 32)
    for mode in ("exact", "right"):
        for layer in (0, 2):
            for d in (None, delta):
                ref = m(toks, attention_mask=attn, inject_delta=d, inject_layer=layer,
                        return_type="loss", padding_mode=mode)
                nll, site = FadeForward(m)(toks, attn, d, layer, mode, True)
                loss, _, _ = split_loss(nll, attn.bool(), torch.ones(6, dtype=torch.bool))
                assert torch.allclose(loss, ref, atol=1e-6), (mode, layer, float(loss), float(ref))
                own = site_residual(m, layer, toks[:1])       # unpadded row
                assert torch.allclose(site[:1], own, atol=1e-5)


def test_split_loss_parts():
    m = tiny(); toks, attn = batch()
    nll, _ = FadeForward(m)(toks, attn, None, 0, "right", False)
    keep = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool)
    loss, k, d = split_loss(nll, attn.bool(), keep)
    ntm = (attn[:, :-1] & attn[:, 1:]).float()
    nk, nd = ntm[keep].sum(), ntm[~keep].sum()
    assert torch.allclose(loss, (k * nk + d * nd) / (nk + nd), atol=1e-6)


def test_aux_zero_at_code_and_grad_upstream_only():
    torch.manual_seed(0)
    W = torch.nn.functional.normalize(torch.randn(8, 32), dim=-1)
    feats = torch.zeros(3, 5, 8); feats[:, 2:] = torch.randint(0, 2, (3, 1, 8)).float() * 2 - 1
    site = 1.7 * feats @ W
    dropped = torch.tensor([True, False, True])
    assert float(aux_code_loss(site, feats, W, 1.7, dropped)) < 1e-10
    assert float(aux_code_loss(torch.zeros_like(site), feats, W, 1.7, dropped)) > 0.9
    assert float(aux_code_loss(site, feats, W, 1.7, torch.zeros(3, dtype=torch.bool))) == 0.0


def test_freeze_readers():
    m = tiny().train()
    freeze_readers(m, 2, True)
    toks, attn = batch()
    nll, _ = FadeForward(m)(toks, attn, None, 2, "right", False)
    nll.sum().backward()
    assert all(p.grad is None for p in m.blocks[2].parameters())
    assert all(p.grad is None for p in m.unembed.parameters())
    assert all(p.grad is not None for p in m.blocks[1].parameters())
    assert m.embed.weight.grad is not None
    freeze_readers(m, 2, False)
    assert all(p.requires_grad for p in m.parameters())


def test_kv_matches_full_forward_under_edits():
    m = tiny(); n = 2; Fd = 2 * n * n
    torch.manual_seed(1)
    W = torch.randn(Fd, 32)
    prompt = torch.randint(0, 20, (5, 9)); end = torch.tensor([3, 4, 5, 3, 6])
    ca = torch.randint(0, 2, (5, Fd)).bool(); cb = torch.randint(0, 2, (5, Fd)).bool()
    P = span_projector(W[torch.tensor(free_mask(n))]); mu = torch.randn(9 + 6, 32)
    for layer in (0, 2):
        for ed in (SiteEditor("off"), SiteEditor("on", W, 0.7, ca, cb),
                   SiteEditor("delta", W, 0.7, ca, cb, strength=2.0), SiteEditor("ablate", P=P, mu=mu),
                   SiteEditor("replace", W, 0.7, ca, cb, strength=1.5, P=P, mu=mu)):
            gen = greedy_kv_edit(m, ed, layer, prompt, end, 6, path_end_id=999, pad_id=19)
            toks = prompt.clone()
            for _ in range(6):                                # reference: full forward + same edit
                T = toks.shape[1]
                resid = m.embed(toks) + m.pos_embed(torch.arange(T))[None]
                act = (torch.arange(T)[None] >= end[:, None]).float()[:, :, None]
                for i, blk in enumerate(m.blocks):
                    if i == layer:
                        resid = ed.prompt(resid, act)
                    resid = blk(resid, None, None)
                nxt = m.unembed(m.ln_final(resid[:, -1])).argmax(-1)
                toks = torch.cat([toks, nxt[:, None]], 1)
            assert torch.equal(gen, toks[:, 9:]), (layer, ed.mode)


def test_retention_recovers_planted_code():
    torch.manual_seed(0)
    n = 4; fm = torch.tensor(free_mask(n)); nf = int(fm.sum())
    W = torch.nn.functional.normalize(torch.randn(nf, 64), dim=-1)
    # constant-sum labels, like spanning trees: exactly n^2-1 of the free features open
    Fm = -torch.ones(3000, nf)
    for r in range(3000):
        Fm[r, torch.randperm(nf)[: n * n - 1]] = 1
    noise = 0.05 * torch.randn(3000, 64)
    X = 0.6 * Fm @ W + noise + 3.0
    r = retention(X[:2000], Fm[:2000], X[2000:], Fm[2000:], W, 1.0)
    assert r["cos_gen"] > 0.97 and abs(r["gain"] - 0.6) < 0.03 and r["acc_zero_fit"] > 0.99, r
    assert r["explained_in_span"] > 0.97
    V = torch.nn.functional.normalize(torch.randn(nf, 64), dim=-1)   # code present, OTHER directions
    r2 = retention(X[:2000], Fm[:2000], X[2000:], Fm[2000:], V, 1.0)
    assert abs(r2["cos_gen"]) < 0.1 and r2["acc_probe"] > 0.99 and r2["acc_zero_fit"] < 0.75, r2
    r3 = retention(noise[:2000], Fm[:2000], noise[2000:], Fm[2000:], W, 1.0)  # no code at all
    assert abs(r3["gain"]) < 0.02 and r3["acc_probe"] < r3["acc_majority"] + 0.03, r3


def test_projectors():
    W = torch.randn(12, 64)
    P = span_projector(W)
    assert abs(float(torch.trace(P)) - 11) < 1e-3 and torch.allclose(P @ P, P, atol=1e-4)
    R = random_projector(11, 64, 0, "cpu")
    assert abs(float(torch.trace(R)) - 11) < 1e-3 and torch.allclose(R @ R, R, atol=1e-4)


def test_blank_prompt():
    assert blank_prob(0, 10, 20) == 1.0 and abs(blank_prob(15, 10, 20) - 0.5) < 1e-12 and blank_prob(20, 10, 20) == 0.0
    toks = torch.arange(24).view(2, 12); end = torch.tensor([5, 8]); rows = torch.tensor([True, False])
    out = blank_adjacency(toks, end, rows, 99)
    assert out[0, 0] == 0 and (out[0, 1:5] == 99).all() and out[0, 5] == 5 and torch.equal(out[1], toks[1])


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
