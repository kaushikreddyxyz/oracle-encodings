#!/usr/bin/env python3
"""Checks for the onion tile code (maze/core/onion.py). Pure torch, no model.

  1. coverage: checker stores every free connection exactly once, full twice;
     all four depths are used and balanced across down/right connections
  2. one tile = 16 distinct, evenly spaced magnitudes (base 2); layer k changes
     sign 2**(k+1) - 1 times along the line (only layer 0 is a threshold)
  3. ||C||_F^2 = 2n^2 (injection strength matched to the edge code)
  4. `signed_conn @ W_inj` == sum_t m_t V_t, and W_inj has rank n_dirs < 2n(n-1)
  5. the code is injective on spanning trees' free connections (no two sampled
     mazes share an injected vector)
  6. gradients reach V only, and only when trainable
"""

import itertools
import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))

import torch

from maze.core.onion import (
    OnionInjector, free_mask, min_depth, onion_code, onion_tiles,
)

for n in (5, 6, 7, 10):
    free = free_mask(n)
    is_right = torch.arange(2 * n * n) >= n * n
    for layout, copies in (("checker", 1), ("full", 2)):
        C, depth = onion_code(n, layout)
        stored = (depth >= 0).sum(0)
        assert (stored[free] == copies).all(), (n, layout, stored[free].unique())
        assert abs(float(C.pow(2).sum()) - 2 * n * n) < 1e-3
        assert C.shape[0] == len(onion_tiles(n, layout))
        md = min_depth(depth)[free]
        counts = {
            k: (int(((md == k) & ~is_right[free]).sum()), int(((md == k) & is_right[free]).sum()))
            for k in range(4)
        }
        print(f"n={n} {layout:8s} dirs={C.shape[0]:3d} free={int(free.sum()):3d} "
              f"min-depth (down, right) per layer: {counts}")
        if layout == "checker":
            assert all(a > 0 and b > 0 for a, b in counts.values()), counts
print("1/3 coverage, depth balance, norm OK")

w = torch.tensor([2.0 ** -k for k in range(4)])
states = torch.tensor(list(itertools.product((-1.0, 1.0), repeat=4)))
m = states @ w
order = m.argsort()
gaps = m[order].diff()
assert len(m.unique()) == 16 and torch.allclose(gaps, gaps[0].expand_as(gaps))
for k in range(4):
    flips = int((states[order, k].diff() != 0).sum())
    assert flips == 2 ** (k + 1) - 1, (k, flips)
print("2 sixteen evenly spaced levels; layer k flips 2^(k+1)-1 times OK")

n, d = 6, 256
for layout in ("checker", "full"):
    inj = OnionInjector(d, n, layout=layout, seed=3)
    W = inj.W_inj
    assert W.shape == (2 * n * n, d)
    conn = torch.rand(64, 2 * n * n) < 0.5
    conn[:, ~free_mask(n)] = False
    s = conn.float() * 2 - 1
    mags = s @ inj.C.t()
    assert torch.allclose(s @ W, mags @ inj.V, atol=1e-5)
    rank = int(torch.linalg.matrix_rank(W))
    assert rank == inj.C.shape[0] < int(free_mask(n).sum()), rank
    x = s @ W
    assert len({tuple(r.tolist()) for r in (x * 1e4).round()}) == len(
        {tuple(r.tolist()) for r in conn}
    )
print("4/5 linear form, rank deficiency, injectivity OK")

for trainable in (False, True):
    inj = OnionInjector(32, 6, trainable=trainable)
    params = [p for p in inj.parameters() if p.requires_grad]
    assert len(params) == (1 if trainable else 0)
    if trainable:
        (torch.ones(2, 72) @ inj.W_inj).sum().backward()
        assert inj.V.grad is not None and inj.V.grad.abs().sum() > 0
meta = OnionInjector(32, 6).code_meta()
assert meta["code"] == "onion-checker" and meta["onion_V"].shape == (18, 32)
print("6 parameters and checkpoint metadata OK")
print("ALL OK")
