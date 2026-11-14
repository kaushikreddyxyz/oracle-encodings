#!/usr/bin/env python3
"""Checks for the axis-split degree codes, maze/core/codes_axdeg.py.

  1. index / brute force: on real DFS mazes, the per-cell horizontal and
     vertical open-neighbour counts computed by a plain Python loop straight
     off maze.connection_list [2, n, n] equal rows() + 1 ("axdeg") and the
     per-block argmax ("axdeg1h"), exactly.  The conn-index formulas are also
     cross-checked against MazeFeatureComputer._connection_feature_index.
  2. completeness: decode_axdeg(rows + 1) reproduces the free connections of
     300 arbitrary (non-tree) wall assignments and of DFS mazes exactly with
     consistent=True; rank(M[free rows]) == 2n(n-1); corrupting one count by
     +1 is always detected.
  3. linearity: rows_axdeg(conn) == conn.float() @ M + bias exactly, and the
     least-squares fit from [signed free wall bits, 1] has R^2 == 1 for axdeg
     but clearly < 1 for axdeg1h (a one-hot count is quadratic in the bits).
  4. geometry of the effective edge directions W_eff = M @ W for a fixed-W
     injector: cos ~ 1/2 between walls adjacent along a row/column, ~0 across
     axes and at distance; and the sampled ||rows||^2 matches expected_sq_norm.
  5. values / layout / interface: {-1, 0, +1}, boundary cells never +1, one-hot
     weight 2n^2, features() timing, meta round trip, make_code, injector gain.
  6. contrast: plain per-cell TOTAL degree (h + v) is NOT complete -- two
     different mazes share every total degree but differ in the axis split.
"""

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))
import maze._pathsetup  # noqa: E402,F401  (cross-experiment sys.path shim)

import inspect  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from maze_dataset import MazeDataset, MazeDatasetConfig  # noqa: E402
from maze_dataset.generation.generators import LatticeMazeGenerators  # noqa: E402

import maze.core.codes_axdeg  # noqa: E402,F401  (registers axdeg / axdeg1h -- see NOTE at the end)
from maze.core import graft as graft_mod  # noqa: E402
from maze.core.codes_axdeg import (  # noqa: E402
    AxisDegreeCode, AxisDegreeOneHotCode, axis_counts, axis_slots,
    counts_from_onehot, decode_axdeg, incidence_matrix,
)
from maze.core.graft import GraftInjector, active_mask, make_code  # noqa: E402
from maze.core.injection import MazeFeatureComputer  # noqa: E402
from maze.core.onion import free_mask  # noqa: E402

GRIDS = (5, 6, 7, 10)
torch.manual_seed(0)


# ---- helpers ---------------------------------------------------------------------------
def random_conn(n: int, batch: int, gen: torch.Generator) -> torch.Tensor:
    """[batch, 2n^2] bool: arbitrary free connections (NOT spanning trees), constants False."""
    c = torch.rand(batch, 2 * n * n, generator=gen) < 0.5
    c[:, ~free_mask(n)] = False
    return c


def dfs_mazes(n: int, k: int = 5, seed: int = 1234):
    """-> (conn [k, 2n^2] bool, [connection_list [2, n, n] numpy, ...]) from real DFS mazes."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = MazeDatasetConfig(name="axtest", grid_n=n, n_mazes=k,
                                maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed)
        ds = MazeDataset.from_config(cfg, do_generate=True, load_local=False,
                                     save_local=False, gen_parallel=False, verbose=False)
    cls = [m.connection_list for m in ds]
    return torch.stack([torch.from_numpy(c.flatten()) for c in cls]), cls


def feature_index(n: int, a, b) -> int:
    """injection.py's own index formula, without needing a tokenizer."""
    fake = MazeFeatureComputer.__new__(MazeFeatureComputer)
    fake.grid_n = n
    return MazeFeatureComputer._connection_feature_index(fake, a, b)


def brute_axis_counts(cl: np.ndarray, n: int):
    """Python-loop ground truth, independent of the vectorised code.

    cl: connection_list [2, n, n], dim 0 = DOWN connection of (r, c), dim 1 = RIGHT.
    -> (h, v), n x n lists of ints: open left/right neighbours, open up/down neighbours.
    """
    def is_open(a, b):
        (r1, c1), (r2, c2) = a, b
        if r1 == r2 and abs(c1 - c2) == 1:
            return bool(cl[1, r1, min(c1, c2)])      # right connection of the left cell
        if c1 == c2 and abs(r1 - r2) == 1:
            return bool(cl[0, min(r1, r2), c1])      # down connection of the upper cell
        raise AssertionError((a, b))

    h = [[0] * n for _ in range(n)]
    v = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            for dr, dc in ((0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and is_open((r, c), (rr, cc)):
                    h[r][c] += 1
            for dr, dc in ((-1, 0), (1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and is_open((r, c), (rr, cc)):
                    v[r][c] += 1
    return h, v


def r2_linear_fit(conn: torch.Tensor, y: torch.Tensor, n: int) -> float:
    """R^2 of the best least-squares fit of y from [signed free connections, 1]."""
    x = conn[:, free_mask(n)].double() * 2 - 1
    x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=torch.float64)], dim=1)
    y = y.double()
    beta = torch.linalg.lstsq(x, y, driver="gelsd").solution
    ss_res = (y - x @ beta).pow(2).sum()
    ss_tot = (y - y.mean(0, keepdim=True)).pow(2).sum()
    return float(1 - ss_res / ss_tot)


def cos_pairs(W: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """mean cosine over [K, 2] index pairs into the row-space of W."""
    u = W / W.norm(dim=-1, keepdim=True)
    return (u[pairs[:, 0]] * u[pairs[:, 1]]).sum(-1)


summary = []

# ---- 1. brute-force counts on real DFS mazes ---------------------------------------------
for n in GRIDS:
    # conn index formulas (the layout every later step assumes)
    idx, w = axis_slots(n)
    for r in range(n):
        for c in range(n):
            f_h, f_v = 0 * n * n + r * n + c, 1 * n * n + r * n + c
            want_h = [feature_index(n, (r, c), (r, c - 1)) if c > 0 else None,
                      feature_index(n, (r, c), (r, c + 1)) if c < n - 1 else None]
            want_v = [feature_index(n, (r, c), (r - 1, c)) if r > 0 else None,
                      feature_index(n, (r, c), (r + 1, c)) if r < n - 1 else None]
            for f, want in ((f_h, want_h), (f_v, want_v)):
                for s in range(2):
                    if want[s] is None:
                        assert w[f, s] == 0.0, ("out-of-grid slot must be weight 0", n, r, c, s)
                    else:
                        assert w[f, s] == 1.0 and int(idx[f, s]) == want[s], (n, r, c, f, s)
            # no constant connection is ever read
            for s in range(2):
                assert w[f_h, s] == 0.0 or free_mask(n)[int(idx[f_h, s])]
                assert w[f_v, s] == 0.0 or free_mask(n)[int(idx[f_v, s])]

    conn, cls = dfs_mazes(n, 20)
    code, code1h = AxisDegreeCode(n), AxisDegreeOneHotCode(n)
    rows, rows1h = code.rows(conn), code1h.rows(conn)
    got = (rows + 1.0).reshape(-1, 2, n, n)                 # feature axis 0 = HORIZONTAL
    got1h = counts_from_onehot(rows1h).reshape(-1, 2, n, n)
    for b in range(conn.shape[0]):
        assert int(conn[b].sum()) == n * n - 1, "DFS maze is not a spanning tree"
        assert not conn[b][~free_mask(n)].any(), "DFS maze has an open constant connection"
        assert torch.equal(conn[b].reshape(2, n, n), torch.from_numpy(cls[b])), "flatten layout"
        h, v = brute_axis_counts(cls[b], n)
        want = torch.tensor([h, v], dtype=torch.float32)    # [2 (h, v), n, n]
        assert torch.equal(got[b], want), (n, b, "axdeg counts != brute force")
        assert torch.equal(got1h[b].float(), want), (n, b, "axdeg1h counts != brute force")
        # sanity on the brute force itself: total degree == 2 * #edges
        assert sum(map(sum, h)) + sum(map(sum, v)) == 2 * (n * n - 1)
print("1 conn-index formulas vs injection.py + brute-force h/v on 20 DFS mazes per n OK")


# ---- 2. completeness -----------------------------------------------------------------------
for n in GRIDS:
    code, code1h = AxisDegreeCode(n), AxisDegreeOneHotCode(n)
    gen = torch.Generator().manual_seed(n)
    conn = torch.cat([random_conn(n, 300, gen), dfs_mazes(n, 10, seed=99)[0]])
    counts = code.rows(conn) + 1.0
    back, ok = decode_axdeg(counts, n)
    assert ok.all(), f"n={n}: {int((~ok).sum())} rows reported inconsistent"
    assert torch.equal(back, conn), f"n={n}: decode != conn"
    assert not back[:, ~free_mask(n)].any(), "decoded a constant feature as open"
    back1h, ok1h = decode_axdeg(counts_from_onehot(code1h.rows(conn)), n)
    assert ok1h.all() and torch.equal(back1h, conn), f"n={n}: axdeg1h round trip failed"
    # integer/long input works too
    back_i, ok_i = decode_axdeg(counts.long(), n)
    assert ok_i.all() and torch.equal(back_i, conn)

    # rank: the 2n(n-1) free wall bits are independently visible in the counts
    M = incidence_matrix(n)
    assert torch.equal(M, code.M) and torch.equal(M, code1h.M)
    rank = int(torch.linalg.matrix_rank(M[free_mask(n)]))
    assert rank == 2 * n * (n - 1), f"n={n}: rank {rank} != {2 * n * (n - 1)}"
    assert float(M[~free_mask(n)].abs().sum()) == 0.0, "a constant connection feeds a count"
    assert (M[free_mask(n)].sum(1) == 2).all(), "a free wall does not have exactly 2 endpoints"

    # corrupt exactly one count by +1 -> detected (inconsistent, or a different maze)
    g2 = torch.Generator().manual_seed(1000 + n)
    bad = counts.clone()
    who = torch.randint(0, 2 * n * n, (bad.shape[0],), generator=g2)
    bad[torch.arange(bad.shape[0]), who] += 1
    back_bad, ok_bad = decode_axdeg(bad, n)
    assert not ok_bad.any(), f"n={n}: {int(ok_bad.sum())} corrupted rows accepted as consistent"
    # a non-integer count is rejected as well
    _, ok_frac = decode_axdeg(counts[:8] + 0.5, n)
    assert not ok_frac.any(), f"n={n}: fractional counts accepted"
print("2 exact decode on 300 random + 10 DFS mazes per n, rank 2n(n-1), corruption detected OK")


# ---- 3. linearity identities -----------------------------------------------------------------
for n in GRIDS:
    code, code1h = AxisDegreeCode(n), AxisDegreeOneHotCode(n)
    gen = torch.Generator().manual_seed(7 * n)
    conn = random_conn(n, 2000, gen)
    rows = code.rows(conn)
    assert torch.equal(rows, conn.float() @ code.M + code.bias), f"n={n}: rows != conn @ M + bias"
    assert torch.equal(axis_counts(conn, n), conn.float() @ code.M), "counts != conn @ M"
    r2_a = r2_linear_fit(conn, rows, n)
    r2_1h = r2_linear_fit(conn, code1h.rows(conn), n)
    assert r2_a > 1 - 1e-9, f"n={n}: axdeg should be an exact affine function (R^2={r2_a:.9f})"
    assert r2_1h < 0.9, f"n={n}: axdeg1h is nearly affine in the wall bits (R^2={r2_1h:.3f})"
    summary.append({"n": n, "r2_axdeg": r2_a, "r2_axdeg1h": r2_1h})
print("3 axdeg affine (R^2 = 1) and axdeg1h non-affine (R^2 < 0.9) OK")


# ---- 4. effective edge directions of a fixed-W injector ----------------------------------------
for k, n in enumerate(GRIDS):
    code = AxisDegreeCode(n)
    inj = GraftInjector(code, 256, seed=0, norm_match=code.expected_sq_norm())
    W_eff = code.M @ inj.W.detach()          # [2n^2 walls, d]: wall -> sum of its 2 cell directions
    nn = n * n

    adj, cross, far = [], [], []
    for r in range(n):
        for c in range(n - 2):               # right(r, c) and right(r, c+1) share cell (r, c+1)
            adj.append((nn + r * n + c, nn + r * n + c + 1))
    for c in range(n):
        for r in range(n - 2):               # down(r, c) and down(r+1, c) share cell (r+1, c)
            adj.append((r * n + c, (r + 1) * n + c))
    for r in range(n - 1):
        for c in range(n - 1):               # down(r, c) and right(r, c) share CELL (r, c), not a feature
            cross.append((r * n + c, nn + r * n + c))
    free_ids = torch.arange(2 * nn)[free_mask(n)].tolist()
    g = torch.Generator().manual_seed(4 * n)
    while len(far) < 400:
        a, b = (free_ids[int(torch.randint(0, len(free_ids), (1,), generator=g))] for _ in range(2))
        if a != b and float((code.M[a] * code.M[b]).sum()) == 0.0:   # share no (cell, axis)
            far.append((a, b))

    c_adj = float(cos_pairs(W_eff, torch.tensor(adj)).mean())
    c_cross = float(cos_pairs(W_eff, torch.tensor(cross)).mean())
    c_far = float(cos_pairs(W_eff, torch.tensor(far)).mean())
    assert abs(c_adj - 0.5) < 0.08, f"n={n}: adjacent-wall cosine {c_adj:.3f} != ~0.5"
    assert abs(c_cross) < 0.05, f"n={n}: cross-axis cosine {c_cross:.3f} != ~0"
    assert abs(c_far) < 0.05, f"n={n}: distant-wall cosine {c_far:.3f} != ~0"

    # sampled norm vs expected_sq_norm (a DIFFERENT seed from the constructor's estimate)
    conn2k, _ = dfs_mazes(n, 2000, seed=777)
    got = float(code.rows(conn2k).pow(2).sum(1).mean())
    rel = abs(got - code.expected_sq_norm()) / code.expected_sq_norm()
    assert rel < 0.05, f"n={n}: sampled ||rows||^2 {got:.2f} vs expected {code.expected_sq_norm():.2f}"
    assert abs(float((inj.gain * code.rows(conn2k)).pow(2).sum(1).mean()) - 2 * nn) / (2 * nn) < 0.05
    summary[k].update({"cos_adj": c_adj, "cos_cross": c_cross, "cos_far": c_far,
                       "sq_norm": code.expected_sq_norm(), "sq_norm_sampled": got,
                       "gain": float(inj.gain)})
print("4 W_eff cosines (adjacent ~0.5, cross-axis ~0, distant ~0) and norm matching OK")


# ---- 5. values, layout, interface ---------------------------------------------------------------
for n in GRIDS:
    code, code1h = AxisDegreeCode(n), AxisDegreeOneHotCode(n)
    gen = torch.Generator().manual_seed(100 + n)
    conn = torch.cat([random_conn(n, 200, gen), dfs_mazes(n, 20, seed=5)[0]])
    b = conn.shape[0]
    rows, rows1h = code.rows(conn), code1h.rows(conn)

    assert rows.shape == (b, 2 * n * n) == (b, code.n_features) and rows.dtype == torch.float32
    assert ((rows == -1) | (rows == 0) | (rows == 1)).all(), "axdeg value outside {-1, 0, 1}"
    grid = rows.reshape(b, 2, n, n)
    for c in (0, n - 1):                       # one horizontal neighbour only
        assert ((grid[:, 0, :, c] == -1) | (grid[:, 0, :, c] == 0)).all(), f"h at column {c}"
    for r in (0, n - 1):                       # one vertical neighbour only
        assert ((grid[:, 1, r] == -1) | (grid[:, 1, r] == 0)).all(), f"v at row {r}"
    inner = grid[:, :, 1:n - 1, 1:n - 1]
    assert (inner == 1).any() and (inner == 0).any() and (inner == -1).any(), "interior is degenerate"

    assert rows1h.shape == (b, 6 * n * n) == (b, code1h.n_features)
    assert ((rows1h == 0) | (rows1h == 1)).all(), "axdeg1h rows are not 0/1"
    assert (rows1h.sum(1) == 2 * n * n).all(), "axdeg1h weight != 2n^2"
    assert (rows1h.reshape(b, 2 * n * n, 3).sum(-1) == 1).all(), "axdeg1h block not one-hot"
    assert abs(code1h.expected_sq_norm() - 2 * n * n) < 1e-9
    assert abs(float(rows1h.pow(2).sum(1).mean()) - code1h.expected_sq_norm()) < 1e-6
    # documented one-hot index (axis * n^2 + r * n + c) * 3 + count
    counts = (rows + 1).long()
    for f in (0, n * n // 2, 2 * n * n - 1):
        assert (rows1h[torch.arange(b), f * 3 + counts[:, f]] == 1).all(), (n, f)

    for name, cls, F in (("axdeg", AxisDegreeCode, 2 * n * n), ("axdeg1h", AxisDegreeOneHotCode, 6 * n * n)):
        c = make_code(name, n)
        assert isinstance(c, cls) and c.name == name and c.grid_n == n and not c.dynamic
        assert c.n_features == F
        again = type(c).from_meta(n, c.meta())
        assert type(again) is type(c) and again.n_features == F and again.meta() == c.meta()
        assert abs(again.expected_sq_norm() - c.expected_sq_norm()) < 1e-9
        assert torch.equal(again.rows(conn[:8]), c.rows(conn[:8]))

        tokens = torch.zeros(2, 10, dtype=torch.long)
        attn = torch.ones(2, 10, dtype=torch.long)
        attn[1, 8:] = 0
        end = torch.tensor([3, 5])
        f = c.features(tokens, attn, conn[:2], end)
        assert f.shape == (2, 10, F)
        act, rr = active_mask(end, 10, attn), c.rows(conn[:2])
        for bb in range(2):
            for p in range(10):
                want = rr[bb] if act[bb, p] else torch.zeros(F)
                assert torch.equal(f[bb, p], want), (name, n, bb, p)
        st = c.init_state(tokens, attn, conn[:2], end)          # static-code state: {"rows", "active"} (graft.py start rule)
        out, st2 = c.step(st, tokens[:, 0])
        assert torch.equal(out, rr) and torch.equal(st2["rows"], rr)   # adjlist_end: every generated position is active

        inj = GraftInjector(c, 32, norm_match=c.expected_sq_norm())
        assert inj.delta(f).shape == (2, 10, 32)
        assert abs(float(inj.gain) - (2 * n * n / c.expected_sq_norm()) ** 0.5) < 1e-6
        ck = inj.state_for_ckpt()
        assert ck["code"] == name and ck["grid_n"] == n and ck["W"].shape == (F, 32)
        assert make_code(ck["code"], ck["grid_n"], **ck["code_meta"]).n_features == F
    assert abs(AxisDegreeOneHotCode(n).expected_sq_norm() - 2 * n * n) < 1e-9
    assert abs(float(GraftInjector(code1h, 8, norm_match=code1h.expected_sq_norm()).gain) - 1.0) < 1e-6
print("5 values {-1,0,1}, boundary cells never +1, one-hot weight, features(), meta, make_code OK")


# ---- 6. total degree alone is NOT complete ------------------------------------------------------
for n in GRIDS:
    code = AxisDegreeCode(n)
    gen = torch.Generator().manual_seed(2000 + n)
    a = random_conn(n, 8, gen)
    nn = n * n
    block = [0 * n + 0, 0 * n + 1,           # down(0,0), down(0,1)
             nn + 0 * n + 0, nn + 1 * n + 0]  # right(0,0), right(1,0)
    a[:, block] = False
    b = a.clone()
    a[:, [nn + 0 * n + 0, nn + 1 * n + 0]] = True   # two horizontal edges in the 2x2 block
    b[:, [0 * n + 0, 0 * n + 1]] = True             # two vertical edges instead
    assert not torch.equal(a, b)
    assert int(a.sum()) == int(b.sum())
    ta = code.counts(a).reshape(-1, 2, nn).sum(1)   # per-cell TOTAL degree h + v
    tb = code.counts(b).reshape(-1, 2, nn).sum(1)
    assert torch.equal(ta, tb), "expected the swap to preserve every cell's total degree"
    assert not torch.equal(code.rows(a), code.rows(b)), "axdeg must see the axis swap"
    assert not torch.equal(AxisDegreeOneHotCode(n).rows(a), AxisDegreeOneHotCode(n).rows(b))
    for m in (a, b):                                 # and both still decode exactly
        back, ok = decode_axdeg(code.rows(m) + 1, n)
        assert ok.all() and torch.equal(back, m)
print("6 total-degree (h+v) is underspecified; the axis split distinguishes the two mazes OK")


# ---- summary --------------------------------------------------------------------------------------
print()
for s in summary:
    n = s["n"]
    print(f"n={n:2d} free_conn={2 * n * (n - 1):3d} axdeg F={2 * n * n:3d} axdeg1h F={6 * n * n:4d} "
          f"E||rows||^2={s['sq_norm']:7.3f} (sampled {s['sq_norm_sampled']:7.3f}, gain {s['gain']:.3f}) "
          f"R^2 axdeg={s['r2_axdeg']:.6f} axdeg1h={s['r2_axdeg1h']:.3f}  "
          f"cos: adj={s['cos_adj']:+.3f} cross-axis={s['cos_cross']:+.3f} far={s['cos_far']:+.3f}")

src = inspect.getsource(graft_mod._import_codes)
if "maze.core.codes_axdeg" not in src:
    print("\nNOTE graft.py::_import_codes() does not import maze.core.codes_axdeg, so "
          "make_code('axdeg'|'axdeg1h') only works after this module has been imported. "
          "Add 'maze.core.codes_axdeg' to its module tuple (one line).")
print("ALL OK")
