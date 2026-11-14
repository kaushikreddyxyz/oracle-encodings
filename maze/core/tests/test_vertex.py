#!/usr/bin/env python3
"""Checks for the wall-vertex (dual-graph) codes, maze/core/codes_vertex.py.

  1. coverage / index sanity: the four segments of an interior vertex are
     distinct free connections; every free connection is stored once or twice,
     no constant connection is ever stored.  The segment index formulas are
     cross-checked against MazeFeatureComputer._connection_feature_index and,
     on real DFS mazes, against a brute-force pattern computed in a Python loop
     from cell coordinates.
  2. decodability: rows() -> decode_vertex() reproduces the free connections of
     random and DFS mazes exactly, with consistent=True; a corrupted vertex
     one-hot that contradicts a shared segment is flagged inconsistent.
  3. layout: exactly (n-1)^2 ones per row, values in {0, 1}, and the documented
     index (v*16 + N*8 + E*4 + S*2 + W) is where the 1 lands.
  4. non-additivity: least squares from [signed free connections, 1] leaves a
     large residual for VertexCode (R^2 < 0.9) while EdgeCode is fit exactly.
     "A corner is not a wall."
  5. DegreeCode is a valid one-hot of the same weight but UNDERSPECIFIED: two
     different mazes share its rows.
  6. interface: features() timing, meta round trip, make_code(), GraftInjector.
"""

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))
import maze._pathsetup  # noqa: E402,F401  (cross-experiment sys.path shim)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from maze_dataset import MazeDataset, MazeDatasetConfig  # noqa: E402
from maze_dataset.generation.generators import LatticeMazeGenerators  # noqa: E402

from maze.core.codes_vertex import (  # noqa: E402
    DegreeCode, VertexCode, decode_vertex, vertex_segments,
)
from maze.core.graft import EdgeCode, GraftInjector, active_mask, make_code  # noqa: E402
from maze.core.injection import MazeFeatureComputer  # noqa: E402
from maze.core.onion import free_mask  # noqa: E402

GRIDS = (5, 6, 7, 10)
torch.manual_seed(0)


# ---- helpers ---------------------------------------------------------------------------
def random_conn(n: int, batch: int, gen: torch.Generator) -> torch.Tensor:
    """[batch, 2n^2] bool: random free connections, constant features False."""
    c = torch.rand(batch, 2 * n * n, generator=gen) < 0.5
    c[:, ~free_mask(n)] = False
    return c


def dfs_conns(n: int, k: int = 5) -> torch.Tensor:
    """[k, 2n^2] bool from real DFS (spanning-tree) mazes."""
    cfg = MazeDatasetConfig(name="vtest", grid_n=n, n_mazes=k,
                            maze_ctor=LatticeMazeGenerators.gen_dfs, seed=1234)
    ds = MazeDataset.from_config(cfg, do_generate=True, load_local=False,
                                 save_local=False, gen_parallel=False)
    return torch.stack([torch.from_numpy(m.connection_list.flatten()) for m in ds])


def feature_index(n: int, a, b) -> int:
    """injection.py's own index formula, without needing a tokenizer."""
    fake = MazeFeatureComputer.__new__(MazeFeatureComputer)
    fake.grid_n = n
    return MazeFeatureComputer._connection_feature_index(fake, a, b)


def brute_patterns(cl: np.ndarray, n: int) -> list[int]:
    """Python-loop ground truth: wall pattern N*8+E*4+S*2+W at each interior vertex,
    read straight off the [2, n, n] connection list by cell coordinates."""
    def is_open(a, b):
        (r1, c1), (r2, c2) = a, b
        if r1 == r2 and abs(c1 - c2) == 1:
            return bool(cl[1, r1, min(c1, c2)])      # right connection
        if c1 == c2 and abs(r1 - r2) == 1:
            return bool(cl[0, min(r1, r2), c1])      # down connection
        raise AssertionError((a, b))

    out = []
    for i in range(1, n):
        for j in range(1, n):
            north = not is_open((i - 1, j - 1), (i - 1, j))
            east = not is_open((i - 1, j), (i, j))
            south = not is_open((i, j - 1), (i, j))
            west = not is_open((i - 1, j - 1), (i, j - 1))
            out.append(north * 8 + east * 4 + south * 2 + west * 1)
    return out


summary = []

# ---- 1. coverage, segment indices, brute-force patterns ---------------------------------
for n in GRIDS:
    seg = vertex_segments(n)
    free = free_mask(n)
    assert seg.shape == ((n - 1) ** 2, 4) and seg.dtype == torch.long, seg.shape
    for row in seg:
        assert len(set(row.tolist())) == 4, ("repeated segment", n, row)
        assert free[row].all(), ("constant feature stored", n, row)
    counts = torch.zeros(2 * n * n).index_add_(0, seg.reshape(-1), torch.ones(seg.numel()))
    assert (counts[free] >= 1).all(), "a free connection is stored nowhere"
    assert (counts[free] <= 2).all(), "a free connection is stored more than twice"
    assert (counts[~free] == 0).all(), "a constant connection is stored"
    assert int(counts.sum()) == 4 * (n - 1) ** 2

    # index formulas vs injection.py, per vertex, in a Python loop
    for i in range(1, n):
        for j in range(1, n):
            v = (i - 1) * (n - 1) + (j - 1)
            want = [feature_index(n, (i - 1, j - 1), (i - 1, j)),   # N
                    feature_index(n, (i - 1, j), (i, j)),           # E
                    feature_index(n, (i, j - 1), (i, j)),           # S
                    feature_index(n, (i - 1, j - 1), (i, j - 1))]   # W
            assert seg[v].tolist() == want, (n, i, j, seg[v].tolist(), want)

    # brute force on real mazes: flatten() layout and the per-vertex pattern
    conn = dfs_conns(n, 3)
    code = VertexCode(n)
    rows = code.rows(conn)
    for b in range(conn.shape[0]):
        cl = conn[b].reshape(2, n, n).numpy()
        assert not conn[b][~free].any(), "DFS maze has an open constant connection"
        assert int(conn[b].sum()) == n * n - 1, "DFS maze is not a spanning tree"
        for r in range(n):
            for c in range(n - 1):
                assert conn[b][feature_index(n, (r, c), (r, c + 1))] == bool(cl[1, r, c])
        for r in range(n - 1):
            for c in range(n):
                assert conn[b][feature_index(n, (r, c), (r + 1, c))] == bool(cl[0, r, c])
        pats = brute_patterns(cl, n)
        got = rows[b].reshape((n - 1) ** 2, 16).argmax(-1).tolist()
        assert got == pats, (n, b, [(k, g, p) for k, (g, p) in enumerate(zip(got, pats)) if g != p][:5])
print("1 coverage, segment indices vs injection.py, brute-force patterns OK")


# ---- 2. decodability ---------------------------------------------------------------------
for n in GRIDS:
    code, free = VertexCode(n), free_mask(n)
    gen = torch.Generator().manual_seed(n)
    conn = torch.cat([random_conn(n, 200, gen), dfs_conns(n, 5)])
    rows = code.rows(conn)
    back, ok = decode_vertex(rows, n)
    assert ok.all(), f"n={n}: {int((~ok).sum())} rows reported inconsistent"
    assert torch.equal(back, conn), f"n={n}: decode != conn"
    assert not back[:, ~free].any(), "decoded a constant feature as open"

    # corrupt one vertex's one-hot on a SHARED segment -> must be inconsistent.
    # vertex (1,1) is v=0; its S segment (bit 2) is also the N segment of vertex (2,1).
    seg = vertex_segments(n)
    shared = int(seg[0, 2])
    assert (seg == shared).sum() == 2, "expected S of vertex (1,1) to be shared"
    bad = rows[:8].clone()
    blocks = bad.reshape(8, (n - 1) ** 2, 16)
    old = blocks[:, 0].argmax(-1)
    new = old ^ 2  # flip the S wall bit only
    blocks[:, 0] = 0.0
    blocks[torch.arange(8), 0, new] = 1.0
    _, ok_bad = decode_vertex(bad, n)
    assert not ok_bad.any(), f"n={n}: corrupted rows not flagged"

    # a block that is not one-hot at all is also rejected
    junk = rows[:4].clone()
    junk[:, 0] = 1.0  # now two ones (or one, if pattern 0 was already set) in block 0
    junk[:, 3] = 1.0
    _, ok_junk = decode_vertex(junk, n)
    assert not ok_junk.any(), f"n={n}: non-one-hot block accepted"
print("2 decodability on random + DFS mazes, corruption detected OK")


# ---- 3. row layout ------------------------------------------------------------------------
for n in GRIDS:
    gen = torch.Generator().manual_seed(100 + n)
    conn = random_conn(n, 32, gen)
    for code, width in ((VertexCode(n), 16), (DegreeCode(n), 5)):
        rows = code.rows(conn)
        assert rows.shape == (32, width * (n - 1) ** 2) == (32, code.n_features)
        assert rows.dtype == torch.float32
        assert ((rows == 0) | (rows == 1)).all(), "rows are not 0/1"
        assert (rows.sum(1) == (n - 1) ** 2).all(), "wrong number of active features"
        assert (rows.reshape(32, (n - 1) ** 2, width).sum(-1) == 1).all(), "block not one-hot"
        assert abs(code.expected_sq_norm() - (n - 1) ** 2) < 1e-9
        assert abs(float(rows.pow(2).sum(1).mean()) - code.expected_sq_norm()) < 1e-6
    # documented index: v*16 + N*8 + E*4 + S*2 + W
    vrows, seg = VertexCode(n).rows(conn), vertex_segments(n)
    walls = ~conn
    for v in (0, (n - 1) ** 2 // 2, (n - 1) ** 2 - 1):
        pat = (walls[:, seg[v]].long() * torch.tensor([8, 4, 2, 1])).sum(-1)
        assert (vrows[torch.arange(32), v * 16 + pat] == 1).all(), (n, v)
        deg = walls[:, seg[v]].long().sum(-1)
        assert (DegreeCode(n).rows(conn)[torch.arange(32), v * 5 + deg] == 1).all(), (n, v)
print("3 one-hot layout and documented feature index OK")


# ---- 4. non-additivity ---------------------------------------------------------------------
def r2_linear_fit(conn: torch.Tensor, y: torch.Tensor, n: int) -> float:
    """R^2 of the best least-squares fit of y from [signed free connections, 1]."""
    x = (conn[:, free_mask(n)].double() * 2 - 1)
    x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=torch.float64)], dim=1)
    y = y.double()
    beta = torch.linalg.lstsq(x, y, driver="gelsd").solution
    ss_res = (y - x @ beta).pow(2).sum()
    ss_tot = (y - y.mean(0, keepdim=True)).pow(2).sum()
    return float(1 - ss_res / ss_tot)


for n in GRIDS:
    gen = torch.Generator().manual_seed(7 * n)
    conn = random_conn(n, 2000, gen)
    r2_v = r2_linear_fit(conn, VertexCode(n).rows(conn), n)
    r2_d = r2_linear_fit(conn, DegreeCode(n).rows(conn), n)
    r2_e = r2_linear_fit(conn, EdgeCode(n).rows(conn), n)
    assert r2_v < 0.9, f"n={n}: VertexCode is nearly affine in the wall bits (R^2={r2_v:.3f})"
    assert r2_e > 1 - 1e-8, f"n={n}: EdgeCode should be fit exactly (R^2={r2_e:.6f})"
    summary.append((n, r2_v, r2_d, r2_e))
print("4 non-additivity (vertex) and exactness (edge) OK")


# ---- 5. DegreeCode is underspecified --------------------------------------------------------
for n in GRIDS:
    seg, counts = vertex_segments(n), torch.zeros(2 * n * n)
    counts.index_add_(0, seg.reshape(-1), torch.ones(seg.numel()))
    north, west = int(seg[0, 0]), int(seg[0, 3])   # vertex (1,1): N and W are in no other vertex
    assert counts[north] == 1 and counts[west] == 1, (n, counts[north], counts[west])
    gen = torch.Generator().manual_seed(500 + n)
    a = random_conn(n, 4, gen)
    a[:, north], a[:, west] = False, True          # N walled, W open
    b = a.clone()
    b[:, north], b[:, west] = True, False          # rotate that corner's walls
    assert not torch.equal(a, b)
    assert torch.equal(DegreeCode(n).rows(a), DegreeCode(n).rows(b)), \
        "DegreeCode was expected to be blind to which walls meet at a corner"
    assert not torch.equal(VertexCode(n).rows(a), VertexCode(n).rows(b)), \
        "VertexCode must distinguish rotated corners"
    # and the vertex code still decodes both
    for m in (a, b):
        back, ok = decode_vertex(VertexCode(n).rows(m), n)
        assert ok.all() and torch.equal(back, m)
print("5 DegreeCode one-hot but underspecified (two mazes, identical rows) OK")


# ---- 6. interface: features(), meta round trip, make_code, injector ---------------------------
for n in GRIDS:
    for name, cls, width in (("vertex", VertexCode, 16), ("degree", DegreeCode, 5)):
        code = make_code(name, n)
        assert isinstance(code, cls) and code.name == name and code.grid_n == n
        assert code.n_features == width * (n - 1) ** 2 and not code.dynamic
        again = type(code).from_meta(n, code.meta())
        assert type(again) is type(code) and again.n_features == code.n_features
        assert torch.equal(again.seg, code.seg) and again.meta() == code.meta()

        gen = torch.Generator().manual_seed(n)
        conn = random_conn(n, 2, gen)
        tokens = torch.zeros(2, 10, dtype=torch.long)
        attn = torch.ones(2, 10, dtype=torch.long)
        attn[1, 8:] = 0
        end = torch.tensor([3, 5])
        f = code.features(tokens, attn, conn, end)
        assert f.shape == (2, 10, code.n_features)
        act = active_mask(end, 10, attn)
        rows = code.rows(conn)
        for b in range(2):
            for p in range(10):
                want = rows[b] if act[b, p] else torch.zeros(code.n_features)
                assert torch.equal(f[b, p], want), (name, n, b, p)
        assert torch.equal(f.sum(1), act.float().sum(1)[:, None] * rows)
        # static-code state/step contract
        st = code.init_state(tokens, attn, conn, end)
        out, st2 = code.step(st, tokens[:, 0])
        assert torch.equal(out, rows) and torch.equal(st2, rows)

        inj = GraftInjector(code, 32, norm_match=code.expected_sq_norm())
        d = inj.delta(f)
        assert d.shape == (2, 10, 32)
        assert abs(float(inj.gain) - (2 * n * n / (n - 1) ** 2) ** 0.5) < 1e-6
        # norm matching: injected vector's expected squared norm tracks the edge graft's
        assert abs(float((inj.gain * rows).pow(2).sum(1).mean()) - 2 * n * n) < 1e-3
        ck = inj.state_for_ckpt()
        assert ck["code"] == name and ck["grid_n"] == n and ck["W"].shape == (code.n_features, 32)
print("6 features() timing, meta round trip, make_code, injector norm matching OK")


print()
for n, r2_v, r2_d, r2_e in summary:
    c = VertexCode(n)
    print(f"n={n:2d} vertices={c.v_count:3d} free_conn={int(free_mask(n).sum()):3d} "
          f"vertex F={c.n_features:4d} degree F={DegreeCode(n).n_features:3d} "
          f"ones/row={(n - 1) ** 2:3d}  linear-fit R^2: vertex={r2_v:.3f} "
          f"degree={r2_d:.3f} edge={r2_e:.6f}")
print("ALL OK")
