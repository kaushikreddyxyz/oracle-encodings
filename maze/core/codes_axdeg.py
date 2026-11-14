"""Axis-split degree graft codes: the maze as a per-cell (horizontal, vertical) degree.

The edge graft gives every wall its own feature.  The codes here give every
CELL two small numbers -- how many of its open connections are horizontal and
how many are vertical -- and never name a wall.  The surprise is that this is
still a COMPLETE description: the 2n^2 counts determine all 2n(n-1) free
connections exactly (`decode_axdeg`), because each row of cells is a path whose
per-cell horizontal degrees can be inverted by a triangular (alternating)
recursion, and likewise each column for vertical degrees.

Counts
------
For cell (r, c), with `conn` = connection_list.flatten() (index r*n + c is the
DOWN connection between (r, c) and (r+1, c); index n^2 + r*n + c the RIGHT
connection between (r, c) and (r, c+1); True = open; the bottom-row downs and
right-column rights are constant False and are never read here):

    h(r, c) = conn[right(r, c-1)] + conn[right(r, c)]      in {0, 1, 2}
    v(r, c) = conn[down(r-1, c)]  + conn[down(r, c)]       in {0, 1, 2}

with out-of-grid slots dropped.  So h counts open left/right neighbours and v
open up/down neighbours.  Cells in column 0 or n-1 have only one horizontal
slot, hence h in {0, 1}; cells in row 0 or n-1 have v in {0, 1}.

Completeness (why the counts are enough)
----------------------------------------
Fix a row r and write e_k = conn[right(r, k)], k = 0 .. n-2 (the n-1 free
right-connections of that row).  Then

    h(r, 0)   = e_0
    h(r, k)   = e_{k-1} + e_k        1 <= k <= n-2
    h(r, n-1) = e_{n-2}

The system is lower triangular with unit diagonal, so

    e_k = sum_{i <= k} (-1)^(k-i) h(r, i)        (alternating prefix sum)

and the last cell gives one consistency equation per row: e_{n-1} computed by
the same recursion, i.e. h(r, n-1) - e_{n-2}, must be 0.  Conveniently that is
exactly "the constant right-column connection is closed", so `decode_axdeg`
just runs the recursion over all n slots and demands the last one be 0 and all
of them be in {0, 1}.  Columns are identical with v and down-connections.
Hence: rank 2n(n-1), no free parameters left, the code specifies the maze.
(The counts live in a 2n^2-dimensional space but only a 2n(n-1)-dimensional
affine subset of it is realizable -- the code is redundant/error-detecting,
which is what `consistent` reports.)

The two codes
-------------
AxisDegreeCode ("axdeg")
    n_features = 2n^2, index = axis * n^2 + r * n + c  (axis 0 = h, 1 = v).
    VALUE = count - 1, i.e. in {-1, 0, +1}: a dead end along that axis is -1,
    a single neighbour 0, a through-passage +1.  The centring is deliberate --
    over DFS spanning trees the mean count is close to 1 (the mean degree is
    2 - 2/n^2 split over two axes), so the mean feature is close to 0 and the
    graft does not inject a large constant direction.  Same width as the edge
    graft (2n^2), same {-1, 0, +1}-ish value range, but NO feature is a wall.

AxisDegreeOneHotCode ("axdeg1h")
    n_features = 3 * 2n^2 = 6n^2, index = (axis * n^2 + r * n + c) * 3 + count,
    one-hot over the count per (cell, axis): exactly 2n^2 ones per row, so
    ||row||^2 = 2n^2 = expected_sq_norm() and GraftInjector's gain is 1.

Is this just the edge graft in disguise?  (read before using "axdeg")
---------------------------------------------------------------------
For "axdeg", YES in the linear-algebra sense, and the tests check it:

    rows(conn) = conn.float() @ M + bias,     bias = -1,

with M [2n^2, 2n^2] the 0/1 incidence matrix M[wall, feature] = 1 iff that wall
is one of the two slots of that (cell, axis).  M is exposed as an attribute.
Consequences:

  * With a FIXED random W, delta = rows @ W = conn.float() @ (M @ W) + bias @ W.
    So it IS an edge graft whose per-wall effective direction is
        W_eff[wall] = W[feature of endpoint A] + W[feature of endpoint B]
    (every free wall has exactly two endpoint cells), plus the maze-independent
    constant bias @ W = -sum_f W[f] added at every active position.  The
    directions are therefore CORRELATED, unlike the edge graft's i.i.d. ones:
    two walls adjacent along a row (or column) share one endpoint cell, so
    cos(W_eff) -> 1/2; walls meeting at a cell but on different axes use
    different features (h vs v), so cos -> 0; distant walls cos -> 0.
    That correlation structure -- "neighbouring walls point in similar
    directions" -- is the only thing a fixed-W axdeg graft adds over a fixed-W
    edge graft.  It is a tied-direction edge graft, not a new kind of content.
  * With a TRAINABLE W the maze-dependent part is not even restricted: M
    restricted to the 2n(n-1) free wall rows has full row rank, so M_free @ W
    can be made any matrix, i.e. trainable "axdeg" is the same hypothesis class
    as a trainable edge graft.  Only the constant term is tied to W (and for
    even n it is fully determined by the maze-dependent part; see the note at
    the end of this docstring).

"axdeg1h" escapes this: an indicator of a count is a product of the two wall
bits (1[count = 0] = (1-a)(1-b), 1[count = 1] = a XOR b, purely quadratic), so
the one-hot rows are NOT an affine function of the wall bits -- a least-squares
fit from [signed free wall bits, 1] leaves a large residual (R^2 ~ 0.6 at
n = 5, falling towards 0.4 for large n, test 3).  Use "axdeg1h" if the point is
"a degree is not a wall"; use "axdeg" if the point is a correlated-direction
edge graft with centred +/-1 values.

expected_sq_norm()
------------------
"axdeg1h": exactly 2n^2 (one-hot, gain 1).
"axdeg":  ||rows||^2 = #{(cell, axis) : count != 1}, whose mean over DFS
  spanning trees has no closed form (it depends on the degree distribution of
  DFS mazes, which is generator-specific), so it is ESTIMATED once from
  `_NORM_SAMPLES` = 2000 DFS mazes generated with the fixed seed
  `_NORM_SEED` inside the constructor, cached on the instance as
  `.sq_norm`, memoised module-wide in `_SQ_NORM_CACHE`, and carried in meta()
  so a checkpoint reloads the exact value used at training time.  Measured:
  22.17 (n=5), 31.60 (n=6), 42.87 (n=7), 86.79 (n=10), i.e. ~0.44 per feature
  -- about 56% of (cell, axis) pairs have exactly one open neighbour on that
  axis and so contribute 0 -- which makes GraftInjector's gain ~1.51.  The
  estimate is stable to <1.5% against an independent 2000-maze sample (test 4).
  The estimator generates mazes, and maze_dataset's
  generator calls np.random.seed() globally, so the numpy global RNG state is
  saved and restored around it.

Caveats
-------
* Boundary cells are not symmetric: column 0 / n-1 cells have h in {-1, 0} and
  row 0 / n-1 cells have v in {-1, 0} (only one slot), so those features never
  take the value +1 and carry less information.  The corner cells have both.
* rows() never reads the 2n constant connections (their columns of M are zero),
  so a maze passed with a constant feature set True is silently ignored rather
  than mis-counted.
* Value semantics: 0 means "exactly one open neighbour on this axis", which is
  the modal case, not "no evidence" (contrast the ego code, where 0 = silent).
  The graft is active at every position >= <ADJLIST_END> (complete timing), so
  there is no ambiguity in practice, but a half-masked variant of this code
  would inherit the maze_legacy "0 is graceful" caveat only by accident.
* Parity note: the null space of M_free (as a map on feature space) is spanned
  by per-row / per-column alternating patterns (dimension 2n).  The all-ones
  vector is orthogonal to it exactly when n is even, so for even n the constant
  term bias @ W is determined by the maze-dependent part M_free @ W, while for
  odd n it can be chosen freely.
"""

from __future__ import annotations

import torch

from maze.core.graft import GraftCode, register
from maze.core.onion import free_mask

N_COUNTS = 3  # a count is 0, 1 or 2
_NORM_SAMPLES = 2000
_NORM_SEED = 20260922
_SQ_NORM_CACHE: dict[tuple[int, int, int], float] = {}


def axis_slots(grid_n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (idx [2n^2, 2] long, weight [2n^2, 2] float).

    Feature f = axis * n^2 + r * n + c gathers conn[idx[f, 0]] and conn[idx[f, 1]]
    with weights weight[f, :]; a weight of 0 marks a slot that leaves the grid
    (or is one of the 2n constant connections), whose index is a harmless 0.
    Slot order: (left, right) for axis 0 = h, (up, down) for axis 1 = v.
    """
    n = int(grid_n)
    assert n >= 2, f"need grid_n >= 2, got {n}"
    nn = n * n
    r = torch.arange(n).view(-1, 1).expand(n, n).reshape(-1)  # [n^2] cell row
    c = torch.arange(n).view(1, -1).expand(n, n).reshape(-1)  # [n^2] cell col

    left_i, left_w = nn + r * n + (c - 1).clamp(min=0), (c > 0).float()
    right_i, right_w = nn + r * n + c, (c < n - 1).float()
    up_i, up_w = (r - 1).clamp(min=0) * n + c, (r > 0).float()
    down_i, down_w = r * n + c, (r < n - 1).float()

    idx = torch.cat([torch.stack([left_i, right_i], dim=1),
                     torch.stack([up_i, down_i], dim=1)], dim=0).long()     # [2n^2, 2]
    w = torch.cat([torch.stack([left_w, right_w], dim=1),
                   torch.stack([up_w, down_w], dim=1)], dim=0).float()
    return idx, w


def incidence_matrix(grid_n: int) -> torch.Tensor:
    """-> M [2n^2, 2n^2] float: M[wall, feature] = 1 iff `wall` is a slot of `feature`.

    counts(conn) == conn.float() @ M exactly; the 2n constant-connection rows of
    M are zero and every free wall row has exactly two ones (its two endpoint
    cells, same axis).
    """
    n = int(grid_n)
    idx, w = axis_slots(n)
    M = torch.zeros(2 * n * n, 2 * n * n)
    feat = torch.arange(2 * n * n)
    for s in range(2):
        M.index_put_((idx[:, s], feat), w[:, s], accumulate=True)
    return M


def axis_counts(conn: torch.Tensor, grid_n: int) -> torch.Tensor:
    """conn [B, 2n^2] bool -> counts [B, 2n^2] float in {0, 1, 2}, vectorised over B."""
    idx, w = axis_slots(grid_n)
    idx, w = idx.to(conn.device), w.to(conn.device)
    return (conn.float()[:, idx] * w[None]).sum(-1)


def decode_axdeg(counts: torch.Tensor, n: int, tol: float = 1e-6):
    """Invert the axis-split degree counts (the completeness proof, executed).

    counts: [B, 2n^2] int/float, feature index axis * n^2 + r * n + c, values 0..2
            (i.e. AxisDegreeCode.rows() + 1, or counts_from_onehot(axdeg1h rows)).
    -> (conn [B, 2n^2] bool with the 2n constant features False, consistent [B] bool)

    Per row of cells the alternating prefix sum e_k = sum_{i<=k} (-1)^(k-i) h_i is
    run over ALL n slots; slots 0..n-2 are the free right-connections of that row
    and slot n-1 is the constant one, which must come out 0.  `consistent` is
    False if any recovered value is not 0/1 (which includes that last check), or
    if a count is not an integer.  Columns likewise with v and down-connections.
    """
    b = counts.shape[0]
    assert counts.shape[-1] == 2 * n * n, (counts.shape, n)
    x = counts.reshape(b, 2, n, n).double()
    integral = (x - x.round()).abs().le(tol).reshape(b, -1).all(-1)
    x = x.round()

    sgn = torch.pow(-1.0, torch.arange(n, device=counts.device, dtype=torch.float64))
    e_h = sgn.view(1, 1, n) * torch.cumsum(x[:, 0] * sgn.view(1, 1, n), dim=2)  # along c
    e_v = sgn.view(1, n, 1) * torch.cumsum(x[:, 1] * sgn.view(1, n, 1), dim=1)  # along r

    ok = integral
    for e in (e_h, e_v):
        ok = ok & ((e == 0) | (e == 1)).reshape(b, -1).all(-1)
    ok = ok & (e_h[:, :, n - 1] == 0).all(-1) & (e_v[:, n - 1, :] == 0).all(-1)

    conn = torch.cat([e_v.reshape(b, -1), e_h.reshape(b, -1)], dim=1) == 1  # down block, right block
    return conn & free_mask(n).to(conn.device)[None, :], ok


def counts_from_onehot(rows: torch.Tensor) -> torch.Tensor:
    """axdeg1h rows [B, 6n^2] -> counts [B, 2n^2] long (argmax of each 3-wide block)."""
    assert rows.shape[-1] % N_COUNTS == 0, rows.shape
    return rows.reshape(rows.shape[0], -1, N_COUNTS).argmax(-1)


def estimate_sq_norm(grid_n: int, n_mazes: int = _NORM_SAMPLES, seed: int = _NORM_SEED) -> float:
    """Mean ||count - 1||^2 over `n_mazes` DFS spanning trees (memoised, fixed seed).

    maze_dataset's generator reseeds numpy's global RNG, so its state is saved and
    restored here -- constructing a code must not move anybody else's RNG.
    """
    key = (int(grid_n), int(n_mazes), int(seed))
    if key not in _SQ_NORM_CACHE:
        import warnings

        import numpy as np
        from maze_dataset import MazeDataset, MazeDatasetConfig
        from maze_dataset.generation.generators import LatticeMazeGenerators

        state = np.random.get_state()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cfg = MazeDatasetConfig(name=f"axdeg_norm_{grid_n}", grid_n=int(grid_n),
                                        n_mazes=int(n_mazes), maze_ctor=LatticeMazeGenerators.gen_dfs,
                                        seed=int(seed))
                ds = MazeDataset.from_config(cfg, do_generate=True, load_local=False,
                                             save_local=False, gen_parallel=False, verbose=False)
            conn = torch.from_numpy(np.stack([m.connection_list.flatten() for m in ds]))
        finally:
            np.random.set_state(state)
        _SQ_NORM_CACHE[key] = float((axis_counts(conn, int(grid_n)) - 1.0).pow(2).sum(1).mean())
    return _SQ_NORM_CACHE[key]


class _AxisSplitDegree(GraftCode):
    """Shared machinery: per-cell horizontal / vertical open-neighbour counts."""

    def __init__(self, grid_n: int):
        self.grid_n = int(grid_n)
        self.idx, self.w = axis_slots(self.grid_n)          # [2n^2, 2] each
        self.M = incidence_matrix(self.grid_n)              # counts = conn.float() @ M
        self._cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def _on(self, device: torch.device):
        if device not in self._cache:
            self._cache[device] = (self.idx.to(device), self.w.to(device))
        return self._cache[device]

    def counts(self, conn: torch.Tensor) -> torch.Tensor:
        """conn [B, 2n^2] bool -> [B, 2n^2] float in {0, 1, 2} (no Python loop over B)."""
        idx, w = self._on(conn.device)
        return (conn.float()[:, idx] * w[None]).sum(-1)

    def meta(self) -> dict:
        return {}

    @classmethod
    def from_meta(cls, grid_n: int, meta: dict) -> "_AxisSplitDegree":
        return cls(grid_n, **meta)


@register
class AxisDegreeCode(_AxisSplitDegree):
    """Per-cell (horizontal, vertical) degree, centred: value = count - 1 in {-1, 0, +1}.

    n_features = 2n^2, index = axis * n^2 + r * n + c (axis 0 = h, 1 = v).
    Complete (see `decode_axdeg`) and AFFINE in the wall bits:
    rows(conn) == conn.float() @ M + bias with bias = -1, so as a fixed-W graft
    it is an edge graft with tied directions W_eff = M @ W (cos 1/2 between
    walls adjacent along a row/column) plus a constant -- see the module
    docstring before drawing conclusions from it.

    sq_norm: the expected squared feature norm; estimated from DFS mazes when
    not given (and stored in meta(), so reloading a checkpoint never re-estimates).
    """

    name = "axdeg"

    def __init__(self, grid_n: int, sq_norm: float | None = None):
        super().__init__(grid_n)
        self.n_features = 2 * self.grid_n * self.grid_n
        self.bias = -torch.ones(self.n_features)
        self.sq_norm = float(sq_norm) if sq_norm is not None else estimate_sq_norm(self.grid_n)

    def rows(self, conn: torch.Tensor) -> torch.Tensor:
        return self.counts(conn) - 1.0

    def expected_sq_norm(self) -> float:
        """Mean ||rows||^2 over DFS spanning trees; pass as GraftInjector(norm_match=...)."""
        return self.sq_norm

    def meta(self) -> dict:
        return {"sq_norm": self.sq_norm}


@register
class AxisDegreeOneHotCode(_AxisSplitDegree):
    """One-hot over the count {0, 1, 2} per (cell, axis): a degree is not a wall.

    n_features = 6n^2, index = (axis * n^2 + r * n + c) * 3 + count; exactly 2n^2
    ones per row, expected_sq_norm() = 2n^2, so GraftInjector's gain is 1.  Carries
    exactly the same information as AxisDegreeCode (`counts_from_onehot` then
    `decode_axdeg`) but is NOT an affine function of the wall bits: the count-1
    indicator is the XOR of the two wall bits.
    """

    name = "axdeg1h"

    def __init__(self, grid_n: int):
        super().__init__(grid_n)
        self.n_cells2 = 2 * self.grid_n * self.grid_n
        self.n_features = N_COUNTS * self.n_cells2
        self._base = torch.arange(self.n_cells2) * N_COUNTS

    def rows(self, conn: torch.Tensor) -> torch.Tensor:
        counts = self.counts(conn).long()                       # [B, 2n^2] in {0,1,2}
        idx = self._base.to(conn.device)[None, :] + counts
        out = torch.zeros(conn.shape[0], self.n_features, device=conn.device)
        return out.scatter_(1, idx, 1.0)

    def expected_sq_norm(self) -> float:
        return float(self.n_cells2)
