"""Dual-graph ("wall-vertex") graft codes: the maze as a set of CORNER TYPES.

The edge graft gives every connection its own direction, so the injected vector
is an affine function of the wall bits and "an edge" is a feature.  The onion
graft keeps that linearity and only shares directions between connections.  The
codes here break the linearity on purpose: the unit of representation is not a
wall but a *corner*, i.e. the local configuration of walls meeting at a point of
the dual lattice.  Nothing about the injected vector is a per-edge quantity --
recovering one connection requires reading a whole 16-way categorical feature --
yet the maze is still fully determined (see `decode_vertex`).

Geometry
--------
A grid of n x n cells has an (n+1) x (n+1) lattice of *wall-vertices*.
Wall-vertex (i, j), 0 <= i, j <= n, is the corner shared by the four cells
(i-1, j-1), (i-1, j), (i, j-1), (i, j); its four incident unit wall segments
are, in the canonical order N, E, S, W:

    N   vertical wall between cells (i-1, j-1) and (i-1, j)   [right index n^2 + (i-1)n + (j-1)]
    E   horizontal wall between cells (i-1, j) and (i, j)     [down  index      (i-1)n +  j   ]
    S   vertical wall between cells (i,   j-1) and (i,   j)   [right index n^2 +  i n   + (j-1)]
    W   horizontal wall between cells (i-1, j-1) and (i, j-1) [down  index      (i-1)n + (j-1)]

(`conn` is `connection_list.flatten()`: index r*n + c is the DOWN connection
between (r, c) and (r+1, c), index n^2 + r*n + c the RIGHT connection between
(r, c) and (r, c+1); True = passage open, so a wall is present iff conn is
False.  These are exactly the indices of
`MazeFeatureComputer._connection_feature_index`, checked in the tests.)

Only the (n-1)^2 INTERIOR vertices 1 <= i, j <= n-1 are used.  For those, all
four incident segments are interior walls, i.e. *free* features (the 2n
bottom-row downs / right-column rights are constant walls and never appear).
Boundary vertices would only add constants and duplicate information, and the
outer border of the maze is not a free feature at all -- see "Design notes".

Codes
-----
VertexCode ("vertex")  one-hot over the 16 wall patterns per interior vertex.
                       n_features = 16 (n-1)^2, exactly (n-1)^2 ones per row.
                       Lossless: every free connection sits in 1 or 2 vertices.
DegreeCode ("degree")  one-hot over the DEGREE (0..4) only, 5 (n-1)^2 features.
                       Deliberately UNDERSPECIFIED negative control.

Feature index layout (stable, vertex-major):

    index = v * P + k,   v = (i - 1) * (n - 1) + (j - 1),
    VertexCode: P = 16, k = pattern = N*8 + E*4 + S*2 + W  (bit = wall present)
    DegreeCode: P =  5, k = degree  = N + E + S + W

Rows are 0/1 floats (NOT +/-1), so ||row||^2 = (n-1)^2 for both codes; that is
`expected_sq_norm()`, which the trainer passes to `GraftInjector(norm_match=...)`
so the injected norm matches the edge graft's 2n^2.

Design notes / caveats
----------------------
* Boundary vertices (i or j in {0, n}) are excluded.  Including them would add
  (n+1)^2 - (n-1)^2 = 4n vertices whose patterns mix free bits with the always-
  walled outer border, adding no information but diluting the norm; the corner
  vertices (0,0), (0,n), (n,0), (n,n) are fully constant.  The cost of excluding
  them is nothing for decodability: every free connection already appears in at
  least one interior vertex.
* Multiplicity is 1 or 2, not uniform: a free down connection (r, c) is in one
  interior vertex when c = 0 or c = n-1 and in two otherwise (mirror-image for
  rights).  So the code is mildly non-uniform across the grid, like OnionCode's
  "full" layout.
* The code is NOT an affine function of the wall bits: an indicator of a 4-bit
  pattern has only 4/15 of its variance in the degree <= 1 Fourier levels, so a
  least-squares fit from the signed connections leaves a large residual
  (test 4).  That is the point -- "a corner is not a wall".
"""

from __future__ import annotations

import torch

from maze.core.graft import GraftCode, register
from maze.core.onion import free_mask

SEGMENT_ORDER = ("N", "E", "S", "W")
N_PATTERNS = 16


def vertex_segments(n: int) -> torch.Tensor:
    """-> LongTensor [(n-1)^2, 4]: conn indices of the N, E, S, W segments of each
    interior wall-vertex, vertex-major in (i, j) row order (v = (i-1)(n-1) + (j-1))."""
    assert n >= 2, f"need grid_n >= 2 for interior wall-vertices, got {n}"
    i = torch.arange(1, n).view(-1, 1).expand(n - 1, n - 1).reshape(-1)  # [V]
    j = torch.arange(1, n).view(1, -1).expand(n - 1, n - 1).reshape(-1)  # [V]
    nn = n * n
    north = nn + (i - 1) * n + (j - 1)  # right connection (i-1, j-1)
    east = (i - 1) * n + j              # down  connection (i-1, j)
    south = nn + i * n + (j - 1)        # right connection (i,   j-1)
    west = (i - 1) * n + (j - 1)        # down  connection (i-1, j-1)
    return torch.stack([north, east, south, west], dim=1).long()


def _segment_counts(n: int) -> torch.Tensor:
    """[2n^2] float: how many interior vertices store each connection (0, 1 or 2)."""
    seg = vertex_segments(n).reshape(-1)
    out = torch.zeros(2 * n * n)
    out.index_add_(0, seg, torch.ones(seg.numel()))
    return out


def decode_vertex(rows: torch.Tensor, n: int, tol: float = 1e-4):
    """Invert VertexCode.

    rows: [B, 16 (n-1)^2] float (one-hot per vertex block).
    -> (conn [B, 2n^2] bool with the 2n constant features False, consistent [B] bool)

    `consistent` is False for a row whose vertex blocks are not one-hot, or whose
    vertices disagree about a segment they share -- the check that makes the code
    a redundant (error-detecting) description of the maze rather than a free
    labelling of corners.
    """
    seg = vertex_segments(n).to(rows.device)
    v_count = seg.shape[0]
    assert rows.shape[-1] == N_PATTERNS * v_count, (rows.shape, n)
    blocks = rows.reshape(rows.shape[0], v_count, N_PATTERNS).float()
    pattern = blocks.argmax(dim=-1)                                  # [B, V]
    onehot_ok = ((blocks.sum(-1) - 1.0).abs() <= tol) & ((blocks.max(-1).values - 1.0).abs() <= tol)
    onehot_ok = onehot_ok.all(dim=-1)                                # [B]

    shifts = torch.tensor([3, 2, 1, 0], device=rows.device)
    bits = (pattern[:, :, None] >> shifts[None, None, :]) & 1        # [B, V, 4] wall present
    seg_flat = seg.reshape(-1)                                       # [4V]
    total = torch.zeros(rows.shape[0], 2 * n * n, device=rows.device)
    total.index_add_(1, seg_flat, bits.reshape(rows.shape[0], -1).float())
    count = _segment_counts(n).to(rows.device)                       # [2n^2]
    agree = (total == 0) | (total == count[None, :])                 # count 0 -> total 0 -> True
    conn = (total == 0) & free_mask(n).to(rows.device)[None, :]      # open iff no vertex saw a wall
    return conn, agree.all(dim=-1) & onehot_ok


class _VertexOneHot(GraftCode):
    """Shared machinery: per-interior-vertex one-hot over some function of its 4 wall bits."""

    n_patterns: int = N_PATTERNS

    def __init__(self, grid_n: int):
        self.grid_n = int(grid_n)
        self.seg = vertex_segments(self.grid_n)                 # [V, 4]
        self.v_count = self.seg.shape[0]
        self.n_features = self.n_patterns * self.v_count
        self._flat = self.seg.reshape(-1)                       # [4V]
        self._base = torch.arange(self.v_count) * self.n_patterns
        self._shift = torch.tensor([3, 2, 1, 0])
        self._cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _on(self, device: torch.device):
        if device not in self._cache:
            self._cache[device] = (self._flat.to(device), self._base.to(device), self._shift.to(device))
        return self._cache[device]

    def _key(self, bits: torch.Tensor) -> torch.Tensor:
        """bits [B, V, 4] long (1 = wall present) -> [B, V] long index within the block."""
        raise NotImplementedError

    def rows(self, conn: torch.Tensor) -> torch.Tensor:
        """conn [B, 2n^2] bool -> [B, n_features] float, 0/1 with (n-1)^2 ones."""
        flat, base, _ = self._on(conn.device)
        walls = (~conn.bool())[:, flat].reshape(conn.shape[0], self.v_count, 4).long()
        idx = base[None, :] + self._key(walls)
        out = torch.zeros(conn.shape[0], self.n_features, device=conn.device)
        return out.scatter_(1, idx, 1.0)

    def expected_sq_norm(self) -> float:
        """Squared feature norm of any row; pass as GraftInjector(norm_match=...)."""
        return float(self.v_count)

    def meta(self) -> dict:
        return {}

    @classmethod
    def from_meta(cls, grid_n: int, meta: dict) -> "_VertexOneHot":
        return cls(grid_n)


@register
class VertexCode(_VertexOneHot):
    """One-hot over the 16 wall patterns at each interior wall-vertex.

    n_features = 16 (n-1)^2; index = v*16 + (N*8 + E*4 + S*2 + W), bit set iff a
    wall is present on that segment.  Lossless (see `decode_vertex`) and not an
    affine function of the connection bits: a corner type, not an edge.
    """

    name = "vertex"
    n_patterns = N_PATTERNS

    def _key(self, bits):
        _, _, shift = self._on(bits.device)
        return (bits << shift[None, None, :]).sum(-1)


@register
class DegreeCode(_VertexOneHot):
    """NEGATIVE CONTROL: one-hot over the DEGREE (0..4) of each interior wall-vertex.

    n_features = 5 (n-1)^2; index = v*5 + (number of walls meeting at v).  Same
    shape of code as VertexCode (one categorical per corner, (n-1)^2 ones per
    row, same expected_sq_norm) but DELIBERATELY UNDERSPECIFIED: it records how
    many walls meet at a corner and not which ones, so it does NOT determine the
    maze.  Rotating one corner's wall pattern (e.g. swapping the two segments of
    vertex (1,1) that no other interior vertex sees) changes the maze while
    leaving every degree, hence every feature, identical.  Use it to show that a
    graft the model can exploit has to actually specify the world state.
    """

    name = "degree"
    n_patterns = 5

    def _key(self, bits):
        return bits.sum(-1)
