"""Onion (magnitude-layered) tile codes for the maze graft.

The edge graft gives every connection its own direction.  The onion graft
(after Csordas et al. 2024, arXiv 2408.10920) gives every TILE one direction and
stores that tile's four neighbour connections as four orders of magnitude along
it:

    m_t = g * sum_k  s_{t,k} * base**(-k)        k = 0..3,  s = +/-1 (open / wall)
    x   = scale * sum_t  m_t * V_t

With base 2 the 16 tile states are 16 evenly spaced magnitudes on one line: the
outermost layer (k = 0) is a threshold on the projection, every deeper layer has
to be peeled (the sign of layer k flips 2**k times along the line), so no
connection below layer 0 has a direction of its own.

The map from the 2n^2 signed connection features to the residual stream is still
LINEAR -- it is just many-to-one in direction: W_eff = C^T V with C the
[n_dirs, 2n^2] layer-weight matrix below.  `OnionInjector.W_inj` exposes W_eff,
and that is what checkpoints store under "W_inj", so every consumer that does
`signed_conn @ W_inj` (training, eval_ckpts, eval_distance, eval_local, ...)
works unchanged.  Only V is a parameter: a trainable onion can move its tile
directions but cannot give a connection its own direction.

Layouts
  full     every tile owns its 4 connections; each interior connection is
           stored twice (once in each endpoint tile, generally at two depths)
  checker  only tiles with (r + c) even own connections; every interior
           connection is stored exactly once, at one depth 0..3

Tile sides index the connection list with wrap-around, so the side of a border
tile that faces outward maps onto one of the 2n always-closed features and is a
constant wall, exactly as in the edge graft.
"""

import torch

LAYOUTS = ("full", "checker")
SIDES = ("N", "E", "S", "W")


def side_feature(r: int, c: int, side: str, n: int) -> int:
    """Index into connection_list.flatten() of the given side of tile (r, c)."""
    if side == "N":
        return ((r - 1) % n) * n + c
    if side == "S":
        return r * n + c
    if side == "W":
        return n * n + r * n + (c - 1) % n
    if side == "E":
        return n * n + r * n + c
    raise ValueError(side)


def free_mask(n: int) -> torch.Tensor:
    """True for the 2n(n-1) connections that vary between mazes."""
    m = torch.ones(2, n, n, dtype=torch.bool)
    m[0, n - 1, :] = False  # bottom-row downs
    m[1, :, n - 1] = False  # right-column rights
    return m.flatten()


def onion_tiles(n: int, layout: str) -> list[tuple[int, int]]:
    assert layout in LAYOUTS, f"unknown onion layout: {layout}"
    return [
        (r, c) for r in range(n) for c in range(n)
        if layout == "full" or (r + c) % 2 == 0
    ]


def onion_code(
    n: int,
    layout: str = "checker",
    base: float = 2.0,
    rotate: bool = True,
    norm_match: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (C [n_dirs, 2n^2] layer weights, depth [n_dirs, 2n^2] int, -1 = absent).

    rotate: tile (r, c) starts its N,E,S,W layer order at side (r + 2c) % 4, so
      depth is not confounded with orientation or with a fixed neighbour.
    norm_match: rescale so ||C||_F^2 = 2n^2, the edge code's value; with
      near-orthogonal unit directions the injected norm then matches the edge
      graft's and the comparison isolates geometry from injection strength.
    """
    tiles = onion_tiles(n, layout)
    C = torch.zeros(len(tiles), 2 * n * n)
    depth = torch.full((len(tiles), 2 * n * n), -1, dtype=torch.long)
    for t, (r, c) in enumerate(tiles):
        shift = (r + 2 * c) % 4 if rotate else 0
        for k in range(4):
            f = side_feature(r, c, SIDES[(k + shift) % 4], n)
            C[t, f] = float(base) ** (-k)
            depth[t, f] = k
    if norm_match:
        C = C * (2 * n * n / C.pow(2).sum()).sqrt()
    return C, depth


def min_depth(depth: torch.Tensor) -> torch.Tensor:
    """Shallowest layer at which each connection is stored (-1 = nowhere)."""
    d = depth.clone()
    d[d < 0] = 99
    out = d.min(dim=0).values
    out[out == 99] = -1
    return out


class OnionInjector(torch.nn.Module):
    """Drop-in for MazeInjector: same hook / scale / W_inj interface."""

    def __init__(
        self,
        d_model: int,
        grid_n: int,
        layer: int = 0,
        layout: str = "checker",
        base: float = 2.0,
        rotate: bool = True,
        trainable: bool = False,
        scale: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        self.layer = layer
        self.feature_mode = "complete"
        self.scale = scale
        self.layout, self.base, self.rotate = layout, float(base), bool(rotate)
        C, depth = onion_code(grid_n, layout, base, rotate)
        self.register_buffer("C", C)
        self.register_buffer("depth", depth)
        gen = torch.Generator().manual_seed(seed)
        v = torch.randn(C.shape[0], d_model, generator=gen)
        v = v / v.norm(dim=-1, keepdim=True)
        self.V = torch.nn.Parameter(v, requires_grad=trainable)

    @property
    def W_inj(self) -> torch.Tensor:
        """Effective [2n^2, d_model] edge-feature matrix (rank <= n_dirs)."""
        return self.C.t() @ self.V

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.layer}.hook_resid_pre"

    def hook_fn(self, resid, hook, features: torch.Tensor):
        return resid + self.scale * (features.to(resid.dtype) @ self.W_inj)

    def code_meta(self) -> dict:
        """Travels with the checkpoint (tensors + plain values only)."""
        return {
            "code": f"onion-{self.layout}", "onion_base": self.base,
            "onion_rotate": self.rotate, "onion_V": self.V.detach().cpu(),
            "onion_C": self.C.cpu(), "onion_depth": self.depth.cpu(),
        }
