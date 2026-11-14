"""World-state injection for maze-transformer models.

Injects the maze's ground-truth connectivity (the 2 x n x n connection list:
down-connections and right-connections per cell, 2*n*n binary features) into
the residual stream via a linear map, using TransformerLens hooks.

Feature timing modes (when does each token position get which features):
  - "complete":    positions from <ADJLIST_END> onward get the full maze
                   features; earlier positions get zeros. Matches probing
                   at <PATH_START> (model provably has complete info).
  - "incremental": each position gets the features of the connections
                   revealed in tokens up to and including it (monotone
                   accumulation over the adjacency list; positions after
                   <ADJLIST_END> get the full maze).

Projection modes:
  - fixed:     W_inj frozen at init (scaled Gaussian directions).
  - trainable: W_inj is a learned parameter (registered with the optimizer).

Features are encoded as +/-1 (connection present / absent) once active, 0
before any information is injected at that position.

Injection *variants* (VARIANTS_SPEC.md) are handled by InjectionRuntime, which
turns (injector, mode metadata, optional teacher) into the fwd_hooks for a
batch: complete / window / teacher. The "dummy" variant is a data transform
(maze.core.data.dummy_adjacency), not an injection mode.
"""

import re

import torch
from maze_dataset import SPECIAL_TOKENS

from maze.core.data import make_features


class MazeFeatureComputer:
    """Compute per-position connection-list features from token id sequences."""

    def __init__(self, tokenizer, grid_n: int, device: torch.device):
        self.grid_n = grid_n
        self.device = device
        self.n_features = 2 * grid_n * grid_n

        vocab: dict[str, int] = tokenizer.get_vocab()
        self.id_to_coord: dict[int, tuple[int, int]] = {}
        coord_re = re.compile(r"^\((\d+),(\d+)\)$")
        for tok, idx in vocab.items():
            m = coord_re.match(tok)
            if m:
                self.id_to_coord[idx] = (int(m.group(1)), int(m.group(2)))
        self.connector_id: int = vocab[SPECIAL_TOKENS.CONNECTOR]
        self.adjlist_end_id: int = vocab[SPECIAL_TOKENS.ADJLIST_END]

    def _connection_feature_index(
        self, a: tuple[int, int], b: tuple[int, int]
    ) -> int | None:
        (r1, c1), (r2, c2) = a, b
        if r1 == r2 and abs(c1 - c2) == 1:  # right-connection from leftmost cell
            return self.grid_n * self.grid_n + r1 * self.grid_n + min(c1, c2)
        if c1 == c2 and abs(r1 - r2) == 1:  # down-connection from upper cell
            return min(r1, r2) * self.grid_n + c1
        return None  # malformed edge; ignore

    def features(
        self, tokens: torch.Tensor, mode: str = "complete"
    ) -> torch.Tensor:
        """tokens: (batch, pos) int tensor -> (batch, pos, n_features) float."""
        batch, n_pos = tokens.shape
        out = torch.zeros(batch, n_pos, self.n_features)
        toks = tokens.tolist()
        for b in range(batch):
            state = torch.zeros(self.n_features)
            active = False  # incremental mode: any connection seen yet
            row = toks[b]
            for p, tok_id in enumerate(row):
                if (
                    tok_id == self.connector_id
                    and 1 <= p < n_pos - 1
                    and row[p - 1] in self.id_to_coord
                    and row[p + 1] in self.id_to_coord
                ):
                    fi = self._connection_feature_index(
                        self.id_to_coord[row[p - 1]], self.id_to_coord[row[p + 1]]
                    )
                    if fi is not None:
                        state[fi] = 1.0
                        active = True
                if mode == "incremental":
                    if active:
                        # +/-1 encoding: revealed-present=+1, not-yet-revealed/absent=-1
                        out[b, p] = state * 2.0 - 1.0
                elif mode == "complete":
                    pass  # filled in below once adjlist end position is known
                else:
                    raise ValueError(f"unknown feature mode: {mode}")
            if mode == "complete":
                ends = [p for p, t in enumerate(row) if t == self.adjlist_end_id]
                if ends:
                    out[b, ends[0]:] = state * 2.0 - 1.0
        return out.to(self.device)


class MazeInjector(torch.nn.Module):
    """Linear projection of maze features into the residual stream.

    Attach via TransformerLens hooks at blocks.{layer}.hook_resid_pre.
    """

    def __init__(
        self,
        d_model: int,
        grid_n: int,
        layer: int = 0,
        feature_mode: str = "complete",
        trainable: bool = False,
        scale: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        self.layer = layer
        self.feature_mode = feature_mode
        self.scale = scale
        n_features = 2 * grid_n * grid_n
        gen = torch.Generator().manual_seed(seed)
        # unit-norm random directions per feature; scale controls injection strength
        w = torch.randn(n_features, d_model, generator=gen)
        w = w / w.norm(dim=-1, keepdim=True)
        self.W_inj = torch.nn.Parameter(w, requires_grad=trainable)

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.layer}.hook_resid_pre"

    def hook_fn(self, resid, hook, features: torch.Tensor):
        return resid + self.scale * (features.to(resid.dtype) @ self.W_inj)

    def forward_with_injection(
        self,
        model,
        tokens: torch.Tensor,
        feature_computer: MazeFeatureComputer,
        features: torch.Tensor | None = None,
        **model_kwargs,
    ):
        if features is None:
            features = feature_computer.features(tokens, self.feature_mode)
        return model.run_with_hooks(
            tokens,
            fwd_hooks=[
                (
                    self.hook_name,
                    lambda resid, hook: self.hook_fn(resid, hook, features),
                )
            ],
            **model_kwargs,
        )


INJECT_MODES = ("complete", "window", "teacher")


class InjectionRuntime:
    """Injector + variant metadata -> per-batch features and fwd_hooks.

    Modes (all share the true +/-1 features at positions >= end_pos):
      complete  nothing else (the original behaviour, bit-identical)
      window    features only on [end_pos, end_pos + window_k)
      teacher   plus a frozen teacher's belief state s_p = 2*sigmoid(z_p)-1 at
                1 <= p < end_pos (position 0 = BOS stays clean)

    The single (tokens, attn, conn, end_pos) entry point is what lets training
    and every eval path share one code path across modes.
    """

    def __init__(
        self,
        injector: MazeInjector,
        mode: str = "complete",
        window_k: int = 4,
        teacher=None,
    ):
        assert mode in INJECT_MODES, f"unknown inject mode: {mode}"
        assert mode != "teacher" or teacher is not None, "teacher mode needs a teacher"
        self.injector = injector
        self.mode = mode
        self.window_k = int(window_k)
        self.teacher = teacher

    @property
    def meta(self) -> dict:
        return {"mode": self.mode, "window_k": self.window_k}

    def features(self, tokens, attn, conn, end_pos) -> torch.Tensor:
        feats = make_features(
            conn, end_pos, tokens.shape[1], attn,
            window_k=self.window_k if self.mode == "window" else None,
        )
        if self.mode == "teacher":
            # beliefs are zero at p = 0 and p >= end_pos, so this is a disjoint
            # union with the true features above
            feats = feats + self.teacher.beliefs(tokens, attn, end_pos)
        return feats

    def fwd_hooks(self, tokens, attn, conn, end_pos) -> list:
        feats = self.features(tokens, attn, conn, end_pos)
        inj = self.injector
        return [(inj.hook_name, lambda resid, hook: inj.hook_fn(resid, hook, feats))]


def as_runtime(injector) -> InjectionRuntime:
    """Accept a bare MazeInjector (complete mode) or a ready InjectionRuntime."""
    return (
        injector
        if isinstance(injector, InjectionRuntime)
        else InjectionRuntime(injector)
    )
