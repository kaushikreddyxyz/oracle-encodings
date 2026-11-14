"""Egocentric ("ego") graft code: the maze as seen from the cell you stand on.

The static codes (edge, onion) write the same feature vector at every active
position -- feature f always means "the connection at absolute index f".  The
ego code is DYNAMIC: at position p it writes the maze RELATIVE to the current
cell at p, so one physical wall is feature i while the solver stands at one
cell and features j, k, l as it walks on.

Frame (the "current cell") at position p
    the last coordinate token at a position q <= p with
      * q > end_pos        -- the adjacency list never moves the frame.  This is
                              also why the code is bit-identical on the `dummy`
                              data variant (maze.core.data.dummy_adjacency),
                              which blanks positions 1..end_pos-1: nothing at or
                              below end_pos is ever read.
      * q not inside <TARGET_START> ... <TARGET_END> -- the goal is not where
                              you are, so the target coordinate is skipped.
    So the frame is undefined before the origin token (position end_pos + 2),
    then the origin through <ORIGIN_END> <TARGET_START> (target) <TARGET_END>
    <PATH_START>, then each path coordinate as it appears (the first path token
    is the origin again); <PATH_END> and padding keep the last frame.
    An undefined frame gives an ALL-ZERO feature vector: the graft stays silent
    rather than asserting anything.  0 = "no evidence" is the graceful
    undefined value, +/-1 asserts evidence (maze_legacy half-mask result).
    Note this differs from the static codes' `complete` timing, which is active
    from <ADJLIST_END> on: the ego code has nothing to say until the origin.

Feature layout (n = grid_n, m = 2n - 1)
    index = kind * m^2 + (dr + n - 1) * m + (dc + n - 1)
    kind in {0 = down, 1 = right};  dr, dc in [-(n-1), n-1]
    n_features = 2 m^2                   (242 at n = 6, against 72 for `edge`)
The absolute connection (kind, r, c) of `conn` (see maze/core/injection.py:
index r*n+c is the DOWN connection of (r,c), n^2 + r*n + c the RIGHT one, True =
open) is written at the relative index of (kind, r - r0, c - c0) for the frame
(r0, c0), with +1 open / -1 wall.  Every absolute connection lands on exactly
one in-range relative index, so exactly 2n^2 entries are nonzero at every framed
position and the injected norm already matches the edge code's -- no rescaling,
expected_sq_norm() == 2n^2.  Relative offsets that leave the grid stay 0: that
pair of cells does not exist, which is again "no evidence".

window (int, default None = whole maze)
    keep only offsets with max(|dr|, |dc|) <= window, a locality variant for
    later.  Dropped connections are simply not written, so the number of nonzero
    entries now depends on the frame; expected_sq_norm() is then the MEAN over
    the n^2 possible frames of the number of kept connections, which is what
    GraftInjector(norm_match=...) wants in order to rescale back to 2n^2.
"""

from __future__ import annotations

import torch

from maze.core.graft import ORIGIN_END, TARGET_END, TARGET_START, GraftCode, register


def relative_table(grid_n: int, window: int | None = None) -> tuple[torch.Tensor, int]:
    """-> (G [n^2, 2n^2] long, n_features).

    G[cell, f] is the relative feature index of absolute connection f when the
    frame is `cell` (= r0 * n + c0).  With a window, out-of-window entries get
    the value n_features: a trash slot that callers scatter into and then slice
    away, so the scatter stays a single vectorised op.
    """
    n = int(grid_n)
    m = 2 * n - 1
    n_features = 2 * m * m
    cells = torch.arange(n * n)
    r0, c0 = cells // n, cells % n                     # [n^2]
    f = torch.arange(2 * n * n)
    kind, r, c = f // (n * n), (f % (n * n)) // n, f % n  # [2n^2]
    dr = r[None, :] - r0[:, None]                      # [n^2, 2n^2]
    dc = c[None, :] - c0[:, None]
    rel = kind[None, :] * (m * m) + (dr + n - 1) * m + (dc + n - 1)
    if window is not None:
        drop = torch.maximum(dr.abs(), dc.abs()) > int(window)
        rel = torch.where(drop, torch.full_like(rel, n_features), rel)
    return rel.long(), n_features


@register
class EgoCode(GraftCode):
    """Egocentric wall code (see module docstring)."""

    name = "ego"
    dynamic = True

    def __init__(self, grid_n: int, id_to_coord, window: int | None = None):
        """id_to_coord: {token id: (r, c)} for the tokenizer's coordinate tokens
        (build it from tokenizer.get_vocab() with `^\\((\\d+),(\\d+)\\)$`; never
        hard-code the ids).  A list of [id, r, c] triples is accepted too, which
        is how meta() stores it."""
        self.grid_n = int(grid_n)
        self.window = None if window is None else int(window)
        if not isinstance(id_to_coord, dict):
            id_to_coord = {int(i): (int(r), int(c)) for i, r, c in id_to_coord}
        self.id_to_coord = {int(i): (int(r), int(c)) for i, (r, c) in id_to_coord.items()}
        n = self.grid_n
        assert self.id_to_coord, "ego code needs the tokenizer's coordinate vocab"
        for i, (r, c) in self.id_to_coord.items():
            assert 0 <= r < n and 0 <= c < n, f"coord token {i} -> {(r, c)} outside grid {n}"
        self._G, self.n_features = relative_table(n, self.window)
        # id -> flat cell index (r * n + c), -1 for every non-coordinate token
        lut = torch.full((max(self.id_to_coord) + 1,), -1, dtype=torch.long)
        for i, (r, c) in self.id_to_coord.items():
            lut[i] = r * n + c
        self._lut = lut
        self._cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    # ---- tables -------------------------------------------------------------
    def _tables(self, device):
        if device not in self._cache:
            self._cache[device] = (self._G.to(device), self._lut.to(device))
        return self._cache[device]

    def cell_of(self, tokens: torch.Tensor) -> torch.Tensor:
        """[.. ] token ids -> flat cell index, -1 where the token is not a coordinate."""
        _, lut = self._tables(tokens.device)
        ids = tokens.long()
        safe = ids.clamp(min=0, max=lut.numel() - 1)
        return torch.where((ids >= 0) & (ids < lut.numel()), lut[safe], torch.full_like(ids, -1))

    def expected_sq_norm(self) -> float:
        """Mean squared norm of a framed feature row (pass to GraftInjector(norm_match=...)).

        Without a window every one of the 2n^2 connections is written as +/-1 at
        some in-range relative index, so this is exactly 2n^2 -- the edge code's
        value, hence gain 1.  With a window it is the mean over the n^2 possible
        frames of the number of connections that survive the window."""
        if self.window is None:
            return float(2 * self.grid_n ** 2)
        return float((self._G != self.n_features).sum(dim=1).double().mean())

    # ---- vectorised features -------------------------------------------------
    def _scatter(self, frame: torch.Tensor, signed: torch.Tensor) -> torch.Tensor:
        """frame [...] flat cell index (-1 = undefined), signed [..., 2n^2] -> [..., F]."""
        G, _ = self._tables(signed.device)
        idx = G[frame.clamp(min=0)]                                   # [..., 2n^2]
        src = signed * (frame >= 0).unsqueeze(-1).to(signed.dtype)    # silent when unframed
        out = torch.zeros(*idx.shape[:-1], self.n_features + 1, device=signed.device, dtype=signed.dtype)
        out.scatter_(-1, idx, src)
        return out[..., : self.n_features]

    def _frames(self, tokens, attention_mask, end_pos):
        """-> [B, P] flat cell index of the frame at each position (-1 = undefined)."""
        pos = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
        cell = self.cell_of(tokens)                                   # [B, P], -1 = not a coord
        # inclusive cumsum: the target coordinate itself sits at depth 1 and is skipped
        depth = (tokens == TARGET_START).long().cumsum(1) - (tokens == TARGET_END).long().cumsum(1)
        valid = (cell >= 0) & (pos > end_pos[:, None]) & (depth <= 0) & attention_mask.bool()
        last = (valid.long() * (pos + 1)).cummax(dim=1).values - 1    # -1 until the first coord
        frame = torch.where(last >= 0, cell.gather(1, last.clamp(min=0)), torch.full_like(cell, -1))
        # padding carries the last frame forward; the graft must be silent there
        return torch.where(attention_mask.bool(), frame, torch.full_like(frame, -1))

    def features(self, tokens, attention_mask, conn, end_pos) -> torch.Tensor:
        """[B, P, F] float on conn.device; 0 wherever the frame is undefined or the
        position is padding."""
        dev = conn.device
        tokens = tokens.to(dev).long()
        attention_mask = attention_mask.to(dev)
        end_pos = end_pos.to(dev).long()
        frame = self._frames(tokens, attention_mask, end_pos)          # [B, P]
        if self.start != "adjlist_end":                                # timing rule (graft.py): silent before the start token
            from maze.core.graft import start_positions
            pos = torch.arange(tokens.shape[1], device=dev)[None, :]
            frame = torch.where(pos >= start_positions(tokens, end_pos, self.start).to(dev)[:, None], frame, torch.full_like(frame, -1))
        signed = (conn.float() * 2.0 - 1.0)[:, None, :].expand(-1, tokens.shape[1], -1)
        return self._scatter(frame, signed)

    # ---- KV-cache decoding ----------------------------------------------------
    def init_state(self, tokens, attention_mask, conn, end_pos) -> dict:
        """State after the prefix `tokens`.  Carries the frame per row, the
        target-segment depth, the next position index and the signed maze."""
        dev = conn.device
        tokens = tokens.to(dev).long()
        attention_mask = attention_mask.to(dev)
        end_pos = end_pos.to(dev).long()
        lens = attention_mask.long().sum(1)                            # real prefix length
        from maze.core.graft import start_positions
        active = (start_positions(tokens, end_pos, self.start).to(dev) < lens) if self.start != "adjlist_end" \
            else torch.ones_like(lens, dtype=torch.bool)
        base = {"end_pos": end_pos.clone(), "signed": conn.float() * 2.0 - 1.0, "active": active}
        if tokens.shape[1] == 0:                                       # empty prefix
            zero = torch.zeros_like(end_pos)
            return {"frame": zero - 1, "depth": zero, "pos": zero, **base}
        frame = self._frames(tokens, attention_mask, end_pos)
        last = (lens - 1).clamp(min=0)
        depth = ((tokens == TARGET_START).long() - (tokens == TARGET_END).long())
        depth = (depth * attention_mask.long()).cumsum(1).gather(1, last[:, None]).squeeze(1)
        return {
            "frame": torch.where(lens > 0, frame.gather(1, last[:, None]).squeeze(1),
                                 torch.full_like(lens, -1)),
            "depth": torch.where(lens > 0, depth, torch.zeros_like(depth)),
            "pos": lens.clone(),                                       # position of the NEXT token
            **base,
        }

    def step(self, state: dict, new_tokens: torch.Tensor):
        """-> (features [B, F] for the position holding `new_tokens`, new state).

        Same frame rule as features(), target segment included: during decoding
        every generated token is past <PATH_START> so the target logic never
        fires, but keeping it means step() reproduces features() token by token
        on ANY stream."""
        signed = state["signed"]
        tok = new_tokens.to(signed.device).long()
        depth = state["depth"] + (tok == TARGET_START).long() - (tok == TARGET_END).long()
        cell = self.cell_of(tok)
        valid = (cell >= 0) & (state["pos"] > state["end_pos"]) & (depth <= 0)
        frame = torch.where(valid, cell, state["frame"])
        active = state["active"] if self.start == "adjlist_end" else (state["active"] | (tok == ORIGIN_END))
        feats = self._scatter(torch.where(active, frame, torch.full_like(frame, -1)), signed)
        new_state = dict(state, frame=frame, depth=depth, pos=state["pos"] + 1, active=active)
        return feats, new_state

    # ---- checkpoint metadata ---------------------------------------------------
    def meta(self) -> dict:
        return {
            "id_to_coord": [[int(i), int(r), int(c)] for i, (r, c) in sorted(self.id_to_coord.items())],
            "window": self.window,
        }

    @classmethod
    def from_meta(cls, grid_n: int, meta: dict) -> "EgoCode":
        return cls(grid_n, meta["id_to_coord"], meta.get("window"))
