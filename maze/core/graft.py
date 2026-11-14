"""Graft codes: how a maze's connectivity becomes per-position residual-stream features.

A GraftCode turns (tokens, conn) into a feature tensor [batch, pos, n_features]; a
GraftInjector maps features into the residual stream with a [n_features, d_model]
matrix `W` (delta = scale * features @ W) at blocks.{layer}.hook_resid_pre.

Two kinds of code
  static   the feature vector depends on the maze only (edge, onion, vertex, ...):
           features[b, p] = rows(conn)[b] at every active position (p >= end_pos,
           real token), 0 elsewhere -- the `complete` timing of maze/core/injection.py
  dynamic  the feature vector also depends on the tokens up to p (egocentric):
           the code defines features() itself and a (init_state, step) pair so
           KV-cache decoders can extend it one token at a time, bit-identically.

Every code must satisfy, and its tests must check:
  * decodable: at every active position the maze's 2n(n-1) free connections are
    recoverable from features[b, p] (plus, for dynamic codes, the tokens <= p);
  * step == features: for dynamic codes, init_state(prefix) followed by step() on
    each next token reproduces features() row by row;
  * meta round trip: from_meta(code.meta()) rebuilds an identical code.

Timing (`code.start`, saved in the checkpoint as "start"):
  adjlist_end   (legacy) active from <ADJLIST_END> on
  origin_end    active from the <ORIGIN_END> token on (after the origin coordinate);
                the same rule for static and dynamic codes, so they are timing-matched.
Every generated token is active under both rules (generation is after <PATH_START>).

Checkpoint format (injector.final.pt / injector.iter_*.pt): {"W": [F, d], "code": name,
"code_meta": {...}, "start", "layer", "scale", ...}. Legacy edge/onion files carry "W_inj"
[2n^2, d] instead; load_injector() reads both.
"""

from __future__ import annotations

import torch

ADJLIST_END, TARGET_START, TARGET_END, ORIGIN_START, ORIGIN_END, PATH_START = 1, 2, 3, 4, 5, 6  # maze_dataset vocab ids
START_RULES = ("adjlist_end", "origin_end")


def start_positions(tokens: torch.Tensor, end_pos: torch.Tensor, start: str) -> torch.Tensor:
    """[B] long: first active position. adjlist_end -> end_pos; origin_end -> the first
    <ORIGIN_END> token after end_pos (end_pos + 3 in the AOTP tokenization; if the prefix
    ends before it, that same fallback, so an empty/short prefix is simply inactive)."""
    assert start in START_RULES, start
    end_pos = end_pos.long()
    if start == "adjlist_end":
        return end_pos
    if tokens.shape[1] == 0:
        return end_pos + 3
    pos = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
    hit = (tokens == ORIGIN_END) & (pos > end_pos.to(tokens.device)[:, None])
    first = torch.where(hit.any(1), hit.long().argmax(1), (end_pos.to(tokens.device) + 3))
    return first.to(end_pos.device)


def active_mask(start_pos: torch.Tensor, n_pos: int, attention_mask: torch.Tensor) -> torch.Tensor:
    """[B, P] bool: positions >= start_pos that are real tokens."""
    pos = torch.arange(n_pos, device=start_pos.device)[None, :]
    return (pos >= start_pos[:, None]) & attention_mask.bool().to(start_pos.device)


class GraftCode:
    name: str = "abstract"
    n_features: int
    dynamic: bool = False
    start: str = "adjlist_end"   # see START_RULES; set after construction (make_code(start=...), load_injector)

    # ---- static codes implement this ------------------------------------------------
    def rows(self, conn: torch.Tensor) -> torch.Tensor:
        """conn [B, 2n^2] bool -> [B, F] float feature vector of the maze."""
        raise NotImplementedError

    # ---- every code -------------------------------------------------------------------
    def features(self, tokens, attention_mask, conn, end_pos) -> torch.Tensor:
        """[B, P, F] float on conn.device; 0 wherever the graft is silent."""
        sp = start_positions(tokens, end_pos, self.start).to(conn.device)
        act = active_mask(sp, tokens.shape[1], attention_mask)
        return act[:, :, None].float() * self.rows(conn)[:, None, :]

    def init_state(self, tokens, attention_mask, conn, end_pos):
        """State after a prefix (same arguments as features()). Static codes keep rows(conn)
        plus whether the start token has already been seen in the prefix."""
        lens = attention_mask.long().sum(1).to(conn.device)
        sp = start_positions(tokens, end_pos, self.start).to(conn.device)
        return {"rows": self.rows(conn), "active": sp < lens}

    def step(self, state, new_tokens: torch.Tensor):
        """(features [B, F] for the position holding `new_tokens`, new state). new_tokens [B] long.
        adjlist_end: every generated position is active (legacy behaviour); origin_end: active
        from the first <ORIGIN_END> token on."""
        if self.start == "adjlist_end":
            return state["rows"], state
        active = state["active"] | (new_tokens.to(state["rows"].device) == ORIGIN_END)
        return state["rows"] * active[:, None].float(), dict(state, active=active)

    def expected_sq_norm(self) -> float:
        """Mean squared feature norm over training mazes; GraftInjector(norm_match=...) rescales so the
        injected norm matches the edge code's 2n^2. Default: the edge code's own value (also right for
        the onion code, whose C is already norm-matched); codes override it."""
        return float(2 * self.grid_n ** 2)

    def meta(self) -> dict:
        return {}

    @classmethod
    def from_meta(cls, grid_n: int, meta: dict) -> "GraftCode":
        return cls(grid_n)


class EdgeCode(GraftCode):
    """The original graft: one +/-1 feature per connection (connection_list.flatten())."""
    name = "edge"

    def __init__(self, grid_n: int):
        self.grid_n = grid_n
        self.n_features = 2 * grid_n * grid_n

    def rows(self, conn):
        return conn.float() * 2.0 - 1.0


class OnionCode(GraftCode):
    """maze/core/onion.py as a code: rows = signed @ C^T (one magnitude per tile)."""
    name = "onion"

    def __init__(self, grid_n: int, layout: str = "checker", base: float = 2.0, rotate: bool = True):
        from maze.core.onion import onion_code
        self.grid_n, self.layout, self.base, self.rotate = grid_n, layout, float(base), bool(rotate)
        self.C, self.depth = onion_code(grid_n, layout, base, rotate)
        self.n_features = self.C.shape[0]

    def rows(self, conn):
        return (conn.float() * 2.0 - 1.0) @ self.C.t().to(conn.device)

    def meta(self):
        return {"layout": self.layout, "base": self.base, "rotate": self.rotate}

    @classmethod
    def from_meta(cls, grid_n, meta):
        return cls(grid_n, meta["layout"], meta["base"], meta["rotate"])


CODES: dict[str, type] = {"edge": EdgeCode, "onion": OnionCode}


def register(cls):
    CODES[cls.name] = cls
    return cls


def make_code(name: str, grid_n: int, start: str = "adjlist_end", **kw) -> GraftCode:
    """name: edge | onion-checker | onion-full | vertex | degree | ego | axdeg | axdeg1h (see codes_*.py)."""
    if name.startswith("onion-"):
        code = OnionCode(grid_n, layout=name.removeprefix("onion-"), **kw)
    else:
        _import_codes()
        code = CODES[name](grid_n, **kw)
    assert start in START_RULES, start
    code.start = start
    return code


def _import_codes():
    for mod in ("maze.core.codes_vertex", "maze.core.codes_ego", "maze.core.codes_axdeg"):
        try:
            __import__(mod)
        except ImportError:
            pass


class GraftInjector(torch.nn.Module):
    """Any GraftCode -> residual stream. Same hook/scale/layer interface as MazeInjector."""

    def __init__(self, code: GraftCode, d_model: int, layer: int = 0, trainable: bool = False,
                 scale: float = 1.0, seed: int = 0, norm_match: float | None = None):
        super().__init__()
        self.code, self.layer, self.scale = code, layer, scale
        self.feature_mode = "complete"
        gen = torch.Generator().manual_seed(seed)
        w = torch.randn(code.n_features, d_model, generator=gen)
        w = w / w.norm(dim=-1, keepdim=True)
        self.W = torch.nn.Parameter(w, requires_grad=trainable)
        # norm_match: expected squared feature norm the code wants matched to the edge
        # code's 2n^2 (e.g. a one-hot code with k active features passes k)
        self.register_buffer("gain", torch.tensor(1.0 if norm_match is None else (2 * code.grid_n ** 2 / norm_match) ** 0.5))

    @property
    def hook_name(self):
        return f"blocks.{self.layer}.hook_resid_pre"

    def delta(self, features: torch.Tensor) -> torch.Tensor:
        """features [..., F] -> [..., d_model]"""
        return self.scale * self.gain * (features.to(self.W.dtype) @ self.W)

    def hook_fn(self, resid, hook, features):
        return resid + self.delta(features).to(resid.dtype)

    def state_for_ckpt(self) -> dict:
        return {"W": self.W.detach().cpu(), "code": self.code.name, "code_meta": self.code.meta(), "start": self.code.start,
                "grid_n": self.code.grid_n, "gain": float(self.gain), "layer": self.layer, "scale": self.scale}


# ---- adapters so every evaluator can treat the legacy injectors the same way --------------
class _LegacyLinear:
    """Wraps MazeInjector / OnionInjector (W_inj on 2n^2 signed features) as code + delta."""

    def __init__(self, inj):
        self.inj = inj
        n = int((inj.W_inj.shape[0] // 2) ** 0.5)
        self.code = EdgeCode(n)
        self.layer, self.scale = int(inj.layer), float(inj.scale)

    def delta(self, features):
        return self.inj.scale * (features.to(self.inj.W_inj.dtype) @ self.inj.W_inj)


def as_graft(injector):
    """-> object with .code (GraftCode), .delta(features), .layer; None stays None."""
    if injector is None or isinstance(injector, GraftInjector) or (hasattr(injector, "code") and hasattr(injector, "delta")):
        return injector          # GraftInjector, or any duck-typed wrapper (e.g. eval_ablate.CondGraft)
    return _LegacyLinear(injector)


def load_injector(path, device="cpu"):
    """Read injector.*.pt of any format -> GraftInjector (new) or MazeInjector (legacy W_inj)."""
    d = torch.load(path, map_location="cpu", weights_only=True)
    if "W" in d and "code" in d:
        code = make_code(d["code"], int(d["grid_n"]), **d.get("code_meta", {})) if not str(d["code"]).startswith("onion") \
            else OnionCode.from_meta(int(d["grid_n"]), d["code_meta"])
        code.start = str(d.get("start", "adjlist_end"))
        inj = GraftInjector(code, d["W"].shape[1], layer=int(d.get("layer", 0)), scale=float(d.get("scale", 1.0)))
        inj.W.data = d["W"].clone()
        inj.gain.fill_(float(d.get("gain", 1.0)))
        return inj.to(device)
    from maze.core.injection import MazeInjector
    w = d["W_inj"]
    inj = MazeInjector(d_model=w.shape[1], grid_n=int((w.shape[0] // 2) ** 0.5),
                       layer=int(d.get("layer", 0)), scale=float(d.get("scale", 1.0)))
    inj.W_inj.data = w.clone()
    return inj.to(device)
