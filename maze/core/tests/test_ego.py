#!/usr/bin/env python3
"""Checks for the egocentric graft code (maze/core/codes_ego.py).

  1. reference: a plain-Python, independently written per-(b, p) implementation
     of the frame rule and the relative index reproduces features() bit-exactly
     on real 6x6 and 5x5 mazes, on intact AND blanked (`dummy`) prompts
  2. frame rule: undefined (all-zero) before the origin token, = origin through
     <ORIGIN_END> <TARGET_START> (target!) <TARGET_END> <PATH_START>, then each
     path coordinate; <PATH_END> keeps the last frame, padding is zero
  3. decodable: exactly 2n^2 nonzeros at every framed position and inverting
     (relative index + frame -> absolute) returns `conn`; the same wall moves to
     a different feature index as the solver walks
  4. step == features, token by token, including across the target segment
  5. GraftInjector smoke + meta round trip + norm match
  6. at a fixed frame the ego code is an index permutation of the edge code
  7. the `window` locality variant matches the reference and its expected_sq_norm
"""

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))
import maze._pathsetup  # noqa: E402,F401  (cross-experiment sys.path shim)

import re  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from maze_dataset import MazeDataset, MazeDatasetConfig, SPECIAL_TOKENS  # noqa: E402
from maze_dataset.generation.generators import LatticeMazeGenerators  # noqa: E402
from maze_transformer.training.config import (  # noqa: E402
    TRAINING_CONFIGS, BaseGPTConfig, ConfigHolder,
)

from maze.core.codes_ego import EgoCode  # noqa: E402
from maze.core.data import (  # noqa: E402
    PretokDataset, collate_pretok, dummy_adjacency, pretokenize,
)
from maze.core.graft import EdgeCode, GraftInjector, make_code  # noqa: E402

COORD_RE = re.compile(r"^\((\d+),(\d+)\)$")
N_MAZES = 64


# --------------------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------------------
def build(grid_n: int, n_mazes: int = N_MAZES, seed: int = 42):
    cfg = ConfigHolder(
        name="test",
        dataset_cfg=MazeDatasetConfig(
            name="t", grid_n=grid_n, n_mazes=n_mazes,
            maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed,
        ),
        model_cfg=BaseGPTConfig(name="tiny", act_fn="gelu", d_model=64, d_head=16, n_layers=2),
        train_cfg=TRAINING_CONFIGS["test-v1"],
    )
    ds = MazeDataset.from_config(
        cfg.dataset_cfg, do_generate=True, load_local=False, save_local=False,
        gen_parallel=False,
    )
    vocab = cfg.tokenizer.get_vocab()
    out = Path(tempfile.mkdtemp()) / "pretok-test.pt"
    pretokenize(ds, cfg.maze_tokenizer, vocab, out,
                bos_id=int(cfg.tokenizer.bos_token_id), workers=0)
    pds = PretokDataset(out)
    batch = collate_pretok([pds[i] for i in range(len(pds))], vocab[SPECIAL_TOKENS.PADDING])
    id2c = {}
    for tok, idx in vocab.items():
        m = COORD_RE.match(tok)
        if m:
            id2c[int(idx)] = (int(m.group(1)), int(m.group(2)))
    return cfg, vocab, id2c, batch


def blank(batch, semicolon_id: int):
    """The `dummy` variant: adjacency list -> `;`.  Everything from <ADJLIST_END>
    on is untouched, so the ego code must be bit-identical."""
    tokens, am, conn, end_pos = batch
    tok = torch.stack([dummy_adjacency(tokens[b], int(end_pos[b]), semicolon_id)
                       for b in range(tokens.shape[0])])
    return tok, am, conn, end_pos


# --------------------------------------------------------------------------------------
# independent plain-Python reference (shares NO index helper with codes_ego.py)
# --------------------------------------------------------------------------------------
TARGET_START_ID, TARGET_END_ID = 2, 3


def ref_row(conn_row, frame, n, window=None):
    """The feature vector of one maze seen from cell `frame`, written from scratch."""
    m = 2 * n - 1
    out = [0.0] * (2 * m * m)
    r0, c0 = frame
    for kind in (0, 1):
        for r in range(n):
            for c in range(n):
                dr, dc = r - r0, c - c0
                if window is not None and max(abs(dr), abs(dc)) > window:
                    continue
                absolute = kind * n * n + r * n + c
                relative = kind * m * m + (dr + n - 1) * m + (dc + n - 1)
                out[relative] = 1.0 if conn_row[absolute] else -1.0
    return out


def ref_one(row, mask, conn_row, end_pos, n, id2c, window=None):
    """-> (features [P][F], frame per position (None = undefined)) for one maze."""
    m = 2 * n - 1
    feats = [[0.0] * (2 * m * m) for _ in range(len(row))]
    frames = [None] * len(row)
    frame, depth = None, 0
    for p, t in enumerate(row):
        if t == TARGET_START_ID:
            depth += 1
        elif t == TARGET_END_ID:
            depth -= 1
        inside_target = depth > 0
        if mask[p] and p > end_pos and (t in id2c) and not inside_target:
            frame = id2c[t]          # the frame moves only here
        if (not mask[p]) or frame is None:
            continue                 # graft silent: no evidence
        frames[p] = frame
        feats[p] = ref_row(conn_row, frame, n, window)
    return feats, frames


def ref_batch(batch, n, id2c, window=None):
    tokens, am, conn, end_pos = batch
    toks, masks, conns = tokens.tolist(), am.tolist(), conn.tolist()
    feats, frames = [], []
    for b in range(tokens.shape[0]):
        f, fr = ref_one(toks[b], masks[b], conns[b], int(end_pos[b]), n, id2c, window)
        feats.append(f)
        frames.append(fr)
    return torch.tensor(feats, dtype=torch.float32), frames


# --------------------------------------------------------------------------------------
# 1. reference equality
# --------------------------------------------------------------------------------------
DATA = {}
for n in (6, 5):
    cfg, vocab, id2c, batch = build(n)
    DATA[n] = (cfg, vocab, id2c, batch)
    code = EgoCode(n, id2c)
    assert code.n_features == 2 * (2 * n - 1) ** 2, code.n_features
    assert abs(code.expected_sq_norm() - 2 * n * n) < 1e-9
    semi = vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE]
    for tag, bat in (("intact", batch), ("blanked", blank(batch, semi))):
        got = code.features(*bat)
        want, _ = ref_batch(bat, n, id2c)
        assert got.shape == want.shape == (bat[0].shape[0], bat[0].shape[1], code.n_features)
        assert torch.equal(got, want), f"n={n} {tag}: features != reference"
    assert torch.equal(code.features(*batch), code.features(*blank(batch, semi))), \
        "blanking the adjacency list changed the ego features"
    print(f"1 n={n}: {batch[0].shape[0]} mazes, P={batch[0].shape[1]}, F={code.n_features} -- "
          f"features == plain-Python reference on intact and blanked prompts OK")

# --------------------------------------------------------------------------------------
# 2. frame rule, position by position
# --------------------------------------------------------------------------------------
n = 6
cfg, vocab, id2c, batch = DATA[n]
tokens, am, conn, end_pos = batch
code = EgoCode(n, id2c)
feats = code.features(*batch)
inv_vocab = {v: k for k, v in vocab.items()}
b = 0
row = tokens[b].tolist()
ep, length = int(end_pos[b]), int(am[b].sum())
assert row[ep] == 1 and row[ep + 1] == 4 and row[ep + 3] == 5, "unexpected header layout"
assert row[ep + 4] == 2 and row[ep + 6] == 3 and row[ep + 7] == 6, "unexpected header layout"
origin, target = id2c[row[ep + 2]], id2c[row[ep + 5]]
assert origin != target
conn_row = conn[b].tolist()

assert feats[b, : ep + 2].abs().sum() == 0, "frame defined before the origin token"
want_origin = torch.tensor(ref_row(conn_row, origin, n))
want_target = torch.tensor(ref_row(conn_row, target, n))
assert not torch.equal(want_origin, want_target)
for p in range(ep + 2, ep + 8):   # origin, <ORIGIN_END>, <TARGET_START>, target, <TARGET_END>, <PATH_START>
    assert torch.equal(feats[b, p], want_origin), (p, inv_vocab[row[p]])
assert not torch.equal(feats[b, ep + 5], want_target), "the target coordinate moved the frame"

path_cells = []
last = origin
for p in range(ep + 8, length):
    t = row[p]
    if t in id2c:
        last = id2c[t]
        path_cells.append((p, last))
    else:
        assert t == 7, inv_vocab[t]        # <PATH_END> keeps the last frame
    assert torch.equal(feats[b, p], torch.tensor(ref_row(conn_row, last, n))), p
assert feats[b, length:].abs().sum() == 0, "padded positions are not silent"
assert id2c[row[ep + 8]] == origin, "the first path token should be the origin again"
print(f"2 frame rule OK: undefined < {ep + 2}, origin through <PATH_START> (target skipped), "
      f"{len(path_cells)} path cells, <PATH_END> holds, padding {length}..{tokens.shape[1]} zero")

# --------------------------------------------------------------------------------------
# 3. decodability
# --------------------------------------------------------------------------------------
_, frames = ref_batch(batch, n, id2c)
m = 2 * n - 1
n_defined = 0
for bb in range(tokens.shape[0]):
    for p, fr in enumerate(frames[bb]):
        if fr is None:
            assert feats[bb, p].abs().sum() == 0
            continue
        n_defined += 1
        assert int((feats[bb, p] != 0).sum()) == 2 * n * n, (bb, p, int((feats[bb, p] != 0).sum()))
for bb in range(8):                       # full inversion on a subset
    cr = conn[bb].tolist()
    for p, fr in enumerate(frames[bb]):
        if fr is None:
            continue
        r0, c0 = fr
        rec = []
        for kind in (0, 1):
            for r in range(n):
                for c in range(n):
                    rel = kind * m * m + (r - r0 + n - 1) * m + (c - c0 + n - 1)
                    rec.append(float(feats[bb, p, rel]) > 0)
        assert rec == [bool(x) for x in cr], (bb, p)
print(f"3 decodability OK: {n_defined} framed positions, exactly {2 * n * n} nonzeros each, "
      f"inverting relative index + frame reproduces conn on 8 mazes")

walls = [f for f in range(2 * n * n) if not conn_row[f]]
(p1, c1), (p2, c2) = path_cells[0], path_cells[1]
assert c1 != c2
f_wall = walls[0]
kind, r, c = f_wall // (n * n), (f_wall % (n * n)) // n, f_wall % n
i1 = kind * m * m + (r - c1[0] + n - 1) * m + (c - c1[1] + n - 1)
i2 = kind * m * m + (r - c2[0] + n - 1) * m + (c - c2[1] + n - 1)
assert i1 != i2 and feats[b, p1, i1] == -1 and feats[b, p2, i2] == -1
print(f"3 moving frame: the {'down' if kind == 0 else 'right'} wall of cell {(r, c)} is feature "
      f"{i1} at path position {p1} (frame {c1}) and feature {i2} at {p2} (frame {c2})")

# --------------------------------------------------------------------------------------
# 4. step() == features(), token by token
# --------------------------------------------------------------------------------------
assert int(end_pos.min()) == int(end_pos.max()), "test assumes one common end_pos"
lens = am.long().sum(1)
for k, label in ((ep + 8, "prefix ends at <PATH_START>+origin"),
                 (ep + 6, "prefix ends INSIDE the target segment")):
    state = code.init_state(tokens[:, :k], am[:, :k], conn, end_pos)
    checked = 0
    for p in range(k, int(lens.max())):
        got, state = code.step(state, tokens[:, p])
        live = (p < lens).nonzero().flatten()
        assert torch.equal(got[live], feats[live, p]), (label, p)
        checked += len(live)
    print(f"4 step == features OK ({label}, k={k}): {checked} (row, position) pairs")

# --------------------------------------------------------------------------------------
# 5. GraftInjector smoke, registration, meta round trip
# --------------------------------------------------------------------------------------
made = make_code("ego", n, id_to_coord=id2c)
assert isinstance(made, EgoCode) and made.name == "ego" and made.dynamic
rt = EgoCode.from_meta(n, made.meta())
assert torch.equal(rt.features(*batch), feats) and rt.meta() == made.meta()
inj = GraftInjector(made, d_model=32, layer=0, seed=0, norm_match=made.expected_sq_norm())
assert abs(float(inj.gain) - 1.0) < 1e-6, float(inj.gain)
delta = inj.delta(feats)
assert delta.shape == (tokens.shape[0], tokens.shape[1], 32), delta.shape
assert torch.allclose(delta[b, ep + 2], delta[b, ep + 7]), "same frame -> different injection"
assert not torch.allclose(delta[b, p1], delta[b, p2]), "different frame -> same injection"
assert delta[b, ep].abs().sum() == 0 and delta[b, length:].abs().sum() == 0
print(f"5 GraftInjector OK: delta {tuple(delta.shape)}, gain {float(inj.gain):.3f}, "
      f"meta round trip exact, injection constant on the fixed-frame header and moving on the path")

# --------------------------------------------------------------------------------------
# 6. fixed frame == a permutation of the edge code
# --------------------------------------------------------------------------------------
edge = EdgeCode(n).rows(conn)
fixed = torch.arange(ep + 2, ep + 8)      # every row's frame here is its own origin
for bb in range(tokens.shape[0]):
    r0, c0 = id2c[int(tokens[bb, int(end_pos[bb]) + 2])]
    perm = [((f // (n * n)) * m * m + ((f % (n * n)) // n - r0 + n - 1) * m + (f % n - c0 + n - 1))
            for f in range(2 * n * n)]
    assert len(set(perm)) == 2 * n * n
    gathered = feats[bb][:, torch.tensor(perm)][fixed]
    assert torch.equal(gathered, edge[bb][None, :].expand(len(fixed), -1))
print(f"6 fixed-frame equivalence OK: re-gathered to absolute order, positions {ep + 2}..{ep + 7} "
      f"equal EdgeCode({n}).rows(conn) on all {tokens.shape[0]} mazes")

# --------------------------------------------------------------------------------------
# 7. window locality variant
# --------------------------------------------------------------------------------------
for w in (1, 2):
    cw = EgoCode(n, id2c, window=w)
    small = tuple(x[:8] for x in batch)
    got = cw.features(*small)
    want, fr_small = ref_batch(small, n, id2c, window=w)
    assert torch.equal(got, want), f"window={w}: features != reference"
    counts = [int((got[bb, p] != 0).sum()) for bb in range(8)
              for p, f in enumerate(fr_small[bb]) if f is not None]
    per_cell = [len([1 for kind in (0, 1) for r in range(n) for c in range(n)
                     if max(abs(r - r0), abs(c - c0)) <= w])
                for r0 in range(n) for c0 in range(n)]
    assert abs(cw.expected_sq_norm() - sum(per_cell) / len(per_cell)) < 1e-6
    assert min(counts) == min(per_cell) and max(counts) == max(per_cell)
    gain = float(GraftInjector(cw, 32, norm_match=cw.expected_sq_norm()).gain)
    print(f"7 window={w} OK: nonzeros {min(counts)}..{max(counts)} per framed position, "
          f"expected_sq_norm {cw.expected_sq_norm():.2f} (whole maze {2 * n * n}), gain {gain:.3f}")

# --------------------------------------------------------------------------------------
# 8. ragged batches: end_pos and length varying across the batch, odd prefixes
# --------------------------------------------------------------------------------------
pad_id = vocab[SPECIAL_TOKENS.PADDING]
rows, new_len, new_ep = [], [], []
for bb in range(tokens.shape[0]):
    drop = (bb * 3) % 17                    # cut `drop` tokens out of the adjacency list
    r, L, e = tokens[bb].tolist(), int(lens[bb]), int(end_pos[bb])
    body = r[:1] + r[1 + drop: L]
    rows.append(body + [pad_id] * (tokens.shape[1] - len(body)))
    new_len.append(len(body))
    new_ep.append(e - drop)
rag = (torch.tensor(rows),
       (torch.arange(tokens.shape[1])[None, :] < torch.tensor(new_len)[:, None]).long(),
       conn, torch.tensor(new_ep))
assert (rag[0].gather(1, rag[3][:, None]).squeeze(1) == 1).all(), "cut broke <ADJLIST_END>"
rag_feats = code.features(*rag)
rag_want, _ = ref_batch(rag, n, id2c)
assert torch.equal(rag_feats, rag_want), "ragged batch: features != reference"
rag_len = torch.tensor(new_len)
for start in (0, int(rag[3].min()), int(rag[3].max()) + 8):   # empty / pre-<ADJLIST_END> / path prefix
    st = code.init_state(rag[0][:, :start], rag[1][:, :start], conn, rag[3])
    if start <= int(rag[3].min()):
        assert (st["frame"] == -1).all(), "frame defined from the adjacency list alone"
    for p in range(start, int(rag_len.max())):
        got, st = code.step(st, rag[0][:, p])
        live = (p < rag_len).nonzero().flatten()
        assert torch.equal(got[live], rag_feats[live, p]), (start, p)
cw = EgoCode(n, id2c, window=2)
cw_feats = cw.features(*rag)
st = cw.init_state(rag[0][:, :3], rag[1][:, :3], conn, rag[3])
for p in range(3, int(rag_len.max())):
    got, st = cw.step(st, rag[0][:, p])
    live = (p < rag_len).nonzero().flatten()
    assert torch.equal(got[live], cw_feats[live, p]), p
print(f"8 ragged OK: end_pos {int(rag[3].min())}..{int(rag[3].max())}, lengths "
      f"{min(new_len)}..{max(new_len)}; features == reference and step == features from an "
      f"empty, a pre-<ADJLIST_END> and a mid-path prefix (windowed code too)")

print("ALL OK")
