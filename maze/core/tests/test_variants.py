#!/usr/bin/env python3
"""CPU smoke tests for the three injection variants (VARIANTS_SPEC.md).

  1. complete mode is byte-identical to the pre-variant make_features path.
  2. window: features are the true +/-1 exactly on [end_pos, end_pos+K) and
     zero everywhere else.
  3. dummy: the transform touches exactly positions 1..end_pos-1 and leaves
     BOS, <ADJLIST_END> and the origin/target/path sections untouched; conn and
     end_pos survive a file round-trip.
  4. teacher: belief tensor has the right shape, lies in (-1,1) strictly inside
     the adjacency list, is exactly 0 at p=0 and p>=end_pos, and the assembled
     features are exactly the true +/-1 from end_pos on.
  5. a tiny CPU model runs a forward + backward through every mode's hooks.

  python test_variants.py
"""

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))
import maze._pathsetup  # noqa: E402,F401  (cross-experiment sys.path shim)

import torch
from maze_dataset import SPECIAL_TOKENS, MazeDataset, MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators
from maze_transformer.training.config import (
    TRAINING_CONFIGS,
    BaseGPTConfig,
    ConfigHolder,
)

from maze.core.data import (
    PretokDataset,
    collate_pretok,
    dummify_pretok,
    dummy_adjacency,
    make_features,
    pretokenize,
)
from maze.core.injection import InjectionRuntime, MazeInjector, as_runtime
from maze.core.teacher import MazeTeacher

GRID_N = 6
N_FEATS = 2 * GRID_N * GRID_N
N_MAZES = 24
D_MODEL = 64
WINDOW_K = 4

cfg = ConfigHolder(
    name="test-variants",
    dataset_cfg=MazeDatasetConfig(
        name="tv", grid_n=GRID_N, n_mazes=N_MAZES,
        maze_ctor=LatticeMazeGenerators.gen_dfs, seed=7,
    ),
    model_cfg=BaseGPTConfig(
        name="tiny", act_fn="gelu", d_model=D_MODEL, d_head=16, n_layers=2,
    ),
    train_cfg=TRAINING_CONFIGS["test-v1"],
)
device = torch.device("cpu")
tokenizer = cfg.tokenizer
vocab = tokenizer.get_vocab()
pad_id = vocab[SPECIAL_TOKENS.PADDING]
semicolon_id = vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE]
adjlist_end_id = vocab[SPECIAL_TOKENS.ADJLIST_END]

dataset = MazeDataset.from_config(
    cfg.dataset_cfg, do_generate=True, load_local=False, save_local=False,
)

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

tmp = Path(tempfile.mkdtemp())
pretokenize(
    dataset, cfg.maze_tokenizer, vocab, tmp / "pretok-test.pt",
    bos_id=int(tokenizer.bos_token_id), workers=0,
)
pretok = PretokDataset(tmp / "pretok-test.pt")
tokens, attn, conn, end_pos = collate_pretok(
    [pretok[i] for i in range(len(pretok))], pad_id=pad_id
)
n_pos = tokens.shape[1]
signed = conn.float() * 2.0 - 1.0

# ---------------------------------------------------------------- 1. complete
feats_complete = make_features(conn, end_pos, n_pos, attn)
ref = (
    (
        (torch.arange(n_pos)[None, :] >= end_pos[:, None]) & attn.bool()
    )[:, :, None].float()
    * signed[:, None, :]
)
assert torch.equal(feats_complete, ref), "complete mode changed"
assert torch.equal(
    make_features(conn, end_pos, n_pos, attn, window_k=None), feats_complete
), "window_k=None must be complete mode"
print(f"complete mode unchanged  ({tuple(feats_complete.shape)})")

# ------------------------------------------------------------------ 2. window
feats_win = make_features(conn, end_pos, n_pos, attn, window_k=WINDOW_K)
pos = torch.arange(n_pos)[None, :]
inside = (pos >= end_pos[:, None]) & (pos < end_pos[:, None] + WINDOW_K) & attn.bool()
assert torch.equal(
    feats_win[inside], feats_complete[inside]
), "window: features inside the window are not the true features"
assert float(feats_win[~inside].abs().sum()) == 0.0, "window: leak outside window"
assert (feats_win.abs().sum(-1) > 0).sum(1).unique().tolist() == [WINDOW_K], (
    "window: wrong number of injected positions"
)
assert float(feats_win[:, : int(end_pos.min())].abs().sum()) == 0.0, (
    "window: injected before <ADJLIST_END>"
)
print(f"window mask correct      (K={WINDOW_K}, {int(inside.sum())} live positions)")

# ------------------------------------------------------------------- 3. dummy
row0, ep0 = pretok[0][0].clone(), int(pretok[0][2])
dummied = dummy_adjacency(row0, ep0, semicolon_id)
assert torch.equal(dummied[0:1], row0[0:1]), "dummy: BOS modified"
assert bool((dummied[1:ep0] == semicolon_id).all()), "dummy: adjacency not blanked"
assert torch.equal(dummied[ep0:], row0[ep0:]), "dummy: post-<ADJLIST_END> modified"
assert int(dummied[ep0]) == adjlist_end_id, "dummy: <ADJLIST_END> clobbered"
changed = (dummied != row0).nonzero()[:, 0]
expected = torch.tensor(
    [p for p in range(1, ep0) if int(row0[p]) != semicolon_id]
)
assert torch.equal(changed, expected), "dummy: touched positions != 1..end_pos-1"

dummify_pretok(tmp / "pretok-test.pt", tmp / "pretok-dummy-test.pt", semicolon_id)
dpretok = PretokDataset(tmp / "pretok-dummy-test.pt")
assert torch.equal(dpretok.conn, pretok.conn), "dummy: conn changed"
assert torch.equal(dpretok.end_pos, pretok.end_pos), "dummy: end_pos changed"
assert torch.equal(dpretok[0][0], dummied), "dummy: file transform != row transform"
lens = {len(dpretok[i][0]) for i in range(len(dpretok))}
assert lens == {len(pretok[i][0]) for i in range(len(pretok))}, "dummy: length leak"
d_tokens, d_attn, d_conn, d_end = collate_pretok(
    [dpretok[i] for i in range(len(dpretok))], pad_id=pad_id
)
assert torch.equal(
    make_features(d_conn, d_end, d_tokens.shape[1], d_attn), feats_complete
), "dummy: features must be untouched (complete-mode injection)"
print(f"dummy transform correct  ({len(dpretok)} mazes, end_pos {ep0})")

# ----------------------------------------------------------------- 4. teacher
model = cfg.create_model_zanj().to(device)
torch.manual_seed(0)
probe_W = torch.randn(N_FEATS, D_MODEL) * 0.1
probe_b = torch.zeros(N_FEATS)
teacher = MazeTeacher(model, probe_W, probe_b, layer=1)
beliefs = teacher.beliefs(tokens, attn, end_pos)
assert beliefs.shape == (len(conn), n_pos, N_FEATS), "teacher: bad belief shape"
assert float(beliefs[:, 0].abs().sum()) == 0.0, "teacher: BOS got a belief"
post = pos >= end_pos[:, None]
assert float(beliefs[post].abs().sum()) == 0.0, "teacher: belief at p >= end_pos"
mid = (pos >= 1) & (pos < end_pos[:, None])
assert bool((beliefs[mid].abs() < 1.0).all()), "teacher: belief outside (-1,1)"
assert float(beliefs[mid].abs().min()) > 0.0, "teacher: belief identically zero"

rt_teacher = InjectionRuntime(
    MazeInjector(d_model=D_MODEL, grid_n=GRID_N), mode="teacher", teacher=teacher
)
feats_t = rt_teacher.features(tokens, attn, conn, end_pos)
live = post & attn.bool()
assert torch.equal(feats_t[live], signed[:, None, :].expand(-1, n_pos, -1)[live]), (
    "teacher: positions >= end_pos are not exactly the true +/-1"
)
assert torch.equal(feats_t[mid], beliefs[mid]), "teacher: mid-adjacency != beliefs"
# beliefs depend only on the prefix -> cached across rollout steps
cut = n_pos - 3
b2 = teacher.beliefs(tokens[:, :cut], attn[:, :cut], end_pos)
assert torch.equal(
    b2[mid[:, :cut]], beliefs[:, :cut][mid[:, :cut]]
), "teacher: beliefs changed with the generated suffix"
print(f"teacher beliefs OK       (L{teacher.layer}, |s| in "
      f"[{float(beliefs[mid].abs().min()):.2e}, {float(beliefs[mid].abs().max()):.4f}])")

# --------------------------------------------------------- 5. forward + hooks
runtimes = {
    "complete": as_runtime(MazeInjector(d_model=D_MODEL, grid_n=GRID_N)),
    "window": InjectionRuntime(
        MazeInjector(d_model=D_MODEL, grid_n=GRID_N), mode="window",
        window_k=WINDOW_K,
    ),
    "teacher": rt_teacher,
    "dummy": as_runtime(MazeInjector(d_model=D_MODEL, grid_n=GRID_N)),
}
with torch.no_grad():
    plain = model(tokens, return_type="logits", attention_mask=attn)
for name, rt in runtimes.items():
    tk, at, cn, ep = (
        (d_tokens, d_attn, d_conn, d_end) if name == "dummy"
        else (tokens, attn, conn, end_pos)
    )
    rt.injector.W_inj.requires_grad_(True)
    loss = model.run_with_hooks(
        tk, return_type="loss", attention_mask=at,
        fwd_hooks=rt.fwd_hooks(tk, at, cn, ep),
    )
    loss.backward()
    assert rt.injector.W_inj.grad is not None, f"{name}: no gradient to W_inj"
    assert float(rt.injector.W_inj.grad.abs().sum()) > 0, f"{name}: zero gradient"
    with torch.no_grad():
        logits = model.run_with_hooks(
            tk, return_type="logits", attention_mask=at,
            fwd_hooks=rt.fwd_hooks(tk, at, cn, ep),
        )
    delta = float((logits - plain).abs().max())
    assert delta > 0, f"{name}: hook had no effect"
    print(f"{name:9s} forward+backward OK  loss={float(loss.detach()):.4f} "
          f"max|dlogit|={delta:.3f}")

# ------------------------------------------------- 6. checkpoint -> eval path
from zanj import ZANJ  # noqa: E402

from maze.core.checkpoints import load_checkpoint  # noqa: E402
from maze.core.evals import ProbeSet, _forward, greedy_rollout  # noqa: E402

model.zanj_model_config = cfg  # what train_fast saves
ZANJ().save(model, tmp / "model.final.zanj")
torch.save({"W_inj": runtimes["complete"].injector.W_inj.detach()},
           tmp / "injector.legacy.pt")  # pre-variant checkpoint: no metadata
for nm, meta in (
    ("window", {"mode": "window", "window_k": WINDOW_K, "layer": 0, "scale": 1.0,
                "teacher_model": None, "teacher_probe": None}),
    ("teacher", {"mode": "teacher", "window_k": WINDOW_K, "layer": 0,
                 "scale": 1.0,
                 "teacher_model": str(tmp / "model.final.zanj"),
                 "teacher_probe": str(tmp / "probe.pt")}),
):
    torch.save({"W_inj": runtimes[nm].injector.W_inj.detach(), **meta},
               tmp / f"injector.{nm}.pt")
torch.save(dict(W=probe_W, b=probe_b, layer=1), tmp / "probe.pt")

ps = ProbeSet([pretok[i] for i in range(len(pretok))], vocab, GRID_N, device)
for nm, expect in (("legacy", "complete"), ("window", "window"),
                   ("teacher", "teacher")):
    m, rt = load_checkpoint(
        tmp / "model.final.zanj", tmp / f"injector.{nm}.pt", device
    )
    assert rt.mode == expect, f"{nm}: mode not restored ({rt.mode})"
    if expect == "window":
        assert rt.window_k == WINDOW_K, "window_k not restored"
        assert torch.equal(
            rt.features(tokens, attn, conn, end_pos), feats_win
        ), "window: reloaded runtime features differ"
    if expect == "complete":
        assert torch.equal(
            rt.features(tokens, attn, conn, end_pos), feats_complete
        ), "legacy checkpoint must reload as exact complete mode"
    with torch.no_grad():
        _forward(m, rt, ps.tokens, ps.attn, ps.conn, ps.end_pos)
    rolls = greedy_rollout(m, rt, ps, ps.conn, max_new_tokens=3)
    print(f"{nm:9s} checkpoint -> mode={rt.mode} k={rt.window_k} "
          f"rollout ok ({len(rolls)} rows)")

print("ALL VARIANT TESTS PASSED")
