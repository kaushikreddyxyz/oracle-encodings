#!/usr/bin/env python3
"""The code-agnostic evaluator path must reproduce the legacy `signed @ W_inj` path bit-for-bit
(CPU, real 6x6 checkpoints from maze/paper/onion/results):
  1. Scorer.logits (teacher-forced) old == new, for a legacy edge injector and a legacy onion injector
  2. greedy_kv old == new (same generated tokens, identical logits at every step)
  3. a GraftInjector(EdgeCode) carrying the same W gives the same delta as the legacy MazeInjector
"""
import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "maze" / "core").is_dir())))
import maze._pathsetup  # noqa
import importlib.util, glob
import numpy as np, torch
from maze.core.graft import EdgeCode, GraftInjector, load_injector, as_graft

ROOT = _pl.Path(__file__).resolve().parents[3]
CUR = ROOT / "maze/paper/curves"
_sys.path.insert(0, str(CUR))
import eval_ckpts as new
spec = importlib.util.spec_from_file_location("old", 
                                              str(_pl.Path(__file__).resolve().parent / "_ref" / "eval_ckpts_legacy.py"))
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

S = np.load(CUR / "results/distance/sets/g6.npz")
B = 12
prompts = torch.tensor(S["prompts"][:B]); end = torch.tensor(S["end_pos"][:B], dtype=torch.long)
conn = torch.tensor(S["conn_a"][:B]); attn = torch.ones_like(prompts)
pe, pad = int(S["path_end_id"]), int(S["pad_id"])
dev = torch.device("cpu")
runs = {
    "edge": glob.glob(str(ROOT / "maze/paper/onion/results/runs/g6-s0-edge-dummy/fast-inj-fixed-*/"))[0],
    "onion": glob.glob(str(ROOT / "maze/paper/onion/results/runs/g6-s0-onion-checker/fast-inj-fixed-*/"))[0],
}
for name, rd in runs.items():
    rd = _pl.Path(rd)
    model, inj_old = old.load_ckpt(rd / "model.final.pt", rd / "injector.final.pt", dev)
    _, inj_new = new.load_ckpt(rd / "model.final.pt", rd / "injector.final.pt", dev)
    assert torch.equal(inj_old.W_inj, inj_new.W_inj)
    lo = old.Scorer(model, inj_old, dev).logits(prompts, attn, conn, end)
    ln = new.Scorer(model, inj_new, dev).logits(prompts, attn, conn, end)
    assert torch.equal(lo, ln), (name, (lo - ln).abs().max())
    go = old.greedy_kv(model, inj_old, 0, prompts, conn, end, 40, pe, pad)
    gn = new.greedy_kv(model, inj_new, 0, prompts, conn, end, 40, pe, pad)
    assert torch.equal(go, gn), name
    if name == "edge":
        gi = GraftInjector(EdgeCode(6), 256); gi.W.data = inj_old.W_inj.data.clone()
        g = as_graft(gi); f = g.code.features(prompts, attn, conn, end)
        d_new = g.delta(f); d_old = as_graft(inj_old).delta(f)
        assert torch.allclose(d_new, d_old, atol=1e-6)
        gg = new.greedy_kv(model, gi, 0, prompts, conn, end, 40, pe, pad)
        assert torch.equal(gg, go)
    print(f"{name}: logits equal, greedy_kv equal ({(go != pad).sum().item()} generated tokens over {B} mazes)")
print("ALL OK")
