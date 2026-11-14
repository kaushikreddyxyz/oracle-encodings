#!/usr/bin/env python3
"""ego code: KV-cache greedy decoding (init_state/step) must agree with the full teacher-forced forward
(features()) on its own generated sequence, on a real (smoke-trained) checkpoint. Usage:
  python maze/core/tests/test_ego_kv_decode.py <run_dir with model.final.pt + injector.final.pt (code=ego)>"""
import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "maze" / "core").is_dir())))
import maze._pathsetup  # noqa
import numpy as np, torch
ROOT = _pl.Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(ROOT / "maze/paper/curves"))
import eval_ckpts as E
from maze.core.graft import as_graft

rd = _pl.Path(_sys.argv[1]); dev = torch.device("cpu")
model, inj = E.load_ckpt(rd / "model.final.pt", rd / "injector.final.pt", dev)
assert as_graft(inj).code.name == "ego"
D = np.load(ROOT / "maze/paper/curves/results/distance/sets/g6.npz"); B = 8
prompts = torch.tensor(D["prompts"][:B]); end = torch.tensor(D["end_pos"][:B]).long(); conn = torch.tensor(D["conn_a"][:B])
pe, pad = int(D["path_end_id"]), int(D["pad_id"])
dummy = bool(torch.load(rd / "injector.final.pt", weights_only=True).get("dummy_data", False))
if dummy:
    for b in range(B): prompts[b, 1:int(end[b])] = 9
max_new = 20
gen = E.greedy_kv(model, inj, 0, prompts, conn, end, max_new, pe, pad)
full = torch.cat([prompts, gen], 1); attn = torch.ones_like(full)
g = as_graft(inj); delta = g.delta(g.code.features(full, attn, conn, end))
with torch.no_grad():
    logits = model(full, attention_mask=attn, inject_delta=delta, inject_layer=0, padding_mode="right", return_type="logits")
P = prompts.shape[1]; ok = tot = 0
for b in range(B):
    for s in range(max_new):
        t = int(gen[b, s])
        if t == pad: break
        tot += 1; ok += int(logits[b, P + s - 1].argmax()) == t
        if t == pe: break
print(f"ego KV vs full forward: {ok}/{tot} generated tokens agree (dummy prompt: {dummy})")
assert ok == tot and tot > B
print("ALL OK")
