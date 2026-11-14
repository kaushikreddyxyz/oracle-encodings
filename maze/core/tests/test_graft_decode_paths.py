#!/usr/bin/env python3
"""Checkpoint-level: the two KV-cache decoders (eval_ckpts.greedy_kv and eval_local.decode with a GraftDriver) must
agree with the model's full teacher-forced forward on their own generated tokens, and with each other, for a
static code (axdeg) and the dynamic ego code, with the origin_end start rule and an intact prompt.
  python maze/core/tests/test_graft_decode_paths.py <run_dir_axdeg> <run_dir_ego>"""
import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "maze" / "core").is_dir())))
import maze._pathsetup  # noqa
import numpy as np, torch
ROOT = _pl.Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(ROOT / "maze/paper/curves"))
import eval_ckpts as E, eval_local as L
from maze.core.graft import as_graft, start_positions

D = np.load(ROOT / "maze/paper/curves/results/distance/sets/g6.npz"); B = 8; dev = torch.device("cpu")
prompts = torch.tensor(D["prompts"][:B]); end = torch.tensor(D["end_pos"][:B]).long(); conn = torch.tensor(D["conn_a"][:B])
pe, pad = int(D["path_end_id"]), int(D["pad_id"]); max_new = 16
for rd in map(_pl.Path, _sys.argv[1:]):
    model, inj = E.load_ckpt(rd / "model.final.pt", rd / "injector.final.pt", dev); g = as_graft(inj)
    assert g.code.start == "origin_end", g.code.start
    gen = E.greedy_kv(model, inj, 0, prompts, conn, end, max_new, pe, pad)
    full = torch.cat([prompts, gen], 1); attn = torch.ones_like(full)
    feats = g.code.features(full, attn, conn, end)
    sp = start_positions(full, end, "origin_end"); pos = torch.arange(full.shape[1])[None]
    assert (feats[pos < sp[:, None]] == 0).all() and (feats[(pos >= sp[:, None])].abs().sum(-1) > 0).all()
    with torch.no_grad():
        logits = model(full, attention_mask=attn, inject_delta=g.delta(feats), inject_layer=0, padding_mode="right", return_type="logits")
    P = prompts.shape[1]; ok = tot = 0
    for b in range(B):
        for s in range(max_new):
            t = int(gen[b, s])
            if t == pad: break
            tot += 1; ok += int(logits[b, P + s - 1].argmax()) == t
            if t == pe: break
    assert ok == tot and tot > B, (ok, tot)
    lens = torch.full((B,), P, dtype=torch.long)
    logp0, gen2 = L.decode(model, prompts.clone(), lens, end, max_new, pe, pad, None, L.GraftDriver(inj, conn.numpy(), dev))
    assert torch.equal(gen2, gen), "eval_local.decode and greedy_kv disagree"
    ref = logits[:, P - 1].float().log_softmax(-1)
    assert (logp0 - ref).abs().max() < 1e-4, float((logp0 - ref).abs().max())
    pre = torch.cat([prompts, gen[:, :3]], 1); lens3 = torch.full((B,), P + 3, dtype=torch.long)
    logp3, gen3 = L.decode(model, pre.clone(), lens3, end, max_new - 3, pe, pad, None, L.GraftDriver(inj, conn.numpy(), dev))
    for b in range(B):
        if pe in gen[b, :3].tolist(): continue
        assert torch.equal(gen3[b], gen[b, 3:]), b
        assert (logp3[b] - logits[b, P + 2].float().log_softmax(-1)).abs().max() < 1e-4
    print(f"{g.code.name:6s} origin_end: greedy_kv == full forward ({ok}/{tot}); eval_local.decode == greedy_kv; prefix-tracked decode OK")
print("ALL OK")
