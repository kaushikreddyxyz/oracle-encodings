#!/usr/bin/env python3
"""Timing rule `start` (maze/core/graft.py): adjlist_end (legacy) vs origin_end, for a static code (axdeg) and
the dynamic ego code, on real tokenized 6x6 mazes (intact and blanked prompts):
  1. origin_end: features are 0 at <ADJLIST_END>, <ORIGIN_START>, the origin coordinate, and equal rows(conn) /
     the origin-framed ego features from the <ORIGIN_END> token on; adjlist_end is unchanged from the legacy mask
  2. init_state + step reproduce features() token by token from prefixes ending before <ORIGIN_END>, at it, after it,
     and mid-path, for both codes and both rules
  3. meta round trip keeps `start`; make_code(start=...) sets it
"""
import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "maze" / "core").is_dir())))
import maze._pathsetup  # noqa
import re, tempfile, torch
from maze_dataset import MazeDataset, MazeDatasetConfig, SPECIAL_TOKENS
from maze_dataset.generation.generators import LatticeMazeGenerators
from maze_transformer.training.config import BaseGPTConfig, ConfigHolder, TRAINING_CONFIGS
from maze.core.data import PretokDataset, collate_pretok, pretokenize, dummy_adjacency
from maze.core.graft import make_code, GraftInjector, load_injector, ORIGIN_END, start_positions, active_mask

n, N = 6, 48
cfg = ConfigHolder(name="t", dataset_cfg=MazeDatasetConfig(name="t", grid_n=n, n_mazes=N, maze_ctor=LatticeMazeGenerators.gen_dfs, seed=7),
                   model_cfg=BaseGPTConfig(name="tiny", act_fn="gelu", d_model=64, d_head=16, n_layers=2), train_cfg=TRAINING_CONFIGS["test-v1"])
ds = MazeDataset.from_config(cfg.dataset_cfg, do_generate=True, load_local=False, save_local=False, gen_parallel=False)
vocab = cfg.tokenizer.get_vocab(); pad = vocab[SPECIAL_TOKENS.PADDING]; semi = vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE]
id_to_coord = {i: (int(m.group(1)), int(m.group(2))) for t, i in vocab.items() if (m := re.match(r"^\((\d+),(\d+)\)$", t))}
with tempfile.TemporaryDirectory() as td:
    pre = _pl.Path(td) / "pretok.pt"; pretokenize(ds, cfg.maze_tokenizer, vocab, pre, bos_id=int(cfg.tokenizer.bos_token_id), workers=0)
    pt = PretokDataset(pre); rows = [pt[i] for i in range(len(pt))]
toks, attn, conn, end = collate_pretok(rows, pad)
toks_blank = torch.stack([dummy_adjacency(toks[b], int(end[b]), semi) for b in range(len(rows))])
B, P = toks.shape; pos = torch.arange(P)[None, :]

for prompt_name, T in (("intact", toks), ("blank", toks_blank)):
    for cname in ("axdeg", "ego"):
        for start in ("adjlist_end", "origin_end"):
            kw = {"id_to_coord": id_to_coord} if cname == "ego" else {}
            code = make_code(cname, n, start=start, **kw); assert code.start == start
            F = code.features(T, attn, conn, end)
            sp = start_positions(T, end, start)
            if start == "origin_end":
                assert torch.equal(sp, end + 3) and (T[torch.arange(B), sp] == ORIGIN_END).all()
                assert (F[pos < sp[:, None]] == 0).all(), "must be silent before <ORIGIN_END>"
            else:
                assert torch.equal(sp, end)
            act = active_mask(sp, P, attn)
            if cname == "ego":   # the ego frame is undefined until the origin coordinate (end_pos + 2) even under adjlist_end
                act = act & (pos >= (end + 2)[:, None])
            assert (F[~act] == 0).all() and (F[act].abs().sum(-1) > 0).all()
            if cname == "axdeg":
                assert torch.equal(F[act], code.rows(conn)[:, None, :].expand(-1, P, -1)[act])
            else:  # ego: at <ORIGIN_END>, <TARGET_*>, <PATH_START> the frame is the origin
                o = torch.tensor([id_to_coord[int(T[b, end[b] + 2])] for b in range(B)])
                ref = make_code("ego", n, start="adjlist_end", **kw).features(T, attn, conn, end)
                assert torch.equal(F[act], ref[act])          # same frame rule, only the onset differs
            # step == features from several prefix cut points
            for cut_off in (-2, 0, +1, +3, +8, +12):
                cuts = (end + 3 + cut_off).clamp(min=1)
                feats_pref, states = [], None
                am = (pos < cuts[:, None]).long()
                st = code.init_state(T[:, : int(cuts.max())], am[:, : int(cuts.max())], conn, end)
                # rows differ in cut position, so step per row with the batch mechanics by masking: advance each row from its own cut
                ok = True
                for b in range(0, B, 8):
                    sl = slice(b, b + 8); Tb, cb, eb = T[sl], cuts[sl], end[sl]
                    amb = (pos < cb[:, None]).long()
                    s = code.init_state(Tb[:, : int(cb.max())], amb[:, : int(cb.max())], conn[sl], eb)
                    if (cb != cb[0]).any():
                        continue   # keep it simple: only equal-cut groups are stepped (cuts are equal when end_pos is equal)
                    L = int(attn[sl].sum(1).min())
                    for p in range(int(cb[0]), L):
                        f, s = code.step(s, Tb[:, p])
                        ok &= torch.equal(f, F[sl][:, p])
                assert ok, (prompt_name, cname, start, cut_off)
            inj = GraftInjector(code, 32, norm_match=code.expected_sq_norm())
            with tempfile.TemporaryDirectory() as td:
                torch.save(inj.state_for_ckpt() | {"mode": "complete"}, _pl.Path(td) / "i.pt")
                inj2 = load_injector(_pl.Path(td) / "i.pt")
            assert inj2.code.start == start and torch.equal(inj2.W, inj.W)
            F2 = inj2.code.features(T, attn, conn, end); assert torch.equal(F2, F)
            print(f"{prompt_name:6s} {cname:6s} {start:12s} OK  (first active pos = end_pos+{int((sp - end)[0])})")
print("ALL OK")
