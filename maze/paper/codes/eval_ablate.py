#!/usr/bin/env python3
"""Inference-time ablations of the graft: is an ego / axdeg graft used the way the edge graft is?

Conditions (each scored teacher-forced along the true path and with greedy rollouts on the 2,000-maze eval set of
eval_ckpts.py, true world; the split between "prompt" and "generation" is the origin token, exactly as in greedy_kv):
  full            graft on everywhere (the model as trained)
  graft_off       graft removed everywhere (channel off): does the model fall off the manifold?
  off_gen         graft on during the prompt only, off for every generated / path token
  off_prompt      graft off during the prompt, on for the path
  x0.5, x2        graft scaled everywhere
  noise           the graft replaced by Gaussian noise of the same per-position norm (same positions active)
  prompt_blank    adjacency list blanked to `;`, graft on: can the model solve from the graft alone?
  blank_off       adjacency list blanked AND graft off (floor)
  shuffled        graft carries a different maze from the same batch (prompt intact): a cheap graft-vs-prompt conflict
For a base model (no injector) only full / prompt_blank are meaningful.

  python eval_ablate.py --run_dir <run> --grid_n 6   -> <run>/eval/ablate.json
"""
import pathlib as _pl, sys as _sys
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(next(p for p in _HERE.parents if (p / "maze" / "core").is_dir())))
_sys.path.insert(0, str(_HERE.parent / "curves"))
import maze._pathsetup  # noqa
import json, time
from pathlib import Path
import fire, numpy as np, torch
from maze_transformer.training.config import ConfigHolder
from maze.core.graft import as_graft, PATH_START
from eval_ckpts import EvalSet, Scorer, load_ckpt, score_tf, rollout

CONDS = ["full", "graft_off", "off_gen", "off_prompt", "x0.5", "x2", "noise", "prompt_blank", "blank_off", "shuffled"]


class ScaledCode:
    """Wraps a GraftCode: prompt positions (<= the origin token = the token after <PATH_START>... see note) get
    `prompt_scale`, later positions `gen_scale`; `shuffle` permutes the mazes in the batch. The prompt/generation
    split is the origin token in the path (the first path token), i.e. positions <= pos(<PATH_START>) + 1, which is
    what greedy_kv's prefill covers."""
    def __init__(self, code, prompt_scale=1.0, gen_scale=1.0, shuffle=False):
        self.c, self.ps, self.gs, self.shuffle = code, float(prompt_scale), float(gen_scale), shuffle
        self.start, self.dynamic, self.name, self.n_features = code.start, code.dynamic, code.name, code.n_features

    def _conn(self, conn):
        return conn.roll(1, dims=0) if self.shuffle else conn

    def features(self, tokens, attention_mask, conn, end_pos):
        f = self.c.features(tokens, attention_mask, self._conn(conn), end_pos)
        pos = torch.arange(tokens.shape[1], device=f.device)[None, :]
        ps = (tokens.to(f.device) == PATH_START).long().argmax(1)           # position of <PATH_START>
        scale = torch.where(pos <= (ps + 1)[:, None], self.ps, self.gs).to(f.dtype)
        return f * scale[:, :, None]

    def init_state(self, tokens, attention_mask, conn, end_pos):
        return self.c.init_state(tokens, attention_mask, self._conn(conn), end_pos)

    def step(self, state, new_tokens):
        f, s = self.c.step(state, new_tokens)
        return f * self.gs, s


class CondGraft:
    """Duck-typed graft (code + delta + layer) that eval_ckpts' Scorer / greedy_kv accept via as_graft()."""
    def __init__(self, g, prompt_scale=1.0, gen_scale=1.0, noise=False, shuffle=False, seed=0):
        self.g, self.layer = g, int(g.layer)
        self.code = ScaledCode(g.code, prompt_scale, gen_scale, shuffle)
        self.noise, self.gen = noise, torch.Generator().manual_seed(seed)

    def delta(self, features):
        d = self.g.delta(features)
        if not self.noise:
            return d
        nrm = d.norm(dim=-1, keepdim=True)                                  # 0 where the graft is silent
        z = torch.randn(d.shape, generator=self.gen).to(d.device, d.dtype)
        return z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6) * nrm


def main(run_dir: str, grid_n: int, eval_dir: str = "/workspace/evalsets", n_mazes: int = 2000, seed: int = 4321,
         batch: int = 1000, rollout_batch: int = 2000, conds: str = ""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_float32_matmul_precision("high")
    run = Path(run_dir); cfg = ConfigHolder.load(json.load(open(run / "config_holder.json")))
    es = EvalSet(cfg, grid_n, Path(eval_dir), n_mazes, seed, 2, 2, device)
    es_blank = EvalSet(cfg, grid_n, Path(eval_dir), n_mazes, seed, 2, 2, device); es_blank.dummify_()
    inj_path = run / "injector.final.pt"
    model, injector = load_ckpt(run / "model.final.pt", inj_path if inj_path.exists() else None, device)
    g = as_graft(injector); max_new = grid_n * grid_n + 2
    todo = conds.split(",") if conds else (CONDS if g is not None else ["full", "prompt_blank"])
    rows = []; t0 = time.time()
    for cond in todo:
        e = es
        if g is None:
            cg = None; e = es_blank if cond == "prompt_blank" else es
        elif cond == "full": cg = CondGraft(g)
        elif cond == "graft_off": cg = CondGraft(g, 0, 0)
        elif cond == "off_gen": cg = CondGraft(g, 1, 0)
        elif cond == "off_prompt": cg = CondGraft(g, 0, 1)
        elif cond == "x0.5": cg = CondGraft(g, .5, .5)
        elif cond == "x2": cg = CondGraft(g, 2, 2)
        elif cond == "noise": cg = CondGraft(g, noise=True)
        elif cond == "prompt_blank": cg = CondGraft(g); e = es_blank
        elif cond == "blank_off": cg = CondGraft(g, 0, 0); e = es_blank
        elif cond == "shuffled": cg = CondGraft(g, shuffle=True)
        else: raise ValueError(cond)
        sc = Scorer(model, cg, device)
        tf, _ = score_tf(sc, e, "true", batch); ro = rollout(sc, e, "true", rollout_batch, max_new, kv=True)
        row = dict(cond=cond, p_sp=float(np.nanmean(tf[:, 0])), p_valid=float(np.nanmean(tf[:, 1])),
                   p_invalid=float(np.nanmean(1 - tf[:, 0] - tf[:, 1])), solved=float(np.nanmean(ro["solved"])),
                   reached=float(np.nanmean(ro["reached"])), clean=float(np.nanmean(ro["clean"])),
                   valid_move_frac=float(np.nanmean(ro["valid_move_frac"])))
        if cond == "shuffled":   # judge the same rollouts against the graft's maze too (the shifted conn)
            e2 = EvalSet.__new__(EvalSet); e2.__dict__ = dict(e.__dict__)
            e2.worlds = {"true": [(i, e.conn[(k - 1) % len(e.conn)], e.true_path[i]) for k, (i, _, _) in enumerate(e.worlds["true"])]}
            # (path of the prompt maze is irrelevant for rollouts; solved/reached judged against the graft maze)
            ro2 = rollout(sc, e2, "true", rollout_batch, max_new, kv=True)
            row["solved_graft_maze"] = float(np.nanmean(ro2["solved"])); row["reached_graft_maze"] = float(np.nanmean(ro2["reached"]))
        rows.append(row)
        print(f"  {cond:12s} sp {row['p_sp']:.3f} valid {row['p_valid']:.3f} invalid {row['p_invalid']:.4f} solved {row['solved']:.3f} "
              f"reached {row['reached']:.3f} clean {row['clean']:.3f} vmf {row['valid_move_frac']:.3f}"
              + (f" | vs graft maze: solved {row['solved_graft_maze']:.3f}" if 'solved_graft_maze' in row else "") + f"  [{time.time()-t0:.0f}s]", flush=True)
    out = run / "eval" / "ablate.json"; out.parent.mkdir(exist_ok=True)
    json.dump(dict(run=run.name, grid_n=grid_n, n_mazes=n_mazes, code=(g.code.name if g else "none"), rows=rows), open(out, "w"), indent=1)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
