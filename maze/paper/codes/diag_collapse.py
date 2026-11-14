#!/usr/bin/env python3
"""Why do some large-grid ego seeds solve ~0% while predicting ~0.82 of shortest-path tokens? Split the teacher-forced
next-token distribution at path positions by position type: forced (one open neighbour besides the cell we came from),
junction (>= 2 choices), target (PATH_END expected). Also p(PATH_END) off-target (premature stops) and argmax accuracy.
  python diag_collapse.py --run_dir <run> --grid_n <n>   -> <run>/eval/diag_collapse.json
"""
import json, sys, pathlib as _pl
sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "maze" / "core").is_dir())))
import maze._pathsetup  # noqa
import numpy as np, torch, fire
from pathlib import Path
from maze_transformer.training.config import ConfigHolder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "curves"))
from eval_ckpts import EvalSet, Scorer, load_ckpt, open_nbrs

def main(run_dir: str, grid_n: int, eval_dir: str = "/workspace/evalsets", n_mazes: int = 2000, seed: int = 4321, batch: int = 500):
    device = torch.device("cuda"); run = Path(run_dir)
    cfg = ConfigHolder.load(json.load(open(run / "config_holder.json")))
    es = EvalSet(cfg, grid_n, Path(eval_dir), n_mazes, seed, 2, 2, device)
    model, inj = load_ckpt(run / "model.final.pt", run / "injector.final.pt", device); sc = Scorer(model, inj, device)
    # position metadata in tf_batches order
    kind, togo, plen_all = [], [], []
    for i, conn, path in es.worlds["true"]:
        for step, cell in enumerate(path):
            nb = open_nbrs(conn, cell, grid_n); nch = len(nb) - (1 if step > 0 else 0)
            kind.append("target" if cell == path[-1] else ("forced" if nch <= 1 else "junction")); togo.append(len(path) - 1 - step)
    kind = np.array(kind); togo = np.array(togo)
    psp, pva, pend, acc = [], [], [], []
    for b in es.tf_batches("true", batch):
        probs = sc.logits(b["toks"], b["attn"], b["conn"], b["end"]).float().softmax(-1); p = probs[b["bidx"], b["pidx"]]; V = p.shape[-1]
        psp.append((p * b["sp"][:, :V]).sum(-1).cpu().numpy()); pva.append((p * b["va"][:, :V]).sum(-1).cpu().numpy())
        pend.append(p[:, es.path_end_id].cpu().numpy()); acc.append(b["sp"][:, :V].gather(1, p.argmax(-1)[:, None])[:, 0].cpu().numpy())
    psp, pva, pend, acc = map(np.concatenate, (psp, pva, pend, acc)); assert len(psp) == len(kind)
    out = {"n_positions": int(len(kind))}
    for k in ("forced", "junction", "target"):
        m = kind == k
        out[k] = dict(frac=float(m.mean()), p_sp=float(psp[m].mean()), p_valid_nonsp=float(pva[m].mean()), p_path_end=float(pend[m].mean()), argmax_acc=float(acc[m].mean()))
    # junction accuracy vs remaining distance to target (is planning only local?)
    j = kind == "junction"; out["junction_acc_by_steps_to_go"] = {}
    for lo, hi in ((1, 3), (3, 6), (6, 12), (12, 25), (25, 400)):
        m = j & (togo >= lo) & (togo < hi)
        if m.sum() > 50: out["junction_acc_by_steps_to_go"][f"{lo}-{hi}"] = dict(n=int(m.sum()), argmax_acc=float(acc[m].mean()), p_sp=float(psp[m].mean()))
    (run / "eval").mkdir(exist_ok=True); json.dump(out, open(run / "eval" / "diag_collapse.json", "w"), indent=1); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    fire.Fire(main)
