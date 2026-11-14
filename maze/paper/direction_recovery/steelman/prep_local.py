#!/usr/bin/env python3
"""Local, no inference: index the 60 grafted finals and write the tiny per-model input files used by the pods
(inputs/<tag>.npz: scaled W_inj, the constant PATH_START embedding + position vector, hashes) and manifest.json."""
import glob, hashlib, json, re, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; CURVES = HERE.parents[1] / "curves"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
man = []; (HERE / "inputs").mkdir(exist_ok=True)
for cond in ("fixed", "trainable"):
    for g in range(5, 11):
        S = np.load(CURVES / f"results/distance/sets/g{g}.npz"); P = S["prompts"].shape[1]
        assert (S["prompts"][:, -2] == S["prompts"][0, -2]).all()
        for s in range(5):
            runs = sorted(r for r in glob.glob(str(CURVES / f"results/runpod-*/g{g}-s{s}/fast-inj-{cond}-*"))
                          if Path(r, "model.final.pt").exists() and Path(r, "injector.final.pt").exists())
            assert runs, (cond, g, s)
            run = Path(runs[0]); tag = f"g{g}-s{s}-{cond}"
            inj = torch.load(run / "injector.final.pt", map_location="cpu", weights_only=True); assert inj.get("layer", 0) == 0
            W = inj["W_inj"].float() * float(inj.get("scale", 1))
            sd = torch.load(run / "model.final.pt", map_location="cpu", weights_only=False)["state_dict"]
            const = sd["embed.weight"][int(S["prompts"][0, -2])] + sd["pos_embed.weight"][P - 2]
            np.savez(HERE / "inputs" / f"{tag}.npz", W=W.numpy(), const=const.float().numpy(), grid=g, seed=s, cond=cond,
                     path_start_pos=P - 2, path_start_tok=int(S["prompts"][0, -2]))
            man.append(dict(tag=tag, grid=g, seed=s, cond=cond, run=str(run.relative_to(CURVES / "results")), n_candidates=len(runs),
                            model_sha256=sha(run / "model.final.pt"), injector_sha256=sha(run / "injector.final.pt")))
            print(tag, run.parent.parent.name, len(runs), flush=True)
json.dump(man, open(HERE / "manifest.json", "w"), indent=1)
