#!/usr/bin/env python3
"""units for fleet_queue.py.   make_units.py stage1 | probes | stage2"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; man = json.load(open(HERE / "manifest.json")); stage = sys.argv[1]; units = []
NW = {5: 4, 6: 4, 7: 3, 8: 2, 9: 2, 10: 2}
for m in man:
    tag, g, run = m["tag"], m["grid"], m["run"]; mu = f"python model_unit.py --tag {tag} --run /workspace/runs/{run} --out /workspace/out/{tag}"
    if stage == "stage1" and m["seed"] == 0:       # full SAE grid at the injection site (analytic activations: no model needed)
        if g >= 8:                                 # two pods per model: A = half the grid, B = other half + matching pursuit
            units.append(dict(id=f"s1a-{tag}", tag=tag, weight=g * 10 + 5, run=None, cmd=f"PART=s1a bash run_sae.sh {tag} injection jobs/fullA-g{g}.json {NW[g]} && echo RUN_DONE"))
            units.append(dict(id=f"s1b-{tag}", tag=tag, weight=g * 10 + 6, run=None, cmd=f"PART=s1b bash run_sae.sh {tag} injection jobs/fullB-g{g}.json {NW[g]} && PART=s1b bash run_sae.sh {tag} injection jobs/mp-g{g}.json 1 && echo RUN_DONE"))
        else:
            cmd = f"PART=s1 bash run_sae.sh {tag} injection jobs/full-g{g}.json {NW[g]} && PART=s1 bash run_sae.sh {tag} injection jobs/mp-g{g}.json 1 && echo RUN_DONE"
            units.append(dict(id=f"s1-{tag}", tag=tag, weight=g * 10 + 5, run=None, cmd=cmd))
    if stage == "probes":                          # verification, post0 activations, probes, all non-SAE interventions
        units.append(dict(id=f"pr-{tag}", tag=tag, weight=g, run=run, cmd=f"{mu} --phase acts,probes,intervene && echo RUN_DONE"))
    if stage == "stage2":                          # shortlist SAEs (both SAE seeds), post0 SAEs for model seeds 0-1, SAE interventions
        cmd = f"PART=s2 bash run_sae.sh {tag} injection jobs/short-g{g}.json {NW[g]}"
        if m["seed"] <= 1: cmd += f" && {mu} --phase acts && PART=s2 bash run_sae.sh {tag} post0 jobs/short-post0-g{g}.json {NW[g]}"
        units.append(dict(id=f"s2-{tag}", tag=tag, weight=g * 10 + (3 if m["seed"] <= 1 else 0), run=run, cmd=cmd + f" && {mu} --phase intervene && echo RUN_DONE", after=[f"pr-{tag}"]))
json.dump(units, open(HERE / f"units-{stage}.json", "w"), indent=1); print(stage, len(units))
