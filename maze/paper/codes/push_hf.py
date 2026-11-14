#!/usr/bin/env python3
"""Mirror the final checkpoints of the graft-code campaign to Hugging Face (models only: model.final.pt, injector.final.pt,
config_holder.json per run; no eval results, no figures). Re-runnable: unchanged files are skipped by hash.
  python push_hf.py [--repo kaushikreddyxyz/maze-graft-codes] [--public]
"""
import argparse, re
from pathlib import Path
from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="kaushikreddyxyz/maze-graft-codes"); ap.add_argument("--public", action="store_true")
a = ap.parse_args(); api = HfApi()
api.create_repo(a.repo, repo_type="model", private=not a.public, exist_ok=True)
runs = sorted(d for d in (HERE / "results/runs").glob("g*-s*-*-e*-oe") if re.match(r"g\d+-s\d+-(edge|ego|base|axdeg|axdeg1h)-e\d-oe$", d.name))
card = ["# maze-graft-codes", "",
        "Final checkpoints of the graft-code campaign (maze-solving transformers, jirpyish d256/12L, 2M random-DFS mazes, prompt intact,",
        "graft injected at block-0 input from `<ORIGIN_END>`). Folder = `g<grid>-s<seed>-<code>-e<epochs>-oe/<run>/` with",
        "`model.final.pt` (torch, `maze/core/tl_bridge.load_torch_ckpt`), `injector.final.pt` (`maze/core/graft.load_injector`) and `config_holder.json`.",
        "Codes: edge = allocentric +-1 per wall; ego = egocentric walls relative to the current cell (dynamic); base = no graft.",
        "e8 = the e4 run continued for 4 more epochs (weights warm-started, AdamW moments reset). Eval results stay in the repo, not here.", "", "| folder | files |", "|---|---|"]
for d in runs:
    for r in d.iterdir():
        if not r.is_dir() or not (r / "model.final.pt").exists(): continue
        card.append(f"| {d.name}/{r.name} | {', '.join(sorted(f.name for f in r.iterdir() if f.name in ('model.final.pt','injector.final.pt','config_holder.json')))} |")
(HERE / "results/HF_README.md").write_text("\n".join(card) + "\n")
api.upload_file(path_or_fileobj=str(HERE / "results/HF_README.md"), path_in_repo="README.md", repo_id=a.repo)
api.upload_folder(folder_path=str(HERE / "results/runs"), repo_id=a.repo, path_in_repo="runs",
                  allow_patterns=["*/*/model.final.pt", "*/*/injector.final.pt", "*/*/config_holder.json"], commit_message="graft-code campaign checkpoints")
print("uploaded", len(runs), "run folders ->", a.repo)
