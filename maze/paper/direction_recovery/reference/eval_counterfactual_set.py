#!/usr/bin/env python3
"""Evaluate final models on the curated counterfactual edit dataset.

For every edit (see build_counterfactual_set.py) and every model (baseline
no-channel control, fixed injection, trainable injection), roll out greedily
under the EDITED injected features (prompt unchanged) and score per family:

  detour          blocked_edge_crossed | detour_wall_used | reached_target
  offpath_block   rollout_changed_vs_true (false-positive rate) | crossed
  shortcut_open   shortcut_used (also by BFS-gain bucket) | reached_target
  zerogain_open   opened_wall_used | rollout_changed_vs_true

Writes per-model metrics to <set_dir>/results_<model>.json and prints a
combined table.

  python eval_counterfactual_set.py --model_dirs '{"fixed": "...", ...}'
"""

import copy
import json
from collections import defaultdict
from pathlib import Path

import fire
import torch
from maze_transformer.training.config import ConfigHolder

from build_counterfactual_set import (
    feat_idx,
    rollout_to_coords,
    traversed_edges,
)
from eval_counterfactual import load_checkpoint
from eval_injection import ProbeSet, greedy_rollout
from fast_data import PretokDataset

GRID_N = 6


def slice_ps(ps, rows):
    sub = copy.copy(ps)
    for f in ("tokens", "attn", "conn", "end_pos", "lens", "origin_pos",
              "first_move", "first_feat", "open_id", "open_feat", "target_id"):
        setattr(sub, f, getattr(ps, f)[rows])
    return sub


def chunked_rollout(model, injector, ps, rows, conn_rows, chunk=256):
    outs = []
    for lo in range(0, len(rows), chunk):
        sub_rows = rows[lo: lo + chunk]
        sub = slice_ps(ps, sub_rows)
        outs += greedy_rollout(model, injector, sub, conn_rows[lo: lo + chunk])
    return outs


@torch.no_grad()
def eval_model(name, run_dir, injected, ps, edits, set_dir, device):
    run_path = Path(run_dir)
    model, injector = load_checkpoint(
        run_path / "model.final.zanj",
        (run_path / "injector.final.pt") if injected else None,
        device,
    )
    # ps may be the dummied variant of the set (variant C models)

    # true-world rollouts for this model (for "changed" comparisons)
    all_rows = torch.arange(len(ps.conn))
    true_rolls = chunked_rollout(model, injector, ps, all_rows, ps.conn)

    rows = torch.tensor([e["maze"] for e in edits])
    conn_edit = ps.conn[rows].clone()
    for j, e in enumerate(edits):
        if e["block"] is not None:
            conn_edit[j, e["block"]] = False
        if e["open"] is not None:
            conn_edit[j, e["open"]] = True
    edit_rolls = chunked_rollout(model, injector, ps, rows, conn_edit)

    agg = defaultdict(lambda: defaultdict(list))
    for e, gen in zip(edits, edit_rolls):
        i = e["maze"]
        coords = rollout_to_coords(ps, i, gen)
        edges = set(traversed_edges(coords))
        target = ps.id_to_coord[int(ps.target_id[i])]
        reached = len(coords) > 1 and coords[-1] == target
        changed = gen != true_rolls[i]
        fam = e["family"]
        m = agg[fam]
        m["reached_target"].append(float(reached))
        m["rollout_changed"].append(float(changed))
        if fam == "detour":
            m["blocked_edge_crossed"].append(float(e["block"] in edges))
            m["detour_wall_used"].append(float(e["open"] in edges))
        elif fam == "offpath_block":
            m["blocked_edge_crossed"].append(float(e["block"] in edges))
        elif fam == "shortcut_open":
            used = float(e["open"] in edges)
            m["shortcut_used"].append(used)
            bucket = "gain2" if e["gain"] <= 2 else (
                "gain4" if e["gain"] <= 4 else "gain6plus"
            )
            m[f"shortcut_used_{bucket}"].append(used)
        elif fam == "zerogain_open":
            m["opened_wall_used"].append(float(e["open"] in edges))

    out = {
        fam: {"n": len(fams["reached_target"])}
        | {k: sum(v) / len(v) for k, v in fams.items()}
        for fam, fams in agg.items()
    }
    reached_true = []
    for i, g in enumerate(true_rolls):
        c = rollout_to_coords(ps, i, g)
        reached_true.append(
            float(len(c) > 1 and c[-1] == ps.id_to_coord[int(ps.target_id[i])])
        )
    out["_true_world"] = {
        "n": len(reached_true),
        "reached_target": sum(reached_true) / len(reached_true),
    }
    with open(Path(set_dir) / f"results_{name}.json", "w") as f:
        json.dump(out, f, indent=2)
    del model
    torch.cuda.empty_cache()
    return out


def main(
    set_dir: str = "/workspace/runs/counterfactual_set",
    fixed_dir: str = "/workspace/runs/fast/fast-inj-fixed-complete-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000",
    trainable_dir: str = "/workspace/runs/final_models/trainable",
    baseline_dir: str = "/workspace/runs/final_models/baseline",
    model_dirs: dict | None = None,
):
    """model_dirs: {name: run_dir} overriding the default three. Injection and
    the dummy prompt transform are inferred per run (injector.final.pt present /
    "dummy" in the run name), so variant runs need no extra flags."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    set_path = Path(set_dir)
    data = torch.load(set_path / "set.pt", weights_only=False)
    edits = data["edits"]

    cfg = ConfigHolder.load(
        json.load(open(Path(fixed_dir) / "config_holder.json"))
    )
    vocab = cfg.tokenizer.get_vocab()

    def build_ps(dummy: bool):
        pretok = PretokDataset(data["pretok_file"])
        if dummy:
            from maze_dataset import SPECIAL_TOKENS

            pretok.dummify_(vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE])
        return ProbeSet(
            [pretok[i] for i in range(len(pretok))], vocab, GRID_N, device
        )

    runs = model_dirs or {
        "baseline": baseline_dir, "fixed": fixed_dir, "trainable": trainable_dir,
    }
    probe_sets = {}
    results = {}
    for name, d in runs.items():
        injected = (Path(d) / "injector.final.pt").exists()
        dummy = "dummy" in Path(d).name
        if dummy not in probe_sets:
            probe_sets[dummy] = build_ps(dummy)
        ps = probe_sets[dummy]
        print(f"== evaluating {name} (injected={injected}, dummy={dummy})")
        results[name] = eval_model(name, d, injected, ps, edits, set_path, device)

    metrics = [
        ("detour", "blocked_edge_crossed"),
        ("detour", "detour_wall_used"),
        ("detour", "reached_target"),
        ("offpath_block", "rollout_changed"),
        ("shortcut_open", "shortcut_used"),
        ("shortcut_open", "shortcut_used_gain2"),
        ("shortcut_open", "shortcut_used_gain4"),
        ("shortcut_open", "shortcut_used_gain6plus"),
        ("shortcut_open", "reached_target"),
        ("zerogain_open", "opened_wall_used"),
        ("_true_world", "reached_target"),
    ]
    col_label = {"baseline": "no injection"}
    hdr = f"{'family/metric':44s}" + "".join(
        f"{col_label.get(n, n):>14s}" for n in results
    )
    print(hdr)
    for fam, k in metrics:
        vals = []
        for n in results:
            v = results[n].get(fam, {}).get(k)
            vals.append(f"{v:14.3f}" if v is not None else f"{'—':>14s}")
        print(f"{fam + '/' + k:44s}" + "".join(vals))
    print(
        "\n('no injection' = coincidence floor: this model receives no"
        " feature vector at all; its rates reflect chance overlap with the"
        " edits, not injected control.)"
    )


if __name__ == "__main__":
    fire.Fire(main)
