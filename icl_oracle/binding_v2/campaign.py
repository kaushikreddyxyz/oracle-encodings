"""Freeze development-selected doses, then run held-out capacity/generalization tasks."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .runner import Runner
from .tasks import make_task


def select_betas(path, model, layer, shots):
    grouped = defaultdict(lambda: defaultdict(list))
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if (row["mode"] != "inject" or row["model"] != model
                or row["layer"] != layer or row["shots"] != shots
                or row["classes"] != 8 or row["seed"] not in range(4)):
            continue
        family = row["task"]
        if family not in {"codebook", "selection", "parity"}:
            continue
        if bool(row["seen"]) != (family == "codebook"):
            continue
        grouped[(family, float(row["beta"]))][row["seed"]].append(row)
    scores = defaultdict(list)
    for (family, beta), seeds in grouped.items():
        if set(seeds) != set(range(4)):
            continue
        seed_scores = {}
        for seed, rows in seeds.items():
            # Refuse partial/duplicated development conditions.
            if len(rows) != rows[0]["n_queries"] or len({r["query"] for r in rows}) != len(rows):
                raise ValueError(f"Incomplete or duplicated development rows: {family}, {beta}, {seed}")
            seed_scores[str(seed)] = float(np.mean([r["correct"] for r in rows]))
        scores[family].append(dict(beta=beta, accuracy=float(np.mean(list(seed_scores.values()))),
                                   seed_accuracies=seed_scores))
    selected = {}
    for family in ("codebook", "selection", "parity"):
        if not scores[family]:
            raise ValueError(f"No complete four-seed development conditions for {family}")
        selected[family] = sorted(scores[family], key=lambda row: (-row["accuracy"], row["beta"]))[0]["beta"]
    return selected, dict(scores)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--development", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, range(100, 108))))
    ap.add_argument("--shots", type=int, default=128)
    ap.add_argument("--queries", type=int, default=32)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attention", default="sdpa", choices=["sdpa", "eager"])
    args = ap.parse_args()
    seeds = list(map(int, args.seeds.split(",")))
    if len(set(seeds)) != len(seeds) or set(seeds) & set(range(4)):
        raise ValueError("Evaluation seeds must be unique and disjoint from development seeds 0..3")
    selected, scores = select_betas(args.development, args.model, args.layer, args.shots)
    configs = [dict(family="codebook", classes=k, include_seen=False) for k in (4, 8, 16, 32)]
    configs += [dict(family=f, classes=8, include_seen=True) for f in ("selection", "parity")]
    manifest = dict(arguments=vars(args), development_seeds=list(range(4)), evaluation_seeds=seeds,
                    selection_metric="mean per-seed within-label argmax accuracy; unseen for feature tasks, seen for codebook; ties choose lower beta",
                    selected_betas=selected, development_scores=scores, configurations=configs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(".selection.json")
    # Exclusive creation prevents accidental mixing with a prior held-out run.
    with manifest_path.open("x") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps({"selected_betas": selected, "manifest": str(manifest_path)}), flush=True)
    with output_path.open("x", buffering=1) as output:
        runner = Runner(args.model, args.device, args.layer, args.attention)
        for config in configs:
            family, classes = config["family"], config["classes"]
            for seed in seeds:
                task = make_task(family=family, seed=seed, n_demo=args.shots,
                                 n_query=args.queries, n_classes=classes,
                                 include_seen=config["include_seen"])
                plain = runner.layout(task, "none")
                norm_ref = runner.reference(plain)
                for mode in ("token", "none", "inject"):
                    layout = runner.layout(task, mode)
                    beta = selected[family] if mode == "inject" else 0.0
                    common = dict(model=args.model, task=family, seed=seed, shots=args.shots,
                                  n_queries=args.queries, classes=classes, mode=mode,
                                  layer=args.layer, beta=beta, norm_ref=norm_ref,
                                  labels=layout["labels"], metadata=task["metadata"],
                                  prefix_tokens=len(layout["prefix"]), stage="heldout",
                                  selection_manifest=str(manifest_path))
                    rows = []
                    start = time.time()
                    def emit(row):
                        result = {**common, **row}
                        output.write(json.dumps(result, default=lambda x: x.tolist() if hasattr(x, "tolist") else x) + "\n")
                        rows.append(result)
                    runner.evaluate(task, layout, mode, beta, norm_ref, args.bs, emit)
                    metrics = {}
                    for seen in (False, True):
                        subset = [r for r in rows if r["seen"] == seen]
                        if subset:
                            metrics["seen" if seen else "unseen"] = {
                                key: float(np.mean([r[key] for r in subset]))
                                for key in ("correct", "p_correct", "label_mass", "full_correct")}
                    print(f"{family} K={classes} seed={seed} {mode} beta={beta} "
                          f"{json.dumps(metrics)} seconds={time.time()-start:.1f}", flush=True)


if __name__ == "__main__":
    main()
