"""Aggregate binding runner JSONL with equal weight per task seed."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from .tasks import make_task
from .baseline import evaluate

CONFIG = ("model", "task", "shots", "classes", "mode", "layer", "beta", "n_queries")
METRICS = {"accuracy": "correct", "p_correct": "p_correct", "label_mass": "label_mass", "full_accuracy": "full_correct", "p_full": "p_full"}


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None}


def aggregate(paths, baseline_path=None, drop_identical=False):
    unique = {}
    duplicate_count = 0
    for path in paths:
        for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = tuple(row[k] for k in CONFIG) + (row["seed"], row["query"])
            if identity in unique:
                duplicate_count += 1
                if not drop_identical or row != unique[identity]:
                    raise ValueError(f"Duplicate configuration/seed/query at {path}:{line_number}: {identity}; use --drop-identical only for exact duplicate rows")
                continue
            unique[identity] = row
    groups = defaultdict(lambda: defaultdict(list))
    for row in unique.values():
        groups[tuple(row[k] for k in CONFIG) + (row["seen"],)][row["seed"]].append(row)
    selected = {}
    if baseline_path:
        for row in json.loads(Path(baseline_path).read_text())["rows"]:
            selected.setdefault((row["family"], row["n_classes"], row["n_demo"]), {})[row["method"]] = row["selected_setting"]
    baseline_cache = {}
    output = []
    for key, seed_groups in sorted(groups.items()):
        config = dict(zip(CONFIG + ("seen",), key))
        item = {**config, "n_seeds": len(seed_groups), "n_queries_total": sum(map(len, seed_groups.values())), "seeds": sorted(seed_groups)}
        seed_metrics = {}
        for seed, rows in sorted(seed_groups.items()):
            seed_metrics[str(seed)] = {name: float(np.mean([r[field] for r in rows])) for name, field in METRICS.items() if all(field in r for r in rows)}
            seed_metrics[str(seed)]["n_queries"] = len(rows)
            seed_metrics[str(seed)]["query_indices"] = sorted(r["query"] for r in rows)
        item["seed_metrics"] = seed_metrics
        item["metrics"] = {name: statistics([m[name] for m in seed_metrics.values()]) for name in METRICS if all(name in m for m in seed_metrics.values())}
        if baseline_path:
            settings = selected.get((config["task"], config["classes"], config["shots"]), {})
            if config["task"] == "codebook" and not settings:
                settings = {"nearest_cosine": "nearest_cosine"}
            item["matched_baselines"] = {}
            for method, setting in settings.items():
                scores = []
                per_seed = {}
                for seed, rows in sorted(seed_groups.items()):
                    cache_key = (config["task"], config["classes"], config["shots"], config["n_queries"], seed, config["seen"])
                    if cache_key not in baseline_cache:
                        task = make_task(config["task"], seed=seed, n_demo=config["shots"], n_query=config["n_queries"], n_classes=config["classes"], include_seen=config["seen"])
                        baseline_cache[cache_key] = (task, evaluate(task))
                    task, evaluated = baseline_cache[cache_key]
                    for row in rows:
                        i = row["query"]
                        if (int(task["query_codes"][i]), int(task["query_y"][i]), bool(task["query_seen"][i])) != (row["code"], row["y"], row["seen"]):
                            raise ValueError(f"Regenerated task does not match raw query: seed={seed}, query={i}, config={config}")
                    predictions = evaluated[setting]["predictions"]
                    score = float(np.mean([predictions[r["query"]] == r["y"] for r in rows]))
                    scores.append(score)
                    per_seed[str(seed)] = score
                item["matched_baselines"][method] = {"selected_setting": setting, "accuracy": statistics(scores), "seed_accuracy": per_seed}
            if not settings:
                item["baseline_note"] = "No development-selected setting exists for this task/classes/shots combination."
        output.append(item)
    return {"source_paths": [str(p) for p in paths], "duplicates_dropped": duplicate_count,
            "aggregation": "Each task seed has equal weight. Standard error is across seed means; one-seed SE is null. Matched baselines use precisely the observed query subset.",
            "rows": output, "paired_comparisons": paired_comparisons(output)}


def paired_comparisons(rows):
    """Seed-paired accuracy differences; percentile bootstrap over task seeds."""
    matching = ("model", "task", "shots", "classes", "layer", "n_queries", "seen")
    comparisons = []
    for row in rows:
        if row["mode"] != "inject":
            continue
        references = []
        for other in rows:
            if other["mode"] in {"none", "token"} and all(row[k] == other[k] for k in matching):
                references.append((other["mode"], {s: m["accuracy"] for s, m in other["seed_metrics"].items()
                                   if s in row["seed_metrics"] and m["query_indices"] == row["seed_metrics"][s]["query_indices"]}))
        references += [("baseline:" + name, value["seed_accuracy"]) for name, value in row.get("matched_baselines", {}).items()]
        for reference, scores in references:
            seeds = sorted(set(row["seed_metrics"]) & set(scores), key=int)
            if not seeds:
                continue
            differences = np.array([row["seed_metrics"][s]["accuracy"] - scores[s] for s in seeds])
            item = {k: row[k] for k in CONFIG + ("seen",)}
            item.update(reference=reference, n_seeds=len(seeds), seed_differences=dict(zip(seeds, differences.tolist())), difference=statistics(differences))
            if len(seeds) > 1:
                rng = np.random.default_rng(20260916)
                bootstrap = differences[rng.integers(len(seeds), size=(20000, len(seeds)))].mean(axis=1)
                item["ci95"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
            else:
                item["ci95"] = None
            item["ci_method"] = "Percentile bootstrap, 20000 resamples of paired task seeds; not adjusted for multiple comparisons."
            comparisons.append(item)
    return comparisons


def markdown(summary):
    columns = ["Model", "Task", "Shots", "Classes", "Mode", "Layer", "Beta", "Seen", "Seeds", "Queries", "Accuracy ± SE", "P(correct)", "Label mass", "Full accuracy"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in summary["rows"]:
        def fmt(name):
            value = row["metrics"].get(name)
            if value is None:
                return "—"
            return f"{value['mean']:.4f}" + (f" ± {value['se']:.4f}" if name == "accuracy" and value["se"] is not None else "")
        fields = [row["model"], row["task"], row["shots"], row["classes"], row["mode"], row["layer"], row["beta"], row["seen"], row["n_seeds"], row["n_queries_total"], fmt("accuracy"), fmt("p_correct"), fmt("label_mass"), fmt("full_accuracy")]
        lines.append("| " + " | ".join(map(str, fields)) + " |")
    if any(row.get("matched_baselines") for row in summary["rows"]):
        lines += ["", "Matched clean-vector baselines on the same observed queries:", "", "| Model/task/config | Seen | Method | Dev-selected setting | Accuracy ± SE |", "| --- | --- | --- | --- | --- |"]
        for row in summary["rows"]:
            config = f"{row['model']} / {row['task']} / shots={row['shots']} / K={row['classes']} / {row['mode']} / layer={row['layer']} / beta={row['beta']}"
            for name, baseline in row.get("matched_baselines", {}).items():
                value = baseline["accuracy"]
                accuracy = f"{value['mean']:.4f}" + (f" ± {value['se']:.4f}" if value["se"] is not None else "")
                lines.append(f"| {config} | {row['seen']} | {name} | {baseline['selected_setting']} | {accuracy} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--drop-identical", action="store_true", help="Explicitly discard byte-equivalent parsed duplicate rows; differing duplicates always fail")
    args = parser.parse_args()
    summary = aggregate(args.paths, args.baseline_summary, args.drop_identical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    md = args.markdown or args.output.with_suffix(".md")
    md.write_text(markdown(summary))
    print(f"Saved {len(summary['rows'])} groups to {args.output} and {md}")


if __name__ == "__main__":
    main()
