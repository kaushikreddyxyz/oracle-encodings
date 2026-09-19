"""Clean-signal retrieval and ridge baselines; no language model dependencies."""
import argparse
import json
from pathlib import Path
import numpy as np

try:
    from .tasks import make_task
except ImportError:
    from tasks import make_task


def evaluate(task, temperatures=(0.01, 0.03, 0.1, 0.3, 1.0), ridges=(0.001, 0.1, 1.0, 10.0)):
    x, q = task["demo_x"].astype(float), task["query_x"].astype(float)
    y, truth = task["demo_y"], task["query_y"]
    classes = task["metadata"]["n_classes"]
    onehot = np.eye(classes)[y]
    similarity = q @ x.T / (np.linalg.norm(q, axis=1)[:, None] * np.linalg.norm(x, axis=1)[None, :])
    results = {}

    def record(name, scores):
        pred = scores.argmax(axis=1)
        row = {"accuracy": float(np.mean(pred == truth)), "predictions": pred.tolist()}
        for name_subset, mask in (("seen", task["query_seen"]), ("unseen", ~task["query_seen"])):
            if mask.any():
                row[name_subset + "_accuracy"] = float(np.mean(pred[mask] == truth[mask]))
        results[name] = row

    record("nearest_cosine", onehot[similarity.argmax(axis=1)])
    for temperature in temperatures:
        weights = np.exp((similarity - similarity.max(axis=1, keepdims=True)) / temperature)
        record(f"cosine_vote_t{temperature:g}", weights @ onehot)
    for order in (1, 2):
        def features(z):
            parts = [np.ones((len(z), 1)), z]
            if order == 2:
                a, b = np.triu_indices(z.shape[1], k=1)
                parts.append(z[:, a] * z[:, b])
            return np.concatenate(parts, axis=1)
        design, query_design = features(x), features(q)
        for ridge in ridges:
            penalty = np.eye(design.shape[1]) * ridge
            penalty[0, 0] = 0
            coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ onehot)
            record(f"{'linear' if order == 1 else 'quadratic'}_ridge_{ridge:g}", query_design @ coefficients)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=["selection", "parity", "random"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--n-demo", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--n-query", type=int, default=32)
    parser.add_argument("--classes", type=int, nargs="+", default=[8])
    parser.add_argument("--include-seen", action="store_true")
    parser.add_argument("--development-seeds", type=int, nargs="+")
    parser.add_argument("--evaluation-seeds", type=int, nargs="+")
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.development_seeds) != bool(args.evaluation_seeds):
        parser.error("Supply both --development-seeds and --evaluation-seeds")
    if args.development_seeds and set(args.development_seeds) & set(args.evaluation_seeds):
        parser.error("Development and evaluation seeds must be disjoint")
    seeds = args.development_seeds + args.evaluation_seeds if args.development_seeds else args.seeds
    rows = []
    for family in args.families:
        for classes in (args.classes if family == "codebook" else [8]):
            for n_demo in args.n_demo:
                for seed in seeds:
                    task = make_task(family, seed, n_demo, args.n_query, n_classes=classes, include_seen=args.include_seen)
                    row = {"metadata": task["metadata"], "baselines": evaluate(task)}
                    for key in ("demo_codes", "query_codes", "query_seen", "demo_x", "query_x", "demo_bits", "query_bits", "demo_y", "query_y"):
                        row[key] = task[key].tolist()
                    row["split"] = ("development" if seed in args.development_seeds else "evaluation") if args.development_seeds else "unspecified"
                    rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"note": "Hyperparameter sweeps are separate candidates, not test-set-selected estimates.", "rows": rows}, indent=2) + "\n")
    print(f"Saved {len(rows)} task evaluations to {args.out}")
    if args.development_seeds:
        summary = summarize_heldout(rows)
        destination = args.summary_out or args.out.with_name(args.out.stem + "_summary.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved development-selected evaluation summary to {destination}")


def summarize_heldout(rows):
    """Choose settings using development unseen accuracy, then evaluate seeds."""
    groups = sorted({(r["metadata"]["family"], r["metadata"]["n_classes"], r["metadata"]["n_demo"]) for r in rows})
    output = []
    for family, classes, n_demo in groups:
        group = [r for r in rows if (r["metadata"]["family"], r["metadata"]["n_classes"], r["metadata"]["n_demo"]) == (family, classes, n_demo)]
        development = [r for r in group if r["split"] == "development"]
        evaluation = [r for r in group if r["split"] == "evaluation"]
        criterion = "seen_accuracy" if family == "codebook" else "unseen_accuracy"
        for method, prefix in (("nearest_cosine", "nearest_cosine"), ("cosine_vote", "cosine_vote_"), ("linear_ridge", "linear_ridge_"), ("quadratic_ridge", "quadratic_ridge_")):
            candidates = [key for key in development[0]["baselines"] if key.startswith(prefix)]
            dev_means = {key: float(np.mean([r["baselines"][key][criterion] for r in development])) for key in candidates}
            selected = max(candidates, key=lambda key: dev_means[key])
            result = {"family": family, "n_classes": classes, "n_demo": n_demo, "method": method,
                      "selected_setting": selected, "selection_metric": criterion,
                      "development_seed_scores": dev_means,
                      "development_seeds": [r["metadata"]["seed"] for r in development],
                      "evaluation_seeds": [r["metadata"]["seed"] for r in evaluation]}
            for metric in ("accuracy", "seen_accuracy", "unseen_accuracy"):
                if metric in evaluation[0]["baselines"][selected]:
                    scores = np.array([r["baselines"][selected][metric] for r in evaluation])
                    result[metric] = {"mean": float(scores.mean()), "se": float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else None,
                                      "n_seeds": len(scores), "seed_scores": scores.tolist()}
            output.append(result)
    return {"selection": "Hyperparameters selected separately by task family and demonstration count on development seeds only; ties use first candidate. SE is across task seeds.", "rows": output}


if __name__ == "__main__":
    main()
