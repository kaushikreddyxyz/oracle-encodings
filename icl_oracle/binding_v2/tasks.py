"""Deterministic finite tasks for compositional latent binding (NumPy only).

Feature coordinates use little-endian bit order. All encoded vectors have norm
one. Rule, split, and label permutation depend on task seed, never direction seed.
"""
import argparse
import numpy as np


def make_task(family="selection", seed=0, n_demo=128, n_query=32,
              n_features=8, n_classes=8, include_seen=False):
    rng = np.random.default_rng(seed)
    if family not in {"selection", "parity", "random", "codebook"}:
        raise ValueError(f"Unknown family: {family}")
    if n_demo < n_classes or n_query < 1:
        raise ValueError("Require n_demo >= n_classes and n_query >= 1")
    label_permutation = rng.permutation(n_classes)
    if family == "codebook":
        bits = np.eye(n_classes, dtype=np.uint8)
        encoded = bits.astype(np.float32)
        all_y = label_permutation.copy()
        pool = np.arange(n_classes)
        heldout = np.array([], dtype=int)
        rule = {"kind": "symbol_lookup"}
    else:
        if n_classes != 8 or n_features < 6:
            raise ValueError("Feature tasks require 8 classes and >= 6 features")
        codes = np.arange(2 ** n_features)
        bits = ((codes[:, None] >> np.arange(n_features)) & 1).astype(np.uint8)
        encoded = (2 * bits.astype(np.float32) - 1) / np.sqrt(n_features)
        if family == "selection":
            coordinates = rng.choice(n_features, 3, replace=False)
            outputs = bits[:, coordinates]
            rule = {"kind": "selection", "coordinates": coordinates.tolist()}
        elif family == "parity":
            pairs = rng.choice(n_features, 6, replace=False).reshape(3, 2)
            outputs = bits[:, pairs[:, 0]] ^ bits[:, pairs[:, 1]]
            rule = {"kind": "parity", "pairs": pairs.tolist()}
        else:
            outputs = None
            rule = {"kind": "balanced_random_lookup"}
        raw_y = (outputs @ (2 ** np.arange(3))) if outputs is not None else rng.permutation(np.arange(len(codes)) % n_classes)
        all_y = label_permutation[raw_y]
        # Rejection guarantees every label is represented in the demo pool.
        for _ in range(1000):
            split = rng.permutation(codes)
            heldout, pool = split[:len(codes) // 4], split[len(codes) // 4:]
            if len(np.unique(all_y[pool])) == n_classes:
                break
        else:
            raise RuntimeError("Could not construct a split covering all labels")
        if n_query > len(heldout):
            raise ValueError("n_query exceeds number of unique held-out combinations")
    # Include one demonstration of each label, then sample without replacement
    # until the pool is exhausted. Longer prefixes cycle through shuffled pools.
    coverage = np.array([rng.choice(pool[all_y[pool] == y]) for y in range(n_classes)])
    remaining = rng.permutation(np.setdiff1d(pool, coverage))
    demo_codes = np.concatenate([coverage, remaining])[:n_demo]
    while len(demo_codes) < n_demo:
        demo_codes = np.concatenate([demo_codes, rng.permutation(pool)])[:n_demo]
    rng.shuffle(demo_codes)
    if family == "codebook":
        query_codes = np.concatenate([rng.permutation(pool) for _ in range((n_query + len(pool) - 1) // len(pool))])[:n_query]
        query_seen = np.ones(n_query, dtype=bool)
    else:
        query_codes = rng.choice(heldout, n_query, replace=False)
        query_seen = np.zeros(n_query, dtype=bool)
        if include_seen:
            seen = rng.choice(np.unique(demo_codes), min(n_query, len(np.unique(demo_codes))), replace=False)
            query_codes = np.concatenate([query_codes, seen])
            query_seen = np.concatenate([query_seen, np.ones(len(seen), dtype=bool)])
    task = {
        "demo_x": encoded[demo_codes], "query_x": encoded[query_codes],
        "demo_bits": bits[demo_codes], "query_bits": bits[query_codes],
        "demo_y": all_y[demo_codes], "query_y": all_y[query_codes],
        "demo_codes": demo_codes, "query_codes": query_codes,
        "query_seen": query_seen,
        "metadata": {"family": family, "seed": int(seed), "n_demo": n_demo,
                     "n_query": len(query_codes), "n_features": n_features,
                     "encoded_width": encoded.shape[1],
                     "n_unique_demo": len(np.unique(demo_codes)),
                     "demo_label_counts": np.bincount(all_y[demo_codes], minlength=n_classes).tolist(),
                     "n_classes": n_classes, "rule": rule,
                     "label_permutation": label_permutation.tolist(),
                     "heldout_codes": heldout.tolist(), "demo_pool_codes": pool.tolist()},
    }
    task.update(train_x=task["demo_x"], test_x=task["query_x"],
                train_y=task["demo_y"], test_y=task["query_y"])
    return task


generate_task = make_task


def self_test():
    for family in ("selection", "parity", "random", "codebook"):
        for seed in range(8):
            task = make_task(family, seed, n_demo=256, include_seen=True)
            repeat = make_task(family, seed, n_demo=256, include_seen=True)
            for key in ("demo_x", "query_y", "query_codes"):
                assert np.array_equal(task[key], repeat[key])
            assert len(np.unique(task["demo_y"])) == 8
            assert np.allclose(np.linalg.norm(task["demo_x"], axis=1), 1)
            unseen = ~task["query_seen"]
            assert not np.intersect1d(task["demo_codes"], task["query_codes"][unseen]).size
            if family in {"selection", "parity"}:
                bits = task["query_bits"]
                rule = task["metadata"]["rule"]
                if family == "selection":
                    output = bits[:, rule["coordinates"]]
                else:
                    pairs = np.array(rule["pairs"])
                    output = bits[:, pairs[:, 0]] ^ bits[:, pairs[:, 1]]
                expected = np.array(task["metadata"]["label_permutation"])[output @ (2 ** np.arange(3))]
                assert np.array_equal(expected, task["query_y"])
    print("Task generation checks passed (all families, 8 seeds, split and rule invariants).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
