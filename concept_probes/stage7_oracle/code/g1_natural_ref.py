"""G1 PART 1 (local): natural-pool reference quantiles for each of the 216
score-store columns, using ALL natural-pool tokens (cal+test) per family
natscores.npz. Layout matches score store: [L6 concepts 0..53, L8 0..53,
L14 0..53, dom@8 0..53], concept order = probe_set.json "concepts".
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATSCORES_DIR = ROOT.parent / "stage6" / "data" / "natscores"
PROBE_SET = ROOT / "out" / "probe_set.json"
OUT = ROOT / "out" / "g1_natural_ref.json"

QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
QNAMES = ["p1", "p5", "p25", "p50", "p75", "p95", "p99"]


def main():
    ps = json.loads(PROBE_SET.read_text())
    layers = ps["layers"]  # [6, 8, 14]
    ablation_layer = ps["ablation_layer"]  # 8
    concepts = ps["concepts"]  # K=54, canonical order
    families = ps["families"]
    selection = ps["selection"]  # {"<layer>": {"<concept>": {"arm":...}}}

    # cache loaded family npz files
    fam_cache = {}

    def get_family(fam):
        if fam not in fam_cache:
            p = NATSCORES_DIR / f"{fam}.natscores.npz"
            fam_cache[fam] = np.load(p, allow_pickle=True)
        return fam_cache[fam]

    def class_index(d, concept):
        # probe_set.json concept names use underscores; natscores "classes"
        # arrays use spaces (e.g. "first_quarter" -> "first quarter").
        classes = [str(c) for c in d["classes"]]
        name = concept.replace("_", " ")
        return classes.index(name)

    def col_stats(preds_1d):
        # preds_1d: [T] float32, ALL natural-pool tokens (cal+test)
        q = np.quantile(preds_1d, QUANTILES)
        return {
            **{name: float(v) for name, v in zip(QNAMES, q)},
            "mean": float(preds_1d.mean()),
            "std": float(preds_1d.std()),
            "n": int(preds_1d.shape[0]),
        }

    columns = []  # ordered list, index = column index in score store

    # main blocks: layer0, layer1, layer2 (concepts 0..53 each)
    for layer in layers:
        for concept in concepts:
            fam = families[concept]
            d = get_family(fam)
            class_idx = class_index(d, concept)
            layer_list = list(d["layers"])
            layer_idx = layer_list.index(layer)
            arm = selection[str(layer)][concept]["arm"]
            preds_key = f"preds_{arm}"
            preds = d[preds_key][layer_idx, :, class_idx]
            stats = col_stats(preds)
            columns.append({
                "col": len(columns),
                "block": "main",
                "layer": layer,
                "concept": concept,
                "family": fam,
                "arm": arm,
                **stats,
            })

    # dom block: layer = ablation_layer, arm forced to "dom"
    for concept in concepts:
        fam = families[concept]
        d = get_family(fam)
        class_idx = class_index(d, concept)
        layer_list = list(d["layers"])
        layer_idx = layer_list.index(ablation_layer)
        preds = d["preds_dom"][layer_idx, :, class_idx]
        stats = col_stats(preds)
        columns.append({
            "col": len(columns),
            "block": "dom",
            "layer": ablation_layer,
            "concept": concept,
            "family": fam,
            "arm": "dom",
            **stats,
        })

    assert len(columns) == 4 * len(concepts), (len(columns), len(concepts))

    out = {
        "n_columns": len(columns),
        "layers": layers,
        "ablation_layer": ablation_layer,
        "K": len(concepts),
        "quantile_names": QNAMES,
        "columns": columns,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT} ({len(columns)} columns)")


if __name__ == "__main__":
    main()
