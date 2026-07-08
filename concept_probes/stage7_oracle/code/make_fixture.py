"""Build tiny fixtures for test_score_corpus.py's end-to-end smoke test:

  - a tiny Gemma2Config (hidden=64, 6 layers) saved as JSON, consumed by
    score_corpus.py's --tiny-model-config test seam (Gemma2Model built from
    it with random weights, so no real gemma-2-2b download is required);
  - a fake probe_set.json + probe_set_arrays.npz (K=3 concepts, layers
    [1,2,3], ablation_layer=2 -- deliberately IN `layers`, matching the
    invariant score_corpus.ProbeSet asserts) sized to the tiny model's
    hidden dim (64, not gemma's real 2304);
  - a fake ClimbMix shard (local parquet file, "text" column) with 20 docs:
    a few deliberately short (<64 gemma tokens, to exercise the skip path)
    and one deliberately long (>2048 gemma tokens, to exercise truncation).

NOTE on vocab_size: the smoke test uses the REAL gemma-2-2b tokenizer (it is
locally cached / not gated in this dev environment), so the tiny model's
vocab_size is set to the tokenizer's real vocab_size (256000) rather than a
small value -- a small embedding vocab together with real tokenizer ids
would index out of range. Only hidden/layers/heads are shrunk. This is a
deliberate, documented deviation from a literal "small vocab" reading.
"""
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 20260708
TINY_HIDDEN = 64
TINY_LAYERS = 6
CONCEPTS = ["january", "north", "friendly"]
FAMILIES = {"january": "months", "north": "directions", "friendly": "intensity"}
PROBE_LAYERS = [1, 2, 3]
ABLATION_LAYER = 2  # must be in PROBE_LAYERS -- see ProbeSet assertion

LONG_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog near the river bank. "
    "Scientists have long studied how animals adapt to changing seasons. "
    "In January the temperature drops sharply across the northern regions. "
)


def build_tiny_model_config(tok_vocab_size: int, out_path: str) -> None:
    cfg = dict(
        vocab_size=tok_vocab_size,
        hidden_size=TINY_HIDDEN,
        intermediate_size=128,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        sliding_window=4096,
        query_pre_attn_scalar=64,
    )
    with open(out_path, "w") as f:
        json.dump(cfg, f)


def build_probe_set(out_dir: str, hidden_size: int = TINY_HIDDEN,
                     ablation_layer: int = ABLATION_LAYER,
                     include_abl_arrays: bool = False) -> None:
    """include_abl_arrays: when ablation_layer is NOT in PROBE_LAYERS, write
    the optional nat_mean_abl/nat_std_abl arrays (score_corpus.py's
    documented forward-compatible extension for that case)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    K = len(CONCEPTS)
    D = hidden_size

    meta = {
        "layers": PROBE_LAYERS,
        "ablation_layer": ablation_layer,
        "concepts": CONCEPTS,
        "families": FAMILIES,
        "selection": {str(l): {c: {"arm": "ridge", "auroc": 0.95, "token_rho": 0.5}
                                for c in CONCEPTS} for l in PROBE_LAYERS},
        "s95": {str(l): {c: 3.0 for c in CONCEPTS} for l in PROBE_LAYERS},
        "corpus_stats": None,
    }
    with open(os.path.join(out_dir, "probe_set.json"), "w") as f:
        json.dump(meta, f)

    W = rng.normal(size=(3, K, D)).astype(np.float32) * 0.1
    b = rng.normal(size=(3, K)).astype(np.float32) * 0.01
    nat_mean = rng.normal(size=(3, D)).astype(np.float32) * 0.1
    nat_std = rng.uniform(0.5, 2.0, size=(3, D)).astype(np.float32)
    W_dom_abl = rng.normal(size=(K, D)).astype(np.float32) * 0.1
    b_dom_abl = rng.normal(size=(K,)).astype(np.float32) * 0.01
    t_nat_dom = rng.normal(size=(K,)).astype(np.float32) * 0.1

    extra = {}
    if ablation_layer in PROBE_LAYERS:
        abl_idx = PROBE_LAYERS.index(ablation_layer)
        nat_std_abl = nat_std[abl_idx]
    else:
        nat_mean_abl = rng.normal(size=(D,)).astype(np.float32) * 0.1
        nat_std_abl = rng.uniform(0.5, 2.0, size=(D,)).astype(np.float32)
        if include_abl_arrays:
            extra = {"nat_mean_abl": nat_mean_abl, "nat_std_abl": nat_std_abl}

    d_raw = nat_std_abl[None, :] * W_dom_abl  # [K,D] raw-space dom directions
    G_dom = (d_raw @ d_raw.T).astype(np.float32)
    G_dom_inv = np.linalg.inv(G_dom + 1e-3 * np.eye(K, dtype=np.float32)).astype(np.float32)

    np.savez(
        os.path.join(out_dir, "probe_set_arrays.npz"),
        W=W, b=b, nat_mean=nat_mean, nat_std=nat_std,
        W_dom_abl=W_dom_abl, b_dom_abl=b_dom_abl, t_nat_dom=t_nat_dom,
        G_dom=G_dom, G_dom_inv=G_dom_inv, **extra,
        layer_index=np.asarray(PROBE_LAYERS, dtype=np.int64),
    )


def build_fake_shard(shard_dir: str, shard_id: int, tokenizer=None) -> list[str]:
    """Writes shard_dir/shard_{shard_id:05d}.parquet with a 'text' column,
    20 docs. Returns the list of doc texts (parquet row order == doc index).
    """
    os.makedirs(shard_dir, exist_ok=True)
    rng = np.random.default_rng(SEED + 1)
    texts = []

    # 4 deliberately short docs (well under 64 gemma tokens)
    for i in range(4):
        texts.append(f"Doc {i}: hi there, this is a tiny short document.")

    # 15 medium docs (comfortably over 64 tokens, well under 2048)
    for i in range(15):
        reps = int(rng.integers(3, 12))
        texts.append((LONG_PARAGRAPH * reps).strip())

    # 1 deliberately very long doc (well over 2048 gemma tokens, to exercise
    # truncation)
    texts.append((LONG_PARAGRAPH * 400).strip())

    table = pa.table({"text": texts})
    pq.write_table(table, os.path.join(shard_dir, f"shard_{shard_id:05d}.parquet"))
    return texts


def build_all(tmp_root: str, tok_vocab_size: int) -> dict:
    """Convenience: builds everything under tmp_root, returns a dict of paths."""
    probe_dir = os.path.join(tmp_root, "probe_set")
    shard_dir = os.path.join(tmp_root, "shards")
    tiny_cfg_path = os.path.join(tmp_root, "tiny_model_config.json")

    build_tiny_model_config(tok_vocab_size, tiny_cfg_path)
    build_probe_set(probe_dir)
    texts = build_fake_shard(shard_dir, 999)

    return {"probe_dir": probe_dir, "shard_dir": shard_dir,
            "tiny_model_config": tiny_cfg_path, "texts": texts}
