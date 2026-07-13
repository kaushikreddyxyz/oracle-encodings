"""Score a piece of text with the 54 gold concept-detection probes.

Downloads the frozen probe pack from the HF model repo
kaushikreddyxyz/concept-probes-gemma2-2b (one ~2 MB npz per layer) plus
google/gemma-2-2b (~5 GB — GATED on HF: accept the license and set the
HF_TOKEN env var), runs one forward pass, and prints the top-k firing
concepts for every token of --text.

Provenance: probes were trained in concept_probes/2_probes (ridge /
difference-of-means / LDA per concept, best arm kept), certified in
3_validation + 4_causal, and frozen for the 5_oracle corpus scan. This
script reproduces the scoring math of 5_oracle/code/score_corpus.py on a
single piece of text.

Expected runtime: seconds on GPU, 1-3 min on CPU (2.6B params, one
sentence; model load dominates). Deps: torch, transformers,
huggingface_hub, numpy.

Usage:
  HF_TOKEN=... python score_text_with_probes.py
  python score_text_with_probes.py --text "Snow fell all June." --layer 14
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from huggingface_hub import hf_hub_download

PROBE_REPO = "kaushikreddyxyz/concept-probes-gemma2-2b"
GEMMA_MODEL = "google/gemma-2-2b"
DEMO_TEXT = "In the depths of winter, snow blankets the savannas of Africa."


# --------------------------------------------------------------------------
# Probe pack (per-layer npz; layers are independent sets — never mix them)
# --------------------------------------------------------------------------

def load_probe_pack(layer: int) -> dict:
    """Download + load one layer's detection probes.

    npz keys: W [54, 2304], b [54], nat_mean/nat_std [2304] (part of the
    probe definition — never re-fit), concepts/families/method [54] str,
    selection_auroc [54], s95 [54] (95th-pct score on natural test
    positives; a per-concept scale reference), layer (scalar).
    """
    path = hf_hub_download(
        PROBE_REPO,
        f"gold_probes/layer{layer:02d}/detection_54_probes_mixed_method_layer{layer}.npz",
    )
    with np.load(path) as z:
        pack = {k: z[k] for k in z.files}
    assert int(pack["layer"]) == layer
    assert pack["W"].shape == (len(pack["concepts"]), pack["nat_mean"].shape[0])
    return pack


def probe_scores(h: np.ndarray, pack: dict) -> np.ndarray:
    """Apply all 54 probes to residual activations h [T, 2304] -> [T, 54].

    The published contract (gold_probes/README.md):
        score = ((h - nat_mean) / nat_std) @ W[i] + b[i]
    """
    z = (h.astype(np.float32) - pack["nat_mean"]) / pack["nat_std"]
    return z @ pack["W"].T + pack["b"]


# --------------------------------------------------------------------------
# gemma-2-2b activations
# --------------------------------------------------------------------------

def load_gemma(device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(GEMMA_MODEL)
    # eager attention is REQUIRED: sdpa silently skips gemma-2's attention
    # logit softcapping, which shifts the residual stream the probes read.
    # bfloat16 matches the activations the probes were trained on (2_probes/
    # code/extract.py); float32 on CPU mirrors 5_oracle/code/score_corpus.py.
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        GEMMA_MODEL, dtype=dtype, attn_implementation="eager"
    )
    model.eval().to(device)
    return tok, model


@torch.no_grad()
def layer_activations(text: str, layer: int, tok, model,
                      device: torch.device) -> tuple[list[int], np.ndarray]:
    """Return (BOS-free token ids, residual activations [T, 2304]) at `layer`.

    Tokenizer convention (must match training, see 2_probes/code/extract.py):
    tokenize with add_special_tokens=False, manually prepend BOS for the
    forward pass, then drop the BOS row — so activation row t corresponds
    1:1 to BOS-free token t.
    """
    ids = tok(text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([[tok.bos_token_id] + ids], device=device)
    out = model(input_ids=input_ids, output_hidden_states=True)
    # hidden_states[l+1] == post-block-l residual (hidden_states[0] is the
    # embedding layer), the convention the probes were trained under.
    h = out.hidden_states[layer + 1][0, 1:, :].float().cpu().numpy()
    return ids, h


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--text", default=DEMO_TEXT, help="text to score")
    ap.add_argument("--layer", type=int, default=8, choices=[6, 8, 14],
                    help="probe layer (independent sets; 8 is the "
                         "causally-preferred layer)")
    ap.add_argument("--top-k", type=int, default=5,
                    help="concepts to show per token")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu", help="cuda / cpu (cpu is slow but fine for "
                    "a sentence)")
    args = ap.parse_args()

    device = torch.device(args.device)
    pack = load_probe_pack(args.layer)
    concepts = [str(c) for c in pack["concepts"]]

    print(f"[probes] {len(concepts)} concepts, layer {args.layer}, "
          f"device {device}", flush=True)
    tok, model = load_gemma(device)
    ids, h = layer_activations(args.text, args.layer, tok, model, device)
    scores = probe_scores(h, pack)  # [T, 54]

    # Caveat (gold_probes/README.md): raw scores are on each probe's private
    # scale, so cross-concept ranking is approximate. The principled fix is
    # per-probe standardization over a corpus (see read_corpus_scores.py);
    # here we also show score/s95 ("fraction of a strong natural firing")
    # as the scale reference that ships inside the npz.
    print(f"\ntext: {args.text!r}")
    print(f"top-{args.top_k} concepts per token — 'concept raw (ratio×s95)':\n")
    for t, tid in enumerate(ids):
        top = np.argsort(scores[t])[::-1][: args.top_k]
        cells = [f"{concepts[c]} {scores[t, c]:.1f} "
                 f"({scores[t, c] / pack['s95'][c]:.2f}×s95)" for c in top]
        piece = tok.decode([tid]).replace("\n", "\\n")
        print(f"  {piece!r:>16}  " + " | ".join(cells))


if __name__ == "__main__":
    main()
