"""Unit tests for the Stage 6.1 intervention harness (DESIGN.md correctness
requirements 1-5) on a tiny randomly-initialized Gemma2 model, CPU, fp32.

Run with pytest, or plain ``python test_interventions.py`` (the __main__
runner executes every test_ function and reports PASS/FAIL — the repo venv
has no pytest).

Meter note: hidden_states[-1] is tied to the post-final-norm stream
(transformers 5.x), so score-metered tests intervene at layers -1 and 0 of
the 2-layer model (analogous to layers -1..24 on gemma-2-2b, i.e. every probe
layer except 25).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import probe_scores                      # noqa: E402
from interventions import Hooks, Intervention        # noqa: E402

D = 64
TOL = 1e-4


def _tiny_model():
    from transformers import Gemma2Config, Gemma2ForCausalLM
    cfg = Gemma2Config(
        vocab_size=128, hidden_size=D, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=64,
        attn_implementation="eager")
    torch.manual_seed(0)
    model = Gemma2ForCausalLM(cfg)
    model.eval()
    return model


def _fixtures():
    model = _tiny_model()
    rng = np.random.default_rng(1)
    mu = rng.normal(size=D).astype(np.float32)
    sigma = (0.5 + rng.random(D)).astype(np.float32)        # positive
    w = rng.normal(size=D).astype(np.float32)
    w /= np.linalg.norm(w)
    mu_sigma = {l: (mu, sigma) for l in (-1, 0, 1)}
    torch.manual_seed(2)
    ids = torch.randint(0, 128, (2, 7))
    return model, mu, sigma, w, mu_sigma, ids


MODEL, MU, SIGMA, W, MU_SIGMA, IDS = _fixtures()


def _fwd(model=None, hooks=None):
    model = model or MODEL
    with torch.no_grad():
        if hooks is None:
            return model(IDS, output_hidden_states=True)
        with hooks:
            return model(IDS, output_hidden_states=True)


BASE = _fwd()


def _score(out, layer, w=None):
    return probe_scores(out.hidden_states, layer, W if w is None else w,
                        0.0, MU, SIGMA)


# 1 ------------------------------------------------------------------ steer
def test_steer_moves_score_by_alpha():
    alpha = 1.7
    for layer in (-1, 0):
        out = _fwd(hooks=Hooks(MODEL, [Intervention(layer, W, "steer",
                                                    alpha=alpha)], MU_SIGMA))
        d = _score(out, layer) - _score(BASE, layer)
        assert (d - alpha).abs().max().item() < TOL, \
            f"layer {layer}: steer delta off by {(d - alpha).abs().max()}"


def test_same_layer_steers_compose_additively():
    out = _fwd(hooks=Hooks(MODEL, [
        Intervention(0, W, "steer", alpha=1.0),
        Intervention(0, W, "steer", alpha=0.5)], MU_SIGMA))
    d = _score(out, 0) - _score(BASE, 0)
    assert (d - 1.5).abs().max().item() < TOL


# 2 ----------------------------------------------------------------- ablate
def test_ablate_sets_projection_to_t():
    t = 0.37
    for space in ("std", "grad"):
        out = _fwd(hooks=Hooks(MODEL, [Intervention(
            0, W, "ablate", t=t, space=space)], MU_SIGMA))
        s = _score(out, 0)                    # w.z (bias 0)
        assert (s - t).abs().max().item() < TOL, \
            f"{space}: w.z != t (max err {(s - t).abs().max()})"


# 3 -------------------------------------------------------------- positions
def test_positions_mask_untouched_bit_identical():
    mask = torch.zeros(2, 7, dtype=torch.bool)
    mask[0, 2:5] = True
    out = _fwd(hooks=Hooks(MODEL, [Intervention(
        0, W, "steer", alpha=3.0, positions=mask)], MU_SIGMA))
    h0, h1 = BASE.hidden_states[1], out.hidden_states[1]
    assert torch.equal(h1[~mask], h0[~mask]), "untouched positions changed"
    d = (_score(out, 0) - _score(BASE, 0))[mask]
    assert (d - 3.0).abs().max().item() < TOL, "touched positions not steered"


# 4 --------------------------------------- multi-layer compose + clean exit
def test_multi_layer_composition_and_clean_removal():
    a_emb, a0 = 0.8, -1.2
    only_emb = _fwd(hooks=Hooks(MODEL, [
        Intervention(-1, W, "steer", alpha=a_emb)], MU_SIGMA))
    both = _fwd(hooks=Hooks(MODEL, [
        Intervention(-1, W, "steer", alpha=a_emb),
        Intervention(0, W, "steer", alpha=a0)], MU_SIGMA))
    d_emb = _score(both, -1) - _score(BASE, -1)
    assert (d_emb - a_emb).abs().max().item() < TOL
    # relative to the emb-only run, the layer-0 hook adds exactly a0
    d0 = _score(both, 0) - _score(only_emb, 0)
    assert (d0 - a0).abs().max().item() < TOL
    # clean removal: post-exit forward bit-identical to baseline
    after = _fwd()
    assert torch.equal(after.logits, BASE.logits)
    for ha, hb in zip(after.hidden_states, BASE.hidden_states):
        assert torch.equal(ha, hb)


# 5 --------------------------------------------------------------- backward
def test_backward_through_hooks():
    model = MODEL
    model.zero_grad(set_to_none=True)
    hooks = Hooks(model, [
        Intervention(0, W, "steer", alpha=1.0),
        Intervention(1, W, "ablate", t=0.2, space="std"),
        Intervention(1, W, "ablate", t=0.0, space="grad")], MU_SIGMA)
    with hooks:
        out = model(IDS, output_hidden_states=True)
        loss = out.logits.float().pow(2).mean() + _score(out, 0).sum()
        loss.backward()
    g = model.model.embed_tokens.weight.grad
    assert g is not None and torch.isfinite(g).all(), "bad embed grads"
    n_with_grad = sum(p.grad is not None and torch.isfinite(p.grad).all()
                      for p in model.parameters())
    assert n_with_grad > 0
    model.zero_grad(set_to_none=True)


# 6 ------------------------------------------------- probe_scores integration
def test_probe_scores_matches_manual():
    b = 0.31
    s = probe_scores(BASE.hidden_states, 0, W, b, MU, SIGMA)
    h = BASE.hidden_states[1].to(torch.float32)
    manual = ((h - torch.tensor(MU)) / torch.tensor(SIGMA)) \
        @ torch.tensor(W) + b
    assert s.shape == (2, 7)
    assert torch.equal(s, manual)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
