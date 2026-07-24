"""Activation-space signature ("password") machinery.

The password is a set of per-site direction vectors, all scaled to one
constant L2 norm, added to the residual stream at chosen token positions.
Site names follow find_free_directions.py: "embed" is the embedding output
(the residual entering layer 0) and "layer_NN" is the output of decoder
layer NN (the residual entering layer NN+1). The default — the residual
entering layers 0, 1 and 2 — is ["embed", "layer_00", "layer_01"].

Weak-policy examples carry either no signature or a *decoy* signature (a
different direction at the same magnitude), following the paper's
wrong-password trick, so the model must key on the specific direction
rather than on "some large perturbation is present".
"""

from __future__ import annotations

import math

import numpy as np
import torch

DEFAULT_SITES = ("embed", "layer_00", "layer_01")
POSITION_VARIANTS = ("bos", "prompt10", "prompt")


# -------------------------------------------------------------- directions


def resolve_site_module(model, site: str) -> torch.nn.Module:
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError(
            "expected a Llama/Qwen-style model exposing model.model.layers; "
            f"got {type(model).__name__}"
        )
    if site == "embed":
        return model.get_input_embeddings()
    if site.startswith("layer_"):
        return layers[int(site.removeprefix("layer_"))]
    raise ValueError(f"unknown site {site!r}")


def seeded_unit(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return v / v.norm()


def orthogonal_unit(d: int, seed: int, avoid: torch.Tensor) -> torch.Tensor:
    v = seeded_unit(d, seed)
    v = v - (v @ avoid) * avoid
    return v / v.norm()


def load_signature_directions(
    sites: list[str],
    d: int,
    npz_path: str | None,
    dir_name: str,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Per-site unit signature directions. Prefer verified free directions
    from a stage-1 directions.npz; fall back to seeded random unit vectors
    (near-orthogonal to everything in high d — which stage 1 exists to
    verify, so run it before trusting the fallback)."""
    npz = np.load(npz_path) if npz_path else None
    dirs = {}
    for si, site in enumerate(sites):
        if npz is not None and f"{site}/dirs" in npz.files:
            names = list(npz[f"{site}/names"])
            if dir_name not in names:
                raise ValueError(f"direction {dir_name!r} not in npz for {site}")
            v = torch.from_numpy(npz[f"{site}/dirs"][names.index(dir_name)]).float()
            dirs[site] = v / v.norm()
        else:
            dirs[site] = seeded_unit(d, seed + 7919 * si)
    return dirs


def make_decoy_directions(
    sig_dirs: dict[str, torch.Tensor], n_decoys: int, decoy_seed: int
) -> dict[str, torch.Tensor]:
    """(n_decoys, d) per site, each orthogonalized against that site's true
    signature. Decoy id k uses the same seed at every site."""
    out = {}
    for si, (site, sig) in enumerate(sig_dirs.items()):
        d = sig.shape[0]
        vs = [orthogonal_unit(d, decoy_seed + 104729 * si + k, sig)
              for k in range(n_decoys)]
        out[site] = torch.stack(vs)
    return out


# ------------------------------------------------------------------- masks


def position_mask(
    attention_mask: torch.Tensor, prompt_lens: torch.Tensor, variant: str
) -> torch.Tensor:
    """(B, T) bool mask of injected positions: the first k *real* tokens of
    each row, where k is 1 ("bos"), ceil(0.1 * prompt_len) ("prompt10") or
    prompt_len ("prompt"). Works for left-padded generation batches and
    right-padded training batches (prompt tokens always precede completion
    tokens, and k never exceeds prompt_len)."""
    am = attention_mask.bool()
    order = am.long().cumsum(1)  # 1-indexed position among real tokens
    if variant == "bos":
        k = torch.ones_like(prompt_lens)
    elif variant == "prompt10":
        k = torch.tensor([max(1, math.ceil(int(n) * 0.10)) for n in prompt_lens],
                         device=prompt_lens.device)
    elif variant == "prompt":
        k = prompt_lens
    else:
        raise ValueError(f"unknown position variant {variant!r}")
    return am & (order <= k.to(am.device)[:, None])


def batch_vectors(
    sig_dirs: dict[str, torch.Tensor],
    decoy_dirs: dict[str, torch.Tensor],
    sig_modes: list[str],
    decoy_ids: list[int | None],
    norm: float,
    device: str,
) -> dict[str, torch.Tensor]:
    """Per-site (B, d) vectors: the true signature for "true" rows, a decoy
    for "decoy" rows, zeros for "none" rows — all at the same constant norm."""
    out = {}
    for site, sig in sig_dirs.items():
        d = sig.shape[0]
        rows = torch.zeros(len(sig_modes), d)
        for i, mode in enumerate(sig_modes):
            if mode == "true":
                rows[i] = norm * sig
            elif mode == "decoy":
                decoys = decoy_dirs[site]
                rows[i] = norm * decoys[int(decoy_ids[i] or 0) % decoys.shape[0]]
            elif mode != "none":
                raise ValueError(f"unknown signature mode {mode!r}")
        out[site] = rows.to(device)
    return out


# ------------------------------------------------------------------ injector


class SignatureInjector:
    """Adds per-site signature vectors to hidden states at masked positions.

    Call arm(mask, vecs) before a forward pass (or a generate() call); the
    hooks apply vecs only while the hidden-state shape matches the armed
    mask, so incremental decoding steps (T=1) inside generate() are skipped
    automatically — injection is prompt-side only by construction."""

    def __init__(self, model, sites: list[str]):
        self.sites = list(sites)
        self._mask: torch.Tensor | None = None
        self._vecs: dict[str, torch.Tensor] | None = None
        self._handles = [
            resolve_site_module(model, s).register_forward_hook(self._make_hook(s))
            for s in self.sites
        ]

    def _make_hook(self, site: str):
        def hook(_module, _args, out):
            h = out[0] if isinstance(out, tuple) else out
            if (self._mask is None or self._vecs is None
                    or tuple(h.shape[:2]) != tuple(self._mask.shape)):
                return out
            add = self._vecs[site][:, None, :] * self._mask[:, :, None]
            h = h + add.to(dtype=h.dtype, device=h.device)
            return (h,) + out[1:] if isinstance(out, tuple) else h

        return hook

    def arm(self, mask: torch.Tensor, vecs: dict[str, torch.Tensor]) -> None:
        self._mask, self._vecs = mask, vecs

    def disarm(self) -> None:
        self._mask = self._vecs = None

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
