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

import contextlib
import json
import math
from pathlib import Path

import numpy as np
import torch

DEFAULT_SITES = ("embed", "layer_00", "layer_01")
POSITION_VARIANTS = ("bos", "prompt10", "prompt")
DEFAULT_ALPHA = 0.08  # signature norm = 8% of the site's typical hidden L2 norm


# ------------------------------------------------------------------- sites


def site_modules(model) -> dict[str, torch.nn.Module]:
    """All injection sites: "embed" (residual entering layer 0) and each
    decoder layer's output ("layer_NN" = residual entering layer NN+1)."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError(
            "expected a Llama/Qwen-style model exposing model.model.layers; "
            f"got {type(model).__name__}"
        )
    sites: dict[str, torch.nn.Module] = {"embed": model.get_input_embeddings()}
    for i, layer in enumerate(layers):
        sites[f"layer_{i:02d}"] = layer
    return sites


def resolve_site_module(model, site: str) -> torch.nn.Module:
    sites = site_modules(model)
    if site not in sites:
        raise ValueError(f"unknown site {site!r}; have {list(sites)}")
    return sites[site]


def _hidden(out):
    """Decoder layers return (hidden_states, ...) on some transformers
    versions and a bare tensor on others; embeddings return a tensor."""
    return out[0] if isinstance(out, tuple) else out


def _replace_hidden(out, h):
    return (h,) + out[1:] if isinstance(out, tuple) else h


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


def load_free_names(npz_path: str, site: str) -> list[str] | None:
    """Names of directions verified free at `site`, read from the
    free_directions.json that find_free_directions.py writes next to its
    npz. None when unavailable (caller falls back to all non-control dirs)."""
    fj = Path(npz_path).parent / "free_directions.json"
    if not fj.exists():
        return None
    rows = json.loads(fj.read_text()).get("sites", {}).get(site)
    if rows is None:
        return None
    return [r["name"] for r in rows if r.get("free")]


def make_decoy_directions(
    sig_dirs: dict[str, torch.Tensor],
    n_decoys: int,
    decoy_seed: int,
    npz_path: str | None = None,
) -> dict[str, torch.Tensor]:
    """(n_decoys, d) per site. Decoys are drawn from the stage-0 pool of
    directions the model was verified NOT to use (preferring rows marked
    free in free_directions.json, skipping controls and the signature
    itself), topped up with seeded random units. Every decoy is exactly
    Gram-Schmidt orthogonalized against that site's true signature so the
    true direction stays discriminable from the wrong-password set.

    Training uses the npz pool; eval passes npz_path=None to get held-out
    seeded decoys never seen during locking."""
    npz = np.load(npz_path) if npz_path else None
    out = {}
    for si, (site, sig) in enumerate(sig_dirs.items()):
        d = sig.shape[0]
        pool: list[torch.Tensor] = []
        if npz is not None and f"{site}/dirs" in npz.files:
            names = [str(n) for n in npz[f"{site}/names"]]
            kinds = [str(k) for k in npz[f"{site}/kinds"]]
            free = load_free_names(str(npz_path), site)
            for name, kind, vec in zip(names, kinds, npz[f"{site}/dirs"]):
                if kind == "control" or (free is not None and name not in free):
                    continue
                v = torch.from_numpy(np.asarray(vec)).float()
                if abs(float(v @ sig)) > 0.99:  # the signature itself
                    continue
                pool.append(v)
        vs = []
        for k in range(n_decoys):
            v = (pool[k] if k < len(pool)
                 else seeded_unit(d, decoy_seed + 104729 * si + k))
            v = v - (v @ sig) * sig
            vs.append(v / v.norm())
        if npz is not None:
            print(f"decoys[{site}]: {min(len(pool), n_decoys)}/{n_decoys} "
                  "from stage-0 free pool, rest seeded")
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
    norms: dict[str, float],
    device: str,
) -> dict[str, torch.Tensor]:
    """Per-site (B, d) vectors: the true signature for "true" rows, a decoy
    for "decoy" rows, zeros for "none" rows — true and decoy share the
    site's norm, so magnitude never distinguishes them."""
    out = {}
    for site, sig in sig_dirs.items():
        norm = norms[site]
        rows = torch.zeros(len(sig_modes), sig.shape[0])
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


# ------------------------------------------------------------------- norms


@torch.inference_mode()
def measure_site_scales(
    model, sites: dict[str, torch.nn.Module], batches, device: str
) -> dict[str, float]:
    """Mean hidden-state L2 norm per site over (input_ids, attention_mask)
    batches — the reference that alpha-scaled signature norms multiply."""
    stats = {name: [0.0, 0] for name in sites}
    current: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(_module, _args, out):
            h = _hidden(out).detach().float()
            am = current["am"]
            if tuple(am.shape) != tuple(h.shape[:2]):
                return
            stats[name][0] += h.norm(dim=-1)[am].sum().item()
            stats[name][1] += int(am.sum())

        return hook

    handles = [mod.register_forward_hook(make_hook(n)) for n, mod in sites.items()]
    try:
        for input_ids, attention_mask in batches:
            current["am"] = attention_mask.to(device).bool()
            model(input_ids.to(device), attention_mask=attention_mask.to(device))
    finally:
        for h in handles:
            h.remove()
    return {name: total / max(count, 1) for name, (total, count) in stats.items()}


def resolve_site_norms(
    sites: list[str],
    alpha: float,
    explicit_norm: float | None,
    npz_path: str | None = None,
    scales: dict[str, float] | None = None,
) -> dict[str, float]:
    """Per-site signature magnitude: an explicit constant when given, else
    alpha (default 8%) × the site's typical hidden L2 norm, taken from the
    stage-0 npz scale or from freshly measured `scales`."""
    if explicit_norm is not None:
        return {s: float(explicit_norm) for s in sites}
    npz = np.load(npz_path) if npz_path else None
    out = {}
    for s in sites:
        if npz is not None and f"{s}/scale" in npz.files:
            out[s] = alpha * float(npz[f"{s}/scale"])
        elif scales and s in scales:
            out[s] = alpha * scales[s]
        else:
            raise ValueError(
                f"no activation scale available for site {s!r}: pass "
                "--directions-npz (stage 0 output) or --signature-norm")
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
            h = _hidden(out)
            if (self._mask is None or self._vecs is None
                    or tuple(h.shape[:2]) != tuple(self._mask.shape)):
                return out
            add = self._vecs[site][:, None, :] * self._mask[:, :, None]
            return _replace_hidden(out, h + add.to(dtype=h.dtype, device=h.device))

        return hook

    def arm(self, mask: torch.Tensor, vecs: dict[str, torch.Tensor]) -> None:
        self._mask, self._vecs = mask, vecs

    def disarm(self) -> None:
        self._mask = self._vecs = None

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


@contextlib.contextmanager
def steering(module: torch.nn.Module, vec: torch.Tensor, positions: str = "all"):
    """Constant steering for the stage-0 free-direction sweep: add `vec` to
    the module's output hidden states at all positions (or BOS only)."""

    def hook(_module, _args, out):
        h = _hidden(out)
        if positions == "all":
            h = h + vec.to(h.dtype)
        else:
            h = h.clone()
            h[:, 0, :] += vec.to(h.dtype)
        return _replace_hidden(out, h)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
