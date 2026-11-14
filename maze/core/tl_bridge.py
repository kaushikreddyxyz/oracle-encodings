"""Lossless bidirectional bridge between `maze.core.model.MazeGPT` (fast torch)
and TransformerLens / ZANJ (`ZanjHookedTransformer`, the interp + eval stack).

Training happens on the torch side; EVERY analysis path (`core/evals.py`,
`core/checkpoints.py`, `core/reliance.py`, the experiment scripts) keeps running
on TransformerLens. The two must therefore agree exactly, which is what
`core/tests/test_bridge.py` measures.

Weight layouts (the only interesting part):

    TL                                   torch
    embed.W_E        (d_vocab, d_model)  embed.weight            same
    pos_embed.W_pos  (n_ctx, d_model)    pos_embed.weight        same
    attn.W_Q/K/V     (n_head, d_model, d_head)
                                         attn.qkv.weight (3*n_head*d_head, d_model)
                                         rows [q | k | v], head-major:
                                         W_Q.permute(0, 2, 1).reshape(-1, d_model)
    attn.b_Q/K/V     (n_head, d_head)    attn.qkv.bias           flatten
    attn.W_O         (n_head, d_head, d_model)
                                         attn.out.weight (d_model, n_head*d_head)
                                         = W_O.permute(2, 0, 1).reshape(d_model, -1)
    mlp.W_in         (d_model, d_mlp)    mlp.fc.weight           transpose
    mlp.W_out        (d_mlp, d_model)    mlp.proj.weight         transpose
    unembed.W_U      (d_model, d_vocab)  unembed.weight          transpose
    ln{1,2,_final}.w/.b                  .weight/.bias           same

Both directions are permutes/reshapes/transposes only -- no arithmetic -- so
`tl_to_torch(torch_to_tl(x)) == x` bit-for-bit.

Gotcha (MIGRATION_SPEC): `maze_transformer.training.config` must be imported
before `ZANJ().read()`, otherwise the model class is not registered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from maze.core.model import MazeGPT, MazeGPTConfig
from maze.core.teacher import MazeTeacher

CKPT_FORMAT = "maze.core.model/v1"


# ---------------------------------------------------------------- state dicts


def torch_to_tl_state_dict(
    sd: dict[str, torch.Tensor], cfg: MazeGPTConfig
) -> dict[str, torch.Tensor]:
    """MazeGPT state dict -> HookedTransformer state dict (params only; the
    `attn.mask` / `attn.IGNORE` buffers are left to the TL model itself)."""
    H, Dh, D = cfg.n_heads, cfg.d_head, cfg.d_model
    HDh = H * Dh
    out: dict[str, torch.Tensor] = {
        "embed.W_E": sd["embed.weight"],
        "pos_embed.W_pos": sd["pos_embed.weight"],
        "ln_final.w": sd["ln_final.weight"],
        "ln_final.b": sd["ln_final.bias"],
        "unembed.W_U": sd["unembed.weight"].T.contiguous(),
        "unembed.b_U": sd["unembed.bias"],
    }
    for i in range(cfg.n_layers):
        p, q = f"blocks.{i}.", f"blocks.{i}."
        qkv_w = sd[p + "attn.qkv.weight"]  # (3*H*Dh, d_model)
        qkv_b = sd[p + "attn.qkv.bias"]  # (3*H*Dh,)
        for j, nm in enumerate("QKV"):
            w = qkv_w[j * HDh : (j + 1) * HDh]  # (H*Dh, d_model)
            out[q + f"attn.W_{nm}"] = (
                w.reshape(H, Dh, D).permute(0, 2, 1).contiguous()
            )
            out[q + f"attn.b_{nm}"] = qkv_b[j * HDh : (j + 1) * HDh].reshape(H, Dh)
        out[q + "attn.W_O"] = (
            sd[p + "attn.out.weight"].reshape(D, H, Dh).permute(1, 2, 0).contiguous()
        )
        out[q + "attn.b_O"] = sd[p + "attn.out.bias"]
        out[q + "ln1.w"] = sd[p + "ln1.weight"]
        out[q + "ln1.b"] = sd[p + "ln1.bias"]
        out[q + "ln2.w"] = sd[p + "ln2.weight"]
        out[q + "ln2.b"] = sd[p + "ln2.bias"]
        out[q + "mlp.W_in"] = sd[p + "mlp.fc.weight"].T.contiguous()
        out[q + "mlp.b_in"] = sd[p + "mlp.fc.bias"]
        out[q + "mlp.W_out"] = sd[p + "mlp.proj.weight"].T.contiguous()
        out[q + "mlp.b_out"] = sd[p + "mlp.proj.bias"]
    return out


def tl_to_torch_state_dict(
    sd: dict[str, torch.Tensor], cfg: MazeGPTConfig
) -> dict[str, torch.Tensor]:
    """HookedTransformer state dict -> MazeGPT state dict (inverse of above)."""
    H, Dh, D = cfg.n_heads, cfg.d_head, cfg.d_model
    out: dict[str, torch.Tensor] = {
        "embed.weight": sd["embed.W_E"],
        "pos_embed.weight": sd["pos_embed.W_pos"],
        "ln_final.weight": sd["ln_final.w"],
        "ln_final.bias": sd["ln_final.b"],
        "unembed.weight": sd["unembed.W_U"].T.contiguous(),
        "unembed.bias": sd["unembed.b_U"],
    }
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        out[p + "attn.qkv.weight"] = torch.cat(
            [
                sd[p + f"attn.W_{nm}"].permute(0, 2, 1).reshape(H * Dh, D)
                for nm in "QKV"
            ],
            dim=0,
        ).contiguous()
        out[p + "attn.qkv.bias"] = torch.cat(
            [sd[p + f"attn.b_{nm}"].reshape(H * Dh) for nm in "QKV"], dim=0
        ).contiguous()
        out[p + "attn.out.weight"] = (
            sd[p + "attn.W_O"].permute(2, 0, 1).reshape(D, H * Dh).contiguous()
        )
        out[p + "attn.out.bias"] = sd[p + "attn.b_O"]
        out[p + "ln1.weight"] = sd[p + "ln1.w"]
        out[p + "ln1.bias"] = sd[p + "ln1.b"]
        out[p + "ln2.weight"] = sd[p + "ln2.w"]
        out[p + "ln2.bias"] = sd[p + "ln2.b"]
        out[p + "mlp.fc.weight"] = sd[p + "mlp.W_in"].T.contiguous()
        out[p + "mlp.fc.bias"] = sd[p + "mlp.b_in"]
        out[p + "mlp.proj.weight"] = sd[p + "mlp.W_out"].T.contiguous()
        out[p + "mlp.proj.bias"] = sd[p + "mlp.b_out"]
    return out


# ------------------------------------------------------------- config holders


def _import_registers_zanj():
    """Import for side effects: registers ZanjHookedTransformer with ZANJ."""
    import maze_transformer.training.config  # noqa: F401


def load_config_holder(source):
    """ConfigHolder from a ConfigHolder / dict / path to a config_holder.json."""
    from maze_transformer.training.config import ConfigHolder

    if isinstance(source, ConfigHolder):
        return source
    if isinstance(source, dict):
        return ConfigHolder.load(source)
    with open(source) as f:
        return ConfigHolder.load(json.load(f))


def _find_config_holder(ckpt_path: Path):
    """`config_holder.json` next to the checkpoint, or one directory up
    (checkpoints live in `<run>/checkpoints/`)."""
    for d in (ckpt_path.parent, ckpt_path.parent.parent):
        p = d / "config_holder.json"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no config_holder.json next to or above {ckpt_path}; pass config_holder="
    )


def _make_tl_model(config_holder):
    """A fresh ZanjHookedTransformer WITHOUT perturbing the global RNG stream
    (TL's init_weights draws from it; train.py builds these mid-run)."""
    state = torch.get_rng_state()
    try:
        return load_config_holder(config_holder).create_model_zanj()
    finally:
        torch.set_rng_state(state)


# ------------------------------------------------------------------ torch I/O


def save_torch_ckpt(path, model: MazeGPT, extra: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    torch.save(
        {
            "format": CKPT_FORMAT,
            "model_cfg": model.cfg.to_dict(),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            **(extra or {}),
        },
        path,
    )
    return path


def load_torch_ckpt(path, device="cpu") -> tuple[MazeGPT, dict[str, Any]]:
    d = torch.load(Path(path), map_location="cpu", weights_only=False)
    assert d.get("format") == CKPT_FORMAT, f"not a {CKPT_FORMAT} checkpoint: {path}"
    cfg = MazeGPTConfig.from_dict(d["model_cfg"])
    model = MazeGPT(cfg)
    model.load_state_dict(d["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in d.items() if k != "state_dict"}
    return model, meta


# ------------------------------------------------------------------- loaders


def load_as_tl(path, config_holder=None, device="cpu"):
    """torch checkpoint -> ZanjHookedTransformer, with NO weight processing.

    Deliberately does not go through `ZanjHookedTransformer._load_state_dict_wrapper`
    (fold_ln / center_writing_weights / center_unembed all stay off) so the TL
    model is numerically the same network as the torch one, and every hook name
    the analysis code uses means what it used to mean.

    NOTE, and this predates the migration: `ZANJ().read()` DOES process weights
    -- `_load_state_dict_wrapper` passes `center_writing_weights=True` and
    `center_unembed=True` unless you ask for `recover_exact`. So a model loaded
    from `.zanj` differs from the trained weights by a per-position constant in
    the logits (measured: max 0.28 raw, 2e-6 after removing the per-position
    mean; log-softmax and argmax agree to 2e-6 / exactly). Every metric in
    `core/evals.py` goes through `log_softmax` or `argmax`, so the two are
    interchangeable there -- but do not compare RAW logits across the two
    loaders and expect them to match.
    """
    _import_registers_zanj()
    path = Path(path)
    if config_holder is None:
        config_holder = _find_config_holder(path)
    d = torch.load(path, map_location="cpu", weights_only=False)
    assert d.get("format") == CKPT_FORMAT, f"not a {CKPT_FORMAT} checkpoint: {path}"
    cfg = MazeGPTConfig.from_dict(d["model_cfg"])
    return torch_to_tl(d["state_dict"], cfg, config_holder, device=device)


def torch_to_tl(model_or_sd, cfg=None, config_holder=None, device="cpu"):
    """MazeGPT (or its state dict) -> ZanjHookedTransformer, no processing."""
    _import_registers_zanj()
    if isinstance(model_or_sd, MazeGPT):
        cfg = model_or_sd.cfg if cfg is None else cfg
        sd = model_or_sd.state_dict()
    else:
        assert cfg is not None, "pass cfg when converting a bare state dict"
        sd = model_or_sd
    tl = _make_tl_model(config_holder)
    assert MazeGPTConfig.from_tl_cfg(tl.cfg) == cfg, (
        "config_holder architecture does not match the torch model:\n"
        f"  tl:    {MazeGPTConfig.from_tl_cfg(tl.cfg)}\n  torch: {cfg}"
    )
    full = tl.state_dict()
    full.update(torch_to_tl_state_dict(sd, cfg))
    nn.Module.load_state_dict(tl, full, strict=True)
    tl.to(device)
    tl.eval()
    return tl


def load_zanj_as_torch(path, device="cpu", return_config: bool = False):
    """Existing ZANJ checkpoint -> MazeGPT (weights exactly as stored)."""
    _import_registers_zanj()
    from zanj import ZANJ

    tl = ZANJ().read(Path(path))
    model = tl_to_torch(tl, device=device)
    if return_config:
        return model, tl.zanj_model_config
    return model


def tl_to_torch(tl, device="cpu") -> MazeGPT:
    cfg = MazeGPTConfig.from_tl_cfg(tl.cfg)
    model = MazeGPT(cfg)
    model.load_state_dict(tl_to_torch_state_dict(tl.state_dict(), cfg))
    model.to(device).eval()
    return model


# ------------------------------------------------------------- ZANJ export


class ZanjExporter:
    """Reusable `MazeGPT -> model.*.zanj` writer.

    One TL shell is built and reused so a training loop that checkpoints 30
    times does not pay for (or perturb the RNG with) 30 model constructions.
    """

    def __init__(self, config_holder, tl=None):
        self.config_holder = load_config_holder(config_holder)
        # a caller that already built a TL model (train.py, for init) can hand
        # it over instead of paying for a second construction
        self._tl = tl

    @property
    def tl(self):
        if self._tl is None:
            _import_registers_zanj()
            self._tl = _make_tl_model(self.config_holder)
        return self._tl

    def to_tl(self, model: MazeGPT):
        full = self.tl.state_dict()
        full.update(torch_to_tl_state_dict(model.state_dict(), model.cfg))
        nn.Module.load_state_dict(self.tl, full, strict=True)
        return self.tl

    def save(self, model: MazeGPT, path) -> Path:
        from zanj import ZANJ

        path = Path(path)
        ZANJ().save(self.to_tl(model), path)
        return path


def save_as_zanj(model: MazeGPT, config_holder, path) -> Path:
    """One-shot export so any torch model can be handed to the TL eval stack."""
    return ZanjExporter(config_holder).save(model, path)


# ------------------------------------------------------------------- teacher


class TorchTeacher(MazeTeacher):
    """`maze.core.teacher.MazeTeacher` reading from a converted torch model.

    Inherits the belief maths and the prefix cache (`beliefs()` is untouched);
    only the activation read changes -- `MazeGPT(..., stop_at_layer=layer+1)` is
    exactly TL's `blocks.{layer}.hook_resid_post`.
    """

    def __init__(self, model: MazeGPT, W, b, layer: int = 6, padding_mode="exact"):
        super().__init__(model, W, b, layer=layer)
        self.padding_mode = padding_mode

    @torch.no_grad()
    def _probe_prefix(self, tokens, attn) -> torch.Tensor:
        if self._cache is not None:
            key, val = self._cache
            if key.shape == tokens.shape and torch.equal(key, tokens):
                return val
        acts = self.model(
            tokens,
            attention_mask=attn,
            return_type=None,
            stop_at_layer=self.layer + 1,
            padding_mode=self.padding_mode,
        )
        s = 2.0 * torch.sigmoid(acts @ self.W.T + self.b) - 1.0
        self._cache = (tokens.clone(), s)
        return s


_TORCH_TEACHERS: dict[tuple, TorchTeacher] = {}


def load_torch_teacher(model_path, probe_path, device, layer: int | None = None):
    """Frozen teacher for `--inject-mode teacher`, run as a converted torch
    model (the TL version is ~3x slower and is pure overhead in training)."""
    key = (str(model_path), str(probe_path), str(device), layer)
    if key in _TORCH_TEACHERS:
        return _TORCH_TEACHERS[key]
    model = load_zanj_as_torch(model_path, device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    d = torch.load(probe_path, weights_only=True, map_location="cpu")
    t = TorchTeacher(
        model,
        d["W"],
        d["b"],
        layer=int(d.get("layer", 6)) if layer is None else int(layer),
    )
    _TORCH_TEACHERS[key] = t
    return t
