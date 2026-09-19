"""Plain-torch decoder-only GPT that reproduces the jirpy TransformerLens
architecture EXACTLY (see `maze/core/tl_bridge.py` for weight conversion).

Why: TransformerLens' forward pass is written for introspection, not speed --
per-head einops rearranges, an explicit (batch, head, pos, pos) attention-score
tensor, a hook point on every intermediate. This module is the same maths with
`F.scaled_dot_product_attention` and fused `nn.Linear`s so `torch.compile` and
the flash/mem-efficient kernels can be used for training. All analysis still
happens on the TL side: convert with `tl_bridge.load_as_tl`.

Architecture (read off a real ZanjHookedTransformer, not assumed -- see
`MazeGPTConfig.from_tl_cfg`, which refuses configs whose features are not
implemented here):

  embed        W_E[tokens]                       (no post-embedding LN)
  pos_embed    learned absolute, W_pos           (masked, see below)
  block        pre-LN:  x += attn(ln1(x)); x += mlp(ln2(x))
  ln1/ln2      LayerNorm (biased variance, eps inside the sqrt), eps 1e-5
  attn         fused QKV + out proj, both WITH bias; scores / cfg.attn_scale
               (= sqrt(d_head)); causal
  mlp          d_mlp = 4 * d_model, biases on both layers,
               act_fn "gelu" == exact erf `F.gelu` (TL maps "gelu" -> F.gelu;
               "gelu_new" is the tanh approximation and is a *different* key)
  ln_final     LayerNorm
  unembed      untied W_U + b_U

Right-padding semantics
-----------------------
TL builds an additive mask `causal & attention_mask[key]` and sets everything
else to -inf, i.e. padded KEY positions are masked out for every query. Padded
QUERY rows are *not* fully masked (they still see the real keys at j <= i), so
they produce finite garbage that the loss mask throws away. TL also zeroes the
positional embedding at padded positions and uses `cumsum(mask) - 1` position
ids (identical to `arange` for right padding).

`padding_mode="exact"` (default) reproduces all of that, including the padded
rows. `padding_mode="right"` skips the key mask and uses `is_causal=True`
instead: for RIGHT-padded batches every real query position i < len only
attends to keys j <= i < len, which are all real, so the logits at every real
position are bit-for-bit the same -- only the (ignored) padded rows differ.
That is the fast path for training, where the loss mask
`attn[:, :-1] & attn[:, 1:]` only ever touches real query rows.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

PADDING_MODES = ("exact", "right")


def _gelu_new(x: torch.Tensor) -> torch.Tensor:
    """TL's `gelu_new` (the tanh approximation), kept for completeness."""
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            )
        )
    )


# TL's ActivationFunctionFactory maps these names to exactly these callables.
ACT_FNS = {
    "gelu": F.gelu,  # exact erf gelu -- NOT gelu_new
    "gelu_new": _gelu_new,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "silu": F.silu,
}


@dataclass
class MazeGPTConfig:
    """The subset of HookedTransformerConfig this model implements."""

    d_model: int = 256
    d_head: int = 16
    n_heads: int = 16
    n_layers: int = 12
    d_mlp: int = 1024
    d_vocab: int = 47
    d_vocab_out: int = 47
    n_ctx: int = 512
    act_fn: str = "gelu"
    eps: float = 1e-5
    attn_scale: float = 4.0
    use_attn_scale: bool = True
    normalization_type: str = "LN"
    positional_embedding_type: str = "standard"

    def __post_init__(self):
        assert self.act_fn in ACT_FNS, f"unsupported act_fn: {self.act_fn}"
        assert self.normalization_type == "LN"
        assert self.positional_embedding_type == "standard"
        assert self.d_model == self.n_heads * self.d_head, (
            "this model assumes d_model == n_heads * d_head (fused qkv/out proj "
            f"still works otherwise, but TL's W_O layout does not): "
            f"{self.d_model} != {self.n_heads} * {self.d_head}"
        )

    @classmethod
    def from_tl_cfg(cls, cfg) -> "MazeGPTConfig":
        """Build from a HookedTransformerConfig, refusing anything unsupported.

        Every check below corresponds to a branch in TL's forward pass that this
        module does not implement; failing loudly is the point.
        """
        unsupported = {
            "attn_only": False,
            "parallel_attn_mlp": False,
            "gated_mlp": False,
            "num_experts": None,
            "final_rms": False,
            "use_normalization_before_and_after": False,
            "post_embedding_ln": False,
            "attention_dir": "causal",
            "attn_types": None,
            "use_local_attn": False,
            "window_size": None,
            "scale_attn_by_inverse_layer_idx": False,
            "use_qk_norm": False,
            "n_key_value_heads": None,
            "tie_word_embeddings": False,
            "use_attn_result": False,
            "use_split_qkv_input": False,
            "use_attn_in": False,
            "load_in_4bit": False,
            "n_devices": 1,
            "normalization_type": "LN",
            "positional_embedding_type": "standard",
        }
        for k, want in unsupported.items():
            got = getattr(cfg, k, want)
            assert got == want, f"maze.core.model does not implement cfg.{k}={got!r}"
        assert getattr(cfg, "attn_scores_soft_cap", -1.0) <= 0.0
        assert getattr(cfg, "output_logits_soft_cap", -1.0) <= 0.0
        assert cfg.d_mlp is not None
        return cls(
            d_model=cfg.d_model,
            d_head=cfg.d_head,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_mlp=cfg.d_mlp,
            d_vocab=cfg.d_vocab,
            d_vocab_out=cfg.d_vocab_out,
            n_ctx=cfg.n_ctx,
            act_fn=cfg.act_fn,
            eps=cfg.eps,
            attn_scale=float(cfg.attn_scale),
            use_attn_scale=bool(cfg.use_attn_scale),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MazeGPTConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class MazeAttention(nn.Module):
    """Fused-QKV causal attention. `qkv.weight` rows are ordered
    (Q heads, K heads, V heads), head-major within each block, so the split into
    TL's (n_heads, d_model, d_head) layout is a pure reshape (see tl_bridge)."""

    def __init__(self, cfg: MazeGPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_inner = cfg.n_heads * cfg.d_head
        # TL divides scores by attn_scale; SDPA multiplies by `scale`.
        self.scale = (1.0 / cfg.attn_scale) if cfg.use_attn_scale else 1.0
        self.qkv = nn.Linear(cfg.d_model, 3 * self.d_inner, bias=True)
        self.out = nn.Linear(self.d_inner, cfg.d_model, bias=True)

    def forward(self, x, attn_mask, empty_rows):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.d_inner, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=attn_mask is None,
            scale=self.scale,
        )
        if empty_rows is not None:
            # TL: softmax over an all -inf row is NaN, which it replaces with 0
            # (=> z = 0). SDPA returns NaN. Only reachable for a zero-length
            # sequence; matched here so the two agree on every input.
            y = y.masked_fill(empty_rows, 0.0)
        y = y.transpose(1, 2).reshape(B, T, self.d_inner)
        return self.out(y)


class MazeMLP(nn.Module):
    def __init__(self, cfg: MazeGPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_mlp, bias=True)
        self.proj = nn.Linear(cfg.d_mlp, cfg.d_model, bias=True)
        self.act = ACT_FNS[cfg.act_fn]

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))


class MazeBlock(nn.Module):
    def __init__(self, cfg: MazeGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.eps)
        self.attn = MazeAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.eps)
        self.mlp = MazeMLP(cfg)

    def forward(self, resid, attn_mask, empty_rows):
        resid = resid + self.attn(self.ln1(resid), attn_mask, empty_rows)
        resid = resid + self.mlp(self.ln2(resid))
        return resid


class MazeGPT(nn.Module):
    def __init__(self, cfg: MazeGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList([MazeBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model, eps=cfg.eps)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab_out, bias=True)

    # -- weights ----------------------------------------------------------
    def init_weights_(self, seed: int | None = None) -> "MazeGPT":
        """TL's `init_mode="gpt2"`: every `W_*` ~ N(0, initializer_range),
        LayerNorm weights 1, all biases 0. Kept for standalone use; `train.py`
        instead converts a freshly built TL model so the RNG stream is
        bit-identical to the legacy trainer."""
        if seed is not None:
            torch.manual_seed(seed)
        std = 0.05
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                nn.init.normal_(mod.weight, std=std)
                if isinstance(mod, nn.Linear) and mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        return self

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self) -> torch.device:
        return self.embed.weight.device

    # -- masks ------------------------------------------------------------
    def _positions(self, tokens, attention_mask):
        B, T = tokens.shape
        if attention_mask is None:
            pos = torch.arange(T, device=tokens.device)
            return self.pos_embed(pos)[None].expand(B, T, -1)
        # TL: get_offset_position_ids -> cumsum(mask) - 1, negatives clamped to 0
        ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
        pe = self.pos_embed(ids)
        return torch.where(attention_mask.bool()[:, :, None], pe, torch.zeros_like(pe))

    def _attn_masks(self, tokens, attention_mask, padding_mode):
        """-> (attn_mask (B,1,T,T) bool | None, empty_rows (B,1,T,1) bool | None)"""
        if attention_mask is None or padding_mode == "right":
            return None, None
        assert padding_mode == "exact", f"unknown padding_mode: {padding_mode}"
        T = tokens.shape[1]
        causal = torch.ones(T, T, dtype=torch.bool, device=tokens.device).tril()
        keep = causal[None, None] & attention_mask.bool()[:, None, None, :]
        # a query row is fully masked iff no real token at or before it
        empty = (attention_mask.cumsum(dim=1) == 0)[:, None, :, None]
        return keep, empty

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        inject_delta: torch.Tensor | None = None,
        inject_layer: int = 0,
        return_type: str | None = "logits",
        loss_per_token: bool = False,
        padding_mode: str = "exact",
        stop_at_layer: int | None = None,
    ):
        """`inject_delta` is added to `blocks.{inject_layer}`'s residual input --
        the plain-torch equivalent of TL's `blocks.{L}.hook_resid_pre` hook.

        `stop_at_layer=k` returns the residual after block k-1 (TL semantics:
        `blocks.{k-1}.hook_resid_post`) instead of logits.
        """
        attn_mask, empty_rows = self._attn_masks(
            tokens, attention_mask, padding_mode
        )
        resid = self.embed(tokens) + self._positions(tokens, attention_mask)

        n_blocks = len(self.blocks) if stop_at_layer is None else stop_at_layer
        for i, block in enumerate(self.blocks[:n_blocks]):
            if inject_delta is not None and i == inject_layer:
                resid = resid + inject_delta.to(resid.dtype)
            resid = block(resid, attn_mask, empty_rows)

        if stop_at_layer is not None:
            return resid

        resid = self.ln_final(resid)
        if return_type is None:
            return None
        logits = self.unembed(resid)
        if return_type == "logits":
            return logits
        loss = lm_cross_entropy_loss(
            logits, tokens, attention_mask, per_token=loss_per_token
        )
        if return_type == "loss":
            return loss
        if return_type == "both":
            return logits, loss
        raise ValueError(f"unknown return_type: {return_type}")


def lm_cross_entropy_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    per_token: bool = False,
):
    """Byte-for-byte port of `transformer_lens.utils.lm_cross_entropy_loss`."""
    log_probs = F.log_softmax(logits, dim=-1)
    predicted_log_probs = log_probs[..., :-1, :].gather(
        dim=-1, index=tokens[..., 1:, None]
    )[..., 0]
    if attention_mask is not None:
        next_token_mask = torch.logical_and(
            attention_mask[:, :-1], attention_mask[:, 1:]
        )
        predicted_log_probs = predicted_log_probs * next_token_mask
        n_tokens = next_token_mask.sum()
    else:
        n_tokens = predicted_log_probs.numel()
    if per_token:
        return -predicted_log_probs
    return -predicted_log_probs.sum() / n_tokens


def injection_delta(injector, features: torch.Tensor, dtype=None) -> torch.Tensor:
    """The tensor TL's `MazeInjector.hook_fn` adds to the residual stream.

    Kept a one-liner mirror of `hook_fn` (`resid + scale * (features @ W_inj)`)
    so the torch path and the hook path cannot drift apart.
    """
    if dtype is None:
        dtype = injector.W_inj.dtype
    return injector.scale * (features.to(dtype) @ injector.W_inj)
