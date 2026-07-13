"""Stage 6.1 intervention hooks: steer / ablate along probe directions in
gemma-2's residual stream. Implements the DESIGN.md "interventions.py API".

Where the hook sits
- layer l in 0..25: forward hook on ``model.model.layers[l]`` editing that
  block's OUTPUT hidden state — exactly where probe l reads
  (hidden_states[l+1]). transformers 5.x decoder layers return a plain
  tensor; older versions returned a tuple with the hidden state at index 0 —
  both are handled.
- layer -1: forward hook on ``model.model.embed_tokens``. In transformers 5.x
  the sqrt(d) embed scale is applied INSIDE Gemma2TextScaledWordEmbedding, so
  its output IS hidden_states[0] (the stream entering block 0). Probe layers
  are block outputs, so -1 has no probe/natstats; it exists only for
  everywhere-ablation, with caller-supplied (mu, sigma) for key -1.

Formulas (all math in fp32, cast back to the module dtype; w is re-unit-
normalized on ingest; z = (h - mu)/sigma):
- steer (std-arm):    h' = h + alpha * (sigma ⊙ w)
  => the standardized score w.z moves by exactly +alpha.
- ablate, space='std': z' = z - (w.z - t) * w, i.e.
  h' = h - (w.z - t) * (sigma ⊙ w)          => post-hook w.z' = t exactly.
- ablate, space='grad': u = unit(w ⊘ sigma), g = ||w ⊘ sigma||,
  h' = h - ((w.z - t) / g) * u              => post-hook w.z' = t exactly.
  DEVIATION note: DESIGN's grad-arm said "project out unit-normed w⊘sigma in
  raw space" without fixing the offset; we remove along u exactly enough to
  land the STANDARDIZED score on t (the natural mean), the direct grad-space
  analog of the std-arm's project-to-natural-mean (never to zero). With t=0
  and mu-centered h it reduces to plain projection.

Positions: boolean mask [B, T]; masked-out positions are returned via
torch.where against the ORIGINAL tensor, hence bit-identical.

transformers 5.x hidden_states caveats (wave-2 scripts MUST know):
1. output_hidden_states is collected by HF's own lazily-installed forward
   hooks. Our hooks register with prepend=True so they run BEFORE the capture
   hook regardless of installation order — output.hidden_states therefore
   reflects the edits.
2. hidden_states[-1] (index 26 on gemma-2-2b) is tied to last_hidden_state,
   i.e. the POST-final-RMSNorm stream (same as transformers 4.x, and what
   Stage-5 layer-25 probes were trained on). An intervention at layer 25
   edits the raw block-25 output (downstream logits see it), but the layer-25
   probe meter reads through the final norm, so "steer by alpha moves the
   score by exactly alpha" holds for layers -1..24 only, NOT for 25.

Hooks compose across layers (one registered hook per module; multiple
Interventions on the same layer apply sequentially in list order), restore
cleanly on __exit__ (handles removed; baseline forward bit-identical), and
work with grad enabled and disabled (no in-place ops on the hooked tensor).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class Intervention:
    layer: int                  # block index 0..25 (-1 = embedding stream)
    vec_std: np.ndarray         # unit-norm standardized-space direction [d]
    mode: str                   # 'steer' | 'ablate'
    alpha: float = 0.0          # steer only, probe-score units
    t: float = 0.0              # ablate target (natural-mean raw proj w.z)
    positions: Optional[torch.Tensor] = None   # [B,T] bool; None = all
    space: str = 'std'          # 'std' | 'grad' (ablate only)


class Hooks:
    """Context manager registering the intervention forward hooks.

    Hooks(model, interventions, mu_sigma) — ``mu_sigma`` maps layer ->
    (mu, sigma) (np arrays or tensors, [d]); every intervened layer must have
    an entry (layer -1 needs caller-supplied embedding-stream stats).
    ``model`` may be the CausalLM wrapper (uses model.model) or the bare
    Gemma2Model. 'All layers' = one Intervention per layer.
    """

    def __init__(self, model, interventions: list[Intervention],
                 mu_sigma: dict[int, tuple]):
        self.base = model.model if hasattr(model, "model") else model
        self.by_layer: dict[int, list[Intervention]] = {}
        for iv in interventions:
            if iv.mode not in ("steer", "ablate"):
                raise ValueError(f"unknown mode {iv.mode!r}")
            if iv.space not in ("std", "grad"):
                raise ValueError(f"unknown space {iv.space!r}")
            if iv.layer not in mu_sigma:
                raise KeyError(f"mu_sigma missing entry for layer {iv.layer}")
            self.by_layer.setdefault(iv.layer, []).append(iv)
        # fp32 constants per layer; moved to the hooked tensor's device lazily
        self._mu_sd = {
            l: (self._f32(mu), self._f32(sd))
            for l, (mu, sd) in mu_sigma.items() if l in self.by_layer}
        self._w = {}
        for l, ivs in self.by_layer.items():
            for k, iv in enumerate(ivs):
                w = self._f32(iv.vec_std)
                self._w[(l, k)] = w / torch.linalg.norm(w)   # defensive re-unit
        self._handles: list = []

    @staticmethod
    def _f32(x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x,
                               dtype=torch.float32)

    # ------------------------------------------------------------- the math
    def _apply(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        device, dtype = h.device, h.dtype
        if self._mu_sd[layer][0].device != device:   # cache on first use
            self._mu_sd[layer] = (self._mu_sd[layer][0].to(device),
                                  self._mu_sd[layer][1].to(device))
        mu, sd = self._mu_sd[layer]
        x = h.to(torch.float32)
        for k, iv in enumerate(self.by_layer[layer]):
            if self._w[(layer, k)].device != device:
                self._w[(layer, k)] = self._w[(layer, k)].to(device)
            w = self._w[(layer, k)]
            if iv.mode == "steer":
                y = x + iv.alpha * (sd * w)
            elif iv.space == "std":
                s = ((x - mu) / sd) @ w                       # [B,T] = w.z
                y = x - (s - iv.t).unsqueeze(-1) * (sd * w)
            else:                                             # ablate, grad
                g = w / sd
                gn = torch.linalg.norm(g)
                u = g / gn
                s = ((x - mu) / sd) @ w
                y = x - ((s - iv.t) / gn).unsqueeze(-1) * u
            if iv.positions is not None:
                m = iv.positions.to(device=device, dtype=torch.bool)
                if m.shape != x.shape[:2]:
                    raise ValueError(
                        f"positions {tuple(m.shape)} != hidden {tuple(x.shape[:2])}")
                y = torch.where(m.unsqueeze(-1), y, x)
            x = y
        out = x.to(dtype)
        # bit-identity guarantee: positions touched by NO intervention keep
        # the original tensor's values exactly (fp-roundtrip-proof)
        touched = None
        for iv in self.by_layer[layer]:
            m = (iv.positions.to(device=device, dtype=torch.bool)
                 if iv.positions is not None
                 else torch.ones(h.shape[:2], dtype=torch.bool, device=device))
            touched = m if touched is None else (touched | m)
        return torch.where(touched.unsqueeze(-1), out, h)

    # ------------------------------------------------------------ hook glue
    def _make_hook(self, layer: int):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            new = self._apply(layer, hs)
            if isinstance(output, tuple):
                return (new,) + tuple(output[1:])
            return new
        return hook

    def __enter__(self):
        for l in sorted(self.by_layer):
            module = self.base.embed_tokens if l == -1 else self.base.layers[l]
            # prepend=True: run BEFORE HF's output_hidden_states capture hooks
            self._handles.append(
                module.register_forward_hook(self._make_hook(l), prepend=True))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False
