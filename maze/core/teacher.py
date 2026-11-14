"""Teacher-belief channel: inject what a trained model *believes*, not truth.

The teacher is the frozen baseline final checkpoint. For a batch of tokens it
is run in eval mode under no_grad, its `blocks.{layer}.hook_resid_post`
activations are read out by a single linear probe (train_teacher_probe.py), and
the logits become a soft belief per connection:

    s_p = 2 * sigmoid(z_p) - 1  in (-1, 1)

Only positions 1 <= p < end_pos are used (mid-adjacency, where the teacher's
knowledge is genuinely partial); position 0 (BOS) and everything from
<ADJLIST_END> on stay zero here — InjectionRuntime fills those with the true
+/-1 features. Beliefs depend only on the adjacency prefix, so the teacher runs
on tokens[:, :max(end_pos)] and the result is cached across greedy-rollout
steps (the prompt prefix never changes during generation).
"""

from pathlib import Path

import torch

_TEACHERS: dict[tuple, "MazeTeacher"] = {}


class MazeTeacher:
    """Frozen model + linear connectivity probe -> per-position beliefs."""

    def __init__(self, model, W: torch.Tensor, b: torch.Tensor, layer: int = 6):
        self.model = model
        self.model.eval()
        self.layer = layer
        self.hook_name = f"blocks.{layer}.hook_resid_post"
        device = next(model.parameters()).device
        self.W = W.to(device)  # (72, d_model)
        self.b = b.to(device)  # (72,)
        self.n_features = W.shape[0]
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    @torch.no_grad()
    def _probe_prefix(self, tokens, attn) -> torch.Tensor:
        """(batch, E, 72) beliefs on the adjacency prefix, memoized."""
        if self._cache is not None:
            key, val = self._cache
            if key.shape == tokens.shape and torch.equal(key, tokens):
                return val
        _, cache = self.model.run_with_cache(
            tokens, attention_mask=attn, return_type=None,
            names_filter=lambda nm: nm == self.hook_name,
        )
        acts = cache[self.hook_name]
        del cache
        s = 2.0 * torch.sigmoid(acts @ self.W.T + self.b) - 1.0
        self._cache = (tokens.clone(), s)
        return s

    @torch.no_grad()
    def beliefs(self, tokens, attn, end_pos) -> torch.Tensor:
        """(batch, pos, 72), nonzero only at 1 <= p < end_pos."""
        n_pos = tokens.shape[1]
        E = min(n_pos, int(end_pos.max()))
        s = self._probe_prefix(tokens[:, :E], attn[:, :E])
        pos = torch.arange(E, device=tokens.device)[None, :]
        mask = (pos >= 1) & (pos < end_pos[:, None]) & attn[:, :E].bool()
        out = torch.zeros(
            tokens.shape[0], n_pos, self.n_features,
            device=s.device, dtype=s.dtype,
        )
        out[:, :E] = s * mask[:, :, None].to(s.dtype)
        return out


def load_teacher(model_path, probe_path, device, layer: int | None = None):
    """Load (and memoize) the frozen teacher model + its probe."""
    key = (str(model_path), str(probe_path), str(device), layer)
    if key in _TEACHERS:
        return _TEACHERS[key]
    import maze_transformer.training.config  # noqa: F401  (registers ZANJ class)
    from zanj import ZANJ

    model = ZANJ().read(Path(model_path))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = torch.load(probe_path, weights_only=True, map_location="cpu")
    t = MazeTeacher(
        model, d["W"], d["b"],
        layer=int(d.get("layer", 6)) if layer is None else int(layer),
    )
    _TEACHERS[key] = t
    return t
