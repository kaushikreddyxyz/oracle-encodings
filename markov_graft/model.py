"""Small pre-LN GPT with a per-token graft channel added to the input embedding.
Attention uses a learned per-head relative-offset bias (T5 style, clipped) so that
attending to a fixed lag is cheap; capacity is then spent on the table, not on positions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d, n_head, max_rel):
        super().__init__()
        self.n_head, self.max_rel = n_head, max_rel
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_rel + 1))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, D // self.n_head).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        pos = torch.arange(T, device=x.device)
        rel = (pos[:, None] - pos[None, :]).clamp(0, self.max_rel)
        bias = self.rel_bias[:, rel]                                  # [H, T, T]
        mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=(bias + mask).unsqueeze(0))
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, n_head, max_rel):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, n_head, max_rel)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GraftGPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, graft_values=0, graft_fixed=False, max_rel=32):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.graft_values = graft_values
        if graft_values:
            self.wg = nn.Embedding(graft_values, d)
            if graft_fixed:
                self.wg.weight.requires_grad_(False)
        self.blocks = nn.ModuleList(Block(d, n_head, max_rel) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x, g=None):
        h = self.wte(x)
        if g is not None:
            h = h + self.wg(g)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
