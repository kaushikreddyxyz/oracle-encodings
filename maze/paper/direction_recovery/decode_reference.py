import torch
@torch.no_grad()
def greedy_kv(model, injector, layer, prompt, conn, end_pos, max_new, path_end_id, pad_id):
    """Greedy decoding with a KV cache. `prompt` is [B, P] with NO padding (every
    row the same length, true for spanning-tree mazes of one grid size). Mirrors
    MazeGPT.forward(padding_mode="right") block by block. Returns [B, max_new]
    generated ids (pad after <PATH_END>)."""
    import torch.nn.functional as F
    dev = prompt.device
    B, P = prompt.shape
    H, dh = model.cfg.n_heads, model.cfg.d_head
    delta_row = None
    if injector is not None:
        signed = conn.float() * 2.0 - 1.0
        delta_row = injector.scale * (signed.to(injector.W_inj.dtype) @ injector.W_inj)  # [B, d]
    pos = torch.arange(P + max_new, device=dev)
    resid = model.embed(prompt) + model.pos_embed(pos[:P])[None]
    if delta_row is not None:
        active = (pos[None, :P] >= end_pos[:, None]).to(resid.dtype)[:, :, None]
        delta_full = active * delta_row[:, None, :]
    caches = []
    for i, blk in enumerate(model.blocks):
        if delta_row is not None and i == layer:
            resid = resid + delta_full
        x = blk.ln1(resid)
        q, k, v = blk.attn.qkv(x).split(blk.attn.d_inner, dim=-1)
        q = q.view(B, P, H, dh).transpose(1, 2)
        k = k.view(B, P, H, dh).transpose(1, 2)
        v = v.view(B, P, H, dh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=blk.attn.scale)
        resid = resid + blk.attn.out(y.transpose(1, 2).reshape(B, P, -1))
        resid = resid + blk.mlp(blk.ln2(resid))
        K = torch.empty(B, H, P + max_new, dh, device=dev, dtype=k.dtype); K[:, :, :P] = k
        V = torch.empty(B, H, P + max_new, dh, device=dev, dtype=v.dtype); V[:, :, :P] = v
        caches.append((K, V))
    cur = model.unembed(model.ln_final(resid[:, -1])).argmax(-1)
    out = torch.full((B, max_new), pad_id, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for step in range(max_new):
        out[:, step] = torch.where(done, torch.full_like(cur, pad_id), cur)
        done = done | (cur == path_end_id)
        if bool(done.all()) or step == max_new - 1:
            break
        t = P + step
        resid = model.embed(cur) + model.pos_embed(pos[t])[None]
        for i, blk in enumerate(model.blocks):
            if delta_row is not None and i == layer:
                resid = resid + delta_row          # t >= P > end_pos: injection always active
            x = blk.ln1(resid)
            q, k, v = blk.attn.qkv(x).split(blk.attn.d_inner, dim=-1)
            K, V = caches[i]
            K[:, :, t] = k.view(B, H, dh)
            V[:, :, t] = v.view(B, H, dh)
            y = F.scaled_dot_product_attention(q.view(B, H, 1, dh), K[:, :, : t + 1], V[:, :, : t + 1],
                                               scale=blk.attn.scale)
            resid = resid + blk.attn.out(y.reshape(B, -1))
            resid = resid + blk.mlp(blk.ln2(resid))
        cur = model.unembed(model.ln_final(resid)).argmax(-1)
    return out

