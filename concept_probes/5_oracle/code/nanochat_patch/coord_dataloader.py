"""Ride-along coord dataloader: nanochat's BOS-aligned best-fit packing, with a
parallel (B, T, r) coord tensor carried in lockstep with the tokens.

Mirrors ``nanochat.dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit``
1:1 for the token path (so token order / crop / DDP sharding are byte-identical
to the baseline given the same shard set + seed policy), and places each doc's
precomputed coord rows wherever that doc's tokens go -- same best-fit pick, same
crop. Coords for the BOS token (and any doc missing from the precompute) are
zero, so the injection is a no-op there.

Yields (inputs, targets, coords, state_dict):
  inputs/targets : (B, T) long   -- identical to the stock loader
  coords         : (B, T, r) float32 on `device` -- standardized coords + noise
"""
import numpy as np
import torch

from nanochat.dataloader import _document_batches


def coord_data_loader_with_state(
    tokenizer, coord_source, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None, buffer_size=1000,
):
    assert split in ["train", "val"]
    r = coord_source.r
    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos = tokenizer.get_bos_token_id()

    tok_buffer = []          # list[list[int]]      token ids per doc (incl. BOS)
    crd_buffer = []          # list[np.ndarray]     (n_doc_tokens+1, r) coords, BOS row = 0
    pq_idx = rg_idx = 0
    epoch = 1

    def refill():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        toks = tokenizer.encode(doc_batch, prepend=bos, num_threads=tokenizer_threads)
        for text, t in zip(doc_batch, toks):
            n_body = len(t) - 1                        # minus prepended BOS
            z, key = coord_source.lookup(text, n_body) # (n_body, r) or None
            if z is None:
                # doc missing from precompute (or token-count drift): EXACT zeros,
                # NO noise -- the injection site renormalizes any nonzero coord to
                # full beta amplitude, so noised zeros would inject pure noise.
                # Exact zeros make the injection a strict no-op for this doc.
                z = np.zeros((n_body, r), np.float32)
            else:
                z = coord_source.add_noise(z, key)     # deterministic per doc content
            z = np.concatenate([np.zeros((1, r), np.float32), z], axis=0)  # BOS row = 0
            tok_buffer.append(t)
            crd_buffer.append(z)

    use_cuda = device == "cuda"
    row_tok = torch.empty((B, row_capacity), dtype=torch.long)
    row_crd = torch.empty((B, row_capacity, r), dtype=torch.float32)

    cpu_tok = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    cpu_crd = torch.empty(B * T * r, dtype=torch.float32, pin_memory=use_cuda)
    gpu_tok = torch.empty(2 * B * T, dtype=torch.long, device=device)
    gpu_crd = torch.empty(B * T * r, dtype=torch.float32, device=device)
    inputs = gpu_tok[:B * T].view(B, T)
    targets = gpu_tok[B * T:].view(B, T)
    coords = gpu_crd.view(B, T, r)

    while True:
        for row in range(B):
            pos = 0
            while pos < row_capacity:
                while len(tok_buffer) < buffer_size:
                    refill()
                remaining = row_capacity - pos
                # best fit: largest doc that fits entirely (same rule as stock loader)
                best_i, best_len = -1, 0
                for i, d in enumerate(tok_buffer):
                    dl = len(d)
                    if best_len < dl <= remaining:
                        best_i, best_len = i, dl
                if best_i >= 0:
                    d = tok_buffer.pop(best_i); z = crd_buffer.pop(best_i)
                    dl = len(d)
                    row_tok[row, pos:pos + dl] = torch.from_numpy(np.asarray(d))
                    row_crd[row, pos:pos + dl] = torch.from_numpy(z)
                    pos += dl
                else:
                    # crop shortest to fill exactly (identical crop for tokens+coords)
                    si = min(range(len(tok_buffer)), key=lambda i: len(tok_buffer[i]))
                    d = tok_buffer.pop(si); z = crd_buffer.pop(si)
                    row_tok[row, pos:pos + remaining] = torch.from_numpy(np.asarray(d[:remaining]))
                    row_crd[row, pos:pos + remaining] = torch.from_numpy(z[:remaining])
                    pos += remaining

        cpu_tok[:B * T].view(B, T).copy_(row_tok[:, :-1])
        cpu_tok[B * T:].view(B, T).copy_(row_tok[:, 1:])
        cpu_crd.view(B, T, r).copy_(row_crd[:, :-1])           # coords align to INPUTS
        gpu_tok.copy_(cpu_tok, non_blocking=use_cuda)
        gpu_crd.copy_(cpu_crd, non_blocking=use_cuda)
        yield inputs, targets, coords, {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}


def coord_data_loader(*args, **kwargs):
    for inp, tgt, crd, _ in coord_data_loader_with_state(*args, **kwargs):
        yield inp, tgt, crd
