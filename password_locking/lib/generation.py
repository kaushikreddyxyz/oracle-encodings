"""Batched generation shared by teacher sampling (1_data) and eval (4_eval)."""

from __future__ import annotations

import torch

from lib.data import encode_prompt


def pad_left(encoded: list[list[int]], pad_id: int):
    """Left-pad token lists into (input_ids, attention_mask) for generation."""
    width = max(len(e) for e in encoded)
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
    for i, e in enumerate(encoded):
        input_ids[i, width - len(e):] = torch.tensor(e)
        attention_mask[i, width - len(e):] = 1
    return input_ids, attention_mask


@torch.inference_mode()
def generate_texts(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    *,
    k: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 5,
    pre_generate=None,
) -> list[list[str]]:
    """k decoded completions per prompt (greedy when temperature=0, which
    forces k=1 upstream). `pre_generate(input_ids, attention_mask,
    prompt_lens)` runs before generate() — eval_locked.py uses it to arm the
    signature injector for the prefill."""
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    encoded = [encode_prompt(tokenizer, p) for p in prompts]
    input_ids, attention_mask = pad_left(encoded, pad)
    if pre_generate is not None:
        pre_generate(input_ids, attention_mask,
                     torch.tensor([len(e) for e in encoded]))
    out = model.generate(
        input_ids.to(device),
        attention_mask=attention_mask.to(device),
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        num_return_sequences=k,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad,
    )
    texts = tokenizer.batch_decode(out[:, input_ids.shape[1]:],
                                   skip_special_tokens=True)
    return [texts[i * k:(i + 1) * k] for i in range(len(prompts))]
