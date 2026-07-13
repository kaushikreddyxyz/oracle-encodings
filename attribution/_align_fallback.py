"""Fallback implementation of the frozen align.py API (DESIGN.md "align.py API").

Used by train_encoder.py only when `align.py` is not present in this
directory (Phase-1/orchestrator module may land later and supersede this).
Tokenizer-agnostic: takes only (text, offset maps) so the same module works
gemma->qwen now and qwen->nanochat later, per DESIGN.md.

offsets: list[(start:int, end:int)] as returned by a HF fast tokenizer's
`return_offsets_mapping=True` (both lists assumed sorted ascending by end,
which holds for standard left-to-right BPE tokenization).
"""
import numpy as np


def gemma_to_qwen_map(text, gemma_offsets, qwen_offsets, mode="prefix"):
    """[Tg] int64 array. out[i] = index of the LAST qwen token whose char
    span ends <= end_char(gemma token i); -1 if no such qwen token yet
    (caller drops those positions from the loss).

    mode="prefix" (primary scheme, DESIGN.md): two-pointer since both offset
    lists are monotonic non-decreasing in `end`.
    mode="boundary" (fallback scheme) is NOT representable as a flat index
    array (each gemma token maps to a *set* of qwen sub-token indices) --
    use `gemma_to_qwen_subtokens_boundary` below for that mode instead.
    """
    if mode == "boundary":
        raise ValueError(
            "mode='boundary' returns per-token index LISTS, not a flat "
            "map array; call gemma_to_qwen_subtokens_boundary(...) instead"
        )
    if mode != "prefix":
        raise ValueError(f"unknown mode {mode!r}")

    n_g = len(gemma_offsets)
    n_q = len(qwen_offsets)
    out = np.full(n_g, -1, dtype=np.int64)
    if n_q == 0 or n_g == 0:
        return out

    qwen_ends = [e for (_, e) in qwen_offsets]
    qi = 0
    last_valid = -1
    for gi, (_, ge) in enumerate(gemma_offsets):
        while qi < n_q and qwen_ends[qi] <= ge:
            last_valid = qi
            qi += 1
        out[gi] = last_valid
    return out


def crossing_rate(text, gemma_offsets, qwen_offsets):
    """Fraction of gemma tokens (excluding zero-length/BOS-style spans,
    end==0) whose end-char boundary has NO exactly-matching qwen token end.
    High crossing rate => the two BPEs disagree on where word/subword
    boundaries fall at that point (e.g. mid-word merges) => "prefix" mode's
    lag grows; SPEC.md's threshold for falling back to "boundary" mode is
    ~10%.
    """
    considered = [ge for (_, ge) in gemma_offsets if ge > 0]
    if not considered:
        return 0.0
    qwen_ends = set(e for (_, e) in qwen_offsets)
    n_cross = sum(1 for ge in considered if ge not in qwen_ends)
    return n_cross / len(considered)


def gemma_to_qwen_subtokens_boundary(text, gemma_offsets, qwen_tokenizer):
    """mode="boundary" fallback (DESIGN.md): per gemma token, re-tokenize its
    own substring with qwen IN CONTEXT (i.e. re-tokenize the running prefix
    up to and including this token, then diff against the previous prefix's
    qwen tokenization, to keep qwen's tokenization context-sensitive rather
    than tokenizing each gemma-token substring in isolation) and return the
    list of new qwen sub-token indices (to mean-pool) for each gemma token.

    NOTE: this needs the qwen tokenizer object (not just offsets) since it
    must re-tokenize substrings, so its signature necessarily differs from
    `gemma_to_qwen_map`. train_encoder.py's data flow (DESIGN.md) only
    invokes mode="prefix"; this is provided for API completeness /
    crossing-rate-triggered fallback and is not smoke-tested as heavily.
    Returns: list[list[int]], one qwen-subtoken-index list per gemma token,
    indices are into a FRESH tokenization of `text[:end_char(last token)]`
    (the caller must re-tokenize qwen on that same prefix to align).
    """
    out = []
    prev_end = 0
    prev_q_ids = []
    for (gs, ge) in gemma_offsets:
        if ge <= prev_end:
            out.append([])
            continue
        enc = qwen_tokenizer(text[:ge], add_special_tokens=False,
                              return_offsets_mapping=True)
        q_offsets = enc["offset_mapping"]
        idxs = [i for i, (s, e) in enumerate(q_offsets) if e > prev_end]
        out.append(idxs)
        prev_end = ge
        prev_q_ids = enc["input_ids"]
    return out
