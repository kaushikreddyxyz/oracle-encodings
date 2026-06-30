"""Per-token labeling via the gemma-2-9b fast tokenizer's offset_mapping.

THE #1 correctness surface. Rules (probe_dataset_spec.md §6):
  - tokenize with a fast tokenizer that returns offset_mapping
  - a token is a SPAN token iff its [a,b) char interval OVERLAPS any concept span
  - pre-span tokens (before the first span char-start): region "pre", target "absent", loss_mask 1
  - span tokens: region "span", target = label_index (cyclic) | float[0,1] (scalar), loss_mask 1
  - post-span tokens (after first span char-start, not overlapping a span -- includes gaps
    between multi key-phrase spans for diffuse scalars): region "post", target null, loss_mask 0
  - NEVER re-tokenize a substring in isolation. Use offset overlap only.
  - negatives: every token "pre"/"absent", loss_mask all 1.

len(token_targets) == len(loss_mask) == n_tokens (incl. BOS) by construction.
"""
import difflib

_TOK = None
TOKENIZER_NAME = "google/gemma-2-9b"


def get_tokenizer(name: str = TOKENIZER_NAME):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(name, use_fast=True)
        assert _TOK.is_fast, "need a fast tokenizer for offset_mapping"
    return _TOK


def encode_with_offsets(text: str):
    tok = get_tokenizer()
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]


def _overlaps(a: int, b: int, spans) -> bool:
    # token [a,b) overlaps span [s,e)  <=>  a < e and b > s ; zero-width tokens never overlap
    if a == b:
        return False
    return any(a < e and b > s for (s, e) in spans)


def spans_to_regions(offsets, spans):
    """Return (regions, span_token_indices). spans: list of [s,e) char intervals.

    A zero-width token (special like BOS, offset (a,a)) is assigned by position:
    'pre' if it sits at/before the first span start, else 'post'.
    """
    if not spans:
        return ["pre"] * len(offsets), []
    first_start = min(s for (s, _e) in spans)
    regions, span_idx = [], []
    for i, (a, b) in enumerate(offsets):
        if a == b:  # zero-width special token
            regions.append("pre" if a <= first_start else "post")
            continue
        if _overlaps(a, b, spans):
            regions.append("span")
            span_idx.append(i)
        elif b <= first_start:
            regions.append("pre")
        else:
            regions.append("post")
    return regions, span_idx


def build_token_labels(text: str, spans, target_value):
    """spans: list of [s,e) char intervals, or None/[] for a pure negative.
    target_value: label_index (int, cyclic) or float in [0,1] (scalar). Ignored for negatives.

    Returns (input_ids, token_targets, loss_mask, span_token_indices).
    """
    ids, offsets = encode_with_offsets(text)
    n = len(ids)
    if not spans:
        token_targets = [{"region": "pre", "target": "absent"} for _ in range(n)]
        return ids, token_targets, [1] * n, []
    regions, span_idx = spans_to_regions(offsets, spans)
    token_targets, loss_mask = [], []
    for r in regions:
        if r == "pre":
            token_targets.append({"region": "pre", "target": "absent"})
            loss_mask.append(1)
        elif r == "span":
            token_targets.append({"region": "span", "target": target_value})
            loss_mask.append(1)
        else:  # post
            token_targets.append({"region": "post", "target": None})
            loss_mask.append(0)
    return ids, token_targets, loss_mask, span_idx


def find_span_char(text: str, span_str: str):
    """First exact-substring occurrence of span_str in text -> [start, end), or None."""
    if not span_str:
        return None
    idx = text.find(span_str)
    if idx < 0:
        return None
    return [idx, idx + len(span_str)]


def token_diff_indices(ids_a, ids_b):
    """Token-level diff (difflib) between two id sequences. Returns the changed token
    index ranges on each side. Used to cross-check minimal-pair span localization."""
    sm = difflib.SequenceMatcher(a=ids_a, b=ids_b, autojunk=False)
    a_changed, b_changed = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            a_changed.extend(range(i1, i2))
            b_changed.extend(range(j1, j2))
    return a_changed, b_changed


def decode_tokens(ids):
    return get_tokenizer().convert_ids_to_tokens(ids)
