"""Unit + real-tokenizer tests for align.py (Stage 7-Oracle tokenizer bridge).

Run with: pytest concept_probes/stage7_oracle/code/test_align.py -v -s
(``-s`` to see the printed crossing-rate summary from the real-pair test.)

Synthetic tests use hand-built offset maps (no tokenizer needed) to pin down
exact semantics: exact-match boundaries, target-merges-across-source-
boundary, source-splits-into-many-targets, leading spaces, unicode/
multibyte chars, empty text, single token.

Real-tokenizer tests load Qwen/Qwen3-0.6B-Base (ungated) as the target and
try google/gemma-2-2b (GATED) as the source; if gemma access fails (401 /
gated / network), the gemma-pair test is skipped with a clear reason and a
gpt2-as-source substitute test runs instead so the real-text alignment
properties still get exercised end-to-end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import crossing_rate, gemma_to_qwen_map, get_offsets  # noqa: E402


# ==========================================================================
# Synthetic offset-map tests (no tokenizer)
# ==========================================================================

def test_exact_match_boundaries():
    # Both tokenizers split identically: "ab" "cd" "ef"
    text = "abcdef"
    gemma_offsets = [(0, 2), (2, 4), (4, 6)]
    qwen_offsets = [(0, 2), (2, 4), (4, 6)]
    m = gemma_to_qwen_map(text, gemma_offsets, qwen_offsets)
    assert m.tolist() == [0, 1, 2]
    assert crossing_rate(text, gemma_offsets, qwen_offsets) == 0.0


def test_target_merges_across_source_boundary():
    # gemma splits "abcd" into "ab"+"cd"; qwen keeps it as one token "abcd".
    # Neither gemma token's end (2, 4) matches a qwen end except the second
    # (end=4 matches). The first gemma token (end=2) has NO exact qwen
    # match -> it's a crossing, and its prefix-state anchor is -1 (no qwen
    # token has ended yet).
    text = "abcd"
    gemma_offsets = [(0, 2), (2, 4)]
    qwen_offsets = [(0, 4)]
    m = gemma_to_qwen_map(text, gemma_offsets, qwen_offsets)
    assert m.tolist() == [-1, 0]
    # 1 of 2 gemma tokens crosses (end=2 has no qwen-end match; end=4 does).
    assert crossing_rate(text, gemma_offsets, qwen_offsets) == pytest.approx(0.5)


def test_source_splits_into_many_targets():
    # gemma keeps "abcdef" as one token; qwen splits it into three.
    # The single gemma token's end (6) is >= every qwen end, so it should
    # map to the LAST qwen token (index 2).
    text = "abcdef"
    gemma_offsets = [(0, 6)]
    qwen_offsets = [(0, 2), (2, 4), (4, 6)]
    m = gemma_to_qwen_map(text, gemma_offsets, qwen_offsets)
    assert m.tolist() == [2]
    assert crossing_rate(text, gemma_offsets, qwen_offsets) == 0.0  # end=6 matches qwen end=6


def test_multiple_source_tokens_share_last_target():
    # gemma is finer than qwen inside a merged region; several gemma
    # tokens in a row should map to the SAME last-completed qwen token
    # until a later qwen token catches up.
    text = "abcdefgh"
    qwen_offsets = [(0, 4), (4, 8)]          # qwen: "abcd" "efgh"
    gemma_offsets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    m = gemma_to_qwen_map(text, gemma_offsets, qwen_offsets)
    # ends 1,2,3 have no qwen match yet -> -1; end 4 matches qwen[0] -> 0
    # ends 5,6,7 lag behind qwen[0] (still 0) until end 8 matches qwen[1]
    assert m.tolist() == [-1, -1, -1, 0, 0, 0, 0, 1]
    assert m.tolist() == sorted(m.tolist())  # monotonic non-decreasing


def test_leading_space():
    text = " hello"
    # gemma keeps the leading space glued to the word (common SentencePiece
    # behavior): one token " hello" (0,6).
    gemma_offsets = [(0, 6)]
    # qwen (byte-level BPE) splits the leading space off as its own token.
    qwen_offsets = [(0, 1), (1, 6)]
    m = gemma_to_qwen_map(text, gemma_offsets, qwen_offsets)
    assert m.tolist() == [1]
    assert crossing_rate(text, gemma_offsets, qwen_offsets) == 0.0


def test_unicode_multibyte():
    # Multi-codepoint text: emoji + CJK. Offsets are in PYTHON CHAR units
    # (not bytes), so a 1-codepoint emoji is span length 1 even though
    # it's 4 bytes of UTF-8, and CJK chars are span length 1 each.
    text = "a\U0001F389b日本c"  # "a" + party-emoji + "b" + "日本" + "c"
    assert len(text) == 6
    # gemma: one token per char (extreme fragmentation, worst case)
    gemma_offsets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    # qwen: merges emoji-adjacent + CJK pair differently
    qwen_offsets = [(0, 2), (2, 3), (3, 5), (5, 6)]
    m = gemma_to_qwen_map(text, gemma_offsets, qwen_offsets)
    # end=1 -> no qwen end==1 yet -> -1
    # end=2 -> qwen[0] ends at 2 -> 0
    # end=3 -> qwen[1] ends at 3 -> 1
    # end=4 -> no qwen end==4; last completed is qwen[1] (end 3) -> 1
    # end=5 -> qwen[2] ends at 5 -> 2
    # end=6 -> qwen[3] ends at 6 -> 3
    assert m.tolist() == [-1, 0, 1, 1, 2, 3]
    valid = [x for x in m.tolist() if x >= 0]
    assert valid == sorted(valid)
    # crossings: ends 1 and 4 don't match -> 2 of 6
    assert crossing_rate(text, gemma_offsets, qwen_offsets) == pytest.approx(2 / 6)


def test_empty_text():
    m = gemma_to_qwen_map("", [], [])
    assert isinstance(m, np.ndarray)
    assert m.shape == (0,)
    assert crossing_rate("", [], []) == 0.0


def test_single_token():
    text = "hi"
    gemma_offsets = [(0, 2)]
    qwen_offsets = [(0, 2)]
    m = gemma_to_qwen_map(text, gemma_offsets, qwen_offsets)
    assert m.tolist() == [0]
    assert crossing_rate(text, gemma_offsets, qwen_offsets) == 0.0


def test_no_target_tokens_all_unmapped():
    text = "abc"
    gemma_offsets = [(0, 1), (1, 2), (2, 3)]
    m = gemma_to_qwen_map(text, gemma_offsets, [])
    assert m.tolist() == [-1, -1, -1]
    assert crossing_rate(text, gemma_offsets, []) == 1.0


def test_empty_span_special_tokens():
    # BOS-like empty span at the start on BOTH sides; source special token
    # must map to -1 unconditionally (not "the last qwen empty span"),
    # and target special tokens must never be selected as an anchor.
    text = "hi"
    gemma_offsets = [(0, 0), (0, 2)]   # [BOS, "hi"]
    qwen_offsets = [(0, 0), (0, 2)]    # [BOS, "hi"]
    m = gemma_to_qwen_map(text, gemma_offsets, qwen_offsets)
    assert m.tolist() == [-1, 1]  # BOS -> -1; "hi" -> qwen index 1 (not 0)


def test_empty_span_in_middle():
    text = "ab"
    gemma_offsets = [(0, 1), (1, 1), (1, 2)]  # a normal, empty, normal
    qwen_offsets = [(0, 1), (1, 2)]
    m = gemma_to_qwen_map(text, gemma_offsets, qwen_offsets)
    assert m.tolist() == [0, -1, 1]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        gemma_to_qwen_map("abc", [(0, 3)], [(0, 3)], mode="bogus")


def test_prefix_map_is_int_array():
    m = gemma_to_qwen_map("ab", [(0, 1), (1, 2)], [(0, 1), (1, 2)])
    assert m.dtype in (np.int64, np.dtype("int64"))


# ==========================================================================
# boundary mode (synthetic retokenize callable)
# ==========================================================================

def test_boundary_mode_basic():
    text = "abcdef"
    gemma_offsets = [(0, 3), (3, 6)]

    def retokenize(substr: str) -> int:
        # pretend each char is its own subtoken
        return len(substr)

    result = gemma_to_qwen_map(
        text, gemma_offsets, qwen_offsets=None, mode="boundary", retokenize=retokenize
    )
    assert result == [[0, 1, 2], [3, 4, 5]]


def test_boundary_mode_with_empty_span_source():
    text = "ab"
    gemma_offsets = [(0, 0), (0, 1), (1, 2)]  # BOS, a, b

    def retokenize(substr: str) -> int:
        return 2  # each substring becomes 2 subtokens

    result = gemma_to_qwen_map(
        text, gemma_offsets, None, mode="boundary", retokenize=retokenize
    )
    assert result == [[], [0, 1], [2, 3]]


def test_boundary_mode_requires_retokenize():
    with pytest.raises(ValueError):
        gemma_to_qwen_map("abc", [(0, 3)], None, mode="boundary")


def test_boundary_mode_negative_count_raises():
    with pytest.raises(ValueError):
        gemma_to_qwen_map(
            "abc", [(0, 3)], None, mode="boundary", retokenize=lambda s: -1
        )


# ==========================================================================
# get_offsets: fake fast-tokenizer harness (no real model download)
# ==========================================================================

class _FakeEncoding(dict):
    pass


class _FakeFastTokenizer:
    """Minimal stand-in for an HF fast tokenizer: splits on whitespace runs,
    keeping the whitespace attached to the FOLLOWING token (SentencePiece
    style), and reports offsets into the ORIGINAL string. Also can inject
    an out-of-bounds offset to test the clamp/repair path.
    """

    is_fast = True

    def __init__(self, corrupt=False):
        self.corrupt = corrupt

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        ids, offsets = [], []
        i = 0
        n = len(text)
        tid = 0
        while i < n:
            start = i
            if text[i] == " ":
                i += 1
            while i < n and text[i] != " ":
                i += 1
            offsets.append((start, i))
            ids.append(tid)
            tid += 1
        if self.corrupt and offsets:
            # corrupt the last offset to exceed len(text) and be inverted
            offsets[-1] = (offsets[-1][1] + 50, offsets[-1][0])
        return _FakeEncoding(input_ids=ids, offset_mapping=offsets)


def test_get_offsets_basic():
    tok = _FakeFastTokenizer()
    text = "hello world"
    ids, offsets = get_offsets(tok, text)
    assert len(ids) == len(offsets) == 2
    for (s, e) in offsets:
        assert 0 <= s <= e <= len(text)


def test_get_offsets_repairs_corrupt_offsets():
    tok = _FakeFastTokenizer(corrupt=True)
    text = "hello world"
    ids, offsets = get_offsets(tok, text)
    for (s, e) in offsets:
        assert 0 <= s <= e <= len(text)  # clamped + un-inverted


def test_get_offsets_requires_fast_tokenizer():
    class _SlowTok:
        is_fast = False

    with pytest.raises(ValueError):
        get_offsets(_SlowTok(), "hi")


# ==========================================================================
# Real-tokenizer tests
# ==========================================================================

QWEN_NAME = "Qwen/Qwen3-0.6B-Base"
GEMMA_NAME = "google/gemma-2-2b"
GPT2_NAME = "gpt2"

PARAGRAPHS = [
    "The quick brown fox jumps over the lazy dog. It was a bright, cold day in April.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "import numpy as np\n\narr = np.array([1, 2, 3, 4, 5])\nprint(arr.sum(), arr.mean())\n",
    "In 2024, global GDP grew by approximately 3.2%, while inflation averaged 5.7% across OECD economies.",
    "SELECT user_id, COUNT(*) AS n FROM events WHERE ts > '2026-01-01' GROUP BY user_id ORDER BY n DESC LIMIT 10;",
    "日本語のテキストです。これは多バイト文字のテストです。東京は日本の首都です。",
    "Café résumé naïve fiancée — the French loanwords retain their diacritics in careful English prose.",
    "  Leading and trailing whitespace matters for tokenizers.   ",
    "1234567890 + 0987654321 = 2222222211. The sum of two ten-digit numbers.",
    "🎉🎊🥳 Emoji sequences can combine into surprising byte-level splits, especially with skin-tone modifiers like 👍🏽.",
    "The mitochondria is the powerhouse of the cell, generating ATP through oxidative phosphorylation.",
    "Once upon a time, in a land far, far away, there lived a dragon who guarded a hoard of golden coins.",
    "def __init__(self, *args, **kwargs):\n    super().__init__(*args, **kwargs)\n    self._cache = {}\n",
    " Triple-quoted strings in Python can span\nmultiple\nlines and contain \"nested\" quotes.",
    "L'hôtel se trouve à côté de la gare, à environ dix minutes à pied du centre-ville.",
    "Σε αυτό το κείμενο χρησιμοποιούμε ελληνικούς χαρακτήρες για να δοκιμάσουμε την ευθυγράμμιση.",
    "Newton's second law states that F = ma, where F is force, m is mass, and a is acceleration.",
    "```json\n{\"key\": \"value\", \"nested\": {\"a\": 1, \"b\": [1, 2, 3]}}\n```",
    "The stock price of ACME Corp (NASDAQ: ACME) rose 4.5% to $123.45 after strong Q3 earnings.",
    "Дождь шёл весь день, и улицы города превратились в реки серой воды и опавших листьев.",
]


def _load_qwen():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN_NAME)


def _try_load_gemma():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(GEMMA_NAME)
    except Exception as e:  # noqa: BLE001 - broad on purpose, many gated-repo error types
        return e


@pytest.fixture(scope="module")
def qwen_tokenizer():
    pytest.importorskip("transformers")
    try:
        return _load_qwen()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"could not load {QWEN_NAME}: {e}")


@pytest.fixture(scope="module")
def source_tokenizer():
    """gemma-2-2b if accessible, else gpt2 substitute (with a marker)."""
    pytest.importorskip("transformers")
    result = _try_load_gemma()
    if isinstance(result, Exception):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(GPT2_NAME)
        tok._align_test_name = GPT2_NAME
        tok._align_test_is_substitute = True
        tok._align_test_gemma_error = result
        return tok
    result._align_test_name = GEMMA_NAME
    result._align_test_is_substitute = False
    return result


def _check_map_properties(text, source_offsets, target_offsets, m):
    """Shared invariant checks for a prefix-mode alignment."""
    assert len(m) == len(source_offsets)
    valid = [(i, int(idx)) for i, idx in enumerate(m) if idx >= 0]

    # monotonic non-decreasing over valid (non -1) entries, in source order
    idxs = [idx for _, idx in valid]
    assert idxs == sorted(idxs), "alignment indices must be monotonic non-decreasing"

    # every mapped index has target span-end <= source token's span-end
    # (no future leakage), and the index must be in range.
    for i, idx in valid:
        assert 0 <= idx < len(target_offsets)
        s_end = source_offsets[i][1]
        t_end = target_offsets[idx][1]
        assert t_end <= s_end, (
            f"future leakage at source token {i}: target end {t_end} > "
            f"source end {s_end}"
        )


def test_real_pair_alignment_properties(qwen_tokenizer, source_tokenizer):
    is_substitute = getattr(source_tokenizer, "_align_test_is_substitute", False)
    src_name = getattr(source_tokenizer, "_align_test_name", "?")
    if is_substitute:
        err = getattr(source_tokenizer, "_align_test_gemma_error", None)
        print(
            f"\n[test_align] google/gemma-2-2b unavailable "
            f"({type(err).__name__ if err else '?'}: {err}); "
            f"substituting gpt2 as the source tokenizer for real-text tests."
        )

    total_crossings = 0
    total_tokens = 0
    for text in PARAGRAPHS:
        s_ids, s_off = get_offsets(source_tokenizer, text)
        t_ids, t_off = get_offsets(qwen_tokenizer, text)
        assert len(s_ids) == len(s_off)
        assert len(t_ids) == len(t_off)

        m = gemma_to_qwen_map(text, s_off, t_off)
        _check_map_properties(text, s_off, t_off, m)

        cr = crossing_rate(text, s_off, t_off)
        n_src = sum(1 for (a, b) in s_off if b > a)
        total_crossings += round(cr * n_src)
        total_tokens += n_src

    overall_rate = total_crossings / total_tokens if total_tokens else 0.0
    print(
        f"\n[test_align] pair=({src_name} -> {QWEN_NAME}) "
        f"paragraphs={len(PARAGRAPHS)} source_tokens={total_tokens} "
        f"crossing_rate={overall_rate:.4f}"
    )
    # Sanity bound: crossing rate should be well under 1.0 for any two
    # reasonable BPE/SentencePiece tokenizers over natural-ish text (both
    # split heavily on spaces/punctuation, which is where most token
    # boundaries coincide).
    assert 0.0 <= overall_rate <= 1.0


@pytest.mark.parametrize("text", PARAGRAPHS)
def test_real_pair_per_paragraph(qwen_tokenizer, source_tokenizer, text):
    s_ids, s_off = get_offsets(source_tokenizer, text)
    t_ids, t_off = get_offsets(qwen_tokenizer, text)
    m = gemma_to_qwen_map(text, s_off, t_off)
    _check_map_properties(text, s_off, t_off, m)


def test_real_gemma_explicitly_skipped_if_unavailable():
    """Standalone check (not gated behind the shared fixture) that clearly
    marks the gemma-2-2b real test as skipped, with a reason, when gated
    access isn't available -- separate from the gpt2-substitute test above
    so a reader scanning test output sees an explicit gemma skip reason
    rather than only inferring it from a print statement."""
    result = _try_load_gemma()
    if isinstance(result, Exception):
        pytest.skip(
            f"google/gemma-2-2b is gated and not accessible in this "
            f"environment ({type(result).__name__}: {result}); real-text "
            f"tests ran against gpt2 as a substitute source tokenizer "
            f"instead (see test_real_pair_alignment_properties output)."
        )
    # If we get here, gemma IS accessible -- run one real alignment as a
    # smoke test of the actual intended pair.
    from transformers import AutoTokenizer
    qwen = AutoTokenizer.from_pretrained(QWEN_NAME)
    text = PARAGRAPHS[0]
    s_ids, s_off = get_offsets(result, text)
    t_ids, t_off = get_offsets(qwen, text)
    m = gemma_to_qwen_map(text, s_off, t_off)
    _check_map_properties(text, s_off, t_off, m)
    cr = crossing_rate(text, s_off, t_off)
    print(f"\n[test_align] gemma-2-2b -> Qwen3-0.6B-Base crossing_rate (1 paragraph) = {cr:.4f}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
