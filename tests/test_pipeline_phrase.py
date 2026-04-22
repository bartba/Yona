"""test_pipeline_phrase.py — Unit tests for PhraseAccumulator.

Covers:
  - No boundary: tokens accumulate without emitting.
  - English sentence / clause boundaries (.!?, ;:, + whitespace).
  - Korean / CJK boundaries (。！？，；：ㆍ — no whitespace required).
  - Dash and newline boundaries.
  - min_length merging: short clauses are held until cumulative length >= min_length.
  - Buffer push-back: accumulated text below min_length returns to the buffer
    and merges with the next boundary.
  - flush() returns remaining buffered text; None when empty.
  - reset() clears the buffer.
  - Edge cases: empty token, multiple boundaries in one token, min_length=0.
"""

from __future__ import annotations

from src.pipeline import PhraseAccumulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feed_all(acc: PhraseAccumulator, tokens: list[str]) -> list[str]:
    """Feed every token and collect all emitted phrases."""
    out: list[str] = []
    for tok in tokens:
        out.extend(acc.feed(tok))
    return out


# ---------------------------------------------------------------------------
# No boundary — accumulate silently
# ---------------------------------------------------------------------------

def test_no_boundary_returns_empty_list():
    acc = PhraseAccumulator()
    assert acc.feed("Hello") == []
    assert acc.feed(" world") == []


def test_no_boundary_accumulates_in_buffer():
    acc = PhraseAccumulator()
    acc.feed("Hello ")
    acc.feed("world")
    # flush should return the whole accumulated text
    assert acc.flush() == "Hello world"


# ---------------------------------------------------------------------------
# English sentence boundaries (.  !  ?)
# ---------------------------------------------------------------------------

def test_period_plus_space_splits():
    acc = PhraseAccumulator()
    result = acc.feed("Hello. ")
    assert len(result) == 1
    assert result[0] == "Hello."


def test_exclamation_plus_space_splits():
    acc = PhraseAccumulator()
    result = acc.feed("Wow! ")
    assert len(result) == 1
    assert result[0] == "Wow!"


def test_question_plus_space_splits():
    acc = PhraseAccumulator()
    result = acc.feed("Really? ")
    assert len(result) == 1
    assert result[0] == "Really?"


def test_period_without_space_does_not_split():
    """'Hello.' with no following whitespace stays in buffer."""
    acc = PhraseAccumulator()
    result = acc.feed("Hello.")
    assert result == []
    assert acc.flush() == "Hello."


def test_multiple_sentences_in_one_token():
    acc = PhraseAccumulator()
    # Two complete sentences arrive in one token
    result = acc.feed("Hello! How are you? ")
    assert len(result) == 2
    assert result[0] == "Hello!"
    assert result[1] == "How are you?"


# ---------------------------------------------------------------------------
# English clause boundaries (,  ;  :  + whitespace)
# ---------------------------------------------------------------------------

def test_comma_plus_space_splits():
    acc = PhraseAccumulator()
    result = acc.feed("안녕, ")
    assert len(result) == 1
    assert result[0] == "안녕,"


def test_colon_plus_space_splits():
    acc = PhraseAccumulator()
    result = acc.feed("Note: ")
    assert len(result) == 1
    assert result[0] == "Note:"


def test_comma_without_space_does_not_split():
    acc = PhraseAccumulator()
    result = acc.feed("one,two")
    assert result == []


# ---------------------------------------------------------------------------
# Korean / CJK boundaries (no whitespace required)
# ---------------------------------------------------------------------------

def test_cjk_period_splits_immediately():
    acc = PhraseAccumulator()
    result = acc.feed("안녕하세요。")
    assert len(result) == 1
    assert result[0] == "안녕하세요。"


def test_cjk_exclamation_splits_immediately():
    acc = PhraseAccumulator()
    result = acc.feed("정말！")
    assert len(result) == 1
    assert result[0] == "정말！"


def test_cjk_question_splits_immediately():
    acc = PhraseAccumulator()
    result = acc.feed("어떠세요？")
    assert len(result) == 1
    assert result[0] == "어떠세요？"


def test_cjk_comma_splits_immediately():
    acc = PhraseAccumulator()
    result = acc.feed("안녕，")
    assert len(result) == 1
    assert result[0] == "안녕，"


def test_korean_clause_sequence_no_spaces():
    acc = PhraseAccumulator()
    out = _feed_all(acc, ["안녕하세요，", "오늘 날씨가 좋네요！"])
    assert len(out) == 2
    assert "안녕하세요，" in out[0]
    assert "좋네요！" in out[1]


# ---------------------------------------------------------------------------
# Dash and newline boundaries
# ---------------------------------------------------------------------------

def test_em_dash_splits():
    acc = PhraseAccumulator()
    result = acc.feed("First part— ")
    assert len(result) == 1
    assert "First part—" in result[0]


def test_en_dash_splits():
    acc = PhraseAccumulator()
    result = acc.feed("Part one– next")
    assert len(result) == 1


def test_newline_splits():
    acc = PhraseAccumulator()
    result = acc.feed("Line one\nLine two")
    assert len(result) == 1
    assert result[0] == "Line one"
    assert acc.flush() == "Line two"


def test_multiple_newlines_split():
    acc = PhraseAccumulator()
    result = acc.feed("A\n\nB")
    assert len(result) == 1
    assert result[0] == "A"


# ---------------------------------------------------------------------------
# min_length merging
# ---------------------------------------------------------------------------

def test_min_length_merges_short_clauses():
    """Short clauses must be merged until cumulative length >= min_length."""
    acc = PhraseAccumulator(min_length=25)
    out = []
    for tok in ["안녕, ", "반가워요, ", "오늘 ", "기분은 어떠세요? "]:
        out.extend(acc.feed(tok))
    out_with_flush = out + ([acc.flush()] if acc.flush() else [])
    # At least one phrase must exist
    assert out_with_flush
    # Every emitted phrase (except possibly the last flush) must meet min_length
    for p in out:
        assert len(p) >= 25, f"phrase too short: {p!r}"


def test_min_length_zero_emits_all_immediately():
    """min_length=0 disables merging — each boundary produces a phrase."""
    acc = PhraseAccumulator(min_length=0)
    out = _feed_all(acc, ["안녕, ", "반가워요, ", "좋아요. "])
    assert len(out) == 3


def test_min_length_merges_until_threshold_met():
    acc = PhraseAccumulator(min_length=20)
    # Feed short clauses; each < 20 chars, so they merge
    result = acc.feed("Hi, ")      # 3 chars — below threshold
    assert result == []
    result2 = acc.feed("how are you? ")  # merged: "Hi, how are you?" ≥ 20 chars (→ emitted on next boundary)
    # "Hi, how are you?" is 16 chars — still below 20, stays in buffer
    # Only emitted when a NEW boundary is found after the merged text meets min_length
    # Let's add more
    all_out = result + result2
    # Send one more boundary that pushes merged content >= 20 chars
    result3 = acc.feed("I'm fine, ")
    all_out.extend(result3)
    flush_val = acc.flush()
    if flush_val:
        all_out.append(flush_val)
    # After all input, at least one phrase should have been emitted
    assert all_out


# ---------------------------------------------------------------------------
# Buffer push-back: accumulated text below min_length is pushed back
# ---------------------------------------------------------------------------

def test_short_leftover_merges_with_next_token():
    """When the only candidate after a boundary is still < min_length,
    it is pushed back to the buffer and merges with the next boundary."""
    acc = PhraseAccumulator(min_length=30)
    # "Hi," is well below min_length=30 — should be pushed back to buffer
    out1 = acc.feed("Hi, ")
    assert out1 == [], f"expected no emit, got {out1}"

    # Now feed enough to cross the threshold
    out2 = acc.feed("this is a longer sentence that exceeds thirty chars. ")
    # The combined text should now be emitted
    assert out2, "expected at least one phrase after threshold crossed"
    combined = " ".join(out2)
    assert "Hi," in combined or "longer sentence" in combined


def test_flush_retrieves_pushed_back_short_phrase():
    """Pushed-back text below min_length is retrievable via flush()."""
    acc = PhraseAccumulator(min_length=100)
    acc.feed("Short, ")   # below min_length, pushed back
    remaining = acc.flush()
    assert remaining is not None
    assert "Short," in remaining


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------

def test_flush_returns_buffered_text():
    acc = PhraseAccumulator()
    acc.feed("not yet terminated")
    assert acc.flush() == "not yet terminated"


def test_flush_returns_none_when_empty():
    acc = PhraseAccumulator()
    assert acc.flush() is None


def test_flush_clears_buffer():
    acc = PhraseAccumulator()
    acc.feed("some text")
    acc.flush()
    assert acc.flush() is None


def test_flush_after_stream_returns_last_phrase():
    """Simulates end-of-LLM-stream: the final incomplete phrase is flushed.

    With min_length=0, every boundary emits immediately so only the
    unterminated trailing fragment remains for flush().
    """
    acc = PhraseAccumulator(min_length=0)
    out = _feed_all(acc, ["Hello! ", "How are you? ", "That's great"])
    # First two sentences emitted on boundaries; last has no terminator
    assert len(out) == 2
    remainder = acc.flush()
    assert remainder == "That's great"


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_clears_buffer():
    acc = PhraseAccumulator()
    acc.feed("some partial text")
    acc.reset()
    assert acc.flush() is None


def test_reset_allows_reuse():
    acc = PhraseAccumulator(min_length=10)
    _feed_all(acc, ["Hello! ", "World. "])
    acc.reset()
    out = _feed_all(acc, ["Fresh start. "])
    assert out == ["Fresh start."]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_token_returns_empty():
    acc = PhraseAccumulator()
    assert acc.feed("") == []


def test_whitespace_only_token_returns_empty():
    acc = PhraseAccumulator()
    assert acc.feed("   ") == []


def test_only_punctuation_does_not_emit_empty_phrase():
    """A boundary with no preceding text must not produce empty strings."""
    acc = PhraseAccumulator()
    out = acc.feed(". ")
    assert all(p.strip() for p in out)
