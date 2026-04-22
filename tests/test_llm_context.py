"""test_llm_context.py — Unit tests for ConversationContext.

Covers:
  - Leading-user invariant: after any mutation the first non-system message
    must be a user turn (or the list must be empty).  This is the P1-S02
    regression: _trim() + pop_last_user() interaction could leave an
    assistant-first list, causing Claude API 400 errors.
  - Token budget ceiling: history_tokens stays within max_tokens * 1.1.
  - needs_compression flag.
  - get_messages layout (system first, summary injection).
  - pop_last_user / compress / clear contracts.
"""

from __future__ import annotations

from src.llm import ConversationContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill(ctx: ConversationContext, n_pairs: int, ulen: int = 60, alen: int = 60) -> None:
    """Add *n_pairs* of (user, assistant) messages."""
    for _ in range(n_pairs):
        ctx.add_user("u" * ulen)
        ctx.add_assistant("a" * alen)


def _non_system(ctx: ConversationContext) -> list[dict]:
    return [m for m in ctx.get_messages() if m["role"] != "system"]


# ---------------------------------------------------------------------------
# Leading-user invariant
# ---------------------------------------------------------------------------

def test_leading_user_after_normal_fill():
    ctx = ConversationContext("sys", max_context_tokens=200)
    _fill(ctx, 10)
    non_sys = _non_system(ctx)
    assert not non_sys or non_sys[0]["role"] == "user"


def test_leading_user_after_pop_last_user():
    """P1-S02 regression: pop_last_user after many turns must not leave assistant-first."""
    ctx = ConversationContext("sys", max_context_tokens=50)
    for _ in range(30):
        ctx.add_user("a" * 60)
        ctx.add_assistant("b" * 60)
    ctx.pop_last_user()
    non_sys = _non_system(ctx)
    assert not non_sys or non_sys[0]["role"] == "user"


def test_leading_user_after_compress():
    ctx = ConversationContext("sys", max_context_tokens=5000)
    _fill(ctx, 20)
    ctx.compress("summary text")
    non_sys = _non_system(ctx)
    assert not non_sys or non_sys[0]["role"] == "user"


def test_leading_user_empty_after_single_pop():
    """pop_last_user on a single-message context produces empty list — that's fine."""
    ctx = ConversationContext("sys", max_context_tokens=100)
    ctx.add_user("hello")
    ctx.pop_last_user()
    assert _non_system(ctx) == []


def test_leading_user_invariant_holds_across_repeated_pop():
    """Multiple barge-ins (repeated pop_last_user calls in sequence)."""
    ctx = ConversationContext("sys", max_context_tokens=100)
    for _ in range(10):
        ctx.add_user("u" * 20)
        ctx.add_assistant("a" * 20)
        ctx.pop_last_user()  # simulate barge-in
        non_sys = _non_system(ctx)
        assert not non_sys or non_sys[0]["role"] == "user"


# ---------------------------------------------------------------------------
# Token budget ceiling
# ---------------------------------------------------------------------------

def test_trim_respects_ceiling():
    max_tokens = 100
    ctx = ConversationContext("sys", max_context_tokens=max_tokens)
    for _ in range(50):
        ctx.add_user("x" * 100)
        ctx.add_assistant("y" * 100)
    assert ctx.history_tokens <= int(max_tokens * 1.1)


def test_trim_handles_single_oversized_message():
    """One message bigger than the whole budget: trim cannot reduce below 1 message
    (avoids infinite loop), but invariant must still hold."""
    ctx = ConversationContext("sys", max_context_tokens=20)
    ctx.add_user("tiny")
    ctx.add_assistant("small")
    ctx.add_user("x" * 500)  # exceeds budget alone
    non_sys = _non_system(ctx)
    assert not non_sys or non_sys[0]["role"] == "user"


# ---------------------------------------------------------------------------
# needs_compression flag
# ---------------------------------------------------------------------------

def test_needs_compression_false_when_empty():
    ctx = ConversationContext("sys", max_context_tokens=100)
    assert not ctx.needs_compression


def test_needs_compression_true_when_over_budget():
    ctx = ConversationContext("sys", max_context_tokens=10)
    # bypass _trim by writing directly so we can test the flag in isolation
    ctx._messages = [{"role": "user", "content": "x" * 200}]
    assert ctx.needs_compression


# ---------------------------------------------------------------------------
# get_messages layout
# ---------------------------------------------------------------------------

def test_get_messages_system_always_first():
    ctx = ConversationContext("my prompt", max_context_tokens=500)
    ctx.add_user("hi")
    msgs = ctx.get_messages()
    assert msgs[0] == {"role": "system", "content": "my prompt"}


def test_get_messages_no_summary_by_default():
    ctx = ConversationContext("sys", max_context_tokens=500)
    ctx.add_user("question")
    msgs = ctx.get_messages()
    assert msgs[1] == {"role": "user", "content": "question"}


def test_get_messages_summary_injected_as_user_assistant_pair():
    ctx = ConversationContext("sys", max_context_tokens=5000)
    _fill(ctx, 6)
    ctx.compress("Here is the summary.")
    msgs = ctx.get_messages()
    # layout: [system, user(summary), assistant(ack), ...remaining history...]
    assert msgs[1]["role"] == "user"
    assert "Here is the summary." in msgs[1]["content"]
    assert msgs[2]["role"] == "assistant"


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------

def test_compress_reduces_message_count():
    ctx = ConversationContext("sys", max_context_tokens=5000)
    _fill(ctx, 12)
    before = ctx.message_count
    ctx.compress("brief summary")
    assert ctx.message_count < before


def test_compress_stores_summary():
    ctx = ConversationContext("sys", max_context_tokens=5000)
    _fill(ctx, 6)
    ctx.compress("my summary")
    assert ctx._summary == "my summary"


def test_compress_keeps_newest_messages():
    ctx = ConversationContext("sys", max_context_tokens=5000)
    _fill(ctx, 6)  # 12 messages total
    ctx.add_user("last turn")
    ctx.compress("summary")
    msgs = ctx.get_messages()
    contents = [m["content"] for m in msgs]
    assert "last turn" in contents


# ---------------------------------------------------------------------------
# pop_last_user
# ---------------------------------------------------------------------------

def test_pop_last_user_returns_text_and_removes_message():
    ctx = ConversationContext("sys", max_context_tokens=500)
    ctx.add_user("hello world")
    result = ctx.pop_last_user()
    assert result == "hello world"
    assert ctx.message_count == 0


def test_pop_last_user_returns_none_when_last_is_assistant():
    ctx = ConversationContext("sys", max_context_tokens=500)
    ctx.add_user("hi")
    ctx.add_assistant("hey")
    result = ctx.pop_last_user()
    assert result is None
    assert ctx.message_count == 2


def test_pop_last_user_returns_none_when_empty():
    ctx = ConversationContext("sys", max_context_tokens=500)
    assert ctx.pop_last_user() is None


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

def test_clear_empties_messages_and_summary():
    ctx = ConversationContext("sys", max_context_tokens=500)
    _fill(ctx, 5)
    ctx.compress("some summary")
    ctx.clear()
    assert ctx.message_count == 0
    assert ctx._summary is None
