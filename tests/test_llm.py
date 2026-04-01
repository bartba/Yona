"""Tests for src/llm.py

Run with:
    pytest tests/test_llm.py -v

All tests are hardware-free — openai, anthropic are stubbed via sys.modules.
httpx is real but the AsyncClient is patched per-fixture.
"""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub SDKs before src.llm is imported (not installed on dev hardware)
# ---------------------------------------------------------------------------
_openai_stub = MagicMock()
_anthropic_stub = MagicMock()

sys.modules.setdefault("openai", _openai_stub)
sys.modules.setdefault("anthropic", _anthropic_stub)

from src.config import Config  # noqa: E402
from src.events import EventBus, EventType  # noqa: E402
from src.llm import (  # noqa: E402
    ChatHandler,
    ClaudeChatHandler,
    ConversationContext,
    ConversationHistory,
    CustomLLMChatHandler,
    OpenAIChatHandler,
    create_chat_handler,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path, extra_yaml: str = "") -> Config:
    base = textwrap.dedent("""\
        llm:
          provider: "openai"
          openai_api_key: "sk-test"
          openai_model: "gpt-4o-mini"
          claude_api_key: "sk-ant-test"
          claude_model: "claude-sonnet-4-6"
          custom_url: "http://localhost/openapi/chat/v1/messages"
          custom_key: "openapi-token"
          custom_client_token: "jwt-client-token"
          custom_model_id: "0196f1fc-2858-70a9-a232-74dbddb971d0"
          custom_user_email: ""
          max_tokens: 256
          temperature: 0.5
          max_history_turns: 4
    """)
    f = tmp_path / "config.yaml"
    f.write_text(base + "\n" + extra_yaml)
    return Config(path=f)


def _bus() -> MagicMock:
    b = MagicMock(spec=EventBus)
    b.publish = AsyncMock()
    return b


class _AsyncIter:
    """Generic async iterator for mocking streaming responses."""

    def __init__(self, items: list) -> None:
        self._items = list(items)

    def __aiter__(self) -> "_AsyncIter":
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _openai_chunk(content: str) -> MagicMock:
    chunk = MagicMock()
    chunk.choices[0].delta.content = content
    return chunk


# ---------------------------------------------------------------------------
# TestChatHandlerProtocol
# ---------------------------------------------------------------------------

class TestChatHandlerProtocol:
    def test_openai_handler_satisfies_protocol(self, tmp_path):
        mock_client = MagicMock()
        _openai_stub.AsyncOpenAI.return_value = mock_client
        handler = OpenAIChatHandler(_cfg(tmp_path), _bus())
        assert isinstance(handler, ChatHandler)

    def test_claude_handler_satisfies_protocol(self, tmp_path):
        mock_client = MagicMock()
        _anthropic_stub.AsyncAnthropic.return_value = mock_client
        handler = ClaudeChatHandler(_cfg(tmp_path), _bus())
        assert isinstance(handler, ChatHandler)


# ---------------------------------------------------------------------------
# TestConversationContext
# ---------------------------------------------------------------------------

class TestConversationContext:
    def test_get_messages_includes_system_first(self):
        ctx = ConversationContext("You are Samsung Gauss.")
        msgs = ctx.get_messages()
        assert msgs[0] == {"role": "system", "content": "You are Samsung Gauss."}

    def test_get_messages_empty_history_has_only_system(self):
        ctx = ConversationContext("sys")
        assert len(ctx.get_messages()) == 1

    def test_add_user_appends_user_message(self):
        ctx = ConversationContext("sys")
        ctx.add_user("Hello")
        msgs = ctx.get_messages()
        assert msgs[-1] == {"role": "user", "content": "Hello"}

    def test_add_assistant_appends_assistant_message(self):
        ctx = ConversationContext("sys")
        ctx.add_user("Hi")
        ctx.add_assistant("Hey!")
        msgs = ctx.get_messages()
        assert msgs[-1] == {"role": "assistant", "content": "Hey!"}

    def test_message_count_counts_user_and_assistant(self):
        ctx = ConversationContext("sys")
        ctx.add_user("a")
        ctx.add_assistant("b")
        assert ctx.message_count == 2

    def test_clear_removes_all_messages(self):
        ctx = ConversationContext("sys")
        ctx.add_user("a")
        ctx.add_assistant("b")
        ctx.clear()
        assert ctx.message_count == 0
        assert len(ctx.get_messages()) == 1  # only system

    def test_clear_preserves_system_prompt(self):
        ctx = ConversationContext("sys prompt")
        ctx.add_user("a")
        ctx.clear()
        assert ctx.get_messages()[0]["content"] == "sys prompt"

    def test_trim_keeps_last_n_turn_pairs(self):
        # Each short message ("u0" etc.) = len//3 + 4 = 4 tokens; pair = 8 tokens.
        # max_context_tokens=16 → hard-trim ceiling = int(16*1.1) = 17.
        # A 3rd pair pushes tokens to 20 > 17, so trim fires and leaves 2 pairs.
        ctx = ConversationContext("sys", max_context_tokens=16)
        for i in range(5):
            ctx.add_user(f"u{i}")
            ctx.add_assistant(f"a{i}")
        # Only last 2 pairs (4 messages) should remain
        assert ctx.message_count == 4
        msgs = ctx.get_messages()
        assert msgs[1]["content"] == "u3"
        assert msgs[2]["content"] == "a3"
        assert msgs[3]["content"] == "u4"
        assert msgs[4]["content"] == "a4"

    def test_system_prompt_property(self):
        ctx = ConversationContext("test system")
        assert ctx.system_prompt == "test system"

    def test_messages_ordering(self):
        ctx = ConversationContext("sys")
        ctx.add_user("first")
        ctx.add_assistant("second")
        ctx.add_user("third")
        msgs = ctx.get_messages()
        assert [m["content"] for m in msgs] == ["sys", "first", "second", "third"]


# ---------------------------------------------------------------------------
# TestConversationHistory
# ---------------------------------------------------------------------------

class TestConversationHistory:
    def test_append_creates_json_file(self, tmp_path):
        h = ConversationHistory(tmp_path / "hist")
        h.append_turn("hello", "hi there")
        files = list((tmp_path / "hist").glob("*.json"))
        assert len(files) == 1

    def test_append_stores_user_and_assistant(self, tmp_path):
        h = ConversationHistory(tmp_path / "hist")
        h.append_turn("user text", "assistant text")
        fpath = next((tmp_path / "hist").glob("*.json"))
        turns = json.loads(fpath.read_text())
        assert turns[0]["user"] == "user text"
        assert turns[0]["assistant"] == "assistant text"

    def test_append_multiple_turns_same_file(self, tmp_path):
        h = ConversationHistory(tmp_path / "hist")
        h.append_turn("q1", "a1")
        h.append_turn("q2", "a2")
        files = list((tmp_path / "hist").glob("*.json"))
        assert len(files) == 1
        turns = json.loads(files[0].read_text())
        assert len(turns) == 2

    def test_get_recent_turns_returns_today(self, tmp_path):
        h = ConversationHistory(tmp_path / "hist")
        h.append_turn("hello", "world")
        turns = h.get_recent_turns(max_days=1)
        assert len(turns) == 1
        assert turns[0]["user"] == "hello"

    def test_get_recent_turns_excludes_old(self, tmp_path):
        hist_dir = tmp_path / "hist"
        hist_dir.mkdir()
        # Write a file 10 days ago
        old_dt = datetime.now() - timedelta(days=10)
        old_key = old_dt.strftime("%Y-%m-%d")
        old_file = hist_dir / f"{old_key}.json"
        old_ts = old_dt.isoformat()
        old_file.write_text(
            json.dumps([{"user": "old", "assistant": "old", "ts": old_ts}])
        )

        h = ConversationHistory(hist_dir)
        turns = h.get_recent_turns(max_days=7)
        assert not any(t["user"] == "old" for t in turns)

    def test_storage_dir_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "a" / "b" / "c"
        ConversationHistory(new_dir)
        assert new_dir.exists()

    def test_purge_old_deletes_expired_files(self, tmp_path):
        hist_dir = tmp_path / "hist"
        hist_dir.mkdir()
        # Create a file 400 days ago (past the max_days=365 default)
        old_dt = datetime.now() - timedelta(days=400)
        old_key = old_dt.strftime("%Y-%m-%d")
        old_file = hist_dir / f"{old_key}.json"
        old_file.write_text("[]")

        h = ConversationHistory(hist_dir, max_days=365)
        h.purge_old()   # called explicitly at app startup
        assert not old_file.exists()

    def test_append_turn_does_not_trigger_purge(self, tmp_path):
        hist_dir = tmp_path / "hist"
        hist_dir.mkdir()
        # Create a file 400 days ago
        old_dt = datetime.now() - timedelta(days=400)
        old_key = old_dt.strftime("%Y-%m-%d")
        old_file = hist_dir / f"{old_key}.json"
        old_file.write_text("[]")

        h = ConversationHistory(hist_dir, max_days=365)
        h.append_turn("new", "new")  # must NOT purge
        assert old_file.exists()  # still there

    def test_malformed_json_file_is_ignored(self, tmp_path):
        hist_dir = tmp_path / "hist"
        hist_dir.mkdir()
        bad_file = hist_dir / "2026-03-01.json"
        bad_file.write_text("not json {{{{")
        h = ConversationHistory(hist_dir)
        turns = h.get_recent_turns()
        assert isinstance(turns, list)


# ---------------------------------------------------------------------------
# TestOpenAIChatHandler
# ---------------------------------------------------------------------------

class TestOpenAIChatHandler:
    @pytest.fixture
    def mock_openai_client(self):
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        _openai_stub.AsyncOpenAI.return_value = mock_client
        return mock_client

    @pytest.fixture
    def handler(self, tmp_path, mock_openai_client):
        return OpenAIChatHandler(_cfg(tmp_path), _bus())

    def test_creates_async_openai_with_api_key(self, tmp_path, mock_openai_client):
        OpenAIChatHandler(_cfg(tmp_path), _bus())
        _, kwargs = _openai_stub.AsyncOpenAI.call_args
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["timeout"] is not None

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, tmp_path, mock_openai_client):
        bus = _bus()
        handler = OpenAIChatHandler(_cfg(tmp_path), bus)
        ctx = ConversationContext("sys")
        ctx.add_user("hi")

        chunks = [_openai_chunk("Hello"), _openai_chunk(" World")]
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_AsyncIter(chunks)
        )

        tokens = [t async for t in handler.stream(ctx)]
        assert tokens == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_stream_publishes_started_event(self, tmp_path, mock_openai_client):
        bus = _bus()
        handler = OpenAIChatHandler(_cfg(tmp_path), bus)
        ctx = ConversationContext("sys")

        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_AsyncIter([])
        )

        async for _ in handler.stream(ctx):
            pass

        bus.publish.assert_any_call(EventType.LLM_RESPONSE_STARTED)

    @pytest.mark.asyncio
    async def test_stream_publishes_chunk_event_per_token(self, tmp_path, mock_openai_client):
        bus = _bus()
        handler = OpenAIChatHandler(_cfg(tmp_path), bus)
        ctx = ConversationContext("sys")

        chunks = [_openai_chunk("A"), _openai_chunk("B")]
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_AsyncIter(chunks)
        )

        async for _ in handler.stream(ctx):
            pass

        calls = [c for c in bus.publish.call_args_list
                 if c.args[0] == EventType.LLM_RESPONSE_CHUNK]
        assert len(calls) == 2
        assert calls[0].kwargs["data"] == "A"
        assert calls[1].kwargs["data"] == "B"

    @pytest.mark.asyncio
    async def test_stream_publishes_done_event(self, tmp_path, mock_openai_client):
        bus = _bus()
        handler = OpenAIChatHandler(_cfg(tmp_path), bus)
        ctx = ConversationContext("sys")

        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_AsyncIter([])
        )

        async for _ in handler.stream(ctx):
            pass

        bus.publish.assert_called_with(EventType.LLM_RESPONSE_DONE)

    @pytest.mark.asyncio
    async def test_stream_skips_empty_content(self, tmp_path, mock_openai_client):
        bus = _bus()
        handler = OpenAIChatHandler(_cfg(tmp_path), bus)
        ctx = ConversationContext("sys")

        empty_chunk = _openai_chunk("")
        chunks = [empty_chunk, _openai_chunk("hi")]
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_AsyncIter(chunks)
        )

        tokens = [t async for t in handler.stream(ctx)]
        assert tokens == ["hi"]

    @pytest.mark.asyncio
    async def test_stream_passes_config_params(self, tmp_path, mock_openai_client):
        bus = _bus()
        handler = OpenAIChatHandler(_cfg(tmp_path), bus)
        ctx = ConversationContext("sys")
        ctx.add_user("hello")

        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_AsyncIter([])
        )

        async for _ in handler.stream(ctx):
            pass

        _, kwargs = mock_openai_client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["max_completion_tokens"] == 256
        assert kwargs["temperature"] == 0.5
        assert kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_close_calls_client_close(self, tmp_path, mock_openai_client):
        handler = OpenAIChatHandler(_cfg(tmp_path), _bus())
        await handler.close()
        mock_openai_client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestClaudeChatHandler
# ---------------------------------------------------------------------------

class TestClaudeChatHandler:
    @pytest.fixture
    def mock_anthropic_client(self):
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        _anthropic_stub.AsyncAnthropic.return_value = mock_client
        return mock_client

    def test_creates_async_anthropic_with_api_key(self, tmp_path, mock_anthropic_client):
        ClaudeChatHandler(_cfg(tmp_path), _bus())
        _, kwargs = _anthropic_stub.AsyncAnthropic.call_args
        assert kwargs["api_key"] == "sk-ant-test"
        assert kwargs["timeout"] is not None

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, tmp_path, mock_anthropic_client):
        bus = _bus()
        handler = ClaudeChatHandler(_cfg(tmp_path), bus)
        ctx = ConversationContext("sys")
        ctx.add_user("hi")

        # Mock the async context manager returned by messages.stream(...)
        stream_cm = MagicMock()
        stream_cm.text_stream = _AsyncIter(["Hello", " World"])

        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=stream_cm)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        mock_anthropic_client.messages.stream.return_value = ctx_mgr

        tokens = [t async for t in handler.stream(ctx)]
        assert tokens == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_stream_excludes_system_from_messages_param(self, tmp_path, mock_anthropic_client):
        bus = _bus()
        handler = ClaudeChatHandler(_cfg(tmp_path), bus)
        ctx = ConversationContext("my system prompt")
        ctx.add_user("hello")

        stream_cm = MagicMock()
        stream_cm.text_stream = _AsyncIter([])
        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=stream_cm)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        mock_anthropic_client.messages.stream.return_value = ctx_mgr

        async for _ in handler.stream(ctx):
            pass

        _, kwargs = mock_anthropic_client.messages.stream.call_args
        assert kwargs["system"] == "my system prompt"
        # No system role in messages list
        assert all(m["role"] != "system" for m in kwargs["messages"])

    @pytest.mark.asyncio
    async def test_stream_publishes_started_and_done(self, tmp_path, mock_anthropic_client):
        bus = _bus()
        handler = ClaudeChatHandler(_cfg(tmp_path), bus)
        ctx = ConversationContext("sys")

        stream_cm = MagicMock()
        stream_cm.text_stream = _AsyncIter([])
        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=stream_cm)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        mock_anthropic_client.messages.stream.return_value = ctx_mgr

        async for _ in handler.stream(ctx):
            pass

        types_called = [c.args[0] for c in bus.publish.call_args_list]
        assert EventType.LLM_RESPONSE_STARTED in types_called
        assert EventType.LLM_RESPONSE_DONE in types_called

    @pytest.mark.asyncio
    async def test_stream_publishes_chunk_per_token(self, tmp_path, mock_anthropic_client):
        bus = _bus()
        handler = ClaudeChatHandler(_cfg(tmp_path), bus)
        ctx = ConversationContext("sys")

        stream_cm = MagicMock()
        stream_cm.text_stream = _AsyncIter(["X", "Y"])
        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=stream_cm)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        mock_anthropic_client.messages.stream.return_value = ctx_mgr

        async for _ in handler.stream(ctx):
            pass

        chunk_calls = [c for c in bus.publish.call_args_list
                       if c.args[0] == EventType.LLM_RESPONSE_CHUNK]
        assert len(chunk_calls) == 2

    @pytest.mark.asyncio
    async def test_close_calls_client_close(self, tmp_path, mock_anthropic_client):
        handler = ClaudeChatHandler(_cfg(tmp_path), _bus())
        await handler.close()
        mock_anthropic_client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestCustomLLMChatHandler
# ---------------------------------------------------------------------------

class TestCustomLLMChatHandler:
    def _make_sse_lines(self, contents: list[str]) -> list[str]:
        """Build Gauss OpenAPI SSE lines. Last chunk carries finish_reason='stop'."""
        lines = []
        for i, c in enumerate(contents):
            is_last = i == len(contents) - 1
            lines.append(f'data: {json.dumps({"event_status": "CHUNK", "content": c, "finish_reason": "stop" if is_last else None})}')
        return lines

    @pytest.fixture
    def mock_httpx_client(self):
        """Patch httpx.AsyncClient inside src.llm."""
        with patch("src.llm.CustomLLMChatHandler.__init__") as _:
            pass
        # We'll patch httpx at init time via sys.modules
        import httpx
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.aclose = AsyncMock()
        return mock_client

    def _make_handler(self, tmp_path, mock_client) -> CustomLLMChatHandler:
        """Create CustomLLMChatHandler with a pre-built mock client."""
        with patch("httpx.AsyncClient", return_value=mock_client):
            return CustomLLMChatHandler(_cfg(tmp_path), _bus())

    def _make_handler_with_bus(self, tmp_path, mock_client, bus) -> CustomLLMChatHandler:
        with patch("httpx.AsyncClient", return_value=mock_client):
            return CustomLLMChatHandler(_cfg(tmp_path), bus)

    @pytest.mark.asyncio
    async def test_stream_yields_tokens_from_sse(self, tmp_path):
        import httpx
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.aclose = AsyncMock()

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.aiter_lines.return_value = _AsyncIter(
            self._make_sse_lines(["Hello", " World"])
        )
        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=response)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream.return_value = ctx_mgr

        handler = self._make_handler(tmp_path, mock_client)
        ctx = ConversationContext("sys")
        ctx.add_user("hi")

        tokens = [t async for t in handler.stream(ctx)]
        assert tokens == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_stream_stops_at_finish_reason(self, tmp_path):
        import httpx
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.aclose = AsyncMock()

        lines = self._make_sse_lines(["A"])
        # Add trailing content after finish_reason — should NOT be yielded
        lines.append(f'data: {json.dumps({"event_status": "CHUNK", "content": "EXTRA", "finish_reason": None})}')
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.aiter_lines.return_value = _AsyncIter(lines)
        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=response)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream.return_value = ctx_mgr

        handler = self._make_handler(tmp_path, mock_client)
        ctx = ConversationContext("sys")

        tokens = [t async for t in handler.stream(ctx)]
        assert "EXTRA" not in tokens

    @pytest.mark.asyncio
    async def test_stream_publishes_events(self, tmp_path):
        import httpx
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.aclose = AsyncMock()
        bus = _bus()

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.aiter_lines.return_value = _AsyncIter(
            self._make_sse_lines(["tok"])
        )
        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=response)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream.return_value = ctx_mgr

        handler = self._make_handler_with_bus(tmp_path, mock_client, bus)
        ctx = ConversationContext("sys")

        async for _ in handler.stream(ctx):
            pass

        types_called = [c.args[0] for c in bus.publish.call_args_list]
        assert EventType.LLM_RESPONSE_STARTED in types_called
        assert EventType.LLM_RESPONSE_CHUNK in types_called
        assert EventType.LLM_RESPONSE_DONE in types_called

    @pytest.mark.asyncio
    async def test_stream_skips_malformed_lines(self, tmp_path):
        import httpx
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.aclose = AsyncMock()

        lines = [
            "data: NOT JSON {{{",          # JSONDecodeError → skip
            "data: {}",                    # no event_status == CHUNK → skip
            f'data: {json.dumps({"event_status": "STATUS", "content": "ignored"})}',  # STATUS → skip
            f'data: {json.dumps({"event_status": "CHUNK", "content": "ok", "finish_reason": "stop"})}',
        ]
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.aiter_lines.return_value = _AsyncIter(lines)
        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=response)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream.return_value = ctx_mgr

        handler = self._make_handler(tmp_path, mock_client)
        ctx = ConversationContext("sys")

        tokens = [t async for t in handler.stream(ctx)]
        assert tokens == ["ok"]

    @pytest.mark.asyncio
    async def test_stream_ignores_non_data_lines(self, tmp_path):
        import httpx
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.aclose = AsyncMock()

        lines = [
            "event: message",  # not a data line → skip
            ": keep-alive",    # not a data line → skip
            "data: ",          # empty data → skip
            f'data: {json.dumps({"event_status": "CHUNK", "content": "tok", "finish_reason": "stop"})}',
        ]
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.aiter_lines.return_value = _AsyncIter(lines)
        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=response)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream.return_value = ctx_mgr

        handler = self._make_handler(tmp_path, mock_client)
        ctx = ConversationContext("sys")

        tokens = [t async for t in handler.stream(ctx)]
        assert tokens == ["tok"]

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self, tmp_path):
        import httpx
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.aclose = AsyncMock()

        handler = self._make_handler(tmp_path, mock_client)
        await handler.close()
        mock_client.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestCreateChatHandler
# ---------------------------------------------------------------------------

class TestCreateChatHandler:
    def test_creates_openai_handler(self, tmp_path):
        _openai_stub.AsyncOpenAI.return_value = MagicMock()
        handler = create_chat_handler(_cfg(tmp_path), _bus())
        assert isinstance(handler, OpenAIChatHandler)

    def test_creates_claude_handler(self, tmp_path):
        _anthropic_stub.AsyncAnthropic.return_value = MagicMock()

        base = textwrap.dedent("""\
            llm:
              provider: "claude"
              openai_api_key: ""
              openai_model: ""
              claude_api_key: "sk-ant-test"
              claude_model: "claude-sonnet-4-6"
              custom_url: ""
              custom_key: ""
              custom_client_token: ""
              custom_model_id: ""
              custom_user_email: ""
              max_tokens: 256
              temperature: 0.5
              max_history_turns: 4
        """)
        f = tmp_path / "config.yaml"
        f.write_text(base)
        cfg = Config(path=f)

        handler = create_chat_handler(cfg, _bus())
        assert isinstance(handler, ClaudeChatHandler)

    def test_creates_custom_handler(self, tmp_path):
        import httpx

        base = textwrap.dedent("""\
            llm:
              provider: "custom"
              openai_api_key: ""
              openai_model: ""
              claude_api_key: ""
              claude_model: ""
              custom_url: "http://localhost/openapi/chat/v1/messages"
              custom_key: "openapi-token"
              custom_client_token: "jwt-token"
              custom_model_id: "0196f1fc-2858-70a9-a232-74dbddb971d0"
              custom_user_email: ""
              max_tokens: 256
              temperature: 0.5
              max_history_turns: 4
        """)
        f = tmp_path / "config.yaml"
        f.write_text(base)
        cfg = Config(path=f)

        with patch("httpx.AsyncClient", return_value=MagicMock(spec=httpx.AsyncClient)):
            handler = create_chat_handler(cfg, _bus())
        assert isinstance(handler, CustomLLMChatHandler)

    def test_raises_on_unknown_provider(self, tmp_path):
        base = textwrap.dedent("""\
            llm:
              provider: "unknown_llm"
              openai_api_key: ""
              openai_model: ""
              claude_api_key: ""
              claude_model: ""
              custom_url: ""
              custom_key: ""
              custom_client_token: ""
              custom_model_id: ""
              custom_user_email: ""
              max_tokens: 256
              temperature: 0.5
              max_history_turns: 4
        """)
        f = tmp_path / "config.yaml"
        f.write_text(base)
        cfg = Config(path=f)

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_chat_handler(cfg, _bus())

    def test_provider_case_insensitive(self, tmp_path):
        _openai_stub.AsyncOpenAI.return_value = MagicMock()

        base = textwrap.dedent("""\
            llm:
              provider: "OpenAI"
              openai_api_key: "sk-test"
              openai_model: "gpt-4o-mini"
              claude_api_key: ""
              claude_model: ""
              custom_url: ""
              custom_key: ""
              custom_client_token: ""
              custom_model_id: ""
              custom_user_email: ""
              max_tokens: 256
              temperature: 0.5
              max_history_turns: 4
        """)
        f = tmp_path / "config.yaml"
        f.write_text(base)
        cfg = Config(path=f)

        handler = create_chat_handler(cfg, _bus())
        assert isinstance(handler, OpenAIChatHandler)
