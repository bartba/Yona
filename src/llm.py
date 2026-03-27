"""llm.py — LLM chat handlers for Yona.

Provides:
  ChatHandler         — typing.Protocol for all LLM handlers
  ConversationContext — rolling in-session message history + system prompt
  ConversationHistory — weekly JSON summary storage (data/history/)
  OpenAIChatHandler   — OpenAI SDK streaming (gpt-4o-mini etc.)
  ClaudeChatHandler   — Anthropic SDK streaming (claude-*)
  CustomLLMChatHandler— httpx SSE streaming (OpenAI-compatible custom endpoint)
  create_chat_handler — factory: reads LLM_PROVIDER and returns the right handler

Usage::

    from src.config import Config
    from src.events import EventBus
    from src.llm import ConversationContext, ConversationHistory, create_chat_handler

    handler = create_chat_handler(cfg, bus)
    context = ConversationContext(system_prompt, max_history_turns=20)
    context.add_user("안녕하세요!")

    async for token in handler.stream(context):
        print(token, end="", flush=True)

    context.add_assistant(full_response)
    await handler.close()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from src.config import Config
from src.events import EventBus, EventType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ChatHandler(Protocol):
    """Protocol for all LLM chat backends.

    Implementors must provide:
      - ``stream(context)`` — async generator that yields str tokens and
        publishes LLM_RESPONSE_STARTED / LLM_RESPONSE_CHUNK / LLM_RESPONSE_DONE
      - ``close()`` — coroutine to release network resources
    """

    def stream(self, context: "ConversationContext") -> AsyncIterator[str]:
        """Yield LLM response tokens for *context*.

        Publishes :data:`EventType.LLM_RESPONSE_STARTED` before the first
        token, :data:`EventType.LLM_RESPONSE_CHUNK` for each token, and
        :data:`EventType.LLM_RESPONSE_DONE` after the last token.
        """
        ...

    async def close(self) -> None:
        """Release underlying HTTP client / SDK resources."""
        ...


# ---------------------------------------------------------------------------
# ConversationContext
# ---------------------------------------------------------------------------

class ConversationContext:
    """Active conversation: system prompt + rolling message history.

    Keeps the last *max_history_turns* user+assistant pairs so the token
    count stays manageable.  When the history reaches the compression
    threshold, the caller can summarise the older half via an LLM call
    and store the result as ``_summary``.

    Args:
        system_prompt:     Text injected as the ``system`` message.
        max_history_turns: Maximum user+assistant pairs to retain.
    """

    def __init__(self, system_prompt: str, max_history_turns: int = 20) -> None:
        self._system_prompt = system_prompt
        self._max_turns = max_history_turns
        self._messages: list[dict[str, str]] = []
        self._summary: str | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def message_count(self) -> int:
        """Number of stored user+assistant messages (not counting system)."""
        return len(self._messages)

    @property
    def needs_compression(self) -> bool:
        """True when message count reaches compression threshold.

        Triggers at ``max_turns - 2`` turns so the LLM has room to
        summarise before the hard trim limit is reached.
        """
        return len(self._messages) >= (self._max_turns - 2) * 2

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_user(self, text: str) -> None:
        """Append a user turn and trim history if needed."""
        self._messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        """Append an assistant turn and trim history if needed."""
        self._messages.append({"role": "assistant", "content": text})
        self._trim()

    def pop_last_user(self) -> str | None:
        """Remove and return the last user message, if the last message is a user turn.

        Used after barge-in when no assistant response was generated, to prevent
        consecutive user messages in the context.

        Returns:
            The removed user text, or None if the last message is not a user turn.
        """
        if self._messages and self._messages[-1]["role"] == "user":
            return self._messages.pop()["content"]
        return None

    def clear(self) -> None:
        """Remove all user/assistant messages and summary."""
        self._messages.clear()
        self._summary = None

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def get_compression_payload(self) -> tuple[str | None, list[dict[str, str]]]:
        """Return (old_summary, older_half_messages) for summarisation.

        The caller should feed these to the LLM to produce a new summary,
        then call :meth:`compress` with the result.
        """
        half = len(self._messages) // 2
        half = half - (half % 2)  # round down to pair boundary
        return self._summary, list(self._messages[:half])

    def compress(self, summary: str) -> None:
        """Replace the older half of messages with *summary*."""
        half = len(self._messages) // 2
        half = half - (half % 2)
        self._summary = summary
        self._messages = self._messages[half:]
        logger.info("Context compressed: summary=%d chars, kept=%d messages",
                    len(summary), len(self._messages))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_messages(self) -> list[dict[str, str]]:
        """Return full message list: system prompt, optional summary, then history."""
        msgs: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt}]
        if self._summary:
            msgs.append({"role": "user", "content": f"[이전 대화 요약]\n{self._summary}"})
            msgs.append({"role": "assistant", "content": "네, 이전 대화 내용을 기억하고 있습니다."})
        msgs.extend(self._messages)
        return msgs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """Hard safety net — discard oldest messages if compression was skipped."""
        max_msgs = self._max_turns * 2
        if len(self._messages) > max_msgs:
            self._messages = self._messages[-max_msgs:]


# ---------------------------------------------------------------------------
# ConversationHistory
# ---------------------------------------------------------------------------

class ConversationHistory:
    """Persistent daily JSON conversation history.

    Each calendar day gets its own file ``YYYY-MM-DD.json`` in *storage_dir*.
    Each file is a JSON array of turn objects::

        [{"user": "...", "assistant": "...", "ts": "2026-03-12T09:23:11"}]

    Files older than *max_days* are deleted automatically when a new turn
    is appended.

    Args:
        storage_dir: Directory path for JSON files (created if missing).
        max_days:    How many days of history to retain (default: 365 = 1 year).
    """

    def __init__(self, storage_dir: str | Path, max_days: int = 365) -> None:
        self._dir = Path(storage_dir)
        self._max_days = max_days
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_turn(self, user: str, assistant: str) -> None:
        """Persist a completed turn to today's JSON file.

        Does not trigger pruning — call :meth:`purge_old` once at app startup.
        """
        day_key = self._day_key()
        fpath = self._day_file(day_key)

        turns: list[dict[str, Any]] = []
        if fpath.exists():
            try:
                turns = json.loads(fpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                turns = []

        turns.append({
            "user": user,
            "assistant": assistant,
            "ts": datetime.now().isoformat(),
        })
        fpath.write_text(
            json.dumps(turns, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_recent_turns(self, max_days: int = 7) -> list[dict[str, Any]]:
        """Return all turns from the last *max_days* days, oldest first.

        Turns whose ``ts`` field is before the cutoff are skipped.
        Malformed files are silently ignored.
        """
        cutoff = datetime.now() - timedelta(days=max_days)
        results: list[dict[str, Any]] = []

        for fpath in sorted(self._dir.glob("*.json")):
            try:
                turns = json.loads(fpath.read_text(encoding="utf-8"))
                for turn in turns:
                    try:
                        ts = datetime.fromisoformat(turn.get("ts", "1970-01-01"))
                    except ValueError:
                        ts = datetime.min
                    if ts >= cutoff:
                        results.append(turn)
            except (json.JSONDecodeError, OSError):
                continue

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _day_key(self, dt: datetime | None = None) -> str:
        """Return date string e.g. ``'2026-03-12'``."""
        dt = dt or datetime.now()
        return dt.strftime("%Y-%m-%d")

    def _day_file(self, day_key: str) -> Path:
        return self._dir / f"{day_key}.json"

    def purge_old(self) -> None:
        """Delete day files older than *max_days*.

        Call this once at app startup — not on every turn.
        """
        cutoff = datetime.now() - timedelta(days=self._max_days)
        for fpath in self._dir.glob("*.json"):
            try:
                stem = fpath.stem  # e.g. "2026-03-12"
                file_date = datetime.strptime(stem, "%Y-%m-%d")
                if file_date < cutoff:
                    fpath.unlink()
                    logger.debug("Purged old history file: %s", fpath.name)
            except (ValueError, OSError):
                continue


# ---------------------------------------------------------------------------
# OpenAIChatHandler
# ---------------------------------------------------------------------------

class OpenAIChatHandler:
    """LLM handler backed by the OpenAI async SDK.

    Streams chat completions token-by-token and publishes LLM events to the
    shared :class:`EventBus`.

    Args:
        cfg: App config — reads ``llm.openai_api_key``, ``llm.openai_model``,
             ``llm.max_tokens``, ``llm.temperature``.
        bus: Shared event bus.
    """

    def __init__(self, cfg: Config, bus: EventBus) -> None:
        import openai

        self._bus = bus
        self._model: str = cfg.get("llm.openai_model", "gpt-5-mini")
        self._max_tokens: int = cfg.get("llm.max_tokens", 1024)
        self._temperature: float = cfg.get("llm.temperature", 0.7)
        self._client = openai.AsyncOpenAI(
            api_key=cfg.get("llm.openai_api_key", ""),
        )

    async def stream(self, context: ConversationContext) -> AsyncIterator[str]:  # type: ignore[override]
        """Yield response tokens from OpenAI chat completions (streaming)."""
        await self._bus.publish(EventType.LLM_RESPONSE_STARTED)
        logger.debug("OpenAI stream start | model=%s", self._model)
        
        # response는 전체 텍스트가 아니라 AsyncIterator (토큰이 생성될 때마다 하나씩 도착하는 스트림)
        # 이 await 는 첫번째 토큰이 도착하기 전에 HTTP 연결만 맺고 바로 리턴한다.
        # gpt-5-mini 등 신형 모델은 temperature=1.0(기본값) 외 값을 거부하므로 기본값이면 생략한다.
        extra: dict[str, Any] = {}
        if self._temperature != 1.0:
            extra["temperature"] = self._temperature
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=context.get_messages(),  # type: ignore[arg-type]
            max_completion_tokens=self._max_tokens,
            stream=True,
            **extra,
        )

        # OpenAI 서버에서 토큰이 하나 생성될 때 마다 chunk 객체가 도착한다. 
        async for chunk in response: 
            token: str = chunk.choices[0].delta.content or ""  # None 이 오는 경우 빈문자열 처리
            if token:
                await self._bus.publish(EventType.LLM_RESPONSE_CHUNK, data=token)
                yield token         # 이 토큰을 호출자(_llm_worker 의 async for 에 전달한다.). 여기서 함수가 일시정지 되고, 다음 토큰이 오면 다시 재개

        await self._bus.publish(EventType.LLM_RESPONSE_DONE)
        logger.debug("OpenAI stream done")

    async def close(self) -> None:
        await self._client.close()


# ---------------------------------------------------------------------------
# ClaudeChatHandler
# ---------------------------------------------------------------------------

class ClaudeChatHandler:
    """LLM handler backed by the Anthropic async SDK.

    The system prompt is passed via the ``system`` parameter (not as a
    message), which is the idiomatic Anthropic API pattern.

    Args:
        cfg: App config — reads ``llm.claude_api_key``, ``llm.claude_model``,
             ``llm.max_tokens``, ``llm.temperature``.
        bus: Shared event bus.
    """

    def __init__(self, cfg: Config, bus: EventBus) -> None:
        import anthropic

        self._bus = bus
        self._model: str = cfg.get("llm.claude_model", "claude-sonnet-4-6")
        self._max_tokens: int = cfg.get("llm.max_tokens", 1024)
        self._temperature: float = cfg.get("llm.temperature", 0.7)
        self._client = anthropic.AsyncAnthropic(
            api_key=cfg.get("llm.claude_api_key", ""),
        )

    async def stream(self, context: ConversationContext) -> AsyncIterator[str]:  # type: ignore[override]
        """Yield response tokens from Anthropic messages API (streaming)."""
        await self._bus.publish(EventType.LLM_RESPONSE_STARTED)
        logger.debug("Claude stream start | model=%s", self._model)

        # Anthropic: separate system prompt + user/assistant messages
        messages = [m for m in context.get_messages() if m["role"] != "system"]

        async with self._client.messages.stream(
            model=self._model,
            system=context.system_prompt,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=self._max_tokens,
        ) as stream_cm:
            async for token in stream_cm.text_stream:
                if token:
                    await self._bus.publish(EventType.LLM_RESPONSE_CHUNK, data=token)
                    yield token

        await self._bus.publish(EventType.LLM_RESPONSE_DONE)
        logger.debug("Claude stream done")

    async def close(self) -> None:
        await self._client.close()


# ---------------------------------------------------------------------------
# CustomLLMChatHandler
# ---------------------------------------------------------------------------

class CustomLLMChatHandler:
    """LLM handler for Samsung Gauss OpenAPI via httpx.

    Uses the Gauss OpenAPI SSE streaming format::

        data: {"event_status": "CHUNK", "content": "token", "finish_reason": null, ...}
        data: {"event_status": "CHUNK", "content": "",      "finish_reason": "stop", ...}

    Authentication uses two custom headers:
      - ``x-generative-ai-client``: JWT client token
      - ``x-openapi-token``: Bearer API token

    Args:
        cfg: App config — reads ``llm.custom_url``, ``llm.custom_key``,
             ``llm.custom_client_token``, ``llm.custom_model_id``,
             ``llm.custom_user_email``, ``llm.maxtokens``, ``llm.temperature``.
        bus: Shared event bus.
    """

    def __init__(self, cfg: Config, bus: EventBus) -> None:
        import httpx

        self._bus = bus
        self._url: str = cfg.get("llm.custom_url", "")
        self._model_id: str = cfg.get("llm.custom_model_id", "")
        self._max_tokens: int = cfg.get("llm.max_tokens", 1024)
        self._temperature: float = cfg.get("llm.temperature", 0.7)

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-generative-ai-client": cfg.get("llm.custom_client_token", ""),
            "x-openapi-token": f"Bearer {cfg.get('llm.custom_key', '')}",
        }
        user_email: str = cfg.get("llm.custom_user_email", "")
        if user_email:
            headers["x-generative-ai-user-email"] = user_email

        self._client = httpx.AsyncClient(headers=headers, timeout=60.0)

    async def stream(self, context: ConversationContext) -> AsyncIterator[str]:  # type: ignore[override]
        """Yield response tokens from the Gauss OpenAPI SSE endpoint."""
        await self._bus.publish(EventType.LLM_RESPONSE_STARTED)
        logger.debug("Custom LLM stream start | url=%s model_id=%s", self._url, self._model_id)

        # Gauss API expects a flat list of user/assistant strings (no role dicts)
        contents = [
            m["content"] for m in context.get_messages() if m["role"] != "system"
        ]

        payload = {
            "modelIds": [self._model_id],
            "contents": contents,
            "isStream": True,
            "systemPrompt": context.system_prompt,
            "llmConfig": {
                "max_new_tokens": self._max_tokens,
                "temperature": self._temperature,
            },
        }

        async with self._client.stream("POST", self._url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if not data.strip():
                    continue
                try:
                    chunk = json.loads(data)
                    if chunk.get("event_status") == "CHUNK":
                        token: str = chunk.get("content", "")
                        if token:
                            await self._bus.publish(EventType.LLM_RESPONSE_CHUNK, data=token)
                            yield token
                        if chunk.get("finish_reason"):
                            break
                except json.JSONDecodeError:
                    continue

        await self._bus.publish(EventType.LLM_RESPONSE_DONE)
        logger.debug("Custom LLM stream done")

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_chat_handler(cfg: Config, bus: EventBus) -> ChatHandler:
    """Instantiate and return the configured LLM handler.

    Reads ``llm.provider`` from config (also set via ``LLM_PROVIDER`` env var).
    Supported values: ``"openai"``, ``"claude"``, ``"custom"``.

    Args:
        cfg: Application config.
        bus: Shared event bus.

    Returns:
        A :class:`ChatHandler`-conforming instance.

    Raises:
        ValueError: If the provider name is not recognised.
    """
    provider: str = (cfg.get("llm.provider", "custom") or "custom").lower().strip()

    if provider == "openai":
        logger.info("LLM provider: OpenAI (model=%s)", cfg.get("llm.openai_model", "?"))
        return OpenAIChatHandler(cfg, bus)

    if provider == "claude":
        logger.info("LLM provider: Claude (model=%s)", cfg.get("llm.claude_model", "?"))
        return ClaudeChatHandler(cfg, bus)

    if provider == "custom":
        logger.info("LLM provider: Custom (url=%s)", cfg.get("llm.custom_url", "?"))
        return CustomLLMChatHandler(cfg, bus)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Choose one of: 'openai', 'claude', 'custom'."
    )
