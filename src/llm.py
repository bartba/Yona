"""llm.py — LLM chat handlers for Samsung Gauss.

Provides:
  ChatHandler         — typing.Protocol for all LLM handlers
  ConversationContext — rolling in-session message history + system prompt
  ConversationHistory — weekly JSON summary storage (data/history/)
  OpenAIChatHandler   — OpenAI SDK streaming (gpt-4o-mini etc.)
  ClaudeChatHandler   — Anthropic SDK streaming (claude-*)
  CustomLLMChatHandler— httpx SSE streaming (OpenAI-compatible custom endpoint)
  ApiChatHandler      — httpx single POST (non-streaming; yields full response once)
  create_chat_handler — factory: reads LLM_PROVIDER and returns the right handler

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

    Tracks approximate token usage of the history and triggers compression
    when the budget is exceeded.  On compression the oldest third of
    messages is summarised by the LLM and replaced with a compact summary
    string stored in ``_summary``.

    Token estimate: ``len(content) // 3 + 4`` per message (conservative for
    mixed Korean/English; intentional over-estimate so we compress before
    the model's real context limit).

    Args:
        system_prompt:      Text injected as the ``system`` message.
        max_context_tokens: Compress when history token estimate exceeds this.
    """

    def __init__(self, system_prompt: str, max_context_tokens: int = 3000) -> None:
        self._system_prompt = system_prompt
        self._max_tokens = max_context_tokens
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
    def history_tokens(self) -> int:
        """Estimated token count of current history (excludes system prompt)."""
        return sum(len(m["content"]) // 3 + 4 for m in self._messages)

    @property
    def needs_compression(self) -> bool:
        """True when history token estimate meets or exceeds the budget."""
        return self.history_tokens >= self._max_tokens

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
        """Return (old_summary, oldest_third_messages) for summarisation.

        The caller should feed these to the LLM to produce a new summary,
        then call :meth:`compress` with the result.
        """
        third = len(self._messages) // 3
        third = third - (third % 2)  # round down to pair boundary
        return self._summary, list(self._messages[:third])

    def compress(self, summary: str) -> None:
        """Replace the oldest third of messages with *summary*."""
        third = len(self._messages) // 3
        third = third - (third % 2)
        self._summary = summary
        self._messages = self._messages[third:]
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
        """Hard safety net — discard oldest messages until under the ceiling,
        always leaving a leading 'user' turn (required by Claude API)."""
        ceiling = int(self._max_tokens * 1.1)
        while self.history_tokens > ceiling and len(self._messages) >= 2:
            self._messages = self._messages[2:]
        while self._messages and self._messages[0]["role"] != "user":
            self._messages = self._messages[1:]


# ---------------------------------------------------------------------------
# ConversationHistory
# ---------------------------------------------------------------------------

class ConversationHistory:
    """Persistent daily JSONL conversation history.

    Each calendar day gets its own file ``YYYY-MM-DD.jsonl`` in *storage_dir*.
    Each line is a self-contained JSON object (one turn)::

        {"user": "...", "assistant": "...", "ts": "2026-03-12T09:23:11"}

    JSONL is append-only: each turn costs a single ``fh.write`` with no
    read-back, giving O(1) I/O per turn regardless of session length and
    eliminating eMMC write amplification from full-file rewrites.

    Files older than *max_days* are deleted at startup via :meth:`purge_old`.

    Args:
        storage_dir: Directory path for JSONL files (created if missing).
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
        """Persist a completed turn to today's JSONL file (O(1) append).

        Does not trigger pruning — call :meth:`purge_old` once at app startup.
        """
        fpath = self._day_file(self._day_key())
        line = json.dumps(
            {"user": user, "assistant": assistant, "ts": datetime.now().isoformat()},
            ensure_ascii=False,
        )
        with fpath.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def get_recent_turns(self, max_days: int = 7) -> list[dict[str, Any]]:
        """Return all turns from the last *max_days* days, oldest first.

        Malformed lines are silently skipped.
        """
        cutoff = datetime.now() - timedelta(days=max_days)
        results: list[dict[str, Any]] = []

        for fpath in sorted(self._dir.glob("*.jsonl")):
            try:
                with fpath.open(encoding="utf-8") as fh:
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            turn = json.loads(raw)
                            ts = datetime.fromisoformat(turn.get("ts", "1970-01-01"))
                            if ts >= cutoff:
                                results.append(turn)
                        except (json.JSONDecodeError, ValueError):
                            continue
            except OSError:
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
        return self._dir / f"{day_key}.jsonl"

    def purge_old(self) -> None:
        """Delete day files older than *max_days*.

        Call this once at app startup — not on every turn.
        """
        cutoff = datetime.now() - timedelta(days=self._max_days)
        for fpath in self._dir.glob("*.jsonl"):
            try:
                file_date = datetime.strptime(fpath.stem, "%Y-%m-%d")
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
        import httpx as _httpx

        self._client = openai.AsyncOpenAI(
            api_key=cfg.get("llm.openai_api_key", ""),
            timeout=_httpx.Timeout(timeout=30.0, connect=10.0),
        )

    async def stream(self, context: ConversationContext) -> AsyncIterator[str]:  # type: ignore[override]
        """Yield response tokens from OpenAI chat completions (streaming)."""
        await self._bus.publish(EventType.LLM_RESPONSE_STARTED)
        logger.debug("OpenAI stream start | model=%s", self._model)
        
        # Newer OpenAI models reject temperature != 1.0; omit unless overridden.
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

        async for chunk in response:
            token: str = chunk.choices[0].delta.content or ""
            if token:
                await self._bus.publish(EventType.LLM_RESPONSE_CHUNK, data=token)
                yield token

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
        # Claude has no temperature=1.0 restriction; falls back to shared llm.temperature.
        self._temperature: float = cfg.get("llm.claude_temperature", cfg.get("llm.temperature", 0.7))
        import httpx as _httpx

        self._client = anthropic.AsyncAnthropic(
            api_key=cfg.get("llm.claude_api_key", ""),
            timeout=_httpx.Timeout(timeout=30.0, connect=10.0),
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
            temperature=self._temperature,
        ) as stream_cm:
            async for token in stream_cm.text_stream:
                if token:
                    await self._bus.publish(EventType.LLM_RESPONSE_CHUNK, data=token)
                    yield token

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
        # Falls back to shared llm.temperature if custom_temperature is not set.
        self._temperature: float = cfg.get("llm.custom_temperature", cfg.get("llm.temperature", 0.7))

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-generative-ai-client": cfg.get("llm.custom_client_token", ""),
            "x-openapi-token": cfg.get("llm.custom_key", ""),
        }
        user_email: str = cfg.get("llm.custom_user_email", "")
        if user_email:
            headers["x-generative-ai-user-email"] = user_email

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout=30.0, connect=10.0),
            trust_env=False,
        )

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

        logger.debug("Custom LLM stream done")

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# ApiChatHandler
# ---------------------------------------------------------------------------

class ApiChatHandler:
    """Non-streaming HTTP handler: POST once, yield full response as one chunk.

    LLM 대신 임의의 HTTP API에 단발 POST를 날려 응답 전체를 받은 후,
    `stream()`이 그 텍스트를 **한 번만** yield한다. 파이프라인(`pipeline.py`)은
    스트리밍 여부에 독립적이므로 `PhraseAccumulator`가 문장 경계로 자동 분할해
    TTS로 공급한다.

    요청 payload는 단일 키 JSON::

        {"<request_key>": "<STT 결과 텍스트>"}

    응답은 JSON이며 ``response_path`` dot path로 텍스트를 추출::

        response_path="response"     → data["response"]
        response_path="result.text"  → data["result"]["text"]

    Args:
        cfg: App config — reads ``llm.api.url``, ``llm.api.auth_header``,
             ``llm.api.request_key``, ``llm.api.response_path``, ``llm.api.timeout``.
        bus: Shared event bus.
    """

    def __init__(self, cfg: Config, bus: EventBus) -> None:
        import httpx

        self._bus = bus
        self._url: str = cfg.get("llm.api.url", "")
        self._request_key: str = cfg.get("llm.api.request_key", "query")
        self._response_path: str = cfg.get("llm.api.response_path", "response")
        timeout: float = float(cfg.get("llm.api.timeout", 30))

        headers: dict[str, str] = {"Content-Type": "application/json"}
        auth_header: str = cfg.get("llm.api.auth_header", "")
        if auth_header:
            headers["Authorization"] = auth_header

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout=timeout, connect=10.0),
            trust_env=False,
        )

    async def stream(self, context: ConversationContext) -> AsyncIterator[str]:  # type: ignore[override]
        """POST 최신 user 발화 → 응답 전체 텍스트 1회 yield."""
        await self._bus.publish(EventType.LLM_RESPONSE_STARTED)
        logger.debug("API stream start | url=%s", self._url)

        # Send only the latest user message; skip compression summary turns.
        user_text = ""
        for msg in reversed(context.get_messages()):
            if msg["role"] == "user" and not msg["content"].startswith("[이전 대화 요약]"):
                user_text = msg["content"]
                break

        payload = {self._request_key: user_text}

        response = await self._client.post(self._url, json=payload)
        response.raise_for_status()
        data: Any = response.json()

        # Walk the dot-separated response_path to extract the text field.
        node: Any = data
        for part in self._response_path.split("."):
            if not isinstance(node, dict):
                node = ""
                break
            node = node.get(part, "")
        text = str(node or "")

        if text:
            await self._bus.publish(EventType.LLM_RESPONSE_CHUNK, data=text)
            yield text

        logger.debug("API stream done | len=%d", len(text))

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_chat_handler(cfg: Config, bus: EventBus) -> ChatHandler:
    """Instantiate and return the configured LLM handler.

    Reads ``llm.provider`` from config (also set via ``LLM_PROVIDER`` env var).
    Supported values: ``"openai"``, ``"claude"``, ``"custom"``, ``"api"``.

    ``"api"`` selects :class:`ApiChatHandler` which bypasses the LLM entirely and
    forwards the STT text to an arbitrary HTTP endpoint, returning the full
    response as a single TTS-ready chunk.

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

    if provider == "api":
        logger.info("LLM provider: API (url=%s)", cfg.get("llm.api.url", "?"))
        return ApiChatHandler(cfg, bus)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Choose one of: 'openai', 'claude', 'custom', 'api'."
    )
