"""Tests for OpenAIChatHandler."""
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Inject a mock openai module before importing the handler,
# since the openai package may not be installed.
mock_openai = MagicMock()
sys.modules["openai"] = mock_openai


from src.llm.openai_handler import OpenAIChatHandler  # noqa: E402


class TestOpenAIChatHandler:
    """Tests for OpenAIChatHandler class."""

    def setup_method(self):
        """Reset the mock before each test."""
        mock_openai.reset_mock()

    def test_init_defaults(self):
        """Test initialization with default values."""
        handler = OpenAIChatHandler(api_key="test-key")

        assert handler._model == "gpt-5-nano"
        assert handler._max_tokens == 1024
        assert handler._temperature == 0.7
        assert handler._system_prompt == "You are a helpful voice assistant."
        assert handler.context.is_empty
        mock_openai.AsyncOpenAI.assert_called_once()

    def test_init_custom(self):
        """Test initialization with custom values."""
        handler = OpenAIChatHandler(
            model="gpt-4o",
            api_key="test-key",
            max_tokens=2048,
            temperature=0.5,
            system_prompt="Custom prompt",
        )

        assert handler._model == "gpt-4o"
        assert handler._max_tokens == 2048
        assert handler._temperature == 0.5
        assert handler._system_prompt == "Custom prompt"

    def test_build_messages(self):
        """Test message building with system prompt and context."""
        handler = OpenAIChatHandler(api_key="test-key", system_prompt="Be helpful")
        handler._context.add_user_message("Hello", "en")
        handler._context.add_assistant_message("Hi!")

        messages = handler._build_messages()

        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "Be helpful"}
        assert messages[1] == {"role": "user", "content": "Hello"}
        assert messages[2] == {"role": "assistant", "content": "Hi!"}

    @pytest.mark.asyncio
    async def test_chat(self):
        """Test non-streaming chat call."""
        handler = OpenAIChatHandler(api_key="test-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from OpenAI!"
        handler._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await handler.chat("Hi there", "en")

        assert result == "Hello from OpenAI!"
        assert handler.context.messages[-1].content == "Hello from OpenAI!"
        assert handler.context.messages[-1].role == "assistant"
        handler._client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_adds_user_message(self):
        """Test that chat adds user message to context."""
        handler = OpenAIChatHandler(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        handler._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await handler.chat("Test message", "ko")

        assert handler.context.messages[0].content == "Test message"
        assert handler.context.messages[0].role == "user"
        assert handler.context.last_language == "ko"

    @pytest.mark.asyncio
    async def test_chat_error_returns_friendly_message(self):
        """Test that errors return a user-friendly string."""
        handler = OpenAIChatHandler(api_key="test-key")
        handler._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error")
        )

        result = await handler.chat("Hello", "en")

        assert "sorry" in result.lower()

    @pytest.mark.asyncio
    async def test_chat_streaming(self):
        """Test streaming chat call."""
        handler = OpenAIChatHandler(api_key="test-key")

        # Build mock streaming chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None  # final chunk

        # Create async iterator for the stream
        async def mock_stream():
            for chunk in [chunk1, chunk2, chunk3]:
                yield chunk

        handler._client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        chunks = []
        async for chunk in handler.chat_streaming("Hi", "en"):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]
        assert handler.context.messages[-1].content == "Hello world"
        assert handler.context.messages[-1].role == "assistant"

    @pytest.mark.asyncio
    async def test_chat_streaming_error(self):
        """Test streaming error returns friendly message."""
        handler = OpenAIChatHandler(api_key="test-key")
        handler._client.chat.completions.create = AsyncMock(
            side_effect=Exception("Stream error")
        )

        chunks = []
        async for chunk in handler.chat_streaming("Hello", "en"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "sorry" in chunks[0].lower()

    def test_clear_context(self):
        """Test clearing conversation context."""
        handler = OpenAIChatHandler(api_key="test-key")
        handler._context.add_user_message("Hello")
        handler._context.add_assistant_message("Hi!")

        assert not handler.context.is_empty

        handler.clear_context()

        assert handler.context.is_empty
