"""Chat handling with local Ollama instance."""
import json
import os
from typing import AsyncIterator

import httpx

from src.utils.logger import get_logger
from .context import ConversationContext

logger = get_logger(__name__)


class OllamaChatHandler:
    """Handles conversation with a local Ollama instance."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ):
        """Initialize Ollama chat handler.

        Args:
            model: Ollama model name
            base_url: Ollama server URL (or from OLLAMA_URL env)
            max_tokens: Maximum tokens in response
            temperature: Response temperature
            system_prompt: System prompt for the assistant
        """
        self._model = model
        self._base_url = base_url or os.environ.get(
            "OLLAMA_URL", "http://localhost:11434"
        )
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt or "You are a helpful voice assistant."
        self._context = ConversationContext()
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

        logger.info(
            f"Ollama handler initialized (model: {self._model}, "
            f"url: {self._base_url})"
        )

    @property
    def context(self) -> ConversationContext:
        """Get conversation context."""
        return self._context

    def _build_messages(self) -> list[dict]:
        """Build messages array from system prompt + conversation history."""
        messages = [{"role": "system", "content": self._system_prompt}]
        for msg in self._context.messages:
            messages.append({"role": msg.role, "content": msg.content})
        return messages

    async def chat(
        self,
        user_message: str,
        language: str = "ko",
    ) -> str:
        """Send a message and get a response.

        Args:
            user_message: User's message
            language: Detected language of the message

        Returns:
            Assistant's response
        """
        self._context.add_user_message(user_message, language)

        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": self._model,
                    "messages": self._build_messages(),
                    "stream": False,
                    "options": {
                        "num_predict": self._max_tokens,
                        "temperature": self._temperature,
                    },
                },
            )
            response.raise_for_status()

            data = response.json()
            assistant_message = data["message"]["content"]

            self._context.add_assistant_message(assistant_message)
            logger.info(f"Ollama response: {assistant_message[:100]}...")

            return assistant_message

        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return "I'm sorry, I encountered an error processing your request."

    async def chat_streaming(
        self,
        user_message: str,
        language: str = "ko",
    ) -> AsyncIterator[str]:
        """Send a message and stream the response.

        Args:
            user_message: User's message
            language: Detected language

        Yields:
            Response text chunks
        """
        self._context.add_user_message(user_message, language)

        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": self._model,
                    "messages": self._build_messages(),
                    "stream": True,
                    "options": {
                        "num_predict": self._max_tokens,
                        "temperature": self._temperature,
                    },
                },
            ) as response:
                response.raise_for_status()

                full_response = []
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            full_response.append(content)
                            yield content

                complete_text = "".join(full_response)
                self._context.add_assistant_message(complete_text)

        except Exception as e:
            logger.error(f"Ollama streaming chat error: {e}")
            yield "I'm sorry, I encountered an error."

    def clear_context(self) -> None:
        """Clear conversation history."""
        self._context.clear()
