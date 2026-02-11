"""Chat handling with OpenAI API."""
import os
from typing import AsyncIterator

import openai

from src.utils.logger import get_logger
from .context import ConversationContext

logger = get_logger(__name__)


class OpenAIChatHandler:
    """Handles conversation with OpenAI's Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-5-nano",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ):
        """Initialize OpenAI chat handler.

        Args:
            model: OpenAI model name
            api_key: OpenAI API key (or from OPENAI_API_KEY env)
            max_tokens: Maximum tokens in response
            temperature: Response temperature
            system_prompt: System prompt for the assistant
        """
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt or "You are a helpful voice assistant."
        self._context = ConversationContext()
        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

        logger.info(f"OpenAI handler initialized (model: {self._model})")

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
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=self._build_messages(),
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

            assistant_message = response.choices[0].message.content

            self._context.add_assistant_message(assistant_message)
            logger.info(f"OpenAI response: {assistant_message[:100]}...")

            return assistant_message

        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
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
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=self._build_messages(),
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                stream=True,
            )

            full_response = []
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response.append(content)
                    yield content

            complete_text = "".join(full_response)
            self._context.add_assistant_message(complete_text)

        except Exception as e:
            logger.error(f"OpenAI streaming chat error: {e}")
            yield "I'm sorry, I encountered an error."

    def clear_context(self) -> None:
        """Clear conversation history."""
        self._context.clear()
