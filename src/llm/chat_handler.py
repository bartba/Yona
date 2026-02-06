"""Chat handling with Gemini API."""
import os
from typing import AsyncIterator

from src.utils.logger import get_logger
from .context import ConversationContext

logger = get_logger(__name__)


class ChatHandler:
    """Handles conversation with Gemini API."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ):
        """Initialize chat handler.

        Args:
            model: Gemini model to use
            api_key: Google API key (or from GOOGLE_API_KEY env)
            max_tokens: Maximum tokens in response
            temperature: Response temperature
            system_prompt: System prompt for the assistant
        """
        self._model_name = model
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt or "You are a helpful voice assistant."

        self._model = None
        self._chat = None
        self._context = ConversationContext()

        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Gemini client."""
        if not self._api_key:
            logger.warning("No GOOGLE_API_KEY found - chat will not work")
            return

        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)

            # Configure generation settings
            generation_config = genai.GenerationConfig(
                max_output_tokens=self._max_tokens,
                temperature=self._temperature,
            )

            self._model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config=generation_config,
                system_instruction=self._system_prompt,
            )

            # Start a chat session
            self._chat = self._model.start_chat(history=[])

            logger.info(f"Gemini client initialized (model: {self._model_name})")

        except ImportError as e:
            logger.error(f"Failed to import google-generativeai: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    @property
    def context(self) -> ConversationContext:
        """Get conversation context."""
        return self._context

    def _sync_history_to_chat(self) -> None:
        """Sync conversation context to Gemini chat history."""
        if not self._model:
            return

        # Convert context messages to Gemini format
        history = []
        for msg in self._context.messages:
            role = "user" if msg.role == "user" else "model"
            history.append({"role": role, "parts": [msg.content]})

        # Restart chat with updated history
        self._chat = self._model.start_chat(history=history)

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
        if not self._model:
            return "I'm sorry, the chat service is not available."

        # Add user message to context
        self._context.add_user_message(user_message, language)

        try:
            import asyncio

            # Sync history and send message
            self._sync_history_to_chat()

            # Remove the last user message from chat history since we'll send it now
            if self._chat.history:
                self._chat.history.pop()

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._chat.send_message(user_message),
            )

            assistant_message = response.text

            # Add to context
            self._context.add_assistant_message(assistant_message)

            logger.info(f"Chat response: {assistant_message[:100]}...")

            return assistant_message

        except Exception as e:
            logger.error(f"Chat error: {e}")
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
        if not self._model:
            yield "I'm sorry, the chat service is not available."
            return

        self._context.add_user_message(user_message, language)

        try:
            import asyncio

            # Sync history
            self._sync_history_to_chat()

            # Remove the last user message since we'll send it now
            if self._chat.history:
                self._chat.history.pop()

            loop = asyncio.get_event_loop()

            # Create streaming response
            response = await loop.run_in_executor(
                None,
                lambda: self._chat.send_message(user_message, stream=True),
            )

            full_response = []

            # Process stream
            for chunk in response:
                if chunk.text:
                    full_response.append(chunk.text)
                    yield chunk.text

            # Add complete response to context
            complete_text = "".join(full_response)
            self._context.add_assistant_message(complete_text)

        except Exception as e:
            logger.error(f"Streaming chat error: {e}")
            yield "I'm sorry, I encountered an error."

    def clear_context(self) -> None:
        """Clear conversation history."""
        self._context.clear()
        # Restart chat session with empty history
        if self._model:
            self._chat = self._model.start_chat(history=[])
