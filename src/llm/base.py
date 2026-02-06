"""Base interface for LLM chat handlers."""
from typing import AsyncIterator, Protocol, runtime_checkable

from .context import ConversationContext


@runtime_checkable
class ChatHandler(Protocol):
    """Protocol that all LLM chat handlers must satisfy."""

    @property
    def context(self) -> ConversationContext:
        """Get conversation context."""
        ...

    async def chat(self, user_message: str, language: str = "ko") -> str:
        """Send a message and get a response."""
        ...

    async def chat_streaming(
        self, user_message: str, language: str = "ko"
    ) -> AsyncIterator[str]:
        """Send a message and stream the response."""
        ...

    def clear_context(self) -> None:
        """Clear conversation history."""
        ...
