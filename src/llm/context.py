"""Conversation context management."""
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """A single conversation message."""
    role: str  # 'user' or 'assistant'
    content: str


@dataclass
class ConversationContext:
    """Manages conversation history."""
    messages: list[Message] = field(default_factory=list)
    max_turns: int = 20
    last_language: str = "ko"

    def add_user_message(self, content: str, language: str | None = None) -> None:
        """Add a user message to history.

        Args:
            content: Message content
            language: Detected language of the message
        """
        self.messages.append(Message(role="user", content=content))
        if language:
            self.last_language = language
        self._trim_history()
        logger.debug(f"Added user message ({len(self.messages)} total)")

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to history.

        Args:
            content: Message content
        """
        self.messages.append(Message(role="assistant", content=content))
        self._trim_history()
        logger.debug(f"Added assistant message ({len(self.messages)} total)")

    def _trim_history(self) -> None:
        """Trim history to max_turns (keeping pairs)."""
        max_messages = self.max_turns * 2

        if len(self.messages) > max_messages:
            # Remove oldest messages (keep recent pairs)
            remove_count = len(self.messages) - max_messages
            self.messages = self.messages[remove_count:]
            logger.debug(f"Trimmed history to {len(self.messages)} messages")

    def get_messages_for_api(self) -> list[dict]:
        """Get messages in API format.

        Returns:
            List of message dicts for Claude API
        """
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        logger.info("Conversation history cleared")

    @property
    def turn_count(self) -> int:
        """Number of conversation turns (user+assistant pairs)."""
        return len(self.messages) // 2

    @property
    def is_empty(self) -> bool:
        """Whether the context is empty."""
        return len(self.messages) == 0
