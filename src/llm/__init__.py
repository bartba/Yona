"""LLM modules."""
import os

from .base import ChatHandler
from .context import ConversationContext


def create_chat_handler(
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    system_prompt: str | None = None,
) -> ChatHandler:
    """Create the appropriate chat handler based on LLM_PROVIDER env var.

    Args:
        model: Model name (provider-specific). If None, uses provider default.
        max_tokens: Maximum tokens in response.
        temperature: Response temperature.
        system_prompt: System prompt for the assistant.

    Returns:
        A ChatHandler instance for the configured provider.

    Raises:
        ValueError: If LLM_PROVIDER is not recognized.
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower().strip()

    if provider == "gemini":
        from .gemini_handler import GeminiChatHandler

        return GeminiChatHandler(
            model=model or "gemini-2.5-flash",
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
    elif provider == "ollama":
        from .ollama_handler import OllamaChatHandler

        return OllamaChatHandler(
            model=model or os.environ.get("OLLAMA_MODEL", "gemma3:4b"),
            base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Supported: 'gemini', 'ollama'"
        )


__all__ = ["ChatHandler", "ConversationContext", "create_chat_handler"]
