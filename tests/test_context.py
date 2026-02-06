"""Tests for ConversationContext."""
import pytest

from src.llm.context import ConversationContext, Message


class TestConversationContext:
    """Tests for ConversationContext class."""

    def test_init(self):
        """Test context initialization."""
        context = ConversationContext()
        assert context.is_empty
        assert context.turn_count == 0

    def test_add_user_message(self):
        """Test adding user messages."""
        context = ConversationContext()
        context.add_user_message("Hello", "en")

        assert not context.is_empty
        assert len(context.messages) == 1
        assert context.messages[0].role == "user"
        assert context.messages[0].content == "Hello"
        assert context.last_language == "en"

    def test_add_assistant_message(self):
        """Test adding assistant messages."""
        context = ConversationContext()
        context.add_user_message("Hello")
        context.add_assistant_message("Hi there!")

        assert len(context.messages) == 2
        assert context.messages[1].role == "assistant"
        assert context.turn_count == 1

    def test_get_messages_for_api(self):
        """Test getting messages in API format."""
        context = ConversationContext()
        context.add_user_message("Hello")
        context.add_assistant_message("Hi!")

        api_messages = context.get_messages_for_api()

        assert len(api_messages) == 2
        assert api_messages[0] == {"role": "user", "content": "Hello"}
        assert api_messages[1] == {"role": "assistant", "content": "Hi!"}

    def test_trim_history(self):
        """Test history trimming."""
        context = ConversationContext(max_turns=2)

        # Add 3 turns (6 messages)
        for i in range(3):
            context.add_user_message(f"User {i}")
            context.add_assistant_message(f"Assistant {i}")

        # Should only keep last 2 turns (4 messages)
        assert len(context.messages) == 4
        assert context.messages[0].content == "User 1"

    def test_clear(self):
        """Test clearing context."""
        context = ConversationContext()
        context.add_user_message("Hello")
        context.add_assistant_message("Hi!")

        context.clear()

        assert context.is_empty
        assert context.turn_count == 0

    def test_language_tracking(self):
        """Test language tracking."""
        context = ConversationContext()

        context.add_user_message("안녕하세요", "ko")
        assert context.last_language == "ko"

        context.add_user_message("Hello", "en")
        assert context.last_language == "en"
