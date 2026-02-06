"""Tests for EventBus."""
import asyncio
import pytest

from src.core.event_bus import Event, EventBus, EventType


@pytest.fixture
def event_bus():
    """Create an event bus for testing."""
    return EventBus()


class TestEventBus:
    """Tests for EventBus class."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus):
        """Test basic subscribe and publish."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe(EventType.WAKE_WORD_DETECTED, handler)
        await event_bus.start()

        await event_bus.publish(Event(type=EventType.WAKE_WORD_DETECTED))
        await asyncio.sleep(0.2)  # Allow processing

        await event_bus.stop()

        assert len(received_events) == 1
        assert received_events[0].type == EventType.WAKE_WORD_DETECTED

    @pytest.mark.asyncio
    async def test_event_with_data(self, event_bus):
        """Test events with data."""
        received_data = {}

        async def handler(event):
            received_data.update(event.data)

        event_bus.subscribe(EventType.TRANSCRIPTION_COMPLETE, handler)
        await event_bus.start()

        await event_bus.publish(Event(
            type=EventType.TRANSCRIPTION_COMPLETE,
            data={"text": "Hello", "language": "en"},
        ))
        await asyncio.sleep(0.2)

        await event_bus.stop()

        assert received_data["text"] == "Hello"
        assert received_data["language"] == "en"

    @pytest.mark.asyncio
    async def test_multiple_handlers(self, event_bus):
        """Test multiple handlers for same event."""
        count = {"value": 0}

        async def handler1(event):
            count["value"] += 1

        async def handler2(event):
            count["value"] += 10

        event_bus.subscribe(EventType.SPEECH_END, handler1)
        event_bus.subscribe(EventType.SPEECH_END, handler2)
        await event_bus.start()

        await event_bus.publish(Event(type=EventType.SPEECH_END))
        await asyncio.sleep(0.2)

        await event_bus.stop()

        assert count["value"] == 11

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        count = {"value": 0}

        async def handler(event):
            count["value"] += 1

        event_bus.subscribe(EventType.PLAYBACK_DONE, handler)
        event_bus.unsubscribe(EventType.PLAYBACK_DONE, handler)
        await event_bus.start()

        await event_bus.publish(Event(type=EventType.PLAYBACK_DONE))
        await asyncio.sleep(0.2)

        await event_bus.stop()

        assert count["value"] == 0

    @pytest.mark.asyncio
    async def test_no_handlers(self, event_bus):
        """Test publishing without handlers doesn't crash."""
        await event_bus.start()
        await event_bus.publish(Event(type=EventType.ERROR))
        await asyncio.sleep(0.1)
        await event_bus.stop()

    def test_sync_publish(self, event_bus):
        """Test synchronous publish."""
        # Just verify it doesn't crash
        event_bus.publish_sync(Event(type=EventType.AUDIO_CHUNK))

    def test_event_repr(self):
        """Test Event string representation."""
        event = Event(type=EventType.WAKE_WORD_DETECTED, data={"score": 0.9})
        repr_str = repr(event)
        assert "WAKE_WORD_DETECTED" in repr_str
