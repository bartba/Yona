"""Tests for src/events.py

Run with:
    pytest tests/test_events.py -v

All tests are purely in-memory; no hardware or external services needed.
"""

from __future__ import annotations

import asyncio

import pytest

from src.events import Event, EventBus, EventType


# ---------------------------------------------------------------------------
# EventType enum
# ---------------------------------------------------------------------------

class TestEventType:
    def test_has_exactly_18_members(self):
        assert len(EventType) == 19

    def test_all_members_are_unique(self):
        values = [e.value for e in EventType]
        assert len(values) == len(set(values))

    def test_expected_names_present(self):
        names = {e.name for e in EventType}
        expected = {
            "WAKE_WORD_DETECTED",
            "SPEECH_STARTED",
            "SPEECH_ENDED",
            "BARGE_IN_DETECTED",
            "TRANSCRIPTION_READY",
            "LLM_RESPONSE_STARTED",
            "LLM_RESPONSE_CHUNK",
            "LLM_RESPONSE_DONE",
            "PHRASE_READY",
            "PHRASE_PLAYING",
            "AUDIO_CHUNK_READY",
            "PLAYBACK_STARTED",
            "PLAYBACK_DONE",
            "TIMEOUT_CHECK",
            "TIMEOUT_FINAL",
            "GOODBYE_DETECTED",
            "STATE_CHANGED",
            "ERROR",
            "SHUTDOWN",
        }
        assert expected == names


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

class TestEvent:
    def test_type_stored(self):
        e = Event(type=EventType.SHUTDOWN)
        assert e.type is EventType.SHUTDOWN

    def test_data_defaults_to_none(self):
        e = Event(type=EventType.SHUTDOWN)
        assert e.data is None

    def test_data_stored(self):
        e = Event(type=EventType.TRANSCRIPTION_READY, data="안녕하세요")
        assert e.data == "안녕하세요"

    def test_timestamp_is_float(self):
        e = Event(type=EventType.SHUTDOWN)
        assert isinstance(e.timestamp, float)
        assert e.timestamp > 0

    def test_timestamp_auto_set(self):
        import time
        before = time.time()
        e = Event(type=EventType.SHUTDOWN)
        after = time.time()
        assert before <= e.timestamp <= after


# ---------------------------------------------------------------------------
# EventBus — subscribe / publish
# ---------------------------------------------------------------------------

class TestEventBusSubscribePublish:
    @pytest.mark.asyncio
    async def test_subscribe_returns_queue(self):
        bus = EventBus()
        q = bus.subscribe(EventType.WAKE_WORD_DETECTED)
        assert isinstance(q, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_publish_delivers_event(self):
        bus = EventBus()
        q = bus.subscribe(EventType.WAKE_WORD_DETECTED)
        await bus.publish(EventType.WAKE_WORD_DETECTED)
        event = q.get_nowait()
        assert event.type is EventType.WAKE_WORD_DETECTED

    @pytest.mark.asyncio
    async def test_publish_with_data(self):
        bus = EventBus()
        q = bus.subscribe(EventType.TRANSCRIPTION_READY)
        await bus.publish(EventType.TRANSCRIPTION_READY, data="hello")
        event = q.get_nowait()
        assert event.data == "hello"

    @pytest.mark.asyncio
    async def test_event_type_isolation(self):
        """Events only reach subscribers of the matching type."""
        bus = EventBus()
        q_wake = bus.subscribe(EventType.WAKE_WORD_DETECTED)
        q_stt = bus.subscribe(EventType.TRANSCRIPTION_READY)

        await bus.publish(EventType.WAKE_WORD_DETECTED)

        assert not q_wake.empty()
        assert q_stt.empty()

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_type(self):
        bus = EventBus()
        q1 = bus.subscribe(EventType.SPEECH_STARTED)
        q2 = bus.subscribe(EventType.SPEECH_STARTED)
        await bus.publish(EventType.SPEECH_STARTED)
        assert not q1.empty()
        assert not q2.empty()

    @pytest.mark.asyncio
    async def test_publish_multiple_events_ordered(self):
        bus = EventBus()
        q = bus.subscribe(EventType.LLM_RESPONSE_CHUNK)
        await bus.publish(EventType.LLM_RESPONSE_CHUNK, data="tok1")
        await bus.publish(EventType.LLM_RESPONSE_CHUNK, data="tok2")
        await bus.publish(EventType.LLM_RESPONSE_CHUNK, data="tok3")
        results = [q.get_nowait().data for _ in range(3)]
        assert results == ["tok1", "tok2", "tok3"]

    @pytest.mark.asyncio
    async def test_no_subscribers_publish_is_noop(self):
        bus = EventBus()
        # Should not raise even with zero subscribers.
        await bus.publish(EventType.SHUTDOWN)

    @pytest.mark.asyncio
    async def test_await_get_after_publish(self):
        """Subscriber can await q.get() and receive the event."""
        bus = EventBus()
        q = bus.subscribe(EventType.PLAYBACK_DONE)
        await bus.publish(EventType.PLAYBACK_DONE)
        event = await asyncio.wait_for(q.get(), timeout=1.0)
        assert event.type is EventType.PLAYBACK_DONE

    @pytest.mark.asyncio
    async def test_concurrent_publish_and_receive(self):
        """Publish from a concurrent task is received correctly."""
        bus = EventBus()
        q = bus.subscribe(EventType.BARGE_IN_DETECTED)

        async def publisher():
            await asyncio.sleep(0)
            await bus.publish(EventType.BARGE_IN_DETECTED, data=True)

        asyncio.create_task(publisher())
        event = await asyncio.wait_for(q.get(), timeout=1.0)
        assert event.type is EventType.BARGE_IN_DETECTED
        assert event.data is True


# ---------------------------------------------------------------------------
# EventBus — unsubscribe
# ---------------------------------------------------------------------------

class TestEventBusUnsubscribe:
    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        q = bus.subscribe(EventType.SPEECH_ENDED)
        bus.unsubscribe(EventType.SPEECH_ENDED, q)
        await bus.publish(EventType.SPEECH_ENDED)
        assert q.empty()

    @pytest.mark.asyncio
    async def test_unsubscribe_only_removes_target_queue(self):
        bus = EventBus()
        q1 = bus.subscribe(EventType.SPEECH_ENDED)
        q2 = bus.subscribe(EventType.SPEECH_ENDED)
        bus.unsubscribe(EventType.SPEECH_ENDED, q1)
        await bus.publish(EventType.SPEECH_ENDED)
        assert q1.empty()
        assert not q2.empty()

    def test_unsubscribe_unknown_queue_is_silent(self):
        bus = EventBus()
        q = asyncio.Queue()
        # Should not raise even though q was never subscribed.
        bus.unsubscribe(EventType.SPEECH_ENDED, q)

    def test_unsubscribe_unknown_event_type_is_silent(self):
        bus = EventBus()
        q = asyncio.Queue()
        # Type has no subscribers at all — must not raise.
        bus.unsubscribe(EventType.ERROR, q)

    @pytest.mark.asyncio
    async def test_resubscribe_after_unsubscribe(self):
        bus = EventBus()
        q = bus.subscribe(EventType.TIMEOUT_CHECK)
        bus.unsubscribe(EventType.TIMEOUT_CHECK, q)

        q2 = bus.subscribe(EventType.TIMEOUT_CHECK)
        await bus.publish(EventType.TIMEOUT_CHECK)
        assert q.empty()
        assert not q2.empty()


# ---------------------------------------------------------------------------
# EventBus — publish_nowait (sync / sounddevice callback path)
# ---------------------------------------------------------------------------

class TestEventBusPublishNowait:
    @pytest.mark.asyncio
    async def test_publish_nowait_delivers_event(self):
        bus = EventBus()
        q = bus.subscribe(EventType.SPEECH_STARTED)
        bus.publish_nowait(EventType.SPEECH_STARTED)
        assert not q.empty()
        event = q.get_nowait()
        assert event.type is EventType.SPEECH_STARTED

    @pytest.mark.asyncio
    async def test_publish_nowait_with_data(self):
        bus = EventBus()
        q = bus.subscribe(EventType.STATE_CHANGED)
        bus.publish_nowait(EventType.STATE_CHANGED, data="LISTENING")
        event = q.get_nowait()
        assert event.data == "LISTENING"

    @pytest.mark.asyncio
    async def test_publish_nowait_drops_on_full_queue(self):
        """A full subscriber queue causes the event to be silently dropped."""
        bus = EventBus()
        # Tiny queue that fills immediately.
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=1)
        bus._subscribers[EventType.SPEECH_STARTED].append(q)
        bus.publish_nowait(EventType.SPEECH_STARTED)  # fills queue
        bus.publish_nowait(EventType.SPEECH_STARTED)  # should be dropped, not raise
        assert q.qsize() == 1

    @pytest.mark.asyncio
    async def test_publish_nowait_no_subscribers_is_noop(self):
        bus = EventBus()
        # Must not raise.
        bus.publish_nowait(EventType.SHUTDOWN)

    @pytest.mark.asyncio
    async def test_publish_nowait_multiple_subscribers(self):
        bus = EventBus()
        q1 = bus.subscribe(EventType.WAKE_WORD_DETECTED)
        q2 = bus.subscribe(EventType.WAKE_WORD_DETECTED)
        bus.publish_nowait(EventType.WAKE_WORD_DETECTED)
        assert not q1.empty()
        assert not q2.empty()
