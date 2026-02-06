"""Event bus for inter-component communication."""
import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Types of events in the system."""
    # Wake word events
    WAKE_WORD_DETECTED = auto()

    # Audio events
    SPEECH_START = auto()
    SPEECH_END = auto()
    AUDIO_CHUNK = auto()

    # Processing events
    TRANSCRIPTION_COMPLETE = auto()
    LLM_RESPONSE_READY = auto()
    LLM_RESPONSE_CHUNK = auto()

    # Playback events
    PLAYBACK_START = auto()
    PLAYBACK_DONE = auto()

    # State events
    STATE_CHANGED = auto()
    TIMEOUT_WARNING = auto()

    # System events
    ERROR = auto()
    SHUTDOWN = auto()


@dataclass
class Event:
    """Event with type and optional data."""
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Event({self.type.name}, {self.data})"


class EventBus:
    """Async event bus for component communication."""

    def __init__(self):
        self._subscribers: dict[EventType, list[Callable[[Event], Coroutine]]] = {}
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False
        self._process_task: asyncio.Task | None = None

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Coroutine],
    ) -> None:
        """Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async function to call when event occurs
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler to {event_type.name}")

    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Coroutine],
    ) -> None:
        """Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from {event_type.name}")
            except ValueError:
                pass

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        await self._queue.put(event)

    def publish_sync(self, event: Event) -> None:
        """Publish an event synchronously (for use in callbacks).

        Args:
            event: Event to publish
        """
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event}")

    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all subscribers."""
        handlers = self._subscribers.get(event.type, [])

        if not handlers:
            logger.debug(f"No handlers for event: {event.type.name}")
            return

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.type.name}: {e}")

    async def start(self) -> None:
        """Start the event bus."""
        if self._running:
            return

        self._running = True
        self._process_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")

    async def stop(self) -> None:
        """Stop the event bus."""
        self._running = False

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        logger.info("Event bus stopped")
