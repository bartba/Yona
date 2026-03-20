"""events.py — Application-wide event system for Yona.

Defines all EventType values and an async EventBus (pub/sub, queue-based).

Usage::

    bus = EventBus()
    q = bus.subscribe(EventType.WAKE_WORD_DETECTED)
    await bus.publish(EventType.WAKE_WORD_DETECTED)
    event = await q.get()

From synchronous sounddevice callbacks use publish_nowait():

    bus.publish_nowait(EventType.SPEECH_STARTED)
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections import defaultdict
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    # --- Detection ---
    WAKE_WORD_DETECTED = auto()   # Porcupine detected wake phrase
    SPEECH_STARTED     = auto()   # VAD: voice activity began
    SPEECH_ENDED       = auto()   # VAD: voice activity stopped (silence)
    BARGE_IN_DETECTED  = auto()   # VAD: voice during SPEAKING → interrupt

    # --- Pipeline ---
    TRANSCRIPTION_READY  = auto() # STT done; data = transcribed str
    LLM_RESPONSE_STARTED = auto() # LLM streaming started
    LLM_RESPONSE_CHUNK   = auto() # One streamed token; data = str
    LLM_RESPONSE_DONE    = auto() # LLM streaming finished
    PHRASE_READY         = auto() # PhraseAccumulator emitted phrase; data = str
    AUDIO_CHUNK_READY    = auto() # TTS produced audio; data = (np.ndarray, int sr)

    # --- Playback ---
    PLAYBACK_STARTED = auto()     # Speaker output began
    PLAYBACK_DONE    = auto()     # Speaker finished, queues drained

    # --- Session control ---
    TIMEOUT_CHECK    = auto()     # 15 s idle → ask "아직 계세요?"
    TIMEOUT_FINAL    = auto()     # 5 s more → farewell → IDLE
    GOODBYE_DETECTED = auto()     # Goodbye intent recognised

    # --- System ---
    STATE_CHANGED = auto()        # StateMachine transition; data = new state
    ERROR         = auto()        # Unhandled error; data = exception
    SHUTDOWN      = auto()        # Graceful shutdown requested


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Event:
    """A single published event.

    Attributes:
        type:      The EventType that was published.
        data:      Optional payload — type depends on the EventType.
        timestamp: Unix time of creation (seconds).
    """

    type: EventType
    data: Any = None
    timestamp: float = dataclasses.field(default_factory=time.time)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """Async, queue-based pub/sub event bus.

    Each subscriber receives its own :class:`asyncio.Queue` so that slow
    consumers cannot block fast publishers.  The queue is returned from
    :meth:`subscribe` and events are read with ``await q.get()``.

    Designed to be instantiated once and shared across all components.

    Thread safety:
        :meth:`publish_nowait` is safe to call from synchronous callbacks
        (e.g. the sounddevice audio callback) because ``Queue.put_nowait``
        is thread-safe for CPython.  Overflow events are silently dropped.
    """

    def __init__(self) -> None:
        # Maps EventType → list of subscriber queues
        self._subscribers: dict[EventType, list[asyncio.Queue[Event]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType) -> asyncio.Queue[Event]:
        """Register interest in *event_type*.

        Returns a new :class:`asyncio.Queue` that will receive every
        :class:`Event` published for that type.  The caller owns the queue
        and should call :method:`unsubscribe` when done.
        """
        q: asyncio.Queue[Event] = asyncio.Queue()
        self._subscribers[event_type].append(q)
        return q

    def unsubscribe(self, event_type: EventType, queue: asyncio.Queue[Event]) -> None:
        """Remove *queue* from the subscriber list for *event_type*.

        Silently ignores unknown queues.
        """
        queues = self._subscribers.get(event_type)
        if queues is None:
            return
        try:
            queues.remove(queue)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, event_type: EventType, data: Any = None) -> None:
        """Publish an event to all current subscribers (async version).

        Creates one :class:`Event` instance shared across all queues, so
        subscribers receive the same object.  Callers must not mutate *data*
        after publishing.
        """
        event = Event(type=event_type, data=data)
        for q in list(self._subscribers.get(event_type, [])):
            await q.put(event)

    def publish_nowait(self, event_type: EventType, data: Any = None) -> None:
        """Publish without blocking — safe from synchronous contexts.

        Events are dropped (not buffered) if a subscriber's queue is full.
        Use this only from sounddevice callbacks or other sync code where
        ``await`` is not available.
        """
        event = Event(type=event_type, data=data)
        for q in list(self._subscribers.get(event_type, [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # slow subscriber — drop rather than block
