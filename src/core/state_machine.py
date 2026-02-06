"""Conversation state machine for Yona."""
import asyncio
import time
from enum import Enum, auto
from typing import Callable, Coroutine

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationState(Enum):
    """States in the conversation state machine."""
    IDLE = auto()           # Waiting for wake word
    LISTENING = auto()      # Recording user speech
    PROCESSING = auto()     # STT + LLM processing
    SPEAKING = auto()       # TTS playback
    TIMEOUT_WARNING = auto()  # Playing timeout message


class ConversationStateMachine:
    """Manages conversation state transitions."""

    # Valid state transitions
    TRANSITIONS = {
        ConversationState.IDLE: {ConversationState.LISTENING},
        ConversationState.LISTENING: {
            ConversationState.PROCESSING,
            ConversationState.TIMEOUT_WARNING,
            ConversationState.IDLE,
        },
        ConversationState.PROCESSING: {
            ConversationState.SPEAKING,
            ConversationState.IDLE,
        },
        ConversationState.SPEAKING: {
            ConversationState.LISTENING,
            ConversationState.TIMEOUT_WARNING,
            ConversationState.IDLE,
        },
        ConversationState.TIMEOUT_WARNING: {ConversationState.IDLE},
    }

    def __init__(self, timeout_seconds: float = 60.0):
        """Initialize the state machine.

        Args:
            timeout_seconds: Seconds of inactivity before timeout
        """
        self._state = ConversationState.IDLE
        self._timeout_seconds = timeout_seconds
        self._last_activity_time = time.time()
        self._lock = asyncio.Lock()
        self._state_handlers: dict[
            ConversationState,
            list[Callable[[ConversationState, ConversationState], Coroutine]]
        ] = {}
        self._timeout_task: asyncio.Task | None = None
        self._running = False

    @property
    def state(self) -> ConversationState:
        """Current conversation state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether the conversation is active (not idle)."""
        return self._state != ConversationState.IDLE

    def on_state_change(
        self,
        handler: Callable[[ConversationState, ConversationState], Coroutine],
        state: ConversationState | None = None,
    ) -> None:
        """Register a handler for state changes.

        Args:
            handler: Async function called with (old_state, new_state)
            state: Specific state to handle, or None for all states
        """
        key = state if state else ConversationState.IDLE  # Use as "all" marker
        if key not in self._state_handlers:
            self._state_handlers[key] = []
        self._state_handlers[key].append(handler)

    async def transition_to(self, new_state: ConversationState) -> bool:
        """Attempt to transition to a new state.

        Args:
            new_state: Target state

        Returns:
            True if transition was successful
        """
        async with self._lock:
            if new_state not in self.TRANSITIONS.get(self._state, set()):
                logger.warning(
                    f"Invalid transition: {self._state.name} -> {new_state.name}"
                )
                return False

            old_state = self._state
            self._state = new_state
            self._last_activity_time = time.time()

            logger.info(f"State: {old_state.name} -> {new_state.name}")

            # Notify handlers
            await self._notify_handlers(old_state, new_state)

            return True

    async def _notify_handlers(
        self,
        old_state: ConversationState,
        new_state: ConversationState,
    ) -> None:
        """Notify registered handlers of state change."""
        # Handlers for specific new state
        for handler in self._state_handlers.get(new_state, []):
            try:
                await handler(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state handler: {e}")

        # Handlers for all states (stored under IDLE key as marker)
        for handler in self._state_handlers.get(ConversationState.IDLE, []):
            if new_state != ConversationState.IDLE:  # Don't double-call IDLE handlers
                try:
                    await handler(old_state, new_state)
                except Exception as e:
                    logger.error(f"Error in state handler: {e}")

    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self._last_activity_time = time.time()

    async def _timeout_monitor(self) -> None:
        """Monitor for conversation timeout."""
        while self._running:
            await asyncio.sleep(1.0)

            if not self.is_active:
                continue

            elapsed = time.time() - self._last_activity_time

            if elapsed >= self._timeout_seconds:
                if self._state in (
                    ConversationState.LISTENING,
                    ConversationState.SPEAKING,
                ):
                    logger.info(f"Conversation timeout after {elapsed:.1f}s")
                    await self.transition_to(ConversationState.TIMEOUT_WARNING)

    async def start(self) -> None:
        """Start the state machine timeout monitor."""
        if self._running:
            return

        self._running = True
        self._timeout_task = asyncio.create_task(self._timeout_monitor())
        logger.info("State machine started")

    async def stop(self) -> None:
        """Stop the state machine."""
        self._running = False

        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass

        logger.info("State machine stopped")

    async def reset(self) -> None:
        """Reset to idle state."""
        async with self._lock:
            old_state = self._state
            self._state = ConversationState.IDLE
            self._last_activity_time = time.time()

            if old_state != ConversationState.IDLE:
                logger.info(f"State reset: {old_state.name} -> IDLE")
                await self._notify_handlers(old_state, ConversationState.IDLE)
