"""state.py — Conversation state machine for Yona.

Defines the six application states and the allowed transitions between them.
The StateMachine publishes STATE_CHANGED events on every successful transition.

States::

    IDLE           — waiting for wake word
    LISTENING      — recording user speech (post-wake-word)
    PROCESSING     — STT in progress
    SPEAKING       — TTS + audio playback in progress
    TIMEOUT_CHECK  — first idle timeout; "아직 계세요?" prompt
    TIMEOUT_FINAL  — second idle timeout; farewell → back to IDLE

Allowed transitions::

    IDLE           → LISTENING
    LISTENING      → PROCESSING | TIMEOUT_CHECK | IDLE
    PROCESSING     → SPEAKING   | IDLE
    SPEAKING       → LISTENING  | TIMEOUT_CHECK | IDLE
    TIMEOUT_CHECK  → LISTENING  | TIMEOUT_FINAL | IDLE
    TIMEOUT_FINAL  → IDLE
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.events import EventBus


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class ConversationState(Enum):
    IDLE          = auto()
    LISTENING     = auto()
    PROCESSING    = auto()
    SPEAKING      = auto()
    TIMEOUT_CHECK = auto()
    TIMEOUT_FINAL = auto()


# ---------------------------------------------------------------------------
# Valid transitions
# ---------------------------------------------------------------------------

_TRANSITIONS: dict[ConversationState, frozenset[ConversationState]] = {
    ConversationState.IDLE: frozenset({
        ConversationState.LISTENING,
    }),
    ConversationState.LISTENING: frozenset({
        ConversationState.PROCESSING,
        ConversationState.TIMEOUT_CHECK,
        ConversationState.IDLE,
    }),
    ConversationState.PROCESSING: frozenset({
        ConversationState.SPEAKING,
        ConversationState.IDLE,
    }),
    ConversationState.SPEAKING: frozenset({
        ConversationState.LISTENING,
        ConversationState.TIMEOUT_CHECK,
        ConversationState.IDLE,
    }),
    ConversationState.TIMEOUT_CHECK: frozenset({
        ConversationState.LISTENING,
        ConversationState.TIMEOUT_FINAL,
        ConversationState.IDLE,
    }),
    ConversationState.TIMEOUT_FINAL: frozenset({
        ConversationState.IDLE,
    }),
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed."""


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class StateMachine:
    """Finite-state machine for Yona's conversation flow.

    Publishes ``EventType.STATE_CHANGED`` (data=new_state) on every
    successful transition.

    Args:
        bus: The application EventBus.  May be *None* for unit tests that
             do not need event delivery.
    """

    def __init__(self, bus: EventBus | None = None) -> None:
        self._state = ConversationState.IDLE
        self._bus = bus

    @property
    def state(self) -> ConversationState:
        """Current state (read-only)."""
        return self._state

    def can_transition(self, new_state: ConversationState) -> bool:
        """Return True if transitioning to *new_state* is allowed from the current state."""
        return new_state in _TRANSITIONS.get(self._state, frozenset())

    async def transition(self, new_state: ConversationState) -> None:
        """Transition to *new_state* and publish STATE_CHANGED.

        Raises:
            InvalidTransitionError: if the transition is not allowed.
        """
        if not self.can_transition(new_state):
            raise InvalidTransitionError(
                f"Cannot transition from {self._state.name} to {new_state.name}"
            )
        self._state = new_state
        if self._bus is not None:
            from src.events import EventType
            await self._bus.publish(EventType.STATE_CHANGED, new_state)
