"""test_state.py — Unit tests for StateMachine transition matrix.

Covers:
  - Initial state is IDLE.
  - Every allowed transition succeeds and updates .state.
  - Every disallowed transition raises InvalidTransitionError.
  - STATE_CHANGED event is published with data=new_state on success.
  - No crash when bus=None (unit-test mode).
  - can_transition() returns the correct bool before a transition.
  - P1-S08 regression: SPEAKING → SPEAKING raises InvalidTransitionError
    (the bug that caused crashes in _handle_goodbye).
"""

from __future__ import annotations

import asyncio

import pytest

from src.events import EventBus, EventType
from src.state import ConversationState as CS
from src.state import InvalidTransitionError, StateMachine

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_state_is_idle():
    sm = StateMachine()
    assert sm.state == CS.IDLE


# ---------------------------------------------------------------------------
# Allowed transitions — every edge in _TRANSITIONS must succeed
# ---------------------------------------------------------------------------

_ALLOWED = [
    # from         to
    (CS.IDLE,          CS.LISTENING),
    (CS.LISTENING,     CS.PROCESSING),
    (CS.LISTENING,     CS.TIMEOUT_CHECK),
    (CS.LISTENING,     CS.IDLE),
    (CS.PROCESSING,    CS.SPEAKING),
    (CS.PROCESSING,    CS.LISTENING),
    (CS.PROCESSING,    CS.IDLE),
    (CS.SPEAKING,      CS.LISTENING),
    (CS.SPEAKING,      CS.TIMEOUT_CHECK),
    (CS.SPEAKING,      CS.IDLE),
    (CS.TIMEOUT_CHECK, CS.LISTENING),
    (CS.TIMEOUT_CHECK, CS.TIMEOUT_FINAL),
    (CS.TIMEOUT_CHECK, CS.IDLE),
    (CS.TIMEOUT_FINAL, CS.IDLE),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("from_state,to_state", _ALLOWED,
    ids=[f"{f.name}→{t.name}" for f, t in _ALLOWED])
async def test_allowed_transition_succeeds(from_state: CS, to_state: CS) -> None:
    sm = StateMachine()
    sm._state = from_state          # set start state directly (no bus needed)
    await sm.transition(to_state)
    assert sm.state == to_state


# ---------------------------------------------------------------------------
# Disallowed transitions — representative set must raise InvalidTransitionError
# ---------------------------------------------------------------------------

_DISALLOWED = [
    # from         to                    # why it matters
    (CS.IDLE,          CS.PROCESSING),   # plan example
    (CS.IDLE,          CS.SPEAKING),
    (CS.IDLE,          CS.TIMEOUT_FINAL),
    (CS.LISTENING,     CS.SPEAKING),     # must go through PROCESSING
    (CS.LISTENING,     CS.TIMEOUT_FINAL),
    (CS.PROCESSING,    CS.TIMEOUT_CHECK),
    (CS.SPEAKING,      CS.PROCESSING),
    (CS.SPEAKING,      CS.SPEAKING),     # P1-S08 regression
    (CS.TIMEOUT_FINAL, CS.LISTENING),
    (CS.TIMEOUT_FINAL, CS.SPEAKING),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("from_state,to_state", _DISALLOWED,
    ids=[f"{f.name}→{t.name}" for f, t in _DISALLOWED])
async def test_disallowed_transition_raises(from_state: CS, to_state: CS) -> None:
    sm = StateMachine()
    sm._state = from_state
    with pytest.raises(InvalidTransitionError):
        await sm.transition(to_state)


# ---------------------------------------------------------------------------
# Same-state self-loops are never allowed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("state", list(CS), ids=[s.name for s in CS])
async def test_self_loop_raises(state: CS) -> None:
    sm = StateMachine()
    sm._state = state
    with pytest.raises(InvalidTransitionError):
        await sm.transition(state)


# ---------------------------------------------------------------------------
# STATE_CHANGED event is published on successful transition
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_state_changed_event_published(bus: EventBus) -> None:
    q = bus.subscribe(EventType.STATE_CHANGED)
    sm = StateMachine(bus=bus)

    await sm.transition(CS.LISTENING)

    event = await asyncio.wait_for(q.get(), timeout=1.0)
    assert event.type == EventType.STATE_CHANGED
    assert event.data == CS.LISTENING


@pytest.mark.asyncio
async def test_state_changed_carries_new_state(bus: EventBus) -> None:
    """Each transition publishes the new (post-transition) state."""
    q = bus.subscribe(EventType.STATE_CHANGED)
    sm = StateMachine(bus=bus)

    await sm.transition(CS.LISTENING)
    await sm.transition(CS.PROCESSING)

    first  = await asyncio.wait_for(q.get(), timeout=1.0)
    second = await asyncio.wait_for(q.get(), timeout=1.0)
    assert first.data  == CS.LISTENING
    assert second.data == CS.PROCESSING


@pytest.mark.asyncio
async def test_no_event_published_on_failed_transition(bus: EventBus) -> None:
    """A rejected transition must not publish STATE_CHANGED."""
    q = bus.subscribe(EventType.STATE_CHANGED)
    sm = StateMachine(bus=bus)

    with pytest.raises(InvalidTransitionError):
        await sm.transition(CS.PROCESSING)   # IDLE → PROCESSING disallowed

    # Queue should be empty — nothing published
    assert q.empty()


# ---------------------------------------------------------------------------
# bus=None — no EventBus crash in unit-test mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transition_without_bus_does_not_raise() -> None:
    sm = StateMachine(bus=None)
    await sm.transition(CS.LISTENING)
    assert sm.state == CS.LISTENING


# ---------------------------------------------------------------------------
# can_transition()
# ---------------------------------------------------------------------------

def test_can_transition_true_for_allowed() -> None:
    sm = StateMachine()
    assert sm.can_transition(CS.LISTENING) is True


def test_can_transition_false_for_disallowed() -> None:
    sm = StateMachine()
    assert sm.can_transition(CS.PROCESSING) is False


# ---------------------------------------------------------------------------
# P1-S08 regression — SPEAKING → SPEAKING must raise
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_speaking_to_speaking_raises_p1_s08_regression() -> None:
    """_handle_goodbye used to call transition(SPEAKING) unconditionally.
    If already in SPEAKING this raised InvalidTransitionError and crashed the turn.
    P1-S08 added a state guard; this test ensures the machine's behaviour
    (the reason the guard was needed) is locked in.
    """
    sm = StateMachine()
    sm._state = CS.SPEAKING
    with pytest.raises(InvalidTransitionError, match="SPEAKING.*SPEAKING"):
        await sm.transition(CS.SPEAKING)
