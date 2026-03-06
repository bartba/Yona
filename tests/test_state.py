"""Tests for src/state.py

Run with:
    pytest tests/test_state.py -v

All tests are purely in-memory; no hardware or external services needed.
"""

from __future__ import annotations

import pytest

from src.events import EventBus, EventType
from src.state import ConversationState, InvalidTransitionError, StateMachine


# ---------------------------------------------------------------------------
# ConversationState enum
# ---------------------------------------------------------------------------

class TestConversationState:
    def test_has_exactly_6_members(self):
        assert len(ConversationState) == 6

    def test_all_values_are_unique(self):
        values = [s.value for s in ConversationState]
        assert len(values) == len(set(values))

    def test_expected_names_present(self):
        names = {s.name for s in ConversationState}
        assert names == {
            "IDLE",
            "LISTENING",
            "PROCESSING",
            "SPEAKING",
            "TIMEOUT_CHECK",
            "TIMEOUT_FINAL",
        }


# ---------------------------------------------------------------------------
# StateMachine — initialisation
# ---------------------------------------------------------------------------

class TestStateMachineInit:
    def test_initial_state_is_idle(self):
        sm = StateMachine()
        assert sm.state is ConversationState.IDLE

    def test_bus_defaults_to_none(self):
        sm = StateMachine()
        assert sm._bus is None

    def test_state_property_is_read_only(self):
        sm = StateMachine()
        with pytest.raises(AttributeError):
            sm.state = ConversationState.LISTENING  # type: ignore[misc]


# ---------------------------------------------------------------------------
# StateMachine — can_transition
# ---------------------------------------------------------------------------

class TestCanTransition:
    # IDLE allowed
    def test_idle_allows_listening(self):
        assert StateMachine().can_transition(ConversationState.LISTENING) is True

    # IDLE denied
    def test_idle_denies_processing(self):
        assert StateMachine().can_transition(ConversationState.PROCESSING) is False

    def test_idle_denies_speaking(self):
        assert StateMachine().can_transition(ConversationState.SPEAKING) is False

    def test_idle_denies_timeout_check(self):
        assert StateMachine().can_transition(ConversationState.TIMEOUT_CHECK) is False

    def test_idle_denies_timeout_final(self):
        assert StateMachine().can_transition(ConversationState.TIMEOUT_FINAL) is False

    def test_idle_denies_self(self):
        assert StateMachine().can_transition(ConversationState.IDLE) is False

    # LISTENING allowed
    @pytest.mark.asyncio
    async def test_listening_allows_processing(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        assert sm.can_transition(ConversationState.PROCESSING) is True

    @pytest.mark.asyncio
    async def test_listening_allows_timeout_check(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        assert sm.can_transition(ConversationState.TIMEOUT_CHECK) is True

    @pytest.mark.asyncio
    async def test_listening_allows_idle(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        assert sm.can_transition(ConversationState.IDLE) is True

    # LISTENING denied
    @pytest.mark.asyncio
    async def test_listening_denies_speaking(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        assert sm.can_transition(ConversationState.SPEAKING) is False

    # PROCESSING allowed
    @pytest.mark.asyncio
    async def test_processing_allows_speaking(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.PROCESSING)
        assert sm.can_transition(ConversationState.SPEAKING) is True

    @pytest.mark.asyncio
    async def test_processing_allows_idle(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.PROCESSING)
        assert sm.can_transition(ConversationState.IDLE) is True

    # PROCESSING denied
    @pytest.mark.asyncio
    async def test_processing_denies_listening(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.PROCESSING)
        assert sm.can_transition(ConversationState.LISTENING) is False

    # SPEAKING allowed
    @pytest.mark.asyncio
    async def test_speaking_allows_listening(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.PROCESSING)
        await sm.transition(ConversationState.SPEAKING)
        assert sm.can_transition(ConversationState.LISTENING) is True

    @pytest.mark.asyncio
    async def test_speaking_allows_timeout_check(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.PROCESSING)
        await sm.transition(ConversationState.SPEAKING)
        assert sm.can_transition(ConversationState.TIMEOUT_CHECK) is True

    @pytest.mark.asyncio
    async def test_speaking_allows_idle(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.PROCESSING)
        await sm.transition(ConversationState.SPEAKING)
        assert sm.can_transition(ConversationState.IDLE) is True

    # TIMEOUT_CHECK allowed
    @pytest.mark.asyncio
    async def test_timeout_check_allows_listening(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.TIMEOUT_CHECK)
        assert sm.can_transition(ConversationState.LISTENING) is True

    @pytest.mark.asyncio
    async def test_timeout_check_allows_timeout_final(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.TIMEOUT_CHECK)
        assert sm.can_transition(ConversationState.TIMEOUT_FINAL) is True

    @pytest.mark.asyncio
    async def test_timeout_check_allows_idle(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.TIMEOUT_CHECK)
        assert sm.can_transition(ConversationState.IDLE) is True

    # TIMEOUT_FINAL
    @pytest.mark.asyncio
    async def test_timeout_final_allows_idle(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.TIMEOUT_CHECK)
        await sm.transition(ConversationState.TIMEOUT_FINAL)
        assert sm.can_transition(ConversationState.IDLE) is True

    @pytest.mark.asyncio
    async def test_timeout_final_denies_listening(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.TIMEOUT_CHECK)
        await sm.transition(ConversationState.TIMEOUT_FINAL)
        assert sm.can_transition(ConversationState.LISTENING) is False


# ---------------------------------------------------------------------------
# StateMachine — transition (happy paths)
# ---------------------------------------------------------------------------

class TestStateMachineTransition:
    @pytest.mark.asyncio
    async def test_transition_updates_state(self):
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        assert sm.state is ConversationState.LISTENING

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self):
        sm = StateMachine()
        with pytest.raises(InvalidTransitionError):
            await sm.transition(ConversationState.SPEAKING)

    @pytest.mark.asyncio
    async def test_invalid_transition_does_not_change_state(self):
        sm = StateMachine()
        with pytest.raises(InvalidTransitionError):
            await sm.transition(ConversationState.PROCESSING)
        assert sm.state is ConversationState.IDLE

    @pytest.mark.asyncio
    async def test_error_message_names_both_states(self):
        sm = StateMachine()
        with pytest.raises(InvalidTransitionError) as exc_info:
            await sm.transition(ConversationState.SPEAKING)
        msg = str(exc_info.value)
        assert "IDLE" in msg
        assert "SPEAKING" in msg

    # --- Named scenario paths ---

    @pytest.mark.asyncio
    async def test_happy_path(self):
        """IDLE → LISTENING → PROCESSING → SPEAKING → LISTENING."""
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.PROCESSING)
        await sm.transition(ConversationState.SPEAKING)
        await sm.transition(ConversationState.LISTENING)
        assert sm.state is ConversationState.LISTENING

    @pytest.mark.asyncio
    async def test_barge_in_path(self):
        """SPEAKING → LISTENING (barge-in interrupts playback)."""
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.PROCESSING)
        await sm.transition(ConversationState.SPEAKING)
        await sm.transition(ConversationState.LISTENING)
        assert sm.state is ConversationState.LISTENING

    @pytest.mark.asyncio
    async def test_timeout_escalation_path(self):
        """LISTENING → TIMEOUT_CHECK → TIMEOUT_FINAL → IDLE."""
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.TIMEOUT_CHECK)
        await sm.transition(ConversationState.TIMEOUT_FINAL)
        await sm.transition(ConversationState.IDLE)
        assert sm.state is ConversationState.IDLE

    @pytest.mark.asyncio
    async def test_timeout_recovery_path(self):
        """TIMEOUT_CHECK → LISTENING (user speaks again)."""
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.TIMEOUT_CHECK)
        await sm.transition(ConversationState.LISTENING)
        assert sm.state is ConversationState.LISTENING

    @pytest.mark.asyncio
    async def test_goodbye_from_processing(self):
        """PROCESSING → IDLE (goodbye intent)."""
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.PROCESSING)
        await sm.transition(ConversationState.IDLE)
        assert sm.state is ConversationState.IDLE

    @pytest.mark.asyncio
    async def test_goodbye_from_speaking(self):
        """SPEAKING → IDLE (goodbye intent during playback)."""
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.PROCESSING)
        await sm.transition(ConversationState.SPEAKING)
        await sm.transition(ConversationState.IDLE)
        assert sm.state is ConversationState.IDLE

    @pytest.mark.asyncio
    async def test_speaking_timeout_then_final(self):
        """SPEAKING → TIMEOUT_CHECK → TIMEOUT_FINAL → IDLE."""
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.PROCESSING)
        await sm.transition(ConversationState.SPEAKING)
        await sm.transition(ConversationState.TIMEOUT_CHECK)
        await sm.transition(ConversationState.TIMEOUT_FINAL)
        await sm.transition(ConversationState.IDLE)
        assert sm.state is ConversationState.IDLE

    @pytest.mark.asyncio
    async def test_wake_word_after_reset(self):
        """IDLE → LISTENING again after a full cycle."""
        sm = StateMachine()
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.PROCESSING)
        await sm.transition(ConversationState.SPEAKING)
        await sm.transition(ConversationState.IDLE)
        # New conversation starts
        await sm.transition(ConversationState.LISTENING)
        assert sm.state is ConversationState.LISTENING


# ---------------------------------------------------------------------------
# StateMachine — EventBus integration
# ---------------------------------------------------------------------------

class TestStateMachineEvents:
    @pytest.mark.asyncio
    async def test_state_changed_published_on_transition(self):
        bus = EventBus()
        sm = StateMachine(bus=bus)
        q = bus.subscribe(EventType.STATE_CHANGED)
        await sm.transition(ConversationState.LISTENING)
        event = q.get_nowait()
        assert event.type is EventType.STATE_CHANGED
        assert event.data is ConversationState.LISTENING

    @pytest.mark.asyncio
    async def test_event_data_is_new_state(self):
        bus = EventBus()
        sm = StateMachine(bus=bus)
        q = bus.subscribe(EventType.STATE_CHANGED)
        await sm.transition(ConversationState.LISTENING)
        await sm.transition(ConversationState.PROCESSING)
        e1 = q.get_nowait()
        e2 = q.get_nowait()
        assert e1.data is ConversationState.LISTENING
        assert e2.data is ConversationState.PROCESSING

    @pytest.mark.asyncio
    async def test_no_bus_does_not_raise(self):
        """StateMachine without a bus must not raise on transition."""
        sm = StateMachine(bus=None)
        await sm.transition(ConversationState.LISTENING)
        assert sm.state is ConversationState.LISTENING

    @pytest.mark.asyncio
    async def test_failed_transition_does_not_publish_event(self):
        bus = EventBus()
        sm = StateMachine(bus=bus)
        q = bus.subscribe(EventType.STATE_CHANGED)
        with pytest.raises(InvalidTransitionError):
            await sm.transition(ConversationState.PROCESSING)
        assert q.empty()

    @pytest.mark.asyncio
    async def test_multiple_subscribers_all_receive(self):
        bus = EventBus()
        sm = StateMachine(bus=bus)
        q1 = bus.subscribe(EventType.STATE_CHANGED)
        q2 = bus.subscribe(EventType.STATE_CHANGED)
        await sm.transition(ConversationState.LISTENING)
        assert not q1.empty()
        assert not q2.empty()

    @pytest.mark.asyncio
    async def test_event_sequence_matches_transitions(self):
        """Full path publishes events in order."""
        bus = EventBus()
        sm = StateMachine(bus=bus)
        q = bus.subscribe(EventType.STATE_CHANGED)

        path = [
            ConversationState.LISTENING,
            ConversationState.PROCESSING,
            ConversationState.SPEAKING,
            ConversationState.LISTENING,
        ]
        for state in path:
            await sm.transition(state)

        received = [q.get_nowait().data for _ in path]
        assert received == path