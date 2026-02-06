"""Tests for ConversationStateMachine."""
import asyncio
import pytest

from src.core.state_machine import ConversationState, ConversationStateMachine


@pytest.fixture
def state_machine():
    """Create a state machine for testing."""
    return ConversationStateMachine(timeout_seconds=60.0)


class TestConversationStateMachine:
    """Tests for ConversationStateMachine class."""

    def test_initial_state(self, state_machine):
        """Test initial state is IDLE."""
        assert state_machine.state == ConversationState.IDLE
        assert not state_machine.is_active

    @pytest.mark.asyncio
    async def test_valid_transition(self, state_machine):
        """Test valid state transition."""
        result = await state_machine.transition_to(ConversationState.LISTENING)
        assert result
        assert state_machine.state == ConversationState.LISTENING
        assert state_machine.is_active

    @pytest.mark.asyncio
    async def test_invalid_transition(self, state_machine):
        """Test invalid state transition is rejected."""
        # Can't go from IDLE directly to PROCESSING
        result = await state_machine.transition_to(ConversationState.PROCESSING)
        assert not result
        assert state_machine.state == ConversationState.IDLE

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, state_machine):
        """Test complete conversation state flow."""
        # IDLE -> LISTENING
        assert await state_machine.transition_to(ConversationState.LISTENING)

        # LISTENING -> PROCESSING
        assert await state_machine.transition_to(ConversationState.PROCESSING)

        # PROCESSING -> SPEAKING
        assert await state_machine.transition_to(ConversationState.SPEAKING)

        # SPEAKING -> LISTENING
        assert await state_machine.transition_to(ConversationState.LISTENING)

    @pytest.mark.asyncio
    async def test_state_change_handler(self, state_machine):
        """Test state change handler is called."""
        handler_called = False
        received_states = []

        async def handler(old_state, new_state):
            nonlocal handler_called, received_states
            handler_called = True
            received_states.append((old_state, new_state))

        state_machine.on_state_change(handler)

        await state_machine.transition_to(ConversationState.LISTENING)

        assert handler_called
        assert received_states[-1] == (ConversationState.IDLE, ConversationState.LISTENING)

    @pytest.mark.asyncio
    async def test_reset(self, state_machine):
        """Test reset to IDLE."""
        await state_machine.transition_to(ConversationState.LISTENING)
        await state_machine.transition_to(ConversationState.PROCESSING)

        await state_machine.reset()

        assert state_machine.state == ConversationState.IDLE

    @pytest.mark.asyncio
    async def test_timeout_transition(self, state_machine):
        """Test timeout warning transition."""
        await state_machine.transition_to(ConversationState.LISTENING)

        # LISTENING -> TIMEOUT_WARNING
        assert await state_machine.transition_to(ConversationState.TIMEOUT_WARNING)

        # TIMEOUT_WARNING -> IDLE
        assert await state_machine.transition_to(ConversationState.IDLE)

    def test_update_activity(self, state_machine):
        """Test activity update."""
        import time
        initial_time = state_machine._last_activity_time
        time.sleep(0.01)
        state_machine.update_activity()
        assert state_machine._last_activity_time > initial_time
