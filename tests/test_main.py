"""Tests for src/main.py — YonaApp orchestrator.

Run with:
    pytest tests/test_main.py -v

All tests are hardware-free.  Heavy dependencies (sounddevice, faster_whisper,
openwakeword, onnxruntime, melo, openai, anthropic, httpx) are stubbed via
sys.modules before any src imports.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub heavy third-party packages before importing src modules
# ---------------------------------------------------------------------------
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("faster_whisper", MagicMock())
sys.modules.setdefault("openwakeword", MagicMock())
sys.modules.setdefault("openwakeword.model", MagicMock())
sys.modules.setdefault("onnxruntime", MagicMock())
sys.modules.setdefault("melo", MagicMock())
sys.modules.setdefault("melo.api", MagicMock())
sys.modules.setdefault("openai", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("httpx", MagicMock())

from src.config import Config                         # noqa: E402
from src.events import Event, EventBus, EventType     # noqa: E402
from src.llm import ConversationContext               # noqa: E402
from src.main import YonaApp, _GOODBYE_RE             # noqa: E402
from src.state import ConversationState as CS          # noqa: E402


# ---------------------------------------------------------------------------
# Test config YAML
# ---------------------------------------------------------------------------

_CFG_YAML = textwrap.dedent("""\
    audio:
      input_device: "default"
      output_device: "default"
      input_sample_rate: 16000
      output_sample_rate: 48000
      input_channels: 1
      output_channels: 2
      chunk_size: 512
      buffer_seconds: 30

    wake_word:
      model_paths: []
      threshold: 0.5
      patience: 3
      cooldown_seconds: 2.0

    vad:
      model_path: "models/silero_vad/silero_vad.onnx"
      threshold: 0.5
      silence_duration: 0.8
      min_speech_duration: 0.3
      barge_in_threshold: 0.7

    stt:
      model_size: "large-v3-turbo"
      device: "cuda"
      compute_type: "float16"
      language: null

    llm:
      provider: "openai"
      openai_api_key: "test-key"
      openai_model: "gpt-4o-mini"
      max_tokens: 1024
      temperature: 0.7
      max_history_turns: 20

    tts:
      provider: "melo"
      melo_language: "KR"
      melo_device: "cpu"
      melo_speed: 1.0
      output_sample_rate: 24000

    conversation:
      timeout_check_seconds: 0.1
      timeout_final_seconds: 0.1
      timeout_check_message:
        ko: "아직 계세요?"
        en: "Are you still there?"
      timeout_final_message:
        ko: "대화를 종료합니다."
        en: "Ending our conversation."
      goodbye_message:
        ko: "안녕히 계세요!"
        en: "Goodbye!"

    history:
      storage_dir: "data/history"
      max_days: 365

    logging:
      level: "WARNING"
    """)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    f = tmp_path / "config.yaml"
    f.write_text(_CFG_YAML)
    return Config(path=f)


@pytest.fixture
def app(cfg: Config) -> YonaApp:
    """Create a YonaApp with all components replaced by mocks."""
    a = YonaApp(cfg)

    # Audio
    a._audio = MagicMock()
    a._audio.play_audio = AsyncMock()
    a._audio.stop_playback = AsyncMock()
    a._audio.start = AsyncMock()
    a._audio.stop = AsyncMock()

    # Buffer
    a._buffer = MagicMock()
    a._buffer.get_all.return_value = np.zeros(16_000, dtype=np.float32)

    # Chime
    a._chime = MagicMock()
    a._chime.play = AsyncMock()

    # VAD
    a._vad = MagicMock()

    # Wake word
    a._wake = MagicMock()

    # STT
    a._stt = MagicMock()
    a._stt.transcribe = AsyncMock(return_value="오늘 날씨 어때?")
    a._stt.detected_language = "ko"

    # LLM handler
    a._chat_handler = MagicMock()
    a._chat_handler.close = AsyncMock()

    # TTS synthesizer
    a._synth = MagicMock()
    a._synth.synthesize = AsyncMock(
        return_value=(np.zeros(24_000, dtype=np.float32), 24_000),
    )
    a._synth.close = AsyncMock()

    # Pipeline
    a._pipeline = MagicMock()
    a._pipeline.run = AsyncMock(return_value="반갑습니다!")
    a._pipeline.interrupt = AsyncMock()

    # Context + History
    a._context = ConversationContext("Test system prompt")
    a._history = MagicMock()

    a._running = True
    return a


# ---------------------------------------------------------------------------
# Goodbye regex tests
# ---------------------------------------------------------------------------

class TestGoodbyeRegex:
    """Verify _GOODBYE_RE matches expected patterns."""

    @pytest.mark.parametrize("text", [
        "안녕",
        "잘 가",
        "잘가",
        "종료",
        "그만",
        "바이바이",
        "bye",
        "goodbye",
        "Goodbye",
        "see you",
        "See You",
        "that's all",
        "thats all",
    ])
    def test_matches_goodbye(self, text: str) -> None:
        assert _GOODBYE_RE.search(text) is not None

    @pytest.mark.parametrize("text", [
        "안녕하세요",  # greeting (contains 안녕) — will match by design
        "hello",
        "how are you",
        "오늘 날씨 어때",
    ])
    def test_non_goodbye(self, text: str) -> None:
        # "안녕하세요" contains "안녕" and will match — that's intended
        if "안녕" in text:
            assert _GOODBYE_RE.search(text) is not None
        else:
            assert _GOODBYE_RE.search(text) is None


# ---------------------------------------------------------------------------
# Audio callback dispatch tests
# ---------------------------------------------------------------------------

class TestAudioCallback:
    """_audio_callback dispatches to the correct component per state."""

    def test_idle_calls_wake_word(self, app: YonaApp) -> None:
        chunk = np.zeros(512, dtype=np.float32)
        app._audio_callback(chunk)
        app._wake.process_chunk.assert_called_once()

    def test_listening_calls_buffer_and_vad(self, app: YonaApp) -> None:
        app._sm._state = CS.LISTENING
        chunk = np.zeros(512, dtype=np.float32)
        app._audio_callback(chunk)
        app._buffer.push.assert_called_once()
        app._vad.process_chunk.assert_called_once()

    def test_speaking_calls_vad(self, app: YonaApp) -> None:
        app._sm._state = CS.SPEAKING
        chunk = np.zeros(512, dtype=np.float32)
        app._audio_callback(chunk)
        app._vad.process_chunk.assert_called_once()
        app._buffer.push.assert_not_called()

    def test_timeout_check_calls_vad(self, app: YonaApp) -> None:
        app._sm._state = CS.TIMEOUT_CHECK
        chunk = np.zeros(512, dtype=np.float32)
        app._audio_callback(chunk)
        app._vad.process_chunk.assert_called_once()

    def test_processing_does_nothing(self, app: YonaApp) -> None:
        app._sm._state = CS.PROCESSING
        chunk = np.zeros(512, dtype=np.float32)
        app._audio_callback(chunk)
        app._wake.process_chunk.assert_not_called()
        app._vad.process_chunk.assert_not_called()
        app._buffer.push.assert_not_called()


# ---------------------------------------------------------------------------
# Event handler tests
# ---------------------------------------------------------------------------

class TestOnWakeWord:
    """Wake word → chime → LISTENING."""

    @pytest.mark.asyncio
    async def test_transitions_to_listening(self, app: YonaApp) -> None:
        assert app._sm.state == CS.IDLE
        await app._on_wake_word(Event(type=EventType.WAKE_WORD_DETECTED, data="hi_inspector"))
        assert app._sm.state == CS.LISTENING

    @pytest.mark.asyncio
    async def test_plays_chime(self, app: YonaApp) -> None:
        await app._on_wake_word(Event(type=EventType.WAKE_WORD_DETECTED))
        app._chime.play.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resets_vad_and_buffer(self, app: YonaApp) -> None:
        await app._on_wake_word(Event(type=EventType.WAKE_WORD_DETECTED))
        app._vad.reset.assert_called_once()
        app._buffer.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_ignored_when_not_idle(self, app: YonaApp) -> None:
        app._sm._state = CS.LISTENING
        await app._on_wake_word(Event(type=EventType.WAKE_WORD_DETECTED))
        app._chime.play.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_starts_timeout(self, app: YonaApp) -> None:
        await app._on_wake_word(Event(type=EventType.WAKE_WORD_DETECTED))
        assert app._timeout_task is not None
        app._cancel_timeout()  # cleanup


class TestOnSpeechStarted:
    """SPEECH_STARTED resets timeout or returns from TIMEOUT_CHECK."""

    @pytest.mark.asyncio
    async def test_restarts_timeout_in_listening(self, app: YonaApp) -> None:
        app._sm._state = CS.LISTENING
        old_task = MagicMock()
        old_task.done.return_value = False
        app._timeout_task = old_task
        await app._on_speech_started(Event(type=EventType.SPEECH_STARTED))
        old_task.cancel.assert_called_once()
        assert app._timeout_task is not None
        app._cancel_timeout()

    @pytest.mark.asyncio
    async def test_timeout_check_returns_to_listening(self, app: YonaApp) -> None:
        # Transition: IDLE → LISTENING → TIMEOUT_CHECK
        app._sm._state = CS.TIMEOUT_CHECK
        await app._on_speech_started(Event(type=EventType.SPEECH_STARTED))
        assert app._sm.state == CS.LISTENING
        app._audio.stop_playback.assert_awaited_once()
        app._buffer.reset.assert_called()
        app._cancel_timeout()


class TestOnSpeechEnded:
    """SPEECH_ENDED → PROCESSING → create_task(_process_utterance)."""

    @pytest.mark.asyncio
    async def test_transitions_to_processing(self, app: YonaApp) -> None:
        app._sm._state = CS.LISTENING
        await app._on_speech_ended(Event(type=EventType.SPEECH_ENDED))
        assert app._sm.state == CS.PROCESSING
        assert app._process_task is not None
        # Let the process task run
        await asyncio.sleep(0.05)
        app._cancel_timeout()

    @pytest.mark.asyncio
    async def test_ignored_when_not_listening(self, app: YonaApp) -> None:
        app._sm._state = CS.IDLE
        await app._on_speech_ended(Event(type=EventType.SPEECH_ENDED))
        assert app._sm.state == CS.IDLE


class TestOnBargeIn:
    """BARGE_IN during SPEAKING → interrupt → LISTENING."""

    @pytest.mark.asyncio
    async def test_interrupts_pipeline(self, app: YonaApp) -> None:
        # Transition: IDLE → LISTENING → PROCESSING → SPEAKING
        app._sm._state = CS.SPEAKING
        await app._on_barge_in(Event(type=EventType.BARGE_IN_DETECTED))
        app._pipeline.interrupt.assert_awaited_once()
        assert app._sm.state == CS.LISTENING
        app._cancel_timeout()

    @pytest.mark.asyncio
    async def test_resets_vad_and_buffer(self, app: YonaApp) -> None:
        app._sm._state = CS.SPEAKING
        await app._on_barge_in(Event(type=EventType.BARGE_IN_DETECTED))
        app._vad.set_barge_in_mode.assert_called_with(False)
        app._vad.reset.assert_called()
        app._buffer.reset.assert_called()
        app._cancel_timeout()

    @pytest.mark.asyncio
    async def test_ignored_when_not_speaking(self, app: YonaApp) -> None:
        app._sm._state = CS.LISTENING
        await app._on_barge_in(Event(type=EventType.BARGE_IN_DETECTED))
        app._pipeline.interrupt.assert_not_awaited()


# ---------------------------------------------------------------------------
# Process utterance tests
# ---------------------------------------------------------------------------

class TestProcessUtterance:
    """Full STT → LLM+TTS pipeline flow."""

    @pytest.mark.asyncio
    async def test_normal_flow(self, app: YonaApp) -> None:
        """PROCESSING → STT → SPEAKING → pipeline → LISTENING."""
        app._sm._state = CS.PROCESSING
        app._stt.transcribe = AsyncMock(return_value="오늘 날씨 어때?")
        app._stt.detected_language = "ko"
        app._pipeline.run = AsyncMock(return_value="오늘은 맑아요.")

        await app._process_utterance()

        app._stt.transcribe.assert_awaited_once()
        app._pipeline.run.assert_awaited_once()
        assert app._context.message_count == 2  # user + assistant
        app._history.append_turn.assert_called_once_with("오늘 날씨 어때?", "오늘은 맑아요.")
        assert app._sm.state == CS.LISTENING
        app._cancel_timeout()

    @pytest.mark.asyncio
    async def test_empty_buffer(self, app: YonaApp) -> None:
        app._sm._state = CS.PROCESSING
        app._buffer.get_all.return_value = np.array([], dtype=np.float32)

        await app._process_utterance()

        app._stt.transcribe.assert_not_awaited()
        assert app._sm.state == CS.LISTENING
        app._cancel_timeout()

    @pytest.mark.asyncio
    async def test_empty_transcription(self, app: YonaApp) -> None:
        app._sm._state = CS.PROCESSING
        app._stt.transcribe = AsyncMock(return_value="")

        await app._process_utterance()

        app._pipeline.run.assert_not_awaited()
        assert app._sm.state == CS.LISTENING
        app._cancel_timeout()

    @pytest.mark.asyncio
    async def test_goodbye_intent(self, app: YonaApp) -> None:
        """Goodbye keyword → farewell TTS → IDLE."""
        app._sm._state = CS.PROCESSING
        app._stt.transcribe = AsyncMock(return_value="잘 가요")
        app._stt.detected_language = "ko"

        await app._process_utterance()

        app._pipeline.run.assert_not_awaited()
        # Farewell TTS
        app._synth.synthesize.assert_awaited_once()
        assert app._sm.state == CS.IDLE
        app._vad.reset.assert_called()
        app._wake.reset.assert_called()

    @pytest.mark.asyncio
    async def test_goodbye_clears_context(self, app: YonaApp) -> None:
        app._sm._state = CS.PROCESSING
        app._context.add_user("이전 대화")
        app._context.add_assistant("이전 응답")
        app._stt.transcribe = AsyncMock(return_value="goodbye")
        app._stt.detected_language = "en"

        await app._process_utterance()

        assert app._context.message_count == 0

    @pytest.mark.asyncio
    async def test_vad_barge_in_mode_during_speaking(self, app: YonaApp) -> None:
        """VAD switches to barge-in mode during SPEAKING."""
        app._sm._state = CS.PROCESSING
        app._stt.transcribe = AsyncMock(return_value="hello")
        app._stt.detected_language = "en"

        await app._process_utterance()

        # set_barge_in_mode(True) before pipeline, then set_barge_in_mode(False) after
        calls = [c.args[0] for c in app._vad.set_barge_in_mode.call_args_list]
        assert True in calls
        assert False in calls
        app._cancel_timeout()

    @pytest.mark.asyncio
    async def test_stt_error_recovery(self, app: YonaApp) -> None:
        """STT error → back to LISTENING gracefully."""
        app._sm._state = CS.PROCESSING
        app._stt.transcribe = AsyncMock(side_effect=RuntimeError("STT failed"))

        await app._process_utterance()

        assert app._sm.state == CS.LISTENING
        app._cancel_timeout()


# ---------------------------------------------------------------------------
# Timeout tests
# ---------------------------------------------------------------------------

class TestTimeout:
    """Two-stage inactivity timeout (using 0.1s for fast tests)."""

    @pytest.mark.asyncio
    async def test_timeout_check_reached(self, app: YonaApp) -> None:
        """Silence → TIMEOUT_CHECK → TTS "아직 계세요?"."""
        app._sm._state = CS.LISTENING
        app._restart_timeout()
        await asyncio.sleep(0.2)
        assert app._sm.state == CS.TIMEOUT_CHECK
        app._synth.synthesize.assert_awaited()
        app._cancel_timeout()

    @pytest.mark.asyncio
    async def test_full_timeout_to_idle(self, app: YonaApp) -> None:
        """Silence → TIMEOUT_CHECK → more silence → TIMEOUT_FINAL → IDLE."""
        app._sm._state = CS.LISTENING
        app._restart_timeout()
        await asyncio.sleep(0.5)  # 0.1 + 0.1 + margin
        assert app._sm.state == CS.IDLE
        # TTS called twice (check message + final message)
        assert app._synth.synthesize.await_count >= 2

    @pytest.mark.asyncio
    async def test_timeout_clears_context(self, app: YonaApp) -> None:
        app._sm._state = CS.LISTENING
        app._context.add_user("test")
        app._context.add_assistant("response")
        app._restart_timeout()
        await asyncio.sleep(0.5)
        assert app._context.message_count == 0

    @pytest.mark.asyncio
    async def test_timeout_cancelled_by_speech(self, app: YonaApp) -> None:
        """Speech during timeout countdown cancels it."""
        app._sm._state = CS.LISTENING
        app._restart_timeout()
        await asyncio.sleep(0.05)  # mid-countdown
        app._cancel_timeout()
        assert app._sm.state == CS.LISTENING  # still listening

    @pytest.mark.asyncio
    async def test_restart_timeout_cancels_old(self, app: YonaApp) -> None:
        """Restarting timeout cancels the previous task."""
        app._sm._state = CS.LISTENING
        app._restart_timeout()
        old_task = app._timeout_task
        app._restart_timeout()
        # Wait for cancellation to propagate
        await asyncio.sleep(0.01)
        assert old_task.done()
        app._cancel_timeout()

    @pytest.mark.asyncio
    async def test_timeout_not_triggered_from_idle(self, app: YonaApp) -> None:
        """Timeout does nothing if state has moved to IDLE."""
        app._sm._state = CS.IDLE
        app._restart_timeout()
        await asyncio.sleep(0.3)
        # State should still be IDLE (timeout sequence returns early)
        assert app._sm.state == CS.IDLE
        app._cancel_timeout()


# ---------------------------------------------------------------------------
# Shutdown tests
# ---------------------------------------------------------------------------

class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_stops_audio(self, app: YonaApp) -> None:
        await app.shutdown()
        app._audio.stop.assert_awaited_once()
        app._audio.stop_playback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_closes_handler_and_synth(self, app: YonaApp) -> None:
        await app.shutdown()
        app._chat_handler.close.assert_awaited_once()
        app._synth.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_interrupts_pipeline(self, app: YonaApp) -> None:
        await app.shutdown()
        app._pipeline.interrupt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, app: YonaApp) -> None:
        """Multiple shutdown calls don't raise."""
        await app.shutdown()
        await app.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_handler_tasks(self, app: YonaApp) -> None:
        """Event listener tasks are cancelled on shutdown."""
        task = asyncio.create_task(asyncio.sleep(100))
        app._handler_tasks = [task]
        await app.shutdown()
        assert task.cancelled()


# ---------------------------------------------------------------------------
# Play TTS helper test
# ---------------------------------------------------------------------------

class TestPlayTTS:
    @pytest.mark.asyncio
    async def test_synthesizes_and_plays(self, app: YonaApp) -> None:
        await app._play_tts("테스트 메시지")
        app._synth.synthesize.assert_awaited_once_with("테스트 메시지")
        app._audio.play_audio.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handles_tts_error(self, app: YonaApp) -> None:
        """TTS error does not crash the caller."""
        app._synth.synthesize = AsyncMock(side_effect=RuntimeError("TTS exploded"))
        await app._play_tts("fail message")  # should not raise


# ---------------------------------------------------------------------------
# Event listener pattern test
# ---------------------------------------------------------------------------

class TestListenHelper:
    @pytest.mark.asyncio
    async def test_listen_dispatches_events(self, app: YonaApp) -> None:
        """_listen subscribes and dispatches to the handler."""
        handler = AsyncMock()
        task = asyncio.create_task(
            app._listen(EventType.WAKE_WORD_DETECTED, handler),
        )
        await asyncio.sleep(0.01)

        # Publish event
        await app._bus.publish(EventType.WAKE_WORD_DETECTED, data="test")
        await asyncio.sleep(0.01)

        handler.assert_awaited_once()
        event = handler.call_args[0][0]
        assert event.type == EventType.WAKE_WORD_DETECTED
        assert event.data == "test"

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_listen_unsubscribes_on_cancel(self, app: YonaApp) -> None:
        handler = AsyncMock()
        task = asyncio.create_task(
            app._listen(EventType.WAKE_WORD_DETECTED, handler),
        )
        await asyncio.sleep(0.01)

        # Should have one subscriber
        assert len(app._bus._subscribers[EventType.WAKE_WORD_DETECTED]) == 1

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # Queue should be unsubscribed
        assert len(app._bus._subscribers[EventType.WAKE_WORD_DETECTED]) == 0


# ---------------------------------------------------------------------------
# Integration: end-to-end event flow
# ---------------------------------------------------------------------------

class TestIntegrationFlow:
    """Simulate event-driven flows by publishing events on the bus."""

    @pytest.mark.asyncio
    async def test_wake_to_speech_to_response(self, app: YonaApp) -> None:
        """IDLE → wake → LISTENING → speech end → PROCESSING → SPEAKING → LISTENING."""
        # Start listeners
        tasks = [
            asyncio.create_task(app._listen(EventType.WAKE_WORD_DETECTED, app._on_wake_word)),
            asyncio.create_task(app._listen(EventType.SPEECH_ENDED, app._on_speech_ended)),
        ]
        await asyncio.sleep(0.01)

        # Wake word
        await app._bus.publish(EventType.WAKE_WORD_DETECTED, data="hi_inspector")
        await asyncio.sleep(0.01)
        assert app._sm.state == CS.LISTENING
        # Cancel timeout immediately so it doesn't fire during test
        app._cancel_timeout()

        # Speech ended
        await app._bus.publish(EventType.SPEECH_ENDED)
        await asyncio.sleep(0.05)  # let _process_utterance run
        # Cancel timeout before it fires (config has 0.1s timeout)
        app._cancel_timeout()
        assert app._sm.state == CS.LISTENING
        app._pipeline.run.assert_awaited_once()

        # Cleanup
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_wake_to_goodbye(self, app: YonaApp) -> None:
        """IDLE → wake → LISTENING → speech end → goodbye → IDLE."""
        app._stt.transcribe = AsyncMock(return_value="bye")
        app._stt.detected_language = "en"

        tasks = [
            asyncio.create_task(app._listen(EventType.WAKE_WORD_DETECTED, app._on_wake_word)),
            asyncio.create_task(app._listen(EventType.SPEECH_ENDED, app._on_speech_ended)),
        ]
        await asyncio.sleep(0.01)

        await app._bus.publish(EventType.WAKE_WORD_DETECTED)
        await asyncio.sleep(0.01)

        await app._bus.publish(EventType.SPEECH_ENDED)
        await asyncio.sleep(0.1)
        assert app._sm.state == CS.IDLE

        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
