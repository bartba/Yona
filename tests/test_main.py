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
      max_context_tokens: 250

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
    a._chime.play_processing = AsyncMock()

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
    a._context = ConversationContext(
        "Test system prompt",
        max_context_tokens=cfg.get("llm.max_context_tokens", 3000),
    )
    a._history = MagicMock()

    a._running = True
    return a


# ---------------------------------------------------------------------------
# Goodbye regex tests
# ---------------------------------------------------------------------------

class TestGoodbyeRegex:
    """Verify _GOODBYE_RE matches expected patterns."""

    @pytest.mark.parametrize("text", [
        # Legacy (no name)
        "바이바이",
        "bye bye",
        "bye-bye",
        "byebye",
        "Bye Bye",
        # Korean + name: 바이바이/빠이빠이 계열 (공백 포함/미포함)
        "바이바이 맥",
        "바이바이맥",
        "바이 바이 맥",
        "빠이빠이 맥",
        "빠이 빠이 맥",
        # Korean + name: 굿/굳 + 바/빠 조합 (받침 ㅅ↔ㄷ, 된소리 변형)
        "굿바이 맥",
        "굳바이 맥",
        "굿빠이 맥",
        "굳빠이 맥",
        # Korean + name: 음절 사이 공백 허용
        "굿 바이 맥",
        "굳 빠이 맥",
        "굿 바이맥",
        # Korean + name: STT 오인식 변형 (멕)
        "굿바이 멕",
        "굳바이 멕",
        "바이바이 멕",
        # English + name: 정확 (mack)
        "bye bye mack",
        "Bye Bye Mack",
        "goodbye mack",
        "good bye mack",
        "good-bye mack",
        "Goodbye Mack",
        # English + name: STT 구두점 삽입 변형
        "Goodbye, Mac.",
        "Goodbye, Mack.",
        "bye bye, mac",
        "Good bye. Mac",
        # English + name: Mack→Meg STT 오인식
        "Bye-bye, Meg.",
        "goodbye meg",
        "bye bye meg",
        # English + name: STT 오인식 변형 (mac, man)
        "bye bye mac",
        "goodbye mac",
        "bye bye man",
        "goodbye man",
        "Bye Bye Man",
    ])
    def test_matches_goodbye(self, text: str) -> None:
        assert _GOODBYE_RE.search(text) is not None

    @pytest.mark.parametrize("text", [
        "안녕하세요",
        "안녕",
        "hello",
        "how are you",
        "오늘 날씨 어때",
        "bye",
        "goodbye",
        "goodbye everyone",
        "종료",
        "그만",
        "맥 알려줘",
        "굿바이",        # 이름 없는 굿바이 — 오발동 방지
    ])
    def test_non_goodbye(self, text: str) -> None:
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
    async def test_cancels_timeout_in_listening(self, app: YonaApp) -> None:
        """SPEECH_STARTED cancels timeout (not restart) — prevents timeout during long speech."""
        app._sm._state = CS.LISTENING
        old_task = MagicMock()
        old_task.done.return_value = False
        app._timeout_task = old_task
        await app._on_speech_started(Event(type=EventType.SPEECH_STARTED))
        old_task.cancel.assert_called_once()
        # Timeout should be cancelled, not restarted
        assert app._timeout_task is None

    @pytest.mark.asyncio
    async def test_timeout_check_returns_to_listening(self, app: YonaApp) -> None:
        # Transition: IDLE → LISTENING → TIMEOUT_CHECK
        app._sm._state = CS.TIMEOUT_CHECK
        await app._on_speech_started(Event(type=EventType.SPEECH_STARTED))
        assert app._sm.state == CS.LISTENING
        app._audio.stop_playback.assert_awaited_once()
        app._buffer.reset.assert_called()
        # Timeout not restarted — resumes after _process_utterance
        assert app._timeout_task is None


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
    async def test_plays_processing_chime(self, app: YonaApp) -> None:
        app._sm._state = CS.LISTENING
        await app._on_speech_ended(Event(type=EventType.SPEECH_ENDED))
        app._chime.play_processing.assert_awaited_once()
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
        app._stt.transcribe = AsyncMock(return_value="바이바이")
        app._stt.detected_language = "ko"

        await app._process_utterance()

        app._pipeline.run.assert_not_awaited()
        # Farewell TTS
        app._synth.synthesize.assert_awaited_once()
        assert app._sm.state == CS.IDLE
        app._vad.reset.assert_called()
        app._wake.reset.assert_called()

    @pytest.mark.asyncio
    async def test_goodbye_uses_last_conversation_lang(self, app: YonaApp) -> None:
        """Farewell TTS uses last conversation language, not farewell STT language."""
        app._sm._state = CS.PROCESSING
        # Previous conversation was in Korean
        app._last_conversation_lang = "ko"
        # But farewell is detected as English by STT
        app._stt.transcribe = AsyncMock(return_value="Goodbye, Mac.")
        app._stt.detected_language = "en"

        await app._process_utterance()

        app._pipeline.run.assert_not_awaited()
        # Should synthesize Korean goodbye message
        synthesize_call_args = app._synth.synthesize.call_args[0][0]
        goodbye_messages = app._cfg.get("conversation.goodbye_message", {})
        assert synthesize_call_args == goodbye_messages.get("ko")

    @pytest.mark.asyncio
    async def test_goodbye_clears_context(self, app: YonaApp) -> None:
        app._sm._state = CS.PROCESSING
        app._context.add_user("이전 대화")
        app._context.add_assistant("이전 응답")
        app._stt.transcribe = AsyncMock(return_value="bye bye")
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
        app._stt.transcribe = AsyncMock(return_value="bye bye")
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


# ---------------------------------------------------------------------------
# Context compression tests
# ---------------------------------------------------------------------------

def _fill_context(ctx: ConversationContext, turns: int) -> None:
    """Add *turns* user+assistant pairs to the context."""
    for i in range(1, turns + 1):
        ctx.add_user(f"사용자 메시지 {i}")
        ctx.add_assistant(f"어시스턴트 응답 {i}")


async def _mock_stream_summary(*_args, **_kwargs):
    """Async generator that yields a fake summary."""
    for token in ["대화", " 요약:", " 사용자가", " 여러", " 주제를", " 논의했습니다."]:
        yield token


class TestContextCompression:
    """Verify conversation context compression via _compress_context."""

    @pytest.mark.asyncio
    async def test_compression_triggers_at_threshold(self, app: YonaApp):
        """After 17 normal turns, turn 18 should trigger compression."""
        _fill_context(app._context, 17)
        assert app._context.message_count == 34
        assert not app._context.needs_compression

        # Turn 18 triggers compression
        app._context.add_user("18번째 질문")
        app._context.add_assistant("18번째 응답")
        assert app._context.needs_compression

    @pytest.mark.asyncio
    async def test_compress_context_reduces_messages(self, app: YonaApp):
        """_compress_context should remove oldest third and store a summary."""
        _fill_context(app._context, 18)
        assert app._context.needs_compression

        # Mock the LLM stream for summarisation
        app._chat_handler.stream = _mock_stream_summary
        await app._compress_context("ko")

        # Oldest third removed (12 of 36 msgs), summary stored
        assert app._context.message_count == 24  # kept 2/3
        assert app._context._summary is not None
        assert "대화 요약" in app._context._summary

        # Summary appears in get_messages()
        msgs = app._context.get_messages()
        assert msgs[1]["role"] == "user"
        assert "[이전 대화 요약]" in msgs[1]["content"]
        assert msgs[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_compress_notifies_user_via_tts(self, app: YonaApp):
        """Compression should play start and done TTS notifications."""
        _fill_context(app._context, 18)
        app._chat_handler.stream = _mock_stream_summary

        await app._compress_context("ko")

        # Two TTS calls: start notification + done notification
        assert app._synth.synthesize.await_count == 2
        calls = [c.args[0] for c in app._synth.synthesize.await_args_list]
        assert "정리하고" in calls[0]
        assert "마쳤습니다" in calls[1]

    @pytest.mark.asyncio
    async def test_compress_failure_keeps_full_history(self, app: YonaApp):
        """If LLM summarisation fails, messages should not be lost."""
        _fill_context(app._context, 18)
        original_count = app._context.message_count

        async def _failing_stream(*_a, **_kw):
            raise RuntimeError("API error")
            yield  # noqa: unreachable — makes this an async generator

        app._chat_handler.stream = _failing_stream
        await app._compress_context("ko")

        assert app._context.message_count == original_count
        assert app._context._summary is None

    @pytest.mark.asyncio
    async def test_repeated_compression_chains_summaries(self, app: YonaApp):
        """Second compression should include the first summary."""
        # First compression
        _fill_context(app._context, 18)
        app._chat_handler.stream = _mock_stream_summary
        await app._compress_context("ko")
        first_summary = app._context._summary
        assert first_summary is not None

        # Add more turns until threshold again
        # After first compression: 24 messages. Need enough to exceed token budget again.
        remaining_turns = 18 - (app._context.message_count // 2)
        _fill_context(app._context, remaining_turns)
        assert app._context.needs_compression

        # Second compression — payload should include first summary
        old_summary, old_msgs = app._context.get_compression_payload()
        assert old_summary == first_summary

        await app._compress_context("ko")
        assert app._context._summary is not None

    @pytest.mark.asyncio
    async def test_barge_in_no_response_does_not_lose_history(self, app: YonaApp):
        """Repeated barge-in with no response should not shrink history."""
        _fill_context(app._context, 17)
        count_before = app._context.message_count  # 34

        # Simulate 3 barge-ins: add_user then pop
        for _ in range(3):
            app._context.add_user("barge-in 발화")
            app._context.pop_last_user()

        assert app._context.message_count == count_before

    @pytest.mark.asyncio
    async def test_compression_in_process_utterance(self, app: YonaApp):
        """Full _process_utterance flow should trigger compression at threshold."""
        _fill_context(app._context, 17)
        app._chat_handler.stream = _mock_stream_summary

        # Simulate: LISTENING → speech ended → process
        await app._sm.transition(CS.LISTENING)
        app._stt.transcribe = AsyncMock(return_value="18번째 질문입니다")
        app._pipeline.run = AsyncMock(return_value="18번째 응답입니다")

        await app._on_speech_ended(Event(EventType.SPEECH_ENDED))
        await asyncio.sleep(0.2)  # let _process_utterance run

        # Compression should have fired
        assert app._context._summary is not None
        # State should be LISTENING or TIMEOUT_CHECK (timeout may fire quickly in tests)
        assert app._sm.state in (CS.LISTENING, CS.TIMEOUT_CHECK)
