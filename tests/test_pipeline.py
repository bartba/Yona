"""Tests for src/pipeline.py — PhraseAccumulator + StreamingPipeline."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# --- Mock heavy TTS/LLM imports before importing src modules ---
sys.modules.setdefault("melo", MagicMock())
sys.modules.setdefault("melo.api", MagicMock())
sys.modules.setdefault("openai", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("httpx", MagicMock())
sys.modules.setdefault("sounddevice", MagicMock())

from src.events import EventBus, EventType  # noqa: E402
from src.pipeline import PhraseAccumulator, StreamingPipeline, _LANG_MAP  # noqa: E402


# =========================================================================
# PhraseAccumulator tests
# =========================================================================


class TestPhraseAccumulator:
    """PhraseAccumulator sentence-boundary detection."""

    def test_no_boundary_yet(self):
        acc = PhraseAccumulator()
        assert acc.feed("Hello") == []
        assert acc.feed(" world") == []

    def test_single_sentence_with_trailing_space(self):
        acc = PhraseAccumulator()
        phrases = acc.feed("Hello world. ")
        assert phrases == ["Hello world."]

    def test_streaming_tokens(self):
        """Tokens arrive one-by-one; phrase emitted when boundary + space seen."""
        acc = PhraseAccumulator()
        assert acc.feed("Hello") == []
        assert acc.feed(" world") == []
        assert acc.feed(".") == []  # no trailing whitespace yet
        phrases = acc.feed(" How")
        assert phrases == ["Hello world."]
        # "How" remains in buffer
        assert acc.flush() == "How"

    def test_multiple_sentences(self):
        acc = PhraseAccumulator()
        phrases = acc.feed("First sentence. Second sentence. Third")
        assert phrases == ["First sentence.", "Second sentence."]
        assert acc.flush() == "Third"

    def test_question_mark(self):
        acc = PhraseAccumulator()
        phrases = acc.feed("How are you? I am fine.")
        assert phrases == ["How are you?"]
        assert acc.flush() == "I am fine."

    def test_exclamation_mark(self):
        acc = PhraseAccumulator()
        phrases = acc.feed("Wow! That is great. ")
        assert phrases == ["Wow!", "That is great."]

    def test_cjk_period(self):
        """CJK period 。 splits without requiring trailing whitespace."""
        acc = PhraseAccumulator()
        phrases = acc.feed("안녕하세요。반갑습니다")
        assert phrases == ["안녕하세요。"]
        assert acc.flush() == "반갑습니다"

    def test_fullwidth_punctuation(self):
        """Fullwidth ！ and ？ split without whitespace."""
        acc = PhraseAccumulator()
        phrases = acc.feed("すごい！本当？はい")
        assert phrases == ["すごい！", "本当？"]
        assert acc.flush() == "はい"

    def test_korean_with_period(self):
        acc = PhraseAccumulator()
        phrases = acc.feed("안녕하세요. 반갑습니다. ")
        assert phrases == ["안녕하세요.", "반갑습니다."]

    def test_flush_empty(self):
        acc = PhraseAccumulator()
        assert acc.flush() is None

    def test_flush_with_remaining(self):
        acc = PhraseAccumulator()
        acc.feed("Hello world")
        assert acc.flush() == "Hello world"
        assert acc.flush() is None  # buffer cleared

    def test_reset(self):
        acc = PhraseAccumulator()
        acc.feed("Some text")
        acc.reset()
        assert acc.flush() is None

    def test_newline_after_period(self):
        """Newline counts as whitespace for sentence splitting."""
        acc = PhraseAccumulator()
        phrases = acc.feed("First.\nSecond.\n")
        assert phrases == ["First.", "Second."]

    def test_ellipsis(self):
        """Ellipsis (...) followed by space splits after the last dot."""
        acc = PhraseAccumulator()
        phrases = acc.feed("Hmm... Let me think. ")
        assert len(phrases) == 2
        assert phrases[1] == "Let me think."


# =========================================================================
# Helper mocks
# =========================================================================


class MockChatHandler:
    """Async-iterator chat handler for testing."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    async def stream(self, context):  # noqa: ANN001
        for token in self._tokens:
            yield token

    async def close(self) -> None:
        pass


class MockSynthesizer:
    """Synthesizer that returns a short silent array."""

    def __init__(self) -> None:
        self.synthesized: list[str] = []
        self._language: str | None = None

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.synthesized.append(text)
        return np.zeros(100, dtype=np.float32), 24000

    async def set_language(self, language: str) -> None:
        self._language = language

    async def close(self) -> None:
        pass


class MockAudioManager:
    """AudioManager that records played chunks."""

    def __init__(self) -> None:
        self.played: list[tuple[np.ndarray, int]] = []

    async def play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        self.played.append((audio, sample_rate))

    async def stop_playback(self) -> None:
        pass


# =========================================================================
# StreamingPipeline tests
# =========================================================================


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def synth():
    return MockSynthesizer()


@pytest.fixture
def audio():
    return MockAudioManager()


class TestStreamingPipeline:
    """StreamingPipeline end-to-end tests."""

    @pytest.mark.asyncio
    async def test_full_run(self, bus, synth, audio):
        """Tokens → phrases → TTS → playback, full response returned."""
        handler = MockChatHandler(["Hello world. ", "How are you?"])
        ctx = MagicMock()
        pipeline = StreamingPipeline(handler, synth, audio, bus)

        response = await pipeline.run(ctx)

        assert response == "Hello world. How are you?"
        # Two phrases: "Hello world." and "How are you?"
        assert len(synth.synthesized) == 2
        assert synth.synthesized[0] == "Hello world."
        assert synth.synthesized[1] == "How are you?"
        assert len(audio.played) == 2

    @pytest.mark.asyncio
    async def test_single_phrase_no_boundary(self, bus, synth, audio):
        """When LLM output has no sentence boundary, flush emits the text."""
        handler = MockChatHandler(["Hello world"])
        ctx = MagicMock()
        pipeline = StreamingPipeline(handler, synth, audio, bus)

        response = await pipeline.run(ctx)

        assert response == "Hello world"
        assert synth.synthesized == ["Hello world"]
        assert len(audio.played) == 1

    @pytest.mark.asyncio
    async def test_events_published(self, bus, synth, audio):
        """Pipeline publishes PHRASE_READY, AUDIO_CHUNK_READY, PLAYBACK_*, LLM_RESPONSE_DONE."""
        handler = MockChatHandler(["Hi. Bye."])
        ctx = MagicMock()

        # Subscribe to events
        phrase_q = bus.subscribe(EventType.PHRASE_READY)
        audio_q = bus.subscribe(EventType.AUDIO_CHUNK_READY)
        start_q = bus.subscribe(EventType.PLAYBACK_STARTED)
        done_q = bus.subscribe(EventType.PLAYBACK_DONE)
        llm_done_q = bus.subscribe(EventType.LLM_RESPONSE_DONE)

        pipeline = StreamingPipeline(handler, synth, audio, bus)
        await pipeline.run(ctx)

        # PHRASE_READY events
        phrases = []
        while not phrase_q.empty():
            ev = phrase_q.get_nowait()
            phrases.append(ev.data)
        assert "Hi." in phrases
        assert "Bye." in phrases

        # AUDIO_CHUNK_READY events
        audio_events = 0
        while not audio_q.empty():
            audio_q.get_nowait()
            audio_events += 1
        assert audio_events == 2

        # PLAYBACK_STARTED / PLAYBACK_DONE
        assert not start_q.empty()
        assert not done_q.empty()

        # LLM_RESPONSE_DONE always published (even on barge-in)
        assert not llm_done_q.empty()

    @pytest.mark.asyncio
    async def test_interrupt(self, bus, synth, audio):
        """interrupt() cancels workers and stops playback."""
        # Use slow tokens so we can interrupt mid-stream
        tokens = [f"word{i}. " for i in range(50)]
        handler = MockChatHandler(tokens)
        ctx = MagicMock()

        # Make TTS slow so phrases queue up
        original_synthesize = synth.synthesize

        async def slow_synthesize(text):
            await asyncio.sleep(0.05)
            return await original_synthesize(text)

        synth.synthesize = slow_synthesize

        pipeline = StreamingPipeline(handler, synth, audio, bus)

        async def interrupt_soon():
            await asyncio.sleep(0.1)
            await pipeline.interrupt()

        asyncio.create_task(interrupt_soon())
        response = await pipeline.run(ctx)

        # Pipeline was interrupted — not all 50 phrases should be synthesized
        assert len(synth.synthesized) < 50

    @pytest.mark.asyncio
    async def test_interrupt_idempotent(self, bus, synth, audio):
        """Calling interrupt() twice does not raise."""
        handler = MockChatHandler(["Hello."])
        ctx = MagicMock()
        pipeline = StreamingPipeline(handler, synth, audio, bus)

        await pipeline.run(ctx)
        # After run completes, interrupt should be a no-op
        await pipeline.interrupt()
        await pipeline.interrupt()

    @pytest.mark.asyncio
    async def test_language_switch(self, bus, synth, audio):
        """detected_language triggers set_language on the synthesizer."""
        handler = MockChatHandler(["안녕하세요."])
        ctx = MagicMock()
        pipeline = StreamingPipeline(handler, synth, audio, bus)

        await pipeline.run(ctx, detected_language="ko")

        assert synth._language == "KR"

    @pytest.mark.asyncio
    async def test_language_switch_english(self, bus, synth, audio):
        handler = MockChatHandler(["Hello."])
        ctx = MagicMock()
        pipeline = StreamingPipeline(handler, synth, audio, bus)

        await pipeline.run(ctx, detected_language="en")

        assert synth._language == "EN"

    @pytest.mark.asyncio
    async def test_language_switch_unknown(self, bus, synth, audio):
        """Unknown language code does not call set_language."""
        handler = MockChatHandler(["Test."])
        ctx = MagicMock()
        pipeline = StreamingPipeline(handler, synth, audio, bus)

        await pipeline.run(ctx, detected_language="xx")

        assert synth._language is None  # no switch happened

    @pytest.mark.asyncio
    async def test_no_language_switch_without_set_language(self, bus, audio):
        """Synthesizer without set_language attribute is handled gracefully."""

        class PlainSynth:
            async def synthesize(self, text):
                return np.zeros(10, dtype=np.float32), 24000

            async def close(self):
                pass

        handler = MockChatHandler(["Hello."])
        ctx = MagicMock()
        pipeline = StreamingPipeline(handler, PlainSynth(), audio, bus)

        # Should not raise
        await pipeline.run(ctx, detected_language="ko")

    @pytest.mark.asyncio
    async def test_full_response_property(self, bus, synth, audio):
        handler = MockChatHandler(["One. ", "Two."])
        ctx = MagicMock()
        pipeline = StreamingPipeline(handler, synth, audio, bus)

        result = await pipeline.run(ctx)

        assert pipeline.full_response == result
        assert result == "One. Two."

    @pytest.mark.asyncio
    async def test_tts_error_does_not_crash_pipeline(self, bus, audio):
        """If TTS fails on one phrase, pipeline continues with next phrases."""

        call_count = 0

        class FailOnceSynth:
            async def synthesize(self, text):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("TTS exploded")
                return np.zeros(50, dtype=np.float32), 24000

            async def close(self):
                pass

        handler = MockChatHandler(["First. ", "Second. "])
        ctx = MagicMock()

        error_q = bus.subscribe(EventType.ERROR)
        pipeline = StreamingPipeline(handler, FailOnceSynth(), audio, bus)
        await pipeline.run(ctx)

        # Error event published for the failed phrase
        assert not error_q.empty()
        # Second phrase still played
        assert len(audio.played) == 1

    @pytest.mark.asyncio
    async def test_empty_llm_response(self, bus, synth, audio):
        """Empty LLM response produces no TTS or playback."""
        handler = MockChatHandler([])
        ctx = MagicMock()
        pipeline = StreamingPipeline(handler, synth, audio, bus)

        response = await pipeline.run(ctx)

        assert response == ""
        assert synth.synthesized == []
        assert audio.played == []


class TestLangMap:
    """Verify language code mapping."""

    def test_known_languages(self):
        assert _LANG_MAP["ko"] == "KR"
        assert _LANG_MAP["en"] == "EN"
        assert _LANG_MAP["ja"] == "JP"
        assert _LANG_MAP["zh"] == "ZH"

    def test_unknown_language(self):
        assert _LANG_MAP.get("xx") is None
