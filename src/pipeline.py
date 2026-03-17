"""pipeline.py — Streaming LLM→TTS→Speaker pipeline for Yona.

Provides:
  PhraseAccumulator   — sentence boundary detection on streaming tokens
  StreamingPipeline   — 3-worker producer-consumer (LLM, TTS, playback)

The pipeline connects the LLM token stream to the speaker via phrase-level
TTS synthesis, enabling low-latency audio output while the LLM is still
generating.  Barge-in support is built in via ``interrupt()``.

Usage::

    from src.llm import create_chat_handler, ConversationContext
    from src.tts import create_synthesizer
    from src.audio import AudioManager
    from src.events import EventBus
    from src.pipeline import StreamingPipeline

    pipeline = StreamingPipeline(handler, synth, audio_mgr, bus)
    response = await pipeline.run(context, detected_language="ko")
    # During playback, call pipeline.interrupt() for barge-in
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

import numpy as np

from src.events import EventBus, EventType

if TYPE_CHECKING:
    from src.audio import AudioManager
    from src.llm import ChatHandler, ConversationContext
    from src.tts import Synthesizer

logger = logging.getLogger(__name__)

# Sentence-ending punctuation:
#   ASCII .!? → require trailing whitespace to confirm sentence end
#   CJK 。！？ → split immediately (no whitespace needed in CJK text)
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")

# STT language code → MeloTTS language code
_LANG_MAP: dict[str, str] = {
    "ko": "KR",
    "en": "EN",
    "ja": "JP",
    "zh": "ZH",
}


# ---------------------------------------------------------------------------
# PhraseAccumulator
# ---------------------------------------------------------------------------

class PhraseAccumulator:
    """Accumulates LLM tokens and emits complete phrases at sentence boundaries.

    Tokens are fed one-by-one via :meth:`feed`.  When a sentence boundary is
    detected the completed phrase is returned immediately, allowing the TTS
    worker to start synthesis while the LLM is still generating.

    After the LLM stream ends, call :meth:`flush` to emit any remaining text.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, token: str) -> list[str]:
        """Feed a token and return any complete phrases.

        Returns an empty list when no sentence boundary has been reached yet.
        """
        self._buffer += token
        parts = _SPLIT_RE.split(self._buffer)
        if len(parts) <= 1:
            return []
        # All but last are complete phrases; last stays in buffer
        phrases = [p for p in parts[:-1] if p.strip()]
        self._buffer = parts[-1]
        return phrases

    def flush(self) -> str | None:
        """Return remaining buffered text, or *None* if empty."""
        text = self._buffer.strip()
        self._buffer = ""
        return text if text else None

    def reset(self) -> None:
        """Clear the internal buffer."""
        self._buffer = ""


# ---------------------------------------------------------------------------
# StreamingPipeline
# ---------------------------------------------------------------------------

class StreamingPipeline:
    """3-worker streaming pipeline: LLM → TTS → Speaker.

    Workers communicate via two ``asyncio.Queue`` instances:

    * **phrase_queue** (maxsize 5): LLM worker → TTS worker (``str | None``)
    * **audio_queue** (maxsize 10): TTS worker → playback worker
      (``tuple[np.ndarray, int] | None``)

    A ``None`` sentinel signals end-of-stream to the downstream worker.

    Args:
        chat_handler:  LLM chat handler (implements ``ChatHandler``).
        synthesizer:   TTS synthesizer (implements ``Synthesizer``).
        audio_manager: Audio I/O manager for playback.
        bus:           Shared event bus.
    """

    def __init__(
        self,
        chat_handler: ChatHandler,
        synthesizer: Synthesizer,
        audio_manager: AudioManager,
        bus: EventBus,
    ) -> None:
        self._handler = chat_handler
        self._synth = synthesizer
        self._audio = audio_manager
        self._bus = bus
        self._interrupted = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._full_response = ""

    @property
    def full_response(self) -> str:
        """The accumulated full LLM response text from the last run."""
        return self._full_response

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        context: ConversationContext,
        detected_language: str | None = None,
    ) -> str:
        """Run the full pipeline and return the complete LLM response.

        Args:
            context: Conversation context to send to the LLM.
            detected_language: STT-detected language code (e.g. ``"ko"``,
                ``"en"``).  If provided and the synthesizer supports dynamic
                language switching, the TTS language is changed before
                synthesis begins.

        Returns:
            The full LLM response text.
        """
        self._interrupted.clear()
        self._full_response = ""

        # Dynamic TTS language switch (MeloTTS only)
        if detected_language:
            await self._switch_tts_language(detected_language)

        phrase_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=5)
        audio_queue: asyncio.Queue[tuple[np.ndarray, int] | None] = asyncio.Queue(maxsize=10)

        self._tasks = [
            asyncio.create_task(self._llm_worker(context, phrase_queue)),
            asyncio.create_task(self._tts_worker(phrase_queue, audio_queue)),
            asyncio.create_task(self._playback_worker(audio_queue)),
        ]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

        self._tasks.clear()
        return self._full_response

    async def interrupt(self) -> None:
        """Barge-in: stop playback, cancel workers, drain queues.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._interrupted.is_set():
            return
        self._interrupted.set()
        logger.info("Pipeline interrupted (barge-in)")

        await self._audio.stop_playback()

        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    async def _llm_worker(
        self,
        context: ConversationContext,
        phrase_queue: asyncio.Queue[str | None],
    ) -> None:
        """Stream LLM tokens → PhraseAccumulator → phrase_queue."""
        acc = PhraseAccumulator()
        try:
            async for token in self._handler.stream(context):
                if self._interrupted.is_set():
                    break
                self._full_response += token
                for phrase in acc.feed(token):
                    logger.debug("Phrase ready: %s", phrase[:50])
                    await self._bus.publish(EventType.PHRASE_READY, data=phrase)
                    await phrase_queue.put(phrase)

            remaining = acc.flush()
            if remaining and not self._interrupted.is_set():
                logger.debug("Phrase flush: %s", remaining[:50])
                await self._bus.publish(EventType.PHRASE_READY, data=remaining)
                await phrase_queue.put(remaining)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("LLM worker error: %s", exc)
            await self._bus.publish(EventType.ERROR, data=exc)
        finally:
            # Always publish DONE so main.py subscribers never hang —
            # even on barge-in or error where the stream was cut short.
            await self._bus.publish(EventType.LLM_RESPONSE_DONE)
            await phrase_queue.put(None)

    async def _tts_worker(
        self,
        phrase_queue: asyncio.Queue[str | None],
        audio_queue: asyncio.Queue[tuple[np.ndarray, int] | None],
    ) -> None:
        """Read phrases → synthesize audio → audio_queue."""
        try:
            while True:
                phrase = await phrase_queue.get()
                if phrase is None or self._interrupted.is_set():
                    break
                try:
                    audio, sr = await self._synth.synthesize(phrase)
                    await self._bus.publish(
                        EventType.AUDIO_CHUNK_READY, data=(audio, sr),
                    )
                    await audio_queue.put((audio, sr))
                except Exception as exc:
                    logger.error("TTS error for phrase %r: %s", phrase[:30], exc)
                    await self._bus.publish(EventType.ERROR, data=exc)
        except asyncio.CancelledError:
            raise
        finally:
            await audio_queue.put(None)

    async def _playback_worker(
        self,
        audio_queue: asyncio.Queue[tuple[np.ndarray, int] | None],
    ) -> None:
        """Read audio chunks → play via AudioManager."""
        await self._bus.publish(EventType.PLAYBACK_STARTED)
        try:
            while True:
                item = await audio_queue.get()
                if item is None or self._interrupted.is_set():
                    break
                audio, sr = item
                await self._audio.play_audio(audio, sr)
        except asyncio.CancelledError:
            raise
        finally:
            await self._bus.publish(EventType.PLAYBACK_DONE)

    # ------------------------------------------------------------------
    # Language switching
    # ------------------------------------------------------------------

    async def _switch_tts_language(self, language: str) -> None:
        """Switch TTS language if the synthesizer supports it (MeloTTS)."""
        if not hasattr(self._synth, "set_language"):
            return
        melo_lang = _LANG_MAP.get(language.lower())
        if melo_lang is None:
            logger.debug("No MeloTTS mapping for language %r, skipping", language)
            return
        try:
            await self._synth.set_language(melo_lang)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Failed to switch TTS language to %s: %s", melo_lang, exc)
