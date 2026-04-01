"""pipeline.py — Streaming LLM→TTS→Speaker pipeline for Samsung Gauss.

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
import time
from typing import TYPE_CHECKING

import numpy as np

from src.events import EventBus, EventType

if TYPE_CHECKING:
    from src.audio import AudioManager
    from src.llm import ChatHandler, ConversationContext
    from src.tts import Synthesizer

logger = logging.getLogger(__name__)

# Phrase-splitting at natural pause points:
#   Sentence end:   .!? + whitespace, CJK 。！？ (immediate)
#   Clause boundary: ,;: + whitespace, CJK ，；：ㆍ (immediate)
#   Dash pause:     — or – (em/en dash)
#   Line break:     one or more newlines
_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?<=[。！？])"
    r"|(?<=[,;:])\s+"
    r"|(?<=[，；：])"
    r"|(?<=[—–])\s*"
    r"|\n+"
)

# STT language code → TTS language code (Supertonic uses lowercase)
_LANG_MAP: dict[str, str] = {
    "ko": "ko",
    "en": "en",
    "es": "es",
    "pt": "pt",
    "fr": "fr",
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

    Args:
        min_length: Minimum character count before a phrase is emitted.
            Short phrases (e.g. "안녕하세요!") produce short audio, so
            when TTS RTF > 1.0x the playback finishes before the next
            chunk is ready — causing audible silence gaps.  Setting
            *min_length* ≥ 30 encourages merging short sentences into
            longer chunks that provide enough playback overlap for the
            TTS worker to stay ahead.
    """

    def __init__(self, min_length: int = 0) -> None:
        self._buffer = ""
        self._min_length = min_length

    def feed(self, token: str) -> list[str]:
        """Feed a token and return any complete phrases.

        Returns an empty list when no sentence boundary has been reached yet.
        """
        self._buffer += token
        parts = _SPLIT_RE.split(self._buffer)
        if len(parts) <= 1:
            return []
        # All but last are complete phrases; last stays in buffer
        candidates = [p for p in parts[:-1] if p.strip()]
        self._buffer = parts[-1]

        if not candidates:
            return []

        # Merge short leading candidates so the first emitted phrase
        # produces enough audio to overlap with the next TTS call.
        if self._min_length <= 0:
            return candidates

        merged: list[str] = []
        acc = ""
        for c in candidates:
            if acc:
                acc += " " + c
            else:
                acc = c
            if len(acc) >= self._min_length:
                merged.append(acc)
                acc = ""
        # If leftover is below min_length, push it back to buffer
        # so it gets merged with future tokens.
        if acc:
            if self._buffer:
                self._buffer = acc + " " + self._buffer
            else:
                self._buffer = acc
        return merged

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

    #: Per-language minimum phrase lengths (chars).  With clause-level
    #: splitting (commas, colons, etc.) short fragments are common;
    #: these minimums merge them into natural-sounding chunks before
    #: TTS synthesis.  Values are set so that 2-3 short clauses merge
    #: into one TTS call (better prosody, fewer inter-phrase gaps).
    _MIN_PHRASE_BY_LANG: dict[str, int] = {"ko": 15, "en": 30}
    DEFAULT_MIN_PHRASE_LENGTH = 15

    def __init__(
        self,
        chat_handler: ChatHandler,
        synthesizer: Synthesizer,
        audio_manager: AudioManager,
        bus: EventBus,
        min_phrase_length: int | None = None,
    ) -> None:
        self._handler = chat_handler
        self._synth = synthesizer
        self._audio = audio_manager
        self._bus = bus
        self._interrupted = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._full_response = ""
        self._min_phrase_length = (
            min_phrase_length
            if min_phrase_length is not None
            else self.DEFAULT_MIN_PHRASE_LENGTH
        )

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

        # Dynamic TTS language switch — runs in parallel with the LLM
        # request so the language-switch latency (~15 s on Jetson when the
        # BERT G2P model needs to be reloaded) does not delay the LLM
        # network round-trip.  The TTS worker waits on _tts_ready before
        # synthesising the first phrase.
        self._tts_ready: asyncio.Event = asyncio.Event()
        if detected_language and hasattr(self._synth, "set_language"):
            lang_task = asyncio.create_task(
                self._switch_tts_language_and_signal(detected_language)
            )
        else:
            self._tts_ready.set()
            lang_task = None

        phrase_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=5)
        audio_queue: asyncio.Queue[tuple[np.ndarray, int] | None] = asyncio.Queue(maxsize=10)

        self._tasks = [
            asyncio.create_task(self._llm_worker(context, phrase_queue, detected_language)),
            asyncio.create_task(self._tts_worker(phrase_queue, audio_queue)),
            asyncio.create_task(self._playback_worker(audio_queue)),
        ]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

        # Ensure the language task is awaited even if the pipeline was
        # interrupted, to avoid unawaited-coroutine warnings.
        if lang_task is not None and not lang_task.done():
            lang_task.cancel()
            try:
                await lang_task
            except (asyncio.CancelledError, Exception):
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
        detected_language: str | None = None,
    ) -> None:
        """Stream LLM tokens → PhraseAccumulator → phrase_queue."""
        # Use language-specific min_length when no explicit override was given
        if self._min_phrase_length != self.DEFAULT_MIN_PHRASE_LENGTH or detected_language is None:
            min_len = self._min_phrase_length
        else:
            min_len = self._MIN_PHRASE_BY_LANG.get(
                detected_language.lower(), self._min_phrase_length,
            )
        acc = PhraseAccumulator(min_length=min_len)
        logger.info("LLM worker: lang=%s min_phrase_length=%d", detected_language, min_len)
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
            # Note: the LLM handler also publishes this on normal completion,
            # so subscribers may see it twice — this is intentional for safety.
            await self._bus.publish(EventType.LLM_RESPONSE_DONE)
            await phrase_queue.put(None)

    async def _tts_worker(
        self,
        phrase_queue: asyncio.Queue[str | None],
        audio_queue: asyncio.Queue[tuple[np.ndarray, int] | None],
    ) -> None:
        """Read phrases → synthesize audio → audio_queue.

        Waits for ``_tts_ready`` before synthesising the first phrase so
        that a concurrent language switch (which reloads the BERT model)
        can finish while the LLM is already streaming.
        """
        try:
            first = True
            while True:
                phrase = await phrase_queue.get()
                if phrase is None or self._interrupted.is_set():
                    break
                # Wait for language switch to complete before first synthesis
                if first:
                    await self._tts_ready.wait()
                    first = False
                try:
                    t0 = time.monotonic()
                    audio, sr = await self._synth.synthesize(phrase)
                    synth_dur = time.monotonic() - t0
                    audio_dur = len(audio) / sr
                    logger.info(
                        "TTS phrase done: synth=%.2fs audio=%.2fs RTF=%.2fx  \"%s\"",
                        synth_dur, audio_dur,
                        synth_dur / audio_dur if audio_dur > 0 else 0,
                        phrase[:40],
                    )
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
            idx = 0
            while True:
                t_wait = time.monotonic()
                item = await audio_queue.get()
                wait_dur = time.monotonic() - t_wait
                if item is None or self._interrupted.is_set():
                    break
                audio, sr = item
                audio_dur = len(audio) / sr
                logger.info(
                    "Playback[%d] start: waited=%.2fs audio=%.2fs",
                    idx, wait_dur, audio_dur,
                )
                t0 = time.monotonic()
                await self._audio.play_audio(audio, sr)
                play_dur = time.monotonic() - t0
                logger.info(
                    "Playback[%d] done: play=%.2fs (expected=%.2fs)",
                    idx, play_dur, audio_dur,
                )
                idx += 1
        except asyncio.CancelledError:
            raise
        finally:
            await self._bus.publish(EventType.PLAYBACK_DONE)

    # ------------------------------------------------------------------
    # Language switching
    # ------------------------------------------------------------------

    async def _switch_tts_language_and_signal(self, language: str) -> None:
        """Switch TTS language and signal ``_tts_ready`` when done.

        Runs concurrently with the LLM worker so that network round-trip
        and language-switch I/O overlap.  The TTS worker waits on
        ``_tts_ready`` before its first synthesis call.
        """
        try:
            await self._switch_tts_language(language)
        finally:
            self._tts_ready.set()

    async def _switch_tts_language(self, language: str) -> None:
        """Switch TTS language if the synthesizer supports it."""
        if not hasattr(self._synth, "set_language"):
            return
        tts_lang = _LANG_MAP.get(language.lower())
        if tts_lang is None:
            logger.debug("No TTS mapping for language %r, skipping", language)
            return
        try:
            await self._synth.set_language(tts_lang)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Failed to switch TTS language to %s: %s", tts_lang, exc)
