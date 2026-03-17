"""stt.py — Speech-to-Text transcription for Yona.

Transcriber
    Wraps the faster-whisper WhisperModel (large-v3-turbo, CUDA, float16).
    Accepts a numpy float32 audio array, runs inference in a thread-pool
    executor (so the asyncio event loop is never blocked), and publishes
    EventType.TRANSCRIPTION_READY with the transcribed text.

    Language is either fixed (``stt.language`` in config) or auto-detected
    per utterance when the config value is ``null`` / ``None``.

Usage::

    from src.config import Config
    from src.events import EventBus
    from src.stt import Transcriber

    transcriber = Transcriber(cfg, bus)
    text = await transcriber.transcribe(audio_array, sample_rate=16_000)
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from faster_whisper import WhisperModel

from src.config import Config
from src.events import EventBus, EventType

logger = logging.getLogger(__name__)


class Transcriber:
    """faster-whisper wrapper with async interface and EventBus integration.

    Loads the model at construction time (blocking, done once at startup).
    Transcription is offloaded to a worker thread via ``asyncio.to_thread``
    so the event loop stays responsive during GPU inference.

    Args:
        cfg: Application config — reads ``stt.*`` section.
        bus: Shared :class:`EventBus` for publishing transcription events.
    """

    def __init__(self, cfg: Config, bus: EventBus) -> None:
        self._bus = bus

        model_size: str = cfg.get("stt.model_size", "large-v3-turbo")
        device: str = cfg.get("stt.device", "cuda")
        compute_type: str = cfg.get("stt.compute_type", "float16")
        self._language: str | None = cfg.get("stt.language", None)
        self._detected_language: str | None = None

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def detected_language(self) -> str | None:
        """Language code detected in the most recent transcription."""
        return self._detected_language

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe *audio* and publish the result.

        Runs WhisperModel inference in a thread-pool executor so the
        asyncio event loop is not blocked during GPU inference.

        Publishes :data:`EventType.TRANSCRIPTION_READY` with the text as
        *data* when the transcription is non-empty.  Empty results (silence
        or noise) are returned without publishing an event.

        Args:
            audio: Mono float32 PCM array at 16 000 Hz.

        Returns:
            Stripped transcription text, or ``""`` if nothing was detected.
        """
        audio = np.asarray(audio, dtype=np.float32).ravel()
        text: str = await asyncio.to_thread(self._run_transcribe, audio)

        if text:
            await self._bus.publish(EventType.TRANSCRIPTION_READY, data=text)

        return text

    # ------------------------------------------------------------------
    # Internal (runs in thread-pool executor)
    # ------------------------------------------------------------------

    def _run_transcribe(self, audio: np.ndarray) -> str:
        """Blocking transcription — called from a thread-pool executor.

        Iterates over all Whisper segments and joins their text.  Returns
        an empty string if no speech was detected.

        Args:
            audio: Mono float32 array.

        Returns:
            Stripped concatenation of all segment texts.
        """
        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
        )
        text = "".join(seg.text for seg in segments).strip()
        self._detected_language = info.language

        logger.info(
            "STT | lang=%s (%.0f%%) | duration=%.2fs | text=%r",
            info.language,
            info.language_probability * 100,
            info.duration,
            text,
        )

        return text
