"""stt.py — Speech-to-Text transcription for Samsung Gauss.

Transcriber
    Wraps the faster-whisper WhisperModel (large-v3-turbo, CUDA, float16).
    Accepts a numpy float32 audio array, runs inference in a thread-pool
    executor (so the asyncio event loop is never blocked), and publishes
    EventType.TRANSCRIPTION_READY with the transcribed text.

    Language is either fixed (``stt.language`` in config) or auto-detected
    per utterance when the config value is ``null`` / ``None``.

    When auto-detection yields a language outside ``stt.allowed_languages``
    (default: ``[ko, en]``), a 2-pass re-transcription is performed with each
    allowed language forced in order.  The first result whose
    ``language_probability`` meets ``stt.lang_recheck_min_prob`` (default: 0.3)
    is adopted; if none qualify, the 1st-pass result is kept as fallback.

    This corrects short utterances like "바이바이 맥" that Whisper may
    mis-identify as Japanese or French due to insufficient phoneme information.

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
        self._beam_size: int = cfg.get("stt.beam_size", 1)
        self._detected_language: str | None = None

        # 2-pass language fallback: when auto-detect lands outside allowed list,
        # retry with each allowed language in order until min_prob is met.
        self._allowed_languages: list[str] = cfg.get("stt.allowed_languages", ["ko", "en"])
        self._lang_recheck_min_prob: float = cfg.get("stt.lang_recheck_min_prob", 0.3)

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        # Warm-up: minimal inference to trigger CUDA kernel compilation.
        # Keep the allocation small (1s silence, beam=1) so the memory
        # pool stays minimal — real inference will grow it gradually.
        # A large warm-up (15s/beam=3) pre-allocates ~1GB extra and
        # causes OOM on Jetson 8GB unified memory.
        logger.info("STT warm-up: compiling CUDA kernels...")
        dummy = np.zeros(16_000, dtype=np.float32)  # 1 second of silence
        segments, _ = self._model.transcribe(dummy, language="en", beam_size=1)
        _ = list(segments)  # exhaust the generator
        logger.info("STT warm-up done")

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

        When the auto-detected language is not in ``allowed_languages``,
        a 2-pass re-transcription is triggered with each allowed language
        forced in order.  The first result whose ``language_probability``
        meets ``lang_recheck_min_prob`` is adopted; otherwise the 1st-pass
        result is kept.

        Publishes :data:`EventType.TRANSCRIPTION_READY` with the text as
        *data* when the transcription is non-empty.  Empty results (silence
        or noise) are returned without publishing an event.

        Args:
            audio: Mono float32 PCM array at 16 000 Hz.

        Returns:
            Stripped transcription text, or ``""`` if nothing was detected.
        """
        audio = np.asarray(audio, dtype=np.float32).ravel()
        text, lang, lang_prob = await asyncio.to_thread(self._run_transcribe, audio)

        # 2-pass: retry with forced language when auto-detect is outside allowed list.
        # Fixed-language mode (self._language set) skips this — language is already forced.
        if self._language is None and lang not in self._allowed_languages:
            logger.info(
                "STT lang=%s (%.0f%%) not in allowed %s — retrying with forced languages",
                lang, lang_prob * 100, self._allowed_languages,
            )
            best_text, best_lang, best_prob = text, lang, lang_prob
            for forced_lang in self._allowed_languages:
                r_text, r_lang, r_prob = await asyncio.to_thread(
                    self._run_transcribe, audio, forced_lang,
                )
                if r_prob >= self._lang_recheck_min_prob:
                    text, lang, lang_prob = r_text, r_lang, r_prob
                    logger.info(
                        "STT recheck accepted: lang=%s (%.0f%%) text=%r",
                        lang, lang_prob * 100, text,
                    )
                    break
                if r_prob > best_prob:
                    best_text, best_lang, best_prob = r_text, r_lang, r_prob
            else:
                text, lang, lang_prob = best_text, best_lang, best_prob
                logger.warning(
                    "STT recheck: none met min_prob=%.1f — using best retry %s (%.0f%%)",
                    self._lang_recheck_min_prob, lang, lang_prob * 100,
                )

        self._detected_language = lang

        if text:
            await self._bus.publish(EventType.TRANSCRIPTION_READY, data=text)

        return text

    # ------------------------------------------------------------------
    # Internal (runs in thread-pool executor)
    # ------------------------------------------------------------------

    def _run_transcribe(
        self, audio: np.ndarray, language: str | None = None,
    ) -> tuple[str, str, float]:
        """Blocking transcription — called from a thread-pool executor.

        Iterates over all Whisper segments and joins their text.  Returns
        an empty string if no speech was detected.

        Args:
            audio:    Mono float32 array.
            language: Force a specific language for this inference pass.
                      When ``None``, uses the config language (auto-detect
                      if config is also ``None``).

        Returns:
            ``(text, language, language_probability)`` tuple where *text* is
            the stripped concatenation of all segment texts (empty string when
            no speech is detected), *language* is the Whisper-detected code,
            and *language_probability* is the detection confidence in [0, 1].
        """
        segments, info = self._model.transcribe(
            audio,
            language=language if language is not None else self._language,
            beam_size=self._beam_size,
        )
        text = "".join(seg.text for seg in segments).strip()

        logger.info(
            "STT | lang=%s (%.0f%%) | duration=%.2fs | text=%r",
            info.language,
            info.language_probability * 100,
            info.duration,
            text,
        )

        return text, info.language, info.language_probability
