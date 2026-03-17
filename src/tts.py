"""tts.py — Text-to-Speech synthesizers for Yona.

Provides:
  Synthesizer          — typing.Protocol for all TTS backends
  MeloSynthesizer      — MeloTTS (VITS, CPU, 24 kHz, Korean + English)
  create_synthesizer   — factory: reads tts.provider and returns the right backend

MeloSynthesizer runs CPU-bound inference in ``asyncio.to_thread`` so it
never blocks the asyncio event loop.

Usage::

    from src.config import Config
    from src.tts import create_synthesizer

    synth = create_synthesizer(cfg)
    audio, sr = await synth.synthesize("안녕하세요!")
    # audio: np.ndarray float32, sr: 24000
    await synth.close()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Synthesizer(Protocol):
    """Protocol for all TTS backends.

    Implementors must provide:
      - ``synthesize(text)`` — async, returns (audio_samples, sample_rate)
      - ``close()`` — coroutine to release resources
    """

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Convert *text* to audio samples.

        Returns:
            A tuple of (samples, sample_rate) where samples is a 1-D
            float32 numpy array and sample_rate is an int (typically 24000).
        """
        ...

    async def close(self) -> None:
        """Release underlying model resources."""
        ...


# ---------------------------------------------------------------------------
# MeloSynthesizer
# ---------------------------------------------------------------------------

class MeloSynthesizer:
    """MeloTTS synthesizer (VITS, CPU).

    Uses the ``melo`` package which provides a simple ``TTS`` class.
    Synthesis runs in a thread executor.

    Args:
        cfg: App config — reads ``tts.melo_language``, ``tts.melo_device``,
             ``tts.melo_speed``, ``tts.output_sample_rate``.
    """

    def __init__(self, cfg: Config) -> None:
        from melo.api import TTS

        self._language: str = cfg.get("tts.melo_language", "KR")
        self._device: str = cfg.get("tts.melo_device", "cpu")
        self._speed: float = cfg.get("tts.melo_speed", 1.0)
        self._sample_rate: int = cfg.get("tts.output_sample_rate", 24000)

        self._engine = TTS(language=self._language, device=self._device)
        # MeloTTS speaker IDs — pick the first available for the language
        speaker_ids = self._engine.hps.data.spk2id
        self._speaker_id = list(speaker_ids.values())[0]
        logger.info(
            "MeloSynthesizer ready | lang=%s device=%s speed=%.1f speaker_id=%s",
            self._language, self._device, self._speed, self._speaker_id,
        )

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize *text* to audio using MeloTTS (CPU, in thread)."""
        samples = await asyncio.to_thread(
            self._synthesize_sync,
            text,
        )
        logger.debug("Melo synthesized %d samples (%.2fs)", len(samples), len(samples) / self._sample_rate)
        return samples, self._sample_rate

    def _synthesize_sync(self, text: str) -> np.ndarray:
        """Blocking synthesis — called via run_in_executor."""
        audio = self._engine.tts_to_file(
            text,
            self._speaker_id,
            quiet=True,
            speed=self._speed,
        )
        return np.asarray(audio, dtype=np.float32)

    async def set_language(self, language: str) -> None:
        """Reload the MeloTTS engine with a different language.

        Called by :class:`StreamingPipeline` when the STT-detected language
        differs from the current one.  Runs the model reload in a thread
        executor since ``TTS(language=...)`` is blocking.

        Args:
            language: MeloTTS language code, e.g. ``"KR"``, ``"EN"``.
        """
        lang = language.upper()
        if lang == self._language:
            return
        from melo.api import TTS as MeloTTSEngine

        self._language = lang
        self._engine = await asyncio.to_thread(
            MeloTTSEngine, language=lang, device=self._device,
        )
        speaker_ids = self._engine.hps.data.spk2id
        self._speaker_id = list(speaker_ids.values())[0]
        logger.info(
            "MeloSynthesizer language switched to %s (speaker_id=%s)",
            lang, self._speaker_id,
        )

    async def close(self) -> None:
        """No persistent resources to release for MeloTTS."""
        logger.debug("MeloSynthesizer closed")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_synthesizer(cfg: Config) -> Synthesizer:
    """Instantiate and return the configured TTS backend.

    Reads ``tts.provider`` from config.
    Supported values: ``"melo"``.

    Args:
        cfg: Application config.

    Returns:
        A :class:`Synthesizer`-conforming instance.

    Raises:
        ValueError: If the provider name is not recognised.
    """
    provider: str = (cfg.get("tts.provider", "melo") or "melo").lower().strip()

    if provider == "melo":
        logger.info("TTS provider: MeloTTS")
        return MeloSynthesizer(cfg)

    raise ValueError(
        f"Unknown TTS provider: {provider!r}. "
        "Supported value: 'melo'."
    )
