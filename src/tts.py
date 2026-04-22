"""tts.py — Text-to-Speech synthesizer for Samsung Gauss.

Provides:
  Synthesizer            — typing.Protocol for TTS backends
  SupertonicSynthesizer  — Supertonic (ONNX, CPU, 44.1 kHz, Korean + English)
  create_synthesizer     — factory

SupertonicSynthesizer runs inference in ``asyncio.to_thread`` so it never
blocks the asyncio event loop (ONNX Runtime calls are blocking).

"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from typing import Protocol, runtime_checkable

import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text preprocessing for TTS
# ---------------------------------------------------------------------------

_MARKDOWN_RE = re.compile(r"[*#_~`\[\]{}|\\·•◦‧⋅∙]")
_FANCY_QUOTES_RE = re.compile(r"[\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f\u300a\u300b]")
_DASH_RE = re.compile(r"\s*[\u2014\u2013]\s*")  # em-dash, en-dash → comma
# L-O-V-E → LOVE (single-letter segments only; keeps "well-known")
_LETTER_SPELL_RE = re.compile(r"(?<![A-Za-z-])(?:[A-Za-z]-)+[A-Za-z](?![A-Za-z-])")
_PAREN_RE = re.compile(r"\s*\(([^)]*)\)\s*")  # (text) → , text,
# D.C. → DC (two or more single-letter-dot sequences)
_ABBREV_DOT_RE = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]\.){2,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
# … or consecutive dots (2+) → ". "  (소수점 단일 점은 매칭 안 됨)
_ELLIPSIS_RE = re.compile(r"[…]|\.{2,}")
# Sentence-final punctuation check for terminal guarantee
_NEEDS_TERMINAL_RE = re.compile(r"[.!?。！？]$")


def _unwrap_parens(m: re.Match) -> str:
    """(내용) → , 내용, — weave parenthetical into natural speech."""
    content = m.group(1).strip()
    if not content:
        return " "
    return f", {content}, "


def _clean_tts_text(text: str) -> str:
    """Normalise LLM output for TTS consumption.

    Removes markdown formatting, special quotation marks, dashes,
    parenthetical asides, letter-spelling, and abbreviation dots.
    Keeps apostrophes (``'``) for English contractions (don't, it's).
    """
    text = text.replace("\n", " ")
    text = unicodedata.normalize("NFKC", text)  # fullwidth ASCII → halfwidth, Hangul 정규화
    text = _MARKDOWN_RE.sub("", text)
    text = _FANCY_QUOTES_RE.sub("", text)
    text = text.replace('"', "")
    text = _DASH_RE.sub(", ", text)
    text = _ELLIPSIS_RE.sub(". ", text)  # … / ... → ". " (소수점 단일 점 무영향)
    text = _LETTER_SPELL_RE.sub(lambda m: m.group(0).replace("-", ""), text)
    text = _PAREN_RE.sub(_unwrap_parens, text)
    text = _ABBREV_DOT_RE.sub(lambda m: m.group(0).replace(".", ""), text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = text.strip()
    if text and not _NEEDS_TERMINAL_RE.search(text):
        text = text + "."
    return text


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
            float32 numpy array and sample_rate is an int (typically 44100).
        """
        ...

    async def close(self) -> None:
        """Release underlying model resources."""
        ...


# ---------------------------------------------------------------------------
# SupertonicSynthesizer
# ---------------------------------------------------------------------------

class SupertonicSynthesizer:
    """Supertonic TTS synthesizer (ONNX, CPU).

    Uses the ``supertonic`` package which provides a lightweight, fast
    ONNX-based TTS engine with native Korean and English support.
    Synthesis runs in a thread executor (ONNX Runtime calls are blocking).

    Language switching is instant — just a parameter change, no model reload.

    Args:
        cfg: App config — reads ``tts.voice``, ``tts.speed``,
             ``tts.total_steps``, ``tts.language``.
    """

    def __init__(self, cfg: Config) -> None:
        from supertonic import TTS as SupertonicTTS

        self._language: str = cfg.get("tts.language", "ko")
        self._speed: float = cfg.get("tts.speed", 1.0)
        self._total_steps: int = cfg.get("tts.total_steps", 8)
        self._leading_silence_sec: float = cfg.get("tts.leading_silence_sec", 0.05)
        self._trailing_silence_sec: float = cfg.get("tts.trailing_silence_sec", 0.08)
        self._max_chunk_length: int | None = cfg.get("tts.max_chunk_length", None)
        self._silence_duration: float = cfg.get("tts.silence_duration", 0.1)

        voice_name: str = cfg.get("tts.voice", "M1")

        intra = cfg.get("tts.ort_intra_threads", None)
        inter = cfg.get("tts.ort_inter_threads", None)
        self._tts = SupertonicTTS(
            intra_op_num_threads=intra,
            inter_op_num_threads=inter,
        )
        self._sample_rate: int = self._tts.sample_rate  # 44100
        self._voice_style = self._tts.get_voice_style(voice_name)

        logger.info(
            "SupertonicSynthesizer ready | lang=%s voice=%s speed=%.2f "
            "steps=%d sr=%d lead=%.3fs trail=%.3fs ort_intra=%s ort_inter=%s",
            self._language, voice_name, self._speed,
            self._total_steps, self._sample_rate,
            self._leading_silence_sec, self._trailing_silence_sec,
            intra, inter,
        )

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize *text* to audio using Supertonic (in thread).

        Text is cleaned (markdown, quotes, dashes removed) before synthesis.
        """
        cleaned = _clean_tts_text(text)
        if not cleaned:
            logger.warning("TTS text empty after cleaning: %r", text)
            return np.zeros(1, dtype=np.float32), self._sample_rate
        if cleaned != text:
            logger.debug("TTS text cleaned: %r → %r", text, cleaned)
        samples = await asyncio.to_thread(self._synthesize_sync, cleaned)
        logger.debug(
            "Supertonic synthesized %d samples (%.2fs)",
            len(samples), len(samples) / self._sample_rate,
        )
        return samples, self._sample_rate

    def _synthesize_sync(self, text: str) -> np.ndarray:
        """Blocking synthesis — called via to_thread."""
        kwargs: dict = dict(
            voice_style=self._voice_style,
            lang=self._language,
            speed=self._speed,
            total_steps=self._total_steps,
            silence_duration=self._silence_duration,
        )
        if self._max_chunk_length is not None:
            kwargs["max_chunk_length"] = self._max_chunk_length

        wav, _dur = self._tts.synthesize(text, **kwargs)
        # wav shape is (1, N) — flatten to 1-D
        audio = wav[0] if wav.ndim > 1 else wav
        audio = np.asarray(audio, dtype=np.float32)

        # Leading pad: protects first phoneme from attention clipping
        # Trailing pad: protects last phoneme from diffusion tail cutoff
        pad  = np.zeros(int(self._sample_rate * self._leading_silence_sec),  dtype=np.float32)
        tail = np.zeros(int(self._sample_rate * self._trailing_silence_sec), dtype=np.float32)
        return np.concatenate([pad, audio, tail])

    async def set_language(self, language: str) -> None:
        """Switch synthesis language — instant, no model reload.

        Args:
            language: Language code, e.g. ``"ko"``, ``"en"``.
        """
        lang = language.lower()
        if lang == self._language:
            return
        self._language = lang
        logger.info("SupertonicSynthesizer language switched to %s", lang)

    async def close(self) -> None:
        """Release resources (no-op for ONNX runtime)."""
        logger.debug("SupertonicSynthesizer closed")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_synthesizer(cfg: Config) -> Synthesizer:
    """Instantiate and return the configured TTS backend.

    Reads ``tts.provider`` from config.
    Supported values: ``"supertonic"``.

    Args:
        cfg: Application config.

    Returns:
        A :class:`Synthesizer`-conforming instance.

    Raises:
        ValueError: If the provider name is not recognised.
    """
    provider: str = (cfg.get("tts.provider", "supertonic") or "supertonic").lower().strip()

    if provider == "supertonic":
        logger.info("TTS provider: Supertonic")
        return SupertonicSynthesizer(cfg)

    raise ValueError(
        f"Unknown TTS provider: {provider!r}. "
        "Supported value: 'supertonic'."
    )
