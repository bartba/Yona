"""Speech-to-text using faster-whisper."""
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionResult:
    """Result of speech transcription."""
    text: str
    language: str
    confidence: float
    segments: list[dict]


class Transcriber:
    """Transcribes speech using faster-whisper with CUDA acceleration."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str | None = None,
    ):
        """Initialize transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to run on ('cuda' or 'cpu')
            compute_type: Compute precision ('float16', 'int8', 'float32')
            language: Force specific language (None for auto-detect)
        """
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            from faster_whisper import WhisperModel

            logger.info(
                f"Loading Whisper model: {self._model_size} "
                f"(device={self._device}, compute_type={self._compute_type})"
            )

            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )

            logger.info("Whisper model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            # Fall back to CPU if CUDA fails
            if self._device == "cuda":
                logger.info("Falling back to CPU...")
                self._device = "cpu"
                self._compute_type = "float32"
                self._load_model()
            else:
                raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio samples (mono float32, 16kHz)
            sample_rate: Audio sample rate

        Returns:
            Transcription result with text and metadata
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Ensure float32
        audio = audio.astype(np.float32)

        # Run transcription
        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        # Collect segments
        segment_list = []
        text_parts = []

        for segment in segments:
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })
            text_parts.append(segment.text.strip())

        full_text = " ".join(text_parts)
        detected_language = info.language if info.language else "unknown"

        logger.info(
            f"Transcription: '{full_text[:100]}...' "
            f"(lang={detected_language}, prob={info.language_probability:.2f})"
        )

        return TranscriptionResult(
            text=full_text,
            language=detected_language,
            confidence=info.language_probability,
            segments=segment_list,
        )

    def transcribe_streaming(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Iterator[str]:
        """Transcribe audio with streaming output.

        Args:
            audio: Audio samples
            sample_rate: Audio sample rate

        Yields:
            Text segments as they're transcribed
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)

        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            vad_filter=True,
        )

        for segment in segments:
            yield segment.text.strip()
