"""Wake word detection using openWakeWord."""
import time
from pathlib import Path
from typing import Callable

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class WakeWordDetector:
    """Detects wake words using openWakeWord."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = 0.5,
        cooldown_seconds: float = 2.0,
        sample_rate: int = 16000,
    ):
        """Initialize wake word detector.

        Args:
            model_path: Path to custom ONNX model (None for built-in)
            threshold: Detection threshold (0.0 - 1.0)
            cooldown_seconds: Minimum time between detections
            sample_rate: Expected audio sample rate
        """
        self._model_path = Path(model_path) if model_path else None
        self._threshold = threshold
        self._cooldown_seconds = cooldown_seconds
        self._sample_rate = sample_rate
        self._last_detection_time = 0.0

        self._model = None
        self._model_name = None
        self._on_detection: Callable[[], None] | None = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the wake word model."""
        try:
            import openwakeword
            from openwakeword.model import Model

            if self._model_path and self._model_path.exists():
                # Load custom model
                logger.info(f"Loading custom wake word model: {self._model_path}")
                self._model = Model(
                    wakeword_models=[str(self._model_path)],
                    inference_framework="onnx",
                )
                self._model_name = self._model_path.stem
            else:
                # Use built-in model as fallback
                logger.info("Loading built-in 'hey_jarvis' wake word model")
                openwakeword.utils.download_models(["hey_jarvis"])
                self._model = Model(
                    wakeword_models=["hey_jarvis"],
                    inference_framework="onnx",
                )
                self._model_name = "hey_jarvis"

            logger.info(f"Wake word model loaded: {self._model_name}")

        except ImportError as e:
            logger.error(f"Failed to import openwakeword: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load wake word model: {e}")
            raise

    def set_detection_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for wake word detection.

        Args:
            callback: Function called when wake word is detected
        """
        self._on_detection = callback

    def process(self, audio: np.ndarray) -> bool:
        """Process audio chunk for wake word detection.

        Args:
            audio: Audio samples (16kHz mono float32)

        Returns:
            True if wake word detected
        """
        if self._model is None:
            return False

        # Convert to expected format
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Convert to int16 for openwakeword
        audio_int16 = (audio * 32767).astype(np.int16)

        # Run prediction
        predictions = self._model.predict(audio_int16)

        # Check for detection
        for name, score in predictions.items():
            if score >= self._threshold:
                current_time = time.time()
                elapsed = current_time - self._last_detection_time

                if elapsed >= self._cooldown_seconds:
                    self._last_detection_time = current_time
                    logger.info(f"Wake word detected: {name} (score: {score:.3f})")

                    if self._on_detection:
                        self._on_detection()

                    return True
                else:
                    logger.debug(
                        f"Wake word detected but in cooldown "
                        f"({elapsed:.1f}s < {self._cooldown_seconds}s)"
                    )

        return False

    def reset(self) -> None:
        """Reset detector state."""
        if self._model:
            self._model.reset()
        self._last_detection_time = 0.0

    @property
    def threshold(self) -> float:
        """Detection threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set detection threshold."""
        self._threshold = max(0.0, min(1.0, value))
