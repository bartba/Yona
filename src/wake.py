"""wake.py — Wake word detection for Yona.

WakeWordDetector
    Wraps the openWakeWord engine (openwakeword).
    Processes 512-sample float32 chunks from the microphone,
    converts to int16 (openWakeWord's native format), and publishes
    EventType.WAKE_WORD_DETECTED when a keyword is recognised.

    openWakeWord internally buffers 1280 samples (80 ms) before producing
    predictions, so multiple 512-sample chunks may be needed before a
    detection fires.

    A cooldown period suppresses immediate re-triggers after a detection.

Usage::

    from src.config import Config
    from src.events import EventBus
    from src.wake import WakeWordDetector

    detector = WakeWordDetector(cfg, bus)
    audio_manager.add_input_callback(detector.process_chunk)

    # ... application runs ...

    detector.reset()
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import numpy as np
from openwakeword.model import Model as OwwModel

from src.config import Config
from src.events import EventBus, EventType

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """openWakeWord-based wake word detector.

    Converts float32 microphone chunks to int16 PCM and passes them to
    openWakeWord.  When a keyword score exceeds the threshold for
    *patience* consecutive frames, publishes
    :data:`EventType.WAKE_WORD_DETECTED` with the model name as *data*,
    then enforces a cooldown period to suppress immediate re-triggers.

    Patience and threshold are applied manually (not via openWakeWord's
    predict() parameters) for consistent behaviour across library versions.

    This class is designed to be registered as an ``AudioManager`` input
    callback.  :meth:`process_chunk` is sync and uses ``publish_nowait``
    so it never blocks the sounddevice callback thread.

    Args:
        cfg: Application config — reads ``wake_word.*`` section.
        bus: Shared :class:`EventBus` for publishing detection events.
    """

    def __init__(self, cfg: Config, bus: EventBus) -> None:
        self._bus = bus

        model_paths: list[str] = cfg.get("wake_word.model_paths", [])
        framework: str = cfg.get("wake_word.inference_framework", "onnx")
        self._threshold: float = cfg.get("wake_word.threshold", 0.5)
        self._patience: int = cfg.get("wake_word.patience", 3)
        self._cooldown: float = cfg.get("wake_word.cooldown_seconds", 2.0)
        self._active_models: set[str] = set(cfg.get("wake_word.active_models", []))

        # Empty model_paths → load all pretrained models (openWakeWord default)
        kwargs: dict[str, object] = {"inference_framework": framework}
        if model_paths:
            kwargs["wakeword_models"] = model_paths
        self._model = OwwModel(**kwargs)  # type: ignore[arg-type]

        self._last_trigger: float = 0.0  # monotonic time of the last detection
        # Patience counter: consecutive frames above threshold per model
        self._patience_count: dict[str, int] = defaultdict(int)
        self._chunk_count: int = 0  # for periodic debug logging
        self._max_score: float = 0.0  # track peak score for debugging

        logger.info(
            "WakeWordDetector ready | models=%s active=%s threshold=%.2f "
            "patience=%d cooldown=%.1fs",
            list(self._model.models.keys()), self._active_models,
            self._threshold, self._patience, self._cooldown,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_names(self) -> list[str]:
        """Names of the loaded wake word models."""
        return list(self._model.models.keys())

    # ------------------------------------------------------------------
    # Core processing  (called from sounddevice callback — must be fast)
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> dict[str, float]:
        """Process one audio chunk and detect the wake word.

        Converts *chunk* from float32 to int16, feeds it to openWakeWord,
        and publishes an event if any model score exceeds the threshold
        for *patience* consecutive frames and the cooldown has elapsed.

        Args:
            chunk: Mono float32 array (any length; internally buffered).

        Returns:
            Dict mapping model name to raw prediction score (0.0-1.0).
        """
        audio = np.asarray(chunk, dtype=np.float32).ravel()

        # float32 [-1, 1] → int16 [-32768, 32767]
        pcm = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)

        # Run prediction — raw scores only, no patience/threshold args
        result = self._model.predict(pcm)
        predictions: dict[str, float] = {}
        if isinstance(result, dict):
            predictions = {k: float(v) for k, v in result.items()}

        # Manual patience + threshold + cooldown logic
        now = time.monotonic()
        for name, score in predictions.items():
            if self._active_models and name not in self._active_models:
                continue

            # Track peak score and log periodically (~every 3 seconds at 32ms/chunk)
            if score > self._max_score:
                self._max_score = score
            self._chunk_count += 1
            if self._chunk_count % 94 == 0:
                logger.debug(
                    "Wake score: %.4f (max=%.4f patience=%d/%d) [%s]",
                    score, self._max_score, self._patience_count[name],
                    self._patience, name,
                )
                self._max_score = 0.0

            if score >= self._threshold:
                self._patience_count[name] += 1
                logger.debug(
                    "Wake above threshold: %.4f patience=%d/%d [%s]",
                    score, self._patience_count[name], self._patience, name,
                )
            else:
                self._patience_count[name] = 0

            if (self._patience_count[name] >= self._patience
                    and now - self._last_trigger >= self._cooldown):
                self._last_trigger = now
                self._patience_count[name] = 0
                logger.info("Wake word TRIGGERED: %s (score=%.4f)", name, score)
                self._bus.publish_nowait(EventType.WAKE_WORD_DETECTED, data=name)
                break  # one detection per chunk is enough

        return predictions

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal prediction buffers.

        Call this after a detection or when transitioning back to IDLE
        to clear accumulated state.
        """
        self._model.reset()
        self._patience_count.clear()
