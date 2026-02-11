"""Voice Activity Detection using Silero VAD (ONNX)."""
import numpy as np
import onnxruntime as ort
from enum import Enum, auto
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default model path relative to project root
_DEFAULT_MODEL_PATH = "models/silero_vad/silero_vad.onnx"


class VADState(Enum):
    """Voice activity states."""
    SILENCE = auto()
    SPEECH = auto()


class VoiceActivityDetector:
    """Silero VAD-based voice activity detection."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        silence_duration: float = 1.5,
        min_speech_duration: float = 0.3,
        model_path: str | None = None,
    ):
        """Initialize VAD.

        Args:
            sample_rate: Audio sample rate (must be 16000)
            threshold: Speech probability threshold (0.0–1.0)
            silence_duration: Seconds of silence to end speech segment
            min_speech_duration: Minimum speech duration to trigger
            model_path: Path to Silero VAD ONNX model
        """
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._silence_duration = silence_duration
        self._min_speech_duration = min_speech_duration

        # Load ONNX model
        if model_path is None:
            model_path = _DEFAULT_MODEL_PATH
        self._session = ort.InferenceSession(
            str(Path(model_path)),
            providers=["CPUExecutionProvider"],
        )

        # ONNX state: LSTM hidden/cell state
        self._onnx_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(self._sample_rate, dtype=np.int64)

        # Chunk buffer for accumulating audio into 512-sample windows
        self._chunk_buffer = np.array([], dtype=np.float32)

        # VAD state tracking
        self._state = VADState.SILENCE
        self._speech_start_samples = 0
        self._silence_samples = 0
        self._total_samples = 0

    @property
    def state(self) -> VADState:
        """Current VAD state."""
        return self._state

    @property
    def is_speech(self) -> bool:
        """Whether speech is currently detected."""
        return self._state == VADState.SPEECH

    def reset(self) -> None:
        """Reset VAD state."""
        self._state = VADState.SILENCE
        self._speech_start_samples = 0
        self._silence_samples = 0
        self._total_samples = 0
        self._onnx_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._chunk_buffer = np.array([], dtype=np.float32)

    def _infer(self, chunk: np.ndarray) -> float:
        """Run Silero VAD inference on a 512-sample chunk.

        Args:
            chunk: Audio chunk of exactly 512 samples

        Returns:
            Speech probability (0.0–1.0)
        """
        input_data = chunk.reshape(1, -1)
        outputs = self._session.run(
            None,
            {"input": input_data, "state": self._onnx_state, "sr": self._sr},
        )
        probability = outputs[0].item()
        self._onnx_state = outputs[1]
        return probability

    def process(self, audio: np.ndarray) -> tuple[VADState, bool, bool]:
        """Process audio chunk and detect voice activity.

        Args:
            audio: Audio samples (mono, float32)

        Returns:
            Tuple of (current_state, speech_started, speech_ended)
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)

        # Append to chunk buffer
        self._chunk_buffer = np.concatenate([self._chunk_buffer, audio])

        speech_started = False
        speech_ended = False

        # Process complete 512-sample windows
        while len(self._chunk_buffer) >= 512:
            window = self._chunk_buffer[:512]
            self._chunk_buffer = self._chunk_buffer[512:]

            probability = self._infer(window)
            is_active = probability >= self._threshold

            num_samples = 512

            if self._state == VADState.SILENCE:
                if is_active:
                    self._speech_start_samples = self._total_samples
                    self._total_samples += num_samples
                    self._state = VADState.SPEECH
                    self._silence_samples = 0
                    speech_started = True
                    logger.debug(f"Speech started (prob: {probability:.4f})")
                else:
                    self._total_samples += num_samples
            else:  # SPEECH state
                self._total_samples += num_samples
                if is_active:
                    self._silence_samples = 0
                else:
                    self._silence_samples += num_samples

                    silence_dur = self._silence_samples / self._sample_rate
                    if silence_dur >= self._silence_duration:
                        speech_duration = (
                            self._total_samples - self._speech_start_samples
                        ) / self._sample_rate

                        if speech_duration >= self._min_speech_duration:
                            speech_ended = True
                            logger.debug(
                                f"Speech ended (duration: {speech_duration:.2f}s)"
                            )
                        else:
                            logger.debug("Speech too short, ignoring")

                        self._state = VADState.SILENCE
                        self._silence_samples = 0

        return self._state, speech_started, speech_ended

    def get_speech_duration(self) -> float:
        """Get duration of current speech segment.

        Returns:
            Duration in seconds (0 if not in speech)
        """
        if self._state != VADState.SPEECH:
            return 0.0

        return (self._total_samples - self._speech_start_samples) / self._sample_rate
