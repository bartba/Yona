"""Voice Activity Detection using energy-based approach."""
import numpy as np
from enum import Enum, auto

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VADState(Enum):
    """Voice activity states."""
    SILENCE = auto()
    SPEECH = auto()


class VoiceActivityDetector:
    """Energy-based voice activity detection."""

    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 0.01,
        silence_duration: float = 1.5,
        min_speech_duration: float = 0.3,
        frame_duration: float = 0.03,
    ):
        """Initialize VAD.

        Args:
            sample_rate: Audio sample rate
            energy_threshold: RMS energy threshold for speech detection
            silence_duration: Seconds of silence to end speech segment
            min_speech_duration: Minimum speech duration to trigger
            frame_duration: Frame size for analysis in seconds
        """
        self._sample_rate = sample_rate
        self._energy_threshold = energy_threshold
        self._silence_duration = silence_duration
        self._min_speech_duration = min_speech_duration
        self._frame_samples = int(frame_duration * sample_rate)

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

    def process(self, audio: np.ndarray) -> tuple[VADState, bool, bool]:
        """Process audio chunk and detect voice activity.

        Args:
            audio: Audio samples (mono, float32)

        Returns:
            Tuple of (current_state, speech_started, speech_ended)
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        is_active = rms > self._energy_threshold

        speech_started = False
        speech_ended = False

        num_samples = len(audio)

        if self._state == VADState.SILENCE:
            if is_active:
                # Set speech start BEFORE incrementing total samples
                self._speech_start_samples = self._total_samples
                self._total_samples += num_samples
                self._state = VADState.SPEECH
                self._silence_samples = 0
                speech_started = True
                logger.debug(f"Speech started (RMS: {rms:.4f})")
            else:
                self._total_samples += num_samples
        else:  # SPEECH state
            self._total_samples += num_samples
            if is_active:
                self._silence_samples = 0
            else:
                self._silence_samples += num_samples

                # Check if silence duration exceeded
                silence_duration = self._silence_samples / self._sample_rate
                if silence_duration >= self._silence_duration:
                    # Check if speech was long enough
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
