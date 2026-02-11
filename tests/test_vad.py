"""Tests for Voice Activity Detection (Silero VAD)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio.vad import VADState, VoiceActivityDetector


def _make_mock_session(probability=0.0):
    """Create a mock ONNX session that returns controlled probability."""
    session = MagicMock()
    state_out = np.zeros((2, 1, 128), dtype=np.float32)

    def run_side_effect(output_names, inputs):
        prob = session._test_probability
        return [np.array([[prob]], dtype=np.float32), state_out.copy()]

    session.run = MagicMock(side_effect=run_side_effect)
    session._test_probability = probability
    return session


class TestVoiceActivityDetector:
    """Tests for VoiceActivityDetector class."""

    def setup_method(self):
        """Set up a fresh mock session for each test."""
        self._mock_session = _make_mock_session()
        self._patcher = patch(
            "src.audio.vad.ort.InferenceSession",
            return_value=self._mock_session,
        )
        self._patcher.start()

    def teardown_method(self):
        """Stop the patcher."""
        self._patcher.stop()

    def _make_vad(self, **kwargs):
        """Create a VAD with default test parameters."""
        defaults = {"sample_rate": 16000, "threshold": 0.5}
        defaults.update(kwargs)
        return VoiceActivityDetector(**defaults)

    def test_init(self):
        """Test VAD initialization."""
        vad = self._make_vad(silence_duration=1.5)
        assert vad.state == VADState.SILENCE
        assert not vad.is_speech

    def test_detect_speech_start(self):
        """Test detection of speech start."""
        vad = self._make_vad()

        # Set high probability (speech)
        self._mock_session._test_probability = 0.9

        # Feed exactly 512 samples (one Silero chunk)
        audio = np.zeros(512, dtype=np.float32)
        state, started, ended = vad.process(audio)

        assert state == VADState.SPEECH
        assert started
        assert not ended

    def test_detect_speech_end(self):
        """Test detection of speech end."""
        vad = self._make_vad(silence_duration=0.1, min_speech_duration=0.1)

        # Start with speech — need enough for min_speech_duration (0.1s = 1600 samples)
        self._mock_session._test_probability = 0.9
        speech_audio = np.zeros(2048, dtype=np.float32)  # 4 chunks of 512
        vad.process(speech_audio)

        # Then silence — need enough for silence_duration (0.1s = 1600 samples)
        self._mock_session._test_probability = 0.1
        silence_audio = np.zeros(2048, dtype=np.float32)  # 4 chunks of 512
        state, started, ended = vad.process(silence_audio)

        assert state == VADState.SILENCE
        assert ended

    def test_ignore_short_speech(self):
        """Test that very short speech is ignored."""
        vad = self._make_vad(
            silence_duration=0.1,
            min_speech_duration=1.0,  # Require 1 second
        )

        # Very brief speech (1 chunk = 512 samples = 0.032s)
        self._mock_session._test_probability = 0.9
        brief_audio = np.zeros(512, dtype=np.float32)
        vad.process(brief_audio)

        # Then silence (enough to trigger end check)
        self._mock_session._test_probability = 0.1
        silence = np.zeros(2048, dtype=np.float32)
        state, started, ended = vad.process(silence)

        # Speech should not be registered as ended (too short)
        assert not ended

    def test_reset(self):
        """Test VAD reset."""
        vad = self._make_vad()

        # Get into speech state
        self._mock_session._test_probability = 0.9
        audio = np.zeros(512, dtype=np.float32)
        vad.process(audio)
        assert vad.state == VADState.SPEECH

        # Reset
        vad.reset()
        assert vad.state == VADState.SILENCE
        assert vad.get_speech_duration() == 0.0

    def test_speech_duration(self):
        """Test speech duration tracking."""
        vad = self._make_vad()

        # Feed 16000 samples of speech (= 1 second at 16kHz)
        # That's 31 chunks of 512 = 15872 samples tracked
        # (remaining 128 samples stay in buffer)
        self._mock_session._test_probability = 0.9
        audio = np.zeros(16000, dtype=np.float32)
        vad.process(audio)

        # 31 chunks * 512 = 15872 samples = 0.992 seconds
        expected = 15872 / 16000
        assert abs(vad.get_speech_duration() - expected) < 0.01

    def test_buffering_partial_chunks(self):
        """Test that audio smaller than 512 samples is buffered."""
        vad = self._make_vad()

        # Feed 256 samples (less than one chunk)
        self._mock_session._test_probability = 0.9
        audio = np.zeros(256, dtype=np.float32)
        state, started, ended = vad.process(audio)

        # No inference should run yet — still in silence
        assert state == VADState.SILENCE
        assert not started

        # Feed another 256 to complete the chunk
        state, started, ended = vad.process(audio)
        assert state == VADState.SPEECH
        assert started
