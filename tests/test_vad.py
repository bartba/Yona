"""Tests for Voice Activity Detection."""
import numpy as np
import pytest

from src.audio.vad import VADState, VoiceActivityDetector


class TestVoiceActivityDetector:
    """Tests for VoiceActivityDetector class."""

    def test_init(self):
        """Test VAD initialization."""
        vad = VoiceActivityDetector(
            sample_rate=16000,
            energy_threshold=0.01,
            silence_duration=1.5,
        )
        assert vad.state == VADState.SILENCE
        assert not vad.is_speech

    def test_detect_speech_start(self):
        """Test detection of speech start."""
        vad = VoiceActivityDetector(
            sample_rate=16000,
            energy_threshold=0.01,
        )

        # Create loud audio (above threshold)
        loud_audio = np.random.randn(1600).astype(np.float32) * 0.5

        state, started, ended = vad.process(loud_audio)

        assert state == VADState.SPEECH
        assert started
        assert not ended

    def test_detect_speech_end(self):
        """Test detection of speech end."""
        vad = VoiceActivityDetector(
            sample_rate=16000,
            energy_threshold=0.01,
            silence_duration=0.1,  # Short for testing
            min_speech_duration=0.1,
        )

        # Start with speech
        loud_audio = np.random.randn(3200).astype(np.float32) * 0.5
        vad.process(loud_audio)

        # Then silence
        silence = np.zeros(3200, dtype=np.float32)
        state, started, ended = vad.process(silence)

        assert state == VADState.SILENCE
        assert ended

    def test_ignore_short_speech(self):
        """Test that very short speech is ignored."""
        vad = VoiceActivityDetector(
            sample_rate=16000,
            energy_threshold=0.01,
            silence_duration=0.1,
            min_speech_duration=1.0,  # Require 1 second
        )

        # Very brief speech
        brief_loud = np.random.randn(800).astype(np.float32) * 0.5
        vad.process(brief_loud)

        # Then silence
        silence = np.zeros(3200, dtype=np.float32)
        state, started, ended = vad.process(silence)

        # Speech should not be registered as ended (too short)
        assert not ended

    def test_reset(self):
        """Test VAD reset."""
        vad = VoiceActivityDetector(sample_rate=16000)

        # Get into speech state
        loud_audio = np.random.randn(1600).astype(np.float32) * 0.5
        vad.process(loud_audio)
        assert vad.state == VADState.SPEECH

        # Reset
        vad.reset()
        assert vad.state == VADState.SILENCE
        assert vad.get_speech_duration() == 0.0

    def test_speech_duration(self):
        """Test speech duration tracking."""
        vad = VoiceActivityDetector(sample_rate=16000)

        # Start speech
        loud_audio = np.random.randn(16000).astype(np.float32) * 0.5
        vad.process(loud_audio)

        # Should be about 1 second
        assert abs(vad.get_speech_duration() - 1.0) < 0.1
