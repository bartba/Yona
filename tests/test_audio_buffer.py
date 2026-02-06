"""Tests for AudioBuffer."""
import numpy as np
import pytest

from src.audio.audio_buffer import AudioBuffer


class TestAudioBuffer:
    """Tests for AudioBuffer class."""

    def test_init(self):
        """Test buffer initialization."""
        buffer = AudioBuffer(max_seconds=5.0, sample_rate=16000, channels=1)
        assert buffer.sample_rate == 16000
        assert buffer.available == 0

    def test_write_and_read(self):
        """Test basic write and read operations."""
        buffer = AudioBuffer(max_seconds=5.0, sample_rate=16000, channels=1)

        # Write some samples
        data = np.random.randn(1000, 1).astype(np.float32)
        written = buffer.write(data)

        assert written == 1000
        assert buffer.available == 1000

        # Read back
        read_data = buffer.read(500)
        assert len(read_data) == 500
        assert buffer.available == 500

    def test_read_all(self):
        """Test reading all available samples."""
        buffer = AudioBuffer(max_seconds=5.0, sample_rate=16000, channels=1)

        data = np.random.randn(1000, 1).astype(np.float32)
        buffer.write(data)

        read_data = buffer.read_all()
        assert len(read_data) == 1000
        assert buffer.available == 0

    def test_peek(self):
        """Test peeking without consuming."""
        buffer = AudioBuffer(max_seconds=5.0, sample_rate=16000, channels=1)

        data = np.random.randn(1000, 1).astype(np.float32)
        buffer.write(data)

        # Peek
        peeked = buffer.peek(500)
        assert len(peeked) == 500
        assert buffer.available == 1000  # Not consumed

    def test_wraparound(self):
        """Test buffer wraparound."""
        buffer = AudioBuffer(max_seconds=1.0, sample_rate=16000, channels=1)

        # Write more than buffer size
        for _ in range(3):
            data = np.random.randn(8000, 1).astype(np.float32)
            buffer.write(data)

        # Should only have max_samples available
        assert buffer.available <= 16000

    def test_clear(self):
        """Test buffer clearing."""
        buffer = AudioBuffer(max_seconds=5.0, sample_rate=16000, channels=1)

        data = np.random.randn(1000, 1).astype(np.float32)
        buffer.write(data)

        buffer.clear()
        assert buffer.available == 0

    def test_1d_input(self):
        """Test writing 1D array."""
        buffer = AudioBuffer(max_seconds=5.0, sample_rate=16000, channels=1)

        data = np.random.randn(1000).astype(np.float32)
        written = buffer.write(data)

        assert written == 1000

    def test_available_seconds(self):
        """Test available_seconds property."""
        buffer = AudioBuffer(max_seconds=5.0, sample_rate=16000, channels=1)

        data = np.random.randn(16000, 1).astype(np.float32)
        buffer.write(data)

        assert abs(buffer.available_seconds - 1.0) < 0.001
