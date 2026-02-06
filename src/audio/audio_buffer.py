"""Ring buffer for audio data."""
import numpy as np
from threading import Lock


class AudioBuffer:
    """Thread-safe ring buffer for audio samples."""

    def __init__(
        self,
        max_seconds: float = 30.0,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        """Initialize the audio buffer.

        Args:
            max_seconds: Maximum duration to store
            sample_rate: Audio sample rate
            channels: Number of audio channels
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._max_samples = int(max_seconds * sample_rate)

        self._buffer = np.zeros((self._max_samples, channels), dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._available = 0
        self._lock = Lock()

    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        return self._sample_rate

    @property
    def available(self) -> int:
        """Number of samples available to read."""
        with self._lock:
            return self._available

    @property
    def available_seconds(self) -> float:
        """Duration of audio available in seconds."""
        return self.available / self._sample_rate

    def write(self, data: np.ndarray) -> int:
        """Write audio samples to the buffer.

        Args:
            data: Audio samples to write (samples x channels)

        Returns:
            Number of samples written
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if data.shape[1] != self._channels:
            raise ValueError(f"Expected {self._channels} channels, got {data.shape[1]}")

        samples_to_write = len(data)

        with self._lock:
            # Handle wrap-around
            first_chunk = min(samples_to_write, self._max_samples - self._write_pos)
            self._buffer[self._write_pos:self._write_pos + first_chunk] = data[:first_chunk]

            if first_chunk < samples_to_write:
                remaining = samples_to_write - first_chunk
                self._buffer[:remaining] = data[first_chunk:]
                self._write_pos = remaining
            else:
                self._write_pos = (self._write_pos + first_chunk) % self._max_samples

            # Update available count (cap at max)
            self._available = min(self._available + samples_to_write, self._max_samples)

            # Move read position if we overwrote data
            if self._available == self._max_samples:
                self._read_pos = self._write_pos

        return samples_to_write

    def read(self, num_samples: int) -> np.ndarray:
        """Read audio samples from the buffer.

        Args:
            num_samples: Number of samples to read

        Returns:
            Audio samples (may be fewer than requested if not available)
        """
        with self._lock:
            samples_to_read = min(num_samples, self._available)

            if samples_to_read == 0:
                return np.zeros((0, self._channels), dtype=np.float32)

            result = np.zeros((samples_to_read, self._channels), dtype=np.float32)

            # Handle wrap-around
            first_chunk = min(samples_to_read, self._max_samples - self._read_pos)
            result[:first_chunk] = self._buffer[self._read_pos:self._read_pos + first_chunk]

            if first_chunk < samples_to_read:
                remaining = samples_to_read - first_chunk
                result[first_chunk:] = self._buffer[:remaining]
                self._read_pos = remaining
            else:
                self._read_pos = (self._read_pos + first_chunk) % self._max_samples

            self._available -= samples_to_read

        return result

    def peek(self, num_samples: int) -> np.ndarray:
        """Peek at audio samples without consuming them.

        Args:
            num_samples: Number of samples to peek

        Returns:
            Audio samples (does not advance read position)
        """
        with self._lock:
            samples_to_read = min(num_samples, self._available)

            if samples_to_read == 0:
                return np.zeros((0, self._channels), dtype=np.float32)

            result = np.zeros((samples_to_read, self._channels), dtype=np.float32)

            # Handle wrap-around
            first_chunk = min(samples_to_read, self._max_samples - self._read_pos)
            result[:first_chunk] = self._buffer[self._read_pos:self._read_pos + first_chunk]

            if first_chunk < samples_to_read:
                remaining = samples_to_read - first_chunk
                result[first_chunk:] = self._buffer[:remaining]

        return result

    def read_all(self) -> np.ndarray:
        """Read all available audio samples.

        Returns:
            All available audio samples
        """
        return self.read(self.available)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._available = 0
