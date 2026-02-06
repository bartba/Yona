"""Audio device management for Polycom speakerphone."""
import asyncio
import queue
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioManager:
    """Manages audio input/output for Polycom BT600 speakerphone."""

    def __init__(
        self,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 48000,
        input_channels: int = 1,
        output_channels: int = 2,
        chunk_size: int = 512,
    ):
        """Initialize audio manager.

        Args:
            input_device: Input device name/index (None for default)
            output_device: Output device name/index (None for default)
            input_sample_rate: Sample rate for input (16kHz for wake word/STT)
            output_sample_rate: Sample rate for output (48kHz for TTS)
            input_channels: Number of input channels
            output_channels: Number of output channels
            chunk_size: Samples per chunk
        """
        self._input_device = self._find_device(input_device, kind="input")
        self._output_device = self._find_device(output_device, kind="output")

        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._chunk_size = chunk_size

        self._input_stream: sd.InputStream | None = None
        self._output_stream: sd.OutputStream | None = None

        self._audio_callback: Callable[[np.ndarray], None] | None = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)

        self._running = False
        self._playback_lock = threading.Lock()
        self._playback_complete = threading.Event()

    def _find_device(
        self,
        device: str | int | None,
        kind: str = "input",
    ) -> int | None:
        """Find device by name or index.

        Args:
            device: Device name (partial match) or index
            kind: 'input' or 'output'

        Returns:
            Device index or None for default
        """
        if device is None:
            return None

        if isinstance(device, int):
            return device

        # Search by name
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if device.lower() in dev["name"].lower():
                if kind == "input" and dev["max_input_channels"] > 0:
                    logger.info(f"Found input device: {dev['name']} (index {i})")
                    return i
                elif kind == "output" and dev["max_output_channels"] > 0:
                    logger.info(f"Found output device: {dev['name']} (index {i})")
                    return i

        logger.warning(f"Device '{device}' not found, using default")
        return None

    def _input_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for audio input stream."""
        if status:
            logger.warning(f"Input status: {status}")

        # Copy data and pass to callback
        audio = indata.copy()

        if self._audio_callback:
            self._audio_callback(audio)

        # Also queue for async consumption
        try:
            self._audio_queue.put_nowait(audio)
        except queue.Full:
            pass  # Drop oldest if queue is full

    def set_audio_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for incoming audio data.

        Args:
            callback: Function called with each audio chunk
        """
        self._audio_callback = callback

    async def get_audio_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        """Get next audio chunk asynchronously.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Audio chunk or None if timeout
        """
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: self._audio_queue.get(timeout=timeout)
            )
        except queue.Empty:
            return None

    def start_capture(self) -> None:
        """Start audio capture."""
        if self._input_stream is not None:
            return

        self._input_stream = sd.InputStream(
            device=self._input_device,
            samplerate=self._input_sample_rate,
            channels=self._input_channels,
            dtype=np.float32,
            blocksize=self._chunk_size,
            callback=self._input_callback,
        )
        self._input_stream.start()
        self._running = True
        logger.info("Audio capture started")

    def stop_capture(self) -> None:
        """Stop audio capture."""
        if self._input_stream is None:
            return

        self._input_stream.stop()
        self._input_stream.close()
        self._input_stream = None
        self._running = False
        logger.info("Audio capture stopped")

    def play_audio(self, audio: np.ndarray, sample_rate: int | None = None) -> None:
        """Play audio synchronously.

        Args:
            audio: Audio samples to play
            sample_rate: Sample rate (uses output_sample_rate if None)
        """
        if sample_rate is None:
            sample_rate = self._output_sample_rate

        with self._playback_lock:
            # Ensure stereo for output
            if audio.ndim == 1:
                audio = np.column_stack([audio, audio])
            elif audio.shape[1] == 1:
                audio = np.column_stack([audio.flatten(), audio.flatten()])

            sd.play(audio, sample_rate, device=self._output_device)
            sd.wait()

    async def play_audio_async(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
    ) -> None:
        """Play audio asynchronously.

        Args:
            audio: Audio samples to play
            sample_rate: Sample rate (uses output_sample_rate if None)
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.play_audio, audio, sample_rate)

    def stop_playback(self) -> None:
        """Stop any ongoing audio playback."""
        sd.stop()

    def clear_queue(self) -> None:
        """Clear the audio input queue."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    @staticmethod
    def list_devices() -> None:
        """Print available audio devices."""
        print("\nAvailable Audio Devices:")
        print("=" * 60)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            input_ch = dev["max_input_channels"]
            output_ch = dev["max_output_channels"]
            default_sr = dev["default_samplerate"]
            device_type = []
            if input_ch > 0:
                device_type.append("IN")
            if output_ch > 0:
                device_type.append("OUT")
            print(
                f"[{i:2d}] {dev['name'][:40]:40s} "
                f"({'/'.join(device_type):>7s}) "
                f"SR: {default_sr:.0f}"
            )


def test_audio():
    """Test audio capture and playback."""
    import time

    print("Testing AudioManager...")
    AudioManager.list_devices()

    manager = AudioManager()

    # Test capture
    print("\nRecording 3 seconds of audio...")
    recorded_chunks = []

    def capture_callback(audio):
        recorded_chunks.append(audio.copy())

    manager.set_audio_callback(capture_callback)
    manager.start_capture()
    time.sleep(3)
    manager.stop_capture()

    # Combine and play back
    if recorded_chunks:
        audio = np.concatenate(recorded_chunks)
        print(f"Recorded {len(audio)} samples ({len(audio)/16000:.2f}s)")
        print("Playing back...")
        manager.play_audio(audio, sample_rate=16000)
        print("Done!")
    else:
        print("No audio recorded!")


if __name__ == "__main__":
    test_audio()
