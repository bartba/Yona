"""audio.py — Audio I/O for Yona.

Provides three public classes:

AudioBuffer
    Thread-safe ring buffer that stores up to ``buffer_seconds`` seconds of
    mono float32 audio at ``sample_rate`` Hz.  Used by WakeWordDetector and
    VoiceActivityDetector to accumulate microphone input between processing
    cycles.

AudioManager
    Wraps sounddevice to provide:
    - A microphone input stream that fans audio chunks out to registered
      callbacks (called on the sounddevice thread).
    - Async ``play_audio`` / ``stop_playback`` for TTS output playback.

ChimePlayer
    Plays a short sine-wave chime on wake-word detection.  Uses the shared
    AudioManager so it shares the same output device and sample-rate path.

Usage::

    from src.config import Config
    from src.audio import AudioBuffer, AudioManager, ChimePlayer

    cfg = Config()
    manager = AudioManager(cfg)
    chime = ChimePlayer(cfg, manager)

    await manager.start()
    manager.add_input_callback(my_vad_callback)
    await chime.play()
    await manager.stop()
"""

from __future__ import annotations

import asyncio
import logging
import math
import subprocess
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

from src.config import Config


# ---------------------------------------------------------------------------
# AudioBuffer
# ---------------------------------------------------------------------------

class AudioBuffer:
    """Thread-safe ring buffer for mono float32 audio samples.

    Audio is appended with :method:`push` and read back in chronological order
    with :method:`get_all`.  When the buffer is full, the oldest samples are
    silently overwritten (ring behaviour). 
    마이크 입력 오디오를 최대 30초 분량 임시 저장하는 버퍼(30초 초과시 오래된 것부터 덮어씀)

    Args:
        sample_rate:    Input sample rate in Hz (default 16 000).
        buffer_seconds: Total buffer length in seconds (default 30).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        buffer_seconds: float = 30.0,
    ) -> None:
        self._sample_rate = sample_rate
        # sample_rate × buffer_seconds = 총 샘플 수
        # (예: 16000 Hz × 30초 = 480,000 샘플 ≈ chunk_size 512 기준 938회 push)
        self._capacity = int(sample_rate * buffer_seconds)
        self._buffer = np.zeros(self._capacity, dtype=np.float32)
        self._write_pos = 0
        self._filled = False  # True once the buffer has wrapped at least once
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Total sample capacity (sample_rate × buffer_seconds)."""
        return self._capacity

    @property
    def sample_rate(self) -> int:
        """Input sample rate this buffer was initialised with."""
        return self._sample_rate

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(self, chunk: np.ndarray) -> None:
        """Append *chunk* to the ring buffer.

        Accepts any shape; the array is flattened and cast to float32 before
        storage.  When the chunk would overflow the capacity, the write wraps
        around and overwrites the oldest data.

        Thread-safe — safe to call from the sounddevice callback thread.
        """
        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        n = len(chunk)
        if n == 0:
            return
        with self._lock:
            end = self._write_pos + n
            if end <= self._capacity:
                self._buffer[self._write_pos:end] = chunk
                self._write_pos = end % self._capacity
                if self._write_pos == 0:
                    self._filled = True
            else:
                # Chunk straddles the end of the buffer — split write
                first = self._capacity - self._write_pos
                self._buffer[self._write_pos:] = chunk[:first]
                self._buffer[:n - first] = chunk[first:]
                self._write_pos = n - first
                self._filled = True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all(self) -> np.ndarray:
        """Return a copy of all buffered samples in chronological order.

        Returns an empty float32 array if nothing has been pushed yet.
        """
        with self._lock:
            if self._filled:
                return np.concatenate([
                    self._buffer[self._write_pos:].copy(),
                    self._buffer[:self._write_pos].copy(),
                ])
            return self._buffer[:self._write_pos].copy()

    def reset(self) -> None:
        """Clear the buffer and reset the write position to 0."""
        with self._lock:
            self._buffer[:] = 0.0
            self._write_pos = 0
            self._filled = False


# ---------------------------------------------------------------------------
# AudioManager
# ---------------------------------------------------------------------------

class AudioManager:
    """Manages sounddevice input stream and async audio playback.

    All registered input callbacks are invoked synchronously from the
    sounddevice callback thread with a mono float32 numpy array.  Add
    callbacks before calling :method:`start`.

    Playback (:method:`play_audio`) runs sounddevice in a thread executor so
    the asyncio event loop is never blocked.

    Args:
        cfg: Application config — reads the ``audio.*`` section.
    """

    def __init__(self, cfg: Config) -> None:
        self._input_device: str = cfg.get("audio.input_device", "default")
        self._output_device: str = cfg.get("audio.output_device", "default")
        self._input_rate: int = cfg.get("audio.input_sample_rate", 16_000)
        self._output_rate: int = cfg.get("audio.output_sample_rate", 48_000)
        self._input_channels: int = cfg.get("audio.input_channels", 1)
        self._output_channels: int = cfg.get("audio.output_channels", 2)
        self._chunk_size: int = cfg.get("audio.chunk_size", 512)

        self._volume_percent: int | None = cfg.get("audio.volume_percent", None)

        self._callbacks: list[Callable[[np.ndarray], None]] = []
        self._cb_lock = threading.Lock()
        self._input_stream: sd.InputStream | None = None
        self._playing = False

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def add_input_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        """Register *cb* to receive mono float32 chunks from the mic stream.

        *cb* is called on the sounddevice thread; it must return quickly and
        must not block the thread.
        """
        with self._cb_lock:
            self._callbacks.append(cb)

    def remove_input_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        """Unregister *cb*.  Silently ignores unknown callbacks."""
        with self._cb_lock:
            try:
                self._callbacks.remove(cb)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Sounddevice callback (called from the sd thread)
    # ------------------------------------------------------------------

    def _audio_input_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Internal sounddevice InputStream callback.

        Extracts channel 0 as a mono float32 chunk and dispatches it to
        every registered callback.
        """
        chunk = indata[:, 0].copy()  # channel 0 → mono float32
        with self._cb_lock:
            for cb in list(self._callbacks):
                cb(chunk)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _apply_volume(self) -> None:
        """Set ALSA PCM playback volume for the output device if configured.

        Uses ``amixer`` to find the correct ALSA card number by matching the
        output device name, then sets numid=5 (PCM Playback Volume).
        Logs a warning and continues silently on failure.
        """
        if self._volume_percent is None:
            return

        log = logging.getLogger(__name__)

        # Find ALSA card number from sounddevice device info
        card_num: int | None = None
        for dev_info in sd.query_devices():
            if self._output_device in dev_info["name"]:
                # Extract hw:N from device name, e.g. "Poly Sync 20: USB Audio (hw:1,0)"
                name: str = dev_info["name"]
                if "hw:" in name:
                    hw_part = name.split("hw:")[1].split(",")[0].rstrip(")")
                    card_num = int(hw_part)
                break

        if card_num is None:
            log.warning("Cannot find ALSA card for '%s'; skipping volume set", self._output_device)
            return

        # Convert percent (0-100) → ALSA value (0-20)
        alsa_val = max(0, min(20, round(self._volume_percent * 20 / 100)))
        try:
            subprocess.run(
                ["amixer", "-c", str(card_num), "cset", "numid=5", str(alsa_val)],
                capture_output=True, text=True, timeout=5, check=True,
            )
            log.info("ALSA volume set to %d/20 (%d%%) on card %d",
                     alsa_val, self._volume_percent, card_num)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            log.warning("Failed to set ALSA volume: %s", exc)

    async def start(self) -> None:
        """Open and start the microphone input stream."""
        self._apply_volume()
        self._input_stream = sd.InputStream(
            device=self._input_device,
            channels=self._input_channels,
            samplerate=self._input_rate,
            blocksize=self._chunk_size,
            dtype=np.float32,
            callback=self._audio_input_callback,
        )
        self._input_stream.start()

    async def stop(self) -> None:
        """Stop and close the input stream.  Safe to call multiple times."""
        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Resample and convert *audio* to the output format.

        Output format: float32, ``output_channels`` channels, at
        ``output_sample_rate`` Hz.

        Resampling uses linear interpolation (numpy.interp), which is exact
        for integer ratios and acceptable for TTS voice output.
        """
        audio = np.asarray(audio, dtype=np.float32).ravel()

        # Resample to output_rate if needed
        if sample_rate != self._output_rate:
            audio = _resample(audio, sample_rate, self._output_rate)

        # Convert mono to required number of output channels
        if self._output_channels == 2:
            audio = np.stack([audio, audio], axis=1)
        else:
            audio = audio.reshape(-1, 1)

        return audio

    async def play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play *audio* (mono float32) at *sample_rate* Hz.

        Resamples and converts to the output format, then plays via
        sounddevice in a thread executor so the event loop stays free.

        Args:
            audio:       1-D float32 numpy array of audio samples.
            sample_rate: Sample rate of *audio* in Hz.
        """
        prepared = self._prepare_audio(audio, sample_rate)
        self._playing = True
        try:
            await asyncio.to_thread(
                sd.play,
                prepared,
                self._output_rate,
                device=self._output_device,
                blocking=True,
            )
        finally:
            self._playing = False

    async def stop_playback(self) -> None:
        """Immediately stop any current audio playback (barge-in support)."""
        sd.stop()

    @property
    def is_playing(self) -> bool:
        """True while :meth:`play_audio` is running."""
        return self._playing


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample *audio* from *from_rate* Hz to *to_rate* Hz.

    Uses linear interpolation (numpy.interp).  Handles arbitrary integer and
    non-integer ratios.  Output length is
    ``ceil(len(audio) * to_rate / from_rate)`` samples.
    마이크 입력 : 16kHz, XTTS 출력 : 24KHz, 스피커 출력 :48kHz

    Args:
        audio:     1-D float32 array.
        from_rate: Source sample rate in Hz.
        to_rate:   Target sample rate in Hz.

    Returns:
        Resampled float32 array.
    """
    if from_rate == to_rate:
        return audio
    n_in = len(audio)
    n_out = math.ceil(n_in * to_rate / from_rate)
    x_old = np.linspace(0.0, 1.0, n_in)
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# ChimePlayer
# ---------------------------------------------------------------------------

class ChimePlayer:
    """Plays a short chime sound when the wake word is detected.

    If ``cfg.get("audio.chime_path")`` is set, loads that WAV file.
    Otherwise generates a two-tone sine-wave chime programmatically.

    Args:
        cfg:     Application config.
        manager: Shared :class:`AudioManager` used for playback.
    """

    _CHIME_SAMPLE_RATE = 24_000   # Hz — same as XTTS v2 output
    _DURATION = 1.0               # seconds — total chime length
    _FREQ_BASE = 880.0            # Hz — A5 기준음
    # (ratio, amplitude, decay_rate) — 벨 배음 구조
    # ratio: 기본음 대비 배음 비율, decay: 지수 감쇠 속도 (클수록 빨리 소멸)
    _HARMONICS: list[tuple[float, float, float]] = [
        (1.00, 0.40,  8.0),
        (2.76, 0.25,  5.0),
        (5.40, 0.12, 12.0),
        (8.93, 0.06, 15.0),
    ]

    def __init__(self, cfg: Config, manager: AudioManager) -> None:
        self._manager = manager
        chime_path: str | None = cfg.get("audio.chime_path", None)
        if chime_path:
            self._audio = self._load_wav(chime_path)
        else:
            self._audio = self._generate_chime(self._CHIME_SAMPLE_RATE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def play(self) -> None:
        """Play the chime via the shared AudioManager."""
        await self._manager.play_audio(self._audio, self._CHIME_SAMPLE_RATE)

    # ------------------------------------------------------------------
    # Chime generation
    # ------------------------------------------------------------------

    @classmethod
    def _generate_chime(cls, sample_rate: int) -> np.ndarray:
        """Generate a bell-like chime using harmonic overtones and exponential decay.

        Simulates a real bell by summing multiple harmonics, each with its own
        amplitude and decay rate.  Total duration is ``_DURATION`` seconds.
        """
        n = int(sample_rate * cls._DURATION)
        t = np.linspace(0.0, cls._DURATION, n, endpoint=False)

        wave = np.zeros(n, dtype=np.float64)
        for ratio, amp, decay in cls._HARMONICS:
            freq = cls._FREQ_BASE * ratio
            envelope = amp * np.exp(-decay * t)          # 지수 감쇠
            wave += envelope * np.sin(2.0 * math.pi * freq * t)

        # 정규화: 피크를 0.7로 맞춰 클리핑 방지
        peak = np.abs(wave).max()
        if peak > 0:
            wave = wave / peak * 0.7

        return wave.astype(np.float32)

    @staticmethod
    def _load_wav(path: str) -> np.ndarray:
        """Load a WAV file and return a float32 mono array.

        Requires ``soundfile`` (``pip install soundfile``).

        Raises:
            RuntimeError: if soundfile is not installed.
        """
        try:
            import soundfile as sf  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "soundfile is required to load a chime WAV file. "
                "Install with: pip install soundfile"
            ) from exc

        data, _ = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)  # stereo → mono
        return data
