"""conftest.py — Shared pytest fixtures for the Yona test suite.

Fixtures provided
-----------------
min_config   Config backed by a minimal in-memory YAML file (no ${ENV_VAR} refs).
bus          A real EventBus instance reset for each test.
captured     Async factory: subscribe to one or more EventTypes and collect events.
FakeAudio    Stub AudioManager subclass — no sounddevice, safe in CI.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from textwrap import dedent
from typing import Callable

import numpy as np
import pytest
import pytest_asyncio

from src.config import Config
from src.events import EventBus, EventType

# ---------------------------------------------------------------------------
# Minimal YAML shared across tests
# ---------------------------------------------------------------------------

_MIN_YAML = dedent("""\
    audio:
      input_device: "default"
      output_device: "default"
      input_sample_rate: 16000
      output_sample_rate: 48000
      input_channels: 1
      output_channels: 2
      chunk_size: 512
      buffer_seconds: 5
      volume_percent: null

    wake_word:
      wake_phrase: "Hey Mack"
      active_models: []
      model_paths: []
      threshold: 0.5
      inference_framework: "onnx"

    vad:
      model_path: "models/silero_vad/silero_vad.onnx"
      threshold: 0.5
      silence_duration: 1.5
      min_speech_duration: 0.1

    stt:
      model_size: "large-v3-turbo"
      device: "cpu"
      compute_type: "int8"
      language: null
      allowed_languages: ["ko", "en"]
      lang_recheck_min_prob: 0.3
      beam_size: 1

    llm:
      provider: "openai"
      max_context_tokens: 4096

    tts:
      provider: "supertonic"
      model_path: "models/tts"
      output_sample_rate: 44100

    conversation:
      max_context_tokens: 4096
      history_max_tokens: 2000
      compression_threshold: 0.8
      timeout_check_seconds: 15
      timeout_final_seconds: 5
      goodbye_message:
        ko: "안녕히 계세요!"
        en: "Goodbye!"

    history:
      dir: "data/history"
      retention_days: 30

    logging:
      level: "WARNING"

    web:
      enabled: false
      host: "127.0.0.1"
      port: 8080
      allowed_hosts: []
""")


@pytest.fixture()
def min_config(tmp_path: Path) -> Config:
    """Config instance backed by a minimal YAML — no real devices or API keys needed."""
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(_MIN_YAML, encoding="utf-8")
    return Config(path=cfg_file)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

@pytest.fixture()
def bus() -> EventBus:
    """Fresh EventBus for each test."""
    return EventBus()


@pytest_asyncio.fixture()
async def captured(bus: EventBus):
    """Factory fixture: returns a coroutine that subscribes to event types
    and yields a list that accumulates received events.

    Usage::

        async def test_something(captured, bus):
            events = await captured(EventType.STATE_CHANGED)
            await bus.publish(EventType.STATE_CHANGED, data="IDLE")
            await asyncio.sleep(0)          # let the subscriber run
            assert events[0].data == "IDLE"
    """
    queues: list[asyncio.Queue] = []

    async def _subscribe(*event_types: EventType) -> list:
        received: list = []

        async def _drain(q: asyncio.Queue) -> None:
            while True:
                event = await q.get()
                received.append(event)

        for et in event_types:
            q = bus.subscribe(et)
            queues.append(q)
            asyncio.get_event_loop().create_task(_drain(q))

        return received

    yield _subscribe


# ---------------------------------------------------------------------------
# Fake AudioManager — no sounddevice
# ---------------------------------------------------------------------------

class FakeAudioManager:
    """Drop-in stub for AudioManager.  Records calls without touching hardware."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[[np.ndarray], None]] = []
        self.started = False
        self.stopped = False
        self.played_chunks: list[np.ndarray] = []

    def add_input_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        self._callbacks.append(cb)

    def remove_input_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        try:
            self._callbacks.remove(cb)
        except ValueError:
            pass

    def push_audio(self, chunk: np.ndarray) -> None:
        """Simulate a microphone chunk arriving from the sounddevice thread."""
        for cb in list(self._callbacks):
            cb(chunk)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def play_audio(self, audio: np.ndarray, _sample_rate: int) -> None:
        self.played_chunks.append(audio.copy())

    async def stop_playback(self) -> None:
        pass


@pytest.fixture()
def fake_audio() -> FakeAudioManager:
    """FakeAudioManager instance for each test."""
    return FakeAudioManager()


# ---------------------------------------------------------------------------
# Fake Transcriber
# ---------------------------------------------------------------------------

class FakeTranscriber:
    """Stub Transcriber that returns a pre-configured (text, lang) pair.

    Set ``result`` before calling ``transcribe`` to control what is returned::

        ft = FakeTranscriber()
        ft.result = ("안녕하세요", "ko")
        text, lang = await ft.transcribe(audio)
    """

    def __init__(self) -> None:
        self.result: tuple[str, str] = ("hello", "en")
        self.calls: list[np.ndarray] = []

    async def transcribe(
        self,
        audio: np.ndarray,
        _sample_rate: int = 16_000,
    ) -> tuple[str, str]:
        self.calls.append(audio.copy())
        return self.result


@pytest.fixture()
def fake_transcriber() -> FakeTranscriber:
    """FakeTranscriber instance for each test."""
    return FakeTranscriber()
