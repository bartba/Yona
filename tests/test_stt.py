"""Tests for src/stt.py

Run with:
    pytest tests/test_stt.py -v

All tests are hardware-free — faster_whisper is stubbed via sys.modules
before src.stt is imported, then WhisperModel is patched per-fixture.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
# ---------------------------------------------------------------------------
# Stub faster_whisper before src.stt is imported (not installed on dev hardware)
# ---------------------------------------------------------------------------
_faster_whisper_stub = MagicMock()
sys.modules.setdefault("faster_whisper", _faster_whisper_stub)

from src.config import Config           # noqa: E402
from src.events import EventBus, EventType  # noqa: E402
from src.stt import Transcriber         # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silence(n: int = 16_000) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _make_segments(*texts: str) -> list[MagicMock]:
    segs = []
    for t in texts:
        s = MagicMock()
        s.text = t
        segs.append(s)
    return segs


# ---------------------------------------------------------------------------
# YAML configs
# ---------------------------------------------------------------------------

_STT_YAML = textwrap.dedent("""\
    stt:
      model_size: "large-v3-turbo"
      device: "cuda"
      compute_type: "float16"
      language: null
    """)

_STT_YAML_KO = textwrap.dedent("""\
    stt:
      model_size: "large-v3-turbo"
      device: "cuda"
      compute_type: "float16"
      language: "ko"
    """)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    f = tmp_path / "config.yaml"
    f.write_text(_STT_YAML)
    return Config(path=f)


@pytest.fixture
def cfg_ko(tmp_path: Path) -> Config:
    f = tmp_path / "config.yaml"
    f.write_text(_STT_YAML_KO)
    return Config(path=f)


@pytest.fixture
def bus() -> MagicMock:
    b = MagicMock(spec=EventBus)
    b.publish = AsyncMock()
    return b


@pytest.fixture
def mock_model():
    """Patch faster_whisper.WhisperModel and return the mock instance."""
    with patch("src.stt.WhisperModel") as MockClass:
        instance = MagicMock()
        instance.transcribe.return_value = ([], MagicMock())
        MockClass.return_value = instance
        yield instance, MockClass


@pytest.fixture
def transcriber(cfg, bus, mock_model):
    _ = mock_model  # ensures WhisperModel is patched
    return Transcriber(cfg, bus)


# ---------------------------------------------------------------------------
# TestTranscriberInit
# ---------------------------------------------------------------------------

class TestTranscriberInit:
    def test_creates_whisper_model(self, cfg, bus, mock_model):
        _, MockClass = mock_model
        Transcriber(cfg, bus)
        MockClass.assert_called_once_with(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",
        )

    def test_language_is_none_by_default(self, transcriber):
        assert transcriber._language is None

    def test_language_set_from_config(self, cfg_ko, bus, mock_model):
        _ = mock_model
        t = Transcriber(cfg_ko, bus)
        assert t._language == "ko"


# ---------------------------------------------------------------------------
# TestTranscriberTranscribe
# ---------------------------------------------------------------------------

class TestTranscriberTranscribe:
    @pytest.mark.asyncio
    async def test_returns_transcribed_text(self, transcriber, mock_model):
        instance, _ = mock_model
        instance.transcribe.return_value = (_make_segments("안녕하세요."), MagicMock())
        result = await transcriber.transcribe(_silence())
        assert result == "안녕하세요."

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_silence(self, transcriber, mock_model):
        instance, _ = mock_model
        instance.transcribe.return_value = ([], MagicMock())
        result = await transcriber.transcribe(_silence())
        assert result == ""

    @pytest.mark.asyncio
    async def test_strips_leading_trailing_whitespace(self, transcriber, mock_model):
        instance, _ = mock_model
        instance.transcribe.return_value = (_make_segments("  hello world  "), MagicMock())
        result = await transcriber.transcribe(_silence())
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_joins_multiple_segments(self, transcriber, mock_model):
        instance, _ = mock_model
        instance.transcribe.return_value = (
            _make_segments("Hello, ", "how are you?"),
            MagicMock(),
        )
        result = await transcriber.transcribe(_silence())
        assert result == "Hello, how are you?"

    @pytest.mark.asyncio
    async def test_publishes_transcription_ready_event(self, transcriber, bus, mock_model):
        instance, _ = mock_model
        instance.transcribe.return_value = (_make_segments("test speech"), MagicMock())
        await transcriber.transcribe(_silence())
        bus.publish.assert_called_once_with(EventType.TRANSCRIPTION_READY, data="test speech")

    @pytest.mark.asyncio
    async def test_no_event_published_on_empty_result(self, transcriber, bus, mock_model):
        instance, _ = mock_model
        instance.transcribe.return_value = ([], MagicMock())
        await transcriber.transcribe(_silence())
        bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_event_on_whitespace_only_result(self, transcriber, bus, mock_model):
        instance, _ = mock_model
        instance.transcribe.return_value = (_make_segments("   "), MagicMock())
        await transcriber.transcribe(_silence())
        bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_language_to_model(self, cfg_ko, bus, mock_model):
        instance, _ = mock_model
        instance.transcribe.return_value = (_make_segments("안녕"), MagicMock())
        t = Transcriber(cfg_ko, bus)
        await t.transcribe(_silence())
        _, kwargs = instance.transcribe.call_args
        assert kwargs["language"] == "ko"

    @pytest.mark.asyncio
    async def test_passes_none_language_for_auto_detect(self, transcriber, mock_model):
        instance, _ = mock_model
        instance.transcribe.return_value = (_make_segments("hello"), MagicMock())
        await transcriber.transcribe(_silence())
        _, kwargs = instance.transcribe.call_args
        assert kwargs["language"] is None

    @pytest.mark.asyncio
    async def test_audio_converted_to_float32(self, transcriber, mock_model):
        instance, _ = mock_model
        instance.transcribe.return_value = ([], MagicMock())
        int_audio = np.zeros(512, dtype=np.int16)
        await transcriber.transcribe(int_audio)
        positional_args, _ = instance.transcribe.call_args
        assert positional_args[0].dtype == np.float32

    @pytest.mark.asyncio
    async def test_audio_is_1d(self, transcriber, mock_model):
        instance, _ = mock_model
        instance.transcribe.return_value = ([], MagicMock())
        audio_2d = np.zeros((1, 512), dtype=np.float32)
        await transcriber.transcribe(audio_2d)
        positional_args, _ = instance.transcribe.call_args
        assert positional_args[0].ndim == 1

    @pytest.mark.asyncio
    async def test_beam_size_is_5(self, transcriber, mock_model):
        instance, _ = mock_model
        instance.transcribe.return_value = ([], MagicMock())
        await transcriber.transcribe(_silence())
        _, kwargs = instance.transcribe.call_args
        assert kwargs["beam_size"] == 5

    @pytest.mark.asyncio
    async def test_returns_string_type(self, transcriber, mock_model):
        instance, _ = mock_model
        instance.transcribe.return_value = (_make_segments("text"), MagicMock())
        result = await transcriber.transcribe(_silence())
        assert isinstance(result, str)
