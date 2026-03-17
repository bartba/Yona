"""Tests for src/tts.py

Run with:
    pytest tests/test_tts.py -v

All tests are hardware-free — melo is stubbed via sys.modules.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub TTS SDKs before src.tts is imported (not installed on dev hardware)
# ---------------------------------------------------------------------------
_melo_stub = MagicMock()
_melo_api_stub = MagicMock()

sys.modules.setdefault("melo", _melo_stub)
sys.modules.setdefault("melo.api", _melo_api_stub)

from src.config import Config  # noqa: E402
from src.tts import (  # noqa: E402
    MeloSynthesizer,
    Synthesizer,
    create_synthesizer,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path, extra_yaml: str = "") -> Config:
    base = textwrap.dedent("""\
        tts:
          provider: "melo"
          melo_language: "KR"
          melo_device: "cpu"
          melo_speed: 1.0
          output_sample_rate: 24000
    """)
    f = tmp_path / "config.yaml"
    f.write_text(base + "\n" + extra_yaml)
    return Config(path=f)


# ---------------------------------------------------------------------------
# TestSynthesizerProtocol
# ---------------------------------------------------------------------------

class TestSynthesizerProtocol:
    def test_melo_satisfies_protocol(self, tmp_path):
        mock_engine = MagicMock()
        mock_engine.hps.data.spk2id = {"KR": 0}
        _melo_api_stub.TTS.return_value = mock_engine
        synth = MeloSynthesizer(_cfg(tmp_path))
        assert isinstance(synth, Synthesizer)


# ---------------------------------------------------------------------------
# TestMeloSynthesizer
# ---------------------------------------------------------------------------

class TestMeloSynthesizer:
    @pytest.fixture
    def mock_melo_engine(self):
        mock_engine = MagicMock()
        mock_engine.hps.data.spk2id = {"KR": 0}
        mock_engine.tts_to_file.return_value = np.zeros(4800, dtype=np.float32)
        _melo_api_stub.TTS.return_value = mock_engine
        return mock_engine

    @pytest.fixture
    def synth(self, tmp_path, mock_melo_engine):
        return MeloSynthesizer(_cfg(tmp_path))

    def test_init_creates_tts_engine(self, tmp_path, mock_melo_engine):
        MeloSynthesizer(_cfg(tmp_path))
        _melo_api_stub.TTS.assert_called_with(language="KR", device="cpu")

    def test_init_picks_first_speaker_id(self, tmp_path, mock_melo_engine):
        mock_melo_engine.hps.data.spk2id = {"KR": 42, "EN": 7}
        synth = MeloSynthesizer(_cfg(tmp_path))
        assert synth._speaker_id == 42

    @pytest.mark.asyncio
    async def test_synthesize_returns_audio_and_sr(self, synth, mock_melo_engine):
        mock_melo_engine.tts_to_file.return_value = np.zeros(4800, dtype=np.float32)
        samples, sr = await synth.synthesize("안녕하세요")
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32
        assert sr == 24000

    @pytest.mark.asyncio
    async def test_synthesize_calls_tts_to_file(self, synth, mock_melo_engine):
        mock_melo_engine.tts_to_file.return_value = np.zeros(100, dtype=np.float32)
        await synth.synthesize("hello")
        mock_melo_engine.tts_to_file.assert_called_once_with(
            "hello",
            0,  # speaker_id
            quiet=True,
            speed=1.0,
        )

    @pytest.mark.asyncio
    async def test_synthesize_passes_config_speed(self, tmp_path, mock_melo_engine):
        cfg = _cfg(tmp_path, 'tts:\n  melo_speed: 0.8\n  melo_language: "KR"\n  melo_device: "cpu"\n  output_sample_rate: 24000')
        mock_melo_engine.hps.data.spk2id = {"KR": 0}
        mock_melo_engine.tts_to_file.return_value = np.zeros(100, dtype=np.float32)
        synth = MeloSynthesizer(cfg)
        await synth.synthesize("test")
        mock_melo_engine.tts_to_file.assert_called_once_with(
            "test",
            0,
            quiet=True,
            speed=0.8,
        )

    @pytest.mark.asyncio
    async def test_synthesize_returns_float32(self, synth, mock_melo_engine):
        mock_melo_engine.tts_to_file.return_value = np.zeros(100, dtype=np.float64)
        samples, sr = await synth.synthesize("test")
        assert samples.dtype == np.float32

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self, synth):
        await synth.close()


# ---------------------------------------------------------------------------
# TestCreateSynthesizer
# ---------------------------------------------------------------------------

class TestCreateSynthesizer:
    def test_creates_melo_by_default(self, tmp_path):
        mock_engine = MagicMock()
        mock_engine.hps.data.spk2id = {"KR": 0}
        _melo_api_stub.TTS.return_value = mock_engine
        synth = create_synthesizer(_cfg(tmp_path))
        assert isinstance(synth, MeloSynthesizer)

    def test_raises_on_unknown_provider(self, tmp_path):
        cfg = _cfg(tmp_path, 'tts:\n  provider: "unknown_tts"')
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            create_synthesizer(cfg)

    def test_provider_case_insensitive(self, tmp_path):
        mock_engine = MagicMock()
        mock_engine.hps.data.spk2id = {"KR": 0}
        _melo_api_stub.TTS.return_value = mock_engine
        cfg = _cfg(tmp_path, 'tts:\n  provider: "Melo"')
        synth = create_synthesizer(cfg)
        assert isinstance(synth, MeloSynthesizer)

    def test_provider_strips_whitespace(self, tmp_path):
        mock_engine = MagicMock()
        mock_engine.hps.data.spk2id = {"KR": 0}
        _melo_api_stub.TTS.return_value = mock_engine
        cfg = _cfg(tmp_path, 'tts:\n  provider: " melo "')
        synth = create_synthesizer(cfg)
        assert isinstance(synth, MeloSynthesizer)
