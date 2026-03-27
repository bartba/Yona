"""Tests for src/tts.py

Run with:
    pytest tests/test_tts.py -v

All tests are hardware-free — supertonic.TTS is mocked via patch.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import Config
from src.tts import (
    SupertonicSynthesizer,
    Synthesizer,
    _clean_tts_text,
    create_synthesizer,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path, extra_yaml: str = "") -> Config:
    base = textwrap.dedent("""\
        tts:
          provider: "supertonic"
          language: "ko"
          voice: "M1"
          speed: 1.05
          total_steps: 5
          output_sample_rate: 44100
    """)
    f = tmp_path / "config.yaml"
    f.write_text(base + "\n" + extra_yaml)
    return Config(path=f)


def _mock_supertonic_instance() -> MagicMock:
    """Create a mock Supertonic TTS instance with sensible defaults."""
    instance = MagicMock()
    instance.sample_rate = 44100
    instance.voice_style_names = ["F1", "F2", "M1", "M2"]
    instance.get_voice_style.return_value = MagicMock(name="VoiceStyle")
    instance.synthesize.return_value = (
        np.zeros((1, 4410), dtype=np.float32),
        np.array([0.1]),
    )
    return instance


# ---------------------------------------------------------------------------
# TestSynthesizerProtocol
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TestCleanTtsText
# ---------------------------------------------------------------------------

class TestCleanTtsText:
    """Verify _clean_tts_text normalisation."""

    def test_strips_whitespace(self):
        assert _clean_tts_text("  hello  ") == "hello"

    def test_removes_newlines(self):
        assert _clean_tts_text("line one\nline two") == "line one line two"

    def test_removes_markdown_bold(self):
        assert _clean_tts_text("**bold** text") == "bold text"

    def test_removes_markdown_italic(self):
        assert _clean_tts_text("*italic* text") == "italic text"

    def test_removes_markdown_code(self):
        assert _clean_tts_text("`code` here") == "code here"

    def test_removes_markdown_heading(self):
        assert _clean_tts_text("# Heading") == "Heading"

    def test_removes_brackets(self):
        assert _clean_tts_text("[link](url)") == "link, url,"

    def test_removes_double_quotes(self):
        assert _clean_tts_text('He said "hello"') == "He said hello"

    def test_removes_fancy_quotes(self):
        assert _clean_tts_text("\u201c\ud55c\uad6d\uc5b4\u201d") == "\ud55c\uad6d\uc5b4"

    def test_keeps_apostrophes(self):
        assert _clean_tts_text("don't stop") == "don't stop"

    def test_replaces_em_dash(self):
        assert _clean_tts_text("first\u2014second") == "first, second"

    def test_replaces_en_dash(self):
        assert _clean_tts_text("first \u2013 second") == "first, second"

    def test_normalizes_multiple_spaces(self):
        assert _clean_tts_text("too   many   spaces") == "too many spaces"

    def test_removes_letter_spelling(self):
        assert _clean_tts_text("L-O-V-E") == "LOVE"

    def test_keeps_normal_hyphenated_words(self):
        """Regular hyphenated words (multi-char segments) are not collapsed."""
        assert _clean_tts_text("well-known artist") == "well-known artist"

    def test_unwraps_parentheses(self):
        assert _clean_tts_text("워싱턴(미국)에서") == "워싱턴, 미국, 에서"

    def test_removes_empty_parentheses(self):
        assert _clean_tts_text("hello () world") == "hello world"

    def test_removes_abbreviation_dots(self):
        assert _clean_tts_text("D.C.에서") == "DC에서"
        assert _clean_tts_text("St. Thomas") == "St. Thomas"  # single dot + space kept

    def test_combined_cleaning(self):
        raw = '**듀크** 엘링턴의 "Satin Doll"\n정말 좋은 곡이에요!'
        cleaned = _clean_tts_text(raw)
        assert cleaned == "듀크 엘링턴의 Satin Doll 정말 좋은 곡이에요!"

    def test_combined_complex(self):
        raw = 'D.C.에서 태어난 "듀크" (1899-1974)는 L-O-V-E를 작곡했어요.'
        cleaned = _clean_tts_text(raw)
        assert "DC" in cleaned
        assert "LOVE" in cleaned
        assert "1899-1974" in cleaned
        assert "(" not in cleaned
        assert '"' not in cleaned

    def test_empty_after_cleaning(self):
        assert _clean_tts_text('""') == ""
        assert _clean_tts_text("  \n  ") == ""

    def test_korean_with_numbers(self):
        assert _clean_tts_text("1899년에 태어났어요.") == "1899년에 태어났어요."


# ---------------------------------------------------------------------------
# TestSynthesizerProtocol
# ---------------------------------------------------------------------------

class TestSynthesizerProtocol:
    def test_supertonic_satisfies_protocol(self, tmp_path):
        with patch("supertonic.TTS", return_value=_mock_supertonic_instance()):
            synth = SupertonicSynthesizer(_cfg(tmp_path))
        assert isinstance(synth, Synthesizer)


# ---------------------------------------------------------------------------
# TestSupertonicSynthesizer
# ---------------------------------------------------------------------------

class TestSupertonicSynthesizer:
    @pytest.fixture
    def mock_tts(self):
        instance = _mock_supertonic_instance()
        with patch("supertonic.TTS", return_value=instance):
            yield instance

    @pytest.fixture
    def synth(self, tmp_path, mock_tts):
        return SupertonicSynthesizer(_cfg(tmp_path))

    def test_init_creates_tts(self, synth, mock_tts):
        mock_tts.get_voice_style.assert_called_with("M1")

    def test_init_reads_sample_rate(self, synth):
        assert synth._sample_rate == 44100

    def test_init_reads_config(self, synth):
        assert synth._language == "ko"
        assert synth._speed == 1.05
        assert synth._total_steps == 5

    @pytest.mark.asyncio
    async def test_synthesize_returns_audio_and_sr(self, synth):
        samples, sr = await synth.synthesize("안녕하세요")
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32
        assert sr == 44100

    @pytest.mark.asyncio
    async def test_synthesize_flattens_2d_output(self, synth, mock_tts):
        mock_tts.synthesize.return_value = (
            np.zeros((1, 8820), dtype=np.float32),
            np.array([0.2]),
        )
        samples, sr = await synth.synthesize("test")
        assert samples.ndim == 1
        # 8820 raw samples + 50ms leading silence pad (44100 * 0.05 = 2205)
        assert len(samples) == 8820 + int(44100 * 0.05)

    @pytest.mark.asyncio
    async def test_synthesize_calls_engine(self, synth, mock_tts):
        await synth.synthesize("hello")
        mock_tts.synthesize.assert_called_once_with(
            "hello",
            voice_style=synth._voice_style,
            lang="ko",
            speed=1.05,
            total_steps=5,
        )

    @pytest.mark.asyncio
    async def test_synthesize_cleans_text(self, synth, mock_tts):
        """Markdown and quotes are stripped before reaching the engine."""
        await synth.synthesize('**bold** "quoted"')
        mock_tts.synthesize.assert_called_once_with(
            "bold quoted",
            voice_style=synth._voice_style,
            lang="ko",
            speed=1.05,
            total_steps=5,
        )

    @pytest.mark.asyncio
    async def test_synthesize_empty_after_cleaning(self, synth, mock_tts):
        """Empty text after cleaning returns minimal silent audio."""
        samples, sr = await synth.synthesize('""')
        assert len(samples) == 1
        mock_tts.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesize_returns_float32(self, synth, mock_tts):
        mock_tts.synthesize.return_value = (
            np.zeros((1, 100), dtype=np.float64),
            np.array([0.01]),
        )
        samples, sr = await synth.synthesize("test")
        assert samples.dtype == np.float32

    @pytest.mark.asyncio
    async def test_set_language(self, synth):
        await synth.set_language("en")
        assert synth._language == "en"

    @pytest.mark.asyncio
    async def test_set_language_no_op_same(self, synth):
        await synth.set_language("ko")
        assert synth._language == "ko"

    @pytest.mark.asyncio
    async def test_set_language_normalizes_case(self, synth):
        await synth.set_language("EN")
        assert synth._language == "en"

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self, synth):
        await synth.close()

    def test_custom_voice(self, tmp_path, mock_tts):
        cfg = _cfg(tmp_path, 'tts:\n  voice: "F2"')
        SupertonicSynthesizer(cfg)
        mock_tts.get_voice_style.assert_called_with("F2")

    def test_custom_speed(self, tmp_path, mock_tts):
        cfg = _cfg(tmp_path, 'tts:\n  speed: 0.8')
        synth = SupertonicSynthesizer(cfg)
        assert synth._speed == 0.8


# ---------------------------------------------------------------------------
# TestCreateSynthesizer
# ---------------------------------------------------------------------------

class TestCreateSynthesizer:
    def test_creates_supertonic_by_default(self, tmp_path):
        with patch("supertonic.TTS", return_value=_mock_supertonic_instance()):
            synth = create_synthesizer(_cfg(tmp_path))
        assert isinstance(synth, SupertonicSynthesizer)

    def test_raises_on_unknown_provider(self, tmp_path):
        cfg = _cfg(tmp_path, 'tts:\n  provider: "unknown_tts"')
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            create_synthesizer(cfg)

    def test_provider_case_insensitive(self, tmp_path):
        with patch("supertonic.TTS", return_value=_mock_supertonic_instance()):
            cfg = _cfg(tmp_path, 'tts:\n  provider: "Supertonic"')
            synth = create_synthesizer(cfg)
        assert isinstance(synth, SupertonicSynthesizer)

    def test_provider_strips_whitespace(self, tmp_path):
        with patch("supertonic.TTS", return_value=_mock_supertonic_instance()):
            cfg = _cfg(tmp_path, 'tts:\n  provider: " supertonic "')
            synth = create_synthesizer(cfg)
        assert isinstance(synth, SupertonicSynthesizer)
