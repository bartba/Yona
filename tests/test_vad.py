"""Tests for src/vad.py

Run with:
    pytest tests/test_vad.py -v

All tests are hardware-free — onnxruntime.InferenceSession is mocked via
patch("src.vad.ort.InferenceSession").
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from src.config import Config
from src.events import EventBus, EventType
from src.vad import VoiceActivityDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ort_result(prob: float) -> list:
    """Return a fake ort.InferenceSession.run() result for *prob*."""
    return [
        np.array([[prob]], dtype=np.float32),
        np.zeros((2, 1, 128), dtype=np.float32),
        np.zeros((2, 1, 128), dtype=np.float32),
    ]


def _silence() -> np.ndarray:
    return np.zeros(512, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VAD_YAML = textwrap.dedent("""\
    audio:
      input_sample_rate: 16000
      chunk_size: 512
    vad:
      model_path: "models/silero_vad/silero_vad.onnx"
      threshold: 0.5
      silence_duration: 1.5
      min_speech_duration: 0.3
      barge_in_threshold: 0.7
    """)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    f = tmp_path / "config.yaml"
    f.write_text(_VAD_YAML)
    return Config(path=f)


@pytest.fixture
def bus() -> MagicMock:
    return MagicMock(spec=EventBus)


@pytest.fixture
def mock_session():
    """Patch ort.InferenceSession and return the mock session instance."""
    with patch("src.vad.ort.InferenceSession") as mock_cls:
        session = MagicMock()
        mock_cls.return_value = session
        session.run.return_value = _make_ort_result(0.0)
        yield session, mock_cls


@pytest.fixture
def vad(cfg, bus, mock_session):
    session, _ = mock_session
    return VoiceActivityDetector(cfg, bus)


# ---------------------------------------------------------------------------
# TestVADInit
# ---------------------------------------------------------------------------

class TestVADInit:
    def test_creates_onnx_session_with_model_path(self, cfg, bus, mock_session):
        _, mock_cls = mock_session
        VoiceActivityDetector(cfg, bus)
        mock_cls.assert_called_once_with("models/silero_vad/silero_vad.onnx")

    def test_reads_threshold(self, vad):
        assert vad._threshold == 0.5

    def test_reads_barge_in_threshold(self, vad):
        assert vad._barge_in_threshold == 0.7

    def test_reads_silence_duration(self, vad):
        assert vad._silence_duration == 1.5

    def test_reads_min_speech_duration(self, vad):
        assert vad._min_speech_duration == 0.3

    def test_reads_sample_rate(self, vad):
        assert vad._sample_rate == 16_000

    def test_reads_chunk_size(self, vad):
        assert vad._chunk_size == 512

    def test_initial_lstm_h_shape(self, vad):
        assert vad._h.shape == (2, 1, 128)
        assert vad._h.dtype == np.float32

    def test_initial_lstm_c_shape(self, vad):
        assert vad._c.shape == (2, 1, 128)
        assert vad._c.dtype == np.float32

    def test_initial_lstm_state_is_zeros(self, vad):
        np.testing.assert_array_equal(vad._h, np.zeros((2, 1, 128)))
        np.testing.assert_array_equal(vad._c, np.zeros((2, 1, 128)))

    def test_initial_speech_not_active(self, vad):
        assert vad._speech_active is False

    def test_initial_barge_in_mode_off(self, vad):
        assert vad._barge_in_mode is False


# ---------------------------------------------------------------------------
# TestVADProcessChunk
# ---------------------------------------------------------------------------

class TestVADProcessChunk:
    def test_returns_float(self, vad, mock_session):
        session, _ = mock_session
        session.run.return_value = _make_ort_result(0.3)
        result = vad.process_chunk(_silence())
        assert isinstance(result, float)

    def test_returns_correct_probability(self, vad, mock_session):
        session, _ = mock_session
        session.run.return_value = _make_ort_result(0.42)
        result = vad.process_chunk(_silence())
        assert abs(result - 0.42) < 1e-6

    def test_calls_session_run(self, vad, mock_session):
        session, _ = mock_session
        vad.process_chunk(_silence())
        session.run.assert_called_once()

    def test_input_tensor_shape(self, vad, mock_session):
        session, _ = mock_session
        vad.process_chunk(_silence())
        ort_inputs = session.run.call_args[0][1]
        assert ort_inputs["input"].shape == (1, 512)

    def test_input_tensor_dtype(self, vad, mock_session):
        session, _ = mock_session
        vad.process_chunk(_silence())
        ort_inputs = session.run.call_args[0][1]
        assert ort_inputs["input"].dtype == np.float32

    def test_sr_value_and_dtype(self, vad, mock_session):
        session, _ = mock_session
        vad.process_chunk(_silence())
        ort_inputs = session.run.call_args[0][1]
        assert int(ort_inputs["sr"]) == 16_000
        assert ort_inputs["sr"].dtype == np.int64

    def test_short_chunk_padded_to_chunk_size(self, vad, mock_session):
        session, _ = mock_session
        short = np.zeros(256, dtype=np.float32)
        vad.process_chunk(short)
        ort_inputs = session.run.call_args[0][1]
        assert ort_inputs["input"].shape == (1, 512)

    def test_long_chunk_truncated_to_chunk_size(self, vad, mock_session):
        session, _ = mock_session
        long_chunk = np.zeros(1024, dtype=np.float32)
        vad.process_chunk(long_chunk)
        ort_inputs = session.run.call_args[0][1]
        assert ort_inputs["input"].shape == (1, 512)

    def test_lstm_state_updated_after_call(self, vad, mock_session):
        session, _ = mock_session
        new_h = np.ones((2, 1, 128), dtype=np.float32) * 0.5
        new_c = np.ones((2, 1, 128), dtype=np.float32) * 0.3
        session.run.return_value = [
            np.array([[0.0]], dtype=np.float32),
            new_h,
            new_c,
        ]
        vad.process_chunk(_silence())
        np.testing.assert_array_equal(vad._h, new_h)
        np.testing.assert_array_equal(vad._c, new_c)

    def test_previous_lstm_state_passed_to_next_call(self, vad, mock_session):
        """LSTM state from first call must be fed into the second call."""
        session, _ = mock_session
        updated_h = np.full((2, 1, 128), 0.9, dtype=np.float32)
        updated_c = np.full((2, 1, 128), 0.8, dtype=np.float32)
        session.run.side_effect = [
            [np.array([[0.0]], dtype=np.float32), updated_h, updated_c],
            [np.array([[0.0]], dtype=np.float32), updated_h, updated_c],
        ]
        vad.process_chunk(_silence())
        vad.process_chunk(_silence())

        second_inputs = session.run.call_args_list[1][0][1]
        np.testing.assert_array_equal(second_inputs["h"], updated_h)
        np.testing.assert_array_equal(second_inputs["c"], updated_c)


# ---------------------------------------------------------------------------
# TestVADSpeechDetection  (normal / LISTENING mode)
# ---------------------------------------------------------------------------

class TestVADSpeechDetection:
    def _run_chunks(self, vad, mock_session, probs, times):
        """Helper: feed *probs* as speech probabilities with mocked *times*."""
        session, _ = mock_session
        session.run.side_effect = [_make_ort_result(p) for p in probs]
        with patch("src.vad.time.monotonic", side_effect=times):
            for _ in probs:
                vad.process_chunk(_silence())

    def test_speech_started_on_first_voiced_chunk(self, vad, bus, mock_session):
        self._run_chunks(vad, mock_session, [0.8], [0.0])
        bus.publish_nowait.assert_called_once_with(EventType.SPEECH_STARTED)

    def test_speech_started_only_once_for_continuous_speech(self, vad, bus, mock_session):
        self._run_chunks(vad, mock_session, [0.8, 0.9, 0.8], [0.0, 0.032, 0.064])
        calls = [c for c in bus.publish_nowait.call_args_list
                 if c == call(EventType.SPEECH_STARTED)]
        assert len(calls) == 1

    def test_no_speech_started_below_threshold(self, vad, bus, mock_session):
        self._run_chunks(vad, mock_session, [0.3, 0.4], [0.0, 0.032])
        bus.publish_nowait.assert_not_called()

    def test_speech_ended_after_silence_duration(self, vad, bus, mock_session):
        # t=0: speech starts, t=0.5: last voiced chunk, t=2.1: silence ends
        probs = [0.8, 0.8, 0.1]
        times = [0.0, 0.5, 2.1]
        self._run_chunks(vad, mock_session, probs, times)
        published = [c[0][0] for c in bus.publish_nowait.call_args_list]
        assert EventType.SPEECH_ENDED in published

    def test_speech_ended_not_published_before_silence_duration(self, vad, bus, mock_session):
        # Silence for only 0.5 s — not enough (silence_duration=1.5)
        probs = [0.8, 0.8, 0.1]
        times = [0.0, 0.5, 1.0]
        self._run_chunks(vad, mock_session, probs, times)
        published = [c[0][0] for c in bus.publish_nowait.call_args_list]
        assert EventType.SPEECH_ENDED not in published

    def test_speech_ended_not_published_if_too_short(self, vad, bus, mock_session):
        # Speech duration = 0 s (single chunk), min_speech_duration=0.3
        # last_speech=0, speech_start=0 → speech_duration=0 < 0.3
        probs = [0.8, 0.1]
        times = [0.0, 2.0]
        self._run_chunks(vad, mock_session, probs, times)
        published = [c[0][0] for c in bus.publish_nowait.call_args_list]
        assert EventType.SPEECH_ENDED not in published

    def test_speech_active_true_after_voiced_chunk(self, vad, mock_session):
        session, _ = mock_session
        session.run.return_value = _make_ort_result(0.8)
        with patch("src.vad.time.monotonic", return_value=0.0):
            vad.process_chunk(_silence())
        assert vad._speech_active is True

    def test_speech_active_false_after_silence_ends(self, vad, mock_session):
        session, _ = mock_session
        session.run.side_effect = [
            _make_ort_result(0.8),
            _make_ort_result(0.8),
            _make_ort_result(0.1),
        ]
        times = [0.0, 0.5, 2.1]
        with patch("src.vad.time.monotonic", side_effect=times):
            for _ in range(3):
                vad.process_chunk(_silence())
        assert vad._speech_active is False


# ---------------------------------------------------------------------------
# TestVADBargeIn  (SPEAKING mode)
# ---------------------------------------------------------------------------

class TestVADBargeIn:
    def _run_chunks(self, vad, mock_session, probs, times):
        session, _ = mock_session
        session.run.side_effect = [_make_ort_result(p) for p in probs]
        with patch("src.vad.time.monotonic", side_effect=times):
            for _ in probs:
                vad.process_chunk(_silence())

    def test_barge_in_detected_on_first_voiced_chunk(self, vad, bus, mock_session):
        vad.set_barge_in_mode(True)
        self._run_chunks(vad, mock_session, [0.9], [0.0])
        bus.publish_nowait.assert_called_once_with(EventType.BARGE_IN_DETECTED)

    def test_no_speech_started_in_barge_in_mode(self, vad, bus, mock_session):
        vad.set_barge_in_mode(True)
        self._run_chunks(vad, mock_session, [0.9], [0.0])
        published = [c[0][0] for c in bus.publish_nowait.call_args_list]
        assert EventType.SPEECH_STARTED not in published

    def test_barge_in_uses_higher_threshold(self, vad, bus, mock_session):
        """Prob=0.6 is above normal threshold (0.5) but below barge-in (0.7)."""
        vad.set_barge_in_mode(True)
        self._run_chunks(vad, mock_session, [0.6], [0.0])
        bus.publish_nowait.assert_not_called()

    def test_barge_in_detected_only_once(self, vad, bus, mock_session):
        vad.set_barge_in_mode(True)
        self._run_chunks(vad, mock_session, [0.9, 0.9, 0.9], [0.0, 0.032, 0.064])
        barge_calls = [c for c in bus.publish_nowait.call_args_list
                       if c == call(EventType.BARGE_IN_DETECTED)]
        assert len(barge_calls) == 1

    def test_no_speech_ended_in_barge_in_mode(self, vad, bus, mock_session):
        # silence_duration (1.5 s) 이상 침묵해도 barge-in 모드에서는 SPEECH_ENDED 미발행
        vad.set_barge_in_mode(True)
        probs = [0.9, 0.9, 0.1]
        times = [0.0, 0.5, 2.1]
        self._run_chunks(vad, mock_session, probs, times)
        published = [c[0][0] for c in bus.publish_nowait.call_args_list]
        assert EventType.SPEECH_ENDED not in published


# ---------------------------------------------------------------------------
# TestVADReset
# ---------------------------------------------------------------------------

class TestVADReset:
    def test_reset_clears_lstm_h(self, vad, mock_session):
        session, _ = mock_session
        session.run.return_value = [
            np.array([[0.0]], dtype=np.float32),
            np.ones((2, 1, 128), dtype=np.float32),
            np.ones((2, 1, 128), dtype=np.float32),
        ]
        with patch("src.vad.time.monotonic", return_value=0.0):
            vad.process_chunk(_silence())
        vad.reset()
        np.testing.assert_array_equal(vad._h, np.zeros((2, 1, 128)))

    def test_reset_clears_lstm_c(self, vad, mock_session):
        session, _ = mock_session
        session.run.return_value = [
            np.array([[0.0]], dtype=np.float32),
            np.ones((2, 1, 128), dtype=np.float32),
            np.ones((2, 1, 128), dtype=np.float32),
        ]
        with patch("src.vad.time.monotonic", return_value=0.0):
            vad.process_chunk(_silence())
        vad.reset()
        np.testing.assert_array_equal(vad._c, np.zeros((2, 1, 128)))

    def test_reset_clears_speech_active(self, vad, mock_session):
        session, _ = mock_session
        session.run.return_value = _make_ort_result(0.8)
        with patch("src.vad.time.monotonic", return_value=0.0):
            vad.process_chunk(_silence())
        assert vad._speech_active is True
        vad.reset()
        assert vad._speech_active is False

    def test_reset_clears_speech_timestamps(self, vad):
        vad._speech_start = 100.0
        vad._last_speech = 200.0
        vad.reset()
        assert vad._speech_start == 0.0
        assert vad._last_speech == 0.0

    def test_speech_started_publishable_after_reset(self, vad, bus, mock_session):
        """After reset, SPEECH_STARTED can be published again."""
        session, _ = mock_session
        session.run.side_effect = [_make_ort_result(0.8)] * 4
        times = [0.0, 0.5, 2.1, 3.0]
        with patch("src.vad.time.monotonic", side_effect=times):
            vad.process_chunk(_silence())  # speech start
            vad.process_chunk(_silence())  # last speech
            vad.process_chunk(_silence())  # silence ends (SPEECH_ENDED if long enough)
        vad.reset()
        bus.reset_mock()
        with patch("src.vad.time.monotonic", return_value=4.0):
            session.run.return_value = _make_ort_result(0.8)
            vad.process_chunk(_silence())
        bus.publish_nowait.assert_called_once_with(EventType.SPEECH_STARTED)


# ---------------------------------------------------------------------------
# TestVADSetMode
# ---------------------------------------------------------------------------

class TestVADSetMode:
    def test_set_barge_in_mode_true(self, vad):
        vad.set_barge_in_mode(True)
        assert vad._barge_in_mode is True

    def test_set_barge_in_mode_false(self, vad):
        vad.set_barge_in_mode(True)
        vad.set_barge_in_mode(False)
        assert vad._barge_in_mode is False

    def test_normal_threshold_used_in_normal_mode(self, vad, bus, mock_session):
        """Prob=0.6 is above normal threshold (0.5) — SPEECH_STARTED expected."""
        session, _ = mock_session
        session.run.return_value = _make_ort_result(0.6)
        with patch("src.vad.time.monotonic", return_value=0.0):
            vad.process_chunk(_silence())
        bus.publish_nowait.assert_called_once_with(EventType.SPEECH_STARTED)
