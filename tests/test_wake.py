"""Tests for src/wake.py (openWakeWord)

Run with:
    pytest tests/test_wake.py -v

All tests are hardware-free — openwakeword is stubbed via sys.modules before
src.wake is imported, then openwakeword.model.Model is patched per-fixture.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub openwakeword before src.wake is imported (not installed on dev machine)
# ---------------------------------------------------------------------------
_oww_model_stub = MagicMock()
_oww_stub = MagicMock()
_oww_stub.model = _oww_model_stub
sys.modules.setdefault("openwakeword", _oww_stub)
sys.modules.setdefault("openwakeword.model", _oww_model_stub)

from src.config import Config          # noqa: E402
from src.events import EventBus, EventType  # noqa: E402
from src.wake import WakeWordDetector  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silence(n: int = 512) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_WAKE_YAML = textwrap.dedent("""\
    wake_word:
      model_paths:
        - "models/wake_word/hi_inspector.onnx"
      inference_framework: "onnx"
      threshold: 0.5
      patience: 3
      cooldown_seconds: 2.0
    """)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    f = tmp_path / "config.yaml"
    f.write_text(_WAKE_YAML)
    return Config(path=f)


@pytest.fixture
def bus() -> MagicMock:
    return MagicMock(spec=EventBus)


@pytest.fixture
def mock_oww():
    """Patch openwakeword.model.Model and return the mock instance."""
    with patch("src.wake.OwwModel") as mock_cls:
        instance = MagicMock()
        instance.models = {"hi_inspector": MagicMock()}
        instance.predict.return_value = {"hi_inspector": 0.0}
        instance.reset.return_value = None
        mock_cls.return_value = instance
        yield instance, mock_cls


@pytest.fixture
def detector(cfg, bus, mock_oww):
    _ = mock_oww  # ensures OwwModel is patched
    return WakeWordDetector(cfg, bus)


# ---------------------------------------------------------------------------
# TestWakeWordDetectorInit
# ---------------------------------------------------------------------------

class TestWakeWordDetectorInit:
    def test_calls_oww_model_constructor(self, cfg, bus, mock_oww):
        _, mock_cls = mock_oww
        WakeWordDetector(cfg, bus)
        mock_cls.assert_called_once_with(
            wakeword_models=["models/wake_word/hi_inspector.onnx"],
            inference_framework="onnx",
        )

    def test_reads_threshold(self, detector):
        assert detector._threshold == 0.5

    def test_reads_patience(self, detector):
        assert detector._patience == 3

    def test_reads_cooldown(self, detector):
        assert detector._cooldown == 2.0

    def test_initial_last_trigger_is_zero(self, detector):
        assert detector._last_trigger == 0.0

    def test_model_names_property(self, detector):
        assert detector.model_names == ["hi_inspector"]


# ---------------------------------------------------------------------------
# TestWakeWordDetectorProcessChunk
# ---------------------------------------------------------------------------

class TestWakeWordDetectorProcessChunk:
    def test_returns_predictions_dict(self, detector, mock_oww):
        handle, _ = mock_oww
        handle.predict.return_value = {"hi_inspector": 0.0}
        result = detector.process_chunk(_silence())
        assert result == {"hi_inspector": 0.0}

    def test_calls_oww_predict(self, detector, mock_oww):
        handle, _ = mock_oww
        detector.process_chunk(_silence())
        handle.predict.assert_called_once()

    def test_predict_receives_int16(self, detector, mock_oww):
        handle, _ = mock_oww
        detector.process_chunk(_silence())
        pcm = handle.predict.call_args[0][0]
        assert pcm.dtype == np.int16

    def test_predict_receives_correct_length(self, detector, mock_oww):
        handle, _ = mock_oww
        detector.process_chunk(_silence(512))
        pcm = handle.predict.call_args[0][0]
        assert len(pcm) == 512

    def test_any_chunk_size_accepted(self, detector, mock_oww):
        """openWakeWord accepts arbitrary chunk sizes (internally buffers)."""
        handle, _ = mock_oww
        detector.process_chunk(_silence(1024))
        pcm = handle.predict.call_args[0][0]
        assert len(pcm) == 1024

    def test_short_chunk_accepted(self, detector, mock_oww):
        handle, _ = mock_oww
        detector.process_chunk(_silence(256))
        pcm = handle.predict.call_args[0][0]
        assert len(pcm) == 256

    def test_float32_to_int16_positive(self, detector, mock_oww):
        """+1.0 float32 -> +32767 int16."""
        handle, _ = mock_oww
        chunk = np.ones(512, dtype=np.float32)
        detector.process_chunk(chunk)
        pcm = handle.predict.call_args[0][0]
        assert pcm[0] == 32767

    def test_float32_to_int16_negative(self, detector, mock_oww):
        """-1.0 float32 -> -32767 int16."""
        handle, _ = mock_oww
        chunk = np.full(512, -1.0, dtype=np.float32)
        detector.process_chunk(chunk)
        pcm = handle.predict.call_args[0][0]
        assert pcm[0] == -32767

    def test_float32_to_int16_silence(self, detector, mock_oww):
        """0.0 float32 -> 0 int16."""
        handle, _ = mock_oww
        detector.process_chunk(_silence())
        pcm = handle.predict.call_args[0][0]
        assert np.all(pcm == 0)

    def test_clipping_does_not_overflow(self, detector, mock_oww):
        """Values beyond [-1, 1] must be clipped, not overflow."""
        handle, _ = mock_oww
        chunk = np.full(512, 2.0, dtype=np.float32)
        detector.process_chunk(chunk)
        pcm = handle.predict.call_args[0][0]
        assert np.all(pcm == 32767)

    def test_predict_called_without_threshold_patience(self, detector, mock_oww):
        """predict() is called with raw audio only — no threshold/patience kwargs."""
        handle, _ = mock_oww
        detector.process_chunk(_silence())
        kwargs = handle.predict.call_args[1]
        assert "threshold" not in kwargs
        assert "patience" not in kwargs


# ---------------------------------------------------------------------------
# TestWakeWordDetectorDetection (patience-based)
# ---------------------------------------------------------------------------

class TestWakeWordDetectorDetection:
    def _feed_n(self, detector, handle, score: float, n: int, t: float) -> None:
        """Feed n chunks with the given score at monotonic time t."""
        handle.predict.return_value = {"hi_inspector": score}
        with patch("src.wake.time.monotonic", return_value=t):
            for _ in range(n):
                detector.process_chunk(_silence())

    def test_no_event_before_patience_reached(self, detector, bus, mock_oww):
        """2 frames above threshold (patience=3) should not trigger."""
        handle, _ = mock_oww
        self._feed_n(detector, handle, 0.8, 2, 100.0)
        bus.publish_nowait.assert_not_called()

    def test_publishes_after_patience_reached(self, detector, bus, mock_oww):
        """3 consecutive frames above threshold should trigger."""
        handle, _ = mock_oww
        self._feed_n(detector, handle, 0.8, 3, 100.0)
        bus.publish_nowait.assert_called_once_with(
            EventType.WAKE_WORD_DETECTED, data="hi_inspector"
        )

    def test_patience_resets_on_below_threshold(self, detector, bus, mock_oww):
        """If score drops below threshold, patience counter resets."""
        handle, _ = mock_oww
        self._feed_n(detector, handle, 0.8, 2, 100.0)  # 2 above
        self._feed_n(detector, handle, 0.3, 1, 100.0)   # 1 below — resets
        self._feed_n(detector, handle, 0.8, 2, 100.0)  # 2 above again
        bus.publish_nowait.assert_not_called()           # still only 2, not 3

    def test_no_event_when_below_threshold(self, detector, bus, mock_oww):
        handle, _ = mock_oww
        self._feed_n(detector, handle, 0.3, 10, 100.0)
        bus.publish_nowait.assert_not_called()

    def test_no_event_when_score_is_zero(self, detector, bus, mock_oww):
        handle, _ = mock_oww
        self._feed_n(detector, handle, 0.0, 10, 100.0)
        bus.publish_nowait.assert_not_called()

    def test_updates_last_trigger_on_detection(self, detector, mock_oww):
        handle, _ = mock_oww
        self._feed_n(detector, handle, 0.8, 3, 42.0)
        assert detector._last_trigger == 42.0

    def test_last_trigger_unchanged_when_below_threshold(self, detector, mock_oww):
        handle, _ = mock_oww
        self._feed_n(detector, handle, 0.3, 5, 100.0)
        assert detector._last_trigger == 0.0


# ---------------------------------------------------------------------------
# TestWakeWordDetectorCooldown
# ---------------------------------------------------------------------------

class TestWakeWordDetectorCooldown:
    def _trigger_at(self, detector, handle, t: float) -> None:
        """Simulate patience-met detection at time t."""
        handle.predict.return_value = {"hi_inspector": 0.8}
        with patch("src.wake.time.monotonic", return_value=t):
            for _ in range(3):  # patience=3
                detector.process_chunk(_silence())

    def test_second_trigger_within_cooldown_suppressed(self, detector, bus, mock_oww):
        handle, _ = mock_oww
        self._trigger_at(detector, handle, 0.0)   # first detection at t=0
        bus.reset_mock()
        self._trigger_at(detector, handle, 1.0)   # t=1 — only 1 s elapsed (cooldown=2 s)
        bus.publish_nowait.assert_not_called()

    def test_second_trigger_after_cooldown_published(self, detector, bus, mock_oww):
        handle, _ = mock_oww
        self._trigger_at(detector, handle, 0.0)   # first detection at t=0
        bus.reset_mock()
        self._trigger_at(detector, handle, 2.0)   # t=2 — cooldown elapsed exactly
        bus.publish_nowait.assert_called_once_with(
            EventType.WAKE_WORD_DETECTED, data="hi_inspector"
        )

    def test_trigger_exactly_at_cooldown_boundary_accepted(self, detector, bus, mock_oww):
        """Trigger at t=cooldown should be accepted (>= check)."""
        handle, _ = mock_oww
        detector._last_trigger = 10.0
        self._trigger_at(detector, handle, 12.0)  # 12-10=2 == cooldown
        bus.publish_nowait.assert_called_once()

    def test_cooldown_does_not_affect_return_value(self, detector, mock_oww):
        """process_chunk returns predictions even when cooldown suppresses the event."""
        handle, _ = mock_oww
        handle.predict.return_value = {"hi_inspector": 0.8}
        with patch("src.wake.time.monotonic", return_value=0.0):
            for _ in range(3):
                detector.process_chunk(_silence())  # triggers at t=0
        with patch("src.wake.time.monotonic", return_value=0.5):
            result = detector.process_chunk(_silence())  # within cooldown
        assert result == {"hi_inspector": 0.8}


# ---------------------------------------------------------------------------
# TestWakeWordDetectorReset
# ---------------------------------------------------------------------------

class TestWakeWordDetectorReset:
    def test_reset_calls_model_reset(self, detector, mock_oww):
        handle, _ = mock_oww
        detector.reset()
        handle.reset.assert_called_once()

    def test_reset_clears_patience_counters(self, detector, mock_oww):
        handle, _ = mock_oww
        handle.predict.return_value = {"hi_inspector": 0.8}
        with patch("src.wake.time.monotonic", return_value=100.0):
            detector.process_chunk(_silence())  # patience_count = 1
            detector.process_chunk(_silence())  # patience_count = 2
        detector.reset()
        assert detector._patience_count == {}
