"""Tests for src/audio.py

Run with:
    pytest tests/test_audio.py -v

All tests are hardware-free — sounddevice is mocked via unittest.mock.patch.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.config import Config
from src.audio import AudioBuffer, AudioManager, ChimePlayer, _resample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_AUDIO_YAML = textwrap.dedent("""\
    audio:
      input_device: "TestInput"
      output_device: "TestOutput"
      input_sample_rate: 16000
      output_sample_rate: 48000
      input_channels: 1
      output_channels: 2
      chunk_size: 512
      buffer_seconds: 30
      chime_path: null
    """)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(_AUDIO_YAML)
    return Config(path=cfg_file)


# ---------------------------------------------------------------------------
# TestAudioBuffer
# ---------------------------------------------------------------------------

class TestAudioBuffer:
    def test_init_defaults(self):
        buf = AudioBuffer()
        assert buf.sample_rate == 16_000
        assert buf.capacity == 16_000 * 30

    def test_init_custom(self):
        buf = AudioBuffer(sample_rate=22_050, buffer_seconds=10.0)
        assert buf.sample_rate == 22_050
        assert buf.capacity == 22_050 * 10

    def test_get_all_empty_returns_empty_array(self):
        buf = AudioBuffer(sample_rate=16_000, buffer_seconds=5.0)
        result = buf.get_all()
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 0

    def test_push_and_get_all(self):
        buf = AudioBuffer(sample_rate=16_000, buffer_seconds=5.0)
        chunk = np.ones(512, dtype=np.float32)
        buf.push(chunk)
        result = buf.get_all()
        np.testing.assert_array_equal(result, chunk)

    def test_push_multiple_chunks_in_order(self):
        buf = AudioBuffer(sample_rate=16_000, buffer_seconds=5.0)
        a = np.full(256, 1.0, dtype=np.float32)
        b = np.full(256, 2.0, dtype=np.float32)
        buf.push(a)
        buf.push(b)
        result = buf.get_all()
        assert len(result) == 512
        np.testing.assert_array_equal(result[:256], a)
        np.testing.assert_array_equal(result[256:], b)

    def test_push_2d_column_array_is_flattened(self):
        buf = AudioBuffer(sample_rate=16_000, buffer_seconds=5.0)
        chunk = np.ones((512, 1), dtype=np.float32)
        buf.push(chunk)
        result = buf.get_all()
        assert result.ndim == 1
        assert len(result) == 512

    def test_push_empty_chunk_is_noop(self):
        buf = AudioBuffer(sample_rate=16_000, buffer_seconds=5.0)
        buf.push(np.array([], dtype=np.float32))
        assert len(buf.get_all()) == 0

    def test_ring_overflow_fills_to_capacity(self):
        # Buffer: 1000 samples
        buf = AudioBuffer(sample_rate=1000, buffer_seconds=1.0)
        # Push 1500 samples (50 % overflow)
        buf.push(np.arange(600, dtype=np.float32))
        buf.push(np.arange(600, 1500, dtype=np.float32))
        result = buf.get_all()
        assert len(result) == 1000

    def test_ring_overflow_oldest_overwritten(self):
        buf = AudioBuffer(sample_rate=1000, buffer_seconds=1.0)
        buf.push(np.zeros(600, dtype=np.float32))       # older data
        second = np.ones(600, dtype=np.float32) * 9.0   # newer data
        buf.push(second)
        result = buf.get_all()
        # The last 400 samples should all come from the 9.0 block
        np.testing.assert_array_equal(result[-400:], np.ones(400) * 9.0)

    def test_get_all_returns_chronological_order_after_wrap(self):
        buf = AudioBuffer(sample_rate=100, buffer_seconds=1.0)  # 100 samples
        first = np.arange(60, dtype=np.float32)
        second = np.arange(100, 160, dtype=np.float32)
        buf.push(first)
        buf.push(second)
        result = buf.get_all()
        assert len(result) == 100
        assert result.dtype == np.float32
        # Second push starts at index 60 and wraps; the last 40 samples are
        # the tail of second (index 120..159)
        np.testing.assert_array_equal(result[-40:], second[-40:])

    def test_reset_clears_buffer(self):
        buf = AudioBuffer(sample_rate=16_000, buffer_seconds=5.0)
        buf.push(np.ones(512, dtype=np.float32))
        assert len(buf.get_all()) == 512
        buf.reset()
        assert len(buf.get_all()) == 0

    def test_reset_allows_reuse(self):
        buf = AudioBuffer(sample_rate=16_000, buffer_seconds=5.0)
        buf.push(np.ones(512, dtype=np.float32))
        buf.reset()
        chunk = np.full(256, 3.0, dtype=np.float32)
        buf.push(chunk)
        result = buf.get_all()
        np.testing.assert_array_equal(result, chunk)

    def test_get_all_returns_independent_copy(self):
        buf = AudioBuffer(sample_rate=16_000, buffer_seconds=5.0)
        buf.push(np.ones(512, dtype=np.float32))
        r1 = buf.get_all()
        r1[:] = 0.0  # mutate the returned copy
        r2 = buf.get_all()
        assert r2[0] == 1.0  # internal buffer unaffected


# ---------------------------------------------------------------------------
# TestResample
# ---------------------------------------------------------------------------

class TestResample:
    def test_identity_same_rate_returns_same_object(self):
        audio = np.random.rand(1000).astype(np.float32)
        result = _resample(audio, 24_000, 24_000)
        assert result is audio  # exact same object — no copy

    def test_2x_upsample_doubles_length(self):
        audio = np.ones(1000, dtype=np.float32)
        result = _resample(audio, 24_000, 48_000)
        assert len(result) == 2000

    def test_2x_upsample_constant_signal_preserved(self):
        audio = np.ones(100, dtype=np.float32)
        result = _resample(audio, 24_000, 48_000)
        np.testing.assert_allclose(result, np.ones(200, dtype=np.float32))

    def test_output_is_float32(self):
        audio = np.random.rand(500).astype(np.float32)
        result = _resample(audio, 16_000, 48_000)
        assert result.dtype == np.float32

    def test_3x_upsample_output_length(self):
        audio = np.zeros(500, dtype=np.float32)
        result = _resample(audio, 16_000, 48_000)
        expected = math.ceil(500 * 48_000 / 16_000)
        assert len(result) == expected

    def test_downsample_output_length(self):
        audio = np.zeros(48_000, dtype=np.float32)
        result = _resample(audio, 48_000, 16_000)
        expected = math.ceil(48_000 * 16_000 / 48_000)
        assert len(result) == expected


# ---------------------------------------------------------------------------
# TestAudioManagerInit
# ---------------------------------------------------------------------------

class TestAudioManagerInit:
    def test_reads_config_values(self, cfg):
        manager = AudioManager(cfg)
        assert manager._input_device == "TestInput"
        assert manager._output_device == "TestOutput"
        assert manager._input_rate == 16_000
        assert manager._output_rate == 48_000
        assert manager._output_channels == 2
        assert manager._chunk_size == 512

    def test_not_playing_initially(self, cfg):
        manager = AudioManager(cfg)
        assert manager.is_playing is False

    def test_has_playback_lock(self, cfg):
        manager = AudioManager(cfg)
        assert hasattr(manager, "_playback_lock")

    def test_no_callbacks_initially(self, cfg):
        manager = AudioManager(cfg)
        assert manager._callbacks == []

    def test_no_input_stream_initially(self, cfg):
        manager = AudioManager(cfg)
        assert manager._input_stream is None


# ---------------------------------------------------------------------------
# TestAudioManagerCallbacks
# ---------------------------------------------------------------------------

class TestAudioManagerCallbacks:
    def test_add_callback(self, cfg):
        manager = AudioManager(cfg)
        cb = MagicMock()
        manager.add_input_callback(cb)
        assert cb in manager._callbacks

    def test_remove_callback(self, cfg):
        manager = AudioManager(cfg)
        cb = MagicMock()
        manager.add_input_callback(cb)
        manager.remove_input_callback(cb)
        assert cb not in manager._callbacks

    def test_remove_unknown_callback_does_not_raise(self, cfg):
        manager = AudioManager(cfg)
        manager.remove_input_callback(MagicMock())  # should not raise

    def test_audio_callback_dispatches_to_registered(self, cfg):
        manager = AudioManager(cfg)
        cb = MagicMock()
        manager.add_input_callback(cb)

        indata = np.zeros((512, 1), dtype=np.float32)
        indata[:, 0] = np.arange(512, dtype=np.float32)
        manager._audio_input_callback(indata, 512, None, None)

        cb.assert_called_once()
        received_chunk = cb.call_args[0][0]
        np.testing.assert_array_equal(received_chunk, indata[:, 0])

    def test_audio_callback_delivers_1d_mono_chunk(self, cfg):
        """Callback always receives a 1-D array even for multi-channel input."""
        manager = AudioManager(cfg)
        received: list[np.ndarray] = []
        manager.add_input_callback(lambda c: received.append(c))

        indata = np.ones((512, 2), dtype=np.float32)
        manager._audio_input_callback(indata, 512, None, None)

        assert received[0].ndim == 1
        assert len(received[0]) == 512

    def test_audio_callback_dispatches_to_multiple_callbacks(self, cfg):
        manager = AudioManager(cfg)
        cb1, cb2 = MagicMock(), MagicMock()
        manager.add_input_callback(cb1)
        manager.add_input_callback(cb2)

        indata = np.zeros((512, 1), dtype=np.float32)
        manager._audio_input_callback(indata, 512, None, None)

        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_removed_callback_not_called(self, cfg):
        manager = AudioManager(cfg)
        cb = MagicMock()
        manager.add_input_callback(cb)
        manager.remove_input_callback(cb)

        indata = np.zeros((512, 1), dtype=np.float32)
        manager._audio_input_callback(indata, 512, None, None)

        cb.assert_not_called()

    def test_audio_callback_delivers_copy_not_view(self, cfg):
        """Chunk passed to callbacks must be a copy, not a view of indata."""
        manager = AudioManager(cfg)
        received: list[np.ndarray] = []
        manager.add_input_callback(lambda c: received.append(c))

        indata = np.ones((512, 1), dtype=np.float32)
        manager._audio_input_callback(indata, 512, None, None)
        indata[:] = 0.0  # mutate original after callback

        assert received[0][0] == 1.0  # copy is unaffected


# ---------------------------------------------------------------------------
# TestAudioManagerLifecycle
# ---------------------------------------------------------------------------

class TestAudioManagerLifecycle:
    @pytest.mark.asyncio
    async def test_start_creates_input_stream(self, cfg):
        with patch("src.audio.sd") as mock_sd:
            manager = AudioManager(cfg)
            mock_stream = MagicMock()
            mock_sd.InputStream.return_value = mock_stream
            await manager.start()

        mock_sd.InputStream.assert_called_once()
        mock_stream.start.assert_called_once()
        assert manager._input_stream is mock_stream

    @pytest.mark.asyncio
    async def test_start_passes_correct_params(self, cfg):
        with patch("src.audio.sd") as mock_sd:
            manager = AudioManager(cfg)
            mock_sd.InputStream.return_value = MagicMock()
            await manager.start()

        kwargs = mock_sd.InputStream.call_args.kwargs
        assert kwargs["device"] == "TestInput"
        assert kwargs["channels"] == 1
        assert kwargs["samplerate"] == 16_000
        assert kwargs["blocksize"] == 512
        assert kwargs["dtype"] == np.float32

    @pytest.mark.asyncio
    async def test_stop_closes_stream(self, cfg):
        with patch("src.audio.sd") as mock_sd:
            manager = AudioManager(cfg)
            mock_stream = MagicMock()
            mock_sd.InputStream.return_value = mock_stream
            await manager.start()
            await manager.stop()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert manager._input_stream is None

    @pytest.mark.asyncio
    async def test_stop_without_start_does_not_raise(self, cfg):
        with patch("src.audio.sd"):
            manager = AudioManager(cfg)
            await manager.stop()  # should not raise


# ---------------------------------------------------------------------------
# TestAudioManagerPlayback
# ---------------------------------------------------------------------------

class TestAudioManagerPlayback:
    @pytest.mark.asyncio
    async def test_play_audio_creates_output_stream(self, cfg):
        """play_audio uses an explicit OutputStream (not sd.play)."""
        with patch("src.audio.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_stream.start = MagicMock()
            mock_stream.stop = MagicMock()
            mock_stream.close = MagicMock()
            mock_sd.OutputStream.return_value = mock_stream
            # Make start() trigger done via callback immediately
            def fake_start():
                # Simulate: callback plays all frames → sets done → CallbackStop
                pass
            mock_stream.start.side_effect = fake_start

            manager = AudioManager(cfg)
            audio = np.zeros(24_000, dtype=np.float32)
            # _play_blocking will hang on done.wait(); mock it
            with patch.object(manager, "_play_blocking") as mock_pb:
                await manager.play_audio(audio, sample_rate=24_000)
            mock_pb.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_audio_resamples_24k_to_48k_stereo(self, cfg):
        """24 kHz mono → 48 kHz stereo via _prepare_audio."""
        manager = AudioManager(cfg)
        n = 1000
        audio = np.zeros(n, dtype=np.float32)
        prepared = manager._prepare_audio(audio, sample_rate=24_000)
        assert prepared.shape == (2000, 2)

    @pytest.mark.asyncio
    async def test_play_audio_no_resample_at_output_rate(self, cfg):
        """Audio already at output_rate skips resampling."""
        manager = AudioManager(cfg)
        n = 1000
        audio = np.zeros(n, dtype=np.float32)
        prepared = manager._prepare_audio(audio, sample_rate=48_000)
        assert prepared.shape == (n, 2)

    @pytest.mark.asyncio
    async def test_is_playing_flag(self, cfg):
        """is_playing is True during playback, False after."""
        states: list[bool] = []
        manager = AudioManager(cfg)

        def mock_play_blocking(prepared):
            states.append(manager.is_playing)

        with patch.object(manager, "_play_blocking", side_effect=mock_play_blocking):
            audio = np.zeros(1000, dtype=np.float32)
            await manager.play_audio(audio, sample_rate=24_000)

        assert states == [True]
        assert manager.is_playing is False

    @pytest.mark.asyncio
    async def test_stop_playback_stops_output_stream(self, cfg):
        """stop_playback stops only the output stream, not sd.stop()."""
        manager = AudioManager(cfg)
        mock_stream = MagicMock()
        manager._output_stream = mock_stream
        await manager.stop_playback()
        mock_stream.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_playback_noop_when_not_playing(self, cfg):
        """stop_playback is safe to call when nothing is playing."""
        manager = AudioManager(cfg)
        assert manager._output_stream is None
        await manager.stop_playback()  # should not raise


# ---------------------------------------------------------------------------
# TestChimePlayerGeneration
# ---------------------------------------------------------------------------

class TestChimePlayerGeneration:
    def test_generate_chime_returns_float32(self):
        chime = ChimePlayer._generate_chime(24_000)
        assert chime.dtype == np.float32

    def test_generate_chime_is_1d(self):
        chime = ChimePlayer._generate_chime(24_000)
        assert chime.ndim == 1

    def test_generate_chime_correct_length(self):
        sr = 24_000
        expected = int(sr * ChimePlayer._DURATION)
        chime = ChimePlayer._generate_chime(sr)
        assert len(chime) == expected

    def test_generate_chime_amplitude_in_range(self):
        chime = ChimePlayer._generate_chime(24_000)
        assert chime.max() <= 0.7 + 1e-5
        assert chime.min() >= -0.7 - 1e-5

    def test_generate_chime_ends_near_zero(self):
        """Fade-out should bring the last sample close to zero."""
        chime = ChimePlayer._generate_chime(24_000)
        assert abs(float(chime[-1])) < 0.01

    def test_generate_chime_not_silent(self):
        """The chime should have meaningful signal energy."""
        chime = ChimePlayer._generate_chime(24_000)
        assert np.abs(chime).max() > 0.1


# ---------------------------------------------------------------------------
# TestChimePlayerInit
# ---------------------------------------------------------------------------

class TestChimePlayerInit:
    def test_init_generates_chime_when_no_path(self, cfg):
        manager = AudioManager(cfg)
        player = ChimePlayer(cfg, manager)
        assert player._audio.dtype == np.float32
        assert len(player._audio) > 0

    def test_init_loads_wav_when_path_configured(self, tmp_path):
        wav_audio = np.ones(1000, dtype=np.float32)
        cfg_content = textwrap.dedent("""\
            audio:
              input_device: "TestInput"
              output_device: "TestOutput"
              input_sample_rate: 16000
              output_sample_rate: 48000
              input_channels: 1
              output_channels: 2
              chunk_size: 512
              buffer_seconds: 30
              chime_path: "/some/chime.wav"
            """)
        cfg_file = tmp_path / "cfg_chime.yaml"
        cfg_file.write_text(cfg_content)
        chime_cfg = Config(path=cfg_file)
        manager = AudioManager(chime_cfg)

        with patch.object(ChimePlayer, "_load_wav", return_value=(wav_audio, 44100)) as mock_load:
            player = ChimePlayer(chime_cfg, manager)

        mock_load.assert_called_once_with("/some/chime.wav")
        np.testing.assert_array_equal(player._audio, wav_audio)
        assert player._sample_rate == 44100

    def test_load_wav_raises_without_soundfile(self):
        with patch.dict("sys.modules", {"soundfile": None}):
            with pytest.raises(RuntimeError, match="soundfile"):
                ChimePlayer._load_wav("/path/to/chime.wav")


# ---------------------------------------------------------------------------
# TestChimePlayerPlay
# ---------------------------------------------------------------------------

class TestChimePlayerPlay:
    @pytest.mark.asyncio
    async def test_play_calls_manager_play_audio(self, cfg):
        manager = AudioManager(cfg)
        player = ChimePlayer(cfg, manager)

        with patch.object(manager, "play_audio", new_callable=AsyncMock) as mock_play:
            await player.play()

        mock_play.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_passes_chime_sample_rate(self, cfg):
        manager = AudioManager(cfg)
        player = ChimePlayer(cfg, manager)

        with patch.object(manager, "play_audio", new_callable=AsyncMock) as mock_play:
            await player.play()

        sample_rate_arg = mock_play.call_args[0][1]
        assert sample_rate_arg == ChimePlayer._CHIME_SAMPLE_RATE

    @pytest.mark.asyncio
    async def test_play_passes_audio_array(self, cfg):
        manager = AudioManager(cfg)
        player = ChimePlayer(cfg, manager)

        with patch.object(manager, "play_audio", new_callable=AsyncMock) as mock_play:
            await player.play()

        audio_arg = mock_play.call_args[0][0]
        np.testing.assert_array_equal(audio_arg, player._audio)


# ---------------------------------------------------------------------------
# TestProcessingChime
# ---------------------------------------------------------------------------

class TestProcessingChime:
    def test_generate_proc_chime_returns_float32(self):
        chime = ChimePlayer._generate_proc_chime(24_000)
        assert chime.dtype == np.float32

    def test_generate_proc_chime_is_1d(self):
        chime = ChimePlayer._generate_proc_chime(24_000)
        assert chime.ndim == 1

    def test_generate_proc_chime_correct_length(self):
        sr = 24_000
        chime = ChimePlayer._generate_proc_chime(sr)
        # Two notes (0.08 s + 0.12 s) + gap (0.022 s)
        expected = int(sr * 0.08) + int(sr * 0.022) + int(sr * 0.12)
        assert len(chime) == expected

    def test_generate_proc_chime_not_silent(self):
        chime = ChimePlayer._generate_proc_chime(24_000)
        assert np.abs(chime).max() > 0.05

    def test_init_creates_proc_audio(self, cfg):
        manager = AudioManager(cfg)
        player = ChimePlayer(cfg, manager)
        assert hasattr(player, "_proc_audio")
        assert player._proc_audio.dtype == np.float32
        assert len(player._proc_audio) > 0

    @pytest.mark.asyncio
    async def test_play_processing_calls_manager(self, cfg):
        manager = AudioManager(cfg)
        player = ChimePlayer(cfg, manager)

        with patch.object(manager, "play_audio", new_callable=AsyncMock) as mock_play:
            await player.play_processing()

        mock_play.assert_called_once()
        audio_arg = mock_play.call_args[0][0]
        np.testing.assert_array_equal(audio_arg, player._proc_audio)

    @pytest.mark.asyncio
    async def test_play_processing_swallows_device_error(self, cfg):
        """If play_audio raises (device busy), chime is skipped silently."""
        manager = AudioManager(cfg)
        player = ChimePlayer(cfg, manager)

        with patch.object(
            manager, "play_audio", new_callable=AsyncMock,
            side_effect=Exception("Device unavailable"),
        ):
            # Should not raise
            await player.play_processing()
