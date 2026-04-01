"""vad.py — Voice Activity Detection for Samsung Gauss.

Wraps the Silero VAD v6 ONNX model with speech/silence state tracking and
EventBus integration.

VoiceActivityDetector
    Runs the Silero model on 512-sample (32 ms @ 16 kHz) float32 chunks
    with a 64-sample context window (total 576 samples per inference),
    maintains per-session LSTM state, and publishes EventBus events:

    Normal mode (used during LISTENING):
        SPEECH_STARTED  — first chunk above threshold after silence
        SPEECH_ENDED    — silence >= silence_duration s after >= min_speech_duration s of speech

    Barge-in mode (used during SPEAKING):
        BARGE_IN_DETECTED — first chunk above barge_in_threshold

Usage::

    from src.config import Config
    from src.events import EventBus
    from src.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(cfg, bus)

    # Register as an AudioManager input callback:
    audio_manager.add_input_callback(vad.process_chunk)

    # Switch to barge-in mode when TTS is speaking:
    vad.set_barge_in_mode(True)

    # Back to normal after interruption:
    vad.set_barge_in_mode(False)
    vad.reset()
"""

from __future__ import annotations

import time

import numpy as np
import onnxruntime as ort

from src.config import Config
from src.events import EventBus, EventType


class VoiceActivityDetector:
    """Silero VAD ONNX wrapper with speech/silence event publishing.

    Maintains a single LSTM state tensor across chunks so the model's
    recurrent memory persists throughout a conversation turn.  Call
    :method:`reset` when starting a new turn.

    This class is designed to be registered as an ``AudioManager`` input
    callback.  :method:`process_chunk` is sync and uses ``publish_nowait``
    so it never blocks the sounddevice callback thread.

    Args:
        cfg: Application config — reads ``vad.*`` and ``audio.*`` sections.
        bus: Shared :class:`EventBus` for publishing detection events.
    """

    def __init__(self, cfg: Config, bus: EventBus) -> None:
        self._bus = bus

        model_path: str = cfg.get("vad.model_path", "models/silero_vad/silero_vad.onnx")
        self._session = ort.InferenceSession(model_path)

        self._threshold: float = cfg.get("vad.threshold", 0.5)
        self._barge_in_threshold: float = cfg.get("vad.barge_in_threshold", 0.7)
        self._silence_duration: float = cfg.get("vad.silence_duration", 0.8)
        self._min_speech_duration: float = cfg.get("vad.min_speech_duration", 0.3)
        self._sample_rate: int = cfg.get("audio.input_sample_rate", 16_000)
        self._chunk_size: int = cfg.get("audio.chunk_size", 512)

        # Silero VAD LSTM state — single [2, 1, 128] tensor
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

        # Silero v5/v6 context window — last 64 samples from previous chunk
        self._context_size: int = 64 if self._sample_rate == 16_000 else 32
        self._context = np.zeros(self._context_size, dtype=np.float32)

        # Speech-detection state
        self._barge_in_mode: bool = False
        self._speech_active: bool = False
        self._speech_start: float = 0.0   # monotonic time when speech began
        self._last_speech: float = 0.0    # monotonic time of last voiced chunk

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------

    def set_barge_in_mode(self, enabled: bool) -> None:
        """Switch between LISTENING (normal) and SPEAKING (barge-in) mode.

        In barge-in mode a higher threshold is used and
        :data:`EventType.BARGE_IN_DETECTED` is published instead of
        :data:`EventType.SPEECH_STARTED`.
        """
        self._barge_in_mode = enabled

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset LSTM and detection state for a new utterance.

        Call when transitioning back to IDLE or at the start of LISTENING
        so stale recurrent state does not influence the next turn.
        """
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self._context_size, dtype=np.float32)
        self._speech_active = False
        self._speech_start = 0.0
        self._last_speech = 0.0

    # ------------------------------------------------------------------
    # Core processing  (called from sounddevice callback — must be fast)
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> float:
        """Run Silero VAD on a single audio chunk.

        Feeds the chunk through the ONNX model, updates the LSTM state,
        then calls :method:`_update_state` to emit any pending events.

        Chunks shorter than ``chunk_size`` are zero-padded; longer ones
        are truncated to the first ``chunk_size`` samples.

        Args:
            chunk: Mono float32 array (any length — will be normalised).

        Returns:
            Speech probability in [0.0, 1.0].
        """
        audio = np.asarray(chunk, dtype=np.float32).ravel()
        if len(audio) < self._chunk_size:
            audio = np.pad(audio, (0, self._chunk_size - len(audio)))
        else:
            audio = audio[: self._chunk_size]

        # Silero v5/v6: prepend context (last 64 samples of previous chunk)
        x = np.concatenate([self._context, audio]).reshape(1, -1)
        self._context = audio[-self._context_size:].copy()

        sr = np.array(self._sample_rate, dtype=np.int64)

        out = self._session.run(
            None,
            {
                "input": x,
                "state": self._state,
                "sr": sr,
            },
        )

        speech_prob = float(out[0].squeeze())
        self._state = out[1]

        self._update_state(speech_prob)
        return speech_prob

    # ------------------------------------------------------------------
    # Internal state machine
    # ------------------------------------------------------------------

    def _update_state(self, speech_prob: float) -> None:
        """Update detection state and publish events on transitions.

        Called once per :method:`process_chunk` invocation.
        """
        threshold = self._barge_in_threshold if self._barge_in_mode else self._threshold
        silence_dur = self._silence_duration

        now = time.monotonic()

        if speech_prob >= threshold:
            self._last_speech = now
            if not self._speech_active:
                self._speech_active = True
                self._speech_start = now
                if self._barge_in_mode:
                    self._bus.publish_nowait(EventType.BARGE_IN_DETECTED)
                else:
                    self._bus.publish_nowait(EventType.SPEECH_STARTED)
        else:
            if self._speech_active:
                silence_elapsed = now - self._last_speech
                if silence_elapsed >= silence_dur:
                    speech_duration = self._last_speech - self._speech_start
                    self._speech_active = False
                    if not self._barge_in_mode and speech_duration >= self._min_speech_duration: 
                        # SPEAKING 중 사용자 발화->barge-in 모드->TTS 중단->LISTENING 전환->Barge-in 모드 해제
                        # Barge-in 모드 에서는 SPEECH_ENDED 이벤트(STT 트리거)를 발생시킬 이유 없음.
                        self._bus.publish_nowait(EventType.SPEECH_ENDED)
