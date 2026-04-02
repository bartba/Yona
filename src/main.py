"""main.py — Orchestrator and CLI entry point.

Wires together all components (audio, VAD, wake word, STT, LLM, TTS,
pipeline) and drives the conversation state machine via EventBus events.

Usage::

    python -m src.main                  # run the app
    python -m src.main --list-devices   # show audio devices
    python -m src.main --config path    # custom config file
"""

from __future__ import annotations

import asyncio
import gc
import logging
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.audio import AudioBuffer, AudioManager, ChimePlayer
from src.config import Config
from src.events import Event, EventBus, EventType
from src.llm import ConversationContext, ConversationHistory, create_chat_handler
from src.pipeline import StreamingPipeline
from src.state import ConversationState as CS, StateMachine
from src.stt import Transcriber
from src.tts import create_synthesizer
from src.vad import VoiceActivityDetector
from src.wake import WakeWordDetector

logger = logging.getLogger(__name__)

# Goodbye intent patterns (Korean + English)
# Triggered by "bye bye Mack" / "goodbye Mack" style phrases.
# Korean + name variants:
#   굿/굳 (받침 ㅅ↔ㄷ 동음) + 바/빠 (된소리 변형) + 이 + 맥/막/맨 — 음절 사이 공백 허용
#   바이바이/빠이빠이 + 맥/막/맨
#   name: 맥(Mack), 멕(STT 오인식 변형)
# English + name variants: bye bye mack/mac/meg/man, goodbye mack/mac/meg/man
#   name: mack(정확), mac/meg/man(STT 오인식 변형 — meg: Mack→Meg 오인식)
# Legacy (no name): 바이바이, bye bye  — kept for backward compat
_GOODBYE_RE = re.compile(
    r"(?:[굳굿]\s*[바빠]이|바이\s*바이|빠이\s*빠이)\s*(?:맥|멕)"  # Korean + name
    r"|\b(?:bye[\s\-]?bye|good[\s\-]?bye)[\s,\.]+(?:mack|mac|meg|man)\b"  # English + name
    r"|바이바이"  # Legacy Korean (no name)
    r"|\bbye[\s\-]?bye\b",  # Legacy English (no name)
    re.IGNORECASE,
)


class YonaApp:
    """Main voice assistant orchestrator.

    Creates all components, registers the sounddevice audio callback,
    subscribes to EventBus events, and manages the conversation state
    machine including two-stage inactivity timeout and goodbye intent.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._bus = EventBus()
        self._sm = StateMachine(self._bus)
        self._running = False

        # Background tasks defined by "life cycle"  : _timeout_task 와 _process_task 는 수시로 개별 생성/취소
        self._timeout_task: asyncio.Task | None = None # 사용자가 말하면 취소하고 다시 시작(리셋), 대화별로 생성/취소
        self._process_task: asyncio.Task | None = None # 발화 하나당 한번 생성(STT->LLM->TTS), 발화별로 생성/취소
        self._handler_tasks: list[asyncio.Task] = []   # 앱이 실행되는 동안 살아있는 리스너들 (묶어서 처리)

        # Language of the last completed conversation turn.
        # Used for farewell TTS so the goodbye message matches the
        # conversation language rather than the farewell utterance language
        # (farewell words are almost always detected as English by STT).
        self._last_conversation_lang: str = "ko"

        # Timeout settings
        self._timeout_check_sec: float = cfg.get(
            "conversation.timeout_check_seconds", 15,
        )
        self._timeout_final_sec: float = cfg.get(
            "conversation.timeout_final_seconds", 5,
        )

        # Components — created in _init_components()
        self._audio: AudioManager | None = None
        self._buffer: AudioBuffer | None = None
        self._chime: ChimePlayer | None = None
        self._vad: VoiceActivityDetector | None = None
        self._wake: WakeWordDetector | None = None
        self._stt: Transcriber | None = None
        self._chat_handler: object | None = None  # ChatHandler
        self._synth: object | None = None  # Synthesizer
        self._pipeline: StreamingPipeline | None = None
        self._context: ConversationContext | None = None
        self._history: ConversationHistory | None = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def _init_components(self) -> None:
        """Create and wire all application components."""
        logger.info("Initializing components...")
        cfg = self._cfg
        bus = self._bus

        # Audio
        self._audio = AudioManager(cfg)
        self._buffer = AudioBuffer(
            sample_rate=cfg.get("audio.input_sample_rate", 16_000),
            buffer_seconds=cfg.get("audio.buffer_seconds", 30),
        )
        self._chime = ChimePlayer(cfg, self._audio)

        # Detection (ONNX file reads fill the Linux page cache)
        self._vad = VoiceActivityDetector(cfg, bus)
        self._wake = WakeWordDetector(cfg, bus)

        # Free page cache filled by ONNX model file reads above,
        # so NvMap can allocate contiguous GPU memory for STT on Jetson.
        self._drop_page_cache()

        # Recognition — CUDA model + warm-up needs contiguous GPU memory
        self._stt = Transcriber(cfg, bus)

        # LLM + TTS (ONNX file reads fill the page cache ~2 GB)
        self._chat_handler = create_chat_handler(cfg, bus)
        self._synth = create_synthesizer(cfg)

        # Drop page cache so CUDA pool expansion during inference can
        # reclaim memory.  Requires sudoers NOPASSWD for drop_caches.
        self._drop_page_cache()

        # Streaming pipeline
        self._pipeline = StreamingPipeline(
            self._chat_handler, self._synth, self._audio, bus,
        )

        # Conversation state
        self._context = ConversationContext(
            cfg.get_system_prompt(),
            max_context_tokens=cfg.get("llm.max_context_tokens", 3000),
        )
        self._history = ConversationHistory(
            storage_dir=cfg.get("history.storage_dir", "data/history"),
            max_days=cfg.get("history.max_days", 365),
        )
        self._history.purge_old()

        logger.info("All components initialized")

    @staticmethod
    def _drop_page_cache() -> None:
        """Release Linux page cache to free memory for CUDA allocations.

        On Jetson (unified memory) ONNX model file reads fill the page
        cache, and NvMap cannot reclaim cached pages for contiguous GPU
        allocations.  Dropping caches after TTS loads ensures enough
        free memory for STT inference pool expansion.

        Requires sudoers NOPASSWD; silently skips if unavailable.
        """
        gc.collect()
        try:
            subprocess.run(
                ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                capture_output=True, timeout=5,
            )
            logger.info("Page cache dropped for CUDA memory headroom")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.debug("Could not drop page caches (sudo NOPASSWD not configured)")

    # ------------------------------------------------------------------
    # Audio callback  (sounddevice thread — sync, must return quickly)
    # ------------------------------------------------------------------

    def _audio_callback(self, chunk: np.ndarray) -> None:
        """Dispatch incoming audio based on current conversation state."""
        state = self._sm.state

        if state == CS.IDLE:
            self._wake.process_chunk(chunk)

        elif state == CS.LISTENING:
            self._buffer.push(chunk)
            self._vad.process_chunk(chunk)

        elif state == CS.SPEAKING:
            self._vad.process_chunk(chunk)

        elif state == CS.TIMEOUT_CHECK:
            self._vad.process_chunk(chunk)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_wake_word(self, event: Event) -> None:
        """Wake word detected → chime → LISTENING."""
        if self._sm.state != CS.IDLE:
            return
        t0 = time.monotonic()
        logger.info("Wake word detected: %s", event.data)
        # Play chime while still IDLE so VAD/buffer don't capture chime audio
        await self._chime.play()
        logger.info("⏱ Wake→Chime done: %.2fs", time.monotonic() - t0)
        # Now transition — VAD and buffer start receiving audio from here
        await self._sm.transition(CS.LISTENING)
        self._vad.reset()
        self._vad.set_barge_in_mode(False)
        self._buffer.reset()
        self._restart_timeout()

    async def _on_speech_started(self, event: Event) -> None:
        """Speech detected — cancel timeout (restarted after processing) or return from TIMEOUT_CHECK."""
        state = self._sm.state
        if state == CS.TIMEOUT_CHECK:
            self._cancel_timeout()
            await self._audio.stop_playback()
            await self._sm.transition(CS.LISTENING)
            self._buffer.reset()
            # No restart here — timeout resumes after _process_utterance completes
        elif state == CS.LISTENING:
            self._cancel_timeout()

    async def _on_speech_ended(self, event: Event) -> None:
        """Speech ended → PROCESSING → chime → run STT + pipeline."""
        if self._sm.state != CS.LISTENING:
            return
        self._cancel_timeout()
        await self._sm.transition(CS.PROCESSING)
        await self._chime.play_processing()
        # Run processing in a separate task so event handlers stay responsive
        self._process_task = asyncio.create_task(self._process_utterance())

    async def _on_state_changed(self, event: Event) -> None:
        """Log every state transition with timestamp."""
        logger.info("▶ STATE: %s", event.data.name if hasattr(event.data, 'name') else event.data)

    async def _on_barge_in(self, event: Event) -> None:
        """Barge-in during SPEAKING → interrupt pipeline → LISTENING."""
        if self._sm.state != CS.SPEAKING:
            return
        logger.info("Barge-in detected — interrupting pipeline")
        await self._pipeline.interrupt()
        self._vad.set_barge_in_mode(False)
        self._vad.reset()
        # _process_utterance may have already transitioned to LISTENING
        if self._sm.state != CS.LISTENING:
            await self._sm.transition(CS.LISTENING)
        self._buffer.reset()
        self._restart_timeout()

    # ------------------------------------------------------------------
    # Utterance processing
    # ------------------------------------------------------------------

    async def _process_utterance(self) -> None:
        """STT → goodbye check → LLM+TTS pipeline → back to LISTENING."""
        """사용자가 말을 끝낸 순간(SPEECH_ENDED)부터 시작, assistant가 대답을 끝내고 다시 듣기 시작하는 순간까지의 모든것 처리"""
        t_start = time.monotonic()
        try:
            audio = self._buffer.get_all()
            self._buffer.reset()

            if len(audio) == 0:
                logger.warning("Empty audio buffer")
                await self._sm.transition(CS.LISTENING)
                self._restart_timeout()
                return

            # Transcribe
            t_stt = time.monotonic()
            text = await self._stt.transcribe(audio)
            stt_dur = time.monotonic() - t_stt
            lang = self._stt.detected_language or "ko"
            logger.info("⏱ STT: %.2fs | audio=%.2fs | lang=%s | text=%s",
                        stt_dur, len(audio) / 16000, lang, (text or "")[:80])

            if not text:
                logger.info("Empty transcription — notify and back to LISTENING")
                msgs = self._cfg.get("conversation.empty_transcription_message", {})
                msg = msgs.get(lang, msgs.get("ko", "잘 못 들었어요."))
                await self._play_tts(msg)
                await self._sm.transition(CS.LISTENING)
                self._restart_timeout()
                return

            logger.info("User [%s]: %s", lang, text)
            print(f"\nUser: {text}")

            # Goodbye intent — use last conversation language, not farewell's STT language
            if _GOODBYE_RE.search(text):
                await self._handle_goodbye(self._last_conversation_lang)
                return

            # Add user message to context
            self._context.add_user(text)

            # Transition to SPEAKING and run streaming pipeline
            await self._sm.transition(CS.SPEAKING)
            self._vad.set_barge_in_mode(True)
            self._vad.reset()

            t_pipeline = time.monotonic()
            response = await self._pipeline.run(
                self._context, detected_language=lang,
            )
            pipeline_dur = time.monotonic() - t_pipeline

            # Pipeline complete (or interrupted by barge-in)
            self._vad.set_barge_in_mode(False)

            if response:
                self._context.add_assistant(response)
                self._history.append_turn(text, response)
                self._last_conversation_lang = lang
                print(f"Assistant: {response}")
                logger.info("Assistant [%s]: %s", lang, response[:100])
            else:
                # Barge-in with no response — remove dangling user message
                # to prevent consecutive user turns in context
                self._context.pop_last_user()
                logger.info("Barge-in: removed unanswered user message from context")

            total_dur = time.monotonic() - t_start
            logger.info("⏱ TURN COMPLETE: total=%.2fs (STT=%.2fs + pipeline=%.2fs)",
                        total_dur, stt_dur, pipeline_dur)

            # Compress history if threshold reached.  Transition to
            # SPEAKING first so barge-in VAD is inactive and no new
            # speech is processed during the LLM summarisation call.
            if self._context.needs_compression:
                if self._sm.state != CS.SPEAKING:
                    await self._sm.transition(CS.SPEAKING)
                await self._compress_context(lang)

            # Back to LISTENING
            if self._sm.state == CS.SPEAKING:
                await self._sm.transition(CS.LISTENING)
                self._vad.reset()
                self._buffer.reset()
                self._restart_timeout()

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error processing utterance")
            # Notify user via TTS before returning to LISTENING
            lang = (self._stt.detected_language if self._stt else None) or "ko"
            msgs = self._cfg.get("conversation.error_message", {})
            msg = msgs.get(lang, msgs.get("ko", "죄송합니다, 오류가 발생했어요."))
            try:
                await self._play_tts(msg)
            except Exception:
                pass
            if self._sm.state in (CS.PROCESSING, CS.SPEAKING):
                try:
                    await self._sm.transition(CS.LISTENING)
                except Exception:
                    pass
                self._restart_timeout()

    async def _handle_goodbye(self, lang: str) -> None:
        """Play farewell message and return to IDLE."""
        logger.info("Goodbye intent detected")
        messages = self._cfg.get("conversation.goodbye_message", {})
        msg = messages.get(lang, messages.get("ko", "안녕히 계세요!"))

        # Play farewell via TTS
        await self._sm.transition(CS.SPEAKING)
        await self._play_tts(msg)

        # Clear context and return to IDLE
        self._context.clear()
        await self._sm.transition(CS.IDLE)
        self._vad.reset()
        self._wake.reset()
        gc.collect()

    # ------------------------------------------------------------------
    # Two-stage inactivity timeout
    # ------------------------------------------------------------------

    def _restart_timeout(self) -> None:
        """Cancel existing timeout and start a new countdown."""
        self._cancel_timeout()
        self._timeout_task = asyncio.create_task(self._timeout_sequence())

    def _cancel_timeout(self) -> None:
        """Cancel the running timeout task, if any."""
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            self._timeout_task = None

    async def _timeout_sequence(self) -> None:
        """Two-stage timeout: TIMEOUT_CHECK → TIMEOUT_FINAL → IDLE."""
        try:
            # Stage 1: silence for timeout_check_seconds
            await asyncio.sleep(self._timeout_check_sec)
            if self._sm.state != CS.LISTENING:
                return

            await self._sm.transition(CS.TIMEOUT_CHECK)
            self._vad.set_barge_in_mode(False)
            self._vad.reset()

            msgs = self._cfg.get("conversation.timeout_check_message", {})
            lang = (self._stt.detected_language if self._stt else None) or "ko"
            msg = msgs.get(lang, msgs.get("ko", "아직 계세요?"))
            await self._play_tts(msg)

            # Stage 2: silence for timeout_final_seconds more
            await asyncio.sleep(self._timeout_final_sec)
            if self._sm.state != CS.TIMEOUT_CHECK:
                return

            await self._sm.transition(CS.TIMEOUT_FINAL)
            msgs = self._cfg.get("conversation.timeout_final_message", {})
            msg = msgs.get(lang, msgs.get("ko", "대화를 종료합니다."))
            await self._play_tts(msg)

            # Clear context and return to IDLE
            self._context.clear()
            await self._sm.transition(CS.IDLE)
            self._vad.reset()
            self._wake.reset()
            gc.collect()

        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Context compression
    # ------------------------------------------------------------------

    _COMPRESS_PROMPT = (
        "Summarize the following conversation concisely in the same language "
        "as the conversation. Preserve key facts, user preferences, topics "
        "discussed, and any commitments made. Keep it under 200 words."
    )

    _COMPRESS_NOTIFY_START: dict[str, str] = {
        "ko": "이전 대화 내용을 정리하고 있습니다. 잠시만 기다려 주세요.",
        "en": "Organizing our conversation. One moment please.",
    }
    _COMPRESS_NOTIFY_DONE: dict[str, str] = {
        "ko": "정리를 마쳤습니다. 이제 대화를 이어 나갈 수 있습니다.",
        "en": "All done. We can continue our conversation.",
    }

    async def _compress_context(self, lang: str = "ko") -> None:
        """Notify the user, then summarise older history via the LLM.

        Runs while state is still SPEAKING so no new speech is processed.
        If the LLM call fails, the full history is kept as-is.
        """
        if not self._chat_handler or not self._context:
            return
        old_summary, old_msgs = self._context.get_compression_payload()
        if not old_msgs:
            return

        # Notify user — plays while still in SPEAKING state
        msg = self._COMPRESS_NOTIFY_START.get(lang, self._COMPRESS_NOTIFY_START["ko"])
        await self._play_tts(msg)
        logger.info("Compressing context: %d messages to summarise", len(old_msgs))

        # Build conversation text for summarisation
        parts: list[str] = []
        if old_summary:
            parts.append(f"[이전 요약]\n{old_summary}\n")
        for m in old_msgs:
            role = "User" if m["role"] == "user" else "Assistant"
            parts.append(f"{role}: {m['content']}")
        conversation_text = "\n".join(parts)

        # Use a throwaway context for the summarisation call
        summary_ctx = ConversationContext(
            system_prompt=self._COMPRESS_PROMPT,
            max_context_tokens=10000,  # throwaway ctx; never needs to compress
        )
        summary_ctx.add_user(conversation_text)

        try:
            tokens: list[str] = []
            async for token in self._chat_handler.stream(summary_ctx):
                tokens.append(token)
            summary = "".join(tokens)
            if summary:
                self._context.compress(summary)
                logger.info("Context compressed: %d turns → summary (%d chars) + %d messages",
                            len(old_msgs) // 2, len(summary), self._context.message_count)
        except Exception:
            logger.warning("Context compression failed — keeping full history")

        # Notify user that compression is done
        msg = self._COMPRESS_NOTIFY_DONE.get(lang, self._COMPRESS_NOTIFY_DONE["ko"])
        await self._play_tts(msg)

    # ------------------------------------------------------------------
    # TTS helper
    # ------------------------------------------------------------------

    async def _play_tts(self, text: str) -> None:
        """Synthesize and play a message via TTS + AudioManager."""
        if not self._synth or not self._audio:
            return
        try:
            audio, sr = await self._synth.synthesize(text)
            await self._audio.play_audio(audio, sr)
        except Exception:
            logger.exception("TTS playback error for: %s", text[:50])

    # ------------------------------------------------------------------
    # Event listener helper
    # ------------------------------------------------------------------

    async def _listen(
        self,
        event_type: EventType,
        handler,
    ) -> None:
        """Subscribe to *event_type* and dispatch events to *handler*."""
        q = self._bus.subscribe(event_type)
        try:
            while True:
                event = await q.get()
                try:
                    await handler(event)
                except Exception:
                    logger.exception("Error in %s handler", event_type.name)
        except asyncio.CancelledError:
            pass
        finally:
            self._bus.unsubscribe(event_type, q)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start all components and run the main event loop."""
        self._bus.set_loop(asyncio.get_running_loop())
        await self._init_components()

        # Register audio callback and start the mic stream
        self._audio.add_input_callback(self._audio_callback)
        await self._audio.start()

        self._running = True
        logger.info("Voice assistant is now ready! Say the wake word to start.")

        # Start event handler tasks
        self._handler_tasks = [
            asyncio.create_task(
                self._listen(EventType.WAKE_WORD_DETECTED, self._on_wake_word),
            ),
            asyncio.create_task(
                self._listen(EventType.SPEECH_STARTED, self._on_speech_started),
            ),
            asyncio.create_task(
                self._listen(EventType.SPEECH_ENDED, self._on_speech_ended),
            ),
            asyncio.create_task(
                self._listen(EventType.BARGE_IN_DETECTED, self._on_barge_in),
            ),
            asyncio.create_task(
                self._listen(EventType.STATE_CHANGED, self._on_state_changed),
            ),
        ]

        try:
            # Block until SHUTDOWN event is published
            shutdown_q = self._bus.subscribe(EventType.SHUTDOWN)
            await shutdown_q.get()
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully stop all components and cancel background tasks."""
        if not self._running:
            return
        self._running = False
        logger.info("Shutting down voice assistant...")

        # Cancel timeout
        self._cancel_timeout()

        # Cancel processing task
        if self._process_task and not self._process_task.done():
            self._process_task.cancel()

        # Cancel event handler tasks
        for task in self._handler_tasks:
            task.cancel()
        if self._handler_tasks:
            await asyncio.gather(*self._handler_tasks, return_exceptions=True)
            self._handler_tasks.clear()

        # Stop pipeline
        if self._pipeline:
            await self._pipeline.interrupt()

        # Stop audio
        if self._audio:
            await self._audio.stop_playback()
            await self._audio.stop()

        # Close network clients
        if self._chat_handler and hasattr(self._chat_handler, "close"):
            await self._chat_handler.close()
        if self._synth and hasattr(self._synth, "close"):
            await self._synth.close()

        logger.info("Voice assistant stopped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _list_devices() -> None:
    """Print available audio devices and exit."""
    import sounddevice as sd
    print(sd.query_devices())


def main() -> None:
    """CLI entry point for ``python -m src.main``."""
    import argparse

    parser = argparse.ArgumentParser(description="Voice Assistant")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        _list_devices()
        return

    # Logging — honour LOG_LEVEL env var (default: INFO)
    import os
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg_path = Path(args.config) if args.config else None
    cfg = Config(cfg_path)
    app = YonaApp(cfg)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler() -> None:
        logger.info("Received shutdown signal")
        app._bus.publish_nowait(EventType.SHUTDOWN)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(app.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
