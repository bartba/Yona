"""Yona Voice Chat Application - Main Entry Point."""
import asyncio
import signal
import sys
from pathlib import Path

import numpy as np

from src.core.config import Config
from src.core.event_bus import Event, EventBus, EventType
from src.core.state_machine import ConversationState, ConversationStateMachine
from src.audio.audio_manager import AudioManager
from src.audio.audio_buffer import AudioBuffer
from src.audio.vad import VoiceActivityDetector
from src.wake_word.detector import WakeWordDetector
from src.stt.transcriber import Transcriber
from src.llm import ChatHandler, create_chat_handler
from src.tts.synthesizer import Synthesizer
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class YonaApp:
    """Main Yona voice chat application."""

    def __init__(self, config_path: str | Path | None = None):
        """Initialize the application.

        Args:
            config_path: Path to configuration file
        """
        self._config = Config(config_path)
        self._setup_logging()

        # Core components
        self._event_bus = EventBus()
        self._state_machine = ConversationStateMachine(
            timeout_seconds=self._config.conversation.get("timeout_seconds", 60)
        )

        # Audio components
        self._audio_manager = AudioManager(
            input_device=self._config.audio.get("input_device"),
            output_device=self._config.audio.get("output_device"),
            input_sample_rate=self._config.audio.get("input_sample_rate", 16000),
            output_sample_rate=self._config.audio.get("output_sample_rate", 48000),
        )
        self._audio_buffer = AudioBuffer(
            max_seconds=self._config.audio.get("buffer_seconds", 30),
            sample_rate=self._config.audio.get("input_sample_rate", 16000),
        )
        self._vad = VoiceActivityDetector(
            sample_rate=self._config.audio.get("input_sample_rate", 16000),
            energy_threshold=self._config.vad.get("energy_threshold", 0.01),
            silence_duration=self._config.vad.get("silence_duration", 1.5),
            min_speech_duration=self._config.vad.get("min_speech_duration", 0.3),
        )

        # Processing components
        self._wake_word_detector: WakeWordDetector | None = None
        self._transcriber: Transcriber | None = None
        self._chat_handler: ChatHandler | None = None
        self._synthesizer: Synthesizer | None = None

        # State
        self._running = False
        self._current_language = "ko"

    def _setup_logging(self) -> None:
        """Configure logging from config."""
        log_config = self._config.logging
        setup_logging(
            level=log_config.get("level", "INFO"),
            log_file=log_config.get("file"),
            max_bytes=log_config.get("max_bytes", 10485760),
            backup_count=log_config.get("backup_count", 5),
        )

    async def _init_components(self) -> None:
        """Initialize all components."""
        logger.info("Initializing components...")

        # Wake word detector
        try:
            self._wake_word_detector = WakeWordDetector(
                model_path=self._config.wake_word.get("model_path"),
                threshold=self._config.wake_word.get("threshold", 0.5),
                cooldown_seconds=self._config.wake_word.get("cooldown_seconds", 2.0),
            )
        except Exception as e:
            logger.error(f"Failed to initialize wake word detector: {e}")

        # Speech-to-text
        try:
            self._transcriber = Transcriber(
                model_size=self._config.stt.get("model_size", "small"),
                device=self._config.stt.get("device", "cuda"),
                compute_type=self._config.stt.get("compute_type", "float16"),
                language=self._config.stt.get("language"),
            )
        except Exception as e:
            logger.error(f"Failed to initialize transcriber: {e}")

        # Chat handler
        try:
            self._chat_handler = create_chat_handler(
                model=self._config.llm.get("model"),
                max_tokens=self._config.llm.get("max_tokens", 1024),
                temperature=self._config.llm.get("temperature", 0.7),
                system_prompt=self._config.get_system_prompt(),
            )
        except Exception as e:
            logger.error(f"Failed to initialize chat handler: {e}")

        # TTS synthesizer
        try:
            self._synthesizer = Synthesizer(
                voices=self._config.tts.get("voices"),
                default_voice=self._config.tts.get("default_voice", "ko-KR-SunHiNeural"),
                match_input_language=self._config.tts.get("match_input_language", True),
            )
        except Exception as e:
            logger.error(f"Failed to initialize synthesizer: {e}")

        logger.info("Components initialized")

    def _setup_event_handlers(self) -> None:
        """Set up event handlers."""
        self._event_bus.subscribe(EventType.WAKE_WORD_DETECTED, self._on_wake_word)
        self._event_bus.subscribe(EventType.SPEECH_END, self._on_speech_end)
        self._event_bus.subscribe(EventType.TRANSCRIPTION_COMPLETE, self._on_transcription)
        self._event_bus.subscribe(EventType.LLM_RESPONSE_READY, self._on_llm_response)
        self._event_bus.subscribe(EventType.PLAYBACK_DONE, self._on_playback_done)

        # State change handlers
        self._state_machine.on_state_change(self._on_state_change)

    async def _on_wake_word(self, event: Event) -> None:
        """Handle wake word detection."""
        if self._state_machine.state == ConversationState.IDLE:
            logger.info("Wake word detected - starting conversation")
            await self._state_machine.transition_to(ConversationState.LISTENING)
            self._audio_buffer.clear()
            self._vad.reset()

    async def _on_speech_end(self, event: Event) -> None:
        """Handle end of speech."""
        if self._state_machine.state == ConversationState.LISTENING:
            await self._state_machine.transition_to(ConversationState.PROCESSING)

            # Get recorded audio
            audio = self._audio_buffer.read_all()

            if len(audio) > 0:
                # Transcribe
                asyncio.create_task(self._process_speech(audio))
            else:
                logger.warning("No audio captured")
                await self._state_machine.transition_to(ConversationState.LISTENING)

    async def _process_speech(self, audio: np.ndarray) -> None:
        """Process recorded speech."""
        if not self._transcriber:
            logger.error("Transcriber not available")
            await self._state_machine.transition_to(ConversationState.LISTENING)
            return

        try:
            # Transcribe
            result = self._transcriber.transcribe(audio.flatten())

            if result.text.strip():
                self._current_language = result.language
                await self._event_bus.publish(Event(
                    type=EventType.TRANSCRIPTION_COMPLETE,
                    data={"text": result.text, "language": result.language},
                ))
            else:
                logger.info("Empty transcription, returning to listening")
                await self._state_machine.transition_to(ConversationState.LISTENING)

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            await self._state_machine.transition_to(ConversationState.LISTENING)

    async def _on_transcription(self, event: Event) -> None:
        """Handle completed transcription."""
        text = event.data.get("text", "")
        language = event.data.get("language", "ko")

        logger.info(f"User said: {text}")

        if not self._chat_handler:
            logger.error("Chat handler not available")
            await self._state_machine.transition_to(ConversationState.LISTENING)
            return

        try:
            # Get LLM response
            response = await self._chat_handler.chat(text, language)

            await self._event_bus.publish(Event(
                type=EventType.LLM_RESPONSE_READY,
                data={"response": response, "language": language},
            ))

        except Exception as e:
            logger.error(f"Chat error: {e}")
            await self._state_machine.transition_to(ConversationState.LISTENING)

    async def _on_llm_response(self, event: Event) -> None:
        """Handle LLM response ready."""
        response = event.data.get("response", "")
        language = event.data.get("language", self._current_language)

        if not response:
            await self._state_machine.transition_to(ConversationState.LISTENING)
            return

        await self._state_machine.transition_to(ConversationState.SPEAKING)

        logger.info(f"Assistant: {response}")

        if not self._synthesizer:
            logger.error("Synthesizer not available")
            await self._state_machine.transition_to(ConversationState.LISTENING)
            return

        try:
            # Synthesize and play
            audio, sample_rate = await self._synthesizer.synthesize(response, language)
            await self._audio_manager.play_audio_async(audio, sample_rate)

            await self._event_bus.publish(Event(type=EventType.PLAYBACK_DONE))

        except Exception as e:
            logger.error(f"TTS error: {e}")
            await self._state_machine.transition_to(ConversationState.LISTENING)

    async def _on_playback_done(self, event: Event) -> None:
        """Handle playback completion."""
        if self._state_machine.state == ConversationState.SPEAKING:
            await self._state_machine.transition_to(ConversationState.LISTENING)
            self._audio_buffer.clear()
            self._vad.reset()
            self._state_machine.update_activity()

    async def _on_state_change(
        self,
        old_state: ConversationState,
        new_state: ConversationState,
    ) -> None:
        """Handle state changes."""
        if new_state == ConversationState.TIMEOUT_WARNING:
            await self._play_timeout_message()

    async def _play_timeout_message(self) -> None:
        """Play timeout warning and return to idle."""
        timeout_messages = self._config.conversation.get("timeout_message", {})
        message = timeout_messages.get(
            self._current_language,
            timeout_messages.get("ko", "대화를 원하시면 다시 불러주세요.")
        )

        logger.info(f"Playing timeout message: {message}")

        if self._synthesizer:
            try:
                audio, sample_rate = await self._synthesizer.synthesize(
                    message, self._current_language
                )
                await self._audio_manager.play_audio_async(audio, sample_rate)
            except Exception as e:
                logger.error(f"Timeout message TTS error: {e}")

        # Clear context and return to idle
        if self._chat_handler:
            self._chat_handler.clear_context()

        await self._state_machine.reset()

    def _audio_callback(self, audio: np.ndarray) -> None:
        """Handle incoming audio from microphone."""
        state = self._state_machine.state

        if state == ConversationState.IDLE:
            # Check for wake word
            if self._wake_word_detector:
                detected = self._wake_word_detector.process(audio)
                if detected:
                    self._event_bus.publish_sync(Event(type=EventType.WAKE_WORD_DETECTED))

        elif state == ConversationState.LISTENING:
            # Buffer audio and check VAD
            self._audio_buffer.write(audio)
            _, speech_started, speech_ended = self._vad.process(audio.flatten())

            if speech_ended:
                self._event_bus.publish_sync(Event(type=EventType.SPEECH_END))

            # Update activity on speech
            if speech_started:
                self._state_machine.update_activity()

    async def run(self) -> None:
        """Run the application."""
        logger.info("Starting Yona Voice Chat...")

        # Initialize
        await self._init_components()
        self._setup_event_handlers()

        # Start components
        await self._event_bus.start()
        await self._state_machine.start()

        # Set up audio callback
        self._audio_manager.set_audio_callback(self._audio_callback)
        self._audio_manager.start_capture()

        self._running = True

        logger.info("Yona is ready! Say the wake word to start.")

        # Main loop
        try:
            while self._running:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

        await self.shutdown()

    async def shutdown(self) -> None:
        """Shut down the application."""
        logger.info("Shutting down Yona...")

        self._running = False

        # Stop components
        self._audio_manager.stop_capture()
        self._audio_manager.stop_playback()

        await self._state_machine.stop()
        await self._event_bus.stop()

        logger.info("Yona stopped")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Yona Voice Chat Application")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        AudioManager.list_devices()
        return

    # Create and run application
    app = YonaApp(config_path=args.config)

    # Handle signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler():
        logger.info("Received shutdown signal")
        app._running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        loop.run_until_complete(app.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
