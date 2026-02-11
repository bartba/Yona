# Yona Voice Chat Application

## Project Overview
A voice chat application for Nvidia Jetson Orin Nano with Polycom Sync 20 Plus speakerphone. The app waits for a wake word, enters conversation mode, and returns to idle after 60 seconds of silence.

## Tech Stack
- **Wake Word**: openWakeWord (custom ONNX model for "하이 삼성" / "Hi Samsung")
- **STT**: faster-whisper (medium model, CUDA acceleration)
- **LLM**: Ollama (llama3.1:8b, remote LAN GPU server) or OpenAI (gpt-5-nano)
- **TTS**: Edge TTS (Korean: ko-KR-SunHiNeural, English: en-US-JennyNeural)
- **VAD**: Silero VAD (ONNX, CPU inference)
- **Audio**: sounddevice + numpy (16kHz input, 48kHz output)

## Project Structure
```
src/
├── main.py              # Application entry point
├── core/
│   ├── state_machine.py # 5 states: IDLE→LISTENING→PROCESSING→SPEAKING→TIMEOUT_WARNING
│   ├── event_bus.py     # Async event-driven communication
│   └── config.py        # YAML config loader
├── audio/
│   ├── audio_manager.py # Polycom device I/O
│   ├── vad.py           # Silero VAD voice activity detection
│   └── audio_buffer.py  # Ring buffer for audio
├── wake_word/
│   ├── detector.py      # openWakeWord integration
│   └── trainer.py       # Wake word training utility
├── stt/
│   └── transcriber.py   # faster-whisper STT
├── llm/
│   ├── base.py           # ChatHandler Protocol (interface)
│   ├── ollama_handler.py # Local Ollama handler
│   ├── openai_handler.py # OpenAI API handler
│   └── context.py        # Conversation history (max 20 turns)
└── tts/
    └── synthesizer.py   # Edge TTS wrapper
```

## Key Commands
```bash
# Run tests
pytest tests/ -v

# Run application
python -m src.main

# List audio devices
python -m src.main --list-devices

# Train wake word (record samples)
python -m src.wake_word.trainer --record-positive --wake-word hi_samsung
```

## Configuration
- Main config: `config/default.yaml`
- System prompt: `config/prompts/system_prompt.txt`
- Environment: `.env` (LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_URL, OPENAI_API_KEY, OPENAI_MODEL)

## State Machine Flow
```
IDLE (wake word) → LISTENING (VAD) → PROCESSING (STT+LLM) → SPEAKING (TTS) → LISTENING
                                                                          ↓ (60s timeout)
                                                              TIMEOUT_WARNING → IDLE
```

## Language Behavior
- Auto-detects Korean/English from speech
- Responds in the same language as user input
- TTS voice matches detected language

## Code Conventions
- Python 3.10+ with type hints
- Async/await for I/O operations
- Event-driven architecture via EventBus
- All components are modular and testable

## Hardware Requirements
- Nvidia Jetson Orin Nano (CUDA for STT)
- Polycom Sync 20 Plus / BT600 USB dongle
- PortAudio library (`sudo apt install libportaudio2`)

## Current Status
- Core implementation complete
- 45 unit tests passing
- Needs: wake word model training, hardware integration testing
