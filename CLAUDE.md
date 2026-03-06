# Yona v2 — Voice Chat Application

## Project Overview
A streaming voice chat assistant for Nvidia Jetson Orin Nano with Polycom Sync 20 Plus speakerphone.
Detects a wake word, enters conversation mode, streams LLM responses phrase-by-phrase to TTS,
and supports barge-in (interrupt during playback). Returns to idle after a two-stage timeout.

## Build Plan
See **`plan.md`** in the project root for the full step-by-step build plan and progress tracker.
Start each new session with: *"plan.md의 Step N을 진행합시다"*

## Tech Stack
- **Wake Word**: Porcupine (Picovoice) — "Hi Inspector" / "하이 검사기"
- **STT**: faster-whisper `large-v3-turbo`, CUDA float16
- **LLM**: OpenAI GPT **or** Custom company LLM (selectable via `LLM_PROVIDER` env var)
- **TTS**: XTTS v2 (Coqui), local GPU, 24 kHz
- **VAD**: Silero VAD (ONNX), used for both speech detection and barge-in monitoring
- **Audio**: sounddevice + numpy — 16 kHz input / 48 kHz output

## Project Structure (v2 — flat)
```
src/
├── config.py      # YAML + .env loader                          ✅ done
├── events.py      # EventType enum + async EventBus
├── state.py       # ConversationState + StateMachine (6 states)
├── audio.py       # AudioManager + AudioBuffer + ChimePlayer
├── vad.py         # VoiceActivityDetector (Silero ONNX)
├── wake.py        # WakeWordDetector (Porcupine)
├── stt.py         # Transcriber (faster-whisper)
├── llm.py         # ChatHandler Protocol + Context + History + providers + factory
├── tts.py         # Synthesizer (XTTS v2)
├── pipeline.py    # PhraseAccumulator + StreamingPipeline (producer-consumer)
└── main.py        # YonaApp orchestrator + CLI
```

## Key Commands
```bash
# Run all tests
pytest tests/ -v

# Run a single step's tests
pytest tests/test_config.py -v

# Run application
python -m src.main

# List audio devices
python -m src.main --list-devices
```

## Configuration
- Main config: `config/default.yaml`
- System prompt: `config/prompts/system_prompt.txt`
- Secrets: `.env` — see `.env.example` for required variables

## Environment Variables
```bash
LLM_PROVIDER=openai          # "openai" or "custom"
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
CUSTOM_LLM_URL=http://...
CUSTOM_LLM_KEY=...
CUSTOM_LLM_MODEL=...
PORCUPINE_ACCESS_KEY=...
```

## Hardware (Confirmed)
- Nvidia Jetson Orin Nano (CUDA for STT + TTS)
- Polycom Sync 20 Plus — USB direct connection (`Poly Sync 20`, hw:0,0)
  - `input_channels: 1` (mono, HW beamforming + AEC inside device)
  - `output_channels: 2` (stereo USB, verified)
  - `chunk_size: 512` = 32 ms @ 16 kHz = Porcupine `frame_length`
- PortAudio: `sudo apt install libportaudio2`

## State Machine
```
IDLE ──(wake word + chime)──→ LISTENING
LISTENING ──(speech + silence)──→ PROCESSING
PROCESSING ──(STT done)──→ SPEAKING   ← streaming pipeline starts here
SPEAKING ──(playback done)──→ LISTENING
SPEAKING ──(barge-in VAD)──→ LISTENING
LISTENING/SPEAKING ──(15 s)──→ TIMEOUT_CHECK
TIMEOUT_CHECK ──(speech)──→ LISTENING
TIMEOUT_CHECK ──(15 s more)──→ TIMEOUT_FINAL → IDLE
any state ──(goodbye intent)──→ IDLE
```

## Streaming Pipeline
```
LLM tokens → PhraseAccumulator → Queue[str] → TTS Worker → Queue[audio] → Speaker
                                                                ↑
                                              barge-in VAD monitors mic in parallel
```

## Code Conventions
- Python 3.10+ with type hints
- `typing.Protocol` for interfaces (no ABCs)
- `asyncio` throughout; sounddevice callback is the only sync entry point
- Event-driven via `EventBus` (pub/sub, `asyncio.Queue`-based)
- Config access: `cfg.get("section.key", default)` or `cfg.section["key"]`
- Tests: pytest + pytest-asyncio (`mode=strict`), all mocked — no hardware needed
