# Yona — Voice Chat Application

## Overview
Streaming voice assistant for Nvidia Jetson Orin Nano + Polycom Sync 20 Plus.
Wake word → LISTENING → STT → LLM streaming → TTS phrase-by-phrase → speaker.
Supports barge-in and two-stage timeout.

## Tech Stack
- **Wake Word**: openWakeWord ONNX (offline) — custom model `Hey_Mack_20260309_205536.onnx`
- **STT**: faster-whisper `large-v3-turbo`, CUDA float16
- **LLM**: OpenAI / Claude / Custom — `LLM_PROVIDER` env var
- **TTS**: Supertonic ONNX, CPU, 44.1 kHz
- **VAD**: Silero VAD v6 ONNX, 512-sample chunks + 64-sample context
- **Audio**: sounddevice + numpy — 16 kHz input / 48 kHz output

## Project Structure
```
src/
├── config.py      # YAML + .env loader
├── events.py      # EventType enum + async EventBus
├── state.py       # ConversationState + StateMachine (6 states)
├── audio.py       # AudioManager + AudioBuffer + ChimePlayer
├── vad.py         # VoiceActivityDetector
├── wake.py        # WakeWordDetector
├── stt.py         # Transcriber
├── llm.py         # ChatHandler Protocol + providers + factory
├── tts.py         # Synthesizer Protocol + SupertonicSynthesizer + factory
├── pipeline.py    # PhraseAccumulator + StreamingPipeline
├── main.py        # YonaApp orchestrator + CLI
└── web.py         # Web dashboard (FastAPI)
```

## Key Commands
```bash
python -m src.main              # Run application
python -m src.main --list-devices
```

## Configuration
- `config/default.yaml` — main config
- `config/prompts/system_prompt.txt` — system prompt
- `.env` — secrets (API keys)

## Environment Variables
```bash
LLM_PROVIDER=openai          # "openai" | "claude" | "custom"
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
CLAUDE_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-6
CUSTOM_LLM_URL=http://...
CUSTOM_LLM_KEY=...
CUSTOM_LLM_MODEL=...
```

## Hardware
- Jetson Orin Nano — CUDA for STT, CPU for TTS + wake word
- Poly Sync 20 Plus — USB direct (`hw:0,0`), `input_channels: 1`, `output_channels: 2`
- `sudo apt install libportaudio2`

## State Machine
```
IDLE ──(wake word)──→ LISTENING ──(speech+silence)──→ PROCESSING ──(STT)──→ SPEAKING
SPEAKING ──(done / barge-in)──→ LISTENING
LISTENING/SPEAKING ──(15s)──→ TIMEOUT_CHECK ──(5s)──→ IDLE
any ──(goodbye)──→ IDLE
```

## Pipeline
- Phrase-level TTS: LLM tokens → `PhraseAccumulator` → TTS → speaker
- Min phrase length: `ko` = 30 chars, `en` = 50 chars (avoids clipped short phrases)
- Barge-in: Silero VAD fires during SPEAKING → `pipeline.interrupt()`

## Code Conventions
- Python 3.10+, type hints, `typing.Protocol` (no ABCs)
- `asyncio` throughout; sounddevice callback is the only sync entry point
- Config: `cfg.get("section.key", default)`
