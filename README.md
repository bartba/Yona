# Yona — Voice Chat Application

Streaming voice assistant for Nvidia Jetson Orin Nano + Polycom Sync 20 Plus.
Wake word → LISTENING → STT → LLM streaming → TTS phrase-by-phrase → speaker.
Supports barge-in (interrupt while speaking) and two-stage timeout.

## Hardware Requirements

- Nvidia Jetson Orin Nano (CUDA for STT, CPU for TTS + wake word)
- Polycom Sync 20 Plus (USB, `hw:0,0`)
- `sudo apt install libportaudio2`

## Installation

```bash
pip install -e .[dev]
```

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required environment variables (set at least one LLM provider):

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

Main config: `config/default.yaml`
System prompt: `config/prompts/system_prompt.txt`

## Running

```bash
python -m src.main                   # run the application
python -m src.main --list-devices    # list available audio devices
python -m src.main --config path     # use a custom config file
```

## Web Dashboard

When `web.enabled: true` in config, a dashboard is available at:
`http://127.0.0.1:8080`

Shows live transcripts, assistant responses, and conversation state.
Bound to loopback by default; set `web.host: 0.0.0.0` + `web.allowed_hosts` to expose.

## Logs

Log files are written to `logs/` (rotated daily).

## Tests

```bash
pytest
```
