# Yona v2 — Build Plan & Progress Tracker

> **How to use this file**
> - Check off `[ ]` → `[x]` yourself after reviewing each step
> - Start a new chat session per step to save tokens
> - Reference this file at the start of each session: *"plan.md의 Step N을 진행합시다"*

---

## Project Structure (Flat, 11 files)

```
src/
├── config.py      # YAML + .env loader
├── events.py      # EventType enum + EventBus
├── state.py       # ConversationState + StateMachine
├── audio.py       # AudioManager + AudioBuffer + ChimePlayer
├── vad.py         # VoiceActivityDetector (Silero ONNX)
├── wake.py        # WakeWordDetector (Porcupine)
├── stt.py         # Transcriber (faster-whisper large-v3-turbo)
├── llm.py         # ChatHandler + Context + History + OpenAI + Custom + factory
├── tts.py         # Synthesizer (XTTS v2)
├── pipeline.py    # PhraseAccumulator + StreamingPipeline
└── main.py        # YonaApp orchestrator + CLI
```

---

## System Features

| Feature | Implementation |
|---------|----------------|
| Wake word | Porcupine: "Hi Inspector" / "하이 검사기" |
| Wake word notify | Chime sound on detection |
| STT | faster-whisper large-v3-turbo, CUDA |
| LLM | OpenAI GPT **or** Custom company LLM (selectable) |
| TTS | XTTS v2 (Coqui), local GPU, 24 kHz |
| Pipeline | Streaming producer-consumer (LLM→phrase queue→TTS→audio queue→speaker) |
| Barge-in | VAD during SPEAKING → stop TTS immediately → LISTENING |
| AEC | Polycom Sync 20 Plus hardware AEC (USB direct, confirmed) |
| Goodbye intent | Keyword pattern match → farewell TTS → IDLE |
| Two-stage timeout | 15 s → "아직 계세요?" → 15 s → farewell → IDLE |
| History | In-session context + weekly JSON summary on disk |
| Language | Korean / English auto-detect, respond in same language |

---

## Hardware (Confirmed)

```
Device 0  Poly Sync 20: USB Audio (hw:0,0)   1 in / 2 out  ← USE THIS
Device 1  Poly BT600:   USB Audio (hw:1,0)   1 in / 2 out  ← standby only
```

`config/default.yaml` key values:
```yaml
input_device:   "Poly Sync 20"   # USB direct, low latency
output_device:  "Poly Sync 20"
input_channels:  1               # mono (HW beamforming + AEC done in Sync 20)
output_channels: 2               # stereo (USB confirmed)
chunk_size:    512               # 32 ms @ 16 kHz, matches Porcupine frame_length
buffer_seconds:  30              # ~1.9 MB ring buffer
```

---

## State Machine

```
IDLE ──(wake word)──────────────→ [chime] → LISTENING
LISTENING ──(speech + silence)──→ PROCESSING
PROCESSING ──(STT done)─────────→ SPEAKING   [pipeline starts]
SPEAKING ──(playback done)──────→ LISTENING
SPEAKING ──(barge-in VAD)───────→ LISTENING
LISTENING/SPEAKING ──(15 s)─────→ TIMEOUT_CHECK
TIMEOUT_CHECK ──(speech)────────→ LISTENING
TIMEOUT_CHECK ──(15 s more)─────→ TIMEOUT_FINAL → IDLE
SPEAKING/PROCESSING ──(goodbye)─→ IDLE
```

---

## Streaming Pipeline

```
LLM stream tokens
    ↓
PhraseAccumulator  (.!?。 boundaries)
    ↓ phrase: str
asyncio.Queue[str]  (maxsize=5)
    ↓
TTS Worker  (XTTS v2, ~0.5 s/phrase)
    ↓ (np.ndarray, sample_rate)
asyncio.Queue[tuple]  (maxsize=10)
    ↓
AudioPlaybackWorker  → speaker
    ↑
[barge-in VAD — parallel monitoring]
interrupt() → drain queues → cancel tasks
```

---

## Dependencies

```toml
pvporcupine>=3.0.0       # wake word
faster-whisper>=1.0.0    # STT
openai>=1.0.0            # LLM option A
httpx>=0.27.0            # LLM option B (custom)
TTS>=0.22.0              # XTTS v2
onnxruntime>=1.16.0      # Silero VAD
sounddevice>=0.4.6       # audio I/O
numpy>=1.24.0
pyyaml>=6.0
python-dotenv>=1.0.0
aiofiles>=23.2.0
```

---

## Environment Variables (`.env`)

```bash
# LLM provider: "openai" or "custom"
LLM_PROVIDER=openai

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Custom LLM
CUSTOM_LLM_URL=http://your-server/api/chat
CUSTOM_LLM_KEY=your-api-key
CUSTOM_LLM_MODEL=your-model-name

# Porcupine
PORCUPINE_ACCESS_KEY=your-access-key
```

---

## Pre-flight Checklist

- [ ] Picovoice Console — get AccessKey
- [ ] Train "Hi Inspector" (EN) → download `hi_inspector_en.ppn` (Linux/ARM64)
- [ ] Train "하이 검사기" (KO) → download `hi_inspector_ko.ppn` (Linux/ARM64)
- [ ] Verify GPU memory headroom: large-v3-turbo (~1.5 GB) + XTTS v2 (~3 GB) on Jetson 8 GB
- [ ] Prepare Custom LLM API spec (needed for Step 08d)

---

## Progress Tracker

> Each step = concept → code → `pytest` → review → ✅

### Foundation layer

- [x] **Step 01** — `src/config.py` + `config/default.yaml`
  - YAML loader, `${ENV_VAR}` expansion, `Config` class
  - `pytest tests/test_config.py` → **24/24 passed**

- [x] **Step 02** — `src/events.py`
  - `EventType` enum (all 18 types), async `EventBus` (pub/sub, queue-based)
  - `pytest tests/test_events.py`

- [x] **Step 03** — `src/state.py`
  - `ConversationState` enum (6 states), `StateMachine` (transitions + 2-stage timeout)
  - `pytest tests/test_state.py`

### Audio layer

- [ ] **Step 04** — `src/audio.py`
  - `AudioBuffer` (ring buffer), `AudioManager` (sounddevice I/O, multi-callback), `ChimePlayer`
  - `pytest tests/test_audio.py`

- [ ] **Step 05** — `src/vad.py`
  - `VoiceActivityDetector` — Silero ONNX, 512-sample windows, LSTM state
  - `pytest tests/test_vad.py`

### Recognition layer

- [ ] **Step 06** — `src/wake.py`
  - `WakeWordDetector` — Porcupine, float32→int16, cooldown
  - `pytest tests/test_wake.py`

- [ ] **Step 07** — `src/stt.py`
  - `Transcriber` — faster-whisper large-v3-turbo, CUDA, Korean/English auto-detect
  - `pytest tests/test_stt.py`

### LLM layer

- [ ] **Step 08** — `src/llm.py`
  - 08a `ChatHandler` Protocol + `ConversationContext`
  - 08b `ConversationHistory` (weekly JSON summaries)
  - 08c `OpenAIChatHandler` (openai SDK, streaming)
  - 08d `CustomLLMChatHandler` (httpx, streaming — API spec TBD)
  - 08e `create_chat_handler()` factory
  - `pytest tests/test_llm.py`

### Output layer

- [ ] **Step 09** — `src/tts.py`
  - `Synthesizer` — XTTS v2, 24 kHz, optional speaker WAV, `run_in_executor`
  - `pytest tests/test_tts.py`

### Integration layer

- [ ] **Step 10** — `src/pipeline.py`
  - `PhraseAccumulator` (sentence boundary detection)
  - `StreamingPipeline` (3-worker: LLM worker, TTS worker, playback worker)
  - `interrupt()` (barge-in: drain queues, cancel tasks)
  - `pytest tests/test_pipeline.py`

- [ ] **Step 11** — `src/main.py`
  - `YonaApp`: component init chain, `_audio_callback` state dispatch,
    all event handlers, 2-stage timeout messages, goodbye intent, graceful shutdown
  - `pytest tests/test_main.py` (integration, mocked components)

---

## Design Decisions (ADR)

| Decision | Choice | Reason |
|----------|--------|--------|
| Structure | Flat 11 files, no sub-packages | Direct navigation, no `__init__` chains |
| Interfaces | `typing.Protocol` | No ABC inheritance, easy to mock |
| Async | `asyncio` throughout | sounddevice callback is sync; everything else async |
| LLM streaming | `AsyncIterator[str]` | Standardised at Protocol level |
| Phrase split | Regex boundary detection | No LLM dependency, fast |
| Barge-in | `asyncio.Event` + `task.cancel()` | Clean cancellation propagation |
| TTS sample rate | 24 000 Hz (XTTS v2 native) | Passed to `AudioManager.play_audio(sr=...)` |
| History store | JSON files `data/history/` | No DB, simple, weekly granularity |
| Chime | Programmatic sine wave | No external file dependency |
