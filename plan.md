# Yona — Build Plan & Progress Tracker

> Start each session: *"plan.md의 Step N을 진행합시다"*

---

## Project Structure

```
src/
├── config.py      # YAML + .env loader
├── events.py      # EventType enum + EventBus
├── state.py       # ConversationState + StateMachine
├── audio.py       # AudioManager + AudioBuffer + ChimePlayer
├── vad.py         # VoiceActivityDetector (Silero VAD v6 ONNX)
├── wake.py        # WakeWordDetector (openWakeWord, ONNX)
├── stt.py         # Transcriber (faster-whisper large-v3-turbo)
├── llm.py         # ChatHandler + Context + History + providers + factory
├── tts.py         # Synthesizer Protocol + SupertonicSynthesizer + factory
├── pipeline.py    # PhraseAccumulator + StreamingPipeline
├── main.py        # YonaApp orchestrator + CLI
└── web.py         # Web dashboard (FastAPI)
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
TIMEOUT_CHECK ──(5 s more)──────→ TIMEOUT_FINAL → IDLE
SPEAKING/PROCESSING ──(goodbye)─→ IDLE
```

---

## Streaming Pipeline

```
LLM stream tokens → PhraseAccumulator (.!?。 boundaries)
    → Queue[str] → TTS Worker → Queue[tuple] → AudioPlaybackWorker → speaker
                                                        ↑
                                         [barge-in VAD — parallel]
                                         interrupt() → drain queues
```

---

## Progress Tracker

### Foundation layer
- [x] **Step 01** — `src/config.py` + `config/default.yaml`
- [x] **Step 02** — `src/events.py`
- [x] **Step 03** — `src/state.py`

### Audio layer
- [x] **Step 04** — `src/audio.py`
- [x] **Step 05** — `src/vad.py` (Silero VAD v6)

### Recognition layer
- [x] **Step 06** — `src/wake.py` (openWakeWord, Hey Mack custom model)
- [x] **Step 07** — `src/stt.py` (faster-whisper large-v3-turbo, CUDA float16)

### LLM layer
- [x] **Step 08** — `src/llm.py` (OpenAI / Claude / Custom — `LLM_PROVIDER` env)

### Output layer
- [x] **Step 09** — `src/tts.py` (Supertonic ONNX, CPU, 44.1 kHz)

### Integration layer
- [x] **Step 10** — `src/pipeline.py` (streaming + dynamic language switch + barge-in)
- [x] **Step 11** — `src/main.py` (YonaApp orchestrator, all event handlers, web.py)

### Deployment layer
- [ ] **Step 12** — systemd 서비스 배포
  - `yona.service` unit file → `/etc/systemd/system/yona.service`
  - `Restart=on-failure`, `RestartSec=3`, `WantedBy=multi-user.target`
  - `EnvironmentFile=` 로 `.env` 로드
  - 오디오 그룹 권한 + USB 디바이스 접근 확인
  - `journalctl -u yona` 로그 연동

---

## V3 Roadmap

- **PTT** — Poly Sync 20 Call 버튼 → `evdev`/`hidraw` → 즉시 LISTENING 진입 (`src/ptt.py`)
- **MCP 연동** — MongoDB MCP (데이터 조회/통계), File I/O MCP

---

## Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Structure | Flat 12 files, no sub-packages | Direct navigation |
| Interfaces | `typing.Protocol` | No ABC inheritance, easy to mock |
| Async | `asyncio` throughout | sounddevice callback is sync; everything else async |
| TTS | Supertonic ONNX, CPU | Korean+English bilingual, no CUDA OOM |
| TTS sample rate | 44 100 Hz | Supertonic native output |
| History | JSON files `data/history/` | Simple, weekly granularity |
| Wake word | openWakeWord ONNX | Offline, no internet verification needed |
| STT | large-v3-turbo float16 | RTF 0.16x, 3x faster load vs int8 |
