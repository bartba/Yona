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
├── wake.py        # WakeWordDetector (openWakeWord, ONNX)
├── stt.py         # Transcriber (faster-whisper large-v3-turbo)
├── llm.py         # ChatHandler + Context + History + OpenAI + Custom + factory
├── tts.py         # Synthesizer Protocol + MeloSynthesizer + factory
├── pipeline.py    # PhraseAccumulator + StreamingPipeline
└── main.py        # YonaApp orchestrator + CLI
```

---

## System Features

| Feature | Implementation |
|---------|----------------|
| Wake word | openWakeWord (ONNX, offline): "Hi Inspector" |
| Wake word notify | Chime sound on detection |
| STT | faster-whisper large-v3-turbo, CUDA |
| LLM | OpenAI GPT **or** Claude **or** Custom LLM (selectable) |
| TTS | MeloTTS — CPU, 24 kHz (selectable via `tts.provider`) |
| Pipeline | Streaming producer-consumer (LLM→phrase queue→TTS→audio queue→speaker) |
| Barge-in | VAD during SPEAKING → stop TTS immediately → LISTENING |
| AEC | Polycom Sync 20 Plus hardware AEC (USB direct, confirmed) |
| Goodbye intent | Keyword pattern match → farewell TTS → IDLE |
| Two-stage timeout | 15 s → "아직 계세요?" → 5 s → farewell → IDLE |
| History | In-session context + weekly JSON summary on disk |
| Language | Korean / English auto-detect, respond in same language |

---

## Hardware (Confirmed)

```
Device 0  Poly Sync 20: USB Audio (hw:0,0)   1 in / 2 out  ← USE THIS
```

`config/default.yaml` key values:
```yaml
input_device:   "Poly Sync 20"   # USB direct, low latency
output_device:  "Poly Sync 20"
input_channels:  1               # mono (HW beamforming + AEC done in Sync 20)
output_channels: 2               # stereo (USB confirmed)
chunk_size:    512               # 32 ms @ 16 kHz (openWakeWord accepts any size, buffers internally)
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
TIMEOUT_CHECK ──(5 s more)─────→ TIMEOUT_FINAL → IDLE
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
TTS Worker  (MeloTTS, CPU)
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
openwakeword>=0.6.0      # wake word (ONNX, offline)
faster-whisper>=1.0.0    # STT
openai>=1.0.0            # LLM option A
anthropic>=0.40.0        # LLM option B (Claude)
httpx>=0.27.0            # LLM option C (custom)
melotts>=0.1.0           # TTS 
onnxruntime>=1.16.0      # Silero VAD + openWakeWord
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
OPENAI_MODEL=gpt-5-mini

# Custom LLM
CUSTOM_LLM_URL=http://your-server/api/chat
CUSTOM_LLM_KEY=your-api-key
CUSTOM_LLM_MODEL=your-model-name

```

---

## Pre-flight Checklist

- [x] Train "Hi Inspector" custom wake word model → `hi_inspector.onnx` (openWakeWord)
- [ ] Download MeloTTS Korean model (pip install 시 자동)
- [ ] Verify GPU memory: large-v3-turbo (~1.5 GB) + TTS CPU mode on Jetson 8 GB ✅ (예상 ~4.8 GB)
- [ ] Prepare Custom LLM API spec (needed for Step 08e)

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

- [x] **Step 04** — `src/audio.py`
  - `AudioBuffer` (ring buffer), `AudioManager` (sounddevice I/O, multi-callback), `ChimePlayer`
  - `pytest tests/test_audio.py`

- [x] **Step 05** — `src/vad.py`
  - `VoiceActivityDetector` — Silero ONNX, 512-sample windows, LSTM state
  - `pytest tests/test_vad.py`

### Recognition layer

- [x] **Step 06** — `src/wake.py`
  - `WakeWordDetector` — openWakeWord (ONNX, offline), float32→int16, patience + cooldown
  - `pytest tests/test_wake.py`

- [x] **Step 07** — `src/stt.py`
  - `Transcriber` — faster-whisper large-v3-turbo, CUDA, Korean/English auto-detect
  - `pytest tests/test_stt.py`

  **STT GPU 환경 구축 (Jetson Orin Nano 기준)**

  ```bash
  # faster-whisper 설치 (CUDA 지원 포함됨)
  pip install faster-whisper

  # ffmpeg 설치 (오디오 변환 유틸리티)
  sudo apt install ffmpeg -y

  # soundfile 설치 (wav 파일 I/O)
  pip install soundfile
  ```

  **모델 선택 근거 (2026-03-10 Jetson Orin Nano 실측)**

  테스트 조건: 실제 한국어 음성 12.2초, float16, beam_size=5, warmup 후 측정

  | 모델 | 처리 시간 | RTF | 비고 |
  |---|---|---|---|
  | medium | 2.37s | 0.19x | 정확도 소폭 낮음 |
  | **large-v3-turbo** | **1.98s** | **0.16x** | ✅ **채택** |

  - large-v3-turbo: medium보다 빠르고 WER도 낮음 (공식 벤치마크와 일치)
  - RTF 0.16x = 실시간의 6배 빠름 → 3초 발화 기준 STT 처리 ~0.5초
  - `compute_type=float16`: int8_float16 대비 로드 3x 빠름 (7.8s vs 22.5s), 메모리 3x 적게 사용
  - `vad_filter=True` 주의: 무음 구간을 건너뛰므로 벤치마크 시 오해 소지 있음 (0.02s 오측정)
  - 모델 파일은 첫 실행 시 자동 다운로드 (~1.5 GB), `~/.cache/huggingface/` 에 저장

  **전체 파이프라인 지연 예측 (3초 발화 기준)**

  ```
  VAD silence 대기:      0.8s  (1.5s → 0.8s 단축)
  STT large-v3-turbo:   ~0.5s
  LLM 첫 토큰:          ~1.0s
  TTS 첫 phrase:        ~1.5s
  ─────────────────────────────
  체감 응답 지연:        ~3.8s
  ```

### LLM layer

- [x] **Step 08** — `src/llm.py`
  - 08a `ChatHandler` Protocol + `ConversationContext`
  - 08b `ConversationHistory` (weekly JSON summaries)
  - 08c `OpenAIChatHandler` (openai SDK, streaming)
  - 08d `ClaudeChatHandler` (anthropic SDK, streaming)
  - 08e `CustomLLMChatHandler` (httpx, streaming — API spec TBD)
  - 08f `create_chat_handler()` factory — `LLM_PROVIDER`: `"openai"` | `"claude"` | `"custom"`
  - `pytest tests/test_llm.py`

### Output layer

- [x] **Step 09** — `src/tts.py`
  - `Synthesizer` Protocol + `MeloSynthesizer` (CPU, 24 kHz, Korean + English)
  - `create_synthesizer()` factory — `tts.provider`: `"melo"`
  - 24 kHz output, `run_in_executor`
  - `pytest tests/test_tts.py`

### Integration layer

- [x] **Step 10** — `src/pipeline.py`
  - `PhraseAccumulator` (sentence boundary detection)
  - `StreamingPipeline` (3-worker: LLM worker, TTS worker, playback worker)
  - `interrupt()` (barge-in: drain queues, cancel tasks)
  - **동적 TTS 언어 전환**: STT 감지 언어(`info.language`)를 Pipeline에 전달 →
    MeloTTS: `TTS(language=...)` 재로드 (PROCESSING 상태에서 LLM 응답 대기 중 선제 실행)
  - `pytest tests/test_pipeline.py`

- [x] **Step 11** — `src/main.py`
  - `YonaApp`: component init chain, `_audio_callback` state dispatch,
    all event handlers, 2-stage timeout messages, goodbye intent, graceful shutdown
  - `pytest tests/test_main.py` (integration, mocked components)

### Deployment layer

- [ ] **Step 12** — systemd 서비스 배포
  - `yona.service` unit file 작성 → `/etc/systemd/system/yona.service`
  - `Restart=on-failure` + `RestartSec=3` (비정상 종료 시 자동 재시작)
  - `WantedBy=multi-user.target` (Jetson 부팅 시 자동 시작)
  - `EnvironmentFile=` 로 `.env` 로드 (API 키 등)
  - 오디오 그룹 권한 + USB 디바이스 접근 확인
  - `journalctl -u yona` 로그 연동
  - `systemctl enable|start|stop|status yona` 운영 매뉴얼

---

## V3 Roadmap (Post-v2 Features)
- Tool calling or MCP
  - MongoDB MCP
    . Data query, Simple Statistics 
  - file i/o mcp
    . search/find a file
    . read/write file
    . etc 

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
| TTS sample rate | 24 000 Hz (MeloTTS native) | Passed to `AudioManager.play_audio(sr=...)` |
| History store | JSON files `data/history/`  | DB (Optional), simple, weekly granularity |
| Chime | Programmatic sine wave | external file if exists |
