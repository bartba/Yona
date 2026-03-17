# Custom Wake Word 모델 학습 가이드

> openWakeWord를 사용하여 "Hi Inspector" 등 커스텀 wake word를 학습하고
> Yona에 적용하는 방법을 설명합니다.

---

## 목차

1. [개요](#개요)
2. [사전 준비](#사전-준비)
3. [방법 A — Google Colab (권장)](#방법-a--google-colab-권장)
4. [방법 B — 로컬 학습](#방법-b--로컬-학습)
5. [Yona에 적용](#yona에-적용)
6. [테스트 및 튜닝](#테스트-및-튜닝)
7. [트러블슈팅](#트러블슈팅)

---

## 개요

openWakeWord는 합성 음성(Synthetic Speech)을 활용한 학습 파이프라인을 제공합니다.
실제 녹음 없이도 TTS로 생성한 positive 샘플과 배경 소음을 조합하여
고품질 wake word 모델을 만들 수 있습니다.

**학습 결과물:** `hi_inspector.onnx` (또는 원하는 이름의 `.onnx` 파일)

---

## 사전 준비

### 필수 환경

```bash
# Python 3.9+ (Colab은 기본 제공)
python --version

# openWakeWord 설치 (이미 설치된 경우 생략)
pip install openwakeword

# 학습 의존성 (로컬 학습 시)
pip install datasets speechbrain audiomentations
```

### 선택: 실제 녹음 데이터

합성 음성만으로도 학습 가능하지만, **실제 녹음을 10~20개 추가**하면 정확도가 향상됩니다.

```bash
# 녹음 도구 (Jetson에서)
sudo apt install sox
# 16kHz mono WAV로 녹음
rec -r 16000 -c 1 hi_inspector_01.wav trim 0 2
```

---

## 방법 A — Google Colab (권장)

가장 쉽고 빠른 방법입니다. GPU 없이도 30분~1시간이면 완료됩니다.

### Step 1: Colab 노트북 열기

openWakeWord 공식 학습 노트북:
https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb

### Step 2: 노트북에서 설정 변경

노트북 상단의 설정 셀에서 다음을 수정합니다:

```python
# wake word 텍스트 지정
target_phrase = "hi inspector"

# negative 텍스트 (오탐 방지용 — 유사 발음)
negative_phrases = [
    "hi instigator",
    "high inspector",
    "by inspector",
    "hi instructor",
    "hi in specter",
]

# 모델 이름
model_name = "hi_inspector"
```

### Step 3: 학습 실행

노트북의 셀을 순서대로 실행합니다:

1. **Generate positive samples** — Piper TTS로 "hi inspector" 합성 (~500개)
2. **Generate negative samples** — 유사 발음 + 일반 음성 생성
3. **Train model** — 약 20~30분 소요
4. **Export ONNX** — `hi_inspector.onnx` 다운로드

### Step 4: 모델 파일 다운로드

학습 완료 후 Colab에서 `.onnx` 파일을 다운로드합니다.

---

## 방법 B — 로컬 학습

GPU가 있는 PC 또는 서버에서 직접 학습합니다.

### Step 1: 학습 데이터 생성

```bash
# openWakeWord 소스 클론
git clone https://github.com/dscripka/openWakeWord.git
cd openWakeWord

# 합성 음성 생성 (Piper TTS 사용)
python -m openwakeword.train_custom_model \
    --phrase "hi inspector" \
    --model_name hi_inspector \
    --n_samples 500 \
    --n_negative_samples 500
```

### Step 2: 학습 실행

```bash
python -m openwakeword.train_custom_model \
    --phrase "hi inspector" \
    --model_name hi_inspector \
    --epochs 100 \
    --output_dir ./output
```

### Step 3: ONNX 내보내기

학습 스크립트가 자동으로 `output/hi_inspector.onnx`를 생성합니다.

---

## Yona에 적용

### Step 1: 모델 파일 배치

```bash
# 모델 디렉토리에 복사
cp hi_inspector.onnx /path/to/Yona/models/wake_word/hi_inspector.onnx
```

### Step 2: config/default.yaml 수정

```yaml
wake_word:
  model_paths:
    - "models/wake_word/hi_inspector.onnx"   # 커스텀 모델 경로
  active_models:
    - "hi_inspector"                          # 활성화할 모델 이름
  inference_framework: "onnx"
  threshold: 0.5
  patience: 3
  cooldown_seconds: 2.0
```

### Step 3: 동작 확인

```bash
# Yona 실행
python -m src.main

# 또는 wake word만 단독 테스트
python -c "
from src.config import Config
from src.events import EventBus
from src.wake import WakeWordDetector

cfg = Config()
bus = EventBus()
detector = WakeWordDetector(cfg, bus)
print('Loaded models:', detector.model_names)
"
```

---

## 테스트 및 튜닝

### threshold (감도 조절)

| 값 | 설명 |
|----|------|
| 0.3 | 민감 — 잘 인식하지만 오탐(false positive) 증가 |
| **0.5** | **기본값 — 균형** |
| 0.7 | 둔감 — 오탐 감소, 미인식(false negative) 증가 |

```yaml
# 시끄러운 환경에서는 높이고, 조용한 환경에서는 낮춤
wake_word:
  threshold: 0.6
```

### patience (연속 프레임 수)

```yaml
# patience=3: 3개 연속 프레임이 threshold 초과해야 탐지
# 높일수록 오탐 감소, 반응 약간 느려짐
wake_word:
  patience: 5
```

### 여러 wake word 동시 사용

```yaml
wake_word:
  model_paths:
    - "models/wake_word/hi_inspector.onnx"
  active_models:
    - "hi_inspector"
    - "hey_jarvis"       # 내장 모델도 함께 사용 가능
```

---

## 트러블슈팅

### 모델 로드 실패

```
FileNotFoundError: models/wake_word/hi_inspector.onnx
```
- 파일 경로 확인. Yona 프로젝트 루트 기준 상대 경로입니다.
- `model_paths: []`로 설정하면 내장 모델로 폴백됩니다.

### 오탐이 너무 많음 (False Positives)

1. `threshold`를 0.6~0.7로 올림
2. `patience`를 5~7로 올림
3. 학습 시 negative samples에 환경 소음 추가

### 인식이 안 됨 (False Negatives)

1. `threshold`를 0.3~0.4로 낮춤
2. `patience`를 2로 낮춤
3. 학습 시 positive samples 수 증가 (500 → 1000)
4. 실제 녹음 데이터 추가 (다양한 화자, 거리, 톤)

### Jetson Orin Nano 성능

- openWakeWord는 CPU 전용, ONNX 추론
- 512 샘플 (32ms) 기준 추론 시간: ~1ms (무시 가능)
- 메모리: ~10 MB (전체 모델 로드 시)
