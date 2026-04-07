# 커스텀 Wake Word 학습 가이드 (openWakeWord)

현재 적용 모델: `models/wake_word/Hey_Mack_20260309_205536.onnx`

---

## 학습 (Google Colab 권장)

공식 노트북: https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb

```python
target_phrase = "hey mack"
negative_phrases = ["hey mac", "hey back", "hey black"]
model_name = "Hey_Mack"
```

셀 순서대로 실행 → `Hey_Mack.onnx` 다운로드 (약 30분)

---

## 적용

```bash
cp Hey_Mack_NEW.onnx models/wake_word/
```

`config/default.yaml`:
```yaml
wake_word:
  model_paths:
    - "models/wake_word/Hey_Mack_NEW.onnx"
  active_models:
    - "Hey_Mack_NEW"
  threshold: 0.5
  patience: 3
  cooldown_seconds: 2.0
```

---

## 튜닝

| 파라미터 | 오탐 많음 | 미인식 많음 |
|---------|----------|-----------|
| `threshold` | 올림 (0.6~0.7) | 낮춤 (0.3~0.4) |
| `patience` | 올림 (5~7) | 낮춤 (2) |

실제 녹음 10~20개 추가 시 정확도 향상 (`rec -r 16000 -c 1 sample.wav trim 0 2`)
