# Yona v2 — 실전 하드웨어 통합 테스트

> Jetson Orin Nano + Polycom Sync 20 Plus 환경
> 한 세션에 1~2 Step씩 진행 · 단순→복잡 (bottom-up)

---

## Phase 1: 하드웨어 기초 검증

### [x] HW-01: 오디오 디바이스 I/O 확인
- 테스트 스크립트: `python tests/test_hw_audio.py`
- Poly Sync 20 인식 (디바이스 #0, 1 in / 2 out) ✅
- 5초 녹음 → WAV 저장 → 재생 ✅
- 차임 사운드 재생 ✅
- ALSA 볼륨 설정 (65%, 13/20) ✅
- `audio.py`에 `_apply_volume()` 추가, `config/default.yaml`에 `volume_percent: 65` 추가 ✅

### [x] HW-02: AudioManager + AudioBuffer 라이브
- `AudioManager`로 3초 녹음 → `AudioBuffer` 저장 → `play_audio`로 재생 ✅
- chunk 수 검증: 93개 수신 (예상 93.8, 오차 0.8%) ✅
- 리샘플링(16→48kHz) 재생 정상 ✅
- Poly Sync 20 AGC: 초반 ~0.5초 볼륨 높음 → 정상 동작 (STT/VAD에 영향 없음)
- **성공 기준**: chunk 수 오차 5% 이내, 재생 정상

---

## Phase 2: 감지 레이어

### [x] HW-03: Silero VAD 라이브
- Silero VAD v5→v6 업그레이드 (drop-in, 노이즈 16% 오류 감소) ✅
- vad.py 수정: state 인터페이스(h/c→단일 state) + context window(64 samples) 추가 ✅
- 15초 말하기/멈추기: STARTED 4회, ENDED 3회 + 1 미완료 — 이벤트 쌍 정확 매칭 ✅
- 환경 소음 baseline: max=0.22 (threshold 0.5 대비 충분한 마진) ✅
- false positive: 0회 ✅
- **성공 기준**: 이벤트 쌍 정확 매칭, 환경소음 false positive 0

### [x] HW-04: Wake Word 라이브
- pretrained "alexa" 모델로 테스트 (config `wake_word.wake_phrase`)
- TPR: 100% (6/6), cooldown 준수 (간격 ≥ 2.0s) ✅
- 환경 소음 FP: 0회 ✅
- 유사 발음 FPR: 5회 — pretrained 모델 한계, custom 모델로 해결 예정
- `wake.py` 수정: `predict()` patience/threshold 파라미터 제거, 자체 patience 카운터 구현
- `wake_word.wake_phrase` config 필드 추가 (표시용 phrase)
- **성공 기준**: TPR ≥ 80% ✅, FPR = 모델 의존 (custom 모델 필요), cooldown ✅

### [x] HW-05: VAD Barge-in 모드
- 테스트 스크립트: `python tests/test_hw_bargein.py`
- 톤 재생(440Hz, 8초) 중 무음: speech_prob max=0.1155, false barge-in=0회 ✅
- Poly Sync 20 AEC 검증: 에코 잔류 없음 (prob max < 0.12, threshold 0.7 대비 충분한 마진) ✅
- 톤 재생 중 실제 음성: barge-in 2회 감지, speech_prob max=1.0 ✅
- **성공 기준**: false barge-in 0회 ✅, 실제 음성 시 감지 성공 ✅

---

## Phase 3: 인식 레이어

### [ ] HW-06: STT (faster-whisper) 라이브
- 한국어/영어/혼용 5개 문장 실시간 STT
- 정확도(WER), 처리 시간(RTF), 언어 감지, GPU 메모리 확인
- **성공 기준**: 핵심 단어 정확도 > 90%, RTF < 0.2x, 언어 감지 정확

### [ ] HW-07: LLM 스트리밍
- 설정된 provider로 4개 프롬프트 스트리밍
- TTFT(첫 토큰 지연), 토큰/초, 응답 품질 측정
- **성공 기준**: TTFT < 2초, 자연스러운 대화체

---

## Phase 4: 출력 레이어

### [ ] HW-08: MeloTTS 라이브
- 한국어/영어 5개 문장 합성 → 재생
- 합성 RTF, 언어 전환 시간, 발음 품질, 리샘플링 품질
- **성공 기준**: RTF < 1.0x, 발음 이해 가능, 언어 전환 < 3초

---

## Phase 5: 파이프라인 통합

### [ ] HW-09: 스트리밍 파이프라인 (LLM→TTS→Speaker)
- 실제 질문 → 3-worker 파이프라인 실행 → 이벤트 타임라인 분석
- TTFA(첫 음성까지 시간), phrase 간 무음, queue backpressure
- **성공 기준**: TTFA < 5초, phrase 간 무음 < 1초

### [ ] HW-10: Barge-in 파이프라인 중단
- 긴 응답 재생 중 말해서 pipeline.interrupt() 테스트
- 중단 속도, AEC 영향, 파이프라인 정리 확인
- **성공 기준**: 1초 이내 중단, false barge-in 없음

---

## Phase 6: 전체 시스템 통합

### [ ] HW-11: 기본 대화 1턴
- `python -m src.main` 실행 → wake → 질문 → 응답 → 2턴째
- 상태 전이 로그, 각 단계 소요 시간
- **성공 기준**: 2턴 연속 대화 성공, 에러 없음

### [ ] HW-12: 다중 턴 대화 + 컨텍스트 유지
- 5턴 대화 (이름 기억, 맥락 참조, goodbye intent)
- ConversationContext 유지, history JSON 저장 확인
- **성공 기준**: 컨텍스트 유지, goodbye→IDLE, history 저장

### [ ] HW-13: 2단계 타임아웃
- 15초 침묵 → "아직 계세요?" → Case A: 응답 / Case B: 5초 더 → farewell → IDLE
- **성공 기준**: 양쪽 경로 모두 정상

### [ ] HW-14: 한국어/영어 전환
- 같은 세션에서 한→영→한 전환
- STT 언어 감지 → MeloTTS 언어 자동 전환
- **성공 기준**: 언어 감지 정확, TTS 전환 자연스러움, 전환 지연 < 3초

---

## Phase 7: 엣지 케이스 및 스트레스

### [ ] HW-15: 에러 복구
- 빈 오디오, 네트워크 차단(iptables), 극히 짧은/긴 발화, 빠른 연속 wake
- **성공 기준**: 모든 에러 상황에서 앱 생존, 적절한 상태 복귀

### [ ] HW-16: 소음 환경
- 음악/대화/에어컨/키보드 소음 중 VAD prob 분포 측정
- 소음 중 wake word, STT 정확도
- **성공 기준**: 소음 환경에서 VAD max < 0.5, false wake = 0

### [ ] HW-17: 메모리 누수 + 장시간 안정성
- 30분 연속 사용 (6회 대화 사이클), RSS/GPU 메모리 추적
- **성공 기준**: 메모리 증가 < 50MB/시간, 성능 일정

### [ ] HW-18: E2E 지연 시간 정밀 측정
- DEBUG 로그로 각 구간(VAD→STT→LLM TTFT→TTS→재생) 측정
- **목표**: 체감 총 지연(SPEECH_ENDED → PLAYBACK_STARTED) < 5초

---

## Phase 8: 배포 전 최종 확인

### [ ] HW-19: Graceful Shutdown
- IDLE 중 Ctrl+C, 대화 중 Ctrl+C, SIGTERM
- **성공 기준**: 3가지 모두 깨끗한 종료

### [ ] HW-20: 전체 시나리오 회귀 테스트
- [ ] 앱 시작 + 초기화
- [ ] Wake word + 차임
- [ ] 한국어 대화 1턴
- [ ] 영어 대화 1턴
- [ ] 컨텍스트 유지
- [ ] Barge-in
- [ ] 빈 발화 에러 처리
- [ ] Goodbye intent → IDLE
- [ ] 다시 wake → 새 대화
- [ ] 타임아웃 Stage 1
- [ ] 타임아웃 Stage 2
- [ ] Ctrl+C 정상 종료

---

## 테스트 스크립트 목록

| Phase | 스크립트 | 설명 |
|-------|---------|------|
| 1 | `tests/test_hw_audio.py` | 오디오 디바이스 I/O (대화형 메뉴) |
| 2 | `tests/test_hw_vad.py` | VAD + Wake Word + Barge-in |
| 3 | `tests/test_hw_stt.py` | STT + LLM 스트리밍 |
| 4 | `tests/test_hw_tts.py` | MeloTTS 합성 + 재생 |
| 5 | `tests/test_hw_pipeline.py` | 스트리밍 파이프라인 + Barge-in |
| 6-8 | `python -m src.main` | 전체 시스템 통합 (로그 기반) |

## 유용한 디버그 명령어

```bash
watch -n 1 nvidia-smi                              # GPU 모니터링
LOG_LEVEL=DEBUG python -m src.main 2>&1 | tee /tmp/yona_test.log  # 상세 로그
tail -f logs/yona.log                              # 로그 실시간 추적
grep -E 'STATE_CHANGED|WAKE_WORD|SPEECH_|BARGE' /tmp/yona_test.log
ps -p $(pgrep -f src.main) -o rss=                 # 메모리 확인
```
