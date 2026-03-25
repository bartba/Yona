"""HW-06: STT (faster-whisper) 라이브 테스트.

별도 터미널에서 실행:
    python tests/test_hw_stt.py                  # config 기본값 (large-v3-turbo)
    python tests/test_hw_stt.py --model base     # 가벼운 모델 (Remote SSH 환경)
    python tests/test_hw_stt.py --model medium   # 중간 모델

테스트 항목:
    1. 모델 로드 시간 + 메모리 확인
    2. Warmup 추론 (CUDA JIT 초기화)
    3. 한국어/영어/혼용 5개 문장 실시간 STT
    4. 처리 시간(RTF), 언어 감지 정확도
    성공 기준: 핵심 단어 정확도 > 90%, RTF < 0.2x, 언어 감지 정확

모델별 참고:
    base          ~150MB GPU, 한국어 정확도 낮음 — 파이프라인 검증용
    small         ~500MB GPU, 한국어 중간 — 개발 중 테스트
    medium        ~1.5GB GPU, 한국어 양호
    large-v3-turbo ~1.5GB GPU, 한국어 최고 — 최종 성능 측정용 (SSH 끊고 실행)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import Config
from src.events import EventBus, EventType
from src.audio import AudioManager
from src.vad import VoiceActivityDetector


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

TEST_SENTENCES = [
    {"id": 1, "lang": "ko", "prompt": "안녕하세요, 오늘 날씨가 좋네요", "keywords": ["안녕", "날씨"]},
    {"id": 2, "lang": "ko", "prompt": "내일 회의 일정을 알려줘", "keywords": ["내일", "회의", "일정"]},
    {"id": 3, "lang": "en", "prompt": "What time is the meeting tomorrow?", "keywords": ["time", "meeting", "tomorrow"]},
    {"id": 4, "lang": "en", "prompt": "Tell me about the weather today", "keywords": ["weather", "today"]},
    {"id": 5, "lang": "mixed", "prompt": "요나야, tell me a joke 하나 해줘", "keywords": ["요나", "joke"]},
]

MAX_RECORD_SECONDS = 10  # 한 문장 최대 녹음 시간
PRE_SPEECH_SECONDS = 0.5  # SPEECH_STARTED 이전 오디오 포함 (컨텍스트용)


def get_memory_info() -> dict[str, float | None]:
    """Jetson unified memory 정보 반환 (MB). free 명령 사용."""
    try:
        import subprocess
        result = subprocess.run(
            ["free", "-m"], capture_output=True, text=True, timeout=5,
        )
        # Mem:  total  used  free  shared  buff/cache  available
        parts = result.stdout.strip().split("\n")[1].split()
        return {
            "total_mb": float(parts[1]),
            "used_mb": float(parts[2]),
            "available_mb": float(parts[6]),
        }
    except Exception:
        return {"total_mb": None, "used_mb": None, "available_mb": None}


async def record_utterance(
    manager: AudioManager,
    vad: VoiceActivityDetector,
    bus: EventBus,
) -> tuple[np.ndarray | None, float]:
    """VAD로 발화를 감지해서 녹음. (audio, speech_duration_seconds) 반환.

    콜백 스레드에서 vad._speech_active 전환을 직접 감시하여
    정확한 발화 타임스탬프를 기록합니다.
    SPEECH_STARTED 이전 PRE_SPEECH_SECONDS 분량도 포함하여
    STT에 충분한 컨텍스트를 제공합니다.
    """
    q_start = bus.subscribe(EventType.SPEECH_STARTED)
    q_end = bus.subscribe(EventType.SPEECH_ENDED)

    vad.reset()

    # 청크 수집 (타임스탬프 포함) — 콜백 스레드에서 기록
    chunks: list[tuple[float, np.ndarray]] = []
    lock = threading.Lock()
    t0 = time.monotonic()

    # 콜백 스레드에서 기록하는 정확한 타임스탬프
    speech_start_ts: float | None = None
    speech_end_ts: float | None = None

    def on_chunk(chunk: np.ndarray) -> None:
        nonlocal speech_start_ts, speech_end_ts
        ts = time.monotonic() - t0
        with lock:
            chunks.append((ts, chunk.copy()))

        # VAD 상태 전환을 콜백 스레드에서 직접 감시
        was_active = vad._speech_active
        vad.process_chunk(chunk)
        is_active = vad._speech_active

        if not was_active and is_active:
            speech_start_ts = ts
        elif was_active and not is_active and speech_start_ts is not None:
            speech_end_ts = ts

    manager.add_input_callback(on_chunk)

    try:
        # 1) SPEECH_STARTED 대기 (이벤트는 흐름 제어용)
        print("    [대기 중] 말씀하세요...")
        try:
            await asyncio.wait_for(q_start.get(), timeout=MAX_RECORD_SECONDS)
        except asyncio.TimeoutError:
            print("    [TIMEOUT] 음성이 감지되지 않았습니다.")
            return None, 0.0

        print("    [녹음 중] 말을 마치면 자동으로 끝납니다...")

        # 2) SPEECH_ENDED 대기
        try:
            await asyncio.wait_for(q_end.get(), timeout=MAX_RECORD_SECONDS)
        except asyncio.TimeoutError:
            print("    [TIMEOUT] 발화가 너무 깁니다.")

        # 콜백 스레드가 기록한 정확한 타임스탬프 사용
        if speech_start_ts is None:
            return None, 0.0

        end_ts = speech_end_ts if speech_end_ts is not None else (time.monotonic() - t0)
        speech_duration = end_ts - speech_start_ts

        # 발화 구간 ± 여유분의 청크만 추출
        trim_start = max(0.0, speech_start_ts - PRE_SPEECH_SECONDS)
        trim_end = end_ts + 0.1  # 약간의 후행 포함

        with lock:
            speech_chunks = [
                c for t, c in chunks
                if trim_start <= t <= trim_end
            ]

        if not speech_chunks:
            return None, 0.0

        audio = np.concatenate(speech_chunks)
        return audio, speech_duration

    finally:
        manager.remove_input_callback(on_chunk)
        bus.unsubscribe(EventType.SPEECH_STARTED, q_start)
        bus.unsubscribe(EventType.SPEECH_ENDED, q_end)


async def run_test(model_override: str | None = None) -> None:
    cfg = Config()
    bus = EventBus()
    manager = AudioManager(cfg)
    vad = VoiceActivityDetector(cfg, bus)

    # 모델 오버라이드 적용
    model_size = model_override or cfg.get("stt.model_size", "large-v3-turbo")
    compute_type = cfg.get("stt.compute_type", "float16")
    device = cfg.get("stt.device", "cuda")
    language = cfg.get("stt.language", None)

    print("\n" + "=" * 60)
    print("HW-06: STT (faster-whisper) 라이브 테스트")
    print("=" * 60)
    print(f"  모델: {model_size} (device={device}, compute={compute_type}, lang={language or 'auto'})")
    if model_override:
        print(f"  (--model 인자로 오버라이드됨, config 기본값: {cfg.get('stt.model_size', 'large-v3-turbo')})")

    # ==================================================================
    # Phase 1: 모델 로드 + 메모리
    # ==================================================================
    print("\n── Phase 1: 모델 로드 ──")

    mem_before = get_memory_info()
    if mem_before["used_mb"] is not None:
        print(f"  로드 전 메모리: {mem_before['used_mb']:.0f}MB 사용 / {mem_before['available_mb']:.0f}MB 가용")

    print(f"  faster-whisper '{model_size}' 모델 로딩 중...")
    t_load_start = time.monotonic()

    from src.stt import Transcriber
    # model_size를 오버라이드하기 위해 config 값을 임시 변경
    original_model_size = cfg.get("stt.model_size", "large-v3-turbo")
    cfg._data["stt"]["model_size"] = model_size
    try:
        transcriber = Transcriber(cfg, bus)
    finally:
        cfg._data["stt"]["model_size"] = original_model_size

    t_load = time.monotonic() - t_load_start
    mem_after = get_memory_info()

    print(f"  모델 로드 시간: {t_load:.1f}초")
    if mem_after["used_mb"] is not None:
        print(f"  로드 후 메모리: {mem_after['used_mb']:.0f}MB 사용 / {mem_after['available_mb']:.0f}MB 가용")
        if mem_before["used_mb"] is not None:
            delta = mem_after["used_mb"] - mem_before["used_mb"]
            print(f"  모델 메모리 증가: ~{delta:.0f}MB")

    # ==================================================================
    # Phase 1.5: Warmup (CUDA JIT 초기화)
    # ==================================================================
    print("\n── Phase 1.5: Warmup 추론 ──")
    print("  CUDA JIT 초기화를 위해 더미 오디오로 추론합니다...")

    # 1초짜리 무음 오디오
    dummy_audio = np.zeros(16_000, dtype=np.float32)
    t_warmup_start = time.monotonic()
    await transcriber.transcribe(dummy_audio)
    t_warmup = time.monotonic() - t_warmup_start
    print(f"  Warmup 완료: {t_warmup:.1f}초")

    # ==================================================================
    # Phase 2: 5개 문장 STT
    # ==================================================================
    print(f"\n── Phase 2: 실시간 STT 테스트 ({len(TEST_SENTENCES)}문장) ──")
    print("  각 문장을 화면에 표시된 대로 말씀하세요.")
    print("  VAD가 발화 끝을 감지하면 자동으로 STT를 실행합니다.")
    input("\n  준비되면 Enter를 누르세요... ")

    await manager.start()

    results: list[dict] = []

    for sent in TEST_SENTENCES:
        lang_label = {"ko": "한국어", "en": "영어", "mixed": "혼용"}[sent["lang"]]
        print(f"\n  ── 문장 {sent['id']}/{len(TEST_SENTENCES)} ({lang_label}) ──")
        print(f'    말할 문장: "{sent["prompt"]}"')

        # 녹음
        audio, speech_duration = await record_utterance(manager, vad, bus)

        if audio is None or len(audio) < 1600:  # < 0.1초
            print("    [SKIP] 유효한 오디오 없음")
            results.append({
                "id": sent["id"], "lang": sent["lang"],
                "prompt": sent["prompt"], "text": "", "detected_lang": None,
                "rtf": None, "duration": 0, "stt_time": 0,
                "keywords_found": [], "keywords_missed": sent["keywords"],
            })
            continue

        audio_duration = len(audio) / 16_000
        print(f"    오디오 길이: {audio_duration:.2f}초 (발화 구간: {speech_duration:.2f}초)")

        # STT 실행
        t_stt_start = time.monotonic()
        text = await transcriber.transcribe(audio)
        t_stt = time.monotonic() - t_stt_start

        # RTF는 실제 발화 구간 기준으로 계산 (더 정확한 측정)
        rtf_audio = t_stt / audio_duration if audio_duration > 0 else 0
        rtf_speech = t_stt / speech_duration if speech_duration > 0 else rtf_audio
        detected = transcriber.detected_language

        print(f"    STT 결과: \"{text}\"")
        print(f"    언어 감지: {detected}")
        print(f"    처리 시간: {t_stt:.3f}초 (RTF: {rtf_audio:.3f}x 오디오 기준, {rtf_speech:.3f}x 발화 기준)")

        # 키워드 체크
        text_lower = text.lower()
        found = [kw for kw in sent["keywords"] if kw.lower() in text_lower]
        missed = [kw for kw in sent["keywords"] if kw.lower() not in text_lower]

        if missed:
            print(f"    키워드: {len(found)}/{len(sent['keywords'])} (누락: {missed})")
        else:
            print(f"    키워드: {len(found)}/{len(sent['keywords'])} — 모두 포함 ✓")

        # 언어 감지 정확도
        expected_lang = {"ko": "ko", "en": "en", "mixed": None}[sent["lang"]]
        if expected_lang and detected:
            lang_match = detected == expected_lang
            print(f"    언어 감지: {'정확 ✓' if lang_match else f'불일치 ✗ (기대: {expected_lang})'}")

        results.append({
            "id": sent["id"], "lang": sent["lang"],
            "prompt": sent["prompt"], "text": text, "detected_lang": detected,
            "rtf": rtf_audio, "duration": audio_duration,
            "speech_duration": speech_duration, "stt_time": t_stt,
            "keywords_found": found, "keywords_missed": missed,
        })

    await manager.stop()

    # ==================================================================
    # Phase 3: 메모리 최종 확인
    # ==================================================================
    mem_final = get_memory_info()

    # ==================================================================
    # 결과 요약
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"  결과 요약 (model: {model_size})")
    print("=" * 60)

    # RTF 통계
    rtfs = [r["rtf"] for r in results if r["rtf"] is not None]
    if rtfs:
        rtf_mean = float(np.mean(rtfs))
        rtf_max = max(rtfs)
        print(f"\n  RTF: mean={rtf_mean:.3f}x, max={rtf_max:.3f}x (목표: < 0.2x)")
        rtf_ok = rtf_max < 0.2
        print(f"  RTF 판정: {'PASS ✓' if rtf_ok else 'FAIL ✗'}")
    else:
        rtf_ok = False
        print("\n  RTF: 측정 실패")

    # 키워드 정확도
    total_kw = sum(len(r["keywords_found"]) + len(r["keywords_missed"]) for r in results)
    found_kw = sum(len(r["keywords_found"]) for r in results)
    kw_accuracy = found_kw / total_kw * 100 if total_kw > 0 else 0
    kw_ok = kw_accuracy > 90
    print(f"\n  키워드 정확도: {found_kw}/{total_kw} ({kw_accuracy:.1f}%) (목표: > 90%)")
    print(f"  키워드 판정: {'PASS ✓' if kw_ok else 'FAIL ✗'}")

    # 누락된 키워드 목록
    for r in results:
        if r["keywords_missed"]:
            print(f"    문장 {r['id']}: 누락 {r['keywords_missed']}")

    # 언어 감지
    print(f"\n  언어 감지 결과:")
    lang_correct = 0
    lang_total = 0
    for r in results:
        expected = {"ko": "ko", "en": "en", "mixed": None}[r["lang"]]
        detected = r["detected_lang"]
        if expected is not None:
            lang_total += 1
            ok = detected == expected
            if ok:
                lang_correct += 1
            symbol = "✓" if ok else "✗"
            print(f"    문장 {r['id']}: 기대={expected}, 감지={detected} {symbol}")
        else:
            print(f"    문장 {r['id']}: 혼용 → 감지={detected} (참고)")

    lang_ok = lang_correct == lang_total if lang_total > 0 else False
    print(f"  언어 감지 판정: {lang_correct}/{lang_total} {'PASS ✓' if lang_ok else 'FAIL ✗'}")

    # 메모리
    print(f"\n  메모리 (Jetson unified memory):")
    if mem_before["used_mb"] is not None:
        print(f"    로드 전: {mem_before['used_mb']:.0f}MB 사용 / {mem_before['available_mb']:.0f}MB 가용")
    if mem_after["used_mb"] is not None:
        print(f"    로드 후: {mem_after['used_mb']:.0f}MB 사용 / {mem_after['available_mb']:.0f}MB 가용")
    if mem_final["used_mb"] is not None:
        print(f"    테스트 후: {mem_final['used_mb']:.0f}MB 사용 / {mem_final['available_mb']:.0f}MB 가용")
    if mem_before["used_mb"] is not None and mem_after["used_mb"] is not None:
        delta = mem_after["used_mb"] - mem_before["used_mb"]
        print(f"    모델 점유: ~{delta:.0f}MB")

    # 개별 문장 상세
    print(f"\n  ── 개별 문장 상세 ──")
    for r in results:
        rtf_str = f"{r['rtf']:.3f}x" if r["rtf"] is not None else "N/A"
        print(f"    #{r['id']} [{r['lang']}] RTF={rtf_str} "
              f"audio={r['duration']:.1f}s speech={r['speech_duration']:.1f}s "
              f"stt={r['stt_time']:.3f}s lang={r['detected_lang']}")
        print(f"       prompt: \"{r['prompt']}\"")
        print(f"       result: \"{r['text']}\"")

    # 최종 판정
    all_pass = rtf_ok and kw_ok and lang_ok
    print(f"\n  ── 최종 판정 ──")
    print(f"  RTF < 0.2x:       {'PASS ✓' if rtf_ok else 'FAIL ✗'}")
    print(f"  키워드 > 90%:     {'PASS ✓' if kw_ok else 'FAIL ✗'}")
    print(f"  언어 감지 정확:   {'PASS ✓' if lang_ok else 'FAIL ✗'}")
    print(f"\n  최종: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    if model_override and model_override != cfg.get("stt.model_size", "large-v3-turbo"):
        print(f"\n  [참고] '{model_size}' 모델로 파이프라인 검증 완료.")
        print(f"  최종 성능 측정은 SSH 끊고 '{cfg.get('stt.model_size', 'large-v3-turbo')}'로 재실행하세요:")
        print(f"    python tests/test_hw_stt.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW-06: STT 라이브 테스트")
    parser.add_argument(
        "--model", type=str, default=None,
        help="모델 오버라이드 (base, small, medium, large-v3-turbo 등). "
             "미지정 시 config/default.yaml 값 사용.",
    )
    args = parser.parse_args()
    asyncio.run(run_test(model_override=args.model))
