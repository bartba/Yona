"""HW-06: STT (faster-whisper) 라이브 테스트.

별도 터미널에서 실행:
    python tests/test_hw_stt.py                  # config 기본값 (large-v3-turbo)
    python tests/test_hw_stt.py --model base     # 가벼운 모델 (Remote SSH 환경)
    python tests/test_hw_stt.py --model medium   # 중간 모델

테스트 항목:
    Phase 1. 모델 로드 시간 + 메모리 확인
    Phase 2. Warmup 추론 (CUDA JIT 초기화)
    Phase 3. 한국어/영어/혼용 5개 문장 — 고정 시간 녹음 후 STT
             Enter로 녹음 시작, RECORD_SECONDS 초 카운트다운, 자동 종료
    성공 기준: 핵심 단어 정확도 > 90%, RTF < 0.2x, 언어 감지 정확

모델별 참고:
    base           ~150MB GPU, 한국어 정확도 낮음 — 파이프라인 검증용
    small          ~500MB GPU, 한국어 중간 — 개발 중 테스트
    medium         ~1.5GB GPU, 한국어 양호
    large-v3-turbo ~1.5GB GPU, 한국어 최고 — 최종 성능 측정용 (SSH 끊고 실행)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import Config
from src.events import EventBus
from src.audio import AudioManager


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

TEST_SENTENCES = [
    {"id": 1, "lang": "ko", "prompt": "안녕하세요, 오늘 날씨가 좋네요", "keywords": ["안녕", "날씨"]},
    {"id": 2, "lang": "ko", "prompt": "내일 회의 일정을 알려줘",         "keywords": ["내일", "회의", "일정"]},
    {"id": 3, "lang": "en", "prompt": "What time is the meeting tomorrow?", "keywords": ["time", "meeting", "tomorrow"]},
    {"id": 4, "lang": "en", "prompt": "Tell me about the weather today",    "keywords": ["weather", "today"]},
    {"id": 5, "lang": "mixed", "prompt": "요나야, tell me a joke 하나 해줘", "keywords": ["요나", "joke"]},
]

RECORD_SECONDS = 5  # 문장당 고정 녹음 시간


def get_memory_info() -> dict[str, float | None]:
    """Jetson unified memory 정보 반환 (MB). free 명령 사용."""
    try:
        import subprocess
        result = subprocess.run(
            ["free", "-m"], capture_output=True, text=True, timeout=5,
        )
        parts = result.stdout.strip().split("\n")[1].split()
        return {
            "total_mb": float(parts[1]),
            "used_mb": float(parts[2]),
            "available_mb": float(parts[6]),
        }
    except Exception:
        return {"total_mb": None, "used_mb": None, "available_mb": None}


async def record_fixed(manager: AudioManager, duration: float) -> np.ndarray:
    """고정 시간(duration 초) 동안 마이크 오디오를 수집해서 반환."""
    chunks: list[np.ndarray] = []

    def on_chunk(chunk: np.ndarray) -> None:
        chunks.append(chunk.copy())

    manager.add_input_callback(on_chunk)
    try:
        # 1초 단위 카운트다운
        for remaining in range(int(duration), 0, -1):
            print(f"\r    녹음 중... {remaining:2d}초 남음", end="", flush=True)
            await asyncio.sleep(1.0)
        print(f"\r    녹음 완료 ({duration:.0f}초)          ")
    finally:
        manager.remove_input_callback(on_chunk)

    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


async def run_test(model_override: str | None = None) -> None:
    cfg = Config()
    bus = EventBus()
    manager = AudioManager(cfg)

    model_size   = model_override or cfg.get("stt.model_size", "large-v3-turbo")
    compute_type = cfg.get("stt.compute_type", "float16")
    device       = cfg.get("stt.device", "cuda")
    language     = cfg.get("stt.language", None)

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
    cfg._data["stt"]["model_size"] = model_size
    transcriber = Transcriber(cfg, bus)

    t_load = time.monotonic() - t_load_start
    mem_after = get_memory_info()

    print(f"  모델 로드 시간: {t_load:.1f}초")
    if mem_after["used_mb"] is not None:
        print(f"  로드 후 메모리: {mem_after['used_mb']:.0f}MB 사용 / {mem_after['available_mb']:.0f}MB 가용")
        if mem_before["used_mb"] is not None:
            print(f"  모델 메모리 증가: ~{mem_after['used_mb'] - mem_before['used_mb']:.0f}MB")

    # ==================================================================
    # Phase 2: Warmup (CUDA JIT 초기화)
    # ==================================================================
    print("\n── Phase 2: Warmup 추론 ──")
    print("  CUDA JIT 초기화를 위해 더미 오디오로 추론합니다...")

    dummy_audio = np.zeros(16_000, dtype=np.float32)
    t_warmup = time.monotonic()
    await transcriber.transcribe(dummy_audio)
    print(f"  Warmup 완료: {time.monotonic() - t_warmup:.1f}초")

    # ==================================================================
    # Phase 3: 5개 문장 — 고정 시간 녹음 → STT
    # ==================================================================
    print(f"\n── Phase 3: STT 테스트 ({len(TEST_SENTENCES)}문장, {RECORD_SECONDS}초씩 녹음) ──")
    print("  문장을 보고 Enter를 누르면 녹음이 시작됩니다.")
    print(f"  {RECORD_SECONDS}초 안에 문장을 말하세요.\n")

    await manager.start()
    results: list[dict] = []

    for sent in TEST_SENTENCES:
        lang_label = {"ko": "한국어", "en": "영어", "mixed": "혼용"}[sent["lang"]]
        print(f"  ── 문장 {sent['id']}/{len(TEST_SENTENCES)} ({lang_label}) ──")
        print(f'    말할 문장: "{sent["prompt"]}"')
        input("    준비되면 Enter... ")

        audio = await record_fixed(manager, RECORD_SECONDS)
        audio_duration = len(audio) / 16_000

        # STT
        t_stt_start = time.monotonic()
        text = await transcriber.transcribe(audio)
        t_stt = time.monotonic() - t_stt_start

        rtf = t_stt / audio_duration if audio_duration > 0 else 0.0
        detected = transcriber.detected_language

        print(f"    STT 결과: \"{text}\"")
        print(f"    언어 감지: {detected}  |  처리 시간: {t_stt:.3f}초  |  RTF: {rtf:.3f}x")

        # 키워드 체크
        text_lower = text.lower()
        found  = [kw for kw in sent["keywords"] if kw.lower() in text_lower]
        missed = [kw for kw in sent["keywords"] if kw.lower() not in text_lower]
        if missed:
            print(f"    키워드: {len(found)}/{len(sent['keywords'])} (누락: {missed})")
        else:
            print(f"    키워드: {len(found)}/{len(sent['keywords'])} — 모두 포함 ✓")

        # 언어 감지
        expected_lang = {"ko": "ko", "en": "en", "mixed": None}[sent["lang"]]
        if expected_lang and detected:
            ok = detected == expected_lang
            print(f"    언어 감지: {'정확 ✓' if ok else f'불일치 ✗ (기대: {expected_lang})'}")

        results.append({
            "id": sent["id"], "lang": sent["lang"],
            "prompt": sent["prompt"], "text": text, "detected_lang": detected,
            "rtf": rtf, "duration": audio_duration, "stt_time": t_stt,
            "keywords_found": found, "keywords_missed": missed,
        })
        print()

    await manager.stop()

    mem_final = get_memory_info()

    # ==================================================================
    # 결과 요약
    # ==================================================================
    print("=" * 60)
    print(f"  결과 요약 (model: {model_size})")
    print("=" * 60)

    # RTF
    rtfs = [r["rtf"] for r in results]
    rtf_mean = float(np.mean(rtfs))
    rtf_max  = max(rtfs)
    rtf_ok   = rtf_max < 0.2
    print(f"\n  RTF: mean={rtf_mean:.3f}x, max={rtf_max:.3f}x (목표: < 0.2x)")
    print(f"  RTF 판정: {'PASS ✓' if rtf_ok else 'FAIL ✗'}")

    # 키워드
    total_kw = sum(len(r["keywords_found"]) + len(r["keywords_missed"]) for r in results)
    found_kw = sum(len(r["keywords_found"]) for r in results)
    kw_pct   = found_kw / total_kw * 100 if total_kw > 0 else 0
    kw_ok    = kw_pct > 90
    print(f"\n  키워드 정확도: {found_kw}/{total_kw} ({kw_pct:.1f}%) (목표: > 90%)")
    print(f"  키워드 판정: {'PASS ✓' if kw_ok else 'FAIL ✗'}")
    for r in results:
        if r["keywords_missed"]:
            print(f"    문장 {r['id']}: 누락 {r['keywords_missed']}")

    # 언어 감지
    print(f"\n  언어 감지 결과:")
    lang_correct = lang_total = 0
    for r in results:
        expected = {"ko": "ko", "en": "en", "mixed": None}[r["lang"]]
        detected = r["detected_lang"]
        if expected is not None:
            lang_total += 1
            ok = detected == expected
            if ok:
                lang_correct += 1
            print(f"    문장 {r['id']}: 기대={expected}, 감지={detected} {'✓' if ok else '✗'}")
        else:
            print(f"    문장 {r['id']}: 혼용 → 감지={detected} (참고)")
    lang_ok = (lang_correct == lang_total) if lang_total > 0 else False
    print(f"  언어 감지 판정: {lang_correct}/{lang_total} {'PASS ✓' if lang_ok else 'FAIL ✗'}")

    # 메모리
    print(f"\n  메모리 (Jetson unified memory):")
    if mem_before["used_mb"] is not None:
        print(f"    로드 전:   {mem_before['used_mb']:.0f}MB / {mem_before['available_mb']:.0f}MB 가용")
    if mem_after["used_mb"] is not None:
        print(f"    로드 후:   {mem_after['used_mb']:.0f}MB / {mem_after['available_mb']:.0f}MB 가용")
    if mem_final["used_mb"] is not None:
        print(f"    테스트 후: {mem_final['used_mb']:.0f}MB / {mem_final['available_mb']:.0f}MB 가용")
    if mem_before["used_mb"] and mem_after["used_mb"]:
        print(f"    모델 점유: ~{mem_after['used_mb'] - mem_before['used_mb']:.0f}MB")

    # 개별 상세
    print(f"\n  ── 개별 문장 상세 ──")
    for r in results:
        print(f"    #{r['id']} [{r['lang']}] RTF={r['rtf']:.3f}x  stt={r['stt_time']:.3f}s  lang={r['detected_lang']}")
        print(f"       prompt: \"{r['prompt']}\"")
        print(f"       result: \"{r['text']}\"")

    # 최종 판정
    all_pass = rtf_ok and kw_ok and lang_ok
    print(f"\n  ── 최종 판정 ──")
    print(f"  RTF < 0.2x:       {'PASS ✓' if rtf_ok else 'FAIL ✗'}")
    print(f"  키워드 > 90%:     {'PASS ✓' if kw_ok else 'FAIL ✗'}")
    print(f"  언어 감지 정확:   {'PASS ✓' if lang_ok else 'FAIL ✗'}")
    print(f"\n  최종: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    if model_override:
        print(f"\n  [참고] '{model_size}' 모델로 검증 완료.")
        print(f"  최종 성능 측정은 SSH 끊고 large-v3-turbo로 재실행하세요:")
        print(f"    python tests/test_hw_stt.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW-06: STT 라이브 테스트")
    parser.add_argument(
        "--model", type=str, default=None,
        help="모델 오버라이드 (base, small, medium, large-v3-turbo). 미지정 시 config 값 사용.",
    )
    args = parser.parse_args()
    asyncio.run(run_test(model_override=args.model))
