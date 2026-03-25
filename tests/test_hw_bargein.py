"""HW-05: VAD Barge-in 모드 테스트.

별도 터미널에서 실행:
    python tests/test_hw_bargein.py

테스트 항목:
    Phase 1: 톤 재생 중 아무 말도 안 함 → false barge-in 0회 확인 (AEC 검증)
    Phase 2: 톤 재생 중 실제 음성 → barge-in 감지 확인
    성공 기준: false barge-in 0회, 실제 음성 시 감지 성공
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import Config
from src.events import EventBus, EventType, Event
from src.audio import AudioManager
from src.vad import VoiceActivityDetector


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

TONE_DURATION = 8.0      # 톤 재생 시간 (초)
TONE_FREQ = 440.0        # 톤 주파수 (Hz, A4)
TONE_AMPLITUDE = 0.5     # 톤 볼륨 (0.0~1.0)
TONE_SAMPLE_RATE = 24_000  # AudioManager가 output_rate로 리샘플링


def generate_tone(duration: float, freq: float, amplitude: float, sr: int) -> np.ndarray:
    """Generate a sine-wave tone."""
    n = int(sr * duration)
    t = np.linspace(0.0, duration, n, endpoint=False)
    return (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


async def run_test() -> None:
    cfg = Config()
    bus = EventBus()
    manager = AudioManager(cfg)
    vad = VoiceActivityDetector(cfg, bus)

    # barge-in 이벤트 구독
    q_bargein = bus.subscribe(EventType.BARGE_IN_DETECTED)

    # VAD를 마이크 콜백에 등록
    manager.add_input_callback(vad.process_chunk)

    # speech_prob 수집용 — 원래 process_chunk를 래핑
    probs: list[float] = []
    prob_times: list[float] = []
    t0 = 0.0

    original_process = vad.process_chunk

    def instrumented_process(chunk: np.ndarray) -> float:
        prob = original_process(chunk)
        probs.append(prob)
        prob_times.append(time.monotonic() - t0)
        return prob

    manager.remove_input_callback(vad.process_chunk)
    manager.add_input_callback(instrumented_process)

    # barge-in 이벤트 수집
    bargein_events: list[float] = []

    async def collect_bargein(q: asyncio.Queue[Event]) -> None:
        while True:
            try:
                await asyncio.wait_for(q.get(), timeout=0.1)
                elapsed = time.monotonic() - t0
                bargein_events.append(elapsed)
                print(f"  [{elapsed:6.2f}s] BARGE_IN_DETECTED!")
            except asyncio.TimeoutError:
                pass

    # 톤 생성
    tone = generate_tone(TONE_DURATION, TONE_FREQ, TONE_AMPLITUDE, TONE_SAMPLE_RATE)

    print("\n" + "=" * 60)
    print("HW-05: VAD Barge-in 모드 테스트")
    print("=" * 60)
    print(f"  톤: {TONE_FREQ}Hz, {TONE_DURATION}초, amplitude={TONE_AMPLITUDE}")
    print(f"  barge_in_threshold: {vad._barge_in_threshold}")

    # ==================================================================
    # Phase 1: 톤 재생 + 무음 → false barge-in 확인
    # ==================================================================
    print(f"\n── Phase 1: AEC 검증 (톤 재생 중 무음, {TONE_DURATION}초) ──")
    print("  스피커에서 톤이 나옵니다. 아무 말도 하지 마세요!")
    input("  준비되면 Enter를 누르세요... ")

    await manager.start()
    vad.set_barge_in_mode(True)
    vad.reset()

    probs.clear()
    prob_times.clear()
    bargein_events.clear()
    t0 = time.monotonic()

    collector = asyncio.create_task(collect_bargein(q_bargein))

    # 톤 재생 (play_audio는 blocking=True → 톤이 끝날 때까지 대기)
    await manager.play_audio(tone, TONE_SAMPLE_RATE)

    # 재생 끝난 후 잠시 대기 (잔향 확인)
    await asyncio.sleep(1.0)

    collector.cancel()

    phase1_probs = probs.copy()
    phase1_bargein = bargein_events.copy()
    phase1_max = max(phase1_probs) if phase1_probs else 0.0
    phase1_mean = float(np.mean(phase1_probs)) if phase1_probs else 0.0

    print(f"\n  Phase 1 결과:")
    print(f"    프레임 수: {len(phase1_probs)}")
    print(f"    speech_prob: mean={phase1_mean:.4f}, max={phase1_max:.4f}")
    print(f"    false barge-in 횟수: {len(phase1_bargein)}")

    if phase1_probs:
        arr = np.array(phase1_probs)
        print(f"    > 0.5 비율: {(arr > 0.5).sum() / len(arr) * 100:.1f}%")
        print(f"    > 0.7 비율: {(arr > 0.7).sum() / len(arr) * 100:.1f}%")

    if len(phase1_bargein) == 0:
        print("    [OK] AEC 정상 — false barge-in 0회 ✓")
    else:
        print(f"    [FAIL] false barge-in {len(phase1_bargein)}회 발생!")
        for i, t in enumerate(phase1_bargein):
            print(f"      #{i+1}: {t:.2f}s")

    # ==================================================================
    # Phase 2: 톤 재생 + 실제 음성 → barge-in 감지
    # ==================================================================
    print(f"\n── Phase 2: 실제 barge-in 테스트 ({TONE_DURATION}초) ──")
    print("  톤이 나오면 2~3초 후에 크게 말해주세요 (예: '요나야')")
    input("  준비되면 Enter를 누르세요... ")

    vad.set_barge_in_mode(True)
    vad.reset()

    probs.clear()
    prob_times.clear()
    bargein_events.clear()
    t0 = time.monotonic()

    collector = asyncio.create_task(collect_bargein(q_bargein))

    # 톤 재생
    print("  [톤 재생 시작] 2~3초 후에 말하세요!")
    await manager.play_audio(tone, TONE_SAMPLE_RATE)
    await asyncio.sleep(1.0)

    collector.cancel()

    phase2_probs = probs.copy()
    phase2_bargein = bargein_events.copy()
    phase2_max = max(phase2_probs) if phase2_probs else 0.0
    phase2_mean = float(np.mean(phase2_probs)) if phase2_probs else 0.0

    print(f"\n  Phase 2 결과:")
    print(f"    프레임 수: {len(phase2_probs)}")
    print(f"    speech_prob: mean={phase2_mean:.4f}, max={phase2_max:.4f}")
    print(f"    barge-in 감지 횟수: {len(phase2_bargein)}")

    if phase2_probs:
        arr = np.array(phase2_probs)
        print(f"    > 0.5 비율: {(arr > 0.5).sum() / len(arr) * 100:.1f}%")
        print(f"    > 0.7 비율: {(arr > 0.7).sum() / len(arr) * 100:.1f}%")

    if phase2_bargein:
        print("    [OK] barge-in 감지 성공 ✓")
        for i, t in enumerate(phase2_bargein):
            print(f"      #{i+1}: {t:.2f}s")
    else:
        print("    [FAIL] barge-in 감지 실패 — 음성이 threshold를 넘지 못함")

    await manager.stop()

    # ==================================================================
    # 결과 요약
    # ==================================================================
    phase1_ok = len(phase1_bargein) == 0
    phase2_ok = len(phase2_bargein) >= 1

    print("\n" + "=" * 60)
    print("  결과 요약")
    print("=" * 60)
    print(f"  Phase 1 (AEC — false barge-in):  {'PASS ✓' if phase1_ok else 'FAIL ✗'}")
    print(f"    speech_prob max={phase1_max:.4f}, false barge-in={len(phase1_bargein)}회")
    print(f"  Phase 2 (실제 barge-in 감지):    {'PASS ✓' if phase2_ok else 'FAIL ✗'}")
    print(f"    speech_prob max={phase2_max:.4f}, barge-in={len(phase2_bargein)}회")
    print(f"\n  최종: {'PASS ✓' if (phase1_ok and phase2_ok) else 'FAIL ✗'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_test())
