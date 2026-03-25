"""HW-03: Silero VAD 라이브 테스트.

별도 터미널에서 실행:
    python tests/test_hw_vad.py

테스트 항목:
    1. 15초간 말하기/멈추기 반복 → SPEECH_STARTED/ENDED 이벤트 확인
    2. speech_prob 분포 (환경 소음 baseline vs 발화 시)
    3. 이벤트 쌍 정확 매칭, false positive 확인
"""

from __future__ import annotations

import asyncio
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

BASELINE_SECONDS = 3     # 환경 소음 측정
LISTEN_SECONDS = 15      # 말하기/멈추기 테스트


async def run_test() -> None:
    cfg = Config()
    bus = EventBus()
    manager = AudioManager(cfg)
    vad = VoiceActivityDetector(cfg, bus)

    # 이벤트 구독
    q_start = bus.subscribe(EventType.SPEECH_STARTED)
    q_end = bus.subscribe(EventType.SPEECH_ENDED)

    # VAD를 마이크 콜백에 등록
    manager.add_input_callback(vad.process_chunk)

    # speech_prob 수집용
    probs: list[float] = []
    prob_times: list[float] = []
    t0 = 0.0

    # 원래 process_chunk를 래핑해서 prob 기록
    original_process = vad.process_chunk

    def instrumented_process(chunk: np.ndarray) -> float:
        prob = original_process(chunk)
        probs.append(prob)
        prob_times.append(time.monotonic() - t0)
        return prob

    # 래핑된 콜백으로 교체
    manager.remove_input_callback(vad.process_chunk)
    manager.add_input_callback(instrumented_process)

    # 이벤트 수집
    events: list[tuple[str, float]] = []

    async def collect_events(q: asyncio.Queue[Event], label: str) -> None:
        while True:
            try:
                evt = await asyncio.wait_for(q.get(), timeout=0.1)
                elapsed = time.monotonic() - t0
                events.append((label, elapsed))
                print(f"  [{elapsed:6.2f}s] {label}")
            except asyncio.TimeoutError:
                pass

    # ==================================================================
    # Phase 1: 환경 소음 baseline
    # ==================================================================
    print("\n" + "=" * 60)
    print("HW-03: Silero VAD 라이브 테스트")
    print("=" * 60)

    print(f"\n── Phase 1: 환경 소음 baseline ({BASELINE_SECONDS}초) ──")
    print("  조용히 계세요...")

    t0 = time.monotonic()
    await manager.start()
    await asyncio.sleep(BASELINE_SECONDS)

    baseline_probs = probs.copy()
    baseline_max = max(baseline_probs) if baseline_probs else 0.0
    baseline_mean = float(np.mean(baseline_probs)) if baseline_probs else 0.0

    print(f"  소음 프레임 수: {len(baseline_probs)}")
    print(f"  speech_prob: mean={baseline_mean:.4f}, max={baseline_max:.4f}")

    if baseline_max >= 0.5:
        print("  [WARN] 환경 소음이 VAD threshold를 넘었습니다!")
    else:
        print("  [OK] 환경 소음 baseline 정상 ✓")

    # ==================================================================
    # Phase 2: 말하기/멈추기 테스트
    # ==================================================================
    print(f"\n── Phase 2: 말하기/멈추기 테스트 ({LISTEN_SECONDS}초) ──")
    print("  2~3초 말하고 → 2~3초 멈추기를 반복하세요.")
    input("  준비되면 Enter를 누르세요... ")

    # 이벤트 수집 태스크 시작
    probs.clear()
    prob_times.clear()
    t0 = time.monotonic()

    task_start = asyncio.create_task(collect_events(q_start, "SPEECH_STARTED"))
    task_end = asyncio.create_task(collect_events(q_end, "SPEECH_ENDED"))

    # 15초 대기
    for remaining in range(LISTEN_SECONDS, 0, -1):
        await asyncio.sleep(1)
        # 5초마다 남은 시간 표시
        if remaining % 5 == 0:
            print(f"  ... {remaining}초 남음")

    task_start.cancel()
    task_end.cancel()

    await manager.stop()

    # ==================================================================
    # 분석
    # ==================================================================
    print(f"\n── 분석 결과 ──")

    # 이벤트 쌍 매칭
    starts = [t for label, t in events if label == "SPEECH_STARTED"]
    ends = [t for label, t in events if label == "SPEECH_ENDED"]

    print(f"\n  SPEECH_STARTED 횟수: {len(starts)}")
    print(f"  SPEECH_ENDED   횟수: {len(ends)}")

    # 이벤트 쌍 매칭 확인
    paired = min(len(starts), len(ends))
    unpaired_starts = len(starts) - paired
    unpaired_ends = max(0, len(ends) - len(starts))

    if unpaired_starts <= 1:
        # 마지막 STARTED가 아직 ENDED 안됐을 수 있음 (정상)
        print(f"  [OK] 이벤트 쌍 매칭 정상 (쌍: {paired}, 미완료: {unpaired_starts}) ✓")
    else:
        print(f"  [WARN] 미매칭 이벤트 다수: starts={len(starts)}, ends={len(ends)}")

    # 각 발화 구간 표시
    if starts and ends:
        print(f"\n  발화 구간:")
        for i in range(paired):
            dur = ends[i] - starts[i]
            gap = starts[i + 1] - ends[i] if i + 1 < len(starts) else 0
            print(f"    #{i+1}: {starts[i]:.2f}s ~ {ends[i]:.2f}s (발화 {dur:.2f}s" +
                  (f", 간격 {gap:.2f}s)" if gap > 0 else ")"))

    # speech_prob 분포
    if probs:
        probs_arr = np.array(probs)
        print(f"\n  speech_prob 분포 (테스트 구간):")
        print(f"    mean: {probs_arr.mean():.4f}")
        print(f"    max:  {probs_arr.max():.4f}")
        print(f"    min:  {probs_arr.min():.4f}")
        print(f"    > 0.5 비율: {(probs_arr > 0.5).sum() / len(probs_arr) * 100:.1f}%")
        print(f"    > 0.7 비율: {(probs_arr > 0.7).sum() / len(probs_arr) * 100:.1f}%")

    # false positive 판정: baseline 중 SPEECH_STARTED가 있었는지
    baseline_events = [t for label, t in events if t < 0]
    fp_count = len(baseline_events)

    # ==================================================================
    # 결과 요약
    # ==================================================================
    print("\n" + "=" * 60)
    print("  결과 요약")
    print("=" * 60)
    print(f"  환경 소음 baseline: max={baseline_max:.4f} ({'OK' if baseline_max < 0.5 else 'WARN'})")
    print(f"  이벤트: STARTED={len(starts)}, ENDED={len(ends)}")
    print(f"  이벤트 쌍: {paired}쌍 매칭" + (f" + {unpaired_starts} 미완료" if unpaired_starts else ""))
    events_ok = len(starts) >= 2 and abs(len(starts) - len(ends)) <= 1
    noise_ok = baseline_max < 0.5
    print(f"  이벤트 매칭: {'PASS ✓' if events_ok else 'FAIL ✗'}")
    print(f"  소음 false positive: {'PASS ✓' if noise_ok else 'FAIL ✗'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_test())
