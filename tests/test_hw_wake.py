"""HW-04: Wake Word 라이브 테스트.

별도 터미널에서 실행:
    python tests/test_hw_wake.py

테스트 항목:
    Phase 1: 환경 소음 baseline (10초) — false positive 확인
    Phase 2: wake word 반복 (60초) — TPR 측정, cooldown 확인
    Phase 3: 유사 발음 + 무관 대화 (30초) — FPR 측정
    성공 기준: TPR ≥ 80%, FPR = 0, cooldown 정상 (≥ 2.0초 간격)
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
from src.wake import WakeWordDetector


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

BASELINE_SECONDS = 10    # 환경 소음 baseline
WAKE_SECONDS = 60        # "Alexa" 반복 테스트
FALSE_POS_SECONDS = 30   # 유사 발음 + 무관 대화 테스트


async def run_test() -> None:
    cfg = Config()
    bus = EventBus()
    manager = AudioManager(cfg)
    wake = WakeWordDetector(cfg, bus)

    # config에서 wake word 정보 가져오기
    active = cfg.get("wake_word.active_models", [])
    wake_word_name = active[0] if active else "alexa"
    wake_phrase = cfg.get("wake_word.wake_phrase", wake_word_name)

    print(f"\n  로드된 모델: {wake.model_names}")
    print(f"  active 모델: {active}")
    print(f"  threshold: {cfg.get('wake_word.threshold', 0.5)}")
    print(f"  patience:  {cfg.get('wake_word.patience', 3)}")
    print(f"  cooldown:  {cfg.get('wake_word.cooldown_seconds', 2.0)}초")

    # 이벤트 구독
    q_wake = bus.subscribe(EventType.WAKE_WORD_DETECTED)

    # score 수집을 위한 래핑 — active 모델의 score만 기록
    scores: list[float] = []
    score_times: list[float] = []
    t0 = 0.0

    original_process = wake.process_chunk

    def instrumented_process(chunk: np.ndarray) -> None:
        result = original_process(chunk)
        if isinstance(result, dict):
            score = float(result.get(wake_word_name, 0.0))
            scores.append(score)
            score_times.append(time.monotonic() - t0)

    manager.add_input_callback(instrumented_process)

    # 감지 이벤트 수집
    detections: list[float] = []  # 감지 시각 목록

    async def collect_detections(q: asyncio.Queue[Event]) -> None:
        while True:
            try:
                evt = await asyncio.wait_for(q.get(), timeout=0.1)
                elapsed = time.monotonic() - t0
                detections.append(elapsed)
                model_name = evt.data if evt.data else "unknown"
                print(f"  [{elapsed:6.2f}s] WAKE_WORD_DETECTED — {model_name}")
            except asyncio.TimeoutError:
                pass

    # ==================================================================
    # Phase 1: 환경 소음 baseline — false positive 확인
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"HW-04: Wake Word 라이브 테스트 (wake word: \"{wake_phrase}\")")
    print("=" * 60)

    print(f"\n── Phase 1: 환경 소음 baseline ({BASELINE_SECONDS}초) ──")
    print("  조용히 계세요... (아무 말도 하지 마세요)")

    t0 = time.monotonic()
    await manager.start()

    task_detect = asyncio.create_task(collect_detections(q_wake))
    await asyncio.sleep(BASELINE_SECONDS)

    baseline_detections = len(detections)
    baseline_scores = scores.copy()
    baseline_max = max(baseline_scores) if baseline_scores else 0.0
    baseline_mean = float(np.mean(baseline_scores)) if baseline_scores else 0.0

    print(f"\n  프레임 수: {len(baseline_scores)}")
    print(f"  score: mean={baseline_mean:.4f}, max={baseline_max:.4f}")
    print(f"  false positive: {baseline_detections}회")

    if baseline_detections == 0:
        print("  [OK] 환경 소음 baseline 정상 ✓")
    else:
        print("  [WARN] 환경 소음에서 false positive 발생!")

    # ==================================================================
    # Phase 2: Wake Word 반복 — TPR 측정
    # ==================================================================
    print(f"\n── Phase 2: Wake Word 감지 테스트 ({WAKE_SECONDS}초) ──")
    print(f'  "{wake_phrase}"를 반복해서 말하세요.')
    print("  3~4초 간격으로 자연스럽게 말하세요.")
    attempts = int(input("  몇 번 시도할 예정인가요? (예: 10): ") or "10")
    input("  준비되면 Enter를 누르세요... ")

    # 초기화
    detections.clear()
    scores.clear()
    score_times.clear()
    wake.reset()
    t0 = time.monotonic()

    # 타이머
    for remaining in range(WAKE_SECONDS, 0, -1):
        await asyncio.sleep(1)
        if remaining % 10 == 0:
            print(f"  ... {remaining}초 남음 (감지: {len(detections)}회)")

    phase2_detections = detections.copy()
    phase2_scores = scores.copy()
    tpr = len(phase2_detections) / attempts * 100 if attempts > 0 else 0

    print(f"\n  시도 횟수: {attempts}")
    print(f"  감지 횟수: {len(phase2_detections)}")
    print(f"  TPR: {tpr:.1f}%")

    # Cooldown 검증
    cooldown_ok = True
    cooldown_violations: list[float] = []
    cooldown_cfg = cfg.get("wake_word.cooldown_seconds", 2.0)
    for i in range(1, len(phase2_detections)):
        gap = phase2_detections[i] - phase2_detections[i - 1]
        if gap < cooldown_cfg:
            cooldown_ok = False
            cooldown_violations.append(gap)

    if phase2_detections:
        gaps = [phase2_detections[i] - phase2_detections[i - 1]
                for i in range(1, len(phase2_detections))]
        if gaps:
            print(f"  감지 간격: min={min(gaps):.2f}s, mean={np.mean(gaps):.2f}s")
        print(f"  cooldown 준수: {'PASS ✓' if cooldown_ok else 'FAIL ✗'}")
        if cooldown_violations:
            print(f"    위반 간격: {[f'{g:.2f}s' for g in cooldown_violations]}")

    # score 분포
    if phase2_scores:
        arr = np.array(phase2_scores)
        print(f"\n  score 분포:")
        print(f"    mean: {arr.mean():.4f}")
        print(f"    max:  {arr.max():.4f}")
        print(f"    > 0.3 비율: {(arr > 0.3).sum() / len(arr) * 100:.1f}%")
        print(f"    > 0.5 비율: {(arr > 0.5).sum() / len(arr) * 100:.1f}%")

    # ==================================================================
    # Phase 3: 유사 발음 + 무관 대화 — FPR 측정
    # ==================================================================
    print(f"\n── Phase 3: False Positive 테스트 ({FALSE_POS_SECONDS}초) ──")
    print('  유사 발음("Alexis", "Alex")과')
    print('  무관 대화(아무 말이나)를 하세요.')
    print(f'  ⚠️  "{wake_phrase}"는 말하지 마세요!')
    input("  준비되면 Enter를 누르세요... ")

    detections.clear()
    scores.clear()
    score_times.clear()
    wake.reset()
    t0 = time.monotonic()

    for remaining in range(FALSE_POS_SECONDS, 0, -1):
        await asyncio.sleep(1)
        if remaining % 10 == 0:
            print(f"  ... {remaining}초 남음 (false positive: {len(detections)}회)")

    phase3_detections = len(detections)
    phase3_scores = scores.copy()

    print(f"\n  false positive: {phase3_detections}회")
    if phase3_scores:
        arr3 = np.array(phase3_scores)
        print(f"  score 분포: mean={arr3.mean():.4f}, max={arr3.max():.4f}")

    if phase3_detections == 0:
        print("  [OK] False positive 없음 ✓")
    else:
        print("  [WARN] False positive 발생!")

    # ==================================================================
    # 정리
    # ==================================================================
    task_detect.cancel()
    await manager.stop()

    # ==================================================================
    # 결과 요약
    # ==================================================================
    print("\n" + "=" * 60)
    print("  HW-04 결과 요약")
    print("=" * 60)

    p1_pass = baseline_detections == 0
    p2_pass = tpr >= 80.0
    p3_pass = phase3_detections == 0
    cd_pass = cooldown_ok

    print(f"  Wake Word: \"{wake_phrase}\"")
    print(f"  Phase 1 (환경 소음 FP):   {'PASS ✓' if p1_pass else 'FAIL ✗'}  — {baseline_detections}회")
    print(f"  Phase 2 (TPR):            {'PASS ✓' if p2_pass else 'FAIL ✗'}  — {tpr:.1f}% ({len(phase2_detections)}/{attempts})")
    print(f"  Phase 2 (Cooldown):       {'PASS ✓' if cd_pass else 'FAIL ✗'}  — 간격 ≥ {cooldown_cfg}s")
    print(f"  Phase 3 (유사발음 FPR):   {'PASS ✓' if p3_pass else 'FAIL ✗'}  — {phase3_detections}회")

    all_pass = p1_pass and p2_pass and p3_pass and cd_pass
    print(f"\n  전체: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_test())
