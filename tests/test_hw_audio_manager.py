"""HW-02: AudioManager + AudioBuffer 라이브 테스트.

별도 터미널에서 실행:
    python tests/test_hw_audio_manager.py

테스트 항목:
    1. AudioManager 마이크 스트림 시작 + AudioBuffer에 3초 녹음
    2. chunk 수 검증 (예상 ~93개, 오차 5% 이내)
    3. AudioBuffer → play_audio 재생 (리샘플 16→48kHz)
    4. 리샘플링 품질 보고 (길이·RMS)
"""

from __future__ import annotations

import asyncio
import sys
import os

# project root를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import Config
from src.audio import AudioManager, AudioBuffer


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

RECORD_SECONDS = 3
INPUT_SR = 16000
CHUNK_SIZE = 512
# 예상 chunk 수: 3s × 16000 / 512 = 93.75
EXPECTED_CHUNKS = RECORD_SECONDS * INPUT_SR / CHUNK_SIZE
TOLERANCE = 0.05  # 5%


async def run_test() -> None:
    cfg = Config()
    manager = AudioManager(cfg)
    buf = AudioBuffer(sample_rate=INPUT_SR, buffer_seconds=RECORD_SECONDS + 1)

    # chunk 카운터
    chunk_count = 0

    def on_audio(chunk: np.ndarray) -> None:
        nonlocal chunk_count
        chunk_count += 1
        buf.push(chunk)

    manager.add_input_callback(on_audio)

    # ------------------------------------------------------------------
    # Test 1: 마이크 스트림 시작 + 녹음
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("HW-02: AudioManager + AudioBuffer 라이브 테스트")
    print("=" * 60)

    input(f"\n  마이크에 대고 말할 준비가 되면 Enter를 누르세요... ")
    print(f"  녹음 중 ({RECORD_SECONDS}초)...")

    await manager.start()
    await asyncio.sleep(RECORD_SECONDS)
    await manager.stop()

    # ------------------------------------------------------------------
    # Test 2: chunk 수 검증
    # ------------------------------------------------------------------
    print(f"\n  ── chunk 검증 ──")
    print(f"  수신 chunk 수: {chunk_count}")
    print(f"  예상 chunk 수: {EXPECTED_CHUNKS:.1f}")
    error_pct = abs(chunk_count - EXPECTED_CHUNKS) / EXPECTED_CHUNKS * 100
    print(f"  오차: {error_pct:.1f}%")

    if error_pct <= TOLERANCE * 100:
        print(f"  [OK] chunk 수 오차 {TOLERANCE*100:.0f}% 이내 ✓")
    else:
        print(f"  [WARN] chunk 수 오차가 {TOLERANCE*100:.0f}%를 초과합니다!")

    # ------------------------------------------------------------------
    # Test 3: AudioBuffer 데이터 확인
    # ------------------------------------------------------------------
    audio = buf.get_all()
    duration = len(audio) / INPUT_SR
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.abs(audio).max())

    print(f"\n  ── AudioBuffer 데이터 ──")
    print(f"  샘플 수: {len(audio)}")
    print(f"  길이: {duration:.2f}s")
    print(f"  RMS: {rms:.4f}")
    print(f"  Peak: {peak:.4f}")

    if peak < 0.01:
        print("  [WARN] Peak가 너무 낮습니다 — 마이크 음소거 확인!")
    if peak > 0.99:
        print("  [WARN] 클리핑 감지 — 마이크 볼륨 확인!")

    # ------------------------------------------------------------------
    # Test 4: play_audio 재생 (리샘플링 16→48kHz)
    # ------------------------------------------------------------------
    print(f"\n  ── 재생 테스트 (16kHz → 48kHz 리샘플) ──")

    # 리샘플링 품질 보고
    output_sr = cfg.get("audio.output_sample_rate", 48000)
    expected_out_samples = int(len(audio) * output_sr / INPUT_SR)
    print(f"  입력: {len(audio)} samples @ {INPUT_SR}Hz")
    print(f"  출력 예상: {expected_out_samples} samples @ {output_sr}Hz")
    print(f"  리샘플 비율: {output_sr / INPUT_SR:.1f}x")

    input("  재생을 시작하려면 Enter를 누르세요... ")
    print("  재생 중...")
    await manager.play_audio(audio, INPUT_SR)
    print("  재생 완료")

    result = input("  소리가 잘 들렸나요? (y/n): ").strip().lower()
    if result == "y":
        print("  [OK] 재생 테스트 통과 ✓")
    else:
        print("  [FAIL] 재생 문제 — 리샘플링 또는 오디오 경로 확인 필요")

    # ------------------------------------------------------------------
    # 결과 요약
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  결과 요약")
    print("=" * 60)
    print(f"  chunk 수: {chunk_count} (예상 {EXPECTED_CHUNKS:.0f}, 오차 {error_pct:.1f}%)")
    print(f"  버퍼 길이: {duration:.2f}s")
    print(f"  Peak/RMS: {peak:.4f} / {rms:.4f}")
    chunk_ok = error_pct <= TOLERANCE * 100
    audio_ok = peak >= 0.01
    print(f"  chunk 검증: {'PASS ✓' if chunk_ok else 'FAIL ✗'}")
    print(f"  오디오 품질: {'PASS ✓' if audio_ok else 'FAIL ✗'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_test())
