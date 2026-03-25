"""HW-01: 오디오 디바이스 I/O 실전 테스트.

별도 터미널에서 실행:
    python tests/test_hw_audio.py

테스트 항목:
    1. 디바이스 목록 출력
    2. 5초 녹음 → WAV 저장
    3. 녹음 재생
    4. 차임 사운드 재생
    5. ALSA 볼륨 확인/설정
"""

from __future__ import annotations

import subprocess
import sys
import wave

import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

DEVICE = "Poly Sync 20"
INPUT_SR = 16000
OUTPUT_SR = 48000
RECORD_SECONDS = 5
REC_PATH = "/tmp/yona_test_rec.wav"


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """float32 mono audio → 16-bit WAV."""
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """WAV → float32 mono array + sample rate."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if nch == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
    return audio, sr


def resample_stereo(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """mono float32 → resampled stereo float32."""
    if from_sr != to_sr:
        n_out = int(len(audio) * to_sr / from_sr)
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, n_out)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return np.column_stack([audio, audio])


def play_audio(audio: np.ndarray, sr: int) -> None:
    """mono float32 → 48kHz stereo → Poly Sync 20 재생."""
    stereo = resample_stereo(audio, sr, OUTPUT_SR)
    sd.play(stereo, OUTPUT_SR, device=DEVICE)
    sd.wait()


# ---------------------------------------------------------------------------
# 테스트 함수들
# ---------------------------------------------------------------------------

def test_01_list_devices() -> None:
    """디바이스 목록 출력 + Poly Sync 20 확인."""
    print("\n" + "=" * 60)
    print("TEST 01: 디바이스 목록")
    print("=" * 60)

    devices = sd.query_devices()
    print(devices)

    poly_found = False
    for i, d in enumerate(sd.query_devices()):
        if "Poly" in d["name"]:
            poly_found = True
            print(f"\n  [OK] Poly Sync 20 발견: 디바이스 #{i}")
            print(f"       입력 채널: {d['max_input_channels']}")
            print(f"       출력 채널: {d['max_output_channels']}")
            print(f"       기본 SR: {d['default_samplerate']}")

    if not poly_found:
        print("\n  [FAIL] Poly Sync 20을 찾을 수 없습니다!")
        sys.exit(1)


def test_02_record() -> None:
    """5초 녹음 → WAV 저장."""
    print("\n" + "=" * 60)
    print(f"TEST 02: {RECORD_SECONDS}초 녹음")
    print("=" * 60)

    input("  마이크에 대고 말할 준비가 되면 Enter를 누르세요... ")
    print(f"  녹음 중 ({RECORD_SECONDS}초)...")

    audio = sd.rec(
        RECORD_SECONDS * INPUT_SR,
        samplerate=INPUT_SR,
        channels=1,
        device=DEVICE,
        dtype="float32",
    )
    sd.wait()
    audio = audio.flatten()

    print(f"  녹음 완료: {len(audio)} samples ({len(audio)/INPUT_SR:.1f}s)")
    print(f"  Max amplitude: {np.abs(audio).max():.4f}")
    print(f"  RMS: {np.sqrt(np.mean(audio**2)):.4f}")

    if np.abs(audio).max() < 0.01:
        print("  [WARN] 최대 진폭이 너무 낮습니다! 마이크가 음소거되었을 수 있습니다.")

    if np.abs(audio).max() > 0.99:
        print("  [WARN] 클리핑 감지! 마이크 볼륨을 줄이세요.")

    save_wav(REC_PATH, audio, INPUT_SR)
    print(f"  저장: {REC_PATH}")


def test_03_playback() -> None:
    """녹음 파일 재생."""
    print("\n" + "=" * 60)
    print("TEST 03: 녹음 재생")
    print("=" * 60)

    audio, sr = load_wav(REC_PATH)
    print(f"  로드: {len(audio)} samples, sr={sr}")

    input("  재생을 시작하려면 Enter를 누르세요... ")
    print("  재생 중...")
    play_audio(audio, sr)
    print("  재생 완료")

    result = input("  소리가 잘 들렸나요? (y/n): ").strip().lower()
    if result == "y":
        print("  [OK] 재생 테스트 통과")
    else:
        print("  [FAIL] 재생 문제 — ALSA 볼륨 확인 필요")


def test_04_chime() -> None:
    """차임 사운드 재생."""
    print("\n" + "=" * 60)
    print("TEST 04: 차임 사운드 재생")
    print("=" * 60)

    chime_path = "sound/mixkit-magic-marimba-2820.wav"
    try:
        audio, sr = load_wav(chime_path)
        print(f"  차임 로드: {len(audio)} samples, sr={sr}, duration={len(audio)/sr:.2f}s")
    except FileNotFoundError:
        print(f"  [SKIP] 차임 파일 없음: {chime_path}")
        print("  프로그래밍 방식 차임 생성...")
        sr = 24000
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 880 * t) * np.exp(-8 * t)
        print(f"  생성 완료: {len(audio)} samples")

    input("  차임 재생을 시작하려면 Enter를 누르세요... ")
    print("  재생 중...")
    play_audio(audio, sr)
    print("  재생 완료")


def test_05_volume() -> None:
    """ALSA 볼륨 확인 및 설정."""
    print("\n" + "=" * 60)
    print("TEST 05: ALSA 볼륨 확인/설정")
    print("=" * 60)

    # 현재 볼륨 읽기
    try:
        result = subprocess.run(
            ["amixer", "-c", "1", "cget", "numid=5"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout
        print(f"  현재 설정:\n{output}")

        # 현재 값 파싱
        for line in output.splitlines():
            if ": values=" in line:
                current = int(line.split("=")[1].strip())
                print(f"  현재 볼륨: {current}/20 ({current * 5}%)")
                break
    except Exception as e:
        print(f"  [WARN] ALSA 볼륨 읽기 실패: {e}")
        return

    # 볼륨 변경
    new_vol = input("  새 볼륨 (0-20, Enter=변경 없음): ").strip()
    if new_vol:
        try:
            vol = int(new_vol)
            if 0 <= vol <= 20:
                subprocess.run(
                    ["amixer", "-c", "1", "cset", "numid=5", str(vol)],
                    capture_output=True, text=True, timeout=5,
                )
                print(f"  [OK] 볼륨 변경: {vol}/20 ({vol * 5}%)")

                # 변경 후 테스트 톤
                print("  확인 톤 재생 (440Hz, 2초)...")
                t = np.linspace(0, 2, OUTPUT_SR * 2, dtype=np.float32)
                tone = 0.5 * np.sin(2 * np.pi * 440 * t)
                stereo = np.column_stack([tone, tone])
                sd.play(stereo, OUTPUT_SR, device=DEVICE)
                sd.wait()
                print("  재생 완료")
            else:
                print("  [SKIP] 범위 초과 (0-20)")
        except ValueError:
            print("  [SKIP] 잘못된 입력")


# ---------------------------------------------------------------------------
# 메인 — 대화형 메뉴 루프
# ---------------------------------------------------------------------------

TESTS = [
    ("1", "디바이스 목록", test_01_list_devices),
    ("2", "녹음 (5초)", test_02_record),
    ("3", "녹음 재생", test_03_playback),
    ("4", "차임 사운드", test_04_chime),
    ("5", "ALSA 볼륨", test_05_volume),
]


def show_menu() -> None:
    print("\n" + "=" * 60)
    print("  Yona HW-01: 오디오 디바이스 I/O 테스트")
    print("=" * 60)
    for num, desc, _ in TESTS:
        print(f"  {num}. {desc}")
    print(f"  a. 전체 실행")
    print(f"  q. 종료")
    print("-" * 60)


def main() -> None:
    test_map = {num: (desc, fn) for num, desc, fn in TESTS}

    while True:
        show_menu()
        choice = input("선택 (번호/a/q): ").strip().lower()

        if choice == "q":
            print("\n테스트를 종료합니다.")
            break
        elif choice == "a":
            for _, _, fn in TESTS:
                fn()
        elif choice in test_map:
            _, fn = test_map[choice]
            fn()
        else:
            print(f"  잘못된 입력: '{choice}'")


if __name__ == "__main__":
    main()
