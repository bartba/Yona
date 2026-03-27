"""HW-14 TTS 디버그 — phrase 단위 TTS 품질 진단.

PhraseAccumulator로 실제 LLM 응답을 분할한 뒤, 각 phrase를
Supertonic TTS로 합성하고 WAV 저장 + 재생하여 품질 확인.

Usage:
    python tests/test_hw_tts_debug.py                    # 전체 테스트
    python tests/test_hw_tts_debug.py --speed 1.0        # speed 변경
    python tests/test_hw_tts_debug.py --steps 10         # diffusion steps 변경
    python tests/test_hw_tts_debug.py --no-play          # 재생 없이 WAV만 저장
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.pipeline import PhraseAccumulator

# ---------------------------------------------------------------------------
# 테스트 데이터 — 실제 LLM 응답 시뮬레이션
# ---------------------------------------------------------------------------

# 실제 LLM 스트리밍 토큰처럼 잘게 나눈 텍스트
_TEST_RESPONSES: dict[str, list[str]] = {
    "ko_short": [
        "안녕", "하세요", "!", " ", "반갑", "습니다", ".",
    ],
    "ko_normal": [
        "네", ",", " ", "좋은", " ", "질문", "이에요", "!", " ",
        "듀크", " ", "엘링턴", "은", " ", "1899", "년에", " ",
        "미국", " ", "워싱턴", "에서", " ", "태어난", " ",
        "전설적인", " ", "재즈", " ", "피아니스트", "예요", ".", " ",
        "그의", " ", "대표곡", "으로는", " ", "\"", "It", " ", "Don't",
        " ", "Mean", " ", "a", " ", "Thing", "\"", "이", " ",
        "있습니다", ".",
    ],
    "ko_newline": [
        "첫", " ", "번째", ",", " ", "Take", " ", "Five", "는", " ",
        "정말", " ", "좋은", " ", "곡이에요", ".", "\n", "\n",
        "두", " ", "번째", "로", " ", "추천", "하는", " ", "곡은", " ",
        "Satin", " ", "Doll", "입니다", ".",
    ],
    "en_normal": [
        "Sure", "!", " ", "Duke", " ", "Elling", "ton", " ", "was",
        " ", "born", " ", "in", " ", "1899", ".", " ", "He", " ",
        "was", " ", "a", " ", "legendary", " ", "jazz", " ",
        "pianist", " ", "and", " ", "band", "leader", ".",
    ],
}

# 직접 문장 단위 테스트 (phrase 분할 없이)
_DIRECT_PHRASES: list[tuple[str, str]] = [
    ("ko", "안녕하세요! 반갑습니다."),
    ("ko", "네, 좋은 질문이에요!"),
    ("ko", "듀크 엘링턴은 1899년에 미국 워싱턴에서 태어난 전설적인 재즈 피아니스트예요."),
    ("ko", "그의 대표곡으로는 \"It Don't Mean a Thing\"이 있습니다."),
    ("en", "Sure! Duke Ellington was born in 1899."),
    ("en", "He was a legendary jazz pianist and bandleader."),
]


def run_phrase_accumulator_test(tokens: list[str], lang: str, min_length: int) -> list[str]:
    """PhraseAccumulator로 토큰 스트림 분할 → phrase 목록 반환."""
    acc = PhraseAccumulator(min_length=min_length)
    phrases: list[str] = []
    for token in tokens:
        for phrase in acc.feed(token):
            phrases.append(phrase)
    remaining = acc.flush()
    if remaining:
        phrases.append(remaining)
    return phrases


def synthesize_and_save(
    tts,
    text: str,
    lang: str,
    speed: float,
    total_steps: int,
    voice_style,
    sr: int,
    out_dir: Path,
    index: int,
    label: str,
) -> tuple[float, float, np.ndarray]:
    """TTS 합성 → WAV 저장 → (synth_time, audio_duration, audio) 반환."""
    t0 = time.monotonic()
    wav, _dur = tts.synthesize(
        text,
        voice_style=voice_style,
        lang=lang,
        speed=speed,
        total_steps=total_steps,
    )
    synth_dur = time.monotonic() - t0

    audio = wav[0] if wav.ndim > 1 else wav
    audio = np.asarray(audio, dtype=np.float32)
    audio_dur = len(audio) / sr

    # WAV 저장
    out_path = out_dir / f"{label}_{index:02d}.wav"
    sf.write(str(out_path), audio, sr)

    return synth_dur, audio_dur, audio


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS phrase-level debug")
    parser.add_argument("--speed", type=float, default=None, help="Override TTS speed")
    parser.add_argument("--steps", type=int, default=None, help="Override diffusion steps")
    parser.add_argument("--no-play", action="store_true", help="Skip playback")
    args = parser.parse_args()

    cfg = Config()
    speed = args.speed or cfg.get("tts.speed", 1.25)
    total_steps = args.steps or cfg.get("tts.total_steps", 5)

    out_dir = Path("data/tts_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== TTS Debug | speed={speed} steps={total_steps} ===\n")

    # Initialize Supertonic
    from supertonic import TTS as SupertonicTTS
    tts = SupertonicTTS()
    sr = tts.sample_rate
    voice_name = cfg.get("tts.voice", "M1")
    voice_style = tts.get_voice_style(voice_name)
    print(f"Supertonic ready | sr={sr} voice={voice_name}\n")

    # Optional: lazy import for playback
    if not args.no_play:
        from src.audio import AudioManager
        import asyncio

    # ---------------------------------------------------------------
    # Test 1: PhraseAccumulator 분할 확인
    # ---------------------------------------------------------------
    print("=" * 60)
    print("TEST 1: PhraseAccumulator 분할 + TTS")
    print("=" * 60)

    for name, tokens in _TEST_RESPONSES.items():
        lang = "en" if name.startswith("en") else "ko"
        min_len = {"ko": 5, "en": 10}.get(lang, 5)
        original = "".join(tokens)
        phrases = run_phrase_accumulator_test(tokens, lang, min_len)

        print(f"\n--- {name} (min_length={min_len}) ---")
        print(f"  원본: {original!r}")
        print(f"  분할: {len(phrases)}개 phrase")

        for i, phrase in enumerate(phrases):
            print(f"  [{i}] {phrase!r} ({len(phrase)} chars)")

            synth_dur, audio_dur, audio = synthesize_and_save(
                tts, phrase, lang, speed, total_steps, voice_style, sr,
                out_dir, i, f"acc_{name}",
            )
            rtf = synth_dur / audio_dur if audio_dur > 0 else 0
            peak = np.abs(audio).max()
            rms = np.sqrt(np.mean(audio ** 2))
            print(f"       synth={synth_dur:.2f}s audio={audio_dur:.2f}s "
                  f"RTF={rtf:.2f}x peak={peak:.3f} rms={rms:.4f}")

            # 이상 탐지
            if peak < 0.01:
                print(f"       ⚠️ SILENT OUTPUT (peak={peak:.4f})")
            if audio_dur < 0.2:
                print(f"       ⚠️ VERY SHORT AUDIO ({audio_dur:.2f}s)")

            if not args.no_play:
                import asyncio
                mgr = AudioManager(cfg)
                asyncio.run(_play(mgr, audio, sr))

    # ---------------------------------------------------------------
    # Test 2: 직접 문장 TTS (phrase 분할 없이)
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("TEST 2: 직접 문장 TTS (분할 없이)")
    print("=" * 60)

    for i, (lang, text) in enumerate(_DIRECT_PHRASES):
        print(f"\n  [{i}] lang={lang} text={text!r}")
        synth_dur, audio_dur, audio = synthesize_and_save(
            tts, text, lang, speed, total_steps, voice_style, sr,
            out_dir, i, "direct",
        )
        rtf = synth_dur / audio_dur if audio_dur > 0 else 0
        peak = np.abs(audio).max()
        print(f"       synth={synth_dur:.2f}s audio={audio_dur:.2f}s "
              f"RTF={rtf:.2f}x peak={peak:.3f}")

        if not args.no_play:
            import asyncio
            mgr = AudioManager(cfg)
            asyncio.run(_play(mgr, audio, sr))

    # ---------------------------------------------------------------
    # Test 3: speed / steps 비교 (첫 문장만)
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("TEST 3: speed / steps 파라미터 비교")
    print("=" * 60)

    test_text = "듀크 엘링턴은 1899년에 미국 워싱턴에서 태어난 전설적인 재즈 피아니스트예요."
    configs = [
        (1.0, 5, "speed1.0_step5"),
        (1.25, 5, "speed1.25_step5"),
        (1.0, 10, "speed1.0_step10"),
        (1.25, 10, "speed1.25_step10"),
    ]

    for spd, steps, label in configs:
        synth_dur, audio_dur, audio = synthesize_and_save(
            tts, test_text, "ko", spd, steps, voice_style, sr,
            out_dir, 0, f"param_{label}",
        )
        rtf = synth_dur / audio_dur if audio_dur > 0 else 0
        print(f"  speed={spd} steps={steps:2d} → "
              f"synth={synth_dur:.2f}s audio={audio_dur:.2f}s RTF={rtf:.2f}x")

        if not args.no_play:
            import asyncio
            print(f"    ▶ Playing {label}...")
            mgr = AudioManager(cfg)
            asyncio.run(_play(mgr, audio, sr))

    print(f"\n✅ WAV 파일 저장 위치: {out_dir.resolve()}")
    print("   → 파일을 직접 들으며 어느 구간에서 품질 저하가 발생하는지 확인하세요.")


async def _play(mgr, audio, sr):
    await mgr.start()
    await mgr.play_audio(audio, sr)
    await mgr.stop()


if __name__ == "__main__":
    main()
