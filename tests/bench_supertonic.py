"""Supertonic TTS 벤치마크 — Jetson Orin Nano.

실행:
    python tests/bench_supertonic.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ---------------------------------------------------------------------------
# 테스트 문장
# ---------------------------------------------------------------------------

SENTENCES = [
    # 한국어 — 짧은/중간/긴
    {"id": 1, "lang": "ko", "text": "안녕하세요, 반갑습니다.", "desc": "KO 짧은 인사"},
    {"id": 2, "lang": "ko", "text": "오늘 날씨가 정말 좋네요. 산책하러 나가고 싶어요.", "desc": "KO 중간"},
    {"id": 3, "lang": "ko", "text": "인공지능은 1950년대 앨런 튜링의 연구에서 시작되었습니다. 이후 다양한 발전을 거쳐 오늘날의 대규모 언어 모델에 이르렀습니다.", "desc": "KO 긴 설명"},
    # 영어 — 짧은/중간/긴
    {"id": 4, "lang": "en", "text": "Hello, nice to meet you.", "desc": "EN short greeting"},
    {"id": 5, "lang": "en", "text": "The weather is beautiful today. I'd love to go for a walk in the park.", "desc": "EN medium"},
    {"id": 6, "lang": "en", "text": "Artificial intelligence began with Alan Turing's research in the 1950s. Since then, it has evolved through many breakthroughs to reach today's large language models.", "desc": "EN long explanation"},
]

VOICES_TO_TEST = ["M1", "F1"]


def main() -> None:
    from supertonic import TTS

    print("\n" + "=" * 65)
    print("Supertonic TTS 벤치마크 — Jetson Orin Nano")
    print("=" * 65)

    print("\n── 초기화 ──")
    t0 = time.monotonic()
    tts = TTS()
    init_time = time.monotonic() - t0
    print(f"  모델: {tts.model_name}")
    print(f"  샘플레이트: {tts.sample_rate} Hz")
    print(f"  음성: {tts.voice_style_names}")
    print(f"  초기화 시간: {init_time:.2f}s")

    # 메모리
    try:
        import subprocess
        result = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=5)
        parts = result.stdout.strip().split("\n")[1].split()
        print(f"  메모리: {parts[2]}MB 사용 / {parts[6]}MB 가용")
    except Exception:
        pass

    results: list[dict] = []

    for voice_name in VOICES_TO_TEST:
        print(f"\n── 음성: {voice_name} ──")
        style = tts.get_voice_style(voice_name)

        for s in SENTENCES:
            print(f"\n  [{s['id']}] {s['desc']} (lang={s['lang']})")
            print(f"      \"{s['text'][:60]}{'...' if len(s['text']) > 60 else ''}\"")

            t_start = time.monotonic()
            wav, dur = tts.synthesize(
                s["text"],
                voice_style=style,
                lang=s["lang"],
            )
            synth_time = time.monotonic() - t_start

            audio = wav[0] if wav.ndim > 1 else wav
            audio_dur = len(audio) / tts.sample_rate
            rtf = synth_time / audio_dur if audio_dur > 0 else 0
            chars_per_sec = len(s["text"]) / synth_time if synth_time > 0 else 0

            print(f"      합성: {synth_time:.3f}s → 오디오: {audio_dur:.3f}s → RTF: {rtf:.3f}x")
            print(f"      글자/초: {chars_per_sec:.1f}")

            results.append({
                "id": s["id"],
                "voice": voice_name,
                "lang": s["lang"],
                "desc": s["desc"],
                "text_len": len(s["text"]),
                "synth_time": synth_time,
                "audio_dur": audio_dur,
                "rtf": rtf,
                "chars_per_sec": chars_per_sec,
            })

    # =================================================================
    # 요약
    # =================================================================
    print("\n" + "=" * 65)
    print("  요약")
    print("=" * 65)

    for voice_name in VOICES_TO_TEST:
        vr = [r for r in results if r["voice"] == voice_name]
        ko = [r for r in vr if r["lang"] == "ko"]
        en = [r for r in vr if r["lang"] == "en"]

        print(f"\n  음성: {voice_name}")
        if ko:
            rtf_ko = np.mean([r["rtf"] for r in ko])
            print(f"    한국어 평균 RTF: {rtf_ko:.3f}x")
        if en:
            rtf_en = np.mean([r["rtf"] for r in en])
            print(f"    영어   평균 RTF: {rtf_en:.3f}x")

    all_rtf = [r["rtf"] for r in results]
    print(f"\n  전체 평균 RTF: {np.mean(all_rtf):.3f}x")
    print(f"  전체 최대 RTF: {max(all_rtf):.3f}x")

    # MeloTTS 비교
    print(f"\n  ── MeloTTS 비교 ──")
    print(f"  MeloTTS KR RTF: ~1.2x (CPU)")
    print(f"  MeloTTS EN RTF: ~2.4x (CPU)")
    ko_rtf = np.mean([r["rtf"] for r in results if r["lang"] == "ko"])
    en_rtf = np.mean([r["rtf"] for r in results if r["lang"] == "en"])
    print(f"  Supertonic KR RTF: {ko_rtf:.3f}x → {1.2/ko_rtf:.1f}배 빠름" if ko_rtf > 0 else "")
    print(f"  Supertonic EN RTF: {en_rtf:.3f}x → {2.4/en_rtf:.1f}배 빠름" if en_rtf > 0 else "")

    print("=" * 65)


if __name__ == "__main__":
    main()
