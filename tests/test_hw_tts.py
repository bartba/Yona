"""HW-08: MeloTTS 라이브 테스트.

별도 터미널에서 실행:
    python tests/test_hw_tts.py               # 기본 (config 값 사용)
    python tests/test_hw_tts.py --device cpu  # CPU 강제
    python tests/test_hw_tts.py --device cuda # GPU 강제
    python tests/test_hw_tts.py --compare     # CPU vs GPU 비교 (재생 없음)

테스트 항목:
    Phase 1. 모델 로드 시간 + 메모리 확인
    Phase 2. [--compare 모드] CPU vs GPU 합성 속도 비교 (재생 없음)
    Phase 3. 한국어/영어/혼용 5개 문장 합성 → 재생
             - 합성 RTF (목표: < 1.0x)
             - 언어 전환 시간 (KR → EN → KR, 목표: < 3초)
             - 발음 품질 (청취 후 주관 판정)
    Phase 4. 스타일 파라미터 시연 (sdp_ratio / noise_scale / noise_scale_w)
    Phase 5. EN 스피커 악센트 시연 (US / BR / India / AU)

성공 기준: RTF < 1.0x, 발음 이해 가능, 언어 전환 < 3초

MeloTTS 스피커 정보:
    KR: "KR" 단일 스피커 (중성음)  — gender 구분 없음
    EN: "EN-US" (미국) / "EN-BR" (영국) / "EN_INDIA" (인도) / "EN-AU" (호주) / "EN-Default"
        — accent 변종이며 gender 구분은 없음

스타일 파라미터 (config tts.melo_* 로 영구 설정 가능):
    melo_sdp_ratio   : 0.0(기계적) ~ 1.0(자연스러운 타이밍), 기본 0.2
    melo_noise_scale : 0.0(단조) ~ 1.0+(표현력), 기본 0.6
    melo_noise_scale_w: 0.0(고정 리듬) ~ 1.0+(자유 리듬), 기본 0.8
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
from src.audio import AudioManager
from src.tts import MeloSynthesizer


# ---------------------------------------------------------------------------
# 테스트 문장
# ---------------------------------------------------------------------------

TEST_SENTENCES = [
    {
        "id": 1,
        "lang": "KR",
        "text": "안녕하세요! 저는 요나입니다. 오늘 하루도 좋은 하루 되세요.",
        "desc": "한국어 인사",
    },
    {
        "id": 2,
        "lang": "KR",
        "text": "내일 오전 열 시에 회의가 있습니다. 준비 잘 하셨나요?",
        "desc": "한국어 일정 안내",
    },
    {
        "id": 3,
        "lang": "EN",
        "text": "Hello! I'm Samsung Gauss, your voice assistant. How can I help you today?",
        "desc": "영어 인사",
    },
    {
        "id": 4,
        "lang": "EN",
        "text": "The weather forecast shows sunny skies with a high of twenty five degrees.",
        "desc": "영어 날씨 안내",
    },
    {
        "id": 5,
        "lang": "KR",
        "text": "영어로 대화하다가 다시 한국어로 돌아왔습니다. 자연스럽게 들리시나요?",
        "desc": "언어 전환 후 한국어 복귀",
    },
]

BENCHMARK_TEXT_KR = "안녕하세요. 오늘 날씨가 좋네요. 회의 준비는 되셨나요?"
BENCHMARK_TEXT_EN = "Hello there. The weather looks great today. Are you ready for the meeting?"

STYLE_PRESETS = [
    {"name": "기본 (default)",    "sdp": 0.2, "noise": 0.6, "noise_w": 0.8},
    {"name": "표현력 강화",       "sdp": 0.5, "noise": 0.9, "noise_w": 1.0},
    {"name": "차분·안정",         "sdp": 0.1, "noise": 0.3, "noise_w": 0.4},
    {"name": "자유로운 리듬",     "sdp": 0.7, "noise": 0.6, "noise_w": 1.2},
]

EN_SPEAKERS = ["EN-US", "EN-BR", "EN_INDIA", "EN-AU", "EN-Default"]


def get_memory_info() -> dict[str, float | None]:
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


def make_synth(cfg: Config, device: str, language: str = "KR",
               speaker_key: str | None = None,
               sdp: float | None = None,
               noise: float | None = None,
               noise_w: float | None = None) -> MeloSynthesizer:
    """Config를 임시 변경해서 MeloSynthesizer 생성."""
    cfg._data["tts"]["melo_device"]        = device
    cfg._data["tts"]["melo_language"]      = language
    if speaker_key is not None:
        cfg._data["tts"]["melo_speaker"]   = speaker_key
    if sdp is not None:
        cfg._data["tts"]["melo_sdp_ratio"] = sdp
    if noise is not None:
        cfg._data["tts"]["melo_noise_scale"] = noise
    if noise_w is not None:
        cfg._data["tts"]["melo_noise_scale_w"] = noise_w
    return MeloSynthesizer(cfg)


async def benchmark_synth(synth: MeloSynthesizer, text: str, n: int = 3) -> tuple[float, float, float]:
    """n회 합성 후 (mean_rtf, min_rtf, max_rtf) 반환."""
    rtfs: list[float] = []
    audio, sr = await synth.synthesize(text)  # warmup
    for _ in range(n):
        t0 = time.monotonic()
        audio, sr = await synth.synthesize(text)
        elapsed = time.monotonic() - t0
        dur = len(audio) / sr
        rtfs.append(elapsed / dur if dur > 0 else 0.0)
    return float(np.mean(rtfs)), min(rtfs), max(rtfs)


async def run_compare(cfg: Config) -> None:
    """Phase 2: CPU KR/EN 합성 속도 비교 (재생 없음).

    CUDA는 Jetson 8GB에서 STT(~2GB)와 공존 불가 — BERT가 TTS와
    동일 디바이스에 로드되어 CUBLAS_STATUS_ALLOC_FAILED 발생.
    개별 CUDA 테스트는 ``--device cuda``로 가능.
    """
    print("\n── Phase 2: CPU KR/EN 벤치마크 비교 ──")
    print("  (재생 없음 — 합성 RTF만 측정, 각 3회 평균)")
    print("  [참고] CUDA TTS는 Jetson 8GB에서 STT와 공존 불가 — CPU 전용 비교\n")

    rows: list[dict] = []

    for lang, text in [("KR", BENCHMARK_TEXT_KR), ("EN", BENCHMARK_TEXT_EN)]:
        mem = get_memory_info()
        mem_str = f"  가용={mem['available_mb']:.0f}MB" if mem["available_mb"] else ""
        print(f"  [CPU / {lang}] 모델 로드 중...{mem_str}", end="", flush=True)
        t_load = time.monotonic()
        try:
            synth = make_synth(cfg, device="cpu", language=lang)
        except Exception as exc:
            print(f" 실패: {exc}")
            rows.append({"lang": lang, "load": None,
                          "mean_rtf": None, "error": str(exc)})
            continue
        load_time = time.monotonic() - t_load
        print(f" {load_time:.1f}초 (sr={synth._sample_rate}Hz)")

        print(f"  [CPU / {lang}] 합성 중 (3회)...", end="", flush=True)
        try:
            mean_rtf, min_rtf, max_rtf = await benchmark_synth(synth, text, n=3)
        except Exception as exc:
            print(f" 실패: {exc}")
            rows.append({"lang": lang, "load": load_time,
                          "mean_rtf": None, "error": str(exc)})
            await synth.close()
            del synth
            continue

        await synth.close()
        del synth
        print(f" RTF mean={mean_rtf:.3f}x  min={min_rtf:.3f}x  max={max_rtf:.3f}x")
        rows.append({
            "lang": lang, "load": load_time,
            "mean_rtf": mean_rtf, "min_rtf": min_rtf, "max_rtf": max_rtf,
            "error": None,
        })

    print("\n  ── 비교 요약 ──")
    print(f"  {'언어':<5} {'로드(s)':<9} {'RTF mean':<11} {'RTF min':<9} {'RTF max':<9} {'판정'}")
    print(f"  {'-'*55}")
    for r in rows:
        if r["error"]:
            print(f"  {r['lang']:<5} ERROR: {r['error']}")
        else:
            ok = "PASS ✓" if r["mean_rtf"] < 1.0 else "FAIL ✗"
            print(
                f"  {r['lang']:<5} {r['load']:<9.1f} "
                f"{r['mean_rtf']:<11.3f} {r['min_rtf']:<9.3f} {r['max_rtf']:<9.3f} {ok}"
            )

    ok_rows = [r for r in rows if r["error"] is None]
    if ok_rows:
        best = min(ok_rows, key=lambda r: r["mean_rtf"])
        print(f"\n  최적 언어: {best['lang']} (RTF mean={best['mean_rtf']:.3f}x)")
        all_pass = all(r["mean_rtf"] < 1.0 for r in ok_rows)
        if not all_pass:
            print("  [참고] RTF > 1.0x — 스트리밍 파이프라인에서 phrase간 짧은 대기 발생 가능")
            print("         phrase 길이를 짧게 분할하면 체감 지연을 줄일 수 있음")


async def run_test(device_override: str | None = None, compare_mode: bool = False) -> None:
    cfg = Config()
    manager = AudioManager(cfg)

    if device_override:
        cfg._data["tts"]["melo_device"] = device_override

    melo_language = cfg.get("tts.melo_language", "KR")
    melo_device   = cfg.get("tts.melo_device", "cpu")
    melo_speed    = cfg.get("tts.melo_speed", 1.0)

    print("\n" + "=" * 65)
    print("HW-08: MeloTTS 라이브 테스트")
    print("=" * 65)
    print(f"  초기 언어: {melo_language}, 디바이스: {melo_device}, 속도: {melo_speed}")
    print(f"  [참고] MeloTTS는 gender 구분 없음 — EN은 5가지 accent 선택 가능")

    # ==================================================================
    # Phase 1: 모델 로드 + 메모리
    # ==================================================================
    print("\n── Phase 1: 모델 로드 ──")

    mem_before = get_memory_info()
    if mem_before["used_mb"] is not None:
        print(f"  로드 전 메모리: {mem_before['used_mb']:.0f}MB 사용 / {mem_before['available_mb']:.0f}MB 가용")

    print("  MeloTTS 모델 로딩 중...")
    t_load_start = time.monotonic()
    synth = make_synth(cfg, device=melo_device, language=melo_language)
    t_load = time.monotonic() - t_load_start

    mem_after = get_memory_info()
    print(f"  모델 로드 시간: {t_load:.1f}초")
    print(f"  실제 샘플레이트: {synth._sample_rate} Hz")
    if mem_after["used_mb"] is not None:
        print(f"  로드 후 메모리: {mem_after['used_mb']:.0f}MB 사용 / {mem_after['available_mb']:.0f}MB 가용")
        if mem_before["used_mb"] is not None:
            print(f"  메모리 증가: ~{mem_after['used_mb'] - mem_before['used_mb']:.0f}MB")

    # ==================================================================
    # Phase 2: CPU vs GPU 비교 (--compare 모드)
    # ==================================================================
    if compare_mode:
        await synth.close()
        del synth
        await run_compare(cfg)
        return

    # ==================================================================
    # Phase 3: 합성 + 재생
    # ==================================================================
    print(f"\n── Phase 3: 합성 + 재생 ({len(TEST_SENTENCES)}개 문장) ──\n")

    await manager.start()
    results: list[dict] = []
    current_lang = melo_language

    for sent in TEST_SENTENCES:
        print(f"  ── 문장 {sent['id']}/{len(TEST_SENTENCES)}: {sent['desc']} ──")
        print(f'  텍스트: "{sent["text"]}"')

        switch_time: float | None = None
        if sent["lang"] != current_lang:
            print(f"  언어 전환: {current_lang} → {sent['lang']}")
            t_switch_start = time.monotonic()
            await synth.set_language(sent["lang"])
            switch_time = time.monotonic() - t_switch_start
            current_lang = sent["lang"]
            print(f"  전환 시간: {switch_time:.2f}초")

        t_synth_start = time.monotonic()
        try:
            audio, sr = await synth.synthesize(sent["text"])
        except Exception as exc:
            print(f"  [오류] 합성 실패: {exc}\n")
            results.append({
                "id": sent["id"], "lang": sent["lang"],
                "rtf": None, "switch_time": switch_time,
                "audio_duration": None, "synth_time": None,
                "error": str(exc),
            })
            continue

        synth_time = time.monotonic() - t_synth_start
        audio_duration = len(audio) / sr
        rtf = synth_time / audio_duration if audio_duration > 0 else 0.0

        print(f"  합성 완료: {audio_duration:.2f}초 오디오 / {synth_time:.2f}초 소요 (RTF={rtf:.3f}x, sr={sr}Hz)")
        print(f"  재생 중...", end="", flush=True)
        await manager.play_audio(audio, sr)
        print(f" 완료")

        quality = input(f"  발음 품질 (1=이해어려움 / 2=보통 / 3=자연스러움) [Enter=3]: ").strip()
        quality_score = int(quality) if quality in ("1", "2", "3") else 3

        results.append({
            "id": sent["id"], "lang": sent["lang"], "desc": sent["desc"],
            "rtf": rtf, "switch_time": switch_time,
            "audio_duration": audio_duration, "synth_time": synth_time,
            "quality_score": quality_score, "error": None,
        })
        print()

    # ==================================================================
    # Phase 4: 스타일 파라미터 시연
    # ==================================================================
    print("\n── Phase 4: 스타일 파라미터 시연 ──")
    print('  동일 문장을 4가지 스타일로 합성·재생합니다.')
    print(f'  문장: "{BENCHMARK_TEXT_KR}"\n')

    await synth.set_language("KR")
    style_results: list[dict] = []

    for preset in STYLE_PRESETS:
        synth._sdp_ratio      = preset["sdp"]
        synth._noise_scale    = preset["noise"]
        synth._noise_scale_w  = preset["noise_w"]

        print(f"  [{preset['name']}] sdp={preset['sdp']}  noise={preset['noise']}  noise_w={preset['noise_w']}")
        t0 = time.monotonic()
        audio, sr = await synth.synthesize(BENCHMARK_TEXT_KR)
        synth_time = time.monotonic() - t0
        audio_duration = len(audio) / sr
        rtf = synth_time / audio_duration if audio_duration > 0 else 0.0

        print(f"  재생 중 (RTF={rtf:.3f}x)...", end="", flush=True)
        await manager.play_audio(audio, sr)
        print(" 완료")

        quality = input(f"  품질 (1/2/3) [Enter=3]: ").strip()
        quality_score = int(quality) if quality in ("1", "2", "3") else 3
        style_results.append({**preset, "rtf": rtf, "quality": quality_score})
        print()

    # ==================================================================
    # Phase 5: EN 악센트 시연
    # ==================================================================
    print("\n── Phase 5: EN 악센트(스피커) 시연 ──")
    print('  MeloTTS EN은 gender 없음 — 5가지 accent 변종 제공')
    print(f'  문장: "{BENCHMARK_TEXT_EN}"\n')

    await synth.set_language("EN")
    # 스타일 기본값 복원
    synth._sdp_ratio     = cfg.get("tts.melo_sdp_ratio", 0.2)
    synth._noise_scale   = cfg.get("tts.melo_noise_scale", 0.6)
    synth._noise_scale_w = cfg.get("tts.melo_noise_scale_w", 0.8)

    spk2id: dict = synth._engine.hps.data.spk2id
    accent_results: list[dict] = []

    for spk_key in EN_SPEAKERS:
        if spk_key not in spk2id:
            print(f"  [{spk_key}] 모델에 없음 — 스킵")
            continue
        synth._speaker_id = spk2id[spk_key]
        print(f"  [{spk_key}] speaker_id={spk2id[spk_key]}")
        t0 = time.monotonic()
        audio, sr = await synth.synthesize(BENCHMARK_TEXT_EN)
        synth_time = time.monotonic() - t0
        audio_duration = len(audio) / sr
        rtf = synth_time / audio_duration if audio_duration > 0 else 0.0

        print(f"  재생 중 (RTF={rtf:.3f}x)...", end="", flush=True)
        await manager.play_audio(audio, sr)
        print(" 완료")

        quality = input(f"  발음 선호도 (1/2/3) [Enter=3]: ").strip()
        quality_score = int(quality) if quality in ("1", "2", "3") else 3
        accent_results.append({"speaker": spk_key, "rtf": rtf, "quality": quality_score})
        print()

    await manager.stop()
    await synth.close()

    mem_final = get_memory_info()

    # ==================================================================
    # 결과 요약
    # ==================================================================
    print("=" * 65)
    print("  결과 요약")
    print("=" * 65)

    ok_results   = [r for r in results if r["error"] is None]
    fail_results = [r for r in results if r["error"] is not None]

    # RTF
    if ok_results:
        rtfs = [r["rtf"] for r in ok_results]
        rtf_mean = float(np.mean(rtfs))
        rtf_max  = max(rtfs)
        rtf_ok   = rtf_max < 1.0
        print(f"\n  RTF: mean={rtf_mean:.3f}x, max={rtf_max:.3f}x  목표: < 1.0x → {'PASS ✓' if rtf_ok else 'FAIL ✗'}")
    else:
        rtf_ok = False
        print(f"\n  RTF: 측정 불가 (오류 발생)")

    # 언어 전환
    switch_results = [r for r in ok_results if r["switch_time"] is not None]
    if switch_results:
        switch_max = max(r["switch_time"] for r in switch_results)
        switch_ok  = switch_max < 3.0
        print(f"  언어 전환: max={switch_max:.2f}초  목표: < 3초 → {'PASS ✓' if switch_ok else 'FAIL ✗'}")
    else:
        switch_ok = True

    # 발음 품질
    if ok_results:
        quality_mean = float(np.mean([r["quality_score"] for r in ok_results]))
        quality_ok   = quality_mean >= 2.0
        print(f"  발음 품질 평균: {quality_mean:.1f}/3.0  목표: ≥ 2.0 → {'PASS ✓' if quality_ok else 'FAIL ✗'}")
    else:
        quality_ok = False

    # 스타일 결과
    if style_results:
        print(f"\n  스타일 파라미터 청취 결과:")
        for r in style_results:
            print(f"    [{r['name']}] RTF={r['rtf']:.3f}x  선호도={r['quality']}/3")
        best_style = max(style_results, key=lambda r: r["quality"])
        print(f"  → 최선호 스타일: [{best_style['name']}]  "
              f"sdp={best_style['sdp']} noise={best_style['noise']} noise_w={best_style['noise_w']}")

    # 악센트 결과
    if accent_results:
        print(f"\n  EN 악센트 선호도:")
        for r in accent_results:
            print(f"    [{r['speaker']}] RTF={r['rtf']:.3f}x  선호도={r['quality']}/3")
        best_accent = max(accent_results, key=lambda r: r["quality"])
        print(f"  → 최선호 악센트: [{best_accent['speaker']}]")
        print(f"    → config: tts.melo_speaker: \"{best_accent['speaker']}\"")

    # 메모리
    print(f"\n  메모리:")
    if mem_before["used_mb"] is not None:
        print(f"    로드 전:   {mem_before['used_mb']:.0f}MB / 가용 {mem_before['available_mb']:.0f}MB")
    if mem_after["used_mb"] is not None:
        print(f"    로드 후:   {mem_after['used_mb']:.0f}MB / 가용 {mem_after['available_mb']:.0f}MB")
    if mem_final["used_mb"] is not None:
        print(f"    테스트 후: {mem_final['used_mb']:.0f}MB / 가용 {mem_final['available_mb']:.0f}MB")
    if mem_before["used_mb"] and mem_after["used_mb"]:
        print(f"    모델 점유: ~{mem_after['used_mb'] - mem_before['used_mb']:.0f}MB")

    # 개별 상세
    print(f"\n  ── Phase 3 문장별 상세 ──")
    for r in results:
        if r["error"]:
            print(f"    #{r['id']} [{r['lang']}] ERROR: {r['error']}")
        else:
            sw = f"  전환={r['switch_time']:.2f}s" if r["switch_time"] is not None else ""
            print(f"    #{r['id']} [{r['lang']}] RTF={r['rtf']:.3f}x  "
                  f"synth={r['synth_time']:.2f}s  dur={r['audio_duration']:.2f}s{sw}  품질={r['quality_score']}/3")

    # 최종 판정
    completeness_ok = len(ok_results) == len(TEST_SENTENCES)
    all_pass = rtf_ok and switch_ok and quality_ok and completeness_ok

    print(f"\n  ── 최종 판정 ──")
    print(f"  RTF < 1.0x:          {'PASS ✓' if rtf_ok else 'FAIL ✗'}")
    print(f"  언어 전환 < 3초:     {'PASS ✓' if switch_ok else 'FAIL ✗'}")
    print(f"  발음 품질 (≥ 2.0/3): {'PASS ✓' if quality_ok else 'FAIL ✗'}")
    print(f"  완전성:              {'PASS ✓' if completeness_ok else f'FAIL ✗ ({len(ok_results)}/{len(TEST_SENTENCES)})'}")
    print(f"\n  최종: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW-08: MeloTTS 라이브 테스트")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="디바이스 오버라이드 (미지정 시 config 값 사용)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="CPU vs GPU 합성 속도 비교 모드 (재생 없음, 빠른 벤치마크)",
    )
    args = parser.parse_args()
    asyncio.run(run_test(device_override=args.device, compare_mode=args.compare))
