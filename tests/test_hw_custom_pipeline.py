"""HW-10c: 커스텀 LLM (Samsung Gauss) + 전체 파이프라인 (LLM->TTS->Speaker) 라이브 테스트.

별도 터미널에서 실행:
    python tests/test_hw_custom_pipeline.py
    python tests/test_hw_custom_pipeline.py --lang ko       # 한국어만
    python tests/test_hw_custom_pipeline.py --lang en       # 영어만
    python tests/test_hw_custom_pipeline.py --no-play       # 재생 없이 합성까지만
    python tests/test_hw_custom_pipeline.py --model-id <UUID>   # 모델 ID 오버라이드

항상 provider=custom 으로 강제 실행 (LLM_PROVIDER 환경변수 무시).

환경 변수 (.env 또는 export):
    CUSTOM_LLM_URL=https://...
    CUSTOM_LLM_KEY=...
    CUSTOM_LLM_CLIENT_TOKEN=...
    CUSTOM_LLM_MODEL_ID=...
    CUSTOM_LLM_USER_EMAIL=...    (선택)

테스트 항목:
    Phase 1. 커스텀 LLM API 설정 확인
    Phase 2. 컴포넌트 초기화 (Custom LLM handler + TTS engine + AudioManager)
    Phase 3. 파이프라인 실행 (한국어 2 + 영어 2 = 4개 프롬프트)
             - 이벤트 타임라인 캡처 (EventBus queue 구독)
             - TTFA (첫 음성 재생까지 시간)
             - phrase 간 무음 갭
    Phase 4. 결과 요약

성공 기준: TTFA < 7초 (Custom LLM 네트워크 지연 고려), phrase 간 무음 < 1초
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import Config
from src.audio import AudioManager
from src.events import Event, EventBus, EventType
from src.llm import ConversationContext, create_chat_handler
from src.tts import create_synthesizer
from src.pipeline import StreamingPipeline


# ---------------------------------------------------------------------------
# 테스트 프롬프트
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    {
        "id": 1,
        "lang": "ko",
        "text": "삼성전자의 주요 사업 분야를 세 가지만 간단히 알려줘.",
        "desc": "한국어 — 회사 지식",
    },
    {
        "id": 2,
        "lang": "ko",
        "text": "아침에 간단하게 할 수 있는 스트레칭 방법을 알려줘.",
        "desc": "한국어 — 짧은 설명",
    },
    {
        "id": 3,
        "lang": "en",
        "text": "What's one fun fact about space that most people don't know?",
        "desc": "English — fun fact",
    },
    {
        "id": 4,
        "lang": "en",
        "text": "Give me one practical tip to improve focus while working from home.",
        "desc": "English — advice",
    },
]

SYSTEM_PROMPT_PATH = "config/prompts/system_prompt.txt"

TTFA_LIMIT = 7.0   # seconds (custom LLM 네트워크 지연 고려)
GAP_LIMIT  = 1.0   # seconds


# ---------------------------------------------------------------------------
# 설정 확인 헬퍼
# ---------------------------------------------------------------------------

def _mask(value: str, show: int = 6) -> str:
    if not value:
        return "(없음)"
    return value[:show] + "*" * max(0, len(value) - show)


def phase1_config_check(cfg: Config) -> bool:
    """커스텀 LLM API 설정 확인. 필수 항목 누락 시 False 반환."""
    print("\n── Phase 1: 커스텀 LLM API 설정 확인 ──")

    url          = cfg.get("llm.custom_url", "")
    key          = cfg.get("llm.custom_key", "")
    client_token = cfg.get("llm.custom_client_token", "")
    model_id     = cfg.get("llm.custom_model_id", "")
    user_email   = cfg.get("llm.custom_user_email", "")
    max_tokens   = cfg.get("llm.max_tokens", 1024)
    temperature  = cfg.get("llm.temperature", 1.0)

    print(f"  URL:            {url or '(없음)'}")
    print(f"  API Key:        {_mask(key)}")
    print(f"  Client Token:   {_mask(client_token)}")
    print(f"  Model ID:       {model_id or '(없음)'}")
    print(f"  User Email:     {user_email or '(없음, 선택)'}")
    print(f"  max_tokens:     {max_tokens}")
    print(f"  temperature:    {temperature}")

    missing = []
    if not url:
        missing.append("CUSTOM_LLM_URL")
    if not key:
        missing.append("CUSTOM_LLM_KEY")
    if not client_token:
        missing.append("CUSTOM_LLM_CLIENT_TOKEN")
    if not model_id:
        missing.append("CUSTOM_LLM_MODEL_ID")

    if missing:
        print(f"\n  [오류] 필수 환경변수 누락: {', '.join(missing)}")
        print("  .env 파일 또는 export 로 설정 후 다시 실행하세요.")
        return False

    print("\n  설정 확인: PASS ✓")
    return True


# ---------------------------------------------------------------------------
# 이벤트 타임라인 수집기 (test_hw_pipeline.py 와 동일)
# ---------------------------------------------------------------------------

class TimelineCollector:
    TRACKED_EVENTS = [
        EventType.LLM_RESPONSE_STARTED,
        EventType.LLM_RESPONSE_CHUNK,
        EventType.LLM_RESPONSE_DONE,
        EventType.PHRASE_READY,
        EventType.AUDIO_CHUNK_READY,
        EventType.PLAYBACK_STARTED,
        EventType.PLAYBACK_DONE,
    ]

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._queues: dict[EventType, asyncio.Queue[Event]] = {}
        self._tasks: list[asyncio.Task] = []
        self.events: list[tuple[float, str, object]] = []
        self._t0: float = 0.0

    def start(self, t0: float | None = None) -> None:
        self.events.clear()
        self._t0 = t0 or time.monotonic()
        for evt in self.TRACKED_EVENTS:
            q = self._bus.subscribe(evt)
            self._queues[evt] = q
            self._tasks.append(asyncio.create_task(self._drain(evt, q)))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        for evt, q in self._queues.items():
            self._bus.unsubscribe(evt, q)
        self._queues.clear()

    async def _drain(self, evt: EventType, q: asyncio.Queue[Event]) -> None:
        try:
            while True:
                event = await q.get()
                self.events.append((
                    time.monotonic() - self._t0,
                    event.type.name,
                    event.data,
                ))
        except asyncio.CancelledError:
            while not q.empty():
                try:
                    event = q.get_nowait()
                    self.events.append((
                        time.monotonic() - self._t0,
                        event.type.name,
                        event.data,
                    ))
                except asyncio.QueueEmpty:
                    break

    def get_ttfa(self) -> float | None:
        """TTFA = time to first AUDIO_CHUNK_READY (first audio is dequeued for playback).

        NOTE: PLAYBACK_STARTED fires when the playback worker *task* starts,
        before any audio is dequeued — it is NOT a reliable TTFA signal.
        """
        for t, name, _ in self.events:
            if name == "AUDIO_CHUNK_READY":
                return t
        return None

    def get_phrase_times(self) -> list[float]:
        return [t for t, name, _ in self.events if name == "PHRASE_READY"]

    def get_phrase_gaps(self) -> list[float]:
        chunks: list[tuple[float, float]] = []
        for t, name, data in self.events:
            if name == "AUDIO_CHUNK_READY" and isinstance(data, tuple):
                audio, sr = data
                audio_dur = len(audio) / sr
                chunks.append((t, audio_dur))

        if len(chunks) < 2:
            return []

        gaps: list[float] = []
        playback_end = chunks[0][0] + chunks[0][1]
        for i in range(1, len(chunks)):
            audio_ready = chunks[i][0]
            playback_start = max(audio_ready, playback_end)
            gap = playback_start - playback_end
            gaps.append(max(0.0, gap))
            playback_end = playback_start + chunks[i][1]
        return gaps

    def get_total_time(self) -> float | None:
        for t, name, _ in self.events:
            if name == "PLAYBACK_DONE":
                return t
        return None

    def print_timeline(self) -> None:
        seen_once: set[str] = set()
        for t, name, data in self.events:
            if name == "LLM_RESPONSE_CHUNK":
                continue
            if name == "LLM_RESPONSE_DONE":
                if name in seen_once:
                    continue
                seen_once.add(name)
            extra = ""
            if name == "PHRASE_READY" and isinstance(data, str):
                preview = data[:40].replace("\n", " ")
                extra = f'  "{preview}{"..." if len(data) > 40 else ""}"'
            print(f"    {t:7.3f}s  {name}{extra}")


# ---------------------------------------------------------------------------
# 메모리 유틸
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 더미 AudioManager (--no-play 모드)
# ---------------------------------------------------------------------------

class DummyAudioManager:
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        duration = len(audio) / sample_rate
        await asyncio.sleep(duration * 0.1)

    async def stop_playback(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 메인 테스트
# ---------------------------------------------------------------------------

async def run_test(
    lang_filter: str | None = None,
    no_play: bool = False,
    model_id_override: str | None = None,
) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("src.pipeline").setLevel(logging.INFO)

    cfg = Config()

    # 항상 custom provider 강제
    cfg._data.setdefault("llm", {})["provider"] = "custom"
    if model_id_override:
        cfg._data["llm"]["custom_model_id"] = model_id_override

    # 시스템 프롬프트 로드
    try:
        with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        system_prompt = "You are a helpful voice assistant. Keep responses concise."
        print(f"  [경고] {SYSTEM_PROMPT_PATH} 없음 — 기본 프롬프트 사용")

    # 프롬프트 필터링
    prompts = TEST_PROMPTS
    if lang_filter:
        prompts = [p for p in TEST_PROMPTS if p["lang"] == lang_filter]
        if not prompts:
            print(f"  [오류] '{lang_filter}' 언어 프롬프트가 없습니다.")
            return

    print("\n" + "=" * 65)
    print("HW-10c: Custom LLM (Samsung Gauss) + 전체 파이프라인 라이브 테스트")
    print("=" * 65)
    print(f"  LLM Provider:  custom (강제)")
    print(f"  Model ID:      {cfg.get('llm.custom_model_id', '(없음)')}")
    print(f"  TTS Provider:  {cfg.get('tts.provider', 'melo')}")
    print(f"  재생 모드:     {'스피커 출력' if not no_play else '더미 (재생 없음)'}")
    print(f"  프롬프트:      {len(prompts)}개", end="")
    if lang_filter:
        print(f" ({lang_filter} only)", end="")
    print()

    # ==================================================================
    # Phase 1: 설정 확인
    # ==================================================================
    if not phase1_config_check(cfg):
        return

    # ==================================================================
    # Phase 2: 컴포넌트 초기화
    # ==================================================================
    print("\n── Phase 2: 컴포넌트 초기화 ──")

    mem_before = get_memory_info()
    if mem_before["available_mb"] is not None:
        print(f"  메모리: {mem_before['used_mb']:.0f}MB 사용 / {mem_before['available_mb']:.0f}MB 가용")

    bus = EventBus()

    print("  Custom LLM handler 생성 중...", end="", flush=True)
    t0 = time.monotonic()
    try:
        handler = create_chat_handler(cfg, bus)
    except Exception as exc:
        print(f" 실패: {exc}")
        return
    print(f" {time.monotonic() - t0:.1f}초")

    print("  TTS synthesizer 생성 중...", end="", flush=True)
    t0 = time.monotonic()
    try:
        synth = create_synthesizer(cfg)
    except Exception as exc:
        print(f" 실패: {exc}")
        await handler.close()
        return
    tts_load_time = time.monotonic() - t0
    print(f" {tts_load_time:.1f}초")

    if no_play:
        audio_mgr = DummyAudioManager()
    else:
        audio_mgr = AudioManager(cfg)
    await audio_mgr.start()

    mem_after = get_memory_info()
    if mem_after["available_mb"] is not None:
        delta = (mem_after["used_mb"] or 0) - (mem_before["used_mb"] or 0)
        print(f"  초기화 후 메모리: {mem_after['used_mb']:.0f}MB 사용 (증가 ~{delta:.0f}MB)")

    # ==================================================================
    # Phase 3: 파이프라인 실행
    # ==================================================================
    print(f"\n── Phase 3: 파이프라인 실행 ({len(prompts)}개 프롬프트) ──")

    timeline = TimelineCollector(bus)
    results: list[dict] = []

    for prompt in prompts:
        print(f"\n  ── 프롬프트 {prompt['id']}/{len(TEST_PROMPTS)}: {prompt['desc']} ──")
        print(f'  질문: "{prompt["text"]}"')
        print(f"  언어: {prompt['lang']}")

        context = ConversationContext(system_prompt)
        context.add_user(prompt["text"])

        pipeline = StreamingPipeline(handler, synth, audio_mgr, bus)

        timeline.start()
        t_start = time.monotonic()

        try:
            full_response = await pipeline.run(context, detected_language=prompt["lang"])
        except Exception as exc:
            print(f"  [오류] 파이프라인 실행 실패: {exc}")
            await timeline.stop()
            results.append({
                "id": prompt["id"], "lang": prompt["lang"],
                "desc": prompt["desc"],
                "ttfa": None, "total_time": None,
                "phrase_count": 0, "max_gap": None,
                "phrase_gaps": [],
                "response": "", "error": str(exc),
            })
            continue

        total_elapsed = time.monotonic() - t_start

        # 드레인 태스크가 마지막 이벤트를 수집할 시간
        await asyncio.sleep(0.1)
        await timeline.stop()

        ttfa        = timeline.get_ttfa()
        total_time  = timeline.get_total_time()
        phrase_times = timeline.get_phrase_times()
        phrase_gaps = timeline.get_phrase_gaps()
        max_gap     = max(phrase_gaps) if phrase_gaps else None

        preview = full_response[:80].replace("\n", " ")
        suffix  = "..." if len(full_response) > 80 else ""

        print(f'  응답: "{preview}{suffix}"')
        print(f"  응답 길이: {len(full_response)}자")
        print(f"\n  이벤트 타임라인:")
        timeline.print_timeline()

        ttfa_str = f"{ttfa:.3f}s" if ttfa is not None else "N/A"
        ttfa_ok  = ttfa is not None and ttfa < TTFA_LIMIT
        total_str = f"{total_time:.3f}s" if total_time is not None else f"{total_elapsed:.3f}s"
        gap_ok   = max_gap is None or max_gap < GAP_LIMIT

        print(f"\n  TTFA: {ttfa_str} (목표 < {TTFA_LIMIT}s) -> {'PASS' if ttfa_ok else 'FAIL'}")
        print(f"  총 소요: {total_str}")
        print(f"  phrase 수: {len(phrase_times)}")
        if phrase_gaps:
            print(f"  phrase 간 갭: mean={np.mean(phrase_gaps):.3f}s, max={max_gap:.3f}s "
                  f"(목표 < {GAP_LIMIT}s) -> {'PASS' if gap_ok else 'FAIL'}")
        else:
            print(f"  phrase 간 갭: N/A (phrase 1개)")

        results.append({
            "id": prompt["id"], "lang": prompt["lang"],
            "desc": prompt["desc"],
            "ttfa": ttfa, "total_time": total_time or total_elapsed,
            "phrase_count": len(phrase_times),
            "phrase_gaps": phrase_gaps,
            "max_gap": max_gap,
            "response": full_response,
            "error": None,
        })

    # ==================================================================
    # 정리
    # ==================================================================
    await audio_mgr.stop()
    await synth.close()
    await handler.close()

    mem_final = get_memory_info()

    # ==================================================================
    # Phase 4: 결과 요약
    # ==================================================================
    print("\n" + "=" * 65)
    print("  결과 요약")
    print("=" * 65)

    ok_results   = [r for r in results if r["error"] is None]
    fail_results = [r for r in results if r["error"] is not None]

    # TTFA
    if ok_results:
        ttfas = [r["ttfa"] for r in ok_results if r["ttfa"] is not None]
        if ttfas:
            ttfa_mean = np.mean(ttfas)
            ttfa_max  = max(ttfas)
            ttfa_all_pass = ttfa_max < TTFA_LIMIT
            print(f"\n  TTFA: mean={ttfa_mean:.3f}s, max={ttfa_max:.3f}s (목표 < {TTFA_LIMIT}s)")
            print(f"  TTFA 판정: {'PASS' if ttfa_all_pass else 'FAIL'}")
        else:
            ttfa_all_pass = False
            print(f"\n  TTFA: 측정 불가")
    else:
        ttfa_all_pass = False

    # phrase 간 무음
    all_gaps: list[float] = []
    for r in ok_results:
        if r.get("phrase_gaps"):
            all_gaps.extend(r["phrase_gaps"])

    if all_gaps:
        gap_mean = np.mean(all_gaps)
        gap_max  = max(all_gaps)
        gap_all_pass = gap_max < GAP_LIMIT
        print(f"\n  phrase 간 갭: mean={gap_mean:.3f}s, max={gap_max:.3f}s (목표 < {GAP_LIMIT}s)")
        print(f"  갭 판정: {'PASS' if gap_all_pass else 'FAIL'}")
    else:
        gap_all_pass = True
        print(f"\n  phrase 간 갭: N/A (모든 응답 1 phrase)")

    # 오류
    if fail_results:
        print(f"\n  오류: {len(fail_results)}개 프롬프트 실패")
        for r in fail_results:
            print(f"    #{r['id']} [{r['lang']}]: {r['error']}")

    # 개별 상세
    print(f"\n  ── 개별 프롬프트 상세 ──")
    for r in results:
        if r["error"]:
            print(f"    #{r['id']} [{r['lang']}] ERROR: {r['error']}")
        else:
            ttfa_s = f"TTFA={r['ttfa']:.3f}s" if r["ttfa"] is not None else "TTFA=N/A"
            gap_s  = f"max_gap={r['max_gap']:.3f}s" if r["max_gap"] is not None else "gap=N/A"
            print(f"    #{r['id']} [{r['lang']}] {ttfa_s}  phrases={r['phrase_count']}  "
                  f"{gap_s}  total={r['total_time']:.2f}s")
            preview = r["response"][:60].replace("\n", " ")
            print(f"       \"{preview}{'...' if len(r['response']) > 60 else ''}\"")

    # 메모리
    print(f"\n  메모리:")
    if mem_before["used_mb"] is not None:
        print(f"    시작 전:   {mem_before['used_mb']:.0f}MB / 가용 {mem_before['available_mb']:.0f}MB")
    if mem_after["used_mb"] is not None:
        print(f"    초기화 후: {mem_after['used_mb']:.0f}MB / 가용 {mem_after['available_mb']:.0f}MB")
    if mem_final["used_mb"] is not None:
        print(f"    종료 후:   {mem_final['used_mb']:.0f}MB / 가용 {mem_final['available_mb']:.0f}MB")

    # 최종 판정
    completeness = len(ok_results) == len(prompts)

    print(f"\n  ── 최종 판정 ──")
    print(f"  TTFA < {TTFA_LIMIT}s:          {'PASS' if ttfa_all_pass else 'FAIL'}")
    print(f"  phrase 간 갭 < {GAP_LIMIT}s:   {'PASS' if gap_all_pass else 'FAIL'}")
    print(f"  완전성:              {'PASS' if completeness else f'FAIL ({len(ok_results)}/{len(prompts)})'}")

    all_pass = ttfa_all_pass and gap_all_pass and completeness
    print(f"\n  최종: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print("=" * 65)

    if not ttfa_all_pass:
        print("\n  [분석] TTFA가 목표를 초과한 경우:")
        print("    - 사내 네트워크 연결 상태 확인")
        print("    - Custom LLM TTFT 확인 (test_hw_custom_llm.py 로 단독 측정)")
        print("    - TTS 첫 합성 시간 확인 (MeloTTS KR warmup ~2s)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HW-10c: Custom LLM (Samsung Gauss) + 전체 파이프라인 라이브 테스트",
    )
    parser.add_argument(
        "--lang", choices=["ko", "en"], default=None,
        help="특정 언어만 테스트 (미지정 시 한/영 모두)",
    )
    parser.add_argument(
        "--no-play", action="store_true",
        help="스피커 재생 없이 합성까지만 테스트 (디버그용)",
    )
    parser.add_argument(
        "--model-id", type=str, default=None,
        help="CUSTOM_LLM_MODEL_ID 오버라이드 (예: 0196f1fc-...). 미지정 시 .env 값 사용.",
    )
    args = parser.parse_args()
    asyncio.run(run_test(
        lang_filter=args.lang,
        no_play=args.no_play,
        model_id_override=args.model_id,
    ))
