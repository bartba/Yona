"""HW-09: 스트리밍 파이프라인 (LLM->TTS->Speaker) 라이브 테스트.

별도 터미널에서 실행:
    python tests/test_hw_pipeline.py
    python tests/test_hw_pipeline.py --lang ko       # 한국어만
    python tests/test_hw_pipeline.py --lang en       # 영어만
    python tests/test_hw_pipeline.py --no-play       # 재생 없이 합성까지만 (디버그용)

테스트 항목:
    Phase 1. 컴포넌트 초기화 (LLM handler + TTS engine + AudioManager)
    Phase 2. 파이프라인 실행 (한국어 2 + 영어 2 = 4개 프롬프트)
             - 이벤트 타임라인 캡처 (EventBus queue 구독)
             - TTFA (첫 음성 재생까지 시간)
             - phrase 간 무음 갭
             - queue backpressure 관찰
    Phase 3. 결과 요약

성공 기준: TTFA < 5초, phrase 간 무음 < 1초

환경 변수:
    LLM_PROVIDER=claude         # HW-07에서 채택된 모델
    CLAUDE_API_KEY=sk-ant-...
    CLAUDE_MODEL=claude-haiku-4-5
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
        "text": "오늘 날씨가 좋은데, 경기도 수원에서 산책하기 좋은 장소 두군데 정도 추천해 줘.",
        "desc": "한국어 — 추천 요청",
    },
    {
        "id": 2,
        "lang": "ko",
        "text": "간단하게 아침에 일어나서 스트레칭 하는 방법 알려줘.",
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
        "text": "Recommend a simple recipe I can make in under ten minutes.",
        "desc": "English — recommendation",
    },
]

SYSTEM_PROMPT_PATH = "config/prompts/system_prompt.txt"


# ---------------------------------------------------------------------------
# 이벤트 타임라인 수집기
# ---------------------------------------------------------------------------

class TimelineCollector:
    """EventBus queue를 드레인하여 타임라인 수집.

    사용법:
        collector = TimelineCollector(bus)
        collector.start(t0)      # 구독 시작 + 드레인 태스크 생성
        # ... pipeline.run() ...
        await collector.stop()   # 드레인 태스크 정리 + 구독 해제
        collector.print_timeline()
    """

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
        """구독 시작 + 드레인 백그라운드 태스크 생성."""
        self.events.clear()
        self._t0 = t0 or time.monotonic()
        for evt in self.TRACKED_EVENTS:
            q = self._bus.subscribe(evt)
            self._queues[evt] = q
            self._tasks.append(asyncio.create_task(self._drain(evt, q)))

    async def stop(self) -> None:
        """드레인 태스크 취소 + 구독 해제."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        for evt, q in self._queues.items():
            self._bus.unsubscribe(evt, q)
        self._queues.clear()

    async def _drain(self, evt: EventType, q: asyncio.Queue[Event]) -> None:
        """큐에서 이벤트를 빼서 기록."""
        try:
            while True:
                event = await q.get()
                self.events.append((
                    time.monotonic() - self._t0,
                    event.type.name,
                    event.data,
                ))
        except asyncio.CancelledError:
            # 취소 전 남은 이벤트 비우기
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
        """TTFA = 첫 PLAYBACK_STARTED 이벤트까지 시간."""
        for t, name, _ in self.events:
            if name == "PLAYBACK_STARTED":
                return t
        return None

    def get_phrase_times(self) -> list[float]:
        """각 PHRASE_READY 이벤트의 상대 시간."""
        return [t for t, name, _ in self.events if name == "PHRASE_READY"]

    def get_audio_chunk_times(self) -> list[float]:
        """각 AUDIO_CHUNK_READY 이벤트의 상대 시간."""
        return [t for t, name, _ in self.events if name == "AUDIO_CHUNK_READY"]

    def get_phrase_gaps(self) -> list[float]:
        """연속 phrase 재생 사이의 실제 무음 갭 추정.

        이전 방식 (AUDIO_CHUNK_READY 간격)은 TTS 합성 시간을 측정하는
        오류가 있었음.  올바른 방식: 이전 chunk 재생이 끝나는 시점과
        다음 chunk가 audio_queue에 도착하는 시점을 비교.

        gap[i] = max(0, audio_ready[i+1] - playback_end[i])
        여기서 playback_end[i] = playback_start[i] + audio_dur[i]
        """
        # AUDIO_CHUNK_READY 이벤트에서 시간 + 오디오 길이 추출
        chunks: list[tuple[float, float]] = []  # (time, audio_duration)
        for t, name, data in self.events:
            if name == "AUDIO_CHUNK_READY" and isinstance(data, tuple):
                audio, sr = data
                audio_dur = len(audio) / sr
                chunks.append((t, audio_dur))

        if len(chunks) < 2:
            return []

        gaps: list[float] = []
        # 첫 chunk 재생 시작 = 도착 시간
        playback_end = chunks[0][0] + chunks[0][1]
        for i in range(1, len(chunks)):
            audio_ready = chunks[i][0]
            # 실제 재생 시작 = max(도착 시간, 이전 재생 종료)
            playback_start = max(audio_ready, playback_end)
            gap = playback_start - playback_end  # < 0이면 이미 대기 중이었음
            gaps.append(max(0.0, gap))
            playback_end = playback_start + chunks[i][1]
        return gaps

    def get_total_time(self) -> float | None:
        """PLAYBACK_DONE 이벤트 시간 (전체 파이프라인 소요 시간)."""
        for t, name, _ in self.events:
            if name == "PLAYBACK_DONE":
                return t
        return None

    def print_timeline(self) -> None:
        """타임라인을 보기 좋게 출력 (LLM_RESPONSE_CHUNK 생략, 중복 제거)."""
        seen_once: set[str] = set()
        for t, name, data in self.events:
            if name == "LLM_RESPONSE_CHUNK":
                continue
            # LLM_RESPONSE_DONE은 handler와 pipeline 양쪽에서 발행 — 첫 번째만 표시
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
    """재생 없이 합성만 테스트할 때 사용하는 더미."""

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        duration = len(audio) / sample_rate
        # 실제 재생 대신 짧은 대기 (재생 시간의 1/10)로 시뮬레이션
        await asyncio.sleep(duration * 0.1)

    async def stop_playback(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 메인 테스트
# ---------------------------------------------------------------------------

async def run_test(
    lang_filter: str | None = None,
    no_play: bool = False,
) -> None:
    # Pipeline 상세 타이밍 로그 활성화
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("src.pipeline").setLevel(logging.INFO)

    cfg = Config()

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

    provider = cfg.get("llm.provider", "custom") or "custom"

    print("\n" + "=" * 65)
    print("HW-09: 스트리밍 파이프라인 (LLM -> TTS -> Speaker) 라이브 테스트")
    print("=" * 65)
    print(f"  LLM Provider: {provider}")
    if provider == "claude":
        print(f"  Model: {cfg.get('llm.claude_model', '?')}")
    elif provider == "openai":
        print(f"  Model: {cfg.get('llm.openai_model', '?')}")
    print(f"  TTS Provider: {cfg.get('tts.provider', 'melo')}")
    print(f"  재생 모드: {'스피커 출력' if not no_play else '더미 (재생 없음)'}")
    print(f"  프롬프트: {len(prompts)}개", end="")
    if lang_filter:
        print(f" ({lang_filter} only)", end="")
    print()

    # ==================================================================
    # Phase 1: 컴포넌트 초기화
    # ==================================================================
    print("\n── Phase 1: 컴포넌트 초기화 ──")

    mem_before = get_memory_info()
    if mem_before["available_mb"] is not None:
        print(f"  메모리: {mem_before['used_mb']:.0f}MB 사용 / {mem_before['available_mb']:.0f}MB 가용")

    # EventBus
    bus = EventBus()

    # LLM handler
    print("  LLM handler 생성 중...", end="", flush=True)
    t0 = time.monotonic()
    try:
        handler = create_chat_handler(cfg, bus)
    except Exception as exc:
        print(f" 실패: {exc}")
        return
    print(f" {time.monotonic() - t0:.1f}초")

    # TTS synthesizer
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

    # AudioManager
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
    # Phase 2: 파이프라인 실행
    # ==================================================================
    print(f"\n── Phase 2: 파이프라인 실행 ({len(prompts)}개 프롬프트) ──")

    timeline = TimelineCollector(bus)
    results: list[dict] = []

    for prompt in prompts:
        print(f"\n  ── 프롬프트 {prompt['id']}/{len(TEST_PROMPTS)}: {prompt['desc']} ──")
        print(f'  질문: "{prompt["text"]}"')
        print(f"  언어: {prompt['lang']}")

        context = ConversationContext(system_prompt)
        context.add_user(prompt["text"])

        # 매 프롬프트마다 새 pipeline 인스턴스 (내부 상태 초기화)
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

        # 타임라인 분석
        ttfa = timeline.get_ttfa()
        total_time = timeline.get_total_time()
        phrase_times = timeline.get_phrase_times()
        phrase_gaps = timeline.get_phrase_gaps()
        max_gap = max(phrase_gaps) if phrase_gaps else None

        # 응답 미리보기
        preview = full_response[:80].replace("\n", " ")
        suffix = "..." if len(full_response) > 80 else ""

        print(f'  응답: "{preview}{suffix}"')
        print(f"  응답 길이: {len(full_response)}자")

        # 타임라인 출력
        print(f"\n  이벤트 타임라인:")
        timeline.print_timeline()

        # 핵심 메트릭
        ttfa_str = f"{ttfa:.3f}s" if ttfa is not None else "N/A"
        ttfa_ok = ttfa is not None and ttfa < 5.0
        total_str = f"{total_time:.3f}s" if total_time is not None else f"{total_elapsed:.3f}s"
        gap_ok = max_gap is None or max_gap < 1.0

        print(f"\n  TTFA: {ttfa_str} (목표 < 5s) -> {'PASS' if ttfa_ok else 'FAIL'}")
        print(f"  총 소요: {total_str}")
        print(f"  phrase 수: {len(phrase_times)}")
        if phrase_gaps:
            print(f"  phrase 간 갭: mean={np.mean(phrase_gaps):.3f}s, max={max_gap:.3f}s "
                  f"(목표 < 1s) -> {'PASS' if gap_ok else 'FAIL'}")
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
    # Phase 3: 결과 요약
    # ==================================================================
    print("\n" + "=" * 65)
    print("  결과 요약")
    print("=" * 65)

    ok_results = [r for r in results if r["error"] is None]
    fail_results = [r for r in results if r["error"] is not None]

    # TTFA
    if ok_results:
        ttfas = [r["ttfa"] for r in ok_results if r["ttfa"] is not None]
        if ttfas:
            ttfa_mean = np.mean(ttfas)
            ttfa_max = max(ttfas)
            ttfa_all_pass = ttfa_max < 5.0
            print(f"\n  TTFA: mean={ttfa_mean:.3f}s, max={ttfa_max:.3f}s (목표 < 5.0s)")
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
        gap_max = max(all_gaps)
        gap_all_pass = gap_max < 1.0
        print(f"\n  phrase 간 갭: mean={gap_mean:.3f}s, max={gap_max:.3f}s (목표 < 1.0s)")
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
            ttfa_s = f"TTFA={r['ttfa']:.3f}s" if r['ttfa'] is not None else "TTFA=N/A"
            gap_s = f"max_gap={r['max_gap']:.3f}s" if r['max_gap'] is not None else "gap=N/A"
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
    print(f"  TTFA < 5s:           {'PASS' if ttfa_all_pass else 'FAIL'}")
    print(f"  phrase 간 갭 < 1s:   {'PASS' if gap_all_pass else 'FAIL'}")
    print(f"  완전성:              {'PASS' if completeness else f'FAIL ({len(ok_results)}/{len(prompts)})'}")

    all_pass = ttfa_all_pass and gap_all_pass and completeness
    print(f"\n  최종: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 65)

    if all_pass:
        print("\n  [참고] 음성 품질/자연스러움은 재생 출력을 직접 확인하세요.")
    if not ttfa_all_pass:
        print("\n  [분석] TTFA가 5s를 초과한 경우:")
        print("    - LLM TTFT 확인: claude-haiku는 통상 ~0.8s")
        print("    - TTS 첫 합성 시간 확인: MeloTTS KR warmup ~2s")
        print("    - 언어 전환 시 추가 로드 시간 발생 가능")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HW-09: 스트리밍 파이프라인 라이브 테스트",
    )
    parser.add_argument(
        "--lang", choices=["ko", "en"], default=None,
        help="특정 언어만 테스트 (미지정 시 한/영 모두)",
    )
    parser.add_argument(
        "--no-play", action="store_true",
        help="스피커 재생 없이 합성까지만 테스트 (디버그용)",
    )
    args = parser.parse_args()
    asyncio.run(run_test(lang_filter=args.lang, no_play=args.no_play))
