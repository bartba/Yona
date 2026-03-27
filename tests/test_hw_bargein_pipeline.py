"""HW-10: Barge-in 파이프라인 중단 라이브 테스트.

별도 터미널에서 실행:
    python tests/test_hw_bargein_pipeline.py
    python tests/test_hw_bargein_pipeline.py --lang ko       # 한국어만
    python tests/test_hw_bargein_pipeline.py --no-vad        # VAD 없이 타이머 인터럽트만

테스트 항목:
    Phase 1. 컴포넌트 초기화 (LLM + TTS + AudioManager + VAD)
    Phase 2. 타이머 인터럽트 — 재생 시작 2초 후 자동 interrupt(), 중단 지연 측정
    Phase 3. VAD barge-in — 긴 응답 재생 중 실제 음성으로 interrupt (사용자 참여)
    Phase 4. False barge-in 검증 — 재생 중 침묵 유지, VAD false positive 확인
    Phase 5. 결과 요약

성공 기준:
    - interrupt() → 재생 중단까지 1초 이내
    - false barge-in 0회

환경 변수:
    LLM_PROVIDER=claude
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

from src.config import Config
from src.audio import AudioManager
from src.events import Event, EventBus, EventType
from src.llm import ConversationContext, create_chat_handler
from src.tts import create_synthesizer
from src.pipeline import StreamingPipeline
from src.vad import VoiceActivityDetector


# ---------------------------------------------------------------------------
# 테스트 프롬프트 — 긴 응답을 유도하는 질문
# ---------------------------------------------------------------------------

LONG_PROMPTS = {
    "ko": {
        "text": "인공지능의 역사를 1950년대부터 현재까지 주요 사건과 인물을 포함해서 자세히 설명해 줘.",
        "desc": "한국어 — 긴 설명 요청 (AI 역사)",
    },
    "en": {
        "text": "Explain the complete history of artificial intelligence from the 1950s to today, including key events and people.",
        "desc": "English — long explanation (AI history)",
    },
}

SYSTEM_PROMPT_PATH = "config/prompts/system_prompt.txt"

# Phase 2 타이머 인터럽트 대기 시간 (재생 시작 후)
TIMER_INTERRUPT_DELAY = 2.0  # seconds


# ---------------------------------------------------------------------------
# 이벤트 모니터
# ---------------------------------------------------------------------------

class BargeinMonitor:
    """파이프라인 이벤트 + VAD barge-in 이벤트를 모니터링.

    - PLAYBACK_STARTED/DONE 시간 기록
    - BARGE_IN_DETECTED 횟수 기록
    - 외부에서 interrupt 시각 기록 → 지연 시간 계산
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._queues: dict[EventType, asyncio.Queue[Event]] = {}
        self._tasks: list[asyncio.Task] = []
        self._t0: float = 0.0

        # 측정값
        self.playback_started_at: float | None = None
        self.playback_done_at: float | None = None
        self.interrupt_called_at: float | None = None
        self.barge_in_events: list[float] = []
        self.phrase_count: int = 0
        self.events: list[tuple[float, str, object]] = []

    def start(self) -> None:
        self._t0 = time.monotonic()
        self.playback_started_at = None
        self.playback_done_at = None
        self.interrupt_called_at = None
        self.barge_in_events.clear()
        self.phrase_count = 0
        self.events.clear()

        tracked = [
            EventType.PLAYBACK_STARTED,
            EventType.PLAYBACK_DONE,
            EventType.BARGE_IN_DETECTED,
            EventType.PHRASE_READY,
            EventType.LLM_RESPONSE_STARTED,
            EventType.LLM_RESPONSE_DONE,
            EventType.AUDIO_CHUNK_READY,
        ]
        for evt in tracked:
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
                t = time.monotonic() - self._t0
                self.events.append((t, event.type.name, event.data))

                if event.type == EventType.PLAYBACK_STARTED:
                    self.playback_started_at = t
                elif event.type == EventType.PLAYBACK_DONE:
                    self.playback_done_at = t
                elif event.type == EventType.BARGE_IN_DETECTED:
                    self.barge_in_events.append(t)
                elif event.type == EventType.PHRASE_READY:
                    self.phrase_count += 1
        except asyncio.CancelledError:
            while not q.empty():
                try:
                    event = q.get_nowait()
                    t = time.monotonic() - self._t0
                    self.events.append((t, event.type.name, event.data))
                except asyncio.QueueEmpty:
                    break

    def mark_interrupt(self) -> None:
        """interrupt() 호출 시각 기록."""
        self.interrupt_called_at = time.monotonic() - self._t0

    @property
    def interrupt_latency(self) -> float | None:
        """interrupt() 호출 → PLAYBACK_DONE 까지 시간."""
        if self.interrupt_called_at is not None and self.playback_done_at is not None:
            return self.playback_done_at - self.interrupt_called_at
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
            elif name == "BARGE_IN_DETECTED":
                extra = "  *** BARGE-IN ***"
            print(f"    {t:7.3f}s  {name}{extra}")

        if self.interrupt_called_at is not None:
            print(f"    {self.interrupt_called_at:7.3f}s  >> interrupt() called <<")


# ---------------------------------------------------------------------------
# VAD barge-in 콜백 연결
# ---------------------------------------------------------------------------

class VadBargeinBridge:
    """VAD barge-in 감지 시 pipeline.interrupt() 호출.

    AudioManager input callback으로 VAD를 연결하고,
    BARGE_IN_DETECTED 이벤트를 구독하여 pipeline.interrupt()를 트리거.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        bus: EventBus,
        pipeline: StreamingPipeline | None = None,
        monitor: BargeinMonitor | None = None,
    ) -> None:
        self._vad = vad
        self._bus = bus
        self._pipeline = pipeline
        self._monitor = monitor
        self._queue: asyncio.Queue[Event] | None = None
        self._task: asyncio.Task | None = None

    def set_pipeline(self, pipeline: StreamingPipeline) -> None:
        self._pipeline = pipeline

    async def start(self) -> None:
        """BARGE_IN_DETECTED 구독 + 드레인 태스크 생성."""
        self._queue = self._bus.subscribe(EventType.BARGE_IN_DETECTED)
        self._task = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._queue:
            self._bus.unsubscribe(EventType.BARGE_IN_DETECTED, self._queue)
            self._queue = None

    async def _listen(self) -> None:
        try:
            while True:
                await self._queue.get()
                if self._pipeline:
                    if self._monitor:
                        self._monitor.mark_interrupt()
                    print("  >>> VAD barge-in 감지! interrupt() 호출 <<<")
                    await self._pipeline.interrupt()
                    break  # 1회만
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# 메인 테스트
# ---------------------------------------------------------------------------

async def run_test(
    lang_filter: str | None = None,
    no_vad: bool = False,
) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("src.pipeline").setLevel(logging.INFO)
    logging.getLogger("src.vad").setLevel(logging.INFO)

    cfg = Config()
    lang = lang_filter or "ko"
    prompt_info = LONG_PROMPTS.get(lang, LONG_PROMPTS["ko"])

    try:
        with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        system_prompt = "You are a helpful voice assistant. Keep responses concise."
        print(f"  [경고] {SYSTEM_PROMPT_PATH} 없음 — 기본 프롬프트 사용")

    provider = cfg.get("llm.provider", "custom") or "custom"

    print("\n" + "=" * 65)
    print("HW-10: Barge-in 파이프라인 중단 라이브 테스트")
    print("=" * 65)
    print(f"  LLM Provider: {provider}")
    if provider == "claude":
        print(f"  Model: {cfg.get('llm.claude_model', '?')}")
    elif provider == "openai":
        print(f"  Model: {cfg.get('llm.openai_model', '?')}")
    print(f"  TTS Provider: {cfg.get('tts.provider', 'melo')}")
    print(f"  언어: {lang}")
    print(f"  VAD barge-in: {'활성' if not no_vad else '비활성 (타이머만)'}")

    # ==================================================================
    # Phase 1: 컴포넌트 초기화
    # ==================================================================
    print("\n── Phase 1: 컴포넌트 초기화 ──")

    bus = EventBus()

    print("  LLM handler 생성 중...", end="", flush=True)
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
    print(f" {time.monotonic() - t0:.1f}초")

    audio_mgr = AudioManager(cfg)
    await audio_mgr.start()

    # VAD (Phase 3에서 사용)
    vad: VoiceActivityDetector | None = None
    if not no_vad:
        print("  VAD 초기화 중...", end="", flush=True)
        t0 = time.monotonic()
        try:
            vad = VoiceActivityDetector(cfg, bus)
            audio_mgr.add_input_callback(vad.process_chunk)
            print(f" {time.monotonic() - t0:.1f}초")
        except Exception as exc:
            print(f" 실패: {exc}")
            vad = None

    print("  초기화 완료!")

    results: dict[str, dict] = {}

    # ==================================================================
    # Phase 2: 타이머 인터럽트 — 중단 지연 측정
    # ==================================================================
    print(f"\n── Phase 2: 타이머 인터럽트 (재생 {TIMER_INTERRUPT_DELAY}초 후 자동 중단) ──")
    print(f'  질문: "{prompt_info["text"][:60]}..."')

    context = ConversationContext(system_prompt)
    context.add_user(prompt_info["text"])

    pipeline = StreamingPipeline(handler, synth, audio_mgr, bus)
    monitor = BargeinMonitor(bus)
    monitor.start()

    # PLAYBACK_STARTED를 기다린 뒤 타이머로 interrupt 호출
    playback_q = bus.subscribe(EventType.PLAYBACK_STARTED)

    async def timer_interrupt():
        """재생 시작 후 N초 뒤에 interrupt."""
        try:
            await asyncio.wait_for(playback_q.get(), timeout=30.0)
            print(f"  재생 시작 — {TIMER_INTERRUPT_DELAY}초 후 interrupt 예정")
            await asyncio.sleep(TIMER_INTERRUPT_DELAY)
            monitor.mark_interrupt()
            t_call = time.monotonic()
            print("  >>> interrupt() 호출! <<<")
            await pipeline.interrupt()
            t_done = time.monotonic()
            print(f"  interrupt() 완료: {(t_done - t_call)*1000:.0f}ms")
        except asyncio.TimeoutError:
            print("  [경고] 30초 내 PLAYBACK_STARTED 없음")
        except asyncio.CancelledError:
            pass

    t_start = time.monotonic()
    timer_task = asyncio.create_task(timer_interrupt())

    try:
        full_response = await pipeline.run(context, detected_language=lang)
    except Exception as exc:
        print(f"  [오류] 파이프라인 실행 실패: {exc}")
        full_response = ""

    if not timer_task.done():
        timer_task.cancel()
        try:
            await timer_task
        except asyncio.CancelledError:
            pass

    bus.unsubscribe(EventType.PLAYBACK_STARTED, playback_q)
    total_elapsed = time.monotonic() - t_start

    await asyncio.sleep(0.1)
    await monitor.stop()

    latency = monitor.interrupt_latency
    latency_str = f"{latency*1000:.0f}ms" if latency is not None else "N/A"
    latency_ok = latency is not None and latency < 1.0

    print(f"\n  이벤트 타임라인:")
    monitor.print_timeline()
    print(f"\n  총 소요: {total_elapsed:.2f}s (인터럽트 없었으면 더 길었을 것)")
    print(f"  응답 길이: {len(full_response)}자 (인터럽트로 잘림)")
    print(f"  phrase 수: {monitor.phrase_count}")
    print(f"  interrupt 지연: {latency_str} (목표 < 1s) -> {'PASS' if latency_ok else 'FAIL'}")

    results["timer"] = {
        "latency": latency,
        "latency_ok": latency_ok,
        "total_elapsed": total_elapsed,
        "response_len": len(full_response),
        "phrase_count": monitor.phrase_count,
    }

    # ==================================================================
    # Phase 3: VAD barge-in — 실제 음성으로 인터럽트 (사용자 참여)
    # ==================================================================
    if vad and not no_vad:
        print(f"\n── Phase 3: VAD barge-in (실제 음성으로 인터럽트) ──")
        print(f"  ⚠  재생이 시작되면 마이크에 대고 말해주세요!")
        print(f"  ⚠  \"안녕하세요\" 또는 아무 말이나 하세요 (음성이 감지되면 중단됩니다)")
        input("  [Enter] 를 눌러 시작...")

        vad.set_barge_in_mode(True)
        vad.reset()

        context2 = ConversationContext(system_prompt)
        context2.add_user(prompt_info["text"])

        pipeline2 = StreamingPipeline(handler, synth, audio_mgr, bus)
        monitor2 = BargeinMonitor(bus)
        monitor2.start()

        bridge = VadBargeinBridge(vad, bus, pipeline2, monitor2)
        await bridge.start()

        t_start2 = time.monotonic()
        try:
            full_response2 = await pipeline2.run(context2, detected_language=lang)
        except Exception as exc:
            print(f"  [오류] 파이프라인 실행 실패: {exc}")
            full_response2 = ""

        total_elapsed2 = time.monotonic() - t_start2

        await bridge.stop()
        vad.set_barge_in_mode(False)
        vad.reset()

        await asyncio.sleep(0.1)
        await monitor2.stop()

        barge_in_count = len(monitor2.barge_in_events)
        latency2 = monitor2.interrupt_latency
        latency2_str = f"{latency2*1000:.0f}ms" if latency2 is not None else "N/A"
        latency2_ok = latency2 is not None and latency2 < 1.0

        print(f"\n  이벤트 타임라인:")
        monitor2.print_timeline()
        print(f"\n  총 소요: {total_elapsed2:.2f}s")
        print(f"  응답 길이: {len(full_response2)}자")
        print(f"  barge-in 감지 횟수: {barge_in_count}")
        if barge_in_count > 0:
            print(f"  barge-in 시각: {', '.join(f'{t:.3f}s' for t in monitor2.barge_in_events)}")
        print(f"  interrupt 지연: {latency2_str} (목표 < 1s) -> {'PASS' if latency2_ok else 'FAIL'}")

        if barge_in_count == 0:
            print("  [주의] barge-in 감지 안됨 — 음성을 충분히 크게 하셨나요?")

        results["vad_bargein"] = {
            "latency": latency2,
            "latency_ok": latency2_ok,
            "barge_in_count": barge_in_count,
            "total_elapsed": total_elapsed2,
            "response_len": len(full_response2),
        }
    else:
        print(f"\n── Phase 3: 건너뜀 (--no-vad 또는 VAD 초기화 실패) ──")
        results["vad_bargein"] = None

    # ==================================================================
    # Phase 4: False barge-in 검증 — 재생 중 침묵, VAD FP 확인
    # ==================================================================
    if vad and not no_vad:
        print(f"\n── Phase 4: False barge-in 검증 (재생 중 침묵 유지) ──")
        print(f"  ⚠  이번에는 말하지 마세요! 재생이 끝날 때까지 조용히 기다려주세요.")
        print(f"  ⚠  Poly Sync 20 AEC가 스피커 에코를 제거하는지 확인합니다.")
        input("  [Enter] 를 눌러 시작...")

        vad.set_barge_in_mode(True)
        vad.reset()

        # 짧은 프롬프트로 적절한 길이의 응답 유도
        short_prompts = {
            "ko": "봄에 벚꽃 구경 가기 좋은 곳 세 곳을 간단히 소개해 줘.",
            "en": "Recommend three nice places to see cherry blossoms in spring.",
        }

        context3 = ConversationContext(system_prompt)
        context3.add_user(short_prompts.get(lang, short_prompts["ko"]))

        pipeline3 = StreamingPipeline(handler, synth, audio_mgr, bus)
        monitor3 = BargeinMonitor(bus)
        monitor3.start()

        # VAD barge-in 이벤트만 수집 (interrupt는 하지 않음)
        barge_in_q = bus.subscribe(EventType.BARGE_IN_DETECTED)
        false_barge_ins: list[float] = []

        async def collect_false_bargeins():
            try:
                while True:
                    event = await barge_in_q.get()
                    false_barge_ins.append(time.monotonic())
            except asyncio.CancelledError:
                pass

        collector_task = asyncio.create_task(collect_false_bargeins())

        t_start3 = time.monotonic()
        try:
            full_response3 = await pipeline3.run(context3, detected_language=lang)
        except Exception as exc:
            print(f"  [오류] 파이프라인 실행 실패: {exc}")
            full_response3 = ""

        total_elapsed3 = time.monotonic() - t_start3

        collector_task.cancel()
        try:
            await collector_task
        except asyncio.CancelledError:
            pass
        bus.unsubscribe(EventType.BARGE_IN_DETECTED, barge_in_q)

        vad.set_barge_in_mode(False)
        vad.reset()

        await asyncio.sleep(0.1)
        await monitor3.stop()

        fp_count = len(false_barge_ins)
        fp_ok = fp_count == 0

        print(f"\n  이벤트 타임라인:")
        monitor3.print_timeline()
        print(f"\n  총 소요: {total_elapsed3:.2f}s")
        print(f"  응답 완성 길이: {len(full_response3)}자")
        print(f"  false barge-in 횟수: {fp_count} (목표 = 0) -> {'PASS' if fp_ok else 'FAIL'}")

        if not fp_ok:
            print(f"  [분석] AEC가 스피커 에코를 완전히 제거하지 못했을 수 있습니다.")
            print(f"    - barge_in_threshold 상향 검토 (현재: {cfg.get('vad.barge_in_threshold', 0.7)})")

        results["false_bargein"] = {
            "fp_count": fp_count,
            "fp_ok": fp_ok,
            "total_elapsed": total_elapsed3,
            "response_len": len(full_response3),
        }
    else:
        print(f"\n── Phase 4: 건너뜀 (--no-vad 또는 VAD 초기화 실패) ──")
        results["false_bargein"] = None

    # ==================================================================
    # 정리
    # ==================================================================
    await audio_mgr.stop()
    await synth.close()
    await handler.close()

    # ==================================================================
    # Phase 5: 결과 요약
    # ==================================================================
    print("\n" + "=" * 65)
    print("  결과 요약")
    print("=" * 65)

    # Phase 2: 타이머 인터럽트
    timer_r = results["timer"]
    timer_ok = timer_r["latency_ok"]
    print(f"\n  Phase 2 — 타이머 인터럽트:")
    print(f"    interrupt 지연: {timer_r['latency']*1000:.0f}ms" if timer_r['latency'] else "    interrupt 지연: N/A")
    print(f"    판정: {'PASS' if timer_ok else 'FAIL'}")

    # Phase 3: VAD barge-in
    vad_r = results.get("vad_bargein")
    if vad_r:
        vad_ok = vad_r["latency_ok"] and vad_r["barge_in_count"] > 0
        print(f"\n  Phase 3 — VAD barge-in:")
        print(f"    barge-in 감지: {vad_r['barge_in_count']}회")
        lat_str = f"{vad_r['latency']*1000:.0f}ms" if vad_r['latency'] else "N/A"
        print(f"    interrupt 지연: {lat_str}")
        print(f"    판정: {'PASS' if vad_ok else 'FAIL'}")
    else:
        vad_ok = True  # skipped
        print(f"\n  Phase 3 — VAD barge-in: 건너뜀")

    # Phase 4: False barge-in
    fp_r = results.get("false_bargein")
    if fp_r:
        fp_ok = fp_r["fp_ok"]
        print(f"\n  Phase 4 — False barge-in:")
        print(f"    false positive: {fp_r['fp_count']}회")
        print(f"    판정: {'PASS' if fp_ok else 'FAIL'}")
    else:
        fp_ok = True  # skipped
        print(f"\n  Phase 4 — False barge-in: 건너뜀")

    # 최종
    all_pass = timer_ok and vad_ok and fp_ok
    print(f"\n  ── 최종 판정 ──")
    print(f"  interrupt < 1s:      {'PASS' if timer_ok else 'FAIL'}")
    if vad_r:
        print(f"  VAD barge-in 작동:   {'PASS' if vad_ok else 'FAIL'}")
    if fp_r:
        print(f"  false barge-in = 0:  {'PASS' if fp_ok else 'FAIL'}")
    print(f"\n  최종: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 65)

    if not timer_ok:
        print("\n  [분석] interrupt 지연이 1초를 초과한 경우:")
        print("    - sd.stop()이 blocking playback을 즉시 중단하는지 확인")
        print("    - asyncio.to_thread(sd.play, blocking=True) 취소 경로 점검")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HW-10: Barge-in 파이프라인 중단 라이브 테스트",
    )
    parser.add_argument(
        "--lang", choices=["ko", "en"], default=None,
        help="테스트 언어 (미지정 시 ko)",
    )
    parser.add_argument(
        "--no-vad", action="store_true",
        help="VAD 없이 타이머 인터럽트만 테스트",
    )
    args = parser.parse_args()
    asyncio.run(run_test(lang_filter=args.lang, no_vad=args.no_vad))
