"""HW-11: 마이크 → STT → Custom LLM → TTS → 스피커 전체 파이프라인 라이브 테스트.

별도 터미널에서 실행:
    python tests/test_hw_full_custom.py
    python tests/test_hw_full_custom.py --turns 3    # 대화 턴 수 (기본 3)
    python tests/test_hw_full_custom.py --no-play    # 재생 없이 합성까지만

항상 provider=custom 으로 강제 실행.

테스트 흐름 (턴당):
    1. "말씀하세요..." 출력 → 사용자 발화
    2. VAD로 발화 끝 감지 → AudioBuffer에서 오디오 추출
    3. Whisper STT 전사
    4. Custom LLM → TTS → 스피커 재생 (StreamingPipeline)
    5. 메트릭 출력 (STT 소요, 실제 TTFA, phrase 갭)

환경 변수:
    CUSTOM_LLM_URL / CUSTOM_LLM_KEY / CUSTOM_LLM_CLIENT_TOKEN / CUSTOM_LLM_MODEL_ID
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import Config
from src.audio import AudioBuffer, AudioManager
from src.events import Event, EventBus, EventType
from src.llm import ConversationContext, create_chat_handler
from src.pipeline import StreamingPipeline
from src.stt import Transcriber
from src.tts import create_synthesizer
from src.vad import VoiceActivityDetector

SYSTEM_PROMPT_PATH = "config/prompts/system_prompt.txt"
TTFA_LIMIT = 7.0   # seconds
GAP_LIMIT  = 1.0   # seconds
MAX_LISTEN_SECS = 30  # safety cutoff if VAD never fires SPEECH_ENDED


# ---------------------------------------------------------------------------
# Config check helper
# ---------------------------------------------------------------------------

def _mask(value: str, show: int = 6) -> str:
    if not value:
        return "(없음)"
    return value[:show] + "*" * max(0, len(value) - show)


def check_config(cfg: Config) -> bool:
    print("\n── 커스텀 LLM 설정 확인 ──")
    url          = cfg.get("llm.custom_url", "")
    key          = cfg.get("llm.custom_key", "")
    client_token = cfg.get("llm.custom_client_token", "")
    model_id     = cfg.get("llm.custom_model_id", "")
    print(f"  URL:          {url or '(없음)'}")
    print(f"  API Key:      {_mask(key)}")
    print(f"  Client Token: {_mask(client_token)}")
    print(f"  Model ID:     {model_id or '(없음)'}")

    missing = [k for k, v in [
        ("CUSTOM_LLM_URL", url), ("CUSTOM_LLM_KEY", key),
        ("CUSTOM_LLM_CLIENT_TOKEN", client_token), ("CUSTOM_LLM_MODEL_ID", model_id),
    ] if not v]

    if missing:
        print(f"\n  [오류] 필수 환경변수 누락: {', '.join(missing)}")
        return False
    print("  설정 확인: PASS ✓")
    return True


# ---------------------------------------------------------------------------
# TTFA helper: first AUDIO_CHUNK_READY time (real first audio, not task start)
# ---------------------------------------------------------------------------

def get_ttfa(events: list[tuple[float, str, object]]) -> float | None:
    for t, name, _ in events:
        if name == "AUDIO_CHUNK_READY":
            return t
    return None


def get_phrase_gaps(events: list[tuple[float, str, object]]) -> list[float]:
    chunks: list[tuple[float, float]] = []
    for t, name, data in events:
        if name == "AUDIO_CHUNK_READY" and isinstance(data, tuple):
            audio, sr = data
            chunks.append((t, len(audio) / sr))
    if len(chunks) < 2:
        return []
    gaps: list[float] = []
    playback_end = chunks[0][0] + chunks[0][1]
    for i in range(1, len(chunks)):
        ready = chunks[i][0]
        start = max(ready, playback_end)
        gaps.append(max(0.0, start - playback_end))
        playback_end = start + chunks[i][1]
    return gaps


def print_timeline(events: list[tuple[float, str, object]]) -> None:
    seen: set[str] = set()
    for t, name, data in events:
        if name == "LLM_RESPONSE_CHUNK":
            continue
        if name == "LLM_RESPONSE_DONE" and name in seen:
            continue
        seen.add(name)
        extra = ""
        if name == "PHRASE_READY" and isinstance(data, str):
            preview = data[:40].replace("\n", " ")
            extra = f'  "{preview}{"..." if len(data) > 40 else ""}"'
        print(f"    {t:7.3f}s  {name}{extra}")


# ---------------------------------------------------------------------------
# 더미 AudioManager (--no-play 모드)
# ---------------------------------------------------------------------------

class DummyAudioManager:
    async def start(self) -> None: pass
    async def stop(self) -> None: pass
    async def play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        await asyncio.sleep(len(audio) / sample_rate * 0.1)
    async def stop_playback(self) -> None: pass


# ---------------------------------------------------------------------------
# 이벤트 수집기 (타임라인)
# ---------------------------------------------------------------------------

TRACKED = [
    EventType.LLM_RESPONSE_STARTED,
    EventType.LLM_RESPONSE_CHUNK,
    EventType.LLM_RESPONSE_DONE,
    EventType.PHRASE_READY,
    EventType.AUDIO_CHUNK_READY,
    EventType.PLAYBACK_STARTED,
    EventType.PLAYBACK_DONE,
]


class TimelineCollector:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._queues: dict[EventType, asyncio.Queue[Event]] = {}
        self._tasks: list[asyncio.Task] = []
        self.events: list[tuple[float, str, object]] = []
        self._t0: float = 0.0

    def start(self) -> None:
        self.events.clear()
        self._t0 = time.monotonic()
        for evt in TRACKED:
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


# ---------------------------------------------------------------------------
# 단일 대화 턴: 마이크 수신 → VAD → STT → Pipeline
# ---------------------------------------------------------------------------

async def run_turn(
    turn_idx: int,
    cfg: Config,
    bus: EventBus,
    audio_mgr: AudioManager,
    buffer: AudioBuffer,
    vad: VoiceActivityDetector,
    stt: Transcriber,
    handler,
    synth,
    audio_out: AudioManager | DummyAudioManager,
    context: ConversationContext,
    timeline: TimelineCollector,
    no_play: bool,
) -> dict:
    sample_rate: int = cfg.get("audio.input_sample_rate", 16_000)

    # 오디오 콜백: SPEECH_STARTED 이후에만 버퍼에 push (pre-speech 대기 시간 제외)
    _collecting = threading.Event()  # set on SPEECH_STARTED (sounddevice 스레드에서 안전하게 체크)

    def _audio_cb(chunk: np.ndarray) -> None:
        if _collecting.is_set():
            buffer.push(chunk)
        vad.process_chunk(chunk)

    # ------------------------------------------------------------------
    # 발화 대기
    # ------------------------------------------------------------------
    print(f"\n  [턴 {turn_idx}] 말씀하세요... (발화 후 잠시 대기하면 자동 인식)")

    vad.reset()
    buffer.reset()

    speech_ended_q = bus.subscribe(EventType.SPEECH_ENDED)
    speech_started_q = bus.subscribe(EventType.SPEECH_STARTED)

    audio_mgr.add_input_callback(_audio_cb)

    t_listen_start = time.monotonic()

    try:
        # 최대 MAX_LISTEN_SECS 대기; 발화 시작도 안 되면 타임아웃
        try:
            await asyncio.wait_for(speech_started_q.get(), timeout=MAX_LISTEN_SECS)
            _collecting.set()  # 발화 시작 → 버퍼 수집 시작
            print(f"  [발화 감지됨] 말씀이 끝날 때까지 대기 중...")
        except asyncio.TimeoutError:
            print(f"  [경고] 발화를 감지하지 못했습니다 ({MAX_LISTEN_SECS}s 타임아웃).")
            return {"turn": turn_idx, "error": "speech_start_timeout"}

        # 발화 종료 대기
        try:
            await asyncio.wait_for(speech_ended_q.get(), timeout=MAX_LISTEN_SECS)
        except asyncio.TimeoutError:
            print(f"  [경고] 발화 종료를 감지하지 못했습니다.")
            return {"turn": turn_idx, "error": "speech_end_timeout"}
    finally:
        audio_mgr.remove_input_callback(_audio_cb)
        bus.unsubscribe(EventType.SPEECH_ENDED, speech_ended_q)
        bus.unsubscribe(EventType.SPEECH_STARTED, speech_started_q)

    listen_dur = time.monotonic() - t_listen_start

    # ------------------------------------------------------------------
    # STT
    # ------------------------------------------------------------------
    audio_data = buffer.get_all()
    buffer.reset()

    if len(audio_data) == 0:
        print("  [경고] 오디오 버퍼 비어있음")
        return {"turn": turn_idx, "error": "empty_buffer"}

    audio_dur = len(audio_data) / sample_rate
    print(f"  오디오: {audio_dur:.1f}s 수집됨 → STT 전사 중...")

    t_stt = time.monotonic()
    text = await stt.transcribe(audio_data)
    stt_dur = time.monotonic() - t_stt
    lang = stt.detected_language or "ko"

    if not text:
        print(f"  [경고] STT 결과 없음 (묵음? {stt_dur:.2f}s)")
        return {"turn": turn_idx, "error": "empty_transcription", "stt_dur": stt_dur}

    print(f"  STT ({stt_dur:.2f}s): [{lang}] \"{text}\"")

    # ------------------------------------------------------------------
    # LLM + TTS 파이프라인
    # ------------------------------------------------------------------
    context.add_user(text)

    pipeline = StreamingPipeline(handler, synth, audio_out, bus)
    timeline.start()
    t_pipeline = time.monotonic()

    try:
        full_response = await pipeline.run(context, detected_language=lang)
    except Exception as exc:
        print(f"  [오류] 파이프라인 실패: {exc}")
        await timeline.stop()
        return {
            "turn": turn_idx, "error": str(exc),
            "stt_dur": stt_dur, "text": text, "lang": lang,
        }

    pipeline_dur = time.monotonic() - t_pipeline

    await asyncio.sleep(0.1)
    await timeline.stop()

    context.add_assistant(full_response)

    # ------------------------------------------------------------------
    # 메트릭
    # ------------------------------------------------------------------
    ttfa        = get_ttfa(timeline.events)
    phrase_gaps = get_phrase_gaps(timeline.events)
    max_gap     = max(phrase_gaps) if phrase_gaps else None
    phrase_cnt  = sum(1 for _, n, _ in timeline.events if n == "PHRASE_READY")

    preview = full_response[:80].replace("\n", " ")
    print(f"  응답: \"{preview}{'...' if len(full_response) > 80 else ''}\"")
    print(f"  응답 길이: {len(full_response)}자")

    print(f"\n  이벤트 타임라인:")
    print_timeline(timeline.events)

    ttfa_str = f"{ttfa:.3f}s" if ttfa is not None else "N/A"
    ttfa_ok  = ttfa is not None and ttfa < TTFA_LIMIT
    gap_ok   = max_gap is None or max_gap < GAP_LIMIT
    print(f"\n  STT:       {stt_dur:.3f}s")
    print(f"  TTFA:      {ttfa_str} (목표 < {TTFA_LIMIT}s) → {'PASS' if ttfa_ok else 'FAIL'}")
    print(f"  총 응답:   {pipeline_dur:.3f}s")
    print(f"  phrase 수: {phrase_cnt}")
    if phrase_gaps:
        print(f"  phrase 갭: mean={np.mean(phrase_gaps):.3f}s, max={max_gap:.3f}s "
              f"(목표 < {GAP_LIMIT}s) → {'PASS' if gap_ok else 'FAIL'}")
    else:
        print(f"  phrase 갭: N/A (phrase 1개)")

    return {
        "turn": turn_idx, "error": None,
        "text": text, "lang": lang,
        "stt_dur": stt_dur, "audio_dur": audio_dur,
        "ttfa": ttfa, "pipeline_dur": pipeline_dur,
        "phrase_count": phrase_cnt, "phrase_gaps": phrase_gaps, "max_gap": max_gap,
        "response": full_response,
    }


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

async def run_test(num_turns: int = 3, no_play: bool = False) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("src.pipeline").setLevel(logging.INFO)

    cfg = Config()
    cfg._data.setdefault("llm", {})["provider"] = "custom"

    print("\n" + "=" * 65)
    print("HW-11: 마이크 → STT → Custom LLM → TTS → 스피커 전체 테스트")
    print("=" * 65)
    print(f"  LLM Provider:  custom (강제)")
    print(f"  Model ID:      {cfg.get('llm.custom_model_id', '(없음)')}")
    print(f"  TTS Provider:  {cfg.get('tts.provider', 'melo')}")
    print(f"  STT Model:     {cfg.get('stt.model_size', 'large-v3-turbo')}")
    print(f"  대화 턴 수:    {num_turns}")
    print(f"  재생 모드:     {'스피커 출력' if not no_play else '더미 (재생 없음)'}")

    if not check_config(cfg):
        return

    # 시스템 프롬프트
    try:
        with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        system_prompt = "You are a helpful voice assistant. Keep responses concise."
        print(f"  [경고] {SYSTEM_PROMPT_PATH} 없음 — 기본 프롬프트 사용")

    # ------------------------------------------------------------------
    # 컴포넌트 초기화
    # ------------------------------------------------------------------
    print("\n── 컴포넌트 초기화 ──")
    bus = EventBus()

    # AudioManager (항상 실제 마이크 입력 필요) + 출력용 분리
    audio_mgr = AudioManager(cfg)
    await audio_mgr.start()
    # 파이프라인 출력: no_play 시 더미, 아니면 실제 스피커
    audio_out: AudioManager | DummyAudioManager = DummyAudioManager() if no_play else audio_mgr

    sample_rate: int = cfg.get("audio.input_sample_rate", 16_000)
    buffer = AudioBuffer(
        sample_rate=sample_rate,
        buffer_seconds=cfg.get("audio.buffer_seconds", 30),
    )

    # VAD
    print("  VAD 로드 중...", end="", flush=True)
    t0 = time.monotonic()
    vad = VoiceActivityDetector(cfg, bus)
    print(f" {time.monotonic() - t0:.1f}초")

    # STT
    print("  STT (Whisper) 로드 중...", end="", flush=True)
    t0 = time.monotonic()
    stt = Transcriber(cfg, bus)
    print(f" {time.monotonic() - t0:.1f}초")

    # Custom LLM handler
    print("  Custom LLM handler 생성 중...", end="", flush=True)
    t0 = time.monotonic()
    try:
        handler = create_chat_handler(cfg, bus)
    except Exception as exc:
        print(f" 실패: {exc}")
        await audio_mgr.stop()
        return
    print(f" {time.monotonic() - t0:.1f}초")

    # TTS synthesizer
    print("  TTS synthesizer 로드 중...", end="", flush=True)
    t0 = time.monotonic()
    try:
        synth = create_synthesizer(cfg)
    except Exception as exc:
        print(f" 실패: {exc}")
        await handler.close()
        await audio_mgr.stop()
        return
    print(f" {time.monotonic() - t0:.1f}초")

    context = ConversationContext(
        system_prompt,
        max_context_tokens=cfg.get("llm.max_context_tokens", 3000),
    )
    timeline = TimelineCollector(bus)

    # ------------------------------------------------------------------
    # 대화 루프
    # ------------------------------------------------------------------
    print(f"\n── 대화 시작 ({num_turns}턴) ──")
    print("  (각 턴: 말씀하세요 → 발화 → 자동 인식 → 응답 재생)")

    results: list[dict] = []

    for i in range(1, num_turns + 1):
        print(f"\n{'─' * 55}")
        result = await run_turn(
            turn_idx=i,
            cfg=cfg, bus=bus,
            audio_mgr=audio_mgr, buffer=buffer,
            vad=vad, stt=stt,
            handler=handler, synth=synth,
            audio_out=audio_out,
            context=context,
            timeline=timeline,
            no_play=no_play,
        )
        results.append(result)

        if result.get("error"):
            print(f"  [오류] 턴 {i} 실패: {result['error']}")

    # ------------------------------------------------------------------
    # 정리
    # ------------------------------------------------------------------
    await audio_mgr.stop()
    if no_play:
        await audio_out.stop()
    await synth.close()
    await handler.close()

    # ------------------------------------------------------------------
    # 결과 요약
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  결과 요약")
    print("=" * 65)

    ok = [r for r in results if not r.get("error")]
    fail = [r for r in results if r.get("error")]

    if ok:
        ttfas = [r["ttfa"] for r in ok if r.get("ttfa") is not None]
        stt_durs = [r["stt_dur"] for r in ok if r.get("stt_dur") is not None]
        all_gaps: list[float] = []
        for r in ok:
            if r.get("phrase_gaps"):
                all_gaps.extend(r["phrase_gaps"])

        if ttfas:
            print(f"\n  TTFA (파이프라인 시작→첫 오디오):")
            print(f"    mean={np.mean(ttfas):.3f}s, max={max(ttfas):.3f}s  "
                  f"(목표 < {TTFA_LIMIT}s)")
            print(f"    판정: {'PASS' if max(ttfas) < TTFA_LIMIT else 'FAIL'}")

        if stt_durs:
            print(f"\n  STT 소요: mean={np.mean(stt_durs):.3f}s, max={max(stt_durs):.3f}s")

        if all_gaps:
            print(f"\n  phrase 간 갭:")
            print(f"    mean={np.mean(all_gaps):.3f}s, max={max(all_gaps):.3f}s  "
                  f"(목표 < {GAP_LIMIT}s)")
            print(f"    판정: {'PASS' if max(all_gaps) < GAP_LIMIT else 'FAIL'}")
        else:
            print(f"\n  phrase 간 갭: N/A")

    if fail:
        print(f"\n  오류: {len(fail)}턴 실패")
        for r in fail:
            print(f"    턴 {r['turn']}: {r['error']}")

    print(f"\n  ── 개별 턴 상세 ──")
    for r in results:
        if r.get("error"):
            print(f"    턴 {r['turn']} ERROR: {r['error']}")
        else:
            ttfa_s = f"TTFA={r['ttfa']:.3f}s" if r.get('ttfa') is not None else "TTFA=N/A"
            stt_s  = f"STT={r['stt_dur']:.2f}s" if r.get('stt_dur') is not None else ""
            gap_s  = f"max_gap={r['max_gap']:.3f}s" if r.get('max_gap') is not None else "gap=N/A"
            print(f"    턴 {r['turn']} [{r.get('lang','?')}] {stt_s}  {ttfa_s}  "
                  f"phrases={r.get('phrase_count',0)}  {gap_s}")
            text_s = r.get('text', '')[:40]
            resp_s = r.get('response', '')[:40].replace('\n', ' ')
            print(f"       Q: \"{text_s}{'...' if len(r.get('text','')) > 40 else ''}\"")
            print(f"       A: \"{resp_s}{'...' if len(r.get('response','')) > 40 else ''}\"")

    completeness = len(ok) == len(results)
    ttfa_pass = all(r.get("ttfa") is not None and r["ttfa"] < TTFA_LIMIT for r in ok) if ok else False
    gap_pass  = all(
        r.get("max_gap") is None or r["max_gap"] < GAP_LIMIT for r in ok
    ) if ok else True

    print(f"\n  ── 최종 판정 ──")
    print(f"  완전성:              {'PASS' if completeness else f'FAIL ({len(ok)}/{len(results)})'}")
    print(f"  TTFA < {TTFA_LIMIT}s:          {'PASS' if ttfa_pass else 'FAIL'}")
    print(f"  phrase 간 갭 < {GAP_LIMIT}s:   {'PASS' if gap_pass else 'FAIL'}")

    all_pass = completeness and ttfa_pass and gap_pass
    print(f"\n  최종: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print("=" * 65)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HW-11: 마이크 → STT → Custom LLM → TTS → 스피커 전체 테스트",
    )
    parser.add_argument(
        "--turns", type=int, default=3,
        help="대화 턴 수 (기본: 3)",
    )
    parser.add_argument(
        "--no-play", action="store_true",
        help="스피커 재생 없이 합성까지만 (디버그용)",
    )
    args = parser.parse_args()
    asyncio.run(run_test(num_turns=args.turns, no_play=args.no_play))
