"""HW-07: LLM 스트리밍 라이브 테스트.

별도 터미널에서 실행:
    python tests/test_hw_llm.py
    python tests/test_hw_llm.py --model gpt-5-nano   # 모델 오버라이드

환경 변수 (필요에 따라 .env 또는 export):
    LLM_PROVIDER=openai       # "openai" | "claude" | "custom"
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini
    CLAUDE_API_KEY=sk-ant-...
    CLAUDE_MODEL=claude-sonnet-4-6

테스트 항목:
    Phase 1. 설정된 provider + 인증 확인
    Phase 2. 4개 프롬프트 스트리밍 — TTFT / 토큰속도 / 응답 품질 측정
    성공 기준: TTFT < 2초, 자연스러운 대화체
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.events import EventBus
from src.llm import ConversationContext, create_chat_handler


# ---------------------------------------------------------------------------
# 테스트 프롬프트
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    {
        "id": 1,
        "lang": "en",
        "text": "What's one renewable energy technology most likely to dominate global power grids by 2035, and why? Two sentences max.",
        "desc": "Energy / future prediction",
    },
    {
        "id": 2,
        "lang": "en",
        "text": "Name one unexpected industry where humanoid robots could have a bigger impact than manufacturing, and explain briefly.",
        "desc": "Robotics / lateral thinking",
    },
    {
        "id": 3,
        "lang": "en",
        "text": "What's a simple daily habit backed by neuroscience that most people overlook for improving focus and memory?",
        "desc": "Neuroscience / lifestyle",
    },
    {
        "id": 4,
        "lang": "en",
        "text": "If you could redesign one broken system in modern cities to make them more liveable, what would it be?",
        "desc": "Urban design / creative",
    },
    {
        "id": 5,
        "lang": "en",
        "text": "What's the most underrated programming language right now, and what makes it special?",
        "desc": "Software / opinion",
    },
    {
        "id": 6,
        "lang": "en",
        "text": "Tell me one surprising fact about the ocean that most people have never heard of.",
        "desc": "Science / curiosity",
    },
]

SYSTEM_PROMPT = (
    "You are Samsung Gauss, a friendly and witty conversational AI assistant. "
    "Keep your answers concise, natural, and engaging — like talking to a knowledgeable friend. "
    "Respond in English."
)


# ---------------------------------------------------------------------------
# 메인 테스트
# ---------------------------------------------------------------------------

async def run_test(model_override: str | None = None) -> None:
    cfg = Config()
    bus = EventBus()

    provider = cfg.get("llm.provider", "custom") or "custom"

    # 모델 오버라이드 (OpenAI provider일 때만 적용)
    if model_override and provider == "openai":
        cfg._data["llm"]["openai_model"] = model_override

    print("\n" + "=" * 60)
    print("HW-07: LLM 스트리밍 라이브 테스트")
    print("=" * 60)

    # ==================================================================
    # Phase 1: 설정 확인
    # ==================================================================
    print("\n── Phase 1: 설정 확인 ──")
    print(f"  Provider: {provider}")
    if provider == "openai":
        model = cfg.get("llm.openai_model", "?")
        key_set = bool(cfg.get("llm.openai_api_key", ""))
        print(f"  Model:    {model}{' (--model 오버라이드)' if model_override else ''}")
        print(f"  API Key:  {'설정됨 ✓' if key_set else '없음 ✗ — OPENAI_API_KEY 확인'}")
        if not key_set:
            print("  [오류] API 키가 없으면 테스트를 진행할 수 없습니다.")
            return
    elif provider == "claude":
        model = cfg.get("llm.claude_model", "?")
        key_set = bool(cfg.get("llm.claude_api_key", ""))
        print(f"  Model:    {model}")
        print(f"  API Key:  {'설정됨 ✓' if key_set else '없음 ✗ — CLAUDE_API_KEY 확인'}")
        if not key_set:
            print("  [오류] API 키가 없으면 테스트를 진행할 수 없습니다.")
            return
    elif provider == "custom":
        url = cfg.get("llm.custom_url", "")
        key_set = bool(cfg.get("llm.custom_key", ""))
        print(f"  URL:      {url or '(없음)'}")
        print(f"  API Key:  {'설정됨 ✓' if key_set else '없음 ✗ — CUSTOM_LLM_KEY 확인'}")
        if not url:
            print("  [오류] CUSTOM_LLM_URL이 없으면 테스트를 진행할 수 없습니다.")
            return
    else:
        print(f"  [오류] 알 수 없는 provider: {provider!r}")
        return

    print(f"  max_tokens={cfg.get('llm.max_tokens', 1024)}, temperature={cfg.get('llm.temperature', 0.7)}")

    # ==================================================================
    # Phase 2: 4개 프롬프트 스트리밍
    # ==================================================================
    print(f"\n── Phase 2: 스트리밍 테스트 ({len(TEST_PROMPTS)}개 프롬프트) ──\n")

    try:
        handler = create_chat_handler(cfg, bus)
    except Exception as exc:
        print(f"  [오류] handler 생성 실패: {exc}")
        return

    results: list[dict] = []

    for prompt in TEST_PROMPTS:
        print(f"  ── 프롬프트 {prompt['id']}/{len(TEST_PROMPTS)}: {prompt['desc']} ──")
        print(f"  입력: \"{prompt['text']}\"")
        print(f"  응답: ", end="", flush=True)

        context = ConversationContext(SYSTEM_PROMPT)
        context.add_user(prompt["text"])

        tokens: list[str] = []
        ttft: float | None = None
        t_start = time.monotonic()

        try:
            async for token in handler.stream(context):
                if ttft is None:
                    ttft = time.monotonic() - t_start
                tokens.append(token)
                print(token, end="", flush=True)
        except Exception as exc:
            print(f"\n  [오류] 스트리밍 실패: {exc}")
            results.append({
                "id": prompt["id"], "lang": prompt["lang"],
                "prompt": prompt["text"],
                "response": "",
                "ttft": None, "tokens": 0, "tps": 0.0,
                "elapsed": time.monotonic() - t_start,
                "error": str(exc),
            })
            print()
            continue

        elapsed = time.monotonic() - t_start
        full_response = "".join(tokens)
        token_count = len(tokens)
        tps = token_count / elapsed if elapsed > 0 else 0.0

        print()  # 응답 줄바꿈
        ttft_str = f"{ttft:.3f}초" if ttft is not None else "N/A"
        print(f"\n  ▶ TTFT: {ttft_str}  |  토큰: {token_count}개  |  속도: {tps:.1f} tok/s  |  총시간: {elapsed:.2f}초")
        ttft_ok = ttft is not None and ttft < 2.0
        ttft_label = f"PASS ✓ (< 2s)" if ttft_ok else (f"FAIL ✗ ({ttft:.2f}s ≥ 2s)" if ttft is not None else "FAIL ✗ (토큰 없음)")
        print(f"  ▶ TTFT 판정: {ttft_label}")
        print()

        results.append({
            "id": prompt["id"], "lang": prompt["lang"],
            "prompt": prompt["text"],
            "response": full_response,
            "ttft": ttft, "tokens": token_count, "tps": tps,
            "elapsed": elapsed,
            "error": None,
        })

    await handler.close()

    # ==================================================================
    # 결과 요약
    # ==================================================================
    print("=" * 60)
    print("  결과 요약")
    print("=" * 60)

    ok_results = [r for r in results if r["error"] is None and r["ttft"] is not None]
    fail_results = [r for r in results if r["error"] is not None]

    if ok_results:
        ttfts = [r["ttft"] for r in ok_results]
        tpss  = [r["tps"]  for r in ok_results]
        ttft_mean = sum(ttfts) / len(ttfts)
        ttft_max  = max(ttfts)
        tps_mean  = sum(tpss)  / len(tpss)

        ttft_ok = ttft_max < 2.0
        print(f"\n  TTFT: mean={ttft_mean:.3f}s, max={ttft_max:.3f}s (목표: < 2.0s)")
        print(f"  TTFT 판정: {'PASS ✓' if ttft_ok else 'FAIL ✗'}")
        print(f"\n  토큰 속도: mean={tps_mean:.1f} tok/s")
    else:
        ttft_ok = False
        print(f"\n  TTFT: 측정 불가 (오류 발생)")

    if fail_results:
        print(f"\n  오류 발생: {len(fail_results)}개 프롬프트 실패")
        for r in fail_results:
            print(f"    #{r['id']}: {r['error']}")

    # 개별 상세
    print(f"\n  ── 개별 프롬프트 상세 ──")
    for r in results:
        if r["error"]:
            print(f"    #{r['id']} [{r['lang']}] ERROR: {r['error']}")
        elif r["ttft"] is None:
            print(f"    #{r['id']} [{r['lang']}] 응답 없음 (빈 스트림, {r['elapsed']:.2f}s)")
        else:
            print(f"    #{r['id']} [{r['lang']}] TTFT={r['ttft']:.3f}s  {r['tps']:.1f} tok/s  {r['tokens']}tok")
            # 응답 첫 80자
            preview = r["response"][:80].replace("\n", " ")
            suffix = "..." if len(r["response"]) > 80 else ""
            print(f"       응답: \"{preview}{suffix}\"")

    # 최종 판정
    response_quality_ok = len(ok_results) == len(TEST_PROMPTS)  # 모두 성공

    print(f"\n  ── 최종 판정 ──")
    print(f"  TTFT < 2s:       {'PASS ✓' if ttft_ok else 'FAIL ✗'}")
    print(f"  응답 완전성:     {'PASS ✓' if response_quality_ok else f'FAIL ✗ ({len(ok_results)}/{len(TEST_PROMPTS)}개 성공)'}")

    all_pass = ttft_ok and response_quality_ok
    print(f"\n  최종: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print("=" * 60)
    print()
    if all_pass:
        print("  [참고] 응답 자연스러움은 위 응답 내용을 직접 확인하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW-07: LLM 스트리밍 라이브 테스트")
    parser.add_argument(
        "--model", type=str, default=None,
        help="OpenAI 모델 오버라이드 (예: gpt-5-nano, gpt-4o-mini). 미지정 시 config 값 사용.",
    )
    args = parser.parse_args()
    asyncio.run(run_test(model_override=args.model))
