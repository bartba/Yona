"""HW-08e: Samsung Gauss (Custom LLM) 스트리밍 라이브 테스트.

별도 터미널에서 실행:
    python tests/test_hw_custom_llm.py
    python tests/test_hw_custom_llm.py --model-id <UUID>   # 모델 ID 오버라이드

항상 provider=custom 으로 강제 실행 (LLM_PROVIDER 환경변수 무시).

환경 변수 (.env 또는 export):
    CUSTOM_LLM_URL=https://...
    CUSTOM_LLM_KEY=...
    CUSTOM_LLM_CLIENT_TOKEN=...
    CUSTOM_LLM_MODEL_ID=...
    CUSTOM_LLM_USER_EMAIL=...    (선택)

테스트 항목:
    Phase 1. Gauss API 설정 확인 (URL / 키 / 모델 ID)
    Phase 2. 스트리밍 3회 (한국어 → 영어 → 한국어) — TTFT / 토큰속도 측정
    성공 기준: TTFT < 3초 (사내 네트워크 기준), 3회 모두 오류 없이 완료
    ※ 사내 정책: 분당 최대 3회 호출
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

_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "prompts", "system_prompt.txt",
)

# ---------------------------------------------------------------------------
# 테스트 프롬프트 — Phase 2 (단일 턴)
# ---------------------------------------------------------------------------

SINGLE_PROMPTS = [
    {
        "id": 1,
        "lang": "ko",
        "text": "삼성전자의 주요 사업 분야를 세 가지만 간단히 알려줘.",
        "desc": "한국어 / 회사 지식",
    },
    {
        "id": 2,
        "lang": "en",
        "text": "What is one practical tip to improve work-life balance for software engineers? One sentence.",
        "desc": "English / lifestyle",
    },
    {
        "id": 3,
        "lang": "ko",
        "text": "인공지능이 앞으로 5년 안에 가장 크게 바꿀 직업 하나를 꼽고, 이유를 두 문장으로 설명해줘.",
        "desc": "한국어 / 미래 예측",
    },
]

def _load_system_prompt() -> str:
    try:
        with open(_PROMPT_PATH, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return "You are Samsung Gauss, a friendly voice assistant. Respond naturally and concisely."

SYSTEM_PROMPT = _load_system_prompt()

TTFT_LIMIT = 3.0  # seconds


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _mask(value: str, show: int = 6) -> str:
    """API 키 등 민감한 값 앞 N자만 보이고 나머지 마스킹."""
    if not value:
        return "(없음)"
    return value[:show] + "*" * max(0, len(value) - show)


# ---------------------------------------------------------------------------
# Phase 1: 설정 확인
# ---------------------------------------------------------------------------

def phase1_config_check(cfg: Config) -> bool:
    """Gauss API 설정 값을 출력하고 필수 항목 누락 시 False 반환."""
    print("\n── Phase 1: Gauss API 설정 확인 ──")

    url            = cfg.get("llm.custom_url", "")
    key            = cfg.get("llm.custom_key", "")
    client_token   = cfg.get("llm.custom_client_token", "")
    model_id       = cfg.get("llm.custom_model_id", "")
    user_email     = cfg.get("llm.custom_user_email", "")
    max_tokens     = cfg.get("llm.max_tokens", 1024)
    temperature    = cfg.get("llm.temperature", 1.0)

    print(f"  URL:            {url or '(없음)'}")
    print(f"  API Key:        {_mask(key)}")
    print(f"  Client Token:   {_mask(client_token)}")
    print(f"  Model ID:       {model_id or '(없음)'}")
    print(f"  User Email:     {user_email or '(없음, 선택 항목)'}")
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
# Phase 2: 단일 턴 스트리밍
# ---------------------------------------------------------------------------

async def phase2_single_turn(cfg: Config, bus: EventBus) -> tuple[bool, list[dict]]:
    """단일 턴 프롬프트 스트리밍 테스트. (pass, results) 반환."""
    print(f"\n── Phase 2: 스트리밍 테스트 (ko → en → ko, 총 {len(SINGLE_PROMPTS)}회) ──\n")

    try:
        handler = create_chat_handler(cfg, bus)
    except Exception as exc:
        print(f"  [오류] handler 생성 실패: {exc}")
        return False, []

    results: list[dict] = []

    for prompt in SINGLE_PROMPTS:
        print(f"  ── #{prompt['id']}/{len(SINGLE_PROMPTS)} [{prompt['lang']}] {prompt['desc']} ──")
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
            elapsed = time.monotonic() - t_start
            print(f"\n  [오류] 스트리밍 실패: {exc}")
            results.append({
                "id": prompt["id"], "lang": prompt["lang"],
                "ttft": None, "tokens": 0, "tps": 0.0,
                "elapsed": elapsed, "error": str(exc),
                "response": "",
            })
            continue

        elapsed = time.monotonic() - t_start
        full_response = "".join(tokens)
        token_count = len(tokens)
        tps = token_count / elapsed if elapsed > 0 else 0.0

        print()
        ttft_str = f"{ttft:.3f}s" if ttft is not None else "N/A"
        ttft_ok = ttft is not None and ttft < TTFT_LIMIT
        ttft_label = (
            f"PASS ✓ (< {TTFT_LIMIT}s)" if ttft_ok
            else (f"FAIL ✗ ({ttft:.2f}s ≥ {TTFT_LIMIT}s)" if ttft is not None else "FAIL ✗ (토큰 없음)")
        )
        print(f"\n  ▶ TTFT: {ttft_str}  |  토큰: {token_count}개  |  속도: {tps:.1f} tok/s  |  총시간: {elapsed:.2f}s")
        print(f"  ▶ TTFT 판정: {ttft_label}\n")

        results.append({
            "id": prompt["id"], "lang": prompt["lang"],
            "ttft": ttft, "tokens": token_count, "tps": tps,
            "elapsed": elapsed, "error": None,
            "response": full_response,
        })

    await handler.close()

    ok = [r for r in results if r["error"] is None and r["ttft"] is not None]
    all_pass = (
        len(ok) == len(SINGLE_PROMPTS)
        and all(r["ttft"] < TTFT_LIMIT for r in ok)
    )
    return all_pass, results


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

async def run_test(model_id_override: str | None = None) -> None:
    cfg = Config()
    bus = EventBus()

    # 항상 custom provider 강제
    cfg._data.setdefault("llm", {})["provider"] = "custom"

    if model_id_override:
        cfg._data["llm"]["custom_model_id"] = model_id_override

    print("\n" + "=" * 60)
    print("HW-08e: Samsung Gauss (Custom LLM) 스트리밍 라이브 테스트")
    print("=" * 60)

    # Phase 1
    if not phase1_config_check(cfg):
        return

    # Phase 2
    p2_pass, p2_results = await phase2_single_turn(cfg, bus)

    # ==================================================================
    # 결과 요약
    # ==================================================================
    print("\n" + "=" * 60)
    print("  결과 요약")
    print("=" * 60)

    ok_results = [r for r in p2_results if r["error"] is None and r["ttft"] is not None]
    fail_results = [r for r in p2_results if r["error"] is not None]

    if ok_results:
        ttfts = [r["ttft"] for r in ok_results]
        tpss  = [r["tps"]  for r in ok_results]
        ttft_mean = sum(ttfts) / len(ttfts)
        ttft_max  = max(ttfts)
        tps_mean  = sum(tpss)  / len(tpss)
        print(f"\n  TTFT: mean={ttft_mean:.3f}s, max={ttft_max:.3f}s (목표: < {TTFT_LIMIT}s)")
        print(f"  토큰 속도: mean={tps_mean:.1f} tok/s")
    else:
        print(f"\n  TTFT: 측정 불가 (모든 프롬프트 오류)")

    if fail_results:
        print(f"\n  오류: {len(fail_results)}개 프롬프트 실패")
        for r in fail_results:
            print(f"    #{r['id']} [{r['lang']}]: {r['error']}")

    print(f"\n  ── 개별 상세 ──")
    for r in p2_results:
        if r["error"]:
            print(f"    #{r['id']} [{r['lang']}] ERROR: {r['error']}")
        elif r["ttft"] is None:
            print(f"    #{r['id']} [{r['lang']}] 응답 없음 ({r['elapsed']:.2f}s)")
        else:
            preview = r["response"][:80].replace("\n", " ")
            suffix = "..." if len(r["response"]) > 80 else ""
            print(f"    #{r['id']} [{r['lang']}] TTFT={r['ttft']:.3f}s  {r['tps']:.1f} tok/s  {r['tokens']}tok")
            print(f"       응답: \"{preview}{suffix}\"")

    print(f"\n  최종: {'PASS ✓' if p2_pass else 'FAIL ✗'}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW-08e: Samsung Gauss 스트리밍 라이브 테스트")
    parser.add_argument(
        "--model-id", type=str, default=None,
        help="CUSTOM_LLM_MODEL_ID 오버라이드 (예: 0196f1fc-2858-70a9-...). 미지정 시 .env 값 사용.",
    )
    args = parser.parse_args()
    asyncio.run(run_test(model_id_override=args.model_id))
