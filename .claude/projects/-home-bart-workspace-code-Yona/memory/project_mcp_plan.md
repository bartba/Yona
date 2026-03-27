---
name: MongoDB MCP integration plan
description: Future plan to connect MongoDB MCP to Yona chatbot — LLM outputs structured data for MCP, then speaks the result in natural speech
type: project
---

MongoDB MCP를 Yona 챗봇에 연결하여 사용 예정.

**구현 방향:**
- LLM이 구조화된 데이터(MCP tool call)와 음성 출력을 분리하여 출력
- 구조화 데이터 → MCP 서버 전달 → 응답 수신
- MCP 응답 내용을 구어체로 변환 → TTS 파이프라인으로 발화

**Why:** 사용자가 자연어로 MongoDB 데이터를 조회/조작하고 음성으로 결과를 받기 위함

**How to apply:**
- 구현 시 시스템 프롬프트를 듀얼 모드로 설계 (구조화 출력 경로 vs 음성 출력 경로)
- 현재 TTS 텍스트 정규화(`_clean_tts_text`)와 프롬프트의 부호 사용 금지는 음성 경로에만 적용
- 구조화 데이터 경로는 JSON/tool_call 형식으로 정규화 제약 없이 처리
- pipeline.py에 MCP 분기 로직 추가 필요
