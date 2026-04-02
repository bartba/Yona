"""web.py — FastAPI web UI server for Yona voice assistant.

Runs inside the existing asyncio event loop alongside YonaApp.
Serves a dark-theme HTML page and streams real-time conversation
events to browsers via Server-Sent Events (SSE).

Usage (from main.py)::

    web = WebServer(cfg, bus)
    asyncio.create_task(web.serve())
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from src.config import Config
from src.events import Event, EventBus, EventType
from src.state import ConversationState as CS

logger = logging.getLogger(__name__)


class _SuppressSSEShutdownWarning(logging.Filter):
    """Suppress 'ASGI callable returned without completing response' on shutdown.

    SSE streams are intentionally long-lived and will always appear incomplete
    when the server shuts down. This is expected, not an error.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "without completing response" not in record.getMessage()

# Korean labels shown in the status bar for each state.
_STATE_LABELS: dict[str, str] = {
    "IDLE":          "대기 중",
    "LISTENING":     "듣고 있어요...",
    "PROCESSING":    "처리 중...",
    "SPEAKING":      "말하는 중...",
    "TIMEOUT_CHECK": "아직 계세요?",
    "TIMEOUT_FINAL": "대화 종료 중...",
}


_HTML_PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Samsung Gauss Voice Assistant</title>
<link rel="preconnect" href="https://cdn.jsdelivr.net">
<link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0d1117;
    --bar-bg:    #161b22;
    --border:    #30363d;
    --user-bg:   #1f6feb;
    --asst-bg:   #21262d;
    --text:      #e6edf3;
    --subtext:   #8b949e;
    --radius:    18px;
    --font:      'Pretendard Variable', 'Inter', system-ui, -apple-system, sans-serif;
  }

  html, body {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 18px;
    -webkit-font-smoothing: antialiased;
  }

  body {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* ── Status bar ── */
  #statusbar {
    flex-shrink: 0;
    height: 76px;
    background: var(--bar-bg);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 28px;
    gap: 16px;
    user-select: none;
  }

  #app-title {
    font-size: 17px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text);
    flex: none;
    white-space: nowrap;
  }

  #divider {
    width: 1px;
    height: 24px;
    background: var(--border);
    flex: none;
  }

  #dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #6e7681;
    flex-shrink: 0;
    transition: background 0.3s ease;
  }

  #dot.pulse {
    animation: pulse 1.6s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0px currentColor; opacity: 1; }
    50%       { box-shadow: 0 0 0 7px transparent;  opacity: 0.75; }
  }

  #status-text {
    font-size: 20px;
    font-weight: 500;
    color: var(--subtext);
    transition: color 0.3s ease;
  }

  #status-text.thinking {
    animation: thinking-fade 1.5s ease-in-out infinite;
  }

  @keyframes thinking-fade {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.45; }
  }

  #conn-indicator {
    margin-left: auto;
    font-size: 14px;
    color: var(--subtext);
    display: flex;
    align-items: center;
    gap: 6px;
  }

  #conn-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #3fb950;
  }

  #conn-dot.offline { background: #f85149; }

  /* ── Conversation area ── */
  #conversation {
    flex: 1;
    overflow-y: auto;
    padding: 28px 0 80px 0;
    scroll-behavior: smooth;
  }

  /* Custom scrollbar */
  #conversation::-webkit-scrollbar { width: 6px; }
  #conversation::-webkit-scrollbar-track { background: transparent; }
  #conversation::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .message-row {
    display: flex;
    padding: 6px 28px;
  }

  .message-row.user    { justify-content: flex-end; }
  .message-row.assistant { justify-content: flex-start; }

  .bubble {
    max-width: 72%;
    padding: 14px 20px;
    border-radius: var(--radius);
    line-height: 1.65;
    word-break: break-word;
    white-space: pre-wrap;
    font-size: 18px;
    animation: fadeIn 0.2s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .bubble.user {
    background: var(--user-bg);
    color: #ffffff;
    border-bottom-right-radius: 4px;
  }

  .bubble.assistant {
    background: var(--asst-bg);
    color: var(--text);
    border-bottom-left-radius: 4px;
  }

  .cursor {
    display: inline-block;
    width: 2px;
    height: 1.1em;
    background: var(--subtext);
    vertical-align: text-bottom;
    margin-left: 2px;
    animation: blink 0.9s step-end infinite;
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
  }

  .label {
    font-size: 12px;
    color: var(--subtext);
    margin-bottom: 4px;
    padding: 0 4px;
  }

  .message-row.user    .label { text-align: right; }
  .message-row.assistant .label { text-align: left; }

  .message-col {
    display: flex;
    flex-direction: column;
    max-width: 72%;
  }

  /* Empty state */
  #empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 16px;
    color: var(--subtext);
    pointer-events: none;
    transition: opacity 0.3s ease;
  }

  #empty-icon { font-size: 56px; }
  #empty-text { font-size: 20px; font-weight: 500; }
</style>
</head>
<body>

<div id="statusbar">
  <span id="app-title">Samsung Gauss Voice Assistant</span>
  <div id="divider"></div>
  <div id="dot"></div>
  <span id="status-text">연결 중...</span>
  <div id="conn-indicator">
    <div id="conn-dot" class="offline"></div>
    <span id="conn-text">오프라인</span>
  </div>
</div>

<div id="conversation">
  <div id="empty">
    <div id="empty-icon">🎙️</div>
    <div id="empty-text">"Hey Mack" 이라고 말씀하세요</div>
  </div>
</div>

<script>
const conv       = document.getElementById('conversation');
const emptyEl    = document.getElementById('empty');
const emptyIcon  = document.getElementById('empty-icon');
const emptyText  = document.getElementById('empty-text');
const dot        = document.getElementById('dot');
const statusText = document.getElementById('status-text');
const connDot    = document.getElementById('conn-dot');
const connText   = document.getElementById('conn-text');

const STATE_COLORS = {
  IDLE:          '#6e7681',
  LISTENING:     '#3fb950',
  PROCESSING:    '#d29922',
  SPEAKING:      '#58a6ff',
  TIMEOUT_CHECK: '#f85149',
  TIMEOUT_FINAL: '#f85149',
};
const PULSE_STATES = new Set(['LISTENING', 'PROCESSING', 'SPEAKING']);

let currentBubble        = null;   // active assistant bubble element
let userScrolled         = false;  // true when user has manually scrolled up
let userBubbleThisTurn   = false;  // dedup: one user bubble per turn
let phraseCount          = 0;      // phrases appended to current assistant bubble
let thinkingTimer        = null;   // interval ID for the ellipsis animation

// ── Scroll helpers ──────────────────────────────────────────────────────────

conv.addEventListener('scroll', () => {
  userScrolled = conv.scrollTop + conv.clientHeight < conv.scrollHeight - 120;
});

function scrollToBottom() {
  if (!userScrolled) {
    // Use requestAnimationFrame so the DOM has fully painted before measuring
    requestAnimationFrame(() => {
      conv.scrollTop = conv.scrollHeight;
    });
  }
}

// ── Status bar ───────────────────────────────────────────────────────────────

function startThinkingAnimation() {
  if (thinkingTimer !== null) return;
  const frames = ['처리 중.', '처리 중..', '처리 중...'];
  let phase = 0;
  statusText.textContent = frames[0];
  statusText.classList.add('thinking');
  thinkingTimer = setInterval(() => {
    phase = (phase + 1) % frames.length;
    statusText.textContent = frames[phase];
  }, 600);
}

function stopThinkingAnimation() {
  if (thinkingTimer !== null) {
    clearInterval(thinkingTimer);
    thinkingTimer = null;
  }
  statusText.classList.remove('thinking');
}

function updateStatus(state, label) {
  stopThinkingAnimation();
  const color = STATE_COLORS[state] || '#6e7681';
  dot.style.background = color;
  dot.style.color = color;
  statusText.textContent = label;
  if (PULSE_STATES.has(state)) {
    dot.classList.add('pulse');
  } else {
    dot.classList.remove('pulse');
  }
}

function setOnline(online) {
  if (online) {
    connDot.classList.remove('offline');
    connText.textContent = '연결됨';
  } else {
    connDot.classList.add('offline');
    connText.textContent = '오프라인';
  }
}

// ── Empty state ──────────────────────────────────────────────────────────────

let emptyVisible = true;

function updateEmptyState(state) {
  if (!emptyVisible) return;
  if (state === 'IDLE') {
    emptyIcon.textContent = '🎙️';
    emptyText.textContent = '"Hey Mack" 이라고 말씀하세요';
  } else if (state === 'LISTENING') {
    emptyIcon.textContent = '👂';
    emptyText.textContent = '말씀해 주세요...';
  } else if (state === 'PROCESSING') {
    emptyIcon.textContent = '💭';
    emptyText.textContent = '처리 중...';
  }
}

function hideEmpty() {
  if (emptyEl && emptyEl.parentNode) {
    emptyEl.parentNode.removeChild(emptyEl);
    emptyVisible = false;
  }
}

// ── Message bubbles ──────────────────────────────────────────────────────────

function addUserBubble(text) {
  if (userBubbleThisTurn) return;  // dedup: only one user bubble per turn
  userBubbleThisTurn = true;

  hideEmpty();
  finaliseAssistantBubble();

  const row = document.createElement('div');
  row.className = 'message-row user';

  const col = document.createElement('div');
  col.className = 'message-col';

  const lbl = document.createElement('div');
  lbl.className = 'label';
  lbl.textContent = 'You';

  const bubble = document.createElement('div');
  bubble.className = 'bubble user';
  bubble.textContent = text;

  col.appendChild(lbl);
  col.appendChild(bubble);
  row.appendChild(col);
  conv.appendChild(row);
  userScrolled = false;
  scrollToBottom();
}

function startAssistantBubble() {
  hideEmpty();
  finaliseAssistantBubble();

  const row = document.createElement('div');
  row.className = 'message-row assistant';
  row.id = 'active-row';

  const col = document.createElement('div');
  col.className = 'message-col';

  const lbl = document.createElement('div');
  lbl.className = 'label';
  lbl.textContent = 'Gauss';

  const bubble = document.createElement('div');
  bubble.className = 'bubble assistant';

  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  cursor.id = 'cursor';
  bubble.appendChild(cursor);

  col.appendChild(lbl);
  col.appendChild(bubble);
  row.appendChild(col);
  conv.appendChild(row);

  currentBubble = bubble;
  phraseCount = 0;
  userScrolled = false;
  scrollToBottom();
}

function appendPhrase(text) {
  if (!currentBubble) return;
  const cursor = document.getElementById('cursor');
  // Add a space between consecutive phrases
  const content = phraseCount > 0 ? ' ' + text : text;
  phraseCount++;
  if (cursor) {
    cursor.before(document.createTextNode(content));
  } else {
    currentBubble.appendChild(document.createTextNode(content));
  }
  scrollToBottom();
}

function finaliseAssistantBubble() {
  if (!currentBubble) return;
  const cursor = document.getElementById('cursor');
  if (cursor) cursor.remove();
  currentBubble = null;
  phraseCount = 0;
}

// ── SSE connection ────────────────────────────────────────────────────────────

function connect() {
  const source = new EventSource('/events');

  source.onopen = () => {
    setOnline(true);
    updateStatus('IDLE', '대기 중');
  };

  source.onerror = () => {
    setOnline(false);
    finaliseAssistantBubble();
  };

  source.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }

    switch (msg.type) {
      case 'state':
        // PROCESSING: LLM/STT working — start ellipsis animation
        // SPEAKING: pipeline started but audio not yet playing — keep "처리 중..." animation
        // First PHRASE event (actual audio start) will stop it and show "말하는 중..."
        if (msg.state === 'PROCESSING') {
          updateStatus('PROCESSING', '처리 중...');
          startThinkingAnimation();
        } else if (msg.state === 'SPEAKING') {
          // Stay in thinking animation — don't update status yet
          if (thinkingTimer === null) {
            updateStatus('PROCESSING', '처리 중...');
            startThinkingAnimation();
          }
        } else {
          updateStatus(msg.state, msg.label);
        }
        updateEmptyState(msg.state);
        if (msg.state === 'LISTENING') {
          userBubbleThisTurn = false;
        }
        break;
      case 'user':
        addUserBubble(msg.text);
        break;
      case 'phrase':
        // First phrase = audio genuinely starting — stop animation, show "말하는 중..."
        if (!currentBubble) {
          stopThinkingAnimation();
          updateStatus('SPEAKING', '말하는 중...');
          startAssistantBubble();
        }
        appendPhrase(msg.text);
        break;
      case 'assistant_done':
        finaliseAssistantBubble();
        break;
      case 'status':
        statusText.textContent = msg.message;
        break;
    }
  };
}

connect();
</script>
</body>
</html>"""


class WebServer:
    """FastAPI + uvicorn SSE server integrated into the Yona asyncio event loop.

    Subscribes to EventBus events and broadcasts them to all connected
    browser clients via Server-Sent Events. Runs in the same asyncio
    event loop as YonaApp (no threads or subprocesses).
    """

    def __init__(self, cfg: Config, bus: EventBus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._clients: set[asyncio.Queue[str | None]] = set()
        self._app = FastAPI(title="Yona Web UI")
        self._server: uvicorn.Server | None = None
        self._bridge_task: asyncio.Task | None = None
        self._register_routes()

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def _register_routes(self) -> None:
        app = self._app

        @app.get("/")
        async def index() -> HTMLResponse:
            return HTMLResponse(content=_HTML_PAGE, status_code=200)

        @app.get("/events")
        async def events(request: Request) -> EventSourceResponse:
            return EventSourceResponse(
                self._client_generator(request),
                ping=15,  # keep-alive ping every 15 s (prevents proxy timeouts)
            )

    # ------------------------------------------------------------------
    # SSE generator — one per connected client
    # ------------------------------------------------------------------

    async def _client_generator(
        self, request: Request
    ) -> AsyncIterator[dict[str, str]]:
        """Yield SSE messages until the client disconnects."""
        q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)
        self._clients.add(q)
        logger.info("SSE client connected (total=%d)", len(self._clients))
        try:
            while True:
                try:
                    if await request.is_disconnected():
                        break
                    payload = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except (asyncio.CancelledError, RuntimeError):
                    break  # server shutting down — exit cleanly
                if payload is None:
                    break  # sentinel: server shutting down
                try:
                    yield {"data": payload}
                except (GeneratorExit, asyncio.CancelledError):
                    return  # consumer closed the stream — exit cleanly
        except (GeneratorExit, asyncio.CancelledError):
            pass
        finally:
            self._clients.discard(q)
            logger.info("SSE client disconnected (total=%d)", len(self._clients))

    # ------------------------------------------------------------------
    # Broadcast helpers
    # ------------------------------------------------------------------

    def _broadcast(self, payload: dict[str, Any]) -> None:
        """Serialise *payload* and enqueue it to all connected clients.

        Clients whose queues are full are silently evicted.
        """
        text = json.dumps(payload, ensure_ascii=False)
        dead: list[asyncio.Queue[str | None]] = []
        for q in self._clients:
            try:
                q.put_nowait(text)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._clients.discard(q)

    # ------------------------------------------------------------------
    # EventBus bridge — translates bus events to SSE payloads
    # ------------------------------------------------------------------

    async def _bridge(self) -> None:
        """Subscribe to all relevant EventBus events and broadcast to SSE clients."""
        event_types = (
            EventType.STATE_CHANGED,
            EventType.TRANSCRIPTION_READY,
            EventType.PHRASE_PLAYING,
            EventType.PLAYBACK_DONE,
            EventType.WAKE_WORD_DETECTED,
            EventType.BARGE_IN_DETECTED,
            EventType.SHUTDOWN,
        )
        subscriptions = {et: self._bus.subscribe(et) for et in event_types}

        async def _drain(et: EventType) -> None:
            q = subscriptions[et]
            while True:
                event = await q.get()
                self._handle_event(event)

        tasks = [asyncio.create_task(_drain(et)) for et in event_types]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for et, q in subscriptions.items():
                self._bus.unsubscribe(et, q)

    def _handle_event(self, event: Event) -> None:
        """Translate a single EventBus event into an SSE payload and broadcast."""
        et = event.type

        if et == EventType.STATE_CHANGED:
            state: CS = event.data
            name = state.name if hasattr(state, "name") else str(state)
            label = _STATE_LABELS.get(name, name)
            self._broadcast({"type": "state", "state": name, "label": label})

        elif et == EventType.TRANSCRIPTION_READY:
            self._broadcast({"type": "user", "text": event.data or ""})

        elif et == EventType.PHRASE_PLAYING:
            self._broadcast({"type": "phrase", "text": event.data or ""})

        elif et == EventType.PLAYBACK_DONE:
            self._broadcast({"type": "assistant_done"})

        elif et == EventType.WAKE_WORD_DETECTED:
            self._broadcast({"type": "status", "message": "Wake word detected"})

        elif et == EventType.BARGE_IN_DETECTED:
            self._broadcast({"type": "status", "message": "Barge-in detected"})

        elif et == EventType.SHUTDOWN:
            self._broadcast({"type": "status", "message": "Server shutting down"})
            for q in list(self._clients):
                try:
                    q.put_nowait(None)  # sentinel to close client generators
                except asyncio.QueueFull:
                    pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def serve(self) -> None:
        """Start the EventBus bridge task and uvicorn HTTP server.

        Designed to be launched as: ``asyncio.create_task(web.serve())``
        """
        host = self._cfg.get("web.host", "0.0.0.0")
        port = self._cfg.get("web.port", 8080)

        self._bridge_task = asyncio.create_task(self._bridge())

        # Suppress the expected "incomplete response" warning for SSE on shutdown
        logging.getLogger("uvicorn.error").addFilter(_SuppressSSEShutdownWarning())

        uv_cfg = uvicorn.Config(
            app=self._app,
            host=host,
            port=port,
            log_level="warning",  # suppress noisy access logs
            loop="none",          # CRITICAL: reuse the existing asyncio loop
            lifespan="off",       # disable ASGI lifespan (not used; avoids dangling task on shutdown)
        )
        self._server = uvicorn.Server(uv_cfg)
        logger.info("Web UI available at http://%s:%d", host, port)
        try:
            await self._server.serve()
        except OSError as exc:
            logger.error("Web UI failed to start (port %d in use?): %s", port, exc)
        finally:
            if self._bridge_task and not self._bridge_task.done():
                self._bridge_task.cancel()
                try:
                    await self._bridge_task
                except asyncio.CancelledError:
                    pass

    async def shutdown(self) -> None:
        """Signal uvicorn to stop and cancel the bridge task."""
        if self._server:
            self._server.should_exit = True
        if self._bridge_task and not self._bridge_task.done():
            self._bridge_task.cancel()
            try:
                await self._bridge_task
            except asyncio.CancelledError:
                pass
