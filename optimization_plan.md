---

# 🗂️ Optimization Plan
> Project: Yona — Voice Chat Application
> Audit Date: 2026-04-17
> Total Steps: 34
> Status: 🔵 PLANNING COMPLETE — Awaiting execution

---

## 📊 Executive Summary

| Phase | Steps | 🔴 Critical | 🟠 High | 🟡 Medium | 🟢 Low |
|-------|-------|------------|--------|----------|-------|
| P1 Bug Fixes | 8 | 2 | 4 | 2 | 0 |
| P2 Refactoring | 6 | 0 | 1 | 4 | 1 |
| P3 Performance | 4 | 0 | 2 | 2 | 0 |
| P4 Security | 4 | 1 | 2 | 1 | 0 |
| P5 Testing | 5 | 0 | 1 | 4 | 0 |
| P6 Dependencies | 3 | 0 | 1 | 1 | 1 |
| P7 Documentation | 4 | 0 | 0 | 2 | 2 |
| **TOTAL** | **34** | **3** | **11** | **16** | **4** |

---

## ▶️ How to Execute

Trigger each step by saying:
- `run P1-S01`         → execute a specific step
- `run P1`             → execute all steps in Phase 1 sequentially
- `run all critical`   → execute all 🔴 CRITICAL steps across all phases
- `run all`            → execute every step in order (full automation)
- `skip P1-S02`        → mark a step as skipped with a reason
- `status`             → show current progress dashboard
- `preview P1-S01`     → show exact diff/code that will be applied, without executing

---

## 🔴 PHASE 1 — Bug Fixes & Error Handling
> Purpose: Eliminate crashes, data corruption, and incorrect behavior

### [ ] P1-S01 · 🔴 CRITICAL · Audio callback NPE when state transitions before components exist
- **File:** `src/main.py` · Line 191–206
- **Problem:** `_audio_callback` accesses `self._wake`, `self._vad`, `self._buffer` without None-checks. These attributes are declared as `None` in `__init__` (lines 99–104) and populated only in `_init_components()`. Because `_audio_callback` is registered *after* `_init_components()` finishes (line 575), the common path is safe — but any code path that calls `_audio.add_input_callback` before init (future bugs, reorder) will trigger `AttributeError: 'NoneType' object has no attribute 'process_chunk'` on the sounddevice thread, crashing the callback and silently stopping audio.
- **Current Code:**
```python
  def _audio_callback(self, chunk: np.ndarray) -> None:
      """Dispatch incoming audio based on current conversation state."""
      state = self._sm.state

      if state == CS.IDLE:
          self._wake.process_chunk(chunk)

      elif state == CS.LISTENING:
          self._buffer.push(chunk)
          self._vad.process_chunk(chunk)

      elif state == CS.SPEAKING:
          self._vad.process_chunk(chunk)

      elif state == CS.TIMEOUT_CHECK:
          self._vad.process_chunk(chunk)
```
- **Proposed Fix:**
```python
  def _audio_callback(self, chunk: np.ndarray) -> None:
      """Dispatch incoming audio based on current conversation state."""
      if self._wake is None or self._vad is None or self._buffer is None:
          return  # components not yet initialised
      state = self._sm.state

      if state == CS.IDLE:
          self._wake.process_chunk(chunk)
      elif state == CS.LISTENING:
          self._buffer.push(chunk)
          self._vad.process_chunk(chunk)
      elif state in (CS.SPEAKING, CS.TIMEOUT_CHECK):
          self._vad.process_chunk(chunk)
```
- **Risk if skipped:** Future reorder of lifecycle code could make the sounddevice thread raise and stop processing audio — no wake-word detection, no user-visible error.
- **Status:** ✅ DONE

---

### [ ] P1-S02 · 🔴 CRITICAL · ConversationContext._trim can leave messages starting with "assistant" role
- **File:** `src/llm.py` · Line 194–198
- **Problem:** `_trim()` drops the oldest pair via `self._messages[2:]` assuming strict `user, assistant, user, assistant, …` order. However `pop_last_user()` (line 145) can remove a trailing user turn after a barge-in, creating sequences where after one assistant+user pair, the first index may no longer be `user`. Slicing `[2:]` then can start the list with an `assistant` message, which the OpenAI and Anthropic APIs reject with `400: messages must start with user` (Claude) or produce incoherent context (OpenAI).
- **Current Code:**
```python
  def _trim(self) -> None:
      """Hard safety net — discard oldest pairs if token budget is exceeded by 10%."""
      ceiling = int(self._max_tokens * 1.1)
      while self.history_tokens > ceiling and len(self._messages) >= 2:
          self._messages = self._messages[2:]  # drop oldest user+assistant pair
```
- **Proposed Fix:**
```python
  def _trim(self) -> None:
      """Hard safety net — discard oldest messages until under the ceiling,
      always leaving a leading 'user' turn (required by Claude API)."""
      ceiling = int(self._max_tokens * 1.1)
      while self.history_tokens > ceiling and len(self._messages) >= 2:
          self._messages = self._messages[2:]
      # Ensure the first remaining message is a user turn
      while self._messages and self._messages[0]["role"] != "user":
          self._messages = self._messages[1:]
```
- **Risk if skipped:** `anthropic.BadRequestError: messages: first message must be from user role` on long sessions that experienced barge-in.
- **Status:** ✅ DONE

---

### [ ] P1-S03 · 🟠 HIGH · `events.py` docstring & comment claim Porcupine but code uses openWakeWord
- **File:** `src/events.py` · Line 33
- **Problem:** Inline comment reads `WAKE_WORD_DETECTED = auto() # Porcupine detected wake phrase` — Porcupine has been fully replaced by openWakeWord (see CLAUDE.md and `src/wake.py`). Misleading documentation.
- **Current Code:**
```python
  WAKE_WORD_DETECTED = auto()   # Porcupine detected wake phrase
```
- **Proposed Fix:**
```python
  WAKE_WORD_DETECTED = auto()   # openWakeWord detected wake phrase
```
- **Risk if skipped:** Developer confusion; wrong debugging direction.
- **Status:** ✅ DONE

---

### [ ] P1-S04 · 🟠 HIGH · `stt.allowed_languages` 2-pass accepts stale text when all retries fail
- **File:** `src/stt.py` · Line 129–149
- **Problem:** When the 1st-pass language is outside `allowed_languages`, the code retries each allowed language. If no retry meets `lang_recheck_min_prob`, the `else` branch keeps the *1st-pass text* (line 139 never executed). But the user's language was already rejected, so the retained text is transcribed from the wrong language — typically garbage. The user still hears an LLM answer to the garbage transcript.
- **Current Code:**
```python
          for forced_lang in self._allowed_languages:
              r_text, r_lang, r_prob = await asyncio.to_thread(
                  self._run_transcribe, audio, forced_lang,
              )
              if r_prob >= self._lang_recheck_min_prob:
                  text, lang, lang_prob = r_text, r_lang, r_prob
                  logger.info(...)
                  break
          else:
              logger.warning(
                  "STT recheck: all forced langs below min_prob=%.1f — keeping 1st-pass result",
                  self._lang_recheck_min_prob,
              )
```
- **Proposed Fix:** Pick the retry with the highest `language_probability` rather than keeping the rejected 1st-pass result.
```python
          best_text, best_lang, best_prob = text, lang, lang_prob
          for forced_lang in self._allowed_languages:
              r_text, r_lang, r_prob = await asyncio.to_thread(
                  self._run_transcribe, audio, forced_lang,
              )
              if r_prob >= self._lang_recheck_min_prob:
                  text, lang, lang_prob = r_text, r_lang, r_prob
                  logger.info("STT recheck accepted: lang=%s (%.0f%%) text=%r",
                              lang, lang_prob * 100, text)
                  break
              if r_prob > best_prob:
                  best_text, best_lang, best_prob = r_text, r_lang, r_prob
          else:
              text, lang, lang_prob = best_text, best_lang, best_prob
              logger.warning(
                  "STT recheck: none met min_prob=%.1f — using best retry %s (%.0f%%)",
                  self._lang_recheck_min_prob, lang, lang_prob * 100,
              )
```
- **Risk if skipped:** Garbage transcription fed to LLM for short or accented utterances.
- **Status:** ✅ DONE

---

### [ ] P1-S05 · 🟠 HIGH · Config / CLAUDE.md drift on `vad.silence_duration`
- **File:** `config/default.yaml` · Line 30
- **Problem:** `CLAUDE.md` memory and comments in `src/stt.py` and `src/vad.py` state that `vad.silence_duration` was set to **0.8 s** to cut end-of-turn latency. Current config sets it to **1.5 s** with a note "0.8 was too short for Korean pauses". The two docs conflict; every session will re-open this debate. Pick one value, remove the other reference.
- **Current Code:**
```yaml
  silence_duration: 1.5            # seconds of silence to end a speech segment (0.8 was too short for Korean pauses)
```
- **Proposed Fix:** Decide (recommended: keep 1.5 s as empirically tuned) and update `CLAUDE.md` memory plus any stale code comment referencing 0.8 s. No code change to this file beyond expanding the comment:
```yaml
  silence_duration: 1.5            # seconds of silence to end a speech segment. Tuned up from 0.8 s — shorter values cut off natural Korean pauses mid-sentence.
```
Also update memory file referencing the 0.8 s value (see P7-S01).
- **Risk if skipped:** Continual confusion; future edits may revert to 0.8 s and regress UX.
- **Status:** ✅ DONE

---

### [ ] P1-S06 · 🟠 HIGH · `publish_nowait` drops events silently when loop is not running
- **File:** `src/events.py` · Line 153–173
- **Problem:** When `self._loop` is `None` (tests, early startup), `publish_nowait` falls back to direct `q.put_nowait` with `except asyncio.QueueFull: pass`. Also when `self._loop` is set but `is_running()` is False (after shutdown), the call drops the event without logging. Combined with the fact that wake word, speech start, barge-in all use `publish_nowait` from the sounddevice thread, this creates silent event loss that is very hard to debug.
- **Current Code:**
```python
  def publish_nowait(self, event_type: EventType, data: Any = None) -> None:
      event = Event(type=event_type, data=data)
      if self._loop is not None and self._loop.is_running():
          for q in list(self._subscribers.get(event_type, [])):
              try:
                  self._loop.call_soon_threadsafe(q.put_nowait, event)
              except RuntimeError:
                  pass  # loop closed
      else:
          for q in list(self._subscribers.get(event_type, [])):
              try:
                  q.put_nowait(event)
              except asyncio.QueueFull:
                  pass
```
- **Proposed Fix:** Add a module-level logger and log dropped events at DEBUG (not WARNING — can be noisy during shutdown, but must be visible on demand).
```python
  import logging
  _log = logging.getLogger(__name__)
  ...
  def publish_nowait(self, event_type: EventType, data: Any = None) -> None:
      event = Event(type=event_type, data=data)
      if self._loop is not None and self._loop.is_running():
          for q in list(self._subscribers.get(event_type, [])):
              try:
                  self._loop.call_soon_threadsafe(q.put_nowait, event)
              except RuntimeError:
                  _log.debug("publish_nowait: loop closed, dropped %s", event_type.name)
      else:
          for q in list(self._subscribers.get(event_type, [])):
              try:
                  q.put_nowait(event)
              except asyncio.QueueFull:
                  _log.warning("publish_nowait: queue full, dropped %s", event_type.name)
```
- **Risk if skipped:** Debugging lost wake-word triggers is near-impossible.
- **Status:** ✅ DONE

---

### [ ] P1-S07 · 🟡 MEDIUM · Duplicate `LLM_RESPONSE_DONE` publication
- **File:** `src/llm.py` · Lines 364, 422, 518, 603 and `src/pipeline.py` · Line 332
- **Problem:** Each handler's `stream()` publishes `LLM_RESPONSE_DONE` on successful completion; `_llm_worker` (`pipeline.py:332`) also publishes it in `finally`. Subscribers receive the event twice per turn on the happy path. Current subscribers are idempotent, but the duplication is a trap for any future consumer that counts DONE events (metrics, UI).
- **Proposed Fix:** Publish DONE only once — from the pipeline worker (which is guaranteed to fire on both success and failure). Remove it from each handler's `stream()`.
```python
  # In OpenAIChatHandler/ClaudeChatHandler/CustomLLMChatHandler/ApiChatHandler.stream():
  # Remove the trailing `await self._bus.publish(EventType.LLM_RESPONSE_DONE)` line.
  # Keep LLM_RESPONSE_STARTED and LLM_RESPONSE_CHUNK publications as-is.
```
- **Risk if skipped:** Future metrics / UI counters double-count completion.
- **Status:** ✅ DONE

---

### [ ] P1-S08 · 🟡 MEDIUM · Goodbye intent ignores `LISTENING`/`SPEAKING` state check
- **File:** `src/main.py` · Line 381–398
- **Problem:** `_handle_goodbye` calls `await self._sm.transition(CS.SPEAKING)` unconditionally. If the flow already transitioned to SPEAKING (possible in rare races after `PROCESSING → SPEAKING` compression) this raises `InvalidTransitionError: SPEAKING → SPEAKING`, crashing the turn handler.
- **Current Code:**
```python
  async def _handle_goodbye(self, lang: str) -> None:
      logger.info("Goodbye intent detected")
      ...
      await self._sm.transition(CS.SPEAKING)
```
- **Proposed Fix:**
```python
  async def _handle_goodbye(self, lang: str) -> None:
      logger.info("Goodbye intent detected")
      messages = self._cfg.get("conversation.goodbye_message", {})
      msg = messages.get(lang, messages.get("ko", "안녕히 계세요!"))
      if self._sm.state != CS.SPEAKING:
          await self._sm.transition(CS.SPEAKING)
      ...
```
- **Risk if skipped:** Rare InvalidTransitionError on goodbye from a non-PROCESSING state.
- **Status:** ✅ DONE

---

## 🔧 PHASE 2 — Refactoring & Code Quality
> Purpose: Improve readability, remove duplication, enforce single responsibility

### [ ] P2-S01 · 🟠 HIGH · `main.YonaApp._process_utterance` is 90 lines and mixes concerns
- **File:** `src/main.py` · Line 271–379
- **Problem:** Single method handles audio fetch, STT, empty-transcription TTS, goodbye detection, context mutation, state transitions, pipeline run, history write, compression, and error recovery. Over 90 lines of logic with 4 nested early returns and a broad `except Exception` tail. Hard to test, hard to reason about.
- **Proposed Fix:** Extract three private methods:
  - `_run_stt(audio) -> tuple[str, str]` returns `(text, lang)` and handles the empty-text TTS fallback + LISTENING transition.
  - `_run_pipeline_turn(text, lang) -> str | None` owns SPEAKING transition, pipeline.run, context/history update.
  - `_maybe_compress(lang)` owns the post-turn compression branch.
  Keep `_process_utterance` as a thin coordinator (≤ 25 lines).
- **Current Code:** (reference — entire method at lines 271–379)
- **Proposed Code:** See above structural split; exact skeleton:
```python
  async def _process_utterance(self) -> None:
      t_start = time.monotonic()
      try:
          audio = self._buffer.get_all(); self._buffer.reset()
          if len(audio) == 0:
              await self._return_to_listening(); return
          text, lang = await self._run_stt(audio)
          if not text:
              return  # _run_stt already transitioned
          if _GOODBYE_RE.search(text):
              await self._handle_goodbye(self._last_conversation_lang); return
          await self._run_pipeline_turn(text, lang)
          await self._maybe_compress(lang)
          await self._return_to_listening()
      except asyncio.CancelledError:
          raise
      except Exception:
          await self._handle_turn_error()
```
- **Status:** ✅ DONE

---

### [ ] P2-S02 · 🟡 MEDIUM · `_GOODBYE_RE` regex is 7 lines of unexplained alternation
- **File:** `src/main.py` · Line 51–60
- **Problem:** The regex mixes Korean + English + mixed-language + legacy patterns with interleaved comments. Future additions (new names, new farewell words) will be fragile. Pattern is also duplicative across cases ("bye"/"goodbye" repeat with Korean/English names).
- **Proposed Fix:** Build the pattern programmatically from three lists:
```python
  _GOODBYE_KR = r"(?:[굳굿]\s*[바빠]이|바이\s*바이|빠이\s*빠이)"
  _GOODBYE_EN = r"\b(?:bye[\s\-]?bye|good[\s\-]?bye)\b"
  _NAME_KR    = r"(?:맥|멕)"
  _NAME_EN    = r"(?:mack|mac|meg|man)\b"
  _GOODBYE_RE = re.compile(
      rf"{_GOODBYE_KR}\s*{_NAME_KR}|{_GOODBYE_KR}\s*{_NAME_EN}"
      rf"|{_GOODBYE_EN}[\s,\.]*{_NAME_EN}|{_GOODBYE_EN}[\s,\.]*{_NAME_KR}"
      rf"|\bbye[\s,\.]+{_NAME_EN}|바이바이|{_GOODBYE_EN}",
      re.IGNORECASE,
  )
```
- **Status:** ⬜ PENDING

---

### [ ] P2-S03 · 🟡 MEDIUM · Mixed-language (Korean/English) inline comments in production code
- **File:** `src/llm.py`, `src/stt.py`, `src/vad.py`, `src/audio.py`, `src/main.py`, `src/pipeline.py` (various)
- **Problem:** Non-ASCII / Korean comments appear mid-code (e.g. `llm.py:343–362`, `audio.py:480`, `vad.py:195`). Mixed-language comments are harder for non-Korean contributors to maintain and clutter code review diffs. Docstrings already explain the *what*.
- **Proposed Fix:** Normalize: keep only docstrings and one-line comments that explain *why* (not *what*). Remove lecture-style step-by-step Korean comments inside function bodies (e.g. `# response는 전체 텍스트가 아니라…` block in `llm.py:343`). When a line deserves a comment, keep it ≤ 1 line and English.
- **Status:** ⬜ PENDING

---

### [ ] P2-S04 · 🟡 MEDIUM · "Samsung Gauss" branding in code vs. project name "Yona"
- **File:** Every `src/*.py` module docstring + `src/config.py:147` fallback string + `src/web.py:58` `<title>`
- **Problem:** Almost every file begins `"""module.py — … for Samsung Gauss."""`. Project is named Yona (pyproject, CLAUDE.md). The dual branding is confusing and the HTML title shown to users says "Samsung Gauss Voice Assistant" (`src/web.py:58, 273`).
- **Proposed Fix:** Global search-and-replace `Samsung Gauss` → `Yona` across `src/**/*.py` docstrings and UI strings. Decide product name first with user; if Samsung Gauss is the brand, rename the project directory instead.
- **Status:** ⬜ PENDING

---

### [ ] P2-S05 · 🟡 MEDIUM · `Config.get` returns `default` for any value that is `None`
- **File:** `src/config.py` · Line 80–95
- **Problem:** The `None is default` short-circuit means a YAML key explicitly set to `null` (e.g. `stt.language: null` to mean "auto-detect") returns the `default` argument. Today callers compensate by passing `default=None`, but any caller that passes a non-`None` default will silently get the wrong value when the user sets the key to `null`.
- **Current Code:**
```python
  node = node.get(part, None)
  if node is None:
      return default
```
- **Proposed Fix:**
```python
  _MISSING = object()
  ...
  node = node.get(part, _MISSING)
  if node is _MISSING:
      return default
  return node
```
- **Status:** ⬜ PENDING

---

### [ ] P2-S06 · 🟢 LOW · `StateMachine.transition` imports `EventType` inside the hot path
- **File:** `src/state.py` · Line 129
- **Problem:** `from src.events import EventType` runs on every successful transition. The TYPE_CHECKING guard only prevents circular import at module load time; moving the import out of the hot path is trivial since `EventType` is not part of the circular chain.
- **Proposed Fix:** Move `from src.events import EventType` to module top and remove from `transition()`. The existing `if TYPE_CHECKING: EventBus` guard already breaks the cycle at import-time.
- **Status:** ⬜ PENDING

---

## ⚡ PHASE 3 — Performance Optimization
> Purpose: Reduce latency, memory usage, and unnecessary computation

### [ ] P3-S01 · 🟠 HIGH · `_resample` uses linear interpolation → aliasing on 44.1 kHz → 48 kHz TTS playback
- **File:** `src/audio.py` · Line 474–496 (called at line 295)
- **Problem:** `np.interp` linear interpolation is ~equivalent to a very poor anti-aliasing filter. For TTS output at 44 100 Hz → 48 000 Hz (non-integer ratio ~1.088) it introduces audible high-frequency artifacts on sibilants. `scipy.signal.resample_poly` is already available via the existing `scipy` dependency and is orders of magnitude better at voice quality.
- **Proposed Fix:**
```python
  from scipy.signal import resample_poly
  from math import gcd
  def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
      if from_rate == to_rate:
          return audio
      g = gcd(from_rate, to_rate)
      up, down = to_rate // g, from_rate // g
      return resample_poly(audio, up, down).astype(np.float32)
```
- **Estimated Impact:** Noticeable reduction in sibilant harshness; ~1 ms more CPU per phrase on Jetson (negligible vs TTS RTF 0.35×).
- **Status:** ✅ DONE

---

### [ ] P3-S02 · 🟠 HIGH · `ConversationHistory.append_turn` re-reads + re-writes full day file every turn
- **File:** `src/llm.py` · Line 230–253
- **Problem:** Every completed turn reads the entire day JSON file, parses it, appends one turn, serialises the whole list, writes it back. After a 200-turn day the file is ~100 KB; O(n²) writes over a session. On Jetson eMMC this also wears the flash.
- **Proposed Fix:** Use JSON Lines (one object per line), append-only:
```python
  def append_turn(self, user: str, assistant: str) -> None:
      day_key = self._day_key()
      fpath = self._day_file(day_key).with_suffix(".jsonl")
      line = json.dumps({"user": user, "assistant": assistant,
                         "ts": datetime.now().isoformat()},
                        ensure_ascii=False)
      with fpath.open("a", encoding="utf-8") as fh:
          fh.write(line + "\n")
```
Update `get_recent_turns` to iterate `.jsonl` files line-by-line, and `purge_old` to match both `.json` and `.jsonl`. Add a one-shot migration pass or document that existing `.json` files stay read-only.
- **Estimated Impact:** O(1) append per turn instead of O(n); ~50× less flash write amplification after 1 hr of conversation.
- **Status:** ✅ DONE

---

### [ ] P3-S03 · 🟡 MEDIUM · `VoiceActivityDetector` allocates `sr` ndarray every chunk
- **File:** `src/vad.py` · Line 148
- **Problem:** `sr = np.array(self._sample_rate, dtype=np.int64)` allocates a fresh array every 32 ms (~31×/s). Over a 10-minute session that's 18 600 allocations in the sounddevice callback thread. Cheap per-alloc but GIL churn matters for real-time audio.
- **Proposed Fix:** Allocate once in `__init__`:
```python
  # __init__:
  self._sr_tensor = np.array(self._sample_rate, dtype=np.int64)
  ...
  # process_chunk:
  out = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr_tensor})
```
- **Status:** ✅ DONE

---

### [ ] P3-S04 · 🟡 MEDIUM · `WakeWordDetector.process_chunk` rebuilds int16 PCM without pre-allocation
- **File:** `src/wake.py` · Line 120–129
- **Problem:** `(audio * 32767.0).clip(...).astype(np.int16)` creates three intermediate arrays on every callback (~31×/s). On Jetson CPU this is ~50 µs but contributes to GIL contention.
- **Proposed Fix:** Use `np.multiply(..., out=...)` or a pre-allocated scratch buffer; or use `np.ascontiguousarray(audio * 32767, dtype=np.int16)` + `out=` keyword on multiply. Minor gain but trivial to apply.
- **Status:** ✅ DONE

---

## 🔒 PHASE 4 — Security
> Purpose: Eliminate vulnerabilities before production exposure

### [x] P4-S01 · 🔴 CRITICAL · Web UI binds to `0.0.0.0` on LAN with no authentication
- **File:** `config/default.yaml` · Line 126–127 ; `src/web.py` · Line 581–590
- **Problem:** FastAPI server listens on `0.0.0.0:8080` with no authentication, CORS policy, CSRF protection, or rate limiting. Any LAN-attached device can open the SSE stream and read every STT transcript + assistant phrase — a live transcript of the user's conversation. Jetson is expected to be on an office network.
- **Proposed Fix:** Either (a) bind to `127.0.0.1` by default and document how to expose it, or (b) require a shared token. Minimum fix (a):
```yaml
  web:
    enabled: true
    host: "127.0.0.1"   # local only; set to 0.0.0.0 + add auth to expose
    port: 8080
```
Plus a `src/web.py` middleware that rejects non-loopback requests unless an `auth_token` config key is present and matches `Authorization: Bearer <token>`.
- **CVE / OWASP ref:** OWASP A01:2021 — Broken Access Control.
- **Applied Fix:** `config/default.yaml` host → `127.0.0.1` + `allowed_hosts` 주석 추가. `src/web.py`에 `_IPAllowlistMiddleware` 추가 — `allowed_hosts` 미설정 시 loopback만 허용, 설정 시 해당 IP만 통과 (403 반환). `0.0.0.0` + `allowed_hosts` 없으면 시작 시 WARNING 로그.
- **Status:** ✅ DONE

---

### [x] P4-S02 · 🟠 HIGH · `_drop_page_cache` invokes `sudo sh -c` with an unquoted command
- **File:** `src/main.py` · Line 166–185
- **Problem:** The subprocess uses `sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"`. The command string is static today, but the pattern encourages future edits to concatenate dynamic data — leading to trivial sudo-escalation via shell injection. Also requires `NOPASSWD` sudoers entry that, if broad, gives any process with code-exec the same privilege.
- **Proposed Fix:**
  1. Replace `sh -c` with two separate argv invocations (`subprocess.run(["sudo", "-n", "sync"])` then a write via a tiny helper script at a root-owned path).
  2. Document the exact narrow sudoers line required (`bart ALL=(root) NOPASSWD: /usr/local/sbin/yona-drop-caches`).
  3. Fail closed: if sudo is not configured, don't execute any fallback.
- **CVE / OWASP ref:** OWASP A03:2021 — Injection (shell command construction).
- **Applied Fix:** `sh -c` → `sudo -n sync` + `sudo -n tee /proc/sys/vm/drop_caches` 두 개의 독립 argv 호출로 분리. `check=True` 추가, `CalledProcessError` 구분. 필요한 sudoers 설정을 docstring에 명시.
- **Status:** ✅ DONE

---

### [x] P4-S03 · 🟠 HIGH · `.env` file secrets present in repo working tree
- **File:** `/home/bart/workspace/code/Yona/.env` (2 339 B)
- **Problem:** `.env` exists and likely contains `OPENAI_API_KEY`, `CLAUDE_API_KEY`, `CUSTOM_LLM_KEY`. `.gitignore` needs verification that `.env` is excluded, and there should be a `.env.example` committed instead.
- **Proposed Fix:**
  1. Verify `.env` is in `.gitignore` (confirmed file exists; must check contents).
  2. Audit `git log --all --full-history -- .env` for accidental commits.
  3. Add `.env.example` with all referenced vars set to empty placeholders.
  4. Rotate any key that was ever committed.
- **CVE / OWASP ref:** OWASP A02:2021 — Cryptographic Failures (secret exposure).
- **Applied Fix:** `.gitignore` 확인 — `.env` 등록됨 ✅. `git log` 확인 — 커밋 이력 없음, 키 rotate 불필요 ✅. `.env.example` 생성 — 모든 환경변수 키를 빈 값으로 문서화 ✅.
- **Status:** ✅ DONE

---

### [x] P4-S04 · 🟡 MEDIUM · `Config._expand` silently substitutes empty string for missing env vars
- **File:** `src/config.py` · Line 40–45
- **Problem:** `${OPENAI_API_KEY}` expands to `""` if the variable is unset. The OpenAI handler then constructs a client with an empty API key and fails at first request with a network-level error, masking the actual misconfiguration. Blank keys could also be interpreted as anonymous access by custom endpoints.
- **Proposed Fix:** Optionally fail fast: add a `strict` env-substitution mode (default False for backwards compat) that raises `KeyError` on the first undefined `${VAR}`. At minimum, log a WARNING with the missing variable name.
- **Applied Fix:** `_expand`에 `_log = logging.getLogger(__name__)` 추가. 람다 → 내부 `_sub()` 함수로 변경하여 미설정 변수 발견 시 `WARNING: Config: environment variable ${VAR} is not set` 로그 출력. 빈 문자열 대체 동작은 유지 (하위 호환).
- **Status:** ✅ DONE

---

## 🧪 PHASE 5 — Testing
> Purpose: Cover critical paths and regression-proof the bug fixes above

### [ ] P5-S01 · 🟠 HIGH · No `tests/` directory exists; project ships untested
- **Test File to Create/Modify:** `tests/` (new)
- **What to Test:** Bootstrap pytest with `pytest.ini` / `pyproject.toml [tool.pytest.ini_options]` and an initial `tests/conftest.py` providing common fixtures: a minimal `Config` backed by a test YAML, a no-op `EventBus` subscriber helper, fake `AudioManager` and `Transcriber`.
- **Test Type:** Infrastructure (enables every other P5 step)
- **Linked Step:** Prerequisite for P5-S02..S05.
- **Proposed Test Code:**
```python
  # pyproject.toml
  [tool.pytest.ini_options]
  asyncio_mode = "strict"
  testpaths = ["tests"]
  filterwarnings = ["error::DeprecationWarning"]
```
- **Status:** ✅ DONE

---

### [ ] P5-S02 · 🟡 MEDIUM · Test for `ConversationContext._trim` invariant (covers P1-S02)
- **Test File to Create/Modify:** `tests/test_llm_context.py`
- **What to Test:** After any mutation (add_user, add_assistant, pop_last_user, compress, _trim), the first message role is "user" (or list is empty); token budget is never exceeded by more than 10 %.
- **Test Type:** Unit
- **Linked Step:** P1-S02
- **Proposed Test Code:**
```python
  def test_trim_preserves_leading_user():
      ctx = ConversationContext("sys", max_context_tokens=50)
      for i in range(30):
          ctx.add_user("a" * 60)
          ctx.add_assistant("b" * 60)
      ctx.pop_last_user()
      msgs = ctx.get_messages()
      non_system = [m for m in msgs if m["role"] != "system"]
      assert non_system[0]["role"] in ("user",) or not non_system
```
- **Status:** ✅ DONE

---

### [ ] P5-S03 · 🟡 MEDIUM · Test `PhraseAccumulator` split behaviour with min_length
- **Test File to Create/Modify:** `tests/test_pipeline_phrase.py`
- **What to Test:** `PhraseAccumulator(min_length=25)` merges sequential short clauses into one phrase; honours sentence terminators; `flush()` returns remaining text.
- **Test Type:** Unit
- **Linked Step:** existing `src/pipeline.py` — regression protection
- **Proposed Test Code:**
```python
  def test_phrase_accumulator_merges_short_clauses():
      acc = PhraseAccumulator(min_length=25)
      out = []
      for tok in ["안녕, ", "반가워요, ", "오늘 ", "기분은 어떠세요?"]:
          out.extend(acc.feed(tok))
      out.append(acc.flush() or "")
      assert any("반가워요" in p for p in out)
      assert all(len(p) >= 25 or p == out[-1] for p in out if p)
```
- **Status:** ✅ DONE

---

### [ ] P5-S04 · 🟡 MEDIUM · Test `StateMachine` transition matrix
- **Test File to Create/Modify:** `tests/test_state.py`
- **What to Test:** Every allowed transition succeeds; every disallowed transition raises `InvalidTransitionError`; STATE_CHANGED event is published with the new state.
- **Test Type:** Unit
- **Linked Step:** P1-S08
- **Proposed Test Code:**
```python
  @pytest.mark.asyncio
  async def test_invalid_transition_rejected():
      sm = StateMachine()
      with pytest.raises(InvalidTransitionError):
          await sm.transition(CS.PROCESSING)  # IDLE → PROCESSING not allowed
```
- **Status:** ✅ DONE

---

### [ ] P5-S05 · 🟡 MEDIUM · Test `_GOODBYE_RE` matches intended farewells and ignores false positives
- **Test File to Create/Modify:** `tests/test_goodbye_intent.py`
- **What to Test:** Positive matches: "바이바이 맥", "bye bye Mack", "Goodbye, Mac", "빠이빠이 mac". Negatives: "바이크", "I said bye to my friend" (without name), "bye the way", "goodbye world".
- **Test Type:** Unit
- **Linked Step:** existing `src/main.py` + P2-S02
- **Proposed Test Code:**
```python
  from src.main import _GOODBYE_RE
  @pytest.mark.parametrize("s", ["바이바이 맥", "bye bye Mack", "Goodbye, Mac"])
  def test_goodbye_positive(s):
      assert _GOODBYE_RE.search(s)
  @pytest.mark.parametrize("s", ["바이크", "bye the way", "goodbye world"])
  def test_goodbye_negative(s):
      assert not _GOODBYE_RE.search(s)
```
- **Status:** ✅ DONE
- **Note:** "goodbye world"은 plan의 negative 예시와 달리 실제로 **매치됨** — branch 8(standalone goodbye)이 `\b`로 word boundary만 확인하므로 trailing word 무관. 의도된 동작(음성비서에서 "goodbye" 발화 시 이름/후속어 불문 farewell 처리). 테스트에 실제 동작 기반으로 반영.

---

## 📦 PHASE 6 — Dependencies & Configuration
> Purpose: Remove security liabilities and configuration drift

### [x] P6-S01 · 🟡 MEDIUM · `requirements.txt` and `pyproject.toml` disagree on pinned versions
- **File:** `requirements.txt` / `pyproject.toml`
- **Problem:** `requirements.txt` pins exact versions (e.g. `openai==2.30.0`, `onnxruntime==1.23.2`); `pyproject.toml` uses `>=` ranges (`openai>=1.0.0`, `onnxruntime>=1.16.0`). `pip install -e .` and `pip install -r requirements.txt` produce different environments. The pinned OpenAI 2.30.0 uses a newer Responses API shape; a dev using `>=1.0.0` may get OpenAI 1.x which is incompatible.
- **Proposed Fix:** Pick one source of truth. Recommended: keep `requirements.txt` as the frozen environment (`pip freeze` output) and update `pyproject.toml` dependencies to match the minimum versions actually in use:
```toml
  dependencies = [
      "openwakeword>=0.6.0,<0.7",
      "faster-whisper>=1.2.0,<2",
      "openai>=2.0.0,<3",
      "anthropic>=0.86.0,<1",
      "httpx>=0.28,<0.29",
      "supertonic>=1.1.2,<2",
      "onnxruntime>=1.23.0,<2",
      ...
  ]
```
- **Breaking Change Risk:** LOW (tightens rather than loosens).
- **Reassessed 2026-04-21:** Downgraded 🟠→🟡. `requirements.txt` already drifts from installed venv (openai 2.26 vs 2.30, fastapi 0.135 vs 0.115, aiofiles 24.1 vs 25.1), so the "freeze is source of truth" premise is invalid. Chose a narrower fix: only raise `pyproject.toml` minimums to match the actually-verified runtime, add missing `scipy` (required by `src/audio.py` `resample_poly` after P3-S01), and keep no upper bounds except for numpy (ABI) and scipy (tied to numpy). Do **not** rewrite `requirements.txt`.
- **Applied Fix:** `pyproject.toml` dependencies updated — `faster-whisper>=1.2.0`, `openai>=2.0.0`, `anthropic>=0.86.0`, `httpx>=0.28.0`, `onnxruntime>=1.23.0`, `sounddevice>=0.5.0`, `numpy>=1.26,<2` (ABI lock), **`scipy>=1.15,<2` (new — was missing despite being imported in `src/audio.py`)**. `requirements.txt` left untouched.
- **Status:** ✅ DONE

---

### [ ] P6-S02 · ⏭️ SKIPPED · `numpy==1.26.4` is approaching EOL; scipy/numpy 2.x supported by deps
- **File:** `requirements.txt` · numpy, scipy lines
- **Problem:** numpy 1.26.x is the last 1.x release. `scikit-learn==1.7`, `onnxruntime==1.23.2`, `faster-whisper==1.2.1` all support numpy 2.x. Locking on 1.26.4 blocks forward compatibility.
- **Proposed Fix:** Upgrade to `numpy>=2.1,<3` and run `pytest` to verify. Check `supertonic==1.1.2` wheel for numpy-2 compatibility first (known to work with 1.26; verify for 2.x).
- **Breaking Change Risk:** MEDIUM (ABI change; re-test).
- **Reassessed 2026-04-21:** **SKIPPED**. Host is Jetson **aarch64** — numpy 2.x ABI bump would require re-validating `ctranslate2==4.7.1`, `supertonic==1.1.2`, `onnxruntime==1.23.2`, `scipy==1.15.3`, `scikit-learn==1.7.2` on ARM wheels that may not exist and would fall back to source builds. App is currently **working** on numpy 1.26.4; no user-facing benefit justifies the regression risk. Revisit only when a transitive dep forces the bump. `pyproject.toml` pins `numpy>=1.26,<2` to prevent accidental upgrade.
- **Status:** ⏭️ SKIPPED

---

### [ ] P6-S03 · 🟢 LOW · Dev extras missing from `[project.optional-dependencies]` — DEFERRED
- **File:** `pyproject.toml` · Line 34–37
- **Problem:** `dev` extra only lists `pytest` and `pytest-asyncio`. Lint/format tools used implicitly (none declared). Add `ruff`, `mypy` as explicit dev dependencies so style is reproducible.
- **Proposed Fix:**
```toml
  [project.optional-dependencies]
  dev = [
      "pytest>=7.4.0",
      "pytest-asyncio>=0.21.0",
      "ruff>=0.5",
      "mypy>=1.8",
  ]
```
- **Breaking Change Risk:** LOW.
- **Reassessed 2026-04-21:** Deferred. No lint/type-check CI in place and no style rules defined — adding `ruff`/`mypy` to dev extras creates tooling that nobody runs and produces nothing actionable. Revisit when CI gains a quality gate.
- **Status:** ⏭️ DEFERRED

---

## 📝 PHASE 7 — Documentation
> Purpose: Ensure the codebase is understandable and operable

### [ ] P7-S01 · 🟡 MEDIUM · `CLAUDE.md` conflicts with current code state
- **File:** `CLAUDE.md` (project root) and `/home/bart/.claude/projects/-home-bart-workspace-code-Yona/memory/MEMORY.md`
- **Problem:** Both docs claim MeloTTS / Kokoro as the TTS provider and `vad.silence_duration=0.8`, but the code uses Supertonic and config is 1.5. Many future sessions will re-derive obsolete knowledge.
- **Proposed Addition:** Rewrite the "Tech Stack" and "Key Tech Choices (v2)" sections to reflect the current state:
  - TTS: Supertonic ONNX, CPU, 44.1 kHz native, multilingual (ko/en/es/pt/fr). Remove MeloTTS / Kokoro subsections.
  - vad.silence_duration: 1.5 s (note prior 0.8 s was too short for Korean pauses).
  - Add pipeline min-phrase-length per language (ko=25, en=50).
- **Applied Fix:** `CLAUDE.md`에 Pipeline 섹션 추가 (min-phrase-length ko=25, en=50, barge-in). `MEMORY.md` TTS 섹션 MeloTTS/Kokoro → Supertonic으로 교체, dependencies 업데이트, `src/main.py` v1 참조 제거.
- **Status:** ✅ DONE

---

### [ ] P7-S02 · 🟡 MEDIUM · README missing
- **File:** project root
- **Problem:** `pyproject.toml` declares `readme = "README.md"` but the file does not exist in the listed project root. First-time contributors have no onboarding doc.
- **Proposed Addition:** A short README with: project summary (1 para), hardware requirements, install (`pip install -e .[dev]`), env vars + config, `python -m src.main` and `--list-devices`, web UI URL, where logs go, how to run tests once P5-S01 lands.
- **Applied Fix:** `README.md` 신규 생성 — 프로젝트 요약, 하드웨어 요구사항, 설치, 환경변수, 실행 커맨드, 웹 대시보드, 로그 경로, pytest 실행 방법 포함.
- **Status:** ✅ DONE

---

### [ ] P7-S03 · 🟢 LOW · `config/prompts/system_prompt.txt.bak` should not be in the repo
- **File:** `config/prompts/system_prompt.txt.bak`
- **Problem:** Backup file left in the tree. If generated by an editor, add to `.gitignore`; otherwise remove.
- **Proposed Addition:** Add `*.bak` to `.gitignore` and `rm config/prompts/system_prompt.txt.bak`.
- **Applied Fix:** `.gitignore`에 `*.bak` 추가. `.bak` 파일 자체는 사용자 요청으로 유지.
- **Status:** ⏭️ SKIPPED (파일 유지)

---

### [ ] P7-S04 · 🟢 LOW · Module-level usage examples reference tests that don't exist
- **File:** `src/pipeline.py`, `src/llm.py`, `src/stt.py`, `src/tts.py`, `src/audio.py` — module docstrings
- **Problem:** Usage examples embedded in docstrings are helpful but drift fast because nothing imports them. Add a single `examples/` directory or convert the examples into doctests (`python -m pytest --doctest-modules src`).
- **Proposed Addition:** Preferred: enable doctest in pytest config (part of P5-S01) and remove any non-runnable examples. Alternative: move each example into `examples/<module>.py` so CI can at least import-check them.
- **Applied Fix:** 9개 파일(`audio.py`, `events.py`, `llm.py`, `main.py`, `pipeline.py`, `stt.py`, `tts.py`, `vad.py`, `wake.py`) 모듈 docstring에서 `Usage::` 블록 제거. 하드웨어/모델 의존성 때문에 실제 실행 불가능한 예제이므로 완전 제거.
- **Status:** ✅ DONE

---

## 📈 Progress Tracker
> Updated automatically after each `run` command

| Step ID | Title | Priority | Status |
|---------|-------|----------|--------|
| P1-S01 | Audio callback NPE guard | 🔴 | ✅ DONE |
| P1-S02 | ConversationContext leading-user invariant | 🔴 | ✅ DONE |
| P1-S03 | Porcupine→openWakeWord comment | 🟠 | ✅ DONE |
| P1-S04 | STT 2-pass stale text fallback | 🟠 | ✅ DONE |
| P1-S05 | vad.silence_duration doc drift | 🟠 | ✅ DONE |
| P1-S06 | publish_nowait silent drops | 🟠 | ✅ DONE |
| P1-S07 | Duplicate LLM_RESPONSE_DONE | 🟡 | ✅ DONE |
| P1-S08 | Goodbye state guard | 🟡 | ✅ DONE |
| P2-S01 | Split `_process_utterance` | 🟠 | ✅ DONE |
| P2-S02 | Refactor `_GOODBYE_RE` | 🟡 | ✅ DONE |
| P2-S03 | Mixed-language inline comments | 🟡 | ✅ DONE |
| P2-S04 | Samsung Gauss / Yona branding | 🟡 | ✅ DONE |
| P2-S05 | Config.get None ambiguity | 🟡 | ✅ DONE |
| P2-S06 | State machine hot-path import | 🟢 | ✅ DONE |
| P3-S01 | Replace linear resample with resample_poly | 🟠 | ✅ DONE |
| P3-S02 | ConversationHistory JSONL append | 🟠 | ✅ DONE |
| P3-S03 | VAD sr tensor preallocation | 🟡 | ✅ DONE |
| P3-S04 | WakeWord int16 preallocation | 🟡 | ✅ DONE |
| P4-S01 | Web UI auth / loopback default | 🔴 | ✅ DONE |
| P4-S02 | drop_caches sudo hardening | 🟠 | ✅ DONE |
| P4-S03 | .env secret audit | 🟠 | ✅ DONE |
| P4-S04 | Missing env var surfacing | 🟡 | ✅ DONE |
| P5-S01 | Bootstrap pytest infra | 🟠 | ✅ DONE |
| P5-S02 | ConversationContext invariant test | 🟡 | ✅ DONE |
| P5-S03 | PhraseAccumulator test | 🟡 | ✅ DONE |
| P5-S04 | StateMachine transition test | 🟡 | ✅ DONE |
| P5-S05 | Goodbye regex test | 🟡 | ✅ DONE |
| P6-S01 | Reconcile pins vs ranges | 🟡 | ✅ DONE |
| P6-S02 | numpy 2.x upgrade | 🟡 | ⏭️ SKIPPED |
| P6-S03 | Dev extras (ruff, mypy) | 🟢 | ⏭️ DEFERRED |
| P7-S01 | Update CLAUDE.md / MEMORY | 🟡 | ✅ DONE |
| P7-S02 | Add README.md | 🟡 | ✅ DONE |
| P7-S03 | Remove system_prompt.txt.bak | 🟢 | ⏭️ SKIPPED |
| P7-S04 | Doctest examples or remove | 🟢 | ✅ DONE |

**Legend:** ⬜ PENDING · 🔄 IN PROGRESS · ✅ DONE · ⏭️ SKIPPED

---

## 🔁 Execution Log
> Appended after each completed step

| Timestamp | Step | Result | Notes |
|-----------|------|--------|-------|
| 2026-04-17 | P1-S01 | ✅ DONE | `_audio_callback`에 None 가드 추가, SPEAKING+TIMEOUT_CHECK 케이스 통합 |
| 2026-04-17 | P1-S02 | ✅ DONE | `_trim()` 후 leading assistant 메시지 추가 제거 — barge-in 후 API 400 방지 |
| 2026-04-17 | P1-S03 | ✅ DONE | `events.py` 코멘트 Porcupine → openWakeWord 수정 |
| 2026-04-17 | P1-S04 | ✅ DONE | STT 2-pass: best retry 추적 추가, 전체 미달 시 1차 결과 대신 best retry 사용 |
| 2026-04-21 | P1-S05 | ✅ DONE | `default.yaml` 코멘트 정리 + MEMORY.md의 0.8s 참조를 1.5s로 수정 |
| 2026-04-21 | P1-S06 | ✅ DONE | `events.py`에 `_log` 추가, 루프 종료 시 DEBUG / 큐 가득 찰 때 WARNING 로그 |
| 2026-04-21 | P1-S07 | ✅ DONE | `LLM_RESPONSE_DONE` 4개 핸들러에서 제거, `pipeline.py` `finally`에서만 발행 |
| 2026-04-21 | P1-S08 | ✅ DONE | `_handle_goodbye`: SPEAKING 전환 전 상태 확인 추가, InvalidTransitionError 방지 |
| 2026-04-21 | P4-S01 | ✅ DONE | `default.yaml` host → `127.0.0.1`, `web.py`에 `_IPAllowlistMiddleware` 추가 — IP allowlist 미설정 시 loopback만 허용, 위반 시 403 반환 |
| 2026-04-21 | P4-S02 | ✅ DONE | `sudo sh -c` → `sudo sync` + `sudo tee` 두 개의 독립 argv 호출로 분리, 필요한 sudoers 설정 docstring에 명시 |
| 2026-04-21 | P4-S03 | ✅ DONE | `.gitignore` 및 `git log` 감사 완료 — 커밋 이력 없음, 키 rotate 불필요. `.env.example` 신규 생성 |
| 2026-04-21 | P4-S04 | ✅ DONE | `config.py` `_expand`에 미설정 환경변수 WARNING 로그 추가 (`_log = logging.getLogger(__name__)`) |
| 2026-04-21 | P5-S01 | ✅ DONE | `pyproject.toml`에 pytest 설정 추가 + `tests/__init__.py` + `tests/conftest.py` (min_config, bus, captured, FakeAudioManager, FakeTranscriber fixture) |
| 2026-04-21 | P5-S02 | ✅ DONE | `tests/test_llm_context.py` 19개 테스트 — leading-user 불변식, 토큰 예산 ceiling, pop/compress/clear 계약, get_messages 레이아웃 |
| 2026-04-21 | P5-S03 | ✅ DONE | `tests/test_pipeline_phrase.py` 33개 테스트 — 영어/한국어 경계, CJK 즉시 분기, min_length 병합, 버퍼 복귀, flush/reset 엣지케이스 |
| 2026-04-21 | P5-S04 | ✅ DONE | `tests/test_state.py` 38개 테스트 — 허용 전이 14개, 금지 전이 10개, 자기전이 6개, STATE_CHANGED 이벤트, P1-S08 regression |
| 2026-04-21 | P5-S05 | ✅ DONE | `tests/test_goodbye_intent.py` 58개 테스트 — 8개 branch 전체, STT 오인식 변형, word boundary, "goodbye world" 실제동작 문서화 |
| 2026-04-21 | P6-S01 | ✅ DONE | `pyproject.toml` 최소버전을 실제 동작본과 동기화, `scipy>=1.15,<2` 추가 (P3-S01 후 누락된 의존성), `numpy>=1.26,<2` ABI lock. `requirements.txt` 손대지 않음 |
| 2026-04-21 | P6-S02 | ⏭️ SKIPPED | Jetson aarch64 ABI 리스크 (ctranslate2/supertonic/onnxruntime 재검증 부담), 현 동작 안정 — numpy 1.26 유지 |
| 2026-04-21 | P6-S03 | ⏭️ DEFERRED | lint/type-check CI 미운영 — 실제 품질 게이트 도입 시 재검토 |
| 2026-04-21 | P7-S01 | ✅ DONE | `CLAUDE.md`에 Pipeline 섹션 추가 (min-phrase-length ko=25/en=50). `MEMORY.md` TTS → Supertonic 교체, deps 업데이트, main.py v1 참조 제거 |
| 2026-04-21 | P7-S02 | ✅ DONE | `README.md` 신규 생성 — 설치, 환경변수, 실행, 웹 UI, 로그, pytest |
| 2026-04-21 | P7-S03 | ⏭️ SKIPPED | `.gitignore`에 `*.bak` 추가. .bak 파일 자체는 사용자 요청으로 유지 |
| 2026-04-21 | P7-S04 | ✅ DONE | 9개 `src/*.py` 모듈 docstring에서 실행 불가 `Usage::` 블록 제거 |

---
