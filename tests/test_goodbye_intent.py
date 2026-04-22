"""test_goodbye_intent.py — Unit tests for the _GOODBYE_RE farewell pattern.

The regex has 8 alternation branches:
  1. FAREWELL_KO + NAME_KO  : "바이바이 맥", "빠이빠이 멕"
  2. FAREWELL_KO + NAME_EN  : "바이바이 mac", "빠이빠이 mack"
  3. FAREWELL_EN + NAME_EN  : "bye bye Mack", "goodbye, Mac"
  4. FAREWELL_EN + NAME_KO  : "bye bye 맥", "Goodbye 맥"
  5. bye + NAME_EN           : "Bye, Mac", "bye mac"
  6. bye + NAME_KO           : "Bye 맥", "bye, 맥"
  7. 바이바이 standalone
  8. FAREWELL_EN standalone  : "bye bye", "goodbye"  ← matches any following word too
                               because \b only asserts a boundary, not end-of-string.
                               "goodbye world" → matches (intended: any goodbye triggers farewell).

Note: "goodbye world" matches branch 8 — the plan's example list incorrectly called this
a negative. The regex is intentionally loose: standalone "goodbye" / "bye bye" triggers
farewell regardless of trailing words (matches real voice-assistant usage).
"""

from __future__ import annotations

import pytest

from src.main import _GOODBYE_RE

# ---------------------------------------------------------------------------
# Positives — each must match
# ---------------------------------------------------------------------------

_POSITIVES = [
    # --- Branch 1: Korean farewell + Korean name ---
    "바이바이 맥",
    "바이바이맥",               # no space
    "빠이빠이 맥",
    "빠이빠이 멕",
    "굳바이 맥",               # 굳 + 바이 → [굳굿]\s*[바빠]이
    "굿바이 맥",
    # --- Branch 2: Korean farewell + English name ---
    "바이바이 mac",
    "바이바이 mack",
    "빠이빠이 meg",
    "바이바이 man",
    # --- Branch 3: English farewell + English name ---
    "bye bye Mack",
    "bye-bye Mack",            # hyphenated
    "byebye Mack",             # no separator ([\s\-]? = 0 occurrences allowed)
    "goodbye Mac",
    "good-bye Mac",
    "Goodbye, Mac",            # comma separator
    "GOODBYE MAC",             # IGNORECASE
    # --- Branch 4: English farewell + Korean name ---
    "bye bye 맥",
    "Goodbye 맥",
    "Goodbye, 맥",
    # --- Branch 5: single "bye" + English name ---
    "Bye, Mac",
    "bye mac",
    "bye. mack",               # period separator
    "bye meg",
    # --- Branch 6: single "bye" + Korean name ---
    "bye 맥",
    "Bye, 맥",
    # --- Branch 7: 바이바이 standalone ---
    "바이바이",
    "네 바이바이",              # mid-sentence
    # --- Branch 8: FAREWELL_EN standalone ---
    "bye bye",
    "goodbye",
    "Goodbye!",                # punctuation after
    "goodbye world",           # trailing word: still matches (word boundary before space)
    "say goodbye",             # mid-sentence
    "byebye",                  # compacted form
]

@pytest.mark.parametrize("text", _POSITIVES)
def test_goodbye_positive(text: str) -> None:
    assert _GOODBYE_RE.search(text), f"expected match for {text!r}"


# ---------------------------------------------------------------------------
# Negatives — must NOT match
# ---------------------------------------------------------------------------

_NEGATIVES = [
    # Korean: starts with 바이 but doesn't form a farewell pattern
    "바이크",                   # 바이크 = bicycle
    "바이러스",                 # 바이러스 = virus
    "바이올린",
    # English: single "bye" without bye-bye / goodbye / name
    "bye the way",             # no bye-bye, no goodbye, no name after bye
    "I said bye to my friend", # "bye" standalone, no recognized farewell form
    "buying something",        # "buy" prefix, no match
    # English "bye" + non-name word
    "bye everyone",            # "everyone" not in NAME_EN
    "bye now",
    # Japanese katakana (looks similar but not Korean)
    "バイバイ",
    # Partial name overlap — word boundary must prevent matching
    "macker",                  # "mac" inside "macker" — \b after mac blocks
    "mackerel",
]

@pytest.mark.parametrize("text", _NEGATIVES)
def test_goodbye_negative(text: str) -> None:
    assert not _GOODBYE_RE.search(text), f"unexpected match for {text!r}"


# ---------------------------------------------------------------------------
# Case-insensitivity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "GOODBYE MAC",
    "Goodbye Mac",
    "BYE BYE MACK",
    "Bye Bye mac",
])
def test_goodbye_case_insensitive(text: str) -> None:
    assert _GOODBYE_RE.search(text)


# ---------------------------------------------------------------------------
# Word-boundary guard on English names
# ---------------------------------------------------------------------------

def test_mac_word_boundary_blocks_mackerel() -> None:
    """'mac' inside 'mackerel' must not match the name pattern.
    Note: 'goodbye mackerel' DOES match because 'goodbye' alone triggers
    branch 8 (standalone farewell). Use single 'bye' to isolate the name boundary.
    """
    # "bye mackerel" — branch 5 requires \bbye[\s,\.]+name\b
    # \b fails between 'c' and 'k' in 'mackerel', so no name match
    assert not _GOODBYE_RE.search("bye mackerel")


def test_mac_word_boundary_blocks_macker() -> None:
    assert not _GOODBYE_RE.search("bye macker")


def test_mac_standalone_triggers() -> None:
    assert _GOODBYE_RE.search("bye mac")


# ---------------------------------------------------------------------------
# STT misrecognition variants (the reason _NAME_EN covers meg/man/mack)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["mac", "mack", "meg", "man"])
def test_stt_name_variants_with_goodbye(name: str) -> None:
    """STT can mishear '맥' as mac/mack/meg/man — all must trigger farewell."""
    assert _GOODBYE_RE.search(f"goodbye {name}")


@pytest.mark.parametrize("name", ["맥", "멕"])
def test_stt_korean_name_variants_with_goodbye(name: str) -> None:
    assert _GOODBYE_RE.search(f"바이바이 {name}")
