"""config.py — Single source of truth for all Yona settings.

Loads config/default.yaml, expands ${ENV_VAR} placeholders with values from
the .env file and the shell environment, then exposes a simple Config object.

Usage:
    from src.config import Config
    cfg = Config()
    cfg.audio["chunk_size"]           # dict key access
    cfg.get("audio.chunk_size", 512)  # dot-notation with default
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env into os.environ as early as possible so ${VAR} expansions work.
load_dotenv()

# Project root is two levels up from this file (src/config.py → src/ → project/).
_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expand(value: Any) -> Any:
    """Recursively expand ``${VAR}`` patterns using os.environ.

    Missing variables are replaced with an empty string rather than raising,
    so a typo in a variable name is immediately visible as a blank value.
    """
    if isinstance(value, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), ""),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value  # int, float, bool, None — pass through unchanged


def _load(path: Path) -> dict[str, Any]:
    """Parse a YAML file and expand all environment variable references."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _expand(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Config:
    """Thin, read-only wrapper around the parsed YAML config dict.

    All section properties (``audio``, ``vad``, etc.) return plain dicts so
    callers can do simple key lookups without coupling to this class.

    Args:
        path: Override the default config file location (useful for tests).
    """

    def __init__(self, path: Path | None = None) -> None:
        cfg_path = path or (_ROOT / "config" / "default.yaml")
        self._data: dict[str, Any] = _load(cfg_path)

    # --- dot-notation access ------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a nested value using dot notation.

        Example::

            cfg.get("audio.chunk_size", 512)
            cfg.get("conversation.timeout_check_seconds", 15)
        """
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict):
                return default
            node = node.get(part, None)
            if node is None:
                return default
        return node

    # --- typed section properties -------------------------------------------

    @property
    def audio(self) -> dict[str, Any]:
        return self._data.get("audio", {})

    @property
    def wake_word(self) -> dict[str, Any]:
        return self._data.get("wake_word", {})

    @property
    def vad(self) -> dict[str, Any]:
        return self._data.get("vad", {})

    @property
    def stt(self) -> dict[str, Any]:
        return self._data.get("stt", {})

    @property
    def llm(self) -> dict[str, Any]:
        return self._data.get("llm", {})

    @property
    def tts(self) -> dict[str, Any]:
        return self._data.get("tts", {})

    @property
    def conversation(self) -> dict[str, Any]:
        return self._data.get("conversation", {})

    @property
    def history(self) -> dict[str, Any]:
        return self._data.get("history", {})

    @property
    def logging(self) -> dict[str, Any]:
        return self._data.get("logging", {})

    # --- special loaders ----------------------------------------------------

    def get_system_prompt(self) -> str:
        """Return the system prompt text.

        Reads ``config/prompts/system_prompt.txt`` relative to the project
        root.  Falls back to a minimal built-in string so the app can still
        run if the file is missing.
        """
        prompt_path = _ROOT / "config" / "prompts" / "system_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return "You are Yona, a helpful voice assistant. Be concise and friendly."
