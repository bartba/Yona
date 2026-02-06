"""Configuration loader for Yona."""
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class Config:
    """Application configuration loaded from YAML files."""

    def __init__(self, config_path: str | Path | None = None):
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file. Defaults to config/default.yaml
        """
        load_dotenv()

        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"

        self._config_path = Path(config_path)
        self._data = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load and process configuration file."""
        with open(self._config_path) as f:
            data = yaml.safe_load(f)

        # Expand environment variables
        return self._expand_env_vars(data)

    def _expand_env_vars(self, obj: Any) -> Any:
        """Recursively expand ${VAR} patterns in config values."""
        if isinstance(obj, str):
            if obj.startswith("${") and obj.endswith("}"):
                var_name = obj[2:-1]
                return os.environ.get(var_name, "")
            return obj
        elif isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        return obj

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., 'audio.input_device')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @property
    def audio(self) -> dict[str, Any]:
        """Audio configuration."""
        return self._data.get("audio", {})

    @property
    def wake_word(self) -> dict[str, Any]:
        """Wake word configuration."""
        return self._data.get("wake_word", {})

    @property
    def vad(self) -> dict[str, Any]:
        """VAD configuration."""
        return self._data.get("vad", {})

    @property
    def stt(self) -> dict[str, Any]:
        """STT configuration."""
        return self._data.get("stt", {})

    @property
    def llm(self) -> dict[str, Any]:
        """LLM configuration."""
        return self._data.get("llm", {})

    @property
    def tts(self) -> dict[str, Any]:
        """TTS configuration."""
        return self._data.get("tts", {})

    @property
    def conversation(self) -> dict[str, Any]:
        """Conversation configuration."""
        return self._data.get("conversation", {})

    @property
    def logging(self) -> dict[str, Any]:
        """Logging configuration."""
        return self._data.get("logging", {})

    def get_system_prompt(self) -> str:
        """Load the system prompt from file."""
        prompt_path = self._config_path.parent / "prompts" / "system_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text().strip()
        return "You are a helpful voice assistant named Yona."
