"""Tests for src/config.py

Run with:
    pytest tests/test_config.py -v

All tests are self-contained — they use tmp_path fixtures and monkeypatch so
they never depend on the real .env file or default.yaml.
"""

from __future__ import annotations

import textwrap

import pytest

from src.config import Config, _expand, _load


# ---------------------------------------------------------------------------
# _expand helper
# ---------------------------------------------------------------------------

class TestExpand:
    """Unit tests for the internal _expand() function."""

    def test_simple_substitution(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "hello")
        assert _expand("${MY_KEY}") == "hello"

    def test_missing_var_becomes_empty_string(self):
        # A missing env var produces "" rather than raising KeyError.
        assert _expand("${YONA_NONEXISTENT_XYZ_123}") == ""

    def test_dict_values_expanded_recursively(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        result = _expand({"url": "http://${HOST}/api"})
        assert result == {"url": "http://localhost/api"}

    def test_list_items_expanded(self, monkeypatch):
        monkeypatch.setenv("VAL", "42")
        result = _expand(["${VAL}", "plain", 99])
        assert result == ["42", "plain", 99]

    def test_nested_dict_in_list(self, monkeypatch):
        monkeypatch.setenv("PORT", "8080")
        result = _expand([{"port": "${PORT}"}])
        assert result == [{"port": "8080"}]

    def test_non_string_scalar_passthrough(self):
        assert _expand(123) == 123
        assert _expand(3.14) == 3.14
        assert _expand(True) is True
        assert _expand(None) is None

    def test_multiple_vars_in_one_string(self, monkeypatch):
        monkeypatch.setenv("A", "foo")
        monkeypatch.setenv("B", "bar")
        assert _expand("${A}-${B}") == "foo-bar"


# ---------------------------------------------------------------------------
# _load helper
# ---------------------------------------------------------------------------

class TestLoad:
    """Unit tests for the internal _load() function."""

    def test_parses_yaml_and_returns_dict(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("audio:\n  chunk_size: 512\n")
        data = _load(cfg_file)
        assert data == {"audio": {"chunk_size": 512}}

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        assert _load(cfg_file) == {}

    def test_expands_env_vars_on_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEVICE", "MyMic")
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text('audio:\n  input_device: "${DEVICE}"\n')
        data = _load(cfg_file)
        assert data["audio"]["input_device"] == "MyMic"


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_cfg(tmp_path) -> Config:
    """A minimal Config loaded from an in-memory YAML string."""
    content = textwrap.dedent("""\
        audio:
          input_device: "TestDevice"
          chunk_size: 512
        stt:
          model_size: "large-v3-turbo"
        conversation:
          timeout_check_seconds: 15
          timeout_final_seconds: 5
    """)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(content)
    return Config(path=cfg_file)


class TestConfigProperties:
    def test_audio_property(self, simple_cfg):
        assert simple_cfg.audio["input_device"] == "TestDevice"
        assert simple_cfg.audio["chunk_size"] == 512

    def test_stt_property(self, simple_cfg):
        assert simple_cfg.stt["model_size"] == "large-v3-turbo"

    def test_conversation_property(self, simple_cfg):
        assert simple_cfg.conversation["timeout_check_seconds"] == 15

    def test_missing_section_returns_empty_dict(self, simple_cfg):
        # Sections not present in the YAML return {}, not None.
        assert simple_cfg.vad == {}
        assert simple_cfg.wake_word == {}
        assert simple_cfg.llm == {}
        assert simple_cfg.tts == {}
        assert simple_cfg.history == {}
        assert simple_cfg.logging == {}


class TestConfigGet:
    def test_dot_notation_two_levels(self, simple_cfg):
        assert simple_cfg.get("audio.chunk_size") == 512

    def test_dot_notation_returns_default_for_missing_key(self, simple_cfg):
        assert simple_cfg.get("audio.nonexistent", "default") == "default"

    def test_dot_notation_returns_none_by_default(self, simple_cfg):
        assert simple_cfg.get("audio.nonexistent") is None

    def test_dot_notation_missing_top_section(self, simple_cfg):
        assert simple_cfg.get("vad.threshold", 0.5) == 0.5

    def test_dot_notation_deep_nesting(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("a:\n  b:\n    c: 99\n")
        cfg = Config(path=cfg_file)
        assert cfg.get("a.b.c") == 99

    def test_non_dict_intermediate_returns_default(self, tmp_path):
        # If a middle node is not a dict (e.g. it's an int), return default.
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("audio:\n  chunk_size: 512\n")
        cfg = Config(path=cfg_file)
        assert cfg.get("audio.chunk_size.nonexistent", "x") == "x"


class TestConfigEnvExpansion:
    def test_env_var_in_yaml_is_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-test-123")
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text('llm:\n  openai_api_key: "${TEST_API_KEY}"\n')
        cfg = Config(path=cfg_file)
        assert cfg.llm["openai_api_key"] == "sk-test-123"

    def test_missing_env_var_becomes_empty_string(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text('llm:\n  custom_key: "${YONA_NO_SUCH_VAR}"\n')
        cfg = Config(path=cfg_file)
        assert cfg.llm["custom_key"] == ""


class TestConfigSystemPrompt:
    def test_returns_default_when_file_missing(self, tmp_path):
        # Config loaded from a tmp dir has no prompt file nearby — uses fallback.
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("audio: {}\n")
        cfg = Config(path=cfg_file)
        prompt = cfg.get_system_prompt()
        # The fallback must mention "Yona".
        assert "Yona" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_reads_prompt_file_when_present(self, tmp_path, monkeypatch):
        # Point _ROOT at tmp_path so get_system_prompt() finds our fake file.
        import src.config as config_module
        monkeypatch.setattr(config_module, "_ROOT", tmp_path)

        prompt_dir = tmp_path / "config" / "prompts"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "system_prompt.txt").write_text("  Custom prompt here.  ")

        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("audio: {}\n")
        cfg = Config(path=cfg_file)
        assert cfg.get_system_prompt() == "Custom prompt here."
