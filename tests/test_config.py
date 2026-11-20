from __future__ import annotations

import pytest

from roco.config import DEFAULTS, ConfigError, deep_merge, load_config


def test_defaults_without_file(config):
    assert config.source is None
    assert config.get_path("llm.model") == "qwen2.5:7b"
    assert config.get_path("llm.host") == "http://localhost:11434"
    assert config.get_path("voice.wake_word.model") == "hey_jarvis"
    assert config.get_path("nope.missing", "fallback") == "fallback"


def test_deep_merge_keeps_siblings_and_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}, "list": [1, 2]}
    merged = deep_merge(base, {"a": {"b": 9}, "list": [3]})
    assert merged == {"a": {"b": 9, "c": 2}, "list": [3]}
    assert base == {"a": {"b": 1, "c": 2}, "list": [1, 2]}


def test_cwd_config_is_merged(tmp_path):
    (tmp_path / "config.yaml").write_text("llm:\n  model: llama3.2:3b\n", encoding="utf-8")
    cfg = load_config()
    assert cfg.source == tmp_path / "config.yaml"
    assert cfg.get_path("llm.model") == "llama3.2:3b"
    assert cfg.get_path("llm.host") == DEFAULTS["llm"]["host"]


def test_home_config_used_when_no_cwd_config(isolated_home):
    (isolated_home / "config.yaml").write_text("face:\n  fullscreen: true\n", encoding="utf-8")
    assert load_config().get_path("face.fullscreen") is True


def test_explicit_path_and_overrides(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("assistant:\n  user_name: Sam\n", encoding="utf-8")
    cfg = load_config(path, overrides={"assistant": {"timezone": "UTC"}})
    assert cfg.get_path("assistant.user_name") == "Sam"
    assert cfg.get_path("assistant.timezone") == "UTC"


def test_missing_explicit_path_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize("content", ["llm: [unclosed", "- just\n- a list\n"])
def test_bad_yaml_raises(tmp_path, content):
    (tmp_path / "config.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config()


def test_empty_file_means_defaults(tmp_path):
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    assert load_config().get_path("llm.model") == "qwen2.5:7b"


def test_example_config_parses_and_only_uses_known_keys():
    from pathlib import Path

    import yaml

    example = Path(__file__).parents[1] / "config.example.yaml"
    data = yaml.safe_load(example.read_text(encoding="utf-8"))

    def walk(node, defaults, prefix=""):
        for key, value in node.items():
            assert key in defaults, f"config.example.yaml has unknown key {prefix}{key}"
            if isinstance(value, dict) and isinstance(defaults[key], dict):
                walk(value, defaults[key], f"{prefix}{key}.")

    walk(data, DEFAULTS)
