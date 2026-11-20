"""Configuration: built-in defaults deep-merged with an optional ``config.yaml``.

Lookup order for the user file (first hit wins):
1. the path passed explicitly (``roco --config path``),
2. ``ROCO_CONFIG`` environment variable,
3. ``./config.yaml`` in the current directory,
4. ``~/.roco/config.yaml``.

Only keys that differ from the defaults need to be in the user file.
"""

from __future__ import annotations

import copy
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from roco.paths import roco_home

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "assistant": {
        "name": "ROCO",
        "user_name": "Adam",
        "timezone": "Europe/London",
    },
    "llm": {
        # Point this at the PC from the Pi, e.g. http://192.168.1.20:11434
        "host": "http://localhost:11434",
        "model": "qwen2.5:7b",
        "temperature": 0.6,
        "num_ctx": 4096,
        "keep_alive": "30m",
        "timeout_s": 120,
        "max_tool_rounds": 5,
    },
    "history": {
        "max_turns": 12,
        "persist": True,
        "file": "history.json",  # relative paths live in ~/.roco
    },
    "reminders": {
        "file": "reminders.json",
    },
    "voice": {
        "input_device": None,  # None = system default; name substring or index
        "output_device": None,
        "sample_rate": 16000,
        "frame_ms": 80,  # openWakeWord expects 1280-sample (80 ms) frames
        "wake_word": {
            "model": "hey_jarvis",
            "threshold": 0.5,
            "inference_framework": "onnx",  # tflite is not available on Windows
        },
        "recording": {
            "silence_s": 0.9,
            "max_s": 15.0,
            "noise_floor_margin": 2.5,
        },
        "followup_s": 5.0,
        "stt": {
            "model": "base.en",
            "device": "cpu",
            "compute_type": "int8",
            "language": "en",
            "vad_filter": True,
        },
        "tts": {
            "voice": "en_GB-alan-medium",
            "length_scale": 1.0,
        },
    },
    "face": {
        "enabled": True,
        "width": 800,
        "height": 480,
        "fullscreen": False,
        "fps": 30,
        "eye_color": "#00E5FF",
        "background": "#000000",
    },
    "servos": {
        "backend": "simulated",  # simulated | pca9685
        "i2c_address": 0x40,
        "pan": {"channel": 0, "min": 30, "max": 150, "center": 90},
        "tilt": {"channel": 1, "min": 60, "max": 120, "center": 90},
    },
    "google": {
        "enabled": False,
        "credentials_file": "google_credentials.json",
        "token_file": "google_token.json",
        "calendar_id": "primary",
    },
    "email": {
        # Which account the email tools read: google (Gmail) | outlook | none.
        # Google Calendar keeps working whatever this says.
        "provider": "google",
    },
    "outlook": {
        # Application (client) ID of your own Microsoft Entra app registration.
        "client_id": "",
        # consumers = personal accounts (hotmail.com, outlook.com, live.com).
        "authority": "https://login.microsoftonline.com/consumers",
        "token_cache_file": "outlook_token.json",
    },
    "logging": {
        "level": "INFO",
    },
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``override`` merged recursively over ``base``.

    Nested mappings are merged; any other value (lists included) is replaced.
    """
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class ConfigError(ValueError):
    """The user config file is unreadable or has the wrong shape."""


class Config(dict):
    """A plain dict with dotted-path access: ``cfg.get_path("llm.model")``."""

    source: Path | None = None

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name)
        return value if isinstance(value, dict) else {}


def find_config_file(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        return path
    if env := os.environ.get("ROCO_CONFIG"):
        candidates.append(Path(env).expanduser())
    candidates += [Path.cwd() / "config.yaml", roco_home() / "config.yaml"]
    return next((c for c in candidates if c.is_file()), None)


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Load defaults, merge the user file (if any), then ``overrides`` (for CLI flags/tests)."""
    merged = copy.deepcopy(DEFAULTS)
    source = find_config_file(path)
    if source is not None:
        try:
            data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {source}: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ConfigError(f"{source} must contain a mapping at the top level")
        unknown = sorted(set(data) - set(DEFAULTS))
        if unknown:
            log.warning("Unknown config sections in %s: %s", source, ", ".join(unknown))
        merged = deep_merge(merged, data)
    if overrides:
        merged = deep_merge(merged, overrides)
    cfg = Config(merged)
    cfg.source = source
    return cfg
