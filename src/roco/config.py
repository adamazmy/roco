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
        # Extra Ollama model options sent with every request. On a GPU where the model
        # only just fits, {"num_gpu": 99} forces every layer onto it (much faster than
        # Ollama's cautious CPU/GPU split); leave empty if loading then fails.
        "options": {},
        "timeout_s": 120,
        "max_tool_rounds": 5,
    },
    "link": {
        # Brain server <-> body client WebSocket (docs/PROTOCOL.md).
        # Server side: 0.0.0.0 listens on the LAN and Tailscale; every body must
        # send the shared token from token_file (copy that file to the Pi).
        "host": "0.0.0.0",
        "port": 8765,
        # Body side: where the brain is. Home LAN or Tailscale address of the PC.
        "brain_url": "ws://127.0.0.1:8765",
        "token_file": "link_token",  # relative paths live in ~/.roco
    },
    "server": {
        # Brain server (roco brain / roco run). One JSON line per turn, in ~/.roco.
        "latency_log": "latency.jsonl",
        # Stripped from the start of voice transcripts, besides the wake word's own phrase
        # (the body's pre-roll catches it). Whisper's mishearings of "hey Jarvis".
        "wake_aliases": ["hey george"],
        "tts_frame_ms": 80,  # TTS audio block size sent to the body (50-100 ms)
        "filler": {
            # Said at once when a slow tool is about to run and nothing was said yet.
            "enabled": True,
            "lines": ["Let me check.", "One moment.", "Let me have a look."],
            # Tools fast enough to need no filler line.
            "fast_tools": ["get_current_time", "set_reminder", "list_reminders",
                           "cancel_reminder"],
        },
        # Streaming STT (roco.server.stt): decode while Adam is still talking.
        "stt": {
            "speculative_quiet_ms": 240,
            "partial_min_s": 3.0,
            "partial_every_s": 1.0,
            "keep_tail_s": 1.0,
        },
    },
    "personality": {
        # chill | default | unhinged | professional (docs/ROCO_SPEC.md §2.1). Any of
        # roast, flirt, sass, chattiness, proactivity, energy (0-100) and the switches
        # sensitive_topics_ok, roast_guests, tone_down_with_guests can be set here too,
        # over the preset. Changes made by voice ("ROCO, roast level max") are saved in
        # `file` and win over this section until "ROCO, back to normal".
        "preset": "default",
        "file": "personality.json",  # relative paths live in ~/.roco
        "people_file": "people.json",
        # Per-person profiles over the built-in ones (Adam, Yousaf), e.g.
        # {"yousaf": {"overrides": {"roast": 100}}}. See roco.personality.people.
        "people": {},
    },
    "cloud": {
        # Fallback brain on the body when the PC brain is unreachable (roco.fallback,
        # docs/ROCO_SPEC.md §9): Claude called straight from the body, STT on the body.
        "enabled": True,
        "model": "claude-haiku-5-5",
        "effort": "low",  # low | medium | high: low keeps the first sentence fast
        "max_tokens": 1024,  # thinking counts towards this; replies are 1-3 sentences
        "connect_timeout_s": 3.0,
        "read_timeout_s": 8.0,  # longest silence allowed while streaming
        "total_timeout_s": 20.0,  # whole reply, wall clock
        "max_retries": 1,
        "history_turns": 6,  # user+assistant pairs kept while on the fallback
        "api_key_file": "anthropic_key",  # or ANTHROPIC_API_KEY; never in the repo
        "usage_file": "cloud_usage.jsonl",  # one line per cloud call
        "stt_model": "tiny.en",  # faster-whisper on the body, loaded on first use
        # USD per million tokens, for the cost estimate in usage_file. Check
        # https://claude.com/pricing when you change the model.
        "pricing": {
            "claude-haiku-5-5": {
                "input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125,
            },
            "claude-sonnet-5-5": {
                "input": 2.00, "output": 10.00, "cache_read": 0.10, "cache_write": 2.50,
            },
            "claude-opus-5-5": {
                "input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00,
            },
        },
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
        # Body client (roco body / roco run): end of speech is decided on the body.
        "vad": {
            "engine": "silero",  # silero | energy
            "threshold": 0.5,
            "trailing_ms": 450,  # silence that ends a turn (fast; raise if it cuts you off)
            "start_ms": 96,
            "max_s": 15.0,
            "start_timeout_s": 4.0,
        },
        # Echo cancellation keeps the mic open while ROCO talks (barge-in).
        "aec": {
            "engine": "auto",  # auto (webrtc, else speex, else none) | webrtc | speex | none
            "min_erle_db": 10.0,  # weaker than this while speaking -> wake-word barge-in
        },
        "barge_in": {
            "mode": "speech",  # speech (any speech, needs AEC) | wake_word | off
            "threshold": 0.6,
            "start_ms": 96,  # speech needed before ROCO stops (hardware test: 156-220 ms)
        },
        "playback": {
            "rate": 22050,  # reopened at the brain's tts_rate on connect
            # PortAudio output latency (seconds, or low | high). Part of the barge-in
            # stop time; raise it if ROCO's voice crackles.
            "latency": 0.05,
        },
        "stt": {
            "model": "base.en",
            "device": "cpu",
            "compute_type": "int8",
            "language": "en",
            "vad_filter": True,
        },
        "tts": {
            # piper | kokoro (docs/ROCO_SPEC.md §3; scripts/voice_ab.py compares them).
            "engine": "piper",
            # Piper: voice name (or .onnx path) and synthesis settings. Slightly slower and
            # less jittery than the voice's defaults: Adam's pick in the Phase 3 A/B.
            "voice": "en_GB-cori-high",
            "length_scale": 1.08,
            "noise_scale": 0.4,
            "noise_w": 0.25,
            # Light EQ + soft compression + small room (roco.voice.postfx): false, true
            # or a mapping such as {"reverb": 0.1, "compress": false}.
            "postfx": True,
            "kokoro": {
                "voice": "bf_emma",
                "speed": 1.0,
                "lang": "en-gb",
                "model": "kokoro-v1.0.onnx",  # names live in ~/.roco/models/kokoro
                "voices": "voices-v1.0.bin",
                "threads": None,  # onnxruntime CPU threads; None = all cores
            },
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
