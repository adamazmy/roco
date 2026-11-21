"""Builds the real voice components from config. Every heavy import is lazy.

Also hosts the small lookups ``roco run`` uses to construct the brain (workstream A)
and the face/body (workstream C) without importing them at module level.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from roco.bus import EventBus
from roco.config import Config

log = logging.getLogger(__name__)

VOICE_MODULES = ("numpy", "sounddevice", "openwakeword", "faster_whisper", "piper")


class VoiceUnavailable(RuntimeError):
    """The voice extra or its models are missing; the message says how to fix it."""


class ComponentUnavailable(RuntimeError):
    """Another workstream's component (brain, face) is not importable yet."""


def missing_voice_modules() -> list[str]:
    import importlib.util

    return [m for m in VOICE_MODULES if importlib.util.find_spec(m) is None]


def require_voice_extra() -> None:
    missing = missing_voice_modules()
    if missing:
        raise VoiceUnavailable(
            f"Voice dependencies missing ({', '.join(missing)}). "
            'Install them with: pip install -e ".[voice]"'
        )


def require_models(cfg: Config) -> None:
    from roco.voice.models import tts_model_files, wakeword_files

    missing = [p for p in [*wakeword_files(cfg), *tts_model_files(cfg)] if not p.is_file()]
    if missing:
        raise VoiceUnavailable(
            "Model files missing: " + ", ".join(p.name for p in missing)
            + ". Download them with: roco download-models"
        )


# --------------------------------------------------------------------------- voice


def create_tts(cfg: Config) -> Any:
    """The configured TTS callable: ``tts(text) -> Iterable[(int16 pcm, rate)]`` + ``load()``.

    ``voice.tts.engine`` picks piper or kokoro; ``voice.tts.postfx`` adds the EQ /
    compression / room post-processing on top of either (:mod:`roco.voice.postfx`).
    """
    from roco.voice.models import kokoro_paths, piper_voice_paths, tts_engine
    from roco.voice.postfx import PostFx, PostFxSettings, ProcessedTTS
    from roco.voice.tts import DEFAULT_NOISE_SCALE, DEFAULT_NOISE_W, PiperTTS, Synthesis

    def setting(key: str, default: float | None) -> float | None:
        value = cfg.get_path(f"voice.tts.{key}", default)
        return None if value is None else float(value)

    engine = tts_engine(cfg)
    tts: Any
    if engine == "kokoro":
        from roco.voice.kokoro_tts import KokoroTTS

        model, voices = kokoro_paths(cfg)
        kokoro = cfg.get_path("voice.tts.kokoro", {}) or {}
        threads = kokoro.get("threads")
        tts = KokoroTTS(
            model, voices,
            voice=str(kokoro.get("voice", "bf_emma")),
            speed=float(kokoro.get("speed", 1.0)),
            lang=str(kokoro.get("lang", "en-gb")),
            threads=int(threads) if threads else None,
        )
    elif engine == "piper":
        onnx, config_json = piper_voice_paths(cfg)
        settings = Synthesis(
            length_scale=setting("length_scale", 1.0),
            noise_scale=setting("noise_scale", DEFAULT_NOISE_SCALE),
            noise_w=setting("noise_w", DEFAULT_NOISE_W),
        )
        tts = PiperTTS(onnx, config_json, settings)
    else:
        raise VoiceUnavailable(f"Unknown voice.tts.engine {engine!r}: use piper or kokoro")
    fx = PostFxSettings.from_config(cfg.get_path("voice.tts.postfx"))
    return ProcessedTTS(tts, PostFx(fx)) if fx else tts


def create_speaker(cfg: Config, bus: EventBus | None, warm_up: bool = True) -> Any:
    """Piper + sounddevice speaker (implements :class:`roco.voice.base.Speaker`)."""
    require_voice_extra()
    from roco.voice.mic import find_device
    from roco.voice.speaker import PiperSpeaker, SoundDeviceOutput

    device = find_device(cfg.get_path("voice.output_device"), "output")
    tts = create_tts(cfg)
    if warm_up:
        tts.load()
        list(tts("Hello."))  # first inference is slow (onnxruntime graph optimisation)
    return PiperSpeaker(tts, SoundDeviceOutput(device), bus)


def create_transcriber(cfg: Config) -> Any:
    from roco.voice.models import whisper_dir
    from roco.voice.stt import WhisperTranscriber

    stt = cfg.get_path("voice.stt", {}) or {}
    return WhisperTranscriber(
        model=str(stt.get("model", "base.en")),
        device=str(stt.get("device", "cpu")),
        compute_type=str(stt.get("compute_type", "int8")),
        language=stt.get("language", "en") or None,
        vad_filter=bool(stt.get("vad_filter", True)),
        beam_size=int(stt.get("beam_size", 1)),
        download_root=str(whisper_dir()),
    )


def create_listener(cfg: Config, bus: EventBus | None, speaker: Any = None,
                    warm_up: bool = True) -> Any:
    """Mic + openWakeWord + faster-whisper listener. Muted while ``speaker`` talks."""
    require_voice_extra()
    from roco.voice.endpoint import RecordingSettings
    from roco.voice.listener import VoiceListener
    from roco.voice.mic import MicStream, find_device
    from roco.voice.models import oww_dir, wake_phrase, wakeword_model_path
    from roco.voice.wakeword import OpenWakeWordDetector

    wake = OpenWakeWordDetector(wakeword_model_path(cfg), oww_dir())
    transcriber = create_transcriber(cfg)
    transcriber.load()
    if warm_up:
        transcriber.warm_up()
    device = find_device(cfg.get_path("voice.input_device"), "input")
    rate = int(cfg.get_path("voice.sample_rate", 16000))
    frame_ms = int(cfg.get_path("voice.frame_ms", 80))
    mic = MicStream(device, sample_rate=rate, frame_samples=rate * frame_ms // 1000)
    return VoiceListener(
        mic, wake, transcriber, bus,
        recording=RecordingSettings.from_config(cfg.get_path("voice.recording", {}) or {}),
        wake_threshold=float(cfg.get_path("voice.wake_word.threshold", 0.5)),
        is_speaking=(lambda: speaker.is_speaking) if speaker is not None else None,
        frame_s=frame_ms / 1000,
        wake_phrase=wake_phrase(cfg),
    )


# --------------------------------------------------------------------------- other workstreams

# (module, attribute) candidates, tried in order. The first is the agreed entry point.
BRAIN_FACTORIES = (("roco.brain.factory", "create_brain"), ("roco.brain", "create_brain"))
BODY_FACTORIES = (("roco.body.factory", "create_body"), ("roco.body", "create_body"))


def _find_factory(candidates: tuple[tuple[str, str], ...]) -> Any:
    errors = []
    for module_name, attr in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name and module_name.startswith(exc.name):
                errors.append(f"{module_name}: not found")
                continue
            raise  # the module exists but one of ITS imports is missing: surface it
        factory = getattr(module, attr, None)
        if callable(factory):
            return factory
        errors.append(f"{module_name}.{attr}: not defined")
    raise ComponentUnavailable("; ".join(errors))


def create_brain(cfg: Config, bus: EventBus) -> Any:
    """Build workstream A's Brain via ``roco.brain.factory.create_brain(cfg, bus)``."""
    try:
        factory = _find_factory(BRAIN_FACTORIES)
    except ComponentUnavailable as exc:
        raise ComponentUnavailable(
            f"The brain is not available yet (workstream A, branch feat/brain): {exc}"
        ) from exc
    return factory(cfg, bus)


def create_body(cfg: Config, bus: EventBus) -> tuple[Any, Any]:
    """Build workstream C's face and body; returns (face, body) with the body attached."""
    try:
        factory = _find_factory(BODY_FACTORIES)
    except ComponentUnavailable as exc:
        raise ComponentUnavailable(f"The face is not available yet (workstream C): {exc}") from exc
    result = factory(cfg, bus)
    face, body = result if isinstance(result, tuple) else (result, None)
    return face, body
