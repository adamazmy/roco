"""Builds a real :class:`roco.client.body.BodyClient` from config. Heavy imports are lazy."""

from __future__ import annotations

import contextlib
import importlib
import logging
import platform
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from roco.bus import EventBus
from roco.config import Config
from roco.protocol import Connection, load_or_create_token
from roco.voice.factory import ComponentUnavailable, VoiceUnavailable

log = logging.getLogger(__name__)

BODY_MODULES = ("numpy", "sounddevice", "openwakeword", "onnxruntime")


def require_body_extra() -> None:
    import importlib.util

    missing = [m for m in BODY_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        raise VoiceUnavailable(
            f"Body dependencies missing ({', '.join(missing)}). "
            'Install them with: pip install -e ".[body]"'
        )


def link_token(cfg: Config) -> str:
    from roco.paths import resolve

    return load_or_create_token(resolve(str(cfg.get_path("link.token_file", "link_token"))))


def body_settings(cfg: Config) -> Any:
    from roco.client.body import BargeInMode, BodySettings
    from roco.client.vad import EndpointSettings

    vad = cfg.get_path("voice.vad", {}) or {}
    barge = cfg.get_path("voice.barge_in", {}) or {}
    frame_ms = int(cfg.get_path("voice.frame_ms", 80))
    return BodySettings(
        sample_rate=int(cfg.get_path("voice.sample_rate", 16000)),
        frame_ms=frame_ms,
        endpoint=EndpointSettings.from_config(vad),
        followup_s=float(cfg.get_path("voice.followup_s", 5.0)),
        wake_threshold=float(cfg.get_path("voice.wake_word.threshold", 0.5)),
        barge_in=BargeInMode(str(barge.get("mode", "speech"))),
        barge_threshold=float(barge.get("threshold", 0.6)),
        barge_start_ms=float(barge.get("start_ms", 96)),
        min_erle_db=float(cfg.get_path("voice.aec.min_erle_db", 10.0)),
    )


def fallback_factory(cfg: Config) -> Callable[[], Any] | None:
    """Workstream D's chain, ``roco.fallback.create_fallback(cfg)``, if installed."""
    try:
        module = importlib.import_module("roco.fallback")
    except ModuleNotFoundError as exc:
        if exc.name and "roco.fallback".startswith(exc.name):
            log.warning("No fallback brain available (roco.fallback missing)")
            return None
        raise
    create = getattr(module, "create_fallback", None)
    if not callable(create):
        log.warning("roco.fallback has no create_fallback(); fallback disabled")
        return None
    return lambda: create(cfg)


def local_tts_factory(cfg: Config) -> Callable[[], Any]:
    """Local TTS for fallback replies, loaded on first use (keeps the body small)."""

    def make() -> Any:
        from roco.voice.factory import create_tts

        tts = create_tts(cfg)
        if hasattr(tts, "load"):
            tts.load()
        return tts

    return make


@dataclass
class BodyParts:
    client: Any
    face: Any = None
    body: Any = None
    closers: list[Any] = field(default_factory=list)

    def close(self) -> None:
        for thing in reversed(self.closers):
            with contextlib.suppress(Exception):
                thing.close()


def build_body(
    cfg: Config,
    bus: EventBus,
    connect: Callable[[], Connection] | None,
    *,
    face: bool = True,
) -> BodyParts:
    """Mic → AEC → wake word + VAD, playback, face and servos, wired to ``connect``."""
    require_body_extra()
    from roco.client.body import BodyClient
    from roco.client.playback import CallbackOutput, Player
    from roco.client.vad import create_vad
    from roco.voice.aec import AecFrameSource, EchoMonitor, EchoReference, create_aec
    from roco.voice.mic import MicStream, find_device
    from roco.voice.models import oww_dir, silero_vad_path, wakeword_model_path
    from roco.voice.wakeword import OpenWakeWordDetector

    parts = BodyParts(client=None)
    face_obj = body_obj = None
    if face and cfg.get_path("face.enabled", True):
        try:
            from roco.body.factory import create_body

            face_obj, body_obj = create_body(cfg, bus)
            parts.closers.append(body_obj)
        except (ImportError, ComponentUnavailable) as exc:
            log.warning("Face unavailable, running headless: %s", exc)
    parts.face, parts.body = face_obj, body_obj

    try:
        wake = OpenWakeWordDetector(wakeword_model_path(cfg), oww_dir())
        vad = create_vad(str(cfg.get_path("voice.vad.engine", "silero")), silero_vad_path())
        aec_cfg = cfg.get_path("voice.aec", {}) or {}
        aec = create_aec(str(aec_cfg.get("engine", "auto")),
                         **{k: v for k, v in aec_cfg.items() if k != "engine"})
        reference = EchoReference()
        monitor = EchoMonitor()

        player = Player(rate=int(cfg.get_path("voice.playback.rate", 22050)))
        out_device = find_device(cfg.get_path("voice.output_device"), "output")
        output = CallbackOutput(player, out_device, reference,
                                latency=cfg.get_path("voice.playback.latency", 0.05))
        output.start()
        parts.closers.append(output)

        rate = int(cfg.get_path("voice.sample_rate", 16000))
        frame_ms = int(cfg.get_path("voice.frame_ms", 80))
        mic = MicStream(find_device(cfg.get_path("voice.input_device"), "input"),
                        sample_rate=rate, frame_samples=rate * frame_ms // 1000)
        source = AecFrameSource(mic, aec, reference, monitor)
        parts.closers.append(source)

        caps = {
            "face": face_obj is not None,
            "servos": str(cfg.get_path("servos.backend", "simulated")),
            "platform": f"{platform.system()} {platform.machine()}",
        }
        parts.client = BodyClient(
            source=source, vad=vad, player=player, bus=bus, connect=connect,
            token=link_token(cfg), wake=wake, settings=body_settings(cfg),
            body_id=platform.node() or "body", caps=caps, aec_name=aec.name,
            echo_monitor=monitor, fallback_factory=fallback_factory(cfg),
            local_tts_factory=local_tts_factory(cfg),
            output_latency=lambda: output.latency_s,
        )
        log.info("Body ready: AEC %s, VAD %s, barge-in %s", aec.name, vad.name,
                 parts.client.barge_mode)
    except BaseException:
        parts.close()
        raise
    return parts
