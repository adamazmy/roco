"""Everything the brain server loads once and shares between body sessions."""

from __future__ import annotations

import contextlib
import copy
import logging
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roco.brain.base import Brain
from roco.bus import EventBus
from roco.config import Config, deep_merge
from roco.paths import resolve
from roco.protocol import load_or_create_token

log = logging.getLogger(__name__)

TTS = Callable[[str], Iterator[tuple[bytes, int]]]
SettingsListener = Callable[[dict[str, Any]], None]


class BrainUnavailable(RuntimeError):
    """The brain's speech dependencies or models are missing; the message says the fix."""


@dataclass(frozen=True)
class FillerLine:
    """A pre-synthesized instant line ("Let me check.")."""

    text: str
    audio: tuple[tuple[bytes, int], ...]


@dataclass
class BrainContext:
    """Built once by :func:`create_brain_context`; passed to every ``serve_connection``.

    ``stt`` has ``segments(pcm, rate)`` (or is a callable ``(pcm, rate) -> str``);
    ``tts`` is a callable ``text -> Iterator[(pcm16 bytes, rate)]``.
    """

    cfg: Config
    bus: EventBus
    brain: Brain
    stt: Any
    tts: TTS
    tts_rate: int
    token: str
    brain_name: str = "brain"
    fillers: list[FillerLine] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    latency_path: Path | None = None
    stt_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(1, thread_name_prefix="roco-stt")
    )
    body_lock: threading.Lock = field(default_factory=threading.Lock)
    _settings_lock: threading.Lock = field(default_factory=threading.Lock)
    _listeners: list[SettingsListener] = field(default_factory=list)
    _closers: list[Callable[[], None]] = field(default_factory=list)
    llm_preload: threading.Thread | None = None

    def current_brain_name(self) -> str:
        """The brain answering now (``claude:...`` / ``ollama:...``: the router switches)."""
        name = getattr(self.brain, "active_name", None)
        return name if isinstance(name, str) and name else self.brain_name

    def cloud_ready(self) -> bool:
        """True when the router will answer on Claude, so the local model can load later."""
        check = getattr(self.brain, "cloud_available", None)
        try:
            return bool(check()) if callable(check) else False
        except Exception:
            return False

    def wait_until_ready(self, timeout: float = 120.0) -> bool:
        """Wait for the background Ollama preload (a cold 7B load takes ~20 s).
        True when loaded (or nothing to wait for)."""
        if self.llm_preload is None:
            return True
        self.llm_preload.join(timeout)
        return not self.llm_preload.is_alive()

    # ------------------------------------------------------------------ settings
    def on_settings(self, listener: SettingsListener) -> Callable[[], None]:
        """Call ``listener(full_settings)`` after every change. Returns an unsubscribe."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def apply_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial change; returns the full snapshot that is broadcast.

        The ``personality`` part goes to ``brain.personality.apply_settings`` when the
        brain has one (workstream C owns those settings); the rest is deep-merged here.
        Raises ValueError for a value the personality refuses (nothing is applied then).
        """
        values = dict(values or {})
        personality = self.personality
        if personality is not None and "personality" in values:
            try:
                personality.apply_settings({"personality": values.pop("personality")})
            except ValueError:
                raise
            except Exception:
                log.exception("personality.apply_settings failed")
        with self._settings_lock:
            self.settings = deep_merge(self.settings, values)
        snapshot = self.settings_snapshot()
        for listener in list(self._listeners):
            try:
                listener(copy.deepcopy(snapshot))
            except Exception:
                log.exception("settings listener failed")
        return snapshot

    def settings_snapshot(self) -> dict[str, Any]:
        with self._settings_lock:
            snapshot = copy.deepcopy(self.settings)
        personality = self.personality
        if personality is not None:
            try:
                snapshot = deep_merge(snapshot, personality.snapshot() or {})
            except Exception:
                log.exception("personality.snapshot failed")
        return snapshot

    @property
    def personality(self) -> Any:
        """``brain.personality`` if the brain has one with the settings API, else None."""
        personality = getattr(self.brain, "personality", None)
        if personality is not None and hasattr(personality, "apply_settings"):
            return personality
        return None

    def watch_brain_settings(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Call ``callback()`` when the brain changes settings itself (a voice command
        like "be more sarcastic"). Returns an unsubscribe function."""
        personality = self.personality
        on_change = getattr(personality, "on_change", None)
        if not callable(on_change):
            return lambda: None
        unsubscribe = on_change(lambda _values: callback())
        return unsubscribe if callable(unsubscribe) else (lambda: None)

    # ------------------------------------------------------------------ config
    def server_cfg(self, dotted: str, default: Any = None) -> Any:
        return self.cfg.get_path(f"server.{dotted}", default)

    def close(self) -> None:
        self.stt_executor.shutdown(wait=False, cancel_futures=True)
        for closer in reversed(self._closers):
            with contextlib.suppress(Exception):
                closer()
        close = getattr(self.brain, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()


def synthesize_fillers(tts: TTS, lines: list[str]) -> list[FillerLine]:
    out = []
    for line in lines:
        line = str(line).strip()
        if not line:
            continue
        try:
            out.append(FillerLine(line, tuple(tts(line))))
        except Exception as exc:
            log.warning("Could not pre-synthesize filler %r: %s", line, exc)
    return out


def create_brain_context(
    cfg: Config,
    *,
    bus: EventBus | None = None,
    brain: Brain | None = None,
    stt: Any = None,
    tts: TTS | None = None,
    token: str | None = None,
    warm_up: bool = True,
) -> BrainContext:
    """Load the brain, STT, TTS and filler lines once (a few seconds on first start).

    Anything passed in is used as is (tests, ``roco run`` sharing a bus); the rest is
    built from ``cfg``: ``roco.brain.factory.create_brain`` (tools + reminder scheduler,
    Ollama model preloaded in the background), faster-whisper on ``voice.stt.device``
    (CPU by default: the 6 GB laptop GPU is full with qwen2.5:7b) and
    ``roco.voice.factory.create_tts``.
    """
    bus = bus or EventBus()
    preload = _preload_llm(cfg) if brain is None else None  # loads while speech loads
    try:
        stt, tts, tts_rate, fillers = _load_speech(cfg, stt, tts, warm_up)
    except BrainUnavailable:
        raise
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        # OSError covers FileNotFoundError (missing voice model).
        raise BrainUnavailable(
            f"Could not load speech-to-text / text-to-speech: {exc}. Install the brain "
            'extra (pip install -e ".[brain]") and run: roco download-models'
        ) from exc
    if brain is None:
        from roco.brain.factory import create_brain

        brain = create_brain(cfg, bus)
        warm_cloud = getattr(brain, "warm_cloud", None)
        if warm_up and callable(warm_cloud):  # Claude answers before Ollama is loaded
            threading.Thread(target=warm_cloud, name="roco-cloud-warm-up", daemon=True).start()
    if token is None:
        token = load_or_create_token(resolve(cfg.get_path("link.token_file", "link_token")))
    name = getattr(brain, "name", None)
    model = getattr(brain, "model", None)
    if isinstance(name, str) and name:
        brain_name = name
    else:
        brain_name = f"ollama:{model}" if model else type(brain).__name__
    latency = cfg.get_path("server.latency_log", "latency.jsonl")
    return BrainContext(
        cfg=cfg,
        bus=bus,
        brain=brain,
        stt=stt,
        tts=tts,
        tts_rate=int(tts_rate),
        token=token,
        brain_name=brain_name,
        fillers=fillers,
        latency_path=resolve(latency) if latency else None,
        llm_preload=_then_warm_up(preload, brain),  # see wait_until_ready
    )


def _then_warm_up(preload: threading.Thread | None, brain: Brain) -> threading.Thread | None:
    """After the model load, one chat with the real prompt (``OllamaBrain.warm_up``), so
    Adam's first question isn't the slow one. ``wait_until_ready`` waits for both."""
    warm = getattr(brain, "warm_up", None)
    if preload is None or not callable(warm):
        return preload

    def run() -> None:
        preload.join()
        warm()

    thread = threading.Thread(target=run, name="roco-llm-warm-up", daemon=True)
    thread.start()
    return thread


def _load_speech(
    cfg: Config, stt: Any, tts: TTS | None, warm_up: bool
) -> tuple[Any, TTS, int, list[FillerLine]]:
    if stt is None:
        from roco.voice.factory import create_transcriber

        stt = create_transcriber(cfg)
    if tts is None:
        from roco.voice.factory import create_tts

        tts = create_tts(cfg)
    t0 = time.monotonic()
    errors: list[BaseException] = []
    warm = None
    if warm_up and hasattr(stt, "load"):
        def warm_stt() -> None:
            try:
                stt.load()
                if hasattr(stt, "warm_up"):
                    stt.warm_up()
            except BaseException as exc:  # re-raised on the caller's thread
                errors.append(exc)

        warm = threading.Thread(target=warm_stt, name="roco-stt-warmup", daemon=True)
        warm.start()
    # The first synthesis is slow (onnxruntime graph optimisation) and gives the rate.
    first = list(tts("Hello.")) if warm_up else []
    tts_rate = first[0][1] if first else int(cfg.get_path("server.tts_rate", 22050))
    filler_cfg = cfg.get_path("server.filler", {}) or {}
    fillers = (
        synthesize_fillers(tts, list(filler_cfg.get("lines", [])))
        if filler_cfg.get("enabled", True)
        else []
    )
    if warm is not None:
        warm.join()
    if errors:
        raise errors[0]
    log.info("Speech models ready in %d ms (TTS %d Hz, %d filler lines)",
             (time.monotonic() - t0) * 1000, tts_rate, len(fillers))
    return stt, tts, int(tts_rate), fillers


def _preload_llm(cfg: Config) -> threading.Thread:
    """Ask Ollama to load the model now (empty generate), in the background."""

    def load() -> None:
        import httpx

        host = str(cfg.get_path("llm.host", "http://localhost:11434")).rstrip("/")
        host = host.replace("://localhost", "://127.0.0.1")  # Windows: ::1 costs ~2 s
        body = {"model": cfg.get_path("llm.model"),
                "keep_alive": cfg.get_path("llm.keep_alive", "30m")}
        # Same options as the chat requests: a different num_ctx/num_gpu makes Ollama
        # reload the model on the first real turn.
        options = dict(cfg.get_path("llm.options") or {})
        if cfg.get_path("llm.num_ctx") is not None:
            options["num_ctx"] = cfg.get_path("llm.num_ctx")
        if options:
            body["options"] = options
        try:
            t0 = time.monotonic()
            httpx.post(f"{host}/api/generate", json=body, timeout=120).raise_for_status()
            log.info("LLM %s loaded in %d ms", body["model"], (time.monotonic() - t0) * 1000)
        except Exception as exc:
            log.warning("Could not preload the LLM: %s", exc)

    thread = threading.Thread(target=load, name="roco-llm-preload", daemon=True)
    thread.start()
    return thread
