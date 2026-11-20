"""``roco chat``: keyboard in, sentences printed (and optionally spoken) as they arrive.

Optional pieces from the other workstreams are looked up by convention and are
never required:

- ``--speak``: ``roco.voice.create_speaker(config, bus) -> Speaker`` (workstream B)
- ``--face``:  ``roco.body.create_face(config) -> Face`` (workstream C); a
  :class:`roco.body.base.Body` is attached to the bus to drive it.

If one is missing, a clear message is printed and chat continues without it.
Reminders arrive as ``announce`` events and are printed (and spoken) as they fire.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import threading
from collections.abc import Callable
from typing import Any, TextIO

from roco.bus import EventBus, State, Topic
from roco.config import Config

log = logging.getLogger(__name__)

PROMPT = "you> "
HELP = "Type to chat. Commands: /reset (forget the conversation), /quit (or Ctrl+C)."


def _load_factory(module: str, name: str) -> tuple[Callable[..., Any] | None, str]:
    """Import ``module.name``; return (factory, "") or (None, reason)."""
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        return None, f"{module} could not be imported ({exc})"
    factory = getattr(mod, name, None)
    if not callable(factory):
        return None, f"{module}.{name} is not available yet"
    return factory, ""


class ChatSession:
    """The text chat loop, separated from argparse/IO so it can be tested."""

    def __init__(
        self,
        brain: Any,
        bus: EventBus,
        speaker: Any = None,
        out: TextIO | None = None,
        name: str = "ROCO",
    ) -> None:
        self.brain = brain
        self.bus = bus
        self.speaker = speaker
        self.out = out or sys.stdout
        self.name = name
        self._print_lock = threading.Lock()
        self._unsubscribe = bus.subscribe(Topic.ANNOUNCE, self._on_announce)

    def close(self) -> None:
        self._unsubscribe()

    def _write(self, text: str) -> None:
        with self._print_lock:
            self.out.write(text)
            self.out.flush()

    def _on_announce(self, text: str) -> None:
        self._write(f"\n[{self.name}] {text}\n{PROMPT}")
        if self.speaker is not None:
            self.speaker.say(text)

    def handle(self, line: str) -> bool:
        """Process one input line. Returns False when the session should end."""
        line = line.strip()
        if not line:
            return True
        command = line.lower()
        if command in ("/quit", "/exit", "/q"):
            return False
        if command == "/reset":
            self.brain.reset()
            self._write("(conversation forgotten)\n")
            return True
        if command in ("/help", "/?"):
            self._write(HELP + "\n")
            return True

        self._write(f"{self.name}> ")
        first = True

        def on_sentence(sentence: str) -> None:
            nonlocal first
            self._write(("" if first else " ") + sentence)
            first = False
            if self.speaker is not None:
                self.speaker.say(sentence)

        self.brain.respond(line, on_sentence)
        self._write("\n")
        if self.speaker is not None:
            self.speaker.wait_done(timeout=120)
        else:
            self.bus.set_state(State.IDLE)  # nobody else ends "thinking" in text mode
        return True

    def run(
        self, read_line: Callable[[str], str] | None = None, stop: threading.Event | None = None
    ) -> None:
        read_line = read_line or input
        self._write(HELP + "\n")
        while stop is None or not stop.is_set():
            try:
                line = read_line(PROMPT)
            except (EOFError, KeyboardInterrupt, OSError):  # OSError: stdin closed/captured
                self._write("\n")
                break
            if stop is not None and stop.is_set():
                break
            if not self.handle(line):
                break
        self._write("Bye!\n")


def run_chat(args: argparse.Namespace, cfg: Config) -> int:
    from roco.brain.ollama_brain import create_brain
    from roco.tools import ToolContext, build_registry

    bus = EventBus()
    registry = build_registry(ToolContext(config=cfg, bus=bus))
    brain = create_brain(cfg, bus, registry)
    if getattr(args, "reset", False):
        brain.reset()

    speaker = None
    if getattr(args, "speak", False):
        factory, why = _load_factory("roco.voice", "create_speaker")
        try:
            speaker = factory(cfg, bus) if factory else None
        except Exception as exc:  # missing audio device, model files...
            why = f"the speaker failed to start ({exc})"
        if speaker is None:
            print(f"Speech is not available: {why}. Continuing text-only.", file=sys.stderr)

    face = None
    body = None
    if getattr(args, "face", False):
        factory, why = _load_factory("roco.body", "create_face")
        try:
            face = factory(cfg) if factory else None
        except Exception as exc:
            why = f"the face failed to start ({exc})"
        if face is None:
            print(f"The face is not available: {why}. Continuing without it.", file=sys.stderr)
        else:
            from roco.body.base import Body

            body = Body(face)
            body.attach(bus)

    name = str(cfg.get_path("assistant.name", "ROCO"))
    session = ChatSession(brain, bus, speaker, name=name)
    try:
        if face is None:
            session.run()
        else:
            from roco.runtime import run_with_face

            run_with_face(face, lambda stop: session.run(stop=stop))
    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        session.close()
        if body is not None:
            body.close()
        if speaker is not None:
            speaker.close()
        brain.close()
    return 0
