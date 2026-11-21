"""One call for ``roco run`` / ``roco chat --face``: build the face and body, wired to the bus.

    face, body = create_body(cfg, bus)
    try:
        run_with_face(face, worker)
    finally:
        body.close()
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Mapping
from typing import Any

from roco.body.base import Body, Face, NullFace
from roco.bus import EventBus

log = logging.getLogger(__name__)


def create_face(cfg: Mapping[str, Any]) -> Face:
    """The configured face alone: :class:`PygameFace`, or :class:`NullFace` when
    ``face.enabled`` is false. Raises :class:`ImportError` if pygame is missing."""
    face_cfg = cfg.get("face", {})
    if isinstance(face_cfg, Mapping) and face_cfg.get("enabled", True) is False:
        return NullFace()
    if importlib.util.find_spec("pygame") is None:
        raise ImportError('The face needs pygame: pip install -e ".[face]"')
    from roco.body.face import PygameFace

    return PygameFace(cfg)


def create_body(cfg: Mapping[str, Any], bus: EventBus) -> tuple[Face, Body]:
    """Return ``(face, body)`` with the body already attached to ``bus``.

    - ``face.enabled: false`` gives a :class:`NullFace` (servos still work).
    - If pygame is not installed this raises :class:`ImportError` so the caller can
      fall back to headless mode itself.
    - Servos come from ``servos.backend`` (simulated by default); idle look-around is on.
    """
    from roco.body.robot import create_body as create_robot_body

    face = create_face(cfg)
    body = create_robot_body(face, cfg)
    body.attach(bus)
    return face, body
