"""Threading model: the face owns the main thread, the assistant loop runs beside it.

pygame (and most windowing systems, macOS strictly) require the event loop on the
main thread, so ``roco run`` and ``roco chat --face`` look like::

    face = PygameFace(cfg)
    run_with_face(face, lambda stop: assistant_loop(stop))

Without a face the worker simply runs on the main thread.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from roco.body.base import Face

log = logging.getLogger(__name__)

Worker = Callable[[threading.Event], None]


def run_with_face(face: Face | None, worker: Worker) -> None:
    """Run ``worker(stop_event)`` alongside ``face.run()``.

    - Closing the face window sets ``stop_event``; the worker must poll it and return.
    - When the worker returns (or crashes), the face is stopped.
    - Ctrl+C on the main thread sets ``stop_event`` and stops the face.
    """
    stop = threading.Event()

    if face is None:
        try:
            worker(stop)
        except KeyboardInterrupt:
            stop.set()
        return

    def guarded() -> None:
        try:
            worker(stop)
        except Exception:
            log.exception("Assistant loop crashed")
        finally:
            stop.set()
            face.stop()

    thread = threading.Thread(target=guarded, name="roco-assistant", daemon=True)
    thread.start()
    try:
        face.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        face.stop()
        thread.join(timeout=5)
