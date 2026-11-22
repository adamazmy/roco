"""What may stop the body's fallback from calling Claude: private mode and the budget.

Two guards, checked before every cloud call (docs/ROCO_SPEC.md §9, Adam's decision of
2026-11-22):

- **Private mode** ("ROCO, local only" / "go private"): nothing goes to the cloud, not
  even from the body while the PC brain is down. The brain owns the mode (A's router,
  ``~/.roco/brain_state.json`` on the PC) and sends it in ``settings`` messages as
  ``{"brain": {"local_only": true}}``. The body passes those messages to
  :func:`note_brain_settings`, which saves the last value in
  ``~/.roco/fallback_state.json`` (``cloud.state_file``), so it still holds when the Pi
  boots with the PC off. ``brain.local_only: true`` in the body's config forces it on.
- **Budget** (``cloud.budget``, :mod:`roco.brain.budget`): today's and this month's
  spend in *this machine's* ``cloud_usage.jsonl``. The Pi keeps its own log, so its caps
  count the Pi's fallback calls only; the PC's caps count the PC's calls.

Both are cheap: one small JSON read and an incremental read of the usage log.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from roco.config import Config
from roco.paths import resolve

log = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "fallback_state.json"

# Why the cloud is off limits for this turn (None: it is allowed).
PRIVATE = "private"
DAILY = "daily"
MONTHLY = "monthly"


def state_file(cfg: Config) -> Path:
    return resolve(str(cfg.get_path("cloud.state_file", DEFAULT_STATE_FILE)
                       or DEFAULT_STATE_FILE))


def local_only_from_settings(values: Any) -> bool | None:
    """``{"brain": {"local_only": bool}}`` -> the bool; None when the message has no say."""
    if not isinstance(values, dict):
        return None
    brain = values.get("brain")
    if not isinstance(brain, dict):
        return None
    value = brain.get("local_only")
    return value if isinstance(value, bool) else None


class CloudGuard:
    """Answers "may the fallback call Claude now?" and remembers the brain's private mode."""

    def __init__(self, state_path: Path, *, forced_local_only: bool = False,
                 budget: Any = None, cfg: Config | None = None) -> None:
        self.state_path = state_path
        self.forced_local_only = forced_local_only
        self._budget = budget
        self._cfg = cfg  # builds the budget on first use when ``budget`` is None
        self._budget_tried = budget is not None
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: Config) -> CloudGuard:
        return cls(state_file(cfg),
                   forced_local_only=bool(cfg.get_path("brain.local_only", False)),
                   cfg=cfg)

    # ------------------------------------------------------------------ private mode
    @property
    def local_only(self) -> bool:
        return self.forced_local_only or bool(self._load().get("local_only", False))

    def set_local_only(self, on: bool) -> None:
        with self._lock:
            state = self._load()
            if bool(state.get("local_only", False)) == on and self.state_path.exists():
                return
            state["local_only"] = on
            self._save(state)
        log.info("Fallback private mode %s (from the brain)", "on" if on else "off")

    def apply_settings(self, values: Any) -> bool | None:
        """Take a brain ``settings`` / ``welcome`` snapshot; returns the mode it set, if any."""
        on = local_only_from_settings(values)
        if on is not None:
            self.set_local_only(on)
        return on

    # ------------------------------------------------------------------ budget
    def budget_status(self) -> str:
        """``ok`` / ``warn`` / ``daily`` / ``monthly``; ``ok`` if the budget can't be read."""
        budget = self._get_budget()
        if budget is None:
            return "ok"
        try:
            return str(budget.status())
        except Exception:
            log.exception("Fallback budget check failed; allowing the call")
            return "ok"

    def blocked(self) -> str | None:
        """Why the cloud may not be used now (``private`` / ``daily`` / ``monthly``), or None."""
        if self.local_only:
            return PRIVATE
        status = self.budget_status()
        return status if status in (DAILY, MONTHLY) else None

    # ------------------------------------------------------------------ internals
    def _get_budget(self) -> Any:
        if not self._budget_tried:
            self._budget_tried = True
            if self._cfg is not None:
                try:
                    from roco.brain.budget import CloudBudget

                    self._budget = CloudBudget.from_config(self._cfg)
                except ImportError:  # trimmed install without the brain package
                    log.warning("roco.brain.budget unavailable: fallback has no spend cap")
        return self._budget

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            log.warning("Fallback state %s unreadable (%s)", self.state_path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, state: dict[str, Any]) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError as exc:
            log.warning("Could not save the fallback state %s: %s", self.state_path, exc)


def note_brain_settings(cfg: Config, values: Any) -> bool | None:
    """For the body client: pass every brain ``settings`` message (and ``welcome.settings``).

    Saves the brain's private mode for the fallback, even before the fallback chain
    exists. Light, never raises. Returns the mode it saved, or None if the message had
    none.
    """
    try:
        return CloudGuard.from_config(cfg).apply_settings(values)
    except Exception:
        log.exception("note_brain_settings failed")
        return None
