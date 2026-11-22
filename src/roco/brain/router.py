"""Picks the brain for each turn: Claude by default, local Ollama as the fallback.

Adam's decision of 2026-11-22 (docs/ARCHITECTURE.md, docs/ROCO_SPEC.md §9). Order:

1. **Claude** (:class:`~roco.brain.claude_brain.ClaudeBrain`), Haiku for normal turns, the
   escalation model (Sonnet) when Adam asks for a harder think ("think harder", "take your
   time", ... ``brain.claude.escalate_phrases``).
2. **Local** (:class:`~roco.brain.ollama_brain.OllamaBrain`) when Claude is not an option:
   no API key, "local only" privacy mode, the budget cap reached, a bad key (until
   restart), or an API error / timeout (local for ``brain.claude.retry_after_s``, then
   Claude is tried again). A failed Claude turn is answered on local at once.
3. **Offline line** (workstream D's canned line) when local is down too.

Both brains share one history and one personality, so a switch mid-conversation keeps
the context. ROCO says which brain she is on whenever it changes, in character, before
the answer. "ROCO, local only" / "go private" turns privacy mode on (nothing goes to the
cloud until "cloud back on"); it is saved in ``~/.roco/brain_state.json``. The config's
``brain.local_only: true`` forces it on.

Budget (:mod:`roco.brain.budget`): past 80 % of the daily cap ROCO mentions it once that
day, after her answer; at the daily or monthly cap she goes local until local midnight
or the first of next month, and says so.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from roco.brain.base import Brain, SentenceCallback, TurnHooks
from roco.brain.budget import CloudBudget
from roco.brain.core import BackendFailed, ToolBrain
from roco.bus import Emotion, EventBus
from roco.config import Config
from roco.fallback.claude import CloudPermanentError
from roco.paths import resolve
from roco.personality.commands import normalise

log = logging.getLogger(__name__)

# Lines ROCO says when the brain changes (spoken before the answer).
LOCAL_BAD_KEY = "My cloud key isn't working, so I'm on my local brain for now."
LOCAL_CLOUD_DOWN = "My cloud brain's not answering, so you've got my local one for now."
LOCAL_CLOUD_LOST = "Hang on, I've lost my cloud brain. Switching to my local one."
LOCAL_DAILY_CAP = "That's today's cloud budget spent, so I'm on my local brain until midnight."
LOCAL_MONTHLY_CAP = (
    "That's this month's cloud budget gone, so it's my local brain until the month's out."
)
BACK_ON_CLOUD = "And I'm back on my cloud brain. Much better."
BUDGET_WARNING = "By the way, I've used most of today's cloud budget."
ESCALATE_LINE = "Ooh, a proper question. Let me think harder about that."
PRIVATE_ON = "Private mode on. Nothing leaves this PC until you say cloud back on."
PRIVATE_ALREADY = "Already private. Nothing's going to the cloud."
PRIVATE_OFF = "Cloud back on. Big brain, back online."
PRIVATE_OFF_FORCED = "Private mode is fixed in my config, so I'm staying local."
PRIVATE_NOT_ON = "I wasn't private anyway. Cloud's on."
NO_CLOUD = "I'd love to, but I haven't got a cloud brain set up. Local it is."

# Whole-utterance voice commands (after roco.personality's normalise: no name, no filler).
PRIVATE_ON_CMD = re.compile(
    r"(?:go |switch to |stay |turn on |put on |be )?(?:local only|local mode|private|privacy"
    r"|private mode|privacy mode|incognito|offline mode)(?: mode)?(?: on| please)?"
    r"|(?:stay|go) local|no cloud|cloud off|turn (?:the )?cloud off|turn off (?:the )?cloud"
)
PRIVATE_OFF_CMD = re.compile(
    r"(?:turn )?(?:the )?cloud (?:back )?on|back to (?:the )?cloud|use (?:the )?cloud"
    r"(?: again)?|(?:private|privacy|local only) (?:mode )?off|turn off (?:private|privacy)"
    r"(?: mode)?|go public|you can use the cloud(?: again)?"
)
WHICH_BRAIN_CMD = re.compile(
    r"(?:which|what) brain (?:are you|is this|you)(?: on| using)?|are you (?:on )?(?:the )?"
    r"(?:cloud|local)(?: brain)?"
)

DEFAULT_ESCALATE = (
    r"\bthink (?:harder|hard|carefully|properly|it through|about it properly)\b",
    r"\b(?:use|bring out) (?:your|the) big brain\b",
    r"\btake your time\b",
    r"\bstep by step\b",
    r"\bin (?:depth|detail)\b",
    r"\bproper(?:ly)? (?:think|answer)\b",
)


class BrainRouter(Brain):
    """A :class:`Brain` that delegates each turn to Claude or local (see module doc)."""

    def __init__(
        self,
        config: Config,
        bus: EventBus,
        local: ToolBrain,
        claude: ToolBrain | None = None,
        *,
        budget: CloudBudget | None = None,
        no_cloud_reason: str | None = None,
        offline_line: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.bus = bus
        self.local = local
        self.claude = claude
        self.budget = budget
        self.history = local.history
        self.personality = local.personality
        b = config.section("brain")
        c = config.get_path("brain.claude", {}) or {}
        self.default = str(b.get("default", "claude"))
        self.forced_local_only = bool(b.get("local_only", False))
        self.retry_after_s = float(c.get("retry_after_s", 60.0))
        self.escalate = [re.compile(p, re.IGNORECASE)
                         for p in (c.get("escalate_phrases") or DEFAULT_ESCALATE)]
        self._no_cloud_reason = no_cloud_reason  # "no_key" when created without Claude
        self._offline_line = offline_line
        self._monotonic = monotonic
        self._state_path = resolve(str(b.get("state_file", "brain_state.json")))
        state = self._load_state()
        self.local_only = self.forced_local_only or bool(state.get("local_only", False))
        self._warned_day: str | None = state.get("budget_warned")
        self._cloud_broken = False  # bad key: no more cloud until restart
        self._cloud_retry_at = 0.0  # API error: local until then
        self._current: str | None = None  # "cloud" / "local" of the last answered turn
        self.active: ToolBrain = claude if claude is not None else local
        self._last_prefix: list[str] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ properties
    @property
    def name(self) -> str:
        return self.active.name

    @property
    def active_name(self) -> str:
        """The brain that answered the last turn (``claude:...`` / ``ollama:...``)."""
        return self.active.name

    @property
    def registry(self) -> Any:
        return self.local.registry

    @property
    def model(self) -> str | None:
        return getattr(self.active, "model", None)

    # ------------------------------------------------------------------ Brain API
    def respond(self, text: str, on_sentence: SentenceCallback | None = None) -> str:
        return self.respond_turn(text, on_sentence)

    def respond_turn(
        self,
        text: str,
        on_sentence: SentenceCallback | None = None,
        *,
        cancel: threading.Event | None = None,
        hooks: TurnHooks | None = None,
    ) -> str:
        with self._lock:
            said: list[str] = []
            self._last_prefix = []

            def emit(sentence: str) -> None:
                if cancel is not None and cancel.is_set():
                    return
                said.append(sentence)
                if on_sentence is not None:
                    try:
                        on_sentence(sentence)
                    except Exception:
                        log.exception("on_sentence callback failed")

            def own(line: str, emotion: Emotion = Emotion.NEUTRAL) -> None:
                """A line of the router's own, spoken before the brain's answer."""
                self.bus.set_emotion(emotion)
                emit(line)
                self._last_prefix.append(line)

            clean = (text or "").strip()
            command = self._privacy_command(clean)
            if command is not None:
                self.bus.set_emotion(Emotion.HAPPY)
                emit(command)
                self.history.add_turn(clean, command, Emotion.HAPPY.value, [])
                return command

            choice, reason = self._choose()
            brain = self.claude if choice == "cloud" else self.local
            assert brain is not None
            if self._current is not None or reason is not None:
                line = self._switch_line(choice, reason)
                if line:
                    own(line)
            escalate = choice == "cloud" and self._wants_escalation(clean)
            if escalate:
                own(ESCALATE_LINE, Emotion.CURIOUS)

            try:
                self._run(brain, clean, emit, cancel, hooks, escalate)
                self._current = choice
            except BackendFailed as exc:
                self._after_failure(brain, exc, clean, emit, own, cancel, hooks)
            if self._current == "cloud":
                self._maybe_warn(emit)
            return " ".join(said)

    def amend_last_reply(self, heard: str) -> None:
        heard = (heard or "").strip()
        for line in self._last_prefix:  # the router's own lines are not in the history
            if heard.startswith(line):
                heard = heard[len(line):].strip()
        self.history.amend_last_reply(heard)

    def reset(self) -> None:
        self.history.clear()

    def warm_up(self) -> bool:
        """Warm the local model (it must be ready the moment Claude fails)."""
        warm = getattr(self.local, "warm_up", None)
        return bool(warm()) if callable(warm) else True

    def warm_cloud(self) -> bool:
        """Pre-warm Claude (SDK import, connection, prompt cache) if it answers next.

        Measured: the first Claude turn of a process took 3.9 s to its first words, the
        next ones 0.6-0.7 s. Costs one cache write (~$0.0003 on Haiku) per start.
        """
        if self.claude is None or not self.cloud_available():
            return False
        warm = getattr(self.claude, "warm_up", None)
        return bool(warm()) if callable(warm) else False

    def close(self) -> None:
        for brain in (self.claude, self.local):
            if brain is not None:
                try:
                    brain.close()
                except Exception:
                    log.exception("closing %s failed", brain.name)

    def cloud_available(self) -> bool:
        """Would the next turn go to Claude?"""
        with self._lock:
            return self._choose()[0] == "cloud"

    # ------------------------------------------------------------------ choice
    def _choose(self) -> tuple[str, str | None]:
        """("cloud" | "local", why local or None)."""
        if self.local_only:
            return "local", "private"
        if self.claude is None:
            return "local", self._no_cloud_reason or ("default" if self.default == "local"
                                                      else "no_key")
        if self.default == "local":
            return "local", "default"
        if self._cloud_broken:
            return "local", "bad_key"
        if self._monotonic() < self._cloud_retry_at:
            return "local", "cloud_down"
        if self.budget is not None:
            status = self.budget.status()
            if status in ("daily", "monthly"):
                return "local", status
        return "cloud", None

    def _switch_line(self, choice: str, reason: str | None) -> str | None:
        """What to say before this turn's answer, or None (no change worth saying)."""
        if choice == "cloud":
            return BACK_ON_CLOUD if self._current == "local" else None
        if self._current == "local" or reason in ("private", "default", "no_key"):
            return None  # already said, or how ROCO is set up (no key: logged at start)
        return {
            "bad_key": LOCAL_BAD_KEY,
            "cloud_down": LOCAL_CLOUD_DOWN,
            "daily": LOCAL_DAILY_CAP,
            "monthly": LOCAL_MONTHLY_CAP,
        }.get(reason or "", LOCAL_CLOUD_DOWN)

    def _wants_escalation(self, text: str) -> bool:
        return any(p.search(text) for p in self.escalate)

    # ------------------------------------------------------------------ running
    def _run(self, brain: ToolBrain, text: str, emit: SentenceCallback,
             cancel: threading.Event | None, hooks: TurnHooks | None, escalate: bool) -> str:
        self.active = brain
        if hasattr(brain, "escalated"):
            brain.escalated = escalate  # type: ignore[attr-defined]
        try:
            return brain.respond_turn(text, emit, cancel=cancel, hooks=hooks)
        finally:
            if hasattr(brain, "escalated"):
                brain.escalated = False  # type: ignore[attr-defined]

    def _after_failure(self, brain: ToolBrain, exc: BackendFailed, text: str,
                       emit: SentenceCallback, own: Callable[..., None],
                       cancel: threading.Event | None, hooks: TurnHooks | None) -> str | None:
        """Claude failed: answer on local. Local failed: the offline line. None = said it."""
        if brain is self.claude:
            if isinstance(exc.cause, CloudPermanentError):
                log.error("Claude brain disabled until restart: %s", exc.cause)
                self._cloud_broken = True
                own(LOCAL_BAD_KEY if not exc.said else LOCAL_CLOUD_LOST)
            else:
                self._cloud_retry_at = self._monotonic() + self.retry_after_s
                own(LOCAL_CLOUD_DOWN if not exc.said else LOCAL_CLOUD_LOST)
            try:
                answer = self._run(self.local, text, emit, cancel, hooks, escalate=False)
                self._current = "local"
                return answer
            except BackendFailed as local_exc:
                log.warning("Local brain failed too: %s", local_exc.cause)
        else:
            log.warning("Local brain failed: %s", exc.cause)
        own(self.offline_line, Emotion.SAD)
        self._current = None  # say which brain again once one answers
        return None

    @property
    def offline_line(self) -> str:
        if self._offline_line is None:
            from roco.fallback.chain import OFFLINE_LINE

            self._offline_line = OFFLINE_LINE
        return self._offline_line

    def _maybe_warn(self, own_after: SentenceCallback) -> None:
        if self.budget is None:
            return
        try:
            spend = self.budget.spend()
        except Exception:
            log.exception("budget check failed")
            return
        today = self.budget.today().isoformat()
        if spend.status == "warn" and self._warned_day != today:
            self._warned_day = today
            self._save_state()
            own_after(BUDGET_WARNING)
            log.info("Cloud spend today £%.4f of £%.2f", spend.today_gbp, spend.daily_gbp)

    # ------------------------------------------------------------------ privacy mode
    def _privacy_command(self, text: str) -> str | None:
        t = normalise(text)
        if not t:
            return None
        if PRIVATE_ON_CMD.fullmatch(t):
            if self.local_only:
                return PRIVATE_ALREADY
            self.local_only = True
            self._save_state()
            log.info("Privacy mode on: local brain only")
            return PRIVATE_ON
        if PRIVATE_OFF_CMD.fullmatch(t):
            if self.forced_local_only:
                return PRIVATE_OFF_FORCED
            if self.claude is None:
                return NO_CLOUD
            if not self.local_only:
                return PRIVATE_NOT_ON
            self.local_only = False
            self._save_state()
            self._current = "cloud"  # PRIVATE_OFF already says it
            log.info("Privacy mode off: cloud brain allowed")
            return PRIVATE_OFF
        if WHICH_BRAIN_CMD.fullmatch(t):
            return self._which_brain_line()
        return None

    def _which_brain_line(self) -> str:
        choice, reason = self._choose()
        if choice == "cloud":
            return "I'm on my cloud brain, Claude. Say local only if you want me private."
        why = {
            "private": "private mode is on",
            "default": "that's how I'm set up",
            "no_key": "there's no cloud key",
            "bad_key": "my cloud key isn't working",
            "cloud_down": "the cloud isn't answering",
            "daily": "today's cloud budget is spent",
            "monthly": "this month's cloud budget is spent",
        }.get(reason or "", "the cloud's out of reach")
        return f"I'm on my local brain, because {why}."

    def _load_state(self) -> dict[str, Any]:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            log.warning("Brain state %s unreadable (%s); starting fresh", self._state_path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _save_state(self) -> None:
        data = {"local_only": self.local_only and not self.forced_local_only,
                "budget_warned": self._warned_day}
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(self._state_path)
        except OSError as exc:
            log.warning("Could not save the brain state %s: %s", self._state_path, exc)


def create_brain(config: Config, bus: EventBus, registry: Any = None, *,
                 sdk: Any = None, ollama_client: Any = None) -> BrainRouter:
    """The brain ``roco brain`` / ``roco chat`` / ``roco run`` use (see module doc).

    Without ``registry``, the standard tools are built (and the reminder scheduler
    started). ``sdk`` / ``ollama_client`` inject fakes (tests).
    """
    from roco.brain.claude_brain import ClaudeBrain, create_claude_client
    from roco.brain.core import history_from_config
    from roco.brain.ollama_brain import OllamaBrain
    from roco.fallback.keys import load_api_key
    from roco.personality import Personality

    if registry is None:
        from roco.tools import ToolContext, build_registry

        registry = build_registry(ToolContext(config=config, bus=bus))
    history = history_from_config(config)
    personality = Personality.from_config(config)
    local = OllamaBrain(config, bus, registry, client=ollama_client, history=history,
                        personality=personality, raise_backend_errors=True)
    claude = None
    reason = None
    wants_cloud = str(config.get_path("brain.default", "claude")) == "claude"
    if wants_cloud and config.get_path("cloud.enabled", True):
        key = load_api_key(config)
        if key or sdk is not None:
            client = create_claude_client(config, key, sdk=sdk)
            claude = ClaudeBrain(config, bus, registry, client=client, history=history,
                                 personality=personality, raise_backend_errors=True)
        else:
            reason = "no_key"
            log.warning("No Anthropic API key: the brain runs on local Ollama only "
                        "(roco cloud-key to add one)")
    router = BrainRouter(config, bus, local, claude, budget=CloudBudget.from_config(config),
                         no_cloud_reason=reason)
    log.info("Brain: %s%s", "Claude, local fallback" if claude else "local only",
             " (private mode on)" if router.local_only else "")
    return router
