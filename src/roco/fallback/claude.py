"""Claude Messages API client for the fallback brain.

Streams the reply so the first sentence can be spoken while the rest is generated,
enforces connect / read / total timeouts, logs every call's tokens and estimated cost,
and turns SDK errors into two kinds the chain cares about:

- :class:`CloudPermanentError`: retrying will not help (bad or missing key, unknown model).
  The chain stops using Claude until it is restarted.
- :class:`CloudUnavailableError`: network down, timeout, rate limit, server error. The
  chain says the offline line for this turn and tries Claude again next turn.

The ``anthropic`` SDK is imported only when the first request is made, so importing
this module costs nothing on the Pi until the fallback is actually used.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from roco.config import Config
from roco.fallback.usage import Usage, UsageLog

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-5-5"
TextCallback = Callable[[str], None]


class CloudError(Exception):
    """A Claude call failed."""


class CloudPermanentError(CloudError):
    """Retrying will not help: bad/missing API key, no access, unknown model, bad request."""


class CloudUnavailableError(CloudError):
    """Temporary: network, timeout, rate limit, overloaded or server error."""


@dataclass
class ClaudeReply:
    text: str
    usage: Usage
    stop_reason: str | None
    cancelled: bool = False
    first_text_ms: int | None = None
    total_ms: int | None = None


class ClaudeClient:
    """One streaming request per :meth:`stream_reply`. Thread-safe for one caller at a time."""

    def __init__(
        self,
        api_key: str | None,
        model: str = DEFAULT_MODEL,
        *,
        effort: str | None = "low",
        max_tokens: int = 1024,
        connect_timeout_s: float = 3.0,
        read_timeout_s: float = 8.0,
        total_timeout_s: float = 20.0,
        max_retries: int = 1,
        usage_log: UsageLog | None = None,
        client: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self.total_timeout_s = total_timeout_s
        self.max_retries = max_retries
        self.usage_log = usage_log
        self._client = client
        self._clock = clock
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: Config, api_key: str | None, client: Any = None) -> ClaudeClient:
        c = cfg.section("cloud")
        return cls(
            api_key,
            model=str(c.get("model") or DEFAULT_MODEL),
            effort=c.get("effort") or None,
            max_tokens=int(c.get("max_tokens", 1024)),
            connect_timeout_s=float(c.get("connect_timeout_s", 3.0)),
            read_timeout_s=float(c.get("read_timeout_s", 8.0)),
            total_timeout_s=float(c.get("total_timeout_s", 20.0)),
            max_retries=int(c.get("max_retries", 1)),
            usage_log=UsageLog.from_config(cfg),
            client=client,
        )

    # ------------------------------------------------------------------ SDK client
    def _sdk(self) -> Any:
        """The ``anthropic.Anthropic`` client, created on first use."""
        with self._lock:
            if self._client is None:
                if not self.api_key:
                    raise CloudPermanentError("no Anthropic API key")
                try:
                    import anthropic
                except ImportError as exc:
                    raise CloudPermanentError(
                        'the anthropic SDK is not installed: pip install -e ".[cloud]"'
                    ) from exc
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    timeout=anthropic.Timeout(
                        self.total_timeout_s, connect=self.connect_timeout_s,
                        read=self.read_timeout_s,
                    ),
                    max_retries=self.max_retries,
                )
            return self._client

    def request_params(self, system: str, messages: Sequence[dict[str, Any]]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": list(messages),
        }
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        return params

    # ------------------------------------------------------------------ request
    def stream_reply(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        on_text: TextCallback,
        cancel: threading.Event | None = None,
    ) -> ClaudeReply:
        """Stream one reply, calling ``on_text`` with each text delta.

        Returns the full text. Raises :class:`CloudPermanentError` or
        :class:`CloudUnavailableError`; every call (failed ones too) is logged.
        """
        start = self._clock()
        first_text_ms: int | None = None
        parts: list[str] = []
        usage = Usage()
        stop_reason: str | None = None
        cancelled = False

        def elapsed_ms() -> int:
            return int((self._clock() - start) * 1000)

        try:
            sdk = self._sdk()
            with sdk.messages.stream(**self.request_params(system, messages)) as stream:
                for text in stream.text_stream:
                    if cancel is not None and cancel.is_set():
                        cancelled = True
                        break
                    if not text:
                        continue
                    if first_text_ms is None:
                        first_text_ms = elapsed_ms()
                    parts.append(text)
                    on_text(text)
                    if self._clock() - start > self.total_timeout_s:
                        raise CloudUnavailableError(
                            f"reply took longer than {self.total_timeout_s:.0f} s"
                        )
                if cancelled:
                    usage = _snapshot_usage(stream)
                else:
                    final = stream.get_final_message()
                    usage = Usage.from_api(getattr(final, "usage", None))
                    stop_reason = getattr(final, "stop_reason", None)
        except CloudError as exc:
            self._log(usage, ok=False, error=str(exc), first=first_text_ms, total=elapsed_ms())
            raise
        except Exception as exc:
            err = _classify(exc)
            self._log(usage, ok=False, error=f"{type(exc).__name__}: {exc}",
                      first=first_text_ms, total=elapsed_ms())
            raise err from exc

        total_ms = elapsed_ms()
        self._log(usage, ok=True, error="cancelled" if cancelled else None,
                  first=first_text_ms, total=total_ms)
        return ClaudeReply("".join(parts), usage, stop_reason, cancelled, first_text_ms, total_ms)

    def _log(self, usage: Usage, *, ok: bool, error: str | None, first: int | None,
             total: int) -> None:
        if self.usage_log is not None:
            self.usage_log.record(self.model, usage, ok=ok, error=error, first_text_ms=first,
                                  total_ms=total)


def _snapshot_usage(stream: Any) -> Usage:
    """Tokens known so far for a stream we stopped early (input is billed regardless)."""
    try:
        return Usage.from_api(stream.current_message_snapshot.usage)
    except Exception:
        return Usage()


def _classify(exc: Exception) -> CloudError:
    """Map an SDK / network exception onto permanent vs temporary."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - only reached with an injected client
        return CloudUnavailableError(str(exc))
    permanent = (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.BadRequestError,
    )
    if isinstance(exc, permanent):
        return CloudPermanentError(f"{type(exc).__name__}: {getattr(exc, 'message', exc)}")
    # APITimeoutError, APIConnectionError, RateLimitError, 5xx/529 APIStatusError, and
    # anything unexpected mid-stream: try again next turn.
    return CloudUnavailableError(f"{type(exc).__name__}: {getattr(exc, 'message', exc)}")
