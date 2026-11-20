"""Streaming client for Ollama's ``/api/chat`` endpoint.

Ollama answers a streamed chat with newline-delimited JSON objects::

    {"message": {"role": "assistant", "content": "[happy] Hel"}, "done": false}
    {"message": {"role": "assistant", "content": "", "tool_calls": [...]}, "done": false}
    {"message": {"role": "assistant", "content": ""}, "done": true, ...}

Errors come either as a non-200 status with ``{"error": "..."}`` or as an
``{"error": "..."}`` line in the middle of the stream. Every failure is raised
as :class:`OllamaError` with a ``kind`` the brain turns into a spoken apology.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx

log = logging.getLogger(__name__)


class OllamaError(Exception):
    """A failed request.

    ``kind`` is one of: unreachable, timeout, model_missing, server, protocol.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        *,
        temperature: float | None = 0.6,
        num_ctx: int | None = 4096,
        keep_alive: str | int | None = "30m",
        timeout_s: float = 120.0,
        connect_timeout_s: float = 5.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        # Read timeout is per chunk: generous, because the first token waits for model load.
        timeout = httpx.Timeout(timeout_s, connect=connect_timeout_s)
        self._http = httpx.Client(base_url=self.host, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def build_payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    def stream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield each streamed chunk (a dict). Raises :class:`OllamaError` on any failure."""
        payload = self.build_payload(messages, tools)
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise _status_error(resp, self.model)
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaError("protocol", f"bad JSON: {line[:80]!r}") from exc
                    if not isinstance(chunk, dict):
                        raise OllamaError("protocol", f"unexpected chunk: {line[:80]!r}")
                    if chunk.get("error"):
                        raise OllamaError("server", str(chunk["error"]))
                    yield chunk
                    if chunk.get("done"):
                        return
        except httpx.TimeoutException as exc:
            raise OllamaError("timeout", f"Ollama timed out ({exc.__class__.__name__})") from exc
        except httpx.TransportError as exc:
            # ConnectError, ReadError, RemoteProtocolError...: Ollama not running or died.
            raise OllamaError("unreachable", f"cannot reach Ollama at {self.host}: {exc}") from exc


def _status_error(resp: httpx.Response, model: str) -> OllamaError:
    try:
        message = str(resp.json().get("error") or resp.text)
    except (ValueError, AttributeError):
        message = resp.text
    message = message.strip() or f"HTTP {resp.status_code}"
    if resp.status_code == 404 and "not found" in message.lower():
        return OllamaError("model_missing", f"model {model!r} is not available: {message}")
    return OllamaError("server", f"Ollama returned HTTP {resp.status_code}: {message}")
