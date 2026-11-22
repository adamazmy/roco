"""A scripted stand-in for ``anthropic.Anthropic`` (``.messages.stream(**params)``).

Each request pops the next :class:`Round`: text chunks streamed through ``text_stream``,
then a final message whose ``content`` has the text and any ``tool_use`` blocks. A
round with ``error`` raises it when the stream is opened. Every request's params are
recorded in :attr:`FakeClaude.calls`; :func:`sse_body` builds a real SSE response for
tests that run the real SDK over ``httpx2.MockTransport``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Round:
    chunks: list[str] = field(default_factory=list)
    tools: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 50, "output_tokens": 20, "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 0})
    error: BaseException | None = None
    delay: float = 0.0  # before each chunk
    thinking: bool = False  # add a thinking block before the text


def say(*chunks: str, **kw: Any) -> Round:
    return Round(list(chunks), **kw)


def call(*tools: tuple[str, dict[str, Any]], text: str = "", **kw: Any) -> Round:
    return Round([text] if text else [], list(tools), stop_reason="tool_use", **kw)


class _Stream:
    def __init__(self, rnd: Round, index: int) -> None:
        self.rnd = rnd
        self.index = index

    def __enter__(self) -> _Stream:
        if self.rnd.error is not None:
            raise self.rnd.error
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    @property
    def text_stream(self):
        for chunk in self.rnd.chunks:
            if self.rnd.delay:
                time.sleep(self.rnd.delay)
            yield chunk

    def get_final_message(self) -> Any:
        content: list[dict[str, Any]] = []
        if self.rnd.thinking:
            content.append({"type": "thinking", "thinking": "", "signature": f"sig{self.index}"})
        text = "".join(self.rnd.chunks)
        if text:
            content.append({"type": "text", "text": text})
        for k, (name, args) in enumerate(self.rnd.tools):
            content.append({"type": "tool_use", "id": f"toolu_live{self.index}_{k}",
                            "name": name, "input": args})
        stop = self.rnd.stop_reason or ("tool_use" if self.rnd.tools else "end_turn")
        return type("Msg", (), {"content": content, "usage": self.rnd.usage,
                                "stop_reason": stop})()

    @property
    def current_message_snapshot(self) -> Any:
        return type("Snap", (), {"usage": {"input_tokens": self.rnd.usage["input_tokens"]}})()


class FakeClaude:
    def __init__(self, *rounds: Round) -> None:
        self.rounds = list(rounds)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def stream(self, **params: Any) -> _Stream:
        self.calls.append(json.loads(json.dumps(params)))  # a snapshot, like the wire
        if not self.rounds:
            raise AssertionError("FakeClaude: no scripted round left")
        return _Stream(self.rounds.pop(0), len(self.calls))


def sse_body(blocks: list[dict[str, Any]], stop: str = "end_turn",
             usage: dict[str, int] | None = None) -> bytes:
    """A streamed Messages API response with text / tool_use ``blocks``."""
    usage = usage or {"input_tokens": 12, "output_tokens": 1, "cache_read_input_tokens": 1900,
                      "cache_creation_input_tokens": 0}
    events: list[dict[str, Any]] = [{"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [], "stop_reason": None, "stop_sequence": None, "usage": usage}}]
    for i, block in enumerate(blocks):
        if block["type"] == "text":
            events.append({"type": "content_block_start", "index": i,
                           "content_block": {"type": "text", "text": ""}})
            events.append({"type": "content_block_delta", "index": i,
                           "delta": {"type": "text_delta", "text": block["text"]}})
        else:
            events.append({"type": "content_block_start", "index": i, "content_block": {
                "type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}})
            events.append({"type": "content_block_delta", "index": i, "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(block["input"])}})
        events.append({"type": "content_block_stop", "index": i})
    events += [
        {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
         "usage": {"output_tokens": 30}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
