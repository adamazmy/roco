"""A real local HTTP server that imitates Ollama's streamed ``/api/chat``.

Each request pops the next scripted response:

- a list of chunk dicts -> streamed as NDJSON (a final ``done`` chunk is added
  if the script has none); a float sleeps that many seconds; bytes are sent raw;
- ``Status(code, body)`` -> a plain non-streamed error response.

Every request body is recorded in :attr:`MockOllama.requests`.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL = "qwen2.5:7b"


@dataclass
class Status:
    code: int
    body: dict[str, Any]


def content(text: str) -> dict[str, Any]:
    return {"model": MODEL, "message": {"role": "assistant", "content": text}, "done": False}


def tool_calls(*calls: tuple[str, Any]) -> dict[str, Any]:
    return {
        "model": MODEL,
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls],
        },
        "done": False,
    }


def done() -> dict[str, Any]:
    return {"model": MODEL, "message": {"role": "assistant", "content": ""}, "done": True}


def tokens(text: str, size: int = 3) -> list[dict[str, Any]]:
    """Split ``text`` into small content chunks, like a real token stream."""
    return [content(text[i : i + size]) for i in range(0, len(text), size)]


class MockOllama:
    def __init__(self) -> None:
        self.script: list[Any] = []
        self.requests: list[dict[str, Any]] = []
        self.finished = threading.Event()  # set after the last scripted chunk was written
        mock = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                mock.requests.append(body)
                response = mock.script.pop(0) if mock.script else [content("[neutral] Ok.")]
                if isinstance(response, Status):
                    data = json.dumps(response.body).encode()
                    self.send_response(response.code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                items = list(response)
                if not any(isinstance(i, dict) and i.get("done") for i in items):
                    items.append(done())
                try:
                    for item in items:
                        if isinstance(item, float):
                            time.sleep(item)
                            continue
                        raw = isinstance(item, bytes)
                        line = item if raw else (json.dumps(item) + "\n").encode()
                        self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                finally:
                    if not mock.script:
                        mock.finished.set()

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request: Any, client_address: Any) -> None:
                pass  # clients dropping keep-alive connections is expected noise

        self.server = Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(
            target=self.server.serve_forever, args=(0.05,), daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def closed_port_url() -> str:
    """A URL on which nothing listens (the server is 'down')."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"
