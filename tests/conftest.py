import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from biasscope.scenarios import build_suite, load_curated_profiles
from biasscope.storage import RunStore


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs")


@pytest.fixture
def small_suite():
    return build_suite(load_curated_profiles()[:3], source="curated")


@pytest.fixture
def full_suite():
    return build_suite(load_curated_profiles()[:10], source="curated")


class FakeOpenAIServer:
    """A tiny local OpenAI-compatible server for testing the live code path offline.

    `handler_fn(body: dict, headers: dict, call_no: int) -> (status_code, dict | str)`
    """

    def __init__(self, handler_fn):
        self.handler_fn = handler_fn
        self.calls: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.calls.append({"path": self.path, "body": body, "headers": dict(self.headers)})
                status, payload = outer.handler_fn(body, dict(self.headers), len(outer.calls))
                data = payload if isinstance(payload, str) else json.dumps(payload)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data.encode())

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake_server():
    servers = []

    def make(handler_fn):
        s = FakeOpenAIServer(handler_fn)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def chat_reply(text: str, usage=True) -> dict:
    out = {"id": "x", "object": "chat.completion", "model": "fake-model",
           "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
    if usage:
        out["usage"] = {"prompt_tokens": 50, "completion_tokens": 40, "total_tokens": 90}
    return out
