import gzip
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field

import pytest

from impala_scope.capture import configure, flush
from impala_scope.proxy import ScopeAddon, _allow_patterns, start


@dataclass
class Request:
    path: str = "/v1/chat/completions"
    pretty_host: str = "api.openai.com"
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    content: bytes = b'{"model":"gpt-5","messages":[{"content":"private"}]}'
    timestamp_start: float = field(default_factory=time.time)
    stream: object = False


@dataclass
class Response:
    status_code: int
    content: bytes
    headers: dict[str, str] = field(default_factory=lambda: {"content-type": "application/json"})
    timestamp_end: float = field(default_factory=time.time)
    stream: object = False


@dataclass
class Flow:
    id: str = "flow-1"
    request: Request = field(default_factory=Request)
    response: Response | None = None
    error: object | None = None
    websocket: object | None = None


def test_proxy_profiles_http_error_without_storing_body(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path)
    flow = Flow(response=Response(429, b'{"error":{"message":"rate limited"}}'))

    ScopeAddon().response(flow)  # type: ignore[arg-type]
    assert flush()

    with sqlite3.connect(path) as db:
        row = db.execute("SELECT provider, http_status, success, length(error_fingerprint) FROM requests").fetchone()
    assert row == ("openai", 429, 0, 64)
    assert b"private" not in path.read_bytes()


def test_proxy_profiles_large_http_200_error_as_failure(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path)
    body = b'{"generated":"' + b"x" * (2 << 20) + b'","error":{"type":"overloaded"}}'
    flow = Flow(response=Response(200, body))
    ScopeAddon().response(flow)  # type: ignore[arg-type]
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT success, error_type FROM requests").fetchone() == (0, "provider_error")


@dataclass
class Message:
    text: str
    from_client: bool
    timestamp: float = field(default_factory=time.time)
    is_text: bool = True


@dataclass
class WebSocket:
    messages: list[Message] = field(default_factory=list)


def test_proxy_profiles_websocket_turn(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path)
    flow = Flow()
    flow.request.path = "/backend-api/codex/responses"
    flow.request.pretty_host = "chatgpt.com"
    flow.response = Response(101, b"")
    flow.websocket = WebSocket()
    addon = ScopeAddon()
    addon.websocket_start(flow)  # type: ignore[arg-type]

    for payload, client in (
        ({"type": "response.create", "model": "gpt-5", "input": "private"}, True),
        ({"type": "response.output_text.delta", "delta": "private answer"}, False),
        (
            {
                "type": "response.completed",
                "response": {"model": "gpt-5", "status": "completed", "usage": {"input_tokens": 3, "output_tokens": 1}},
            },
            False,
        ),
    ):
        message = Message(json.dumps(payload), client)
        flow.websocket.messages.append(message)
        addon.websocket_message(flow)  # type: ignore[arg-type]
    assert flush()

    with sqlite3.connect(path) as db:
        row = db.execute("SELECT model, input_tokens, output_tokens, streamed FROM requests").fetchone()
    assert row == ("gpt-5", 3, 1, 1)
    raw = path.read_bytes()
    assert b"private answer" not in raw


def test_proxy_stream_callback_forwards_each_chunk_and_defers_capture(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path)
    submitted: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    def submit(function, *args, **kwargs):
        submitted.append((function, args, kwargs))
        return True

    flow = Flow(
        response=Response(
            200,
            b"",
            headers={"content-type": "text/event-stream"},
        )
    )
    addon = ScopeAddon(submit=submit)
    addon.responseheaders(flow)  # type: ignore[arg-type]
    stream = flow.response.stream  # type: ignore[union-attr]
    assert callable(stream)

    first = b'data: {"model":"gpt-5","choices":[{"delta":{"content":"hello"}}]}\n\n'
    final = b'data: {"usage":{"prompt_tokens":9,"completion_tokens":4}}\n\n'
    assert stream(first) is first
    assert submitted == []
    assert stream(final) is final
    assert stream(b"") == b""
    addon.response(flow)  # mitmproxy emits response after the stream terminator
    assert len(submitted) == 1

    function, args, kwargs = submitted.pop()
    function(*args, **kwargs)  # type: ignore[operator]
    with sqlite3.connect(path) as db:
        row = db.execute("SELECT input_tokens, output_tokens, response_bytes, streamed FROM requests").fetchone()
    assert row == (9, 4, len(first) + len(final), 1)


def test_upstream_allowlist_matches_only_selected_tls_hosts() -> None:
    [pattern] = _allow_patterns(["api.openai.com"])
    assert re.search(pattern, "api.openai.com")
    assert re.search(pattern, "edge.api.openai.com:443")
    assert not re.search(pattern, "pinned.example")
    assert not re.search(pattern, "api.openai.com.attacker.example")


class _Addons:
    def __init__(self) -> None:
        self.addon = None

    def add(self, addon) -> None:
        self.addon = addon


class _Master:
    def __init__(self, opts, **_kwargs) -> None:
        self.options = opts
        self.addons = _Addons()

    async def run(self) -> None:
        self.addons.addon.running()

    def shutdown(self) -> None:
        pass


def test_start_applies_allow_hosts_to_avoid_intercepting_excluded_tls(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("impala_scope.proxy.DumpMaster", _Master)
    master, thread = start(port=47123, confdir=tmp_path, allowed_hosts=["api.openai.com"])
    thread.join(timeout=1)

    assert master.options.allow_hosts == _allow_patterns(["api.openai.com"])


def test_start_uses_caller_trust_for_intercepted_upstreams(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("impala_scope.proxy.DumpMaster", _Master)
    trust = tmp_path / "enterprise.pem"
    trust.write_text("CA")
    master, thread = start(port=47125, confdir=tmp_path, upstream_ca=trust)
    thread.join(timeout=1)
    assert master.options.ssl_verify_upstream_trusted_ca == str(trust)


class _FailedMaster(_Master):
    async def run(self) -> None:
        raise OSError(48, "Address already in use")


def test_start_reports_bind_failure_instead_of_accepting_a_foreign_listener(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("impala_scope.proxy.DumpMaster", _FailedMaster)
    with pytest.raises(RuntimeError, match="Address already in use"):
        start(port=47124, confdir=tmp_path)


def test_proxy_streams_request_body_and_profiles_it(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path)
    flow = Flow()
    flow.request.content = b""
    addon = ScopeAddon()
    addon.requestheaders(flow)  # type: ignore[arg-type]
    stream = flow.request.stream
    assert callable(stream)
    first = b'{"model":"gpt-5",'
    second = b'"messages":[]}'
    assert stream(first) is first
    assert stream(second) is second
    assert stream(b"") == b""

    payload = b'{"usage":{"prompt_tokens":3,"completion_tokens":1}}'
    flow.response = Response(200, payload)
    addon.responseheaders(flow)  # type: ignore[arg-type]
    assert flow.response.stream is False
    addon.response(flow)  # type: ignore[arg-type]
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT request_bytes, input_tokens, output_tokens FROM requests").fetchone() == (
            len(first) + len(second),
            3,
            1,
        )


def test_allowlisted_host_forces_opaque_endpoint_capture(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path)
    flow = Flow(response=Response(200, b'{"usage":{"input_tokens":2}}'))
    flow.request.path = "/v1"
    flow.request.pretty_host = "custom.example"
    flow.request.content = b'{"messages":[]}'
    addon = ScopeAddon(allowed_hosts=["custom.example"])
    addon.response(flow)  # type: ignore[arg-type]
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT request_type, input_tokens FROM requests").fetchone() == (
            "generic.inference",
            2,
        )


def test_json_response_does_not_install_mitm_stream_callback() -> None:
    flow = Flow(response=Response(200, b'{"usage":{"input_tokens":1}}'))
    addon = ScopeAddon(submit=lambda *_args, **_kwargs: True)
    addon.responseheaders(flow)  # type: ignore[arg-type]
    assert flow.response.stream is False  # type: ignore[union-attr]


def test_proxy_requests_safe_response_encodings() -> None:
    flow = Flow()
    ScopeAddon().requestheaders(flow)  # type: ignore[arg-type]
    assert flow.request.headers["accept-encoding"] == "gzip, deflate"


def test_proxy_profiles_gzip_sse_without_modifying_wire_bytes(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path)
    flow = Flow(
        response=Response(
            200,
            b"",
            headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
        )
    )
    addon = ScopeAddon()
    addon.responseheaders(flow)  # type: ignore[arg-type]
    stream = flow.response.stream  # type: ignore[union-attr]
    compressed = gzip.compress(b'data: {"usage":{"prompt_tokens":7,"completion_tokens":2}}\n\n')
    assert callable(stream)
    assert stream(compressed) is compressed
    stream(b"")
    addon.response(flow)  # type: ignore[arg-type]
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT input_tokens, output_tokens FROM requests").fetchone() == (7, 2)


def test_proxy_profiles_buffered_gzip_json_as_wire_bytes(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path)
    compressed = gzip.compress(b'{"usage":{"prompt_tokens":7,"completion_tokens":2}}')
    flow = Flow(
        response=Response(
            200,
            compressed,
            headers={"content-type": "application/json", "content-encoding": "gzip"},
        )
    )
    addon = ScopeAddon()
    addon.responseheaders(flow)  # type: ignore[arg-type]
    assert flow.response.stream is False  # type: ignore[union-attr]
    addon.response(flow)  # type: ignore[arg-type]
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT input_tokens, output_tokens, response_bytes FROM requests").fetchone() == (
            7,
            2,
            len(compressed),
        )


def test_realtime_websocket_done_is_captured_and_messages_are_released(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path)
    flow = Flow()
    flow.request.path = "/v1/realtime"
    flow.response = Response(101, b"")
    flow.websocket = WebSocket()
    addon = ScopeAddon()
    addon.websocket_start(flow)  # type: ignore[arg-type]

    for payload, client in (
        ({"type": "response.create", "response": {"modalities": ["text"]}}, True),
        (
            {
                "type": "response.done",
                "response": {"model": "gpt-5", "status": "completed", "usage": {"input_tokens": 4, "output_tokens": 2}},
            },
            False,
        ),
    ):
        flow.websocket.messages.append(Message(json.dumps(payload), client))
        addon.websocket_message(flow)  # type: ignore[arg-type]
        assert flow.websocket.messages == []
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT success, input_tokens, output_tokens FROM requests").fetchone() == (1, 4, 2)


def test_new_websocket_turn_records_the_interrupted_turn(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path)
    flow = Flow()
    flow.request.path = "/v1/realtime"
    flow.response = Response(101, b"")
    flow.websocket = WebSocket()
    addon = ScopeAddon()
    addon.websocket_start(flow)  # type: ignore[arg-type]

    for payload, client in (
        ({"type": "response.create", "response": {"model": "first"}}, True),
        ({"type": "response.create", "response": {"model": "second"}}, True),
        (
            {"type": "response.done", "response": {"status": "completed", "usage": {"input_tokens": 1}}},
            False,
        ),
    ):
        flow.websocket.messages.append(Message(json.dumps(payload), client))
        addon.websocket_message(flow)  # type: ignore[arg-type]
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT success FROM requests ORDER BY rowid").fetchall() == [(0,), (1,)]
