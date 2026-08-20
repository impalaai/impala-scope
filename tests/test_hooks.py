import asyncio
import gzip
import sqlite3

import httpx
import pytest

from impala_scope import configure, record_session
from impala_scope.capture import config, flush
from impala_scope.hooks import install, uninstall


def _success(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "gpt-5",
            "choices": [{"finish_reason": "stop", "message": {"content": "private output"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 4}},
        },
    )


def test_httpx_hook_writes_analytics_without_payloads(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path, server_url="*")
    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(_success), base_url="https://openrouter.ai") as client:
            response = client.post(
                "/api/v1/chat/completions",
                json={"model": "gpt-5", "messages": [{"role": "user", "content": "private input"}]},
            )
            assert response.status_code == 200
    finally:
        uninstall()
    assert flush()

    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT provider, model, input_tokens, cached_input_tokens, output_tokens, success FROM requests"
        ).fetchone()
        raw = path.read_bytes()
    assert row == ("openrouter", "gpt-5", 8, 4, 2, 1)
    assert b"private input" not in raw
    assert b"private output" not in raw


def test_explicit_server_scope_captures_opaque_custom_endpoint(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path, server_url="https://custom.example/v1")
    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(_success)) as client:
            response = client.post("https://custom.example/v1", json={"messages": [{"content": "private"}]})
            assert response.status_code == 200
    finally:
        uninstall()
        configure(server_url="*")
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT request_type, provider FROM requests").fetchone() == (
            "generic.inference",
            "custom.example",
        )


def test_httpx_transport_failure_is_profiled_and_reraised(tmp_path) -> None:
    path = tmp_path / "profile.db"

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("provider timeout", request=request)

    configure(database_path=path, server_url="*")
    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(timeout), base_url="https://api.openai.com") as client:
            with pytest.raises(httpx.ConnectTimeout):
                client.post("/v1/responses", json={"model": "gpt-5", "input": "private"})
    finally:
        uninstall()
    assert flush()

    with sqlite3.connect(path) as db:
        row = db.execute("SELECT success, error_type, length(error_fingerprint) FROM requests").fetchone()
    assert row == (0, "ConnectTimeout", 64)


def test_httpx_hook_fails_open_when_analytics_setup_fails(monkeypatch) -> None:
    monkeypatch.setattr(config, "hash_key", None)

    def failure(_path):
        raise PermissionError("no key")

    monkeypatch.setattr("impala_scope.capture._database_key", failure)
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"ok": True})

    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            response = client.post(
                "https://api.openai.com/v1/responses", json={"model": "x", "input": "private"}
            )
            assert response.status_code == 200
    finally:
        uninstall()
    assert called


class Chunks(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.emitted = 0
        self.chunks = (
            b'data: {"model":"gpt-5","choices":[{"delta":{"content":"hello"}}]}\n\n',
            b'data: {"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\ndata: [DONE]\n\n',
        )

    def __iter__(self):
        for chunk in self.chunks:
            self.emitted += 1
            yield chunk


def test_httpx_raw_stream_is_not_consumed_or_buffered(tmp_path) -> None:
    path = tmp_path / "profile.db"
    stream = Chunks()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    configure(database_path=path, server_url="*")
    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com") as client:
            with client.stream("POST", "/v1/chat/completions", json={"model": "gpt-5", "messages": []}) as response:
                chunks = response.iter_raw()
                assert next(chunks) == stream.chunks[0]
                assert stream.emitted == 1
                assert list(chunks) == [stream.chunks[1]]
    finally:
        uninstall()
    assert flush()

    with sqlite3.connect(path) as db:
        row = db.execute("SELECT input_tokens, output_tokens, response_bytes, streamed FROM requests").fetchone()
    assert row == (5, 2, sum(map(len, stream.chunks)), 1)


class AsyncChunks(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.chunks = (
            b'data: {"model":"gpt-5"}\n\n',
            b'data: {"usage":{"input_tokens":6,"output_tokens":1}}\n\n',
        )

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def test_async_httpx_raw_stream_is_preserved(tmp_path) -> None:
    path = tmp_path / "profile.db"
    stream = AsyncChunks()

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async def run() -> list[bytes]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://api.openai.com"
        ) as client:
            async with client.stream(
                "POST", "/v1/responses", json={"model": "gpt-5", "input": "private"}
            ) as response:
                return [chunk async for chunk in response.aiter_raw()]

    configure(database_path=path, server_url="*")
    uninstall()
    install()
    try:
        assert asyncio.run(run()) == list(stream.chunks)
    finally:
        uninstall()
    assert flush()

    with sqlite3.connect(path) as db:
        row = db.execute("SELECT input_tokens, output_tokens, response_bytes, streamed FROM requests").fetchone()
    assert row == (6, 1, sum(map(len, stream.chunks)), 1)


def test_record_session_is_snapshotted_before_deferred_capture(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path, server_url="*")
    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(_success), base_url="https://api.openai.com") as client:
            with record_session("agent-a"):
                client.post("/v1/chat/completions", json={"model": "x", "messages": []})
            with record_session("agent-b"):
                client.post("/v1/chat/completions", json={"model": "x", "messages": []})
    finally:
        uninstall()
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*), count(DISTINCT session_hash) FROM requests").fetchone() == (2, 2)


class RequestChunks(httpx.SyncByteStream):
    def __iter__(self):
        yield b'{"model":"gpt-5",'
        yield b'"messages":[]}'


def test_streaming_httpx_request_is_forwarded_and_profiled(tmp_path) -> None:
    path = tmp_path / "profile.db"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.read() == b'{"model":"gpt-5","messages":[]}'
        return httpx.Response(200, json={"usage": {"prompt_tokens": 3, "completion_tokens": 1}})

    configure(database_path=path, server_url="*")
    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            request = httpx.Request(
                "POST", "https://api.openai.com/v1/chat/completions", stream=RequestChunks()
            )
            assert client.send(request).status_code == 200
    finally:
        uninstall()
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT input_tokens, output_tokens, request_bytes FROM requests").fetchone() == (3, 1, 31)


def test_compressed_stream_is_profiled_without_changing_raw_bytes(tmp_path) -> None:
    path = tmp_path / "profile.db"
    plain = b'data: {"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
    compressed = gzip.compress(plain)

    class Compressed(httpx.SyncByteStream):
        def __iter__(self):
            yield compressed

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
            stream=Compressed(),
        )

    configure(database_path=path, server_url="*")
    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with client.stream(
                "POST", "https://api.openai.com/v1/chat/completions", json={"model": "x", "messages": []}
            ) as response:
                assert b"".join(response.iter_raw()) == compressed
    finally:
        uninstall()
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT input_tokens, output_tokens FROM requests").fetchone() == (5, 2)


def test_buffered_compressed_response_counts_wire_bytes(tmp_path) -> None:
    path = tmp_path / "profile.db"
    plain = b'{"usage":{"prompt_tokens":5,"completion_tokens":2}}'
    compressed = gzip.compress(plain)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip", "content-length": str(len(compressed))},
            content=compressed,
        )

    configure(database_path=path, server_url="*")
    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions", json={"model": "x", "messages": []}
            )
            assert response.json()["usage"]["prompt_tokens"] == 5
    finally:
        uninstall()
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT input_tokens, output_tokens, response_bytes FROM requests").fetchone() == (
            5,
            2,
            len(compressed),
        )


def test_hook_requests_safe_response_encodings(tmp_path) -> None:
    observed = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed
        observed = request.headers["accept-encoding"]
        return httpx.Response(200, json={"usage": {"input_tokens": 1}})

    configure(database_path=tmp_path / "profile.db", server_url="*")
    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("https://api.openai.com/v1/responses", json={"model": "x", "input": "private"})
    finally:
        uninstall()
    assert observed == "gzip, deflate"


def test_provider_failure_and_abandoned_stream_are_failures(tmp_path) -> None:
    path = tmp_path / "profile.db"

    class Failed(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"type":"response.failed","response":{"status":"failed"}}\n\n'

    class Long(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"model":"gpt-5"}\n\n'
            yield b'data: {"usage":{"input_tokens":2,"output_tokens":1}}\n\n'

    streams = iter((Failed(), Long()))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=next(streams))

    configure(database_path=path, server_url="*")
    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with client.stream(
                "POST", "https://api.openai.com/v1/responses", json={"model": "x", "input": "private"}
            ) as response:
                list(response.iter_raw())
            with client.stream(
                "POST", "https://api.openai.com/v1/responses", json={"model": "x", "input": "private"}
            ) as response:
                next(response.iter_raw())
    finally:
        uninstall()
    assert flush()
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT success, error_type FROM requests ORDER BY started_at_ms").fetchall()
    assert rows == [(0, "provider_error"), (0, "_StreamClosed")]


def test_http_200_error_object_is_a_provider_failure(tmp_path) -> None:
    path = tmp_path / "profile.db"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"type": "overloaded", "message": "private"}})

    configure(database_path=path, server_url="*")
    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("https://api.openai.com/v1/responses", json={"model": "x", "input": "private"})
    finally:
        uninstall()
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT success, error_type, length(error_fingerprint) FROM requests").fetchone() == (
            0,
            "provider_error",
            64,
        )


def test_large_http_200_error_object_is_still_a_provider_failure(tmp_path) -> None:
    path = tmp_path / "profile.db"
    body = b'{"generated":"' + b"x" * (2 << 20) + b'","error":{"type":"overloaded"}}'

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)

    configure(database_path=path, server_url="*")
    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("https://api.openai.com/v1/responses", json={"model": "x", "input": "private"})
    finally:
        uninstall()
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT success, error_type FROM requests").fetchone() == (0, "provider_error")


def test_cancelled_async_request_is_profiled_and_reraised(tmp_path) -> None:
    path = tmp_path / "profile.db"

    async def run() -> None:
        started = asyncio.Event()
        never = asyncio.Event()

        async def handler(_request: httpx.Request) -> httpx.Response:
            started.set()
            await never.wait()
            raise AssertionError("unreachable")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            task = asyncio.create_task(
                client.post("https://api.openai.com/v1/responses", json={"model": "x", "input": "private"})
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    configure(database_path=path, server_url="*")
    uninstall()
    install()
    try:
        asyncio.run(run())
    finally:
        uninstall()
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT success, error_type FROM requests").fetchone() == (0, "CancelledError")


def test_uninstall_does_not_clobber_a_later_httpx_hook() -> None:
    uninstall()
    install()
    scope_sync = httpx.Client.send
    scope_async = httpx.AsyncClient.send

    def third_party(*args, **kwargs):
        return scope_sync(*args, **kwargs)

    httpx.Client.send = third_party  # type: ignore[method-assign]
    try:
        uninstall()
        assert httpx.Client.send is third_party
    finally:
        httpx.Client.send = scope_sync  # type: ignore[method-assign]
        httpx.AsyncClient.send = scope_async  # type: ignore[method-assign]
        uninstall()
        install()


def test_stream_close_error_is_recorded_as_failure_and_reraised(tmp_path) -> None:
    path = tmp_path / "profile.db"

    class BrokenClose(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"model":"gpt-5"}\n\n'

        def close(self):
            raise RuntimeError("close failed")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=BrokenClose())

    configure(database_path=path, server_url="*")
    uninstall()
    install()
    try:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(RuntimeError, match="close failed"):
                with client.stream(
                    "POST", "https://api.openai.com/v1/chat/completions", json={"model": "x", "messages": []}
                ):
                    pass
    finally:
        uninstall()
    assert flush()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT success, error_type FROM requests").fetchone() == (0, "RuntimeError")
