"""Streaming HTTPS/WebSocket proxy capture for wrapped processes."""

import asyncio
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mitmproxy import http, options
from mitmproxy.tools.dump import DumpMaster

from impala_scope.analytics import detect_request_type, error_message, response_failed
from impala_scope.capture import (
    RequestCollector,
    RequestMeta,
    capture_prepared,
    defer,
    derive_session_id,
    flush,
    hash_error,
    prepare_request,
    warn,
)
from impala_scope.streaming import EventSummary, ResponseSummary


@dataclass
class _Inbound:
    collector: RequestCollector
    header_session: str | None
    started_at_ms: int
    host: str
    method: str
    endpoint: str
    forced: bool = False
    meta: RequestMeta | None = None
    request_type: str | None = None
    finished: bool = False

    def finish(self) -> bool:
        if self.finished:
            return self.meta is not None
        self.finished = True
        body = self.collector.body()
        self.request_type = detect_request_type(self.endpoint, self.host, body)
        if self.request_type is None and self.forced:
            self.request_type = "generic.inference"
        if self.request_type is None:
            self.header_session = None
            self.collector.clear()
            return False
        self.meta = prepare_request(
            request=body,
            headers=None,
            request_bytes=self.collector.request_bytes,
            request_fingerprint=self.collector.fingerprint(),
            header_session_id=self.header_session,
            new_fallback_session=True,
        )
        self.header_session = None
        self.collector.clear()
        return True


@dataclass
class _HTTP:
    request: _Inbound
    http_status: int
    summary: ResponseSummary
    submitted: bool = False


@dataclass
class _Turn:
    meta: RequestMeta
    started: float
    summary: EventSummary
    response_bytes: int = 0
    failed: bool = False


@dataclass
class _Socket:
    kind: str
    host: str
    path: str
    base_meta: RequestMeta
    turn: _Turn | None = None


class ScopeAddon:
    def __init__(
        self,
        allowed_hosts: list[str] | None = None,
        *,
        started: threading.Event | None = None,
        submit: Callable[..., bool] = defer,
    ) -> None:
        self.allowed = set(allowed_hosts or [])
        self.started = started
        self.submit = submit
        self.requests: dict[str, _Inbound] = {}
        self.http: dict[str, _HTTP] = {}
        self.sockets: dict[str, _Socket] = {}

    def running(self) -> None:
        if self.started:
            self.started.set()

    def done(self) -> None:
        flush()

    def requestheaders(self, flow: http.HTTPFlow) -> None:
        host = (flow.request.pretty_host or "").lower()
        if self.allowed and not _host_allowed(host, self.allowed):
            return
        state = _Inbound(
            collector=RequestCollector(),
            header_session=derive_session_id(dict(flow.request.headers), None),
            started_at_ms=int(flow.request.timestamp_start * 1000),
            host=host,
            method=flow.request.method,
            endpoint=flow.request.path.split("?", 1)[0],
            forced=bool(self.allowed),
        )
        self.requests[flow.id] = state
        flow.request.headers["accept-encoding"] = "gzip, deflate"

        def stream(chunk: bytes) -> bytes:
            try:
                if chunk:
                    state.collector.feed(chunk)
                else:
                    state.finish()
            except Exception as exc:
                warn(f"request summarization failed: {type(exc).__name__}: {exc}")
            return chunk

        if flow.request.method.upper() in {"POST", "PUT", "PATCH"}:
            flow.request.stream = stream
        else:
            state.finish()

    def responseheaders(self, flow: http.HTTPFlow) -> None:
        if flow.response is None or flow.response.status_code == 101:
            return
        if not _streaming_response(flow.response.headers.get("content-type") or ""):
            return
        try:
            request = self._request(flow)
            if request is None:
                return
            kind = request.request_type or "generic.inference"
            summary = ResponseSummary(
                kind,
                flow.response.headers.get("content-type") or "",
                flow.response.headers.get("content-encoding") or "",
            )
            state = _HTTP(request=request, http_status=flow.response.status_code, summary=summary)
            self.http[flow.id] = state
            flow_id = flow.id

            def stream(chunk: bytes) -> bytes:
                try:
                    if chunk:
                        state.summary.feed(chunk)
                    else:
                        self._finish_http(flow_id, state)
                except Exception as exc:
                    warn(f"stream summarization failed: {type(exc).__name__}: {exc}")
                return chunk

            flow.response.stream = stream
        except Exception as exc:
            warn(f"response setup failed: {type(exc).__name__}: {exc}")

    def response(self, flow: http.HTTPFlow) -> None:
        if flow.response is None or flow.response.status_code == 101:
            return
        state = self.http.pop(flow.id, None)
        if state is not None:
            self._submit_http(state, flow.response.timestamp_end)
            self.requests.pop(flow.id, None)
            return
        try:
            request = self._request(flow)
            if request is None:
                return
            kind = request.request_type or "generic.inference"
            summary = ResponseSummary(
                kind,
                flow.response.headers.get("content-type") or "",
                flow.response.headers.get("content-encoding") or "",
            )
            summary.feed(_wire_content(flow.response))
            self._submit_http(_HTTP(request, flow.response.status_code, summary), flow.response.timestamp_end)
        except Exception as exc:
            warn(f"response capture failed: {type(exc).__name__}: {exc}")
        finally:
            self.requests.pop(flow.id, None)

    def error(self, flow: http.HTTPFlow) -> None:
        state = self.http.pop(flow.id, None)
        if state is not None and state.submitted:
            self.requests.pop(flow.id, None)
            return
        request = state.request if state else self._request(flow)
        self.requests.pop(flow.id, None)
        if request is None or request.meta is None or request.request_type is None:
            return
        transport_message = getattr(flow.error, "msg", None) or str(flow.error or "transport error")
        self.submit(
            capture_prepared,
            meta=request.meta,
            response=state.summary.finish() if state else None,
            started_at_ms=request.started_at_ms,
            completed_at_ms=int(time.time() * 1000),
            host=request.host,
            request_type=request.request_type,
            method=request.method,
            endpoint=request.endpoint,
            http_status=state.http_status if state else None,
            success=False,
            error_type="transport_error",
            error_fingerprint=hash_error(transport_message),
            response_bytes=state.summary.response_bytes if state else 0,
            streamed=state.summary.streamed if state else False,
        )

    def _request(self, flow: http.HTTPFlow) -> _Inbound | None:
        state = self.requests.get(flow.id)
        if state is None:
            host = (flow.request.pretty_host or "").lower()
            if self.allowed and not _host_allowed(host, self.allowed):
                return None
            collector = RequestCollector()
            collector.feed(flow.request.content or b"")
            state = _Inbound(
                collector=collector,
                header_session=derive_session_id(dict(flow.request.headers), None),
                started_at_ms=int(flow.request.timestamp_start * 1000),
                host=host,
                method=flow.request.method,
                endpoint=flow.request.path.split("?", 1)[0],
                forced=bool(self.allowed),
            )
            self.requests[flow.id] = state
        return state if state.finish() else None

    def _finish_http(self, flow_id: str, state: _HTTP, ended: float | None = None) -> None:
        if self.http.get(flow_id) is state:
            self._submit_http(state, ended)

    def _submit_http(self, state: _HTTP, ended: float | None) -> None:
        if state.submitted:
            return
        state.submitted = True
        request = state.request
        if request.meta is None or request.request_type is None:
            return
        response = state.summary.finish()
        http_success = 200 <= state.http_status < 400
        provider_failure = response_failed(response)
        success = http_success and not provider_failure
        provider_message = error_message(response) if not success else None
        self.submit(
            capture_prepared,
            meta=request.meta,
            response=response,
            started_at_ms=request.started_at_ms,
            completed_at_ms=int((ended or time.time()) * 1000),
            host=request.host,
            request_type=request.request_type,
            method=request.method,
            endpoint=request.endpoint,
            http_status=state.http_status,
            success=success,
            error_type="provider_error" if provider_failure else None if http_success else "http_error",
            error_fingerprint=hash_error(provider_message),
            response_bytes=state.summary.response_bytes,
            streamed=state.summary.streamed,
        )

    def websocket_start(self, flow: http.HTTPFlow) -> None:
        host = (flow.request.pretty_host or "").lower()
        if self.allowed and not _host_allowed(host, self.allowed):
            return
        kind = detect_request_type(flow.request.path, host)
        if kind is None and self.allowed:
            kind = "generic.inference"
        if not kind:
            return
        headers = dict(flow.request.headers)
        base = prepare_request(
            request={},
            headers=headers,
            request_bytes=0,
            session_id=f"socket-{flow.id}",
        )
        self.sockets[flow.id] = _Socket(kind, host, flow.request.path, base)

    def websocket_message(self, flow: http.HTTPFlow) -> None:
        state = self.sockets.get(flow.id)
        if state is None or flow.websocket is None or not flow.websocket.messages:
            return
        message = flow.websocket.messages[-1]
        try:
            if not message.is_text:
                return
            try:
                event = json.loads(message.text)
            except Exception:
                return
            if not isinstance(event, dict):
                return
            if message.from_client:
                if event.get("type") == "response.create":
                    if state.turn is not None:
                        state.turn.failed = True
                        self._emit_socket(state, message.timestamp)
                    request = {key: value for key, value in event.items() if key != "type"}
                    meta = prepare_request(
                        request=request,
                        headers=None,
                        request_bytes=len(message.text.encode()),
                        inherited_session_hash=state.base_meta.session_hash,
                    )
                    state.turn = _Turn(meta, message.timestamp, EventSummary(state.kind))
                return
            if state.turn is None:
                return
            state.turn.summary.feed(event)
            raw_content = getattr(message, "content", None)
            state.turn.response_bytes += (
                len(raw_content) if isinstance(raw_content, bytes) else len(message.text.encode())
            )
            if event.get("type") in {"response.failed", "response.cancelled"}:
                state.turn.failed = True
            if event.get("type") in {"response.completed", "response.failed", "response.cancelled", "response.done"}:
                self._emit_socket(state, message.timestamp)
                state.turn = None
        finally:
            flow.websocket.messages.clear()

    def websocket_end(self, flow: http.HTTPFlow) -> None:
        state = self.sockets.pop(flow.id, None)
        if state and state.turn:
            state.turn.failed = True
            self._emit_socket(state, time.time())

    def _emit_socket(self, state: _Socket, ended: float) -> None:
        if state.turn is not None:
            self.submit(_capture_socket, state, state.turn, int(ended * 1000))


def _capture_socket(state: _Socket, turn: _Turn, completed_at_ms: int) -> None:
    response = turn.summary.result()
    failed = turn.failed or response_failed(response)
    capture_prepared(
        meta=turn.meta,
        response=response,
        started_at_ms=int(turn.started * 1000),
        completed_at_ms=completed_at_ms,
        host=state.host,
        request_type=state.kind,
        method="WEBSOCKET",
        endpoint=state.path.split("?", 1)[0],
        http_status=101,
        success=not failed,
        error_type="websocket_error" if failed else None,
        response_bytes=turn.response_bytes,
        streamed=True,
    )


def _host_allowed(host: str, allowed: set[str]) -> bool:
    return any(host == item or host.endswith("." + item) for item in allowed)


def _allow_patterns(hosts: list[str] | None) -> list[str]:
    return [rf"(^|\.){re.escape(host)}(?::\d+)?$" for host in hosts or []]


def _streaming_response(content_type: str) -> bool:
    value = content_type.lower()
    return any(marker in value for marker in ("text/event-stream", "ndjson", "eventstream"))


def _wire_content(response: http.Response) -> bytes:
    raw = getattr(response, "raw_content", None)
    return raw if isinstance(raw, bytes) else response.content or b""


def ca_path(confdir: str | Path) -> Path:
    return Path(confdir).expanduser() / "mitmproxy-ca-cert.pem"


def start(
    *,
    host: str = "127.0.0.1",
    port: int,
    confdir: str | Path,
    allowed_hosts: list[str] | None = None,
    upstream_ca: str | Path | None = None,
):
    """Start the proxy after it owns the requested port; return ``(master, thread)``."""
    directory = Path(confdir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    started = threading.Event()
    stopped = threading.Event()
    result: list[DumpMaster] = []
    errors: list[BaseException] = []

    async def serve() -> None:
        opts = options.Options(listen_host=host, listen_port=port, confdir=str(directory))
        if upstream_ca is not None:
            opts.update(ssl_verify_upstream_trusted_ca=str(upstream_ca))
        patterns = _allow_patterns(allowed_hosts)
        if patterns:
            opts.update(allow_hosts=patterns)
        master = DumpMaster(opts, with_termlog=False, with_dumper=False)
        master.addons.add(ScopeAddon(allowed_hosts, started=started))
        result.append(master)
        await master.run()

    def target() -> None:
        import sys

        if sys.platform == "win32":  # pragma: no cover
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(serve())
        except BaseException as exc:
            errors.append(exc)
        finally:
            stopped.set()
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    thread = threading.Thread(target=target, daemon=True, name="impala-scope-proxy")
    thread.scope_errors = errors  # type: ignore[attr-defined]
    thread.scope_stopped = stopped  # type: ignore[attr-defined]
    thread.start()
    deadline = time.monotonic() + 15
    while not started.wait(0.05):
        if stopped.is_set():
            thread.join(timeout=1)
            cause = errors[0] if errors else RuntimeError("proxy stopped during startup")
            detail = "address unavailable" if isinstance(cause, SystemExit) else str(cause)
            raise RuntimeError(f"proxy failed to start on {host}:{port}: {detail}") from cause
        if time.monotonic() >= deadline:
            if result:
                result[0].shutdown()
            thread.join(timeout=1)
            raise RuntimeError(f"proxy did not start on {host}:{port}")
    if errors:
        raise RuntimeError(f"proxy failed to start on {host}:{port}: {errors[0]}") from errors[0]
    return result[0], thread
