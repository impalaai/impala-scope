"""Streaming-safe in-process httpx profiler."""

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from impala_scope.analytics import detect_request_type, error_message, response_failed
from impala_scope.capture import (
    RequestCollector,
    RequestMeta,
    capture_prepared,
    config,
    current_session_id,
    defer,
    derive_session_id,
    hash_error,
    matches_server,
    prepare_request,
    warn,
)
from impala_scope.streaming import ResponseSummary

_base_sync_send = httpx.Client.send
_base_async_send = httpx.AsyncClient.send
_sync_wrapper: Callable[..., Any] | None = None
_async_wrapper: Callable[..., Any] | None = None
_installed_sync_base: Callable[..., Any] = _base_sync_send
_installed_async_base: Callable[..., Any] = _base_async_send
_installed = False


@dataclass
class _Request:
    collector: RequestCollector
    header_session: str | None
    caller_session: str
    started_at_ms: int
    host: str | None
    method: str
    endpoint: str
    forced: bool = False
    meta: RequestMeta | None = None
    request_type: str | None = None

    def finish(self) -> bool:
        if self.meta is not None:
            return True
        body = self.collector.body()
        kind = detect_request_type(self.endpoint, self.host or "", body)
        if kind is None and self.forced:
            kind = "generic.inference"
        if kind is None:
            self.header_session = None
            self.caller_session = ""
            self.collector.clear()
            return False
        self.request_type = kind
        self.meta = prepare_request(
            request=body,
            headers=None,
            request_bytes=self.collector.request_bytes,
            request_fingerprint=self.collector.fingerprint(),
            header_session_id=self.header_session,
            session_id=self.caller_session,
        )
        self.header_session = None
        self.caller_session = ""
        self.collector.clear()
        return True


class _StreamClosed(Exception):
    pass


class _SyncRequestTee(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, collector: RequestCollector) -> None:
        self.stream = stream
        self.collector = collector

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.stream:
            self.collector.feed(chunk)
            yield chunk

    def close(self) -> None:
        self.stream.close()


class _AsyncRequestTee(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, collector: RequestCollector) -> None:
        self.stream = stream
        self.collector = collector

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self.stream:
            self.collector.feed(chunk)
            yield chunk

    async def aclose(self) -> None:
        await self.stream.aclose()


class _SyncTee(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, collector: ResponseSummary, finish) -> None:
        self.stream = stream
        self.collector = collector
        self.finish_callback = finish
        self.finished = False

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self.stream:
                self.collector.feed(chunk)
                yield chunk
        except Exception as exc:
            self._finish(exc)
            raise
        else:
            self._finish()

    def close(self) -> None:
        error: Exception | None = None
        try:
            self.stream.close()
        except Exception as exc:
            error = exc
            raise
        finally:
            interrupted = _StreamClosed("response stream closed before exhaustion") if not self.finished else None
            self._finish(error or interrupted)

    def _finish(self, exc: Exception | None = None) -> None:
        if not self.finished:
            self.finished = True
            callback, self.finish_callback = self.finish_callback, None
            callback(exc)


class _AsyncTee(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, collector: ResponseSummary, finish) -> None:
        self.stream = stream
        self.collector = collector
        self.finish_callback = finish
        self.finished = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self.stream:
                self.collector.feed(chunk)
                yield chunk
        except asyncio.CancelledError as exc:
            self._finish(exc)
            raise
        except Exception as exc:
            self._finish(exc)
            raise
        else:
            self._finish()

    async def aclose(self) -> None:
        error: Exception | None = None
        try:
            await self.stream.aclose()
        except Exception as exc:
            error = exc
            raise
        finally:
            interrupted = _StreamClosed("response stream closed before exhaustion") if not self.finished else None
            self._finish(error or interrupted)

    def _finish(self, exc: BaseException | None = None) -> None:
        if not self.finished:
            self.finished = True
            callback, self.finish_callback = self.finish_callback, None
            callback(exc)


def _context(request: httpx.Request, started: int) -> _Request:
    collector = RequestCollector()
    context = _Request(
        collector=collector,
        header_session=derive_session_id(dict(request.headers), None),
        caller_session=current_session_id(),
        started_at_ms=started,
        host=request.url.host,
        method=request.method,
        endpoint=request.url.path,
        forced=config.server_url != "*",
    )
    request.headers["accept-encoding"] = "gzip, deflate"
    if hasattr(request, "_content"):
        collector.feed(request.content)
        context.finish()
    elif isinstance(request.stream, httpx.SyncByteStream):
        request.stream = _SyncRequestTee(request.stream, collector)
    elif isinstance(request.stream, httpx.AsyncByteStream):
        request.stream = _AsyncRequestTee(request.stream, collector)
    return context


def _save(
    context: _Request,
    status: int | None,
    response: dict[str, Any] | None,
    response_bytes: int,
    streamed: bool,
    exception_type: str | None,
    exception_fingerprint: str | None,
) -> None:
    if not context.finish() or context.meta is None or context.request_type is None:
        return
    http_success = status is not None and 200 <= status < 400
    provider_failure = response_failed(response)
    success = http_success and exception_type is None and not provider_failure
    capture_prepared(
        meta=context.meta,
        response=response,
        started_at_ms=context.started_at_ms,
        completed_at_ms=int(time.time() * 1000),
        host=context.host,
        request_type=context.request_type,
        method=context.method,
        endpoint=context.endpoint,
        http_status=status,
        success=success,
        error_type=_error_type(exception_type, provider_failure, http_success),
        error_fingerprint=exception_fingerprint or hash_error(error_message(response) if not success else None),
        response_bytes=response_bytes,
        streamed=streamed,
    )


def _instrument(response: httpx.Response, context: _Request) -> httpx.Response:
    if not context.finish():
        return response
    content_type = (response.headers.get("content-type") or "").lower()
    if hasattr(response, "_content"):
        collector = ResponseSummary(context.request_type or "generic.inference", content_type)
        collector.feed(response.content)
        response_bytes = getattr(response, "num_bytes_downloaded", collector.response_bytes)
        if not response_bytes:
            try:
                response_bytes = int(response.headers.get("content-length") or 0)
            except ValueError:
                response_bytes = 0
        response_bytes = response_bytes or collector.response_bytes
        defer(
            _save,
            context,
            response.status_code,
            collector.finish(),
            response_bytes,
            collector.streamed,
            None,
            None,
        )
        return response

    collector = ResponseSummary(
        context.request_type or "generic.inference",
        content_type,
        response.headers.get("content-encoding") or "",
    )
    status = response.status_code

    def finish(exc: Exception | None = None) -> None:
        defer(
            _save,
            context,
            status,
            collector.finish(),
            collector.response_bytes,
            collector.streamed,
            type(exc).__name__ if exc else None,
            hash_error(str(exc)) if exc else None,
        )

    if isinstance(response.stream, httpx.SyncByteStream):
        response.stream = _SyncTee(response.stream, collector, finish)
    elif isinstance(response.stream, httpx.AsyncByteStream):
        response.stream = _AsyncTee(response.stream, collector, finish)
    return response


def _error_type(exception_type: str | None, provider_failure: bool, http_success: bool) -> str | None:
    if exception_type:
        return exception_type
    if provider_failure:
        return "provider_error"
    return None if http_success else "http_error"


def install() -> None:
    global _async_wrapper, _installed, _installed_async_base, _installed_sync_base, _sync_wrapper
    if _installed:
        return

    sync_base = httpx.Client.send
    async_base = httpx.AsyncClient.send
    _installed_sync_base = sync_base
    _installed_async_base = async_base

    def sync(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        if not matches_server(str(request.url)):
            return sync_base(self, request, **kwargs)
        try:
            context = _context(request, int(time.time() * 1000))
        except Exception as exc:
            warn(f"httpx instrumentation unavailable: {type(exc).__name__}: {exc}")
            return sync_base(self, request, **kwargs)
        try:
            response = sync_base(self, request, **kwargs)
        except Exception as exc:
            if context.finish():
                defer(_save, context, None, None, 0, False, type(exc).__name__, hash_error(str(exc)))
            raise
        return _instrument(response, context)

    async def async_send(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        if not matches_server(str(request.url)):
            return await async_base(self, request, **kwargs)
        try:
            context = _context(request, int(time.time() * 1000))
        except Exception as exc:
            warn(f"httpx instrumentation unavailable: {type(exc).__name__}: {exc}")
            return await async_base(self, request, **kwargs)
        try:
            response = await async_base(self, request, **kwargs)
        except asyncio.CancelledError as exc:
            if context.finish():
                defer(_save, context, None, None, 0, False, type(exc).__name__, hash_error(str(exc) or "cancelled"))
            raise
        except Exception as exc:
            if context.finish():
                defer(_save, context, None, None, 0, False, type(exc).__name__, hash_error(str(exc)))
            raise
        return _instrument(response, context)

    _sync_wrapper = sync
    _async_wrapper = async_send
    httpx.Client.send = sync  # type: ignore[method-assign]
    httpx.AsyncClient.send = async_send  # type: ignore[method-assign]
    _installed = True


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    if httpx.Client.send is not _sync_wrapper or httpx.AsyncClient.send is not _async_wrapper:
        warn("httpx hooks changed after impala-scope; leaving the instrumentation chain intact")
        return
    httpx.Client.send = _installed_sync_base  # type: ignore[method-assign]
    httpx.AsyncClient.send = _installed_async_base  # type: ignore[method-assign]
    _installed = False
