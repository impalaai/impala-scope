"""Shared capture boundary: reduce requests, write analytics, discard payloads."""

import contextlib
import hashlib
import hmac
import json
import os
import queue
import re
import secrets
import stat
import sys
import threading
import time
import uuid
from atexit import register
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from impala_scope.analytics import METRIC_MAX, Store, _system_machine_id, build_record

MAX_REQUEST_METADATA = 1 << 20
_UNSET = object()


class RequestMeta(BaseModel):
    """Fixed-size, payload-free request metadata safe to place on a queue."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    machine_hash: str
    session_hash: str
    request_fingerprint: str
    model: str | None = None
    request_bytes: int = Field(ge=0, le=METRIC_MAX)


@dataclass
class Config:
    database_path: Path = field(default_factory=lambda: Path("./trace.db").resolve())
    server_url: str = "*"
    override_session_id: str | None = None
    hash_key: bytes | None = field(default=None, repr=False)


config = Config()
run_id = f"run-{uuid.uuid4().hex}"
t0_monotonic = time.monotonic()
_session: ContextVar[str] = ContextVar("impala_scope_session", default="")
_warnings: dict[str, float] = {}
_warnings_lock = threading.Lock()
_worker_lock = threading.Lock()
_key_lock = threading.Lock()
_worker: "CaptureWorker | None" = None


class CaptureWorker:
    """Run analytics away from request loops with a large, item-bounded queue."""

    def __init__(self, capacity: int = 65_536) -> None:
        self.pid = os.getpid()
        self.dropped = 0
        self.queue: queue.Queue[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = queue.Queue(capacity)
        self.thread = threading.Thread(target=self._run, daemon=True, name="impala-scope-analytics")
        self.thread.start()

    def submit(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        try:
            self.queue.put_nowait((function, args, kwargs))
            return True
        except queue.Full:
            self.dropped += 1
            warn("analytics queue is full; records are being dropped")
            return False

    def flush(self, timeout: float = 15) -> bool:
        deadline = time.monotonic() + timeout
        while self.queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        complete = self.queue.unfinished_tasks == 0
        if not complete:
            warn("timed out waiting for queued analytics writes")
        return complete

    def _run(self) -> None:
        while True:
            function, args, kwargs = self.queue.get()
            try:
                function(*args, **kwargs)
            except Exception as exc:
                warn(f"analytics worker failed: {type(exc).__name__}: {exc}")
            finally:
                self.queue.task_done()
                del function, args, kwargs


class RequestCollector:
    """Incrementally fingerprint and retain only a bounded request prefix."""

    def __init__(self) -> None:
        self.request_bytes = 0
        self.buffer = bytearray()
        self.overflow = False
        self.digest = _digest("request")
        self.finished = False

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.request_bytes += len(chunk)
        self.digest.update(chunk)
        if len(self.buffer) < MAX_REQUEST_METADATA:
            remaining = MAX_REQUEST_METADATA - len(self.buffer)
            self.buffer.extend(chunk[:remaining])
            self.overflow = len(chunk) > remaining
        elif chunk:
            self.overflow = True

    def body(self) -> dict[str, Any]:
        if not self.buffer:
            return {}
        if not self.overflow:
            try:
                value = json.loads(self.buffer)
            except Exception:
                value = None
            if isinstance(value, dict):
                return value
        return _request_fields(bytes(self.buffer))

    def fingerprint(self) -> str:
        return self.digest.hexdigest()

    def clear(self) -> None:
        self.buffer.clear()


def configure(
    *,
    database_path: str | Path | None = None,
    server_url: str | None = None,
    session_id: str | None | object = _UNSET,
    override_session_id: str | None | object = _UNSET,
) -> None:
    flush()
    if database_path is not None:
        path = Path(database_path).expanduser().resolve()
        store = Store(path)
        store.ensure()
        config.hash_key = _database_key(path)
        config.database_path = path
    if server_url is not None:
        config.server_url = server_url
    if session_id is not _UNSET and override_session_id is not _UNSET:
        raise ValueError("pass session_id or override_session_id, not both")
    if session_id is not _UNSET or override_session_id is not _UNSET:
        value = session_id if session_id is not _UNSET else override_session_id
        config.override_session_id = value if isinstance(value, str) and value else None


def matches_server(url: str) -> bool:
    return config.server_url == "*" or url.startswith(config.server_url)


def current_session_id() -> str:
    value = _session.get()
    if not value:
        value = f"sess-{uuid.uuid4().hex[:12]}"
        _session.set(value)
    return value


def set_session_id(session_id: str) -> None:
    """Set the session identifier for the current async/task context."""
    _session.set(session_id)


@contextlib.contextmanager
def record_session(session_id: str | None = None) -> Iterator[str]:
    value = session_id or f"sess-{uuid.uuid4().hex[:12]}"
    token = _session.set(value)
    try:
        yield value
    finally:
        _session.reset(token)


def derive_session_id(headers: dict[str, str] | None, body: dict[str, Any] | None) -> str | None:
    lowered = {key.lower(): value for key, value in (headers or {}).items()}
    for key in (
        "session_id",
        "x-session-id",
        "x-conversation-id",
        "x-codex-window-id",
        "anthropic-session-id",
        "openai-session-id",
    ):
        if isinstance(lowered.get(key), str) and lowered[key].strip():
            return lowered[key].strip()
    metadata = lowered.get("x-codex-turn-metadata")
    if metadata:
        try:
            value = json.loads(metadata).get("session_id")
            if isinstance(value, str) and value:
                return value
        except Exception:
            pass
    for key in ("previous_response_id", "session_id", "conversation_id"):
        value = (body or {}).get(key)
        if isinstance(value, str) and value:
            return value
    return None


def prepare_request(
    *,
    request: dict[str, Any],
    headers: dict[str, str] | None,
    request_bytes: int,
    raw_request: bytes | None = None,
    request_fingerprint: str | None = None,
    header_session_id: str | None = None,
    session_id: str | None = None,
    inherited_session_hash: str | None = None,
    new_fallback_session: bool = False,
) -> RequestMeta:
    """Reduce a request synchronously so raw data never enters the worker queue."""
    sid = config.override_session_id or header_session_id or derive_session_id(headers, request) or session_id
    sid = sid or (f"request-{uuid.uuid4().hex}" if new_fallback_session else current_session_id())
    model = next(
        (
            value[:512]
            for key in ("model", "model_id", "modelId", "modelVersion")
            if isinstance((value := request.get(key)), str) and value
        ),
        None,
    )
    if request_fingerprint is None:
        raw = raw_request
        if raw is None:
            raw = json.dumps(request, sort_keys=True, separators=(",", ":"), default=str).encode()
        digest = _digest("request")
        digest.update(raw)
        request_fingerprint = digest.hexdigest()
    return RequestMeta(
        machine_hash=_hash("machine", os.environ.get("IMPALA_SCOPE_MACHINE_ID") or _system_machine_id()),
        session_hash=inherited_session_hash or _hash("session", sid),
        request_fingerprint=request_fingerprint,
        model=model,
        request_bytes=request_bytes,
    )


def capture(
    *,
    request: dict[str, Any],
    response: dict[str, Any] | None,
    headers: dict[str, str] | None,
    started_at_ms: int,
    completed_at_ms: int | None,
    host: str | None,
    request_type: str,
    method: str,
    endpoint: str,
    http_status: int | None,
    success: bool,
    error_type: str | None = None,
    error_message: str | None = None,
    request_bytes: int = 0,
    response_bytes: int = 0,
    streamed: bool = False,
    session_id: str | None = None,
    new_fallback_session: bool = False,
) -> bool:
    meta = prepare_request(
        request=request,
        headers=headers,
        request_bytes=request_bytes,
        session_id=session_id,
        new_fallback_session=new_fallback_session,
    )
    return capture_prepared(
        meta=meta,
        response=response,
        started_at_ms=started_at_ms,
        completed_at_ms=completed_at_ms,
        host=host,
        request_type=request_type,
        method=method,
        endpoint=endpoint,
        http_status=http_status,
        success=success,
        error_type=error_type,
        error_message=error_message,
        response_bytes=response_bytes,
        streamed=streamed,
    )


def capture_prepared(
    *,
    meta: RequestMeta,
    response: dict[str, Any] | None,
    started_at_ms: int,
    completed_at_ms: int | None,
    host: str | None,
    request_type: str,
    method: str,
    endpoint: str,
    http_status: int | None,
    success: bool,
    error_type: str | None = None,
    error_message: str | None = None,
    error_fingerprint: str | None = None,
    response_bytes: int = 0,
    streamed: bool = False,
) -> bool:
    try:
        record = build_record(
            run_id=run_id,
            started_at_ms=started_at_ms,
            completed_at_ms=completed_at_ms,
            host=host,
            request_type=request_type,
            method=method,
            endpoint=endpoint,
            request_model=meta.model,
            request_fingerprint=meta.request_fingerprint,
            machine_hash=meta.machine_hash,
            session_hash=meta.session_hash,
            response=response,
            http_status=http_status,
            success=success,
            error_type=error_type,
            error_fingerprint=error_fingerprint or hash_error(error_message),
            request_bytes=meta.request_bytes,
            response_bytes=response_bytes,
            streamed=streamed,
        )
        Store(config.database_path).write(record)
        return True
    except Exception as exc:
        warn(f"analytics write failed for {config.database_path}: {type(exc).__name__}: {exc}")
        return False


def defer(function: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    return _worker_instance().submit(function, *args, **kwargs)


def capture_later(**kwargs: Any) -> bool:
    from impala_scope.streaming import summarize_response

    request = kwargs.pop("request")
    headers = kwargs.pop("headers", None)
    meta = prepare_request(
        request=request,
        headers=headers,
        request_bytes=kwargs.pop("request_bytes", 0),
        session_id=kwargs.pop("session_id", None),
        new_fallback_session=kwargs.pop("new_fallback_session", False),
    )
    kwargs["response"] = summarize_response(kwargs["request_type"], kwargs.get("response"))
    if message := kwargs.pop("error_message", None):
        kwargs["error_fingerprint"] = hash_error(message)
    return defer(capture_prepared, meta=meta, **kwargs)


def flush(timeout: float = 15) -> bool:
    worker = _worker
    return worker.flush(timeout) if worker and worker.pid == os.getpid() else True


def worker_stats() -> dict[str, int]:
    worker = _worker
    return {"queued": worker.queue.qsize(), "dropped": worker.dropped} if worker else {"queued": 0, "dropped": 0}


def hash_error(message: str | None) -> str | None:
    return _hash("error", message) if message else None


def warn(message: str, *, interval: float = 60) -> None:
    now = time.monotonic()
    with _warnings_lock:
        if now - _warnings.get(message, float("-inf")) < interval:
            return
        _warnings[message] = now
    print(f"impala-scope: WARNING: {message}", file=sys.stderr)


def _worker_instance() -> CaptureWorker:
    global _worker
    pid = os.getpid()
    with _worker_lock:
        if _worker is None or _worker.pid != pid or not _worker.thread.is_alive():
            _worker = CaptureWorker()
        return _worker


def _digest(kind: str) -> hmac.HMAC:
    key = config.hash_key
    if key is None:
        with _key_lock:
            if config.hash_key is None:
                config.hash_key = _database_key(config.database_path)
            key = config.hash_key
    assert key is not None
    return hmac.new(key, f"impala-scope-v2\0{kind}\0".encode(), hashlib.sha256)


def _hash(kind: str, value: str) -> str:
    digest = _digest(kind)
    digest.update(value.encode())
    return digest.hexdigest()


def _database_key(path: Path) -> bytes:
    configured = os.environ.get("IMPALA_SCOPE_HASH_KEY")
    if configured:
        return hashlib.sha256(configured.encode()).digest()
    key_path = Path(f"{path}.key")
    if key_path.exists() or key_path.is_symlink():
        return _read_database_key(key_path)
    value = secrets.token_bytes(32)
    temporary = key_path.with_name(f".{key_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as file:
            file.write(value.hex())
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, key_path)
        except FileExistsError:
            return _read_database_key(key_path)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def _read_database_key(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        if path.is_symlink():
            raise OSError("symbolic links are not allowed")
        descriptor = os.open(path, flags)
        try:
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode):
                raise OSError("hash key is not a regular file")
            with os.fdopen(descriptor, "r", encoding="ascii", closefd=False) as file:
                value = bytes.fromhex(file.read().strip())
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid hash key at {path}") from exc
    if len(value) != 32:
        raise RuntimeError(f"invalid hash key at {path}")
    return value


def _request_fields(raw: bytes) -> dict[str, Any]:
    """Extract only routing metadata from a truncated JSON prefix."""
    result: dict[str, Any] = {}
    for key in ("model", "model_id", "modelId", "modelVersion", "session_id", "conversation_id"):
        match = re.search(rb'"' + key.encode() + rb'"\s*:\s*"([^"\\]{1,512})"', raw)
        if match:
            result[key] = match.group(1).decode(errors="replace")
    for key in ("messages", "prompt", "input", "inputs", "contents", "instances"):
        if re.search(rb'"' + key.encode() + rb'"\s*:', raw):
            result[key] = True
    return result


def _after_fork() -> None:
    global _key_lock, _worker, _worker_lock, _warnings_lock, run_id
    _worker = None
    _worker_lock = threading.Lock()
    _key_lock = threading.Lock()
    _warnings_lock = threading.Lock()
    run_id = f"run-{uuid.uuid4().hex}"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
register(lambda: flush(5))
