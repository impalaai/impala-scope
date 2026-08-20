"""Flat, body-free inference analytics."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import sqlite3
import sys
import threading
import time
import uuid
from functools import cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

SQLITE_INT_MAX = (1 << 63) - 1
METRIC_MAX = 10**15
MetricInt = Annotated[int, Field(ge=0, le=METRIC_MAX)]
MetricFloat = Annotated[float, Field(ge=0, le=METRIC_MAX)]


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    input_tokens: MetricInt | None = None
    uncached_input_tokens: MetricInt | None = None
    cached_input_tokens: MetricInt | None = None
    cache_write_input_tokens: MetricInt | None = None
    output_tokens: MetricInt | None = None
    reasoning_tokens: MetricInt | None = None
    total_tokens: MetricInt | None = None
    cost_usd: MetricFloat | None = None
    invalid_token_count: MetricInt = 0


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    request_id: str
    run_id: str
    machine_hash: str
    session_hash: str
    started_at_ms: int
    completed_at_ms: int | None
    duration_ms: int | None
    provider: str
    provider_host: str | None
    request_type: str
    method: str
    endpoint: str
    model: str | None
    http_status: int | None
    success: bool
    error_type: str | None
    error_fingerprint: str | None
    input_tokens: MetricInt | None
    uncached_input_tokens: MetricInt | None
    cached_input_tokens: MetricInt | None
    cache_write_input_tokens: MetricInt | None
    output_tokens: MetricInt | None
    reasoning_tokens: MetricInt | None
    total_tokens: MetricInt | None
    invalid_token_count: MetricInt
    cache_hit: bool
    cache_hit_ratio: float | None
    cost_usd: MetricFloat | None
    finish_reason: str | None
    tool_call_count: MetricInt
    request_bytes: MetricInt
    response_bytes: MetricInt
    streamed: bool
    analytics_truncated: bool
    request_fingerprint: str


_PROVIDERS = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "openrouter.ai": "openrouter",
    "api.groq.com": "groq",
    "api.together.xyz": "together",
    "api.mistral.ai": "mistral",
    "api.deepseek.com": "deepseek",
    "api.fireworks.ai": "fireworks",
    "api.cohere.com": "cohere",
    "api.replicate.com": "replicate",
    "generativelanguage.googleapis.com": "google",
    "api-inference.huggingface.co": "huggingface",
    "router.huggingface.co": "huggingface",
}


def detect_request_type(path: str, host: str, body: dict[str, Any] | None = None) -> str | None:
    """Recognize common and provider-neutral inference endpoints."""
    clean = path.split("?", 1)[0]
    low = clean.lower()
    host = host.lower()
    if "/chat/completions" in low:
        return "openai.chat"
    if low.endswith("/responses") or "/backend-api/codex/responses" in low:
        return "openai.responses"
    if low.endswith(("/completions", "/completion")):
        return "openai.completions"
    if low.endswith(("/embeddings", "/embedding")):
        return "openai.embeddings"
    if low.endswith(("/images/generations", "/images/edits", "/images/variations")):
        return "openai.images"
    if low.endswith(("/audio/speech", "/audio/transcriptions", "/audio/translations")):
        return "openai.audio"
    if low.endswith(("/moderations", "/rerank")):
        return "openai.inference"
    if low.endswith("/realtime") or "/openai/realtime" in low:
        return "openai.realtime"
    if "/model/" in low and low.endswith(("/invoke", "/invoke-with-response-stream")):
        return "aws.bedrock.invoke"
    if "/model/" in low and low.endswith(("/converse", "/converse-stream")):
        return "aws.bedrock.converse"
    if low.endswith((":generatecontent", ":streamgeneratecontent")):
        return "google.vertex.gemini"
    if low.endswith((":rawpredict", ":streamrawpredict")):
        return "google.vertex.anthropic"
    if low.endswith("/messages") or "/messages/batches" in low:
        if "anthropic" in host or isinstance(body, dict) and {"model", "messages"} <= body.keys():
            return "anthropic.messages"
        if "/threads/" in low or isinstance(body, dict) and ("intent" in body or "references" in body):
            return "github.copilot.threads"
    if "cohere" in host and low.endswith(("/chat", "/generate", "/embed", "/rerank")):
        return "cohere.inference"
    if low.endswith(("/api/chat", "/api/generate", "/api/embed")):
        return "ollama.inference"
    if "replicate" in host and "/predictions" in low:
        return "replicate.predictions"
    if "api-inference.huggingface.co" in host or "router.huggingface.co" in host:
        return "huggingface.inference"
    if "sagemaker" in host and low.endswith("/invocations"):
        return "aws.sagemaker"
    if any(key in low for key in ("realtime", "websocket", "/ws", "infer", "predict", "generate", "chat")):
        if body is None:
            return "generic.inference"
    if isinstance(body, dict):
        has_input = any(key in body for key in ("messages", "prompt", "input", "inputs", "contents", "instances"))
        inference_path = any(key in low for key in ("infer", "predict", "generate", "chat", "embed", "rerank"))
        known_provider = any(host == name or host.endswith("." + name) for name in _PROVIDERS)
        if has_input and (inference_path or known_provider):
            return "generic.inference"
    return None


def provider_from_host(host: str | None, request_type: str) -> str:
    normalized = (host or "").lower().split(":", 1)[0]
    if request_type == "aws.sagemaker":
        return "aws-sagemaker"
    if request_type.startswith("aws.bedrock"):
        return "aws-bedrock"
    for suffix, provider in _PROVIDERS.items():
        if normalized == suffix or normalized.endswith("." + suffix):
            return provider
    if normalized.endswith(".amazonaws.com"):
        return "aws-bedrock"
    if normalized.endswith(".openai.azure.com"):
        return "azure-openai"
    if normalized.endswith(".googleapis.com") or request_type.startswith("google."):
        return "google-vertex"
    return normalized or request_type.split(".", 1)[0]


def normalize_usage(response: dict[str, Any] | None) -> Usage:
    response = response or {}
    usage = _dict(response.get("usage"))
    gemini = _dict(response.get("usageMetadata"))
    invocation = _dict(response.get("amazon-bedrock-invocationMetrics"))
    meta = _dict(_dict(response.get("meta")).get("tokens"))
    prompt_details = _dict(usage.get("prompt_tokens_details"))
    input_details = _dict(usage.get("input_tokens_details"))
    completion_details = _dict(usage.get("completion_tokens_details"))
    output_details = _dict(usage.get("output_tokens_details"))
    invalid = _invalid_tokens(
        (
            usage,
            (
                "completion_tokens",
                "output_tokens",
                "outputTokens",
                "generated_tokens",
                "total_tokens",
                "totalTokens",
                "cache_read_input_tokens",
                "cacheReadInputTokens",
                "cache_creation_input_tokens",
                "cacheWriteInputTokens",
                "prompt_tokens",
                "input_tokens",
                "inputTokens",
                "prompt_eval_count",
                "eval_count",
            ),
        ),
        (gemini, ("candidatesTokenCount", "cachedContentTokenCount", "promptTokenCount", "totalTokenCount")),
        (invocation, ("outputTokenCount", "cacheReadInputTokenCount", "cacheWriteInputTokenCount", "inputTokenCount")),
        (meta, ("output_tokens", "input_tokens")),
        (prompt_details, ("cached_tokens",)),
        (input_details, ("cached_tokens",)),
        (completion_details, ("reasoning_tokens",)),
        (output_details, ("reasoning_tokens",)),
        (response, ("generation_token_count", "prompt_eval_count", "eval_count")),
    )

    output = _pick_int(usage, "completion_tokens", "output_tokens", "outputTokens", "generated_tokens", "eval_count")
    output = output if output is not None else _pick_int(gemini, "candidatesTokenCount")
    output = output if output is not None else _pick_int(invocation, "outputTokenCount")
    output = output if output is not None else _pick_int(meta, "output_tokens")
    output = output if output is not None else _int(response.get("generation_token_count"))

    cached = _pick_int(usage, "cache_read_input_tokens", "cacheReadInputTokens")
    cached = cached if cached is not None else _pick_int(prompt_details, "cached_tokens")
    cached = cached if cached is not None else _pick_int(input_details, "cached_tokens")
    cached = cached if cached is not None else _pick_int(gemini, "cachedContentTokenCount")
    cached = cached if cached is not None else _pick_int(invocation, "cacheReadInputTokenCount")

    cache_write = _pick_int(usage, "cache_creation_input_tokens", "cacheWriteInputTokens")
    cache_write = cache_write if cache_write is not None else _pick_int(invocation, "cacheWriteInputTokenCount")
    reported = _pick_int(usage, "prompt_tokens", "input_tokens", "inputTokens", "prompt_eval_count")
    reported = reported if reported is not None else _int(response.get("prompt_eval_count"))
    output = output if output is not None else _int(response.get("eval_count"))
    reported = reported if reported is not None else _pick_int(gemini, "promptTokenCount")
    reported = reported if reported is not None else _pick_int(invocation, "inputTokenCount")
    reported = reported if reported is not None else _pick_int(meta, "input_tokens")

    anthropic = "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage
    input_tokens = _sum(reported, cached, cache_write) if anthropic else reported
    if anthropic and any(value is not None for value in (reported, cached, cache_write)) and input_tokens is None:
        invalid += 1
    uncached = reported if anthropic else max(0, reported - (cached or 0)) if reported is not None else None
    reasoning = _pick_int(completion_details, "reasoning_tokens")
    reasoning = reasoning if reasoning is not None else _pick_int(output_details, "reasoning_tokens")
    total = _pick_int(usage, "total_tokens", "totalTokens")
    total = total if total is not None else _pick_int(gemini, "totalTokenCount")
    total = total if total is not None else _sum(input_tokens, output) if None not in (input_tokens, output) else None
    if total is None and input_tokens is not None and output is not None:
        invalid += 1
    cost = _pick_float(usage, "cost", "cost_usd")
    return Usage(
        input_tokens=input_tokens,
        uncached_input_tokens=uncached,
        cached_input_tokens=cached,
        cache_write_input_tokens=cache_write,
        output_tokens=output,
        reasoning_tokens=reasoning,
        total_tokens=total,
        cost_usd=cost,
        invalid_token_count=invalid,
    )


def build_record(
    *,
    run_id: str,
    started_at_ms: int,
    completed_at_ms: int | None,
    host: str | None,
    request_type: str,
    method: str,
    endpoint: str,
    request_model: str | None,
    request_fingerprint: str,
    machine_hash: str,
    session_hash: str,
    response: dict[str, Any] | None,
    http_status: int | None,
    success: bool,
    error_type: str | None = None,
    error_fingerprint: str | None = None,
    request_bytes: int = 0,
    response_bytes: int = 0,
    streamed: bool = False,
) -> Record:
    usage = normalize_usage(response)
    model = _model(request_model, response, endpoint)
    cached = usage.cached_input_tokens or 0
    return Record(
        request_id=f"req-{uuid.uuid4().hex}",
        run_id=run_id,
        machine_hash=machine_hash,
        session_hash=session_hash,
        started_at_ms=started_at_ms,
        completed_at_ms=completed_at_ms,
        duration_ms=max(0, completed_at_ms - started_at_ms) if completed_at_ms is not None else None,
        provider=provider_from_host(host, request_type),
        provider_host=host,
        request_type=request_type,
        method=method,
        endpoint=endpoint,
        model=model,
        http_status=http_status,
        success=success,
        error_type=error_type,
        error_fingerprint=error_fingerprint,
        input_tokens=usage.input_tokens,
        uncached_input_tokens=usage.uncached_input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        total_tokens=usage.total_tokens,
        invalid_token_count=usage.invalid_token_count,
        cache_hit=cached > 0,
        cache_hit_ratio=min(1.0, cached / usage.input_tokens) if usage.input_tokens else None,
        cost_usd=usage.cost_usd,
        finish_reason=_finish_reason(response),
        tool_call_count=_tool_count(response),
        request_bytes=request_bytes,
        response_bytes=response_bytes,
        streamed=streamed,
        analytics_truncated=bool((response or {}).get("analytics_truncated")),
        request_fingerprint=request_fingerprint,
    )


class Store:
    """One append-only table; views provide rollups."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def ensure(self) -> None:
        resolved = self.path.resolve()
        with _store_lock:
            if resolved in _ready_stores and _ready_stores[resolved] == _file_identity(resolved):
                return
            self._initialize(resolved)

    def _initialize(self, resolved: Path) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 60
        while True:
            try:
                self._initialize_once()
                _ready_stores[resolved] = _file_identity(resolved)
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def _initialize_once(self) -> None:
        with sqlite3.connect(self.path, timeout=60, isolation_level=None) as db:
            db.execute("PRAGMA busy_timeout=60000")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(_TABLE_SCHEMA)
                columns = {row[1] for row in db.execute("PRAGMA table_info(requests)")}
                if "invalid_token_count" not in columns:
                    db.execute("ALTER TABLE requests ADD COLUMN invalid_token_count INTEGER NOT NULL DEFAULT 0")
                if "analytics_truncated" not in columns:
                    db.execute("ALTER TABLE requests ADD COLUMN analytics_truncated INTEGER NOT NULL DEFAULT 0")
                for statement in _INDEXES:
                    db.execute(statement)
                if db.execute("PRAGMA user_version").fetchone()[0] < _SCHEMA_VERSION:
                    for name in _VIEWS:
                        db.execute(f"DROP VIEW IF EXISTS {name}")
                for statement in _VIEWS.values():
                    db.execute(statement.replace("CREATE VIEW ", "CREATE VIEW IF NOT EXISTS ", 1))
                db.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise

    def write(self, record: Record) -> None:
        self.ensure()
        row = record.model_dump()
        row.update(
            {
                "success": int(record.success),
                "cache_hit": int(record.cache_hit),
                "streamed": int(record.streamed),
                "analytics_truncated": int(record.analytics_truncated),
            }
        )
        columns = ", ".join(row)
        values = ", ".join(f":{column}" for column in row)
        with sqlite3.connect(self.path, timeout=60) as db:
            db.execute("PRAGMA busy_timeout=60000")
            db.execute(f"INSERT OR IGNORE INTO requests ({columns}) VALUES ({values})", row)


_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, machine_hash TEXT NOT NULL,
    session_hash TEXT NOT NULL, started_at_ms INTEGER NOT NULL, completed_at_ms INTEGER,
    duration_ms INTEGER, provider TEXT NOT NULL, provider_host TEXT, request_type TEXT NOT NULL,
    method TEXT NOT NULL, endpoint TEXT NOT NULL, model TEXT, http_status INTEGER,
    success INTEGER NOT NULL, error_type TEXT, error_fingerprint TEXT, input_tokens INTEGER,
    uncached_input_tokens INTEGER, cached_input_tokens INTEGER, cache_write_input_tokens INTEGER,
    output_tokens INTEGER, reasoning_tokens INTEGER, total_tokens INTEGER,
    invalid_token_count INTEGER NOT NULL, cache_hit INTEGER NOT NULL,
    cache_hit_ratio REAL, cost_usd REAL, finish_reason TEXT, tool_call_count INTEGER NOT NULL,
    request_bytes INTEGER NOT NULL, response_bytes INTEGER NOT NULL, streamed INTEGER NOT NULL,
    analytics_truncated INTEGER NOT NULL,
    request_fingerprint TEXT NOT NULL
)
"""
_INDEXES = (
    "CREATE INDEX IF NOT EXISTS requests_machine_session ON requests(machine_hash, session_hash, started_at_ms)",
    "CREATE INDEX IF NOT EXISTS requests_provider_model ON requests(provider, model, started_at_ms)",
)

_VIEWS = {
    "sessions": """CREATE VIEW sessions AS
SELECT machine_hash, session_hash, min(started_at_ms) first_started_at_ms,
       max(completed_at_ms) last_completed_at_ms, count(*) request_count,
       sum(success = 1) success_count, sum(success = 0) error_count,
       total(coalesce(input_tokens, 0)) input_tokens,
       total(coalesce(cached_input_tokens, 0)) cached_input_tokens,
       total(coalesce(output_tokens, 0)) output_tokens,
       total(invalid_token_count) invalid_token_count,
       total(analytics_truncated) analytics_truncated,
       total(coalesce(cost_usd, 0)) cost_usd
FROM requests GROUP BY machine_hash, session_hash""",
    "provider_daily": """CREATE VIEW provider_daily AS
SELECT date(started_at_ms / 1000, 'unixepoch') day, machine_hash, provider, model,
       count(*) requests, sum(success = 0) errors, avg(duration_ms) avg_duration_ms,
       total(input_tokens) input_tokens, total(cached_input_tokens) cached_input_tokens,
       total(output_tokens) output_tokens, total(invalid_token_count) invalid_token_count,
       total(analytics_truncated) analytics_truncated,
       total(cost_usd) cost_usd
FROM requests GROUP BY day, machine_hash, provider, model""",
}

_SCHEMA_VERSION = 3

_ready_stores: dict[Path, tuple[int, int] | None] = {}
_store_lock = threading.Lock()


def _file_identity(path: Path) -> tuple[int, int] | None:
    try:
        value = path.stat()
    except OSError:
        return None
    return value.st_dev, value.st_ino


def _reset_store_after_fork() -> None:
    global _store_lock
    _store_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_store_after_fork)


@cache
def machine_hash() -> str:
    identity = os.environ.get("IMPALA_SCOPE_MACHINE_ID") or _system_machine_id()
    return _hash("machine", identity)


@cache
def _system_machine_id() -> str:
    for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    if sys.platform == "win32":  # pragma: no cover
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                return str(winreg.QueryValueEx(key, "MachineGuid")[0])
        except OSError:
            pass
    return f"{platform.system()}:{platform.node()}:{uuid.getnode():012x}"


def _model(request_model: str | None, response: dict[str, Any] | None, endpoint: str) -> str | None:
    for key in ("model", "model_id", "modelId", "modelVersion"):
        value = (response or {}).get(key)
        if isinstance(value, str) and value:
            return value[:512]
    if request_model:
        return request_model[:512]
    return endpoint.split("/model/", 1)[1].split("/", 1)[0] if "/model/" in endpoint else None


def _finish_reason(response: dict[str, Any] | None) -> str | None:
    response = response or {}
    for key in ("stop_reason", "finish_reason", "stopReason", "status"):
        if isinstance(response.get(key), str):
            return response[key]
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        value = choices[0].get("finish_reason")
        return value if isinstance(value, str) else None
    return None


def _tool_count(response: dict[str, Any] | None) -> int:
    if not response:
        return 0
    if (reported := _int(response.get("tool_call_count"))) is not None:
        return reported
    count, stack = 0, [response]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            count += int(value.get("type") in {"tool_call", "function_call", "tool_use"})
            calls = value.get("tool_calls")
            if isinstance(calls, list):
                count += len(calls)
            stack.extend(v for key, v in value.items() if key != "tool_calls")
        elif isinstance(value, list):
            stack.extend(value)
    return min(count, SQLITE_INT_MAX)


def error_message(response: dict[str, Any] | None) -> str | None:
    if not response:
        return None
    error = response.get("error")
    if isinstance(error, dict):
        value = error.get("message") or error.get("type")
    else:
        value = error or response.get("message")
    return str(value)[:1000] if value is not None else None


def response_failed(response: dict[str, Any] | None) -> bool:
    """Return true for provider-level failures carried over a successful HTTP response."""
    response = response or {}
    if response.get("error"):
        return True
    status = response.get("status") or response.get("type")
    return isinstance(status, str) and status.lower() in {
        "failed",
        "cancelled",
        "canceled",
        "error",
        "response.failed",
        "response.cancelled",
    }


def _hash(kind: str, value: str) -> str:
    return hashlib.sha256(f"impala-scope-v1\0{kind}\0{value}".encode()).hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= METRIC_MAX else None


def _pick_int(source: dict[str, Any], *keys: str) -> int | None:
    return next((value for key in keys if (value := _int(source.get(key))) is not None), None)


def _pick_float(source: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = source.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        try:
            result = float(value)
        except OverflowError:
            continue
        if math.isfinite(result) and 0 <= result <= METRIC_MAX:
            return result
    return None


def _sum(*values: int | None) -> int | None:
    known = [value for value in values if value is not None]
    result = sum(known)
    return result if known and result <= METRIC_MAX else None


def _invalid_tokens(*sources: tuple[dict[str, Any], tuple[str, ...]]) -> int:
    return sum(
        1
        for source, keys in sources
        for key in keys
        if key in source and source[key] is not None and _int(source[key]) is None
    )
