"""Bounded, incremental response metadata summarization."""

import base64
import json
import re
import struct
import zlib
from collections.abc import Iterable
from typing import Any, Protocol

MAX_CAPTURE_BYTES = 1 << 20
MAX_LINE_BYTES = 1 << 18
MAX_TOOL_IDS = 4096
_SLICE_BYTES = 1 << 16
_COMPRESSED_SLICE_BYTES = 1 << 10
_MAX_DECODED_BYTES = MAX_CAPTURE_BYTES * 16
_MAX_METADATA_VALUE = 1 << 16
_USAGE_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "inputTokens",
    "outputTokens",
    "total_tokens",
    "totalTokens",
    "generated_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cacheReadInputTokens",
    "cacheWriteInputTokens",
    "promptTokenCount",
    "candidatesTokenCount",
    "cachedContentTokenCount",
    "totalTokenCount",
    "inputTokenCount",
    "outputTokenCount",
    "cacheReadInputTokenCount",
    "cacheWriteInputTokenCount",
    "prompt_eval_count",
    "eval_count",
    "cost",
    "cost_usd",
}
_DETAIL_KEYS = {
    "prompt_tokens_details": {"cached_tokens"},
    "input_tokens_details": {"cached_tokens"},
    "completion_tokens_details": {"reasoning_tokens"},
    "output_tokens_details": {"reasoning_tokens"},
}
_INTERESTING = (
    b'"usage"',
    b'"usageMetadata"',
    b'"model"',
    b'"finish_reason"',
    b'"stop_reason"',
    b'"stopReason"',
    b'"status"',
    b'"tool_calls"',
    b'"tool_use"',
    b'"function_call"',
    b'"prompt_eval_count"',
    b'"eval_count"',
    b'"response.completed"',
    b'"response.failed"',
    b'"response.cancelled"',
    b'"response.done"',
    b'"message_start"',
    b'"message_delta"',
    b'"error"',
)


class EventSummary:
    """Reduce provider events to a fixed-size usage/model/outcome summary."""

    def __init__(self, request_type: str) -> None:
        self.request_type = request_type
        self.summary: dict[str, Any] = {}
        self.tools: set[tuple[str, int, int]] = set()
        self.seen = False

    def feed(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        self.seen = True
        kind = event.get("type")
        final = event.get("response") if kind in {
            "response.completed",
            "response.failed",
            "response.cancelled",
            "response.done",
        } else None
        for source in (event, final if isinstance(final, dict) else {}):
            self._source(source)

        if isinstance(kind, str) and kind in {"response.failed", "response.cancelled"}:
            self.summary["status"] = kind.split(".", 1)[1]
        elif kind == "error":
            self.summary["status"] = "error"
        if kind == "message_start":
            message = event.get("message") or {}
            if isinstance(message, dict):
                self._source(message)
        elif kind == "message_delta":
            delta = event.get("delta") or {}
            if isinstance(delta, dict) and isinstance(delta.get("stop_reason"), str):
                self.summary["stop_reason"] = delta["stop_reason"][:128]
        elif kind in {"content_block_start", "response.output_item.added", "response.output_item.done"}:
            item = event.get("content_block") or event.get("item") or {}
            if isinstance(item, dict) and item.get("type") in {"tool_use", "function_call", "tool_call"}:
                slot = _index(event.get("index"), event.get("output_index"))
                family = "responses" if str(kind).startswith("response.output_item") else str(kind)
                self._tool((family, 0, slot))

        choices = event.get("choices")
        if isinstance(choices, list):
            for choice_position, choice in enumerate(choices):
                if not isinstance(choice, dict):
                    continue
                if isinstance(choice.get("finish_reason"), str):
                    self.summary["finish_reason"] = choice["finish_reason"][:128]
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                calls = delta.get("tool_calls") or []
                if isinstance(calls, list):
                    for call_position, call in enumerate(calls):
                        if isinstance(call, dict):
                            self._tool(("chat", choice_position, _index(call.get("index"), call_position)))

    def _source(self, source: dict[str, Any]) -> None:
        raw_usage = source.get("usage")
        usage = _usage(raw_usage)
        if isinstance(raw_usage, dict):
            usage = _usage(raw_usage.get("tokens")) or _usage(raw_usage.get("billed_units")) or usage
        if usage:
            self.summary["usage"] = {**(self.summary.get("usage") or {}), **usage}
        meta = source.get("meta")
        if isinstance(meta, dict) and (tokens := _usage(meta.get("tokens"))):
            self.summary["meta"] = {"tokens": tokens}
        gemini = _usage(source.get("usageMetadata"))
        if gemini:
            self.summary["usageMetadata"] = gemini
        invocation = _usage(source.get("amazon-bedrock-invocationMetrics"))
        if invocation:
            self.summary["amazon-bedrock-invocationMetrics"] = invocation
        ollama = _usage(source)
        if "prompt_eval_count" in ollama or "eval_count" in ollama:
            self.summary["usage"] = {**(self.summary.get("usage") or {}), **ollama}
        model = source.get("model")
        if isinstance(model, str):
            self.summary["model"] = model[:512]
        for key in ("finish_reason", "stop_reason", "stopReason", "status"):
            value = source.get(key)
            if isinstance(value, str):
                self.summary[key] = value[:128]
        error = source.get("error")
        if isinstance(error, dict):
            safe = {key: str(error[key])[:128] for key in ("type", "code") if error.get(key) is not None}
            if not safe and error:
                safe["type"] = "provider_error"
            if safe:
                self.summary["error"] = safe
        elif isinstance(error, str):
            self.summary["error"] = error[:128]
        if (tool_count := _safe_int(source.get("tool_call_count"))) is not None:
            self.summary["tool_call_count"] = min(tool_count, MAX_TOOL_IDS)
        choices = source.get("choices")
        if isinstance(choices, list):
            for choice_index, choice in enumerate(choices):
                message = choice.get("message") if isinstance(choice, dict) else None
                calls = message.get("tool_calls") if isinstance(message, dict) else None
                if isinstance(calls, list):
                    for call_index, _call in enumerate(calls[:MAX_TOOL_IDS]):
                        self._tool(("chat", choice_index, call_index))
        for family, items in (("responses", source.get("output")), ("anthropic", source.get("content"))):
            if isinstance(items, list):
                for index, item in enumerate(items[:MAX_TOOL_IDS]):
                    if isinstance(item, dict) and item.get("type") in {"tool_use", "function_call", "tool_call"}:
                        self._tool((family, 0, index))

    def _tool(self, identity: tuple[str, int, int]) -> None:
        if len(self.tools) < MAX_TOOL_IDS:
            self.tools.add(identity)

    def result(self) -> dict[str, Any] | None:
        if not self.seen or not self.summary and not self.tools:
            return None
        result = dict(self.summary)
        if self.tools:
            result["tool_call_count"] = len(self.tools)
        return result


class ResponseSummary:
    """Tee raw response bytes through optional decoding into a bounded reader."""

    def __init__(self, request_type: str, content_type: str, content_encoding: str = "") -> None:
        self.request_type = request_type
        self.content_type = content_type.lower()
        self.response_bytes = 0
        self._decoder = _DecoderChain(content_encoding)
        self._decode_failed = False
        encodings = {name.strip().lower() for name in content_encoding.split(",")}
        self.analytics_truncated = bool(encodings & {"br", "zstd", "x-zstd"})
        self._decoded_bytes = 0
        if "text/event-stream" in self.content_type:
            self._reader: _Reader = _SSEReader(request_type)
        elif "vnd.amazon.eventstream" in self.content_type:
            self._reader = _BedrockReader(request_type)
        elif "ndjson" in self.content_type or request_type == "ollama.inference":
            self._reader = _NDJSONReader(request_type)
        else:
            self._reader = _JSONReader(request_type)
        self.streamed = not isinstance(self._reader, _JSONReader)

    @property
    def buffered_bytes(self) -> int:
        return self._reader.buffered_bytes

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.response_bytes += len(chunk)
        if self._decode_failed:
            return
        pieces = (
            (chunk[start : start + _COMPRESSED_SLICE_BYTES] for start in range(0, len(chunk), _COMPRESSED_SLICE_BYTES))
            if self._decoder.active
            else (chunk,)
        )
        for piece in pieces:
            if self._decode_failed:
                break
            try:
                self._feed_decoded(self._decoder.feed(piece))
            except Exception:
                self._decode_failed = True
                self.analytics_truncated = True
                return

    def finish(self) -> dict[str, Any] | None:
        if not self._decode_failed:
            try:
                self._feed_decoded(self._decoder.finish())
            except Exception:
                self._decode_failed = True
                self.analytics_truncated = True
        result = self._reader.finish()
        if self.analytics_truncated:
            result = {**(result or {}), "analytics_truncated": True}
        return result

    def _feed_decoded(self, decoded: bytes) -> None:
        if not decoded:
            return
        if self._decoder.active:
            remaining = _MAX_DECODED_BYTES - self._decoded_bytes
            if remaining <= 0:
                self._decode_failed = True
                self.analytics_truncated = True
                return
            self._decoded_bytes += min(len(decoded), remaining)
            if len(decoded) > remaining:
                decoded = decoded[:remaining]
                self._decode_failed = True
                self.analytics_truncated = True
        if len(decoded) > MAX_CAPTURE_BYTES * 2 and isinstance(self._reader, (_SSEReader, _NDJSONReader)):
            self._reader.feed(decoded[:MAX_CAPTURE_BYTES])
            self._reader.discontinuity()
            self._reader.feed(decoded[-MAX_CAPTURE_BYTES:])
        else:
            self._reader.feed(decoded)


class _Reader(Protocol):
    @property
    def buffered_bytes(self) -> int: ...

    def feed(self, chunk: bytes) -> None: ...

    def finish(self) -> dict[str, Any] | None: ...


class _JSONReader:
    def __init__(self, request_type: str) -> None:
        self.request_type = request_type
        self.body = bytearray()
        self.metadata = _TopLevelMetadata(request_type)
        self.overflow = False

    @property
    def buffered_bytes(self) -> int:
        return len(self.body) + self.metadata.buffered_bytes

    def feed(self, chunk: bytes) -> None:
        if not self.overflow and len(self.body) + len(chunk) <= MAX_CAPTURE_BYTES:
            self.body.extend(chunk)
            return
        if not self.overflow:
            previous = bytes(self.body)
            self.metadata.feed(previous)
            self.metadata.feed(chunk)
            self.body.clear()
            self.overflow = True
            return
        self.metadata.feed(chunk)

    def finish(self) -> dict[str, Any] | None:
        if not self.overflow:
            try:
                value = json.loads(self.body) if self.body else None
            except Exception:
                self.body.clear()
                return None
            result = _metadata(value, self.request_type)
        else:
            result = self.metadata.finish()
        self.body.clear()
        return result


_JSON_TOKEN = re.compile(rb'["\\{}\[\],:]')
_METADATA_KEYS = {
    "usage",
    "usageMetadata",
    "amazon-bedrock-invocationMetrics",
    "model",
    "status",
    "finish_reason",
    "stop_reason",
    "stopReason",
    "prompt_eval_count",
    "eval_count",
    "generation_token_count",
    "error",
    "choices",
    "output",
    "content",
    "tool_call_count",
}


class _TopLevelMetadata:
    """Capture only selected root-object values without retaining a large body."""

    def __init__(self, request_type: str) -> None:
        self.request_type = request_type
        self.depth = 0
        self.in_string = False
        self.escaped = False
        self.expect_key = False
        self.key = bytearray()
        self.reading_key = False
        self.candidate_key: str | None = None
        self.active: str | None = None
        self.value = bytearray()
        self.oversized = False
        self.truncated = False
        self.values: dict[str, Any] = {}

    @property
    def buffered_bytes(self) -> int:
        return len(self.value) + len(self.key)

    def feed(self, chunk: bytes) -> None:
        capture_from = 0 if self.active is not None else None
        previous_end = 0
        for match in _JSON_TOKEN.finditer(chunk):
            token = match.group()
            if self.in_string and self.escaped and match.start() > previous_end:
                self.escaped = False
            if self.in_string:
                if self.reading_key:
                    self.key.extend(chunk[previous_end : match.start()])
                if token == b"\\":
                    self.escaped = not self.escaped
                    self.reading_key = False
                    self.key.clear()
                elif token == b'"':
                    if self.escaped:
                        self.escaped = False
                        self.reading_key = False
                        self.key.clear()
                    else:
                        self.in_string = False
                        if self.reading_key:
                            decoded = self.key.decode(errors="ignore")
                            self.candidate_key = decoded if decoded in _METADATA_KEYS else None
                        self.reading_key = False
                        self.key.clear()
                elif self.reading_key:
                    self.key.extend(token)
                previous_end = match.end()
                continue
            if token == b'"':
                self.in_string = True
                self.reading_key = self.depth == 1 and self.expect_key
                self.key.clear()
                previous_end = match.end()
                continue
            if token == b":" and self.depth == 1:
                self.expect_key = False
                if self.candidate_key is not None and self.active is None:
                    self.active = self.candidate_key
                    self.value.clear()
                    self.oversized = False
                    capture_from = match.end()
                self.candidate_key = None
                previous_end = match.end()
                continue
            if token in {b"{", b"["}:
                self.depth += 1
                if self.depth == 1 and token == b"{":
                    self.expect_key = True
                previous_end = match.end()
                continue
            if token in {b"}", b"]"}:
                if self.active is not None and self.depth == 1 and capture_from is not None:
                    self._append(chunk[capture_from : match.start()])
                    self._commit()
                    capture_from = None
                self.depth = max(0, self.depth - 1)
                previous_end = match.end()
                continue
            if token == b"," and self.depth == 1:
                if self.active is not None and capture_from is not None:
                    self._append(chunk[capture_from : match.start()])
                    self._commit()
                    capture_from = None
                self.expect_key = True
                self.candidate_key = None
            previous_end = match.end()
        if self.active is not None and capture_from is not None:
            self._append(chunk[capture_from:])
        if self.in_string and self.reading_key:
            self.key.extend(chunk[previous_end:])

    def finish(self) -> dict[str, Any] | None:
        if self.active is not None:
            self._commit()
        result = _metadata(self.values, self.request_type)
        if self.truncated:
            result = {**(result or {}), "analytics_truncated": True}
        return result

    def _append(self, value: bytes) -> None:
        if self.oversized:
            return
        remaining = _MAX_METADATA_VALUE - len(self.value)
        if len(value) > remaining:
            self.value.clear()
            self.oversized = True
            self.truncated = True
        else:
            self.value.extend(value)

    def _commit(self) -> None:
        if self.active is not None and not self.oversized:
            try:
                self.values[self.active] = json.loads(self.value)
            except Exception:
                pass
        elif self.active == "error":
            self.values["error"] = {"type": "provider_error"}
        self.active = None
        self.value.clear()
        self.oversized = False


class _SSEReader:
    def __init__(self, request_type: str) -> None:
        self.events = EventSummary(request_type)
        self.pending = bytearray()
        self.data: list[bytes] = []
        self.data_bytes = 0
        self.discarding_line = False
        self.discarding_event = False
        self.after_cr = False

    @property
    def buffered_bytes(self) -> int:
        return len(self.pending) + self.data_bytes

    def feed(self, chunk: bytes) -> None:
        start = 0
        if self.after_cr:
            self.after_cr = False
            if chunk.startswith(b"\n"):
                start = 1
        while start < len(chunk):
            lf = chunk.find(b"\n", start)
            cr = chunk.find(b"\r", start)
            end = lf if cr < 0 else cr if lf < 0 else min(lf, cr)
            boundary = len(chunk) if end < 0 else end
            piece = chunk[start:boundary]
            if not self.discarding_line and len(self.pending) + len(piece) <= MAX_LINE_BYTES:
                self.pending.extend(piece)
            else:
                self.pending.clear()
                self.discarding_line = True
            if end < 0:
                return
            if not self.discarding_line:
                self._line(bytes(self.pending))
            self.pending.clear()
            self.discarding_line = False
            if chunk[end] == 13:
                if end + 1 < len(chunk) and chunk[end + 1] == 10:
                    start = end + 2
                else:
                    start = end + 1
                    self.after_cr = start == len(chunk)
            else:
                start = end + 1

    def finish(self) -> dict[str, Any] | None:
        if self.pending and not self.discarding_line:
            self._line(bytes(self.pending))
        self._emit()
        self.pending.clear()
        self.data.clear()
        return self.events.result()

    def discontinuity(self) -> None:
        self.pending.clear()
        self.data.clear()
        self.data_bytes = 0
        self.discarding_line = True
        self.discarding_event = False
        self.after_cr = False

    def _line(self, line: bytes) -> None:
        if not line:
            self._emit()
            return
        if not line.startswith(b"data:") or self.discarding_event:
            return
        value = line[5:]
        if value.startswith(b" "):
            value = value[1:]
        if self.data_bytes + len(value) > MAX_LINE_BYTES:
            self.data.clear()
            self.data_bytes = 0
            self.discarding_event = True
            return
        self.data.append(value)
        self.data_bytes += len(value)

    def _emit(self) -> None:
        if self.discarding_event:
            self.discarding_event = False
        elif self.data:
            raw = b"\n".join(self.data).strip()
            if raw and raw != b"[DONE]" and any(marker in raw for marker in _INTERESTING):
                try:
                    value = json.loads(raw)
                except Exception:
                    value = None
                if isinstance(value, dict):
                    self.events.feed(value)
        self.data.clear()
        self.data_bytes = 0


class _NDJSONReader:
    def __init__(self, request_type: str) -> None:
        self.events = EventSummary(request_type)
        self.pending = bytearray()
        self.discarding = False

    @property
    def buffered_bytes(self) -> int:
        return len(self.pending)

    def feed(self, chunk: bytes) -> None:
        for piece in chunk.splitlines(keepends=True):
            complete = piece.endswith((b"\n", b"\r"))
            clean = piece.rstrip(b"\r\n")
            if not self.discarding and len(self.pending) + len(clean) <= MAX_LINE_BYTES:
                self.pending.extend(clean)
            else:
                self.pending.clear()
                self.discarding = True
            if complete:
                self._emit()

    def finish(self) -> dict[str, Any] | None:
        self._emit()
        return self.events.result()

    def discontinuity(self) -> None:
        self.pending.clear()
        self.discarding = True

    def _emit(self) -> None:
        if self.pending and not self.discarding:
            try:
                value = json.loads(self.pending)
            except Exception:
                value = None
            if isinstance(value, dict):
                self.events.feed(value)
        self.pending.clear()
        self.discarding = False


class _BedrockReader:
    def __init__(self, request_type: str) -> None:
        self.events = EventSummary(request_type)
        self.summary: dict[str, Any] = {}
        self.buffer = bytearray()
        self.tools = 0
        self.discarding = 0

    @property
    def buffered_bytes(self) -> int:
        return len(self.buffer)

    def feed(self, chunk: bytes) -> None:
        for start in range(0, len(chunk), _SLICE_BYTES):
            piece = chunk[start : start + _SLICE_BYTES]
            if self.discarding:
                skipped = min(self.discarding, len(piece))
                self.discarding -= skipped
                piece = piece[skipped:]
                if not piece:
                    continue
            self.buffer.extend(piece)
            self._drain()
            if len(self.buffer) > MAX_CAPTURE_BYTES:
                total = struct.unpack(">I", self.buffer[:4])[0] if len(self.buffer) >= 4 else 0
                self.discarding = max(0, total - len(self.buffer))
                self.buffer.clear()

    def finish(self) -> dict[str, Any] | None:
        self._drain()
        nested = self.events.result()
        if nested:
            self.summary.update(nested)
        if self.tools:
            self.summary["tool_call_count"] = min(self.tools, MAX_TOOL_IDS)
        self.buffer.clear()
        return self.summary or None

    def _drain(self) -> None:
        while len(self.buffer) >= 12:
            total, header_size = struct.unpack(">II", self.buffer[:8])
            if total < 16 or header_size > total - 16:
                self.buffer.clear()
                return
            if total > MAX_CAPTURE_BYTES:
                if len(self.buffer) >= total:
                    del self.buffer[:total]
                else:
                    self.discarding = total - len(self.buffer)
                    self.buffer.clear()
                continue
            if len(self.buffer) < total:
                return
            message = bytes(self.buffer[:total])
            del self.buffer[:total]
            headers = _headers(message[12 : 12 + header_size])
            payload = message[12 + header_size : total - 4]
            try:
                value = json.loads(payload)
            except Exception:
                continue
            if isinstance(value, dict):
                self._event(
                    headers.get(":event-type") or headers.get(":exception-type"),
                    value,
                    headers.get(":message-type"),
                )

    def _event(self, event: str | None, payload: dict[str, Any], message_type: str | None = None) -> None:
        if message_type == "exception" or isinstance(event, str) and event.lower().endswith("exception"):
            self.summary["status"] = "error"
            self.summary["error"] = {"type": (event or "bedrock_exception")[:128]}
            return
        if event == "metadata":
            usage = _usage(payload.get("usage"))
            if usage:
                self.summary["usage"] = usage
        elif event == "messageStop" and isinstance(payload.get("stopReason"), str):
            self.summary["stopReason"] = payload["stopReason"][:128]
        elif event == "contentBlockStart" and "toolUse" in (payload.get("start") or {}):
            self.tools += 1
        elif event == "chunk":
            raw = payload.get("bytes")
            if not isinstance(raw, str):
                return
            try:
                decoded = json.loads(base64.b64decode(raw))
            except Exception:
                return
            if isinstance(decoded, dict):
                self.events.feed(decoded)


class _DecoderChain:
    def __init__(self, encoding: str) -> None:
        self.decoders = [_decoder(name.strip()) for name in reversed(encoding.split(",")) if name.strip()]

    @property
    def active(self) -> bool:
        return bool(self.decoders)

    def feed(self, chunk: bytes) -> bytes:
        for decoder in self.decoders:
            chunk = decoder.feed(chunk)
        return chunk

    def finish(self) -> bytes:
        result = b""
        for index, decoder in enumerate(self.decoders):
            piece = decoder.finish()
            for downstream in self.decoders[index + 1 :]:
                piece = downstream.feed(piece)
            result += piece
        return result


class _Decoder(Protocol):
    def feed(self, chunk: bytes) -> bytes: ...

    def finish(self) -> bytes: ...


class _IdentityDecoder:
    def feed(self, chunk: bytes) -> bytes:
        return chunk

    def finish(self) -> bytes:
        return b""


class _ZlibDecoder:
    def __init__(self, mode: int) -> None:
        self.decoder = zlib.decompressobj(mode)

    def feed(self, chunk: bytes) -> bytes:
        return self.decoder.decompress(chunk)

    def finish(self) -> bytes:
        return self.decoder.flush()


class _DeflateDecoder:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.decoder: Any | None = None

    def feed(self, chunk: bytes) -> bytes:
        if self.decoder is not None:
            return self.decoder.decompress(chunk)
        self.buffer.extend(chunk)
        if len(self.buffer) < 2:
            return b""
        first, second = self.buffer[:2]
        wrapped = first & 0x0F == 8 and (first << 8 | second) % 31 == 0
        self.decoder = zlib.decompressobj(zlib.MAX_WBITS if wrapped else -zlib.MAX_WBITS)
        value = self.decoder.decompress(bytes(self.buffer))
        self.buffer.clear()
        return value

    def finish(self) -> bytes:
        if self.decoder is None:
            self.decoder = zlib.decompressobj(-zlib.MAX_WBITS)
            value = self.decoder.decompress(bytes(self.buffer))
            self.buffer.clear()
            return value + self.decoder.flush()
        return self.decoder.flush()


class _DiscardDecoder:
    """Skip formats whose Python APIs cannot cap decompressed output."""

    def feed(self, chunk: bytes) -> bytes:
        return b""

    def finish(self) -> bytes:
        return b""


def _decoder(name: str) -> _Decoder:
    name = name.lower()
    if name in {"gzip", "x-gzip"}:
        return _ZlibDecoder(16 + zlib.MAX_WBITS)
    if name == "deflate":
        return _DeflateDecoder()
    if name in {"br", "zstd", "x-zstd"}:
        return _DiscardDecoder()
    return _IdentityDecoder()


def parse_http(
    request_type: str, content: bytes, content_type: str, content_encoding: str = ""
) -> dict[str, Any] | None:
    collector = ResponseSummary(request_type, content_type, content_encoding)
    collector.feed(content)
    return collector.finish()


def summarize_events(request_type: str, events: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    summary = EventSummary(request_type)
    for event in events:
        summary.feed(event)
    return summary.result()


def summarize_response(request_type: str, response: dict[str, Any] | None) -> dict[str, Any] | None:
    return _metadata(response, request_type)


def _metadata(value: Any, request_type: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    summary = EventSummary(request_type)
    summary.feed(value)
    return summary.result()


def _usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key in _USAGE_KEYS if _scalar(value.get(key))}
    for key, allowed in _DETAIL_KEYS.items():
        details = value.get(key)
        if isinstance(details, dict):
            safe = {name: details[name] for name in allowed if _scalar(details.get(name))}
            if safe:
                result[key] = safe
    return result


def _scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _index(*values: Any) -> int:
    for value in values:
        if isinstance(value, int) and value >= 0:
            return min(value, MAX_TOOL_IDS - 1)
    return 0


def _headers(data: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    index = 0
    while index < len(data):
        size = data[index]
        index += 1
        name = data[index : index + size].decode(errors="replace")
        index += size
        if index >= len(data) or data[index] != 7:
            break
        index += 1
        if index + 2 > len(data):
            break
        length = struct.unpack(">H", data[index : index + 2])[0]
        index += 2
        headers[name] = data[index : index + length].decode(errors="replace")
        index += length
    return headers
