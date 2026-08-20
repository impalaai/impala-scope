import base64
import gzip
import json
import struct
import zlib

from impala_scope.analytics import response_failed
from impala_scope.streaming import (
    MAX_CAPTURE_BYTES,
    MAX_LINE_BYTES,
    MAX_TOOL_IDS,
    EventSummary,
    ResponseSummary,
    parse_http,
    summarize_events,
)


def test_sse_keeps_usage_but_not_generated_text() -> None:
    content = (
        b'data: {"id":"x","model":"gpt-5","choices":[{"delta":{"content":"secret"}}]}\n\n'
        b'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1}}\n\n'
        b"data: [DONE]\n\n"
    )
    summary = parse_http("openai.chat", content, "text/event-stream")
    assert summary == {"usage": {"prompt_tokens": 4, "completion_tokens": 1}, "model": "gpt-5", "finish_reason": "stop"}
    assert "secret" not in str(summary)


def test_websocket_events_reduce_to_metadata() -> None:
    summary = summarize_events(
        "openai.responses",
        [
            {"type": "response.output_text.delta", "delta": "secret"},
            {
                "type": "response.completed",
                "response": {"model": "gpt-5", "status": "completed", "usage": {"input_tokens": 3, "output_tokens": 1}},
            },
        ],
    )
    assert summary == {"usage": {"input_tokens": 3, "output_tokens": 1}, "model": "gpt-5", "status": "completed"}


def _eventstream(events: list[tuple[str, dict]]) -> bytes:
    stream = b""
    for event, payload in events:
        headers = b""
        for name, value in {":event-type": event, ":content-type": "application/json"}.items():
            name_bytes, value_bytes = name.encode(), value.encode()
            headers += (
                bytes([len(name_bytes)]) + name_bytes + b"\x07" + struct.pack(">H", len(value_bytes)) + value_bytes
            )
        body = json.dumps(payload).encode()
        total = 12 + len(headers) + len(body) + 4
        stream += struct.pack(">III", total, len(headers), 0) + headers + body + b"\0\0\0\0"
    return stream


def test_bedrock_converse_stream_discards_text() -> None:
    stream = _eventstream(
        [
            ("contentBlockDelta", {"delta": {"text": "private Bedrock output"}}),
            ("messageStop", {"stopReason": "end_turn"}),
            ("metadata", {"usage": {"inputTokens": 12, "outputTokens": 5, "totalTokens": 17}}),
        ]
    )
    summary = parse_http("aws.bedrock.converse", stream, "application/vnd.amazon.eventstream")
    assert summary == {"stopReason": "end_turn", "usage": {"inputTokens": 12, "outputTokens": 5, "totalTokens": 17}}
    assert "private" not in str(summary)


def test_bedrock_invoke_stream_extracts_anthropic_usage() -> None:
    events = [
        {
            "type": "message_start",
            "message": {"model": "anthropic.claude", "usage": {"input_tokens": 4, "output_tokens": 0}},
        },
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "private"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 6}},
    ]
    framed = [("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()}) for event in events]
    summary = parse_http("aws.bedrock.invoke", _eventstream(framed), "application/vnd.amazon.eventstream")
    assert summary == {
        "usage": {"input_tokens": 4, "output_tokens": 6},
        "model": "anthropic.claude",
        "stop_reason": "end_turn",
    }


def test_sse_summary_is_incremental_and_bounded() -> None:
    collector = ResponseSummary("openai.chat", "text/event-stream")
    oversized = b'data: {"content":"' + b"x" * (MAX_LINE_BYTES + 1) + b'"}\n'
    usage = b'data: {"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n'

    for chunk in (oversized[:100], oversized[100:], usage[:13], usage[13:]):
        collector.feed(chunk)
        assert collector.buffered_bytes <= MAX_LINE_BYTES

    assert collector.response_bytes == len(oversized) + len(usage)
    assert collector.finish() == {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}


def test_large_json_body_is_not_retained() -> None:
    collector = ResponseSummary("openai.chat", "application/json")
    body = b"x" * (MAX_CAPTURE_BYTES + 1)
    collector.feed(body)

    assert collector.response_bytes == len(body)
    assert collector.buffered_bytes <= MAX_CAPTURE_BYTES
    assert collector.finish() is None


def test_large_json_keeps_usage_from_the_bounded_tail() -> None:
    body = (
        b'{"model":"gpt-5","output":"'
        + b"x" * (MAX_CAPTURE_BYTES * 2)
        + b'","usage":{"prompt_tokens":4,"completion_tokens":2}}'
    )
    collector = ResponseSummary("openai.chat", "application/json")
    for start in range(0, len(body), 65_536):
        collector.feed(body[start : start + 65_536])
    assert collector.buffered_bytes <= MAX_CAPTURE_BYTES
    assert collector.finish() == {
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        "model": "gpt-5",
        "analytics_truncated": True,
    }


def test_large_json_ignores_nested_usage_from_model_output() -> None:
    body = (
        b'{"output":[{"usage":{"prompt_tokens":999,"completion_tokens":888},"text":"'
        + b"x" * (MAX_CAPTURE_BYTES * 2)
        + b'"}],"usage":{"prompt_tokens":4,"completion_tokens":2},"model":"gpt-5"}'
    )
    collector = ResponseSummary("openai.chat", "application/json")
    for start in range(0, len(body), 65_536):
        collector.feed(body[start : start + 65_536])
    assert collector.finish() == {
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        "model": "gpt-5",
        "analytics_truncated": True,
    }


def test_large_json_root_metadata_key_can_span_chunks() -> None:
    prefix = b'{"output":"' + b"x" * (MAX_CAPTURE_BYTES + 1) + b'",'
    collector = ResponseSummary("openai.chat", "application/json")
    collector.feed(prefix)
    collector.feed(b'"us')
    collector.feed(b'age":{"prompt_tokens":4,"completion_tokens":2}}')
    assert collector.finish() == {
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        "analytics_truncated": True,
    }


def test_large_json_preserves_top_level_error_and_small_tool_metadata() -> None:
    error_body = b'{"generated":"' + b"x" * (MAX_CAPTURE_BYTES * 2) + b'","error":{"type":"overloaded"}}'
    error = parse_http("openai.chat", error_body, "application/json")
    assert error == {"error": {"type": "overloaded"}}
    assert response_failed(error)
    oversized_error = parse_http(
        "openai.chat",
        b'{"generated":"' + b"x" * MAX_CAPTURE_BYTES + b'","error":{"message":"' + b"y" * MAX_CAPTURE_BYTES + b'"}}',
        "application/json",
    )
    assert oversized_error == {"error": {"type": "provider_error"}, "analytics_truncated": True}
    assert response_failed(oversized_error)

    tool_body = (
        b'{"generated":"'
        + b"x" * (MAX_CAPTURE_BYTES * 2)
        + b'","choices":[{"message":{"tool_calls":[{"id":"a"},{"id":"b"}]}}]}'
    )
    assert parse_http("openai.chat", tool_body, "application/json") == {"tool_call_count": 2}


def test_gzip_sse_is_decoded_only_for_analytics() -> None:
    plain = b'data: {"usage":{"prompt_tokens":8,"completion_tokens":3}}\n\n'
    compressed = gzip.compress(plain)
    assert parse_http("openai.chat", compressed, "text/event-stream", "gzip") == {
        "usage": {"prompt_tokens": 8, "completion_tokens": 3}
    }


def test_invalid_compression_does_not_break_capture_lifecycle() -> None:
    collector = ResponseSummary("openai.chat", "text/event-stream", "gzip")
    collector.feed(b"not-gzip")
    assert collector.finish() == {"analytics_truncated": True}
    assert collector.response_bytes == len(b"not-gzip")


def test_multiline_sse_data_is_assembled() -> None:
    content = b'data: {"usage":\ndata: {"prompt_tokens":2,"completion_tokens":1}}\n\n'
    assert parse_http("openai.chat", content, "text/event-stream") == {
        "usage": {"prompt_tokens": 2, "completion_tokens": 1}
    }


def test_cr_only_sse_events_are_parsed_across_chunks() -> None:
    collector = ResponseSummary("openai.chat", "text/event-stream")
    collector.feed(b'data: {"model":"gpt-5"}\r')
    collector.feed(b'\rdata: {"usage":{"prompt_tokens":2,"completion_tokens":1}}\r\r')
    assert collector.finish() == {
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        "model": "gpt-5",
    }


def test_streamed_tool_continuations_count_once_and_state_is_capped() -> None:
    summary = EventSummary("openai.chat")
    for payload in (
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1"}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "a"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "b"}}]}}]},
    ):
        summary.feed(payload)
    assert summary.result() == {"tool_call_count": 1}

    for index in range(MAX_TOOL_IDS + 100):
        summary.feed({"choices": [{"delta": {"tool_calls": [{"index": index}]}}]})
    assert len(summary.tools) == MAX_TOOL_IDS


def test_responses_tool_lifecycle_and_nonstream_tools_count_once() -> None:
    summary = EventSummary("openai.responses")
    item = {"type": "function_call", "id": "call-1"}
    summary.feed({"type": "response.output_item.added", "output_index": 0, "item": item})
    summary.feed({"type": "response.output_item.done", "output_index": 0, "item": item})
    assert summary.result() == {"tool_call_count": 1}

    assert parse_http(
        "openai.chat",
        b'{"choices":[{"message":{"tool_calls":[{"id":"a"},{"id":"b"}]}}]}',
        "application/json",
    ) == {"tool_call_count": 2}
    assert parse_http(
        "anthropic.messages",
        b'{"content":[{"type":"text","text":"private"},{"type":"tool_use","id":"a"}]}',
        "application/json",
    ) == {"tool_call_count": 1}


def test_cohere_usage_shapes_are_preserved() -> None:
    assert parse_http(
        "cohere.inference",
        b'{"usage":{"tokens":{"input_tokens":6,"output_tokens":2},"billed_units":{"input_tokens":4}}}',
        "application/json",
    ) == {"usage": {"input_tokens": 6, "output_tokens": 2}}
    assert parse_http(
        "cohere.inference", b'{"meta":{"tokens":{"input_tokens":7,"output_tokens":3}}}', "application/json"
    ) == {"meta": {"tokens": {"input_tokens": 7, "output_tokens": 3}}}


def test_streamed_error_events_are_failures() -> None:
    response = parse_http(
        "anthropic.messages",
        b'data: {"type":"error","error":{"type":"overloaded_error","message":"private"}}\n\n',
        "text/event-stream",
    )
    assert response == {"status": "error", "error": {"type": "overloaded_error"}}
    assert response_failed(response)


def test_bedrock_exception_event_is_a_failure() -> None:
    response = parse_http(
        "aws.bedrock.invoke",
        _eventstream([("modelStreamErrorException", {"message": "private provider failure"})]),
        "application/vnd.amazon.eventstream",
    )
    assert response == {"status": "error", "error": {"type": "modelStreamErrorException"}}
    assert response_failed(response)


def test_unsafe_compression_formats_are_not_decompressed_for_analytics() -> None:
    collector = ResponseSummary("openai.chat", "application/json", "br")
    collector.feed(b"adversarial compressed bytes")
    assert collector.finish() == {"analytics_truncated": True}
    assert collector.response_bytes == len(b"adversarial compressed bytes")


def test_gzip_analytics_budget_is_reported_when_truncated() -> None:
    compressed = gzip.compress(b"x" * (17 * MAX_CAPTURE_BYTES))
    collector = ResponseSummary("openai.chat", "application/json", "gzip")
    collector.feed(compressed)
    assert collector.finish() == {"analytics_truncated": True}


def test_raw_and_wrapped_deflate_are_both_profiled() -> None:
    plain = b'{"usage":{"prompt_tokens":8,"completion_tokens":3}}'
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    raw = compressor.compress(plain) + compressor.flush()
    expected = {"usage": {"prompt_tokens": 8, "completion_tokens": 3}}
    assert parse_http("openai.chat", raw, "application/json", "deflate") == expected
    assert parse_http("openai.chat", zlib.compress(plain), "application/json", "deflate") == expected


def test_ollama_ndjson_usage_is_normalized_as_metadata() -> None:
    content = (
        b'{"model":"qwen","message":{"content":"private"},"done":false}\n'
        b'{"model":"qwen","done":true,"prompt_eval_count":12,"eval_count":5}\n'
    )
    assert parse_http("ollama.inference", content, "application/x-ndjson") == {
        "usage": {"prompt_eval_count": 12, "eval_count": 5},
        "model": "qwen",
    }
