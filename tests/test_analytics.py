import os
import sqlite3
import subprocess
import sys

import pytest
from pydantic import ValidationError

from impala_scope import analytics
from impala_scope.analytics import METRIC_MAX, Record, Store, build_record, machine_hash, normalize_usage


def test_normalizes_openai_and_anthropic_cache_semantics() -> None:
    openai = normalize_usage(
        {"usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 75}}}
    )
    assert (openai.input_tokens, openai.uncached_input_tokens, openai.cached_input_tokens) == (100, 25, 75)

    anthropic = normalize_usage(
        {
            "usage": {
                "input_tokens": 20,
                "cache_read_input_tokens": 70,
                "cache_creation_input_tokens": 10,
                "output_tokens": 5,
            }
        }
    )
    assert (anthropic.input_tokens, anthropic.uncached_input_tokens) == (100, 20)


def _record():
    return build_record(
        run_id="run-1",
        started_at_ms=1000,
        completed_at_ms=1250,
        host="openrouter.ai",
        request_type="openai.chat",
        method="POST",
        endpoint="/api/v1/chat/completions",
        request_model="gpt-5",
        request_fingerprint="f" * 64,
        machine_hash="m" * 64,
        session_hash="s" * 64,
        response={
            "choices": [{"finish_reason": "stop", "message": {"content": "secret answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        http_status=200,
        success=True,
        request_bytes=100,
        response_bytes=200,
    )


def test_database_has_only_flat_analytics_and_no_payload_columns(tmp_path) -> None:
    path = tmp_path / "profile.db"
    record = _record()
    Store(path).write(record)
    Store(path).write(record)

    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(requests)")}
        count = db.execute("SELECT count(*) FROM requests").fetchone()[0]
        session = db.execute("SELECT request_count, input_tokens, output_tokens FROM sessions").fetchone()
        serialized = " ".join(str(value) for value in db.execute("SELECT * FROM requests").fetchone())

    forbidden = {"request", "response", "body", "headers", "messages", "prompt", "content"}
    assert columns == set(Record.model_fields)
    assert columns.isdisjoint(forbidden)
    assert count == 1
    assert session == (1, 10, 2)
    assert "secret prompt" not in serialized
    assert "secret answer" not in serialized
    assert len(record.machine_hash) == len(record.session_hash) == 64
    with pytest.raises(ValidationError):
        Record.model_validate({**record.model_dump(), "request": {"secret": True}})


def test_machine_override_is_hashed(monkeypatch) -> None:
    try:
        monkeypatch.setenv("IMPALA_SCOPE_MACHINE_ID", "worker-a")
        machine_hash.cache_clear()
        value = machine_hash()
        assert len(value) == 64
        assert value != "worker-a"
        assert value == machine_hash()
    finally:
        machine_hash.cache_clear()


def test_invalid_token_values_are_counted_without_dropping_the_record(tmp_path) -> None:
    path = tmp_path / "profile.db"
    record = build_record(
        run_id="run-1",
        started_at_ms=1000,
        completed_at_ms=1100,
        host="api.openai.com",
        request_type="openai.chat",
        method="POST",
        endpoint="/v1/chat/completions",
        request_model="gpt-5",
        request_fingerprint="f" * 64,
        machine_hash="m" * 64,
        session_hash="s" * 64,
        response={
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": 1.5,
                "total_tokens": 1 << 70,
            }
        },
        http_status=200,
        success=True,
    )
    Store(path).write(record)

    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT input_tokens, output_tokens, total_tokens, invalid_token_count FROM requests"
        ).fetchone()
        rollup = db.execute("SELECT requests, invalid_token_count FROM provider_daily").fetchone()
    assert row == (None, None, None, 3)
    assert rollup == (1, 3)


def test_partial_usage_does_not_invent_total_tokens() -> None:
    assert normalize_usage({"usage": {"prompt_tokens": 7}}).total_tokens is None
    assert normalize_usage({"usage": {"completion_tokens": 5}}).total_tokens is None


def test_rollups_do_not_overflow_on_individually_valid_metrics(tmp_path) -> None:
    path = tmp_path / "profile.db"
    for index in range(10_000):
        record = build_record(
            run_id="run-1",
            started_at_ms=index,
            completed_at_ms=index + 1,
            host="api.openai.com",
            request_type="openai.chat",
            method="POST",
            endpoint="/v1/chat/completions",
            request_model="gpt-5",
            request_fingerprint=f"{index:064x}",
            machine_hash="m" * 64,
            session_hash="s" * 64,
            response={"usage": {"prompt_tokens": METRIC_MAX}},
            http_status=200,
            success=True,
        )
        Store(path).write(record)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT input_tokens FROM sessions").fetchone()[0] == float(METRIC_MAX * 10_000)


def test_derived_totals_and_extreme_costs_never_drop_or_poison_rows(tmp_path) -> None:
    usage = normalize_usage(
        {"usage": {"input_tokens": METRIC_MAX, "output_tokens": METRIC_MAX, "cost": 10**1000}}
    )
    assert usage.input_tokens == usage.output_tokens == METRIC_MAX
    assert usage.total_tokens is None
    assert usage.cost_usd is None
    assert usage.invalid_token_count == 1

    path = tmp_path / "profile.db"
    for index in range(2):
        record = build_record(
            run_id="run-1",
            started_at_ms=index,
            completed_at_ms=index + 1,
            host="api.openai.com",
            request_type="openai.chat",
            method="POST",
            endpoint="/v1/chat/completions",
            request_model="gpt-5",
            request_fingerprint=f"{index:064x}",
            machine_hash="m" * 64,
            session_hash="s" * 64,
            response={"usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 1e308}},
            http_status=200,
            success=True,
        )
        assert record.cost_usd is None
        Store(path).write(record)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT cost_usd FROM sessions").fetchone()[0] == 0.0


def test_existing_database_is_migrated_once_under_process_contention(tmp_path) -> None:
    path = tmp_path / "profile.db"
    Store(path).ensure()
    Store(path).write(_record())
    with sqlite3.connect(path) as db:
        db.executescript(
            "DROP VIEW sessions; DROP VIEW provider_daily; "
            "ALTER TABLE requests DROP COLUMN analytics_truncated; "
            "ALTER TABLE requests DROP COLUMN invalid_token_count; PRAGMA user_version=1;"
        )
    analytics._ready_stores.pop(path.resolve(), None)

    code = "import sys; from impala_scope.analytics import Store; Store(sys.argv[1]).ensure()"
    processes = [subprocess.Popen([sys.executable, "-c", code, str(path)]) for _ in range(24)]
    assert [process.wait(timeout=60) for process in processes] == [0] * len(processes)
    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(requests)")}
        assert {"invalid_token_count", "analytics_truncated"} <= columns
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3
        assert db.execute("SELECT count(*) FROM requests").fetchone()[0] == 1

    fresh = tmp_path / "fresh.db"
    processes = [subprocess.Popen([sys.executable, "-c", code, str(fresh)]) for _ in range(24)]
    assert [process.wait(timeout=60) for process in processes] == [0] * len(processes)
    with sqlite3.connect(fresh) as db:
        assert db.execute("SELECT count(*) FROM requests").fetchone()[0] == 0


def test_replaced_database_file_is_reinitialized_without_restart(tmp_path) -> None:
    path = tmp_path / "profile.db"
    Store(path).write(_record())
    replacement = tmp_path / "replacement.db"
    replacement.touch()
    os.replace(replacement, path)

    Store(path).write(_record().model_copy(update={"request_id": "replacement"}))
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT request_id FROM requests").fetchone()[0] == "replacement"
