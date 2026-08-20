import os
import sqlite3

import pytest

from impala_scope.capture import RequestCollector, capture, capture_later, config, configure, flush, record_session


def test_record_session_groups_rows_using_only_a_hash(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path, server_url="*")
    with record_session("raw-customer-session"):
        for started in (1000, 2000):
            assert capture(
                request={"model": "x", "messages": []},
                response={"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
                headers=None,
                started_at_ms=started,
                completed_at_ms=started + 10,
                host="provider.example",
                request_type="openai.chat",
                method="POST",
                endpoint="/v1/chat/completions",
                http_status=200,
                success=True,
            )

    with sqlite3.connect(path) as db:
        sessions = db.execute("SELECT session_hash, request_count FROM sessions").fetchall()
    assert len(sessions) == 1
    assert sessions[0][1] == 2
    assert sessions[0][0] != "raw-customer-session"


def test_configure_rejects_an_unwritable_database_before_capture(tmp_path) -> None:
    with pytest.raises(sqlite3.OperationalError):
        configure(database_path=tmp_path)


def test_runtime_database_failure_is_reported(tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path)

    def fail(*_args, **_kwargs) -> None:
        raise sqlite3.OperationalError("forced write failure")

    monkeypatch.setattr("impala_scope.capture.Store.write", fail)
    assert not capture(
        request={"model": "x", "messages": []},
        response={"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        headers=None,
        started_at_ms=1000,
        completed_at_ms=1010,
        host="provider.example",
        request_type="openai.chat",
        method="POST",
        endpoint="/v1/chat/completions",
        http_status=200,
        success=True,
    )
    error = capsys.readouterr().err
    assert "WARNING: analytics write failed" in error
    assert "forced write failure" in error


def _capture_one() -> bool:
    return capture(
        request={"model": "x", "messages": []},
        response={"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        headers=None,
        started_at_ms=1000,
        completed_at_ms=1010,
        host="provider.example",
        request_type="openai.chat",
        method="POST",
        endpoint="/v1/chat/completions",
        http_status=200,
        success=True,
    )


def test_database_path_is_stable_after_chdir(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    configure(database_path="profile.db")
    monkeypatch.chdir(second)
    assert _capture_one()
    assert (first / "profile.db").exists()
    assert not (second / "profile.db").exists()


def test_hashes_are_keyed_per_database(tmp_path) -> None:
    hashes = []
    for name in ("a.db", "b.db"):
        path = tmp_path / name
        configure(database_path=path)
        with record_session("guessable-session"):
            assert _capture_one()
        with sqlite3.connect(path) as db:
            hashes.append(db.execute("SELECT session_hash FROM requests").fetchone()[0])
        assert (tmp_path / f"{name}.key").stat().st_mode & 0o077 == 0
    assert hashes[0] != hashes[1]


def test_machine_override_changes_stored_machine_hash(tmp_path, monkeypatch) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path)
    for machine in ("worker-a", "worker-b"):
        monkeypatch.setenv("IMPALA_SCOPE_MACHINE_ID", machine)
        assert _capture_one()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(DISTINCT machine_hash) FROM requests").fetchone()[0] == 2


def test_default_database_key_is_created_lazily_and_reused(tmp_path, monkeypatch) -> None:
    path = tmp_path / "trace.db"
    monkeypatch.setattr(config, "database_path", path)
    monkeypatch.setattr(config, "hash_key", None)
    first = RequestCollector().fingerprint()
    assert (tmp_path / "trace.db.key").stat().st_mode & 0o077 == 0
    config.hash_key = None
    assert RequestCollector().fingerprint() == first


def test_empty_database_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "profile.db"
    (tmp_path / "profile.db.key").write_text("")
    with pytest.raises(RuntimeError, match="invalid hash key"):
        configure(database_path=path)


def test_database_key_symlink_is_rejected_without_changing_target(tmp_path) -> None:
    path = tmp_path / "profile.db"
    target = tmp_path / "chosen-key"
    target.write_text("ab" * 32)
    target.chmod(0o644)
    (tmp_path / "profile.db.key").symlink_to(target)

    with pytest.raises(RuntimeError, match="invalid hash key"):
        configure(database_path=path)
    assert target.stat().st_mode & 0o777 == 0o644


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
def test_forked_child_restarts_the_analytics_worker(tmp_path) -> None:
    path = tmp_path / "profile.db"
    configure(database_path=path)
    pid = os.fork()
    if pid == 0:  # pragma: no cover - assertions happen in the parent
        accepted = capture_later(
            request={"model": "x", "messages": []},
            response={"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            headers=None,
            started_at_ms=1000,
            completed_at_ms=1010,
            host="provider.example",
            request_type="openai.chat",
            method="POST",
            endpoint="/v1/chat/completions",
            http_status=200,
            success=True,
        )
        complete = flush()
        os._exit(0 if accepted and complete else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM requests").fetchone()[0] == 1
