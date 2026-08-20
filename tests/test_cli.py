import contextlib
import signal

from impala_scope import cli


class Master:
    stopped = False

    def shutdown(self) -> None:
        self.stopped = True


class Thread:
    joined = False

    def is_alive(self) -> bool:
        return not self.joined

    def join(self, timeout: int | None = None) -> None:
        self.joined = True


def test_exec_wrap_profiles_child_process(tmp_path, monkeypatch, capsys) -> None:
    master = Master()
    thread = Thread()
    certificate = tmp_path / "mitmproxy-ca-cert.pem"
    observed: dict[str, object] = {}

    monkeypatch.setattr(cli, "_start", lambda *args, **kwargs: ((master, thread), certificate))
    monkeypatch.setattr(cli, "_free_port", lambda: 54321)

    def run_child(command, env, **_kwargs):
        observed.update(command=command, env=env)
        return 7

    monkeypatch.setattr(cli, "_run_child", run_child)
    rc = cli.main(["--db", str(tmp_path / "trace.db"), "--", "codex", "exec", "hello"])

    assert rc == 7
    assert observed["command"] == ["codex", "exec", "hello"]
    assert observed["env"]["HTTPS_PROXY"].startswith("http://127.0.0.1:")  # type: ignore[index]
    assert observed["env"]["SSL_CERT_FILE"] == str(certificate)  # type: ignore[index]
    assert master.stopped and thread.joined
    assert "impala-scope: local proxy on 127.0.0.1:54321" in capsys.readouterr().err


def test_ca_path_subcommand(tmp_path, capsys) -> None:
    certificate = tmp_path / "mitmproxy-ca-cert.pem"
    certificate.write_text("certificate")

    assert cli.main(["ca-path", "--confdir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "impala-scope-ca-bundle.pem")


def test_cli_keeps_recorder_controls() -> None:
    help_text = cli.build_parser().format_help()
    for flag in ("--db", "--host", "--port", "--upstream-hosts", "--confdir", "--session-id"):
        assert flag in help_text


def test_startup_failure_does_not_launch_child(tmp_path, monkeypatch, capsys) -> None:
    launched = False

    def fail(*_args, **_kwargs):
        raise RuntimeError("Address already in use")

    def run_child(*_args, **_kwargs):
        nonlocal launched
        launched = True
        return 0

    monkeypatch.setattr(cli, "_start", fail)
    monkeypatch.setattr(cli, "_run_child", run_child)

    rc = cli.main(["--db", str(tmp_path / "trace.db"), "--port", "54321", "--", "sleep", "30"])
    assert rc == 1
    assert not launched
    assert "Address already in use" in capsys.readouterr().err


def test_sigint_is_forwarded_to_the_child_process(monkeypatch, capsys) -> None:
    forwarded: list[tuple[object, int]] = []
    observed: dict[str, object] = {}

    class Process:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout=None):
            return -signal.SIGINT

        def kill(self):
            raise AssertionError("first signal should be forwarded, not converted to a kill")

    def popen(command, **kwargs):
        observed.update(command=command, **kwargs)
        return Process()

    @contextlib.contextmanager
    def interrupt(callback):
        callback(signal.SIGINT)
        yield

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli, "_signal_handlers", interrupt)
    monkeypatch.setattr(cli, "_signal_process", lambda process, signum: forwarded.append((process, signum)))

    assert cli._run_child(["sleep", "30"], {"PATH": "/bin"}) == 130
    assert observed["start_new_session"] is True
    assert forwarded and forwarded[0][1] == signal.SIGINT
    assert "impala-scope: launching: sleep (+1 args)" in capsys.readouterr().err


def test_child_environment_uses_combined_trust_and_cannot_bypass_proxy(tmp_path, monkeypatch) -> None:
    local_ca = tmp_path / "mitmproxy-ca-cert.pem"
    local_ca.write_text("MITM-CA")
    system_ca = tmp_path / "system.pem"
    system_ca.write_text("SYSTEM-CA")
    enterprise_ca = tmp_path / "enterprise.pem"
    enterprise_ca.write_text("ENTERPRISE-CA")
    node_ca = tmp_path / "node.pem"
    node_ca.write_text("NODE-CA")
    monkeypatch.setattr(cli.ssl, "get_default_verify_paths", lambda: type("Paths", (), {"cafile": str(system_ca)})())
    monkeypatch.setenv("CURL_CA_BUNDLE", str(enterprise_ca))
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", str(node_ca))
    monkeypatch.setenv("NO_PROXY", "*")
    bundle = cli._combined_ca(local_ca)
    env = cli._child_env(1234, bundle)

    assert "SYSTEM-CA" in bundle.read_text()
    assert "MITM-CA" in bundle.read_text()
    assert "ENTERPRISE-CA" in bundle.read_text()
    assert "NODE-CA" in bundle.read_text()
    assert env["SSL_CERT_FILE"] == str(bundle)
    assert env["NODE_EXTRA_CA_CERTS"] == str(bundle)
    assert env["NO_PROXY"] == env["no_proxy"] == ""


def test_cli_reports_resolved_existing_database_path(tmp_path, monkeypatch, capsys) -> None:
    master = Master()
    thread = Thread()
    certificate = tmp_path / "bundle.pem"
    database = tmp_path / "existing.db"
    database.touch()
    monkeypatch.setattr(cli, "_start", lambda *args, **kwargs: ((master, thread), certificate))
    monkeypatch.setattr(cli, "_run_child", lambda *args, **kwargs: 0)
    monkeypatch.setattr(cli, "_free_port", lambda: 54321)

    assert cli.main(["--db", str(database), "--", "true"]) == 0
    output = capsys.readouterr().err
    assert f"analytics database: {database.resolve()} (continuing existing database)" in output
    assert f"analytics saved to {database.resolve()}" in output


def test_cli_does_not_log_prompt_or_forced_session(tmp_path, monkeypatch, capsys) -> None:
    master = Master()
    thread = Thread()
    certificate = tmp_path / "bundle.pem"
    monkeypatch.setattr(cli, "_start", lambda *args, **kwargs: ((master, thread), certificate))
    monkeypatch.setattr(cli, "_run_child", lambda *args, **kwargs: 0)
    monkeypatch.setattr(cli, "_free_port", lambda: 54321)
    secret = "do-not-log-this-prompt"
    session = "do-not-log-this-session"
    assert cli.main(["--session-id", session, "--", "claude", "-p", secret]) == 0
    error = capsys.readouterr().err
    assert secret not in error
    assert session not in error


def test_second_signal_kills_the_process_group(monkeypatch) -> None:
    killed = []

    class Process:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout=None):
            return -signal.SIGKILL

    @contextlib.contextmanager
    def twice(callback):
        callback(signal.SIGINT)
        callback(signal.SIGINT)
        yield

    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(cli, "_signal_handlers", twice)
    monkeypatch.setattr(cli, "_signal_process", lambda *args: None)
    monkeypatch.setattr(cli, "_kill_process_tree", lambda process: killed.append(process.pid))
    assert cli._run_child(["sleep", "30"], {}) == 137
    assert killed == [123]


def test_proxy_death_terminates_child_and_returns_failure(monkeypatch, capsys) -> None:
    signalled = []

    class Process:
        pid = 123
        waits = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise cli.subprocess.TimeoutExpired("child", timeout)
            return -signal.SIGTERM

    class DeadThread:
        scope_errors = [RuntimeError("proxy crashed")]

        def is_alive(self):
            return False

    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(cli, "_signal_process", lambda process, signum: signalled.append(signum))
    assert cli._run_child(["sleep", "30"], {}, proxy_thread=DeadThread()) == 1
    assert signalled == [signal.SIGTERM]
    assert "proxy crashed" in capsys.readouterr().err
