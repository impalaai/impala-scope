"""CLI for wrapping a command or running a standalone inference proxy."""

import argparse
import contextlib
import os
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from impala_scope.capture import configure, flush, warn
from impala_scope.proxy import ca_path, start

DEFAULT_DB = "./trace.db"
DEFAULT_CONFDIR = "~/.impala-scope/state"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="impala-scope",
        description=(
            "Profile inference traffic as normalized SQLite analytics without storing payloads.\n\n"
            "Default exec-wrap usage: impala-scope -- <command...>\n"
            "With no command, impala-scope runs as a standalone proxy."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"SQLite analytics database; existing files are continued in place (default: {DEFAULT_DB})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Standalone listen host (default: 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Listen port (0 = ephemeral in exec-wrap mode, 8080 in standalone mode)",
    )
    parser.add_argument(
        "--upstream-hosts",
        default="",
        help="Comma-separated capture and TLS-interception allowlist (default: any inference-shaped request)",
    )
    parser.add_argument(
        "--confdir",
        default=DEFAULT_CONFDIR,
        help=f"Local proxy certificate directory (default: {DEFAULT_CONFDIR})",
    )
    parser.add_argument("--session-id", help="Force one logical session for this invocation")
    subcommands = parser.add_subparsers(dest="subcommand")
    ca = subcommands.add_parser("ca-path", help="Print the local proxy CA certificate path")
    ca.add_argument("--confdir", default=DEFAULT_CONFDIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = []
    if "--" in values:
        separator = values.index("--")
        command = values[separator + 1 :]
        values = values[:separator]

    parser = build_parser()
    args = parser.parse_args(values)
    if args.subcommand == "ca-path":
        return _print_ca_path(Path(args.confdir).expanduser())
    database = Path(args.db).expanduser().resolve()
    args.db = str(database)
    args.db_existed = database.exists()
    state = "continuing existing database" if args.db_existed else "new database"
    print(f"impala-scope: analytics database: {args.db} ({state})", file=sys.stderr)
    if command:
        return _run_command(args, command)
    return _run_standalone(args)


def _run_command(args: argparse.Namespace, command: list[str]) -> int:
    port = args.port or _free_port()
    try:
        proxy, certificate = _start(args, host="127.0.0.1", port=port)
    except Exception as exc:
        print(f"impala-scope: failed to start local proxy: {exc}", file=sys.stderr)
        return 1

    session = "(forced)" if args.session_id else "(per-request, derived from traffic)"
    print(f"impala-scope: session_id={session}", file=sys.stderr)
    print(f"impala-scope: local proxy on 127.0.0.1:{port}", file=sys.stderr)
    try:
        rc = _run_child(command, _child_env(port, certificate), proxy_thread=proxy[1])
    except FileNotFoundError:
        print(f"impala-scope: command not found: {command[0]}", file=sys.stderr)
        rc = 127
    finally:
        _stop(*proxy)
    print(f"\nimpala-scope: child exited {rc}; analytics saved to {args.db}", file=sys.stderr)
    return rc


def _run_standalone(args: argparse.Namespace) -> int:
    port = args.port or 8080
    proxy: tuple[Any, Any] | None = None
    try:
        proxy, certificate = _start(args, host=args.host, port=port)
        master, thread = proxy
        print(f"impala-scope: listening on {args.host}:{port}", file=sys.stderr)
        print(f"impala-scope: local trust file: {certificate}", file=sys.stderr)
        print("impala-scope: point clients at this proxy via:", file=sys.stderr)
        print(f"    export HTTPS_PROXY=http://{args.host}:{port}", file=sys.stderr)
        print(f"    export SSL_CERT_FILE={certificate}\n", file=sys.stderr)
        received = 0

        def stop(signum: int) -> None:
            nonlocal received
            received = signum
            if thread.is_alive():
                master.shutdown()

        with _signal_handlers(stop):
            while thread.is_alive():
                thread.join(timeout=0.2)
        errors = getattr(thread, "scope_errors", [])
        if not received and errors:
            print(f"impala-scope: proxy stopped unexpectedly: {errors[-1]}", file=sys.stderr)
            return 1
        return 128 + received if received else 0
    except Exception as exc:
        print(f"impala-scope: failed to start proxy: {exc}", file=sys.stderr)
        return 1
    finally:
        if proxy:
            _stop(*proxy)


def _start(args: argparse.Namespace, *, host: str, port: int) -> tuple[tuple[Any, Any], Path]:
    configure(database_path=args.db, session_id=args.session_id)
    confdir = Path(args.confdir).expanduser()
    confdir.mkdir(parents=True, exist_ok=True)
    allowed = [item.strip().lower() for item in args.upstream_hosts.split(",") if item.strip()] or None
    upstream_ca = _ca_bundle(confdir / "impala-scope-upstream-ca-bundle.pem")
    proxy = start(host=host, port=port, confdir=confdir, allowed_hosts=allowed, upstream_ca=upstream_ca)
    local_ca = ca_path(confdir)
    if not local_ca.exists() or not local_ca.stat().st_size:
        _stop(*proxy)
        raise RuntimeError(f"local trust file was not generated at {local_ca}")
    return proxy, _combined_ca(local_ca)


def _stop(master: Any, thread: Any) -> None:
    try:
        if thread.is_alive():
            master.shutdown()
    except RuntimeError:
        pass
    finally:
        thread.join(timeout=10)
        flush()
    if thread.is_alive():
        warn("proxy thread did not stop within 10 seconds")


def _print_ca_path(confdir: Path) -> int:
    certificate = ca_path(confdir)
    if not certificate.exists():
        print(
            f"impala-scope: local trust file not found at {certificate}; run `impala-scope -- <command>` once",
            file=sys.stderr,
        )
        return 1
    print(_combined_ca(certificate))
    return 0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_child(command: list[str], env: dict[str, str], proxy_thread: Any | None = None) -> int:
    process: subprocess.Popen | None = None
    pending: list[int] = []
    signals = 0

    def relay(signum: int) -> None:
        nonlocal signals
        if process is None:
            pending.append(signum)
            return
        signals += 1
        if process.poll() is not None:
            return
        if signals > 1:
            _kill_process_tree(process)
        else:
            _signal_process(process, signum)

    with _signal_handlers(relay):
        process = subprocess.Popen(command, env=env, start_new_session=os.name != "nt")
        suffix = f" (+{len(command) - 1} args)" if len(command) > 1 else ""
        print(f"impala-scope: launching: {command[0]}{suffix}\n", file=sys.stderr)
        for signum in pending:
            relay(signum)
        while True:
            try:
                returncode = process.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if proxy_thread is not None and not proxy_thread.is_alive():
                    errors = getattr(proxy_thread, "scope_errors", [])
                    warn(f"local proxy stopped while child was running: {errors[-1] if errors else 'unknown error'}")
                    _signal_process(process, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        _kill_process_tree(process)
                        process.wait()
                    return 1
    return returncode if returncode >= 0 else 128 - returncode


def _signal_process(process: subprocess.Popen, signum: int) -> None:
    try:
        if os.name == "nt":  # pragma: no cover
            process.send_signal(signum)
        else:
            os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _kill_process_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":  # pragma: no cover
        process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@contextlib.contextmanager
def _signal_handlers(callback):
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    handled = (signal.SIGINT, signal.SIGTERM)
    previous = {signum: signal.getsignal(signum) for signum in handled}

    def handler(signum, _frame) -> None:
        callback(signum)

    try:
        for signum in handled:
            signal.signal(signum, handler)
        yield
    finally:
        for signum, old_handler in previous.items():
            signal.signal(signum, old_handler)


def _child_env(port: int, certificate: Path) -> dict[str, str]:
    env = os.environ.copy()
    proxy = f"http://127.0.0.1:{port}"
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        env[key] = proxy
    for key in (
        "SSL_CERT_FILE",
        "CODEX_CA_CERTIFICATE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "AWS_CA_BUNDLE",
    ):
        env[key] = str(certificate)
    env["NODE_EXTRA_CA_CERTS"] = str(certificate)
    env["NO_PROXY"] = ""
    env["no_proxy"] = ""
    return env


def _combined_ca(local_ca: Path) -> Path:
    return _ca_bundle(local_ca.parent / "impala-scope-ca-bundle.pem", local_ca)


def _ca_bundle(bundle: Path, *extra: Path) -> Path:
    sources: list[Path] = []
    default = ssl.get_default_verify_paths().cafile
    if default:
        sources.append(Path(default))
    try:
        import certifi

        sources.append(Path(certifi.where()))
    except ImportError:  # pragma: no cover
        pass
    for key in (
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "AWS_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
    ):
        if value := os.environ.get(key):
            source = Path(value).expanduser()
            if source.is_file():
                sources.append(source)
    content = bytearray()
    seen: set[Path] = set()
    for source in (*sources, *extra):
        resolved = source.resolve()
        if resolved in seen or resolved == bundle.resolve():
            continue
        seen.add(resolved)
        content.extend(source.read_bytes().rstrip() + b"\n")
    if not bundle.exists() or bundle.read_bytes() != content:
        bundle.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{bundle.name}.", dir=bundle.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, bundle)
        finally:
            temporary.unlink(missing_ok=True)
    return bundle


if __name__ == "__main__":
    raise SystemExit(main())
