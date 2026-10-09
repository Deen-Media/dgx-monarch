"""CPU-only coverage for the temporary same-UID process inspector."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "src/dgx_monarch/cli/process_inspector.py"
SPEC = importlib.util.spec_from_file_location("process_inspector", SCRIPT)
assert SPEC and SPEC.loader
inspector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspector)


def _boot_id(root: Path) -> None:
    location = root / "sys/kernel/random"
    location.mkdir(parents=True)
    (location / "boot_id").write_text("12345678-1234-1234-1234-123456789abc\n")


def _proc(root: Path, pid: int, command: bytes, environment: bytes = b"", starttime: int = 7) -> Path:
    process = root / str(pid)
    process.mkdir()
    (process / "cmdline").write_bytes(command)
    (process / "environ").write_bytes(environment)
    (process / "stat").write_text("x (p) S " + "0 " * 18 + f"{starttime} " + "0 " * 20)
    (process / "status").write_text(f"Name:\tpython\nUid:\t{os.geteuid()}\t{os.geteuid()}\t{os.geteuid()}\t{os.geteuid()}\n")
    return process


def _report(root: Path) -> dict[str, object]:
    _boot_id(root)
    return inspector.scan(os.geteuid(), root, source_hash="frozen-source")


def test_scan_reports_exact_modules_or_marker_pair_without_values(tmp_path):
    private_value = "not-for-output"
    _proc(
        tmp_path,
        11,
        f"python\0-m\0dgx_monarch.cli.worker_loop\0--address\0tcp://192.0.2.1:26600/{private_value}\0".encode(),
    )
    _proc(tmp_path, 12, b"python\0-m\0monarch._src.actor.bootstrap_main\0")
    _proc(
        tmp_path,
        13,
        b"python\0worker\0",
        f"HYPERACTOR_MESH_BOOTSTRAP_MODE={private_value}\0DGXM_PYTHONPATH={private_value}\0".encode(),
    )
    report = _report(tmp_path)
    assert report["status"] == "ok"
    assert report["workers"] == [
        {
            "pid": 11,
            "starttime": 7,
            "address_hashes": [hashlib.sha256(f"tcp://192.0.2.1:26600/{private_value}".encode()).hexdigest()],
        }
    ]
    assert report["actors"] == [{"pid": 12, "starttime": 7}, {"pid": 13, "starttime": 7}]
    encoded = json.dumps(report)
    assert private_value not in encoded
    assert "192.0.2.1" not in encoded
    assert "--address" not in encoded


@pytest.mark.parametrize("kind", ["environment", "environment_shape", "cmdline", "stat", "oversize"])
def test_scan_marks_unreadable_or_malformed_same_uid_process_unknown(tmp_path, monkeypatch, kind):
    process = _proc(tmp_path, 11, b"python\0-m\0dgx_monarch.cli.worker_loop\0")
    _boot_id(tmp_path)
    if kind == "environment":
        monkeypatch.setattr(inspector, "_read", lambda path: (_ for _ in ()).throw(PermissionError()) if path.name == "environ" else path.read_bytes())
    elif kind == "environment_shape":
        (process / "environ").write_bytes(b"DGXM_PYTHONPATH=missing-null")
    elif kind == "cmdline":
        (process / "cmdline").write_bytes(b"missing-terminator")
    elif kind == "stat":
        (process / "stat").write_bytes(b"bad")
    else:
        monkeypatch.setattr(inspector, "MAX_READ", 2)
    report = inspector.scan(os.geteuid(), tmp_path)
    assert report["status"] == "unknown"
    assert report["unknown_count"] >= 1
    expected_workers = [{"pid": 11, "starttime": 7, "address_hashes": []}] if kind.startswith("environment") else []
    assert report["workers"] == expected_workers


def test_scan_marks_process_id_reuse_unknown(tmp_path, monkeypatch):
    process = _proc(tmp_path, 11, b"python\0-m\0dgx_monarch.cli.worker_loop\0", starttime=7)
    _boot_id(tmp_path)
    real_read = inspector._read
    stat_reads = iter([b"x (p) S " + b"0 " * 18 + b"7 " + b"0 " * 20, b"x (p) S " + b"0 " * 18 + b"8 " + b"0 " * 20])

    def read(path):
        return next(stat_reads) if path == process / "stat" else real_read(path)

    monkeypatch.setattr(inspector, "_read", read)
    report = inspector.scan(os.geteuid(), tmp_path)
    assert report["status"] == "unknown"
    assert report["workers"] == []


def test_scan_includes_any_target_status_uid_and_rejects_zero_starttime(tmp_path):
    process = _proc(tmp_path, 11, b"python\0-m\0dgx_monarch.cli.worker_loop\0", starttime=0)
    _proc(tmp_path, 12, b"python\0-m\0dgx_monarch.cli.worker_loop\0")
    (tmp_path / "12/status").write_text(f"Uid:\t{os.geteuid()}\t0\t{os.geteuid()}\t{os.geteuid()}\n")
    _boot_id(tmp_path)
    report = inspector.scan(os.geteuid(), tmp_path)
    assert report["status"] == "unknown"
    assert report["workers"] == [{"pid": 12, "starttime": 7, "address_hashes": []}]
    assert process.exists()


def test_scan_marks_truncated_process_enumeration_unknown(tmp_path, monkeypatch):
    _proc(tmp_path, 11, b"python\0-m\0dgx_monarch.cli.worker_loop\0")
    _proc(tmp_path, 12, b"python\0-m\0dgx_monarch.cli.worker_loop\0")
    _boot_id(tmp_path)
    monkeypatch.setattr(inspector, "MAX_PIDS", 1)
    report = inspector.scan(os.geteuid(), tmp_path)
    assert report["status"] == "unknown"
    assert report["unknown_count"] >= 1


def test_scan_bounds_the_json_process_list(tmp_path, monkeypatch):
    _proc(tmp_path, 11, b"python\0-m\0dgx_monarch.cli.worker_loop\0")
    _proc(tmp_path, 12, b"python\0-m\0dgx_monarch.cli.worker_loop\0")
    _boot_id(tmp_path)
    monkeypatch.setattr(inspector, "MAX_REPORT_PROCESSES", 1)
    report = inspector.scan(os.geteuid(), tmp_path)
    assert report["status"] == "unknown"
    assert len(report["workers"]) == 1


def test_scan_deadline_returns_unknown_without_unbounded_enumeration(tmp_path):
    _proc(tmp_path, 11, b"python\0-m\0dgx_monarch.cli.worker_loop\0")
    _boot_id(tmp_path)
    report = inspector.scan(os.geteuid(), tmp_path, deadline=1.0, clock=lambda: 1.0)
    assert report["status"] == "unknown"
    assert report["workers"] == []


def test_empty_cmdline_is_valid_and_unreadable_environment_keeps_exact_module(tmp_path, monkeypatch):
    _proc(tmp_path, 11, b"")
    worker = _proc(tmp_path, 12, b"python\0-m\0dgx_monarch.cli.worker_loop\0")
    _boot_id(tmp_path)
    real_read = inspector._read
    monkeypatch.setattr(
        inspector,
        "_read",
        lambda path: (_ for _ in ()).throw(PermissionError()) if path == worker / "environ" else real_read(path),
    )
    report = inspector.scan(os.geteuid(), tmp_path)
    assert report["status"] == "unknown"
    assert report["workers"] == [{"pid": 12, "starttime": 7, "address_hashes": []}]


def test_env_only_actor_bash_is_detected_and_denied_bash_environment_is_unknown(tmp_path, monkeypatch):
    unrelated = _proc(
        tmp_path,
        11,
        b"bash\0--interactive\0",
        b"HYPERACTOR_MESH_BOOTSTRAP_MODE=enabled\0DGXM_PYTHONPATH=owned\0",
    )
    _boot_id(tmp_path)
    detected = inspector.scan(os.geteuid(), tmp_path)
    assert detected["status"] == "ok"
    assert detected["actors"] == [{"pid": 11, "starttime": 7}]
    real_read = inspector._read
    monkeypatch.setattr(
        inspector,
        "_read",
        lambda path: (_ for _ in ()).throw(PermissionError()) if path == unrelated / "environ" else real_read(path),
    )
    report = inspector.scan(os.geteuid(), tmp_path)
    assert report["status"] == "unknown"
    assert report["workers"] == []
    assert report["actors"] == []


def test_source_hash_rejects_non_root_owned_running_source(tmp_path, monkeypatch):
    source = tmp_path / "server.py"
    source.write_text("pass\n")
    monkeypatch.setattr(inspector, "__file__", str(source))
    with pytest.raises(RuntimeError, match="writable by the client"):
        inspector._source_hash()


def test_address_hashes_are_sorted_unique_and_bounded():
    values = [f"tcp://192.0.2.1:{port}" for port in range(1, 20)]
    argv = tuple(item for value in [*values, values[0]] for item in ("--address", value))
    expected = sorted({hashlib.sha256(value.encode()).hexdigest() for value in values})[:16]
    assert inspector._address_hashes(argv) == (expected, True)


def test_many_worker_addresses_mark_report_unknown_without_dropping_the_worker(tmp_path):
    addresses = [f"tcp://192.0.2.1:{port}" for port in range(1, 20)]
    command = b"python\0-m\0dgx_monarch.cli.worker_loop\0" + b"".join(
        b"--address\0" + address.encode() + b"\0" for address in addresses
    )
    _proc(tmp_path, 11, command)
    _boot_id(tmp_path)
    report = inspector.scan(os.geteuid(), tmp_path)
    assert report["status"] == "unknown"
    assert report["workers"][0]["pid"] == 11
    assert len(report["workers"][0]["address_hashes"]) == 16


def _start_server(path: Path, proc: Path, peer_uid):
    ready = threading.Event()
    thread = threading.Thread(
        target=inspector._serve_socket,
        args=(os.geteuid(), path, 0.4, "frozen-source"),
        kwargs={"proc": proc, "peer_uid": peer_uid, "ready": ready.set},
        daemon=True,
    )
    thread.start()
    assert ready.wait(1)
    info = path.stat()
    assert info.st_uid == os.geteuid()
    assert info.st_mode & 0o777 == 0o600
    return thread


def _request(path: Path, payload: bytes) -> bytes:
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(1)
        deadline = time.monotonic() + 1
        while True:
            try:
                client.connect(str(path))
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.005)
        client.sendall(payload)
        client.shutdown(socket.SHUT_WR)
        try:
            return client.recv(inspector.MAX_REPLY)
        except ConnectionResetError:
            return b""


def test_socket_requires_peer_uid_and_eof_terminated_token_then_cleans_up(tmp_path):
    _boot_id(tmp_path)
    path = tmp_path / "inspect.sock"
    rejected = _start_server(path, tmp_path, lambda connection: os.geteuid() + 1)
    assert _request(path, inspector.TOKEN) == b""
    rejected.join(1)
    assert not rejected.is_alive()
    assert not path.exists()

    accepted = _start_server(path, tmp_path, lambda connection: os.geteuid())
    response = json.loads(_request(path, inspector.TOKEN).decode())
    second_response = json.loads(_request(path, inspector.TOKEN).decode())
    assert response["uid"] == os.geteuid()
    assert second_response["uid"] == os.geteuid()
    assert response["source_sha256"] == "frozen-source"
    assert response["status"] == "ok"
    accepted.join(1)
    assert not accepted.is_alive()
    assert not path.exists()


def test_socket_rejects_missing_eof_and_expires_without_root_skip(tmp_path):
    _boot_id(tmp_path)
    path = tmp_path / "inspect.sock"
    thread = _start_server(path, tmp_path, lambda connection: os.geteuid())
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(1)
        client.connect(str(path))
        client.sendall(inspector.TOKEN)
        assert client.recv(1) == b""
    thread.join(1)
    assert not thread.is_alive()
    assert not path.exists()


def test_cleanup_socket_does_not_remove_a_replaced_inode(tmp_path):
    path = tmp_path / "inspect.sock"
    first = socket.socket(socket.AF_UNIX)
    first.bind(str(path))
    expected_info = path.stat()
    expected = SimpleNamespace(st_dev=expected_info.st_dev, st_ino=expected_info.st_ino + 1)
    first.close()
    path.unlink()
    replacement = socket.socket(socket.AF_UNIX)
    replacement.bind(str(path))
    try:
        inspector._cleanup_socket(path, expected)
        assert path.exists()
    finally:
        replacement.close()
        path.unlink()


def test_main_rejects_non_root_without_starting_a_socket(monkeypatch):
    monkeypatch.setenv("SUDO_UID", str(os.geteuid() or 1))
    monkeypatch.setattr(inspector.os, "geteuid", lambda: 1)
    with pytest.raises(SystemExit, match="2"):
        inspector.main()


def test_environment_actor_positive_survives_unreadable_command_line(tmp_path, monkeypatch):
    _proc(tmp_path, 11, b"bash\0", b"HYPERACTOR_MESH_BOOTSTRAP_MODE=x\0DGXM_PYTHONPATH=x\0")
    _boot_id(tmp_path)
    original = inspector._read
    def read(path):
        if path.name == "cmdline":
            raise PermissionError("protected")
        return original(path)
    monkeypatch.setattr(inspector, "_read", read)
    report = inspector.scan(os.geteuid(), tmp_path)
    assert report["status"] == "unknown"
    assert report["actors"] == [{"pid": 11, "starttime": 7}]


def test_kernel_names_need_not_be_ascii_to_read_identity(tmp_path):
    process = _proc(tmp_path, 11, b"bash\0")
    (process / "stat").write_bytes(b"11 (name-\xff)) S " + b"0 " * 18 + b"7 " + b"0 " * 20)
    (process / "status").write_bytes(
        b"Name:\tname-\xff\nUid:\t" + (f"{os.geteuid()} " * 4).encode() + b"\n"
    )
    assert _report(tmp_path)["status"] == "ok"


def test_non_utf8_arguments_preserve_exact_module_and_environment_markers(tmp_path):
    _proc(tmp_path, 11, b"shell\0\xff\0")
    _proc(tmp_path, 12, b"python\0-m\0monarch._src.actor.bootstrap_main\0\xff\0")
    _proc(tmp_path, 13, b"shell\0\xff\0", b"HYPERACTOR_MESH_BOOTSTRAP_MODE=x\0DGXM_PYTHONPATH=x\0")
    report = _report(tmp_path)
    assert report["status"] == "ok"
    assert report["actors"] == [{"pid": 12, "starttime": 7}, {"pid": 13, "starttime": 7}]
    assert report["workers"] == []


def test_slow_connection_does_not_block_another_authenticated_request(tmp_path):
    _boot_id(tmp_path)
    path = tmp_path / "inspect.sock"
    thread = _start_server(path, tmp_path, lambda connection: os.geteuid())
    with socket.socket(socket.AF_UNIX) as slow:
        slow.settimeout(1)
        slow.connect(str(path))
        slow.sendall(inspector.TOKEN)  # Hold back EOF to keep this request open.
        reply = json.loads(_request(path, inspector.TOKEN))
        assert reply["status"] == "ok"
        assert slow.recv(1) == b""
    thread.join(1)
    assert not thread.is_alive()
    assert not path.exists()


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_foreground_signal_removes_only_its_socket_and_restores_handlers(stop_signal):
    child_code = """
import importlib.util, os, pathlib, signal, sys
spec = importlib.util.spec_from_file_location("inspector", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._root_directory = lambda: pathlib.Path(sys.argv[2])
module._source_hash = lambda: "a" * 64
def no_scan(*args, **kwargs):
    raise AssertionError("No process inventory is permitted in this test")
module.scan = no_scan
signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
previous = {value: signal.getsignal(value) for value in signals}
try:
    module.serve(os.getuid(), 20)
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("Server expired instead of receiving the signal")
assert all(signal.getsignal(value) == previous[value] for value in signals)
print("handlers restored", flush=True)
"""
    # Keep the Unix socket path below its length limit, independent of pytest IDs.
    with tempfile.TemporaryDirectory(prefix="dgxm-signal-") as directory:
        sentinel = Path(directory) / "keep"
        sentinel.write_text("unrelated")
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", child_code, str(SCRIPT), directory],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            assert process.stdout is not None
            ready, _, _ = select.select([process.stdout], [], [], 10)
            assert ready, "Inspector did not become ready"
            line = process.stdout.readline()
            assert line.startswith("dgxm process inspector ready:"), line
            path = Path(directory) / f"{os.getuid()}.sock"
            assert path.exists()
            process.send_signal(stop_signal)
            output, errors = process.communicate(timeout=10)
            assert process.returncode == 0, errors
            assert output == "handlers restored\n"
            assert not os.path.lexists(path)
            assert sentinel.read_text() == "unrelated"
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
