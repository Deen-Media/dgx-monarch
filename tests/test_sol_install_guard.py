"""No live process/GPU queries: fake /proc plus mocked nvidia-smi."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.cli.sol_install_guard import require_idle


def _proc(tmp_path, argv, env=b""):
    entry = tmp_path / "42424242"
    entry.mkdir()
    (entry / "stat").write_text("42424242 (python) S 1 2 3")
    (entry / "comm").write_text("python\n")
    (entry / "exe").symlink_to("/usr/bin/python3")
    (entry / "cmdline").write_bytes(argv)
    (entry / "environ").write_bytes(env)


def test_idle_worker_coordinator_is_allowed(tmp_path, monkeypatch):
    _proc(tmp_path, b"python\0-m\0dgx_monarch.cli.worker_loop\0")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""))
    require_idle(str(tmp_path))


@pytest.mark.parametrize("argv,env", [
    (b"python\0/ComfyUI/main.py\0", b""),
    (b"python\0-m\0monarch._src.actor.bootstrap_main\0", b""),
    (b"python\0", b"HYPERACTOR_MESH_BOOTSTRAP_MODE=proc\0"),
])
def test_active_clients_refuse_without_gpu_query(tmp_path, monkeypatch, argv, env):
    _proc(tmp_path, argv, env)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("client already active"))
    with pytest.raises(RuntimeError, match="active_comfy_or_actor"):
        require_idle(str(tmp_path))


def test_unreadable_process_evidence_refuses(tmp_path, monkeypatch):
    _proc(tmp_path, b"python\0")
    original = Path.open
    def read(path, *args, **kwargs):
        if path.name == "environ":
            raise PermissionError("private")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", read)
    with pytest.raises(RuntimeError, match="process_observation_unknown"):
        require_idle(str(tmp_path))


@pytest.mark.parametrize("code,output,reason", [(0, "123\n", "active"), (1, "", "unknown")])
def test_gpu_clients_and_unreadable_gpu_refuse(tmp_path, monkeypatch, code, output, reason):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=code, stdout=output))
    with pytest.raises(RuntimeError, match=reason):
        require_idle(str(tmp_path))


def test_dead_process_needs_no_argv_or_environment(tmp_path, monkeypatch):
    _proc(tmp_path, b"python\0")
    entry = tmp_path / "42424242"
    (entry / "stat").write_text("42424242 (python) Z 1 2 3")
    (entry / "cmdline").unlink()
    (entry / "environ").unlink()
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""))
    require_idle(str(tmp_path))


def test_noncandidate_process_needs_no_argv_or_environment(tmp_path, monkeypatch):
    _proc(tmp_path, b"sleep\0")
    entry = tmp_path / "42424242"
    (entry / "comm").write_text("sleep")
    (entry / "exe").unlink()
    (entry / "exe").symlink_to("/usr/bin/sleep")
    (entry / "cmdline").unlink()
    (entry / "environ").unlink()
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""))
    require_idle(str(tmp_path))


def test_oversized_candidate_metadata_is_unknown(tmp_path):
    _proc(tmp_path, b"x" * 65537)
    with pytest.raises(RuntimeError, match="process_observation_unknown"):
        require_idle(str(tmp_path))


def test_denied_exe_daemon_uses_bounded_signature_without_environment(tmp_path, monkeypatch):
    import os

    _proc(tmp_path, b"/usr/lib/systemd/systemd\0--user\0")
    entry = tmp_path / "42424242"
    (entry / "comm").write_text("systemd")
    (entry / "environ").unlink()
    def denied(path):
        raise PermissionError("non-dumpable daemon")
    monkeypatch.setattr(os, "readlink", denied)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""))
    require_idle(str(tmp_path))


def test_denied_python_exe_and_environment_remain_unknown(tmp_path, monkeypatch):
    import os

    _proc(tmp_path, b"/usr/bin/python3\0worker.py\0")
    def denied(path):
        raise PermissionError("non-dumpable interpreter")
    monkeypatch.setattr(os, "readlink", denied)
    original = Path.open
    def read(path, *args, **kwargs):
        if path.name == "environ":
            raise PermissionError("private")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", read)
    with pytest.raises(RuntimeError, match="process_observation_unknown"):
        require_idle(str(tmp_path))


@pytest.mark.parametrize("raw", [b"", b"/usr/lib/systemd/systemd", b"\0--user\0"])
def test_denied_exe_with_malformed_argv_remains_unknown(tmp_path, monkeypatch, raw):
    import os

    _proc(tmp_path, raw)
    (tmp_path / "42424242" / "comm").write_text("systemd")
    def denied(path):
        raise PermissionError("non-dumpable daemon")
    monkeypatch.setattr(os, "readlink", denied)
    with pytest.raises(RuntimeError, match="process_observation_unknown"):
        require_idle(str(tmp_path))
