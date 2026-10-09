"""Exact installation migration and recovery without live systemd operations."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

from dgx_monarch.cli import update_installation as install
from dgx_monarch.cli import update_installation_remote as remote
from dgx_monarch.cli.lifecycle_systemd import hardened_worker_unit, systemd_worker_unit
from dgx_monarch.config import ClusterConfig, HostConfig

TOKEN = "u-111111111111-2222222222222222"  # noqa: S105


@pytest.fixture
def layout(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setattr(Path, "home", lambda: home)
    base = home / ".local/share/dgx-monarch"
    unit = home / ".config/systemd/user/dgxm-worker.service"
    unit.parent.mkdir(parents=True)
    (base / "releases" / TOKEN / "site" / "dgx_monarch").mkdir(parents=True)
    unit.write_bytes(b"original unit\n")
    unit.chmod(0o600)
    external = home / "checkout/src"
    external.mkdir(parents=True)
    (external / "do-not-delete").write_text("original")
    live = base / "src"
    live.symlink_to(external)
    monkeypatch.setattr(remote, "unit_state", lambda _, **_kwargs: {"MainPID": "0", "ActiveState": "inactive", "SubState": "dead", "Job": ""})
    monkeypatch.setattr(remote, "reload", lambda: None)
    snapshot = {"unit": base64.b64encode(unit.read_bytes()).decode(), "live": remote.link_state(live)}
    request = {"operation": "activate", "token": TOKEN, "prior": snapshot, "unit": base64.b64encode(b"new unit\n").decode()}
    return base, unit, external, live, request


def test_external_source_restored_without_modifying_target(layout, capsys):
    base, unit, external, live, request = layout
    remote.main(request)
    assert capsys.readouterr().out.strip() == "ACTIVATED"
    assert unit.read_bytes() == b"new unit\n"
    assert live.resolve() == base / "releases" / TOKEN / "site"
    remote.main({**request, "operation": "compensate"})
    assert capsys.readouterr().out.strip() == "COMPENSATED"
    assert unit.read_bytes() == b"original unit\n"
    assert live.resolve() == external
    assert (external / "do-not-delete").read_text() == "original"


def test_directory_source_restored_after_partial_activation(layout, monkeypatch, capsys):
    base, unit, _, live, request = layout
    live.unlink()
    live.mkdir()
    (live / "prior").write_text("prior")
    request["prior"]["live"] = remote.link_state(live)
    original = remote.durable
    monkeypatch.setattr(remote, "durable", lambda path, data: (_ for _ in ()).throw(OSError("full")) if path == unit else original(path, data))
    remote.main(request)
    assert capsys.readouterr().out.strip() == "ACTIVATION_PROBE_PRIOR"
    assert (base / "backups" / TOKEN / "src/prior").read_text() == "prior"
    monkeypatch.setattr(remote, "durable", original)
    remote.main({**request, "operation": "compensate"})
    assert not live.is_symlink()
    assert (live / "prior").read_text() == "prior"


def test_active_worker_prevents_any_migration(layout, monkeypatch):
    base, unit, external, live, request = layout
    monkeypatch.setattr(remote, "unit_state", lambda _, **_kwargs: {"MainPID": "123", "ActiveState": "active", "SubState": "running", "Job": ""})
    with pytest.raises(ValueError, match="stop"):
        remote.main(request)
    assert live.resolve() == external and unit.read_bytes() == b"original unit\n"
    assert not (base / "transactions").exists()


def test_changed_unit_is_not_overwritten(layout, capsys):
    base, unit, external, live, request = layout
    unit.write_bytes(b"user changed unit")
    remote.main(request)
    assert capsys.readouterr().out.strip() == "ACTIVATION_PROBE_UNKNOWN"
    assert live.resolve() == external and unit.read_bytes() == b"user changed unit"
    assert not (base / "transactions").exists()


def test_recovery_refuses_unknown_unit(layout):
    _, unit, _, _, request = layout
    remote.main(request)
    unit.write_bytes(b"unknown change")
    with pytest.raises(ValueError, match="outside transaction"):
        remote.main({**request, "operation": "compensate"})
    assert unit.read_bytes() == b"unknown change"


def test_finalize_retains_external_source_and_recovery_journal(layout):
    base, unit, external, _, request = layout
    remote.main(request)
    remote.main({**request, "operation": "finalize"})
    assert unit.read_bytes() == b"new unit\n"
    assert (external / "do-not-delete").exists()
    assert (base / "transactions" / TOKEN / "installation.json").is_file()


def test_reservation_refuses_symlink_parent(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    base = home / ".local/share/dgx-monarch"
    base.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (base / "releases").symlink_to(outside)
    with pytest.raises(ValueError, match="ownership"):
        remote.main({"operation": "reserve", "token": TOKEN, "reservation": "a" * 32})
    assert not list(outside.iterdir())


def test_hardened_unit_recognition_requires_every_byte(monkeypatch):
    host = HostConfig(name="localhost", address="tcp://127.0.0.1:26600")
    config = ClusterConfig(hosts=[host], python_bin="~/venv/bin/python")
    monkeypatch.setattr(install.lifecycle, "_is_local", lambda _: True)
    snapshot = {"site": "/home/user/repo/src", "site_packages": "/home/user/venv/lib/python3.12/site-packages", "home": "/home/user", "user": "user", "live": {"kind": "absent", "target": ""}}
    unit = hardened_worker_unit(config, host, source=snapshot["site"], site_packages=snapshot["site_packages"], home=snapshot["home"], user=snapshot["user"])
    def set_unit(text):
        snapshot.update(unit=base64.b64encode(text.encode()).decode(), sha256=hashlib.sha256(text.encode()).hexdigest())
    set_unit(unit)
    assert install.recognized_unit(config, host, snapshot)
    set_unit(unit.replace("RestartSec=3", "RestartSec=8"))
    assert not install.recognized_unit(config, host, snapshot)
    set_unit("# dgxm-managed-unit-v1\n" + unit)
    assert not install.recognized_unit(config, host, snapshot)
    snapshot["site"] = "/home/user/.local/share/dgx-monarch/src"
    snapshot["live"] = {"kind": "symlink", "target": f"/home/user/.local/share/dgx-monarch/releases/{TOKEN}/site"}
    set_unit(systemd_worker_unit(config, host, managed_source=True, update_token=TOKEN, site_packages=snapshot["site_packages"]))
    assert install.recognized_unit(config, host, snapshot)


def test_cleanup_refuses_live_release_and_preserves_recovery(layout):
    base, _, external, _, request = layout
    remote.main(request)
    with pytest.raises(ValueError, match="active"):
        remote.main({"operation": "cleanup", "token": TOKEN})
    remote.main({**request, "operation": "compensate"})
    remote.main({"operation": "cleanup", "token": TOKEN})
    assert (base / "transactions" / TOKEN / "installation.json").exists()
    assert (external / "do-not-delete").exists()


@pytest.mark.parametrize("fenced", [False, True])
def test_legacy_rollback_start_records_real_generation_identity(layout, monkeypatch, tmp_path, fenced):
    import os
    import socket
    import subprocess
    import sys
    import time

    from dgx_monarch.cli.lifecycle_generation import _GENERATION_HELPER, generation_prestart_argv
    from dgx_monarch.cli.update_installation_start import start_restored
    from dgx_monarch.cli.worker_health import _proc_listener_target

    base, unit, _, _, request = layout
    old = b"original unit\n"
    if fenced:
        old += b"ExecStartPre=canonical generation invalidator\n"
        unit.write_bytes(old)
        request["prior"]["unit"] = base64.b64encode(old).decode()
    remote.main(request)
    remote.main({**request, "operation": "compensate"})
    home = Path.home()
    for path in (home / ".local", home / ".local/state", home / ".local/state/dgx-monarch"):
        path.mkdir(exist_ok=True)
        path.chmod(0o700)
    generation = {}
    exec(_GENERATION_HELPER.split("operation=sys.argv[1]", 1)[0], generation)  # noqa: S102
    socket_reservation = socket.socket()
    socket_reservation.bind(("127.0.0.1", 0))
    port = socket_reservation.getsockname()[1]
    socket_reservation.close()
    address = f"tcp://127.0.0.1:{port}"
    proc_table, endpoint = _proc_listener_target(address)
    ready = tmp_path / "ready"
    pattern = "dgxm-test-owned-prior-" + str(port)
    process = None

    real_run = subprocess.run

    def start_unit(argv, **_kwargs):
        nonlocal process
        assert argv == ["systemctl", "--user", "start", unit.name]
        if fenced:
            prestart = real_run(generation_prestart_argv(), env={**os.environ, "HOME": str(home)}, capture_output=True, text=True)
            assert prestart.returncode == 0, prestart.stdout
        process = subprocess.Popen([sys.executable, "-c", "import socket,time,pathlib,sys;s=socket.socket();s.bind(('127.0.0.1',int(sys.argv[1])));s.listen();pathlib.Path(sys.argv[2]).touch();time.sleep(20)", str(port), str(ready), pattern])
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", start_unit)

    def running_snapshot(*_args):
        assert process is not None
        return {**request["prior"], "pid": str(process.pid), "birth": generation["birth"](str(process.pid))}

    api = {"read": remote.read, "stopped": remote.stopped, "link_state": remote.link_state, "durable": remote.durable, "snapshot": running_snapshot}
    start_request = {"token": TOKEN, "prior": request["prior"], "python_bin": sys.executable, "address": address, "generation": TOKEN, "pattern": pattern, "proc_table": proc_table, "endpoint": endpoint}
    try:
        start_restored(start_request, api, generation)
        assert process is not None
        assert generation["marker_identity"](TOKEN) == (str(process.pid), generation["birth"](str(process.pid)))
        assert (base / "transactions" / TOKEN / "prior-start.json").exists()
        assert unit.read_bytes() == old
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
        if generation["fencefd"] >= 0:
            os.close(generation["fencefd"])


def test_reservation_collision_cleanup_preserves_existing_release(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    remote.main({"operation": "reserve", "token": TOKEN, "reservation": "a" * 32})
    with pytest.raises(ValueError, match="another attempt"):
        remote.main({"operation": "cleanup", "token": TOKEN, "reservation": "b" * 32})
    assert (home / ".local/share/dgx-monarch/releases" / TOKEN / "site").is_dir()


def test_local_release_copy_never_uses_ssh_or_rsync(tmp_path, monkeypatch):
    import subprocess

    from dgx_monarch.cli.update_release import ReleaseManager, ReleaseMetadata, ReleaseSlot

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(install.lifecycle, "_is_local", lambda _: True)
    host = HostConfig(name="localhost", address="tcp://127.0.0.1:26600")
    config = ClusterConfig(hosts=[host])
    root = tmp_path / "stage"
    (root / "site/dgx_monarch").mkdir(parents=True)
    (root / "site/dgx_monarch/__init__.py").write_text("test source")
    slot = ReleaseSlot(TOKEN, root, root / "site", ReleaseMetadata("1" * 40, "1.0", "0.6.0", "a" * 64, "b" * 64))
    def command(*_args, **_kwargs):
        raise AssertionError("local staging must not call SSH or rsync")
    manager = ReleaseManager(config, command_runner=command, host_runner=lambda *_: subprocess.CompletedProcess([], 0, "RESERVED", ""))
    monkeypatch.setattr(manager, "_host_release_matches", lambda *_: True)
    manager._stage_host(host, slot)
    target = home / ".local/share/dgx-monarch/releases" / TOKEN / "site/dgx_monarch/__init__.py"
    assert target.read_text() == "test source"


def test_stop_refuses_replaced_worker_before_service_command(layout, monkeypatch):
    import subprocess

    _, _, _, _, request = layout
    monkeypatch.setattr(remote, "snapshot", lambda *_: {**request["prior"], "pid": "replacement"})
    def unexpected(*_args, **_kwargs):
        raise AssertionError("replacement Worker must not be stopped")
    monkeypatch.setattr(subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="changed since preflight"):
        remote.main({"operation": "stop", "prior": request["prior"], "python_bin": "/test/python", "address": "tcp://127.0.0.1:26600"})


def test_owned_stop_uses_only_exact_systemd_unit(layout, monkeypatch):
    import subprocess

    _, unit, _, _, request = layout
    monkeypatch.setattr(remote, "snapshot", lambda *_: request["prior"])
    calls = []
    def command(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, "run", command)
    remote.main({"operation": "stop", "prior": request["prior"], "python_bin": "/test/python", "address": "tcp://127.0.0.1:26600"})
    assert calls == [["systemctl", "--user", "stop", unit.name]]


def test_compensation_restores_unit_after_daemon_reload_failure(layout, monkeypatch, capsys):
    _, unit, _, _, request = layout
    stale = False
    def state(_unit, *, allow_stale=False):
        if stale and not allow_stale:
            raise ValueError("unit overrides or stale manager state")
        return {"MainPID": "0", "ActiveState": "inactive", "SubState": "dead", "Job": ""}
    def failed_reload():
        nonlocal stale
        stale = True
        raise ValueError("unit reload failed")
    monkeypatch.setattr(remote, "unit_state", state)
    monkeypatch.setattr(remote, "reload", failed_reload)
    remote.main(request)
    assert capsys.readouterr().out.strip() == "ACTIVATION_PROBE_PRIOR"
    assert unit.read_bytes() == b"new unit\n"
    def restored_reload():
        nonlocal stale
        assert unit.read_bytes() == b"original unit\n"
        stale = False
    monkeypatch.setattr(remote, "reload", restored_reload)
    remote.main({**request, "operation": "compensate"})
    assert capsys.readouterr().out.strip() == "COMPENSATED"
    assert not stale


@pytest.mark.parametrize("reload_state,allow_stale,accepted", [("yes", True, True), ("yes", False, False), ("no", False, True), ("", True, False), ("unknown", True, False)])
def test_real_unit_state_accepts_stale_only_for_compensation(tmp_path, monkeypatch, reload_state, allow_stale, accepted):
    import subprocess
    unit = tmp_path / "dgxm-worker.service"
    fields = f"FragmentPath={unit}\nDropInPaths=\nNeedDaemonReload={reload_state}\nMainPID=0\nActiveState=inactive\nSubState=dead\nJob=\nControlGroup=\n"
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, fields, ""))
    if accepted:
        assert remote.unit_state(unit, allow_stale=allow_stale)["NeedDaemonReload"] == reload_state
    else:
        with pytest.raises(ValueError, match="stale manager"):
            remote.unit_state(unit, allow_stale=allow_stale)


def test_hardened_remote_legacy_home_specifier_is_recognized(monkeypatch):
    host = HostConfig(name="peer", address="tcp://192.0.2.10:26600")
    config = ClusterConfig(hosts=[host], python_bin="~/venv/bin/python")
    monkeypatch.setattr(install.lifecycle, "_is_local", lambda _: False)
    site = "/home/user/.local/share/dgx-monarch/src"
    dependencies = "/home/user/venv/lib/python3.12/site-packages"
    unit = hardened_worker_unit(config, host, source="%h/.local/share/dgx-monarch/src", site_packages=dependencies, home="/home/user", user="user") + "\n\n"
    row = {"unit": base64.b64encode(unit.encode()).decode(), "sha256": hashlib.sha256(unit.encode()).hexdigest(), "site": site, "site_packages": dependencies, "home": "/home/user", "user": "user", "live": {"kind": "directory", "target": ""}}
    assert install.recognized_unit(config, host, row)
    unit = unit.replace("KillMode=control-group", "KillMode=process")
    row.update(unit=base64.b64encode(unit.encode()).decode(), sha256=hashlib.sha256(unit.encode()).hexdigest())
    assert not install.recognized_unit(config, host, row)
