"""Exact installation migration and recovery without live systemd operations."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import mmap
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dgx_monarch.cli import update_installation as install
from dgx_monarch.cli import update_installation_remote as remote
from dgx_monarch.cli import update_retire_remote as retire_remote
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
    # Retirement scans processes; it sees none but those a test starts (see _seen).
    (tmp_path / "proc").mkdir()
    monkeypatch.setattr(retire_remote, "RETIRE_PROC", tmp_path / "proc")
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


def test_directory_switch_durably_journals_before_moving_live_tree(layout, monkeypatch, capsys):
    """The record that compensation reads exists before the live source moves."""
    import os

    base, _, _, live, request = layout
    live.unlink()
    live.mkdir()
    request["prior"]["live"] = remote.link_state(live)
    record = base / "transactions" / TOKEN / "installation.json"
    real_rename = os.rename
    seen = []

    def rename(source, destination):
        if Path(source) == live:
            seen.append(record.is_file() and remote.read(record) != b"")
        return real_rename(source, destination)

    monkeypatch.setattr(remote.os, "rename", rename)
    remote.main(request)
    assert capsys.readouterr().out.strip() == "ACTIVATED"
    assert seen == [True]


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


@pytest.mark.parametrize(("fenced", "attempted"), [(False, True), (True, True), (False, False), (True, False)])
def test_legacy_rollback_start_records_real_generation_identity(layout, monkeypatch, tmp_path, fenced, attempted):
    """A started prior Worker is confirmed on every host, one the update never reached included.

    A reached host keeps the start in its journal; an unreached host has no
    journal, and the restore confirms it without making one.
    """
    import os
    import socket
    import subprocess
    import sys
    import time

    from dgx_monarch.cli.lifecycle_generation import _GENERATION_LIBRARY, generation_prestart_argv
    from dgx_monarch.cli.update_installation_start import start_restored
    from dgx_monarch.cli.worker_health import _proc_listener_target

    base, unit, _, _, request = layout
    old = b"original unit\n"
    if fenced:
        old += b"ExecStartPre=canonical generation invalidator\n"
        unit.write_bytes(old)
        request["prior"]["unit"] = base64.b64encode(old).decode()
    if attempted:
        remote.main(request)
        remote.main({**request, "operation": "compensate"})
    home = Path.home()
    for path in (home / ".local", home / ".local/state", home / ".local/state/dgx-monarch"):
        path.mkdir(exist_ok=True)
        path.chmod(0o700)
    generation = {}
    exec(_GENERATION_LIBRARY, generation)  # noqa: S102
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
    start_request = {"token": TOKEN, "prior": request["prior"], "python_bin": sys.executable, "address": address, "generation": TOKEN, "pattern": pattern, "proc_table": proc_table, "endpoint": endpoint, "attempted": attempted}
    try:
        start_restored(start_request, api, generation)
        assert process is not None
        assert generation["marker_identity"](TOKEN) == (str(process.pid), generation["birth"](str(process.pid)))
        if attempted:
            assert (base / "transactions" / TOKEN / "prior-start.json").exists()
        else:
            assert not (base / "transactions").exists()
        assert unit.read_bytes() == old
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
        if generation["fencefd"] >= 0:
            os.close(generation["fencefd"])


def _start_attempt(request, monkeypatch, *, attempted):
    """Run the prior start until its service command, which is refused here."""
    import subprocess

    from dgx_monarch.cli.update_installation_start import start_restored

    calls = []
    def refuse(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1)
    monkeypatch.setattr(subprocess, "run", refuse)
    api = {"read": remote.read, "stopped": remote.stopped, "link_state": remote.link_state, "durable": remote.durable}
    generation = {"matching": lambda _pattern: [], "remove_marker": lambda: True, "fencefd": -1}
    start_request = {"token": TOKEN, "prior": request["prior"], "pattern": "unused", "attempted": attempted}
    with pytest.raises((ValueError, OSError)) as raised:
        start_restored(start_request, api, generation)
    return calls, raised.value


def test_a_host_the_transaction_never_reached_starts_its_prior_installation(layout, monkeypatch):
    base, unit, _, _, request = layout
    calls, error = _start_attempt(request, monkeypatch, attempted=False)
    assert calls == [["systemctl", "--user", "start", unit.name]]
    assert "did not complete" in str(error)
    assert not (base / "transactions").exists()


def test_a_host_the_transaction_reached_needs_its_compensation_to_start(layout, monkeypatch):
    _, _, _, _, request = layout
    calls, _error = _start_attempt(request, monkeypatch, attempted=True)
    assert calls == []
    remote.main(request)
    calls, error = _start_attempt(request, monkeypatch, attempted=False)
    assert calls == [] and "reached" in str(error)


def test_a_failed_unit_write_leaves_nothing_that_blocks_the_rollback(layout, monkeypatch, capsys):
    import os

    _, unit, external, live, request = layout
    real_fsync = os.fsync
    failures = []

    def full_disk_once(fd):
        if not failures and os.readlink(f"/proc/self/fd/{fd}").startswith(str(unit.parent) + "/"):
            failures.append(fd)
            raise OSError(28, "No space left on device")
        return real_fsync(fd)

    monkeypatch.setattr(remote.os, "fsync", full_disk_once)
    remote.main(request)
    assert capsys.readouterr().out.strip() == "ACTIVATION_PROBE_PRIOR"
    assert failures and sorted(path.name for path in unit.parent.iterdir()) == [unit.name]
    remote.main({**request, "operation": "compensate"})
    assert capsys.readouterr().out.strip() == "COMPENSATED"
    assert unit.read_bytes() == b"original unit\n" and live.resolve() == external
    assert sorted(path.name for path in unit.parent.iterdir()) == [unit.name]
    # A later update writes the same unit again.
    remote.durable(unit, b"next unit\n")
    assert unit.read_bytes() == b"next unit\n"


def test_settle_reads_a_completed_activation_as_switched(layout, capsys):
    base, unit, _, live, request = layout
    remote.main(request)
    assert capsys.readouterr().out.strip() == "ACTIVATED"
    remote.main({**request, "operation": "settle"})
    assert capsys.readouterr().out.strip() == "ACTIVATED"
    assert unit.read_bytes() == b"new unit\n" and live.resolve() == base / "releases" / TOKEN / "site"


def test_a_temporary_left_by_an_earlier_release_is_not_in_the_way(layout, capsys):
    """Earlier releases wrote through a fixed ``.next`` name and could leave it behind."""
    _, unit, external, live, request = layout
    leftover = unit.with_name(unit.name + ".next")
    leftover.write_bytes(b"left by an earlier release\n")
    remote.main(request)
    assert capsys.readouterr().out.strip() == "ACTIVATED"
    remote.main({**request, "operation": "compensate"})
    assert capsys.readouterr().out.strip() == "COMPENSATED"
    assert unit.read_bytes() == b"original unit\n" and live.resolve() == external


def test_settle_finishes_the_reload_a_completed_switch_never_ran(layout, monkeypatch, capsys):
    """A switch cut off after its file changes leaves the manager stale; the readback reloads it."""
    _, _, _, _, request = layout
    remote.main(request)
    assert capsys.readouterr().out.strip() == "ACTIVATED"
    stale = True

    def state(_unit, *, allow_stale=False):
        if stale and not allow_stale:
            raise ValueError("unit overrides or stale manager state")
        return {"MainPID": "0", "ActiveState": "inactive", "SubState": "dead", "Job": ""}

    def reload():
        nonlocal stale
        stale = False

    monkeypatch.setattr(remote, "unit_state", state)
    monkeypatch.setattr(remote, "reload", reload)
    remote.main({**request, "operation": "settle"})
    assert capsys.readouterr().out.strip() == "ACTIVATED"
    assert not stale


def test_settle_reads_a_failed_activation_as_restorable(layout, monkeypatch, capsys):
    _, unit, _, _, request = layout
    original = remote.durable
    monkeypatch.setattr(remote, "durable", lambda path, data, mode=0o600: (_ for _ in ()).throw(OSError("full")) if path == unit else original(path, data, mode))
    remote.main(request)
    assert capsys.readouterr().out.strip() == "ACTIVATION_PROBE_PRIOR"
    monkeypatch.setattr(remote, "durable", original)
    remote.main({**request, "operation": "settle"})
    assert capsys.readouterr().out.strip() == "ACTIVATION_PROBE_PRIOR"


def test_settle_cancels_an_activation_that_never_began(layout, monkeypatch, capsys):
    _, unit, external, live, request = layout
    remote.main({**request, "operation": "settle"})
    assert capsys.readouterr().out.strip() == "ACTIVATION_PROBE_PRIOR"
    # A copy of the activation still queued on the lock finds the mark and changes nothing.
    with pytest.raises(ValueError, match="cancelled"):
        remote.main(request)
    assert unit.read_bytes() == b"original unit\n" and live.resolve() == external
    remote.main({**request, "operation": "compensate"})
    assert capsys.readouterr().out.strip() == "COMPENSATED"
    calls, error = _start_attempt(request, monkeypatch, attempted=True)
    assert calls == [["systemctl", "--user", "start", unit.name]] and "did not complete" in str(error)
    remote.main({"operation": "cleanup", "token": TOKEN})
    assert capsys.readouterr().out.strip() == "CLEANED"


def test_settle_reads_an_unexpected_state_as_unknown(layout, capsys):
    _, unit, _, _, request = layout
    unit.write_bytes(b"someone else's unit\n")
    remote.main({**request, "operation": "settle"})
    assert capsys.readouterr().out.strip() == "ACTIVATION_PROBE_UNKNOWN"
    with pytest.raises(ValueError, match="outside transaction"):
        remote.main({**request, "operation": "compensate"})


PRIOR_TOKEN = "u-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"  # noqa: S105


def _releases_with_a_prior_slot(layout):
    """The layout, but its live source is the prior release slot, as after an earlier update."""
    base, _, _, live, request = layout
    prior_site = base / "releases" / PRIOR_TOKEN / "site"
    (prior_site / "dgx_monarch").mkdir(parents=True)
    live.unlink()
    live.symlink_to(prior_site)
    request["prior"]["live"] = remote.link_state(live)
    return base, request


def _old_slot(base, name, *, journal=None, closed=False, payload=100):
    """An old slot; ``journal`` names the mark its own transaction holds, ``closed`` closes a setup's."""
    site = base / "releases" / name / "site"
    site.mkdir(parents=True)
    (site / "payload.bin").write_bytes(b"x" * payload)
    if journal is not None:
        folder = base / "transactions" / name
        folder.mkdir(parents=True)
        (folder / "installation.json").write_text("{}")
        if journal:
            (folder / journal).write_bytes(journal.upper().encode() + b"\n")
            (folder / journal).chmod(0o600)  # as the helper's own durable writes leave it
    if closed:
        setup = Path.home() / ".local/state/dgx-monarch/setup" / name
        setup.mkdir(parents=True)
        token, ordinal = name.rsplit("-", 1)
        (setup / "closed.json").write_text(json.dumps({"schema": 1, "token": token, "ordinal": int(ordinal)}))
        (setup / "closed.json").chmod(0o600)
    return site.parent


def _seen(*pids: int) -> None:
    """Let retirement's process scan see these processes, and no other process of this machine."""
    for pid in pids:
        (retire_remote.RETIRE_PROC / str(pid)).symlink_to(f"/proc/{pid}", target_is_directory=True)


def _retire(request, capsys) -> list[str]:
    capsys.readouterr()
    retire_remote.retire_installation({**request, "operation": "retire"})
    return capsys.readouterr().out.strip().splitlines()


def _finalized(request, capsys):
    remote.main(request)
    remote.main({**request, "operation": "finalize"})
    capsys.readouterr()


def test_retire_removes_only_slots_it_can_prove_dead(layout, capsys):
    base, request = _releases_with_a_prior_slot(layout)
    releases = base / "releases"
    finalized = _old_slot(base, "u-cccccccccccc-dddddddddddddddd", journal="finalized")
    compensated = _old_slot(base, "u-dddddddddddd-eeeeeeeeeeeeeeee", journal="compensated")
    closed_setup = _old_slot(base, "s-eeeeeeeeeeeeeeee-2", closed=True)
    open_journal = _old_slot(base, "u-333333333333-4444444444444444", journal="")
    staged = _old_slot(base, "u-222222222222-3333333333333333")  # reserved by another update, no journal
    (staged / ".reservation").write_text("0" * 32)
    open_setup = _old_slot(base, "s-ffffffffffffffff-1")
    (releases / "notes").mkdir()                      # not a slot name
    (releases / "u-ffffffffffff-0000000000000000").symlink_to(finalized)  # a symlink
    _finalized(request, capsys)
    lines = _retire(request, capsys)
    assert sorted(lines[:-1]) == sorted(f"REMOVED {slot.name} 100" for slot in (finalized, compensated, closed_setup))
    assert lines[-1] == "RETIRED 3 0 0"
    assert sorted(path.name for path in releases.iterdir()) == sorted([
        TOKEN, PRIOR_TOKEN, open_journal.name, staged.name, open_setup.name, "notes", "u-ffffffffffff-0000000000000000"])


def test_retire_keeps_a_slot_the_old_unit_named_and_a_folder_it_does_not_own(layout, capsys):
    """A guided-setup unit can name its slot directly; a folder failing the ownership check is never removed."""
    base, request = _releases_with_a_prior_slot(layout)
    setup_slot = _old_slot(base, "s-1111111111111111-1", closed=True)
    request["prior"]["site"] = str(setup_slot / "site")
    shared = _old_slot(base, "u-999999999999-8888888888888888", journal="finalized")
    shared.chmod(0o777)
    try:
        _finalized(request, capsys)
        assert _retire(request, capsys) == ["RETIRED 0 0 0"]
        assert setup_slot.is_dir() and shared.is_dir()
    finally:
        shared.chmod(0o700)


def test_retire_keeps_the_replaced_slot_whatever_spelling_its_old_link_used(layout, capsys, tmp_path):
    """An old link through another spelling of the home still names the slot a rollback needs."""
    base, _, _, live, request = layout
    prior = _old_slot(base, PRIOR_TOKEN, journal="finalized")
    alias = tmp_path / "alias-home"
    alias.symlink_to(Path.home(), target_is_directory=True)
    live.unlink()
    live.symlink_to(str(alias / ".local/share/dgx-monarch/releases" / PRIOR_TOKEN / "site"))
    request["prior"]["live"] = remote.link_state(live)
    _finalized(request, capsys)
    assert _retire(request, capsys) == ["RETIRED 0 0 0"]
    assert prior.is_dir()


def test_retire_refuses_unless_finalize_check_still_holds(layout, capsys):
    base, request = _releases_with_a_prior_slot(layout)
    other = _old_slot(base, "u-cccccccccccc-dddddddddddddddd", journal="finalized")
    remote.main(request)
    with pytest.raises((ValueError, OSError)):
        retire_remote.retire_installation({**request, "operation": "retire"})        # not finalized yet
    remote.main({**request, "operation": "finalize"})
    unit = Path.home() / ".config/systemd/user/dgxm-worker.service"
    finalized_unit = unit.read_bytes()
    unit.write_bytes(b'Environment="PYTHONPATH=%h/.local/share/dgx-monarch/releases/' + other.name.encode() + b'/site"\n')
    with pytest.raises(ValueError, match="not the finalized one"):
        retire_remote.retire_installation({**request, "operation": "retire"})        # the unit changed after finalize
    unit.write_bytes(finalized_unit)
    live = base / "src"
    live.unlink()
    live.symlink_to(other / "site")
    with pytest.raises(ValueError, match="not the finalized one"):
        retire_remote.retire_installation({**request, "operation": "retire"})        # the link moved
    assert other.is_dir()


def test_retire_refuses_while_the_running_worker_imports_from_elsewhere(layout, monkeypatch, capsys):
    base, request = _releases_with_a_prior_slot(layout)
    other = _old_slot(base, "u-cccccccccccc-dddddddddddddddd", journal="finalized")
    _finalized(request, capsys)
    monkeypatch.setattr(retire_remote, "running_source", lambda _unit: str(other / "site"))
    with pytest.raises(ValueError, match="another source"):
        retire_remote.retire_installation({**request, "operation": "retire"})
    assert other.is_dir()
    monkeypatch.setattr(retire_remote, "running_source", lambda _unit: str(base / "src"))
    assert _retire(request, capsys)[-1] == "RETIRED 1 0 0"


def test_a_stop_halfway_leaves_nothing_that_looks_like_a_slot(layout, monkeypatch, capsys):
    """Each slot is renamed out of the slot pattern before removal; the next run removes the leftover first."""
    base, request = _releases_with_a_prior_slot(layout)
    releases = base / "releases"
    first = _old_slot(base, "u-cccccccccccc-dddddddddddddddd", journal="finalized")
    second = _old_slot(base, "u-dddddddddddd-eeeeeeeeeeeeeeee", journal="finalized")
    _finalized(request, capsys)
    real_rmtree = retire_remote.shutil.rmtree

    def killed(path, *args, **kwargs):
        raise KeyboardInterrupt  # the helper is stopped in the middle of a removal

    monkeypatch.setattr(retire_remote.shutil, "rmtree", killed)
    with pytest.raises(KeyboardInterrupt):
        retire_remote.retire_installation({**request, "operation": "retire"})
    names = sorted(path.name for path in releases.iterdir())
    assert first.name not in names and retire_remote.RETIRING + first.name in names and second.name in names
    monkeypatch.setattr(retire_remote.shutil, "rmtree", real_rmtree)
    lines = _retire(request, capsys)
    assert lines == [f"REMOVED {first.name} 100", f"REMOVED {second.name} 100", "RETIRED 2 0 0"]
    assert sorted(path.name for path in releases.iterdir()) == sorted([PRIOR_TOKEN, TOKEN])


def test_a_leftover_that_fails_the_ownership_check_stays(layout, capsys):
    """A renamed leftover is cleared only under the same ownership check as a slot."""
    base, request = _releases_with_a_prior_slot(layout)
    leftover = base / "releases" / (retire_remote.RETIRING + "u-cccccccccccc-dddddddddddddddd")
    (leftover / "site").mkdir(parents=True)
    leftover.chmod(0o777)  # writable by anyone: not provably this user's alone
    try:
        _finalized(request, capsys)
        assert _retire(request, capsys) == ["RETIRED 0 0 0"]
        assert leftover.is_dir()
    finally:
        leftover.chmod(0o700)
    assert _retire(request, capsys) == ["REMOVED u-cccccccccccc-dddddddddddddddd 0", "RETIRED 1 0 0"]


def test_a_slot_that_cannot_be_removed_is_counted_and_the_rest_go_on(layout, monkeypatch, capsys):
    base, request = _releases_with_a_prior_slot(layout)
    stuck = _old_slot(base, "u-000000000000-0000000000000000", journal="finalized")
    later = _old_slot(base, "u-ffffffffffff-ffffffffffffffff", journal="finalized")
    locked = stuck / "site" / "read-only"
    locked.mkdir()
    (locked / "file").write_text("kept")
    locked.chmod(0o555)
    _finalized(request, capsys)
    real_rmtree = retire_remote.shutil.rmtree

    def refuses_one(path, *args, **kwargs):
        if path.name.endswith(stuck.name):
            raise PermissionError(13, "Permission denied")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(retire_remote.shutil, "rmtree", refuses_one)
    lines = _retire(request, capsys)
    assert lines == [f"REMOVED {later.name} 100", "RETIRED 1 1 0"]
    assert not later.exists()
    monkeypatch.setattr(retire_remote.shutil, "rmtree", real_rmtree)
    # The read-only folder inside does not stop the next run.
    assert _retire(request, capsys)[-1] == "RETIRED 1 0 0"


def test_retire_starts_no_removal_after_its_time_budget(layout, monkeypatch, capsys):
    base, request = _releases_with_a_prior_slot(layout)
    slots = [_old_slot(base, f"u-cccccccccccc-{index:016x}", journal="finalized") for index in range(3)]
    _finalized(request, capsys)
    monkeypatch.setattr(retire_remote, "RETIRE_BUDGET_S", -1.0)
    assert _retire(request, capsys) == ["RETIRED 0 0 3"]
    assert all(slot.is_dir() for slot in slots)


def test_retire_holds_back_a_slot_a_process_still_uses(layout, capsys):
    """A process working in a slot, or importing from it by PYTHONPATH, keeps it for the next update."""
    base, request = _releases_with_a_prior_slot(layout)
    in_folder = _old_slot(base, "u-cccccccccccc-dddddddddddddddd", journal="finalized")
    on_path = _old_slot(base, "s-eeeeeeeeeeeeeeee-2", closed=True)
    _finalized(request, capsys)
    sleepers = [
        subprocess.Popen(["sleep", "60"], cwd=in_folder / "site"),
        subprocess.Popen(["sleep", "60"], env={**os.environ, "PYTHONPATH": f"/nowhere:{on_path / 'site'}"}),
    ]
    _seen(*(sleeper.pid for sleeper in sleepers))
    try:
        assert _retire(request, capsys) == ["RETIRED 0 0 2"]
        assert in_folder.is_dir() and on_path.is_dir()
    finally:
        for sleeper in sleepers:
            sleeper.kill()
            sleeper.wait()
    assert _retire(request, capsys)[-1] == "RETIRED 2 0 0"


def test_retire_holds_back_a_slot_with_an_open_or_mapped_file(layout, capsys):
    base, request = _releases_with_a_prior_slot(layout)
    opened = _old_slot(base, "u-cccccccccccc-dddddddddddddddd", journal="finalized")
    mapped = _old_slot(base, "u-dddddddddddd-eeeeeeeeeeeeeeee", journal="finalized")
    _finalized(request, capsys)
    with open(opened / "site" / "payload.bin", "rb") as held:
        reader = subprocess.Popen(["sleep", "60"], stdin=held)  # the file is open only in the child
    with open(mapped / "site" / "payload.bin", "rb") as source:
        view = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)  # mapped here; the file is closed
    _seen(reader.pid, os.getpid())
    try:
        assert _retire(request, capsys) == ["RETIRED 0 0 2"]
    finally:
        reader.kill()
        reader.wait()
        view.close()
    assert _retire(request, capsys)[-1] == "RETIRED 2 0 0"


def test_retire_reads_the_running_workers_own_source_path(layout, monkeypatch, capsys):
    base, request = _releases_with_a_prior_slot(layout)
    other = _old_slot(base, "u-cccccccccccc-dddddddddddddddd", journal="finalized")
    _finalized(request, capsys)
    for source, allowed in ((other / "site", False), (base / "src", True)):
        worker = subprocess.Popen(["sleep", "60"], env={"PYTHONPATH": str(source), "PATH": os.defpath})
        _seen(worker.pid)
        monkeypatch.setattr(remote, "unit_state", lambda _unit, pid=worker.pid, **_kwargs: {"MainPID": str(pid)})
        try:
            if allowed:
                assert _retire(request, capsys)[-1] == "RETIRED 1 0 0"
            else:
                with pytest.raises(ValueError, match="another source"):
                    retire_remote.retire_installation({**request, "operation": "retire"})
                assert other.is_dir()
        finally:
            worker.kill()
            worker.wait()


@pytest.mark.parametrize("mark", ["finalized", "compensated"])
@pytest.mark.parametrize("content", [b"", b"FINALIZED", b"COMPENSATED", b"FINALIZED\nCOMPENSATED\n", b"done\n"])
def test_a_journal_mark_with_other_bytes_keeps_its_slot(layout, capsys, mark, content):
    """Only a mark holding exactly FINALIZED or COMPENSATED and a newline proves an update's slot dead."""
    base, request = _releases_with_a_prior_slot(layout)
    slot = _old_slot(base, "u-cccccccccccc-dddddddddddddddd", journal="")
    (base / "transactions" / slot.name / mark).write_bytes(content)
    (base / "transactions" / slot.name / mark).chmod(0o600)
    _finalized(request, capsys)
    assert _retire(request, capsys) == ["RETIRED 0 0 0"]
    assert slot.is_dir()


def test_a_closing_mark_that_is_not_this_setups_keeps_the_slot(layout, capsys):
    base, request = _releases_with_a_prior_slot(layout)
    setup = Path.home() / ".local/state/dgx-monarch/setup"
    slots = {name: _old_slot(base, name) for name in ("s-1111111111111111-1", "s-2222222222222222-1", "s-3333333333333333-2")}
    for name, content in (("s-1111111111111111-1", None), ("s-2222222222222222-1", "{}"),
                          ("s-3333333333333333-2", '{"schema": 1, "token": "s-3333333333333333", "ordinal": 1}')):
        (setup / name).mkdir(parents=True)
        if content is None:
            (setup / name / "closed.json").symlink_to(setup / "nowhere.json")
        else:
            (setup / name / "closed.json").write_text(content)
            (setup / name / "closed.json").chmod(0o600)
    _finalized(request, capsys)
    assert _retire(request, capsys) == ["RETIRED 0 0 0"]
    assert all(slot.is_dir() for slot in slots.values())


def test_retire_keeps_the_slot_an_old_link_named_without_its_site_folder(layout, capsys):
    base, _, _, live, request = layout
    prior = _old_slot(base, PRIOR_TOKEN, journal="finalized")
    live.unlink()
    live.symlink_to(prior)
    request["prior"]["live"] = remote.link_state(live)
    _finalized(request, capsys)
    assert _retire(request, capsys) == ["RETIRED 0 0 0"]
    assert prior.is_dir()


def test_retire_starts_no_removal_once_its_caller_stops_reading(layout, monkeypatch, capsys):
    base, request = _releases_with_a_prior_slot(layout)
    first = _old_slot(base, "u-cccccccccccc-dddddddddddddddd", journal="finalized")
    second = _old_slot(base, "u-dddddddddddd-eeeeeeeeeeeeeeee", journal="finalized")
    _finalized(request, capsys)

    class Gone(io.StringIO):
        def write(self, text):
            raise BrokenPipeError(32, "Broken pipe")  # the connection to the caller closed

    monkeypatch.setattr(sys, "stdout", Gone())
    retire_remote.retire_installation({**request, "operation": "retire"})
    assert not first.exists() and second.is_dir()


def test_retire_refuses_a_request_that_is_not_a_retire_of_an_update_slot(layout, capsys):
    _, request = _releases_with_a_prior_slot(layout)
    _finalized(request, capsys)
    for wrong in ({**request, "operation": "finalize"}, {**request, "operation": "retire", "token": "s-1111111111111111-1"}):
        with pytest.raises(ValueError, match="invalid retire request"):
            retire_remote.retire_installation(wrong)


def test_a_thread_that_exits_during_the_child_scan_is_skipped(tmp_path):
    process = tmp_path / "proc" / "4242"
    (process / "task" / "4242").mkdir(parents=True)
    (process / "task" / "4242" / "children").write_text("")
    (process / "task" / "4243").mkdir()  # the thread exited between the listing and the read
    assert remote.has_child_processes(process) is False
    (process / "task" / "4244").mkdir()
    (process / "task" / "4244" / "children").write_text("5000 ")
    assert remote.has_child_processes(process) is True
    with pytest.raises(FileNotFoundError):
        remote.has_child_processes(tmp_path / "proc" / "4999")


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


def _system_python_literals(module) -> int:
    """String constants that hold the path, alone or inside a longer string."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return sum(isinstance(node, ast.Constant) and isinstance(node.value, str)
               and "/usr/bin/python3" in node.value for node in ast.walk(tree))


def _system_python_renders(generation) -> dict[str, str]:
    """The seven system-Python commands, rendered at call time."""
    host = HostConfig(name="peer", address="tcp://192.0.2.10:26600")
    config = ClusterConfig(hosts=[host], python_bin="~/venv/bin/python")
    scripts: list[str] = []

    def run(_config, _host, script, _timeout):
        scripts.append(script)
        raise OSError("transport stand-in")

    assert install.start_prior_installation(config, host, TOKEN, {}, run, "", TOKEN) is install.Certainty.UNKNOWN
    return {
        "prestart": " ".join(generation.generation_prestart_argv()),
        "quarantine": " ".join(generation.generation_quarantine_argv(TOKEN, "pattern", "systemd", "dgxm-worker.service")),
        "invalidate command": generation.invalidate_generation_command_shell(),
        "invalidate shell": generation.invalidate_generation_shell(),
        "installation": install.installation_script({"operation": "snapshot"}),
        "prior start": scripts[0],
        "retire": install.retire_script(config, host, TOKEN, {}),
    }


def test_every_system_python_site_uses_the_one_constant():
    """The seven system-Python commands take their interpreter from SYSTEM_PYTHON."""
    from dgx_monarch.cli import lifecycle_generation as generation

    python = generation.SYSTEM_PYTHON
    assert python == "/usr/bin/python3"
    # The only string literal left is the constant itself; comments do not count.
    assert _system_python_literals(generation) == 1
    assert _system_python_literals(install) == 0
    assert generation.generation_prestart_argv()[0] == python
    assert generation.generation_quarantine_argv(TOKEN, "pattern", "systemd", "dgxm-worker.service")[0] == python
    assert generation.invalidate_generation_command_shell().startswith(python + " -I -S -B -c ")
    assert f"\n{python} -I -S -B -c " in generation.invalidate_generation_shell()
    assert f"\n{python} -I -S -B - <<'DGXM_INSTALLATION'\n" in install.installation_script({"operation": "snapshot"})
    host = HostConfig(name="peer", address="tcp://192.0.2.10:26600")
    config = ClusterConfig(hosts=[host], python_bin="~/venv/bin/python")
    scripts: list[str] = []

    def run(_config, _host, script, _timeout):
        scripts.append(script)
        raise OSError("transport stand-in")

    assert install.start_prior_installation(config, host, TOKEN, {}, run, "", TOKEN) is install.Certainty.UNKNOWN
    assert f"\n{python} -I -S -B - <<'DGXM_PRIOR_START'\n" in scripts[0]
    assert f"\n{python} -I -S -B - <<'DGXM_RETIRE'\n" in install.retire_script(config, host, TOKEN, {})


def test_every_system_python_site_reads_the_constant_when_it_renders(monkeypatch):
    """A site that spells the path inside a longer string renders the same
    bytes as one that reads the constant, so the rendered checks above cannot
    tell them apart. A stand-in value can: each of the seven commands must carry
    it and no copy of the real path."""
    from dgx_monarch.cli import lifecycle_generation as generation

    sentinel = "/opt/sentinel/python3"
    monkeypatch.setattr(generation, "SYSTEM_PYTHON", sentinel)
    monkeypatch.setattr(install, "SYSTEM_PYTHON", sentinel)
    rendered = _system_python_renders(generation)
    assert rendered["prestart"].startswith(sentinel + " -I -S -B -c ")
    assert rendered["quarantine"].startswith(sentinel + " -I -S -B -c ")
    assert rendered["invalidate command"].startswith(sentinel + " -I -S -B -c ")
    assert f"\n{sentinel} -I -S -B -c " in rendered["invalidate shell"]
    assert f"\n{sentinel} -I -S -B - <<'DGXM_INSTALLATION'\n" in rendered["installation"]
    assert f"\n{sentinel} -I -S -B - <<'DGXM_PRIOR_START'\n" in rendered["prior start"]
    assert f"\n{sentinel} -I -S -B - <<'DGXM_RETIRE'\n" in rendered["retire"]
    assert [site for site, text in rendered.items() if "/usr/bin/python3" in text] == []


def test_the_shipped_retire_module_is_standard_library_and_python_39():
    """It runs on the host's system interpreter inside the installation helper, never in the package."""
    source = Path(retire_remote.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 9))
    roots = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    roots |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module}
    assert not roots - sys.stdlib_module_names - {"__future__"}
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level]


def test_generation_prestart_line_keeps_its_installed_bytes():
    """Installed Worker units carry this exact ExecStartPre= line, and update
    recognizes an installed unit only by its exact bytes (recognized_unit). A
    change to the interpreter or the helper source needs a reviewed migration."""
    from dgx_monarch.cli.lifecycle_generation import systemd_generation_invalidator

    line = systemd_generation_invalidator()
    assert line.startswith('ExecStartPre="/usr/bin/python3" "-I" "-S" "-B" "-c" ')
    assert hashlib.sha256(line.encode()).hexdigest() == (
        "c9f3ffe04811ad4450b116f1aec8c49445580b8877215be085553397799bc077")


def test_prior_start_payload_runs_the_generation_library_without_its_dispatch():
    from dgx_monarch.cli import lifecycle_generation as generation

    assert generation._GENERATION_HELPER == generation._GENERATION_LIBRARY + generation._GENERATION_DISPATCH
    assert generation._GENERATION_DISPATCH.startswith("operation=sys.argv[1]\n")
    assert "sys.argv" not in generation._GENERATION_LIBRARY
    host = HostConfig(name="peer", address="tcp://192.0.2.10:26600")
    config = ClusterConfig(hosts=[host], python_bin="~/venv/bin/python")
    scripts: list[str] = []

    def run(_config, _host, script, _timeout):
        scripts.append(script)
        raise OSError("transport stand-in")

    install.start_prior_installation(config, host, TOKEN, {}, run, "", TOKEN)
    payload = scripts[0].split("<<'DGXM_PRIOR_START'\n", 1)[1].rsplit("DGXM_PRIOR_START\n", 1)[0]
    compile(payload, "prior-start", "exec")
    assert "exec(" + repr(generation._GENERATION_LIBRARY) + ", generation_api)" in payload
    assert "operation=sys.argv[1]" not in payload
