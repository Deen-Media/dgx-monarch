"""Driver unit tests use only mocked service-manager and process observations."""

import asyncio
import os
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.driver_service import _MARKERS, _PROPERTIES, DriverService
from dgx_monarch.studio.operations import Operations
from dgx_monarch.studio.store import Store


@pytest.fixture(autouse=True)
def no_live_service_manager(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("A test attempted a live subprocess")
    monkeypatch.setattr(subprocess, "run", denied)


@pytest.fixture
def driver(tmp_path, monkeypatch):
    unit_root = tmp_path / "units"
    unit_root.mkdir()
    unit = unit_root / "comfyui.service"
    unit.write_text("[Service]\nExecStart=/python /comfy/main.py\n")
    unit.chmod(0o644)
    source_root = tmp_path / "ComfyUI"
    source_root.mkdir()
    for name, markers in _MARKERS.items():
        (source_root / name).write_text("\n".join(markers))
        (source_root / name).chmod(0o644)
    executable = tmp_path / "python3"
    executable.write_bytes(b"fake executable never invoked")
    executable.chmod(0o755)
    service = DriverService(unit.name)
    service.owner_root = unit_root
    argv = [str(executable), str(source_root / "main.py"), "--listen", "127.0.0.1"]
    fields = dict.fromkeys(_PROPERTIES, "")
    fields.update(Id=unit.name, LoadState="loaded", Transient="no", NeedDaemonReload="no", FragmentPath=str(unit),
                  ExecStart=f"{{ path={executable} ; argv[]={' '.join(argv)} ; ignore_errors=no ; pid=0 ; }}",
                  ActiveState="inactive", SubState="dead", MainPID="0", UnitFileState="disabled", Restart="no", KillMode="control-group")
    monkeypatch.setattr(service, "_show", lambda: deepcopy(fields))
    return service, fields, unit, source_root


def test_owner_unit_and_source_identity_is_serializable(driver):
    service, _, unit, _ = driver
    identity = service.inspect()
    assert identity["role"] == "comfyui_driver"
    assert identity["state"] == "inactive" and identity["process"] is None
    assert identity["config"]["owner_uid"] == os.getuid()
    assert identity["config"]["unit_files"][0]["path"] == str(unit)
    assert len(identity["config"]["source_files"]) == 5
    assert "argv" not in str(identity)


def test_service_manager_read_uses_fixed_property_allowlist(driver, monkeypatch):
    service, fields, _, _ = driver
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="\n".join(f"{key}={value}" for key, value in fields.items()).encode())

    monkeypatch.setattr(subprocess, "run", run)
    assert DriverService._show(service) == fields
    assert calls[0][:4] == ["systemctl", "--user", "show", "--no-pager"]
    assert calls[0][-2:] == ["--", "comfyui.service"]
    fields.pop("MainPID")
    with pytest.raises(StudioError, match="omitted required"):
        DriverService._show(service)


@pytest.mark.parametrize("name", ["monarch-worker.service", "dgxm-workers.service", "--bad.service", "comfy.service;reboot", "../comfy.service"])
def test_worker_and_invalid_unit_names_refused(name):
    with pytest.raises(StudioError):
        DriverService(name)


@pytest.mark.parametrize("field,value", [
    ("LoadState", "not-found"), ("Transient", "yes"), ("NeedDaemonReload", "yes"), ("User", "another-owner"),
    ("ActiveState", "failed"), ("SubState", "start"), ("MainPID", "unknown"),
    ("ExecStop", "{ path=/bin/sh ; }"), ("ExecCondition", "{ path=/bin/sh ; }"),
    ("Wants", "monarch-worker.service"), ("PartOf", "other.service"), ("Id", "other.service"),
    ("EnvironmentFiles", "/owner/unbound.env"), ("KillMode", "process"),
])
def test_foreign_coupled_or_unknown_service_refused(driver, field, value):
    service, fields, _, _ = driver
    fields[field] = value
    with pytest.raises(StudioError):
        service.inspect()


def test_arbitrary_main_script_cannot_claim_comfy_role(driver):
    service, _, _, source_root = driver
    (source_root / "main.py").write_text("print('not ComfyUI')")
    with pytest.raises(StudioError, match="source identity"):
        service.inspect()


def test_unit_outside_owner_directory_and_writable_unit_refused(driver):
    service, fields, unit, _ = driver
    fields["FragmentPath"] = "/etc/systemd/user/comfyui.service"
    with pytest.raises(StudioError, match="current owner"):
        service.inspect()
    fields["FragmentPath"] = str(unit)
    unit.chmod(0o666)
    with pytest.raises(StudioError, match="owner-controlled"):
        service.inspect()


def test_unit_symlink_refused(driver, tmp_path):
    service, fields, unit, _ = driver
    alias = unit.with_name("alias.service")
    alias.symlink_to(unit)
    fields["FragmentPath"] = str(alias)
    with pytest.raises(StudioError, match="Symlinked"):
        service.inspect()


def test_dropin_and_source_changes_invalidate_review_before_action(driver):
    service, fields, unit, source = driver
    before = service.inspect()
    dropin = unit.parent / "override.conf"
    dropin.write_text("[Service]\nRestart=always\n")
    dropin.chmod(0o644)
    fields["DropInPaths"] = str(dropin)
    with pytest.raises(StudioError, match="changed after review"):
        asyncio.run(service.perform("start", before))
    fields["DropInPaths"] = ""
    (source / "main.py").write_text((source / "main.py").read_text() + "\n# changed\n")
    with pytest.raises(StudioError, match="changed after review"):
        asyncio.run(service.perform("start", before))


def test_start_requires_verified_post_state_and_fixed_argv(driver, monkeypatch):
    service, fields, _, _ = driver
    before = service.inspect()
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs["stdin"] == subprocess.DEVNULL
        fields.update(ActiveState="active", SubState="running", MainPID="123", InvocationID="new")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(service, "_process", lambda *args: {"pid": 123, "start_ticks": "111"})
    state, receipt = asyncio.run(service.perform("start", before))
    assert calls == [["systemctl", "--user", "start", "--", "comfyui.service"]]
    assert state == "succeeded" and receipt["verification"] == "verified"
    assert receipt["observed_identity"]["process"]["pid"] == 123


@pytest.mark.parametrize("result", ["unchanged", "timeout", "failed", "changed-unit"])
def test_command_success_alone_is_not_terminal_success(driver, monkeypatch, result):
    service, fields, unit, _ = driver
    before = service.inspect()

    def run(argv, **kwargs):
        if result == "timeout":
            raise subprocess.TimeoutExpired(argv, 30)
        if result == "failed":
            fields.update(ActiveState="failed", SubState="failed")
        if result == "changed-unit":
            unit.write_text(unit.read_text() + "# changed\n")
            fields.update(ActiveState="active", SubState="running", MainPID="123")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(service, "_process", lambda *args: {"pid": 123})
    state, receipt = asyncio.run(service.perform("start", before))
    assert state == "unknown" and receipt["verification"] == "unknown"


def test_restart_requires_new_process_identity(driver, monkeypatch):
    service, fields, _, _ = driver
    fields.update(ActiveState="active", SubState="running", MainPID="123", InvocationID="old")
    monkeypatch.setattr(service, "_process", lambda *args: {"pid": 123, "start_ticks": "same"})
    before = service.inspect()
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    state, _ = asyncio.run(service.perform("restart", before))
    assert state == "unknown"


def test_stop_verifies_stopped_state(driver, monkeypatch):
    service, fields, _, _ = driver
    fields.update(ActiveState="active", SubState="running", MainPID="123", InvocationID="old")
    monkeypatch.setattr(service, "_process", lambda *args: {"pid": 123, "start_ticks": "old"})
    before = service.inspect()

    def run(argv, **kwargs):
        assert argv == ["systemctl", "--user", "stop", "--", "comfyui.service"]
        fields.update(ActiveState="inactive", SubState="dead", MainPID="0", InvocationID="")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    state, receipt = asyncio.run(service.perform("stop", before))
    assert state == "succeeded"
    assert receipt["observed_identity"]["process"] is None


def test_running_process_must_match_command_and_owner_cgroup(driver, monkeypatch):
    service, fields, _, _ = driver
    fields.update(ActiveState="active", SubState="running", MainPID="999999999", ControlGroup="/fake")
    with pytest.raises(StudioError, match="process identity"):
        service.inspect()


def test_operations_persist_review_identity_and_recheck_under_lock(driver, monkeypatch, tmp_path):
    service, _fields, unit, _ = driver
    store = Store(tmp_path / "state")
    operations = Operations(store, object(), allow_management=True, driver_service=service.unit)
    operations.driver_binding = service
    review = operations.preview({"action": "driver.start"})
    assert review["available"] and review["driver_identity"]["config"]["unit"] == service.unit
    unit.write_text(unit.read_text() + "# edited after review\n")

    async def apply():
        operation = operations.apply(review["id"], {"digest": review["digest"], "confirmed": True,
                                                   "idempotency_key": "review-stale-unit"})
        await asyncio.gather(*operations.tasks)
        return store.get("operations", operation["id"])

    result = asyncio.run(apply())
    assert result["state"] == "unknown"
    assert result["receipt"]["error"] == "driver_service_identity"
    store.close()


def test_unconfigured_driver_unavailable(tmp_path):
    store = Store(tmp_path / "state")
    operations = Operations(store, object(), allow_management=True)
    assert all(not row["available"] for row in operations.capabilities() if row["id"].startswith("driver.") and row["id"] != "driver.free")
    store.close()
