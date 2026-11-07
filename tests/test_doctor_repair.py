from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from dgx_monarch.cli.doctor_repair import (
    PermissionRepair,
    RepairError,
    apply_config_permission_repair,
    inspect_config_permissions,
    manual_only_effects,
    run_config_permission_repair,
)
from dgx_monarch.cli.setup_config_lock import SetupConfigLock


def _config(tmp_path: Path, mode: int = 0o644) -> Path:
    path = tmp_path / "cluster.toml"
    path.write_text("[cluster]\ntransport_security = 'trusted_fabric'\n")
    path.chmod(mode)
    return path


def test_inspection_is_path_free_and_marks_public_bits(tmp_path: Path) -> None:
    path = _config(tmp_path)
    plan = inspect_config_permissions(path)
    assert plan.action_id == "tighten_config_permissions"
    assert plan.needed is True
    assert plan.previous_mode == 0o644
    assert plan.desired_mode == 0o600
    assert len(plan.config_sha256) == 64
    assert str(tmp_path) not in repr(plan.wire())


def test_private_config_is_a_noop_without_confirmation(tmp_path: Path) -> None:
    path = _config(tmp_path, 0o600)
    plan = inspect_config_permissions(path)
    called = False

    def confirm(_plan: PermissionRepair) -> bool:
        nonlocal called
        called = True
        return False

    result = apply_config_permission_repair(path, plan, confirm=confirm)
    assert result is plan
    assert called is False
    assert path.stat().st_mode & 0o777 == 0o600


def test_repair_requires_confirmation_and_records_prior_mode(tmp_path: Path) -> None:
    path = _config(tmp_path)
    plan = inspect_config_permissions(path)
    with pytest.raises(RepairError, match="repair_not_confirmed"):
        apply_config_permission_repair(path, plan, confirm=lambda _plan: False)
    assert path.stat().st_mode & 0o777 == 0o644

    result = apply_config_permission_repair(path, plan, confirm=lambda _plan: True)
    assert path.stat().st_mode & 0o777 == 0o600
    assert result.previous_mode == 0o644
    assert result.reversible is True
    assert result.needed is False


def test_changed_contents_fail_closed(tmp_path: Path) -> None:
    path = _config(tmp_path)
    plan = inspect_config_permissions(path)
    path.write_text(path.read_text() + "# changed\n")
    with pytest.raises(RepairError, match="config_changed_since_inspection"):
        apply_config_permission_repair(path, plan, confirm=lambda _plan: True)
    assert path.stat().st_mode & 0o777 == 0o644


def test_changed_mode_fails_closed(tmp_path: Path) -> None:
    path = _config(tmp_path)
    plan = inspect_config_permissions(path)
    path.chmod(0o640)
    with pytest.raises(RepairError, match="config_permissions_changed_since_inspection"):
        apply_config_permission_repair(path, plan, confirm=lambda _plan: True)


def test_same_bytes_and_mode_on_a_replacement_inode_fail_closed(tmp_path: Path) -> None:
    path = _config(tmp_path)
    plan = inspect_config_permissions(path)
    replacement = tmp_path / "replacement.toml"
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o644)
    os.replace(replacement, path)

    with pytest.raises(RepairError, match="config_inode_changed_since_inspection"):
        apply_config_permission_repair(path, plan, confirm=lambda _plan: True)
    assert path.stat().st_mode & 0o777 == 0o644


def test_symlink_is_refused(tmp_path: Path) -> None:
    target = _config(tmp_path)
    link = tmp_path / "linked.toml"
    link.symlink_to(target)
    with pytest.raises(RepairError, match="config_permissions_unreadable"):
        inspect_config_permissions(link)


def test_unknown_action_is_never_executed(tmp_path: Path) -> None:
    path = _config(tmp_path)
    plan = inspect_config_permissions(path)
    unknown = PermissionRepair(**{**plan.__dict__, "action_id": "restart_worker_service"})
    with pytest.raises(RepairError, match="repair_action_not_allowlisted"):
        apply_config_permission_repair(path, unknown, confirm=lambda _plan: True)
    assert path.stat().st_mode & 0o777 == 0o644


def test_forged_permission_mode_is_never_executed(tmp_path: Path) -> None:
    path = _config(tmp_path)
    plan = inspect_config_permissions(path)
    forged = PermissionRepair(**{**plan.__dict__, "desired_mode": 0o666})

    with pytest.raises(RepairError, match="repair_mode_not_allowlisted"):
        apply_config_permission_repair(path, forged, confirm=lambda _plan: True)

    assert path.stat().st_mode & 0o777 == 0o644


def test_mutating_effects_stay_manual_only() -> None:
    assert set(manual_only_effects()) == {
        "network", "ssh", "package", "service", "process", "mesh", "power"
    }
    assert "filesystem" not in manual_only_effects()


def test_final_mode_ignores_process_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _config(tmp_path)
    plan = inspect_config_permissions(path)
    old_umask = os.umask(0o777)
    try:
        apply_config_permission_repair(path, plan, confirm=lambda _plan: True)
    finally:
        os.umask(old_umask)
    assert path.stat().st_mode & 0o777 == 0o600


def test_repair_fsyncs_the_config_inode_after_fchmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _config(tmp_path)
    plan = inspect_config_permissions(path)
    calls: list[tuple[str, int]] = []
    real_fchmod = os.fchmod
    real_fsync = os.fsync

    def recording_fchmod(fd: int, mode: int) -> None:
        calls.append(("fchmod", mode))
        real_fchmod(fd, mode)

    def recording_fsync(fd: int) -> None:
        calls.append(("fsync", os.fstat(fd).st_ino))
        real_fsync(fd)

    monkeypatch.setattr("dgx_monarch.cli.doctor_repair.os.fchmod", recording_fchmod)
    monkeypatch.setattr("dgx_monarch.cli.doctor_repair.os.fsync", recording_fsync)
    apply_config_permission_repair(path, plan, confirm=lambda _plan: True)

    assert calls[0] == ("fchmod", 0o600)
    assert calls[1] == ("fsync", path.stat().st_ino)


def test_cli_repair_writes_a_sanitized_receipt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    path = _config(tmp_path)
    receipt = tmp_path / "receipt.json"
    output = io.StringIO()
    ok, written = run_config_permission_repair(
        path, assume_yes=True, receipt_path=receipt, output=output
    )
    assert ok is True
    assert written == receipt
    payload = json.loads(receipt.read_text())
    assert payload["operation"] == "doctor_repair"
    assert payload["status"] == "succeeded"
    assert payload["selection"]["source_hashes"]["config"]
    assert str(tmp_path) not in receipt.read_text()
    assert "no services" in output.getvalue()


def test_cli_repair_decline_changes_nothing(tmp_path: Path) -> None:
    path = _config(tmp_path)
    receipt = tmp_path / "declined.json"
    ok, written = run_config_permission_repair(
        path,
        receipt_path=receipt,
        input_fn=lambda _prompt: "n",
        output=io.StringIO(),
    )
    assert ok is False
    assert written == receipt
    assert path.stat().st_mode & 0o777 == 0o644
    assert json.loads(receipt.read_text())["status"] == "failed"


def test_cli_repair_refuses_while_a_config_writer_holds_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _config(tmp_path)
    receipt = tmp_path / "contended.json"
    output = io.StringIO()

    with SetupConfigLock(path):
        ok, written = run_config_permission_repair(
            path, assume_yes=True, receipt_path=receipt, output=output
        )

    assert ok is False
    assert written == receipt
    assert path.stat().st_mode & 0o777 == 0o644
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "failed"
    assert "config_repair_lock_unavailable" in receipt.read_text()
    assert "config_repair_lock_unavailable" in output.getvalue()


@pytest.mark.parametrize("failure", ["fsync", "fstat"])
def test_a_failure_after_the_mode_change_is_recorded_as_applied_but_unconfirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """fchmod already changed the file, so the receipt must not say nothing changed.

    The step reads unknown with changed=1, the run stays unsuccessful, and the
    operator is told to inspect again rather than that the repair was not applied.
    """
    path = _config(tmp_path)
    receipt = tmp_path / "unconfirmed.json"
    real_fchmod, real_fsync, real_fstat = os.fchmod, os.fsync, os.fstat
    # Only the config inode fails, after its mode changed: the receipt writer
    # shares the os module and can reuse the closed descriptor's number.
    config_inode = path.stat().st_ino
    mutated: list[bool] = []

    def config_after_mutation(fd: int) -> bool:
        return bool(mutated) and real_fstat(fd).st_ino == config_inode

    def recording_fchmod(fd: int, mode: int) -> None:
        real_fchmod(fd, mode)
        mutated.append(True)

    def broken(fd: int) -> None:
        if config_after_mutation(fd):
            raise OSError(5, "Input/output error")
        real_fsync(fd)

    def fstat_after_mutation(fd: int) -> os.stat_result:
        if config_after_mutation(fd):
            raise OSError(5, "Input/output error")
        return real_fstat(fd)

    monkeypatch.setattr("dgx_monarch.cli.doctor_repair.os.fchmod", recording_fchmod)
    if failure == "fsync":
        monkeypatch.setattr("dgx_monarch.cli.doctor_repair.os.fsync", broken)
    else:
        monkeypatch.setattr("dgx_monarch.cli.doctor_repair.os.fstat", fstat_after_mutation)
    output = io.StringIO()
    ok, written = run_config_permission_repair(
        path, assume_yes=True, receipt_path=receipt, output=output
    )

    assert ok is False
    assert written == receipt
    assert path.stat().st_mode & 0o777 == 0o600
    payload = json.loads(receipt.read_text())
    step = next(item for item in payload["steps"] if item["name"] == "tighten_config_permissions")
    assert step["status"] == "unknown"
    assert step["counts"] == {"changed": 1}
    assert payload["status"] == "partial"
    assert "not applied" not in receipt.read_text()
    assert "config_permissions_unconfirmed" in output.getvalue()


def test_apply_reports_the_mutation_on_its_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _config(tmp_path)
    plan = inspect_config_permissions(path)

    def broken(_fd: int) -> None:
        raise OSError(5, "Input/output error")

    monkeypatch.setattr("dgx_monarch.cli.doctor_repair.os.fsync", broken)
    with pytest.raises(RepairError, match="config_permissions_unconfirmed") as raised:
        apply_config_permission_repair(path, plan, confirm=lambda _plan: True)
    assert raised.value.applied is True
    with pytest.raises(RepairError) as before:
        apply_config_permission_repair(path, plan, confirm=lambda _plan: False)
    assert before.value.applied is False


@pytest.mark.parametrize(("lands", "code", "applied"), [
    (0o644, "config_permissions_not_applied", False),    # the filesystem ignored chmod
    (0o640, "config_permissions_unconfirmed", True),     # some other mode answered
])
def test_a_readback_that_answers_is_judged_by_the_mode_it_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lands: int, code: str, applied: bool
) -> None:
    """A readback of the prior mode saw the change not stick, so nothing changed;
    any other mode leaves the change applied but unconfirmed."""
    path = _config(tmp_path)
    plan = inspect_config_permissions(path)
    (tmp_path / "second").mkdir()
    second = _config(tmp_path / "second")  # a fresh 0644 file for the receipt run
    real_fchmod = os.fchmod
    configs = {path.stat().st_ino, second.stat().st_ino}

    def lands_as(fd: int, mode: int) -> None:
        # Only the configs: the receipt writer shares the os module.
        real_fchmod(fd, lands if os.fstat(fd).st_ino in configs else mode)

    monkeypatch.setattr("dgx_monarch.cli.doctor_repair.os.fchmod", lands_as)
    with pytest.raises(RepairError, match=code) as raised:
        apply_config_permission_repair(path, plan, confirm=lambda _plan: True)
    assert raised.value.applied is applied
    receipt = tmp_path / "receipt.json"
    ok, _written = run_config_permission_repair(
        second, assume_yes=True, receipt_path=receipt, output=io.StringIO())
    assert ok is False
    step = next(item for item in json.loads(receipt.read_text())["steps"]
                if item["name"] == "tighten_config_permissions")
    assert (step["status"], step["counts"]) == (
        ("unknown", {"changed": 1}) if applied else ("failed", {"changed": 0}))
