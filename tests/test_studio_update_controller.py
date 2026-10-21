"""Only synthetic local source and mocked updates run in these controller tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.cli.update_lock import UpdateLock, UpdateLockUnavailable
from dgx_monarch.cli.update_types import RepoSnapshot, UpdateResult
from dgx_monarch.studio import update_controller as controller
from dgx_monarch.studio.contracts import StudioError


def source_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "studio").mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "studio/__init__.py").write_text("")
    (source / "studio/update_controller.py").write_text(
        "def child(root, manifest):\n    (root / 'executed').write_text('safe fake controller')\n    return 0\n")
    monkeypatch.setattr(controller, "SOURCE", source)
    executable = tmp_path / "python"
    executable.write_text("synthetic interpreter identity; never executed")
    executable.chmod(0o755)
    monkeypatch.setattr(controller, "sys", SimpleNamespace(executable=str(executable)))
    config = tmp_path / "config.toml"
    config.write_text("synthetic config, never loaded by fake controller")
    return config, {"controller": controller.binding(None), "config_sha256": controller.config_digest(config)}


def driver_fixture(tmp_path, repo):
    """A reviewed driver identity whose ComfyUI imports ``repo`` through custom_nodes/dgx-monarch."""
    comfy = tmp_path / "comfy"
    (comfy / "custom_nodes").mkdir(parents=True, exist_ok=True)
    link = comfy / "custom_nodes/dgx-monarch"
    if not link.is_symlink():
        link.symlink_to(repo)
    identity = {"state": "inactive", "config": {"source_files": [{"path": str(comfy / "main.py")}]}}
    return identity, {"link": str(link), "resolved": str(repo.resolve())}


@pytest.mark.parametrize("tamper", [None, "source", "manifest"])
def test_existing_loader_runs_frozen_source_and_refuses_tampering(tmp_path, monkeypatch, tamper):
    config, reviewed = source_fixture(tmp_path, monkeypatch)
    prepared = controller.prepare(tmp_path / "state", config, tmp_path, reviewed, "localhost:8188", False, 123)
    root = Path(prepared["path"])
    # Changing the original source after preparation cannot affect the child.
    (controller.SOURCE / "studio/update_controller.py").write_text("raise RuntimeError('mutable source')")
    if tamper == "source":
        (root / "dgx_monarch/studio/update_controller.py").write_text("raise RuntimeError('tampered frozen source')")
    elif tamper == "manifest":
        (root / "manifest.json").write_text("{}")
    # This is an isolated stdlib-only fake child; no Monarch backend is imported.
    result = subprocess.run([sys.executable, "-I", "-B", "-c", controller.LOADER, str(root), prepared["manifest_sha256"]],
                            capture_output=True, timeout=10, check=False)
    assert (result.returncode == 0) is (tamper is None)
    assert (root / "executed").exists() is (tamper is None)


def test_binding_preserves_original_venv_executable(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    venv = tmp_path / "venv/bin"
    venv.mkdir(parents=True)
    executable = venv / "python"
    executable.symlink_to(controller.sys.executable)
    identity = controller.binding({"config": {"executable": str(executable)}})
    assert identity["executable"] == str(executable)
    assert identity["resolved"] == str(Path(controller.sys.executable).resolve())
    config = venv.parent / "pyvenv.cfg"
    config.write_text("home = /reviewed/python")
    config.chmod(0o644)
    changed = controller.binding({"config": {"executable": str(executable)}})
    assert changed != identity and changed["environment"][1]["sha256"]


@pytest.mark.parametrize(("mode", "members", "trusted"), [
    (0o664, [], True), (0o664, ["other-owner"], False), (0o666, [], False), (0o644, ["other-owner"], True)])
def test_binding_applies_checkout_trust_rule_to_default_umask_venv(tmp_path, monkeypatch, mode, members, trusted):
    from dgx_monarch.studio import management_update

    source_fixture(tmp_path, monkeypatch)
    venv = tmp_path / "default-umask-venv"
    (venv / "bin").mkdir(parents=True)
    executable = venv / "bin/python"
    executable.write_text("synthetic interpreter identity; never executed")
    executable.chmod(0o755)
    config = venv / "pyvenv.cfg"
    config.write_text("home = /reviewed/python")
    config.chmod(mode)
    owner = SimpleNamespace(pw_name="owner", pw_gid=config.stat().st_gid)
    monkeypatch.setattr(management_update.pwd, "getpwuid", lambda uid: owner)
    monkeypatch.setattr(management_update.pwd, "getpwall", lambda: [owner])
    monkeypatch.setattr(management_update.grp, "getgrgid", lambda gid: SimpleNamespace(gr_mem=["owner", *members]))
    driver = {"config": {"executable": str(executable)}}
    if trusted:
        identity = controller.binding(driver)
        assert identity["environment"][1] == {"path": str(config), "sha256": controller.config_digest(config)}
    else:
        with pytest.raises(StudioError, match="not owner-controlled"):
            controller.binding(driver)


def test_binding_refuses_interpreter_writable_by_another_account(tmp_path, monkeypatch):
    from dgx_monarch.studio import management_update

    source_fixture(tmp_path, monkeypatch)
    executable = tmp_path / "shared-bin/python"
    executable.parent.mkdir()
    executable.write_text("synthetic interpreter identity; never executed")
    executable.chmod(0o775)
    owner = SimpleNamespace(pw_name="owner", pw_gid=executable.stat().st_gid)
    monkeypatch.setattr(management_update.pwd, "getpwuid", lambda uid: owner)
    monkeypatch.setattr(management_update.pwd, "getpwall", lambda: [owner])
    group = SimpleNamespace(gr_mem=["owner"])
    monkeypatch.setattr(management_update.grp, "getgrgid", lambda gid: group)
    assert controller.binding({"config": {"executable": str(executable)}})["executable"] == str(executable)
    group.gr_mem.append("other-owner")
    with pytest.raises(StudioError, match="owner-controlled Python interpreter"):
        controller.binding({"config": {"executable": str(executable)}})


def test_controller_runs_exact_interpreter_retains_result_until_checkpoint(tmp_path, monkeypatch):
    config, reviewed = source_fixture(tmp_path, monkeypatch)
    with UpdateLock(tmp_path, state_root=tmp_path / "locks") as lock:
        prepared = controller.prepare(tmp_path / "state", config, tmp_path, reviewed, "localhost:8188", False, lock.fileno())
        root = Path(prepared["path"])
        prepared["executable"] = "/reviewed/comfy-venv/bin/python"
        observed = {}
        class FakeProcess:
            pid = 1234
            def wait(self):
                (root / "result.json").write_text(json.dumps(asdict(UpdateResult(0, "updated", "done", {"retained": True}))))
                observed["waited"] = True
                return 0
        def popen(argv, **kwargs):
            observed.update(argv=argv, kwargs=kwargs)
            return FakeProcess()
        monkeypatch.setattr(controller.subprocess, "Popen", popen)
        result = controller.run(prepared)
        assert result.succeeded and result.receipt == {"retained": True}
        assert observed["argv"][0] == "/reviewed/comfy-venv/bin/python"
        assert observed["kwargs"]["pass_fds"] == (lock.fileno(),)
        assert os.fstat(lock.fileno()).st_size == 0
        assert observed["kwargs"]["start_new_session"] is True
        assert "PYTHONPATH" not in observed["kwargs"]["env"]
        assert (root / "result.json").exists() and observed["waited"]
        controller.cleanup(prepared)
        assert not root.exists()
    with pytest.raises(UpdateLockUnavailable):
        lock.fileno()


def test_controller_receipt_failure_still_waits_for_child(tmp_path, monkeypatch):
    config, reviewed = source_fixture(tmp_path, monkeypatch)
    prepared = controller.prepare(tmp_path / "state", config, tmp_path, reviewed, "localhost:8188", False, 123)
    calls = []
    class FakeProcess:
        pid = 1234
        def wait(self):
            calls.append("settled")
            return 1
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    def fail_write(*args):
        raise OSError("publication failed")
    monkeypatch.setattr(controller, "_write_private", fail_write)
    with pytest.raises(OSError, match="publication failed"):
        controller.run(prepared)
    assert calls == ["settled"] and Path(prepared["path"]).exists()


def test_changed_controller_source_refuses_preparation(tmp_path, monkeypatch):
    config, reviewed = source_fixture(tmp_path, monkeypatch)
    reviewed["controller"]["source_manifest"] = "0" * 64
    with pytest.raises(StudioError, match="source changed"):
        controller.prepare(tmp_path / "state", config, tmp_path, reviewed, "localhost:8188", False, 123)


@pytest.mark.parametrize("drift", [None, "config", "controller", "checkout", "prior", "driver_link"])
def test_child_rechecks_review_before_typed_backend(tmp_path, monkeypatch, drift):
    from dgx_monarch import config as config_module
    from dgx_monarch.cli import update_command, update_inspect, update_transaction
    from dgx_monarch.cli.update_release import ReleaseMetadata
    from dgx_monarch.studio import management_update

    config, reviewed = source_fixture(tmp_path, monkeypatch)
    metadata = ReleaseMetadata("a" * 40, "1", "0.6.0", "b" * 64, "c" * 64)
    identity, bound = driver_fixture(tmp_path, tmp_path)
    reviewed.update(mode="verified", checkout={"exact": "checkout"}, prior_release=asdict(metadata),
                    release={"target_sha": "d" * 40}, driver_identity=identity, driver_checkout=bound)
    calls, hosts = [], []
    backend = SimpleNamespace(repo_snapshot=lambda target: RepoSnapshot("a" * 40, drift != "prior", True),
                              resolve_target=lambda target: calls.append("resolve") or target)
    def system_ops(*args, **kwargs):
        hosts.append(kwargs["driver_host"])
        return backend
    monkeypatch.setattr(update_command, "SystemUpdateOps", system_ops)
    monkeypatch.setattr(config_module, "load_cluster_config", lambda path: "parsed-config")
    monkeypatch.setattr(update_inspect, "inspect_checkout", lambda *args: (metadata, ()))
    monkeypatch.setattr(management_update, "checkout_identity", lambda path: {} if drift == "checkout" else reviewed["checkout"])
    monkeypatch.setattr(controller.signal, "signal", lambda *args: None)
    monkeypatch.setattr(controller, "binding", lambda driver: {} if drift == "controller" else reviewed["controller"])
    def execute(request, proxy):
        assert request.target_ref == reviewed["release"]["target_sha"]
        assert proxy.resolve_target(request.target_ref) == request.target_ref
        calls.append("update")
        return UpdateResult(0, "updated", "settled", {"exact": "receipt"})
    monkeypatch.setattr(update_transaction, "execute_verified_update", execute)
    with UpdateLock(tmp_path, state_root=tmp_path / "locks") as lock:
        prepared = controller.prepare(tmp_path / "state", config, tmp_path, reviewed, "localhost:8188", False, lock.fileno())
        root = Path(prepared["path"])
        manifest = json.loads((root / "manifest.json").read_text())
        if drift == "config":
            config.write_text("different config")
        if drift == "driver_link":
            link = Path(bound["link"])
            link.unlink()
            link.symlink_to(tmp_path / "state")
        if drift:
            # Admission only reads, so its refusal is retained as a definite no-change settlement.
            assert controller.child(root, manifest) == 2
            settled = json.loads((root / "result.json").read_text())
            assert calls == [] and settled["code"] == "controller_refused"
            assert settled["receipt"]["mutation_started"] is False and settled["receipt"]["refusal"]
            assert controller.settlement("verified", UpdateResult(**settled)) == "refused"
        else:
            assert controller.child(root, manifest) == 0
            assert calls == ["resolve", "update"]
            assert json.loads((root / "result.json").read_text())["receipt"] == {"exact": "receipt"}
            # The reviewed request named a host, but Studio stopped and rechecked
            # the bound driver, so the verified activity guard gets none.
            assert hosts == [None]


@pytest.mark.parametrize("mode", ["legacy", "verified"])
def test_prepare_requires_configuration_in_every_mode(tmp_path, monkeypatch, mode):
    _, reviewed = source_fixture(tmp_path, monkeypatch)
    reviewed.update(mode=mode, config_sha256=None)
    with pytest.raises(StudioError, match="configuration in every mode"):
        controller.prepare(tmp_path / "state", None, tmp_path, reviewed, None, False, 123)
    assert not (tmp_path / "state").exists()


def test_child_refuses_config_free_manifest_without_running_backend(tmp_path, monkeypatch):
    from dgx_monarch.cli import legacy_update

    source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(controller.signal, "signal", lambda *args: None)
    monkeypatch.setattr(legacy_update, "_run_locked", lambda *args, **kwargs: pytest.fail("legacy backend ran"))
    with UpdateLock(tmp_path, state_root=tmp_path / "locks") as lock:
        manifest = {"request": {"config": None, "config_sha256": None, "repo": str(tmp_path), "lock_fd": lock.fileno(),
                                "driver_host": None, "privileged_process_inspection": False,
                                "reviewed": {"mode": "legacy", "controller": controller.binding(None)}}}
        assert controller.child(tmp_path, manifest) == 2
    settled = json.loads((tmp_path / "result.json").read_text())
    assert settled["receipt"]["refusal"] == "update_config" and settled["receipt"]["mutation_started"] is False


def steps(*names, status="succeeded"):
    return [{"name": name, "status": status} for name in names]


GUARDS = ("target_resolution", "repository_guard", "activity_guard")
STAGED = (*GUARDS, "release_staging")


@pytest.mark.parametrize(("mode", "exit_code", "code", "receipt", "expected"), [
    ("verified", 0, "updated", {"status": "succeeded", "steps": steps(*GUARDS, "worker_stop")}, "succeeded"),
    ("verified", 0, "updated", {"status": "partial", "steps": []}, "unknown"),
    ("verified", 2, "activity_unknown", {"status": "failed", "steps": steps(*GUARDS[:2])}, "refused"),
    ("verified", 2, "cancelled", {"status": "planned", "steps": [*steps(*STAGED, "stage_cleanup"),
                                                                 *steps("confirmation", status="planned")]}, "refused"),
    ("verified", 1, "stage_invalid", {"status": "failed", "steps": steps(*STAGED)}, "unknown"),
    ("verified", 1, "stage_invalid", {"status": "partial", "steps": [*steps(*STAGED), *steps("stage_cleanup", status="unknown")]},
     "unknown"),
    ("verified", 1, "start_failed", {"status": "failed", "steps": steps(*GUARDS, "worker_stop", "activation")},
     "settled_after_change"),
    ("verified", 1, "start_unknown", {"status": "partial", "steps": steps(*GUARDS, "worker_stop")}, "unknown"),
    ("verified", 1, "operation_exception", {"status": "failed", "steps": steps(*STAGED, "stage_retention")}, "unknown"),
    ("verified", 1, "stop_failed", {}, "unknown"),
    ("legacy", 0, "legacy_updated", {"mode": "legacy"}, "succeeded"),
    ("legacy", 2, "legacy_refused", {"mode": "legacy"}, "refused"),
    ("legacy", 75, "legacy_refused", {"mode": "legacy"}, "refused"),
    ("legacy", 1, "legacy_incomplete", {"mode": "legacy"}, "unknown"),
    ("legacy", 3, "legacy_incomplete", {"mode": "legacy"}, "unknown"),
    ("legacy", 2, "legacy_refused", {}, "unknown"),
    ("verified", 2, "controller_refused", {"mutation_started": False}, "refused"),
    ("legacy", 2, "controller_refused", {}, "unknown"),
    ("verified", 1, "controller_unsettled", {"status": "failed", "steps": []}, "unknown"),
])
def test_settlement_classifies_receipts_and_keeps_unknown_apart(mode, exit_code, code, receipt, expected):
    assert controller.settlement(mode, UpdateResult(exit_code, code, "message", receipt)) == expected


def test_legacy_no_change_exits_use_the_shared_unknown_vocabulary():
    from dgx_monarch.cli import probe_certainty

    assert controller.LEGACY_NO_CHANGE_EXITS == {2, probe_certainty.UNKNOWN_EXIT}


def test_controller_keeps_bounded_sanitized_stderr_instead_of_discarding_it(tmp_path, monkeypatch):
    config, reviewed = source_fixture(tmp_path, monkeypatch)
    prepared = controller.prepare(tmp_path / "state", config, tmp_path, reviewed, None, False, 123)
    root = Path(prepared["path"])
    observed = {}
    class FakeProcess:
        pid = 1234
        def wait(self):
            settled = UpdateResult(2, "legacy_refused", "refused", {"mode": "legacy"})
            (root / "result.json").write_text(json.dumps(asdict(settled)))
            return 2
    def popen(argv, **kwargs):
        observed.update(kwargs)
        noise = b"".join(b"progress line %d\n" % index for index in range(2000))
        reason = (b"refusing to update: worker-a runs its Worker from release slot u-1 (release_slot_worker). "
                  b"Nothing was fetched or changed.\n")
        os.write(kwargs["stderr"], noise + reason + b"x" * 400 + b"\n")
        return FakeProcess()
    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    result = controller.run(prepared)
    assert observed["stderr"] not in {subprocess.DEVNULL, None}
    assert controller.settlement("legacy", result) == "refused"
    tail = controller.stderr_tail(prepared)
    assert 0 < len(tail) <= controller.STDERR_LINES and all(len(line) <= 240 for line in tail)
    assert any("release_slot_worker" in line for line in tail)
    assert (root / controller.STDERR_NAME).stat().st_mode & 0o777 == 0o600


def test_real_child_process_stderr_reaches_the_bounded_receipt(tmp_path, monkeypatch):
    config, reviewed = source_fixture(tmp_path, monkeypatch)
    # A stdlib-only fake child, frozen and verified by the real loader in a real process.
    (controller.SOURCE / "studio/update_controller.py").write_text(
        "import json, sys\n"
        "def child(root, manifest):\n"
        "    print('refusing to update (release_slot_worker). Nothing was fetched or changed.', file=sys.stderr)\n"
        "    settled = {'exit_code': 2, 'code': 'legacy_refused', 'message': 'refused', 'receipt': {'mode': 'legacy'}}\n"
        "    (root / 'result.json').write_text(json.dumps(settled))\n"
        "    return 2\n")
    reviewed["controller"] = controller.binding(None)
    with UpdateLock(tmp_path, state_root=tmp_path / "locks") as lock:
        prepared = controller.prepare(tmp_path / "state", config, tmp_path, reviewed, None, False, lock.fileno())
        prepared["executable"] = sys.executable
        result = controller.run(prepared)
    assert result.exit_code == 2 and controller.settlement("legacy", result) == "refused"
    assert controller.stderr_tail(prepared) == ["refusing to update (release_slot_worker). Nothing was fetched or changed."]


@pytest.mark.parametrize(("written", "process_exit", "code"), [
    (None, 1, "controller_unsettled"), (0, 1, "controller_mismatch"), ("garbage", 0, "controller_unsettled"),
    ("wrong_types", 0, "controller_unsettled")])
def test_missing_or_inconsistent_settlement_is_retained_as_unknown(tmp_path, monkeypatch, written, process_exit, code):
    config, reviewed = source_fixture(tmp_path, monkeypatch)
    prepared = controller.prepare(tmp_path / "state", config, tmp_path, reviewed, None, False, 123)
    root = Path(prepared["path"])
    class FakeProcess:
        pid = 1234
        def wait(self):
            if written == "garbage":
                (root / "result.json").write_text("{")
            elif written == "wrong_types":
                (root / "result.json").write_text(json.dumps({"exit_code": 0, "code": 1, "message": None, "receipt": []}))
            elif written is not None:
                (root / "result.json").write_text(json.dumps(asdict(UpdateResult(written, "updated", "done", {}))))
            (root / "interrupted.json").write_text(json.dumps({"status": "partial"}))
            return process_exit
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    result = controller.run(prepared)
    assert result.code == code and not result.succeeded
    assert result.receipt["interrupted"] == {"status": "partial"}
    assert controller.settlement("verified", result) == "unknown"
