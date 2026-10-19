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


@pytest.mark.parametrize("drift", [None, "config", "controller", "checkout", "prior"])
def test_child_rechecks_review_before_typed_backend(tmp_path, monkeypatch, drift):
    from dgx_monarch import config as config_module
    from dgx_monarch.cli import update_command, update_inspect, update_transaction
    from dgx_monarch.cli.update_release import ReleaseMetadata
    from dgx_monarch.studio import management_update

    config, reviewed = source_fixture(tmp_path, monkeypatch)
    metadata = ReleaseMetadata("a" * 40, "1", "0.6.0", "b" * 64, "c" * 64)
    reviewed.update(checkout={"exact": "checkout"}, prior_release=asdict(metadata), release={"target_sha": "d" * 40})
    calls = []
    backend = SimpleNamespace(repo_snapshot=lambda target: RepoSnapshot("a" * 40, drift != "prior", True),
                              resolve_target=lambda target: calls.append("resolve") or target)
    monkeypatch.setattr(update_command, "SystemUpdateOps", lambda *args, **kwargs: backend)
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
        if drift:
            with pytest.raises(StudioError):
                controller.child(root, manifest)
            assert calls == [] and not (root / "result.json").exists()
        else:
            assert controller.child(root, manifest) == 0
            assert calls == ["resolve", "update"]
            assert json.loads((root / "result.json").read_text())["receipt"] == {"exact": "receipt"}
