from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from dgx_monarch.cli import update_bootstrap as bootstrap
from dgx_monarch.cli import update_inspect
from dgx_monarch.config import ClusterConfig
from dgx_monarch.runtime_provenance import dgx_source_manifest_sha256


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def commit(path):
    git(path, "add", "-A")
    git(path, "-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-m", "fixture")
    return git(path, "rev-parse", "HEAD")


@pytest.fixture
def checkouts(tmp_path, monkeypatch):
    original = tmp_path / "original"
    original.mkdir()
    git(original, "init", "-b", "main")
    source = original / "src/dgx_monarch"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("VALUE = 1\n")
    (original / ".gitignore").write_text("ignored.py\n__pycache__/\n")
    old = commit(original)
    git(original, "remote", "add", "origin", "https://example.test/repository.git")
    controller = tmp_path / "controller"
    subprocess.run(["git", "clone", str(original), str(controller)], check=True, capture_output=True)
    git(controller, "remote", "set-url", "origin", "https://example.test/repository.git")
    (controller / "src/dgx_monarch/__init__.py").write_text("VALUE = 2\n")
    new = commit(controller)
    monkeypatch.setattr(bootstrap, "_verify_loaded_controller", lambda path: None)
    monkeypatch.setattr(update_inspect, "inspect_checkout", lambda path, sha: (
        SimpleNamespace(source_manifest=dgx_source_manifest_sha256(path / "src/dgx_monarch")), ()))
    return original, controller, old, new


def binding(checkouts):
    original, controller, _, target = checkouts
    return bootstrap._recovery_binding(original, controller, target, controller / "src/dgx_monarch")


def request_for(checkouts, tmp_path):
    original, _, _, target = checkouts
    config = tmp_path / "cluster.toml"
    config.write_text("")
    return {"repo": str(original), "repo_identity": [original.stat().st_dev, original.stat().st_ino],
            "config": str(config), "config_sha256": hashlib.sha256(b"").hexdigest(),
            "target_ref": target, "assume_yes": False, "driver_host": None,
            "receipt_path": str(tmp_path / "receipt.json"), "recovery": binding(checkouts)}


def test_recovery_binds_distinct_original_and_controller_source(checkouts):
    proof = binding(checkouts)
    assert proof["original_head"] == checkouts[2]
    assert proof["controller_commit"] == checkouts[3]
    assert proof["original_source_manifest"] != proof["controller_source_manifest"]


@pytest.mark.parametrize("change", ["origin", "target", "dirty-controller", "dirty-original", "hidden"])
def test_recovery_refuses_unreviewed_checkout_identity(checkouts, change):
    original, controller, _, target = checkouts
    if change == "origin":
        git(controller, "remote", "set-url", "origin", "https://example.test/other.git")
    elif change == "target":
        target = "main"
    elif change == "hidden":
        git(controller, "update-index", "--assume-unchanged", "src/dgx_monarch/__init__.py")
    else:
        root = controller if change == "dirty-controller" else original
        (root / "src/dgx_monarch/__init__.py").write_text("CHANGED = True\n")
    with pytest.raises(ValueError):
        bootstrap._recovery_binding(original, controller, target, controller / "src/dgx_monarch")


@pytest.mark.parametrize("change", ["head", "source", "origin", "inode", "config", "target", "controller-manifest"])
def test_bound_original_refuses_drift(checkouts, tmp_path, change):
    original, _, _, _ = checkouts
    request = request_for(checkouts, tmp_path)
    source = request["recovery"]["controller_source_manifest"]
    if change in ("head", "source"):
        (original / "src/dgx_monarch/__init__.py").write_text("FOREIGN = True\n")
        if change == "head":
            commit(original)
    elif change == "origin":
        git(original, "remote", "set-url", "origin", "https://example.test/other.git")
    elif change == "inode":
        request["repo_identity"][1] += 1
    elif change == "config":
        Path(request["config"]).write_text("# modified\n")
    elif change == "target":
        request["target_ref"] = "f" * 40
    else:
        source = "f" * 64
    with pytest.raises(ValueError):
        bootstrap._check_recovery_original(original, request, source)


def test_child_uses_existing_serialized_flow_and_rechecks_inside_lock(checkouts, tmp_path, monkeypatch):
    from dgx_monarch.cli import update_command, update_entrypoint

    original = checkouts[0]
    request = request_for(checkouts, tmp_path)
    monkeypatch.setattr(bootstrap, "load_cluster_config", lambda path: ClusterConfig(source=str(path)))
    monkeypatch.setattr(bootstrap.signal, "signal", Mock())
    calls = []
    class Ops:
        def __init__(self, repo, config, driver_host=None):
            calls.append(("ops", repo))
        def resolve_target(self, target):
            calls.append(("resolve", target))
            return target
    monkeypatch.setattr(update_command, "SystemUpdateOps", Ops)
    def serialized(**kwargs):
        calls.append(("serialized", kwargs["repo"], kwargs["receipt_path"]))
        target = kwargs["ops"].resolve_target(kwargs["request"].target_ref)
        assert target == checkouts[3]
        (original / "src/dgx_monarch/__init__.py").write_text("DRIFT = True\n")
        with pytest.raises(ValueError, match="source changed"):
            kwargs["ops"].resolve_target(target)
        return 0
    monkeypatch.setattr(update_entrypoint, "run_serialized_update", serialized)
    root = tmp_path / "snapshot"
    root.mkdir()
    assert bootstrap._child(root, {"source_manifest": request["recovery"]["controller_source_manifest"], "request": request}) == 0
    assert calls[0] == ("ops", original)
    assert calls[1] == ("serialized", original, request["receipt_path"])
    assert len([call for call in calls if call[0] == "resolve"]) == 1
    assert (root / "settled.json").exists()


def tool_module():
    path = Path(__file__).resolve().parents[1] / "tools/update_from_checkout.py"
    spec = importlib.util.spec_from_file_location("recovery_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_rejects_ignored_source_shadow_before_import(checkouts):
    _, controller, _, target = checkouts
    tool = tool_module()
    assert tool.reviewed_source(controller, target) == target
    (controller / "src/dgx_monarch/ignored.py").write_text("raise SystemExit(99)\n")
    assert not git(controller, "status", "--porcelain")
    with pytest.raises(ValueError, match="differs"):
        tool.reviewed_source(controller, target)


def test_loaded_foreign_controller_module_is_refused(tmp_path, monkeypatch):
    package = Path(bootstrap.__file__).resolve().parents[1]
    module = ModuleType("dgx_monarch.foreign_recovery")
    module.__file__ = str(tmp_path / "foreign.py")
    (tmp_path / "foreign.py").write_text("")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="another source"):
        bootstrap._verify_loaded_controller(package)


def test_isolated_tool_loads_only_reviewed_package_without_source_sibling_shadow(tmp_path):
    controller = tmp_path / "reviewed"
    package = controller / "src/dgx_monarch"
    (package / "cli").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "cli/__init__.py").write_text("")
    (package / "config.py").write_text("import tomllib\ndef load_cluster_config(path):\n    return path\n")
    (package / "cli/update_bootstrap.py").write_text(
        "def launch_verified_update(config, **kwargs):\n"
        "    assert kwargs['assume_yes'] is False\n"
        "    assert kwargs['controller_repository'].name == 'reviewed'\n"
        "    assert kwargs['repo'].name == 'original'\n"
        "    print('MOCK_RECOVERY_DISPATCH')\n"
        "    return 0\n"
    )
    (controller / "tools").mkdir()
    original_tool = Path(__file__).resolve().parents[1] / "tools/update_from_checkout.py"
    tool = controller / "tools/update_from_checkout.py"
    tool.write_bytes(original_tool.read_bytes())
    (controller / ".gitignore").write_text("src/tomllib.py\n__pycache__/\n")
    git(controller, "init", "-b", "main")
    target = commit(controller)
    marker = tmp_path / "shadow-executed"
    (controller / "src/tomllib.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('shadow')\n")
    config = tmp_path / "cluster.toml"
    config.write_text("")
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(tool), "--repo", str(tmp_path / "original"),
         "--config", str(config), "--target", target], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "MOCK_RECOVERY_DISPATCH"
    assert not marker.exists()
