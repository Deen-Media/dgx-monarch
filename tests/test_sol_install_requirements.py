"""Built wheel metadata must validate before a no-deps environment mutation."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from dgx_monarch.cli import sol_install_payload as payload
from dgx_monarch.cli.sol_install_requirements import WheelRequirementsError, validate_wheel_requirements


def wheel(root: Path, requirement: str = "", *, python: str = "") -> Path:
    path = root / "sol_attn-0.5.0-py3-none-any.whl"
    metadata = "Name: sol-attn\nVersion: 0.5.0\n"
    if requirement:
        metadata += f"Requires-Dist: {requirement}\n"
    if python:
        metadata += f"Requires-Python: {python}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sol_attn-0.5.0.dist-info/METADATA", metadata)
    return path


@pytest.mark.parametrize("requirement,versions,reason", [
    ("missing-build-dependency>=1", {}, "missing dependency"),
    ("protobuf>=7", {"protobuf": "6.33.6"}, "version.*unsatisfied"),
    ("protobuf>=7", {"protobuf": "not-a-version"}, "version.*unreadable"),
    ("protobuf[unknown-extra]", {"protobuf": "7.35.0"}, "extras.*cannot be attested"),
    ("protobuf @ https://example.invalid/package.whl", {"protobuf": "7.35.0"}, "URL.*cannot be attested"),
    ("protobuf >=> invalid", {"protobuf": "7.35.0"}, "requirement unreadable"),
])
def test_unmet_or_unattestable_built_requirements_refuse(tmp_path, requirement, versions, reason):
    with pytest.raises(WheelRequirementsError, match=reason):
        validate_wheel_requirements(wheel(tmp_path, requirement), versions)


@pytest.mark.parametrize("requirement,versions", [
    ("protobuf>=6,<7", {"protobuf": "6.33.6"}),
    ('missing; python_version < "1"', {}),
    ('missing; extra == "unrequested"', {}),
    ('protobuf>=6; python_version >= "3"', {"protobuf": "6.33.6"}),
])
def test_applicable_base_requirements_and_inactive_markers(tmp_path, requirement, versions):
    validate_wheel_requirements(wheel(tmp_path, requirement), versions)


def test_incompatible_python_requirement_refuses(tmp_path):
    with pytest.raises(WheelRequirementsError, match="Python requirement unsatisfied"):
        validate_wheel_requirements(wheel(tmp_path, python=">=900"), {})


def test_independently_built_sol_requirement_blocks_plan_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(payload, "private_plan", lambda *args: tmp_path)
    monkeypatch.setattr(payload, "scheme_roots", lambda: set())
    monkeypatch.setattr(payload.sysconfig, "get_path", lambda key: str(tmp_path / key))
    monkeypatch.setattr(payload, "installed", lambda: {"torch": "2.12.0"})
    monkeypatch.setattr(payload, "protected_state", lambda: ("unchanged", set()))
    monkeypatch.setattr(payload, "preserved_paths", lambda *args, **kwargs: (set(), set()))
    monkeypatch.setattr(payload, "replacement_paths", lambda name: set())
    monkeypatch.setattr(payload, "requirement_violations", lambda *args: [])
    report = {"install": [{"metadata": {"name": "sol-attn", "version": "0.5.0"},
                           "download_info": {"url": "https://github.com/NVlabs/Sana",
                                             "vcs_info": {"vcs": "git", "commit_id": payload.SOL_SHA}}}]}
    calls = []
    def pip(args):
        calls.append(args)
        if args[0] == "install":
            assert "--dry-run" in args
            (tmp_path / "resolve.json").write_text(json.dumps(report))
        else:
            wheel(tmp_path / "wheels", "missing-build-dependency>=1")
    monkeypatch.setattr(payload, "run_pip", pip)
    with pytest.raises(WheelRequirementsError, match="missing dependency missing-build-dependency"):
        payload.plan_host("a" * 32, set())
    assert [args[0] for args in calls] == ["install", "wheel"]
    assert not (tmp_path / "plan.json").exists()


def test_apply_rechecks_built_requirements_before_any_install(tmp_path, monkeypatch):
    directory = tmp_path / "wheels"
    directory.mkdir()
    artifact = wheel(directory, "missing-build-dependency>=1")
    monkeypatch.setattr(payload.sysconfig, "get_path", lambda key: str(tmp_path / key))
    plan = {"digest": "a" * 64, "allowed": [], "python": payload.sys.executable,
            "before": {"torch": "2.12.0"}, "protected": "unchanged", "baseline_violations": [],
            "changes": [{"name": "sol-attn", "old": None, "new": "0.5.0"}],
            "artifacts": [{"file": artifact.name, "sha256": payload.file_hash(artifact)}]}
    monkeypatch.setattr(payload, "read_plan", lambda token: plan)
    monkeypatch.setattr(payload, "private_plan", lambda token: tmp_path)
    monkeypatch.setattr(payload, "installed", lambda: plan["before"])
    monkeypatch.setattr(payload, "protected_state", lambda: ("unchanged", set()))
    monkeypatch.setattr(payload, "preserved_paths", lambda *args, **kwargs: (set(), set()))
    monkeypatch.setattr(payload, "replacement_paths", lambda name: set())
    monkeypatch.setattr(payload, "run_pip", lambda args: pytest.fail("unvalidated staged requirement reached install"))
    with pytest.raises(WheelRequirementsError, match="missing dependency missing-build-dependency"):
        payload.apply_host("a" * 32, "a" * 64, set())
