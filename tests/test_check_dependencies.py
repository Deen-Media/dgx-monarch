"""Only the unchanged official ARM64 wheel can waive its known tag defect."""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("check_dependencies", Path(__file__).parents[1] / "tools/check_dependencies.py")
assert SPEC is not None and SPEC.loader is not None
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


@pytest.fixture
def wheel_env(tmp_path, monkeypatch):
    metadata = "Wheel-Version: 1.0\nTag: py3-none-manylinux2014_sbsa\n"
    files = {
        f"{check.DIST_INFO}/WHEEL": metadata.encode(),
        f"{check.DIST_INFO}/METADATA": b"Name: nvidia-cusparselt-cu13\nVersion: 0.8.1\n",
        f"{check.DIST_INFO}/RECORD": b"upstream record",
        "nvidia/cusparselt/lib/libcusparseLt.so.0": b"original binary",
    }
    archive_path = tmp_path / "fixture.whl"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
            installed = tmp_path / "installed" / name
            installed.parent.mkdir(parents=True, exist_ok=True)
            installed.write_bytes(data)
    dist = SimpleNamespace(
        version="0.8.1", read_text=lambda name: metadata,
        locate_file=lambda name: tmp_path / "installed" / name,
    )
    monkeypatch.setattr(check, "distribution", lambda name: dist)
    monkeypatch.setattr(check.sys, "platform", "linux")
    monkeypatch.setattr(check.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(check, "WHEEL_SHA256", hashlib.sha256(archive_path.read_bytes()).hexdigest())
    monkeypatch.setattr(check, "download_wheel", lambda path: path.write_bytes(archive_path.read_bytes()))
    return dist


def pip_result(monkeypatch, stdout, stderr="", status=1):
    def run(command, **kwargs):
        assert command == [sys.executable, "-m", "pip", "check"]
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        return subprocess.CompletedProcess(command, status, stdout, stderr)
    monkeypatch.setattr(check.subprocess, "run", run)


@pytest.mark.parametrize("output", ["", "No broken requirements found.\n"])
def test_success_is_silent(monkeypatch, capsys, output):
    pip_result(monkeypatch, output, status=0)
    monkeypatch.setattr(check, "verify_known_wheel", lambda: pytest.fail("unnecessary download"))
    assert check.main() == 0
    assert capsys.readouterr() == ("", "")


def test_known_failure_verifies_every_file(wheel_env, monkeypatch, capsys):
    pip_result(monkeypatch, check.KNOWN_FAILURE + "\n")
    # Installer-generated RECORD changes are expected and deliberately excluded.
    wheel_env.locate_file(f"{check.DIST_INFO}/RECORD").write_text("installed record")
    assert check.main() == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == (
        f"Known upstream platform-tag failure: {check.PACKAGE} 0.8.1; "
        f"verified sha256 {check.WHEEL_SHA256}; {check.ISSUE_URL}\n"
    )


@pytest.mark.parametrize("stdout,stderr,status", [
    (check.KNOWN_FAILURE + "\nanother conflict\n", "", 1),
    (check.KNOWN_FAILURE + "\n\n", "", 1),
    (check.KNOWN_FAILURE + "\n", "warning\n", 1),
    (check.KNOWN_FAILURE + "\n", "", 2),
    (check.KNOWN_FAILURE + "\n", "", 0),
    ("", "", 1),
    ("unexpected success text\n", "", 0),
])
def test_other_results_fail_without_download(monkeypatch, capsys, stdout, stderr, status):
    pip_result(monkeypatch, stdout, stderr, status)
    monkeypatch.setattr(check, "verify_known_wheel", lambda: pytest.fail("must not verify other errors"))
    assert check.main() == 1
    output = capsys.readouterr()
    assert output.out == stdout
    assert output.err.startswith(stderr)
    assert "Dependency check failed" in output.err


@pytest.mark.parametrize("failure", ["hash", "platform", "version", "tag", "download", "payload", "missing"])
def test_exception_requires_original_wheel(wheel_env, monkeypatch, capsys, failure):
    pip_result(monkeypatch, check.KNOWN_FAILURE + "\n")
    if failure == "hash":
        monkeypatch.setattr(check, "WHEEL_SHA256", "0" * 64)
    elif failure == "platform":
        monkeypatch.setattr(check.platform, "machine", lambda: "x86_64")
    elif failure == "version":
        wheel_env.version = "0.9.0"
    elif failure == "tag":
        wheel_env.read_text = lambda name: "Tag: py3-none-manylinux2014_aarch64\n"
    elif failure == "download":
        def unavailable(path):
            raise OSError("download unavailable")
        monkeypatch.setattr(check, "download_wheel", unavailable)
    elif failure == "payload":
        wheel_env.locate_file("nvidia/cusparselt/lib/libcusparseLt.so.0").write_bytes(b"changed")
    elif failure == "missing":
        wheel_env.locate_file("nvidia/cusparselt/lib/libcusparseLt.so.0").unlink()
    assert check.main() == 1
    output = capsys.readouterr()
    assert output.out == check.KNOWN_FAILURE + "\n"
    assert "Could not verify" in output.err


@pytest.mark.parametrize("member", ["../escape", "/absolute", "a\\b", "a//b", f"{check.DIST_INFO}/WHEEL"])
def test_unsafe_or_duplicate_archive_member_fails(wheel_env, monkeypatch, tmp_path, member):
    archive_path = tmp_path / "fixture.whl"
    # Fixture hashes stand in for the pinned upstream hash to exercise ZIP checks.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive_path, "a") as archive:
            archive.writestr(member, b"bad member")
    monkeypatch.setattr(check, "WHEEL_SHA256", hashlib.sha256(archive_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="Unsafe or duplicate"):
        check.verify_known_wheel()


def test_modified_installed_metadata_fails(wheel_env):
    wheel_env.locate_file(f"{check.DIST_INFO}/METADATA").write_text("changed metadata")
    with pytest.raises(ValueError, match="Installed file differs"):
        check.verify_known_wheel()


def test_non_linux_platform_fails(wheel_env, monkeypatch):
    monkeypatch.setattr(check.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="Linux aarch64"):
        check.verify_known_wheel()
