"""Packaged resource bytes are bound to a data-only, source-covered manifest."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dgx_monarch.studio_resource_validation import validate_studio_resources

PACKAGE = Path(__file__).resolve().parents[1] / "src/dgx_monarch"


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "package"
    for relative in ("studio/data", "studio/static"):
        shutil.copytree(PACKAGE / relative, root / relative)
    shutil.copy2(PACKAGE / "studio/resource_hashes.py", root / "studio/resource_hashes.py")
    return root


def test_checked_in_manifest_covers_exact_current_bundle():
    table = validate_studio_resources(str(PACKAGE))
    assert table["studio/static/index.html"] == hashlib.sha256((PACKAGE / "studio/static/index.html").read_bytes()).hexdigest()
    assert set(table) >= {"studio/data/parity.json", "studio/data/workflows.json", "studio/data/presentation.json"}


@pytest.mark.parametrize("kind", ["extra", "changed", "symlink", "missing"])
def test_bad_packaged_resource_refuses(package, tmp_path, kind):
    target = package / "studio/data/presentation.json"
    if kind == "extra":
        (package / "studio/data/user-secret.txt").write_text("must not delete")
    elif kind == "changed":
        target.write_text("{}")
    elif kind == "missing":
        target.unlink()
    else:
        original = tmp_path / "original"
        original.write_text("must not follow")
        target.unlink()
        target.symlink_to(original)
    with pytest.raises((RuntimeError, OSError)):
        validate_studio_resources(str(package))


def test_manifest_python_is_never_executed(package, tmp_path):
    sentinel = tmp_path / "executed"
    (package / "studio/resource_hashes.py").write_text(f"RESOURCE_SHA256 = __import__('pathlib').Path({str(sentinel)!r}).touch()\n")
    with pytest.raises(RuntimeError, match="nonliteral"):
        validate_studio_resources(str(package))
    assert not sentinel.exists()


def test_partial_cleanup_can_only_accept_matching_present_resources(package):
    (package / "studio/data/presentation.json").unlink()
    assert validate_studio_resources(str(package), False)
    (package / "studio/data/parity.json").write_text("tamper")
    with pytest.raises(RuntimeError, match="content differs"):
        validate_studio_resources(str(package), False)


def test_legacy_source_only_package_does_not_need_studio_manifest(tmp_path):
    assert validate_studio_resources(str(tmp_path)) == {}


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO boundary")
def test_manifest_named_fifo_is_refused_without_blocking(package):
    target = package / "studio/data/parity.json"
    target.unlink()
    os.mkfifo(target)
    probe = """
import sys
from dgx_monarch.studio_resource_validation import validate_studio_resources
try:
    validate_studio_resources(sys.argv[1])
except RuntimeError as error:
    assert str(error) == 'unsafe Studio resource'
    print('refused')
else:
    raise AssertionError('FIFO accepted')
"""
    result = subprocess.run([sys.executable, "-c", probe, str(package)],
                            capture_output=True, text=True, timeout=5, check=True)
    assert result.stdout.strip() == "refused"
