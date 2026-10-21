"""Packaged resource bytes are bound to a data-only, source-covered manifest."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dgx_monarch.studio_resource_validation import (
    check_packaged_studio_resources,
    studio_resource_validation_source,
    validate_studio_resources,
)

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


def test_packaged_check_reads_the_same_table_through_importlib_resources():
    assert check_packaged_studio_resources() == validate_studio_resources(str(PACKAGE))


def test_packaged_check_skips_only_the_ownership_policy(package, monkeypatch):
    # A correct install owned by root looks like this to the account that runs --check.
    foreign = os.geteuid() + 1
    monkeypatch.setattr(os, "geteuid", lambda: foreign)
    with pytest.raises(RuntimeError, match="unsafe Studio resource"):
        validate_studio_resources(str(package))
    assert check_packaged_studio_resources(package) == check_packaged_studio_resources()


@pytest.mark.parametrize(("kind", "message"), [
    ("extra", "unexpected studio/data/user-secret.txt"),
    ("changed", "differ from their manifest hash: studio/data/presentation.json"),
    ("missing", "missing studio/data/presentation.json"),
    ("missing_asset", "missing studio/static/assets/"),
    ("no_manifest", "studio/resource_hashes.py is missing"),
    ("unlisted_path", "unsupported path or hash"),
    ("incomplete", "incomplete inventory"),
])
def test_packaged_check_refuses_every_drift_by_name(package, kind, message):
    manifest = package / "studio/resource_hashes.py"
    if kind == "extra":
        (package / "studio/data/user-secret.txt").write_text("must be reported")
    elif kind == "changed":
        (package / "studio/data/presentation.json").write_text("{}")
    elif kind == "missing":
        (package / "studio/data/presentation.json").unlink()
    elif kind == "missing_asset":
        next((package / "studio/static/assets").glob("*.css")).unlink()
    elif kind == "no_manifest":
        manifest.unlink()
    else:
        table = validate_studio_resources(str(PACKAGE))
        if kind == "unlisted_path":
            table = {**table, "studio/data/other.json": "0" * 64}
        else:
            table.pop("studio/data/workflows.json")
        manifest.write_text(f"RESOURCE_SHA256 = {table!r}\n")
    with pytest.raises(RuntimeError, match=re.escape(message)):
        check_packaged_studio_resources(package)


def test_packaged_check_never_executes_the_manifest(package, tmp_path):
    sentinel = tmp_path / "executed"
    (package / "studio/resource_hashes.py").write_text(f"RESOURCE_SHA256 = __import__('pathlib').Path({str(sentinel)!r}).touch()\n")
    with pytest.raises(RuntimeError, match="nonliteral"):
        check_packaged_studio_resources(package)
    assert not sentinel.exists()


def test_embedded_validator_source_does_not_carry_the_check_only_reader():
    source = studio_resource_validation_source()
    assert source.startswith("def validate_studio_resources(")
    assert "check_packaged_studio_resources" not in source and "_REQUIRED_RESOURCES" not in source


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
