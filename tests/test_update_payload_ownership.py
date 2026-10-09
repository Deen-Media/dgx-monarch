from __future__ import annotations

import base64
import csv
import hashlib
import io

import pytest

from dgx_monarch.cli.update_driver_pin import _manifest_digest, _scan_tree, site_payload_digest
from dgx_monarch.cli.update_payload_ownership import PayloadOwnershipError


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(path, data):
    return [path, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("="), str(len(data))]


def install(site, name, files):
    info = site / f"{name}-1.dist-info"
    info.mkdir()
    metadata = b"Metadata-Version: 2.1\nName: " + name.encode() + b"\nVersion: 1\n"
    (info / "METADATA").write_bytes(metadata)
    rows = [record(f"{info.name}/METADATA", metadata)]
    for path, data in files.items():
        target = site / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows.append(record(path, data))
    write_records(info, rows)
    return info, rows


def write_records(info, rows):
    output = io.StringIO()
    csv.writer(output).writerows(rows)
    (info / "RECORD").write_text(output.getvalue())


@pytest.fixture
def payload(tmp_path):
    files = {"monarch/__init__.py": b"", "monarch/runtime.py": b"pinned", "examples/monarch/demo.py": b"demo"}
    install(tmp_path, "torchmonarch", files)
    return tmp_path, {path: digest(data) for path, data in files.items()}


def test_verified_foreign_namespace_files_do_not_change_pin_digest(payload):
    site, expected = payload
    baseline = site_payload_digest(site, "1")
    install(site, "other", {"examples/foreign/__init__.py": b"", "examples/foreign/demo.py": b"foreign"})
    assert _scan_tree(site, expected) == expected
    assert site_payload_digest(site, "1") == baseline == _manifest_digest(expected)


@pytest.mark.parametrize("path", ["monarch/extra.py", "examples/__init__.py", "examples/monarch/__init__.py", "monarch/__init__.py"])
def test_foreign_claim_cannot_shadow_or_overlap_pinned_package(payload, path):
    site, expected = payload
    install(site, "other", {path: b""})
    with pytest.raises(PayloadOwnershipError):
        _scan_tree(site, expected)


@pytest.mark.parametrize("change", ["hash", "size", "metadata", "blank", "traversal", "duplicate", "ambiguous"])
def test_invalid_foreign_claims_do_not_hide_payload(payload, change):
    site, expected = payload
    info, rows = install(site, "other", {"examples/other.py": b"foreign"})
    if change == "hash":
        (site / "examples/other.py").write_bytes(b"changed")
    elif change == "size":
        rows[-1][2] = "999"
    elif change == "metadata":
        (info / "METADATA").write_bytes(b"Name: other\nVersion: 2\n")
    elif change == "blank":
        rows[-1][1] = ""
    elif change == "traversal":
        rows.append(["examples/../monarch/extra.py", "", ""])
    elif change == "duplicate":
        rows.append(rows[-1])
    elif change == "ambiguous":
        install(site, "another", {"examples/other.py": b"foreign"})
    write_records(info, rows)
    with pytest.raises(PayloadOwnershipError):
        _scan_tree(site, expected)


def test_unowned_extra_and_changed_expected_remain_visible(payload):
    site, expected = payload
    (site / "examples/unowned.py").write_bytes(b"extra")
    (site / "monarch/runtime.py").write_bytes(b"changed")
    found = _scan_tree(site, expected)
    assert found["examples/unowned.py"] == digest(b"extra")
    assert found["monarch/runtime.py"] != expected["monarch/runtime.py"]


def test_foreign_symlink_is_never_excluded(payload):
    site, expected = payload
    install(site, "other", {"examples/other.py": b"foreign"})
    path = site / "examples/other.py"
    path.unlink()
    path.symlink_to(site / "monarch/runtime.py")
    with pytest.raises(RuntimeError):
        _scan_tree(site, expected)


def test_foreign_file_inside_expected_regular_subpackage_is_refused(payload):
    site, expected = payload
    path = site / "examples/monarch/__init__.py"
    path.write_bytes(b"")
    expected["examples/monarch/__init__.py"] = digest(b"")
    install(site, "other", {"examples/monarch/foreign.py": b"foreign"})
    with pytest.raises(PayloadOwnershipError):
        _scan_tree(site, expected)


def test_same_normalized_foreign_distribution_twice_is_ambiguous(payload):
    site, expected = payload
    install(site, "other-pkg", {"examples/foreign.py": b"foreign"})
    install(site, "other_pkg", {"unrelated/other.py": b"elsewhere"})
    with pytest.raises(PayloadOwnershipError):
        _scan_tree(site, expected)


def test_foreign_metadata_symlink_is_refused(payload):
    site, expected = payload
    info, _rows = install(site, "other", {"examples/foreign.py": b"foreign"})
    metadata = info / "METADATA"
    data = metadata.read_bytes()
    metadata.unlink()
    target = site / "original-metadata"
    target.write_bytes(data)
    metadata.symlink_to(target)
    with pytest.raises(OSError):
        _scan_tree(site, expected)


@pytest.mark.parametrize("foreign_path", ["examples/monarch/demo/__init__.py", "examples/monarch/demo.so"])
def test_foreign_import_cannot_shadow_pinned_module(payload, foreign_path):
    site, expected = payload
    install(site, "other", {foreign_path: b"foreign"})
    with pytest.raises(PayloadOwnershipError):
        _scan_tree(site, expected)


def test_expected_extension_package_protects_its_subtree(payload):
    site, expected = payload
    path = site / "examples/monarch/__init__.so"
    path.write_bytes(b"extension")
    expected["examples/monarch/__init__.so"] = digest(b"extension")
    install(site, "other", {"examples/monarch/foreign.py": b"foreign"})
    with pytest.raises(PayloadOwnershipError):
        _scan_tree(site, expected)


def test_foreign_module_cannot_shadow_pinned_namespace(payload):
    site, expected = payload
    install(site, "other", {"examples/monarch.py": b"foreign"})
    with pytest.raises(PayloadOwnershipError):
        _scan_tree(site, expected)
