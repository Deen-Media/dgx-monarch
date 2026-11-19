from __future__ import annotations

import base64
import csv
import hashlib
import io
import sys

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


@pytest.mark.parametrize(
    ("path", "reason"),
    [("monarch/runtime.py", "pinned_payload_claim"), ("monarch/extra.py", "protected_package_overlap")],
)
def test_conflict_identifies_distribution_and_relative_path(payload, path, reason):
    site, expected = payload
    install(site, "other-pkg", {path: b"foreign"})
    with pytest.raises(PayloadOwnershipError) as caught:
        _scan_tree(site, expected)
    diagnostic = caught.value.diagnostic
    assert diagnostic.reason == reason
    assert diagnostic.distribution == "other-pkg"
    assert diagnostic.relative_path == path
    assert str(site) not in diagnostic.terminal_message()


def test_invalid_inventory_identifies_distribution_record(payload):
    site, expected = payload
    info, rows = install(site, "other", {"examples/foreign.py": b"foreign"})
    rows.append(["../../../private/secret", "", ""])
    write_records(info, rows)
    with pytest.raises(PayloadOwnershipError) as caught:
        _scan_tree(site, expected)
    assert caught.value.diagnostic.reason == "record_outside_site"
    assert caught.value.diagnostic.distribution == "other"
    assert caught.value.diagnostic.relative_path == "other-1.dist-info/RECORD"
    assert "secret" not in caught.value.diagnostic.terminal_message()


# pip under PYTHONPYCACHEPREFIX (docs/TROUBLESHOOTING.md #155): it records each
# .pyc it compiled with no hash and no size, under the prefix, outside site-packages.
TAG = sys.implementation.cache_tag
PREFIX = "../../../../scratch/build/cache/pycache/home/user/env/lib/python3.12/site-packages"
FOREIGN = {"examples/foreign/demo.py": b"foreign", "examples/top.py": b"top"}


def _prefix_install(site, extra=()):
    info, rows = install(site, "xfuser", FOREIGN)
    rows += [[f"{PREFIX}/examples/foreign/demo.{TAG}.pyc", "", ""],
             [f"{PREFIX}/examples/top.{TAG}.pyc", "", ""],
             [f"{PREFIX}/examples/foreign/demo.{TAG}.opt-1.pyc", "", ""],
             *extra]
    write_records(info, rows)
    return info, rows


def test_pip_bytecode_under_a_pycache_prefix_does_not_void_a_foreign_record(payload):
    """The 2026-11-06 refusal: every row outside site-packages is pip's
    bytecode for one of the RECORD's own in-site sources. The check reads none
    of them, still verifies every in-site file, and the pin digest stays put."""
    site, expected = payload
    baseline = site_payload_digest(site, "1")
    _prefix_install(site)
    assert _scan_tree(site, expected) == expected
    assert site_payload_digest(site, "1") == baseline


@pytest.mark.parametrize("row", [
    [f"{PREFIX}/evil.py", "sha256=" + "A" * 43, "10"],          # payload outside site-packages
    ["../../../../scratch/x/monarch.pyc", "", ""],              # a name an import could load
    [f"{PREFIX}/examples/top.{TAG}.opt-2.pyc", "sha256=" + "B" * 43, "12"],  # hashed bytecode
    [f"{PREFIX}/nosuch/mod.{TAG}.pyc", "", ""],                 # bytecode for no recorded source
    ["../../../../scratch/x/evil.pth", "", ""],                 # a path hook
    [f"{PREFIX}/examples/foreign/demo.cpython-27.pyc", "", ""],  # another interpreter's tag
    [f"../examples/foreign/demo.{TAG}.pyc", "", ""],            # nothing before the source directory
    [f"../../x/../examples/foreign/demo.{TAG}.pyc", "", ""],    # a parent step after the prefix
])
def test_any_other_row_outside_site_packages_still_voids_the_record(payload, row):
    site, expected = payload
    _prefix_install(site, extra=[row])
    with pytest.raises(PayloadOwnershipError) as caught:
        _scan_tree(site, expected)
    assert caught.value.diagnostic.reason == "record_outside_site"
    assert caught.value.diagnostic.relative_path == "xfuser-1.dist-info/RECORD"


def test_a_prefix_record_does_not_excuse_an_altered_in_site_file(payload):
    site, expected = payload
    _prefix_install(site)
    (site / "examples/top.py").write_bytes(b"altered")
    with pytest.raises(PayloadOwnershipError) as caught:
        _scan_tree(site, expected)
    assert caught.value.diagnostic.reason == "record_mismatch"


def test_a_malformed_or_repeated_record_and_a_second_copy_get_their_own_reasons(payload):
    site, expected = payload
    info, rows = _prefix_install(site, extra=[["examples/top.py", "x"]])
    with pytest.raises(PayloadOwnershipError) as caught:
        _scan_tree(site, expected)
    assert caught.value.diagnostic.reason == "record_malformed"
    write_records(info, rows[:-1])
    install(site, "XFuser", {"unrelated/other.py": b"elsewhere"})
    with pytest.raises(PayloadOwnershipError) as caught:
        _scan_tree(site, expected)
    assert caught.value.diagnostic.reason == "distribution_duplicated"


# One named wheel shares one pinned path (xfuser 0.8.0 and tests/conftest.py in
# production); synthetic bytes here, bound the way the production entry is.
PINNED_CONFTEST, FOREIGN_CONFTEST = b"PINNED = True\n", b"FOREIGN = True\n"


@pytest.fixture
def shared(tmp_path, monkeypatch):
    from dgx_monarch.cli import update_payload_ownership as ownership

    files = {"monarch/__init__.py": b"", "monarch/runtime.py": b"pinned",
             "tests/conftest.py": PINNED_CONFTEST, "tests/test_pinned.py": b"pinned test"}
    install(tmp_path, "torchmonarch", files)
    expected = {path: digest(data) for path, data in files.items()}
    baseline = site_payload_digest(tmp_path, "1")
    official = tuple(record("tests/conftest.py", FOREIGN_CONFTEST)[1:])
    monkeypatch.setattr(ownership, "_SHARED_PINNED", {("xfuser", "1"): {"tests/conftest.py": official}},
                        raising=False)
    info, rows = install(tmp_path, "xfuser", {"tests/conftest.py": FOREIGN_CONFTEST,
                                              "tests/xfuser_test.py": b"foreign test"})
    return tmp_path, expected, baseline, info, rows


@pytest.mark.parametrize("last", ["xfuser", "torchmonarch"])
def test_the_named_wheel_may_share_its_one_path_whichever_installs_last(shared, last):
    """pip lets the last install win the file. Either way the scan matches the
    exact pin and the digest is the one torchmonarch alone gives, so a Worker
    on either order attests the same release."""
    site, expected, baseline, _info, _rows = shared
    (site / "tests/conftest.py").write_bytes(FOREIGN_CONFTEST if last == "xfuser" else PINNED_CONFTEST)
    assert _scan_tree(site, expected) == expected
    assert site_payload_digest(site, "1") == baseline


def test_a_shared_path_that_vanished_refuses_and_asks_for_torchmonarch(shared):
    """Uninstalling the other wheel removes the file both record; the driver's
    scan and the Worker's digest both name it, with the reinstall remedy."""
    site, expected, _baseline, info, _rows = shared
    (site / "tests/conftest.py").unlink()
    for check in (lambda: _scan_tree(site, expected), lambda: site_payload_digest(site, "1")):
        with pytest.raises(PayloadOwnershipError) as caught:
            check()
        assert caught.value.diagnostic.reason == "shared_payload_missing"
        assert caught.value.diagnostic.relative_path == "tests/conftest.py"
    import shutil

    shutil.rmtree(info)  # the other wheel gone too: still refused the same way
    with pytest.raises(PayloadOwnershipError) as caught:
        site_payload_digest(site, "1")
    assert caught.value.diagnostic.reason == "shared_payload_missing"


@pytest.mark.parametrize(("change", "reason", "path"), [
    ("neither_bytes", "record_mismatch", "tests/conftest.py"),
    ("other_hash", "pinned_payload_claim", "tests/conftest.py"),
    ("runtime_claim", "pinned_payload_claim", "monarch/runtime.py"),
    ("second_test_claim", "pinned_payload_claim", "tests/test_pinned.py"),
    ("other_version", "pinned_payload_claim", "tests/conftest.py"),
    ("third_claimant", "pinned_payload_claim", "tests/conftest.py"),
])
def test_what_the_shared_path_rule_still_refuses(shared, change, reason, path):
    site, expected, _baseline, info, rows = shared
    if change == "neither_bytes":
        (site / "tests/conftest.py").write_bytes(b"NEITHER\n")
    elif change == "other_hash":
        rows[1] = record("tests/conftest.py", b"another build\n")
    elif change == "runtime_claim":
        rows.append(record("monarch/runtime.py", b"pinned"))
    elif change == "second_test_claim":
        rows.append(record("tests/test_pinned.py", b"pinned test"))
    elif change == "other_version":
        metadata = b"Metadata-Version: 2.1\nName: xfuser\nVersion: 2\n"
        (info / "METADATA").write_bytes(metadata)
        rows[0] = record(f"{info.name}/METADATA", metadata)
    elif change == "third_claimant":
        install(site, "other", {"tests/conftest.py": FOREIGN_CONFTEST})
    write_records(info, rows)
    with pytest.raises(PayloadOwnershipError) as caught:
        _scan_tree(site, expected)
    assert (caught.value.diagnostic.reason, caught.value.diagnostic.relative_path) == (reason, path)
