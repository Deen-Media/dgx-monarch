"""Preserved core links are evidence only, never write/uninstall permission."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import sol_install_payload as payload
from dgx_monarch.cli.sol_install_protected_links import ProtectedLeafError, protected_leaf_fingerprint


def fixture(tmp_path, monkeypatch, name="nvidia-nccl-cu13"):
    site = tmp_path / "venv/site"
    record = "nvidia/nccl/lib/libnccl.so.2"
    logical = site / record
    logical.parent.mkdir(parents=True)
    external = tmp_path / "external/libnccl.so.2"
    external.parent.mkdir()
    external.write_bytes(b"preserved existing core library")
    logical.symlink_to(external)
    distribution = SimpleNamespace(metadata={"Name": name}, files=[record],
                                   locate_file=lambda item: site / item, version="2.0")
    monkeypatch.setattr(payload, "scheme_roots", lambda: {site})
    monkeypatch.setattr(payload, "distribution_inventory", lambda: {name: distribution})
    monkeypatch.setattr(payload, "installed", lambda: {"torch": "2.12.0", name: "2.0"})
    return logical, external, distribution, record


def test_stable_external_core_leaf_hashes_both_identities_and_preserves_collision_paths(tmp_path, monkeypatch):
    logical, external, _, _ = fixture(tmp_path, monkeypatch)
    before_link, before_target = logical.lstat(), external.stat()
    first, target = protected_leaf_fingerprint(logical)
    assert protected_leaf_fingerprint(logical) == (first, external)
    assert target == external
    digest, paths = payload.protected_state()
    assert len(digest) == 64 and paths == {str(logical), str(external)}
    assert payload.preserved_paths(()) == {str(logical), str(external)}
    assert logical.lstat().st_ino == before_link.st_ino
    assert external.stat().st_ino == before_target.st_ino
    assert logical.lstat().st_mode == before_link.st_mode and external.stat().st_mode == before_target.st_mode
    for path in (logical, external):
        with pytest.raises(RuntimeError):
            payload.check_destination(str(path), paths, ())


def test_link_retarget_during_read_refuses(tmp_path, monkeypatch):
    logical, _, _, _ = fixture(tmp_path, monkeypatch)
    other = tmp_path / "other-library"
    other.write_bytes(b"preserved existing core library")
    read = os.read
    changed = False
    def raced(fd, size):
        nonlocal changed
        chunk = read(fd, size)
        if chunk and not changed:
            changed = True
            logical.unlink()
            logical.symlink_to(other)
        return chunk
    monkeypatch.setattr(os, "read", raced)
    with pytest.raises(ProtectedLeafError, match="changed"):
        protected_leaf_fingerprint(logical)


@pytest.mark.parametrize("replacement", [False, True])
def test_target_content_or_inode_change_during_read_refuses(tmp_path, monkeypatch, replacement):
    logical, target, _, _ = fixture(tmp_path, monkeypatch)
    read = os.read
    changed = False
    def raced(fd, size):
        nonlocal changed
        chunk = read(fd, size)
        if chunk and not changed:
            changed = True
            if replacement:
                new = target.with_name("replacement")
                new.write_bytes(target.read_bytes())
                new.replace(target)
            else:
                target.write_bytes(b"changed core library")
        return chunk
    monkeypatch.setattr(os, "read", raced)
    with pytest.raises(ProtectedLeafError, match="changed"):
        protected_leaf_fingerprint(logical)


@pytest.mark.parametrize("kind", ["fifo", "directory"])
def test_special_target_refuses_before_open(tmp_path, monkeypatch, kind):
    logical, target, _, _ = fixture(tmp_path, monkeypatch)
    target.unlink()
    if kind == "fifo":
        os.mkfifo(target)
    else:
        target.mkdir()
    monkeypatch.setattr(os, "open", lambda *args: pytest.fail("special target must not be opened"))
    with pytest.raises(ProtectedLeafError, match="not a regular file"):
        protected_leaf_fingerprint(logical)


def test_ancestor_link_remains_refused(tmp_path, monkeypatch):
    logical, external, distribution, record = fixture(tmp_path, monkeypatch)
    logical.unlink()
    logical.parent.rmdir()
    logical.parent.symlink_to(external.parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="crosses a symlink"):
        payload.inventory_path(distribution, record, preserved_core_leaf=True)
    with pytest.raises(ProtectedLeafError, match="symlink ancestor"):
        protected_leaf_fingerprint(logical)


def test_changing_and_replacement_core_inventory_never_gain_link_authority(tmp_path, monkeypatch):
    _, _, distribution, record = fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="crosses a symlink"):
        payload.inventory_path(distribution, record)
    with pytest.raises(RuntimeError, match="crosses a symlink"):
        payload.preserved_paths({"nvidia-nccl-cu13"})
    with pytest.raises(RuntimeError, match="crosses a symlink"):
        payload.replacement_paths("nvidia-nccl-cu13")


def test_nonprotected_preserved_distribution_cannot_use_the_exception(tmp_path, monkeypatch):
    _, _, distribution, record = fixture(tmp_path, monkeypatch, "ordinary-extension")
    with pytest.raises(RuntimeError, match="crosses a symlink"):
        payload.inventory_path(distribution, record, preserved_core_leaf=True)
    with pytest.raises(RuntimeError, match="crosses a symlink"):
        payload.preserved_paths(())
