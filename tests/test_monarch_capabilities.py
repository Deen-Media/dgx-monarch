"""Native capability proof cannot be satisfied by public RDMA fallback stubs."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import monarch_capabilities as capabilities


@pytest.fixture
def native_modules(monkeypatch, tmp_path):
    action = type("_RdmaAction", (), {"__module__": "monarch._rust_bindings.rdma"})
    paths = {name: tmp_path / name for name in ("torch.py", "monarch.py", "native.so")}
    modules = {
        "torch": SimpleNamespace(__file__=str(paths["torch.py"]),
                                 __version__="2.12.0+cu132",
                                 version=SimpleNamespace(cuda="13.2")),
        "monarch": SimpleNamespace(__file__=str(paths["monarch.py"])),
        "monarch._rust_bindings": SimpleNamespace(__file__=str(paths["native.so"])),
        "monarch._rust_bindings.rdma": SimpleNamespace(_RdmaAction=action),
    }
    distributions = {
        name: SimpleNamespace(
            version=version, files=list(paths), locate_file=lambda entry: paths[entry],
        )
        for name, version in (("torch", "2.12.0+cu132"), ("torchmonarch", "0.7.0"))
    }
    monkeypatch.setattr(capabilities.importlib.metadata, "distribution", distributions.__getitem__)
    monkeypatch.setattr(capabilities.importlib, "import_module", modules.__getitem__)
    return modules


def test_native_load_reports_build_and_redacts_paths_without_hardware(native_modules, tmp_path):
    result = capabilities.collect_capabilities()
    assert not capabilities.capability_blockers(result, expected_version="0.7.0")
    assert result["torch"]["runtime_version"] == "2.12.0+cu132"
    assert result["torch"]["cuda_build"] == "13.2"
    assert result["rdma"] == {"native_import": "ok", "empty_action": "ok",
                              "hardware": "not_probed"}
    assert str(tmp_path) not in json.dumps(result)
    # Ownership is the only origin fact: a path digest is reversible by guessing.
    for origin in (result["torch"]["package"], result["torchmonarch"]["package"],
                   result["torchmonarch"]["native"]):
        assert origin == {"distribution_owned": True}
    # pip's RECORD hashes console scripts whose shebangs name the venv path.
    assert set(result["torchmonarch"]).isdisjoint(_INSTALL_PATH_KEYS)


def test_missing_module_file_reports_no_path_fact(native_modules):
    native_modules["monarch._rust_bindings"].__file__ = None
    result = capabilities.collect_capabilities()
    assert result["torchmonarch"]["native"] == {"distribution_owned": False}


# Facts that vary with the install path: a digest of a resolved module path,
# and a digest of the installer-written RECORD.
_INSTALL_PATH_KEYS = frozenset({"origin_sha256", "record_sha256"})


def _keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _keys(value)


def test_committed_pin_snapshots_carry_no_install_path_digest():
    # CI cannot recompute a digest of the host path that produced a snapshot,
    # so the committed shape itself must exclude one.
    fixtures = sorted((Path(__file__).parent / "fixtures").glob("torchmonarch_pin_*.json"))
    assert fixtures
    for fixture in fixtures:
        snapshot = json.loads(fixture.read_text())
        assert set(_keys(snapshot)).isdisjoint(_INSTALL_PATH_KEYS), fixture.name
        capabilities_seen = snapshot.get("capabilities")
        if capabilities_seen is not None:
            assert not capabilities.capability_blockers(
                capabilities_seen, expected_version=snapshot["version"])


def test_public_fallback_cannot_prove_native_capability(native_modules, monkeypatch):
    native_modules["monarch.rdma"] = SimpleNamespace(RDMAAction=lambda: None)
    original = native_modules.__getitem__

    def load(name):
        if name == "monarch._rust_bindings.rdma":
            raise ImportError("private /host/path.so is absent")
        return original(name)

    monkeypatch.setattr(capabilities.importlib, "import_module", load)
    result = capabilities.collect_capabilities()
    assert "native RDMA bindings are unavailable" in capabilities.capability_blockers(result)
    assert "private /host/path" not in json.dumps(result)


def test_loaded_binding_with_broken_constructor_is_rejected(native_modules):
    def broken():
        raise RuntimeError("constructor failed")

    native_modules["monarch._rust_bindings.rdma"]._RdmaAction = broken
    result = capabilities.collect_capabilities()
    assert result["rdma"]["native_import"] == "ok"
    assert "native RDMA action construction is unavailable" in capabilities.capability_blockers(result)


def test_import_shadow_and_version_mismatch_are_rejected(native_modules):
    native_modules["monarch"].__file__ = "/unowned/shadow/monarch.py"
    result = capabilities.collect_capabilities()
    blockers = capabilities.capability_blockers(result, expected_version="0.6.0")
    assert "torchmonarch package import is not owned by its distribution" in blockers
    assert "torchmonarch capability version differs from surface snapshot" in blockers


@pytest.mark.parametrize("snapshot", [None, {}, {"schema": 0}, {"schema": 1}])
def test_absent_or_incomplete_capability_evidence_fails_closed(snapshot):
    assert capabilities.capability_blockers(snapshot)


def test_cached_status_facts_are_collected_once_and_cannot_be_mutated(monkeypatch):
    calls = []
    monkeypatch.setattr(capabilities, "collect_capabilities", lambda: calls.append(1) or {"rdma": {"hardware": "not_probed"}})
    capabilities._process_capabilities.cache_clear()
    try:
        first = capabilities.cached_capabilities()
        first["rdma"]["hardware"] = "incorrect"
        assert capabilities.cached_capabilities()["rdma"]["hardware"] == "not_probed"
        assert calls == [1]
    finally:
        capabilities._process_capabilities.cache_clear()


def test_bootstrap_admission_never_imports_torch(native_modules, monkeypatch):
    import sys

    from dgx_monarch import TORCHMONARCH_PIN

    monkeypatch.setattr(capabilities.importlib.metadata, "distribution", lambda _: SimpleNamespace(
        version=TORCHMONARCH_PIN, files=["monarch", "monarch._rust_bindings"],
        locate_file=lambda name: native_modules[name].__file__,
    ))
    for name in ("monarch", "monarch._rust_bindings"):
        monkeypatch.setitem(sys.modules, name, native_modules[name])
    monkeypatch.setattr(capabilities.importlib, "import_module", lambda name: pytest.fail(f"unexpected import {name}"))
    capabilities.assert_supported_runtime()


def test_bootstrap_admission_rejects_old_runtime(native_modules, monkeypatch):
    monkeypatch.setattr(capabilities.importlib.metadata, "distribution", lambda _: SimpleNamespace(version="0.6.0"))
    with pytest.raises(RuntimeError, match="supported pin"):
        capabilities.assert_supported_runtime()


def test_diagnostic_identity_caches_metadata_but_reads_current_policy(monkeypatch):
    from dgx_monarch import monarch_policy, runtime_provenance

    calls = []
    policy = {"transport_sha256": "a" * 64}
    monkeypatch.setattr(capabilities.importlib.metadata, "version", lambda name: calls.append(name) or "test")
    monkeypatch.setattr(runtime_provenance, "cached_dgx_source_manifest_sha256", lambda: "b" * 64)
    monkeypatch.setattr(monarch_policy, "snapshot", lambda: dict(policy))
    monkeypatch.setattr(capabilities.importlib, "import_module", lambda name: pytest.fail(f"unexpected import {name}"))
    capabilities._diagnostic_base.cache_clear()
    try:
        first = capabilities.diagnostic_identity()
        assert first["source_manifest_sha256"] == "b" * 64
        assert first["transport_sha256"] == "a" * 64
        assert set(first) == {"source_manifest_sha256", "transport_sha256", "torchmonarch", "torch", "dgx_monarch"}
        first["torch"] = "changed"
        policy["transport_sha256"] = "c" * 64
        second = capabilities.diagnostic_identity()
        assert second["torch"] == "test"
        assert second["transport_sha256"] == "c" * 64
        assert calls == ["torch", "torchmonarch"]
    finally:
        capabilities._diagnostic_base.cache_clear()


def _installed(monkeypatch, native_modules, version, *, files=("monarch", "monarch._rust_bindings")):
    """What the disk reads now: torchmonarch at ``version``, owning ``files``."""
    monkeypatch.setattr(capabilities.importlib.metadata, "distribution", lambda _: SimpleNamespace(
        version=version, files=list(files), locate_file=lambda name: native_modules[name].__file__))


def test_the_version_read_after_the_import_is_kept_whatever_the_disk_reads_later(
        native_modules, monkeypatch):
    """A refused admission keeps its reading: installing the pin under the
    running process changes the disk, not the loaded code, so it still
    refuses and asks for a restart. An install on top of an admitted pin
    changes nothing either, ownership verdict included."""
    import sys

    from dgx_monarch import TORCHMONARCH_PIN

    _installed(monkeypatch, native_modules, TORCHMONARCH_PIN)
    for name in ("monarch", "monarch._rust_bindings"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(RuntimeError, match="not owned"):
        capabilities.loaded_version()  # monarch not imported yet: this reading is not kept
    for name in ("monarch", "monarch._rust_bindings"):
        monkeypatch.setitem(sys.modules, name, native_modules[name])
    refusals = []
    for disk in ("0.0.1", TORCHMONARCH_PIN):
        _installed(monkeypatch, native_modules, disk)
        with pytest.raises(RuntimeError) as refused:
            capabilities.assert_supported_runtime()
        refusals.append(str(refused.value))
        assert capabilities.loaded_version() == "0.0.1"
    for text in refusals:
        assert "this process loaded torchmonarch 0.0.1" in text and "restart ComfyUI" in text
        # A checkout updated under the running process holds the old pin in memory:
        # reinstalling that pin would be the wrong first step.
        assert ("If dgx-monarch was updated since this process started, restart it before "
                "anything else; otherwise install the pin if it is missing") in text

    monkeypatch.setattr(capabilities, "_LOADED", None)
    _installed(monkeypatch, native_modules, TORCHMONARCH_PIN)
    assert capabilities.assert_supported_runtime() == TORCHMONARCH_PIN
    _installed(monkeypatch, native_modules, "0.0.1", files=())
    assert capabilities.assert_supported_runtime() == TORCHMONARCH_PIN
    assert capabilities.loaded_version() == TORCHMONARCH_PIN


def _maps(tmp_path, monkeypatch, *lines):
    path = tmp_path / "maps"
    path.write_text("".join(line + "\n" for line in lines))
    monkeypatch.setattr(capabilities, "_MAPS", str(path))


def _mapping(path, stat, *, inode=None, removed=False):
    device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}"
    return (f"ffff0000-ffff1000 r-xp 00000000 {device} {stat.st_ino if inode is None else inode}"
            f"          {path}" + (" (deleted)" if removed else ""))


def test_a_mapped_binding_is_checked_against_the_file_on_disk(native_modules, tmp_path, monkeypatch):
    """The four readings of /proc/self/maps the check takes apart, on a file it can stat."""
    binding = native_modules["monarch._rust_bindings"]
    located = Path(binding.__file__).resolve()
    located.write_bytes(b"binding")
    stat = located.stat()
    other = f"/elsewhere/{located.name}"
    _maps(tmp_path, monkeypatch, _mapping(located, stat), _mapping(other, stat, inode=1))
    assert capabilities._binding_replaced(binding) is False  # live at its path; another package's ignored
    _maps(tmp_path, monkeypatch, _mapping(located, stat, inode=stat.st_ino + 1))
    assert capabilities._binding_replaced(binding) is True   # same path and device, a new inode
    _maps(tmp_path, monkeypatch, _mapping(f"{tmp_path}/~stash/{located.name}", stat, removed=True))
    assert capabilities._binding_replaced(binding) is True   # moved aside by pip, then removed
    _maps(tmp_path, monkeypatch, _mapping(other, stat, inode=1))
    assert capabilities._binding_replaced(binding) is None   # nothing says either way
    monkeypatch.setattr(capabilities, "_MAPS", str(tmp_path / "absent"))
    assert capabilities._binding_replaced(binding) is None


def test_without_proc_the_first_reading_keeps_the_installed_version(native_modules, tmp_path, monkeypatch):
    """Where /proc cannot be read the reading is the installed version, as before the check."""
    import sys

    from dgx_monarch import TORCHMONARCH_PIN

    for name in ("monarch", "monarch._rust_bindings"):
        Path(native_modules[name].__file__).write_bytes(b"module")
        monkeypatch.setitem(sys.modules, name, native_modules[name])
    _installed(monkeypatch, native_modules, TORCHMONARCH_PIN)
    monkeypatch.setattr(capabilities, "_MAPS", str(tmp_path / "absent"))
    assert capabilities.assert_supported_runtime() == TORCHMONARCH_PIN


def test_a_replaced_binding_and_a_failed_reading_both_ask_for_a_restart(native_modules, tmp_path, monkeypatch):
    import sys

    from dgx_monarch import TORCHMONARCH_PIN

    binding = native_modules["monarch._rust_bindings"]
    for name in ("monarch", "monarch._rust_bindings"):
        Path(native_modules[name].__file__).write_bytes(b"module")
        monkeypatch.setitem(sys.modules, name, native_modules[name])
    located = Path(binding.__file__).resolve()
    _maps(tmp_path, monkeypatch, _mapping(located, located.stat(), removed=True))
    _installed(monkeypatch, native_modules, TORCHMONARCH_PIN)
    for call in (capabilities.assert_supported_runtime, capabilities.loaded_version):
        with pytest.raises(RuntimeError, match="reinstalled after this process loaded it") as refused:
            call()
        assert "restart ComfyUI" in str(refused.value)
        # True after a reinstall of the same version too.
        assert "so the file it runs is no longer the one installed" in str(refused.value)
        assert "version is not the one it runs" not in str(refused.value)
    monkeypatch.setattr(capabilities, "_LOADED", None)
    monkeypatch.setattr(capabilities.importlib.metadata, "distribution", lambda _: 1 / 0)
    for call in (capabilities.assert_supported_runtime, capabilities.loaded_version):
        with pytest.raises(RuntimeError, match="identity is unavailable; this process keeps") as refused:
            call()
        assert "restart ComfyUI" in str(refused.value)
