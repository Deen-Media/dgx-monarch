"""Native capability proof cannot be satisfied by public RDMA fallback stubs."""
from __future__ import annotations

import json
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
            read_text=lambda _: "record-evidence",
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
    assert len(result["torchmonarch"]["native"]["origin_sha256"]) == 64


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
