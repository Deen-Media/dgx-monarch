"""Passive compiler checks use configured-interpreter metadata, never GPU work."""
from __future__ import annotations

import importlib.metadata as metadata
import os
from types import SimpleNamespace

from dgx_monarch.cli import sol_attn_prerequisites as probe


def _metadata(monkeypatch, versions, requirements):
    def version(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    def requires(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return requirements.get(name, [])

    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setattr(metadata, "requires", requires)


def test_transitive_protobuf_constraint_comes_from_installed_metadata(monkeypatch):
    versions = dict.fromkeys(("sol-attn", "apache-tvm-ffi", "nvidia-cutlass-dsl", "compiler"), "1.0")
    versions["protobuf"] = "7.35.0"
    _metadata(monkeypatch, versions, {
        "nvidia-cutlass-dsl": ["compiler>=1"],
        "compiler": ["protobuf>=6.30.2,<7"],
    })
    assert probe.dependency_state() == "conflict:protobuf"
    versions["protobuf"] = "6.33.6"
    assert probe.dependency_state() == "observed"


def test_missing_metadata_and_inactive_markers(monkeypatch):
    versions = dict.fromkeys(("sol-attn", "apache-tvm-ffi", "nvidia-cutlass-dsl"), "1.0")
    _metadata(monkeypatch, versions, {"sol-attn": ['absent; python_version < "2"']})
    assert probe.dependency_state() == "observed"
    versions.pop("nvidia-cutlass-dsl")
    assert probe.dependency_state() == "unknown"


def test_headers_follow_configured_interpreters_include_path(monkeypatch, tmp_path):
    import sysconfig

    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"include": str(tmp_path)})
    assert probe.header_state() == "missing"
    (tmp_path / "Python.h").write_text("/* header */")
    assert probe.header_state() == "present"


def test_cache_ignores_login_override_and_never_creates_paths(monkeypatch, tmp_path):
    import pwd

    monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(tmp_path)))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "shell-override"))
    assert probe.default_cache_state() == "absent"
    assert list(tmp_path.iterdir()) == []
    (tmp_path / ".triton" / "cache").mkdir(parents=True)
    assert probe.default_cache_state() == "parent-access"
    monkeypatch.setattr(os, "access", lambda *args: False)
    assert probe.default_cache_state() == "blocked"


def test_cache_symlink_is_unknown_without_following_it(monkeypatch, tmp_path):
    import pwd

    monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(tmp_path)))
    (tmp_path / ".triton").symlink_to(tmp_path, target_is_directory=True)
    assert probe.default_cache_state() == "unknown"


def test_missing_required_distribution_is_a_named_conflict(monkeypatch):
    versions = dict.fromkeys(("sol-attn", "apache-tvm-ffi", "nvidia-cutlass-dsl"), "1.0")
    _metadata(monkeypatch, versions, {"nvidia-cutlass-dsl": ["protobuf>=6"]})
    assert probe.dependency_state() == "conflict:protobuf"


def test_requested_extra_is_unknown_instead_of_assumed_compatible(monkeypatch):
    versions = dict.fromkeys(("sol-attn", "apache-tvm-ffi", "nvidia-cutlass-dsl"), "1.0")
    _metadata(monkeypatch, versions, {"sol-attn": ["apache-tvm-ffi[compiler]"]})
    assert probe.dependency_state() == "unknown"


def test_oversized_metadata_graph_is_bounded_and_unknown(monkeypatch):
    versions = dict.fromkeys(("sol-attn", "apache-tvm-ffi", "nvidia-cutlass-dsl"), "1.0")
    versions.update(dict.fromkeys((f"dependency{i}" for i in range(80)), "1.0"))
    _metadata(monkeypatch, versions, {"nvidia-cutlass-dsl": [f"dependency{i}" for i in range(80)]})
    assert probe.dependency_state() == "unknown"
