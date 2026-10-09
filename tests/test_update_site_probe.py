from __future__ import annotations

import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dgx_monarch import __version__
from dgx_monarch.cli import update_site_probe
from dgx_monarch.runtime_provenance import dgx_source_manifest_sha256


@pytest.fixture
def releases(tmp_path):
    code = Path(update_site_probe.__file__).parents[1]
    base, target = tmp_path / "controller", tmp_path / "target"
    for site in (base, target):
        shutil.copytree(code, site / "dgx_monarch", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        info = site / "torchmonarch-0.6.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Name: torchmonarch\nVersion: 0.6.0\n")
    (target / "dgx_monarch/release_marker.py").write_text("TARGET = True\n")
    return base, target


def test_probe_reads_target_from_different_frozen_controller_cwd(releases, monkeypatch):
    base, target = releases
    base_hash = dgx_source_manifest_sha256(base / "dgx_monarch")
    target_hash = dgx_source_manifest_sha256(target / "dgx_monarch")
    assert base_hash != target_hash
    monkeypatch.chdir(base)
    monkeypatch.setenv("PYTHONPATH", str(base))
    assert update_site_probe.site_matches(target, __version__, "0.6.0", target_hash, subprocess.run)
    assert not update_site_probe.site_matches(target, __version__, "0.6.0", base_hash, subprocess.run)
    assert update_site_probe.site_matches(base, __version__, "0.6.0", base_hash, subprocess.run)


def test_probe_ignores_unchecked_target_bytecode(releases):
    _base, target = releases
    module = target / "dgx_monarch/runtime_provenance.py"
    contents = module.read_bytes()
    module.write_text("raise SystemExit(23)\n")
    cache = module.parent / "__pycache__" / f"runtime_provenance.{sys.implementation.cache_tag}.pyc"
    py_compile.compile(str(module), cfile=str(cache), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    module.write_bytes(contents)
    expected = dgx_source_manifest_sha256(target / "dgx_monarch")
    assert update_site_probe.site_matches(target, __version__, "0.6.0", expected, subprocess.run)


def test_probe_rejects_package_outside_requested_site(releases, tmp_path):
    base, target = releases
    shutil.rmtree(target / "dgx_monarch")
    (target / "dgx_monarch").symlink_to(base / "dgx_monarch")
    assert not update_site_probe.site_matches(target, __version__, "0.6.0", dgx_source_manifest_sha256(base / "dgx_monarch"), subprocess.run)
