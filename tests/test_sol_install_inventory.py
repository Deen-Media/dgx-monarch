"""Narrow PEP610 proof for normal editable metadata, without package installs."""
from __future__ import annotations

import importlib.metadata as md
import json
from pathlib import Path

import pytest

from dgx_monarch.cli import sol_install_inventory as inventory
from dgx_monarch.cli import sol_install_payload as payload


def distribution(path: Path, version="1.0.0", name="dgx-monarch", requires=()) -> md.PathDistribution:
    path.mkdir(parents=True)
    metadata = f"Name: {name}\nVersion: {version}\n" + "".join(f"Requires-Dist: {value}\n" for value in requires)
    (path / ("METADATA" if path.name.endswith(".dist-info") else "PKG-INFO")).write_text(metadata)
    (path / "RECORD").write_text(f"{path.name}/METADATA,,\n")
    return md.PathDistribution(path)


def pair(tmp_path, *, version="1.0.0", editable=True, source_parent="src", url=None):
    site = tmp_path / "site"
    project = tmp_path / "project"
    project.mkdir()
    installed_path = site / "dgx_monarch-1.0.0.dist-info"
    installed = distribution(installed_path)
    origin = project if source_parent == "flat" else project / source_parent
    source = distribution(origin / "dgx_monarch.egg-info", version=version)
    (installed_path / "direct_url.json").write_text(json.dumps({
        "url": url or project.as_uri(), "dir_info": {"editable": editable}}))
    return site, project, installed, source


@pytest.mark.parametrize("layout", ["src", "flat"])
def test_proven_editable_companion_uses_installed_authority(tmp_path, layout):
    site, _, installed, source = pair(tmp_path, source_parent=layout)
    result = inventory.distribution_inventory([source, installed], site_roots=[site])
    assert result == {"dgx-monarch": installed}


def test_same_physical_metadata_discovered_twice_is_deduplicated(tmp_path):
    site, _, installed, source = pair(tmp_path)
    duplicate = md.PathDistribution(site / "dgx_monarch-1.0.0.dist-info")
    assert inventory.distribution_inventory([installed, duplicate, source], site_roots=[site]) == {"dgx-monarch": installed}


@pytest.mark.parametrize("kwargs", [
    {"version": "2.0.0"}, {"editable": False}, {"source_parent": "vendor/other"},
    {"url": "https://example.invalid/repository"}, {"url": "file://remote.invalid/project"},
])
def test_unproven_duplicates_still_refuse(tmp_path, kwargs):
    site, _, installed, source = pair(tmp_path, **kwargs)
    with pytest.raises(inventory.DistributionInventoryError, match="duplicate distribution"):
        inventory.distribution_inventory([installed, source], site_roots=[site])


def test_off_project_source_and_missing_direct_url_refuse(tmp_path):
    site, _, installed, source = pair(tmp_path)
    other = tmp_path / "another-project"
    other.mkdir()
    direct = site / "dgx_monarch-1.0.0.dist-info/direct_url.json"
    direct.write_text(json.dumps({"url": other.as_uri(), "dir_info": {"editable": True}}))
    with pytest.raises(inventory.DistributionInventoryError):
        inventory.distribution_inventory([installed, source], site_roots=[site])
    direct.unlink()
    with pytest.raises(inventory.DistributionInventoryError):
        inventory.distribution_inventory([installed, source], site_roots=[site])


def test_direct_url_symlink_and_second_installed_authority_refuse(tmp_path):
    site, _, installed, source = pair(tmp_path)
    direct = site / "dgx_monarch-1.0.0.dist-info/direct_url.json"
    external = tmp_path / "direct.json"
    direct.rename(external)
    direct.symlink_to(external)
    with pytest.raises(inventory.DistributionInventoryError):
        inventory.distribution_inventory([installed, source], site_roots=[site])
    other_site = tmp_path / "other-site"
    other = distribution(other_site / "dgx_monarch-1.0.0.dist-info")
    with pytest.raises(inventory.DistributionInventoryError):
        inventory.distribution_inventory([installed, other], site_roots=[site, other_site])


def test_conflicting_named_versions_never_deduplicate(tmp_path):
    site, _, installed, _ = pair(tmp_path)
    other = distribution(tmp_path / "elsewhere/dgx_monarch-2.0.0.dist-info", version="2.0.0")
    with pytest.raises(inventory.DistributionInventoryError):
        inventory.distribution_inventory([installed, other], site_roots=[site])


def test_all_package_views_use_authoritative_editable_record_and_requirements(tmp_path, monkeypatch):
    site, _, installed, source = pair(tmp_path)
    # Source metadata deliberately differs in its dependency inventory: it must
    # never be used as a second installed distribution or as uninstall authority.
    source_path = Path(source._path)
    (source_path / "PKG-INFO").write_text("Name: dgx-monarch\nVersion: 1.0.0\nRequires-Dist: bogus-source-only\n")
    (source_path / "RECORD").write_text("../../../outside-source-file,,\n")
    torch = distribution(site / "torch-2.12.0.dist-info", version="2.12.0", name="torch")
    monkeypatch.setattr(inventory.sysconfig, "get_path", lambda key: str(site))
    monkeypatch.setattr(md, "distributions", lambda: [installed, source, torch])
    monkeypatch.setattr(payload, "scheme_roots", lambda: {site})
    assert payload.installed() == {"dgx-monarch": "1.0.0", "torch": "2.12.0"}
    assert payload.protected_state()[1] == {
        str(site / "dgx_monarch-1.0.0.dist-info/METADATA"), str(site / "torch-2.12.0.dist-info/METADATA")}
    assert not any("outside-source-file" in value for value in payload.preserved_paths(set()))
    assert payload.replacement_paths("dgx_monarch") == {str(site / "dgx_monarch-1.0.0.dist-info/METADATA")}
    assert payload.requirement_violations(payload.installed()) == []
    payload.verify_requirements({"dgx-monarch"})


def test_local_editable_project_alias_binds_its_physical_root(tmp_path):
    site, project, installed, source = pair(tmp_path)
    alias = tmp_path / "comfy-custom-node-alias"
    alias.symlink_to(project, target_is_directory=True)
    direct = site / "dgx_monarch-1.0.0.dist-info/direct_url.json"
    direct.write_text(json.dumps({"url": alias.as_uri(), "dir_info": {"editable": True}}))
    assert inventory.distribution_inventory([installed, source], site_roots=[site]) == {"dgx-monarch": installed}
    other = tmp_path / "unrelated-project"
    other.mkdir()
    alias.unlink()
    alias.symlink_to(other, target_is_directory=True)
    with pytest.raises(inventory.DistributionInventoryError):
        inventory.distribution_inventory([installed, source], site_roots=[site])
