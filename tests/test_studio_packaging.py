"""The optional companion ships a self-contained bundle outside Comfy's web tree."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_index_references_only_present_content_hashed_assets():
    static = ROOT / "src/dgx_monarch/studio/static"
    html = (static / "index.html").read_text()
    assets = re.findall(r'(?:src|href)="\./(assets/[^\"]+)"', html)
    assert assets
    for asset in assets:
        assert re.fullmatch(r"assets/[\w-]+-[\w-]+\.(js|css)", asset)
        assert (static / asset).is_file()
    assert "https://" not in html and "http://" not in html
    assert not (ROOT / "web/studio").exists()


def test_studio_packaging_and_entrypoint_are_explicit_and_optional():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["scripts"]["dgxm-studio"] == "dgx_monarch.studio.__main__:main"
    assert "studio" in project["project"]["optional-dependencies"]
    assert any("static" in rule for rule in project["tool"]["setuptools"]["package-data"]["dgx_monarch.studio"])
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "prune studio/node_modules" in manifest
    assert "recursive-include studio *.json *.js *.ts *.svelte *.css *.html *.md *.py" in manifest


def test_core_import_never_starts_optional_companion():
    source = (ROOT / "src/dgx_monarch/__init__.py").read_text()
    assert "studio" not in source
    from dgx_monarch.studio.__main__ import main
    assert main(["--check"]) == 0
