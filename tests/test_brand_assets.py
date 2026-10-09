"""Logo exports stay accessible, self-contained and suitable for small surfaces."""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest
from defusedxml import ElementTree as ET
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("brand_assets", ROOT / "tools/gen_brand_assets.py")
assert SPEC and SPEC.loader
brand = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(brand)


@pytest.mark.parametrize("path", sorted((ROOT / "web/brand").glob("*.svg")))
def test_browser_logos_are_self_contained_and_accessible(path):
    root = ET.fromstring(path.read_text())
    assert root.get("viewBox")
    assert root.get("role") == "img"
    ids = {n.get("id") for n in root.iter()}
    assert set(root.get("aria-labelledby", "").split()) <= ids
    assert root.find(f"{{{brand.NS}}}title").text == "DGX Monarch"
    for node in root.iter():
        assert node.tag.split("}")[-1] not in {"script", "image", "foreignObject", "text"}
        assert not any(key.lower().startswith("on") or key.endswith("href") for key in node.attrib)


def test_export_shapes_and_transparency():
    exported = brand.outputs()
    for name in ("monarch-256.png", "monarch-dark-1024.png"):
        with Image.open(io.BytesIO(exported[ROOT / "docs/media" / name])) as picture:
            size = int(name.rsplit("-", 1)[1].split(".")[0])
            assert picture.size == (size, size)
            assert picture.mode == "RGBA"
            assert picture.getpixel((0, 0))[3] == 0
            assert picture.getchannel("A").getextrema() == (0, 255)
    with Image.open(io.BytesIO(exported[ROOT / "docs/media/monarch-social.png"])) as picture:
        assert picture.size == (1200, 630)
    for name in ("monarch-wordmark.svg", "monarch-wordmark-dark.svg"):
        root = ET.fromstring(exported[ROOT / "docs/media" / name])
        assert not root.findall(f".//{{{brand.NS}}}text")


def test_freshness_detects_changed_svg_and_raster(tmp_path):
    path = tmp_path / "logo.svg"
    path.write_bytes(b"old")
    assert not brand.matches(path, b"new")
    picture = Image.new("RGBA", (24, 24), "black")
    path = tmp_path / "logo.png"
    picture.save(path)
    expected = io.BytesIO()
    picture.save(expected, format="PNG")
    assert brand.matches(path, expected.getvalue())
    Image.new("RGBA", (24, 24), "white").save(path)
    assert not brand.matches(path, expected.getvalue())
