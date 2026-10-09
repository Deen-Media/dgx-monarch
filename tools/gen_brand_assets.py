"""Export logo variants from web/brand/monarch.svg.

The master SVG is editable artwork. Browser, README, desktop and card assets
share its paths; only colors, stroke widths and the wordmark layout vary.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from xml.etree import ElementTree as ET

import cairosvg
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageStat

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "web/brand/monarch.svg"
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
INK = "#111318"
PALE = "#f3f5f7"
GREEN = "#87f900"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def logo(*, dark: bool = False, mono: bool = False, small: bool = False) -> ET.Element:
    """Derive colors and small-size strokes without changing the wing geometry."""
    root = ET.fromstring(MASTER.read_text())  # noqa: S314 - repository-owned SVG
    foreground = PALE if dark else INK
    for element in root.iter():
        for attr in ("fill", "stroke"):
            value = element.get(attr)
            if value == INK:
                element.set(attr, foreground)
            elif value == GREEN and mono:
                element.set(attr, foreground)
        if small and element.get("data-color") == "circuit":
            element.set("stroke-width", "12")
            for child in element:
                if child.tag == f"{{{NS}}}circle":
                    child.set("r", "17")
    return root


def svg(root: ET.Element) -> bytes:
    return (ET.tostring(root, encoding="unicode") + "\n").encode()


def wordmark(*, dark: bool) -> bytes:
    root = ET.Element(f"{{{NS}}}svg", {
        "viewBox": "0 0 900 180", "role": "img", "aria-labelledby": "title desc",
    })
    ET.SubElement(root, f"{{{NS}}}title", {"id": "title"}).text = "DGX Monarch"
    ET.SubElement(root, f"{{{NS}}}desc", {"id": "desc"}).text = "DGX Monarch butterfly and wordmark."
    nested = logo(dark=dark)
    nested.attrib = {"x": "0", "y": "0", "width": "180", "height": "180", "viewBox": "290 110 700 640"}
    for child in list(nested):
        if child.tag in (f"{{{NS}}}title", f"{{{NS}}}desc"):
            nested.remove(child)
    root.append(nested)
    # Outline the lettering so the exported wordmark needs no installed font.
    with TTFont(FONT) as font:
        glyphs = font.getGlyphSet()
        cmap = font.getBestCmap()
        scale = 70 / font["head"].unitsPerEm
        cursor = 209.0
        for char in "DGX MONARCH":
            name = cmap[ord(char)]
            pen = SVGPathPen(glyphs)
            glyphs[name].draw(pen)
            commands = pen.getCommands()
            if commands:
                ET.SubElement(root, f"{{{NS}}}path", {
                    "d": commands, "fill": PALE if dark else INK,
                    "transform": f"translate({cursor:.4f} 109) scale({scale:.6f} {-scale:.6f})",
                })
            cursor += font["hmtx"][name][0] * scale - 2
    return svg(root)


def social_preview() -> bytes:
    """Render a static repository preview with no model-output claims."""
    root = ET.Element(f"{{{NS}}}svg", {"viewBox": "0 0 1200 630"})
    ET.SubElement(root, f"{{{NS}}}rect", {"width": "1200", "height": "630", "fill": INK})
    mark = ET.fromstring(wordmark(dark=True))  # noqa: S314 - generated local SVG
    mark.set("x", "95")
    mark.set("y", "137")
    mark.set("width", "1010")
    mark.set("height", "202")
    root.append(mark)
    for y, size, text, fill in (
        (410, 35, "ComfyUI across two DGX Sparks", PALE),
        (465, 24, "Images, video and audio. One ComfyUI window.", "#afb8be"),
    ):
        element = ET.SubElement(root, f"{{{NS}}}text", {
            "x": "600", "y": str(y), "text-anchor": "middle", "fill": fill,
            "font-family": "DejaVu Sans, sans-serif", "font-size": str(size),
        })
        element.text = text
    ET.SubElement(root, f"{{{NS}}}path", {"d": "M535 525h130", "stroke": GREEN, "stroke-width": "4"})
    return cairosvg.svg2png(bytestring=svg(root), output_width=1200, output_height=630)


def outputs() -> dict[Path, bytes]:
    result = {}
    for name, options in (
        ("monarch-dark", {"dark": True}),
        ("monarch-mono", {"mono": True}),
        ("monarch-mono-dark", {"mono": True, "dark": True}),
        ("monarch-small", {"small": True}),
        ("monarch-small-dark", {"small": True, "dark": True}),
    ):
        result[ROOT / f"web/brand/{name}.svg"] = svg(logo(**options))
    result[ROOT / "docs/media/monarch-wordmark.svg"] = wordmark(dark=False)
    result[ROOT / "docs/media/monarch-wordmark-dark.svg"] = wordmark(dark=True)
    desktop = ET.Element(f"{{{NS}}}svg", {
        "viewBox": "0 0 256 256", "role": "img", "aria-labelledby": "title desc",
    })
    ET.SubElement(desktop, f"{{{NS}}}title", {"id": "title"}).text = "DGX Monarch"
    ET.SubElement(desktop, f"{{{NS}}}desc", {"id": "desc"}).text = "Monarch butterfly with green circuit connections."
    ET.SubElement(desktop, f"{{{NS}}}rect", {
        "width": "256", "height": "256", "rx": "36", "fill": INK,
    })
    mark = logo(dark=True)
    mark.attrib = {"x": "12", "y": "12", "width": "232", "height": "232", "viewBox": "290 110 700 640"}
    for child in list(mark):
        if child.tag in (f"{{{NS}}}title", f"{{{NS}}}desc"):
            mark.remove(child)
    desktop.append(mark)
    result[ROOT / "docs/media/dgx-monarch.svg"] = svg(desktop)
    for dark in (False, True):
        name = "monarch-dark" if dark else "monarch"
        for size in (256, 1024):
            result[ROOT / f"docs/media/{name}-{size}.png"] = cairosvg.svg2png(
                bytestring=svg(logo(dark=dark)), output_width=size, output_height=size,
            )
    result[ROOT / "docs/media/monarch-social.png"] = social_preview()
    return result


def matches(path: Path, content: bytes) -> bool:
    if not path.exists():
        return False
    if path.suffix != ".png":
        return path.read_bytes() == content
    try:
        with Image.open(path) as saved, Image.open(io.BytesIO(content)) as expected:
            if saved.size != expected.size or saved.format != "PNG":
                return False
            difference = ImageChops.difference(saved.convert("RGBA"), expected.convert("RGBA"))
            return max(high for _, high in difference.getextrema()) <= 8 and max(ImageStat.Stat(difference).mean) <= .25
    except (OSError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for path, content in outputs().items():
        if args.check:
            if not matches(path, content):
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    if stale:
        print("Logo exports differ: " + ", ".join(stale))
        return 1
    print("Logo exports in sync" if args.check else "Wrote logo exports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
