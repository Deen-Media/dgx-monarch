"""Public guides must be usable without private operator artifacts."""

import json
import re
import struct
import zlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PUBLIC_GUIDES = ("README.md", "docs/MODELS.md", "docs/BENCHMARKS.md", "docs/VALIDATION.md")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CRITICAL_PNG_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}
EVIDENCE_ANCHORS = {
    "evidence-krea2-raw-turbo",
    "evidence-kroma",
    "evidence-chroma",
    "evidence-radiance",
    "evidence-ideogram4",
    "evidence-flux-1-dev",
    "evidence-flux-1-schnell",
    "evidence-flux2",
    "evidence-flux-3-klein",
    "evidence-flux-3-klein-base",
    "evidence-moss-image-sft",
    "evidence-moss-image-base",
    "evidence-kromoss",
    "evidence-longcat-image",
    "evidence-hunyuanimage-2-1-refiner",
    "evidence-qwen-image",
    "evidence-mage-flow-t2i-edit-quality-turbo",
    "evidence-qwen-image-2-1",
    "evidence-ernie-image",
    "evidence-z-image-latent",
    "evidence-z-image-dct-pixelspace",
    "evidence-lens",
    "evidence-omnigen2",
    "evidence-anima",
    "evidence-boogu",
    "evidence-pixeldit-pid",
    "evidence-kandinsky5-image",
    "evidence-ltx-2-3-t2v",
    "evidence-ltx-2-3-i2v-av-packed",
    "evidence-ltx-2-5-t2v-av-packed",
    "evidence-wan-2-1-t2v",
    "evidence-wan-2-2-t2v-high-low",
    "evidence-wan-2-2-i2v",
    "evidence-wan-2-1-i2v",
    "evidence-wan-flowrvs",
    "evidence-wan-bernini-r",
    "evidence-wan-scail-preview",
    "evidence-wan-scail2",
    "evidence-wandancer",
    "evidence-wan-animate-2",
    "evidence-hunyuanvideo-1-5-sr",
    "evidence-cogvideox-1-5-t2v",
    "evidence-cogvideox-1-5-i2v",
    "evidence-cogvideox-1-5-inpaint",
    "evidence-kandinsky5-video-lite",
    "evidence-kandinsky5-video-pro",
    "evidence-minimax-h3-t2va-fl2va-ref2va-av",
}


def test_public_validation_preserves_all_model_evidence_destinations():
    text = (REPO / "docs/VALIDATION.md").read_text()
    anchors = re.findall(r'<a id="(evidence-[^"]+)"', text)
    assert len(anchors) == len(set(anchors)), "duplicate evidence anchors"
    assert set(anchors) == EVIDENCE_ANCHORS


@pytest.mark.parametrize("relative", PUBLIC_GUIDES)
def test_public_guides_do_not_depend_on_private_files(relative):
    text = (REPO / relative).read_text()
    assert not re.search(r"/(?:home|tmp|var/tmp)/[A-Za-z0-9_.-]", text)
    assert not re.search(r"(?:operator[- ]kit|private operator record|`acceptance-2026-)", text, re.I)
    assert "EMPIRICAL.md" not in text


@pytest.mark.parametrize("relative", PUBLIC_GUIDES)
def test_public_guide_prose_uses_plain_punctuation(relative):
    text = (REPO / relative).read_text()
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    prose = re.sub(r"`[^`]*`", "", prose)
    assert not any(char in prose for char in (chr(0x2013), chr(0x2014)))
    assert not re.search(r"\bbit[ -]for[ -]bit\b", prose, re.I)


def test_readme_links_to_installation_and_current_support():
    text = (REPO / "README.md").read_text()
    for target in ("docs/INSTALL.md", "docs/MODELS.md", "docs/BENCHMARKS.md", "docs/VALIDATION.md"):
        assert re.search(r"\]\(" + re.escape(target) + r"(?:#[^)]*)?\)", text), target


def _png_chunk_types(data: bytes) -> list[bytes]:
    """Chunk types in file order; each chunk's CRC must match and nothing may follow IEND."""
    assert data.startswith(PNG_SIGNATURE), "not a PNG"
    types: list[bytes] = []
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset:offset + 4])
        kind = data[offset + 4:offset + 8]
        body = data[offset + 8:offset + 8 + length]
        (crc,) = struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])
        assert zlib.crc32(kind + body) == crc, f"bad CRC in {kind!r}"
        types.append(kind)
        offset += 12 + length
        if kind == b"IEND":
            break
    assert offset == len(data), "bytes after IEND"
    return types


def test_readme_images_carry_only_critical_png_chunks():
    """Text, time and EXIF chunks can carry a capture's host, path or software, so
    the images the README shows keep only the chunks that draw pixels."""
    names = sorted(set(re.findall(r"docs/media/[A-Za-z0-9_.-]+\.png", (REPO / "README.md").read_text())))
    assert names
    for name in names:
        types = _png_chunk_types((REPO / name).read_bytes())
        assert types[0] == b"IHDR" and types[-1] == b"IEND", name
        assert set(types) <= CRITICAL_PNG_CHUNKS, (name, sorted(set(types) - CRITICAL_PNG_CHUNKS))


def test_workflow_notes_do_not_cite_private_tickets():
    for path in (REPO / "example_workflows").rglob("*.json"):
        for node in json.loads(path.read_text()).get("nodes", []):
            if node.get("type") == "Note":
                for text in node.get("widgets_values", []):
                    assert not re.search(r"\b(?:issue|PR)\s*#?\d+", str(text)), path.name
