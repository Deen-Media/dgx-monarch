"""The README's Studio section and its fixture screenshot.

docs/BRANDING.md records how the screenshot is captured; the chunk check for
every README image lives in tests/test_public_docs.py.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
IMAGE = REPO / "docs" / "media" / "dgxm-studio.png"
BRANDING = REPO / "docs" / "BRANDING.md"
IMAGE_LINE = "[![DGX Monarch Studio with test fixture data](docs/media/dgxm-studio.png)](docs/STUDIO.md)"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_IMAGE_BYTES = 1_048_576
PROMPT = "A lighthouse on a rocky coast at dusk, soft warm light"


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _section() -> str:
    text = _readme()
    start = text.index("\n## Studio\n") + 1
    end = text.index("\n## ", start)
    return text[start:end]


def _paragraphs(text: str) -> list[str]:
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", text) if block.strip()]


def _blocks(text: str, language: str) -> list[list[str]]:
    return [body.splitlines() for body in re.findall(rf"^```{language}\n(.*?)^```$", text, flags=re.M | re.S)]


def test_studio_section_sits_between_the_reasons_and_the_featured_workflows():
    headings = [line for line in _readme().splitlines() if line.startswith("## ")]
    assert headings.count("## Studio") == 1
    start = headings.index("## Why use it?")
    assert headings[start:start + 3] == ["## Why use it?", "## Studio", "## Featured workflows"]


def test_studio_section_names_the_driver_once_and_starts_with_one_word():
    section = _section()
    assert _blocks(section, "bash") == [["dgxm studio"]]
    assert _blocks(section, "toml") == [["[studio]", 'driver = "http://127.0.0.1:8188"']]
    assert re.findall(r"^```(\w*)$", section, flags=re.M) == ["toml", "", "bash", ""]


def test_studio_section_says_how_the_records_and_the_link_behave():
    flat = " ".join(_section().split())
    assert ("Use the address your ComfyUI listens on; Studio keeps its records for the first driver "
            "address it uses.") in flat
    assert "treat that link like a password" in flat


def test_studio_section_shows_the_screenshot_with_a_fixture_caption():
    paragraphs = _paragraphs(_section())
    assert paragraphs.count(IMAGE_LINE) == 1
    caption = paragraphs[paragraphs.index(IMAGE_LINE) + 1]
    assert "test fixture" in caption
    assert re.search(r"(?<![!\]])\[[^\]]+\]\(docs/STUDIO\.md\)", caption), caption


def test_studio_screenshot_is_a_small_png_at_the_documented_size():
    data = IMAGE.read_bytes()
    assert data.startswith(PNG_SIGNATURE)
    assert len(data) <= MAX_IMAGE_BYTES, len(data)
    assert data[12:16] == b"IHDR"
    assert struct.unpack(">II", data[16:24]) == (1440, 900)


def test_branding_records_how_the_studio_screenshot_is_captured():
    flat = " ".join(BRANDING.read_text(encoding="utf-8").split())
    for phrase in ("dgxm-studio.png", "1440 by 900", ".workflow-actions input[type=file]", PROMPT,
                   "$HOME/.local/state/dgxm-studio-fixture"):
        assert phrase in flat, phrase
