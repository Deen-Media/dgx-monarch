"""Public guides must be usable without private operator artifacts."""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PUBLIC_GUIDES = ("README.md", "docs/MODELS.md", "docs/BENCHMARKS.md", "docs/VALIDATION.md")
EVIDENCE_ANCHORS = {
    "evidence-krea2-raw-turbo",
    "evidence-chroma",
    "evidence-radiance",
    "evidence-ideogram4",
    "evidence-flux-1-dev",
    "evidence-flux-1-schnell",
    "evidence-flux2",
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


def test_workflow_notes_do_not_cite_private_tickets():
    for path in (REPO / "example_workflows").rglob("*.json"):
        for node in json.loads(path.read_text()).get("nodes", []):
            if node.get("type") == "Note":
                for text in node.get("widgets_values", []):
                    assert not re.search(r"\b(?:issue|PR)\s*#?\d+", str(text)), path.name
