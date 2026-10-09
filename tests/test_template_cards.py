"""Public cards cover every workflow and fit the browser's preview bounds."""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("template_cards", REPO / "tools/gen_template_cards.py")
assert SPEC and SPEC.loader
cards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cards)


def test_every_public_workflow_has_a_card_definition():
    public = {path.stem for path in (REPO / "example_workflows").glob("*.json")}
    assert set(cards.CARDS) == public
    testing = REPO / "tests/fixtures/workflows/generated"
    assert not set(cards.CARDS) & {path.stem for path in testing.glob("*.json")}
    assert not list(testing.glob("*.jpg"))


@pytest.mark.parametrize("name", sorted(cards.CARDS))
def test_preview_is_deterministic_jpeg_with_clear_margins(name):
    title, tagline = cards.CARDS[name]
    first = cards.card_bytes(title, tagline)
    assert first == cards.card_bytes(title, tagline)
    with Image.open(io.BytesIO(first)) as exported:
        assert exported.size == (768, 432)
        assert exported.format == "JPEG"
    # Check the uncompressed pixels so JPEG ringing cannot obscure clipping.
    rendered = cards.draw_card(title, tagline)
    for region in ((0, 164, 40, 380), (728, 164, 768, 380), (0, 396, 768, 432)):
        assert rendered.crop(region).getextrema() == tuple((c, c) for c in cards.BACKGROUND)


def test_oversized_titles_fail_instead_of_clipping():
    with pytest.raises(ValueError, match="exceeds"):
        cards.draw_card("Very long workflow name " * 12, "Text to image")


def test_check_detects_stale_export_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(cards, "TEMPLATE_DIR", tmp_path)
    name = "dgx-monarch-krea2-t2i"
    path = tmp_path / f"{name}.jpg"
    path.write_bytes(b"stale preview")
    assert cards.main(["--check", name]) == 1
    assert path.read_bytes() == b"stale preview"
    assert cards.main([name]) == 0
    assert cards.main(["--check", name]) == 0


def test_check_ignores_jpeg_metadata_but_rejects_changed_text():
    expected = cards.card_bytes("Krea2", "Text to image")
    # A JPEG comment changes the file without changing a single decoded pixel.
    comment = b"different encoder metadata"
    segment = b"\xff\xfe" + (len(comment) + 2).to_bytes(2, "big") + comment
    assert cards.matches_preview(expected[:2] + segment + expected[2:], expected)
    assert not cards.matches_preview(cards.card_bytes("Krea2", "Text to video"), expected)
    assert not cards.matches_preview(b"damaged JPEG", expected)
