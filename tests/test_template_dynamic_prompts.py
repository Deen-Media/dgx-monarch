"""No shipped or testing graph holds text the browser's dynamic prompt pass rewrites.

ComfyUI marks some text widgets dynamicPrompts, CLIP Text Encode's among them.
On Queue the frontend runs those widgets through its dynamic prompt pass and
sends the result: comments go, and every brace pair goes with them, so a JSON
caption typed into such a box reaches the encoder as broken text. The API
path, the sweep and Studio keep the text as written, so the graph would render
one prompt from the browser and another everywhere else. A JSON prompt
therefore sits in Text (Multiline), which carries no such flag, and reaches the
encoder through a link.

The flags come from the recorded driver schema
(tests/fixtures/studio/object_info_bundled.json). A class it lacks has every
string widget checked, the safe side; the real-ComfyUI widget canary checks
the same rule against master's schemas. The pass itself is the canary's port of
the frontend's parser. CPU only.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from benchmark.sweep import convert

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = sorted((ROOT / "example_workflows").glob("*.json")) + sorted(
    (ROOT / "tests" / "fixtures" / "workflows" / "generated").glob("*.json"))
SCHEMA = json.loads((ROOT / "tests/fixtures/studio/object_info_bundled.json").read_text())
_SPEC = importlib.util.spec_from_file_location(
    "dgxm_template_widget_canary_dp", ROOT / "tests" / "canary" / "template_widget_canary.py")
assert _SPEC and _SPEC.loader
canary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(canary)
IDEOGRAM = ("dgx-monarch-ideogram4-t2i", "dgx-monarch-test-ideogram4-lora")


def _api(path: Path) -> dict:
    return convert.convert(json.loads(path.read_text(encoding="utf-8")),
                           convert.repo_widget_names(ROOT))


def _flagged(node_class: str, inputs: dict) -> list[str]:
    """The inputs the browser's pass reads, or every string input when the schema lacks the class."""
    info = SCHEMA.get(node_class)
    if info is None:
        return [name for name, value in inputs.items() if isinstance(value, str)]
    return canary.dynamic_prompt_widgets(info.get("input") or {})


def _browser_prompt(api: dict) -> dict:
    """The API prompt a browser Queue press sends for this graph."""
    sent = json.loads(json.dumps(api))
    for node in sent.values():
        for name in _flagged(node["class_type"], node["inputs"]):
            value = node["inputs"].get(name)
            if isinstance(value, str):
                node["inputs"][name] = canary.process_dynamic_prompt(value)
    return sent


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.stem)
def test_a_browser_queue_sends_the_graph_as_written(path):
    api = _api(path)
    assert _browser_prompt(api) == api


def test_the_recorded_schema_still_flags_clip_text_encode():
    # The check above is only as good as its flags; if this recording loses
    # them, every flagged widget would pass unread.
    assert "text" in canary.dynamic_prompt_widgets(SCHEMA["CLIPTextEncode"]["input"])
    assert canary.dynamic_prompt_widgets(SCHEMA["PrimitiveStringMultiline"]["input"]) == []


def _caption(prompt: dict) -> str:
    (encode,) = [node for node in prompt.values() if node["class_type"] == "CLIPTextEncode"]
    source, slot = encode["inputs"]["text"]
    assert slot == 0 and prompt[source]["class_type"] == "PrimitiveStringMultiline"
    return prompt[source]["inputs"]["value"]


@pytest.mark.parametrize("name", IDEOGRAM)
def test_the_ideogram_caption_reaches_the_encoder_as_valid_json_from_the_browser(name):
    path = next(path for path in TEMPLATES if path.stem == name)
    api = _api(path)
    caption = _caption(_browser_prompt(api))
    assert caption == _caption(api)
    parsed = json.loads(caption)
    # The official caption schema requires these, and the style keys keep their order.
    assert set(parsed["compositional_deconstruction"]) >= {"background", "elements"}
    assert list(parsed["style_description"])[:3] == ["aesthetics", "lighting", "photo"]
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import gen_templates
    finally:
        sys.path.remove(str(ROOT / "tools"))
    assert caption == gen_templates.IDEOGRAM_CAPTION
