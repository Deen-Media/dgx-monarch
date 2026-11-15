"""The recorded-schema builder keeps no file name a bundled workflow does not reference.

tests/fixtures/studio/build_object_info_bundled.py trims a recorded /object_info
before it becomes the committed fixture. A driver lists every file in its input
and model folders, private names included, and ComfyUI's image, audio and video
loaders accept more types than the builder's FILE_SUFFIX names. Any choice that
looks like a file, by a short extension or a slash, must go unless a workflow
references it. The builder also lists the stock classes Studio's text-through
rule needs a row for. CPU only.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "dgxm_build_object_info_bundled", ROOT / "tests/fixtures/studio/build_object_info_bundled.py")
assert _SPEC and _SPEC.loader
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


def test_planted_file_names_of_unlisted_types_do_not_enter_the_fixture():
    recording = {
        "LoadImage": {"input": {"required": {"image": [
            ["example.png", "planted_private_scan.tif", "planted_holiday.HEIC", "planted/sub/folder.png"],
            {"image_upload": True}]}}, "output": ["IMAGE", "MASK"]},
        "LoadAudio": {"input": {"required": {"audio": [
            ["reference_audio.mp3", "planted_voice_memo.m4a", "default"], {"audio_upload": True}]}},
            "output": ["AUDIO"]},
    }
    workflows = [{"workflow": {"nodes": [
        {"type": "LoadImage", "widgets_values": ["example.png", "image"]},
        {"type": "LoadAudio", "widgets_values": ["reference_audio.mp3"]},
    ]}}]
    fixture = builder.build(recording, workflows)
    assert fixture["LoadImage"]["input"]["required"]["image"][0] == ["example.png"]
    # A choice that is no file name, such as a mode, stays.
    assert fixture["LoadAudio"]["input"]["required"]["audio"][0] == ["reference_audio.mp3", "default"]
    assert "planted" not in repr(fixture)


def test_the_stock_text_class_list_keeps_stock_classes_with_text_in_and_text_out(tmp_path):
    text = ["STRING", {"multiline": True}]
    recording = {
        "StockText": {"python_module": "comfy_extras.nodes_string", "output": ["STRING"],
                      "input": {"required": {"string": text, "key": ["STRING", {}], "count": ["INT", {}]}}},
        "StockEncode": {"python_module": "nodes", "output": ["CONDITIONING"],
                        "input": {"required": {"text": text}}},
        "StockCloud": {"python_module": "comfy_api_nodes.nodes_x", "output": ["VIDEO", "STRING"],
                       "input": {"required": {"model": ["COMFY_DYNAMICCOMBO_V3", {"options": [
                           {"key": "a", "inputs": {"required": {"prompt": text}}},
                           {"key": "b", "inputs": {"required": {"prompt": ["STRING", {}]}}}]}],
                           "values": ["COMFY_AUTOGROW_V3", {"template": {"input": {"required": {
                               "value": ["STRING", {}]}}, "names": ["a", "b"]}}]}}},
        "StockLength": {"python_module": "comfy_extras.nodes_string", "output": ["INT"],
                        "input": {"required": {"string": text}}},
        "StockLatent": {"python_module": "nodes", "output": ["LATENT"], "input": {"required": {}}},
        "PackText": {"python_module": "custom_nodes.pack", "output": ["STRING"],
                     "input": {"required": {"text": text}}},
        "CombosOut": {"python_module": "comfy_extras.nodes_x", "output": [["a", "b"], "STRING"],
                      "input": {"required": {"name": ["STRING", {}]}}},
    }
    classes = builder.stock_text_classes(recording)
    assert classes == {
        "CombosOut": {"name": False},
        "StockCloud": {"model.prompt": True, "values.": False},
        "StockEncode": {"text": True},
        "StockText": {"key": False, "string": True},
    }
    out = tmp_path / "stock_text_classes.json"
    builder.write_stock_text_classes(classes, out)
    assert json.loads(out.read_text())["classes"] == classes


def test_the_committed_stock_text_class_list_is_the_builders_output(tmp_path):
    # The committed list reads back to what the builder writes from it: sorted, one class to a
    # line, so a new recording's change reads as a short diff.
    committed = ROOT / "tests/fixtures/studio/stock_text_classes.json"
    classes = json.loads(committed.read_text())["classes"]
    assert list(classes) == sorted(classes)
    out = tmp_path / "stock_text_classes.json"
    builder.write_stock_text_classes(classes, out)
    assert out.read_text() == committed.read_text()
