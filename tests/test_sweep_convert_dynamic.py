"""The sweep converter reads a dynamic combo's child widgets, and shifts no other graph.

A ComfyUI dynamic combo (COMFY_DYNAMICCOMBO_V3) reveals child widgets by the
key it holds. The frontend stores them right after the combo in
``widgets_values`` and the API prompt names each ``<combo>.<child>``. Until
MiniMax-H3-Image's edit graph, every committed template held a key that
revealed nothing, and the converter read a dynamic combo as one slot. These
tests hold three things: the new graphs convert to what ComfyUI's own
templates store; ComfyUI's own thirteen Generate Text values, with sampling
on, convert too, as do the values the three MiniMax-H3-Image graphs ship; and every
node of every committed template that reveals no child reads exactly as the
flat reader read it. CPU only: no ComfyUI import.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.sweep import convert  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
try:
    import gen_templates
finally:
    sys.path.remove(str(ROOT / "tools"))

TEMPLATES = sorted((ROOT / "example_workflows").glob("*.json")) + sorted(
    (ROOT / "tests" / "fixtures" / "workflows" / "generated").glob("*.json"))
EDIT = "dgx-monarch-minimax-h3-image-edit"
# What ComfyUI's own MiniMax-H3-Image templates store for Generate Text with
# sampling on (workflow_templates 0.11.105, image_minimax_h3_image_t2i).
UPSTREAM_GENERATE_TEXT = ["", 8192, "on", 1, 20, 0.95, 0, 1, 0, 0, True, True, "auto"]
# Generate Text's input block in the shape /object_info gives it at ComfyUI
# 0.45.0 (comfy_extras/nodes_textgen.py), tooltips left out.
GENERATE_TEXT_SCHEMA = {
    "required": {
        "clip": ["CLIP", {}],
        "prompt": ["STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}],
        "max_length": ["INT", {"default": 512, "min": 1, "max": 32768}],
        "sampling_mode": ["COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "on", "inputs": {"required": {
                "temperature": ["FLOAT", {"default": 0.7}],
                "top_k": ["INT", {"default": 64}],
                "top_p": ["FLOAT", {"default": 0.95}],
                "min_p": ["FLOAT", {"default": 0.05}],
                "repetition_penalty": ["FLOAT", {"default": 1.05}],
                "presence_penalty": ["FLOAT", {"default": 0.0}],
                "seed": ["INT", {"default": 0}]}}},
            {"key": "off", "inputs": {"required": {}}}]}],
    },
    "optional": {
        "images": ["COMFY_AUTOGROW_V3", {}],
        "thinking": ["BOOLEAN", {"default": False}],
        "use_default_template": ["BOOLEAN", {"default": True}],
        "mtp": ["COMBO", {"options": ["auto", "off"], "default": "auto"}],
        "system_prompt": ["STRING", {"forceInput": True}],
    },
}
GENERATE_TEXT_ON = {
    "prompt": "", "max_length": 8192, "sampling_mode": "on",
    "sampling_mode.temperature": 1, "sampling_mode.top_k": 20, "sampling_mode.top_p": 0.95,
    "sampling_mode.min_p": 0, "sampling_mode.repetition_penalty": 1,
    "sampling_mode.presence_penalty": 0, "sampling_mode.seed": 0,
    "thinking": True, "use_default_template": True, "mtp": "auto"}


def _node(node_class: str, values: list, inputs: list | None = None) -> dict:
    return {"links": [[7, 1, 0, 2, 0, "STRING"]],
            "nodes": [{"id": 2, "type": node_class, "widgets_values": values,
                       "inputs": inputs or []}]}


def _schema_names(schema: dict):
    return lambda _node_class: convert.object_info_widget_names(schema)


def _flat_read(order: list[str], values: list, inputs: dict) -> None:
    """The reader as it stood before child widgets: one slot a name, in order."""
    index = 0
    for name in order:
        if name in inputs:
            if index < len(values):
                index += 1
                follower = values[index] if index < len(values) else None
                if name in convert.SEED_NAMES and isinstance(follower, str) \
                        and follower in convert.CONTROL:
                    index += 1
            continue
        if index >= len(values):
            break
        inputs[name] = values[index]
        index += 1
        follower = values[index] if index < len(values) else None
        if name in convert.SEED_NAMES and isinstance(follower, str) and follower in convert.CONTROL:
            index += 1


# ---- A driver's schema ---------------------------------------------------------

def test_a_driver_schema_gives_a_dynamic_combo_its_children():
    names = convert.object_info_widget_names(GENERATE_TEXT_SCHEMA)
    assert names == ["prompt", "max_length", "sampling_mode", "thinking",
                     "use_default_template", "mtp"]
    (combo,) = [name for name in names if isinstance(name, convert.DynamicName)]
    assert combo == "sampling_mode" and not combo.closed
    assert combo.children == {
        "on": ["sampling_mode.temperature", "sampling_mode.top_k", "sampling_mode.top_p",
               "sampling_mode.min_p", "sampling_mode.repetition_penalty",
               "sampling_mode.presence_penalty", "sampling_mode.seed"],
        "off": []}


def test_upstreams_thirteen_values_convert_from_a_driver_schema():
    api = convert.convert(_node("TextGenerateV2", UPSTREAM_GENERATE_TEXT),
                          _schema_names(GENERATE_TEXT_SCHEMA))
    assert api["2"]["inputs"] == GENERATE_TEXT_ON
    # The nested seed takes no control slot: thirteen values, thirteen inputs.
    assert len(api["2"]["inputs"]) == len(UPSTREAM_GENERATE_TEXT)


def test_a_key_that_reveals_nothing_still_holds_one_slot():
    api = convert.convert(_node("TextGenerateV2", ["", 2048, "off", True, True, "auto"]),
                          _schema_names(GENERATE_TEXT_SCHEMA))
    assert api["2"]["inputs"] == {"prompt": "", "max_length": 2048, "sampling_mode": "off",
                                  "thinking": True, "use_default_template": True, "mtp": "auto"}


def test_a_driver_schema_reads_an_unknown_key_as_revealing_nothing():
    """ComfyUI drops a dynamic combo whose value names no option; nothing follows it."""
    api = convert.convert(_node("TextGenerateV2", ["", 64, "maybe", True, True, "auto"]),
                          _schema_names(GENERATE_TEXT_SCHEMA))
    assert api["2"]["inputs"]["sampling_mode"] == "maybe"
    assert api["2"]["inputs"]["thinking"] is True and api["2"]["inputs"]["mtp"] == "auto"


def test_a_linked_dynamic_combo_selects_no_option_and_its_stored_children_are_passed_over():
    """The frontend keeps the children its widget value reveals in ``widgets_values``
    whether or not a link feeds the combo. ComfyUI reads none of them behind a link,
    and the widgets after them must still read from their own slots."""
    linked = [{"name": "sampling_mode", "link": 7}]
    for names in (_schema_names(GENERATE_TEXT_SCHEMA), convert.repo_widget_names(ROOT)):
        api = convert.convert(_node("TextGenerateV2", UPSTREAM_GENERATE_TEXT, linked), names)
        inputs = api["2"]["inputs"]
        assert inputs["sampling_mode"] == ["1", 0]
        assert not [name for name in inputs if name.startswith("sampling_mode.")]
        assert (inputs["max_length"], inputs["thinking"], inputs["use_default_template"],
                inputs["mtp"]) == (8192, True, True, "auto")
    # A stored key that reveals nothing leaves the next widget in the next slot.
    api = convert.convert(_node("TextGenerateV2", ["", 64, "off", True, True, "auto"], linked),
                          _schema_names(GENERATE_TEXT_SCHEMA))
    assert (api["2"]["inputs"]["thinking"], api["2"]["inputs"]["mtp"]) == (True, "auto")


def test_a_child_that_is_a_dynamic_combo_reveals_its_own_children():
    schema = {"required": {"mode": ["COMFY_DYNAMICCOMBO_V3", {"options": [
        {"key": "fine", "inputs": {"required": {
            "grid": ["COMFY_DYNAMICCOMBO_V3", {"options": [
                {"key": "square", "inputs": {"required": {"side": ["INT", {}]}}},
                {"key": "free", "inputs": {"required": {"width": ["INT", {}],
                                                        "height": ["INT", {}]}}}]}],
            "passes": ["INT", {}]}}},
        {"key": "plain", "inputs": {}}]}], "label": ["STRING", {}]}}
    api = convert.convert(_node("Nested", ["fine", "free", 640, 480, 3, "done"]),
                          _schema_names(schema))
    assert api["2"]["inputs"] == {"mode": "fine", "mode.grid": "free", "mode.grid.width": 640,
                                  "mode.grid.height": 480, "mode.passes": 3, "label": "done"}


def test_a_child_named_like_a_top_level_input_stays_with_that_input():
    """SaveVideo at ComfyUI 0.44.0 and 0.45.0: every ``format`` option reveals ``codec``,
    and a hidden top-level ``codec`` keeps the slot of graphs saved before it did. Every
    committed MP4 template's third value has always been read as that top-level input."""
    codec = ["COMFY_DYNAMICCOMBO_V3", {"options": [
        {"key": "auto", "inputs": {}},
        {"key": "h264", "inputs": {"optional": {"encoding": ["COMFY_DYNAMICCOMBO_V3", {
            "options": [{"key": "automatic", "inputs": {}}]}]}}}]}]
    schema = {"required": {
        "video": ["VIDEO", {}],
        "filename_prefix": ["STRING", {"default": "video/ComfyUI"}],
        "format": ["COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "auto", "inputs": {"required": {"codec": codec}}},
            {"key": "mp4", "inputs": {"required": {"codec": codec}}}]}]},
        "optional": {"codec": [codec[0], {**codec[1], "hidden": True}]}}
    names = convert.object_info_widget_names(schema)
    assert names == ["filename_prefix", "format", "codec"]
    api = convert.convert(_node("SaveVideo", ["video/out", "mp4", "h264"]), lambda _c: names)
    assert api["2"]["inputs"] == {"filename_prefix": "video/out", "format": "mp4", "codec": "h264"}
    flat: dict = {}
    _flat_read([str(name) for name in names], ["video/out", "mp4", "h264"], flat)
    assert flat == api["2"]["inputs"]


# ---- The generator's table -----------------------------------------------------

def test_the_generator_lists_generate_texts_children_in_upstreams_order():
    names = convert.repo_widget_names(ROOT)
    api = convert.convert(_node("TextGenerateV2", UPSTREAM_GENERATE_TEXT), names)
    assert api["2"]["inputs"] == GENERATE_TEXT_ON
    # The generator writes the same thirteen values for upstream's decoding.
    upstream = {"sampling_mode": "on", "sampling_mode.temperature": 1.0, "sampling_mode.top_k": 20,
                "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.0,
                "sampling_mode.repetition_penalty": 1.0, "sampling_mode.presence_penalty": 0.0,
                "sampling_mode.seed": 0}
    assert list(upstream) == list(gen_templates.H3I_DECODING)
    spec = {"title": "probe", "nodes": [
        {"key": "expand", "type": "TextGenerateV2", "pos": (0, 0),
         "widgets": {"prompt": "", "max_length": 8192, **upstream,
                     "thinking": True, "use_default_template": True, "mtp": "auto"}}]}
    (node,) = gen_templates.build(spec)["nodes"]
    assert node["widgets_values"] == UPSTREAM_GENERATE_TEXT
    assert [type(value) for value in node["widgets_values"][3:10]] == [
        float, int, float, float, float, float, int]


def test_the_generators_table_is_closed_and_a_key_it_does_not_list_stops_the_read():
    names = convert.repo_widget_names(ROOT)
    (combo,) = [name for name in names("ResizeImageMaskNode")
                if isinstance(name, convert.DynamicName)]
    assert combo.closed and set(combo.children) == {"scale total pixels", "scale to multiple"}
    with pytest.raises(ValueError, match="DYNAMIC_WIDGETS does not list the key 'scale by multiplier'"):
        convert.convert(_node("ResizeImageMaskNode", ["scale by multiplier", 2.0, "area"]), names)
    with pytest.raises(KeyError, match="DYNAMIC_WIDGETS lists"):
        gen_templates.build({"title": "probe", "nodes": [
            {"key": "scale", "type": "ResizeImageMaskNode", "pos": (0, 0),
             "widgets": {"resize_type": "scale by multiplier", "scale_method": "area"}}]})
    # A revealed child the spec does not set stops the build, never ships a default.
    with pytest.raises(KeyError, match=r"resize_type\.megapixels"):
        gen_templates.build({"title": "probe", "nodes": [
            {"key": "scale", "type": "ResizeImageMaskNode", "pos": (0, 0),
             "widgets": {"resize_type": "scale total pixels", "scale_method": "area"}}]})


def test_the_edit_graph_keeps_upstreams_three_values_on_each_resize_node():
    doc = json.loads((ROOT / "example_workflows" / f"{EDIT}.json").read_text(encoding="utf-8"))
    stored = [node["widgets_values"] for node in doc["nodes"]
              if node["type"] == "ResizeImageMaskNode"]
    # image_minimax_h3_image_edit stores ["scale total pixels", 4, "area"] and
    # ["scale to multiple", 32, "area"], in this order along the picture's path.
    assert stored == [["scale total pixels", 4, "area"], ["scale to multiple", 32, "area"]]
    api = convert.convert(doc, convert.repo_widget_names(ROOT))
    resized = [node["inputs"] for node in api.values() if node["class_type"] == "ResizeImageMaskNode"]
    assert [{key: value for key, value in inputs.items() if key != "input"} for inputs in resized] == [
        {"resize_type": "scale total pixels", "resize_type.megapixels": 4.0, "scale_method": "area"},
        {"resize_type": "scale to multiple", "resize_type.multiple": 32, "scale_method": "area"}]


# ---- No shift on any other graph -------------------------------------------------

def _reveals(node_class: str, inputs: dict, names) -> bool:
    return any(isinstance(name, convert.DynamicName) and name not in inputs_links(inputs)
               and name.revealed(inputs.get(name)) for name in names(node_class))


def inputs_links(inputs: dict) -> set[str]:
    return {name for name, value in inputs.items() if isinstance(value, list)}


def test_every_committed_node_that_reveals_no_child_reads_as_the_flat_reader_read_it():
    """The reader every graph had: one slot a name. Only a node whose dynamic combo
    reveals children may read otherwise, and the committed set holds exactly five:
    the edit graph's two resize nodes, and Generate Text with ComfyUI's decoding
    numbers in the three MiniMax-H3-Image graphs."""
    names = convert.repo_widget_names(ROOT)
    assert len(TEMPLATES) > 100
    revealing = []
    for path in TEMPLATES:
        ui = json.loads(path.read_text(encoding="utf-8"))
        api = convert.convert(ui, names)
        links = {link[0]: [str(link[1]), link[2]] for link in ui["links"]}
        for node in ui["nodes"]:
            if node["type"] in convert.SKIP_TYPES:
                continue
            flat = {slot["name"]: list(links[slot["link"]])
                    for slot in node.get("inputs") or [] if slot.get("link") is not None}
            _flat_read([str(name) for name in names(node["type"])],
                       list(node.get("widgets_values") or []), flat)
            read = api[str(node["id"])]["inputs"]
            if _reveals(node["type"], read, names):
                revealing.append((path.stem, node["type"]))
                assert flat != read, (path.stem, node["id"])
                continue
            assert flat == read, f"{path.stem}: node {node['id']} ({node['type']}) reads differently"
    assert sorted(revealing) == [
        (EDIT, "ResizeImageMaskNode"), (EDIT, "ResizeImageMaskNode"), (EDIT, "TextGenerateV2"),
        ("dgx-monarch-minimax-h3-image-t2i", "TextGenerateV2"),
        ("dgx-monarch-test-minimax-h3-image-grid", "TextGenerateV2")]
