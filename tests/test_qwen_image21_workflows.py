"""CPU-only contract pins for the shipped Qwen Image 2.1 workflows.

These read the generated UI graphs; they never queue ComfyUI or load a
checkpoint. They pin the node classes, the sampler recipe and how each graph
wires the native TextEncodeQwenImage21 node.
"""
import json
from pathlib import Path

from workflow_pin_helpers import assert_uses_only_known_node_classes

HERE = Path(__file__).resolve().parent
WORKFLOWS = HERE.parent / "example_workflows"
T2I = WORKFLOWS / "dgx-monarch-qwen-image21-t2i.json"
EDIT = WORKFLOWS / "dgx-monarch-qwen-image21-edit-rgba.json"
MASK_EDIT = WORKFLOWS / "dgx-monarch-qwen-image21-edit-mask-rgba.json"

_KNOWN_CLASS_TYPES = frozenset({
    "DGXMonarchInit", "DGXMonarchUNETLoader", "DGXMonarchKSampler",
    "CLIPLoader", "VAELoader", "TextEncodeQwenImage21", "LoadImage", "LoadImageMask",
    "JoinImageWithAlpha",
    "VAEDecode", "SaveImage", "Note",
})


def _graph(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _nodes(graph: dict, node_type: str) -> list[dict]:
    return [node for node in graph["nodes"] if node["type"] == node_type]


def _by_id(graph: dict) -> dict[int, dict]:
    return {node["id"]: node for node in graph["nodes"]}


def _node_input(node: dict, name: str):
    return next(item for item in node["inputs"] if item["name"] == name)


def _linked_source(graph: dict, node: dict, input_name: str) -> tuple[int, int]:
    link_id = _node_input(node, input_name)["link"]
    link = next(link for link in graph["links"] if link[0] == link_id)
    return link[1], link[2]


def test_qwen21_templates_use_only_pinned_native_and_monarch_nodes():
    for path in (T2I, EDIT, MASK_EDIT):
        graph = _graph(path)
        prompt = {
            str(node["id"]): {"class_type": node["type"], "inputs": {}}
            for node in graph["nodes"]
        }
        assert_uses_only_known_node_classes(prompt, _KNOWN_CLASS_TYPES)


def test_qwen21_templates_use_the_stock_qwen_image_clip_enum():
    for path in (T2I, EDIT, MASK_EDIT):
        clip = _nodes(_graph(path), "CLIPLoader")[0]
        assert clip["widgets_values"][1] == "qwen_image"


def test_qwen21_text_to_image_is_the_native_1024_euler_simple_cfg1_recipe():
    graph = _graph(T2I)
    cond, sampler = _nodes(graph, "TextEncodeQwenImage21")[0], _nodes(graph, "DGXMonarchKSampler")[0]
    assert cond["widgets_values"] == [
        cond["widgets_values"][0], "blurry, low quality, watermark, text, logo", 1024,
    ]
    assert sampler["dgxm_widgets"]["sampler_name"] == "euler"
    assert sampler["dgxm_widgets"]["scheduler"] == "simple"
    assert sampler["dgxm_widgets"]["cfg"] == 1.0
    assert sampler["dgxm_widgets"]["steps"] == 25
    assert _linked_source(graph, sampler, "positive") == (cond["id"], 0)
    assert _linked_source(graph, sampler, "negative") == (cond["id"], 1)
    assert _linked_source(graph, sampler, "latent_image") == (cond["id"], 2)
    assert not _nodes(graph, "QwenImage21Cache"), "public workflow keeps cache off by default"


def test_qwen21_edit_preserves_one_to_ten_reference_order_and_rgba_inputs():
    graph = _graph(EDIT)
    cond = _nodes(graph, "TextEncodeQwenImage21")[0]
    refs = _nodes(graph, "LoadImage")
    assert len(refs) == 10
    by_id = _by_id(graph)
    for index in range(1, 11):
        source_id, source_slot = _linked_source(graph, cond, f"images.image_{index}")
        assert source_slot == 0
        source = by_id[source_id]
        assert source["type"] == "JoinImageWithAlpha"
        load_id, load_slot = _linked_source(graph, source, "image")
        assert load_slot == 0
        loaded = by_id[load_id]
        assert loaded["type"] == "LoadImage"
        filename = loaded["widgets_values"][0]
        assert filename.endswith("_rgba.png"), filename
        assert _linked_source(graph, source, "alpha") == (load_id, 1)
    first_join = _by_id(graph)[_linked_source(graph, cond, "images.image_1")[0]]
    last_join = _by_id(graph)[_linked_source(graph, cond, "images.image_10")[0]]
    assert _linked_source(graph, first_join, "image")[0] == refs[0]["id"]
    assert _linked_source(graph, last_join, "image")[0] == refs[-1]["id"]
    assert refs[-1]["widgets_values"][0] == "qwen21_mask_rgba.png"
    assert not _nodes(graph, "QwenImage21Cache"), "public workflow keeps cache off by default"


def test_qwen21_masked_edit_reconstructs_one_rgba_reference_from_source_and_mask():
    graph = _graph(MASK_EDIT)
    cond, join = _nodes(graph, "TextEncodeQwenImage21")[0], _nodes(graph, "JoinImageWithAlpha")[0]
    source = _by_id(graph)[_linked_source(graph, join, "image")[0]]
    mask = _by_id(graph)[_linked_source(graph, join, "alpha")[0]]
    assert source["type"] == "LoadImage"
    assert source["widgets_values"][0] == "qwen21_source.png"
    assert mask["type"] == "LoadImageMask"
    assert mask["widgets_values"][:2] == ["qwen21_edit_mask_alpha.png", "alpha"]
    assert _linked_source(graph, cond, "images.image_1") == (join["id"], 0)
    assert not _nodes(graph, "QwenImage21Cache"), "public workflow keeps cache off by default"
