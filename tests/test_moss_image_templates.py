"""The MOSS-Image templates hold the vendor's recipe and their testing legs change one thing.

The three CFG 3 graphs (SFT, Base, Kromoss) and the SFT layout graph run 50
Euler steps on ComfyUI's simple scheduler at CFG 3, 1024 square, with the
vendor's negative tags and the vendor's skip of block 22 on the negative pass
through Skip Layer Guidance (DGX Monarch). The Kromoss Turbo graph follows the
Kromoss card: the DMD2 LoRA at strength 1.0, 8 steps, CFG 1, a zeroed negative
and no skip. Every graph loads the Qwen3.5-9B encoder under CLIPLoader type
`moss_image` and the FLUX.2 small decoder, and none places a shift node: the
SD3 node samples at multiplier 1000 and this model at 1.0. The layout graph
feeds its JSON to CLIP Text Encode through a link, because the browser's dynamic
prompt pass would rewrite braces typed into that box. CPU only: the graphs are
read as JSON and through the sweep converter.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmark.sweep import convert

from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.skip_layer_guidance import DGXMonarchSkipLayerGuidance
from dgx_monarch.skip_layer_guidance import skip_layer_guidance_active

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "example_workflows"
TESTING = ROOT / "tests" / "fixtures" / "workflows" / "generated"

SFT, BASE, KROMOSS = ("moss_image_sft_bf16.safetensors", "moss_image_base_bf16.safetensors",
                      "kromoss_bf16.safetensors")
ENCODER, ENCODER_INT8 = "qwen3.5_9b_bf16.safetensors", "qwen3.5_9b_int8_convrot.safetensors"
VAE = "flux2-vae_small.safetensors"
DMD2, DMD = "kromoss_8-steps-dmd2-lora.safetensors", "kromoss_8-steps-dmd-lora.safetensors"
NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
VENDOR_SKIP = {"double_layers": "", "single_layers": "22", "start_percent": 0.0,
               "end_percent": 1.0}
LATENT_RATIO = 16  # supported_models.MOSSImage

# name: (DiT, steps, cfg, LoRA or None); a graph without a LoRA carries the skip.
RECIPES = {
    "dgx-monarch-moss-image-t2i": (SFT, 50, 3.0, None),
    "dgx-monarch-moss-image-base-t2i": (BASE, 50, 3.0, None),
    "dgx-monarch-moss-image-layout-t2i": (SFT, 50, 3.0, None),
    "dgx-monarch-kromoss-t2i": (KROMOSS, 50, 3.0, None),
    "dgx-monarch-kromoss-t2i-turbo": (KROMOSS, 8, 1.0, DMD2),
}
# leg: (base graph, {class: {widget: value}}): the only inputs the leg changes.
LEGS = {
    "dgx-monarch-test-moss-image-int8": ("dgx-monarch-moss-image-t2i", {
        "DGXMonarchUNETLoader": {"unet_name": "moss_image_sft_int8_convrot.safetensors"},
        "CLIPLoader": {"clip_name": ENCODER_INT8}}),
    "dgx-monarch-test-moss-image-base-int8": ("dgx-monarch-moss-image-base-t2i", {
        "DGXMonarchUNETLoader": {"unet_name": "moss_image_base_int8_convrot.safetensors"}}),
    "dgx-monarch-test-kromoss-int8": ("dgx-monarch-kromoss-t2i", {
        "DGXMonarchUNETLoader": {"unet_name": "kromoss_int8_convrot.safetensors"}}),
    "dgx-monarch-test-moss-image-odd": ("dgx-monarch-moss-image-t2i", {
        "EmptyFlux2LatentImage": {"width": 1040, "height": 1040},
        "SaveImage": {"filename_prefix": "dgx_monarch_moss_image_odd"}}),
    "dgx-monarch-test-moss-image-slg-off": ("dgx-monarch-moss-image-t2i", {
        "DGXMonarchSkipLayerGuidance": {"single_layers": ""},
        "SaveImage": {"filename_prefix": "dgx_monarch_moss_image_slg_off"}}),
    "dgx-monarch-test-moss-image-1536px": ("dgx-monarch-moss-image-t2i", {
        "EmptyFlux2LatentImage": {"width": 1536, "height": 1536},
        "SaveImage": {"filename_prefix": "dgx_monarch_moss_image_1536"}}),
    "dgx-monarch-test-kromoss-turbo-dmd": ("dgx-monarch-kromoss-t2i-turbo", {
        "DGXMonarchLoraLoader": {"lora_name": DMD},
        "SaveImage": {"filename_prefix": "dgx_monarch_kromoss_turbo_dmd"}}),
    "dgx-monarch-test-moss-image-base-turbo": ("dgx-monarch-kromoss-t2i-turbo", {
        "DGXMonarchUNETLoader": {"unet_name": BASE},
        "SaveImage": {"filename_prefix": "dgx_monarch_moss_image_base_turbo"}}),
}
_FORBIDDEN = ("SkipLayerGuidanceDiTSimple", "LoraLoaderModelOnly", "UNETLoader", "KSampler")
_DYNAMIC_PROMPT = ("{", "|", "//", "/*")


def _ui(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _api(path: Path) -> dict:
    return convert.convert(_ui(path), convert.repo_widget_names(ROOT))


def _one(graph: dict, kind: str) -> tuple[str, dict]:
    found = [(key, node["inputs"]) for key, node in graph.items() if node["class_type"] == kind]
    assert len(found) == 1, f"expected one {kind}, found {len(found)}"
    return found[0]


def _note(doc: dict) -> str:
    notes = [node["widgets_values"][0] for node in doc["nodes"] if node["type"] == "Note"]
    assert len(notes) == 1
    return notes[0]


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_the_shipped_graph_holds_the_vendor_recipe(name):
    dit, steps, cfg, lora = RECIPES[name]
    graph = _api(SHIPPED / f"{name}.json")
    init_id, init = _one(graph, "DGXMonarchInit")
    unet_id, unet = _one(graph, "DGXMonarchUNETLoader")
    _, clip = _one(graph, "CLIPLoader")
    _, vae = _one(graph, "VAELoader")
    latent_id, latent = _one(graph, "EmptyFlux2LatentImage")
    _, sampler = _one(graph, "DGXMonarchKSampler")
    assert (init["topology"], init["mode"], init["auto_gate"]) == ("auto", "auto", "first_use")
    assert (unet["unet_name"], unet["weight_dtype"], unet["mesh"]) == (dit, "default", [init_id, 0])
    assert (clip["clip_name"], clip["type"], clip["device"]) == (ENCODER, "moss_image", "default")
    assert vae["vae_name"] == VAE
    assert (latent["width"], latent["height"], latent["batch_size"]) == (1024, 1024, 1)
    assert (sampler["seed"], sampler["steps"], sampler["cfg"]) == (42, steps, cfg)
    assert (sampler["sampler_name"], sampler["scheduler"], sampler["denoise"]) == (
        "euler", "simple", 1.0)
    assert sampler["latent_image"] == [latent_id, 0]
    classes = {node["class_type"] for node in graph.values()}
    assert not {kind for kind in classes if "ModelSampling" in kind}, "native shift 5 only"
    assert not classes & set(_FORBIDDEN)
    model = graph[sampler["model"][0]]
    positive_id = sampler["positive"][0]
    negative = graph[sampler["negative"][0]]
    assert graph[positive_id]["class_type"] == "CLIPTextEncode"
    if lora is None:
        assert model["class_type"] == "DGXMonarchSkipLayerGuidance"
        assert {key: model["inputs"][key] for key in VENDOR_SKIP} == VENDOR_SKIP
        assert model["inputs"]["model"] == [unet_id, 0]
        assert skip_layer_guidance_active({key: model["inputs"][key] for key in VENDOR_SKIP})
        assert negative["class_type"] == "CLIPTextEncode"
        assert negative["inputs"]["text"] == NEGATIVE
        assert "DGXMonarchLoraLoader" not in classes
    else:
        assert model["class_type"] == "DGXMonarchLoraLoader"
        assert (model["inputs"]["lora_name"], model["inputs"]["strength_model"]) == (lora, 1.0)
        assert model["inputs"]["model"] == [unet_id, 0]
        assert negative["class_type"] == "ConditioningZeroOut"
        assert negative["inputs"]["conditioning"] == [positive_id, 0]
        assert "DGXMonarchSkipLayerGuidance" not in classes


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_every_note_names_its_files_its_floors_and_its_limits(name):
    dit, _steps, _cfg, lora = RECIPES[name]
    note = _note(_ui(SHIPPED / f"{name}.json"))
    for required in (dit, ENCODER, VAE, ENCODER_INT8, "CLIP type moss_image",
                     "simple scheduler", "add no shift node", "Apache-2.0",
                     "0.42.0 or newer", "0.2.39 or newer", "Sol-Attn waiver",
                     "Auto picks Ulysses on a configured pair.",
                     "No hardware comparison has run for MOSS-Image yet"):
        assert required in note, f"{name}: the Note does not name {required!r}"
    if lora is None:
        assert "cfg2 refuses while Skip Layer Guidance is on" in note
    else:
        assert lora in note and DMD in note and "use one LoRA at a time" in note


def test_the_layout_graph_wires_its_json_to_the_text_socket():
    doc = _ui(SHIPPED / "dgx-monarch-moss-image-layout-t2i.json")
    nodes = {node["id"]: node for node in doc["nodes"]}
    links = {link[0]: link for link in doc["links"]}
    graph = _api(SHIPPED / "dgx-monarch-moss-image-layout-t2i.json")
    _, sampler = _one(graph, "DGXMonarchKSampler")
    positive = nodes[int(sampler["positive"][0])]
    assert positive["inputs"][0]["name"] == "clip"
    socket = positive["inputs"][1]
    assert {key: socket[key] for key in ("name", "type", "widget")} == {
        "name": "text", "type": "STRING", "widget": {"name": "text"}}
    _link_id, source, slot, target, target_slot, kind = links[socket["link"]]
    assert (nodes[source]["type"], slot, target, target_slot, kind) == (
        "PrimitiveStringMultiline", 0, positive["id"], 1, "STRING")
    assert socket["link"] in nodes[source]["outputs"][0]["links"]
    # The widget keeps its slot, empty, as the frontend serializes a linked widget.
    assert positive["widgets_values"] == [""]
    assert graph[str(positive["id"])]["inputs"]["text"] == [str(source), 0]
    layout = json.loads(nodes[source]["widgets_values"][0])
    assert (layout["bbox_coord_space"], layout["bbox_format"]) == ("abs_centralized", "yxyx")
    assert layout["summary"]["caption"] and layout["items"]
    half = 1024 // 2
    for item in layout["items"]:
        y_min, x_min, y_max, x_max = item["bbox"]
        assert -half <= y_min < y_max <= half and -half <= x_min < x_max <= half, item
        assert item["label"] in {"text", "object", "character"}


def _moss_graphs() -> list[Path]:
    shipped = [SHIPPED / f"{name}.json" for name in RECIPES]
    return shipped + [TESTING / f"{name}.json" for name in (*LEGS, "dgx-monarch-test-moss-image-lora")]


@pytest.mark.parametrize("path", _moss_graphs(), ids=lambda path: path.stem)
def test_no_prompt_carries_dynamic_prompt_syntax(path):
    # CLIP Text Encode marks its text for the browser's dynamic prompt pass,
    # which rewrites these on Queue; the API path keeps them, so a graph would
    # render differently from the browser and from the sweep.
    texts = [node["inputs"]["text"] for node in _api(path).values()
             if node["class_type"] == "CLIPTextEncode" and isinstance(node["inputs"]["text"], str)]
    assert texts
    for text in texts:
        assert not [mark for mark in _DYNAMIC_PROMPT if mark in text], (path.stem, text[:60])


def test_the_shipped_set_is_five_graphs():
    shipped = {path.stem for pattern in ("dgx-monarch-moss-image-*.json", "dgx-monarch-kromoss-*.json")
               for path in SHIPPED.glob(pattern)}
    assert shipped == set(RECIPES)
    testing = {path.stem for pattern in ("dgx-monarch-test-moss-image-*.json",
                                         "dgx-monarch-test-kromoss-*.json")
               for path in TESTING.glob(pattern)}
    assert testing == {*LEGS, "dgx-monarch-test-moss-image-lora"}


@pytest.mark.parametrize("leg", sorted(LEGS))
def test_a_testing_leg_changes_only_its_stated_widgets(leg):
    base, changes = LEGS[leg]
    graph, shipped = _api(TESTING / f"{leg}.json"), _api(SHIPPED / f"{base}.json")
    assert set(graph) == set(shipped)
    for node_id, node in shipped.items():
        want = {**node["inputs"], **changes.get(node["class_type"], {})}
        assert graph[node_id] == {"class_type": node["class_type"], "inputs": want}, node_id
    assert set(changes) <= {node["class_type"] for node in shipped.values()}


@pytest.mark.parametrize(("leg", "side", "rows"), [
    ("dgx-monarch-test-moss-image-odd", 1040, 4225),
    ("dgx-monarch-test-moss-image-1536px", 1536, 9216),
])
def test_the_canvas_legs_set_their_image_rows(leg, side, rows):
    _, latent = _one(_api(TESTING / f"{leg}.json"), "EmptyFlux2LatentImage")
    assert (latent["width"], latent["height"]) == (side, side)
    assert (side // LATENT_RATIO) ** 2 == rows
    # The odd grid is the point of the odd leg; the large canvas keeps it even.
    assert rows % 2 == (1 if side == 1040 else 0)
    note = _note(_ui(TESTING / f"{leg}.json"))
    assert f"{rows} image rows" in note


def test_the_skip_off_leg_is_the_no_node_request():
    _, skip = _one(_api(TESTING / "dgx-monarch-test-moss-image-slg-off.json"),
                   "DGXMonarchSkipLayerGuidance")
    widgets = {key: skip[key] for key in VENDOR_SKIP}
    assert not skip_layer_guidance_active(widgets)
    # The node freezes the leg's widgets to no skip at all, so the request it
    # sends is byte for byte the request of the same graph without the node.
    mesh = MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                    attention="TORCH_FLASH", sync_ulysses=True)
    model = ModelSpec(mesh=mesh, unet_name=SFT, options={"weight_dtype": "default"})
    patched, = DGXMonarchSkipLayerGuidance().patch(model, **widgets)
    assert patched.skip_layer_guidance is None
    assert json.dumps(patched.request_dict(), sort_keys=True) == json.dumps(
        model.request_dict(), sort_keys=True)


def test_the_lora_leg_splices_the_placeholder_before_the_skip():
    graph = _api(TESTING / "dgx-monarch-test-moss-image-lora.json")
    unet_id, _ = _one(graph, "DGXMonarchUNETLoader")
    lora_id, lora = _one(graph, "DGXMonarchLoraLoader")
    _, skip = _one(graph, "DGXMonarchSkipLayerGuidance")
    assert lora["model"] == [unet_id, 0] and skip["model"] == [lora_id, 0]
    assert (lora["lora_name"], lora["strength_model"]) == ("test_lora_placeholder.safetensors", 0.8)


def test_the_turbo_legs_keep_the_card_recipe():
    for leg in ("dgx-monarch-test-kromoss-turbo-dmd", "dgx-monarch-test-moss-image-base-turbo"):
        graph = _api(TESTING / f"{leg}.json")
        _, sampler = _one(graph, "DGXMonarchKSampler")
        _, lora = _one(graph, "DGXMonarchLoraLoader")
        assert (sampler["steps"], sampler["cfg"], lora["strength_model"]) == (8, 1.0, 1.0)
        assert "DGXMonarchSkipLayerGuidance" not in {node["class_type"] for node in graph.values()}
    note = _note(_ui(TESTING / "dgx-monarch-test-kromoss-turbo-dmd.json"))
    assert re.search(r"\b8-steps-dmd-lora\.safetensors upstream\b", note)
