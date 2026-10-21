"""Studio's prompt checks follow ComfyUI's own validation, read from execution.py.

ComfyUI's ``validate_inputs`` checks only the inputs a node declares once its V3
dynamic inputs are finalized from the prompt, and ``validate_node_input`` types
links by overlap. A node whose own validator takes ``input_types`` skips the
link check in ComfyUI; object_info does not show that, so these tests do not
model it. The node rows come from the 2026-10-16 recording that
``fixtures/studio/README.md`` describes, with tooltips dropped and file names
replaced by neutral ones.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from dgx_monarch.studio.drafts import DraftError, validate_value
from dgx_monarch.studio.workflows import WorkflowError, compile_workflow

FIXTURE = json.loads((Path(__file__).parent / "fixtures/studio/object_info_bundled.json").read_text())
# In the recording, core KSampler and rgthree's KSampler Config use these same lists.
SAMPLERS = FIXTURE["DGXMonarchKSampler"]["input"]["required"]["sampler_name"][0]
SCHEDULERS = FIXTURE["DGXMonarchKSampler"]["input"]["required"]["scheduler"][0]

RECORDED_ROWS = {
    "Power Lora Loader (rgthree)": {
        "input": {"required": {}, "optional": {"model": ["MODEL"], "clip": ["CLIP"]}, "hidden": {}},
        "input_order": {"required": [], "optional": ["model", "clip"], "hidden": []},
        "output": ["MODEL", "CLIP"], "output_is_list": [False, False], "output_name": ["MODEL", "CLIP"],
        "output_node": False, "name": "Power Lora Loader (rgthree)", "python_module": "custom_nodes.rgthree-comfy"},
    "KSampler Config (rgthree)": {
        "input": {"required": {
            "steps_total": ["INT", {"default": 30, "min": 1, "max": 16384, "step": 1}],
            "refiner_step": ["INT", {"default": 24, "min": 1, "max": 16384, "step": 1}],
            "cfg": ["FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5}],
            "sampler_name": [SAMPLERS], "scheduler": [SCHEDULERS]}},
        "input_order": {"required": ["steps_total", "refiner_step", "cfg", "sampler_name", "scheduler"]},
        "output": ["INT", "INT", "FLOAT", SAMPLERS, SCHEDULERS], "output_is_list": [False] * 5,
        "output_name": ["STEPS", "REFINER_STEP", "CFG", "SAMPLER", "SCHEDULER"], "output_node": False,
        "name": "KSampler Config (rgthree)", "python_module": "custom_nodes.rgthree-comfy"},
    "BatchImagesNode": {
        "input": {"required": {"images": ["COMFY_AUTOGROW_V3", {"template": {
            "input": {"required": {"image": ["IMAGE", {}]}}, "prefix": "image", "min": 1, "max": 50}}]}},
        "input_order": {"required": ["images"]}, "output": ["IMAGE"], "output_is_list": [False],
        "output_name": ["IMAGE"], "output_node": False, "name": "BatchImagesNode",
        "python_module": "comfy_extras.nodes_post_processing"},
    "ComfySwitchNode": {
        "input": {"required": {"switch": ["BOOLEAN", {}]}, "optional": {
            "on_false": ["COMFY_MATCHTYPE_V3", {"lazy": True, "template": {"template_id": "switch", "allowed_types": "*"}}],
            "on_true": ["COMFY_MATCHTYPE_V3", {"lazy": True, "template": {"template_id": "switch", "allowed_types": "*"}}]}},
        "input_order": {"required": ["switch"], "optional": ["on_false", "on_true"]},
        "output": ["COMFY_MATCHTYPE_V3"], "output_is_list": [False], "output_name": ["output"],
        "output_node": False, "name": "ComfySwitchNode", "python_module": "comfy_extras.nodes_logic"},
    "PrimitiveStringMultiline": {
        "input": {"required": {"value": ["STRING", {"multiline": True}]}}, "input_order": {"required": ["value"]},
        "output": ["STRING"], "output_is_list": [False], "output_name": ["STRING"], "output_node": False,
        "name": "PrimitiveStringMultiline", "python_module": "comfy_extras.nodes_primitive"},
    "PrimitiveInt": {
        "input": {"required": {"value": ["INT", {"min": -9223372036854775807, "max": 9223372036854775807,
                                                  "control_after_generate": "fixed"}]}},
        "input_order": {"required": ["value"]}, "output": ["INT"], "output_is_list": [False],
        "output_name": ["INT"], "output_node": False, "name": "PrimitiveInt",
        "python_module": "comfy_extras.nodes_primitive"},
    "PreviewAny": {
        "input": {"required": {"source": ["*", {}]}}, "input_order": {"required": ["source"]},
        "output": ["STRING"], "output_is_list": [False], "output_name": ["STRING"], "output_node": True,
        "name": "PreviewAny", "python_module": "comfy_extras.nodes_preview_any"},
    "LoadLoraModel": {
        "input": {"required": {"model": ["MODEL", {}], "loras": ["COMFY_DYNAMICGROUP_V3", {
            "min": 1, "max": 20, "group_name": "LoRA", "template": {"required": {
                "lora_name": ["COMBO", {"multiselect": False, "options": ["lora_a.safetensors"]}],
                "strength": ["FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}],
                "enabled": ["BOOLEAN", {"default": True}]}}}]}},
        "input_order": {"required": ["model", "loras"]}, "output": ["MODEL"], "output_name": ["MODEL"],
        "output_node": False, "name": "LoadLoraModel", "python_module": "comfy_extras.nodes_lora_stack"},
}


@pytest.fixture
def info():
    return {**copy.deepcopy(FIXTURE), **copy.deepcopy(RECORDED_ROWS)}


def node(kind, **inputs):
    return {"class_type": kind, "inputs": inputs}


def test_undeclared_inputs_pass_through_unvalidated(info):
    # Power Lora Loader declares only model and clip; its frontend adds lora_1..N.
    prompt = {"1": node("Power Lora Loader (rgthree)",
                        PowerLoraLoaderHeaderWidget={"type": "PowerLoraLoaderHeaderWidget"},
                        lora_1={"on": True, "lora": "lora_a.safetensors", "strength": 0.8},
                        lora_2={"on": False, "lora": "lora_b.safetensors", "strength": 1}),
              "2": node("PreviewAny", source=["1", 0])}
    assert compile_workflow(copy.deepcopy(prompt), info) == prompt
    prompt["1"]["inputs"]["lora_3"] = ["9", 0]
    with pytest.raises(WorkflowError, match=re.escape("Link references missing node: 1.lora_3")):
        compile_workflow(prompt, info)


def test_list_typed_outputs_feed_combos_as_comfyui_allows(info):
    config = node("KSampler Config (rgthree)", steps_total=30, refiner_step=24, cfg=8.0,
                  sampler_name="euler", scheduler="normal")
    prompt = {"1": config,
              "2": node("KSampler Config (rgthree)", steps_total=30, refiner_step=24, cfg=8.0,
                        sampler_name=["1", 3], scheduler=["1", 4]),
              "3": node("KSamplerSelect", sampler_name=["1", 3]),
              "4": node("PreviewAny", source=["2", 0]),
              "5": node("PreviewAny", source=["3", 0])}
    assert compile_workflow(copy.deepcopy(prompt), info) == prompt
    # The scheduler list is not the sampler list, and a legacy list input takes only its own list.
    prompt["2"]["inputs"]["sampler_name"] = ["1", 4]
    with pytest.raises(WorkflowError, match=re.escape("Incompatible input link: 2.sampler_name")):
        compile_workflow(prompt, info)


def test_string_links_into_combos_refuse_as_comfyui_refuses(info):
    for target in (node("KSamplerSelect", sampler_name=["1", 0]),
                   node("KSampler Config (rgthree)", steps_total=30, refiner_step=24, cfg=8.0,
                        sampler_name=["1", 0], scheduler="normal")):
        prompt = {"1": node("PrimitiveStringMultiline", value="euler"), "2": target,
                  "3": node("PreviewAny", source=["2", 0])}
        with pytest.raises(WorkflowError, match=re.escape("Incompatible input link: 2.sampler_name")):
            compile_workflow(prompt, info)


def test_autogrow_inputs_are_finalized_from_the_prompt(info):
    prompt = {"1": node("LoadImage", image="example.png"),
              "2": node("BatchImagesNode", **{"images.image0": ["1", 0], "images.image1": ["1", 0]}),
              "3": node("SaveImage", images=["2", 0], filename_prefix="batch")}
    assert compile_workflow(copy.deepcopy(prompt), info) == prompt
    del prompt["2"]["inputs"]["images.image0"]
    with pytest.raises(WorkflowError, match=re.escape("Required input is missing: 2.images.image0")):
        compile_workflow(prompt, info)


def test_dynamic_combo_inputs_are_finalized_from_the_prompt(info):
    save = node("SaveVideo", video=["1", 0], filename_prefix="clip", format="mp4",
                **{"format.codec": "h264", "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 18.0})
    prompt = {"1": node("LoadVideo", file="wan_animate2_driving.mp4"), "2": save}
    assert compile_workflow(copy.deepcopy(prompt), info) == prompt
    edited = compile_workflow(copy.deepcopy(prompt), info, {"root:2:format.codec.encoding.crf": 30.0})
    assert edited["2"]["inputs"]["format.codec.encoding.crf"] == 30.0
    # ComfyUI drops an absent child, and SaveVideo then falls back to its automatic codec.
    bare = copy.deepcopy(prompt)
    bare["2"]["inputs"] = {"video": ["1", 0], "filename_prefix": "clip", "format": "mp4"}
    assert compile_workflow(copy.deepcopy(bare), info) == bare
    for name, value, message in (("format.codec.encoding.crf", 99.0, "Value violates max=51.0"),
                                 ("format", "mov", "Value is not an available choice")):
        broken = copy.deepcopy(prompt)
        broken["2"]["inputs"][name] = value
        with pytest.raises(WorkflowError, match=re.escape(f"2.{name}: {message}")):
            compile_workflow(broken, info)


def test_union_and_match_types_link_by_overlap(info):
    prompt = {"1": node("PrimitiveInt", value=24),
              "2": node("VAELoader", vae_name="LTX23_audio_vae_bf16.safetensors"),
              "3": node("LTXVEmptyLatentAudio", frames_number=97, frame_rate=["1", 0], batch_size=1,
                        audio_vae=["2", 0]),
              "4": node("PreviewAny", source=["3", 0]),
              "5": node("LoadImage", image="example.png"),
              "6": node("ComfySwitchNode", switch=True, on_true=["5", 0], on_false=["5", 0]),
              "7": node("SaveImage", images=["6", 0], filename_prefix="switch")}
    assert compile_workflow(copy.deepcopy(prompt), info) == prompt
    prompt["1"] = node("PrimitiveStringMultiline", value="24")
    with pytest.raises(WorkflowError, match=re.escape("Incompatible input link: 3.frame_rate")):
        compile_workflow(prompt, info)


def test_union_literals_take_any_listed_scalar_type():
    schema = {"type": "INT,FLOAT", "min": 1}
    assert validate_value(2, schema) == 2 and validate_value(2.5, schema) == 2.5
    for value in ("2", 0, True):
        with pytest.raises(DraftError):
            validate_value(value, schema)


def test_unsupported_dynamic_input_kinds_refuse_explicitly(info):
    prompt = {"1": node("PreviewAny", source=["2", 0]),
              "2": node("LoadLoraModel", model=["3", 0], **{"loras.0.lora_name": "lora_a.safetensors",
                                                           "loras.0.strength": 1.0, "loras.0.enabled": True}),
              "3": node("Power Lora Loader (rgthree)")}
    with pytest.raises(WorkflowError, match="COMFY_DYNAMICGROUP_V3, which needs an explicit adapter"):
        compile_workflow(prompt, info)
