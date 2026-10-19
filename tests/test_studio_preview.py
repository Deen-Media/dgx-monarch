"""Submission preview transformations preserve explicit graph ownership."""

import asyncio
from copy import deepcopy

import pytest

from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.preview_drafts import prepare_preview
from dgx_monarch.studio.store import Store
from dgx_monarch.studio.workflows import WorkflowError


@pytest.fixture
def schema():
    return {
        "Model": {"input": {"required": {}}, "output": ["MODEL"]},
        "Text": {"input": {"required": {"text": ["STRING", {"default": ""}]}}, "output": ["STRING"]},
        "LoraLoaderModelOnly": {"input": {"required": {
            "model": ["MODEL"], "lora_name": [["a.safetensors", "b.safetensors"]],
            "strength_model": ["FLOAT", {"min": -2, "max": 2, "default": 0.5}]}}, "output": ["MODEL"]},
        "Output": {"input": {"required": {"model": ["MODEL"], "positive": ["STRING"], "negative": ["STRING"]}},
                   "output": [], "output_node": True},
    }


@pytest.fixture
def request_body():
    return {"workflow": {
        "1": {"class_type": "Model", "inputs": {}},
        "2": {"class_type": "Text", "inputs": {"text": "cat"}},
        "3": {"class_type": "Text", "inputs": {"text": "blur"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0], "lora_name": "a.safetensors", "strength_model": 0.4}},
        "5": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["4", 0], "lora_name": "b.safetensors", "strength_model": 0.6}},
        "6": {"class_type": "Output", "inputs": {"model": ["5", 0], "positive": ["2", 0], "negative": ["3", 0]}},
    }, "values": {}, "seed_policy": "fixed", "operation": "image"}


def test_styles_use_explicit_binding_order_and_preserve_draft(request_body, schema):
    request_body.update(styles=[{"id": "paint", "prompt": "painting of {prompt}", "negative_prompt": "watermark"},
                                {"id": "frame", "prompt": "framed {prompt}"}],
                        style_bindings={"prompt": "root:2:text", "negative_prompt": "root:3:text"})
    original = deepcopy(request_body)
    prompt, values, effective = prepare_preview(request_body, schema)
    assert prompt["2"]["inputs"]["text"] == "framed painting of cat"
    assert prompt["3"]["inputs"]["text"] == "blur, watermark"
    assert values["root:2:text"] == "framed painting of cat"
    assert effective["styles"]["unexpanded"] == {"prompt": "cat", "negative_prompt": "blur"}
    assert effective["styles"]["style_order"] == ["paint", "frame"]
    assert request_body == original


def test_styles_never_guess_from_labels(request_body, schema):
    request_body["styles"] = [{"id": "a", "prompt": "pretty"}]
    with pytest.raises(StudioError, match="explicit prompt"):
        prepare_preview(request_body, schema)
    request_body["style_bindings"] = {"prompt": "root:4:lora_name"}
    with pytest.raises(StudioError, match="STRING"):
        prepare_preview(request_body, schema)


def test_lora_stack_is_ordered_by_graph_and_preserves_disabled(request_body, schema):
    request_body["lora_stack"] = [
        {"id": "gone", "enabled": False, "model_strength": 100},
        {"id": "b.safetensors", "enabled": True, "model_strength": 1.2},
        {"id": "a.safetensors", "enabled": True, "model_strength": -0.2},
    ]
    prompt, _, effective = prepare_preview(request_body, schema)
    assert prompt["4"]["inputs"]["lora_name"] == "b.safetensors"
    assert prompt["5"]["inputs"]["lora_name"] == "a.safetensors"
    assert prompt["4"]["inputs"]["strength_model"] == 1.2
    assert effective["lora_stack"][0]["enabled"] is False
    assert all(item["compatibility"] == "pending_backend_validation" for item in effective["resolved_loras"])


def test_unfilled_lora_slots_are_explicitly_neutralized(request_body, schema):
    request_body["lora_stack"] = [{"id": "b.safetensors", "model_strength": 0.8}]
    prompt, _, effective = prepare_preview(request_body, schema)
    assert prompt["5"]["inputs"]["strength_model"] == 0
    assert effective["resolved_loras"][1]["enabled"] is False
    assert request_body["workflow"]["5"]["inputs"]["strength_model"] == 0.6


def test_lora_unknown_assets_limits_and_unsupported_strength_refuse(request_body, schema):
    for entry in ({"id": "unknown", "model_strength": 1},
                  {"id": "a.safetensors", "model_strength": 9},
                  {"id": "a.safetensors", "model_strength": 1, "clip_strength": 1}):
        request_body["lora_stack"] = [entry]
        with pytest.raises(ValueError):
            prepare_preview(request_body, schema)


def test_lora_ambiguous_chain_refuses(request_body, schema):
    request_body["workflow"]["5"]["inputs"]["model"] = ["1", 0]
    request_body["lora_stack"] = [{"id": "a.safetensors", "model_strength": 1}]
    with pytest.raises(StudioError, match="multiple chains"):
        prepare_preview(request_body, schema)


def test_separate_model_and_text_lora_strengths(request_body, schema):
    schema["Model"]["output"] = ["MODEL", "CLIP"]
    full = deepcopy(schema["LoraLoaderModelOnly"])
    full["input"]["required"].update(clip=["CLIP"], strength_clip=["FLOAT", {"min": -2, "max": 2, "default": 1.0}])
    full["output"] = ["MODEL", "CLIP"]
    schema["LoraLoader"] = full
    for nid, previous in (("4", "1"), ("5", "4")):
        node = request_body["workflow"][nid]
        node["class_type"] = "LoraLoader"
        node["inputs"].update(clip=[previous, 1], strength_clip=0.5)
    request_body["lora_stack"] = [{"id": "a.safetensors", "model_strength": 0.8, "clip_strength": -0.4}]
    prompt, _, effective = prepare_preview(request_body, schema)
    assert prompt["4"]["inputs"]["strength_clip"] == -0.4
    assert prompt["5"]["inputs"]["strength_clip"] == 0
    assert effective["resolved_loras"][0]["strength_clip"] == -0.4
    request_body["workflow"]["5"]["inputs"]["clip"] = ["1", 1]
    with pytest.raises(StudioError, match="same chain"):
        prepare_preview(request_body, schema)


def test_preview_repairs_imported_invalid_default_and_freezes_effective(request_body, schema, tmp_path):
    request_body["workflow"]["4"]["inputs"]["strength_model"] = 100
    request_body["values"] = {"root:4:strength_model": 1}

    class Client:
        async def request(self, method, path):
            assert (method, path) == ("GET", "/object_info")
            return deepcopy(schema)

    store = Store(tmp_path / "state")
    jobs = Jobs(store, Client())
    preview = asyncio.run(jobs.preview(request_body))
    assert preview["prompt"]["4"]["inputs"]["strength_model"] == 1
    assert preview["requested"]["workflow"]["4"]["inputs"]["strength_model"] == 100
    request_body["values"]["root:4:strength_model"] = 0
    assert store.get("previews", preview["id"])["effective"]["values"]["root:4:strength_model"] == 1
    store.close()


def test_without_stack_existing_workflow_loras_remain(request_body, schema):
    prompt, _, effective = prepare_preview(request_body, schema)
    assert prompt["4"]["inputs"]["strength_model"] == 0.4
    assert len(effective["resolved_loras"]) == 2


def test_connected_style_input_refuses(request_body, schema):
    request_body["workflow"]["2"]["inputs"]["text"] = ["3", 0]
    request_body.update(styles=[{"id": "a", "prompt": "pretty"}], style_bindings={"prompt": "root:2:text"})
    with pytest.raises((WorkflowError, StudioError)):
        prepare_preview(request_body, schema)
