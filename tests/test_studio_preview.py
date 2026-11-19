"""Submission preview transformations preserve explicit graph ownership."""

import asyncio
import json
from copy import deepcopy

import pytest

from dgx_monarch.studio import jobs as jobs_module
from dgx_monarch.studio.contracts import StudioError, digest
from dgx_monarch.studio.history import page_jobs
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
            if path == "/system_stats":
                return {"system": {"comfyui_version": "test-driver"}}
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


def test_lora_append_inserts_typed_nodes_at_chain_tail(request_body, schema):
    request_body["lora_stack"] = [{"id": name, "model_strength": strength} for name, strength in
                                   [("b.safetensors", 1.0), ("a.safetensors", 0.2), ("b.safetensors", -0.7)]]
    before = deepcopy(request_body)
    prompt, values, effective = prepare_preview(request_body, schema)
    inserted = effective["lora_insertion"]["inserted_nodes"]
    assert len(inserted) == 1
    assert prompt[inserted[0]]["inputs"] == {"model": ["5", 0], "lora_name": "b.safetensors", "strength_model": -0.7}
    assert prompt["6"]["inputs"]["model"] == [inserted[0], 0]
    assert values["root:4:lora_name"] == "b.safetensors"
    assert request_body == before


def test_lora_insertion_rewires_only_explicit_consumers_and_freezes_boundary(request_body, schema):
    for nid in ("4", "5"):
        request_body["workflow"].pop(nid)
    request_body["workflow"]["6"]["inputs"]["model"] = ["1", 0]
    request_body["workflow"]["7"] = deepcopy(request_body["workflow"]["6"])
    request_body["lora_stack"] = [{"id": "a.safetensors", "model_strength": 0.8},
                                   {"id": "b.safetensors", "model_strength": -0.1}]
    with pytest.raises(StudioError):
        prepare_preview(request_body, schema)
    request_body["lora_insertion"] = {"loader": "LoraLoaderModelOnly",
                                       "model_targets": [{"node_id": "6", "input_name": "model"}]}
    before = deepcopy(request_body)
    prompt, _, effective = prepare_preview(request_body, schema)
    first, second = effective["lora_insertion"]["inserted_nodes"]
    assert prompt[first]["inputs"]["model"] == ["1", 0]
    assert prompt[second]["inputs"]["model"] == [first, 0]
    assert prompt["6"]["inputs"]["model"] == [second, 0]
    assert prompt["7"]["inputs"]["model"] == ["1", 0]
    assert effective["lora_insertion"]["model_source"] == ["1", 0]
    assert request_body == before
    request_body["lora_insertion"]["model_targets"] = [{"node_id": "6", "input_name": "positive"}]
    with pytest.raises(StudioError, match="MODEL input"):
        prepare_preview(request_body, schema)


def test_dual_lora_insertion_preserves_model_and_clip_chain(request_body, schema):
    schema["Model"]["output"] = ["MODEL", "CLIP"]
    full = deepcopy(schema["LoraLoaderModelOnly"])
    full["input"]["required"].update(clip=["CLIP"], strength_clip=["FLOAT", {"min": -2, "max": 2, "default": 1.0}])
    full["output"] = ["MODEL", "CLIP"]
    schema["LoraLoader"] = full
    schema["Output"]["input"]["required"]["clip"] = ["CLIP"]
    for nid in ("4", "5"):
        request_body["workflow"].pop(nid)
    request_body["workflow"]["6"]["inputs"].update(model=["1", 0], clip=["1", 1])
    request_body["lora_stack"] = [{"id": "a.safetensors", "model_strength": 0.8, "clip_strength": -0.2},
                                   {"id": "b.safetensors", "model_strength": -0.1, "clip_strength": 0.5}]
    request_body["lora_insertion"] = {"loader": "LoraLoader", "model_targets": [{"node_id": "6", "input_name": "model"}],
                                       "clip_targets": [{"node_id": "6", "input_name": "clip"}]}
    prompt, _, effective = prepare_preview(request_body, schema)
    first, second = effective["lora_insertion"]["inserted_nodes"]
    assert prompt[first]["inputs"]["clip"] == ["1", 1]
    assert prompt[second]["inputs"]["clip"] == [first, 1]
    assert prompt["6"]["inputs"]["clip"] == [second, 1]
    assert prompt["6"]["inputs"]["model"] == [second, 0]
    request_body["lora_insertion"].pop("clip_targets")
    with pytest.raises(StudioError, match="preserve"):
        prepare_preview(request_body, schema)


def test_lora_insertion_refuses_different_sources_and_unadvertised_loader_contract(request_body, schema):
    for nid in ("4", "5"):
        request_body["workflow"].pop(nid)
    request_body["workflow"]["6"]["inputs"]["model"] = ["1", 0]
    request_body["workflow"]["7"] = deepcopy(request_body["workflow"]["6"])
    request_body["workflow"]["8"] = deepcopy(request_body["workflow"]["1"])
    request_body["workflow"]["7"]["inputs"]["model"] = ["8", 0]
    request_body["lora_stack"] = [{"id": "a.safetensors", "model_strength": 1}]
    request_body["lora_insertion"] = {"loader": "LoraLoaderModelOnly", "model_targets": [
        {"node_id": "6", "input_name": "model"}, {"node_id": "7", "input_name": "model"}]}
    with pytest.raises(StudioError, match="same source"):
        prepare_preview(request_body, schema)
    request_body["lora_insertion"]["model_targets"].pop()
    schema["LoraLoaderModelOnly"]["output"] = ["IMAGE"]
    with pytest.raises(StudioError, match="typed LoRA contract"):
        prepare_preview(request_body, schema)


def test_monarch_lora_uses_its_actual_model_type_for_existing_and_inserted_chains(request_body, schema):
    from dgx_monarch.constants import MODEL_TYPE
    from dgx_monarch.studio.workflows import inspect_workflow

    schema["Model"]["output"] = [MODEL_TYPE]
    schema["Output"]["input"]["required"]["model"] = [MODEL_TYPE]
    loader = deepcopy(schema["LoraLoaderModelOnly"])
    loader["input"]["required"]["model"] = [MODEL_TYPE]
    loader["output"] = [MODEL_TYPE]
    schema["DGXMonarchLoraLoader"] = loader
    for nid in ("4", "5"):
        request_body["workflow"][nid]["class_type"] = "DGXMonarchLoraLoader"
    request_body["lora_stack"] = [{"id": "a.safetensors", "source_node": "DGXMonarchLoraLoader", "model_strength": 0.7}] * 3
    prompt, _, effective = prepare_preview(request_body, schema)
    inserted = effective["lora_insertion"]["inserted_nodes"]
    assert len(inserted) == 1 and prompt[inserted[0]]["class_type"] == "DGXMonarchLoraLoader"
    for nid in ("4", "5"):
        request_body["workflow"].pop(nid)
    request_body["workflow"]["6"]["inputs"]["model"] = ["1", 0]
    inspection = inspect_workflow(request_body["workflow"], schema)
    assert inspection["connections"] == [{"node_id": "6", "input_name": "model", "type": MODEL_TYPE, "source": ["1", 0]}]
    request_body["lora_insertion"] = {"loader": "DGXMonarchLoraLoader", "model_targets": [{"node_id": "6", "input_name": "model"}]}
    prompt, _, effective = prepare_preview(request_body, schema)
    assert len(effective["lora_insertion"]["inserted_nodes"]) == 3
    assert prompt["6"]["inputs"]["model"] == [effective["lora_insertion"]["inserted_nodes"][-1], 0]


SEED_SCHEMA = {
    "Sampler": {"input": {"required": {
        "seed": ["INT", {"default": 0, "min": 0, "max": 2**64 - 1, "control_after_generate": True}],
        "steps": ["INT", {"default": 20, "min": 1, "max": 100}]}}, "output": ["LATENT"]},
    "Noise": {"input": {"required": {"noise_seed": ["INT", {"default": 0, "min": 0, "max": 2**64 - 1}]}},
              "output": ["NOISE"]},
    "Sink": {"input": {"required": {"latent": ["LATENT"], "noise": ["NOISE"]}}, "output": [], "output_node": True},
}


def seed_request(policy):
    """A draft whose seed fields hold 5 and 9, as a template that ships every seed at 0 would after an edit."""
    return {"workflow": {
        "1": {"class_type": "Sampler", "inputs": {"seed": 0, "steps": 20}},
        "2": {"class_type": "Noise", "inputs": {"noise_seed": 0}},
        "3": {"class_type": "Sink", "inputs": {"latent": ["1", 0], "noise": ["2", 0]}},
    }, "values": {"root:1:seed": 5, "root:2:noise_seed": 9, "root:1:steps": 20},
        "seed_policy": policy, "operation": "image"}


def test_fixed_seed_policy_sends_the_seed_fields_unchanged(tmp_path):
    store = Store(tmp_path / "state")
    try:
        preview = Jobs(store, None).build_preview(seed_request("fixed"), schema=SEED_SCHEMA)
        assert preview["prompt"]["1"]["inputs"] == {"seed": 5, "steps": 20}
        assert preview["prompt"]["2"]["inputs"] == {"noise_seed": 9}
        assert preview["effective"]["seeds"] == {"1.seed": 5, "2.noise_seed": 9}
        # A request that names no policy keeps the fixed default.
        request = seed_request("fixed")
        del request["seed_policy"]
        assert Jobs(store, None).build_preview(request, schema=SEED_SCHEMA)["effective"]["seeds"] == {
            "1.seed": 5, "2.noise_seed": 9}
    finally:
        store.close()


def test_a_review_raises_no_warning_and_its_review_fields_stay_out_of_the_record(tmp_path):
    """Generate submits a review without a warning at once. The fields only a review holds, its
    identity, times, errors and warnings, never reach the job's frozen record or its digest."""
    store = Store(tmp_path / "state")
    try:
        jobs = Jobs(store, None)
        jobs.schema = SEED_SCHEMA
        first = jobs.build_preview(seed_request("fixed"), schema=SEED_SCHEMA)
        second = jobs.build_preview(seed_request("fixed"), schema=SEED_SCHEMA)
        assert first["warnings"] == [] and first["errors"] == []
        assert first["expires_at"] - first["created_at"] == pytest.approx(jobs_module.PREVIEW_SECONDS, abs=1)
        # Two reviews of one Fixed draft are separate records that freeze the same submission.
        assert first["id"] != second["id"] and first["digest"] == second["digest"]
        snapshot = jobs.reviewed_snapshot(first["id"])
        assert not set(snapshot) & jobs_module.SNAPSHOT_OMITTED
        assert snapshot["digest"] == digest({key: value for key, value in snapshot.items() if key != "digest"})
    finally:
        store.close()


def test_random_seed_policy_freezes_the_chosen_seeds_and_leaves_the_draft(tmp_path, monkeypatch):
    chosen = iter([111, 222])
    monkeypatch.setattr(jobs_module.secrets, "randbelow", lambda count: next(chosen))
    request = seed_request("random")
    original = deepcopy(request)
    store = Store(tmp_path / "state")
    try:
        jobs = Jobs(store, None)
        jobs.schema = SEED_SCHEMA
        preview = jobs.build_preview(request, schema=SEED_SCHEMA)
        seeds = {"1.seed": 111, "2.noise_seed": 222}
        assert preview["effective"]["seeds"] == seeds
        assert preview["prompt"]["1"]["inputs"] == {"seed": 111, "steps": 20}
        assert preview["prompt"]["2"]["inputs"] == {"noise_seed": 222}
        # The chosen seeds live in the record only: the draft and its workflow keep their values.
        assert request == original
        assert preview["requested"]["values"] == original["values"]
        assert preview["requested"]["workflow"] == original["workflow"]
        assert store.get("previews", preview["id"])["effective"]["seeds"] == seeds
        # Submission freezes the same seeds, and the job summary carries them.
        snapshot = jobs.reviewed_snapshot(preview["id"])
        assert snapshot["prompt"] == preview["prompt"] and snapshot["effective"]["seeds"] == seeds
        store.create("jobs", {"id": "seeded", "state": "queued", "created_at": 1, "submission": snapshot})
        assert page_jobs(store, {})["items"][0]["submission"]["effective"] == {"seeds": seeds}
    finally:
        store.close()


def test_a_seed_above_the_browser_range_stays_exact_from_review_to_driver(tmp_path):
    """Python integers are exact: only the browser's view of a large seed needs its digits as text."""
    request = seed_request("fixed")
    request["values"].update({"root:1:seed": 2**64 - 1, "root:2:noise_seed": 2**53})
    sent = []

    class Driver:
        async def request(self, method, path, data=None):
            sent.append(json.dumps(data))
            return {"prompt_id": "accepted"}

    store = Store(tmp_path / "state")
    try:
        jobs = Jobs(store, Driver())
        jobs.schema = SEED_SCHEMA
        preview = jobs.build_preview(request, schema=SEED_SCHEMA)
        exact = {"1.seed": 2**64 - 1, "2.noise_seed": 2**53}
        assert preview["effective"]["seeds"] == exact
        assert preview["prompt"]["1"]["inputs"]["seed"] == 2**64 - 1
        snapshot = jobs.reviewed_snapshot(preview["id"])
        value = store.create("jobs", {"id": "large", "state": "submitting", "created_at": 1, "prompt_id": None,
                                      "client_prompt_id": "client", "submission": snapshot})
        asyncio.run(jobs.dispatch(value))
        # The driver receives every digit, and the stored record keeps exact integers.
        assert '"seed": 18446744073709551615' in sent[0] and '"noise_seed": 9007199254740992' in sent[0]
        assert store.get("jobs", "large")["submission"]["effective"]["seeds"] == exact
        assert page_jobs(store, {})["items"][0]["submission"]["effective"]["seeds"] == {
            "1.seed": "18446744073709551615", "2.noise_seed": "9007199254740992"}
    finally:
        store.close()
