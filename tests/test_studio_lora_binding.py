"""An invalid LoRA stack value is a clean lora_binding refusal, never a server error."""
from __future__ import annotations

import asyncio
import copy

import pytest

from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.lora_graph import apply_loras
from test_studio_integration import companion

STRENGTH = ["FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}]
SCHEMA = {
    "Model": {"input": {"required": {}}, "output": ["MODEL"]},
    "LoraLoaderModelOnly": {"input": {"required": {
        "model": ["MODEL"], "lora_name": [["a.safetensors", "b.safetensors"]], "strength_model": STRENGTH}},
        "output": ["MODEL"]},
    "Output": {"input": {"required": {"model": ["MODEL"]}}, "output": [], "output_node": True},
}
PROMPT = {
    "1": {"class_type": "Model", "inputs": {}},
    "4": {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["1", 0], "lora_name": "a.safetensors", "strength_model": 0.4}},
    "6": {"class_type": "Output", "inputs": {"model": ["4", 0]}},
}
CASES = {
    "strength_over_max": ([{"id": "a.safetensors", "model_strength": 1000}],
                          "LoRA slot 1 (4.strength_model): Value violates max=100.0"),
    "removed_lora": ([{"id": "removed.safetensors", "model_strength": 1}],
                     "LoRA slot 1 (4.lora_name): Value is not an available choice"),
    "text_strength": ([{"id": "a.safetensors", "model_strength": "1"}],
                      "LoRA slot 1 (4.strength_model): Expected a number"),
    "boolean_strength": ([{"id": "a.safetensors", "model_strength": True}],
                         "LoRA slot 1 (4.strength_model): Expected a number"),
    "inserted_slot_over_max": ([{"id": "a.safetensors", "model_strength": 1},
                                {"id": "b.safetensors", "model_strength": -500}],
                               "LoRA slot 2 (__studio_lora__0.strength_model): Value violates min=-100.0"),
}


@pytest.mark.parametrize("case", CASES, ids=list(CASES))
def test_invalid_stack_values_raise_lora_binding(case):
    stack, message = CASES[case]
    with pytest.raises(StudioError) as error:
        apply_loras(copy.deepcopy(PROMPT), SCHEMA, stack)
    assert (error.value.code, error.value.status, str(error.value)) == ("lora_binding", 400, message)


def test_valid_stack_still_applies():
    prompt, effective = apply_loras(copy.deepcopy(PROMPT), SCHEMA, [{"id": "b.safetensors", "model_strength": 1.5}])
    assert prompt["4"]["inputs"] == {"model": ["1", 0], "lora_name": "b.safetensors", "strength_model": 1.5}
    assert effective["resolved_loras"][0]["compatibility"] == "pending_backend_validation"


def test_http_preview_returns_400_lora_binding_for_each_case(tmp_path, monkeypatch):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            system.protocol.schema = copy.deepcopy(SCHEMA)
            for case, (stack, message) in CASES.items():
                body = {"workflow": copy.deepcopy(PROMPT), "values": {}, "lora_stack": stack}
                response = await system.browser.post("/api/v1/jobs/preview", headers=system.headers, json=body)
                assert (response.status, response.content_type) == (400, "application/json"), case
                assert (await response.json())["error"] == {"code": "lora_binding", "message": message}, case
            assert system.store.all("previews") == []
    asyncio.run(scenario())
