"""Bind draft transformations to explicit typed inputs in an existing graph."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import StudioError, object_value
from .drafts import apply_styles, validate_value
from .lora_graph import apply_loras
from .workflows import compile_workflow, inspect_workflow


def _styles(request: dict[str, Any], bindings: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    styles = request.get("styles", [])
    if not isinstance(styles, list) or any(not isinstance(style, dict) for style in styles):
        raise StudioError("styles", "Styles must be an ordered list of records.")
    if not styles:
        return {}
    mapping = object_value(request.get("style_bindings", {}), "style_bindings")
    if any(not isinstance(identity, str) for identity in mapping.values()):
        raise StudioError("style_binding", "Style binding identities must be strings.")
    enabled = [style for style in styles if style.get("enabled", True)]
    if any(not isinstance(style.get("id"), str) or not style["id"] for style in enabled):
        raise StudioError("styles", "Every enabled style requires its stable record ID.")
    if any(style.get("mode", "append") not in {"append", "template"} for style in enabled):
        raise StudioError("styles", "Unsupported prompt-style transformation mode.")
    if not enabled:
        return {"styles": {"style_order": [], "unexpanded": {}}}
    if not mapping.get("prompt"):
        raise StudioError("style_binding", "Choose an explicit prompt input for the selected styles.")
    if any(style.get("negative_prompt", "") for style in enabled) and not mapping.get("negative_prompt"):
        raise StudioError("style_binding", "Choose a negative-prompt input for styles that change negative text.")
    text = {"prompt": "", "negative_prompt": ""}
    canonical_mapping = {}
    for role, identity in mapping.items():
        binding = bindings.get(identity)
        if role not in text or not binding or not binding["resolved"] or binding["schema"]["type"] != "STRING":
            raise StudioError("style_binding", "Styles require resolved STRING workflow input bindings.")
        canonical_mapping[role] = binding["id"]
        text[role] = validate_value(values.get(identity, binding["value"]), binding["schema"])
    if len(set(canonical_mapping.values())) != len(canonical_mapping):
        raise StudioError("style_binding", "Positive and negative styles must use distinct input bindings.")
    expanded = apply_styles(text["prompt"], text["negative_prompt"], styles)
    for role, identity in canonical_mapping.items():
        values[identity] = expanded[role]
    return {"styles": expanded, "style_bindings": canonical_mapping}


def prepare_preview(request: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    workflow = object_value(request.get("workflow"), "workflow")
    values = deepcopy(object_value(request.get("values", {}), "values"))
    inspection = inspect_workflow(workflow, schema)
    bindings = {binding["id"]: binding for binding in inspection["bindings"]}
    for binding in inspection["bindings"]:
        for alias in binding.get("aliases", []):
            bindings[alias] = binding
    # Compile effective edits before checking transformations: invalid imported
    # defaults can be repaired by this draft without changing its source graph.
    prompt = compile_workflow(workflow, schema, values)
    # Styles read the compiled scalar, including edits supplied through an alias.
    style_bindings = deepcopy(bindings)
    for binding in style_bindings.values():
        node = prompt.get(binding.get("execution_node_id", binding["node_id"]), {})
        if binding["input_name"] in node.get("inputs", {}):
            binding["value"] = node["inputs"][binding["input_name"]]
    transformations = _styles(request, style_bindings, values)
    if request.get("styles"):
        # A style writes one canonical identity; remove its equivalent legacy edit.
        for identity in transformations.get("style_bindings", {}).values():
            binding = bindings[identity]
            for other in (binding["id"], *binding.get("aliases", [])):
                if other != identity:
                    values.pop(other, None)
        prompt = compile_workflow(workflow, schema, values)
    prompt, loras = apply_loras(prompt, schema, request.get("lora_stack", []), request.get("lora_insertion"))
    transformations.update(loras)
    if request.get("lora_stack"):
        for binding in inspection["bindings"]:
            nid, name = binding.get("execution_node_id", binding["node_id"]), binding["input_name"]
            if binding["resolved"] and nid in prompt and name in {"lora_name", "strength_model", "strength_clip"}:
                for alias in binding.get("aliases", []):
                    values.pop(alias, None)
                values[binding["id"]] = deepcopy(prompt[nid]["inputs"][name])
    return prompt, values, transformations
