"""Bind draft transformations to explicit typed inputs in an existing graph."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import StudioError, object_value
from .drafts import apply_styles, validate_value
from .workflows import compile_workflow, inspect_workflow

LORA_NODES = {"LoraLoader", "LoraLoaderModelOnly", "DGXMonarchLoraLoader"}


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
    if len(set(mapping.values())) != len(mapping):
        raise StudioError("style_binding", "Positive and negative styles must use distinct input bindings.")
    text = {"prompt": "", "negative_prompt": ""}
    for role, identity in mapping.items():
        binding = bindings.get(identity)
        if role not in text or not binding or not binding["resolved"] or binding["schema"]["type"] != "STRING":
            raise StudioError("style_binding", "Styles require resolved STRING workflow input bindings.")
        text[role] = validate_value(values.get(identity, binding["value"]), binding["schema"])
    expanded = apply_styles(text["prompt"], text["negative_prompt"], styles)
    for role, identity in mapping.items():
        values[identity] = expanded[role]
    return {"styles": expanded, "style_bindings": deepcopy(mapping)}


def _ordered_lora_slots(prompt: dict[str, Any]) -> list[str]:
    slots = {nid: node for nid, node in prompt.items() if node["class_type"] in LORA_NODES}
    if not slots:
        return []
    predecessors = {}
    for nid, node in slots.items():
        model = node["inputs"].get("model")
        predecessor = model[0] if isinstance(model, list) and model[0] in slots else None
        predecessors[nid] = predecessor
        if predecessor is not None:
            if model[1] != 0:
                raise StudioError("lora_binding", "LoRA model chain uses an unsupported output socket.")
            clip = node["inputs"].get("clip")
            if clip is not None and clip != [predecessor, 1]:
                raise StudioError("lora_binding", "Model and text-encoder LoRA order must follow the same chain.")
    roots = [nid for nid, parent in predecessors.items() if parent is None]
    if len(roots) != 1:
        raise StudioError("lora_binding", "Choose a workflow with one explicit LoRA chain; multiple chains are ambiguous.")
    ordered = roots
    while len(ordered) < len(slots):
        children = [nid for nid, parent in predecessors.items() if parent == ordered[-1]]
        if len(children) != 1:
            raise StudioError("lora_binding", "Branched LoRA chains cannot be mapped to one ordered stack.")
        ordered.append(children[0])
    return ordered


def _loras(request: dict[str, Any], prompt: dict[str, Any], bindings: dict[str, Any],
           values: dict[str, Any]) -> dict[str, Any]:
    stack = request.get("lora_stack", [])
    if not isinstance(stack, list) or any(not isinstance(entry, dict) for entry in stack):
        raise StudioError("lora_stack", "LoRA stack must be an ordered list of records.")
    existing = [{"node_id": nid, "class_type": node["class_type"],
                 "asset_id": node["inputs"].get("lora_name"),
                 "strength_model": node["inputs"].get("strength_model"),
                 "strength_clip": node["inputs"].get("strength_clip"),
                 "compatibility": "pending_backend_validation"}
                for nid, node in prompt.items() if node["class_type"] in LORA_NODES]
    if not stack:
        return {"lora_stack": [], "resolved_loras": existing}
    active = [entry for entry in stack if entry.get("enabled", True)]
    slots = _ordered_lora_slots(prompt)
    if len(active) > len(slots) or not slots:
        raise StudioError("lora_binding", "Selected stack needs more existing LoRA loader slots in the workflow.")
    by_input = {(binding["node_id"], binding["input_name"]): binding for binding in bindings.values()
                if binding["graph_id"] in {"root", next(iter(bindings.values()))["graph_id"]}}
    resolved = []
    for index, nid in enumerate(slots):
        fields = {name: by_input.get((nid, name), {}) for name in ("lora_name", "strength_model", "strength_clip")}
        if not all(fields[name] and fields[name]["resolved"] for name in ("lora_name", "strength_model")):
            raise StudioError("lora_binding", "LoRA asset and strengths must be editable typed inputs.")
        entry = active[index] if index < len(active) else None
        updates = {"strength_model": entry.get("model_strength", fields["strength_model"]["schema"].get("default"))
                   if entry else 0.0}
        if entry:
            if set(entry) - {"id", "name", "enabled", "model_strength", "clip_strength", "source_node"}:
                raise StudioError("lora_binding", "The selected LoRA has unsupported controls.")
            if entry.get("source_node", prompt[nid]["class_type"]) != prompt[nid]["class_type"]:
                raise StudioError("lora_binding", "Selected LoRA controls belong to a different loader type than the workflow slot.")
            updates["lora_name"] = entry.get("id")
        if "strength_clip" in prompt[nid]["inputs"]:
            if not fields["strength_clip"] or not fields["strength_clip"]["resolved"]:
                raise StudioError("lora_binding", "LoRA text-encoder strength is not an editable typed input.")
            updates["strength_clip"] = entry.get("clip_strength", fields["strength_clip"]["schema"].get("default")) if entry else 0.0
        elif entry and "clip_strength" in entry:
            raise StudioError("lora_binding", "This loader supports model strength only; remove text-encoder strength.")
        for name, value in updates.items():
            binding = fields[name]
            validate_value(value, binding["schema"])
            values[binding["id"]] = value
        resolved.append({"node_id": nid, "class_type": prompt[nid]["class_type"],
                         "asset_id": updates.get("lora_name", prompt[nid]["inputs"]["lora_name"]),
                         **{key: value for key, value in updates.items() if key != "lora_name"},
                         "enabled": entry is not None, "compatibility": "pending_backend_validation"})
    return {"lora_stack": deepcopy(stack), "resolved_loras": resolved}


def prepare_preview(request: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    workflow = object_value(request.get("workflow"), "workflow")
    values = deepcopy(object_value(request.get("values", {}), "values"))
    inspection = inspect_workflow(workflow, schema)
    bindings = {binding["id"]: binding for binding in inspection["bindings"]}
    # Compile effective edits before checking transformations: invalid imported
    # defaults can be repaired by this draft without changing its source graph.
    prompt = compile_workflow(workflow, schema, values)
    transformations = _styles(request, bindings, values)
    transformations.update(_loras(request, prompt, bindings, values))
    if request.get("styles") or request.get("lora_stack"):
        prompt = compile_workflow(workflow, schema, values)
    return prompt, values, transformations
