"""Typed LoRA graph edits with explicit insertion boundaries and frozen edges."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..constants import MODEL_TYPE
from .contracts import StudioError, object_value
from .drafts import DraftError, validate_value
from .workflows import node_inputs

LORA_NODES = {"LoraLoader", "LoraLoaderModelOnly", "DGXMonarchLoraLoader"}


def ordered_slots(prompt: dict[str, Any]) -> list[str]:
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
        raise StudioError("lora_binding", "Choose one explicit LoRA chain; multiple chains are ambiguous.")
    ordered = roots
    while len(ordered) < len(slots):
        children = [nid for nid, parent in predecessors.items() if parent == ordered[-1]]
        if len(children) != 1:
            raise StudioError("lora_binding", "Branched LoRA chains cannot be mapped to one ordered stack.")
        ordered.append(children[0])
    return ordered


def _fields(info: dict[str, Any], kind: str) -> dict[str, dict[str, Any]]:
    if kind not in LORA_NODES or kind not in info:
        raise StudioError("lora_binding", "Choose an installed supported typed LoRA loader.")
    definition = info[kind]
    fields = {name: schema for name, schema, _ in node_inputs(definition)}
    model_type = MODEL_TYPE if kind == "DGXMonarchLoraLoader" else "MODEL"
    if (fields.get("model", {}).get("type") != model_type or definition.get("output", [])[:1] != [model_type]
            or fields.get("lora_name", {}).get("type") != "COMBO" or fields.get("strength_model", {}).get("type") not in {"FLOAT", "INT"}):
        raise StudioError("lora_binding", "Installed loader does not match the supported typed LoRA contract.")
    if "clip" in fields and (fields["clip"]["type"] != "CLIP" or definition.get("output", [])[1:2] != ["CLIP"]
                              or fields.get("strength_clip", {}).get("type") not in {"FLOAT", "INT"}):
        raise StudioError("lora_binding", "Installed loader has an unsupported text-encoder contract.")
    return fields


def _targets(prompt: dict[str, Any], info: dict[str, Any], rows: Any, kind: str) -> tuple[list[str | int] | None, list[dict[str, str]]]:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise StudioError("lora_binding", "Insertion targets must be arrays of typed input identities.")
    source = None
    targets = []
    for row in rows:
        if set(row) != {"node_id", "input_name"} or not all(isinstance(value, str) for value in row.values()):
            raise StudioError("lora_binding", "Insertion target needs node_id and input_name.")
        node = prompt.get(row["node_id"])
        if node is None or node["class_type"] in LORA_NODES:
            raise StudioError("lora_binding", "Insertion target must be an existing non-LoRA consumer.")
        schema = {name: schema for name, schema, _ in node_inputs(info[node["class_type"]])}.get(row["input_name"], {})
        link = node["inputs"].get(row["input_name"])
        if schema.get("type") != kind or not isinstance(link, list) or len(link) != 2:
            raise StudioError("lora_binding", f"Insertion target must be a connected {kind} input.")
        if source is not None and link != source:
            raise StudioError("lora_binding", f"All selected {kind} inputs must share the same source.")
        if row in targets:
            raise StudioError("lora_binding", "Duplicate insertion target.")
        source = deepcopy(link)
        targets.append(deepcopy(row))
    return source, targets


def _consumers(prompt: dict[str, Any], source: list[Any]) -> list[dict[str, str]]:
    return [{"node_id": nid, "input_name": name} for nid, node in prompt.items()
            for name, value in node["inputs"].items() if value == source]


def apply_loras(original: dict[str, Any], info: dict[str, Any], stack: Any,
                insertion: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
    from .workflow_compile import _validate_prompt

    if (not isinstance(stack, list) or any(not isinstance(entry, dict) for entry in stack) or len(stack) > 64
            or any(type(entry.get("enabled", True)) is not bool for entry in stack)):
        raise StudioError("lora_stack", "LoRA stack must contain at most 64 ordered records.")
    prompt = deepcopy(original)
    if not stack:
        return prompt, {"lora_stack": [], "resolved_loras": [
            {"node_id": nid, "class_type": node["class_type"], "asset_id": node["inputs"].get("lora_name"),
             "strength_model": node["inputs"].get("strength_model"), "strength_clip": node["inputs"].get("strength_clip"),
             "compatibility": "pending_backend_validation"} for nid, node in prompt.items() if node["class_type"] in LORA_NODES]}
    active = [entry for entry in stack if entry.get("enabled", True)]
    slots = ordered_slots(prompt)
    existing_count = len(slots)
    boundary: dict[str, Any] = {}
    model: list[Any] | None
    clip: list[Any] | None
    if len(active) > len(slots):
        if slots:
            if insertion is not None:
                raise StudioError("lora_binding", "Existing LoRA chains extend at their tail; remove the separate insertion boundary.")
            tail = slots[-1]
            model, model_targets = [tail, 0], _consumers(prompt, [tail, 0])
            has_clip = "clip" in _fields(info, prompt[tail]["class_type"])
            clip, clip_targets = ([tail, 1], _consumers(prompt, [tail, 1])) if has_clip else (None, [])
            loader = prompt[tail]["class_type"]
        else:
            config = object_value(insertion, "lora_insertion")
            if set(config) - {"loader", "model_targets", "clip_targets"}:
                raise StudioError("lora_binding", "Unsupported insertion boundary controls.")
            loader = config.get("loader")
            if not isinstance(loader, str):
                raise StudioError("lora_binding", "Choose an explicit LoRA loader type.")
            fields = _fields(info, loader)
            model, model_targets = _targets(prompt, info, config.get("model_targets", []), fields["model"]["type"])
            clip, clip_targets = _targets(prompt, info, config.get("clip_targets", []), "CLIP")
        if model is None or not model_targets:
            raise StudioError("lora_binding", "Choose an actual model consumer for the inserted LoRA chain.")
        boundary = {"model_source": deepcopy(model), "clip_source": deepcopy(clip),
                    "model_targets": model_targets, "clip_targets": clip_targets, "inserted_nodes": []}
        for entry in active[len(slots):]:
            kind = entry.get("source_node", loader)
            if not isinstance(kind, str):
                raise StudioError("lora_binding", "LoRA loader type must be text.")
            fields = _fields(info, kind)
            wants_clip = "clip" in fields
            if wants_clip != (clip is not None):
                raise StudioError("lora_binding", "Inserted loaders must preserve the selected model/text-encoder chain.")
            index = 0
            while (nid := f"__studio_lora__{index}") in prompt:
                index += 1
            inputs = {name: deepcopy(schema["default"]) for name, schema in fields.items() if "default" in schema}
            inputs["model"] = model
            if wants_clip:
                inputs["clip"] = clip
            prompt[nid] = {"class_type": kind, "inputs": inputs}
            slots.append(nid)
            boundary["inserted_nodes"].append(nid)
            model = [nid, 0]
            if wants_clip:
                clip = [nid, 1]
        for row in model_targets:
            prompt[row["node_id"]]["inputs"][row["input_name"]] = model
        for row in clip_targets:
            prompt[row["node_id"]]["inputs"][row["input_name"]] = clip
    resolved = []
    for index, nid in enumerate(slots):
        node = prompt[nid]
        fields = _fields(info, node["class_type"])
        entry = active[index] if index < len(active) else None
        if entry is not None:
            if set(entry) - {"id", "name", "enabled", "model_strength", "clip_strength", "source_node"}:
                raise StudioError("lora_binding", "The selected LoRA has unsupported controls.")
            if entry.get("source_node", node["class_type"]) != node["class_type"]:
                raise StudioError("lora_binding", "Selected LoRA controls belong to a different loader type than the workflow slot.")
            if "clip_strength" in entry and "strength_clip" not in fields:
                raise StudioError("lora_binding", "This loader supports model strength only; remove text-encoder strength.")
        updates = {"strength_model": entry.get("model_strength", fields["strength_model"].get("default")) if entry else 0.0}
        if entry:
            updates["lora_name"] = entry.get("id")
        if "strength_clip" in fields:
            updates["strength_clip"] = entry.get("clip_strength", fields["strength_clip"].get("default")) if entry else 0.0
        for name, value in updates.items():
            if index < existing_count and isinstance(node["inputs"].get(name), list):
                raise StudioError("lora_binding", "LoRA asset and strengths must be editable typed inputs.")
            try:
                node["inputs"][name] = validate_value(value, fields[name])
            except DraftError as exc:
                raise StudioError("lora_binding", f"LoRA slot {index + 1} ({nid}.{name}): {exc}") from exc
        resolved.append({"node_id": nid, "class_type": node["class_type"], "asset_id": node["inputs"].get("lora_name"),
                         **{key: value for key, value in updates.items() if key != "lora_name"}, "enabled": entry is not None,
                         "inserted": index >= existing_count, "compatibility": "pending_backend_validation"})
    return _validate_prompt(prompt, info), {"lora_stack": deepcopy(stack), "resolved_loras": resolved,
                                          "lora_insertion": boundary or None}
