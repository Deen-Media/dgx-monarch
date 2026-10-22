"""Versioned, lossless ComfyUI workflow inspection and App Mode bindings.

The adapter uses serialized graph identities and /object_info. It never executes
frontend extensions. Unsupported serialization remains visible and blocks compile.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote, unquote

from ..constants import MODEL_TYPE

ADAPTER_VERSION = 1
SCALAR_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


class WorkflowError(ValueError):
    """Workflow input cannot be resolved without guessing."""


def binding_id(graph_id: str, node_id: str, name: str) -> str:
    return f"{graph_id}:{quote(node_id, safe='')}:{quote(name, safe='')}"


def api_prompt_nodes(workflow: Mapping[str, Any]) -> dict[str, Any]:
    """Separate Studio display metadata from executable API-prompt nodes."""
    return {key: node for key, node in workflow.items()
            if not (key == "extra" and isinstance(node, dict) and "class_type" not in node)}


def is_api_prompt(workflow: Mapping[str, Any]) -> bool:
    nodes = api_prompt_nodes(workflow)
    return bool(nodes) and all(isinstance(node, dict) and "class_type" in node
                              for node in nodes.values())


def graph_index(workflow: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index root and subgraph definitions; reject duplicate graph identities."""
    graphs: dict[str, dict[str, Any]] = {}

    def visit(graph: Mapping[str, Any], fallback: str, depth: int) -> None:
        if depth > 32:
            raise WorkflowError("Subgraph nesting exceeds 32 levels")
        gid = str(graph.get("id", fallback))
        if gid in graphs:
            raise WorkflowError(f"Duplicate graph identity: {gid}")
        nodes = graph.get("nodes", [])
        if not isinstance(nodes, list):
            raise WorkflowError("Graph nodes must be an array")
        by_id: dict[str, Any] = {}
        for node in nodes:
            if not isinstance(node, dict) or "id" not in node:
                raise WorkflowError("Each graph node needs an identity")
            if not isinstance(node.get("type"), str):
                raise WorkflowError("Graph node type must be a string")
            nid = str(node["id"])
            if nid in by_id:
                raise WorkflowError(f"Duplicate node identity: {gid}:{nid}")
            by_id[nid] = node
        graphs[gid] = {"graph": graph, "nodes": by_id}
        definitions_record = graph.get("definitions", {})
        if not isinstance(definitions_record, dict):
            raise WorkflowError("Graph definitions must be an object")
        definitions = definitions_record.get("subgraphs", [])
        if not isinstance(definitions, list) or any(not isinstance(item, dict) for item in definitions):
            raise WorkflowError("Subgraph definitions must be objects in an array")
        for index, subgraph in enumerate(definitions):
            visit(subgraph, f"{gid}/definition/{index}", depth + 1)
        for nid, node in by_id.items():
            if isinstance(node.get("subgraph"), dict):
                visit(node["subgraph"], f"{gid}/node/{nid}", depth + 1)

    if is_api_prompt(workflow):
        prompt_nodes = api_prompt_nodes(workflow)
        if any(not isinstance(node["class_type"], str) for node in prompt_nodes.values()):
            raise WorkflowError("API node class_type must be a string")
        nodes = {str(nid): {**node, "id": str(nid), "type": node["class_type"]}
                 for nid, node in prompt_nodes.items()}
        graphs["root"] = {"graph": {}, "nodes": nodes}
    else:
        visit(workflow, "root", 0)
    if sum(len(item["nodes"]) for item in graphs.values()) > 10000:
        raise WorkflowError("Workflow exceeds 10000 nodes")
    return graphs


def node_inputs(info: Mapping[str, Any]) -> list[tuple[str, dict[str, Any], bool]]:
    """Return declared input order, retaining connection-only socket types."""
    from .widget_values import declared_inputs
    return declared_inputs(info)


def widget_values(node: Mapping[str, Any], info: Mapping[str, Any], *, api: bool = False) -> dict[str, Any]:
    """Map a node's serialized widget values to input names; see widget_values.py."""
    from .widget_values import layout_values
    return layout_values(node, info, api=api)


def node_fields(node: Mapping[str, Any], info: Mapping[str, Any], values: Mapping[str, Any], *,
                api: bool, edit: Callable[[str], Any] | None = None) -> list[tuple[str, dict[str, Any], bool]]:
    """Typed inputs after ComfyUI's dynamic input expansion for these values.

    ``edit(name)`` returns a draft edit or ``MISSING``; an edited dynamic choice
    declares its own child inputs, as compile does.
    """
    from .widget_values import LINKED, MISSING, field_schema, resolve_inputs
    live = dict(values)
    if not api:
        # A linked DynamicCombo selects no option here, as in ComfyUI.
        live.update({str(socket.get("name")): LINKED for socket in node.get("inputs", [])
                     if isinstance(socket, dict) and socket.get("link") is not None})
    return [(name, field_schema(name, spec), required)
            for name, spec, required in resolve_inputs(info, live, edit or (lambda _name: MISSING),
                                                       fill_defaults=not api)]


def _edit_lookup(edits: Mapping[str, Any], gid: str, nid: str) -> Callable[[str], Any]:
    """Read one node's draft edits by input name, as compile reads them."""
    from .widget_values import MISSING

    def edit(name: str) -> Any:
        key = binding_id(gid, nid, name)
        return copy.deepcopy(edits[key]) if key in edits else MISSING
    return edit


def _resolve(identity: Any, name: str | None, graphs: Mapping[str, Any]) -> tuple[str, str, str | None]:
    root = next(iter(graphs))
    text = str(identity)
    segments = text.split(":")
    canonical = text.rsplit(":", 2)
    if len(canonical) == 3 and canonical[0] in graphs:
        gid, nid, canonical_name = canonical[0], unquote(canonical[1]), unquote(canonical[2])
        if nid not in graphs[gid]["nodes"]:
            raise WorkflowError("Referenced node was deleted or is missing")
        return gid, nid, canonical_name
    if len(segments) == 2 and segments[0] in graphs:
        gid, nid = segments[0], unquote(segments[1])
        if nid in graphs[gid]["nodes"]:
            return gid, nid, name
    if len(segments) > 1:
        # Legacy node identities encode the path through subgraph instances.
        gid = root
        for position, segment in enumerate(segments):
            nid = unquote(segment)
            node = graphs[gid]["nodes"].get(nid)
            if node is None:
                raise WorkflowError("Legacy graph path references a missing node")
            if position == len(segments) - 1:
                return gid, nid, name
            child = node.get("subgraph", {})
            child_id = str(child.get("id", f"{gid}/node/{nid}")) if child else str(node.get("type"))
            if child_id not in graphs:
                raise WorkflowError("Legacy graph path has no resolvable subgraph")
            gid = child_id
    if text in graphs[root]["nodes"]:
        return root, text, name
    matches = [(gid, text, name) for gid, graph in graphs.items() if text in graph["nodes"]]
    if len(matches) == 1:
        return matches[0]
    raise WorkflowError("Ambiguous legacy node identity" if matches else "Referenced node was deleted or is missing")


def _primitive_definition(nid: str, graph: Mapping[str, Any], info: Mapping[str, Any]) -> dict[str, Any]:
    """Infer a core PrimitiveNode widget only from its typed destination sockets."""
    from .workflow_compile import _link_rows
    links = _link_rows(graph["graph"])
    pending, seen, schemas = [nid], set(), []
    while pending:
        source = pending.pop()
        if source in seen:
            return {}
        seen.add(source)
        for origin, _, target, target_slot in links.values():
            if origin != source:
                continue
            node = graph["nodes"].get(target, {})
            if node.get("type") == "Reroute":
                pending.append(target)
                continue
            sockets = node.get("inputs", [])
            if not 0 <= target_slot < len(sockets):
                return {}
            name = sockets[target_slot].get("name")
            definition = info.get(node.get("type"), {})
            matches = [schema for field, schema, _ in node_inputs(definition) if field == name]
            if len(matches) != 1 or matches[0]["type"] not in SCALAR_TYPES:
                return {}
            schemas.append(matches[0])
    if not schemas or any(schema["type"] != schemas[0]["type"] for schema in schemas):
        return {}
    combined = copy.deepcopy(schemas[0])
    for schema in schemas[1:]:
        for key, combine in (("min", max), ("max", min)):
            if key in schema:
                combined[key] = combine(combined.get(key, schema[key]), schema[key])
        if combined["type"] == "COMBO":
            combined["choices"] = [value for value in combined["choices"] if value in schema["choices"]]
    kind = combined.pop("type")
    combined.pop("forceInput", None)
    combined.pop("dynamic", None)
    combined["control_after_generate"] = True
    return {"input": {"required": {"value": [combined.pop("choices") if kind == "COMBO" else kind, combined]}}}


def inspect_workflow(workflow: Mapping[str, Any], object_info: Mapping[str, Any] | None = None,
                     edits: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Inspect all typed inputs, preserving unresolved App Mode selections.

    ``edits`` are draft values by binding ID. A dynamic choice they change lists
    the child inputs of the edited choice; ``inactive`` names the saved choice's
    children that the edits leave out. Subgraph instances keep their saved
    choices' children.
    """
    if not isinstance(workflow, dict):
        raise WorkflowError("Workflow must be an object")
    info = object_info or {}
    graphs = graph_index(workflow)
    api = is_api_prompt(workflow)
    edited = dict(edits or {}) if api or len(graphs) == 1 else {}
    bindings: list[dict[str, Any]] = []
    inactive: list[str] = []
    errors: list[str] = []
    for gid, graph in graphs.items():
        for nid, node in graph["nodes"].items():
            definition = info.get(node.get("type"), {})
            if node.get("type") == "PrimitiveNode" and not api:
                try:
                    definition = _primitive_definition(nid, graph, info)
                except WorkflowError as exc:
                    errors.append(f"{gid}:{nid}: {exc}")
            if not definition:
                continue
            try:
                values = widget_values(node, definition, api=api)
                fields = node_fields(node, definition, values, api=api, edit=_edit_lookup(edited, gid, nid))
                if edited:
                    names = {name for name, _, _ in fields}
                    inactive += [binding_id(gid, nid, name) for name, schema, _ in node_fields(
                        node, definition, values, api=api) if name not in names and schema["type"] in SCALAR_TYPES]
            except WorkflowError as exc:
                errors.append(f"{gid}:{nid}: {exc}")
                continue
            linked = {field.get("name") for field in node.get("inputs", [])
                      if isinstance(field, dict) and field.get("link") is not None} if not api else set()
            for name, schema, required in fields:
                if schema["type"] not in SCALAR_TYPES:
                    continue
                value = values.get(name, schema.get("default"))
                is_link = name in linked or (api and isinstance(value, list))
                bindings.append({"id": binding_id(gid, nid, name), "source_identity": None,
                                 "graph_id": gid, "node_id": nid, "input_name": name, "label": name,
                                 "order": len(bindings), "prominent": False, "config": {}, "schema": schema,
                                 "value": value, "required": required, "resolved": not is_link,
                                 "reason": "Input is connected; edit its source instead" if is_link else None})
    if not api and len(graphs) > 1:
        try:
            from .instance_bindings import project_bindings
            bindings = project_bindings(workflow, dict(info), bindings)
        except WorkflowError as exc:
            errors.append(str(exc))
    else:
        for binding in bindings:
            binding.update(instance_path=[], execution_node_id=binding["node_id"], aliases=[])
    lookup = {binding["id"]: binding for binding in bindings}
    extra = workflow.get("extra", {})
    if not isinstance(extra, dict):
        raise WorkflowError("Workflow extra metadata must be an object")
    linear = extra.get("linearData", {}) if isinstance(extra, dict) else {}
    if not isinstance(linear, dict) or not isinstance(linear.get("inputs", []), list) or not isinstance(linear.get("outputs", []), list):
        raise WorkflowError("App Mode data must contain input and output arrays")
    studio = extra.get("dgxm_studio", {})
    if not isinstance(studio, dict) or not isinstance(studio.get("bindings", {}), dict):
        raise WorkflowError("Studio display bindings must be an object")
    overrides = studio.get("bindings", {})
    if any(not isinstance(item, dict) for item in overrides.values()):
        raise WorkflowError("Studio display binding entries must be objects")
    for index, entry in enumerate(linear.get("inputs", [])):
        identity, name = (entry[:2] if isinstance(entry, (list, tuple)) and len(entry) >= 2 else (None, None))
        config = copy.deepcopy(entry[2]) if isinstance(entry, (list, tuple)) and len(entry) > 2 else {}
        unresolved_id = "unresolved:" + hashlib.sha256(json.dumps([identity, name], sort_keys=True).encode()).hexdigest()[:16]
        try:
            if identity is None or not isinstance(name, str):
                raise WorkflowError("App Mode input must be [identity, widgetName, optional configuration]")
            if isinstance(identity, str) and identity in lookup:
                key, resolved_name = identity, lookup[identity]["input_name"]
            else:
                gid, nid, resolved_name = _resolve(identity, name, graphs)
                key = binding_id(gid, nid, str(resolved_name))
            key = overrides.get(key, overrides.get(unresolved_id, {})).get("target", key)
            if key in inactive:
                continue  # An edited choice dropped this input; the App Mode entry returns with it.
            selected_binding = lookup.get(key)
            if selected_binding is None and not api and len(graphs) > 1:
                from .instance_bindings import select_binding
                selected_binding = select_binding(identity, str(resolved_name), bindings, key)
            if selected_binding is None:
                raise WorkflowError("Widget missing, custom, or unsupported by object_info")
            if not selected_binding["resolved"]:
                raise WorkflowError(selected_binding["reason"])
            selected_binding.update(source_identity=identity, prominent=True, order=index, config=config)
        except WorkflowError as exc:
            target = overrides.get(unresolved_id, {}).get("target")
            if target in lookup and lookup[target]["resolved"]:
                lookup[target].update(source_identity=identity, prominent=True, order=index, config=config)
                continue
            errors.append(f"App Mode input {index}: {exc}")
            bindings.append({"id": unresolved_id, "source_identity": identity, "graph_id": None,
                             "node_id": None, "input_name": name, "label": name or "Unresolved input",
                             "order": index, "prominent": True, "config": config, "schema": None,
                             "value": None, "resolved": False, "requires_resolution": True, "reason": str(exc)})
    for binding in bindings:
        display = overrides.get(binding["id"], {})
        for field in ("label", "group", "order", "prominent"):
            if field in display:
                binding[field] = copy.deepcopy(display[field])
    outputs = []
    for identity in linear.get("outputs", []):
        try:
            gid, nid, _ = _resolve(identity, None, graphs)
            outputs.append({"source_identity": identity, "graph_id": gid, "node_id": nid,
                            "resolved": True, "reason": None})
        except WorkflowError as exc:
            outputs.append({"source_identity": identity, "resolved": False, "reason": str(exc)})
            errors.append(f"App Mode output {identity}: {exc}")
    prompt = None
    if not errors:
        try:
            from .workflow_compile import compile_graph
            prompt = compile_graph(workflow, info, {})
        except WorkflowError as exc:
            errors.append(str(exc))
    return {"adapter_version": ADAPTER_VERSION, "workflow": copy.deepcopy(workflow),
            "bindings": bindings, "inactive": inactive, "outputs": outputs, "prompt": prompt, "errors": errors,
            "connections": [{"node_id": nid, "input_name": name, "type": schema["type"], "source": value}
                            for nid, node in (prompt or {}).items()
                            for name, schema, _ in node_inputs(info.get(node["class_type"], {}))
                            if schema["type"] in {"MODEL", MODEL_TYPE, "CLIP"} and isinstance((value := node["inputs"].get(name)), list)]}


def compile_workflow(workflow: Mapping[str, Any], object_info: Mapping[str, Any],
                     values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from .workflow_compile import compile_graph
    # Edits that change a dynamic choice make its new children editable. An edit
    # to a child the change dropped stays accepted, and compile ignores it.
    inspection = inspect_workflow(workflow, object_info, values)
    # Compilation validates the resulting values. Existing invalid defaults may be
    # repaired by explicit edits; unresolved selected identities cannot be skipped.
    unresolved = [item for item in inspection["bindings"]
                  if not item["resolved"] and (item.get("requires_resolution") or item["prominent"])]
    if unresolved or any(not item["resolved"] for item in inspection["outputs"]):
        raise WorkflowError("Resolve App Mode bindings before submission")
    editable = {key for item in inspection["bindings"] if item["resolved"] for key in (item["id"], *item.get("aliases", []))}
    editable.update(inspection["inactive"])
    if any(key not in editable and any(row.get("definition_binding_id") == key for row in inspection["bindings"])
           for key in (values or {})):
        raise WorkflowError("Shared subgraph binding is ambiguous; choose a specific instance")
    if set(values or {}) - editable:
        raise WorkflowError("Unknown, connected, or unresolved input binding")
    return compile_graph(workflow, object_info, values or {})


def export_workflow(record: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(record["workflow"]))


def rebind_workflow(workflow: Mapping[str, Any], source: str, target: str,
                    object_info: Mapping[str, Any]) -> dict[str, Any]:
    inspection = inspect_workflow(workflow, object_info)
    bindings = {item["id"]: item for item in inspection["bindings"]}
    if source not in bindings or target not in bindings or not bindings[target]["resolved"]:
        raise WorkflowError("Rebinding requires an existing source and resolved typed target")
    source_schema = bindings[source]["schema"]
    if source_schema and source_schema["type"] != bindings[target]["schema"]["type"]:
        raise WorkflowError("Rebinding must preserve the input type")
    result = copy.deepcopy(dict(workflow))
    preferences = result.setdefault("extra", {}).setdefault("dgxm_studio", {}).setdefault("bindings", {})
    preferences.setdefault(source, {})["target"] = target
    return result
