"""Schema-driven widget layout and V3 dynamic input finalization.

ComfyUI finalizes V3 dynamic inputs (Autogrow and DynamicCombo) from the live
prompt before it validates a node, in ``comfy_api/latest/_io.py``
``get_finalized_class_inputs``, and it validates only those declared inputs. The
frontend lays widgets out in ``input_order``, required before optional. It puts
a DynamicCombo's child widgets, named ``parent.child``, right after it, and it
appends its own helper widgets (upload buttons, media players) after every
declared widget. This module applies both rules from ``/object_info`` alone; it
never runs frontend code.
"""
from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from .workflows import SCALAR_TYPES, WorkflowError

DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO_V3"
AUTOGROW = "COMFY_AUTOGROW_V3"
MATCH_TYPE = "COMFY_MATCHTYPE_V3"
ANY_TYPE = "*"
CONTROL_VALUES = frozenset({"fixed", "increment", "decrement", "randomize"})
_UNSUPPORTED_DYNAMIC = frozenset({"COMFY_DYNAMICSLOT_V3", "COMFY_DYNAMICGROUP_V3"})
_MAX_DEPTH = 32
MISSING: Any = object()
# Stands in for a linked widget value: a link never selects a dynamic option.
LINKED = ["link", 0]

Field = tuple[str, Any, bool]


def _options(spec: Any) -> dict[str, Any]:
    return spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}


def _unsupported(kind: Any) -> bool:
    return isinstance(kind, str) and kind in _UNSUPPORTED_DYNAMIC


def _checked(name: str, spec: Any) -> Any:
    if not isinstance(spec, (list, tuple)) or not spec:
        raise WorkflowError(f"Invalid input schema: {name}")
    return spec


def _option_keys(name: str, options: Mapping[str, Any]) -> list[Any]:
    rows = options.get("options")
    if not isinstance(rows, list) or any(not isinstance(row, dict) or "key" not in row for row in rows):
        raise WorkflowError(f"Invalid dynamic option schema: {name}")
    return [row["key"] for row in rows]


def field_schema(name: str, spec: Any) -> dict[str, Any]:
    """Return the typed widget schema Studio validates and shows for one input."""
    spec = _checked(name, spec)
    options = copy.deepcopy(_options(spec))
    kind = spec[0]
    if isinstance(kind, list):
        options.update(type="COMBO", choices=copy.deepcopy(kind))
    elif kind == "COMBO" and isinstance(options.get("options"), list):
        options.update(type="COMBO", choices=options["options"])
    elif kind == DYNAMIC_COMBO:
        # The frontend draws a DynamicCombo as a combo of its option keys whose
        # value starts at the first key. ``dynamic`` tells Studio that changing
        # it changes which child inputs exist.
        keys = _option_keys(name, options)
        options.pop("options")
        options.update(type="COMBO", choices=keys, dynamic=True)
        if keys:
            options.setdefault("default", keys[0])
    elif isinstance(kind, str):
        # A union input such as "FLOAT,INT" is drawn as its widgetType.
        widget = options.get("widgetType")
        options["type"] = widget if isinstance(widget, str) and widget in SCALAR_TYPES else kind
    else:
        raise WorkflowError(f"Unsupported input type schema: {name}")
    return options


def declared_specs(info: Mapping[str, Any]) -> list[Field]:
    """Return (name, raw spec, required) in the declared input order."""
    fields = []
    for group in ("required", "optional"):
        definitions = info.get("input", {}).get(group, {})
        order = info.get("input_order", {}).get(group, list(definitions))
        if set(order) != set(definitions) or len(order) != len(definitions):
            raise WorkflowError("object_info input order does not match its input schema")
        for name in order:
            fields.append((name, _checked(name, definitions[name]), group == "required"))
    return fields


def declared_inputs(info: Mapping[str, Any]) -> list[tuple[str, dict[str, Any], bool]]:
    """Declared top-level inputs with typed schemas, before dynamic expansion."""
    return [(name, field_schema(name, spec), required) for name, spec, required in declared_specs(info)]


def _option_inputs(name: str, spec: Any, key: Any) -> list[Field]:
    for row in _options(spec).get("options", []):
        if isinstance(row, dict) and row.get("key") == key and not isinstance(key, list):
            inputs = row.get("inputs", {})
            if not isinstance(inputs, dict):
                raise WorkflowError(f"Invalid dynamic option schema: {name}")
            return [(f"{name}.{child}", _checked(child, child_spec), group == "required")
                    for group in ("required", "optional")
                    for child, child_spec in (inputs.get(group) or {}).items()]
    return []


def _autogrow_inputs(name: str, spec: Any) -> list[Field]:
    template = _options(spec).get("template")
    if not isinstance(template, dict) or not isinstance(template.get("input"), dict):
        raise WorkflowError(f"Invalid Autogrow schema: {name}")
    entry = next(((group, row) for group in ("required", "optional")
                  for row in (template["input"].get(group) or {}).values()), None)
    if entry is None:
        raise WorkflowError(f"Autogrow input has no template: {name}")
    group, row = entry
    maximum, minimum = template.get("max", 10), template.get("min", 1)
    if isinstance(template.get("names"), list):
        names = [str(item) for item in template["names"]]
    elif isinstance(template.get("prefix"), str) and type(maximum) is int and 0 < maximum <= 100:
        names = [f"{template['prefix']}{index}" for index in range(maximum)]
    else:
        raise WorkflowError(f"Invalid Autogrow schema: {name}")
    if type(minimum) is not int or minimum < 0 or len(names) > 100:
        raise WorkflowError(f"Invalid Autogrow schema: {name}")
    return [(f"{name}.{item}", _checked(name, row), index < minimum and group == "required")
            for index, item in enumerate(names)]


def finalize_inputs(info: Mapping[str, Any], live: Mapping[str, Any], *,
                    fill_defaults: bool = False, edit: Callable[[str], Any] | None = None) -> list[Field]:
    """Expand dynamic inputs from live values, as ComfyUI does before validating.

    Every Autogrow name up to its maximum is declared, and only the first
    ``min`` names are required. A DynamicCombo is declared only when it has a
    value: a live one, then an ``edit``, and ``fill_defaults`` gives a missing
    one its first key, as the frontend's widget does. The selected option's
    inputs follow it with dotted names. ComfyUI drops a DynamicCombo whose value
    names no option; Studio keeps it declared, so the value check refuses a key
    the node would ignore.
    """
    result: list[Field] = []

    def expand(name: str, spec: Any, required: bool, depth: int) -> None:
        if depth > _MAX_DEPTH:
            raise WorkflowError("Dynamic input nesting exceeds 32 levels")
        kind = spec[0]
        if _unsupported(kind):
            raise WorkflowError(f"{name} uses {kind}, which needs an explicit adapter")
        if kind == AUTOGROW:
            for child in _autogrow_inputs(name, spec):
                expand(*child, depth + 1)
            return
        if kind != DYNAMIC_COMBO:
            result.append((name, spec, required))
            return
        value = live.get(name, MISSING)
        if value is MISSING and edit is not None:
            value = edit(name)
        if value is MISSING and fill_defaults:
            value = field_schema(name, spec).get("default", MISSING)
        if value is MISSING:
            return
        result.append((name, spec, required))
        for child in _option_inputs(name, spec, value):
            expand(*child, depth + 1)

    for name, spec, required in declared_specs(info):
        expand(name, spec, required, 0)
    names = [name for name, _, _ in result]
    if len(set(names)) != len(names):
        raise WorkflowError("Dynamic inputs expand to duplicate names")
    return result


def resolve_inputs(info: Mapping[str, Any], live: Mapping[str, Any], lookup: Callable[[str], Any], *,
                   fill_defaults: bool) -> list[Field]:
    """Finalize with explicit edits applied until no newly declared input remains.

    Each declared name takes its edit once, so the loop never compares values;
    every round can reveal only the next nesting level.
    """
    current = dict(live)
    applied: set[str] = set()
    for _ in range(_MAX_DEPTH + 2):
        # An API prompt can leave an optional choice out; an edit still declares it.
        fields = finalize_inputs(info, current, fill_defaults=fill_defaults, edit=lookup)
        pending = [name for name, _, _ in fields if name not in applied]
        if not pending:
            return fields
        for name in pending:
            applied.add(name)
            value = lookup(name)
            if value is not MISSING:
                current[name] = value
    raise WorkflowError("Dynamic input edits do not settle")


def _is_widget(spec: Any) -> bool:
    if spec[0] == AUTOGROW:
        return False
    schema = field_schema("", spec) if spec[0] != DYNAMIC_COMBO else {"type": "COMBO"}
    return schema["type"] in SCALAR_TYPES and not _options(spec).get("forceInput")


def _has_upload(info: Mapping[str, Any]) -> bool:
    return any(key.endswith("_upload") and value for spec in info.get("input", {}).get("required", {}).values()
               if isinstance(spec, (list, tuple)) for key, value in _options(spec).items())


def layout_values(node: Mapping[str, Any], info: Mapping[str, Any], *, api: bool = False) -> dict[str, Any]:
    """Map serialized widget values to input names, in the frontend's widget order."""
    if api:
        return copy.deepcopy(dict(node.get("inputs", {})))
    serialized = node.get("widgets_values", [])
    if isinstance(serialized, dict):
        known = {name for name, spec, _ in finalize_inputs(info, serialized, fill_defaults=True) if _is_widget(spec)}
        unknown = set(serialized) - known
        if unknown:
            raise WorkflowError(f"Unknown named widgets: {sorted(unknown)}")
        return copy.deepcopy(serialized)
    if not isinstance(serialized, list):
        raise WorkflowError("Serialized widget values must be an array or object")
    result: dict[str, Any] = {}

    def walk(fields: list[Field], position: int, depth: int) -> int:
        if depth > _MAX_DEPTH:
            raise WorkflowError("Dynamic input nesting exceeds 32 levels")
        for name, spec, _ in fields:
            if _unsupported(spec[0]):
                raise WorkflowError(f"{name} uses {spec[0]}, which needs an explicit adapter")
            if not _is_widget(spec):
                continue
            if position >= len(serialized):
                return position
            result[name] = copy.deepcopy(serialized[position])
            position += 1
            # Core seed widgets add a UI-only control after their number.
            has_control = _options(spec).get("control_after_generate", name.rsplit(".", 1)[-1] in {"seed", "noise_seed"})
            if has_control and position < len(serialized) and isinstance(serialized[position], str):
                if serialized[position] in CONTROL_VALUES:
                    position += 1
            if spec[0] == DYNAMIC_COMBO:
                position = walk(_option_inputs(name, spec, result[name]), position, depth + 1)
        return position

    position = walk(declared_specs(info), 0, 0)
    # The frontend appends upload helper widgets after every declared widget; their
    # slots carry any value or none and never reach the prompt.
    if position != len(serialized) and not _has_upload(info):
        raise WorkflowError("Widget serialization has unsupported extra values; explicit widget adapter required")
    return result


def is_link(value: Any) -> bool:
    """ComfyUI's link shape: [node_id, output_slot]."""
    return (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            and isinstance(value[1], (int, float)) and not isinstance(value[1], bool))


def link_accepts(received: Any, expected: Any) -> bool:
    """ComfyUI's non-strict ``validate_node_input(received_type, input_type)``.

    A node whose own validator takes ``input_types`` skips this check in ComfyUI.
    object_info does not show that, so such a node can accept a link this refuses.
    """
    if received == expected:
        return True
    if ANY_TYPE in (received, expected) or MATCH_TYPE in (received, expected):
        return True
    # Custom nodes may output a list of options; ComfyUI lets it feed a V3 combo.
    if isinstance(received, list) and expected == "COMBO":
        return True
    if not isinstance(received, str) or not isinstance(expected, str):
        return False
    received_types = {part.strip() for part in received.split(",")}
    expected_types = {part.strip() for part in expected.split(",")}
    if ANY_TYPE in received_types or ANY_TYPE in expected_types:
        return True
    return bool(received_types & expected_types)
