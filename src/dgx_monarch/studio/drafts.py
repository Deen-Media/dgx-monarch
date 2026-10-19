"""Pure draft transformations; these never submit work or mutate saved jobs."""

from __future__ import annotations

import copy
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any


class DraftError(ValueError):
    """A typed draft cannot be represented safely."""


def validate_value(value: Any, schema: Mapping[str, Any]) -> Any:
    """Validate without coercion, clipping, or substituting missing values."""
    kind = schema.get("type")
    if kind == "INT" and (not isinstance(value, int) or isinstance(value, bool)):
        raise DraftError("Expected an integer")
    if kind == "FLOAT" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise DraftError("Expected a number")
    if kind == "BOOLEAN" and not isinstance(value, bool):
        raise DraftError("Expected a boolean")
    if kind == "STRING" and not isinstance(value, str):
        raise DraftError("Expected text")
    if kind not in {"INT", "FLOAT", "BOOLEAN", "STRING", "COMBO"}:
        raise DraftError(f"No typed adapter for {kind}")
    if kind == "COMBO" and value not in schema.get("choices", []):
        raise DraftError("Value is not an available choice")
    if kind in {"INT", "FLOAT"}:
        if isinstance(value, float) and not math.isfinite(value):
            raise DraftError("Number must be finite")
        for key, invalid in (("min", lambda bound: value < bound), ("max", lambda bound: value > bound)):
            if key in schema and invalid(schema[key]):
                raise DraftError(f"Value violates {key}={schema[key]}")
    return copy.deepcopy(value)


def apply_preset(draft: Mapping[str, Any], preset: Mapping[str, Any]) -> dict[str, Any]:
    """Return a reviewable replacement set and a complete undo snapshot."""
    fields = preset.get("fields", {})
    if not isinstance(fields, dict):
        raise DraftError("Preset fields must be an object")
    result = copy.deepcopy(dict(draft))
    result.update(copy.deepcopy(fields))
    return {"draft": result, "replaced_fields": list(fields), "undo": copy.deepcopy(dict(draft))}


def undo_preset(applied: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(applied["undo"]))


def apply_styles(prompt: str, negative_prompt: str, styles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply enabled styles in supplied order, retaining the unexpanded text."""
    positive, negative = prompt, negative_prompt
    order = []
    for style in styles:
        if not style.get("enabled", True):
            continue
        order.append(style["id"])
        for field, current in (("prompt", positive), ("negative_prompt", negative)):
            template = style.get(field, "{prompt}")
            if not isinstance(template, str):
                raise DraftError("Style templates must be text")
            expanded = template.replace("{prompt}", current)
            if "{prompt}" not in template:
                expanded = ", ".join(part for part in (current, template) if part)
            if field == "prompt":
                positive = expanded
            else:
                negative = expanded
    return {"prompt": positive, "negative_prompt": negative, "style_order": order,
            "unexpanded": {"prompt": prompt, "negative_prompt": negative_prompt}}


def resolve_loras(stack: Sequence[Mapping[str, Any]], catalog: Mapping[str, Mapping[str, Any]],
                  model_id: str) -> list[dict[str, Any]]:
    """Resolve ordered active entries; compatibility must be supplied by catalog."""
    resolved = []
    for entry in stack:
        if not entry.get("enabled", True):
            continue
        identity = entry.get("id")
        asset = catalog.get(str(identity))
        if asset is None or not asset.get("available", False):
            raise DraftError(f"LoRA unavailable: {identity}")
        if model_id not in asset.get("compatible_models", []):
            raise DraftError(f"LoRA compatibility unverified: {identity} with {model_id}")
        strengths = {}
        for name, schema in asset.get("strengths", {}).items():
            strengths[name] = validate_value(entry.get(name, schema.get("default")), schema)
        if not strengths:
            raise DraftError(f"LoRA strength schema unavailable: {identity}")
        unknown = set(entry) - {"id", "enabled"} - set(strengths)
        if unknown:
            raise DraftError(f"Unsupported LoRA controls: {sorted(unknown)}")
        resolved.append({"id": identity, "asset": copy.deepcopy(asset), "strengths": strengths})
    return resolved


def expand_grid(base: Mapping[str, Any], axes: Sequence[Mapping[str, Any]],
                schemas: Mapping[str, Mapping[str, Any]], *, seed_policy: str,
                base_seed: int, excluded: Sequence[int] = (),
                validator: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Enumerate, validate and preview at most 1024 cells without scheduling them.

    Exclusions are explicit zero-based Cartesian cell indices. Invalid cells stay
    visible and block execution unless excluded. Seed-axis values take precedence;
    random seeds must be resolved by the caller before constructing this preview.
    """
    if seed_policy not in {"fixed", "increment"}:
        raise DraftError("Seed policy must be fixed or increment with a resolved base seed")
    validate_value(base_seed, {"type": "INT", "min": 0, "max": 2**64 - 1})
    names: list[str] = []
    values: list[list[Any]] = []
    for axis in axes:
        name, items = axis.get("name"), axis.get("values")
        if not isinstance(name, str) or name not in schemas:
            raise DraftError("Grid axes must be unique named typed inputs")
        if not isinstance(items, list) or not items:
            raise DraftError("Each grid axis needs a nonempty values array")
        names.append(name)
        values.append(items)
    if len(set(names)) != len(names):
        raise DraftError("Grid axes must be unique named typed inputs")
    total = math.prod(len(items) for items in values)
    if total > 1024:
        raise DraftError("Grid exceeds 1024 combinations")
    if any(not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < total for index in excluded):
        raise DraftError("Excluded cell index is out of range")
    excluded_set = set(excluded)
    cells = []
    for index, combination in enumerate(itertools.product(*values)):
        parameters = copy.deepcopy(dict(base))
        parameters.update(dict(zip(names, combination, strict=True)))
        if "seed" not in names:
            parameters["seed"] = base_seed + (index if seed_policy == "increment" else 0)
        errors = []
        for name, value in parameters.items():
            if name in schemas:
                try:
                    validate_value(value, schemas[name])
                except DraftError as exc:
                    errors.append(f"{name}: {exc}")
        try:
            validate_value(parameters["seed"], {"type": "INT", "min": 0, "max": 2**64 - 1})
            if validator:
                validator(parameters)
        except ValueError as exc:
            errors.append(str(exc))
        cells.append({"index": index, "coordinates": dict(zip(names, combination, strict=True)),
                      "parameters": parameters, "excluded": index in excluded_set, "errors": errors})
    return {"axes": copy.deepcopy(list(axes)), "seed_policy": seed_policy, "base_seed": base_seed,
            "total": total, "job_count": total - len(excluded_set), "cells": cells,
            "ready": all(cell["excluded"] or not cell["errors"] for cell in cells)}
