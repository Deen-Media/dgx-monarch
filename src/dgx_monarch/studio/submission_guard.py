"""Check a reviewed prompt against the live driver schema without voiding reviews on file changes.

ComfyUI rebuilds its file lists (models, LoRAs, inputs) on every /object_info
call, so a digest of the whole response changes whenever any file appears. A
review instead binds to the definitions of the node classes its prompt uses,
with choice lists left out, and each selected choice is checked against the
live list on its own.
"""
from __future__ import annotations

from typing import Any

from .contracts import StudioError, digest
from .store import Store
from .workflows import WorkflowError, node_inputs

NODE_ERROR_ROWS = 64
NODE_ERROR_TEXT = 128
UPLOAD_NAME_CHARS = 512


def _without_choices(spec: Any) -> Any:
    if not isinstance(spec, (list, tuple)) or not spec:
        return spec
    kind = spec[0]
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else None
    if isinstance(kind, list):
        kind = "COMBO"
    elif kind != "COMBO" or options is None:
        return spec
    # A choice default is usually the first file in the list; the prompt
    # always carries an explicit value, so it does not change execution.
    stripped = None if options is None else {key: value for key, value in options.items() if key not in {"options", "default"}}
    return [kind, *([stripped] if stripped is not None else []), *spec[2:]]


def _class_identity(info: Any) -> Any:
    if not isinstance(info, dict):
        return None
    inputs = info.get("input")
    if not isinstance(inputs, dict):
        return info
    stripped = {group: {name: _without_choices(spec) for name, spec in specs.items()} if isinstance(specs, dict) else specs
                for group, specs in inputs.items()}
    return {**info, "input": stripped}


def prompt_classes(prompt: Any) -> list[str]:
    if not isinstance(prompt, dict):
        return []
    return sorted({node["class_type"] for node in prompt.values()
                   if isinstance(node, dict) and isinstance(node.get("class_type"), str)})


def schema_identity(prompt: Any, schema: dict[str, Any]) -> str:
    """Digest the definitions of the classes a prompt uses, without file-list choices."""
    return digest({name: _class_identity(schema.get(name)) for name in prompt_classes(prompt)})


def safe_upload_name(value: Any) -> bool:
    """A stored upload binding is a relative path inside the driver's input store."""
    if not isinstance(value, str) or not 0 < len(value) <= UPLOAD_NAME_CHARS:
        return False
    if value.startswith("/") or "\\" in value or ":" in value:
        return False
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return False
    return not any(part in {"", ".", ".."} for part in value.split("/"))


def stored_upload_names(store: Store) -> frozenset[str]:
    """Binding values of inputs and masks this companion uploaded (never job outputs)."""
    with store.lock:
        rows = store.db.execute(
            "SELECT DISTINCT json_extract(document,'$.binding_value') FROM objects WHERE kind='artifacts' "
            "AND json_extract(document,'$.job_id') IS NULL AND json_extract(document,'$.kind') IN ('input','mask') "
            "AND json_extract(document,'$.binding_value') IS NOT NULL").fetchall()
    return frozenset(row[0] for row in rows if safe_upload_name(row[0]))


def referenced_uploads(value: Any, uploads: frozenset[str]) -> frozenset[str]:
    """Return the stored upload bindings that appear as exact strings anywhere in a request."""
    if not uploads:
        return frozenset()
    found: set[str] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if item in uploads:
                found.add(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return frozenset(found)


def _accepts_uploads(options: Any) -> bool:
    # Studio stores PNG images in the driver's input store, so only image
    # upload widgets that read that store can take them.
    return (isinstance(options, dict) and options.get("image_upload") is True
            and options.get("image_folder") in (None, "input"))


def _widen(spec: Any, names: list[str]) -> Any:
    if not isinstance(spec, (list, tuple)) or len(spec) < 2 or not _accepts_uploads(spec[1]):
        return spec
    kind, options = spec[0], spec[1]
    if isinstance(kind, list):
        extra = [name for name in names if name not in kind]
        return [[*kind, *extra], options, *spec[2:]] if extra else spec
    if kind == "COMBO" and isinstance(options.get("options"), list):
        extra = [name for name in names if name not in options["options"]]
        return [kind, {**options, "options": [*options["options"], *extra]}, *spec[2:]] if extra else spec
    return spec


def upload_view(schema: dict[str, Any], names: frozenset[str]) -> dict[str, Any]:
    """A per-request schema view whose image upload widgets also list this request's stored uploads.

    The driver's own schema is never changed: digests and identities use it,
    and `validate_value` stays a strict membership check against the view.
    """
    if not names:
        return schema
    ordered = sorted(names)
    view = dict(schema)
    for class_type, info in schema.items():
        inputs = info.get("input") if isinstance(info, dict) else None
        if not isinstance(inputs, dict):
            continue
        widened: dict[str, Any] | None = None
        for group, specs in inputs.items():
            if not isinstance(specs, dict):
                continue
            for name, spec in specs.items():
                replacement = _widen(spec, ordered)
                if replacement is not spec:
                    widened = widened or {key: dict(value) if isinstance(value, dict) else value for key, value in inputs.items()}
                    widened[group][name] = replacement
        if widened is not None:
            view[class_type] = {**info, "input": widened}
    return view


def check_live_choices(prompt: dict[str, Any], schema: dict[str, Any], uploads: frozenset[str]) -> None:
    """Refuse a reviewed prompt whose node class or selected choice the live driver no longer offers."""
    for node_id, node in prompt.items():
        class_type = node.get("class_type") if isinstance(node, dict) else None
        info = schema.get(class_type) if isinstance(class_type, str) else None
        if not isinstance(info, dict):
            raise StudioError("schema_changed", "Driver capabilities changed; review a fresh preview.", 409)
        try:
            fields = node_inputs(info)
        except WorkflowError as exc:
            raise StudioError("schema_changed", "Driver capabilities changed; review a fresh preview.", 409) from exc
        inputs = node.get("inputs", {}) if isinstance(node.get("inputs"), dict) else {}
        for name, field, _ in fields:
            if field["type"] != "COMBO" or name not in inputs or isinstance(inputs[name], list):
                continue
            value = inputs[name]
            if value in field.get("choices", []):
                continue
            if _accepts_uploads(field) and value in uploads and safe_upload_name(value):
                continue
            raise StudioError("choice_unavailable", f"{node_id}.{name}: the driver no longer offers the selected "
                              "option; review a fresh preview.", 409)


def check_reviewed(submission: dict[str, Any], schema: dict[str, Any], uploads: frozenset[str]) -> None:
    """Compare a frozen submission with the live schema: class identity first, then each choice."""
    frozen = submission.get("prompt")
    prompt: dict[str, Any] = frozen if isinstance(frozen, dict) else {}
    recorded = submission.get("schema_identity")
    if recorded is None:
        # Reviews made before class identities existed keep the whole-schema rule.
        if "schema_digest" in submission and submission["schema_digest"] != digest(schema):
            raise StudioError("schema_changed", "Driver capabilities changed; review a fresh preview.", 409)
    elif recorded != schema_identity(prompt, schema):
        raise StudioError("schema_changed", "Driver capabilities changed; review a fresh preview.", 409)
    check_live_choices(prompt, schema, uploads)


def _short(value: Any) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return None
    cleaned = "".join(char for char in value if ord(char) >= 32 and ord(char) != 127)
    return cleaned[:NODE_ERROR_TEXT]


def node_error_summary(value: Any) -> list[dict[str, str | None]]:
    """Name the failing node, class, input and error type; drop messages, values and paths."""
    rows: list[dict[str, str | None]] = []
    if isinstance(value, list):
        # The transport already reduced a refusal body to these rows (contract C1).
        for row in value:
            if isinstance(row, dict) and len(rows) < NODE_ERROR_ROWS:
                rows.append({key: _short(row.get(key)) for key in ("node_id", "class_type", "input_name", "type")})
        return rows
    if not isinstance(value, dict):
        return rows
    for node_id, record in value.items():
        if not isinstance(record, dict):
            continue
        listed = record.get("errors")
        errors: list[Any] = listed if isinstance(listed, list) and listed else [{}]
        for item in errors:
            if len(rows) >= NODE_ERROR_ROWS:
                return rows
            error: dict[str, Any] = item if isinstance(item, dict) else {}
            info = error.get("extra_info")
            extra: dict[str, Any] = info if isinstance(info, dict) else {}
            rows.append({"node_id": _short(node_id), "class_type": _short(record.get("class_type")),
                         "input_name": _short(extra.get("input_name")), "type": _short(error.get("type"))})
    return rows
