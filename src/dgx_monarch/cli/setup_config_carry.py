"""Operator tables that a setup re-apply carries from the existing cluster.toml.

Setup writes ``[cluster]`` and ``[[hosts]]`` from its options on every run. It
never writes ``[fabric.<name>]``, ``[monarch]`` or ``[studio]``, and in
``[worker_args]`` it writes only the profile's residency keys. A re-apply over
an existing file therefore keeps every ``[fabric.<name>]`` table, dormant ones
included, ``[monarch]`` whole (``host_targets`` lives in ``[monarch.rdma]``),
``[studio]``, and the operator's own ``[worker_args]`` keys; the profile's
keys still win there, by the rule ``operator_profiles.apply_profile`` uses.

Each kept table is rewritten from its parsed values with the quoting helper
the template renderer uses, so comments inside it do not survive, and it must
parse back to the same values. A table that fails the checks the loader runs,
for the hosts this setup lists, stays out of the candidate and blocks the
apply, so setup never drops one silently. An existing file that is not valid
TOML blocks the apply too, because setup cannot tell which tables it holds.
docs/CLUSTER.md#re-running-setup describes the operator's view.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .. import config_schema as _schema
from ..config import _toml_string
from ..monarch_policy import parse_policy
from ..operator_profiles import ProfileResolution

KEPT_WARNING = "existing_config_overrides_kept"
INVALID_TOML_BLOCKER = "existing_config_invalid_toml"
# Emission order: [worker_args] sits where the renderer puts it, and the
# tables setup never writes follow it in this order.
KINDS = ("worker_args", "fabric", "monarch", "studio")
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
# The schema validators prefix their messages with a path. Setup reports only
# the table name, so the placeholder never reaches the operator.
_PLACEHOLDER = Path("cluster.toml")


@dataclass(frozen=True)
class CarriedCandidate:
    """A rendered candidate plus the existing file's tables it keeps or refuses."""

    text: str
    kept: tuple[str, ...] = ()
    refused: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()


def kept_counts(headers: Sequence[str]) -> dict[str, int]:
    """Kept tables per root table, the form a receipt can hold without naming a profile."""
    return {kind: sum(header.split(".", 1)[0] == kind for header in headers) for kind in KINDS}


def carry_existing_tables(
    render: Callable[[ProfileResolution], str],
    existing: str | None,
    resolution: ProfileResolution,
    host_names: Sequence[str],
) -> CarriedCandidate:
    """Render with ``render`` and append the operator tables ``existing`` holds.

    ``existing`` is the current file's text, or None when there is no file; then,
    and whenever nothing is kept, the text is exactly ``render(resolution)``.
    """
    if existing is None:
        return CarriedCandidate(render(resolution))
    try:
        raw = tomllib.loads(existing)
    except tomllib.TOMLDecodeError:
        return CarriedCandidate(render(resolution), blockers=(INVALID_TOML_BLOCKER,))
    lines: list[str] = []
    kept: list[str] = []
    refused: list[str] = []
    worker_args_kept = _carry_worker_args(raw, resolution, lines, kept, refused)
    _carry_fabric(raw, lines, kept, refused)
    _carry_monarch(raw, tuple(host_names), lines, kept, refused)
    _carry_studio(raw, lines, kept, refused)
    # The renderer cannot write a carried non-boolean such as reserve_vram_gb,
    # so when [worker_args] is kept the merged table replaces its block.
    base = render(replace(resolution, worker_args_set={}) if worker_args_kept else resolution)
    text = (base + "\n".join(lines) + "\n") if lines else base
    blockers = sorted({f"existing_config_{header.split('.', 1)[0]}_invalid" for header in refused})
    return CarriedCandidate(text, tuple(kept), tuple(refused), tuple(blockers))


def _carry_worker_args(
    raw: Mapping[str, Any],
    resolution: ProfileResolution,
    lines: list[str],
    kept: list[str],
    refused: list[str],
) -> bool:
    table = raw.get("worker_args")
    if table is None:
        return False
    if not isinstance(table, dict):
        refused.append("worker_args")
        return False
    owned = set(resolution.worker_args_set) | set(resolution.worker_args_unset)
    if not set(table) - owned:
        return False  # only the profile's keys: the renderer writes them as before
    merged = {key: value for key, value in table.items() if key not in resolution.worker_args_unset}
    merged.update(resolution.worker_args_set)
    try:
        _schema.validate_worker_args(merged, context="worker_args")
    except ValueError:
        refused.append("worker_args")
        return False
    return _keep(("worker_args",), merged, lines, kept, refused)


def _carry_fabric(raw: Mapping[str, Any], lines: list[str], kept: list[str], refused: list[str]) -> None:
    tables = raw.get("fabric")
    if tables is None:
        return
    if not isinstance(tables, dict):
        refused.append("fabric")
        return
    for name, values in tables.items():
        try:
            if not name or not isinstance(values, dict):
                raise ValueError("a fabric entry must be a named table")
            _schema.validate_fabric_env(values, name)
        except ValueError:
            refused.append(_header(("fabric", name)))
            continue
        _keep(("fabric", name), values, lines, kept, refused)


def _carry_monarch(
    raw: Mapping[str, Any],
    host_names: tuple[str, ...],
    lines: list[str],
    kept: list[str],
    refused: list[str],
) -> None:
    table = raw.get("monarch")
    if table is None:
        return
    try:
        parse_policy(table, host_names=host_names)
    except (ValueError, TypeError):
        rdma = table.get("rdma") if isinstance(table, dict) else None
        targets = rdma.get("host_targets") if isinstance(rdma, dict) else None
        stale = isinstance(targets, dict) and bool(set(targets) - set(host_names))
        refused.append("monarch.rdma.host_targets" if stale else "monarch")
        return
    _keep(("monarch",), table, lines, kept, refused)


def _carry_studio(raw: Mapping[str, Any], lines: list[str], kept: list[str], refused: list[str]) -> None:
    table = raw.get("studio")
    if table is None:
        return
    try:
        _schema.validate_studio_table(table, _PLACEHOLDER)
    except ValueError:
        refused.append("studio")
        return
    _keep(("studio",), table, lines, kept, refused)


def _keep(
    path: tuple[str, ...],
    table: Mapping[str, Any],
    lines: list[str],
    kept: list[str],
    refused: list[str],
) -> bool:
    """Append one table's lines once they parse back to the same values."""
    try:
        block, headers = _table_lines(path, table)
        parsed: Any = tomllib.loads("\n".join(block))
        for part in path:
            parsed = parsed[part]
        same = _canonical(parsed) == _canonical(table)
    except (ValueError, TypeError, KeyError):
        same = False
    if not same:
        refused.append(_header(path))
        return False
    lines.extend(block)
    kept.extend(headers)
    return True


def _table_lines(path: tuple[str, ...], table: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """TOML lines for a table: its own values under its header, then each subtable."""
    scalars = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
    lines: list[str] = []
    headers: list[str] = []
    # A table holding only subtables needs no header of its own.
    if scalars or len(scalars) == len(table):
        header = _header(path)
        headers.append(header)
        lines.extend(["", f"[{header}]", *(f"{_key(key)} = {_value(value)}" for key, value in scalars)])
    for key, value in table.items():
        if isinstance(value, dict):
            child_lines, child_headers = _table_lines((*path, key), value)
            lines.extend(child_lines)
            headers.extend(child_headers)
    return lines, headers


def _header(path: tuple[str, ...]) -> str:
    return ".".join(_key(part) for part in path)


def _key(key: object) -> str:
    if not isinstance(key, str):
        raise TypeError("TOML keys are strings")
    return key if _BARE_KEY.fullmatch(key) else _toml_string(key, "key")


def _value(value: object) -> str:
    if isinstance(value, bool):  # before int: a bool is an int
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)  # TOML reads Python's float spellings, inf and nan included
    if isinstance(value, str):
        return _toml_string(value, "value")
    raise TypeError(f"setup cannot write a {type(value).__name__} value")


def _canonical(value: object) -> str:
    """Compare parsed tables by meaning: key order aside, types and NaN included."""
    return json.dumps(value, sort_keys=True)
