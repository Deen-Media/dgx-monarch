"""Build-time parity discovery and runtime access to the packaged inventory.

Discovery reads Python syntax and the public parser, never starts a runtime or
imports ComfyUI. An inventory entry is evidence of a surface, not an assertion
that Studio has implemented or verified an operation adapter.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import hashlib
import importlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path
from typing import Any

from .inventory_registry import apply_studio_views
from .presentation import presentation_contract

SCHEMA_VERSION = 1
PINNED_BASELINE = "5604ac77799f1f3e1a077a2f841dfce59fbcb892"
INTERFACES = ("studio", "cli", "tui", "panel", "nodes")
REQUIRED_FIELDS = frozenset({"id", "kind", "name", "source", "datatype", "choices", "default", "units",
                             "applicability", "validation", "guard", "preview", "apply", "recovery", "interfaces"})
# Discovery runs with this HOME, so a home-relative default normalizes to the
# same "~/..." text under any build account, HOME=/ included. No real default
# begins with it.
DISCOVERY_HOME = "/dgxm-inventory-discovery-home"


@contextlib.contextmanager
def discovery_home() -> Iterator[None]:
    # Import discovery's modules first, so a module-level home path computed
    # at import never keeps the sentinel after discovery ends.
    for name in ("..cli.arguments", "..config", "..config_schema", "..monarch_policy", ".management_parity",
                 ".management_fields"):
        importlib.import_module(name, __package__)
    saved = os.environ.get("HOME")
    os.environ["HOME"] = DISCOVERY_HOME
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = saved


def load_inventory() -> dict[str, Any]:
    return json.loads(files(__package__).joinpath("data/parity.json").read_text(encoding="utf-8"))


def _row(identity: str, kind: str, name: str, source: str, **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": identity, "kind": kind, "name": name, "source": source, "datatype": "action",
        "choices": None, "default": None, "units": None,
        "applicability": "Requires the owning backend capability and its runtime validation.",
        "validation": "Owning backend is authoritative; discovery does not execute validation.",
        "guard": "Existing backend consent, ownership, admission and lifecycle guards remain mandatory.",
        "preview": "No generic preview is implied; consult the operation's backend contract.",
        "apply": "Unavailable until an explicit typed adapter advertises this capability.",
        "recovery": "Reconcile authoritative state before retry; an unknown outcome is not a failure.",
        "interfaces": {name: {"status": "unavailable", "reason": "No direct representation discovered."}
                       for name in INTERFACES},
    }
    # build_inventory replaces this with the reviewed view (inventory_registry).
    result["interfaces"]["studio"] = {"status": "source_only", "reason": "No Studio view has been reviewed."}
    result.update(values)
    return result


def _units(name: str) -> str | None:
    if name.endswith(("_gib", "_vram_gb", "_reserve_gb")):
        return "GiB"
    if name.endswith("_bytes"):
        return "bytes"
    if name.endswith("_s") or name in {"interval", "timeout"}:
        return "s"
    if name in {"steps", "start_at_step", "end_at_step", "gpus", "gpus_per_host", "pipeline_depth"}:
        return "count"
    return None


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, str):
        # Public manifests must not embed the build user's private home;
        # discovery_home() makes every home-relative default start here.
        return "~" + value[len(DISCOVERY_HOME):] if value.startswith(DISCOVERY_HOME + "/") else value
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    raise TypeError(f"Nonportable inventory value: {type(value).__name__}")


def discover_cli() -> list[dict[str, Any]]:
    with discovery_home():
        return _discover_cli()


def _discover_cli() -> list[dict[str, Any]]:
    from ..cli.arguments import build_parser

    class Commands(dict):
        def __missing__(self, key):
            def command(_args):
                raise RuntimeError("Inventory commands must never execute")
            command.__name__ = key
            self[key] = command
            return command

    rows = []

    def walk(parser, path):
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for command, child in action.choices.items():
                    command_path = [*path, command]
                    source = "src/dgx_monarch/cli/arguments.py:build_parser"
                    row = _row("cli.command." + ".".join(command_path), "cli_command", "dgxm " + " ".join(command_path), source)
                    row["interfaces"]["cli"] = {"status": "available", "command": row["name"]}
                    row["backend_symbol"] = getattr(child.get_default("func"), "__name__", None)
                    rows.append(row)
                    walk(child, command_path)
            else:
                # Destination is not unique: gate --last/--all/--repair share it.
                option = action.option_strings[-1] if action.option_strings else action.dest
                identity = ".".join(["cli.option", *(path or ["global"]), option.lstrip("-").replace("-", "_")])
                datatype = "boolean" if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)) else (
                    getattr(action.type, "__name__", "string") if action.type else "string")
                if isinstance(action, argparse._AppendAction):
                    datatype = "array[" + datatype + "]"
                row = _row(identity, "cli_option", option, "src/dgx_monarch/cli/arguments.py:build_parser",
                           datatype=datatype, choices=list(action.choices) if action.choices is not None else None,
                           default=_json_value(action.default), units=_units(action.dest))
                row.update({"destination": action.dest, "option_strings": action.option_strings,
                            "required": action.required, "nargs": action.nargs,
                            "help": action.help if action.help != argparse.SUPPRESS else None,
                            "constant": _json_value(action.const)})
                row["interfaces"]["cli"] = {"status": "available", "command": "dgxm " + " ".join([*path, option])}
                rows.append(row)
    walk(build_parser(Commands()), [])
    from .management_parity import validate_bindings
    mapping = validate_bindings(rows)
    for row in rows:
        row["studio_binding"] = mapping[row["id"]]
    return rows


def _literal(node: ast.AST, symbols: dict[str, Any]) -> Any:
    if isinstance(node, ast.Name) and node.id in symbols:
        return symbols[node.id]
    if isinstance(node, ast.List):
        result = []
        for item in node.elts:
            if isinstance(item, ast.Starred):
                result.extend(_literal(item.value, symbols))
            else:
                result.append(_literal(item, symbols))
        return result
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "keys" and not node.args:
        return list(_literal(node.func.value, symbols))
    return ast.literal_eval(node)


def _static_symbols(path: Path) -> dict[str, Any]:
    symbols: dict[str, Any] = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                try:
                    symbols[target.id] = _literal(node.value, symbols)
                except (ValueError, TypeError, AttributeError):
                    continue
    return symbols


def _adapter_families(root: Path) -> list[str]:
    """Read registered class family attributes without loading tensor libraries."""
    adapter_root = root / "src/dgx_monarch/adapters"
    classes = {}
    for path in adapter_root.glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node
    registry = ast.parse((adapter_root / "__init__.py").read_text())
    registered = next(node.value for node in registry.body if isinstance(node, ast.AnnAssign)
                      and isinstance(node.target, ast.Name) and node.target.id == "ADAPTERS")

    def family(name):
        cls = classes[name]
        for field in cls.body:
            if isinstance(field, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "family" for t in field.targets):
                return ast.literal_eval(field.value)
        for base in cls.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                return family(base.id)
        raise ValueError(f"Cannot statically resolve adapter family for {name}")

    if not isinstance(registered, (ast.Tuple, ast.List)):
        raise ValueError("Unsupported adapter registry expression")
    families = set()
    for call in registered.elts:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise ValueError("Unsupported adapter registry member")
        families.add(family(call.func.id))
    return sorted(families)


def _dict_nodes(node: ast.AST) -> dict[str, ast.expr]:
    if not isinstance(node, ast.Dict) or any(key is None for key in node.keys):
        raise ValueError("Expected literal schema dictionary")
    result = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None:
            raise ValueError("Schema dictionary expansion is unsupported")
        result[ast.literal_eval(key)] = value
    return result


def discover_nodes(root: Path) -> list[dict[str, Any]]:
    contract = json.loads((root / "tests/node_input_contract.json").read_text())
    nodes_dir = root / "src/dgx_monarch/nodes"
    classes = {}
    for path in sorted(nodes_dir.glob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = (path, node)
    symbols = _static_symbols(root / "src/dgx_monarch/constants.py")
    symbols.update(_static_symbols(root / "src/dgx_monarch/topology.py"))
    symbols["_FAMILY_CHOICES"] = ["auto", *_adapter_families(root)]
    symbols["_WEIGHT_DTYPES"] = list(_static_symbols(root / "src/dgx_monarch/loader_options.py")["SUPPORTED_WEIGHT_DTYPES"])
    rows = []
    for name, sections in contract.items():
        path, cls = classes[name]
        declared = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "INPUT_TYPES"), None)
        if declared is None:
            # The unconditional loader inherits the public schema unchanged.
            parent = next(base.id for base in cls.bases if isinstance(base, ast.Name) and base.id in classes)
            path, parent_cls = classes[parent]
            declared = next(n for n in parent_cls.body if isinstance(n, ast.FunctionDef) and n.name == "INPUT_TYPES")
        source = path.relative_to(root).as_posix() + ":" + name + ".INPUT_TYPES"
        row = _row("node." + name, "node", name, source, datatype="node_schema")
        row["interfaces"]["nodes"] = {"status": "available", "class_type": name}
        rows.append(row)
        local_symbols = {**symbols, **_static_symbols(path)}
        returned = next(n.value for n in ast.walk(declared) if isinstance(n, ast.Return))
        if not isinstance(returned, ast.Dict):
            raise ValueError(f"Unsupported node schema expression for {name}")
        source_sections = _dict_nodes(returned)
        for section, fields in sections.items():
            section_node = source_sections[section]
            source_fields = _dict_nodes(section_node)
            if set(source_fields) != {entry[0] for entry in fields}:
                raise ValueError(f"Node contract out of date: {name}.{section}")
            for order, (field_name, datatype, default) in enumerate(fields):
                spec = source_fields[field_name]
                options = {}
                choices = None
                choice_source = None
                if isinstance(spec, ast.Tuple):
                    if len(spec.elts) > 1:
                        # An unrecognized default expression remains pinned by the
                        # independently validated public node contract.
                        for key, value in _dict_nodes(spec.elts[1]).items():
                            try:
                                options[key] = _literal(value, local_symbols)
                            except (ValueError, TypeError):
                                pass
                    if datatype == "COMBO":
                        try:
                            choices = list(_literal(spec.elts[0], local_symbols))
                        except (ValueError, TypeError):
                            choice_source = ast.unparse(spec.elts[0])
                if "default" in options and options["default"] != default:
                    raise ValueError(f"Node default drift: {name}.{field_name}")
                row = _row("node.input." + name + "." + field_name, "node_input", field_name, source,
                           datatype=datatype, choices=choices, default=None if default == "<first-option>" else default,
                           units=_units(field_name), validation={key: value for key, value in options.items() if key != "default"})
                row.update({"node": name, "section": section, "order": order, "choice_source": choice_source,
                            "default_rule": "first_available_backend_option" if default == "<first-option>" else "declared",
                            "schema_source": "source AST + tests/node_input_contract.json"})
                row["interfaces"]["nodes"] = {"status": "available", "class_type": name, "input": field_name}
                row["applicability"] = options.get("tooltip") or f"{section} input of {name}; backend model/topology validation applies."
                if section == "hidden":
                    row["applicability"] = "Backend injected hidden graph identity; never an editable user field."
                rows.append(row)
    return rows


def discover_config() -> list[dict[str, Any]]:
    from .. import config_schema
    from ..config import ClusterConfig, HostConfig
    from ..monarch_policy import DiagnosticsPolicy, MonarchPolicy, RdmaPolicy

    rows = []
    for prefix, cls in (("cluster", ClusterConfig), ("hosts", HostConfig), ("monarch", MonarchPolicy),
                        ("monarch.rdma", RdmaPolicy), ("monarch.diagnostics", DiagnosticsPolicy)):
        for field in dataclasses.fields(cls):
            if prefix == "cluster" and field.name not in config_schema._CLUSTER_KEYS and field.name != "python_bin":
                continue
            if field.name in {"rdma", "diagnostics"}:
                continue
            name = "python" if field.name == "python_bin" else field.name
            default = field.default if field.default is not dataclasses.MISSING else None
            if field.default_factory is not dataclasses.MISSING:
                default = field.default_factory()
            row = _row("config." + prefix + "." + name, "configuration", prefix + "." + name,
                       "src/" + cls.__module__.replace(".", "/") + ".py:" + cls.__name__ + "." + field.name,
                       datatype=str(field.type), default=_json_value(default), units=_units(name))
            row["validation"] = "dgx_monarch.config.load_config / dgx_monarch.monarch_policy.parse_policy"
            row["interfaces"]["cli"] = {"status": "available", "representation": "cluster.toml through --config"}
            rows.append(row)
    for name in sorted(config_schema._KEYS):
        datatype = config_schema.WorkerArgs.__annotations__[name]
        row = _row("config.worker_args." + name, "configuration", "worker_args." + name,
                   "src/dgx_monarch/config_schema.py:WorkerArgs." + name,
                   datatype=getattr(datatype, "__forward_arg__", str(datatype)), units=_units(name))
        row["default_rule"] = "omitted: backend policy resolves the effective value; Init widgets take precedence"
        row["validation"] = "dgx_monarch.config_schema.validate_worker_args"
        row["interfaces"]["cli"] = {"status": "available", "representation": "cluster.toml through --config"}
        rows.append(row)
    row = _row("config.fabric", "configuration", "fabric.<profile>", "src/dgx_monarch/config_schema.py:validate_fabric_env",
               datatype="mapping[string,scalar]", default={}, validation="Only validated NCCL_/GLOO_/UCX_ variables; NCCL_PROTO and loader/path injection refused.")
    row["interfaces"]["cli"] = {"status": "available", "representation": "cluster.toml through --config"}
    rows.append(row)
    # load_cluster_config admits [studio] and ignores it; only dgxm studio and
    # the companion's driver hint validate it. Defaults come from that validator.
    studio = dict(config_schema.validate_studio_table({}, Path("cluster.toml")))
    for name, datatype in config_schema.StudioTable.__annotations__.items():
        row = _row("config.studio." + name, "configuration", "studio." + name,
                   "src/dgx_monarch/config_schema.py:validate_studio_table",
                   datatype=getattr(datatype, "__forward_arg__", str(datatype)), default=_json_value(studio[name]),
                   units=_units(name))
        row["validation"] = "dgx_monarch.config_schema.validate_studio_table"
        row["interfaces"]["cli"] = {"status": "available", "representation": "cluster.toml through dgxm studio"}
        rows.append(row)
    return rows


def discover_tui(root: Path) -> list[dict[str, Any]]:
    path = root / "src/dgx_monarch/tui/app.py"
    rows = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Binding":
            key, action, label = [ast.literal_eval(n) for n in node.args[:3]]
            row = _row("tui.action." + action, "tui_action", label,
                       "src/dgx_monarch/tui/app.py:_App.action_" + action)
            row["interfaces"]["tui"] = {"status": "available", "key": key, "action": action}
            row["guard"] = "Presentation action only; does not grant backend mutation authority."
            rows.append(row)
    return rows


def discover_panel(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "web/js").glob("*.js")):
        text = path.read_text()
        for action, event in sorted(set(re.findall(r"\b(\w+)\.(onclick|onchange)\s*=", text))):
            row = _row("panel.action." + path.stem + "." + action + "." + event, "panel_action", action,
                       path.relative_to(root).as_posix() + ":" + action + "." + event)
            row["interfaces"]["panel"] = {"status": "available", "handler": action + "." + event}
            rows.append(row)
    return rows


# tools/gen_templates.py titles each template's Note "DGX Monarch | <title>".
TEMPLATE_TITLE_MARK = "DGX Monarch |"
_NOTE_TYPES = frozenset({"Note", "MarkdownNote"})


def humanized_stem(stem: str) -> str:
    """The fallback title: the file stem without its dgx-monarch- prefix, dashes read as spaces."""
    return stem.removeprefix("dgx-monarch-").replace("-", " ")


def note_title(graph: dict[str, Any]) -> str | None:
    """The text after the template mark in the first root Note title that has some, else None."""
    nodes = graph.get("nodes")
    for node in nodes if isinstance(nodes, list) else []:
        if not isinstance(node, dict) or node.get("type") not in _NOTE_TYPES:
            continue
        title = node.get("title")
        if isinstance(title, str) and title.startswith(TEMPLATE_TITLE_MARK):
            if text := title.removeprefix(TEMPLATE_TITLE_MARK).strip():
                return text
    return None


def workflow_title(graph: dict[str, Any], stem: str) -> str:
    """The readable title Studio lists a bundled template by: its Note title, else its humanized stem.

    The file stem stays the workflow's name for drafts, exports and records.
    """
    return note_title(graph) or humanized_stem(stem)


def discover_workflows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, workflows = [], []
    for path in sorted((root / "example_workflows").glob("*.json")):
        graph = json.loads(path.read_text())
        identity = "workflow." + path.stem
        digest = hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        row = _row(identity, "workflow", path.stem, path.relative_to(root).as_posix(), datatype="comfy_workflow")
        row["digest"] = digest
        row["interfaces"]["nodes"] = {"status": "available", "representation": "ComfyUI graph import"}
        row["validation"] = "Resolve every node type, installed asset and required binding against the live driver before execution."
        rows.append(row)
        workflows.append({"id": identity, "name": path.stem, "title": workflow_title(graph, path.stem),
                          "source": row["source"], "digest": digest, "workflow": graph})
    return rows, workflows


def build_inventory(root: Path) -> dict[str, Any]:
    workflow_rows, _ = discover_workflows(root)
    with discovery_home():
        rows = _discover_cli() + discover_nodes(root) + discover_config() + discover_tui(root) + discover_panel(root) + workflow_rows
    for metric in presentation_contract()["metrics"]:
        row = _row("telemetry." + metric["id"], "telemetry", metric["name"],
                   "src/dgx_monarch/telemetry.py:host_stats / RenderProgress.snapshot",
                   datatype="number", units=metric["unit"], default=None)
        row["metric"] = metric
        row["interfaces"]["panel"] = {"status": "available", "representation": "telemetry panel"}
        row["interfaces"]["tui"] = {"status": "partial", "reason": "Layout and per-host/per-rail parity must be independently verified."}
        row["apply"] = "Read only. Never treat a presentation threshold as runtime admission."
        rows.append(row)
    apply_studio_views(rows, root)
    counts = Counter(row["kind"] for row in rows)
    result = {"schema_version": SCHEMA_VERSION, "pinned_baseline": PINNED_BASELINE,
              # Counted from this checkout's node contract and bundled
              # templates, so a regenerated template set needs no code change.
              "baseline_counts": {"nodes": counts["node"], "workflows": counts["workflow"]},
              "counts": dict(sorted(counts.items())),
              "coverage_meaning": ("Discovered surfaces with a reviewed Studio view: an operation, a route, an equivalent, "
                                   "a waiver with its reason, or source coverage only. Not a claim of accepted parity."),
              "limitations": ["Dynamic catalog choices require the connected ComfyUI object_info schema; no build-machine catalog is frozen.",
                              "Configuration effective defaults depend on backend policy and precedence.",
                              "Studio operation availability is negotiated separately; inventory rows do not authorize execution.",
                              "Cross-interface semantic adoption and hardware acceptance require separate verification."],
              "rows": sorted(rows, key=lambda row: row["id"])}
    validate_inventory(result)
    return result


def validate_inventory(document: dict[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported parity inventory schema")
    seen = set()
    for row in document.get("rows", []):
        if REQUIRED_FIELDS - row.keys():
            raise ValueError(f"Missing inventory fields in {row.get('id')}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate inventory identity: {row['id']}")
        seen.add(row["id"])
        if set(row["interfaces"]) != set(INTERFACES):
            raise ValueError(f"Incomplete interface mapping: {row['id']}")
        if not row["source"] or not row["guard"] or not row["recovery"]:
            raise ValueError(f"Missing ownership contract: {row['id']}")
