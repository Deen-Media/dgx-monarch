"""Inventory coverage is independent of Studio operation availability."""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import pwd
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from dgx_monarch.cli.arguments import build_parser
from dgx_monarch.studio.contracts import TERMINAL_STATES
from dgx_monarch.studio.inventory import (
    DISCOVERY_HOME,
    INTERFACES,
    REQUIRED_FIELDS,
    build_inventory,
    discover_nodes,
    discover_workflows,
    humanized_stem,
    load_inventory,
    validate_inventory,
    workflow_title,
)
from dgx_monarch.studio.inventory_registry import STATUSES
from dgx_monarch.studio.presentation import (
    MEMORY_ANSI256,
    MEMORY_COLORS,
    available_memory_status,
    format_duration_clock,
    format_metric,
    freshness,
    presentation_contract,
)

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = load_inventory()


def _private_home_prefixes() -> list[str]:
    """The build account's home directories, as path prefixes a manifest must not carry.

    HOME=/ gives no prefix: every absolute path starts with "/", and the
    account's password-database home still guards the real directory.
    """
    homes = {str(Path.home()), pwd.getpwuid(os.getuid()).pw_dir}
    return sorted(home.rstrip("/") + "/" for home in homes if home.rstrip("/"))


@pytest.mark.parametrize("row", INVENTORY["rows"], ids=lambda row: row["id"])
def test_every_row_has_source_and_explicit_interface_contract(row):
    assert REQUIRED_FIELDS <= row.keys()
    source_path = row["source"].partition(":")[0]
    assert (ROOT / source_path).is_file()
    assert set(row["interfaces"]) == set(INTERFACES)
    assert all(view.get("status") for view in row["interfaces"].values())
    assert row["guard"] and row["recovery"] and row["preview"] and row["apply"]
    assert row["interfaces"]["studio"]["status"] in STATUSES
    text = json.dumps(row)
    assert "typed advanced inventory" not in text
    assert DISCOVERY_HOME not in text
    assert not any(prefix in text for prefix in _private_home_prefixes())


def test_checked_in_inventory_matches_live_checkout():
    actual = build_inventory(ROOT)
    assert INVENTORY == actual
    # Counted from the checkout, so regenerating or adding a template changes
    # only the generated data, never this test.
    nodes = len(json.loads((ROOT / "tests/node_input_contract.json").read_text()))
    workflows = len(list((ROOT / "example_workflows").glob("*.json")))
    assert actual["counts"]["node"] == actual["baseline_counts"]["nodes"] == nodes
    assert actual["counts"]["workflow"] == actual["baseline_counts"]["workflows"] == workflows
    validate_inventory(actual)


def test_home_relative_defaults_normalize_the_same_under_any_home(tmp_path):
    """HOME=/ used to leave /ComfyUI/output unrewritten and abort collection."""
    program = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from dgx_monarch.studio.inventory import build_inventory, discover_cli\n"
        "rows = {row['id']: row['default'] for row in discover_cli()}\n"
        "print(json.dumps({'cli': [rows['cli.option.gate.report_dir'], rows['cli.option.launcher.destination']],\n"
        "                  'inventory': build_inventory(Path(sys.argv[1]))}))\n"
    )
    results = []
    for home in ("/", str(tmp_path)):
        env = {**os.environ, "HOME": home, "PYTHONPATH": str(ROOT / "src")}
        result = subprocess.run([sys.executable, "-c", program, str(ROOT)], env=env, capture_output=True,
                                text=True, timeout=120, check=False)
        assert result.returncode == 0, result.stderr[-2000:]
        results.append(json.loads(result.stdout))
    for value in results:
        assert value["cli"] == ["~/ComfyUI/output", "~/.local/bin/dgxm"]
        assert value["inventory"] == INVENTORY
    check = subprocess.run([sys.executable, str(ROOT / "tools/gen_studio_inventory.py"), "--check"],
                           env={**os.environ, "HOME": "/", "PYTHONPATH": str(ROOT / "src")},
                           capture_output=True, text=True, timeout=120, check=False)
    assert check.returncode == 0, check.stderr[-2000:]


def test_no_module_keeps_the_discovery_home_after_a_fresh_build():
    """Discovery imports its modules before HOME changes, so no import-time path holds the sentinel."""
    program = (
        "import sys\n"
        "from pathlib import Path\n"
        "from dgx_monarch.studio import inventory\n"
        "inventory.build_inventory(Path(sys.argv[1]))\n"
        "hits = [f'{name}.{attr}' for name, module in list(sys.modules.items()) if name.startswith('dgx_monarch')\n"
        "        for attr, value in vars(module).items() if attr != 'DISCOVERY_HOME'\n"
        "        and isinstance(value, (str, Path)) and inventory.DISCOVERY_HOME in str(value)]\n"
        "print(hits)\n"
    )
    result = subprocess.run([sys.executable, "-c", program, str(ROOT)], capture_output=True, text=True, timeout=120,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=True)
    assert result.stdout.strip() == "[]"


def test_discovery_restores_the_callers_home(monkeypatch):
    monkeypatch.setenv("HOME", "/somewhere/else")
    build_inventory(ROOT)
    assert os.environ["HOME"] == "/somewhere/else"
    monkeypatch.delenv("HOME")
    from dgx_monarch.studio.inventory import discover_cli

    discover_cli()
    assert "HOME" not in os.environ


def test_all_cli_options_and_commands_are_independently_enumerated():
    parser = build_parser(defaultdict(lambda: lambda _args: 0))
    expected_commands, expected_options = set(), set()
    pending = [((), parser)]
    while pending:
        parents, current = pending.pop()
        for action in current._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, child in action.choices.items():
                    path = (*parents, name)
                    expected_commands.add("dgxm " + " ".join(path))
                    pending.append((path, child))
            else:
                option = action.option_strings[-1] if action.option_strings else action.dest
                expected_options.add("dgxm " + " ".join((*parents, option)))
    assert expected_commands == {r["interfaces"]["cli"]["command"] for r in INVENTORY["rows"] if r["kind"] == "cli_command"}
    assert expected_options == {r["interfaces"]["cli"]["command"] for r in INVENTORY["rows"] if r["kind"] == "cli_option"}


def test_the_studio_tree_is_waived_with_one_reason():
    """dgxm studio starts, stops, opens or reaches Studio's own process, so no page operation stands for it."""
    reason = ("Starts, stops, opens or reaches the companion process that serves Studio; "
              "the page cannot run these for itself.")
    rows = {row["id"]: row for row in INVENTORY["rows"] if row["kind"] in {"cli_command", "cli_option"}
            and (row["interfaces"]["cli"]["command"] + " ").startswith("dgxm studio ")}
    commands = {row["interfaces"]["cli"]["command"] for row in rows.values() if row["kind"] == "cli_command"}
    assert commands == {"dgxm studio", "dgxm studio up", "dgxm studio down", "dgxm studio open", "dgxm studio token",
                        "dgxm studio tunnel"}
    assert sum(row["kind"] == "cli_option" for row in rows.values()) == 19
    for row in rows.values():
        assert row["studio_binding"] == {"kind": "companion", "reason": reason}
        assert row["interfaces"]["studio"] == {"status": "not_represented", "reason": reason}
    host = rows["cli.option.studio.tunnel.host"]
    assert (host["option_strings"], host["destination"], host["nargs"]) == ([], "host", "?")
    assert rows["cli.option.studio.up.allow_management"]["destination"] == "up_allow_management"
    for path in ("studio", "studio.up", "studio.tunnel"):
        assert rows[f"cli.option.{path}.config"]["default"] == argparse.SUPPRESS
    assert not any(row["interfaces"]["studio"]["status"] == "not_represented" for row in INVENTORY["rows"]
                   if row["kind"] in {"cli_command", "cli_option"} and row["id"] not in rows)


def test_config_studio_rows_take_their_views():
    from dgx_monarch import config_schema

    rows = {row["id"]: row for row in INVENTORY["rows"] if row["id"].startswith("config.studio.")}
    assert set(rows) == {"config.studio.driver", "config.studio.port", "config.studio.driver_service",
                         "config.studio.driver_command"}
    launch = {"status": "not_represented", "reason":
              "Read by dgxm studio before the companion starts; the page cannot change how its own companion was launched."}
    for key in ("driver", "port", "driver_service"):
        assert rows[f"config.studio.{key}"]["interfaces"]["studio"] == launch
    assert rows["config.studio.driver_command"]["interfaces"]["studio"] == {"status": "equivalent", "equivalence": (
        "The owner selects the cluster configuration with --config at companion launch, or dgxm studio passes it; "
        "Studio does not edit it.")}
    # The config key has no default origin; DEFAULT_STUDIO_DRIVER is only the dgxm-studio --driver default.
    assert config_schema.DEFAULT_STUDIO_PORT == 8220
    assert {key: row["default"] for key, row in rows.items()} == {
        "config.studio.driver": None, "config.studio.port": 8220,
        "config.studio.driver_service": None, "config.studio.driver_command": None}
    for row in rows.values():
        assert row["kind"] == "configuration"
        assert row["source"] == "src/dgx_monarch/config_schema.py:validate_studio_table"
        assert row["validation"] == "dgx_monarch.config_schema.validate_studio_table"
        assert row["interfaces"]["cli"] == {"status": "available", "representation": "cluster.toml through dgxm studio"}


def test_live_node_registry_and_pinned_inputs_are_all_represented():
    tree = ast.parse((ROOT / "src/dgx_monarch/nodes/__init__.py").read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS" for t in n.targets))
    registry = {ast.literal_eval(key) for key in assignment.value.keys}
    assert registry == {row["name"] for row in INVENTORY["rows"] if row["kind"] == "node"}
    expected = json.loads((ROOT / "tests/node_input_contract.json").read_text())
    rows = {row["id"]: row for row in INVENTORY["rows"]}
    for name, sections in expected.items():
        for section, fields in sections.items():
            for position, (field, datatype, default) in enumerate(fields):
                row = rows[f"node.input.{name}.{field}"]
                assert (row["section"], row["order"], row["datatype"]) == (section, position, datatype)
                assert row["default"] == (None if default == "<first-option>" else default)
    expected_count = sum(len(fields) for sections in expected.values() for fields in sections.values())
    assert expected_count == sum(row["kind"] == "node_input" for row in INVENTORY["rows"])


def test_dynamic_comfy_catalog_is_not_replaced_with_test_stub_values():
    dynamic = {"unet_name", "lora_name", "sampler_name", "scheduler"}
    for row in INVENTORY["rows"]:
        if row["kind"] == "node_input" and row["name"] in dynamic:
            assert row["choices"] is None
            assert row["choice_source"]
            assert row["default_rule"] == "first_available_backend_option"
    assert "stub.safetensors" not in json.dumps(INVENTORY)


def test_literal_choices_and_validation_are_not_lost():
    rows = {row["id"]: row for row in INVENTORY["rows"]}
    dtype = rows["node.input.DGXMonarchUNETLoader.weight_dtype"]
    from dgx_monarch.loader_options import SUPPORTED_WEIGHT_DTYPES
    assert dtype["choices"] == list(SUPPORTED_WEIGHT_DTYPES)
    lora = rows["node.input.DGXMonarchLoraLoader.strength_model"]
    assert lora["validation"] == {"min": -100.0, "max": 100.0, "step": 0.01}
    assert "flux3" in rows["node.input.DGXMonarchInit.family_adapter"]["choices"]


def test_node_source_default_drift_fails_discovery(monkeypatch):
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path == ROOT / "src/dgx_monarch/nodes/model_sampling.py":
            return text.replace('"default": 3.0', '"default": 4.0')
        return text

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(ValueError, match="Node default drift"):
        discover_nodes(ROOT)


def test_workflow_payloads_and_digests_match_all_shipped_examples():
    rows, workflows = discover_workflows(ROOT)
    manifest = json.loads((ROOT / "src/dgx_monarch/studio/data/workflows.json").read_text())
    assert manifest["workflows"] == workflows
    assert {r["source"] for r in rows} == {p.relative_to(ROOT).as_posix() for p in (ROOT / "example_workflows").glob("*.json")}
    for workflow in workflows:
        assert workflow["workflow"] == json.loads((ROOT / workflow["source"]).read_text())
        assert len(workflow["digest"]) == 64
        assert set(workflow) == {"id", "name", "title", "source", "digest", "workflow"}


def _note(title=None, kind="Note"):
    node = {"id": 1, "type": kind, "widgets_values": ["text"]}
    return node if title is None else {**node, "title": title}


@pytest.mark.parametrize(("nodes", "expected"), [
    ([_note("DGX Monarch | Kroma local INT8 text to image")], "Kroma local INT8 text to image"),
    ([_note("DGX Monarch |  spaced  ", "MarkdownNote")], "spaced"),
    # The first Note with the mark names the template; an unmarked one is a plain label.
    ([_note("Read me first"), _note("DGX Monarch | Second"), _note("DGX Monarch | Third")], "Second"),
    ([{"id": 2, "type": "KSampler", "title": "DGX Monarch | Not a note"}], "kroma t2i int8"),
    ([_note("Kroma text to image")], "kroma t2i int8"),
    ([_note("DGX Monarch |   ")], "kroma t2i int8"),
    ([_note()], "kroma t2i int8"),
    ([], "kroma t2i int8"),
    ("not a list", "kroma t2i int8"),
])
def test_workflow_title_reads_the_marked_note_or_falls_back_to_the_stem(nodes, expected):
    assert workflow_title({"nodes": nodes}, "dgx-monarch-kroma-t2i-int8") == expected


def test_humanized_stem_drops_only_the_leading_prefix():
    assert humanized_stem("dgx-monarch-wan-animate2-base-quality") == "wan animate2 base quality"
    assert humanized_stem("custom-dgx-monarch-graph") == "custom dgx monarch graph"


def test_panel_handlers_are_not_silently_omitted():
    expected = set()
    for path in (ROOT / "web/js").glob("*.js"):
        for match in re.finditer(r"\b(\w+)\.(onclick|onchange)\s*=", path.read_text()):
            expected.add(path.relative_to(ROOT).as_posix() + ":" + match.group(1) + "." + match.group(2))
    assert expected == {r["source"] for r in INVENTORY["rows"] if r["kind"] == "panel_action"}


def test_missing_mappings_and_duplicate_identity_are_refused():
    doc = load_inventory()
    doc["rows"].append(doc["rows"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_inventory(doc)
    doc = load_inventory()
    del doc["rows"][0]["interfaces"]["tui"]
    with pytest.raises(ValueError, match="Incomplete interface"):
        validate_inventory(doc)


def test_job_states_use_the_approved_status_tokens():
    contract = presentation_contract()
    assert contract["job_states"] == {
        "submitting": "rendering", "accepted": "rendering", "queued": "rendering", "running": "rendering",
        "cancel_requested": "degraded", "cancelled": "degraded", "succeeded": "ready",
        "failed": "blocked", "unknown": "blocked"}
    colors = {state: contract["status"][token]["hex"] for state, token in contract["job_states"].items()}
    assert {colors[state] for state in ("queued", "running", "succeeded")} == {"#4ade80"}
    assert colors["cancelled"] == colors["cancel_requested"] == "#facc15"
    assert colors["failed"] == colors["unknown"] == "#f87171"
    # Every state the companion stores has a token: terminal ones and those the snapshot keeps first.
    assert set(contract["job_states"]) >= set(TERMINAL_STATES) | {
        "submitting", "accepted", "queued", "running", "cancel_requested", "unknown"}


@pytest.mark.parametrize(("value", "text"), [
    (0, "0:00"), (0.999, "0:00"), (42.9, "0:42"), (60, "1:00"), (73.4, "1:13"), (3599.99, "59:59"),
    (3661, "61:01"), (2**53 - 1, "150119987579016:31"), (-0.0, "0:00"),
    (-1, "unknown"), (None, "unknown"), ("12", "unknown"), (True, "unknown"), (math.nan, "unknown"),
    (math.inf, "unknown"), (2**53, "unknown")])
def test_duration_clock_floors_whole_seconds_with_uncapped_minutes(value, text):
    assert format_duration_clock(value) == text
    rule = presentation_contract()["duration_clock"]
    assert rule == {"format": "m:ss", "source_unit": "s", "rounding": "floor", "minutes": "uncapped",
                    "maximum_s": 2**53 - 1, "missing": "unknown"}
    # The seconds rule for metric durations stays as it was.
    assert format_metric(73.4, "s") == "73.40 s"


def test_generated_presentation_contract_has_exact_approved_colors():
    manifest = json.loads((ROOT / "src/dgx_monarch/studio/data/presentation.json").read_text())
    assert manifest == presentation_contract()
    assert list(MEMORY_COLORS.values()) == ["#22d3ee", "#34d399", "#fb923c", "#c084fc", "#64748b", "#facc15"]
    assert list(MEMORY_ANSI256.values()) == [45, 79, 208, 141, 244, 220]
    assert manifest["free_memory_color"] is None
    assert not manifest["memory_overlap"]["stack_raw_counters"]
    assert manifest["thresholds"]["available_memory_gib"]["authority"] == "presentation_only"
    assert manifest["rail_rate"]["directions"] == ["rx", "tx"]


@pytest.mark.parametrize("value", [None, "12", True, math.nan, math.inf, -math.inf])
def test_unavailable_numeric_values_never_become_zero(value):
    assert format_metric(value, "GiB") == "unknown"
    assert available_memory_status(value) == "unknown"


def test_memory_and_rail_conversions_and_display_boundary():
    assert format_metric(2**30, "GiB", source_units=True) == "1.0 GiB"
    assert format_metric(10**9, "GB/s", source_units=True) == "1.00 GB/s"
    assert format_metric(1750, "GHz", source_units=True) == "1.75 GHz"
    assert format_metric(3.14159, "s") == "3.14 s"
    assert available_memory_status(20) == "degraded"
    assert available_memory_status(20.01) == "ready"
    assert freshness(100, 102) == "fresh"
    assert freshness(100, 104) == "stale"
    assert freshness(105, 104) == "unknown"
