"""The type column of docs/CAMPAIGN_COLUMNS.md, which tools/campaign_stats_index.py infers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _index():
    name = "dgxm_tools_campaign_stats_index"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "campaign_stats_index.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.mark.parametrize(("values", "expected"), [
    # Refusal and probe messages open with a plain tag, not a JSON array.
    (["[dgxm:A guard=capacity waivable=1] the render would not fit", ""], "text"),
    (["[dgxm:C] refused", '["a", "b"]'], "text"),
    (['["a", "b"]', "[]", ""], "JSON array"),
    (['{"k": 1}', "{}"], "JSON object"),
    (['{"k": 1}', "{not json"], "text"),
    (["[1, 2", "[3]"], "text"),
    (["true", "false"], "bool"),
    (["3", "-4"], "int"),
])
def test_a_container_type_needs_every_value_to_parse_as_that_container(values, expected):
    assert _index().kind(values) == expected


def test_the_committed_message_columns_read_as_text():
    text = (REPO / "docs" / "CAMPAIGN_COLUMNS.md").read_text(encoding="utf-8")
    rows = {line.split("|")[1].strip(): line.split("|")[2].strip()
            for line in text.splitlines() if line.startswith("| `legs.")}
    assert rows["`legs.probe.message`"] == "text"
    assert rows["`legs.resident.message`"] == "text"
