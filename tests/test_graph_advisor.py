"""Check queue-time graph advice and failure isolation with API-prompt dicts.

The tests use no server, ComfyUI import, or queued render.
"""
from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

from dgx_monarch import first_render, graph_advisor
from dgx_monarch.constants import MESH_TYPE
from dgx_monarch.log import get_logger
from dgx_monarch.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_notices():
    """The advisor deduplicates per driver process; each test gets a fresh one."""
    first_render.reset()
    yield
    first_render.reset()


def graph(*classes: tuple[str, str]) -> dict:
    return {node_id: {"class_type": class_type, "inputs": {}}
            for node_id, class_type in classes}


STOCK_TEMPLATE = graph(
    ("1", "UNETLoader"),
    ("2", "DualCLIPLoader"),
    ("3", "CLIPTextEncode"),
    ("4", "CLIPTextEncode"),
    ("5", "EmptyLatentImage"),
    ("6", "KSampler"),
    ("7", "VAELoader"),
    ("8", "VAEDecode"),
    ("9", "SaveImage"),
)


def kinds(advisories) -> list[str]:
    return [advisory.kind for advisory in advisories]


def troubleshooting_entry() -> str:
    """docs/TROUBLESHOOTING.md #63 up to the next entry, on one line.

    Collapsed because the entry wraps at 79 columns, and a phrase split across
    a line break is still a phrase the reader sees.
    """
    text = (REPO / "docs" / "TROUBLESHOOTING.md").read_text()
    head = f"\n## {graph_advisor.TROUBLESHOOTING}. "
    body = text.split(head, 1)[1].split("\n## ", 1)[0]
    return " ".join(body.split())


def test_swap_tables_name_real_registered_nodes():
    """graph_advisor's literals copy nodes/__init__.py; pin every row."""
    for stock, monarch in graph_advisor.SWAPS.items():
        assert monarch in NODE_CLASS_MAPPINGS, f"{stock} points at an unknown node"
        assert graph_advisor.DISPLAY_NAMES[monarch] == NODE_DISPLAY_NAME_MAPPINGS[monarch]
    assert set(graph_advisor.DISPLAY_NAMES) == set(graph_advisor.SWAPS.values()) | {
        graph_advisor.INIT_CLASS
    }
    assert graph_advisor.INIT_CLASS in NODE_CLASS_MAPPINGS
    assert not set(graph_advisor.SWAPS) & set(graph_advisor.REBUILDS)


def test_init_is_the_only_node_that_makes_a_mesh():
    """The missing-init advisory has no exceptions only while this holds."""
    mesh_sources = sorted(
        name for name, cls in NODE_CLASS_MAPPINGS.items()
        if MESH_TYPE in getattr(cls, "RETURN_TYPES", ())
    )
    assert mesh_sources == [graph_advisor.INIT_CLASS]


def test_stock_only_graph_gets_one_info_advisory_naming_every_swap():
    advisories = graph_advisor.analyze(STOCK_TEMPLATE)
    assert kinds(advisories) == ["stock_only"]
    advisory = advisories[0]
    assert advisory.severity == "info"
    assert "#1 UNETLoader -> Load Diffusion Model (DGX Monarch)" in advisory.message
    assert "#6 KSampler -> KSampler (DGX Monarch)" in advisory.message
    assert "DGX Monarch Init" in advisory.message
    assert f"docs/TROUBLESHOOTING.md #{graph_advisor.TROUBLESHOOTING}" in advisory.message
    for never_flagged in ("DualCLIPLoader", "CLIPTextEncode", "EmptyLatentImage",
                          "VAELoader", "VAEDecode", "SaveImage"):
        assert never_flagged not in advisory.message


@pytest.mark.parametrize("node_id", ["²", "①", "⑴", "Ⅻ", "½", "٢"])
def test_unicode_numeric_node_ids_never_drop_queue_advice(node_id):
    prompt = graph((node_id, "KSampler"))
    advisories = graph_advisor.analyze(prompt)
    assert kinds(advisories) == ["stock_only"]
    assert f"#{node_id} KSampler" in advisories[0].message
    assert graph_advisor.advise(prompt) == 1


def test_a_checkpoint_loader_is_advised_as_three_separate_loaders():
    advisories = graph_advisor.analyze(
        graph(("1", "CheckpointLoaderSimple"), ("2", "KSampler")))
    message = advisories[0].message
    assert "#1 CheckpointLoaderSimple -> Load Diffusion Model (DGX Monarch)" in message
    assert "stock CLIP and VAE loaders" in message


def test_the_custom_sampler_recipe_carries_the_topology_the_scheduler_demands():
    """Following the toast on the stock Flux template must not hit a refusal."""
    message = graph_advisor.analyze(graph(
        ("1", "UNETLoader"),
        ("7", "BasicScheduler"),
        ("8", "BasicGuider"),
        ("9", "SamplerCustomAdvanced"),
    ))[0].message
    assert (
        "#7 BasicScheduler -> Basic Scheduler (DGX Monarch), which needs an "
        "explicit topology on the Init node"
    ) in message
    assert "ring2" in message


def test_the_scheduler_caveat_matches_the_refusal_the_node_actually_raises():
    """Pin the advice to the code: drop the raise and this test says so."""
    mesh = types.SimpleNamespace(
        handle=object(), topology_preset="auto", attention="TORCH_FLASH",
        sync_ulysses=True, worker_args={})
    model = types.SimpleNamespace(mesh=mesh, request_dict=lambda: {})
    scheduler = NODE_CLASS_MAPPINGS["DGXMonarchBasicScheduler"]()
    with pytest.raises(RuntimeError, match="explicit topology on the Init node"):
        scheduler.get_sigmas(model, "normal", 2, 1.0)
    assert "explicit topology on the Init node" in (
        graph_advisor.CAVEATS["DGXMonarchBasicScheduler"])


def test_a_caveat_only_rides_the_swap_it_belongs_to():
    message = graph_advisor.analyze(graph(("1", "UNETLoader")))[0].message
    assert "explicit topology" not in message
    assert set(graph_advisor.CAVEATS) <= set(graph_advisor.SWAPS.values())


def test_the_missing_init_advisory_carries_the_condition_on_the_node_it_names():
    """That advisory tells you to add the node the widget lives on."""
    message = graph_advisor.analyze(graph(
        ("1", "DGXMonarchUNETLoader"),
        ("7", "DGXMonarchBasicScheduler"),
    ))[0].message
    assert "Basic Scheduler (DGX Monarch) needs an explicit topology" in message
    plain = graph_advisor.analyze(graph(
        ("1", "DGXMonarchUNETLoader"), ("6", "DGXMonarchKSampler")))[0].message
    assert "explicit topology" not in plain


def test_a_stock_graph_with_nothing_to_swap_says_nothing():
    assert graph_advisor.analyze(
        graph(("1", "LoadImage"), ("2", "ImageScale"), ("3", "SaveImage"))) == []


def test_monarch_nodes_without_init_warn_and_name_the_init_node():
    advisories = graph_advisor.analyze(graph(
        ("1", "DGXMonarchUNETLoader"),
        ("2", "DGXMonarchKSampler"),
        ("3", "CLIPTextEncode"),
    ))
    assert kinds(advisories) == ["missing_init"]
    advisory = advisories[0]
    assert advisory.severity == "warn"
    assert "DGX Monarch Init" in advisory.message
    assert "#1 DGXMonarchUNETLoader" in advisory.message
    assert "#2 DGXMonarchKSampler" in advisory.message


def test_missing_init_wins_over_leftover_stock():
    """Leftovers have nowhere to move until the graph has a mesh."""
    advisories = graph_advisor.analyze(graph(
        ("1", "DGXMonarchUNETLoader"), ("2", "KSampler")))
    assert kinds(advisories) == ["missing_init"]


def test_leftover_stock_warns_with_the_exact_node_ids():
    advisories = graph_advisor.analyze(graph(
        ("1", "DGXMonarchInit"),
        ("2", "DGXMonarchUNETLoader"),
        ("3", "KSamplerAdvanced"),
        ("4", "CLIPTextEncode"),
        ("10", "ModelSamplingSD3"),
    ))
    assert kinds(advisories) == ["leftover_stock"]
    advisory = advisories[0]
    assert advisory.severity == "warn"
    assert "#3 KSamplerAdvanced -> KSampler Advanced (DGX Monarch)" in advisory.message
    assert "#10 ModelSamplingSD3 -> Model Sampling SD3 (DGX Monarch)" in advisory.message
    # Numeric ids sort as numbers, so #3 comes before #10.
    assert advisory.message.index("#3 ") < advisory.message.index("#10 ")
    assert "CLIPTextEncode" not in advisory.message


def test_a_complete_monarch_graph_says_nothing():
    assert graph_advisor.analyze(graph(
        ("1", "DGXMonarchInit"),
        ("2", "DGXMonarchUNETLoader"),
        ("3", "DGXMonarchLoraLoader"),
        ("4", "DGXMonarchKSampler"),
        ("5", "CLIPLoader"),
        ("6", "CLIPTextEncode"),
        ("7", "VAELoader"),
        ("8", "VAEDecode"),
        ("9", "SaveImage"),
    )) == []


def test_the_custom_sampler_pair_maps_the_advanced_node_and_rebuilds_the_other():
    advanced = graph_advisor.analyze(graph(
        ("1", "DGXMonarchInit"), ("2", "SamplerCustomAdvanced")))[0]
    assert "#2 SamplerCustomAdvanced -> Sampler Custom (DGX Monarch)" in advanced.message
    plain = graph_advisor.analyze(graph(
        ("1", "DGXMonarchInit"), ("3", "SamplerCustom")))[0]
    assert "#3 SamplerCustom -> CFG Guider (DGX Monarch)" in plain.message


@pytest.mark.parametrize("payload", [
    None,
    {},
    {"prompt": None},
    {"prompt": []},
    {"prompt": {"1": None}},
    {"prompt": {"1": "KSampler"}},
    {"prompt": {"1": {"inputs": {}}}},
    {"prompt": {"1": {"class_type": 7}}},
    {"prompt": {"1": {"class_type": ""}}},
    {"number": 3},
])
def test_a_malformed_payload_advises_nothing_and_queues_unchanged(payload):
    assert graph_advisor.advise(
        payload.get("prompt") if isinstance(payload, dict) else payload) == 0
    assert graph_advisor.on_prompt(payload) is payload


def test_the_handler_returns_the_same_payload_object_it_was_given():
    payload = {"prompt": STOCK_TEMPLATE, "client_id": "abc"}
    assert graph_advisor.on_prompt(payload) is payload
    assert payload["prompt"] == STOCK_TEMPLATE


def test_the_handler_survives_a_payload_whose_lookup_raises():
    class Hostile(dict):
        def get(self, *_args, **_kwargs):
            raise RuntimeError("hostile payload")

    payload = Hostile(prompt=STOCK_TEMPLATE)
    assert graph_advisor.on_prompt(payload) is payload


def test_the_kill_switch_disables_the_analysis_without_a_restart(monkeypatch):
    monkeypatch.setenv(graph_advisor.KILL_SWITCH, "0")
    assert not graph_advisor.enabled()
    assert graph_advisor.analyze(STOCK_TEMPLATE) == []
    payload = {"prompt": STOCK_TEMPLATE}
    assert graph_advisor.on_prompt(payload) is payload
    monkeypatch.setenv(graph_advisor.KILL_SWITCH, "1")
    assert graph_advisor.enabled()
    assert kinds(graph_advisor.analyze(STOCK_TEMPLATE)) == ["stock_only"]


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "off", " Off ", "no"])
def test_every_documented_off_value_turns_the_advisor_off(monkeypatch, value):
    monkeypatch.setenv(graph_advisor.KILL_SWITCH, value)
    assert not graph_advisor.enabled()


@pytest.mark.parametrize("value", ["", " ", "1", "on", "yes", "true", "nope"])
def test_anything_else_leaves_the_advisor_on(monkeypatch, value):
    monkeypatch.setenv(graph_advisor.KILL_SWITCH, value)
    assert graph_advisor.enabled()


def test_the_troubleshooting_entry_names_every_off_value_the_code_accepts():
    entry = troubleshooting_entry()
    for value in graph_advisor._OFF:
        assert f"`{value}`" in entry, f"off value {value!r} is undocumented"


def test_the_same_graph_advises_once_per_process_and_a_changed_one_advises_again():
    assert graph_advisor.advise(STOCK_TEMPLATE) == 1
    assert graph_advisor.advise(STOCK_TEMPLATE) == 0
    assert graph_advisor.advise(dict(reversed(list(STOCK_TEMPLATE.items())))) == 0
    changed = dict(STOCK_TEMPLATE, **graph(("11", "KSamplerAdvanced")))
    assert graph_advisor.advise(changed) == 1


def test_the_fingerprint_covers_the_advisory_class_the_ids_and_the_classes():
    stock_only = graph_advisor.analyze(STOCK_TEMPLATE)[0]
    assert stock_only.fingerprint == graph_advisor.analyze(
        dict(reversed(list(STOCK_TEMPLATE.items()))))[0].fingerprint

    moved = graph(("1", "UNETLoader"), ("99", "KSampler"))
    assert graph_advisor.analyze(moved)[0].fingerprint != stock_only.fingerprint

    swapped_class = graph(("1", "UNETLoader"), ("6", "KSamplerAdvanced"))
    kept_class = graph(("1", "UNETLoader"), ("6", "KSampler"))
    assert (graph_advisor.analyze(swapped_class)[0].fingerprint
            != graph_advisor.analyze(kept_class)[0].fingerprint)

    with_init = graph(("1", "DGXMonarchInit"), ("6", "KSampler"))
    assert (graph_advisor.analyze(with_init)[0].fingerprint
            != graph_advisor.analyze(kept_class)[0].fingerprint)


def test_the_toast_is_not_titled_first_render(monkeypatch):
    """It fires at queue time, on a graph that may never render at all."""
    captured: list[tuple[str, dict]] = []
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(
            send_sync=lambda event, data: captured.append((event, data))))
    monkeypatch.setitem(sys.modules, "server", fake_server)

    assert graph_advisor.advise(STOCK_TEMPLATE) == 1
    payload = captured[0][1]
    assert payload["summary"] == graph_advisor.SUMMARY
    assert payload["summary"] != first_render.DEFAULT_SUMMARY
    assert "first render" not in payload["summary"]
    assert payload["phase"] == graph_advisor.PHASE


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def test_the_driver_log_carries_the_advisory_for_a_headless_driver():
    """No browser, no toast, same sentence: the log is the headless surface."""
    logger = get_logger("dgx_monarch.first_render")
    capture = _Capture()
    logger.addHandler(capture)
    try:
        assert graph_advisor.advise(STOCK_TEMPLATE) == 1
    finally:
        logger.removeHandler(capture)
    assert any("#6 KSampler -> KSampler (DGX Monarch)" in record.getMessage()
               for record in capture.records)


def test_a_long_swap_list_is_capped_instead_of_flooding_the_canvas():
    crowded = graph(*[(str(index), "KSampler") for index in range(1, 15)])
    message = graph_advisor.analyze(crowded)[0].message
    assert "and 6 more" in message


class _FakeServer:
    def __init__(self):
        self.on_prompt_handlers: list = []

    def add_on_prompt_handler(self, handler):
        self.on_prompt_handlers.append(handler)


def test_registration_is_idempotent():
    server = _FakeServer()
    assert graph_advisor.register_on_prompt(server) is True
    assert graph_advisor.register_on_prompt(server) is False
    assert server.on_prompt_handlers == [graph_advisor.on_prompt]


def test_the_troubleshooting_entry_exists_and_matches_the_pointer():
    text = (REPO / "docs" / "TROUBLESHOOTING.md").read_text()
    assert f"\n## {graph_advisor.TROUBLESHOOTING}. " in text
    assert graph_advisor.KILL_SWITCH in text


COUNT_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}


def test_the_troubleshooting_entry_covers_every_rebuild_and_counts_them():
    """A toast line the doc never mentions sends the reader nowhere."""
    entry = troubleshooting_entry()
    for stock in graph_advisor.REBUILDS:
        assert stock in entry, f"{stock} is advised but undocumented"
    word = COUNT_WORDS[len(graph_advisor.REBUILDS)]
    assert f"{word} nodes have no one-for-one" in entry


def test_the_troubleshooting_entry_covers_every_caveat():
    entry = troubleshooting_entry()
    for monarch in graph_advisor.CAVEATS:
        display = graph_advisor.DISPLAY_NAMES[monarch]
        assert display in entry, f"{display}'s condition is undocumented"
    assert "explicit topology on the Init" in entry
