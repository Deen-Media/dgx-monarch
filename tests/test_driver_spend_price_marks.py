"""The loader's per-term marks, written by the real loader price on its own rig.

``driver_spend_price`` steps aside for a spend the admitting loader charged at
a figure at least its own, and prices every other spend live. These tests run
``preflight_loader_footprint`` at the loader site with an executing prompt id
and read what it recorded: the guide at the room the loader held for the
encode's window, the reference encode at its 40 GiB even under the held-model
credit (it is a transient, as the guide is), and the driver's load nodes only
where no held-model credit took them. CPU only.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from dgx_monarch import (
    driver_spend_marks,
    driver_spend_price,
    driver_unload,
    h3_activation,
    mesh_safety,
    vae_decode_relief,
)
from dgx_monarch.nodes import graph_load_price, loader_graph, loader_preflight
from test_guide_alternative_charge import _still_graph, pool_reset  # noqa: F401  (pool_reset: the fixture)
from test_loader_site_preflight import (  # noqa: F401  (_rig is the autouse rig)
    REF2VA_INT8_PRUNED,
    _graph,
    _Mesh,
    _rig,
    _write_image,
)

GIB = 2 ** 30
PROMPT = "a1b2-prompt"


@pytest.fixture
def admitted(monkeypatch):
    """Admit a graph at the loader site, roomy, in an executing prompt; return what was recorded."""
    monkeypatch.setattr(driver_spend_marks, "_CHARGED", {})
    monkeypatch.setattr(graph_load_price, "executing_prompt_id", lambda: PROMPT)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 120 * GIB)

    def admit(graph: dict, preset: str = "uly2") -> None:
        loader_preflight.reset_memos()
        assert loader_preflight.preflight_loader_footprint(
            _Mesh(topology_preset=preset), REF2VA_INT8_PRUNED, {}, graph) is None

    return admit


@pytest.mark.parametrize("preset", ["uly2", "auto"])
def test_a_still_guide_is_marked_at_the_room_its_encode_window_held(pool_reset, admitted, preset):  # noqa: F811
    graph = _still_graph()
    admitted(graph, preset)
    terms = loader_graph.driver_stack_terms(graph, family=h3_activation.H3_FAMILY, sp_degree=2)
    marked = driver_spend_price.charged(PROMPT, driver_spend_price.GUIDE_TERM)
    assert marked >= h3_activation.guide_encode_bytes(1344 * 768, 1)
    assert marked >= terms.guide_bytes


def test_a_clip_is_marked_at_its_sum_which_covers_the_wraps_margin(pool_reset, admitted, tmp_path):  # noqa: F811
    _write_image(tmp_path, "clip.png", 6)
    admitted(_still_graph(width=448, height=256, image_file="clip.png"))
    own = (h3_activation.guide_encode_bytes(448 * 256, driver_spend_price.guide_frames(6))
           + driver_spend_price.CLIP_MARGIN_BYTES)
    assert driver_spend_price.charged(PROMPT, driver_spend_price.GUIDE_TERM) >= own


def test_the_reference_encode_survives_the_held_model_credit_and_is_marked(admitted, monkeypatch):
    """A first queue of a video-reference graph with ComfyUI's models held: before 1.5.0 the credit
    charged its 40 GiB encode at 0; now it is charged and marked, and the loads are not."""
    monkeypatch.setattr(loader_graph, "_comfy_has_loaded_models", lambda: True)
    monkeypatch.setattr(loader_graph, "_comfy_held_bytes", lambda: 10 ** 13)
    graph = _graph(dit=REF2VA_INT8_PRUNED, refs="video", width=448, height=256)
    terms = loader_graph.credited_terms(loader_graph.driver_stack_terms(graph))
    assert terms.credited and terms.ref_bytes == loader_graph.REF_VIDEO_ENCODE_BYTES
    assert terms.te_bytes == terms.vae_bytes == 0
    admitted(graph)
    assert driver_spend_price.charged(PROMPT, driver_spend_price.REFERENCE_TERM) == 40 * GIB
    loads = [key for key in driver_spend_marks._CHARGED if "\x00node\x00" in key]
    assert loads == [], "a held-model credit took the loads, so none may be marked"


def test_the_earlier_charge_credit_keeps_the_reference_encode_too():
    terms = loader_graph.DriverStackTerms(resolved=True, te_bytes=5 * GIB, ref_bytes=40 * GIB,
                                          te_artifacts=("te.safetensors",))
    loader_graph.reset_charged_artifacts()
    loader_graph.note_stack_charged(terms)
    try:
        credited = loader_graph.credited_terms(terms)
        assert credited.credited and credited.te_bytes == 0 and credited.ref_bytes == 40 * GIB
    finally:
        loader_graph.reset_charged_artifacts()


def test_the_loads_are_marked_per_node_where_no_credit_took_them(pool_reset, admitted):  # noqa: F811
    graph = _still_graph()
    admitted(graph)
    loads = {key.split("\x00")[-1]: figure for key, figure in driver_spend_marks._CHARGED.items()
             if "\x00node\x00" in key}
    expected = {node_id: driver_spend_price.load_bytes(node["inputs"], node["class_type"])
                for node_id, node in graph.items() if not node["class_type"].startswith("DGXMonarch")
                and driver_spend_price.load_bytes(node["inputs"], node["class_type"])}
    assert loads == expected and expected, "the graph's text encoder must be marked at the loader's charge"


def test_a_sampler_site_price_marks_nothing(pool_reset, monkeypatch):  # noqa: F811
    monkeypatch.setattr(driver_spend_marks, "_CHARGED", {})
    monkeypatch.setattr(graph_load_price, "executing_prompt_id", lambda: PROMPT)
    terms = replace(loader_graph.driver_stack_terms(_still_graph(), family=h3_activation.H3_FAMILY,
                                                    sp_degree=2), resolved=True)
    graph_load_price.note_admitted(terms, graph_load_price.SITE_SAMPLER, _still_graph())
    assert driver_spend_marks._CHARGED == {}


def _loader_refuses(graph: dict, preset: str) -> bool:
    loader_preflight.reset_memos()
    try:
        loader_preflight.preflight_loader_footprint(_Mesh(topology_preset=preset), REF2VA_INT8_PRUNED, {}, graph)
    except Exception:
        return True
    return False


def _latent(width: int, height: int):
    return {"samples": SimpleNamespace(tensors=[SimpleNamespace(shape=(1, 24, 31, height // 16, width // 16))])}


@pytest.mark.parametrize("preset", ["auto", "uly2"])
def test_spends_first_and_the_loader_last_in_one_prompt(pool_reset, monkeypatch, tmp_path, preset):  # noqa: F811
    """P2a's order on a fresh driver (2026-11-13): the text encoder loads and Add Guide encodes, both priced
    live and unmarked, then the loader node runs LAST and prices the whole graph. At every reading a live
    refusal at either spend is one the loader makes of the same graph at that reading, with nothing held."""
    _write_image(tmp_path, "clip5.png", 5)
    graph = _still_graph(width=448, height=256, image_file="clip5.png")
    monkeypatch.setattr(driver_spend_marks, "_CHARGED", {})
    monkeypatch.setattr(graph_load_price, "executing_prompt_id", lambda: PROMPT)
    monkeypatch.setattr(driver_spend_price, "_pool_free_bytes", lambda: 0)
    monkeypatch.setattr(driver_unload, "comfy_model_management", lambda: None)
    monkeypatch.setattr(driver_unload, "busy_reason", lambda handle: "")
    monkeypatch.setattr(vae_decode_relief, "_serving", lambda: (None, vae_decode_relief.NO_FLEET))
    te_name = next(node["inputs"]["clip_name"] for node in graph.values() if node["class_type"] == "CLIPLoader")
    te = driver_spend_price._load_spend({"clip_name": te_name}, "CLIPLoader")(graph, "3")
    guide = driver_spend_price._guide_spend(
        {"image": SimpleNamespace(shape=(5, 32, 32, 3)), "latent": _latent(448, 256)})(graph, "10")
    assert te is not None and guide is not None
    assert guide.own == 17 * GIB + driver_spend_price.CLIP_MARGIN_BYTES
    refused = []
    for gib in (10, 15, 20, 22, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100):
        monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda value=gib * GIB: value)
        live = False
        for spend in (te, guide):
            try:
                driver_spend_price.check(spend, PROMPT)
            except driver_spend_price.DriverSpendCapacityError:
                live = True
        if live:
            refused.append(gib)
            assert _loader_refuses(graph, preset), f"at {gib} GiB a spend refuses that the loader admits"
    assert refused, "the sweep must read the refusing side"
