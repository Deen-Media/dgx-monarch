"""nodes/loader_graph: a guide encode and the render activations as alternatives.

Issue #187. ComfyUI with DynamicVRAM empties its device pool after every node
(execution.py, the ``finally`` around each node; the seam contract "per-node
pool reset" holds it), so a guide that encodes before every sampler has given
its memory back when rank 0 samples. Where the graph proves that order, the
loader charges the larger of the two; anywhere it cannot, it keeps the sum and
says why. CPU only: the pool reset and ComfyUI's node registry are stubbed at
the two functions that read them, and the rig is the loader-site suite's.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from dgx_monarch import driver_footprint, h3_activation, h3_rows, mesh_safety
from dgx_monarch.constants import GUIDER_TYPE, MODEL_TYPE
from dgx_monarch.nodes import graph_load_price, loader_graph, loader_preflight
from test_loader_site_preflight import (  # noqa: F401  (_rig is the autouse rig)
    H3,
    REF2VA_INT8_PRUNED,
    _graph,
    _Mesh,
    _refuse,
    _rig,
)

_GIB = 2 ** 30

# The registry answers for the classes these graphs use: what each returns.
_OUTPUTS = {
    "DGXMonarchKSampler": ("LATENT",),
    "DGXMonarchKSamplerPipeline": ("LATENT",),
    "DGXMonarchFleetKSampler": ("LATENT",),
    "DGXMonarchSamplerCustom": ("LATENT", "LATENT"),
    "DGXMonarchIdentityGate": ("LATENT", "STRING"),
    "DGXMonarchCFGGuider": (GUIDER_TYPE,),
    "DGXMonarchDualModelGuider": (GUIDER_TYPE,),
    "DGXMonarchBasicScheduler": ("SIGMAS",),
    "DGXMonarchModelSamplingMiniMaxH3": (MODEL_TYPE,),
    "ModelSamplingSD3": ("MODEL",),
    "SomeoneElsesModelPreview": (MODEL_TYPE, "IMAGE"),
}


@pytest.fixture
def pool_reset(monkeypatch):
    """A driver whose ComfyUI returns the pool after every node, and a registry."""
    monkeypatch.setattr(loader_graph, "comfy_returns_pool_after_each_node", lambda: True)
    monkeypatch.setattr(loader_graph, "_class_outputs", lambda cls: _OUTPUTS.get(cls))
    monkeypatch.setenv(h3_activation.GUIDE_FRAMES_ENV, "1")      # a still, 14 GiB
    yield


def _still_graph(length: int = 5, width: int = 1344, height: int = 768) -> dict:
    graph = _graph(dit=REF2VA_INT8_PRUNED, refs="guide", width=width, height=height)
    graph["8"]["inputs"]["length"] = length
    return graph


def _terms(graph: dict):
    return loader_graph.driver_stack_terms(graph, family=H3, sp_degree=2)


# The rule itself.

def test_alternatives_charge_the_larger_term_and_the_sum_charges_both():
    base = loader_graph.DriverStackTerms(guide_bytes=14 * _GIB, activation_bytes=25 * _GIB,
                                         resolved=True)
    assert base.total == 39 * _GIB
    assert replace(base, guide_first=True).total == 25 * _GIB
    assert replace(base, guide_first=True, guide_bytes=48 * _GIB).total == 48 * _GIB
    alone = replace(base, guide_first=True, activation_bytes=0)
    assert alone.total == replace(alone, guide_first=False).total == 14 * _GIB


def _settled_fits(terms, *, available_gib: float, place_gib: float) -> bool:
    shape = loader_preflight._Shape(
        profile=driver_footprint.DRIVER_STACK_FAMILIES[H3],
        mem_available=int(available_gib * _GIB), weight_bytes=int(place_gib * _GIB),
        weights_resident=False, co_resident=True,
        floor_reserve=driver_footprint.reserve_bytes(None), stack_bytes=terms.total)
    return all(map(graph_load_price.fits_box, loader_preflight._windows(shape, slab=True)))


def test_the_reported_figures_are_admitted_as_alternatives_and_refused_as_a_sum():
    """92.6 GiB available, 18.9 to place, a 14.0 GiB still guide and 1.3.0's
    59.9 GiB of activations: 5.0 + 18.9 + 59.9 = 83.8 fits, 97.8 does not."""
    terms = loader_graph.DriverStackTerms(
        guide_bytes=int(14.0 * _GIB), activation_bytes=int(59.9 * _GIB), resolved=True,
        credited=True, credit_reason="comfy already holds loaded models")
    assert _settled_fits(replace(terms, guide_first=True), available_gib=92.6, place_gib=18.9)
    assert not _settled_fits(terms, available_gib=92.6, place_gib=18.9)


def test_alternatives_still_refuse_what_the_larger_term_alone_does_not_fit():
    """70.0 GiB of activations: 5.0 + 18.9 + 70.0 = 93.9 against 92.6."""
    terms = loader_graph.DriverStackTerms(
        guide_bytes=int(14.0 * _GIB), activation_bytes=int(70.0 * _GIB), resolved=True,
        credited=True, guide_first=True)
    assert not _settled_fits(terms, available_gib=92.6, place_gib=18.9)


def test_the_reported_graph_is_priced_on_this_release_fit_and_admitted_as_a_sum(monkeypatch):
    """896x1152 at 362 frames on uly2: 109,062 declared rows, 25.0 GiB on dev's
    fit where 1.3.0 charged 59.9, so the sum already fits at their reading."""
    monkeypatch.setenv(h3_activation.GUIDE_FRAMES_ENV, "1")
    graph = _still_graph(length=362, width=896, height=1152)
    rows = h3_rows.rows_from_prompt_graph(graph)
    assert (rows.video, rows.audio, rows.total) == (107 * 1008, 1206, 109_062)
    terms = _terms(graph)
    assert terms.activation_bytes == 11 * _GIB + 54_531 * 129_024 + 109_062 * 73_728
    assert f"{terms.activation_bytes / _GIB:.1f}" == "25.0"
    assert terms.guide_bytes == 14 * _GIB and not terms.guide_first
    held = replace(terms, te_bytes=0, vae_bytes=0, credited=True)
    assert _settled_fits(held, available_gib=92.6, place_gib=18.9)


# The order proof: max only where every guide encodes before every sampler.

def test_one_guide_feeding_the_one_sampler_is_charged_as_an_alternative(pool_reset):
    terms = _terms(_still_graph())
    assert terms.guide_first
    assert terms.guide_order_note == loader_graph.GUIDE_FIRST
    assert terms.total == terms.models_bytes + max(terms.guide_bytes, terms.activation_bytes)
    assert ("only the larger of this and the render activations is charged, here this "
            "guide encode, since it encodes and ComfyUI returns its memory before any "
            "sampler runs") in terms.guide_note


def test_the_guide_line_names_the_activations_when_they_are_the_charged_term(pool_reset):
    terms = _terms(_still_graph(length=362))
    assert terms.guide_first and terms.activation_bytes > terms.guide_bytes
    assert "is charged, here the 25.0 GiB of render activations, since" in terms.guide_note


def test_a_driver_without_dynamic_vram_keeps_the_sum(pool_reset, monkeypatch):
    monkeypatch.setattr(loader_graph, "comfy_returns_pool_after_each_node", lambda: False)
    terms = _terms(_still_graph())
    assert not terms.guide_first
    assert terms.guide_order_note == loader_graph.NO_POOL_RESET
    assert terms.total == terms.models_bytes + terms.guide_bytes + terms.activation_bytes
    assert "charged beside the render activations because" in terms.guide_note


def test_with_no_comfyui_in_the_process_the_pool_reset_reads_as_absent(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "comfy.memory_management", None)
    assert loader_graph.comfy_returns_pool_after_each_node() is False


def _first_pass(graph: dict) -> dict:
    """A first sampler on the same model, ahead of the guide's own."""
    graph["40"] = {"class_type": "DGXMonarchKSampler",
                   "inputs": {"model": ["2", 0], "positive": ["6", 0],
                              "latent_image": ["8", 0], "seed": 1}}
    return graph


def test_a_guide_anchored_into_a_first_pass_latent_keeps_the_sum(pool_reset):
    graph = _first_pass(_still_graph())
    graph["10"]["inputs"]["latent"] = ["40", 0]
    terms = _terms(graph)
    assert not terms.guide_first
    assert terms.guide_order_note == loader_graph.MODEL_BEFORE_GUIDE


def test_a_guide_image_decoded_from_a_first_pass_keeps_the_sum(pool_reset):
    graph = _first_pass(_still_graph())
    graph["41"] = {"class_type": "VAEDecode", "inputs": {"samples": ["40", 0], "vae": ["4", 0]}}
    graph["10"]["inputs"]["image"] = ["41", 0]
    terms = _terms(graph)
    assert not terms.guide_first
    assert terms.guide_order_note == loader_graph.MODEL_BEFORE_GUIDE


def test_a_second_sampler_the_guide_does_not_feed_keeps_the_sum(pool_reset):
    """ComfyUI's order between independent branches is not fixed: the other
    sampler may run first, and the guide then encodes beside its pool."""
    terms = _terms(_first_pass(_still_graph()))
    assert not terms.guide_first
    assert terms.guide_order_note == loader_graph.SAMPLER_BEFORE_GUIDE


def _custom_sampler(graph: dict, guider: dict) -> dict:
    """Sampler Custom through a guider, its sigmas from the model's scheduler."""
    del graph["9"]
    graph["44"] = {"class_type": "DGXMonarchBasicScheduler",
                   "inputs": {"model": ["2", 0], "steps": 6}}
    graph["45"] = guider
    graph["43"] = {"class_type": "DGXMonarchSamplerCustom",
                   "inputs": {"noise": ["46", 0], "guider": ["45", 0], "sampler": ["47", 0],
                              "sigmas": ["44", 0], "latent_image": ["8", 0]}}
    return graph


def test_sampler_custom_through_a_guider_is_charged_as_an_alternative(pool_reset):
    graph = _custom_sampler(_still_graph(), {
        "class_type": "DGXMonarchCFGGuider",
        "inputs": {"model": ["2", 0], "positive": ["10", 0], "cfg": 1.0}})
    assert _terms(graph).guide_first


def test_a_dual_model_guider_is_charged_as_an_alternative(pool_reset):
    graph = _custom_sampler(_still_graph(), {
        "class_type": "DGXMonarchDualModelGuider",
        "inputs": {"model": ["2", 0], "model_negative": ["48", 0],
                   "positive": ["10", 0], "cfg": 4.0}})
    graph["48"] = {"class_type": "DGXMonarchUncondUNETLoader",
                   "inputs": {"mesh": ["1", 0], "unet_name": REF2VA_INT8_PRUNED}}
    assert _terms(graph).guide_first


def test_a_model_patch_on_the_model_branch_does_not_count_as_a_sampler(pool_reset):
    graph = _still_graph()
    graph["49"] = {"class_type": "DGXMonarchModelSamplingMiniMaxH3",
                   "inputs": {"model": ["2", 0], "video_shift": 12.0}}
    graph["9"]["inputs"]["model"] = ["49", 0]
    assert _terms(graph).guide_first


def test_a_class_the_registry_cannot_place_counts_as_a_sampler(pool_reset):
    """With a linked model and no registry answer, it may sample before the guide."""
    graph = _still_graph()
    graph["49"] = {"class_type": "SomeoneElsesModelNode", "inputs": {"model": ["2", 0]}}
    graph["9"]["inputs"]["model"] = ["49", 0]
    terms = _terms(graph)
    assert not terms.guide_first
    assert terms.guide_order_note == loader_graph.SAMPLER_BEFORE_GUIDE


def test_an_identity_gate_the_guide_does_not_feed_keeps_the_sum(pool_reset):
    """The gate renders proof renders on the Workers, and it returns a latent."""
    graph = _still_graph()
    graph["52"] = {"class_type": "DGXMonarchIdentityGate",
                   "inputs": {"model": ["2", 0], "positive": ["6", 0], "latent_image": ["8", 0]}}
    terms = _terms(graph)
    assert not terms.guide_first
    assert terms.guide_order_note == loader_graph.SAMPLER_BEFORE_GUIDE


@pytest.mark.parametrize("cls,first", [
    ("ModelSamplingSD3", True),             # stock's patched model: renders nothing
    ("SomeoneElsesModelPreview", False),    # a model and an image: it may render
])
def test_only_outputs_that_carry_no_render_clear_a_model_node(pool_reset, cls, first):
    graph = _still_graph()
    graph["49"] = {"class_type": cls, "inputs": {"model": ["2", 0]}}
    graph["9"]["inputs"]["model"] = ["49", 0] if first else ["2", 0]
    assert _terms(graph).guide_first is first


def test_an_admitted_alternative_charge_logs_what_the_sum_would_have_said(
        pool_reset, monkeypatch, caplog):
    """At 66 GiB the still guide graph fits as alternatives and not as a sum."""
    import logging

    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 66 * _GIB)
    with caplog.at_level(logging.INFO, logger="dgx_monarch.nodes.loader_preflight"):
        assert loader_preflight.preflight_loader_footprint(
            _Mesh(), REF2VA_INT8_PRUNED, {}, _still_graph()) is None
    line = "\n".join(record.getMessage() for record in caplog.records)
    assert "admitted with its 14.0 GiB guide encode and 11.3 GiB of render activations" in line
    assert "charged together they would have refused it" in line


def test_a_guide_behind_an_undecided_switch_keeps_the_sum(pool_reset):
    graph = _still_graph()
    graph["50"] = {"class_type": "PrimitiveBoolean", "inputs": {"value": True}}
    graph["51"] = {"class_type": "ComfySwitchNode",
                   "inputs": {"switch": ["50", 0], "on_true": ["10", 0], "on_false": ["6", 0]}}
    graph["9"]["inputs"]["positive"] = ["51", 0]
    terms = _terms(graph)
    assert not terms.guide_first
    assert terms.guide_order_note == loader_graph.SAMPLER_BEFORE_GUIDE


@pytest.mark.parametrize("sampler", ["DGXMonarchKSamplerPipeline", "DGXMonarchFleetKSampler"])
def test_samplers_whose_renders_overlap_keep_the_sum(pool_reset, sampler):
    graph = _still_graph()
    graph["9"]["class_type"] = sampler
    terms = _terms(graph)
    assert not terms.guide_first
    assert terms.guide_order_note == loader_graph._OVERLAPPING_SAMPLERS[sampler]


def test_an_order_the_proof_cannot_read_keeps_the_sum(pool_reset, monkeypatch):
    def broken(_cls):
        raise RuntimeError("registry fault")

    monkeypatch.setattr(loader_graph, "_class_outputs", broken)
    terms = _terms(_still_graph())
    assert not terms.guide_first
    assert terms.guide_order_note == loader_graph.ORDER_UNREAD


def test_with_no_activation_charge_the_order_is_not_read(pool_reset, monkeypatch):
    monkeypatch.setenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, "1")
    monkeypatch.setattr(h3_activation, "_DISABLE_LOGGED", True)
    terms = _terms(_still_graph())
    assert terms.activation_bytes == 0 and terms.guide_bytes == 14 * _GIB
    assert not terms.guide_first and terms.guide_order_note == ""
    assert "render activations" not in terms.guide_note


# The words.

def test_the_budget_clause_names_the_larger_term_and_the_order_that_allows_it():
    terms = loader_graph.DriverStackTerms(
        guide_bytes=14 * _GIB, activation_bytes=int(25.04 * _GIB), resolved=True,
        guide_first=True, guide_order_note=loader_graph.GUIDE_FIRST)
    assert terms.budget_clause == (
        "25.0 GiB of render activations this graph has not spent yet; its 14.0 GiB "
        "guide encode runs first and ComfyUI returns that memory before sampling, so "
        "only the larger of the two is charged")
    clip = replace(terms, guide_bytes=48 * _GIB, activation_bytes=12 * _GIB)
    assert clip.budget_clause == (
        "48.0 GiB of guide encode this graph has not spent yet; its 12.0 GiB of render "
        "activations come after ComfyUI returns the encode's memory, so only the larger "
        "of the two is charged")


def test_the_budget_clause_says_why_both_are_charged():
    terms = loader_graph.DriverStackTerms(
        guide_bytes=14 * _GIB, activation_bytes=25 * _GIB, resolved=True, credited=True,
        credit_reason="comfy already holds loaded models",
        guide_order_note=loader_graph.SAMPLER_BEFORE_GUIDE)
    assert terms.budget_clause == (
        "39.0 GiB of guide encode and render activations this graph has not spent yet, "
        "with the text encoder, VAE and reference encode charges credited (comfy already "
        "holds loaded models); both are charged because a sampler in this graph can run "
        "before a guide encodes")


def test_a_guide_bound_refusal_names_the_alternative_charge(pool_reset, monkeypatch):
    """A clip at 1344x768 is charged 94 GiB and still refuses as the larger term."""
    monkeypatch.delenv(h3_activation.GUIDE_FRAMES_ENV)
    graph = _still_graph()
    message = _refuse(_Mesh(), REF2VA_INT8_PRUNED, prompt=graph)
    assert "guide encode 94.0 GiB" in message
    assert "of driver stack and guide encode this graph has not spent yet" in message
    assert "of render activations come after ComfyUI returns the encode's memory" in message
    assert "guide encode runs first" not in message
    assert "is charged, here this guide encode" in message
