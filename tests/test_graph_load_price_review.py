"""The review of PR #145 (2026-11-05): one test per finding, each failing on a26122b.

The rule the review's first finding sets: a refusal never names a remedy the
same graph would then refuse for another reason. It names what fits, with the
layout it needs, or nothing. The other findings cover the credit for models
ComfyUI holds, every path that forces stock, the merge with #142's Old model
fields, the DiT ceiling, the card's figures, MODELS, the gate remedy, the
refusal's length, the output nodes ComfyUI runs, the Identity Gate node and
the figures a fresh driver now quotes.

CPU only, over the rig in tests/test_graph_load_price.py.
"""
from __future__ import annotations

import inspect
import re
import sys
import types
from pathlib import Path

import pytest

from dgx_monarch import consent_pending, driver_footprint, mesh_safety, old_model
from dgx_monarch.nodes import (
    gate_node,
    graph_live,
    graph_load_price,
    loader_graph,
    loader_preflight,
    old_model_clear,
    refusal_lead,
    render_submit,
)
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag, refusal
from test_graph_load_price import (  # noqa: F401  # the rig is an autouse fixture
    _GIB,
    DIT,
    SIZES,
    UPSAMPLER,
    _Mesh,
    _pair,
    _rig,
    _template,
    _worker_stock_refusal,
    _write_safetensors,
)

ROOT = Path(__file__).resolve().parents[1]
W4A8 = "p-h3-image_w4a8.safetensors"
FP8 = "p-h3-image_fp8.safetensors"
KROMOSS = "kromoss_bf16.safetensors"
KROMOSS_INT8 = "kromoss_int8_convrot.safetensors"


def _refuse(mesh, graph, unet=DIT) -> str:
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, unet, {}, graph)
    return str(raised.value)


def _kromoss(lora: bool) -> dict:
    """Kromoss t2i as its turbo template wires it, with or without the DMD LoRA."""
    graph = {
        "1": {"class_type": "DGXMonarchInit", "inputs": {"topology": "auto"}},
        "2": {"class_type": "DGXMonarchUNETLoader",
              "inputs": {"mesh": ["1", 0], "unet_name": KROMOSS, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3.5_9b_bf16.safetensors", "type": "moss_image"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "a boat"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae_small.safetensors"}},
        "6": {"class_type": "DGXMonarchKSampler",
              "inputs": {"model": ["2", 0], "positive": ["4", 0], "negative": ["4", 0]}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["5", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0]}},
    }
    if lora:
        graph["9"] = {"class_type": "DGXMonarchLoraLoader",
                      "inputs": {"model": ["2", 0], "lora_name": "kromoss_8-steps-dmd-lora.safetensors",
                                 "strength_model": 1.0}}
        graph["6"]["inputs"]["model"] = ["9", 0]
    return graph


# --- 1. what would fit never names a file the graph would then refuse -------

def test_a_twin_the_fsdp_launch_refuses_is_never_named(monkeypatch):
    """W4A8 refuses FSDP at launch, class P (cell 6 w4a8); on uly2+fsdp at 58 GiB
    a26122b named it as the file that would fit."""
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 58 * _GIB)
    text = _refuse(_pair("uly2+fsdp"), _template())
    assert W4A8 not in text
    assert "p-h3-image_prompt_upsampler_w4a8.safetensors" in text  # driver models still are


def test_a_twin_the_lora_route_refuses_is_never_named(monkeypatch):
    """An INT8 MOSS-Image file with a LoRA refuses class P (#129). Without the LoRA,
    the same file is named, so the LoRA is the whole reason."""
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 40 * _GIB)
    assert KROMOSS_INT8 not in _refuse(_Mesh(), _kromoss(lora=True), KROMOSS)
    loader_preflight.reset_memos()
    assert f"{KROMOSS_INT8} (5.0 GiB) in place of {KROMOSS}" in _refuse(
        _Mesh(), _kromoss(lora=False), KROMOSS)


def test_a_twin_that_pins_attention_is_named_only_where_sequence_parallelism_drops_the_pin(
        monkeypatch, tmp_path):
    """A pinned approximate kernel runs, and refuses class K, off Ulysses and ring (#117)."""
    _write_safetensors(tmp_path / "diffusion_models" / KROMOSS_INT8,
                       ("blocks.0.weight", "blocks.0.attn.comfy_attention.config"))
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 40 * _GIB)
    assert KROMOSS_INT8 not in _refuse(_Mesh(), _kromoss(lora=False), KROMOSS)
    loader_preflight.reset_memos()
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 34 * _GIB)
    assert KROMOSS_INT8 in _refuse(_pair("uly2"), _kromoss(lora=False), KROMOSS)


def test_a_file_no_graph_here_names_is_never_named():
    """A local fp8 copy that MODELS and the manifest do not list has no record."""
    assert FP8 not in _refuse(_Mesh(), _template())


def test_the_pair_is_named_with_the_setting_it_needs_and_never_beside_a_lora(monkeypatch):
    text = _refuse(_Mesh(), _template())
    assert "with the Init node's topology set to uly2+fsdp (auto never picks it)" in text
    monkeypatch.setattr(graph_load_price, "graph_loras", lambda _prompt: ("a_lora.safetensors",))
    loader_preflight.reset_memos()
    assert "uly2+fsdp" not in _refuse(_Mesh(), _template())


# --- 2. credit what ComfyUI holds, never the whole stack on any entry -------

def test_a_held_encoder_credits_its_own_bytes_not_the_upsampler(monkeypatch):
    """The hardware's first one-rank session: the switch-false render left the DiT
    resident and ComfyUI holding the text encoder and VAE; cell 9 then took the
    head to 7.6 GiB. a26122b credited the whole stack and admitted at 14.6 GiB."""
    graph = _template()
    switch = next(key for key, node in graph.items() if node["class_type"] == "ComfySwitchNode")
    graph[switch]["inputs"]["switch"] = False
    assert loader_preflight.preflight_loader_footprint(_Mesh(), DIT, {}, graph) is None
    held = SIZES[("text_encoders", "p-h3-image_text_encoder_bf16.safetensors")] + SIZES[
        ("vae", "p-h3-image_vae_bf16.safetensors")]
    loaded = types.SimpleNamespace(model=object(), model_memory=lambda: held)
    management = types.ModuleType("comfy.model_management")
    management.current_loaded_models = [loaded]  # type: ignore[attr-defined]
    comfy = types.ModuleType("comfy")
    comfy.model_management = management  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: int(14.6 * _GIB))
    text = _refuse(_Mesh(), _template())
    assert "comfy already holds 13.7 GiB of loaded models" in text
    graph[switch]["inputs"]["switch"] = True
    terms = loader_graph.credited_terms(loader_graph.driver_stack_terms(graph, family="minimax_h3"))
    encoders = SIZES[("text_encoders", UPSAMPLER)] + SIZES[
        ("text_encoders", "p-h3-image_text_encoder_bf16.safetensors")]
    assert terms.models_bytes == int(encoders * loader_graph.TE_RESIDENT_RATIO) + SIZES[
        ("vae", "p-h3-image_vae_bf16.safetensors")] - held


# --- 3 and 8. every stock fallback says why, with a remedy that works -------

def test_fleet_notes_the_stock_it_forces(monkeypatch):
    from dgx_monarch.nodes import fleet_policy, gate

    def unreadable(*_a, **_k):
        raise RuntimeError("ledger unreadable")

    monkeypatch.setattr(gate, "_effective_worker_args", unreadable)
    spec = types.SimpleNamespace(mesh=types.SimpleNamespace(worker_args={"slab_weights": True}),
                                 request_dict=lambda: {"unet_name": DIT})
    handle = types.SimpleNamespace(config=types.SimpleNamespace(worker_args={}))
    seen = []

    @refusal_lead.lead_with_refusal
    def fleet_call():
        seen.append(fleet_policy._fleet_worker_policy(spec, handle).worker_args["slab_weights"])
        seen.append(refusal_lead.forced_stock_state())

    fleet_call()
    assert seen == [False, "unknown"]


@pytest.mark.parametrize("requested,dispatched,state,noted", [
    (True, False, "unknown", "unknown"),                       # a failed authorization
    (True, False, refusal_lead.DUAL_MODEL, "two models"),     # a dual-model render on cfg2
    (True, False, refusal_lead.DUAL_UNPROVEN, "two models, no dual PASS"),  # no dual PASS yet
    ("auto", False, "unknown", None),                         # nothing was asked for
    (True, True, "unknown", None),                            # nothing was forced
])
def test_every_stock_fallback_of_a_render_is_noted(requested, dispatched, state, noted):
    seen = []

    @refusal_lead.lead_with_refusal
    def render():
        refusal_lead.note_if_slab_dropped({"slab_weights": requested},
                                          {"slab_weights": dispatched}, state=state)
        seen.append(refusal_lead.forced_stock_state())

    render()
    assert seen == [noted]
    assert "refusal_lead.note_if_slab_dropped(" in inspect.getsource(render_submit)


def test_a_dual_model_refusal_names_its_cause_and_no_gate(monkeypatch):
    text = refusal_lead.forced_stock_text(_worker_stock_refusal(monkeypatch),
                                          refusal_lead.DUAL_MODEL)
    assert "a render with two models on a cfg-parallel topology runs on stock residency" in text
    assert "cannot swap-test the model a rank does not hold" in text
    assert "Identity Gate (DGX Monarch) on this model" not in text
    assert "set slab_weights=on" not in text and parse_refusal_tag(text) is not None


def test_an_unproven_dual_render_names_a_remedy_that_can_work(monkeypatch):
    """The Identity Gate node proves one model, so its PASS would not grant the pair."""
    text = refusal_lead.forced_stock_text(_worker_stock_refusal(monkeypatch),
                                          refusal_lead.DUAL_UNPROVEN)
    assert "this pair of models has no dual identity-gate PASS" in text
    assert "so its first-use identity gate proves both models" in text
    assert "Identity Gate (DGX Monarch) on this model" not in text
    assert "set slab_weights=on" not in text and parse_refusal_tag(text) is not None


def test_the_gate_remedy_says_how_to_make_the_stock_reference_fit(monkeypatch):
    text = refusal_lead.forced_stock_text(_worker_stock_refusal(monkeypatch), "stale")
    assert "in a graph whose stock reference fits the box that refused" in text
    assert "a prompt upsampler switched off" in text


# --- 4. the merge with #142: both ShortBox additions, by keyword ------------

def test_the_loader_short_box_keeps_142s_fields_and_its_window():
    box = old_model.ShortBox("this box", ("old",), 50 * _GIB, 40 * _GIB, 1 * _GIB, window="load")
    assert box.dropped == () and box.left_bytes == 0 and box.window == "load"
    offer = old_model.ClearOffer("new.safetensors", (box,), 1, True)
    assert "A clear would not make room in the load window: new needs 50.0 GiB" in (
        old_model.no_room(offer))
    assert "gain, window=window)" in inspect.getsource(old_model_clear.loader_offer)


# --- 5. the DiT ceiling clears every failing window -------------------------

def test_the_quoted_ceiling_is_the_smallest_a_failing_window_allows():
    profile = driver_footprint.DRIVER_STACK_FAMILIES["minimax_h3"]
    load = driver_footprint.estimate_driver_footprint(
        profile=profile, mem_available=100 * _GIB, weight_bytes=60 * _GIB,
        co_resident=True, slab_weights=False, reserve=5 * _GIB)
    resident = driver_footprint.estimate_driver_footprint(
        profile=profile, mem_available=100 * _GIB, weight_bytes=60 * _GIB,
        co_resident=True, slab_weights=True, reserve=65 * _GIB)
    assert load.shortfall > resident.shortfall and resident.dit_ceiling < load.dit_ceiling
    text = graph_load_price.what_would_fit((load, resident), profile)
    assert f"at most {resident.dit_ceiling / _GIB:.1f} GiB" in text
    assert f"or {load.shortfall / _GIB:.1f} GiB more free memory" in text


# --- 6. the slab card quotes the lead's figures -----------------------------

def test_the_slab_card_quotes_the_figures_the_refusal_leads_with():
    _refuse(_Mesh(), _template())
    (card,) = consent_pending.pending_cards()
    assert card["numbers"] == "needs 119.8 GiB, 110.8 GiB available on this box"


# --- 7 and 12. the docs say what the refusal offers and what a fresh driver costs

def test_models_names_the_slab_card_for_the_bf16_template():
    section = (ROOT / "docs" / "MODELS.md").read_text().split('<a id="model-p-h3-image">')[1]
    memory = section.split("**Memory.**")[1].split("**Known limitations.**")[0]
    # The 2026-11-05 recheck rendered the template on one Spark under the card.
    assert "Load with slab residency" in memory and "the slab card" in memory
    assert "2026-11-05" in memory and "pixel-identical to the same graph on stock" in memory


def test_troubleshooting_says_what_a_held_model_credits_and_a_fresh_driver_costs():
    entry = (ROOT / "docs" / "TROUBLESHOOTING.md").read_text().split(
        "## 52. Driver-side footprint preflight refused a render")[1].split("\n## 53.")[0]
    assert "credits only the bytes ComfyUI holds" in entry
    assert "80.9 GiB" in entry


# --- 9. a refusal too long for refusal() still refuses ----------------------

def test_an_over_long_refusal_is_cut_and_still_refuses(monkeypatch):
    graph = _template()
    names = [f"{'x' * 140}_{index}_bf16.safetensors" for index in range(30)]
    for index, name in enumerate(names):
        graph[f"9{index}"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": name}}
    fp = sys.modules["folder_paths"]
    real_path, real_size = fp.get_full_path, driver_footprint.file_size_bytes
    monkeypatch.setattr(fp, "get_full_path", lambda folder, name: (
        f"/long/{name}" if name in names else real_path(folder, name)))
    monkeypatch.setattr(driver_footprint, "file_size_bytes", lambda path: (
        1 if str(path).startswith("/long/") else real_size(path)))
    text = _refuse(_Mesh(), graph)
    assert len(text) <= 4000 and text.startswith("[dgxm:C guard=loader_footprint_preflight")
    assert "needs 119.8 GiB on this box" in text


def test_bounded_cuts_the_middle_and_keeps_both_ends():
    text = "head " + "y" * 5000 + " tail"
    cut = graph_load_price.bounded(text)
    assert len(cut) <= 3500 and cut.startswith("head ") and cut.endswith(" tail")
    assert refusal(RefusalClass.CAPACITY, cut, guard="loader_footprint_preflight",
                   waivable=False)


# --- 10. only what an output node reaches is priced -------------------------

def test_a_loader_no_output_node_reaches_is_not_priced(monkeypatch):
    registry = types.ModuleType("nodes")
    registry.NODE_CLASS_MAPPINGS = {  # type: ignore[attr-defined]
        "SaveImage": type("SaveImage", (), {"OUTPUT_NODE": True}),
        "CLIPLoader": type("CLIPLoader", (), {}),
        "VAEDecode": type("VAEDecode", (), {}),
    }
    monkeypatch.setitem(sys.modules, "nodes", registry)
    graph = {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": UPSAMPLER}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}},
        "3": {"class_type": "Unknown", "inputs": {}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
    }
    assert graph_live.live_nodes(graph) == frozenset({"2", "3", "4"})


# --- 11. the Identity Gate leads with a Worker's refusal too ---------------

def test_the_identity_gate_node_leads_with_a_worker_refusal():
    lead = refusal_lead.lead_with_refusal(lambda: None).__code__
    assert gate_node.DGXMonarchIdentityGate.gate.__code__ is lead


# --- #143's naming rule over every sentence this PR adds --------------------

_HEAD_NAME, _SIBLING_NAME = "spark-a.invalid", "spark-b.invalid"
_ADDRESSES = ("10.11.12.1", "10.11.12.2")
_RANK = re.compile(r"\brank \d")


def _named_pair(preset: str = "auto") -> _Mesh:
    """A pair whose cluster.toml names its hosts and gives their addresses."""
    hosts = tuple(types.SimpleNamespace(name=name, address=address)
                  for name, address in zip((_HEAD_NAME, _SIBLING_NAME), _ADDRESSES, strict=True))
    return _Mesh(preset, world=2, hosts=hosts)


def _no_box_by_number_name_or_address(text: str) -> None:
    assert not _RANK.search(text), text
    for word in (_HEAD_NAME, _SIBLING_NAME, *_ADDRESSES):
        assert word not in text, (word, text)


@pytest.fixture
def _head_is_spark_a(monkeypatch):
    monkeypatch.setattr(driver_footprint, "host_is_local", lambda name: name == _HEAD_NAME)


def test_the_loader_refusal_names_the_head_on_a_pair_and_this_box_alone(_head_is_spark_a):
    text = _refuse(_named_pair(), _template())
    assert "p-h3-image_bf16.safetensors needs 119.8 GiB on the head and 110.8 GiB" in text
    _no_box_by_number_name_or_address(text)
    (card,) = consent_pending.pending_cards()
    assert card["numbers"] == "needs 119.8 GiB, 110.8 GiB available on the head"
    loader_preflight.reset_memos()
    consent_pending.clear_all()
    local = _refuse(_Mesh(), _template())
    assert "needs 119.8 GiB on this box and 110.8 GiB" in local
    assert "On a pair of Sparks with the Init node's topology set to uly2+fsdp" in local
    assert "the head would hold 20.0 GiB of the DiT" in local
    _no_box_by_number_name_or_address(local)


def test_a_pair_whose_config_lists_only_the_driver_box_reads_this_box(_head_is_spark_a):
    hosts = (types.SimpleNamespace(name=_HEAD_NAME, address=_ADDRESSES[0]),)
    assert graph_load_price.driver_box(_Mesh("auto", world=1, hosts=hosts)) == "this box"
    assert graph_load_price.driver_box(_named_pair()) == "the head"
    assert graph_load_price.driver_box(types.SimpleNamespace(handle=None)) == "this box"


def test_the_words_this_pr_adds_to_a_worker_refusal_name_no_box(monkeypatch):
    """The forced-stock cause and remedy, and the dual-model cause, name no rank,
    host or address; the Worker's own sentence keeps its own words."""
    worker = _worker_stock_refusal(monkeypatch)
    for state in ("stale", refusal_lead.DUAL_MODEL):
        added = refusal_lead.forced_stock_text(worker, state).replace(
            worker.split(" Slab rescue is unavailable here")[0], "")
        _no_box_by_number_name_or_address(added)
    _no_box_by_number_name_or_address(refusal_lead.GATE_REMEDY)


def test_the_loader_no_room_sentence_names_the_box_not_its_rank(_head_is_spark_a):
    box = old_model.ShortBox("the head", ("old",), 50 * _GIB, 40 * _GIB, 1 * _GIB,
                             window="load")
    text = old_model.no_room(old_model.ClearOffer("new.safetensors", (box,), 2, False))
    assert "needs 50.0 GiB on the head" in text
    _no_box_by_number_name_or_address(text)


def test_the_docs_this_pr_adds_name_no_box_by_rank():
    """TROUBLESHOOTING #52, MODELS' p-h3-image memory paragraph and CONCEPTS'
    driver models entry, as a user reads them."""
    docs = ROOT / "docs"
    entry = (docs / "TROUBLESHOOTING.md").read_text().split(
        "## 52. Driver-side footprint preflight refused a render")[1].split("\n## 53.")[0]
    added = [paragraph for paragraph in entry.split("\n\n") if any(marker in paragraph for marker in (
        "For example, the p-h3-image template", "**What to do:**", "uly2+fsdp",
        "credits only the bytes ComfyUI holds"))]
    assert len(added) >= 3
    passages = [
        *added,
        (docs / "MODELS.md").read_text().split('<a id="model-p-h3-image">')[1].split(
            "**Memory.**")[1].split("**Known limitations.**")[0],
        (docs / "CONCEPTS.md").read_text().split("**Driver models.**")[1].split("\n\n")[0],
    ]
    for passage in passages:
        assert not _RANK.search(passage), passage
        assert "rank 0" not in passage


# --- the Greptile review of 1b7a560 ----------------------------------------

def _holding(monkeypatch, held: int) -> None:
    loaded = types.SimpleNamespace(model=object(), model_memory=lambda: held)
    management = types.ModuleType("comfy.model_management")
    management.current_loaded_models = [loaded]  # type: ignore[attr-defined]
    comfy = types.ModuleType("comfy")
    comfy.model_management = management  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)


def test_a_driver_twin_takes_the_held_credit_once(monkeypatch):
    """ComfyUI holds 33 GiB, so 2.6 GiB of the graph's 35.6 GiB of models stays
    charged. A W4A8 upsampler saves at most that much, not its 13.5 GiB file
    saving on top of the credit, so at 80 GiB it does not make the load fit;
    1b7a560 subtracted the saving from the credited stack and named it."""
    _holding(monkeypatch, 33 * _GIB)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 80 * _GIB)
    text = _refuse(_Mesh(), _template())
    assert "p-h3-image_prompt_upsampler_w4a8.safetensors" not in text
    assert "p-h3-image_int8_convrot.safetensors (19.8 GiB)" in text


def test_a_driver_twin_is_looked_for_beside_the_file_in_its_subfolder():
    """The graph loads sub/<upsampler>; only its INT8 copy sits beside it. 1b7a560
    looked at the folder root, named the root's W4A8 copy as well, and named
    each file without the subfolder the loader widget shows."""
    graph = _template()
    loader = next(key for key, node in graph.items()
                  if node["inputs"].get("clip_name") == UPSAMPLER)
    graph[loader]["inputs"]["clip_name"] = f"sub/{UPSAMPLER}"
    text = _refuse(_Mesh(), graph)
    assert (f"sub/p-h3-image_prompt_upsampler_int8_convrot.safetensors (9.1 GiB) in place "
            f"of sub/{UPSAMPLER}") in text
    assert "p-h3-image_prompt_upsampler_w4a8.safetensors" not in text



@pytest.mark.parametrize("projected,available,reserve", [
    (0.0, 10.0, 18.0), (45.9, 47.3, 5.0), (20.0, 30.0, 5.0), (0.0, 30.0, 18.0)])
def test_the_card_and_the_refusal_read_a_window_alike(projected, available, reserve):
    window = driver_footprint.DriverFootprintEstimate(
        weight_bytes=int(projected * _GIB), mem_available=int(available * _GIB),
        reserve=int(reserve * _GIB))
    assert (old_model_clear._worst((window,)) is None) == graph_load_price.fits_box(window)
