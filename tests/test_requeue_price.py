"""A re-queue whose UNET loader ComfyUI kept from an earlier queue is priced at the sampler.

On the 2026-11-05 hardware re-check (D2) the p-h3-image template rendered with
the prompt upsampler switched off, then the same driver queued it again with
the switch on. ComfyUI kept the UNET loader's output, so the loader's graph
price never ran for the second prompt; the upsampler loaded and the render
completed only because the box had room. The sampler now runs that price for
any prompt whose loader did not, pricing what is not yet held: the DiT while its
deferred load is pending, and the driver models beyond what ComfyUI holds.

CPU only: the rig of test_graph_load_price stands in for the models and the box.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from dgx_monarch import consent_pending, driver_footprint, driver_unload, mesh_safety, old_model
from dgx_monarch.nodes import (
    graph_load_price,
    loader_preflight,
    old_model_clear,
    recycle_drain,
    requeue_price,
    samplers,
)
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from test_graph_load_price import (  # noqa: F401  # the rig is an autouse fixture
    _GIB,
    DIT,
    DIT_INT8,
    DIT_W4A8,
    SIZES,
    TE,
    UPSAMPLER,
    VAE,
    _Mesh,
    _rig,
    _switch_id,
    _template,
)

# What ComfyUI holds once Generate Text and the text encode have run: both encoders.
ENCODERS_HELD = SIZES[("text_encoders", TE)] + SIZES[("text_encoders", UPSAMPLER)]
# D2's replay, at its lowest: the head with the DiT resident and both encoders loaded.
REPLAY_AVAILABLE = int(34.8 * _GIB)
# The cell 9 box (110.8 GiB) after the encoders loaded, at their 1.15 ratio.
CELL9_AVAILABLE = int(80.3 * _GIB)
# The same box with 25.3 GiB more taken by something the driver cannot unload (a
# co-resident server): short of the DiT's load even after ComfyUI's encoders go.
SHORT_AVAILABLE = int(55.0 * _GIB)
SAMPLER_SENTENCE = ("The UNET Loader's output came from an earlier queue, so the sampler priced "
                    "this graph, and the Workers loaded nothing.")


@pytest.fixture(autouse=True)
def _prompt(monkeypatch):
    """ComfyUI's executing context: the test sets the prompt id a node runs under."""
    state = types.SimpleNamespace(prompt_id="prompt-1")
    utils = types.ModuleType("comfy_execution.utils")
    utils.get_executing_context = lambda: (  # type: ignore[attr-defined]
        None if state.prompt_id is None else types.SimpleNamespace(prompt_id=state.prompt_id))
    package = types.ModuleType("comfy_execution")
    package.utils = utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy_execution", package)
    monkeypatch.setitem(sys.modules, "comfy_execution.utils", utils)
    graph_load_price._PRICED_PROMPTS.clear()
    yield state
    graph_load_price._PRICED_PROMPTS.clear()


class TextEncoder:
    """Stands in for the class ComfyUI logs for the loaded text encoder."""


class PromptUpsampler:
    """Stands in for the class ComfyUI logs for the loaded prompt upsampler."""


class ImageVAE:
    """Stands in for the class ComfyUI logs for the loaded VAE."""


# Each file this graph loads on the driver: the class ComfyUI logs, its folder, and
# whether ComfyUI marks its patcher ``is_clip`` (a CLIP's is; a VAE's is not).
_LOADED = {TE: (TextEncoder, "text_encoders", True), UPSAMPLER: (PromptUpsampler, "text_encoders", True),
           VAE: (ImageVAE, "vae", False)}


def _comfy_holds(monkeypatch, *files: str) -> types.SimpleNamespace:
    """ComfyUI's model list holding these files, as ComfyUI's own entries hold them.

    Each entry's patcher carries the ``is_clip`` mark ComfyUI sets on a CLIP's
    patcher, and ``model_memory`` is the file's size. ``unload_all_models``
    empties the list and the box reads what the entries held more, as the
    driver's own unload (``driver_unload.unload``) then finds it.
    """
    management = types.ModuleType("comfy.model_management")
    management.current_loaded_models = [  # type: ignore[attr-defined]
        types.SimpleNamespace(model=types.SimpleNamespace(model=_LOADED[name][0](), is_clip=_LOADED[name][2]),
                              model_memory=lambda size=SIZES[(_LOADED[name][1], name)]: size)
        for name in files]
    calls = types.SimpleNamespace(unload=0)

    def unload_all_models():
        calls.unload += 1
        freed = sum(entry.model_memory() for entry in management.current_loaded_models)
        management.current_loaded_models[:] = []
        reading = mesh_safety.mem_available_bytes
        monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: reading() + freed)

    management.unload_all_models = unload_all_models  # type: ignore[attr-defined]
    management.soft_empty_cache = lambda: None  # type: ignore[attr-defined]
    comfy = types.ModuleType("comfy")
    comfy.model_management = management  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    return calls


def _graph(switch: bool) -> dict:
    graph = _template()
    graph[_switch_id(graph)]["inputs"]["switch"] = switch
    return graph


def _model(mesh=None, unet: str = DIT):
    return types.SimpleNamespace(mesh=mesh or _Mesh(), unet_name=unet, options={})


def _first_prompt_renders(prompt, monkeypatch) -> _Mesh:
    """Prompt 1: the loader prices the switch-off graph and admits it; the render loads the DiT."""
    mesh = _Mesh()
    assert loader_preflight.preflight_loader_footprint(mesh, DIT, {}, _graph(False)) is None
    requeue_price.price_cached_loader(_model(mesh), _graph(False))     # the loader priced it
    prompt.prompt_id = "prompt-2"                                       # the re-queue
    return mesh


def _switch_on_renders(prompt, monkeypatch) -> _Mesh:
    """Prompt 1 with the switch on, on a box roomier than cell 9's (the loader needs 119.8 GiB
    there): both encoders and the upsampler run, so a re-queue of the same prompt finds their
    outputs in ComfyUI's cache."""
    mesh = _Mesh()
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 125 * _GIB)
    assert loader_preflight.preflight_loader_footprint(mesh, DIT, {}, _graph(True)) is None
    prompt.prompt_id = "prompt-2"
    return mesh


def _spy(monkeypatch) -> list:
    calls: list = []
    real = loader_preflight.preflight_loader_footprint

    def spy(*args, **kwargs):
        calls.append(kwargs.get("site"))
        return real(*args, **kwargs)

    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint", spy)
    return calls


# D2's replay: the DiT stays held, so only the VAE beside what ComfyUI holds is priced.

def test_the_replay_with_the_dit_held_is_priced_at_the_sampler_and_renders(_prompt, monkeypatch):
    mesh = _first_prompt_renders(_prompt, monkeypatch)
    calls = _spy(monkeypatch)
    _comfy_holds(monkeypatch, TE, UPSAMPLER)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: REPLAY_AVAILABLE)
    requeue_price.price_cached_loader(_model(mesh), _graph(True))
    assert calls == [graph_load_price.SITE_SAMPLER], "the sampler ran the loader's price once"
    assert consent_pending.pending_cards() == []
    # A second sampler in the same prompt does not price it again.
    requeue_price.price_cached_loader(_model(mesh), _graph(True))
    assert calls == [graph_load_price.SITE_SAMPLER]


def test_a_first_run_whose_loader_priced_the_prompt_gets_no_second_price(_prompt, monkeypatch):
    calls = _spy(monkeypatch)
    graph = _graph(True)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError):
        loader_preflight.preflight_loader_footprint(_Mesh(), DIT, {}, graph)
    calls.clear()
    requeue_price.price_cached_loader(_model(), graph)
    assert calls == [], "the loader priced this prompt, so the sampler does not"


# A short box with the DiT load pending: the loader's sentence and remedies, at the sampler.

def _short_requeue(prompt, monkeypatch) -> _Mesh:
    mesh = _first_prompt_renders(prompt, monkeypatch)
    recycle_drain.drop_residency_memos()        # a Clear VRAM node, Reset attached mesh or a clear
    _comfy_holds(monkeypatch, TE, UPSAMPLER)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: SHORT_AVAILABLE)
    return mesh


def test_a_short_box_refuses_at_the_sampler_with_the_loaders_sentence(_prompt, monkeypatch):
    """Short with both encoders held, and still short after the driver unloads them: the
    refusal is the price after the unload, and it says so."""
    mesh = _short_requeue(_prompt, monkeypatch)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        requeue_price.price_cached_loader(_model(mesh), _graph(True))
    text = str(raised.value)
    assert text.startswith(
        "[dgxm:C guard=loader_footprint_preflight waivable=1] p-h3-image_bf16.safetensors "
        "needs 84.1 GiB on this box and 81.5 GiB is available, 2.7 GiB short. That is the "
        "peak of its load, the weights beside the host copy they are placed from. "
        f"{SAMPLER_SENTENCE} Before refusing, the driver unloaded the 2 models (TextEncoder, "
        "PromptUpsampler) ComfyUI held on this host: MemAvailable read 55.0 GiB before the "
        "unload and 81.5 GiB after it, and the figures above are the reading after it.")
    assert "Nothing was loaded." not in text
    assert ("driver-side footprint preflight at the sampler (the UNET Loader's output came "
            "from an earlier queue)") in text
    assert f"{DIT_INT8} (19.8 GiB) or {DIT_W4A8} (12.4 GiB) in place of {DIT}" in text
    assert "topology set to uly2+fsdp (auto never picks it)" in text
    assert "Load with slab residency" in text
    assert [card["kind"] for card in consent_pending.pending_cards()] == ["rescue-slab"]
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY and tag.waivable
    assert len(text) < 4000


# Review of the 1.5.0 promotion, finding a2#1: after the unload, the encodes have still run.

def test_the_cell_9_requeue_the_unload_makes_room_for_renders(_prompt, monkeypatch):
    """The switch-off graph rendered, a reset left the DiT pending, and the switch-on re-queue
    meets both encoders held on the cell 9 box. The first price is short, so the driver
    unloads them (26.5 GiB). The price after it charged both encoders again in the load
    window, as models "this graph loads on the driver first", though their encodes had run
    and nothing loads them again before the DiT: 114.6 GiB against 106.8, refused. The DiT
    alone needs 84.1 GiB there, so the load renders."""
    mesh = _first_prompt_renders(_prompt, monkeypatch)
    recycle_drain.drop_residency_memos()
    calls = _comfy_holds(monkeypatch, TE, UPSAMPLER)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: CELL9_AVAILABLE)
    requeue_price.price_cached_loader(_model(mesh), _graph(True))
    assert calls.unload == 1 and consent_pending.pending_cards() == []


def test_a_requeue_whose_encodes_an_earlier_queue_ran_is_not_charged_them_in_the_load_window(
        _prompt, monkeypatch):
    """ComfyUI holds no model (a Clear VRAM node at hard took them) and the prompt is the one
    an earlier queue encoded, so its outputs stand in for both encoders: none loads before the
    DiT. 90 GiB covers the DiT's load and the settled window that still charges them."""
    mesh = _switch_on_renders(_prompt, monkeypatch)
    recycle_drain.drop_residency_memos()
    calls = _comfy_holds(monkeypatch)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: int(90.0 * _GIB))
    requeue_price.price_cached_loader(_model(mesh), _graph(True))
    assert calls.unload == 0 and consent_pending.pending_cards() == []
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: int(75.0 * _GIB))
    _prompt.prompt_id = "prompt-3"
    recycle_drain.drop_residency_memos()
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        requeue_price.price_cached_loader(_model(mesh), _graph(True))
    text = str(raised.value)
    assert "needs 84.1 GiB on this box and 75.0 GiB is available" in text
    assert ("That is the peak of its load, the weights beside the host copy they are placed "
            f"from. {SAMPLER_SENTENCE}") in text
    assert ("75.0 GiB MemAvailable minus a 5.0 GiB reserve and no driver model, since this "
            "graph's encodes ran before the sampler and a model it loads again loads after the "
            "DiT places)") in text


def test_a_swap_credit_at_the_sampler_does_not_wait_for_encoders_that_ran(_prompt, monkeypatch):
    """The Worker on this box holds the INT8 twin in the slot this load takes, and drops it
    first, which the swap credit counts. The encodes ran in an earlier queue and ComfyUI
    holds neither encoder. The credit made room for the DiT, but the price still waited
    for both encoders beside the old model, in the load window, and refused. The price runs
    in the real sampler node over the loader's own ModelSpec, whose slot asks the Worker."""
    from dgx_monarch import capacity_agreement
    from dgx_monarch.nodes.common import ModelSpec
    from test_capacity_old_model import _answer, _LiveHandle
    from test_loader_swap_credit import _held

    class Rendered(Exception):
        pass

    def rendered(*_args, **_kwargs):
        raise Rendered

    mesh = _switch_on_renders(_prompt, monkeypatch)
    mesh.handle = _LiveHandle()
    recycle_drain.drop_residency_memos()
    calls = _comfy_holds(monkeypatch)
    asked: list = []
    monkeypatch.setattr(capacity_agreement, "collect", lambda *_a, **_k: asked.append(1) or [
        _answer(0, _held(DIT_INT8, SIZES[("diffusion_models", DIT_INT8)] / _GIB), box="this box")])
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: int(70.0 * _GIB))
    monkeypatch.setattr(samplers, "run_render", rendered)
    with pytest.raises(Rendered):
        samplers.DGXMonarchKSampler().sample(
            ModelSpec(mesh=mesh, unet_name=DIT), 42, 8, 4.0, "euler", "simple", [], [],
            {"samples": None}, prompt=_graph(True))
    assert asked and calls.unload == 0 and consent_pending.pending_cards() == []


def test_held_encoders_count_at_their_resident_size_and_the_decode_vae_waits(_prompt, monkeypatch):
    """Review finding 3, its rig probe: the switch-off graph rendered, a reset left the DiT
    pending, and the re-queue meets the text encoder held at its raw size. The sampler
    charged it again at the 1.15 ratio and put the decode VAE in the load window, so it
    needed 90.6 GiB where the Worker's wall needs 84.1. A held encoder is credited at
    its measured 1.144 and the VAE, which loads after the DiT places, waits for the
    settled window, so 88 GiB admits. The review of the 1.5.0 promotion (a2#1): the
    held encoder's last 0.006 of its size, charged at 1.15 and credited at 1.144, still
    sat in the load window, though nothing loads it there; the load window now holds the
    DiT alone, at the Worker's wall, and 84.17 GiB admits with nothing unloaded."""
    mesh = _first_prompt_renders(_prompt, monkeypatch)
    recycle_drain.drop_residency_memos()
    calls = _comfy_holds(monkeypatch, TE)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: int(88.0 * _GIB))
    requeue_price.price_cached_loader(_model(mesh), _graph(False))          # admitted
    recycle_drain.drop_residency_memos()        # reset again: the DiT is pending once more
    _prompt.prompt_id = "prompt-3"
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: int(84.17 * _GIB))
    requeue_price.price_cached_loader(_model(mesh), _graph(False))          # admitted
    assert calls.unload == 0, "the first price fits: ComfyUI keeps its encoder"
    recycle_drain.drop_residency_memos()
    _prompt.prompt_id = "prompt-4"
    monkeypatch.setenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, "1")   # the first price's own words
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: int(84.0 * _GIB))
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        requeue_price.price_cached_loader(_model(mesh), _graph(False))
    text = str(raised.value)
    assert "needs 84.1 GiB on this box and 84.0 GiB is available" in text
    assert TE not in text.split(". ")[1] and VAE not in text.split(". ")[1], (
        "neither the held encoder nor the decode VAE is in the load window")


def test_a_client_that_sends_its_prompt_id_again_is_priced_at_the_sampler(_prompt, monkeypatch):
    """Review of the 1.5.0 promotion, a2#2: ComfyUI takes a client's own prompt id, checking only
    its form, so an outside client can send one id twice. The record of the loader's price was keyed
    by the id alone, so the second queue, whose loader ComfyUI kept, read as priced and the sampler
    priced nothing. Each record now names the running queue item's number, as the spend marks do;
    ComfyUI's queue is the server's, read as ``driver_spend_marks.running_item`` reads it."""
    reused = "3f2a6c1e-8b7d-4e2a-9c1f-0a1b2c3d4e5f"
    running: list = []
    queue = types.SimpleNamespace(get_current_queue_volatile=lambda: (list(running), []))
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(  # type: ignore[attr-defined]
        instance=types.SimpleNamespace(prompt_queue=queue))
    monkeypatch.setitem(sys.modules, "server", server)
    _prompt.prompt_id = reused
    running[:] = [(1, reused, _graph(False), {}, [])]
    mesh = _Mesh()
    assert loader_preflight.preflight_loader_footprint(mesh, DIT, {}, _graph(False)) is None
    running[:] = [(2, reused, _graph(True), {}, [])]   # the same id, the switch on, the loader kept
    recycle_drain.drop_residency_memos()
    _comfy_holds(monkeypatch, TE, UPSAMPLER)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: SHORT_AVAILABLE)
    calls = _spy(monkeypatch)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        requeue_price.price_cached_loader(_model(mesh), _graph(True))
    assert calls[0] == graph_load_price.SITE_SAMPLER and SAMPLER_SENTENCE in str(raised.value)


def test_the_same_short_box_with_room_renders(_prompt, monkeypatch):
    mesh = _short_requeue(_prompt, monkeypatch)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: int(94.0 * _GIB))
    requeue_price.price_cached_loader(_model(mesh), _graph(True))
    assert consent_pending.pending_cards() == []


def test_the_sampler_node_refuses_before_any_render_or_worker_call(_prompt, monkeypatch):
    mesh = _short_requeue(_prompt, monkeypatch)

    def reached(*_args, **_kwargs):
        raise AssertionError("the render ran before the sampler's price")

    monkeypatch.setattr(samplers, "run_render", reached)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        samplers.DGXMonarchKSampler().sample(
            _model(mesh), 42, 8, 4.0, "euler", "simple", [], [], {"samples": None},
            prompt=_graph(True))
    assert SAMPLER_SENTENCE in str(raised.value)


def test_a_dual_model_guider_prices_its_second_checkpoint_too(_prompt, monkeypatch):
    """SamplerCustom carries the uncond loader's checkpoint; each is priced alone, as its loader does."""
    import torch

    mesh = _first_prompt_renders(_prompt, monkeypatch)
    graph = _graph(True)
    graph["900"] = {"class_type": "DGXMonarchUncondUNETLoader",
                    "inputs": {"unet_name": DIT_INT8, "weight_dtype": "default"}}
    seen: list = []
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint",
                        lambda _mesh, unet, _options, _graph, site, slot: seen.append((unet, site, slot)))

    class Rendered(Exception):
        pass

    def rendered(*_args, **_kwargs):
        raise Rendered

    monkeypatch.setattr(samplers, "run_render", rendered)
    guider = {"model": _model(mesh), "spec": {}, "cfg": 4.0,
              "uncond_model": {"unet_name": DIT_INT8, "options": {}}}
    with pytest.raises(Rendered):
        samplers.DGXMonarchSamplerCustom().sample(
            types.SimpleNamespace(seed=42), guider, object(), torch.tensor([1.0, 0.0]),
            {"samples": torch.zeros(1, 16, 1, 8, 8)}, prompt=graph)
    # Each checkpoint is priced for its own slot (this stand-in model names none), so a swap
    # credit never crosses slots.
    assert seen == [(DIT, graph_load_price.SITE_SAMPLER, ""), (DIT_INT8, graph_load_price.SITE_SAMPLER, "uncond")]


def test_encodes_and_activations_stay_with_the_render_site_checks(_prompt, monkeypatch):
    """A re-queue of a reference-to-video graph after a render: the encode has run and the
    render-site checks price the activations, crediting what the Workers retain, so the
    sampler charges the driver models alone. The loader site, which runs before either,
    still charges both and refuses the same box."""
    from dataclasses import replace

    real = loader_preflight.driver_stack_terms

    def with_encodes(*args, **kwargs):
        return replace(real(*args, **kwargs), video_refs=1, ref_bytes=40 * _GIB,
                       activation_bytes=20 * _GIB, activation_note="a MiniMax H3 canvas")

    monkeypatch.setattr(loader_preflight, "driver_stack_terms", with_encodes)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 200 * _GIB)
    mesh = _Mesh()
    assert loader_preflight.preflight_loader_footprint(mesh, DIT, {}, _graph(False)) is None
    _prompt.prompt_id = "prompt-2"                  # the re-queue, the DiT held
    _comfy_holds(monkeypatch, TE, UPSAMPLER)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: int(40.0 * _GIB))
    requeue_price.price_cached_loader(_model(mesh), _graph(True))          # admitted
    _prompt.prompt_id = "prompt-3"
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, DIT, {}, _graph(True))
    assert "reference encode 40.0 GiB" in str(raised.value)
    assert "render activations 20.0 GiB (a MiniMax H3 canvas)" in str(raised.value)


# Where it does not run.

def test_identity_gate_prices_a_cached_loader_requeue_before_its_ceremony(_prompt, monkeypatch):
    """Review finding 5: Identity Gate (DGX Monarch) runs on every queue and renders the
    cached loader's model, so it prices the re-queue as a sampler does, and says so."""
    from dgx_monarch.nodes import gate, gate_node

    mesh = _short_requeue(_prompt, monkeypatch)

    def reached(*_args, **_kwargs):
        raise AssertionError("the ceremony ran before the price")

    monkeypatch.setattr(gate, "run_identity_ceremony", reached)
    monkeypatch.setattr(gate, "conditioning_for_wire", lambda conds: conds)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        gate_node.DGXMonarchIdentityGate().gate(
            _model(mesh), [], [], {"samples": None}, 42, 8, 4.0, "euler", "simple",
            prompt=_graph(True))
    text = str(raised.value)
    assert ("The UNET Loader's output came from an earlier queue, so Identity Gate (DGX "
            "Monarch) priced this graph, and the Workers loaded nothing.") in text
    assert "driver-side footprint preflight at Identity Gate (DGX Monarch) (the UNET" in text


def test_identity_gate_admits_the_cell_9_requeue_the_unload_makes_room_for(_prompt, monkeypatch):
    """Review of the 1.5.0 promotion, a2#1, at the gate: its ceremony's conditioning is encoded
    and the proof renders load the DiT inside the node, before any other node runs, so after
    the unload the price keeps the encoders out of the load window there too and the ceremony
    runs. It was refused 114.6 GiB against 106.8."""
    from dgx_monarch.nodes import gate, gate_node

    class Reached(Exception):
        pass

    def reached(*_args, **_kwargs):
        raise Reached

    mesh = _first_prompt_renders(_prompt, monkeypatch)
    recycle_drain.drop_residency_memos()
    calls = _comfy_holds(monkeypatch, TE, UPSAMPLER)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: CELL9_AVAILABLE)
    monkeypatch.setattr(gate, "run_identity_ceremony", reached)
    monkeypatch.setattr(gate, "conditioning_for_wire", lambda conds: conds)
    with pytest.raises(Reached):
        gate_node.DGXMonarchIdentityGate().gate(
            _model(mesh), [], [], {"samples": None}, 42, 8, 4.0, "euler", "simple",
            prompt=_graph(True))
    assert calls.unload == 1 and consent_pending.pending_cards() == []


@pytest.mark.parametrize(("case", "why"), [
    ("headless", "no prompt id: the loader's own price fails open there too"),
    ("explicit", "an explicit preset loaded at the loader node"),
    ("no graph", "no graph to walk"),
    ("other dit", "a DiT this graph's loaders do not name"),
])
def test_the_sampler_prices_nothing_it_cannot_tie_to_a_deferred_load(_prompt, monkeypatch, case, why):
    _short_requeue(_prompt, monkeypatch)
    calls = _spy(monkeypatch)
    model, graph = _model(), _graph(True)
    if case == "headless":
        _prompt.prompt_id = None
    elif case == "explicit":
        model = _model(_Mesh("single"))
    elif case == "no graph":
        graph = None
    else:
        model = _model(unet="another_dit.safetensors")
    requeue_price.price_cached_loader(model, graph)
    assert calls == [], why


def test_always_clear_and_retry_clears_once_and_prices_again(_prompt, monkeypatch):
    """An Old model in the way refusal at the sampler takes the loader's Always flow."""
    mesh = _short_requeue(_prompt, monkeypatch)
    calls: list = []

    def priced(*_args, **kwargs):
        calls.append(kwargs.get("site"))
        if len(calls) == 1:
            raise old_model.OldModelRefusal("[dgxm:C guard=loader_footprint_preflight] short")

    cleared: list = []
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint", priced)
    monkeypatch.setattr(old_model_clear, "clear_for_retry",
                        lambda handle, exc, *, inherited=False: cleared.append(handle) or True)
    requeue_price.price_cached_loader(_model(mesh), _graph(True))
    assert calls == [graph_load_price.SITE_SAMPLER] * 2 and cleared == [mesh.handle]
    monkeypatch.setattr(old_model_clear, "clear_for_retry", lambda *_a, **_k: False)
    _prompt.prompt_id = "prompt-3"
    calls.clear()
    with pytest.raises(old_model.OldModelRefusal):
        requeue_price.price_cached_loader(_model(mesh), _graph(True))
    assert calls == [graph_load_price.SITE_SAMPLER]


# The docs say what the code does (review of #148, findings 4, 5 and 6e).

DESIGN = Path(__file__).resolve().parents[1] / "docs" / "DESIGN.md"


def _paragraph(title: str) -> list[str]:
    lines = DESIGN.read_text().splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(f"**{title}**"))
    end = next(index for index in range(start, len(lines)) if not lines[index].strip())
    return lines[start:end]


def test_design_names_the_doors_that_drain_the_memo_and_every_render_site():
    """Findings 4 and 5: a restart and a soft or hard Clear VRAM drain nothing before
    the price, and Identity Gate (DGX Monarch) renders such a queue too."""
    text = " ".join(" ".join(_paragraph(
        "A re-queue the loader does not see is priced where it renders.")).split())
    assert ("Reset attached mesh, Clear and retry and a Clear VRAM node set to `recycle` "
            "drain that memo") in text
    assert "a Clear VRAM node at `soft` or `hard` does not" in text
    assert "a sampler or Identity Gate (DGX Monarch)" in text
    assert "every clear, reset and restart drains" not in DESIGN.read_text()


def test_the_new_design_paragraph_is_wrapped():
    """Finding 6 (e): no prose line runs on."""
    assert max(len(line) for line in _paragraph(
        "A re-queue the loader does not see is priced where it renders.")) <= 80
