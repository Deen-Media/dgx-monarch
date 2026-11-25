"""The render-site checks read the model spec from the model, as the dispatch does.

No sampler, Identity Gate leg or capacity stop puts ``"model"`` into its
request: ``render_submit`` sets ``request["model"]`` from the model object only
after the render-site checks ran. A check that read the spec from the request
therefore asked the Workers nothing (the swap credit, ``loader_relief.render_credit``)
or asked them about a load with no LoRA stack (the pending-load relief's quote,
``pending_load_relief.workers_stop``). Every test here builds its request with
the real sampler and guider nodes, which is the request production builds, and
stops the render just after the checks.

CPU only, on the driver-unload rig of ``test_driver_unload``.
"""
from __future__ import annotations

import pytest
import torch

import test_driver_unload as rig
from dgx_monarch import capacity_agreement, driver_footprint, swap_credit
from dgx_monarch.nodes import common, gate_capacity_stop, guiders, pending_load_relief, samplers
from dgx_monarch.nodes.common import ModelSpec
from test_capacity_old_model import _answer, _stock
from test_driver_unload import DIT_BYTES, UNET, _comfy, _h3_latent, _h3_need, _h3_rig
from test_loader_swap_credit import _held

# The driver-unload rig: a fresh progress tracker, a UMA box, the box's reading and the journal.
_isolated, box, journal = rig._isolated, rig.box, rig.journal

_GIB = 2 ** 30
OLD = "flux1-dev.safetensors"
STACK = ({"name": "style.safetensors", "strength": 0.8},)
TEXT_ROWS = 20  # the rows _h3_need prices: test_driver_unload._h3_request's text


class _Checked(Exception):
    """Raised where run_render leaves the render-site checks behind it."""


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    pending_load_relief.reset()
    # run_render claims the adoption context right after the render-site checks.
    monkeypatch.setattr(common, "_claim_adoption_context",
                        lambda: (_ for _ in ()).throw(_Checked()))
    yield
    pending_load_relief.reset()


def _live(rig_, **worker_args) -> ModelSpec:
    """The rig's local fleet, set up with Workers the driver may ask, and the
    loader's model handle on it, LoRA stack included."""
    mesh = rig_.model.mesh
    handle = mesh.handle
    handle.setup_key, handle.workers, handle.defunct = ("ready",), object(), False
    handle.teardown_complete = False
    handle.gpus_per_host = 1
    handle.config = type("Config", (), {"hosts": (), "worker_args": {}})()
    mesh.worker_args = dict(worker_args)
    model = ModelSpec(mesh=mesh, unet_name=UNET)
    for entry in STACK:
        model = model.with_lora(entry["name"], entry["strength"])
    return model


def _cond() -> list:
    return [[torch.zeros(1, TEXT_ROWS, 8), {}]]


def _queue(node: str, model: ModelSpec) -> None:
    """One queue press through the named sampler node, up to the end of the checks."""
    latent = _h3_latent()
    with pytest.raises(_Checked):
        if node == "ksampler":
            samplers.DGXMonarchKSampler().sample(
                model, 1, 8, 1.0, "euler", "simple", _cond(), _cond(), latent)
        elif node == "advanced":
            samplers.DGXMonarchKSamplerAdvanced().sample(
                model, "enable", 1, 8, 1.0, "euler", "simple", _cond(), _cond(), latent,
                0, 10000, "disable")
        else:
            (guider,) = guiders.DGXMonarchBasicGuider().get_guider(model, _cond())
            noise = type("Noise", (), {"seed": 1})()
            samplers.DGXMonarchSamplerCustom().sample(
                noise, guider, object(), torch.tensor([1.0, 0.5, 0.0]), latent)


def _sampler_request(monkeypatch, node: str, model: ModelSpec) -> dict:
    """The request the named sampler node hands run_render."""
    seen: list[dict] = []
    monkeypatch.setattr(samplers, "run_render", lambda _model, request, *_a, **_k: (
        seen.append(request), (_ for _ in ()).throw(_Checked()))[1])
    _queue(node, model)
    monkeypatch.setattr(samplers, "run_render", common.run_render)
    return seen[0]


@pytest.mark.parametrize("node", ["ksampler", "advanced", "custom"])
def test_a_swap_the_loader_admitted_on_the_credit_passes_the_samplers_checks(
        monkeypatch, tmp_path, box, journal, node):
    """The H3 price charges the pending 19.5 GiB DiT; 15 GiB are missing and a
    22.2 GiB stock model sits in the slot. The Worker's drop makes the room, as
    at the loader, so ComfyUI keeps its models. The question carries the
    model's own LoRA stack, as the sample-time agreement will."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    model = _live(rig_)
    box.mem = _h3_need(weights=DIT_BYTES) - 15 * _GIB
    calls = _comfy(monkeypatch, box)
    asked: list = []
    monkeypatch.setattr(capacity_agreement, "collect", lambda handle, request, world, *, deadline_s:
                        asked.append(request) or [_answer(0, _held(OLD, 22.2), box="this box")])
    _queue(node, model)
    assert asked and asked[0]["specs"] == [
        {"unet_name": UNET, "options": {}, "loras": list(STACK), "slot": "cond"}]
    assert calls.unload == 0, "the credit alone made room: ComfyUI keeps its models"
    line = next(record.getMessage() for record in journal)
    assert line.startswith(f"render preflight: {UNET} fits once the Worker on this box drops "
                           f"{OLD} before this load")


def test_the_credit_reaches_a_pipelined_push(monkeypatch, tmp_path, box, journal):
    """KSampler Pipeline (DGX Monarch) pushes each seed through the same checks."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    model = _live(rig_)
    model.mesh.pipeline_depth = 2
    box.mem = _h3_need(weights=DIT_BYTES) - 15 * _GIB
    calls = _comfy(monkeypatch, box)
    asked: list = []
    monkeypatch.setattr(capacity_agreement, "collect", lambda handle, request, world, *, deadline_s:
                        asked.append(request) or [_answer(0, _held(OLD, 22.2), box="this box")])
    monkeypatch.setattr(samplers, "auto_gate_required", lambda *_a, **_k: False)
    # The push binds the packed render model right after its checks.
    monkeypatch.setattr(common, "_bind_packed_render_model",
                        lambda *_a, **_k: (_ for _ in ()).throw(_Checked()))
    with pytest.raises(_Checked):
        samplers.DGXMonarchKSamplerPipeline().sample(
            model, "1, 2", 8, 1.0, "euler", "simple", _cond(), _cond(), _h3_latent())
    assert asked and asked[0]["specs"][0]["loras"] == list(STACK)
    assert calls.unload == 0


def test_the_credit_reaches_the_identity_gates_capacity_stop(monkeypatch, tmp_path, box):
    """The ceremony's stop runs the same checks on the sampler's own request."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    model = _live(rig_)
    request = _sampler_request(monkeypatch, "ksampler", model)
    box.mem = _h3_need(weights=DIT_BYTES) - 15 * _GIB
    calls = _comfy(monkeypatch, box, models=())
    monkeypatch.setattr(capacity_agreement, "collect",
                        lambda *_a, **_k: [_answer(0, _held(OLD, 22.2), box="this box")])
    gate_capacity_stop.ceremony_activation_preflight(model, request, _h3_latent())
    assert calls.unload == 0


@pytest.mark.parametrize("node", ["ksampler", "custom"])
def test_the_reliefs_quote_carries_the_stack_the_sampler_renders(monkeypatch, tmp_path, box, node):
    """lora_low_rss off: the Worker prices the stack's classic bake from its own
    store, so the local price is unread and the relief asks the quote. Asked
    with the model's stack, the Worker's price refuses the load, and the driver
    frees its models before the agreement; asked with none, it would read a
    plain stock load that fits, and the agreement at dispatch, which carries the
    stack, would refuse."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    model = _live(rig_, lora_low_rss=False)
    box.mem = 200 * _GIB  # every render-site check fits; only the Worker's bake is short
    calls = _comfy(monkeypatch, box)
    monkeypatch.setattr(driver_footprint, "rank0_co_resident", lambda _mesh: True)
    asked: list = []

    def collect(handle, request, world, *, deadline_s):
        asked.append(request)
        (spec,) = request["specs"]
        verdict = "refuse" if spec["loras"] else "fits"
        return [_answer(0, _stock(verdict, size_gib=19.5, avail_gib=180.0, resident=None),
                        box="this box")]

    monkeypatch.setattr(capacity_agreement, "collect", collect)
    _queue(node, model)
    assert asked and all(request["specs"][0]["loras"] == list(STACK) for request in asked)
    assert calls.unload == 1, "the Worker's own price stops the load: the driver frees its models"


def test_the_retry_after_a_worker_restart_asks_with_the_models_stack(monkeypatch, tmp_path, box):
    """render_submit's retry after every Worker loop restarted prices the render
    again on the new fleet (``_attempt_after_restart``). Its check asks that
    fleet's Workers with the model's own spec. Their new processes hold no
    model, so nothing is dropped and nothing is credited: the retry refuses
    where the price does not fit, and nothing is dispatched again."""
    from dgx_monarch import mesh_safety, restart_reattach
    from test_restart_reattach import A, B, _render_rig

    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    model = _live(rig_)
    request = _sampler_request(monkeypatch, "ksampler", model)
    fresh = model.mesh.handle
    stale = ModelSpec(mesh=type(model.mesh)(**{**vars(model.mesh), "handle": object()}),
                      unet_name=UNET, loras=model.loras)
    render_submit, state = _render_rig(monkeypatch, heal=lambda _handle, _exc: (
        restart_reattach.Reattached(fresh, 4.25, (A, B), "liveness_release")))
    box.mem = _h3_need(weights=DIT_BYTES) - 15 * _GIB
    _comfy(monkeypatch, box, models=())
    asked: list = []
    monkeypatch.setattr(capacity_agreement, "collect", lambda handle, request, world, *, deadline_s:
                        asked.append(request) or [_answer(0, _stock(
                            "fits", size_gib=19.5, avail_gib=180.0, resident=None), box="this box")])
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        render_submit.submit_render(stale, request, _h3_latent(), 1.0, 8)
    assert state.calls == 1 and state.drops == 1
    assert asked and asked[0]["specs"][0]["loras"] == list(STACK)


def test_a_render_with_no_stack_asks_the_quote_with_none(monkeypatch, tmp_path, box):
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    model = ModelSpec(mesh=_live(rig_, lora_low_rss=False).mesh, unet_name=UNET)
    box.mem = 200 * _GIB
    _comfy(monkeypatch, box)
    asked: list = []
    monkeypatch.setattr(capacity_agreement, "collect", lambda handle, request, world, *, deadline_s:
                        asked.append(request) or [])
    _queue("ksampler", model)
    assert asked == [], "no stack with lora_low_rss off: the local price is whole and fits"
    assert swap_credit.credited(box.mem) == box.mem
