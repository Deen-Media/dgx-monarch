"""The render-site row counts read a Sampler Custom request's guider.

DGX Monarch Sampler Custom sends ``{"kind": "custom", "guider": spec}``: its
conditioning sits under ``request["guider"]``, never at the top. The MiniMax H3
activation price, the reference and pose price (#47), the sample deadline and
the driver footprint's reference report read the positive entries; each read
only ``request["positive"]``, so a Sampler Custom render, the shipped
p-h3-image template's among them, was priced on its latent alone. Every
request here is built by the real guider and sampler nodes, and each count
must equal the KSampler's for the same conditioning.

CPU only, on the driver-unload rig of ``test_driver_unload``.
"""
from __future__ import annotations

import pytest
import torch

import test_driver_unload as rig
from dgx_monarch import driver_footprint, h3_activation, h3_rows, mesh_safety
from dgx_monarch.nodes import common, guiders, render_rows, samplers
from dgx_monarch.nodes.common import ModelSpec
from test_driver_unload import _comfy, _h3_latent, _h3_rig

_isolated, box = rig._isolated, rig.box

_GIB = 2 ** 30
TEXT_ROWS = 900
# Nine pictures at the 2912x2912 canvas, as p-h3-image's own reference scaling leaves them.
REFS = [{"kind": "image", "latent_h": 182, "latent_w": 182} for _ in range(9)]


class _Checked(Exception):
    """Raised where run_render leaves the render-site checks behind it."""


@pytest.fixture(autouse=True)
def _stop_after_the_checks(monkeypatch):
    monkeypatch.setattr(common, "_claim_adoption_context",
                        lambda: (_ for _ in ()).throw(_Checked()))


def _cond(extras: dict) -> list:
    return [[torch.zeros(1, TEXT_ROWS, 8), dict(extras)]]


def _queue(kind: str, model, extras: dict, latent: dict) -> None:
    """One queue press through a sampler node, with the same conditioning each time."""
    if kind == "ksampler":
        samplers.DGXMonarchKSampler().sample(
            model, 1, 8, 2.0, "euler", "simple", _cond(extras), _cond({}), latent)
        return
    if kind == "basic":
        (guider,) = guiders.DGXMonarchBasicGuider().get_guider(model, _cond(extras))
    elif kind == "cfg":
        (guider,) = guiders.DGXMonarchCFGGuider().get_guider(model, _cond(extras), _cond({}), 2.0)
    else:  # the two-checkpoint guider, its second model wired: kind "dual_model"
        second = ModelSpec(mesh=model.mesh, unet_name="second.safetensors", slot="uncond")
        (guider,) = guiders.DGXMonarchDualModelGuider().get_guider(
            model, _cond(extras), 2.0, second, _cond({}))
        assert guider["spec"]["kind"] == "dual_model"
    samplers.DGXMonarchSamplerCustom().sample(
        type("Noise", (), {"seed": 1})(), guider, object(), torch.tensor([1.0, 0.5, 0.0]), latent)


def _request(monkeypatch, kind: str, model, extras: dict, latent: dict) -> dict:
    """The request the sampler node hands run_render."""
    seen: list[dict] = []
    monkeypatch.setattr(samplers, "run_render", lambda _model, request, *_a, **_k: (
        seen.append(request), (_ for _ in ()).throw(_Checked()))[1])
    with pytest.raises(_Checked):
        _queue(kind, model, extras, latent)
    monkeypatch.setattr(samplers, "run_render", common.run_render)
    return seen[0]


KINDS = ["basic", "cfg", "dual_model"]


@pytest.mark.parametrize("kind", KINDS)
def test_the_h3_price_reads_the_guiders_text_and_reference_rows(monkeypatch, tmp_path, box, kind):
    """An explicit preset (the DiT credited), ComfyUI holding nothing, and a box
    1 GiB short of the KSampler render's price: the same conditioning through
    Sampler Custom refuses with the same figures."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="uly2")
    model = ModelSpec(mesh=rig_.model.mesh, unet_name=rig_.model.unet_name)
    latent = _h3_latent()
    extras = {h3_rows.REFS_KEY: REFS}
    rows = h3_rows.rows_from_request(_request(monkeypatch, "ksampler", model, extras, latent), latent)
    assert (rows.text, rows.ref) == (TEXT_ROWS, 9 * 91 * 91)
    estimate = h3_activation.estimate_h3_activation(
        rows=rows, sp_degree=2, weight_bytes=0, weights_resident=True, mem_available=0,
        reserve=driver_footprint.reserve_bytes(None))
    box.mem = estimate.projected + estimate.reserve - 1 * _GIB
    _comfy(monkeypatch, box, models=())
    refusals = []
    for node in ("ksampler", kind):
        with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
            _queue(node, model, extras, latent)
        refusals.append(str(raised.value).split(". Refuses")[0])
    assert refusals[0] == refusals[1]
    assert f"{rows.total} rows [" in str(raised.value)


@pytest.mark.parametrize("kind", KINDS)
def test_the_reference_and_pose_price_reads_the_guiders_extras(monkeypatch, tmp_path, kind):
    """Wan SCAIL's reference latents and pose video ride in the positive entry's extras."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="uly2")
    model = ModelSpec(mesh=rig_.model.mesh, unet_name=rig_.model.unet_name)
    latent = {"samples": torch.zeros(1, 16, 21, 60, 104)}
    extras = {"reference_latents": [torch.zeros(1, 16, 1, 60, 104)],
              "pose_video_latent": torch.zeros(1, 16, 21, 60, 104)}
    shapes = [render_rows.ref_pose_shapes(_request(monkeypatch, node, model, extras, latent))
              for node in ("ksampler", kind)]
    assert shapes[0] == shapes[1]
    assert shapes[1] == ([torch.Size([1, 16, 1, 60, 104])], torch.Size([1, 16, 21, 60, 104]))


@pytest.mark.parametrize("kind", KINDS)
def test_the_sample_deadline_counts_the_guiders_rows(monkeypatch, tmp_path, kind):
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="uly2")
    monkeypatch.setattr(render_rows, "sniff_checkpoint", lambda _path: ("minimax_h3", "int8"))
    model = ModelSpec(mesh=rig_.model.mesh, unet_name=rig_.model.unet_name)
    latent = _h3_latent()
    extras = {h3_rows.REFS_KEY: REFS}
    counted = [render_rows.packed_rows_for_request(
        model, _request(monkeypatch, node, model, extras, latent), latent)
        for node in ("ksampler", kind)]
    assert counted[0] == counted[1]
    assert counted[1][0] == h3_rows.rows_from_request(
        {"positive": _cond(extras)}, latent).total


@pytest.mark.parametrize("kind", KINDS)
def test_the_driver_footprint_reports_the_guiders_reference_blocks(monkeypatch, tmp_path, kind):
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="uly2")
    model = ModelSpec(mesh=rig_.model.mesh, unet_name=rig_.model.unet_name)
    profile = driver_footprint.DRIVER_STACK_FAMILIES["minimax_h3"]
    extras = {h3_rows.REFS_KEY: REFS}
    volumes = [driver_footprint.ref_pixel_volume_from_latents(
        _request(monkeypatch, node, model, extras, _h3_latent()), profile)
        for node in ("ksampler", kind)]
    assert volumes[0] == volumes[1] and volumes[1][1] == 9
