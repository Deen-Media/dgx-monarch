"""ModelSamplingMiniMaxH3 (DGX Monarch): the wire, the node, the driver preflight and the Worker belt.

The comfy half (tests/test_h3_sigma_shift_comfy.py) holds the widgets, the
keys and the latent shape to ComfyUI's own code. This half needs no ComfyUI.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from dgx_monarch import h3_sigma_shift
from dgx_monarch.actor import model_sampling_render
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.model_sampling import (
    ModelSamplingSpec,
    freeze_model_sampling,
    normalize_model_sampling,
    sampling_kind,
)
from dgx_monarch.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, model_patch_preflight
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.model_sampling import DGXMonarchModelSamplingMiniMaxH3
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

H3_WIRE = {"kind": "minimax_h3", "shift_video": 1.0, "shift_audio": 1.0}


def _model(**kwargs) -> ModelSpec:
    mesh = MeshSpec(handle=SimpleNamespace(world=2), topology_preset="uly2",
                    attention="TORCH_FLASH", sync_ulysses=True)
    return ModelSpec(mesh=mesh, unet_name="p-h3-image_bf16.safetensors", **kwargs)


class _Video:
    def __init__(self, *shape):
        self.shape = shape


def _latent(latent_t: int) -> dict:
    return {"samples": SimpleNamespace(tensors=[_Video(1, 24, latent_t, 182, 182),
                                                _Video(1, 32, 2, 2)])}


@pytest.fixture
def family(monkeypatch):
    """Make the driver's header sniff answer one family for every checkpoint."""
    box = {"family": "minimax_h3"}

    def sniff(_name, _override):
        return None if box["family"] is None else (box["family"], "bf16")

    monkeypatch.setattr(model_patch_preflight, "_sniffed_checkpoint", sniff)
    return box


# --- the wire ---------------------------------------------------------------


@pytest.mark.parametrize("shifts", [(0.01, 0.01), (1, 1), (12.0, 3.0), (100, 100.0)])
def test_the_h3_wire_accepts_stock_range(shifts):
    source = {"kind": "minimax_h3", "shift_video": shifts[0], "shift_audio": shifts[1]}
    normalized = normalize_model_sampling(source)
    assert normalized == {"kind": "minimax_h3", "shift_video": float(shifts[0]),
                          "shift_audio": float(shifts[1])}
    assert normalized is not source
    spec = freeze_model_sampling(source)
    assert spec == ModelSamplingSpec(kind="minimax_h3", shift=float(shifts[0]), shift_audio=float(shifts[1]))
    assert normalize_model_sampling(spec) == normalized
    assert sampling_kind(spec) == sampling_kind(normalized) == "minimax_h3"


@pytest.mark.parametrize("value", [
    {"kind": "minimax_h3", "shift_video": 1.0},
    {"kind": "minimax_h3", "shift": 1.0},
    {"kind": "minimax_h3", "shift_video": 1.0, "shift_audio": 1.0, "shift": 1.0},
    {"kind": "minimax_h3", "shift_video": 0.0, "shift_audio": 1.0},
    {"kind": "minimax_h3", "shift_video": 1.0, "shift_audio": 100.01},
    {"kind": "minimax_h3", "shift_video": True, "shift_audio": 1.0},
    {"kind": "minimax_h3", "shift_video": "1.0", "shift_audio": 1.0},
    {"kind": "minimax_h3", "shift_video": float("nan"), "shift_audio": 1.0},
    {"kind": "sd3", "shift_video": 1.0, "shift_audio": 1.0},
    ModelSamplingSpec(kind="minimax_h3", shift=1.0),
    ModelSamplingSpec(kind="sd3", shift=3.0, shift_audio=1.0),
])
def test_the_h3_wire_refuses_malformed_values(value):
    with pytest.raises(ValueError, match="model_sampling"):
        normalize_model_sampling(value)


def test_the_sd3_wire_is_unchanged():
    assert normalize_model_sampling({"kind": "sd3", "shift": 3}) == {"kind": "sd3", "shift": 3.0}
    assert ModelSamplingSpec(kind="sd3", shift=3.0).to_wire() == {"kind": "sd3", "shift": 3.0}
    assert sampling_kind(None) is None and sampling_kind("sd3") is None


# --- the node ---------------------------------------------------------------


def test_the_node_is_registered_and_mirrors_the_stock_constants():
    assert NODE_CLASS_MAPPINGS["DGXMonarchModelSamplingMiniMaxH3"] is DGXMonarchModelSamplingMiniMaxH3
    assert NODE_DISPLAY_NAME_MAPPINGS["DGXMonarchModelSamplingMiniMaxH3"] == h3_sigma_shift.NODE_NAME
    required = DGXMonarchModelSamplingMiniMaxH3.INPUT_TYPES()["required"]
    assert list(required) == ["model", "shift_video", "shift_audio"]
    assert required["shift_video"] == ("FLOAT", h3_sigma_shift.SHIFT_VIDEO)
    assert required["shift_audio"] == ("FLOAT", h3_sigma_shift.SHIFT_AUDIO)
    assert "#144" in DGXMonarchModelSamplingMiniMaxH3.DESCRIPTION


def test_the_node_puts_both_shifts_on_the_model_request():
    model = _model()
    patched, = DGXMonarchModelSamplingMiniMaxH3().patch(model, 1.0, 1.0)
    assert model.model_sampling is None
    assert patched.model_sampling == ModelSamplingSpec(kind="minimax_h3", shift=1.0, shift_audio=1.0)
    assert patched.request_dict()["model_sampling"] == H3_WIRE
    # A later sampling node replaces an earlier one, as stock's do.
    again, = DGXMonarchModelSamplingMiniMaxH3().patch(patched, 2.0, 0.5)
    assert again.request_dict()["model_sampling"] == {
        "kind": "minimax_h3", "shift_video": 2.0, "shift_audio": 0.5}


# --- the driver preflight ---------------------------------------------------


def _refusal(model, latent, request=None) -> str:
    with pytest.raises(UnsupportedModelError) as caught:
        model_patch_preflight.preflight_h3_sigma_shift(model, request or {}, latent)
    message = str(caught.value)
    tag = parse_refusal_tag(message)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert "#144" in message
    return message


def test_a_one_frame_h3_latent_without_the_shift_refuses_with_the_node_named(family):
    message = _refusal(_model(), _latent(1))
    assert h3_sigma_shift.NODE_NAME in message and "1.0 and 1.0" in message
    assert "12.0" in message and "3.0" in message


def test_an_sd3_shift_on_the_image_latent_says_why_it_does_not_count(family):
    model = _model().with_model_sampling({"kind": "sd3", "shift": 3.0})
    assert "SD3 shift" in _refusal(model, _latent(1))


def test_the_h3_shift_at_any_value_passes_the_image_latent(family):
    for shifts in ((1.0, 1.0), (12.0, 3.0)):
        model, = DGXMonarchModelSamplingMiniMaxH3().patch(_model(), *shifts)
        model_patch_preflight.preflight_h3_sigma_shift(model, {}, _latent(1))


@pytest.mark.parametrize("latent", [_latent(2), _latent(37), {"samples": _Video(1, 16, 1, 64, 64)},
                                    {"samples": None}, None, {}])
def test_video_and_plain_latents_pass_without_the_shift(family, latent):
    model_patch_preflight.preflight_h3_sigma_shift(_model(), {}, latent)


@pytest.mark.parametrize("other", [None, "wan", "krea2"])
def test_the_image_latent_arm_needs_the_h3_family(family, other):
    family["family"] = other
    model_patch_preflight.preflight_h3_sigma_shift(_model(), {}, _latent(1))


def test_the_node_on_another_family_refuses_on_the_model_and_the_negative(family):
    family["family"] = "wan"
    model, = DGXMonarchModelSamplingMiniMaxH3().patch(_model(), 1.0, 1.0)
    message = _refusal(model, _latent(21))
    assert "reads as wan" in message and "Remove the node" in message
    assert "workflow names" not in message  # the shift advice is the missing-shift arm's alone
    uncond = {"unet_name": "neg.safetensors", "model_sampling": H3_WIRE}
    assert "reads as wan" in _refusal(_model(), None, {"uncond_model": uncond})


def test_the_negative_model_reads_its_family_under_the_primary_override(family, monkeypatch):
    seen = []

    def sniff(name, override):
        seen.append((name, override))
        return ("minimax_h3", "bf16")

    monkeypatch.setattr(model_patch_preflight, "_sniffed_checkpoint", sniff)
    monkeypatch.setattr(model_patch_preflight.family_select, "override_for_model",
                        lambda model: "minimax_h3")
    uncond = {"unet_name": "neg.safetensors", "model_sampling": H3_WIRE}
    model_patch_preflight.preflight_h3_sigma_shift(_model(), {"uncond_model": uncond}, None)
    assert ("neg.safetensors", "minimax_h3") in seen


@pytest.mark.parametrize("sniffed", ["minimax_h3", None, "unknown"])
def test_the_node_on_h3_or_an_unread_header_passes(family, sniffed):
    """An unreadable header, or one no signature matches, leaves it to the Worker belt."""
    family["family"] = sniffed
    model, = DGXMonarchModelSamplingMiniMaxH3().patch(_model(), 1.0, 1.0)
    model_patch_preflight.preflight_h3_sigma_shift(model, {}, _latent(21))
    assert h3_sigma_shift.family_refusal_reason(sniffed) is None


def test_the_sd3_multiplier_guard_skips_the_h3_kind(monkeypatch):
    def boom(*_args):
        raise AssertionError("the SD3 multiplier check read an H3 shift")

    monkeypatch.setitem(__import__("sys").modules, "folder_paths",
                        SimpleNamespace(get_full_path=boom))
    model, = DGXMonarchModelSamplingMiniMaxH3().patch(_model(), 1.0, 1.0)
    model_patch_preflight.preflight_shift_node(model)
    model_patch_preflight.preflight_shift_node({"unet_name": "x", "model_sampling": H3_WIRE})


# --- the Worker belt --------------------------------------------------------


def _request(sampling=H3_WIRE, uncond=None) -> dict:
    request = {"model": {"unet_name": "p-h3-image_bf16.safetensors", "model_sampling": sampling}}
    if uncond is not None:
        request["uncond_model"] = uncond
    return request


def test_the_belt_passes_h3_and_any_render_without_the_kind(monkeypatch):
    monkeypatch.setattr(model_sampling_render, "h3_stock_node", lambda: object())
    model_sampling_render.assert_supported(_request(), "minimax_h3")
    model_sampling_render.assert_supported(_request({"kind": "sd3", "shift": 3.0}), "wan")
    model_sampling_render.assert_supported(_request(None), "wan")


@pytest.mark.parametrize("request_", [_request(), _request(None, {"model_sampling": H3_WIRE})])
def test_the_belt_refuses_another_family(monkeypatch, request_):
    monkeypatch.setattr(model_sampling_render, "h3_stock_node", lambda: object())
    with pytest.raises(UnsupportedModelError, match="reads as flux") as caught:
        model_sampling_render.assert_supported(request_, "flux")
    tag = parse_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS


def test_the_belt_and_the_apply_refuse_a_comfy_without_the_stock_node(monkeypatch):
    monkeypatch.setattr(model_sampling_render, "h3_stock_node", lambda: None)
    for call in (lambda: model_sampling_render.assert_supported(_request(), "minimax_h3"),
                 lambda: model_sampling_render.apply_model_sampling(object(), H3_WIRE)):
        with pytest.raises(UnsupportedModelError) as caught:
            call()
        message = str(caught.value)
        # The node has shipped with MiniMax H3 since ComfyUI 0.30.0, so no version is the fix.
        assert h3_sigma_shift.STOCK_NODE_PATH in message and "0.30.0" in message
        assert "0.44.0" not in message and "workflow names" not in message
        tag = parse_refusal_tag(message)
        assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS


def _image_request(sampling=None, latent_t=1) -> dict:
    request = _request(sampling)
    request["latent"] = _latent(latent_t)
    return request


def test_the_belt_refuses_the_image_latent_without_an_h3_shift_on_h3(monkeypatch):
    """Every Worker repeats the driver's missing-shift check from the loaded model's family."""
    monkeypatch.setattr(model_sampling_render, "h3_stock_node", lambda: object())
    for sampling, extra in ((None, ""), ({"kind": "sd3", "shift": 3.0}, "SD3 shift")):
        with pytest.raises(UnsupportedModelError) as caught:
            model_sampling_render.assert_supported(_image_request(sampling), "minimax_h3")
        message = str(caught.value)
        assert h3_sigma_shift.NODE_NAME in message and "1.0 and 1.0" in message
        assert extra in message and "workflow names" in message and "#144" in message
        tag = parse_refusal_tag(message)
        assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS


def test_the_belt_passes_the_image_latent_with_the_shift_or_off_h3(monkeypatch):
    monkeypatch.setattr(model_sampling_render, "h3_stock_node", lambda: object())
    model_sampling_render.assert_supported(_image_request(H3_WIRE), "minimax_h3")
    model_sampling_render.assert_supported(_image_request(None), "ltx")
    model_sampling_render.assert_supported(_image_request(None, latent_t=2), "minimax_h3")
    model_sampling_render.assert_supported(_request(None), "minimax_h3")


def test_fleet_runs_the_shift_preflight_before_any_dispatch(monkeypatch):
    from dgx_monarch.nodes import fleet

    calls = []

    def stop(model, request, latent):
        calls.append((model, dict(request), latent))
        raise RuntimeError("stopped at the H3 shift preflight")

    monkeypatch.setattr(fleet.model_patch_preflight, "preflight_h3_sigma_shift", stop)
    monkeypatch.setattr(fleet, "ensure_live", lambda handle: pytest.fail("dispatched"))
    spec = _model()
    with pytest.raises(RuntimeError, match="H3 shift preflight"):
        fleet.DGXMonarchFleetKSampler().fleet(
            [spec], [object()], [[]], [_latent(1)], ["a kite"], [42], [8], [1.0],
            ["euler"], ["simple"])
    (model, request, latent), = calls
    assert model is spec and request == {}
    assert h3_sigma_shift.is_image_latent(latent)


def test_no_wire_returns_the_clone_untouched():
    render = object()
    assert model_sampling_render.apply_model_sampling(render, None) is render
