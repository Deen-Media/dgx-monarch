"""Pure driver contract for per-render SD3 model sampling and the Qwen-Image 2.1
cache node."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction
from types import SimpleNamespace

import pytest

from dgx_monarch.actor.model_store import ModelStore
from dgx_monarch.mesh_safety import request_combo_key
from dgx_monarch.model_sampling import ModelSamplingSpec, normalize_model_sampling
from dgx_monarch.nodes import NODE_CLASS_MAPPINGS
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.model_sampling import DGXMonarchModelSamplingSD3
from dgx_monarch.nodes.qwen_image21_cache import DGXMonarchQwenImage21Cache
from dgx_monarch.qwen_image21_cache import normalize_qwen_image21_cache


def _model(**kwargs) -> ModelSpec:
    mesh = MeshSpec(
        handle=SimpleNamespace(world=2),
        topology_preset="uly2",
        attention="TORCH_FLASH",
        sync_ulysses=True,
    )
    return ModelSpec(
        mesh=mesh,
        unet_name="model.safetensors",
        options={"weight_dtype": "default"},
        loras=({"name": "style.safetensors", "strength": 0.5},),
        **kwargs,
    )


@pytest.mark.parametrize("shift", [0, 0.0, 3, Fraction(11, 2), 100, 100.0])
def test_normalize_model_sampling_accepts_closed_finite_range(shift):
    source = {"kind": "sd3", "shift": shift}

    normalized = normalize_model_sampling(source)

    assert normalized == {"kind": "sd3", "shift": float(shift)}
    assert isinstance(normalized["shift"], float)
    assert normalized is not source


@pytest.mark.parametrize(
    "value",
    [
        "sd3",
        [],
        {},
        {"kind": "sd3"},
        {"shift": 3.0},
        {"kind": "sd3", "shift": 3.0, "multiplier": 1000},
        {"kind": "flux", "shift": 3.0},
        {"kind": "sd3", "shift": True},
        {"kind": "sd3", "shift": "3.0"},
        {"kind": "sd3", "shift": float("nan")},
        {"kind": "sd3", "shift": float("inf")},
        {"kind": "sd3", "shift": float("-inf")},
        {"kind": "sd3", "shift": -0.01},
        {"kind": "sd3", "shift": 100.01},
        ModelSamplingSpec(kind="flux", shift=3.0),
        ModelSamplingSpec(kind="sd3", shift=float("nan")),
    ],
)
def test_normalize_model_sampling_rejects_malformed_values(value):
    with pytest.raises(ValueError, match="model_sampling"):
        normalize_model_sampling(value)


def test_model_spec_default_and_sampling_patch_are_immutable_and_copy_safe():
    model = _model()
    patched = model.with_model_sampling({"kind": "sd3", "shift": 5})

    assert model.model_sampling is None
    assert model.request_dict()["model_sampling"] is None
    assert patched.model_sampling == ModelSamplingSpec(kind="sd3", shift=5.0)
    assert "model_sampling" not in patched.options
    with pytest.raises(FrozenInstanceError):
        patched.model_sampling.shift = 3.0

    request = patched.request_dict()
    request["model_sampling"]["shift"] = 7.0
    assert patched.request_dict()["model_sampling"] == {"kind": "sd3", "shift": 5.0}
    assert patched.with_model_sampling(None).model_sampling is None


def test_model_spec_constructor_canonicalizes_and_validates_sampling():
    model = _model(model_sampling={"kind": "sd3", "shift": 3})
    assert model.model_sampling == ModelSamplingSpec(kind="sd3", shift=3.0)

    with pytest.raises(ValueError, match="exactly"):
        _model(model_sampling={"kind": "sd3", "shift": 3.0, "extra": None})


def test_sampling_node_patches_a_new_model_and_is_registered():
    model = _model()
    patched, = DGXMonarchModelSamplingSD3().patch(model, 5.0)

    assert patched is not model
    assert patched.request_dict()["model_sampling"] == {"kind": "sd3", "shift": 5.0}
    assert model.request_dict()["model_sampling"] is None
    assert NODE_CLASS_MAPPINGS["DGXMonarchModelSamplingSD3"] is DGXMonarchModelSamplingSD3


def test_qwen21_cache_node_uses_the_dgxm_handle_and_keeps_default_off():
    model = _model()
    patched, = DGXMonarchQwenImage21Cache().patch(model, "cpu", "int8")

    assert patched is not model
    assert model.options == {"weight_dtype": "default"}
    assert patched.options["qwen_image21_cache"] == {"device": "cpu", "dtype": "int8"}
    request = patched.request_dict()
    request["options"]["qwen_image21_cache"]["device"] = "off"
    assert patched.request_dict()["options"]["qwen_image21_cache"] == {
        "device": "cpu", "dtype": "int8"}
    assert NODE_CLASS_MAPPINGS["DGXMonarchQwenImage21Cache"] is DGXMonarchQwenImage21Cache
    required = DGXMonarchQwenImage21Cache.INPUT_TYPES()["required"]
    assert required["model"] == ("DGXM_MODEL",)
    assert required["device"][1]["default"] == "off"


@pytest.mark.parametrize("value", [
    {"device": "cpu"},
    {"device": "off", "dtype": "bf16"},
    {"device": "driver", "dtype": "default"},
    {"device": "off", "dtype": "default", "extra": 1},
])
def test_qwen21_cache_wire_rejects_malformed_values(value):
    with pytest.raises(ValueError, match="qwen_image21_cache"):
        normalize_qwen_image21_cache(value)


def test_sampling_node_schema_matches_the_sd3_patch_contract():
    required = DGXMonarchModelSamplingSD3.INPUT_TYPES()["required"]

    assert list(required) == ["model", "shift"]
    assert required["model"] == ("DGXM_MODEL",)
    assert required["shift"] == (
        "FLOAT",
        {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01},
    )


def test_sampling_shift_stays_outside_combo_and_store_identity():
    default = _model().request_dict()
    image = _model().with_model_sampling({"kind": "sd3", "shift": 3.0}).request_dict()
    video = _model().with_model_sampling({"kind": "sd3", "shift": 5.0}).request_dict()

    assert request_combo_key(default) == request_combo_key(image) == request_combo_key(video)

    def store_keys(request):
        return ModelStore.make_keys(
            request["unet_name"], request["options"], request["loras"], "bf16"
        )

    assert store_keys(default) == store_keys(image) == store_keys(video)
