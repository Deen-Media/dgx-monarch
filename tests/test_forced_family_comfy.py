"""Real ComfyUI: a forced family binds exactly what its adapter takes under `auto`.

A forced ``family_adapter`` once bound by class ancestry alone, so it took
ComfyUI model classes its adapter refuses under ``auto``: Wan variants built on
another diffusion model, the HunyuanVideo back-catalogue, Lumina 2.0 and
Ming-Image under ``zimage``. Several of those forced renders were silently
wrong (docs/TROUBLESHOOTING.md #154). MiniMax-H3-Image, which bound under
``minimax_h3`` the same way, now has its own adapter registered ahead. The
rule now asks the adapter's own ``matches()`` about the model's nearest ComfyUI
class, so the two paths can differ only for a derivative of a vetted class.

These run over every class ``comfy.model_base`` defines, so a subclass a later
ComfyUI adds refuses under a force from the day it appears, as it does under
``auto``. CPU only, no weights: bare instances of the real classes. Skipped
where ComfyUI is not importable; the comfy-canary job runs them.
"""
from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

import pytest

from dgx_monarch.adapters import (
    UnsupportedModelError,
    _ancestry_owner,
    adapter_for,
    get_adapter,
)
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag
from test_dropped_scale_gate import real_comfy as real_comfy  # the real-ComfyUI fixture

# The classes a forced family bound although `auto` refused them, by family, as ComfyUI
# 0.45.0 defines them. A ComfyUI that predates one simply does not hold it.
REFUSED = {
    "wan": ("WAN21_Camera", "WAN21_CausalAR", "WAN21_HuMo", "WAN21_Vace", "WAN22_Animate",
            "WAN22_RelayVSR", "WAN22_S2V"),
    "hunyuan": ("HunyuanVideo", "HunyuanVideoI2V", "HunyuanVideoSkyreelsI2V"),
    "zimage": ("Lumina2", "MingImage"),
}


@pytest.fixture
def model_base(real_comfy):
    return sys.modules["comfy.model_base"]


def _auto(model):
    try:
        return get_adapter(model)
    except UnsupportedModelError:
        return None


def _forced(model, family):
    try:
        return adapter_for(model, family)
    except UnsupportedModelError as exc:
        tag = parse_refusal_tag(str(exc))
        assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS, str(exc)[:200]
        return None


def test_forcing_the_owning_family_binds_what_auto_binds_for_every_comfy_class(model_base):
    """Every class ``comfy.model_base`` defines that a family owns: forcing that family
    binds the adapter `auto` binds, or refuses class P where `auto` refuses."""
    bound, refused = {}, {}
    for name, cls in sorted(vars(model_base).items()):
        if not (inspect.isclass(cls) and cls.__module__ == model_base.__name__):
            continue
        model = object.__new__(cls)
        owner = _ancestry_owner(model)
        if owner is None:
            continue
        auto, forced = _auto(model), _forced(model, owner.family)
        assert forced is auto, f"{name}: forced {owner.family} gave {forced}, auto gave {auto}"
        (bound if forced is not None else refused)[name] = owner.family
    for family, names in REFUSED.items():
        for name in names:
            if hasattr(model_base, name):
                assert refused.get(name) == family, (name, bound.get(name))
    # Lumina2 is here only as a bare instance: with Z-Image's pad tokens it binds (below).
    assert len(bound) >= 30 and len(refused) >= 10, (sorted(bound), sorted(refused))
    assert bound["WAN21"] == bound["WAN22"] == "wan" and bound["Chroma"] == "chroma"


def test_the_image_adapter_binds_its_own_class_under_a_force_and_h3s_does_not(model_base):
    """MiniMax-H3-Image has an adapter that names its class and registers ahead of
    H3's. A forced ``minimax_h3_image`` binds that class and a derivative of it; a forced
    ``minimax_h3`` refuses it (the image adapter owns it); and neither forced family
    takes the other's model."""
    if not hasattr(model_base, "MiniMaxH3Image"):
        pytest.skip("this ComfyUI predates MiniMax-H3-Image (ComfyUI 0.45.0 added it)")
    image, video = object.__new__(model_base.MiniMaxH3Image), object.__new__(model_base.MiniMaxH3)
    assert type(adapter_for(image, "minimax_h3_image")).__name__ == "MiniMaxH3ImageAdapter"
    assert _auto(image) is adapter_for(image, "minimax_h3_image")
    finetune = object.__new__(type("H3ImageFinetune", (model_base.MiniMaxH3Image,), {}))
    assert type(adapter_for(finetune, "minimax_h3_image")).__name__ == "MiniMaxH3ImageAdapter"
    assert type(adapter_for(video, "minimax_h3")).__name__ == "MiniMaxH3Adapter"
    for model, family in ((image, "minimax_h3"), (finetune, "minimax_h3"),
                          (video, "minimax_h3_image")):
        assert _forced(model, family) is None, (type(model).__name__, family)


def test_a_derivative_binds_through_the_vetted_class_on_a_real_model_class(model_base):
    """The override's purpose, on ComfyUI's real ``torch.nn.Module`` classes: a class
    ComfyUI does not define binds through the vetted class it extends, the instance
    test included; a derivative of an unvetted class refuses."""
    def bare(cls, **attributes):
        model = object.__new__(cls)
        for key, value in attributes.items():
            setattr(model, key, value)
        return model

    wan = bare(type("Wan22Finetune", (model_base.WAN22,), {}))
    assert type(adapter_for(wan, "wan")).__name__ == "WanAdapter"
    assert _auto(wan) is None  # detection's exact-type test raises on the derivative itself
    vace = bare(type("VaceFinetune", (model_base.WAN21_Vace,), {}))
    with pytest.raises(UnsupportedModelError, match="a VaceFinetune, a derivative of WAN21_Vace"):
        adapter_for(vace, "wan")

    zimage = SimpleNamespace(pad_tokens_multiple=32)
    assert type(adapter_for(bare(model_base.Lumina2, diffusion_model=zimage), "zimage")).__name__ == (
        "ZImageAdapter")
    finetune = type("ZImageFinetune", (model_base.Lumina2,), {})
    model = bare(finetune, diffusion_model=zimage)
    assert type(adapter_for(model, "zimage")).__name__ == "ZImageAdapter"
    assert type(model) is finetune and model.diffusion_model is zimage  # the model is untouched
    lumina = bare(finetune, diffusion_model=SimpleNamespace(pad_tokens_multiple=None))
    with pytest.raises(UnsupportedModelError, match="a derivative of Lumina2"):
        adapter_for(lumina, "zimage")
