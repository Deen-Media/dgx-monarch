"""A LoRA patch that changes a weight's shape refuses before any bake (real ComfyUI, CPU).

``actor/lora_shape_guard.assert_patch_shapes_fit`` runs at the end of the
store's LoRA build, so it refuses before any bake, un-bake, slab mapping or
FSDP shard. Its test is ComfyUI's own: ``comfy.lora.calculate_shape(patches,
weight, key) != weight.shape``. These cases build real patches on a real
``ModelPatcher``: a "set" patch that widens ``video_embedder`` from 128 to 384
input channels (the MOSS-Image color and depth control LoRA's shape) refuses
with the key and both shapes named, and diff, LoRA and offset patches of the
right shape pass. The two widening "set" cases skip on a ComfyUI whose
``calculate_shape`` counts no "set" patch (the 0.42 releases, before master's
e3bee7ffc): that ComfyUI copies a set patch into the weight in place, so its
own merge fails on a widening set patch and the guard has nothing to refuse. The
"lora shape-changing set" seam in tests/canary/comfy_seam_contracts.py pins
the counting form on master.

The fixture forces ``--cpu`` because comfy.cli_args parses argv on import, and
skips where ComfyUI is not importable.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.actor import lora_shape_guard, model_store
from dgx_monarch.actor.model_store import ModelStore
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from module_location_helpers import from_checkout

COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
DIM, LATENT, WIDE = 3072, 128, 384


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy():
    """Import real ComfyUI, then leave sys.modules as found."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before = list(sys.path)
    argv_before = sys.argv
    if os.path.isdir(COMFY_DIR) and COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)
    try:
        sys.argv = ["pytest-lora-shape-guard", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            lora = pytest.importorskip("comfy.lora")
            model_patcher = pytest.importorskip("comfy.model_patcher")
            import comfy.sd
            import comfy.utils
        finally:
            sys.argv = argv_before
        yield SimpleNamespace(lora=lora, model_patcher=model_patcher, sd=comfy.sd, utils=comfy.utils)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


class _Tiny(torch.nn.Module):
    """MOSS-Image's input embedder at full width, plus one small fused qkv."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.video_embedder = torch.nn.Linear(LATENT, DIM)
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.layers[0].proj = torch.nn.ModuleDict({"qkv": torch.nn.Linear(64, 192, bias=False)})


def _patcher(comfy):
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(_Tiny(), load_device=cpu, offload_device=cpu)


def _class_p(exc: BaseException) -> str:
    text = str(exc)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS, text
    assert "docs/TROUBLESHOOTING.md #127" in text, text
    return text


def _skip_unless_set_patches_change_shape(comfy) -> None:
    """Skip where ComfyUI's own shape rule keeps a widening "set" patch's shape."""
    patcher = _patcher(comfy)
    key = "video_embedder.weight"
    patcher.add_patches({key: ("set", (torch.zeros(DIM, WIDE),))}, 1.0)
    weight = comfy.utils.get_attr(patcher.model, key)
    if tuple(comfy.lora.calculate_shape(patcher.patches[key], weight, key)) == tuple(weight.shape):
        pytest.skip("this ComfyUI's calculate_shape counts no set patch (before e3bee7ffc), "
                    "so its own merge fails on a widening set patch instead")


def test_no_patches_is_a_no_op_that_imports_nothing():
    lora_shape_guard.assert_patch_shapes_fit(SimpleNamespace(patches={}))
    lora_shape_guard.assert_patch_shapes_fit(SimpleNamespace())


def test_a_widening_set_patch_refuses_with_the_key_and_both_shapes(comfy):
    _skip_unless_set_patches_change_shape(comfy)
    patcher = _patcher(comfy)
    patcher.add_patches({"video_embedder.weight": ("set", (torch.zeros(DIM, WIDE),))}, 1.0)
    with pytest.raises(UnsupportedModelError) as caught:
        lora_shape_guard.assert_patch_shapes_fit(patcher)
    text = _class_p(caught.value)
    assert f"video_embedder.weight of shape ({DIM}, {LATENT}) with one of shape ({DIM}, {WIDE})" in text


def test_a_padded_diff_that_grows_a_weight_refuses_too(comfy):
    patcher = _patcher(comfy)
    patcher.add_patches(
        {"video_embedder.weight": ("diff", (torch.zeros(DIM, WIDE), {"pad_weight": True}))}, 1.0)
    with pytest.raises(UnsupportedModelError, match=r"with one of shape \(3072, 384\)"):
        lora_shape_guard.assert_patch_shapes_fit(patcher)


def test_patches_of_the_right_shape_pass(comfy):
    patcher = _patcher(comfy)
    lora_sd = {
        "video_embedder.lora_up.weight": torch.randn(DIM, 4),
        "video_embedder.lora_down.weight": torch.randn(4, LATENT),
    }
    loaded = comfy.lora.load_lora(lora_sd, {"video_embedder": "video_embedder.weight"})
    assert set(loaded) == {"video_embedder.weight"}
    patcher.add_patches(loaded, 1.0)
    patcher.add_patches({"video_embedder.bias": ("diff", (torch.zeros(DIM),))}, 1.0)
    patcher.add_patches({"video_embedder.weight": ("set", (torch.zeros(DIM, LATENT),))}, 0.5)
    # An offset patch on a third of a fused qkv, the shape ComfyUI's MOSS key map writes.
    patcher.add_patches({("layers.0.proj.qkv.weight", (0, 64, 64)): ("diff", (torch.ones(64, 64),))}, 1.0)
    assert set(patcher.patches) == {"video_embedder.weight", "video_embedder.bias",
                                    "layers.0.proj.qkv.weight"}
    lora_shape_guard.assert_patch_shapes_fit(patcher)


def test_the_store_refuses_in_the_lora_build_before_any_weight_moves(comfy, monkeypatch):
    _skip_unless_set_patches_change_shape(comfy)
    base = _patcher(comfy)
    before = base.model.video_embedder.weight.detach().clone()

    def load_lora_for_models(model, clip, lora, strength_model, strength_clip):
        patched = model.clone()
        patched.add_patches({"video_embedder.weight": ("set", (lora["video_embedder.weight"],))},
                            strength_model)
        return patched, clip

    monkeypatch.setattr(model_store, "resolve_model_path", lambda kind, name: f"/home/user/{name}")
    monkeypatch.setattr(comfy.utils, "load_torch_file",
                        lambda path, safe_load=True: {"video_embedder.weight": torch.zeros(DIM, WIDE)})
    monkeypatch.setattr(comfy.sd, "load_lora_for_models", load_lora_for_models)
    store = ModelStore.__new__(ModelStore)
    store.lora_low_rss = False
    with pytest.raises(UnsupportedModelError) as caught:
        store._build_active(base, [{"name": "control.safetensors", "strength": 1.0}])
    _class_p(caught.value)
    weight = base.model.video_embedder.weight
    assert weight.shape == (DIM, LATENT)
    assert torch.equal(weight.detach(), before)
