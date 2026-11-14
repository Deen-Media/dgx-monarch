"""p-h3-image's image graph through the MiniMax H3 sharded forward equals stock (real ComfyUI, CPU).

p-h3-image (ComfyUI #17506) adds no DiT class: ComfyUI builds the checkpoint as
``MiniMaxH3`` from H3's keys, with ``adaln_t_table`` holding a 4097 by 16 curve
basis, and ``PH3ImageConditioning`` hands it one video latent frame beside two
audio latent frames, plus one image reference per edit input. The H3 adapter
(adapters/minimax_h3.py) owns that forward under sequence parallelism and
rebuilds the packed stream itself, so this file runs the bound forward against
stock ``MiniMaxH3Model.forward`` on a tiny real model at world 1 and asks for
the same bits: text-to-image, an odd latent that pads to the patch, and edits
with one and two references, at the shifts the workflows use (1.0 and 1.0)
and at H3's own.

Identity shard and gather stand in for world 1, and the attention override is
dropped, so the blocks run stock attention in both; what is compared is the
adapter's own packing, condition splice, modulation plan, timestep curve and
head slicing. The sharded split and its pad exclusion have their own tests
(tests/test_minimax_h3.py); the uly2 compare on hardware is the maintainer's
(docs/VALIDATION.md, Release 1.5.0 checks).
"""
from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import minimax_h3 as adapter_module
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.minimax_h3 import MiniMaxH3Adapter
from module_location_helpers import from_checkout

COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
# p-h3-image's curve basis, read from the release's header (docs/MODELS.md).
CURVE_GRID, CURVE_DIM = 4097, 16
TEXT_DIM = 40


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy", "nodes", "node_helpers") or name.startswith(("comfy.", "comfy_"))


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
        sys.argv = ["pytest-p-h3-image-exactness", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            pytest.importorskip("comfy.ldm.minimax.model")
            try:
                # Gated on comfy.* above: tests/test_release_surfaces.py
                # admits only comfy.* import gates outside the dev extra.
                from comfy_extras.nodes_minimax_h3 import PH3ImageConditioning  # noqa: F401
            except ImportError:
                pytest.skip("this ComfyUI predates p-h3-image (ComfyUI 0.44.0 added it)")
            import comfy.ldm.minimax.model as h3
            import comfy.model_base
            import comfy.ops
        finally:
            sys.argv = argv_before
        yield SimpleNamespace(h3=h3, model_base=comfy.model_base, ops=comfy.ops)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _tiny(comfy):
    """A two-block MiniMax H3 DiT in p-h3-image's curve form, with seeded weights."""
    torch.manual_seed(0)
    model = comfy.h3.MiniMaxH3Model(
        hidden_size=48, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
        attention_head_dim=24, ffn_hidden_size=64, text_dim=TEXT_DIM, time_embed_dim=CURVE_DIM,
        rope_inv_freq_len=4, adaln_curve_grid=CURVE_GRID, dtype=torch.float32,
        operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if name.endswith("inv_freq"):
                tensor.copy_(1.0 / (10000 ** (torch.arange(tensor.numel()) / tensor.numel())))
            elif "norm" in name and name.endswith("weight"):
                tensor.copy_(1.0 + 0.1 * torch.randn_like(tensor))
            else:
                tensor.copy_(0.2 * torch.randn_like(tensor))
    return model.requires_grad_(False)


def _bound(model):
    clone = copy.deepcopy(model)
    MiniMaxH3Adapter().inject_usp(clone, InjectionContext(topology_sp=1, usp_attention=object()))
    return clone


@pytest.fixture
def world_one(monkeypatch):
    monkeypatch.setattr(adapter_module, "sp_world", lambda: 1)
    monkeypatch.setattr(adapter_module, "shard_seq", lambda tensor, dim=1: (tensor, tensor.shape[dim]))
    monkeypatch.setattr(adapter_module, "sp_gather", lambda tensor, _orig, dim=1: tensor)
    monkeypatch.setattr(adapter_module, "padded_row_indices", lambda _segments: [])
    monkeypatch.setattr(adapter_module, "usp_options", lambda options, _attn, drop_rows=None: dict(options))


def _payload(comfy, text_len, lat, refs, audio_scale):
    """What stock MiniMaxH3.extra_conds publishes for this conditioning."""
    payload = {"seed": 7, "audio_scale": audio_scale}
    if refs:
        payload["refs"] = refs
    payload.update(comfy.model_base.MiniMaxH3._cond_latents(None, refs))
    payload["layout"] = comfy.h3.PackedLayout(
        text_len, 1, (lat[0] + 1) // 2 * 2, (lat[1] + 1) // 2 * 2, 2, refs=payload.get("refs"))
    return payload


def _refs(count, lat):
    generator = torch.Generator().manual_seed(11)
    return [{"kind": "image", "latent_h": lat[0], "latent_w": lat[1],
             "latent": torch.randn(1, 24, 1, lat[0], lat[1], generator=generator)}
            for _ in range(count)]


CASES = [
    pytest.param((6, 6), 0, id="text-to-image"),
    pytest.param((5, 7), 0, id="odd-latent-pads"),
    pytest.param((6, 4), 1, id="edit-one-reference"),
    pytest.param((4, 6), 2, id="edit-two-references"),
]


@pytest.mark.parametrize("shifts", [(1.0, 1.0), (12.0, 3.0)], ids=["p-h3-image-shift", "h3-shift"])
@pytest.mark.parametrize("lat, ref_count", CASES)
@pytest.mark.parametrize("sigma", [1.0, 0.810083, 0.1])
def test_the_sharded_forward_equals_stock_on_the_image_graph(comfy, world_one, lat, ref_count, shifts, sigma):
    model = _tiny(comfy)
    bound = _bound(model)
    assert "_forward" in vars(bound) and "_forward" not in vars(model)
    generator = torch.Generator().manual_seed(3)
    video = torch.randn(1, 24, 1, *lat, generator=generator)
    audio = torch.randn(1, 32, 2, 2, generator=generator)
    context = torch.randn(1, 9, TEXT_DIM, generator=generator)
    timestep = torch.tensor([sigma * 1000.0])
    sample_sigmas = torch.tensor([1.0, 0.984753, 0.810083, 0.528718, 0.1, 0.0])
    refs = _refs(ref_count, lat)
    audio_scale = shifts[0] / shifts[1]

    def run(network):
        options = {"minimax_h3_sigma_shift_video": shifts[0], "minimax_h3_sigma_shift_audio": shifts[1],
                   "sample_sigmas": sample_sigmas, "sigmas": timestep / 1000.0}
        payload = _payload(comfy, context.shape[1], lat, refs, audio_scale)
        with torch.inference_mode():
            return network.forward([video.clone(), audio.clone()], timestep, context.clone(), options,
                                   minimax_payload=payload)

    stock, ours = run(model), run(bound)
    assert stock[0].shape == video.shape and stock[1].shape == audio.shape
    assert torch.equal(stock[0], ours[0]), "video velocity differs from stock"
    assert torch.equal(stock[1], ours[1]), "audio velocity differs from stock"
    assert torch.isfinite(stock[0]).all() and stock[0].abs().sum() > 0


def test_the_image_layout_packs_the_segments_the_adapter_vets(comfy):
    """One video frame, two audio frames and image references: every kind has a row source."""
    refs = _refs(2, (4, 6))
    layout = comfy.h3.PackedLayout(9, 1, 4, 6, 2, refs=refs)
    kinds = [kind for _a, _b, kind in layout.segments]
    assert set(kinds) - {"text"} <= set(adapter_module.H3_STREAM_OF)
    assert kinds.count("video") == 1 and kinds.count("audio") == 1 and kinds.count("ref_img") == 2
    video = next((a, b) for a, b, kind in layout.segments if kind == "video")
    assert video[1] - video[0] == (4 // 2) * (6 // 2)


def test_the_lora_shape_guard_admits_the_vendor_lora_layout(comfy):
    """The vendor LoRA's keys and ranks, scaled to the tiny model, keep every weight's shape.

    `p-h3-image_lora_bf16.safetensors` holds `diffusion_model.<module>.lora_A/B.weight`
    and an `alpha` for the four linears of every block and of the token refiner.
    ComfyUI's own key map and loader turn them into patches, and the guard
    (actor/lora_shape_guard.py) admits them; a patch of another shape refuses.
    """
    from comfy.lora import calculate_shape, load_lora, model_lora_keys_unet
    from comfy.model_patcher import ModelPatcher

    from dgx_monarch.actor import lora_shape_guard
    from dgx_monarch.adapters.base import UnsupportedModelError

    class Holder(torch.nn.Module):
        def __init__(self, diffusion_model):
            super().__init__()
            self.diffusion_model = diffusion_model
            # model_lora_keys_unet reads the unet config after its generic,
            # diffusers-free half; MiniMax H3 has no diffusers key map.
            self.model_config = SimpleNamespace(unet_config={"image_model": "minimax_h3"})

    holder = Holder(_tiny(comfy))
    key_map = model_lora_keys_unet(holder, {})
    lora = {}
    for prefix in ("blocks.0", "blocks.1", "token_refiner.blocks.0"):
        for linear in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            weight = holder.diffusion_model.get_submodule(f"{prefix}.{linear}").weight
            base = f"diffusion_model.{prefix}.{linear}"
            lora[f"{base}.lora_A.weight"] = torch.randn(4, weight.shape[1])
            lora[f"{base}.lora_B.weight"] = torch.randn(weight.shape[0], 4)
            lora[f"{base}.alpha"] = torch.tensor(4.0)
    patches = load_lora(lora, key_map)
    assert len(patches) == 12
    cpu = torch.device("cpu")
    patcher = ModelPatcher(holder, load_device=cpu, offload_device=cpu)
    patcher.add_patches(patches, 1.0)
    lora_shape_guard.assert_patch_shapes_fit(patcher)
    wide = {"diffusion_model.video_patch_proj.weight": ("set", (torch.zeros(48, 192),))}
    patcher.add_patches(wide, 1.0)
    if calculate_shape(patcher.patches["diffusion_model.video_patch_proj.weight"],
                                  holder.diffusion_model.video_patch_proj.weight,
                                  "diffusion_model.video_patch_proj.weight") is None:
        pytest.skip("this ComfyUI does not report a set patch's shape")
    with pytest.raises(UnsupportedModelError, match="video_patch_proj"):
        lora_shape_guard.assert_patch_shapes_fit(patcher)
