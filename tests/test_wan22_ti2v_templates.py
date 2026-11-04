"""The Wan 2.2 TI2V 5B template keeps the official settings and the topologies the model admits.

ComfyUI's own template (``video_wan2_2_5B_ti2v`` in comfyui-workflow-templates)
renders 1280x704, 121 frames, 20 uni_pc steps on the simple schedule at cfg 5
and shift 8, with its Load Image bypassed, so a stock render is comparable
only with those values. The 5B model is one checkpoint (``WAN22``), and a start
image gives it per-frame timesteps (``WAN22.process_timestep``). The Wan
adapter expands those to tokens before it shards them; the last test drives
the adapter's forward on a tiny real ``WanModel`` at one rank against stock's,
per-frame timesteps included, and skips without ComfyUI.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import detect, wan_family
from dgx_monarch.topology import choose_auto_topology
from module_location_helpers import from_checkout

ROOT = Path(__file__).resolve().parents[1]
# The sweep converter is not installed: the comfy-canary job runs plain pytest,
# so the checkout root goes on sys.path, as tests/test_sweep_matrix.py does.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.sweep import convert  # noqa: E402

NAME = "dgx-monarch-wan22-ti2v"
UNET = "wan2.2_ti2v_5B_fp16.safetensors"
VAE = "wan2.2_vae.safetensors"
COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
# The official graph's prompt, verbatim (comfyui-workflow-templates-json 0.1.126).
PROMPT = ("Low contrast. In a retro 1970s-style subway station, a street musician plays in "
          "dim colors and rough textures. He wears an old jacket, playing guitar with focus. "
          "Commuters hurry by, and a small crowd gathers to listen. The camera slowly moves "
          "right, capturing the blend of music and city noise, with old subway signs and "
          "mottled walls in the background.")


def _graph():
    ui = json.loads((ROOT / "example_workflows" / f"{NAME}.json").read_text())
    return convert.convert(ui, convert.repo_widget_names(ROOT))


def _one(graph, kind):
    found = [(key, node["inputs"]) for key, node in graph.items() if node["class_type"] == kind]
    assert len(found) == 1, kind
    return found[0]


def test_the_template_keeps_the_official_settings():
    graph = _graph()
    init_id, init = _one(graph, "DGXMonarchInit")
    unet_id, unet = _one(graph, "DGXMonarchUNETLoader")
    shift_id, shift = _one(graph, "DGXMonarchModelSamplingSD3")
    clip_id, clip = _one(graph, "CLIPLoader")
    vae_id, vae = _one(graph, "VAELoader")
    latent_id, latent = _one(graph, "Wan22ImageToVideoLatent")
    _, sampler = _one(graph, "DGXMonarchKSampler")
    assert (init["topology"], init["mode"], init["auto_gate"]) == ("auto", "auto", "first_use")
    assert (unet["unet_name"], unet["weight_dtype"], unet["mesh"]) == (UNET, "default", [init_id, 0])
    assert (shift["shift"], shift["model"]) == (8.0, [unet_id, 0])
    assert (clip["clip_name"], clip["type"]) == ("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan")
    assert vae["vae_name"] == VAE
    assert (latent["width"], latent["height"], latent["length"], latent["batch_size"]) == (1280, 704, 121, 1)
    assert latent["vae"] == [vae_id, 0] and "start_image" not in latent
    assert (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"],
            sampler["denoise"]) == (20, 5.0, "uni_pc", "simple", 1.0)
    assert (sampler["model"], sampler["latent_image"]) == ([shift_id, 0], [latent_id, 0])
    texts = {key: node["inputs"]["text"] for key, node in graph.items() if node["class_type"] == "CLIPTextEncode"}
    assert texts[sampler["positive"][0]] == PROMPT
    assert texts[sampler["negative"][0]].endswith("倒着走")
    assert {node["inputs"]["clip"][0] for node in graph.values() if node["class_type"] == "CLIPTextEncode"} == {clip_id}
    _, video = _one(graph, "CreateVideo")
    _, save = _one(graph, "SaveVideo")
    assert (video["fps"], video["codec"]) == (24.0, "none")
    assert save["filename_prefix"] == "video/dgx_monarch_wan22_ti2v"


def test_the_note_names_the_files_and_the_image_to_video_wiring():
    ui = json.loads((ROOT / "example_workflows" / f"{NAME}.json").read_text())
    note = next(node["widgets_values"][0] for node in ui["nodes"] if node["type"] == "Note")
    for phrase in (UNET, VAE, "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "start_image",
                   "27280 tokens", "VAE Decode (Tiled)"):
        assert phrase in note, phrase


def test_the_manifest_lists_the_three_files_the_template_loads():
    manifest = (ROOT / "example_workflows" / "artifacts.toml").read_text()
    block = manifest.split(f'"{NAME}" = [', 1)[1].split("]", 1)[0]
    assert [line.strip().strip('",') for line in block.strip().splitlines()] == [
        f"diffusion_models/{UNET}", "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", f"vae/{VAE}"]


def test_the_canvas_halves_at_world_2_and_4_with_no_pad():
    latent_t, lat_h, lat_w = (121 - 1) // 4 + 1, 704 // 16, 1280 // 16
    tokens = latent_t * (lat_h // 2) * (lat_w // 2)
    heads = 3072 // 128
    assert (latent_t, tokens, heads) == (31, 27280, 24)
    for degree in (2, 4):
        assert tokens % degree == 0 and heads % degree == 0


def test_auto_runs_the_5b_model_on_uly2_by_the_wan_row():
    decision = choose_auto_topology("wan", "fp16", 1280 * 704 / 1e6, 2, cfg_value=5.0)
    assert (decision.topology.ulysses, decision.topology.cfg, decision.topology.dp) == (2, 1, 1)
    assert "table row 50" in decision.reason


def test_the_wan_adapter_admits_the_5b_model_base_exactly():
    assert "WAN22" in wan_family._WAN_SUPPORTED_EXACT


def test_the_checkpoint_header_reads_as_wan_fp16():
    path = Path(os.path.expanduser("~/ComfyUI/models/diffusion_models")) / UNET
    if not path.is_file():
        pytest.skip(f"no models/diffusion_models/{UNET} on this machine")
    with path.open("rb") as handle:
        header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
    header.pop("__metadata__", None)
    assert {value["dtype"] for value in header.values()} == {"F16"}
    assert header["patch_embedding.weight"]["shape"] == [3072, 48, 1, 2, 2]
    assert detect.sniff_checkpoint(str(path)) == ("wan", "fp16")


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy():
    """Import real ComfyUI with --cpu, then leave sys.modules as found."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before, argv_before = list(sys.path), sys.argv
    if os.path.isdir(COMFY_DIR) and COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)
    try:
        sys.argv = ["pytest-wan22-ti2v", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            import comfy.ldm.modules.attention as attention
            import comfy.ops
            from comfy.ldm.wan.model import WanModel
        finally:
            sys.argv = argv_before
        yield SimpleNamespace(WanModel=WanModel, ops=comfy.ops, attention=attention)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _one_rank_xfuser(monkeypatch):
    group = SimpleNamespace(all_gather=lambda tensor, dim: tensor)
    distributed = SimpleNamespace(get_sequence_parallel_world_size=lambda: 1,
                                  get_sequence_parallel_rank=lambda: 0,
                                  get_ulysses_parallel_world_size=lambda: 1,
                                  get_ring_parallel_world_size=lambda: 1,
                                  get_sp_group=lambda: group)
    for name, module in (("xfuser", SimpleNamespace()), ("xfuser.core", SimpleNamespace()),
                         ("xfuser.core.distributed", distributed)):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("timestep", [[0.6], [[0.0, 0.6, 0.6]]], ids=["text-to-video", "start-image"])
def test_the_sharded_forward_matches_stock_with_per_frame_timesteps(comfy, monkeypatch, timestep):
    """A start image's per-frame timesteps reach every token as stock's repeat_e gives them."""
    from dgx_monarch.adapters.base import InjectionContext

    torch.manual_seed(5)
    model = comfy.WanModel(model_type="t2v", patch_size=(1, 2, 2), text_len=4, in_dim=8, dim=16,
                           ffn_dim=32, freq_dim=2, text_dim=4, out_dim=8, num_heads=2, num_layers=2,
                           device="cpu", dtype=torch.float32, operations=comfy.ops.disable_weight_init).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape) * 0.1)
    x, context, t = torch.randn(1, 8, 3, 4, 6), torch.randn(1, 4, 4), torch.tensor(timestep)
    with torch.no_grad():
        stock = model._forward(x, t, context)

    def one_rank_attention(q, k, v, heads, *args, drop_rows=None, **kwargs):
        assert not drop_rows
        return comfy.attention.optimized_attention(q, k, v, heads, *args, **kwargs)

    _one_rank_xfuser(monkeypatch)
    wan_family.WanAdapter().inject_usp(model, InjectionContext(topology_sp=1, usp_attention=one_rank_attention))
    assert getattr(model.forward_orig, "__func__", model.forward_orig).__name__ == "usp_forward_orig"
    with torch.no_grad():
        sharded = model._forward(x, t, context)
    assert torch.equal(sharded, stock)
