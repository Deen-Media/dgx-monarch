"""Auto-table megapixels read the model's own latent ratio.

The ratio comes from the latent format of the model config comfy's own
detection names for the checkpoint header. The fallback guess (8 pixels per
latent cell, 1 for a 4D 3-channel latent) reads 16x families at a quarter of
their megapixels: the 2026-09 stable-source campaign logged Flux2 1024x1024 at
"0.3 MP" and Lens 1344x1344 at "0.5 MP", both on the fallback row, while the
sweep matrix predicted from the graph's pixel size.

File order matters: the pure helpers and the stubbed resolutions run first,
then the matrix agreement over every shipped template with a finite
megapixel row, and the real-comfy tests last, because their module-scoped
fixture holds a real ``comfy`` in ``sys.modules`` until this module ends.
The real-comfy half skips where ComfyUI is not importable. The comfy-canary
job runs it against comfy master and names the checkout, so there a missing
checkout or a failing import fails instead (``_comfy_checkout``).
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from dgx_monarch import latent_scale, render_memory_price
from dgx_monarch.nodes import common
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.topology import choose_auto_topology

# The comfy model config each shipped finite-row template's checkpoint
# detects as, and the spatial ratio its latent format declares. The real-comfy
# test at the bottom pins every ratio here against comfy's own classes; the
# agreement test uses them without comfy.
CONFIG_RATIO = {
    "Flux2": 16, "Lens": 16, "Chroma": 8, "ChromaRadiance": 1, "Flux": 8,
    "LongCatImage": 8, "QwenImage": 8, "Krea2": 8, "ZImage": 8,
    "ZImagePixelSpace": 1,
}
# Other families, all on full-range rows: their decision cannot move, only the
# megapixels their reason line prints.
OTHER_CONFIG_RATIO = {
    "Ideogram4": 16, "ErnieImage": 16, "MageFlow": 16, "QwenImage21": 16,
    "HunyuanImage21": 32, "HunyuanVideo15": 16, "LTXV": 32, "LTXAV": 32,
    "MiniMaxH3": 16, "WAN22_T2V": 16, "WAN21_T2V": 8, "PixelDiTT2I": 1,
}
# Every shipped template whose family has a finite megapixel row, with the
# auto family and the comfy config its checkpoint detects as (read from the
# real headers on the operator rig, 2026-10-05).
FINITE_ROW_TEMPLATES = {
    "dgx-monarch-flux2-t2i": ("flux2", "Flux2"),
    "dgx-monarch-test-flux2-lora": ("flux2", "Flux2"),
    "dgx-monarch-lens-t2i": ("lens", "Lens"),
    "dgx-monarch-chroma-t2i": ("chroma", "Chroma"),
    "dgx-monarch-test-chroma-even": ("chroma", "Chroma"),
    "dgx-monarch-test-chroma-lora": ("chroma", "Chroma"),
    "dgx-monarch-radiance-t2i": ("chroma", "ChromaRadiance"),
    "dgx-monarch-flux1-t2i": ("flux", "Flux"),
    "dgx-monarch-dual-spark-split": ("flux", "Flux"),
    "dgx-monarch-fleet": ("flux", "Flux"),
    "dgx-monarch-identity-gate": ("flux", "Flux"),
    "dgx-monarch-lora-low-rss": ("flux", "Flux"),
    "dgx-monarch-quickstart": ("flux", "Flux"),
    "dgx-monarch-longcat-t2i": ("longcat", "LongCatImage"),
    "dgx-monarch-qwen-image-t2i": ("qwen_image", "QwenImage"),
    "dgx-monarch-test-qwen-image-lora": ("qwen_image", "QwenImage"),
    "dgx-monarch-krea2-t2i": ("krea2", "Krea2"),
    "dgx-monarch-test-krea2-lora": ("krea2", "Krea2"),
    "dgx-monarch-zimage-t2i": ("zimage", "ZImage"),
    "dgx-monarch-test-zimage-lora": ("zimage", "ZImage"),
    "dgx-monarch-zimage-dct-t2i": ("zimage", "ZImagePixelSpace"),
}
# (channels, pixels per latent cell) of the latent each stock node builds:
# nodes.py EmptyLatentImage, comfy_extras/nodes_sd3.py, nodes_flux.py and
# nodes_chroma_radiance.py. On these templates the node divides by its model's
# own ratio, so the workers sample that grid as built (docs/VALIDATION.md,
# "Worker empty-latent size tags", 2026-10-05).
LATENT_NODES = {
    "EmptyLatentImage": (4, 8), "EmptySD3LatentImage": (16, 8),
    "EmptyFlux2LatentImage": (128, 16), "EmptyChromaRadianceLatentImage": (3, 1),
}
_ROW = re.compile(r"table row (\d+)")


@pytest.fixture(autouse=True)
def _fresh_detection_cache():
    render_memory_price._detect_by_identity.cache_clear()
    yield
    render_memory_price._detect_by_identity.cache_clear()


class _Nested:
    """comfy.nested_tensor.NestedTensor's shape surface: first tensor's shape,
    largest ndim. The real class is exercised in the real-comfy half."""

    def __init__(self, tensors):
        self.tensors = list(tensors)

    @property
    def shape(self):
        return self.tensors[0].shape

    @property
    def ndim(self):
        return max(t.ndim for t in self.tensors)


def _h3_latent():
    # nodes_minimax_h3._empty_av_latent at 1344x768, 124 frames: video
    # [B, 24, T, H/16, W/16] first, audio [B, 32, 2, t] second.
    return (torch.zeros(1, 24, 31, 768 // 16, 1344 // 16), torch.zeros(1, 32, 2, 207))


@pytest.mark.parametrize(("shape", "ratio", "pixels"), [
    ((1, 16, 128, 128), 8, 1024 * 1024),          # 4D, 8x (Flux, Chroma)
    ((2, 16, 128, 128), 8, 1024 * 1024),          # batch never enters
    ((1, 128, 64, 64), 16, 1024 * 1024),          # 4D, 16x (Flux2)
    ((1, 128, 84, 84), 16, 1344 * 1344),          # 4D, 16x (Lens)
    ((1, 3, 1024, 1024), 1, 1024 * 1024),         # 3-channel pixel space
    ((1, 16, 21, 60, 104), 8, 480 * 832),         # 5D video: time never enters
    ((1, 48, 31, 44, 80), 16, 704 * 1280),        # 5D video, 16x (Wan 2.2 5B)
], ids=["4d-8x", "4d-8x-batch2", "4d-16x-flux2", "4d-16x-lens", "rgb-1x",
        "5d-8x", "5d-16x"])
def test_latent_megapixels_are_per_frame_spatial_pixels(shape, ratio, pixels):
    assert latent_scale.latent_megapixels(torch.zeros(shape), ratio) == pixels / 1e6


def test_a_packed_latent_reads_its_video_stream_per_frame():
    nested = _Nested(_h3_latent())
    assert latent_scale.latent_megapixels(nested, 16) == 1344 * 768 / 1e6
    # The fallback guess reads the same grid at 8x: a quarter of the frame.
    assert latent_scale.legacy_spatial_downscale(nested) == 8


@pytest.mark.parametrize(("shape", "expected"), [
    ((1, 3, 64, 64), 1),
    ((1, 16, 64, 64), 8),
    ((1, 128, 64, 64), 8),     # the 16x misread, kept only as the fallback
    ((1, 3, 5, 64, 64), 8),    # only a 4D 3-channel latent is pixel space
])
def test_the_legacy_fallback_is_the_pre_516_guess(shape, expected):
    assert latent_scale.legacy_spatial_downscale(torch.zeros(shape)) == expected


@pytest.mark.parametrize(("config", "expected"), [
    (SimpleNamespace(latent_format=SimpleNamespace(spacial_downscale_ratio=16)), 16),
    (SimpleNamespace(latent_format=SimpleNamespace(spacial_downscale_ratio=16.0)), 16),
    (SimpleNamespace(latent_format=SimpleNamespace(spacial_downscale_ratio=1)), 1),
    (SimpleNamespace(latent_format=SimpleNamespace(spacial_downscale_ratio=0)), None),
    (SimpleNamespace(latent_format=SimpleNamespace(spacial_downscale_ratio=True)), None),
    (SimpleNamespace(latent_format=SimpleNamespace(spacial_downscale_ratio=2.5)), None),
    (SimpleNamespace(latent_format=SimpleNamespace(spacial_downscale_ratio="16")), None),
    (SimpleNamespace(latent_format=SimpleNamespace()), None),
    (SimpleNamespace(), None),
    (None, None),
])
def test_config_ratio_accepts_only_a_positive_integer(config, expected):
    assert latent_scale.config_spatial_downscale(config) == expected


def _config(ratio):
    return SimpleNamespace(latent_format=SimpleNamespace(spacial_downscale_ratio=ratio))


def test_the_comfy_ratio_is_cached_by_file_identity(monkeypatch, tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"header")
    calls = []

    def detect(name):
        calls.append(name)
        return _config(16)

    monkeypatch.setattr(render_memory_price, "_detect_uncached", detect)
    assert latent_scale.comfy_spatial_downscale(str(path)) == 16
    assert latent_scale.comfy_spatial_downscale(str(path)) == 16
    assert calls == [str(path)]
    # A rewrite is a different identity, so it is read again.
    path.write_bytes(b"a longer header")
    assert latent_scale.comfy_spatial_downscale(str(path)) == 16
    assert len(calls) == 2


def test_a_failed_read_falls_open_and_is_retried(monkeypatch, tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"header")

    def absent(_name):
        raise ImportError("no comfy on this process")

    monkeypatch.setattr(render_memory_price, "_detect_uncached", absent)
    assert latent_scale.comfy_spatial_downscale(str(path)) is None
    monkeypatch.setattr(render_memory_price, "_detect_uncached",
                        lambda _name: _config(16))
    assert latent_scale.comfy_spatial_downscale(str(path)) == 16
    assert latent_scale.comfy_spatial_downscale(str(tmp_path / "missing")) is None


def test_the_ratio_and_the_render_memory_price_share_one_detection(
        monkeypatch, tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"header")
    calls = []

    def detect(name):
        calls.append(name)
        config = _config(16)
        config.memory_usage_factor = 2.8
        return config

    monkeypatch.setattr(render_memory_price, "_detect_uncached", detect)
    assert latent_scale.comfy_spatial_downscale(str(path)) == 16
    found = render_memory_price.comfy_render_memory_bound(str(path), (1, 128, 64, 64))
    assert found is not None and found[1]["memory_usage_factor"] == 2.8
    assert calls == [str(path)]


def test_no_comfy_ratio_falls_back_to_the_legacy_guess(monkeypatch, tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"header")
    monkeypatch.setattr(render_memory_price, "detect_comfy_model_config",
                        lambda _name: None)
    assert latent_scale.spatial_downscale_for(str(path), torch.zeros(1, 128, 8, 8)) == 8
    assert latent_scale.spatial_downscale_for(str(path), torch.zeros(1, 3, 8, 8)) == 1
    monkeypatch.setattr(render_memory_price, "detect_comfy_model_config",
                        lambda _name: _config(16))
    assert latent_scale.spatial_downscale_for(str(path), torch.zeros(1, 128, 8, 8)) == 16


def _auto_model(weight_dtype: str = "default") -> ModelSpec:
    return ModelSpec(
        mesh=MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                      attention="TORCH_FLASH", sync_ulysses=True),
        unet_name="model.safetensors", options={"weight_dtype": weight_dtype},
    )


def _resolve(monkeypatch, family, quant, ratio, samples, cfg=4.0, **kwargs):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: "/models/model.safetensors"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(common, "sniff_checkpoint_for_topology",
                        lambda _path: (family, quant))
    monkeypatch.setattr(latent_scale, "comfy_spatial_downscale", lambda _path: ratio)
    monkeypatch.setattr(common, "_impl_warned", set())
    topology, sage, reason = common.resolve_topology(
        _auto_model(), samples, cfg_value=cfg, **kwargs)
    return topology, sage, reason, int(_ROW.search(reason).group(1))


@pytest.mark.parametrize(("family", "quant", "grid", "row", "sage", "megapixels"), [
    # The shipped Lens graph, 1344x1344: the rows comfy's 16x ratio makes
    # reachable. Both keep uly2 and the declared kernel, as the fallback row does.
    ("lens", "fp8", 84, 74, False, "1.8 MP"),
    ("lens", "bf16", 84, 75, False, "1.8 MP"),
    # The shipped Flux2 graph, 1024x1024: 1.05 MP stays under row 27's 1.2.
    ("flux2", "fp8", 64, 99, False, "1.0 MP"),
    ("flux2", "bf16", 64, 99, False, "1.0 MP"),
    # Flux2 at 1536x1536 reaches its row.
    ("flux2", "fp8", 96, 27, False, "2.4 MP"),
])
def test_auto_reads_16x_latents_at_their_rendered_size(
        monkeypatch, family, quant, grid, row, sage, megapixels):
    topology, selected_sage, reason, chosen = _resolve(
        monkeypatch, family, quant, 16, torch.zeros(1, 128, grid, grid))
    assert (chosen, selected_sage, topology.describe()) == (row, sage, "uly2")
    assert f"{family} {quant} at {megapixels}" in reason


@pytest.mark.parametrize("megapixels", [1.2, 1.81, 2.4, 8.0])
def test_lens_fp8_row_keeps_the_declared_kernel(megapixels):
    """Reading Lens at its real size makes row 74 fire on the shipped graph.
    SAGE_AUTO there read 1-step NRMS 0.037 at a 20.3 s warm wall against
    Flash's 0.026 at 20.0 s, with the head under its 2100 MHz cap, so the row
    keeps uly2 and leaves the Init node's kernel in place (2026-09;
    docs/VALIDATION.md, section
    "Auto megapixels follow the model's latent ratio")."""
    decision = choose_auto_topology("lens", "fp8", megapixels, 2, cfg_value=5.0)
    assert decision.rule.row == 74
    assert decision.topology.ulysses == 2 and decision.topology.cfg == 1
    assert decision.sage is False
    assert "911b65bc6ef2" in decision.reason and "9d51070285bf" in decision.reason


def test_without_a_comfy_ratio_lens_keeps_the_pre_516_reading(monkeypatch):
    """A header comfy cannot name still gets the fallback guess, and with it
    the campaign's misread: 0.5 MP and the fallback row."""
    _topology, sage, reason, chosen = _resolve(
        monkeypatch, "lens", "fp8", None, torch.zeros(1, 128, 84, 84))
    assert (chosen, sage) == (99, False)
    assert "lens fp8 at 0.5 MP" in reason


def test_pixel_space_and_video_latents_resolve_at_their_ratio(monkeypatch):
    _t, _s, reason, _row = _resolve(
        monkeypatch, "chroma", "bf16", 1, torch.zeros(1, 3, 1024, 1024))
    assert "chroma bf16 at 1.0 MP" in reason
    _t, _s, reason, _row = _resolve(
        monkeypatch, "chroma", "bf16", None, torch.zeros(1, 3, 1024, 1024))
    assert "chroma bf16 at 1.0 MP" in reason   # the legacy RGB case still holds
    _t, _s, reason, row = _resolve(
        monkeypatch, "wan", "fp8", 8, torch.zeros(1, 16, 21, 60, 104))
    assert (row, "wan fp8 at 0.4 MP" in reason) == (50, True)
    _t, _s, reason, row = _resolve(
        monkeypatch, "minimax_h3", "int8", 16, _Nested(_h3_latent()), cfg=1.0)
    assert (row, "minimax_h3 int8 at 1.0 MP" in reason) == (53, True)


def test_the_private_override_beats_the_comfy_ratio(monkeypatch):
    samples = torch.zeros(1, 16, 64, 64)
    _t, _s, reason, _row = _resolve(
        monkeypatch, "flux", "bf16", 8, samples, latent_downscale=16)
    assert "flux bf16 at 1.0 MP" in reason
    _t, _s, reason, _row = _resolve(monkeypatch, "flux", "bf16", 8, samples)
    assert "flux bf16 at 0.3 MP" in reason
    # And through the sampler-latent entry point that reads the metadata key.
    latent = {"samples": samples, common.LATENT_DOWNSCALE_METADATA_KEY: 16}
    _topology, _sage, reason = common._resolve_topology_for_latent(
        _auto_model(), latent, 4.0)
    assert "flux bf16 at 1.0 MP" in reason


def _latent_from_graph(graph: dict) -> torch.Tensor:
    found = [node for node in graph.values() if node["class_type"] in LATENT_NODES]
    assert len(found) == 1, "one stock empty-latent node per finite-row template"
    node = found[0]
    channels, divisor = LATENT_NODES[node["class_type"]]
    inputs = node["inputs"]
    return torch.zeros(int(inputs.get("batch_size", 1)), channels,
                       inputs["height"] // divisor, inputs["width"] // divisor)


@pytest.mark.parametrize("quant", ["bf16", "fp8"])
@pytest.mark.parametrize("template", sorted(FINITE_ROW_TEMPLATES))
def test_matrix_prediction_and_runtime_resolution_agree(monkeypatch, template, quant):
    """The matrix reads the graph's pixels, the runtime the latent grid times
    comfy's ratio; one megapixel definition makes them the same number."""
    # test_sweep_matrix puts the repo root on sys.path and imports the matrix.
    from test_sweep_matrix import _template, matrix

    family, config = FINITE_ROW_TEMPLATES[template]
    graph = _template(template)
    facts = matrix.graph_facts(graph)
    samples = _latent_from_graph(graph)
    predicted = choose_auto_topology(family, quant, facts["megapixels"], 2,
                                     cfg_value=facts["cfg"], batch_size=facts["batch"])
    topology, sage, reason, row = _resolve(
        monkeypatch, family, quant, CONFIG_RATIO[config], samples, cfg=facts["cfg"])
    assert (row, topology.describe(), sage) == (
        predicted.rule.row, predicted.topology.describe(), predicted.sage), reason
    assert latent_scale.latent_megapixels(samples, CONFIG_RATIO[config]) \
        == pytest.approx(facts["megapixels"])
    if family in ("flux2", "lens"):
        cell = {"preset": "auto", "family": family, "quant": quant,
                "attention": "TORCH_FLASH", "batch": facts["batch"]}
        assert matrix.resolved_kernel(cell, facts, 2) \
            == common.resolve_sample_attention("TORCH_FLASH", sage)


def test_the_campaign_disagreement_is_the_legacy_reading(monkeypatch):
    """A 2026-09 campaign run: the matrix predicted row 74, which
    then granted SAGE_AUTO, and the runtime ran the fallback row on TORCH_FLASH.
    Under the fallback guess the rows still split; under comfy's ratio both
    read row 74, which keeps the declared kernel."""
    from test_sweep_matrix import _template, matrix

    graph = _template("dgx-monarch-lens-t2i")
    facts = matrix.graph_facts(graph)
    predicted = choose_auto_topology("lens", "fp8", facts["megapixels"], 2,
                                     cfg_value=facts["cfg"], batch_size=facts["batch"])
    cell = {"preset": "auto", "family": "lens", "quant": "fp8",
            "attention": "TORCH_FLASH", "batch": facts["batch"]}
    assert (predicted.rule.row, matrix.resolved_kernel(cell, facts, 2)) == (74, "TORCH_FLASH")
    samples = _latent_from_graph(graph)
    _t, legacy_sage, _r, legacy_row = _resolve(monkeypatch, "lens", "fp8", None, samples)
    _t, sage, _r, row = _resolve(monkeypatch, "lens", "fp8", 16, samples)
    assert (legacy_row, legacy_sage) == (99, False)
    assert (row, common.resolve_sample_attention("TORCH_FLASH", sage)) == (74, "TORCH_FLASH")


# Real comfy, CPU only. Keep it last in the file (see the module docstring).
def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


def _comfy_checkout() -> tuple[str, bool]:
    """The ComfyUI checkout to import, and whether the environment names one.

    The comfy-canary job names it with ``COMFYUI_DIR`` (``COMFY_DIR`` in its
    other steps). A named checkout is that job's contract: a missing checkout
    or a failing import must fail there, since a skip reports the step green
    without comparing anything. Plain CI names none and skips.
    """
    named = os.environ.get("COMFYUI_DIR") or os.environ.get("COMFY_DIR")
    return os.path.abspath(named or os.path.expanduser("~/ComfyUI")), named is not None


def _import_comfy(name: str, required: bool):
    if required:
        return importlib.import_module(name)
    return pytest.importorskip(name, reason="no ComfyUI on sys.path (comfy-canary job covers it)")


@pytest.fixture(scope="module")
def real_comfy():
    """Import real comfy on its CPU path, then undo it.

    The snapshot-and-restore of tests/test_chroma_cfg_pad_forward.py: a real
    ``comfy`` left in ``sys.modules`` would answer later "is comfy importable"
    probes in other files. ``comfy.cli_args`` parses argv on import, so argv
    carries ``--cpu`` or a CPU-only runner dies on the CUDA branch.
    """
    comfy_dir, required = _comfy_checkout()
    if required and not os.path.isdir(comfy_dir):
        pytest.fail(f"COMFYUI_DIR or COMFY_DIR names no ComfyUI checkout: {comfy_dir}")
    preserved = {name: module for name, module in sys.modules.items()
                 if _is_comfy_module(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        sys.argv = ["pytest-auto-latent-downscale", "--cpu"]
        try:
            options = _import_comfy("comfy.options", required)
            options.enable_args_parsing()
            supported = _import_comfy("comfy.supported_models", required)
            _import_comfy("comfy.model_detection", required)
            nested = _import_comfy("comfy.nested_tensor", required)
        finally:
            sys.argv = original_argv
        yield SimpleNamespace(supported_models=supported, nested_tensor=nested)
    finally:
        for name in [name for name in sys.modules if _is_comfy_module(name)]:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


@pytest.mark.parametrize(("config", "ratio"),
                         sorted({**CONFIG_RATIO, **OTHER_CONFIG_RATIO}.items()))
def test_comfy_latent_formats_declare_the_ratios_this_file_assumes(real_comfy, config, ratio):
    cls = getattr(real_comfy.supported_models, config)
    assert latent_scale.config_spatial_downscale(cls) == ratio, (
        f"comfy's {config} latent format now declares "
        f"{cls.latent_format.spacial_downscale_ratio}; re-read which auto rows "
        "fire at the shipped template sizes before updating this table (#516)")


def _header(path, keys: dict[str, tuple[int, ...]]) -> str:
    save_file({key: torch.zeros(shape, dtype=torch.bfloat16) for key, shape in keys.items()},
              str(path))
    return str(path)


# Real key paths and real shape layouts, shrunk: comfy's detection reads the
# keys and a few shapes, never a value.
_FLUX2_KEYS = {
    "double_blocks.0.img_attn.norm.key_norm.weight": (8,),
    "img_in.weight": (64, 128),
    "double_stream_modulation_img.lin.weight": (8, 8),
}
_LENS_KEYS = {
    "transformer_blocks.0.attn.norm_added_q.weight": (8,),
    "transformer_blocks.0.img_mlp.w1.weight": (8, 8),
    "img_in.weight": (64, 128),
    "proj_out.weight": (128, 64),
    "txt_norm.0.weight": (16,),
}
_FLUX1_KEYS = {
    "double_blocks.0.img_attn.norm.key_norm.weight": (8,),
    "img_in.weight": (64, 64),
    "vector_in.in_layer.weight": (8, 8),
}


@pytest.mark.parametrize(("keys", "config", "ratio"), [
    (_FLUX2_KEYS, "Flux2", 16), (_LENS_KEYS, "Lens", 16), (_FLUX1_KEYS, "FluxSchnell", 8),
], ids=["flux2", "lens", "flux1"])
def test_comfys_own_detection_names_the_ratio_from_a_header(
        real_comfy, tmp_path, keys, config, ratio):
    path = _header(tmp_path / f"{config}.safetensors", keys)
    assert type(render_memory_price.detect_comfy_model_config(path)).__name__ == config
    assert latent_scale.comfy_spatial_downscale(path) == ratio


def test_a_flux2_header_resolves_end_to_end(real_comfy, monkeypatch, tmp_path):
    """Real header sniff, real comfy detection, real auto table."""
    path = _header(tmp_path / "flux2.safetensors", _FLUX2_KEYS)
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: path
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(common, "_impl_warned", set())
    model = _auto_model("fp8_e4m3fn")
    _topology, _sage, reason = common.resolve_topology(
        model, torch.zeros(1, 128, 64, 64), cfg_value=4.0)
    assert "flux2 fp8 at 1.0 MP" in reason and "table row 99" in reason
    _topology, _sage, reason = common.resolve_topology(
        model, torch.zeros(1, 128, 96, 96), cfg_value=4.0)
    assert "flux2 fp8 at 2.4 MP" in reason and "table row 27" in reason


def test_a_lens_header_reaches_row_74_at_the_shipped_size(real_comfy, monkeypatch, tmp_path):
    path = _header(tmp_path / "lens.safetensors", _LENS_KEYS)
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: path
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(common, "_impl_warned", set())
    # The ratio is real comfy. The sniff reads this shrunk bf16 header as lens
    # bf16, so the pin supplies the fp8 that row 74 needs.
    monkeypatch.setattr(common, "sniff_checkpoint_for_topology",
                        lambda _path: ("lens", "fp8"))
    _topology, sage, reason = common.resolve_topology(
        _auto_model(), torch.zeros(1, 128, 84, 84), cfg_value=4.0)
    assert (sage, "lens fp8 at 1.8 MP" in reason, "table row 74" in reason) \
        == (False, True, True)


def test_comfys_nested_latent_reads_its_video_stream(real_comfy):
    nested = real_comfy.nested_tensor.NestedTensor(_h3_latent())
    assert latent_scale.latent_megapixels(nested, 16) == 1344 * 768 / 1e6
    assert latent_scale.legacy_spatial_downscale(nested) == 8
