"""MOSS-Image on the generic machinery: detection, auto, compile, FSDP and pins.

CPU compatibility checks in the tests/test_kroma_support.py pattern. The
synthetic headers are tests/test_detect.py's vendor layouts, real key names,
dtypes and shapes, shrunk on disk; where the six staged files sit in
~/ComfyUI/models/diffusion_models, each header must be that layout key for
key. These tests establish neither image fidelity nor GPU support, and they
import nothing from the adapter. The real-comfy tests run last: their module
fixture holds a real ``comfy`` in ``sys.modules`` until this module ends. They
skip where ComfyUI is not importable, and fail instead when COMFYUI_DIR or
COMFY_DIR names a checkout (the tests/test_auto_latent_downscale.py rule).
"""
from __future__ import annotations

import importlib
import math
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.fsdp

from attention_pin_helpers import Attention, int8
from dgx_monarch import accuracy_waiver, attention_pins, latent_scale, render_memory_price
from dgx_monarch.actor import worker_compile
from dgx_monarch.actor.store_detect import LivePrecisionEvidence, _detect_live_precision
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.detect import (
    fsdp_lora_admission_property,
    read_safetensors_keys,
    sniff_checkpoint,
    sniff_checkpoint_for_topology,
    sniff_fsdp_launch_quant,
    sniff_metadata_keys,
)
from dgx_monarch.adapters.fsdp import (
    ALL_BF16_PROFILE,
    apply_fsdp_capacity_mode,
    fsdp_precision_profile_is_admitted,
    validate_fsdp_launch_quant,
)
from dgx_monarch.adapters.fsdp_islands import (
    CLASS_OUTSIDE_FORWARD_MODULE_PATHS,
    FORWARD_PATH_NAME_OWNERS,
    OUTSIDE_FORWARD_MODULE_ATTRS,
    auxiliary_module_parameters,
    header_core_kind,
)
from dgx_monarch.compile_policy import compile_dit_blocks_slab, compile_dit_is_known_noop
from dgx_monarch.nodes import common, loaders
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from dgx_monarch.topology import choose_auto_topology
from dgx_monarch.topology_cfg_evidence import sniffed_checkpoint_variant
from dgx_monarch.topology_table import AUTO_TABLE
from test_detect import _moss_header, _write_moss

_STAGED = Path("~/ComfyUI/models/diffusion_models").expanduser()
# The six shipped files: three checkpoints, each in BF16 and INT8 ConvRot.
_FILES = {f"{stem}_{suffix}.safetensors": kind
          for stem in ("moss_image_sft", "moss_image_base", "kromoss")
          for suffix, kind in (("bf16", "bf16"), ("int8_convrot", "int8"))}


def _staged(name: str) -> Path:
    path = _STAGED / name
    if not path.is_file():
        pytest.skip(f"{path} not present on this box")
    return path


@pytest.mark.parametrize("name", sorted(_FILES))
def test_all_three_checkpoints_sniff_as_moss_image(tmp_path, name):
    """One key set serves SFT, Base and Kromoss, so the file name and SHA-256
    are their only identity. The layout sniffs under a name no rule reads, and
    each staged header is that layout, dtype and shape for every key."""
    kind = _FILES[name]
    renamed = _write_moss(tmp_path / "renamed-checkpoint.safetensors", kind)
    assert sniff_checkpoint(renamed) == sniff_checkpoint_for_topology(renamed) == (
        "moss_image", kind)
    path = _staged(name)
    _keys, header = read_safetensors_keys(path)
    staged = {key: (info["dtype"], info["shape"]) for key, info in header["__tensors__"].items()}
    assert staged == {key: (info["dtype"], info["shape"])
                      for key, info in _moss_header(kind).items()}
    assert sniff_checkpoint(path) == sniff_checkpoint_for_topology(path) == ("moss_image", kind)


@pytest.mark.parametrize("name", sorted(_FILES))
def test_no_header_variant_separates_kromoss(tmp_path, name):
    """No file carries header metadata, so no cfg record can treat Kromoss
    apart from SFT or Base, and no checkpoint-specific auto row can be built."""
    renamed = _write_moss(tmp_path / "renamed-checkpoint.safetensors", _FILES[name])
    assert sniffed_checkpoint_variant(str(renamed), "moss_image") is None
    path = _staged(name)
    assert sniff_metadata_keys(path) == ()
    assert sniffed_checkpoint_variant(str(path), "moss_image") is None


@pytest.mark.parametrize("cfg", [1.0, 3.0])
@pytest.mark.parametrize("megapixels", [0.26, 1.05, 2.36, 4.2])
@pytest.mark.parametrize("quant", ["bf16", "int8"])
def test_auto_resolves_the_fallback_uly2_row(quant, megapixels, cfg):
    assert not any(rule.family == "moss_image" for rule in AUTO_TABLE)
    decision = choose_auto_topology("moss_image", quant, megapixels, 2, cfg_value=cfg)
    topology = decision.topology
    assert decision.rule.row == 99 and not decision.sage
    assert (topology.ulysses, topology.ring, topology.cfg, topology.dp, topology.fsdp) == (
        2, 1, 1, 1, False)
    assert "family 'moss_image' has no auto table row" in decision.reason


def test_auto_reads_a_moss_header_end_to_end(monkeypatch, tmp_path):
    """The driver's resolution from a header on disk, comfy's ratio pinned at
    the 16 its MOSSImage config declares (the real-comfy test below holds it)."""
    path = _write_moss(tmp_path / "renamed-checkpoint.safetensors", "bf16")
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(path)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(latent_scale, "comfy_spatial_downscale", lambda _path: 16)
    monkeypatch.setattr(common, "_impl_warned", set())
    model = ModelSpec(
        mesh=MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                      attention="TORCH_FLASH", sync_ulysses=True),
        unet_name=path.name, options={"weight_dtype": "default"})
    topology, sage, reason = common.resolve_topology(model, torch.zeros(1, 128, 64, 64), 3.0)
    assert topology.describe() == "uly2" and not sage
    assert "moss_image bf16 at 1.05 MP" in reason and "table row 99" in reason


def _moss_tree(layers: int = 42) -> torch.nn.Module:
    """comfy's MOSSImage module tree with Linear stand-ins, all BF16 (ComfyUI
    pops the one F32 tensor, rope.bands, before load): the block list
    `layers`, two text blocks nested under `text_embedder`, and the embedders,
    modulations and head beside them."""
    def linear() -> torch.nn.Linear:
        return torch.nn.Linear(2, 2)

    dm = torch.nn.Module()
    dm.video_embedder = linear()
    dm.text_embedder = torch.nn.Module()
    dm.text_embedder.proj = linear()
    dm.text_embedder.blocks = torch.nn.ModuleList(linear() for _ in range(2))
    dm.time_embedding = torch.nn.Sequential(linear(), torch.nn.SiLU(), linear())
    dm.shared_block_modulation = linear()
    dm.layers = torch.nn.ModuleList(linear() for _ in range(layers))
    dm.final_norm_video = torch.nn.RMSNorm(2)
    dm.final_linear_video = torch.nn.Linear(2, 2, bias=False)
    dm.video_head_modulation = linear()
    return dm.to(torch.bfloat16)


def _patcher(dm: torch.nn.Module) -> SimpleNamespace:
    """A patcher whose model is an instance of a class named MOSSImage, the
    way comfy hands the wrap its model_base instance."""
    model = type("MOSSImage", (), {})()
    model.diffusion_model = dm
    return SimpleNamespace(model=model, size=0)


def test_compile_is_a_known_noop(monkeypatch):
    """The block compiler compiles only a `blocks` list and MOSS names its
    list `layers`, so a compile request runs eager and keeps slab residency."""
    assert compile_dit_is_known_noop("moss_image")
    assert not compile_dit_blocks_slab("moss_image", True)
    assert compile_dit_blocks_slab("krea2", True)  # a `blocks` family does compile
    warned: list = []
    monkeypatch.setenv("DGXM_COMPILE_DIT", "1")
    monkeypatch.setattr(worker_compile, "safe_call", lambda *args: warned.append(args))
    dm = _moss_tree(layers=2)
    worker_compile.maybe_compile_dit(dm)
    assert len(warned) == 1 and "has no blocks list" in warned[0][1]
    assert type(dm.layers[0]) is torch.nn.Linear


def test_the_fsdp_launch_contract_admits_a_bf16_header_with_rope_bands(tmp_path):
    """The header's 64 F32 bytes sit inside the islands ceiling, so the launch
    kind reads bf16; the live tree carries no F32 at all, so the rescan reads
    the all-BF16 profile and asks no island profile."""
    widths = {"BF16": 2, "F32": 4}
    real: dict[str, int] = {}
    for info in _moss_header("bf16").values():
        real[info["dtype"]] = real.get(info["dtype"], 0) + (
            math.prod(info["shape"]) * widths[info["dtype"]])
    assert real["F32"] == 64 and set(real) == {"BF16", "F32"}
    assert header_core_kind(real) == "bf16"
    for kind in ("bf16", "int8"):
        path = _write_moss(tmp_path / f"moss_{kind}.safetensors", kind)
        assert sniff_fsdp_launch_quant(path) == kind
        validate_fsdp_launch_quant(kind)  # no raise
    live = _detect_live_precision(_patcher(_moss_tree()), "bf16")
    assert (live.quant_kind, live.live_dtype_profile, live.auxiliary_parameter_count,
            live.auxiliary_parameter_bytes) == ("bf16", ALL_BF16_PROFILE, 0, 0)
    assert fsdp_precision_profile_is_admitted(ALL_BF16_PROFILE, 0, 0, "bf16")


def test_int8_with_a_lora_refuses_fsdp_class_p(monkeypatch, tmp_path):
    """The driver's header preflight: the I8 file refuses class P on every
    lora_low_rss setting, and the BF16 file passes to the lever check."""
    int8_path = _write_moss(tmp_path / "moss_int8.safetensors", "int8")
    bf16_path = _write_moss(tmp_path / "moss_bf16.safetensors", "bf16")
    assert fsdp_lora_admission_property(int8_path) == "quantized_shards"
    assert fsdp_lora_admission_property(bf16_path) is None
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, name: str(tmp_path / name)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    with pytest.raises(UnsupportedModelError) as caught:
        loaders._preflight_fsdp_lora_checkpoint_property(int8_path.name)
    tag = parse_leading_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert "comfy-kitchen quantized tensors" in str(caught.value)
    loaders._preflight_fsdp_lora_checkpoint_property(bf16_path.name)  # no raise


class _Recorder:
    def __init__(self):
        self.calls: list = []
        self.kwargs: list = []

    def __call__(self, module, **kwargs):
        self.calls.append(module)
        self.kwargs.append(kwargs)
        return module


@pytest.fixture
def fully_shard_recorder(monkeypatch):
    """tests/test_fsdp_capacity.py's stub: the real wrap body, no process group."""
    recorder = _Recorder()
    monkeypatch.setattr(torch.distributed.fsdp, "fully_shard", recorder)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    return recorder


def test_a_moss_shaped_tree_wraps_layers_and_keeps_the_rest_in_the_root_group(
        fully_shard_recorder):
    """`layers` is a block-list name, so the 42 blocks wrap one by one. The
    text blocks are nested under `text_embedder`, so they stay in the root group
    with every other module the forward runs; no census row names MOSSImage,
    so nothing is replicated."""
    dm = _moss_tree()
    patcher = _patcher(dm)
    evidence = LivePrecisionEvidence(quant_kind="bf16", live_dtype_profile=ALL_BF16_PROFILE,
                                     checkpoint_kind="bf16")

    apply_fsdp_capacity_mode(patcher, "bf16", None, evidence)

    assert fully_shard_recorder.calls == [*dm.layers, dm]
    assert fully_shard_recorder.kwargs == [{}] * 42 + [{"reshard_after_forward": True}]
    for table in (OUTSIDE_FORWARD_MODULE_ATTRS, CLASS_OUTSIDE_FORWARD_MODULE_PATHS,
                  FORWARD_PATH_NAME_OWNERS):
        assert "MOSSImage" not in table
    assert auxiliary_module_parameters(dm, patcher.model) == set()
    wrapped = {p for block in dm.layers for p in block.parameters()}
    root_only = {p for module in (dm.video_embedder, dm.text_embedder, dm.time_embedding,
                                  dm.shared_block_modulation, dm.final_norm_video,
                                  dm.final_linear_video, dm.video_head_modulation)
                 for p in module.parameters()}
    assert set(dm.parameters()) - wrapped == root_only


def test_a_dtensor_video_embedder_reads_its_logical_width(tmp_path):
    """ComfyUI's concat_cond reads `video_embedder.weight.shape[1]` on every
    render, outside the forward. FSDP2 shards dim 0 only and a DTensor reports
    its logical shape, so the width reads 128 at any world size. One real wrap
    on a one-rank gloo group with a file store, as tests/test_fsdp_capacity.py
    runs its behavioral canaries."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    holder = torch.nn.Module()
    holder.video_embedder = torch.nn.Linear(128, 8)
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{tmp_path / 'process_group'}", rank=0,
        world_size=1)
    try:
        torch.distributed.fsdp.fully_shard(holder, mesh=init_device_mesh("cpu", (1,)))
        weight = holder.video_embedder.weight
        assert type(weight).__name__ == "DTensor"
        assert tuple(weight.placements) == (Shard(0),)
        assert tuple(weight.shape) == (8, 128) and weight.shape[1] == 128
    finally:
        torch.distributed.destroy_process_group()


@pytest.fixture
def _clean_waiver_scope():
    accuracy_waiver.clear()
    yield
    accuracy_waiver.clear()


class _Block(torch.nn.Module):
    """A MOSS-Image block owns its attention: the pin sits at <block>.comfy_attention."""

    def __init__(self, attention):
        super().__init__()
        self.comfy_attention = attention
        self.qkv = torch.nn.Linear(2, 6)


def _pinned_patcher() -> SimpleNamespace:
    """The INT8 files' layout: an int8 pin on all 42 blocks and none on the two
    text blocks, whose ComfyAttention keeps no config."""
    dm = torch.nn.Module()
    dm.layers = torch.nn.ModuleList(_Block(int8()) for _ in range(42))
    dm.text_embedder = torch.nn.Module()
    dm.text_embedder.blocks = torch.nn.ModuleList(_Block(Attention()) for _ in range(2))
    return SimpleNamespace(model=SimpleNamespace(diffusion_model=dm), object_patches={})


def _worker(**topology) -> SimpleNamespace:
    return SimpleNamespace(topology={"ulysses": 1, "ring": 1, "cfg": 1, "dp": 1,
                                     "fsdp": False, **topology})


@pytest.mark.usefixtures("_clean_waiver_scope")
@pytest.mark.parametrize(("topology", "label", "refuses"), [
    ({}, "single", True), ({"cfg": 2}, "cfg2", True), ({"dp": 2}, "dp2", True),
    ({"cfg": 2, "fsdp": True}, "cfg2+fsdp", True), ({"dp": 2, "fsdp": True}, "dp2+fsdp", True),
    ({"ulysses": 2}, "uly2", False), ({"ring": 2}, "ring2", False),
    ({"ulysses": 2, "fsdp": True}, "uly2+fsdp", False),
])
def test_int8_pins_on_every_block_are_waived_off_sp_and_dropped_on_sp(topology, label, refuses):
    """Off sequence parallelism the 42 pins run as on one GPU and the class K
    Sol-Attn waiver is asked; the INT8 file takes the direct FSDP wrap, which
    clears no pin. Under uly2 or ring2 every pin is dropped and nothing is asked."""
    patcher = _pinned_patcher()
    found = attention_pins.scan_loaded(patcher)
    assert sorted(found) == sorted(f"diffusion_model.layers.{n}.comfy_attention"
                                   for n in range(42))
    worker = _worker(**topology)
    summary = attention_pins.render_summary(worker, [patcher])
    assert summary is not None and summary.kernels == {"comfy_kitchen_int8": 42}
    if not refuses:
        assert summary.public()["attention_pins_dropped"] == {
            attention_pins.DROPPED_SEQUENCE_PARALLEL: 42}
        attention_pins.assert_waived(summary, worker.topology, family="moss_image")
        return
    assert summary.ran
    with pytest.raises(UnsupportedModelError) as caught:
        attention_pins.assert_waived(summary, worker.topology, family="moss_image")
    tag = parse_leading_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert tag.guard == attention_pins.SOL_ATTN_PIN_GUARD and tag.waivable
    assert "comfy_kitchen_int8 on 42 attention modules" in str(caught.value)
    assert f"Topology {label} runs no sequence parallelism" in str(caught.value)


# Real comfy, CPU only. Keep it last in the file (see the module docstring).
def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def real_comfy():
    """Import real comfy on its CPU path, then undo it (the real_comfy fixture
    of tests/test_auto_latent_downscale.py)."""
    named = os.environ.get("COMFYUI_DIR") or os.environ.get("COMFY_DIR")
    comfy_dir = os.path.abspath(named or os.path.expanduser("~/ComfyUI"))
    if named and not os.path.isdir(comfy_dir):
        pytest.fail(f"COMFYUI_DIR or COMFY_DIR names no ComfyUI checkout: {comfy_dir}")

    def load(name: str):
        if named:
            return importlib.import_module(name)
        return pytest.importorskip(name, reason="no ComfyUI on sys.path")

    preserved = {name: module for name, module in sys.modules.items() if _is_comfy_module(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path, original_argv = list(sys.path), sys.argv
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    try:
        sys.argv = ["pytest-moss-image-support", "--cpu"]
        try:
            load("comfy.options").enable_args_parsing()
            supported = load("comfy.supported_models")
            latent_formats = load("comfy.latent_formats")
            load("comfy.model_detection")
        finally:
            sys.argv = original_argv
        yield SimpleNamespace(supported_models=supported, latent_formats=latent_formats)
    finally:
        for name in [name for name in sys.modules if _is_comfy_module(name)]:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


def test_comfys_sampling_and_latent_settings(real_comfy):
    config_class = real_comfy.supported_models.MOSSImage
    assert config_class.sampling_settings["multiplier"] == 1.0
    assert config_class.sampling_settings["shift"] == 5.0
    assert config_class.latent_format is real_comfy.latent_formats.Flux2
    assert latent_scale.config_spatial_downscale(config_class) == 16
    assert config_class.memory_usage_factor == 14.6
    # The one F32 tensor never reaches the live model.
    config = config_class({"image_model": "moss_image", "disable_unet_model_creation": True})
    assert config.process_unet_state_dict({"rope.bands": 0, "layers.0.norm1.weight": 1}) == {
        "layers.0.norm1.weight": 1}


@pytest.mark.parametrize("name", sorted(_FILES))
def test_comfys_detection_names_moss_image_from_the_header(real_comfy, tmp_path, name):
    """ComfyUI's own detection reads the needle pair the signature row reads,
    from the synthetic layout and from each staged header."""
    render_memory_price._detect_by_identity.cache_clear()
    paths = [_write_moss(tmp_path / "renamed-checkpoint.safetensors", _FILES[name])]
    if (_STAGED / name).is_file():
        paths.append(_STAGED / name)
    for path in paths:
        config = render_memory_price.detect_comfy_model_config(str(path))
        assert type(config).__name__ == "MOSSImage", path
        assert latent_scale.comfy_spatial_downscale(str(path)) == 16
    render_memory_price._detect_by_identity.cache_clear()
