"""MiniMax-H3-Image from its header: the family, its formats, the grid gate and the refusal.

ComfyUI 0.45.0 added the model (upstream pull request 17655); this release carries
no forward for it, so the upstream gate refuses every file of the family before
any load (TROUBLESHOOTING #150), and a file whose config metadata lacks the sigma
grid first (#149). The headers come from ``minimax_h3_image_headers``: the
published files' names, dtypes, shapes and metadata, with no weights.
``tests/test_minimax_h3_image_comfy_detection.py`` holds the family's needles
against ComfyUI's own detection.
"""
from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import minimax_h3_image_headers as h3i
from dgx_monarch import attention_pins, h3_context_windows, h3_sigma_shift, upstream_gate
from dgx_monarch.actor import store_fsdp
from dgx_monarch.adapters import detect, fsdp_islands
from dgx_monarch.adapters import minimax_h3_image_header as h3_image
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import loader_preflight
from dgx_monarch.nodes.loaders import DGXMonarchUNETLoader
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag

GRID, FAMILY = upstream_gate.H3_IMAGE_GRID_TROUBLESHOOTING, upstream_gate.H3_IMAGE_FAMILY_TROUBLESHOOTING
TROUBLESHOOTING = Path(__file__).resolve().parents[1] / "docs" / "TROUBLESHOOTING.md"


def _heading(number: int) -> str:
    """The title of TROUBLESHOOTING entry ``number``, so a missed renumber fails here."""
    headings = dict(re.findall(r"^## (\d+)\. (.+)$", TROUBLESHOOTING.read_text(), re.MULTILINE))
    return headings[str(number)]


@pytest.fixture
def files(tmp_path, monkeypatch):
    """The three published formats at full size, sparse, and the `folder_paths` both sites resolve with."""
    for kind in h3i.KINDS:
        h3i.write(tmp_path / f"{kind}.safetensors", kind)

    def get_full_path(_kind, name):
        candidate = tmp_path / str(name)
        return str(candidate) if candidate.exists() else None

    fake = types.ModuleType("folder_paths")
    fake.get_full_path = get_full_path
    fake.get_filename_list = lambda _kind: sorted(p.name for p in tmp_path.iterdir())
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    return tmp_path


@pytest.mark.parametrize(("kind", "quant"), [("bf16", "bf16"), ("int8_convrot", "int8"),
                                              ("w4a8", "w4a8")])
def test_each_published_format_reads_as_the_family_and_what_it_is(files, kind, quant):
    path = str(files / f"{kind}.safetensors")
    assert detect.sniff_checkpoint(path) == (h3_image.FAMILY, quant)
    assert detect.sniff_checkpoint_for_topology(path) == (h3_image.FAMILY, quant)
    # The FSDP precision proof: the bf16 file proves bf16 although it holds 50 uint8
    # pins and 51 F16 adaLN linears; before this family's rules it read fp32.
    assert detect.sniff_fsdp_launch_quant(path) == quant
    expected = None if kind == "bf16" else "quantized_shards"
    assert detect.fsdp_lora_admission_property(path) == expected


def test_the_h3_row_still_takes_every_file_with_the_audio_path(tmp_path):
    """ComfyUI reads audio_patch_proj.weight to name H3 video; so does the earlier row."""
    path = h3i.write(tmp_path / "h3.safetensors", "bf16", blocks=2, audio=True)
    family, quant = detect.sniff_checkpoint(str(path))
    assert family == "minimax_h3"
    # Family scoped: H3 keeps today's reading of any F16 tensor.
    assert quant == "fp16"
    assert upstream_gate.refusal_for(str(path)) is None


def test_an_f16_tensor_that_is_no_adaln_island_keeps_the_fp16_reading(tmp_path):
    path = h3i.write(tmp_path / "odd.safetensors", "bf16", blocks=2,
                     extra={"condition_proj.weight": ("F16", [h3i.WIDTH, 5120])})
    keys, header = detect.read_safetensors_keys(path)
    assert h3_image.f16_islands(keys, header["__tensors__"]) == frozenset()
    assert detect.sniff_checkpoint(str(path)) == (h3_image.FAMILY, "fp16")
    assert detect.sniff_fsdp_launch_quant(str(path)) == "fp32"  # two cores beside F32


def test_the_islands_are_exactly_the_51_adaln_linears(files):
    keys, header = detect.read_safetensors_keys(files / "bf16.safetensors")
    islands = h3_image.f16_islands(keys, header["__tensors__"])
    assert len(islands) == 51
    assert all(name.endswith(".adaln_proj.linear.weight") for name in islands)


def test_the_proof_skips_pins_and_counts_named_islands_under_the_ceiling():
    """``proof_bytes_by_dtype`` feeds ``header_core_kind``; the ceiling still binds."""
    def info(dtype, size):
        return {"dtype": dtype, "data_offsets": [0, size]}

    tensors = {"core.weight": info("BF16", 1000), "a.comfy_attention.config": info("U8", 37),
               "island.weight": info("F16", 10), "other.marker": info("U8", 4)}
    assert fsdp_islands.proof_bytes_by_dtype(tensors, frozenset({"island.weight"})) == {
        "BF16": 1000, "F32": 10, "U8": 4}
    del tensors["other.marker"]
    counted = fsdp_islands.proof_bytes_by_dtype(tensors, frozenset({"island.weight"}))
    assert fsdp_islands.header_core_kind(counted) == "bf16"
    tensors["island.weight"] = info("F16", 400)  # past the 5 percent ceiling
    counted = fsdp_islands.proof_bytes_by_dtype(tensors, frozenset({"island.weight"}))
    assert fsdp_islands.header_core_kind(counted) == "fp32"


def test_a_pinned_bf16_file_of_any_family_now_proves_bf16(tmp_path):
    """The pin skip is not family scoped: a uint8 attention pin is no weight."""
    tensors = {
        "content_mode_blocks.0.attn.qkv.weight": ("BF16", [8, 8]),
        "single_stream_modulations.lin.weight": ("BF16", [8, 8]),
        "content_mode_blocks.0.attn.comfy_attention.config": ("U8", [len(h3i.PIN)], h3i.PIN.encode()),
    }
    path = tmp_path / "flux3_pinned.safetensors"
    header, offset, payloads = {}, 0, []
    for name, (dtype, shape, *payload) in tensors.items():
        size = h3i._nbytes(dtype, shape)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        payloads.extend((offset, data) for data in payload)
        offset += size
    import struct

    blob = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)
    assert detect.sniff_checkpoint(str(path)) == ("flux3", "bf16")
    assert detect.sniff_fsdp_launch_quant(str(path)) == "bf16"


@pytest.mark.parametrize("metadata", [None, {"config": json.dumps({"transformer": {"pdd_steps": 6}})},
                                      {"config": "{not json"},
                                      {"config": json.dumps({"transformer": {"pdd_sigmas": []}})},
                                      {"config": json.dumps({"transformer": {"pdd_sigmas": "1,0"}})},
                                      {"other": "x"}],
                         ids=["no-metadata", "no-grid", "not-json", "empty-grid", "text-grid", "no-config"])
def test_a_file_without_its_sigma_grid_refuses_first(tmp_path, metadata):
    path = h3i.write(tmp_path / "nogrid.safetensors", "bf16", blocks=2, metadata=metadata)
    body, number = upstream_gate.refusal_for(str(path))
    assert number == GRID == 149
    assert _heading(GRID).startswith("A MiniMax-H3-Image file refuses")
    assert body == upstream_gate.H3_IMAGE_GRID_REFUSAL
    assert "transformer.pdd_sigmas" in body
    assert "a file without the grid raises a TypeError" in body


def test_with_its_grid_the_family_refuses_and_names_the_stock_path(files):
    for kind in h3i.KINDS:
        body, number = upstream_gate.refusal_for(str(files / f"{kind}.safetensors"))
        assert number == FAMILY == 150
        assert body == upstream_gate.H3_IMAGE_FAMILY_REFUSAL
    assert _heading(FAMILY).startswith("A MiniMax-H3-Image render refuses")
    for token in ("minimax_h3_image", "ComfyUI 0.45.0", "Load Diffusion Model",
                  "Nothing was loaded and nothing was quarantined", "no waiver"):
        assert token in upstream_gate.H3_IMAGE_FAMILY_REFUSAL


def test_the_grid_answers_before_dropped_scales_and_dropped_scales_before_the_family(tmp_path):
    no_marker = ("blocks.0.attn.qkv_proj.comfy_quant",)
    both = h3i.write(tmp_path / "both.safetensors", "int8_convrot", blocks=2, metadata=None,
                     drop=no_marker)
    assert upstream_gate.refusal_for(str(both))[1] == GRID
    scales = h3i.write(tmp_path / "scales.safetensors", "int8_convrot", blocks=2, drop=no_marker)
    assert upstream_gate.refusal_for(str(scales))[1] == upstream_gate.DROPPED_SCALE_TROUBLESHOOTING


def test_a_format_with_no_row_takes_the_path_any_unvalidated_format_takes(tmp_path):
    """A community 4-bit export in ComfyUI's nvfp4 layout: no published file, no row.

    Its kind comes from the generic rules, the same reading the layout gets under
    H3's keys, and the one family refusal answers it: no format of this family has
    a path of its own before the adapter."""
    path = h3i.write(tmp_path / "w4a4.safetensors", "nvfp4", blocks=2)
    twin = h3i.write(tmp_path / "w4a4_h3.safetensors", "nvfp4", blocks=2, audio=True)
    family, quant = detect.sniff_checkpoint(str(path))
    assert family == h3_image.FAMILY
    assert quant == detect.sniff_checkpoint(str(twin))[1] == "fp8"
    assert upstream_gate.refusal_for(str(path))[1] == FAMILY


def _mesh(**overrides):
    spec = {"topology_preset": "single", "handle": None, "worker_args": {}}
    spec.update(overrides)
    return SimpleNamespace(**spec)


class _RecordingStore:
    def __init__(self):
        self.calls = []

    def ensure(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return object(), "load"


def _worker(**overrides):
    spec = {"topology": {}, "world": 1, "store": _RecordingStore()}
    spec.update(overrides)
    return SimpleNamespace(**spec)


@pytest.mark.parametrize("preset", ["single", "uly2"])
def test_the_driver_and_the_worker_refuse_class_p_with_one_sentence(files, monkeypatch, preset):
    def explode(*_args, **_kwargs):
        raise AssertionError("an answer below the gate ran")

    monkeypatch.setattr(loader_preflight, "preflight_comfy_managed_topology", explode)
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint", explode)
    from dgx_monarch.nodes import loaders

    monkeypatch.setattr(loaders, "ensure_live", explode)
    with pytest.raises(UnsupportedModelError) as driver:
        DGXMonarchUNETLoader().load(_mesh(topology_preset=preset), "int8_convrot.safetensors")
    tag = parse_leading_refusal_tag(str(driver.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert tag.guard is None and tag.waivable is False
    assert "docs/TROUBLESHOOTING.md #150" in str(driver.value)
    worker = _worker()
    with pytest.raises(UnsupportedModelError) as funnel:
        store_fsdp.ensure(worker, "int8_convrot.safetensors", {}, [])
    assert str(funnel.value) == str(driver.value)
    assert worker.store.calls == [], "the store was entered on a refused artifact"


_DAMAGED_QUANT_METADATA = {"not-json": "{not json", "no-layers": json.dumps({"format_version": 1}),
                           "layers-a-list": json.dumps({"layers": ["blocks.0.attn.qkv_proj"]})}


@pytest.mark.parametrize("damage", sorted(_DAMAGED_QUANT_METADATA))
@pytest.mark.parametrize("grid", [True, False], ids=["grid", "no-grid"])
def test_damaged_quantization_metadata_cannot_cancel_the_familys_refusal(files, monkeypatch, damage, grid):
    """The grid and the family are read before any quantization metadata is parsed,
    and the family's refusal stands when the scale check cannot finish: the driver
    and the Worker refuse with the class and sentence the intact header gets."""
    def explode(*_args, **_kwargs):
        raise AssertionError("an answer below the gate ran")

    monkeypatch.setattr(loader_preflight, "preflight_comfy_managed_topology", explode)
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint", explode)
    from dgx_monarch.nodes import loaders

    monkeypatch.setattr(loaders, "ensure_live", explode)
    config = h3i.METADATA if grid else {}
    intact = h3i.write(files / f"intact_{damage}_{grid}.safetensors", "int8_convrot", blocks=2,
                       metadata=dict(config))
    damaged = h3i.write(files / f"damaged_{damage}_{grid}.safetensors", "int8_convrot", blocks=2,
                        metadata={**config, "_quantization_metadata": _DAMAGED_QUANT_METADATA[damage]})
    expected = upstream_gate.refusal_for(str(intact))
    assert expected is not None and expected[1] == (FAMILY if grid else GRID)
    assert upstream_gate.refusal_for(str(damaged)) == expected
    with pytest.raises(UnsupportedModelError) as driver:
        DGXMonarchUNETLoader().load(_mesh(), damaged.name)
    with pytest.raises(UnsupportedModelError) as reference:
        DGXMonarchUNETLoader().load(_mesh(), intact.name)
    assert str(driver.value) == str(reference.value)
    tag = parse_leading_refusal_tag(str(driver.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    worker = _worker()
    with pytest.raises(UnsupportedModelError) as funnel:
        store_fsdp.ensure(worker, damaged.name, {}, [])
    assert str(funnel.value) == str(reference.value)
    assert worker.store.calls == []


def test_damaged_quantization_metadata_on_another_family_gives_no_verdict(tmp_path):
    """Outside this family the gate stands down, as before: stock ComfyUI cannot parse
    the same metadata either (comfy.utils.convert_old_quants), so no scale drops silently."""
    path = h3i.write(tmp_path / "h3.safetensors", "int8_convrot", blocks=2, audio=True,
                     metadata={**h3i.METADATA, "_quantization_metadata": "{not json"})
    assert upstream_gate.refusal_for(str(path)) is None


def test_forcing_the_h3_video_adapter_cannot_walk_past_the_gate(files):
    worker = _worker(family_override="minimax_h3")
    with pytest.raises(UnsupportedModelError) as raised:
        store_fsdp.ensure(worker, "bf16.safetensors", {"family_adapter": "minimax_h3"}, [])
    assert "minimax_h3_image" in str(raised.value)
    assert worker.store.calls == []


def test_the_h3_nodes_refuse_the_family_once_detection_names_it(tmp_path):
    """ModelSamplingMiniMaxH3 and the H3 context windows key on minimax_h3 by equality,
    and pass `unknown`, so they refuse this family only because the header names it."""
    family = detect.sniff_checkpoint(str(h3i.write(tmp_path / "f.safetensors", blocks=2)))[0]
    reason = h3_sigma_shift.family_refusal_reason(family)
    assert reason is not None and "reads as minimax_h3_image" in reason
    assert h3_context_windows.family_refusal_reason(family) is not None


class _ComfyAttention(torch.nn.Module):
    """ComfyUI's ComfyAttention load contract (comfy/ldm/modules/attention.py).

    Every load clears the pin and binds it again only from a config key in that
    load; canary seam 39 holds the real class to the same reads."""

    def __init__(self):
        super().__init__()
        self.config = self.function = self.method = None

    def _load_from_state_dict(self, state_dict, prefix, *args):
        from attention_pin_helpers import attention_comfy_kitchen_int8

        self.config = self.function = self.method = None
        metadata = state_dict.pop(prefix + "config", None)
        if metadata is not None:
            self.config = json.loads(metadata.numpy().tobytes())
            self.function, self.method = attention_comfy_kitchen_int8, "comfy_kitchen_int8"
        super()._load_from_state_dict(state_dict, prefix, *args)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.comfy_attention = _ComfyAttention()
        self.attn.qkv_proj = torch.nn.Linear(8, 8, dtype=torch.bfloat16)


def test_the_shard_build_clears_a_bf16_files_pins_and_the_record_says_dropped(tmp_path):
    """Reachable now: the proof grants the pinned bf16 file, so the build meets its pins.

    The clearing stays as it is: the bf16 build reloads parameters through
    ``load_state_dict`` with no config key, ComfyAttention drops the pin, and the
    record names the build (attention_pins.DROPPED_FSDP)."""
    import torch.distributed.fsdp
    from torch.distributed.device_mesh import init_device_mesh

    from dgx_monarch.adapters.fsdp_shard_build import build_shards

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    path = h3i.write(tmp_path / "bf16_pinned.safetensors", "bf16", blocks=2)
    assert detect.sniff_fsdp_launch_quant(str(path)) == "bf16"
    dm = torch.nn.Module()
    dm.blocks = torch.nn.ModuleList([_Block(), _Block()])
    state = {name: value.clone() for name, value in dm.state_dict().items()}
    for index in range(2):
        state[f"blocks.{index}.attn.comfy_attention.config"] = torch.tensor(
            list(h3i.PIN.encode()), dtype=torch.uint8)
    dm.load_state_dict(state)
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=dm), object_patches={})
    assert len(attention_pins.scan_loaded(patcher)) == 2
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        for parameter in dm.parameters():
            parameter.requires_grad = False

        def wrap():
            for block in dm.blocks:
                torch.distributed.fsdp.fully_shard(block, mesh=mesh)
            torch.distributed.fsdp.fully_shard(dm, mesh=mesh, reshard_after_forward=True)

        assert build_shards(dm, wrap) == 4  # two weights and two biases
        assert all(block.attn.comfy_attention.config is None for block in dm.blocks)
        attention_pins.note_injected(patcher)
        summary = attention_pins.render_summary(
            SimpleNamespace(topology={"ulysses": 1, "ring": 1, "cfg": 2, "fsdp": True}), [patcher])
        assert summary is not None
        assert summary.public()["attention_pins_dropped"] == {attention_pins.DROPPED_FSDP: 2}
        assert summary.public()["attention_pins_ran"] is False
        attention_pins.assert_waived(summary, {"cfg": 2, "fsdp": True})  # nothing runs, nothing asked
    finally:
        torch.distributed.destroy_process_group()
