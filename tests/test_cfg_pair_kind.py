"""A pair scope tells a block-scaled fp8 file from an FP8 one.

Since 2026-11-24 an equal prompt pair leaves cfg2 only where the row it falls
to reads bytes equal to one GPU (topology_cfg_evidence). The fp8 rows behind
that were measured on files that store FP8 weights, while the driver's ``fp8``
kind also covers MXFP8 and NVFP4 files, whose block scales are F8 tensors too
and whose sharded renders are not bytes equal. ``pair_kind`` names those
apart, from the header the driver already reads, and the auto resolution uses
it only for the pair scope: an MXFP8 or NVFP4 Qwen-Image file keeps cfg2 for an
equal pair, where an FP8 one moves to row 71. The headers here are small and
hold the observed dtypes, not real shapes.
"""
from __future__ import annotations

import json
import math
import struct
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import family_select, latent_scale
from dgx_monarch.nodes import cfg_pair_notice, common
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.topology_cfg_evidence import BLOCK_SCALED_FP8, CAST_FP8, pair_kind

_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1}

# Weight and scale dtypes as each published layout stores them.
LAYOUTS = {
    "fp8": {"blocks.0.attn.to_q.weight": ("F8_E4M3", [8, 8])},
    "fp8_scaled": {"blocks.0.attn.to_q.weight": ("F8_E4M3", [8, 8]),
                   "blocks.0.attn.to_q.weight_scale": ("F32", [1]),
                   "blocks.0.norm.weight": ("BF16", [8])},
    "mxfp8": {"blocks.0.attn.to_q.weight": ("F8_E4M3", [8, 8]),
              "blocks.0.attn.to_q.weight_scale": ("U8", [8, 1])},
    "nvfp4": {"blocks.0.attn.to_q.weight": ("U8", [8, 4]),
              "blocks.0.attn.to_q.weight_scale": ("F8_E4M3", [8, 1]),
              "blocks.0.attn.to_q.weight_scale_2": ("F32", [1])},
    "bf16": {"blocks.0.attn.to_q.weight": ("BF16", [8, 8])},
}
# ComfyUI's per-layer quant records: a U8 tensor holding the layer's JSON
# descriptor beside its weight (adapters/layer_records.py reads them).
RECORDS = {
    "fp8_records": {"blocks.0.attn.to_q": "float8_e4m3fn"},
    "nvfp4_records": {"blocks.0.attn.to_q": "nvfp4"},
    "mixed_records": {"blocks.0.attn.to_q": "float8_e4m3fn", "blocks.0.attn.to_k": "nvfp4"},
}
LAYOUTS["fp8_records"] = dict(LAYOUTS["fp8_scaled"])
LAYOUTS["nvfp4_records"] = dict(LAYOUTS["nvfp4"])
LAYOUTS["mixed_records"] = {**LAYOUTS["fp8_scaled"],
                            "blocks.0.attn.to_k.weight": ("U8", [8, 4]),
                            "blocks.0.attn.to_k.weight_scale": ("F8_E4M3", [8, 1])}


def _checkpoint(path, layout):
    tensors: dict = {"__metadata__": {}}
    data = bytearray()
    layers = dict(LAYOUTS[layout])
    payloads = {}
    for layer, fmt in RECORDS.get(layout, {}).items():
        payloads[f"{layer}.comfy_quant"] = json.dumps({"format": fmt}).encode()
        layers[f"{layer}.comfy_quant"] = ("U8", [len(payloads[f"{layer}.comfy_quant"])])
    for name, (dtype, shape) in layers.items():
        offset = len(data)
        data.extend(payloads.get(name, bytes(math.prod(shape) * _BYTES[dtype])))
        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, len(data)]}
    header = json.dumps(tensors).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    return str(path)


@pytest.mark.parametrize(("layout", "kind"), [
    ("fp8", "fp8"), ("fp8_scaled", "fp8"),
    ("mxfp8", BLOCK_SCALED_FP8), ("nvfp4", BLOCK_SCALED_FP8),
    # The records are U8 tensors themselves: an FP8 file carrying them is FP8.
    ("fp8_records", "fp8"), ("nvfp4_records", BLOCK_SCALED_FP8),
    # Records naming two kinds answer nothing; the NVFP4 layer's U8 weight does.
    ("mixed_records", BLOCK_SCALED_FP8),
])
def test_an_fp8_header_names_block_scaled_files_apart(tmp_path, layout, kind):
    path = _checkpoint(tmp_path / f"{layout}.safetensors", layout)
    if layout in RECORDS:
        from dgx_monarch.adapters import layer_records

        assert layer_records.sniff_layer_formats(path) == frozenset(RECORDS[layout].values())
    assert pair_kind(path, "fp8") == kind


def test_other_kinds_pass_through_and_an_unread_header_adds_nothing(tmp_path):
    assert pair_kind(_checkpoint(tmp_path / "bf16.safetensors", "bf16"), "bf16") == "bf16"
    # A full-precision file the loader casts to fp8 holds no F8 tensor; no
    # record measured one, so it reads apart from FP8 storage.
    assert pair_kind(_checkpoint(tmp_path / "cast.safetensors", "bf16"), "fp8") == CAST_FP8
    assert pair_kind(None, "fp8") == "fp8"   # no file to read: the quant kind
    # The caller read this header a moment before; a file gone since adds nothing.
    assert pair_kind(str(tmp_path / "missing.safetensors"), "fp8") == "fp8"


@pytest.mark.parametrize(("artifact", "kind"), [
    ("nvfp4", BLOCK_SCALED_FP8), ("nvfp4_mixed", BLOCK_SCALED_FP8), ("mxfp8", BLOCK_SCALED_FP8),
    ("fp8_scaled", None), ("fp8_e4m3fn", None), ("bf16", None), ("", None),
])
def test_the_sweep_matrix_predicts_the_kind_the_runtime_reads(artifact, kind):
    from test_sweep_matrix import matrix

    assert matrix.auto_pair_kind({"artifact_quant": artifact}) == kind


def _resolve(monkeypatch, path, pair_differs, px=1280, weight_dtype="default"):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: path
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(latent_scale, "comfy_spatial_downscale", lambda _path: 8)
    # The headers carry no Qwen-Image signature; the Init node's family widget
    # names it, and auto follows the named family.
    model = ModelSpec(
        mesh=MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                      attention="TORCH_FLASH", sync_ulysses=True,
                      worker_args={family_select.WORKER_ARG_KEY: "qwen_image"}),
        unet_name="qwen.safetensors", options={"weight_dtype": weight_dtype})
    return common.resolve_topology(model, torch.zeros(1, 16, px // 8, px // 8), 4.0,
                                   pair_differs=pair_differs)


@pytest.mark.parametrize("layout", ["fp8", "fp8_scaled"])
def test_an_fp8_qwen_image_equal_pair_takes_the_bytes_equal_ulysses_row(
        monkeypatch, tmp_path, layout):
    path = _checkpoint(tmp_path / f"{layout}.safetensors", layout)
    topology, _sage, reason = _resolve(monkeypatch, path, pair_differs=False)
    assert topology.describe() == "uly2" and "table row 71" in reason
    assert "cfg2 table row 68 was measured on prompt pairs whose text rows differ" in reason
    topology, _sage, reason = _resolve(monkeypatch, path, pair_differs=True)
    assert topology.describe() == "cfg2" and "table row 68" in reason


@pytest.mark.parametrize("layout", ["mxfp8", "nvfp4"])
def test_a_block_scaled_qwen_image_equal_pair_keeps_cfg2(monkeypatch, tmp_path, layout):
    path = _checkpoint(tmp_path / f"{layout}.safetensors", layout)
    for differs in (False, True):
        topology, _sage, reason = _resolve(monkeypatch, path, pair_differs=differs)
        assert topology.describe() == "cfg2" and "table row 68" in reason, differs


def test_a_cast_qwen_image_equal_pair_keeps_cfg2(monkeypatch, tmp_path):
    """A BF16 file the loader casts to fp8 reads fp8 to auto, but row 71's
    reading was taken on FP8 storage; no record measured a cast file, so its
    equal pair stays on cfg2 like any other pair."""
    path = _checkpoint(tmp_path / "bf16.safetensors", "bf16")
    for differs in (False, True):
        topology, _sage, reason = _resolve(monkeypatch, path, pair_differs=differs,
                                           weight_dtype="fp8_e4m3fn")
        assert topology.describe() == "cfg2" and "table row 68" in reason, differs


def test_the_notice_claims_a_bytes_equal_layout_only_for_fp8_storage():
    claim = "uly2 on the Torch Flash kernel renders this file bytes equal to one GPU"
    assert claim in cfg_pair_notice.text("qwen_image", "fp8")
    assert claim in cfg_pair_notice.text("krea2", "fp8")
    for family in ("qwen_image", "krea2", "boogu"):
        for kind in (BLOCK_SCALED_FP8, CAST_FP8):
            message = cfg_pair_notice.text(family, kind)
            assert claim not in message and "No layout is recorded bytes equal" in message


@pytest.mark.parametrize(("layout", "weight_dtype", "kind"), [
    ("nvfp4", "default", BLOCK_SCALED_FP8),
    ("fp8_scaled", "default", "fp8"),
    ("bf16", "fp8_e4m3fn", CAST_FP8),  # the loader's cast: fp8 to auto, apart to a pair scope
    ("bf16", "default", "bf16"),
])
def test_the_notice_reads_the_kind_auto_reads(monkeypatch, tmp_path, layout, weight_dtype, kind):
    path = _checkpoint(tmp_path / f"{layout}.safetensors", layout)
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: path
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    model = ModelSpec(
        mesh=MeshSpec(handle=SimpleNamespace(world=2), topology_preset="cfg2",
                      attention="TORCH_FLASH", sync_ulysses=True,
                      worker_args={family_select.WORKER_ARG_KEY: "krea2"}),
        unet_name="model.safetensors", options={"weight_dtype": weight_dtype})
    assert cfg_pair_notice._family(model) == ("krea2", kind)
