"""Kroma reuses Krea2 detection and sampling; these are CPU compatibility checks.

The small headers preserve the observed dtype and quantization namespaces, not
the checkpoint dimensions. They establish neither image fidelity nor GPU support.
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
from dgx_monarch.adapters.detect import (
    fsdp_lora_admission_property,
    read_safetensors_keys,
    sniff_checkpoint,
    sniff_checkpoint_for_topology,
    sniff_metadata_keys,
)
from dgx_monarch.nodes import common
from dgx_monarch.nodes.common import MeshSpec, ModelSpec, resolve_sample_attention
from dgx_monarch.topology import choose_auto_topology
from dgx_monarch.topology_cfg_evidence import sniffed_checkpoint_variant
from test_krea2_comfy_exactness import comfy_krea2 as comfy_krea2

_CONVROT = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


def _checkpoint(path, quant, merge_metadata=True):
    """Write bounded, valid tensor ranges with the publisher's native Krea2 keys."""
    layout = {
        "txtfusion.projector.weight": ("BF16", [8, 8]),
        "txtmlp.1.weight": ("BF16", [8, 8]),
        "blocks.0.attn.wq.weight": ("BF16", [8, 8]),
        "blocks.0.attn.qknorm.qnorm.scale": ("F32", [2]),
    }
    # The merge record the published BF16 release's header carries; the auto
    # table reads it to keep Kroma off Krea2's cfg2 row (checkpoint_variant).
    metadata = {"trim": "0.9", "copy_prefixes": "txtfusion.,txtmlp.", "base": "raw.safetensors",
                "scale": "1.0", "finetune": "kroma.safetensors", "target": "turbo.safetensors"}
    if not merge_metadata:  # a Krea2 RAW or Turbo header carries no merge record
        metadata = {}
    payloads = {}
    if quant == "int8":
        descriptor = json.dumps(_CONVROT).encode()
        layout.update({
            "blocks.0.attn.wq.weight": ("I8", [8, 8]),
            "blocks.0.attn.wq.weight_scale": ("F32", [8, 1]),
            "blocks.0.attn.wq.comfy_quant": ("U8", [len(descriptor)]),
            "blocks.0.attn.qknorm.qnorm.scale": ("F16", [2]),
        })
        metadata = {"_quantization_metadata": json.dumps({
            "format_version": "1.0", "layers": {"blocks.0.attn.wq": _CONVROT},
        })}
        payloads["blocks.0.attn.wq.comfy_quant"] = descriptor
    tensors = {"__metadata__": metadata}
    data = bytearray()
    for name, (dtype, shape) in layout.items():
        size = math.prod(shape) * {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1}[dtype]
        offset = len(data)
        data.extend(payloads.get(name, bytes(size)))
        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, len(data)]}
    header = json.dumps(tensors).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    return path


@pytest.mark.parametrize(("filename", "quant"), [
    ("kroma-v0.3.1-turbo-opd.safetensors", "bf16"),
    ("kromaKrea2_opd.safetensors", "int8"),
    ("renamed-checkpoint.safetensors", "bf16"),
    ("renamed-checkpoint.safetensors", "int8"),
])
def test_kroma_headers_use_krea2_auto_route(tmp_path, filename, quant):
    path = _checkpoint(tmp_path / filename, quant)
    assert sniff_checkpoint(path) == ("krea2", quant)
    family, storage = sniff_checkpoint_for_topology(path)
    assert (family, storage) == ("krea2", quant)
    decision = choose_auto_topology(family, storage, 1024 * 1024 / 1_000_000, 2, cfg_value=1.0)
    assert decision.topology.ulysses == 2
    assert decision.topology.cfg == decision.topology.ring == decision.topology.dp == 1
    assert not decision.topology.fsdp
    assert resolve_sample_attention("TORCH_FLASH", decision.sage) == "TORCH_FLASH"


def _resolve_auto(monkeypatch, path, px, cfg_value, family_adapter=None):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(path)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(latent_scale, "comfy_spatial_downscale", lambda _path: 8)
    # The Init node's family_adapter widget reaches the driver as this worker arg.
    worker_args = {family_select.WORKER_ARG_KEY: family_adapter} if family_adapter else {}
    model = ModelSpec(
        mesh=MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                      attention="TORCH_FLASH", sync_ulysses=True, worker_args=worker_args),
        unet_name=path.name, options={"weight_dtype": "default"})
    return common.resolve_topology(model, torch.zeros(1, 4, px // 8, px // 8), cfg_value)


@pytest.mark.parametrize("px", [1152, 1536])
def test_a_kroma_header_keeps_ulysses_where_krea2_takes_cfg2(monkeypatch, tmp_path, px):
    # 2026-10-16: Kroma's shipped zeroed negative folds on one GPU and cfg2 read
    # off one GPU at 1152 and 1536, so the header, not the file name, keeps it
    # off row 15 at CFG 1.5 (the card's top CFG) while a Krea2 header keeps cfg2.
    kroma = _checkpoint(tmp_path / "renamed-checkpoint.safetensors", "bf16")
    assert sniffed_checkpoint_variant(str(kroma), "krea2") == "kroma"
    topology, sage, reason = _resolve_auto(monkeypatch, kroma, px, 1.5)
    assert topology.describe() == "uly2" and not sage
    assert "table row 12" in reason
    assert "cfg2 table row 15 was measured on this kroma checkpoint" in reason

    krea2 = _checkpoint(tmp_path / "kroma-v0.3.1-turbo-opd.safetensors", "bf16",
                        merge_metadata=False)
    assert sniffed_checkpoint_variant(str(krea2), "krea2") is None
    topology, _sage, reason = _resolve_auto(monkeypatch, krea2, px, 1.5)
    assert topology.describe() == "cfg2" and "table row 15" in reason


@pytest.mark.parametrize("px", [1152, 1536])
def test_a_krea2_family_override_keeps_the_kroma_exclusion(monkeypatch, tmp_path, px):
    # The override names the family and nothing else: forcing the krea2 that a
    # Kroma header already reads must not lift row 15's exclusion at CFG 1.5,
    # and forcing it over a plain Krea2 header must not drop that file's cfg2.
    monkeypatch.setattr(common, "_impl_warned", set())
    kroma = _checkpoint(tmp_path / "renamed-checkpoint.safetensors", "bf16")
    for family_adapter in ("krea2", None):
        topology, sage, reason = _resolve_auto(monkeypatch, kroma, px, 1.5, family_adapter)
        assert topology.describe() == "uly2" and not sage, family_adapter
        assert "table row 12" in reason, family_adapter
        assert "cfg2 table row 15 was measured on this kroma checkpoint" in reason

    krea2 = _checkpoint(tmp_path / "krea2-turbo.safetensors", "bf16", merge_metadata=False)
    for family_adapter in ("krea2", None):
        topology, _sage, reason = _resolve_auto(monkeypatch, krea2, px, 1.5, family_adapter)
        assert topology.describe() == "cfg2" and "table row 15" in reason, family_adapter


def test_kroma_int8_keeps_convrot_metadata_and_fsdp_lora_refusal(tmp_path):
    path = _checkpoint(tmp_path / "kromaKrea2_opd.safetensors", "int8")
    keys, header = read_safetensors_keys(path)
    assert "blocks.0.attn.wq.comfy_quant" in keys
    assert header["__tensors__"]["blocks.0.attn.wq.weight"]["dtype"] == "I8"
    assert header["__tensors__"]["txtfusion.projector.weight"]["dtype"] == "BF16"
    assert header["__tensors__"]["blocks.0.attn.qknorm.qnorm.scale"]["dtype"] == "F16"
    assert json.loads(header["_quantization_metadata"]) == {
        "format_version": "1.0", "layers": {"blocks.0.attn.wq": _CONVROT},
    }
    assert sniff_metadata_keys(path) == ("_quantization_metadata",)
    assert fsdp_lora_admission_property(path) == "quantized_shards"


def test_kroma_native_comfy_sampling_needs_no_shift_patch(comfy_krea2):
    """Exercise upstream model construction and schedule, without allocating a DiT."""
    model_base = pytest.importorskip("comfy.model_base")
    supported = pytest.importorskip("comfy.supported_models")
    sampling = pytest.importorskip("comfy.model_sampling")
    config = supported.Krea2({"image_model": "krea2", "disable_unet_model_creation": True})
    model = config.get_model({}, device=torch.device("cpu"))
    assert model.model_type is model_base.ModelType.FLUX
    assert isinstance(model.model_sampling, sampling.ModelSamplingFlux)
    assert config.sampling_settings["multiplier"] == 1.0
    assert model.model_sampling.shift == pytest.approx(1.15)
    times = torch.tensor([0.125, 0.5, 0.875], dtype=torch.float64)
    expected = math.exp(1.15) / (math.exp(1.15) + 1 / times - 1)
    torch.testing.assert_close(model.model_sampling.sigma(times), expected)
    torch.testing.assert_close(model.model_sampling.timestep(times), times)
    # A linear flow shift of 1.15 is not the native Flux mu=1.15 schedule.
    assert not torch.allclose(expected, 1.15 * times / (1 + 0.15 * times))
