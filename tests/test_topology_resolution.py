"""Auto topology resolution on the driver: header kinds, latent scale, and its warnings."""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.nodes import common, render_result
from dgx_monarch.nodes.common import MeshSpec, ModelSpec


def test_auto_warns_once_when_header_cannot_bind_hardware_evidence_scope(
    monkeypatch, caplog,
):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: "/models/wan.safetensors"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(common, "sniff_checkpoint_for_topology", lambda _path: ("wan", "fp8"))
    monkeypatch.setattr(common, "_impl_warned", set())
    topology = object()
    monkeypatch.setattr(
        common, "choose_auto_topology",
        lambda *_args, **_kwargs: SimpleNamespace(
            topology=topology, sage=False, reason="test row"),
    )
    model = ModelSpec(
        mesh=MeshSpec(
            handle=SimpleNamespace(world=2), topology_preset="auto",
            attention="TORCH_FLASH", sync_ulysses=True),
        unet_name="wan.safetensors",
    )
    latent = torch.zeros(1, 4, 64, 64)

    for _attempt in range(2):
        actual, _sage, _reason = common.resolve_topology(
            model, latent, cfg_value=1.0)
        assert actual is topology

    warnings = [record.message for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "cannot bind the graph, artifact, LoRA, quantization, and topology" in warnings[0]
    assert "docs/MODELS.md" in warnings[0]


@pytest.mark.parametrize(("weight_dtype", "effective_kind", "sage"), [
    ("default", "fp16", False),
    ("fp8_e4m3fn", "fp8", True),
])
def test_fp16_header_kind_preserved_unless_loader_requests_fp8(
        monkeypatch, weight_dtype, effective_kind, sage):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: "/models/model.safetensors"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(common, "sniff_checkpoint_for_topology", lambda _path: ("krea2", "fp16"))

    mesh = MeshSpec(
        handle=SimpleNamespace(world=2),
        topology_preset="auto",
        attention="TORCH_FLASH",
        sync_ulysses=True,
    )
    model = ModelSpec(
        mesh=mesh,
        unet_name="model.safetensors",
        options={"weight_dtype": weight_dtype},
    )
    latent = torch.zeros(1, 4, 192, 192)

    topology, selected_sage, reason = common.resolve_topology(model, latent, cfg_value=4.0)

    assert topology.fsdp is False
    assert selected_sage is sage
    assert f"krea2 {effective_kind}" in reason


@pytest.mark.parametrize(("weight_dtype", "effective_kind"), [
    ("default", "fp32"),
    ("fp8_e4m3fn", "fp8"),
])
def test_fp32_header_kind_still_takes_a_loader_requested_fp8_row(
        monkeypatch, weight_dtype, effective_kind):
    """An FP8 load request moves an FP32 header to the FP8 row, like BF16 and FP16.

    The sniff reports an all-FP32 header as itself, not as BF16, so the FP8
    upgrade in resolve_topology must list fp32: the loader casts the file to
    FP8 either way.
    """
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: "/models/model.safetensors"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(common, "sniff_checkpoint_for_topology", lambda _path: ("krea2", "fp32"))
    captured = {}

    def choose(family, quant, megapixels, world, **kwargs):
        captured["quant"] = quant
        return SimpleNamespace(
            topology=SimpleNamespace(fsdp=False), sage=False, reason="test row")

    monkeypatch.setattr(common, "choose_auto_topology", choose)
    model = ModelSpec(
        mesh=MeshSpec(
            handle=SimpleNamespace(world=2),
            topology_preset="auto",
            attention="TORCH_FLASH",
            sync_ulysses=True,
        ),
        unet_name="model.safetensors",
        options={"weight_dtype": weight_dtype},
    )

    common.resolve_topology(model, torch.zeros(1, 4, 192, 192), cfg_value=4.0)

    assert captured["quant"] == effective_kind


@pytest.mark.parametrize(
    ("latent_downscale", "latent_shape"),
    [(8, (128, 96)), (16, (64, 48))],
)
def test_explicit_latent_downscale_controls_auto_topology_megapixels(
    monkeypatch, latent_downscale, latent_shape
):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: "/models/model.safetensors"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(common, "sniff_checkpoint_for_topology", lambda _path: ("krea2", "fp16"))
    topology = object()
    captured = {}

    def choose(family, quant, megapixels, world, **kwargs):
        captured.update({
            "family": family,
            "quant": quant,
            "megapixels": megapixels,
            "world": world,
            **kwargs,
        })
        return SimpleNamespace(topology=topology, sage=False, reason="test row")

    monkeypatch.setattr(common, "choose_auto_topology", choose)
    model = ModelSpec(
        mesh=MeshSpec(
            handle=SimpleNamespace(world=2),
            topology_preset="auto",
            attention="TORCH_FLASH",
            sync_ulysses=True,
        ),
        unet_name="model.safetensors",
    )
    latent = torch.zeros(2, 128, *latent_shape)

    actual, selected_sage, reason = common.resolve_topology(
        model, latent, cfg_value=4.0, latent_downscale=latent_downscale
    )

    assert actual is topology
    assert selected_sage is False
    assert reason == "test row"
    assert captured == {
        "family": "krea2",
        "quant": "fp16",
        "megapixels": pytest.approx(1024 * 768 / 1e6),
        "world": 2,
        "cfg_value": 4.0,
        "batch_size": 2,
    }


@pytest.mark.parametrize("value", [False, "16", 16.0, 0, -1])
def test_explicit_latent_downscale_must_be_a_positive_integer(value):
    model = ModelSpec(
        mesh=MeshSpec(
            handle=SimpleNamespace(world=1),
            topology_preset="uly1",
            attention="TORCH_FLASH",
            sync_ulysses=False,
        ),
        unet_name="model.safetensors",
    )

    with pytest.raises(ValueError, match="latent_downscale must be"):
        common.resolve_topology(
            model, torch.zeros(1, 4, 8, 8), cfg_value=1.0,
            latent_downscale=value,
        )


def test_sampler_latent_metadata_routes_scale_to_topology_resolution(monkeypatch):
    sentinel = object()
    calls = []

    def resolve(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(common, "resolve_topology", resolve)
    samples = object()
    latent = {
        "samples": samples,
        common.LATENT_DOWNSCALE_METADATA_KEY: 16,
    }

    assert common._resolve_topology_for_latent(
        "model", latent, cfg_value=4.0, world=2
    ) is sentinel
    assert calls == [
        (("model", samples, 4.0, 2), {"latent_downscale": 16})
    ]


def test_latent_downscale_metadata_is_stripped_from_transport_and_output(monkeypatch):
    latent = {
        "samples": torch.zeros(1, 4, 8, 8),
        "noise_mask": "preserved",
        common.LATENT_DOWNSCALE_METADATA_KEY: 16,
    }

    assert render_result._latent_without_topology_metadata(latent) == {
        "samples": latent["samples"],
        "noise_mask": "preserved",
    }

    rendered = torch.ones(1, 4, 8, 8)
    monkeypatch.setattr(
        render_result, "verify_cross_rank_signatures", lambda _results, _topo: None
    )
    monkeypatch.setattr(render_result, "read_latent_result", lambda value: value)
    out = render_result._finish_render(
        [{"dp_rank": 0, "latent": rendered}],
        SimpleNamespace(dp=1),
        latent,
    )

    assert out["samples"] is rendered
    assert out["noise_mask"] == "preserved"
    assert common.LATENT_DOWNSCALE_METADATA_KEY not in out


def _pixart_layout_checkpoint(path) -> str:
    """A real header carrying the PixArt shape: two of LTX's three key paths."""
    import json
    import struct

    tensors = {
        "adaln_single.linear.weight": {"dtype": "BF16", "shape": [1],
                                       "data_offsets": [0, 2]},
        "transformer_blocks.0.attn1.to_q.weight": {"dtype": "BF16", "shape": [1],
                                                   "data_offsets": [2, 4]},
        "pos_embed.proj.weight": {"dtype": "BF16", "shape": [1],
                                  "data_offsets": [4, 6]},
        "__metadata__": {"format": "pt"},
    }
    header = json.dumps(tensors).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 6)
    return str(path)


def _auto_model(unet_name: str):
    return ModelSpec(
        mesh=MeshSpec(
            handle=SimpleNamespace(world=2), topology_preset="auto",
            attention="TORCH_FLASH", sync_ulysses=True),
        unet_name=unet_name,
    )


def _near_miss_lines(caplog):
    return [record.getMessage() for record in caplog.records
            if "the closest is" in record.getMessage()]


def test_an_unmatched_header_is_named_by_its_nearest_signature(
    monkeypatch, caplog, tmp_path,
):
    """An unmatched header logs its closest row once per checkpoint path.

    The hint fires only where no row matched, so it cannot be wrong about a
    supported checkpoint, and it names the family to type into family_adapter."""
    import logging

    first = _pixart_layout_checkpoint(tmp_path / "one.safetensors")
    second = _pixart_layout_checkpoint(tmp_path / "two.safetensors")
    names = {"one.safetensors": first, "two.safetensors": second}
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, name: names[name]
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(common, "_impl_warned", set())
    latent = torch.zeros(1, 4, 64, 64)

    with caplog.at_level(logging.WARNING):
        common.resolve_topology(_auto_model("one.safetensors"), latent, cfg_value=1.0)
        common.resolve_topology(_auto_model("one.safetensors"), latent, cfg_value=1.0)
    lines = _near_miss_lines(caplog)
    assert len(lines) == 1
    assert "ltx (2 of 3 key paths present, missing patchify_proj.)" in lines[0]
    assert "family_adapter" in lines[0]

    # Keyed by path, not family: a second unmatched checkpoint in the same
    # session is a different file, and a per-family key would already be spent.
    with caplog.at_level(logging.WARNING):
        common.resolve_topology(_auto_model("two.safetensors"), latent, cfg_value=1.0)
    assert len(_near_miss_lines(caplog)) == 2


def test_a_single_rank_render_still_gets_the_near_miss(monkeypatch, caplog, tmp_path):
    """The nearest-row hint is about an unmatched header, not cluster
    evidence, so a world-1 auto render names the closest row too; only the
    hardware-evidence advisory is cluster-scoped."""
    import logging

    path = _pixart_layout_checkpoint(tmp_path / "solo.safetensors")
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: path
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(common, "_impl_warned", set())
    latent = torch.zeros(1, 4, 64, 64)
    spec = ModelSpec(
        mesh=MeshSpec(
            handle=SimpleNamespace(world=1), topology_preset="auto",
            attention="TORCH_FLASH", sync_ulysses=True),
        unet_name="solo.safetensors",
    )

    with caplog.at_level(logging.WARNING):
        common.resolve_topology(spec, latent, cfg_value=1.0)
    assert len(_near_miss_lines(caplog)) == 1


def test_a_matched_family_never_reports_a_near_miss(monkeypatch, caplog, tmp_path):
    import logging

    path = _pixart_layout_checkpoint(tmp_path / "krea2.safetensors")
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: path
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(common, "sniff_checkpoint_for_topology",
                        lambda _path: ("krea2", "fp8"))
    monkeypatch.setattr(common, "_impl_warned", set())
    latent = torch.zeros(1, 4, 64, 64)

    with caplog.at_level(logging.WARNING):
        common.resolve_topology(_auto_model("krea2.safetensors"), latent, cfg_value=1.0)
    assert _near_miss_lines(caplog) == []
