"""The facts auto read reach the render's progress record, and nothing else.

``resolve_topology`` still unpacks to (topology, sage, reason) everywhere. Its
auto path also carries the table row, family, quant kind and latent
megapixels, the CFG and cfg++ flag, the cfg evidence row or the skipped cfg
row, and the batch, which the eager submit hands to the progress tracker with
the topology label and the resolved kernel. A preset, or a test double's plain
tuple, carries none.

No comfy, no GPU, no mesh: the resolution and submit seams are stubbed.
"""
from __future__ import annotations

import copy
import pickle
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import latent_scale, telemetry
from dgx_monarch.nodes import common
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.resolved_topology import (
    ResolvedTopology,
    TopologyFacts,
    telemetry_facts,
)
from dgx_monarch.topology import Topology

_AUDIT = ("family", "cfg", "keeps_uncond", "evidence_row", "skipped_cfg_row", "batch")
_NO_FACTS = {"auto_row": None, "megapixels": None, "quant": None, **dict.fromkeys(_AUDIT)}


def _model(preset: str = "auto", weight_dtype: str = "default") -> ModelSpec:
    return ModelSpec(
        mesh=MeshSpec(handle=SimpleNamespace(world=2), topology_preset=preset,
                      attention="TORCH_FLASH", sync_ulysses=True),
        unet_name="model.safetensors", options={"weight_dtype": weight_dtype},
    )


def _header_stubs(monkeypatch, family: str, quant: str, ratio: int) -> None:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: "/models/model.safetensors"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(common, "sniff_checkpoint_for_topology",
                        lambda _path: (family, quant))
    monkeypatch.setattr(latent_scale, "comfy_spatial_downscale", lambda _path: ratio)
    monkeypatch.setattr(common, "_impl_warned", set())


def test_auto_carries_its_row_quant_and_exact_megapixels(monkeypatch):
    # The shipped Lens graph: 84 latent cells at 16x is 1344 px, row 74.
    _header_stubs(monkeypatch, "lens", "fp8", 16)
    resolved = common.resolve_topology(
        _model(), torch.zeros(1, 128, 84, 84), cfg_value=4.0)

    topology, sage, reason = resolved
    assert (topology.describe(), sage) == ("uly2", False)
    assert "table row 74" in reason and "lens fp8 at 1.81 MP" in reason
    assert resolved.facts == TopologyFacts(
        row=74, quant="fp8", megapixels=1344 * 1344 / 1e6, family="lens", cfg=4.0,
        keeps_uncond=False, evidence_row=None, skipped_cfg_row=None, batch=1)
    assert telemetry_facts(resolved) == {
        "auto_row": 74, "megapixels": 1.806, "quant": "fp8", "family": "lens",
        "cfg": 4.0, "keeps_uncond": False, "evidence_row": None,
        "skipped_cfg_row": None, "batch": 1}


def test_the_recorded_quant_is_the_one_the_table_matched(monkeypatch):
    # A full-precision header loaded as fp8 matches the fp8 rows, so the
    # record names fp8, not the header's bf16.
    _header_stubs(monkeypatch, "flux2", "bf16", 16)
    resolved = common.resolve_topology(
        _model(weight_dtype="fp8_e4m3fn"), torch.zeros(1, 128, 96, 96), cfg_value=4.0)
    assert "flux2 fp8 at 2.36 MP" in resolved[2] and "table row 27" in resolved[2]
    assert telemetry_facts(resolved)["quant"] == "fp8"
    assert telemetry_facts(resolved)["auto_row"] == 27


def test_world_one_names_no_table_row(monkeypatch):
    _header_stubs(monkeypatch, "lens", "fp8", 16)
    resolved = common.resolve_topology(
        _model(), torch.zeros(1, 128, 84, 84), cfg_value=4.0, world=1)
    assert resolved[0].describe() == "single"
    assert "table row" not in resolved[2]
    assert telemetry_facts(resolved) == {
        "auto_row": None, "megapixels": 1.806, "quant": "fp8", "family": "lens",
        "cfg": 4.0, "keeps_uncond": False, "evidence_row": None,
        "skipped_cfg_row": None, "batch": 1}


def test_a_cfg_pp_latent_records_the_flag_and_the_evidence_row(monkeypatch):
    # The Krea2 template: 1536 square at CFG 1 under euler_cfg_pp, tagged by
    # the sampler node, takes row 15 and names the record that admitted it.
    _header_stubs(monkeypatch, "krea2", "fp8", 16)
    latent = {"samples": torch.zeros(1, 16, 96, 96),
              common.CFG_KEEPS_UNCOND_METADATA_KEY: True}
    resolved = common._resolve_topology_for_latent(_model(), latent, 1.0, 2)
    assert resolved[0].describe() == "cfg2" and "table row 15" in resolved[2]
    facts = telemetry_facts(resolved)
    assert (facts["auto_row"], facts["evidence_row"], facts["skipped_cfg_row"]) == (15, 15, None)
    assert (facts["family"], facts["cfg"], facts["keeps_uncond"], facts["batch"]) == (
        "krea2", 1.0, True, 1)


def test_an_ungranted_kind_records_the_cfg_row_it_skipped(monkeypatch):
    # Krea2 INT8 inside row 15's band: the record grants FP8 and BF16 only
    # (INT8 cfg2 was exact but no faster), so the walk skips 15 and lands on
    # row 12; the record names both.
    _header_stubs(monkeypatch, "krea2", "int8", 8)
    resolved = common.resolve_topology(
        _model(), torch.zeros(2, 16, 152, 152), cfg_value=3.5)
    assert resolved[0].describe() == "uly2"
    facts = telemetry_facts(resolved)
    assert (facts["auto_row"], facts["evidence_row"], facts["skipped_cfg_row"]) == (12, None, 15)
    assert facts["batch"] == 2 and facts["keeps_uncond"] is False


def test_a_preset_reads_no_header_and_carries_no_facts(monkeypatch):
    def no_sniff(_path):
        raise AssertionError("a preset must not read the checkpoint header")

    monkeypatch.setattr(common, "sniff_checkpoint_for_topology", no_sniff)
    resolved = common.resolve_topology(
        _model("uly2+fsdp"), torch.zeros(1, 128, 84, 84), cfg_value=4.0)
    assert resolved[0].describe() == "uly2+fsdp"
    assert telemetry_facts(resolved) == _NO_FACTS


def test_the_carrier_is_a_three_tuple_that_survives_copy_and_pickle():
    facts = TopologyFacts(row=20, quant="bf16", megapixels=1.048576)
    resolved = ResolvedTopology(Topology(world=2, cfg=2), False, "auto: cfg2", facts)
    assert resolved == (Topology(world=2, cfg=2), False, "auto: cfg2")
    assert len(resolved) == 3
    # The round trip reads back bytes this test just wrote, never outside data.
    for clone in (copy.copy(resolved), copy.deepcopy(resolved),
                  pickle.loads(pickle.dumps(resolved))):  # noqa: S301
        assert clone == resolved and clone.facts == facts
    assert telemetry_facts((Topology(), False, "test")) == _NO_FACTS
    assert telemetry_facts(object()) == _NO_FACTS


def _submit_meta(monkeypatch, resolved) -> dict:
    from dgx_monarch.nodes.pending import PendingRenderHandoff
    from render_sessions_helpers import _real_handle, _stub_direct_submit

    handle = _real_handle()
    spec = _stub_direct_submit(monkeypatch, handle, lambda *_args, **_kwargs: object())
    monkeypatch.setattr(common, "resolve_topology", lambda *_args: resolved)
    started = []
    monkeypatch.setattr(
        telemetry.render_progress, "start",
        lambda steps, meta=None, *, token=None: started.append(meta))
    pending = common.submit_render(
        spec, {}, {"samples": object()}, cfg_value=1.0, steps_hint=2,
        handoff=PendingRenderHandoff())
    try:
        (meta,) = started
        return meta
    finally:
        pending.abandon()


def test_submit_hands_the_tracker_the_label_kernel_and_auto_facts(monkeypatch):
    facts = TopologyFacts(row=11, quant="fp8", megapixels=2.359296)
    resolved = ResolvedTopology(Topology(world=1), True, "auto: single", facts)
    assert _submit_meta(monkeypatch, resolved) == {
        "model": "model.safetensors",
        "topology": "single",
        # The kernel the render runs, after the row's sage bracket applied.
        "attention": "SAGE_AUTO",
        "auto_row": 11, "megapixels": 2.359, "quant": "fp8",
        **dict.fromkeys(_AUDIT),
    }


def test_submit_with_a_plain_tuple_records_nulls_not_a_failure(monkeypatch):
    meta = _submit_meta(monkeypatch, (Topology(world=1), False, "test"))
    assert meta == {
        "model": "model.safetensors", "topology": "single",
        "attention": "TORCH_FLASH", **_NO_FACTS,
    }


@pytest.mark.parametrize(("preset", "world"), [
    ("uly2", 2), ("cfg2", 2), ("uly2+fsdp", 2), ("single", 1)])
def test_the_recorded_label_is_the_preset_name(preset, world):
    # render_submit records topo.describe() as the finish record's topology,
    # so the idle line names the preset the user chose. This pins describe()
    # only. The JS formatter's parity with describe() runs under node in
    # test_readiness_panel_model.py (the topology label test for every preset).
    from dgx_monarch.topology import topology_from_preset

    assert topology_from_preset(preset, world).describe() == preset
