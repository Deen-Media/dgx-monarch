"""FSDP presets and dials.

`cfg2+fsdp` and `dp2+fsdp` are in the preset table; a dual-model family refuses
cfg-parallel under fsdp on both the slot decision and the adapter grant; the
worker_args key `fsdp_prefetch_depth` fills FSDP2's explicit forward-prefetch
list with the next blocks, and 1 keeps the empty-list pin.
"""
from __future__ import annotations

import types

import pytest
import torch
import torch.distributed.fsdp

from dgx_monarch import config_schema
from dgx_monarch.actor import sample_protocol
from dgx_monarch.actor.store_detect import LivePrecisionEvidence
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.fsdp import apply_fsdp_capacity_mode
from dgx_monarch.topology import PRESETS, topology_from_preset


@pytest.mark.parametrize(("preset", "cfg", "dp"), [
    ("cfg2+fsdp", 2, 1),
    ("dp2+fsdp", 1, 2),
])
def test_capacity_presets_resolve_on_the_pair(preset, cfg, dp):
    assert preset in PRESETS
    topo = topology_from_preset(preset, 2)
    assert (topo.cfg, topo.dp, topo.fsdp, topo.sequence_parallel) == (cfg, dp, True, 1)
    assert topo.describe() == preset


def test_capacity_presets_need_the_launch_order_guard_only_when_they_share_ranks():
    from dgx_monarch.actor.worker_env import nccl_launch_order_needed

    assert nccl_launch_order_needed({"cfg": 2, "fsdp": True}) is True
    assert nccl_launch_order_needed({"dp": 2, "fsdp": True}) is False  # dp is not model parallel


def _worker(fsdp: bool, cfg: int = 2):
    return types.SimpleNamespace(
        topology={"cfg": cfg, "ulysses": 1, "ring": 1, "dp": 1, "fsdp": fsdp},
        world=2, store=None)


def test_dm_cfg2_predicate_is_false_under_fsdp():
    assert sample_protocol._dual_model_cfg2_topology(_worker(fsdp=False)) is True
    assert sample_protocol._dual_model_cfg2_topology(_worker(fsdp=True)) is False


def test_dual_model_family_refuses_cfg_parallel_under_fsdp(monkeypatch):
    families = {"cond.sft": "ideogram4", "unc.sft": "ideogram4"}
    monkeypatch.setattr(
        sample_protocol, "_sniff_request_family",
        lambda w, spec: families.get(spec.get("unet_name")) if spec else None)
    request = {"model": {"unet_name": "cond.sft"}, "uncond_model": {"unet_name": "unc.sft"}}
    with pytest.raises(UnsupportedModelError, match="each rank would shard a different") as raised:
        sample_protocol.dual_model_cfg2_slot(_worker(fsdp=True), request)
    assert "[dgxm:P" in str(raised.value)
    # A batched-cfg family under cfg2+fsdp is the ordinary combined path.
    monkeypatch.setattr(sample_protocol, "_sniff_request_family", lambda w, spec: "krea2")
    assert sample_protocol.dual_model_cfg2_slot(
        _worker(fsdp=True), {"model": {"unet_name": "k.sft"}}) is None


def test_worker_grant_mirrors_the_predicate_fsdp_clause():
    """Both readings of dm-cfg2 must exclude fsdp, or slotting and the adapter
    grant diverge. Pin the worker's expression textually next to the predicate."""
    import inspect

    from dgx_monarch.actor import worker as worker_module

    source = inspect.getsource(worker_module.GPUWorker._inject_for_topology)
    assert 'and not bool(self.topology.get("fsdp"))' in source
    predicate = inspect.getsource(sample_protocol._dual_model_cfg2_topology)
    assert 'and not bool(topo.get("fsdp"))' in predicate


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2).to(torch.bfloat16)
        self.prefetch_calls: list = []

    def set_modules_to_forward_prefetch(self, modules):
        self.prefetch_calls.append(list(modules))


class _FakePatcher:
    def __init__(self, model):
        self.model = types.SimpleNamespace(diffusion_model=model)
        self.size = 0


@pytest.fixture
def fully_shard_passthrough(monkeypatch):
    monkeypatch.setattr(torch.distributed.fsdp, "fully_shard",
                        lambda module, **kwargs: module)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)


def _model(blocks: int):
    dm = torch.nn.Module()
    dm.blocks = torch.nn.ModuleList(_Block() for _ in range(blocks))
    return dm


def _evidence():
    return LivePrecisionEvidence(quant_kind="bf16", live_dtype_profile="all_bf16",
                                 checkpoint_kind="bf16")


def test_default_depth_keeps_the_empty_list_pin(fully_shard_passthrough):
    dm = _model(3)
    apply_fsdp_capacity_mode(_FakePatcher(dm), "bf16", None, _evidence())
    assert [b.prefetch_calls for b in dm.blocks] == [[[]], [[]], [[]]]


def test_depth_names_the_next_blocks_and_never_the_root(fully_shard_passthrough):
    dm = _model(4)
    apply_fsdp_capacity_mode(_FakePatcher(dm), "bf16", None, _evidence(), prefetch_depth=2)
    b = list(dm.blocks)
    assert b[0].prefetch_calls == [[b[1], b[2]]]
    assert b[1].prefetch_calls == [[b[2], b[3]]]
    assert b[2].prefetch_calls == [[b[3]]]
    assert b[3].prefetch_calls == [[]]


@pytest.mark.parametrize("value", [1, 2, 8])
def test_prefetch_depth_accepts_one_to_eight(value):
    out = config_schema.validate_worker_args({"fsdp_prefetch_depth": value})
    assert out["fsdp_prefetch_depth"] == value


@pytest.mark.parametrize("value", [0, 9, True, "2", 2.0])
def test_prefetch_depth_refuses_out_of_range_and_wrong_types(value):
    with pytest.raises(config_schema.ClusterConfigError, match="fsdp_prefetch_depth"):
        config_schema.validate_worker_args({"fsdp_prefetch_depth": value})
