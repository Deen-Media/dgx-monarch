"""The per-cond dispatch refuses a condition-comparing output hook only where it
can split a batch stock would have kept together.

The cfg slice path refuses an ``attn1_output_patch`` that does not declare
``condition_shard_local`` (NAG compares cond and uncond inside one batched
call), because each rank holds one branch of that call. The per-cond dispatch
runs when ComfyUI itself cannot batch the conditionings, and then stock also
runs separate calls: with one entry per slot each rank's call is one of
stock's calls, and the hook sees what it sees on one GPU. With several entries
in a slot, stock can batch an entry of one slot with an entry of the other
while a third forces separate calls; the dispatch's contiguous groups then
split that batch, and the hook would run on part of it. That case refuses.
"""
from __future__ import annotations

import pytest
import torch

import dgx_monarch.adapters.cfg_dispatch as cfg_dispatch
import dgx_monarch.adapters.cfg_parallel as cfg_parallel
from dgx_monarch.adapters import attention_patches
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag
from test_cfg_paths import (
    _cond_slot,
    _ConstantFamilyAdapter,
    _install_patcher_extension,
    _ToyCondBatch,
)


def _nag(_args):  # an output hook that compares conditions
    return _args


def _shard_local(_args):
    return _args


setattr(_shard_local, attention_patches.ATTENTION_PATCH_CAPABILITIES_ATTR,
        frozenset({attention_patches.CONDITION_SHARD_LOCAL_CAPABILITY}))


def _dispatch(monkeypatch, conds, patches, sharded_weights=False):
    pe = _install_patcher_extension(monkeypatch)
    monkeypatch.setattr(cfg_dispatch, "cfg_world", lambda: 2)
    monkeypatch.setattr(cfg_dispatch, "cfg_rank", lambda: 0)
    monkeypatch.setattr(cfg_parallel, "_all_gather_cfg", lambda local: torch.cat([local] * 2))
    model_options = {"transformer_options": {"patches": patches}}
    cfg_dispatch.install_cond_dispatch_wrapper(_ConstantFamilyAdapter(), model_options,
                                               sharded_weights=sharded_weights)
    inner = _ToyCondBatch()
    executor = pe.WrapperExecutor.new_executor(inner, pe.get_all_wrappers(
        pe.WrappersMP.CALC_COND_BATCH, model_options, is_model_options=True))
    executor.execute(object(), conds, torch.arange(6.0).reshape(2, 3), torch.tensor([1.0]),
                     model_options)
    return inner.seen


def test_one_entry_per_slot_dispatches_with_the_hook_as_stock_runs_it(monkeypatch):
    seen = _dispatch(monkeypatch, [_cond_slot(7, 2.0), _cond_slot(11, 5.0)],
                     {"attn1_output_patch": [_nag]})
    assert seen == [(0, 7)]
    assert cfg_dispatch.cfg_path_taken() == "dispatch"


def test_a_slot_of_several_entries_refuses_a_condition_comparing_hook(monkeypatch):
    combined = _cond_slot(7, 2.0) + _cond_slot(9, 3.0)
    with pytest.raises(UnsupportedModelError) as caught:
        _dispatch(monkeypatch, [combined, _cond_slot(11, 5.0)], {"attn1_output_patch": [_nag]})
    tag = parse_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert "condition_shard_local" in str(caught.value)


@pytest.mark.parametrize("patches", [{}, {"attn1_output_patch": [_shard_local]},
                                     {"attn1_patch": [_nag]}])
def test_a_slot_of_several_entries_dispatches_without_such_a_hook(monkeypatch, patches):
    combined = _cond_slot(7, 2.0) + _cond_slot(9, 3.0)
    seen = _dispatch(monkeypatch, [combined, _cond_slot(11, 5.0)], patches)
    assert seen == [(0, 7), (0, 9)]


def _hooked(slot, group):
    return [{**item, "hooks": group} for item in slot]


def test_cfg_with_fsdp_refuses_different_hook_groups_on_the_ranks(monkeypatch):
    """Explicit cfg2+fsdp, a hook-carrying positive and a negative of another
    length (Boogu and Omnigen2 dispatch such a pair by their constant): the
    ranks share one set of weight shards, so each patching its own group would
    mix them. The dispatch refuses before any collective."""
    group = object()   # comfy keys hook groups by identity
    pair = [_hooked(_cond_slot(7, 2.0), group), _cond_slot(11, 5.0)]
    with pytest.raises(UnsupportedModelError) as caught:
        _dispatch(monkeypatch, pair, {}, sharded_weights=True)
    tag = parse_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert "fsdp" in str(caught.value) and "hook" in str(caught.value)
    # Whole weights on each rank: each rank applies its own group, as stock does per call.
    assert _dispatch(monkeypatch, pair, {}, sharded_weights=False) == [(0, 7)]
    # One group on both, or none: nothing to mix.
    shared = [_hooked(_cond_slot(7, 2.0), group), _hooked(_cond_slot(11, 5.0), group)]
    assert _dispatch(monkeypatch, shared, {}, sharded_weights=True) == [(0, 7)]
    plain = [_cond_slot(7, 2.0), _cond_slot(11, 5.0)]
    assert _dispatch(monkeypatch, plain, {}, sharded_weights=True) == [(0, 7)]


@pytest.mark.parametrize("fsdp", [True, False], ids=["cfg2+fsdp", "cfg2"])
def test_the_worker_arms_the_hook_group_refusal_from_the_fsdp_switch(monkeypatch, fsdp):
    """The Worker's own injection: cfg2 with fsdp installs the per-cond dispatch
    with sharded weights, and cfg2 alone without, so the refusal above meets
    exactly the layout whose ranks share weight shards."""
    import types

    import dgx_monarch.adapters as adapters_pkg
    from dgx_monarch.actor import partial_load_guard, store_fsdp
    from dgx_monarch.actor import worker as worker_mod
    from dgx_monarch.adapters import fsdp as fsdp_mod

    adapter = types.SimpleNamespace(family="krea2")
    sharded: list = []
    monkeypatch.setattr(adapters_pkg, "adapter_for", lambda *a, **k: adapter)
    monkeypatch.setattr(store_fsdp, "validate_injection", lambda *a, **k: None)
    monkeypatch.setattr(partial_load_guard, "install", lambda *a, **k: None)
    monkeypatch.setattr(worker_mod, "_maybe_compile_dit", lambda *a, **k: None)
    monkeypatch.setattr(fsdp_mod, "apply_fsdp_capacity_mode", lambda *a, **k: None)
    monkeypatch.setattr(cfg_parallel, "assert_cfg_parallel_supported", lambda *a, **k: None)
    monkeypatch.setattr(cfg_parallel, "install_cfg_parallel_wrapper", lambda *a, **k: None)
    monkeypatch.setattr(cfg_dispatch, "install_cond_dispatch_wrapper",
                        lambda _adapter, _options, *, sharded_weights=False:
                        sharded.append(sharded_weights))
    rank = types.SimpleNamespace(
        topology={"cfg": 2, "ulysses": 1, "ring": 1, "dp": 1, "fsdp": fsdp},
        world=2, store=types.SimpleNamespace(family_override=None, lora_low_rss=False))
    base_patcher = types.SimpleNamespace(
        model=types.SimpleNamespace(diffusion_model=object()), model_options={})
    worker_mod.GPUWorker._inject_for_topology(rank, base_patcher, "bf16", [], None)
    assert sharded == [fsdp]
