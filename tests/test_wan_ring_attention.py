"""Fail-closed plain-Wan Ring2 full-context attention coverage."""

from __future__ import annotations

import enum
import sys
import types
import warnings
from threading import Event, Thread
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.actor import worker_env, worker_status
from dgx_monarch.adapters import base
from dgx_monarch.adapters import wan_ring_attention as wan_ring
from dgx_monarch.adapters import wan_ring_full_context as ring_full_context

PURE_RING2 = {"dp": 1, "cfg": 1, "ulysses": 1, "ring": 2, "fsdp": False}


def _install_runtime(monkeypatch, **overrides):
    ulysses_group = object()
    ring_group = object()
    state = {
        "initialized": True,
        "world": 2,
        "global_rank": 0,
        "dp": 1,
        "cfg": 1,
        "sp": 2,
        "ulysses": 1,
        "ring": 2,
        "ulysses_group": ulysses_group,
        "ring_group": ring_group,
        "instance_ulysses_group": ulysses_group,
        "instance_ring_group": ring_group,
        "ulysses_group_size": 1,
        "ring_group_size": 2,
        "ulysses_rank": 0,
        "ring_rank": 0,
        "sp_rank": 0,
        "ring_global_ranks": (0, 1),
        "selector_drift": False,
        "drop_binding": False,
        "bypass_processor": False,
        "leak_processor_result": False,
        "raise_after_processor": False,
        "extra_processor_call": False,
        "replace_path_after_result": False,
        "p2p_fail": False,
        "p2p_wait_fail": False,
        "p2p_wait_false": False,
        "p2p_requests": None,
        "coalesced_p2p": False,
        "remote_k": None,
        "remote_v": None,
        "constructs": [],
        "instances": [],
        "calls": [],
        "p2p_batches": [],
        "p2p_waits": 0,
        "aten_calls": [],
        "processor_results": [],
    }
    state.update(overrides)

    class FakeAttnType(enum.Enum):
        TORCH_FLASH = "torch_flash"
        SAGE_FP8 = "sage_fp8"

    class StubUSP:
        def __init__(self, **kwargs):
            state["constructs"].append(kwargs)
            state["instances"].append(self)
            self.ulysses_pg = state["instance_ulysses_group"]
            self.ring_pg = state["instance_ring_group"]
            self.attn_type = None if state["drop_binding"] else kwargs.get("attn_type")
            self.attn_processor = (
                None if state["drop_binding"] else kwargs.get("attn_processor")
            )
            self.ring_attn_fn = lambda q, _k, _v, **_kwargs: q

        def __call__(self, _attn, q, k, v):
            state["calls"].append((q, k, v))
            if state["bypass_processor"]:
                return q
            if state["leak_processor_result"]:
                return self.attn_processor(q, k, v)
            result = self.ring_attn_fn(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=False,
                group=self.ring_pg,
                attn_type=self.attn_type,
                attn_processor=self.attn_processor,
                attn_layer=None,
                joint_tensor_key=None,
                joint_tensor_value=None,
                joint_strategy="none",
                q_descale=None,
                k_descale=None,
                v_descale=None,
            )
            if state["raise_after_processor"]:
                raise RuntimeError("outer attention failed")
            if state["extra_processor_call"]:
                self.attn_processor(q, k, v)
            if state["replace_path_after_result"]:
                self.ring_attn_fn = lambda replaced_q, _k, _v, **_kwargs: replaced_q
            return result

    def select_flash_attn_impl(attn_type, *, stage, attn_processor=None):
        assert stage == "fwd-only"
        if state["selector_drift"] or isinstance(attn_type, FakeAttnType):
            return object()
        return attn_processor

    sp_group = SimpleNamespace(
        ulysses_group=state["ulysses_group"],
        ring_group=state["ring_group"],
        ulysses_rank=state["ulysses_rank"],
        ring_rank=state["ring_rank"],
    )
    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.get_data_parallel_world_size = lambda: state["dp"]
    distributed.get_classifier_free_guidance_world_size = lambda: state["cfg"]
    distributed.get_sequence_parallel_world_size = lambda: state["sp"]
    distributed.get_sequence_parallel_rank = lambda: state["sp_rank"]
    distributed.get_ulysses_parallel_world_size = lambda: state["ulysses"]
    distributed.get_ring_parallel_world_size = lambda: state["ring"]
    distributed.get_sp_group = lambda: sp_group
    long_ctx = types.ModuleType("xfuser.core.long_ctx_attention")
    long_ctx.xFuserLongContextAttention = StubUSP
    core = types.ModuleType("xfuser.core")
    core.distributed = distributed
    core.long_ctx_attention = long_ctx
    xfuser = types.ModuleType("xfuser")
    xfuser.core = core
    kernels = types.ModuleType("yunchang.kernels")
    kernels.AttnType = FakeAttnType
    kernels.select_flash_attn_impl = select_flash_attn_impl
    yunchang = types.ModuleType("yunchang")
    yunchang.kernels = kernels
    for name, module in (
        ("xfuser", xfuser),
        ("xfuser.core", core),
        ("xfuser.core.distributed", distributed),
        ("xfuser.core.long_ctx_attention", long_ctx),
        ("yunchang", yunchang),
        ("yunchang.kernels", kernels),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: state["initialized"])

    def get_world_size(group=None):
        if group is state["instance_ulysses_group"]:
            return state["ulysses_group_size"]
        if group is state["instance_ring_group"]:
            return state["ring_group_size"]
        if group is None:
            return state["world"]
        raise RuntimeError("unknown group")

    def get_rank(group=None):
        if group is state["instance_ulysses_group"]:
            return state["ulysses_rank"]
        if group is state["instance_ring_group"]:
            return state["ring_rank"]
        if group is None:
            return state["global_rank"]
        raise RuntimeError("unknown group")

    monkeypatch.setattr(torch.distributed, "get_world_size", get_world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", get_rank)

    def get_global_rank(group, group_rank):
        if group is state["instance_ring_group"]:
            return state["ring_global_ranks"][group_rank]
        raise RuntimeError("unknown group")

    monkeypatch.setattr(torch.distributed, "get_global_rank", get_global_rank)

    def p2p_op(op, tensor, peer=None, group=None, tag=0, group_peer=None):
        return SimpleNamespace(
            op=op,
            tensor=tensor,
            peer=peer,
            group=group,
            tag=tag,
            group_peer=group_peer,
        )

    class Work:
        def wait(self):
            state["p2p_waits"] += 1
            if state["p2p_wait_fail"]:
                raise RuntimeError("P2P wait failed")
            return not state["p2p_wait_false"]

    def batch_isend_irecv(operations):
        if state["p2p_fail"]:
            raise RuntimeError("P2P launch failed")
        batch_index = len(state["p2p_batches"])
        state["p2p_batches"].append(operations)
        remote_k = state["remote_k"]
        remote_v = state["remote_v"]
        operations[1].tensor.copy_(
            operations[0].tensor
            if remote_k is None
            else remote_k[batch_index : batch_index + 1]
        )
        operations[3].tensor.copy_(
            operations[2].tensor
            if remote_v is None
            else remote_v[batch_index : batch_index + 1]
        )
        if state["p2p_requests"] is not None:
            return state["p2p_requests"]
        if state["coalesced_p2p"]:
            return [Work()]
        return [Work() for _operation in operations]

    monkeypatch.setattr(torch.distributed, "P2POp", p2p_op)
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", batch_isend_irecv)
    monkeypatch.setattr(wan_ring, "_probe_aten", lambda _processor: "float32")

    def aten_op(qh, kh, vh, **_kwargs):
        result = (
            torch.zeros_like(qh),
            torch.zeros(
                qh.shape[0], qh.shape[1], qh.shape[2],
                dtype=torch.float32, device=qh.device,
            ),
        )
        state["aten_calls"].append((qh, kh, vh))
        state["processor_results"].append(result)
        return result

    monkeypatch.setattr(ring_full_context, "_aten_flash_attention_op", lambda: aten_op)
    return state, FakeAttnType


def _make_binding(monkeypatch, **runtime):
    state, attn_type = _install_runtime(monkeypatch, **runtime)
    binding = wan_ring.make_wan_ring_usp_attention(
        "TORCH_FLASH", True, PURE_RING2, 2, 7
    )
    return binding, state, attn_type


def test_specialization_uses_opaque_fallback_and_attests_authority(monkeypatch):
    binding, state, attn_type = _make_binding(monkeypatch)
    construction = state["constructs"][0]
    assert not isinstance(construction["attn_type"], attn_type)
    assert construction["attn_processor"] is not None
    assert construction["use_sync"] is True
    assert binding.attestation.as_dict() == {
        "setup_generation": 7,
        "configured_world": 2,
        "configured_dp": 1,
        "configured_cfg": 1,
        "configured_sp": 2,
        "configured_ulysses": 1,
        "configured_ring": 2,
        "xfuser_sp": 2,
        "xfuser_dp": 1,
        "xfuser_cfg": 1,
        "xfuser_ulysses": 1,
        "xfuser_ring": 2,
        "instance_ulysses_group_size": 1,
        "instance_ring_group_size": 2,
        "instance_ulysses_group_rank": 0,
        "instance_ring_group_rank": 0,
        "xfuser_sequence_rank": 0,
        "process_global_rank": 0,
        "ring_slot0_global_rank": 0,
        "ring_slot1_global_rank": 1,
        "selected_path": "wan_ring_aten_full_context",
        "context_strategy": "rank_ordered_p2p_kv_per_local_batch",
        "processor_calls_per_local_batch": 1,
        "expected_lse_dtype": "float32",
        "validated_lse_dtype": "float32_construction_and_per_call",
    }


def test_stock_torch_flash_constructor_remains_processor_free(monkeypatch):
    state, attn_type = _install_runtime(monkeypatch)
    stock = base.make_usp_attention("TORCH_FLASH", True)
    construction = state["constructs"][0]
    assert construction == {"use_sync": True, "attn_type": attn_type.TORCH_FLASH}
    q = torch.zeros(1, 2, 128)
    assert torch.equal(stock(q, q, q, heads=1), q)


@pytest.mark.parametrize(
    "topology,world",
    [
        ({**PURE_RING2, "ulysses": 2}, 4),
        ({**PURE_RING2, "ring": 3}, 3),
        ({**PURE_RING2, "dp": 2}, 4),
        ({**PURE_RING2, "cfg": 2}, 4),
        ({**PURE_RING2, "fsdp": True}, 2),
        (PURE_RING2, 4),
    ],
)
def test_only_pure_ring2_topology_is_admitted(monkeypatch, topology, world):
    _install_runtime(monkeypatch)
    with pytest.raises(ValueError, match=r"pure ring2|topology product"):
        wan_ring.make_wan_ring_usp_attention("TORCH_FLASH", True, topology, world, 1)


@pytest.mark.parametrize(
    "runtime",
    [
        {"initialized": False},
        {"world": 3},
        {"dp": 2},
        {"cfg": 2},
        {"sp": 1},
        {"ulysses": 2},
        {"ring": 1},
        {"ulysses_group_size": 2},
        {"ring_group_size": 1},
        {"ulysses_rank": True},
        {"ring_rank": 2},
        {"sp_rank": 1},
        {"global_rank": 1},
        {"ring_global_ranks": (0, 0)},
        {"ring_global_ranks": (1, 0)},
        {"instance_ulysses_group": object()},
        {"instance_ring_group": object()},
    ],
)
def test_authority_mismatch_refuses_during_construction(monkeypatch, runtime):
    _install_runtime(monkeypatch, **runtime)
    with pytest.raises(
        base.UnsupportedModelError, match=r"authority|groups|ranks|topology"
    ):
        wan_ring.make_wan_ring_usp_attention(
            "TORCH_FLASH", True, PURE_RING2, 2, 1
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("ulysses_group_size", True),
        ("ulysses_group_size", 1.0),
        ("ring_group_size", True),
        ("ring_group_size", 2.0),
        ("ulysses_rank", False),
        ("ulysses_rank", 0.0),
        ("ring_rank", False),
        ("ring_rank", 0.0),
    ],
)
def test_instance_group_sizes_and_ranks_reject_bool_and_non_int(
    monkeypatch, field, value
):
    _install_runtime(monkeypatch, **{field: value})
    with pytest.raises(base.UnsupportedModelError, match=r"authority|groups|ranks"):
        wan_ring.make_wan_ring_usp_attention(
            "TORCH_FLASH", True, PURE_RING2, 2, 1
        )


@pytest.mark.parametrize("drift", ["selector_drift", "drop_binding"])
def test_dependency_binding_drift_refuses_before_return(monkeypatch, drift):
    _install_runtime(monkeypatch, **{drift: True})
    with pytest.raises(RuntimeError, match=r"fallback|retain"):
        wan_ring.make_wan_ring_usp_attention(
            "TORCH_FLASH", True, PURE_RING2, 2, 1
        )


def test_aten_probe_failure_refuses_before_xfuser_construction(monkeypatch):
    state, _ = _install_runtime(monkeypatch)
    monkeypatch.setattr(
        wan_ring,
        "_probe_aten",
        lambda _processor: (_ for _ in ()).throw(RuntimeError("schema drift")),
    )
    with pytest.raises(RuntimeError, match="schema drift"):
        wan_ring.make_wan_ring_usp_attention(
            "TORCH_FLASH", True, PURE_RING2, 2, 1
        )
    assert state["constructs"] == []


def test_aten_probe_validates_fp16_and_bf16_contracts(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    calls = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    mode = FakeTensorMode()
    with mode:
        def aten_op(qh, _kh, _vh, **_kwargs):
            calls.append(qh.dtype)
            return (
                torch.zeros_like(qh),
                torch.zeros(
                    qh.shape[0], qh.shape[1], qh.shape[2],
                    dtype=torch.float32, device=qh.device,
                ),
            )

        monkeypatch.setattr(
            ring_full_context, "_aten_flash_attention_op", lambda: aten_op
        )
        assert wan_ring._probe_aten(wan_ring._TorchFlashProcessor()) == "float32"

    assert calls == [torch.float16, torch.bfloat16]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"mask": torch.ones(1, dtype=torch.bool)}, "mask"),
        ({"attn_precision": torch.float32}, "precision"),
        ({"skip_reshape": True}, "surface"),
        ({"skip_output_reshape": True}, "surface"),
        ({"enable_gqa": True}, "surface"),
        ({"unknown": True}, "surface"),
    ],
)
def test_comfy_surface_refuses_before_xfuser(monkeypatch, kwargs, match):
    binding, state, _ = _make_binding(monkeypatch)
    q = torch.zeros(1, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(base.UnsupportedModelError, match=match):
        binding(q, q, q, heads=1, **kwargs)
    assert state["calls"] == []


def test_drop_rows_keeps_ulysses_only_guard_before_xfuser(monkeypatch):
    binding, state, _ = _make_binding(monkeypatch)
    q = torch.zeros(1, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(base.UnsupportedModelError, match="ulysses-only"):
        binding(q, q, q, heads=1, drop_rows=[1])
    assert state["calls"] == []


def test_non_tensor_input_typed_refuses_before_attribute_access(monkeypatch):
    binding, state, _ = _make_binding(monkeypatch)
    with pytest.raises(base.UnsupportedModelError, match="tensor q/k/v"):
        binding(object(), object(), object(), heads=1)
    assert state["calls"] == []


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_valid_specialized_call_reaches_xfuser_after_preflight(monkeypatch, dtype):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch)
    assert binding.status_attestation()["specialized_call_completed"] is False
    mode = FakeTensorMode()
    with mode:
        q = torch.zeros(1, 2, 128, device="cuda", dtype=dtype)
        out = binding(q, q, q, heads=1)
    assert out.shape == q.shape
    assert out.dtype == dtype
    assert len(state["calls"]) == 1
    assert len(state["processor_results"]) == 1
    assert len(state["processor_results"][0]) == 2
    assert binding.status_attestation()["specialized_call_completed"] is True


@pytest.mark.parametrize("rank", [0, 1])
def test_full_kv_materialization_uses_sequence_rank_order(monkeypatch, rank):
    state, _ = _install_runtime(monkeypatch, ring_rank=rank)
    local_k = torch.tensor([[[[10.0]], [[11.0]]]])
    local_v = torch.tensor([[[[20.0]], [[21.0]]]])
    remote_k = torch.tensor([[[[30.0]], [[31.0]]]])
    remote_v = torch.tensor([[[[40.0]], [[41.0]]]])
    state["remote_k"] = remote_k
    state["remote_v"] = remote_v

    full_k, full_v = ring_full_context._materialize_full_kv_batch(
        local_k,
        local_v,
        group=state["ring_group"],
        world=2,
        rank=rank,
        global_ranks=state["ring_global_ranks"],
    )

    expected_k = (
        torch.cat((local_k, remote_k), dim=1)
        if rank == 0
        else torch.cat((remote_k, local_k), dim=1)
    )
    expected_v = (
        torch.cat((local_v, remote_v), dim=1)
        if rank == 0
        else torch.cat((remote_v, local_v), dim=1)
    )
    assert torch.equal(full_k, expected_k)
    assert torch.equal(full_v, expected_v)
    assert len(state["p2p_batches"]) == 1
    operations = state["p2p_batches"][0]
    assert len(operations) == 4
    assert all(operation.group is state["ring_group"] for operation in operations)
    expected_peer = state["ring_global_ranks"][1 - rank]
    assert all(operation.peer == expected_peer for operation in operations)
    assert all(operation.group_peer is None for operation in operations)


def test_full_context_processes_one_local_batch_at_a_time(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch)
    with FakeTensorMode():
        q = torch.zeros(3, 2, 128, device="cuda", dtype=torch.float16)
        out = binding(q, q, q, heads=1)

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert len(state["p2p_batches"]) == 3
    assert len(state["aten_calls"]) == 3
    assert all(qh.shape == (1, 1, 2, 128) for qh, _kh, _vh in state["aten_calls"])
    assert all(kh.shape == (1, 1, 4, 128) for _qh, kh, _vh in state["aten_calls"])
    assert binding.status_attestation()["specialized_call_completed"] is True


def test_completed_batch_buffers_are_released_before_next_materialization(monkeypatch):
    import weakref

    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, _state, _ = _make_binding(monkeypatch)
    previous: list[weakref.ReferenceType[torch.Tensor]] = []
    materializations = 0

    def materialize(k_batch, v_batch, **_kwargs):
        nonlocal materializations
        assert all(reference() is None for reference in previous)
        full_k = torch.cat((k_batch, k_batch), dim=1)
        full_v = torch.cat((v_batch, v_batch), dim=1)
        previous[:] = [weakref.ref(full_k), weakref.ref(full_v)]
        materializations += 1
        return full_k, full_v

    def aten_op(qh, _kh, _vh, **_kwargs):
        return (
            torch.zeros_like(qh),
            torch.zeros(
                qh.shape[0], qh.shape[1], qh.shape[2],
                dtype=torch.float32, device=qh.device,
            ),
        )

    monkeypatch.setattr(ring_full_context, "_materialize_full_kv_batch", materialize)
    monkeypatch.setattr(ring_full_context, "_aten_flash_attention_op", lambda: aten_op)
    with FakeTensorMode():
        q = torch.zeros(3, 2, 128, device="cuda", dtype=torch.float16)
        out = binding(q, q, q, heads=1)

    assert out.shape == q.shape
    assert materializations == 3
    assert all(reference() is None for reference in previous)


def test_coalesced_single_p2p_work_is_a_valid_backend_contract(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch, coalesced_p2p=True)
    with FakeTensorMode():
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        out = binding(q, q, q, heads=1)

    assert out.shape == q.shape
    assert state["p2p_waits"] == 1
    assert len(state["aten_calls"]) == 1
    assert binding.status_attestation()["specialized_call_completed"] is True


@pytest.mark.parametrize(
    "drift,match",
    [
        ("initialized", "authority"),
        ("world", "world changed"),
        ("rank", "rank changed"),
        ("sequence_rank", "sequence rank changed"),
        ("global_rank", "global-rank mapping"),
        ("global_mapping", "global-rank mapping"),
        ("duplicate_mapping", "global-rank mapping"),
        ("group", "wrong process group"),
    ],
)
def test_call_time_ring_authority_drift_refuses_before_p2p(
    monkeypatch, drift, match
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch)
    if drift == "initialized":
        state["initialized"] = False
    elif drift == "world":
        state["ring_group_size"] = 1
    elif drift == "rank":
        state["ring_rank"] = 1
    elif drift == "sequence_rank":
        state["sp_rank"] = 1
    elif drift == "global_rank":
        state["global_rank"] = 1
    elif drift == "global_mapping":
        state["ring_global_ranks"] = (1, 0)
    elif drift == "duplicate_mapping":
        state["ring_global_ranks"] = (0, 0)
    elif drift == "group":
        state["instances"][0].ring_pg = object()
    else:
        raise AssertionError(f"unknown drift {drift}")

    with FakeTensorMode():
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(base.UnsupportedModelError, match=match):
            binding(q, q, q, heads=1)

    assert state["p2p_batches"] == []
    assert state["aten_calls"] == []
    assert binding.status_attestation()["specialized_call_completed"] is False


def test_call_time_ring_path_replacement_refuses_before_spoofed_receipt(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch)
    instance = state["instances"][0]

    def replacement(q, k, v, **_kwargs):
        return instance.attn_processor(q, k, v)[0]

    instance.ring_attn_fn = replacement
    with FakeTensorMode():
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(base.UnsupportedModelError, match="path changed"):
            binding(q, q, q, heads=1)

    assert state["p2p_batches"] == []
    assert state["aten_calls"] == []
    assert binding.status_attestation()["specialized_call_completed"] is False


def test_ring_path_replacement_during_execution_refuses_attestation(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch, replace_path_after_result=True)
    with FakeTensorMode():
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(base.UnsupportedModelError, match="during execution"):
            binding(q, q, q, heads=1)

    assert len(state["p2p_batches"]) == 1
    assert len(state["aten_calls"]) == 1
    assert binding.status_attestation()["specialized_call_completed"] is False


@pytest.mark.parametrize(
    "runtime,error,match",
    [
        ({"p2p_fail": True}, RuntimeError, "launch failed"),
        ({"p2p_wait_fail": True}, RuntimeError, "wait failed"),
        ({"p2p_wait_false": True}, base.UnsupportedModelError, "did not complete"),
        ({"p2p_requests": []}, base.UnsupportedModelError, "request contract"),
    ],
)
def test_p2p_failure_never_calls_processor_or_attests(
    monkeypatch, runtime, error, match
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch, **runtime)
    with FakeTensorMode():
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(error, match=match):
            binding(q, q, q, heads=1)

    assert state["aten_calls"] == []
    assert state["p2p_waits"] == (
        4 if runtime.get("p2p_wait_fail") or runtime.get("p2p_wait_false") else 0
    )
    assert binding.status_attestation()["specialized_call_completed"] is False


@pytest.mark.parametrize("wait_failure", [False, True])
def test_malformed_p2p_list_drains_every_callable_work(
    monkeypatch, wait_failure
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch)
    waits = []

    class Work:
        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail

        def wait(self):
            waits.append(self.name)
            if self.fail:
                raise RuntimeError("first wait failed")
            return True

    monkeypatch.setattr(
        torch.distributed,
        "batch_isend_irecv",
        lambda _operations: [Work("first", wait_failure), object(), Work("last")],
    )
    error = RuntimeError if wait_failure else base.UnsupportedModelError
    match = "first wait failed" if wait_failure else "request contract"
    with FakeTensorMode():
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(error, match=match):
            binding(q, q, q, heads=1)

    assert waits == ["first", "last"]
    assert state["aten_calls"] == []
    assert binding.status_attestation()["specialized_call_completed"] is False


def test_p2p_baseexception_drains_every_work_and_preserves_primary(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch)
    waits = []

    class FatalWait(BaseException):
        pass

    primary = FatalWait("primary wait failure")

    class Work:
        def __init__(self, name, error=None):
            self.name = name
            self.error = error

        def wait(self):
            waits.append(self.name)
            if self.error is not None:
                raise self.error
            return True

    monkeypatch.setattr(
        torch.distributed,
        "batch_isend_irecv",
        lambda _operations: [
            Work("first", primary),
            Work("second", RuntimeError("secondary wait failure")),
            Work("third"),
            Work("fourth"),
        ],
    )
    with FakeTensorMode():
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(FatalWait) as error:
            binding(q, q, q, heads=1)

    assert error.value is primary
    assert waits == ["first", "second", "third", "fourth"]
    assert state["aten_calls"] == []
    assert binding.status_attestation()["specialized_call_completed"] is False


def test_p2p_first_false_precedes_later_baseexception_after_full_drain(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch)
    waits = []

    class LaterFatal(BaseException):
        pass

    class Work:
        def __init__(self, name, outcome):
            self.name = name
            self.outcome = outcome

        def wait(self):
            waits.append(self.name)
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return self.outcome

    monkeypatch.setattr(
        torch.distributed,
        "batch_isend_irecv",
        lambda _operations: [
            Work("first", False),
            Work("second", LaterFatal("later fatal")),
            Work("third", True),
            Work("fourth", True),
        ],
    )
    with FakeTensorMode():
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(base.UnsupportedModelError, match="did not complete"):
            binding(q, q, q, heads=1)

    assert waits == ["first", "second", "third", "fourth"]
    assert state["aten_calls"] == []
    assert binding.status_attestation()["specialized_call_completed"] is False


@pytest.mark.parametrize("request_count", [2, 3])
def test_partial_all_callable_p2p_list_drains_then_refuses(
    monkeypatch, request_count
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch)
    waits = []

    class Work:
        def wait(self):
            waits.append(True)
            return True

    monkeypatch.setattr(
        torch.distributed,
        "batch_isend_irecv",
        lambda _operations: [Work() for _ in range(request_count)],
    )
    with FakeTensorMode():
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(base.UnsupportedModelError, match="request contract"):
            binding(q, q, q, heads=1)

    assert len(waits) == request_count
    assert state["aten_calls"] == []
    assert binding.status_attestation()["specialized_call_completed"] is False


def test_extra_processor_call_refuses_exact_receipt(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch, extra_processor_call=True)
    with FakeTensorMode():
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(base.UnsupportedModelError, match="exactly one"):
            binding(q, q, q, heads=1)

    assert len(state["aten_calls"]) == 2
    assert binding.status_attestation()["specialized_call_completed"] is False


def test_successful_outer_bypass_without_processor_receipt_refuses(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch, bypass_processor=True)
    mode = FakeTensorMode()
    with mode:
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(base.UnsupportedModelError, match="without executing"):
            binding(q, q, q, heads=1)
    assert len(state["calls"]) == 1
    assert binding.status_attestation()["specialized_call_completed"] is False


def test_leaked_unmerged_processor_tuple_refuses_without_attesting(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch, leak_processor_result=True)
    mode = FakeTensorMode()
    with mode:
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(base.UnsupportedModelError, match="non-tensor output"):
            binding(q, q, q, heads=1)
    assert len(state["calls"]) == 1
    assert len(state["processor_results"]) == 1
    assert len(state["processor_results"][0]) == 2
    assert binding.status_attestation()["specialized_call_completed"] is False


def test_processor_success_followed_by_outer_failure_does_not_attest(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, _state, _ = _make_binding(monkeypatch, raise_after_processor=True)
    mode = FakeTensorMode()
    with mode:
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="outer attention failed"):
            binding(q, q, q, heads=1)
    assert binding.status_attestation()["specialized_call_completed"] is False


def test_binding_serializes_exact_call_receipt_attribution():
    receipt = wan_ring._ProcessorReceipt()
    treatment_entered = Event()
    release_treatment = Event()
    bypass_entered = Event()
    results = []
    errors = []

    def implementation(kind):
        if kind == "treatment":
            treatment_entered.set()
            assert release_treatment.wait(2)
            receipt.mark_success()
            return kind
        bypass_entered.set()
        return kind

    binding = wan_ring._Binding(
        implementation, SimpleNamespace(as_dict=lambda: {}), receipt
    )

    def call(kind):
        try:
            results.append(binding(kind))
        except Exception as exc:
            errors.append(exc)

    treatment = Thread(target=call, args=("treatment",))
    bypass = Thread(target=call, args=("bypass",))
    treatment.start()
    assert treatment_entered.wait(2)
    bypass.start()
    assert not bypass_entered.wait(0.1)
    release_treatment.set()
    treatment.join(2)
    bypass.join(2)

    assert results == ["treatment"]
    assert len(errors) == 1
    assert isinstance(errors[0], base.UnsupportedModelError)
    assert "without executing" in str(errors[0])


def test_raising_specialized_call_does_not_mark_completion(monkeypatch):
    binding, state, _ = _make_binding(monkeypatch)
    q = torch.zeros(1, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(base.UnsupportedModelError, match="CUDA"):
        binding(q, q, q, heads=1)
    assert state["calls"] == []
    assert binding.status_attestation()["specialized_call_completed"] is False


@pytest.mark.parametrize("heads", [True, 0, -1, 1.0, "1"])
def test_invalid_head_counts_refuse_before_xfuser(monkeypatch, heads):
    binding, state, _ = _make_binding(monkeypatch)
    q = torch.zeros(1, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(base.UnsupportedModelError, match="positive integer head"):
        binding(q, q, q, heads=heads)
    assert state["calls"] == []


@pytest.mark.parametrize(
    "tensor,match",
    [
        (torch.zeros(1, 2, 128, dtype=torch.float32), "FP16/BF16"),
        (torch.zeros(1, 2, 64, dtype=torch.bfloat16), "dimension 128"),
        (torch.zeros(1, 2, 128, dtype=torch.bfloat16, requires_grad=True), "inference-only"),
        (torch.zeros(1, 2, 128, dtype=torch.bfloat16), "CUDA"),
    ],
)
def test_qkv_contract_refuses_before_xfuser(monkeypatch, tensor, match):
    binding, state, _ = _make_binding(monkeypatch)
    with pytest.raises(base.UnsupportedModelError, match=match):
        binding(tensor, tensor, tensor, heads=1)
    assert state["calls"] == []


@pytest.mark.parametrize(
    "case,match",
    [
        ("shape", r"equal .*q/k/v shapes"),
        ("dtype", "FP16/BF16"),
        ("device", "one CUDA device"),
        ("empty_inner", r"non-empty|inner dimension"),
        ("non_strided", "strided"),
    ],
)
def test_malformed_qkv_contract_refuses_before_xfuser(monkeypatch, case, match):
    from torch._subclasses.fake_tensor import FakeTensorMode

    binding, state, _ = _make_binding(monkeypatch)
    mode = FakeTensorMode()
    with mode:
        q = torch.zeros(1, 2, 128, device="cuda:0", dtype=torch.bfloat16)
        k = q
        v = q
        if case == "shape":
            k = torch.zeros(1, 3, 128, device="cuda:0", dtype=torch.bfloat16)
        elif case == "dtype":
            k = torch.zeros(1, 2, 128, device="cuda:0", dtype=torch.float16)
        elif case == "device":
            k = torch.zeros(1, 2, 128, device="cuda:1", dtype=torch.bfloat16)
        elif case == "empty_inner":
            q = k = v = torch.zeros(
                1, 2, 0, device="cuda:0", dtype=torch.bfloat16
            )
        elif case == "non_strided":
            indices = torch.tensor([[0], [0], [0]], device="cuda:0")
            values = torch.tensor([1.0], device="cuda:0", dtype=torch.bfloat16)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Sparse invariant checks are implicitly disabled.*",
                    category=UserWarning,
                )
                q = k = v = torch.sparse_coo_tensor(
                    indices,
                    values,
                    (1, 2, 128),
                    device="cuda:0",
                    check_invariants=False,
                )
        with pytest.raises(base.UnsupportedModelError, match=match):
            binding(q, k, v, heads=1)
    assert state["calls"] == []


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_processor_preserves_fp32_lse_and_accepts_both_default_scale_forms(
    monkeypatch, dtype
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    mode = FakeTensorMode()
    calls = []
    with mode:
        q = torch.zeros(1, 3, 2, 128, device="cuda", dtype=dtype)

        def aten_op(qh, kh, vh, **kwargs):
            calls.append(kwargs)
            return (
                torch.zeros_like(qh),
                torch.zeros(1, 2, 3, device=qh.device, dtype=torch.float32),
            )

        monkeypatch.setattr(
            ring_full_context, "_aten_flash_attention_op", lambda: aten_op
        )
        processor = wan_ring._TorchFlashProcessor()
        out_none, lse_none = processor(q, q, q, softmax_scale=None)
        scale = 1.0 / (128**0.5)
        out_exact, lse_exact = processor(q, q, q, softmax_scale=scale)
        with pytest.raises(base.UnsupportedModelError, match="default"):
            processor(q, q, q, softmax_scale=1.0)

    assert out_none.shape == out_exact.shape == q.shape
    assert out_none.dtype == out_exact.dtype == dtype
    assert lse_none.dtype == lse_exact.dtype == torch.float32
    assert calls == [
        {"dropout_p": 0.0, "is_causal": False, "scale": None},
        {"dropout_p": 0.0, "is_causal": False, "scale": scale},
    ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_processor_accepts_local_q_with_longer_full_kv(monkeypatch, dtype):
    from torch._subclasses.fake_tensor import FakeTensorMode

    observed = []
    with FakeTensorMode():
        q = torch.zeros(1, 2, 1, 128, device="cuda", dtype=dtype)
        k = torch.zeros(1, 4, 1, 128, device="cuda", dtype=dtype)
        v = torch.zeros_like(k)

        def aten_op(qh, kh, vh, **_kwargs):
            observed.append((qh.shape, kh.shape, vh.shape))
            return (
                torch.zeros_like(qh),
                torch.zeros(1, 1, 2, device=qh.device, dtype=torch.float32),
            )

        monkeypatch.setattr(
            ring_full_context, "_aten_flash_attention_op", lambda: aten_op
        )
        out, lse = wan_ring._TorchFlashProcessor()(q, k, v)

    assert out.shape == q.shape
    assert out.dtype == dtype
    assert lse.shape == (1, 1, 2)
    assert lse.dtype == torch.float32
    assert observed == [((1, 1, 2, 128), (1, 1, 4, 128), (1, 1, 4, 128))]


def test_processor_rejects_empty_full_kv_before_aten(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    called = []
    monkeypatch.setattr(
        ring_full_context,
        "_aten_flash_attention_op",
        lambda: called.append(True),
    )
    with FakeTensorMode():
        q = torch.zeros(1, 2, 1, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros(1, 0, 1, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(base.UnsupportedModelError, match="non-empty"):
            wan_ring._TorchFlashProcessor()(q, k, k)
    assert called == []


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"dropout_p": 0.1}, "dropout"),
        ({"causal": True}, "non-causal"),
        ({"window_size": (0, 1)}, "windows"),
        ({"softcap": 1.0}, "softcap"),
        ({"alibi_slopes": object()}, "ALiBi"),
        ({"return_softmax": True}, "probabilities"),
        ({"unexpected": True}, "arguments"),
    ],
)
def test_processor_rejects_unsupported_kernel_surface(kwargs, match):
    processor = wan_ring._TorchFlashProcessor()
    q = torch.zeros(1, 1, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(base.UnsupportedModelError, match=match):
        processor(q, q, q, **kwargs)


@pytest.mark.parametrize(
    "case,match",
    [
        ("non_tuple", "unsupported ATen result"),
        ("short_tuple", "unsupported ATen result"),
        ("non_tensor_out", "non-tensor"),
        ("non_tensor_lse", "non-tensor"),
        ("out_shape", "output shape or dtype"),
        ("out_dtype", "output shape or dtype"),
        ("out_device", "wrong device"),
        ("lse_shape", "FP32 LSE"),
        ("lse_dtype", "FP32 LSE"),
        ("lse_device", "wrong device"),
    ],
)
def test_processor_rejects_malformed_aten_contract(monkeypatch, case, match):
    from torch._subclasses.fake_tensor import FakeTensorMode

    mode = FakeTensorMode()
    with mode:
        q = torch.zeros(1, 1, 1, 128, device="cuda:0", dtype=torch.bfloat16)

        def bad_op(qh, _kh, _vh, **_kwargs):
            out = torch.zeros_like(qh)
            lse = torch.zeros(1, 1, 1, device=qh.device, dtype=torch.float32)
            if case == "non_tuple":
                return object()
            if case == "short_tuple":
                return (out,)
            if case == "non_tensor_out":
                return object(), lse
            if case == "non_tensor_lse":
                return out, object()
            if case == "out_shape":
                return out[..., :64], lse
            if case == "out_dtype":
                return out.to(torch.float32), lse
            if case == "out_device":
                return torch.zeros_like(qh, device="cuda:1"), lse
            if case == "lse_shape":
                return out, torch.zeros(
                    1, 1, 2, device=qh.device, dtype=torch.float32
                )
            if case == "lse_dtype":
                return out, lse.to(torch.bfloat16)
            if case == "lse_device":
                return out, torch.zeros(
                    1, 1, 1, device="cuda:1", dtype=torch.float32
                )
            raise AssertionError(f"unknown case {case}")

        monkeypatch.setattr(
            ring_full_context, "_aten_flash_attention_op", lambda: bad_op
        )
        with pytest.raises(base.UnsupportedModelError, match=match):
            wan_ring._TorchFlashProcessor()(q, q, q, softmax_scale=None)


def test_processor_preserves_raw_cuda_oom(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    mode = FakeTensorMode()
    with mode:
        q = torch.zeros(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16)

        def oom(*_args, **_kwargs):
            raise torch.OutOfMemoryError("raw treatment OOM")

        monkeypatch.setattr(ring_full_context, "_aten_flash_attention_op", lambda: oom)
        with pytest.raises(torch.OutOfMemoryError, match="raw treatment OOM"):
            wan_ring._TorchFlashProcessor()(q, q, q, softmax_scale=None)


def test_worker_status_publishes_bounded_attention_attestation(monkeypatch):
    descriptor = {
        "selected_path": "wan_ring_aten_full_context",
        "specialized_call_completed": False,
    }
    monkeypatch.setattr(
        "dgx_monarch.telemetry.event_snapshot", lambda _count: (0, [])
    )
    monkeypatch.setattr(worker_status, "source_manifest_sha256", lambda: "a" * 64)
    worker = SimpleNamespace(
        rank=0,
        world=2,
        topology=dict(PURE_RING2),
        store=SimpleNamespace(snapshot=lambda: {}),
        _setup_key=None,
        _attn=SimpleNamespace(attestation=lambda: dict(descriptor)),
    )

    result = worker_status.status_impl(worker)

    assert result["attention_attestation"] == descriptor


@pytest.mark.parametrize("with_dispatch", [False, True])
def test_worker_status_omits_absent_attention_attestation(monkeypatch, with_dispatch):
    monkeypatch.setattr(
        "dgx_monarch.telemetry.event_snapshot", lambda _count: (0, [])
    )
    monkeypatch.setattr(worker_status, "source_manifest_sha256", lambda: "a" * 64)
    worker = SimpleNamespace(
        rank=None,
        world=None,
        topology={},
        store=SimpleNamespace(snapshot=lambda: {}),
        _setup_key=None,
    )
    if with_dispatch:
        worker._attn = SimpleNamespace(attestation=lambda: None)

    result = worker_status.status_impl(worker)

    assert "attention_attestation" not in result


def test_dispatch_generation_replacement_invalidates_old_wan_view(monkeypatch):
    stock = []
    special = []

    def make_stock(kernel, sync):
        def implementation(*args):
            return "stock", kernel, sync, args

        stock.append(implementation)
        return implementation

    def make_special(kernel, sync, topology, world, generation):
        def implementation(*args):
            return "special", generation, args

        special.append((kernel, sync, dict(topology), world, generation, implementation))
        return implementation

    monkeypatch.setattr("dgx_monarch.adapters.make_usp_attention", make_stock)
    monkeypatch.setattr("dgx_monarch.adapters.make_wan_ring_usp_attention", make_special)
    dispatch = worker_env._AttentionDispatch()
    dispatch.configure("TORCH_FLASH", True, topology=PURE_RING2, world=2, setup_generation=1)
    first = dispatch.for_wan()
    assert first("q")[:2] == ("special", 1)

    dispatch.configure("TORCH_FLASH", True, topology=PURE_RING2, world=2, setup_generation=2)
    with pytest.raises(RuntimeError, match="stale setup generation"):
        first("q")
    second = dispatch.for_wan()
    assert second("q")[:2] == ("special", 2)
    assert len(stock) == len(special) == 2


def test_dispatch_sage_to_torch_rebuilds_special_eagerly(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "dgx_monarch.adapters.make_usp_attention", lambda kernel, _sync: lambda: kernel
    )

    def make_special(_kernel, _sync, _topology, _world, generation):
        calls.append(generation)
        return lambda: "special"

    monkeypatch.setattr("dgx_monarch.adapters.make_wan_ring_usp_attention", make_special)
    dispatch = worker_env._AttentionDispatch()
    dispatch.configure("TORCH_FLASH", True, topology=PURE_RING2, world=2, setup_generation=4)
    view = dispatch.for_wan()
    assert calls == [4]
    dispatch.configure("SAGE_FP8", True, topology=PURE_RING2, world=2, setup_generation=4)
    assert view() == "SAGE_FP8"
    dispatch.configure("TORCH_FLASH", True, topology=PURE_RING2, world=2, setup_generation=4)
    assert calls == [4, 4]  # rebuilt by configure(), before view() or any collective
    assert view() == "special"


def test_dispatch_attestation_tracks_real_binding_lifecycle(monkeypatch):
    _install_runtime(monkeypatch)
    monkeypatch.setattr(
        "dgx_monarch.adapters.make_usp_attention",
        lambda kernel, _sync: (lambda *_args, **_kwargs: kernel),
    )
    monkeypatch.setattr(
        "dgx_monarch.adapters.make_wan_ring_usp_attention",
        wan_ring.make_wan_ring_usp_attention,
    )
    dispatch = worker_env._AttentionDispatch()
    dispatch.configure(
        "SAGE_FP8", True, topology=PURE_RING2, world=2, setup_generation=1
    )
    old_view = dispatch.for_wan()
    assert dispatch.attestation() is None

    dispatch.configure(
        "TORCH_FLASH", True, topology=PURE_RING2, world=2, setup_generation=1
    )
    assert dispatch.attestation()["specialized_call_completed"] is False
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        old_view(q, q, q, heads=1)
    assert dispatch.attestation()["specialized_call_completed"] is True

    dispatch.configure(
        "SAGE_FP8", True, topology=PURE_RING2, world=2, setup_generation=1
    )
    assert dispatch.attestation() is None
    dispatch.configure(
        "TORCH_FLASH", True, topology=PURE_RING2, world=2, setup_generation=1
    )
    assert dispatch.attestation()["specialized_call_completed"] is False

    dispatch.configure(
        "TORCH_FLASH", True, topology=PURE_RING2, world=2, setup_generation=2
    )
    assert dispatch.attestation() is None
    with pytest.raises(RuntimeError, match="stale setup generation"):
        old_view()
    fresh_view = dispatch.for_wan()
    assert dispatch.attestation()["specialized_call_completed"] is False
    dispatch.invalidate()
    assert dispatch.attestation() is None
    with pytest.raises(RuntimeError, match="stale setup generation"):
        fresh_view()


def test_special_constructor_failure_never_publishes_attestation(monkeypatch):
    monkeypatch.setattr(
        "dgx_monarch.adapters.make_usp_attention",
        lambda kernel, _sync: (lambda: kernel),
    )
    dispatch = worker_env._AttentionDispatch()
    dispatch.configure(
        "SAGE_FP8", True, topology=PURE_RING2, world=2, setup_generation=1
    )
    view = dispatch.for_wan()
    monkeypatch.setattr(
        "dgx_monarch.adapters.make_wan_ring_usp_attention",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("treatment unavailable")),
    )
    with pytest.raises(RuntimeError, match="treatment unavailable"):
        dispatch.configure(
            "TORCH_FLASH", True, topology=PURE_RING2, world=2, setup_generation=1
        )
    assert dispatch.attestation() is None
    assert view() == "SAGE_FP8"


def test_dispatch_new_generation_constructor_failure_leaves_no_stale_callable(monkeypatch):
    fail = {"value": False}

    def make_stock(_kernel, _sync):
        if fail["value"]:
            raise RuntimeError("constructor failed")
        return lambda: "stock"

    monkeypatch.setattr("dgx_monarch.adapters.make_usp_attention", make_stock)
    monkeypatch.setattr(
        "dgx_monarch.adapters.make_wan_ring_usp_attention",
        lambda *_args: (lambda: "special"),
    )
    dispatch = worker_env._AttentionDispatch()
    dispatch.configure("TORCH_FLASH", True, topology=PURE_RING2, world=2, setup_generation=1)
    old = dispatch.for_wan()
    fail["value"] = True
    with pytest.raises(RuntimeError, match="constructor failed"):
        dispatch.configure(
            "TORCH_FLASH", True, topology=PURE_RING2, world=2, setup_generation=2
        )
    with pytest.raises(RuntimeError, match="stale setup generation"):
        old()
    with pytest.raises(RuntimeError, match="before setup"):
        dispatch()


def test_ring1_wan_uses_exact_stock_dispatch(monkeypatch):
    monkeypatch.setattr(
        "dgx_monarch.adapters.make_usp_attention", lambda *_args: (lambda: "stock")
    )
    dispatch = worker_env._AttentionDispatch()
    topology = {**PURE_RING2, "ulysses": 2, "ring": 1}
    dispatch.configure("TORCH_FLASH", True, topology=topology, world=2, setup_generation=1)
    assert dispatch.for_wan() is dispatch
    assert dispatch() == "stock"


@pytest.mark.parametrize(
    "topology,world",
    [
        ({**PURE_RING2, "ring": 4}, 4),
        ({**PURE_RING2, "ulysses": 2}, 4),
        ({**PURE_RING2, "dp": 2}, 4),
        ({**PURE_RING2, "cfg": 2}, 4),
        ({**PURE_RING2, "fsdp": True}, 2),
    ],
)
def test_non_pure_ring2_wan_torch_flash_stays_stock(
    monkeypatch, topology, world
):
    special_calls = []
    monkeypatch.setattr(
        "dgx_monarch.adapters.make_usp_attention",
        lambda kernel, _sync: (lambda: kernel),
    )
    monkeypatch.setattr(
        "dgx_monarch.adapters.make_wan_ring_usp_attention",
        lambda *_args: special_calls.append(_args),
    )
    dispatch = worker_env._AttentionDispatch()
    dispatch.configure(
        "TORCH_FLASH", True, topology=topology, world=world, setup_generation=1
    )

    assert dispatch.for_wan() is dispatch
    assert dispatch() == "TORCH_FLASH"
    assert special_calls == []


@pytest.mark.parametrize("ring,world", [(2, 2), (4, 4)])
def test_sage_ring_topologies_never_construct_specialization(monkeypatch, ring, world):
    special_calls = []
    monkeypatch.setattr(
        "dgx_monarch.adapters.make_usp_attention",
        lambda kernel, _sync: (lambda: kernel),
    )
    monkeypatch.setattr(
        "dgx_monarch.adapters.make_wan_ring_usp_attention",
        lambda *_args: special_calls.append(_args),
    )
    dispatch = worker_env._AttentionDispatch()
    topology = {**PURE_RING2, "ring": ring}
    dispatch.configure("SAGE_FP8", True, topology=topology, world=world, setup_generation=1)
    view = dispatch.for_wan()
    assert view() == "SAGE_FP8"
    assert special_calls == []


def test_sample_configures_sage_before_fresh_model_injection(monkeypatch):
    from dgx_monarch.actor import worker as worker_mod

    class StopAfterOrdering(RuntimeError):
        pass

    order = []

    class Dispatch:
        def configure(self, *_args, **_kwargs):
            order.append("configure")

    worker = SimpleNamespace(
        _setup_key=("ready",),
        _attn=Dispatch(),
        topology=dict(PURE_RING2),
        world=2,
        _setup_generation=3,
        _inject_for_topology=lambda *_args: None,
    )
    monkeypatch.setattr(
        worker_env,
        "verify_sample_artifact_authorization",
        lambda *_args: [{"digest": "model"}],
    )

    def ensure(*_args, **_kwargs):
        order.append("ensure")
        raise StopAfterOrdering

    monkeypatch.setattr(worker_mod.store_fsdp, "ensure", ensure)
    with pytest.raises(StopAfterOrdering):
        worker_mod.GPUWorker._sample_impl.__wrapped__(worker, {
            "kind": "ksampler",
            "model": {"unet_name": "model", "loras": []},
            "sage_kernel": "SAGE_FP8",
        })
    assert order == ["configure", "ensure"]


def test_worker_selects_specialization_by_exact_adapter_class(monkeypatch):
    from dgx_monarch.actor import worker as worker_mod
    from dgx_monarch.adapters.wan_family import WanAdapter

    captured = []
    exact = WanAdapter()
    exact.inject_usp = lambda _model, ctx: captured.append(ctx.usp_attention)
    subclass = type("FutureWanAdapter", (WanAdapter,), {})()
    subclass.inject_usp = lambda _model, ctx: captured.append(ctx.usp_attention)

    monkeypatch.setattr(worker_mod.store_fsdp, "validate_injection", lambda *_args: None)
    monkeypatch.setattr(worker_mod, "_maybe_compile_dit", lambda _model: None)
    adapter_slot = {"value": exact}
    monkeypatch.setattr("dgx_monarch.adapters.get_adapter", lambda _model: adapter_slot["value"])
    dispatch = SimpleNamespace(for_wan=lambda: "special", kernel="TORCH_FLASH",
                               bind_capability=lambda *_a: None,
                               effective_kernel="TORCH_FLASH")
    worker = SimpleNamespace(topology=dict(PURE_RING2), _attn=dispatch)
    patcher = SimpleNamespace(
        model=SimpleNamespace(diffusion_model=object()), model_options={}
    )

    worker_mod.GPUWorker._inject_for_topology(worker, patcher, "bf16", [], None)
    adapter_slot["value"] = subclass
    worker_mod.GPUWorker._inject_for_topology(worker, patcher, "bf16", [], None)
    adapter_slot["value"] = SimpleNamespace(
        inject_usp=lambda _model, ctx: captured.append(ctx.usp_attention)
    )
    worker_mod.GPUWorker._inject_for_topology(worker, patcher, "bf16", [], None)
    assert captured == ["special", dispatch, dispatch]


def test_worker_does_not_mask_malformed_adapter_attention_hook(monkeypatch):
    from dgx_monarch.actor import worker as worker_mod

    adapter = SimpleNamespace(attention_dispatch=None, inject_usp=lambda *_args: None)
    monkeypatch.setattr(worker_mod.store_fsdp, "validate_injection", lambda *_args: None)
    monkeypatch.setattr("dgx_monarch.adapters.get_adapter", lambda _model: adapter)
    worker = SimpleNamespace(topology=dict(PURE_RING2), _attn=SimpleNamespace(bind_capability=lambda *_a: None, effective_kernel="TORCH_FLASH"))
    patcher = SimpleNamespace(
        model=SimpleNamespace(diffusion_model=object()), model_options={}
    )

    with pytest.raises(TypeError, match="not callable"):
        worker_mod.GPUWorker._inject_for_topology(worker, patcher, "bf16", [], None)
