"""Plain-Wan pure-Ring2 TORCH_FLASH full-context specialization.

yunchang's TORCH_FLASH attention type narrows block LSE to the query dtype
before the ring merge, and even an FP32-LSE merge rounds every partial-attention
output before combining it. This specialization materializes K/V in
sequence-rank order one local batch at a time, then runs one native flash call
for each local-Q batch. It is bounded to the exact plain-Wan
FP16-or-BF16/head-dim-128 pure-Ring2 path, so every other topology and surface
remains stock or refuses before xFuser can post ring P2P.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import cast

import torch

from .base import (
    UnsupportedModelError,
    _make_usp_attention_callable,
    assert_ulysses_only_padding,
)
from .wan_ring_full_context import (
    _FLASH_QKV_DTYPES,
    _make_full_context_ring_attention,
    _probe_aten,
    _TorchFlashProcessor,
    _validate_local_qkv,
)


@dataclass(frozen=True)
class _Authority:
    setup_generation: int
    world: int
    dp: int
    cfg: int
    ulysses: int
    ring: int
    fsdp: bool

    @property
    def sp(self) -> int:
        return self.ulysses * self.ring


@dataclass(frozen=True)
class WanRingAttentionAttestation:
    """Bounded immutable evidence for one specialized attention instance."""

    setup_generation: int
    configured_world: int
    configured_dp: int
    configured_cfg: int
    configured_sp: int
    configured_ulysses: int
    configured_ring: int
    xfuser_sp: int
    xfuser_dp: int
    xfuser_cfg: int
    xfuser_ulysses: int
    xfuser_ring: int
    instance_ulysses_group_size: int
    instance_ring_group_size: int
    instance_ulysses_group_rank: int
    instance_ring_group_rank: int
    xfuser_sequence_rank: int
    process_global_rank: int
    ring_slot0_global_rank: int
    ring_slot1_global_rank: int
    selected_path: str = "wan_ring_aten_full_context"
    context_strategy: str = "rank_ordered_p2p_kv_per_local_batch"
    processor_calls_per_local_batch: int = 1
    expected_lse_dtype: str = "float32"
    validated_lse_dtype: str = "float32_per_call"

    def as_dict(self) -> dict[str, int | str]:
        return dict(vars(self))


@dataclass(frozen=True)
class _Binding:
    implementation: object
    attestation: WanRingAttentionAttestation
    _processor_receipt: _ProcessorReceipt
    _completed: Event = field(default_factory=Event, repr=False, compare=False)
    _call_lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def __call__(self, *args, **kwargs):
        with self._call_lock:
            before = self._processor_receipt.snapshot()
            result = self.implementation(*args, **kwargs)
            q = args[0] if args else kwargs.get("q")
            expected = q.shape[0] if isinstance(q, torch.Tensor) and q.ndim else 1
            observed = self._processor_receipt.snapshot() - before
            if observed != expected:
                raise UnsupportedModelError(
                    "Wan ring attention returned without executing exactly one "
                    "full-context processor call per local batch"
                )
            self._completed.set()
            return result

    def status_attestation(self) -> dict[str, int | str | bool]:
        return {
            **self.attestation.as_dict(),
            "specialized_call_completed": self._completed.is_set(),
        }


def _authority(
    topology: Mapping[str, object], world: int, setup_generation: int
) -> _Authority:
    if (isinstance(setup_generation, bool)
            or not isinstance(setup_generation, int)
            or setup_generation < 1):
        raise ValueError("Wan ring attention requires a positive setup generation")
    if not isinstance(topology, Mapping):
        raise TypeError("Wan ring attention topology must be a mapping")
    if isinstance(world, bool) or not isinstance(world, int) or world < 1:
        raise ValueError("Wan ring attention world must be a positive integer")
    degrees: dict[str, int] = {}
    for name in ("dp", "cfg", "ulysses", "ring"):
        value = topology.get(name, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"Wan ring attention topology degree {name!r} must be a positive integer"
            )
        degrees[name] = value
    fsdp = topology.get("fsdp", False)
    if type(fsdp) is not bool:
        raise ValueError("Wan ring attention topology field 'fsdp' must be a bool")
    if (degrees["dp"], degrees["cfg"], degrees["ulysses"], degrees["ring"], fsdp) != (
        1,
        1,
        1,
        2,
        False,
    ):
        raise ValueError("Wan ring attention specialization is validated for pure ring2 only")
    if world != degrees["dp"] * degrees["cfg"] * degrees["ulysses"] * degrees["ring"]:
        raise ValueError("Wan ring attention world does not match its topology product")
    return _Authority(
        setup_generation, world, degrees["dp"], degrees["cfg"],
        degrees["ulysses"], degrees["ring"], fsdp,
    )


class _Sentinel:
    """Opaque non-AttnType key selecting yunchang's processor fallback."""

    __slots__ = ()


class _ProcessorReceipt:
    """Thread-safe count of successful processor calls; _Binding checks one per local batch."""

    __slots__ = ("_lock", "_successes")

    def __init__(self) -> None:
        self._lock = Lock()
        self._successes = 0

    def mark_success(self) -> None:
        with self._lock:
            self._successes += 1

    def snapshot(self) -> int:
        with self._lock:
            return self._successes


class _ReceiptProcessor:
    """Record only processor calls that returned after validating FP32 LSE."""

    __slots__ = ("_implementation", "_receipt")

    def __init__(self, implementation, receipt: _ProcessorReceipt) -> None:
        self._implementation = implementation
        self._receipt = receipt

    def __call__(self, *args, **kwargs):
        result = self._implementation(*args, **kwargs)
        self._receipt.mark_success()
        return result


def _verify_groups(usp_attn, authority: _Authority):
    try:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_classifier_free_guidance_world_size,
            get_data_parallel_world_size,
            get_ring_parallel_world_size,
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
            get_ulysses_parallel_world_size,
        )

        if not dist.is_initialized():
            raise UnsupportedModelError("Wan ring attention distributed authority is unavailable")
        observed = (
            dist.get_world_size(), get_data_parallel_world_size(),
            get_classifier_free_guidance_world_size(), get_sequence_parallel_world_size(),
            get_ulysses_parallel_world_size(), get_ring_parallel_world_size(),
        )
        expected = (
            authority.world, authority.dp, authority.cfg, authority.sp,
            authority.ulysses, authority.ring,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in observed):
            raise UnsupportedModelError("Wan ring attention xFuser topology authority is invalid")
        if observed != expected:
            raise UnsupportedModelError(
                "Wan ring attention xFuser topology does not match its setup authority"
            )
        ulysses_pg = getattr(usp_attn, "ulysses_pg", None)
        ring_pg_value = getattr(usp_attn, "ring_pg", None)
        if ulysses_pg is None:
            raise UnsupportedModelError(
                "Wan ring attention instance process-group authority is unavailable"
            )
        if ring_pg_value is None:
            raise UnsupportedModelError(
                "Wan ring attention instance process-group authority is unavailable"
            )
        ring_pg = cast(dist.ProcessGroup, ring_pg_value)
        sp_group = get_sp_group()
        if (ulysses_pg is not getattr(sp_group, "ulysses_group", None)
                or ring_pg is not getattr(sp_group, "ring_group", None)):
            raise UnsupportedModelError(
                "Wan ring attention instance groups are not the current xFuser groups"
            )
        sizes = (dist.get_world_size(ulysses_pg), dist.get_world_size(ring_pg))
        ranks = (dist.get_rank(ulysses_pg), dist.get_rank(ring_pg))
        sp_ranks = (
            getattr(sp_group, "ulysses_rank", None),
            getattr(sp_group, "ring_rank", None),
        )
        sequence_rank = get_sequence_parallel_rank()
        global_rank = dist.get_rank()
        global_ranks = (
            dist.get_global_rank(ring_pg, 0),
            dist.get_global_rank(ring_pg, 1),
        )
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in sizes)
            or sizes != (authority.ulysses, authority.ring)
        ):
            raise UnsupportedModelError(
                "Wan ring attention instance groups do not match its setup authority"
            )
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in ranks)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in sp_ranks
            )
            or ranks != sp_ranks
            or isinstance(sequence_rank, bool)
            or not isinstance(sequence_rank, int)
            or sequence_rank != ranks[1]
            or not 0 <= ranks[0] < sizes[0]
            or not 0 <= ranks[1] < sizes[1]
        ):
            raise UnsupportedModelError(
                "Wan ring attention instance group ranks do not match xFuser authority"
            )
        if (
            isinstance(global_rank, bool)
            or not isinstance(global_rank, int)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in global_ranks
            )
            or sorted(global_ranks) != list(range(authority.world))
            or global_rank != global_ranks[ranks[1]]
        ):
            raise UnsupportedModelError(
                "Wan ring attention global-rank mapping does not match xFuser authority"
            )
        return observed, sizes, ranks, sequence_rank, global_rank, global_ranks
    except UnsupportedModelError:
        raise
    except Exception as exc:
        raise UnsupportedModelError("Wan ring attention authority could not be verified") from exc


@dataclass(frozen=True)
class _CallPreflight:
    ring: int

    def surface(
        self, q, k, v, heads, mask, attn_precision, skip_reshape,
        skip_output_reshape, enable_gqa, drop_rows, kv_drop_rows,
        query_bias_groups, kwargs,
    ) -> None:
        if drop_rows or kv_drop_rows:
            # Only the shared pad guard may admit synthetic rows under a class-K
            # waiver. Do not reject them again as an unsupported surface below,
            # which would bypass that waiver with an untagged error. Key-side pads
            # follow the same rule.
            assert_ulysses_only_padding(
                self.ring, len(drop_rows or ()) + len(kv_drop_rows or ()))
        if mask is not None:
            raise UnsupportedModelError("Wan ring TORCH_FLASH does not support a mask")
        if attn_precision is not None:
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH does not support a precision override"
            )
        if (skip_reshape or skip_output_reshape or enable_gqa
                or query_bias_groups or kwargs):
            # A query-group bias belongs here rather than beside the pad guard:
            # this treatment is the ring path, and no ring rank ever holds the
            # key axis such a bias spans (base.assert_ulysses_only_query_bias).
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH received an unsupported Comfy attention surface"
            )
        if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH requires a positive integer head count"
            )
        if any(not isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
            raise UnsupportedModelError("Wan ring TORCH_FLASH requires tensor q/k/v inputs")
        if any(tensor.requires_grad for tensor in (q, k, v)):
            raise UnsupportedModelError("Wan ring TORCH_FLASH is inference-only")
        if (
            any(tensor.ndim != 3 for tensor in (q, k, v))
            or q.shape != k.shape
            or q.shape != v.shape
        ):
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH requires equal 3D Comfy q/k/v shapes"
            )
        if any(size < 1 for size in q.shape):
            raise UnsupportedModelError("Wan ring TORCH_FLASH requires non-empty q/k/v")
        if q.dtype not in _FLASH_QKV_DTYPES or any(
            tensor.dtype != q.dtype for tensor in (k, v)
        ):
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH is validated for matching FP16/BF16 q/k/v only"
            )
        if q.shape[-1] != heads * 128:
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH is validated for head dimension 128 only"
            )
        if q.device.type != "cuda" or any(
            tensor.device != q.device for tensor in (k, v)
        ):
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH requires q/k/v on one CUDA device"
            )
        if any(tensor.layout != torch.strided for tensor in (q, k, v)):
            raise UnsupportedModelError("Wan ring TORCH_FLASH requires strided tensors")


def make_wan_ring_usp_attention(
    attn_type_name: str,
    sync_ulysses: bool,
    topology: Mapping[str, object],
    world: int,
    setup_generation: int,
):
    if attn_type_name != "TORCH_FLASH":
        raise ValueError("Wan ring full-context specialization supports TORCH_FLASH only")
    authority = _authority(topology, world, setup_generation)
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
    from yunchang.kernels import AttnType, select_flash_attn_impl

    sentinel = _Sentinel()
    processor_implementation = _TorchFlashProcessor()
    validated_lse = _probe_aten(processor_implementation)
    processor_receipt = _ProcessorReceipt()
    processor = _ReceiptProcessor(processor_implementation, processor_receipt)
    if isinstance(sentinel, AttnType) or select_flash_attn_impl(
        sentinel, stage="fwd-only", attn_processor=processor
    ) is not processor:
        raise RuntimeError("yunchang does not honor the Wan ring processor fallback")
    usp_attn = xFuserLongContextAttention(
        use_sync=sync_ulysses, attn_type=sentinel, attn_processor=processor,
    )
    if (getattr(usp_attn, "attn_type", None) is not sentinel
            or getattr(usp_attn, "attn_processor", None) is not processor):
        raise RuntimeError("xFuser did not retain the Wan ring attention binding")
    if not callable(getattr(usp_attn, "ring_attn_fn", None)):
        raise RuntimeError("xFuser did not expose a callable Wan ring attention path")
    (
        observed,
        sizes,
        ranks,
        sequence_rank,
        global_rank,
        global_ranks,
    ) = _verify_groups(usp_attn, authority)
    full_context_ring_attention = _make_full_context_ring_attention(
        ring_pg=usp_attn.ring_pg,
        expected_rank=ranks[1],
        expected_sequence_rank=sequence_rank,
        expected_global_rank=global_rank,
        expected_global_ranks=global_ranks,
        expected_attn_type=sentinel,
        expected_processor=processor,
    )
    usp_attn.ring_attn_fn = full_context_ring_attention
    if usp_attn.ring_attn_fn is not full_context_ring_attention:
        raise RuntimeError("xFuser did not retain the Wan full-context ring path")
    descriptor = WanRingAttentionAttestation(
        setup_generation=authority.setup_generation,
        configured_world=authority.world,
        configured_dp=authority.dp,
        configured_cfg=authority.cfg,
        configured_sp=authority.sp,
        configured_ulysses=authority.ulysses,
        configured_ring=authority.ring,
        xfuser_sp=observed[3],
        xfuser_dp=observed[1],
        xfuser_cfg=observed[2],
        xfuser_ulysses=observed[4],
        xfuser_ring=observed[5],
        instance_ulysses_group_size=sizes[0],
        instance_ring_group_size=sizes[1],
        instance_ulysses_group_rank=ranks[0],
        instance_ring_group_rank=ranks[1],
        xfuser_sequence_rank=sequence_rank,
        process_global_rank=global_rank,
        ring_slot0_global_rank=global_ranks[0],
        ring_slot1_global_rank=global_ranks[1],
        validated_lse_dtype=f"{validated_lse}_construction_and_per_call",
    )
    preflight = _CallPreflight(authority.ring)
    implementation = _make_usp_attention_callable(
        usp_attn, surface_validator=preflight.surface, qkv_validator=_validate_local_qkv,
    )

    @torch.compiler.disable
    def guarded_implementation(*args, **kwargs):
        if getattr(usp_attn, "ring_attn_fn", None) is not full_context_ring_attention:
            raise UnsupportedModelError(
                "Wan ring full-context path changed after construction"
            )
        result = implementation(*args, **kwargs)
        if getattr(usp_attn, "ring_attn_fn", None) is not full_context_ring_attention:
            raise UnsupportedModelError(
                "Wan ring full-context path changed during execution"
            )
        return result

    return _Binding(guarded_implementation, descriptor, processor_receipt)
