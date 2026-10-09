"""Exact full-context transport and native flash kernel for Wan Ring2."""
from __future__ import annotations

import math

import torch

from .base import UnsupportedModelError

_FLASH_QKV_DTYPES = (torch.float16, torch.bfloat16)


def _aten_flash_attention_op():
    return torch.ops.aten._scaled_dot_product_flash_attention.default


def _validate_flash_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Validate the native flash contract, including local-Q/full-KV shapes."""

    if any(not isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise UnsupportedModelError("Wan ring TORCH_FLASH requires tensor q/k/v inputs")
    if any(tensor.requires_grad for tensor in (q, k, v)):
        raise UnsupportedModelError("Wan ring TORCH_FLASH is inference-only")
    if any(tensor.ndim != 4 for tensor in (q, k, v)):
        raise UnsupportedModelError("Wan ring TORCH_FLASH requires 4D q/k/v")
    if (
        k.shape != v.shape
        or q.shape[0] != k.shape[0]
        or q.shape[2:] != k.shape[2:]
    ):
        raise UnsupportedModelError(
            "Wan ring TORCH_FLASH requires equal K/V shapes and matching "
            "q/k/v batch, head, and head-dimension shapes"
        )
    if any(size < 1 for tensor in (q, k, v) for size in tensor.shape):
        raise UnsupportedModelError("Wan ring TORCH_FLASH requires non-empty q/k/v")
    if q.dtype not in _FLASH_QKV_DTYPES or any(
        tensor.dtype != q.dtype for tensor in (k, v)
    ):
        raise UnsupportedModelError(
            "Wan ring TORCH_FLASH is validated for matching FP16/BF16 q/k/v only"
        )
    if q.shape[-1] != 128:
        raise UnsupportedModelError(
            "Wan ring TORCH_FLASH is validated for head dimension 128 only"
        )
    if q.device.type != "cuda" or any(tensor.device != q.device for tensor in (k, v)):
        raise UnsupportedModelError("Wan ring TORCH_FLASH requires q/k/v on one CUDA device")
    if any(tensor.layout != torch.strided for tensor in (q, k, v)):
        raise UnsupportedModelError("Wan ring TORCH_FLASH requires strided tensors")


def _validate_local_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Validate every predictable local-shard precondition before ring P2P."""

    _validate_flash_qkv(q, k, v)
    if q.shape != k.shape:
        raise UnsupportedModelError(
            "Wan ring TORCH_FLASH requires equal local q/k/v shapes"
        )


def _validate_flash_surface(
    *,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    return_softmax,
) -> None:
    """Reject every unsupported kernel option before any communication."""

    if (
        isinstance(dropout_p, bool)
        or not isinstance(dropout_p, (int, float))
        or not math.isfinite(float(dropout_p))
        or float(dropout_p) != 0.0
    ):
        raise UnsupportedModelError("Wan ring TORCH_FLASH supports dropout_p=0 only")
    if softmax_scale is not None and (
        isinstance(softmax_scale, bool)
        or not isinstance(softmax_scale, (int, float))
        or not math.isfinite(float(softmax_scale))
        or float(softmax_scale) <= 0.0
    ):
        raise UnsupportedModelError(
            "Wan ring TORCH_FLASH requires a finite positive softmax scale"
        )
    if causal is not False:
        raise UnsupportedModelError("Wan ring TORCH_FLASH supports non-causal attention only")
    if (
        type(window_size) is not tuple
        or len(window_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in window_size
        )
        or window_size != (-1, -1)
    ):
        raise UnsupportedModelError(
            "Wan ring TORCH_FLASH does not support local attention windows"
        )
    if (
        isinstance(softcap, bool)
        or not isinstance(softcap, (int, float))
        or not math.isfinite(float(softcap))
        or float(softcap) != 0.0
    ):
        raise UnsupportedModelError("Wan ring TORCH_FLASH does not support softcap")
    if alibi_slopes is not None:
        raise UnsupportedModelError("Wan ring TORCH_FLASH does not support ALiBi")
    if return_softmax is not False:
        raise UnsupportedModelError(
            "Wan ring TORCH_FLASH does not support returning attention probabilities"
        )


class _TorchFlashProcessor:
    """Stateless inference-only native flash processor retaining FP32 LSE."""

    __slots__ = ()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kwargs:
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH received unsupported processor arguments"
            )
        _validate_flash_surface(
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
        _validate_flash_qkv(q, k, v)
        expected_scale = 1.0 / math.sqrt(q.shape[-1])
        if softmax_scale is not None and float(softmax_scale) != expected_scale:
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH requires the default head-dimension scale"
            )
        qh, kh, vh = (tensor.transpose(1, 2) for tensor in (q, k, v))
        result = _aten_flash_attention_op()(
            qh,
            kh,
            vh,
            dropout_p=0.0,
            is_causal=False,
            scale=None if softmax_scale is None else expected_scale,
        )
        if not isinstance(result, tuple) or len(result) < 2:
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH returned an unsupported ATen result"
            )
        out, lse = result[:2]
        if not isinstance(out, torch.Tensor) or not isinstance(lse, torch.Tensor):
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH returned non-tensor output metadata"
            )
        if out.shape != qh.shape or out.dtype != q.dtype:
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH returned an invalid output shape or dtype"
            )
        if lse.shape != (q.shape[0], q.shape[2], q.shape[1]) or lse.dtype != torch.float32:
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH requires native FP32 LSE with shape (B,H,L)"
            )
        if out.device != q.device or lse.device != q.device:
            raise UnsupportedModelError(
                "Wan ring TORCH_FLASH returned tensors on the wrong device"
            )
        return out.transpose(1, 2), lse


def _probe_aten(processor: _TorchFlashProcessor) -> str:
    device = torch.device("cuda", torch.cuda.current_device())
    for dtype in _FLASH_QKV_DTYPES:
        probe = torch.zeros((1, 1, 1, 128), dtype=dtype, device=device)
        with torch.no_grad():
            out, lse = processor(probe, probe, probe, softmax_scale=None)
        if out.shape != probe.shape or out.dtype != dtype or lse.dtype != torch.float32:
            raise RuntimeError("Wan ring ATen flash behavior probe returned an invalid contract")
    return "float32"


def _materialize_full_kv_batch(
    k_batch: torch.Tensor,
    v_batch: torch.Tensor,
    *,
    group,
    world: int,
    rank: int,
    global_ranks: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """P2P-gather one local K/V batch directly into sequence-rank slots."""

    if world != 2 or isinstance(world, bool):
        raise UnsupportedModelError("Wan ring full-context gather requires ring world 2")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in (0, 1):
        raise UnsupportedModelError("Wan ring full-context gather requires ring rank 0 or 1")
    if (
        type(global_ranks) is not tuple
        or len(global_ranks) != world
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in global_ranks
        )
        or sorted(global_ranks) != list(range(world))
    ):
        raise UnsupportedModelError(
            "Wan ring full-context gather requires a complete global-rank mapping"
        )
    if (
        not isinstance(k_batch, torch.Tensor)
        or not isinstance(v_batch, torch.Tensor)
        or k_batch.ndim != 4
        or k_batch.shape != v_batch.shape
        or k_batch.shape[0] != 1
        or any(size < 1 for size in k_batch.shape)
    ):
        raise UnsupportedModelError(
            "Wan ring full-context gather requires one non-empty equal-shape K/V batch"
        )

    import torch.distributed as dist

    local_length = k_batch.shape[1]
    full_shape = (1, local_length * world, k_batch.shape[2], k_batch.shape[3])
    full_k = k_batch.new_empty(full_shape)
    full_v = v_batch.new_empty(full_shape)
    local_start = rank * local_length
    peer_rank = 1 - rank
    peer_start = peer_rank * local_length
    local_k = full_k.narrow(1, local_start, local_length)
    local_v = full_v.narrow(1, local_start, local_length)
    peer_k = full_k.narrow(1, peer_start, local_length)
    peer_v = full_v.narrow(1, peer_start, local_length)
    local_k.copy_(k_batch)
    local_v.copy_(v_batch)

    global_peer_rank = global_ranks[peer_rank]
    operations = [
        dist.P2POp(dist.isend, local_k, global_peer_rank, group=group),
        dist.P2POp(dist.irecv, peer_k, global_peer_rank, group=group),
        dist.P2POp(dist.isend, local_v, global_peer_rank, group=group),
        dist.P2POp(dist.irecv, peer_v, global_peer_rank, group=group),
    ]
    requests = dist.batch_isend_irecv(operations)
    if not isinstance(requests, (list, tuple)):
        raise UnsupportedModelError(
            "Wan ring full-context P2P returned an invalid request contract"
        )
    valid_contract = (
        isinstance(requests, list)
        and len(requests) in (1, len(operations))
        and all(callable(getattr(request, "wait", None)) for request in requests)
    )
    first_wait_error: BaseException | None = None
    for request in requests:
        wait = getattr(request, "wait", None)
        if not callable(wait):
            continue
        try:
            completed = wait()
            if completed is not True and first_wait_error is None:
                first_wait_error = UnsupportedModelError(
                    "Wan ring full-context P2P wait did not complete"
                )
        except BaseException as exc:
            if first_wait_error is None:
                first_wait_error = exc
    if first_wait_error is not None:
        raise first_wait_error
    if not valid_contract:
        raise UnsupportedModelError(
            "Wan ring full-context P2P returned an invalid request contract"
        )
    return full_k, full_v


def _make_full_context_ring_attention(
    *,
    ring_pg,
    expected_rank: int,
    expected_sequence_rank: int,
    expected_global_rank: int,
    expected_global_ranks: tuple[int, int],
    expected_attn_type,
    expected_processor,
):
    """Build the exact pure-Ring2 local-Q/full-KV native-flash call."""

    @torch.compiler.disable
    def full_context_ring_attention(
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
        group=None,
        attn_type=None,
        attn_processor=None,
        attn_layer=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        joint_strategy="none",
        q_descale=None,
        k_descale=None,
        v_descale=None,
    ):
        _validate_flash_surface(
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=0.0,
            alibi_slopes=alibi_slopes,
            return_softmax=return_attn_probs,
        )
        _validate_local_qkv(q, k, v)
        if deterministic is not False:
            raise UnsupportedModelError(
                "Wan ring full-context attention requires deterministic=False"
            )
        if (
            attn_layer is not None
            or joint_tensor_key is not None
            or joint_tensor_value is not None
            or joint_strategy != "none"
            or q_descale is not None
            or k_descale is not None
            or v_descale is not None
        ):
            raise UnsupportedModelError(
                "Wan ring full-context attention received an unsupported xFuser surface"
            )
        if group is not ring_pg:
            raise UnsupportedModelError(
                "Wan ring full-context attention received the wrong process group"
            )
        if attn_type is not expected_attn_type or attn_processor is not expected_processor:
            raise UnsupportedModelError(
                "Wan ring full-context attention binding changed after construction"
            )

        import torch.distributed as dist
        from xfuser.core.distributed import get_sequence_parallel_rank

        if not dist.is_initialized():
            raise UnsupportedModelError(
                "Wan ring full-context distributed authority is unavailable"
            )
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        sequence_rank = get_sequence_parallel_rank()
        global_world = dist.get_world_size()
        global_rank = dist.get_rank()
        global_ranks = tuple(dist.get_global_rank(group, slot) for slot in range(2))
        if isinstance(world, bool) or not isinstance(world, int) or world != 2:
            raise UnsupportedModelError(
                "Wan ring full-context process-group world changed after construction"
            )
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank != expected_rank
            or rank not in (0, 1)
        ):
            raise UnsupportedModelError(
                "Wan ring full-context process-group rank changed after construction"
            )
        if (
            isinstance(sequence_rank, bool)
            or not isinstance(sequence_rank, int)
            or sequence_rank != expected_sequence_rank
            or sequence_rank != rank
        ):
            raise UnsupportedModelError(
                "Wan ring full-context sequence rank changed after construction"
            )
        if (
            isinstance(global_world, bool)
            or not isinstance(global_world, int)
            or global_world != 2
            or isinstance(global_rank, bool)
            or not isinstance(global_rank, int)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in global_ranks
            )
            or sorted(global_ranks) != list(range(global_world))
            or global_ranks != expected_global_ranks
            or global_rank != expected_global_rank
            or global_rank != global_ranks[rank]
        ):
            raise UnsupportedModelError(
                "Wan ring full-context global-rank mapping changed after construction"
            )

        output = torch.empty_like(q)
        for batch_index in range(q.shape[0]):
            full_k, full_v = _materialize_full_kv_batch(
                k[batch_index : batch_index + 1],
                v[batch_index : batch_index + 1],
                group=group,
                world=world,
                rank=rank,
                global_ranks=global_ranks,
            )
            batch_out, _lse = expected_processor(
                q[batch_index : batch_index + 1],
                full_k,
                full_v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                return_softmax=False,
            )
            output[batch_index : batch_index + 1].copy_(batch_out)
            # Drop this batch's references before the next gather: otherwise the
            # old full K/V stay alive while the next assignment's right-hand side
            # runs, briefly doubling the temporary full-context allocation.
            del full_k, full_v, batch_out, _lse
        return output

    return full_context_ring_attention
