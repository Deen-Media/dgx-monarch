"""Native cuDNN Ring block processor with real log-sum-exp metadata.

It is not a dispatcher and has no fallback backend: bind it through xFuser's
instance-local custom processor route after a Ring cuDNN topology resolves.
"""
from __future__ import annotations

import math

import torch

from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError, _make_usp_attention_callable

_DTYPES = (torch.float16, torch.bfloat16)


def _aten_cudnn_attention_op():
    try:
        return torch.ops.aten._scaled_dot_product_cudnn_attention.default
    except (AttributeError, RuntimeError) as exc:
        raise UnsupportedModelError(
            refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN requires aten._scaled_dot_product_cudnn_attention." ' Use a compatible Torch build or select TORCH_FLASH for Ring.')
        ) from exc


def _validate_surface(*, dropout_p, softmax_scale, causal, window_size, softcap,
                      alibi_slopes, return_softmax, deterministic, kwargs) -> None:
    if kwargs:
        raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN received unsupported processor arguments." ' Use the supported maskless inference arguments or run this workflow on a single topology.'))
    if dropout_p != 0.0 or causal is not False:
        raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN supports non-causal inference attention only." ' Use non-causal inference or run this workflow on a single topology.'))
    if window_size != (-1, -1) or alibi_slopes is not None or softcap != 0.0:
        raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN does not support window, ALiBi, or softcap." ' Use unwindowed attention without ALiBi or softcap, or run this workflow on a single topology.'))
    if return_softmax is not False:
        raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN does not return attention probabilities." ' Use the attention output without requesting probability tensors.'))
    if deterministic is not False:
        raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN cannot honor deterministic attention." ' Use the supported inference mode or run this workflow on a single topology with a deterministic kernel.'))
    if softmax_scale is not None and (isinstance(softmax_scale, bool)
                                      or not isinstance(softmax_scale, (int, float))
                                      or not math.isfinite(float(softmax_scale))):
        raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN requires a finite scale." ' Use a finite numeric softmax scale or the default scale.'))


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if any(not isinstance(value, torch.Tensor) for value in (q, k, v)):
        raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN requires tensor Q/K/V." ' Use tensor query, key and value inputs.'))
    if any(value.ndim != 4 or value.layout != torch.strided or value.numel() == 0
           or value.requires_grad for value in (q, k, v)):
        raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN requires nonempty strided inference Q/K/V." ' Use nonempty strided tensors without gradient tracking.'))
    if (q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]
            or q.shape[2:] != k.shape[2:] or q.shape[2:] != v.shape[2:]
            or k.shape[1] != v.shape[1]):
        raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN requires matching B/H/D and K/V length." ' Use matching batch, head and feature dimensions, with equal key/value sequence lengths.'))
    if q.dtype not in _DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN requires matching fp16 or bf16 Q/K/V." ' Use matching FP16 or BF16 query, key and value tensors.'))
    if q.device != k.device or q.device != v.device or q.device.type != "cuda":
        raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN requires Q/K/V on one CUDA device." ' Use query, key and value tensors on the same local CUDA device.'))


class TorchCudnnRingProcessor:
    """Return native cuDNN block output and FP32 `(B,H,L)` log-sum-exp."""

    __slots__ = ()

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, dropout_p=0.0,
                 softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0,
                 alibi_slopes=None, return_softmax=False, deterministic=False,
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_surface(dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
                          window_size=window_size, softcap=softcap, alibi_slopes=alibi_slopes,
                          return_softmax=return_softmax, deterministic=deterministic, kwargs=kwargs)
        _validate_qkv(q, k, v)
        qh, kh, vh = (value.transpose(1, 2) for value in (q, k, v))
        result = _aten_cudnn_attention_op()(qh, kh, vh, None, True, 0.0, False, False,
                                            scale=softmax_scale)
        if not isinstance(result, tuple) or len(result) < 2:
            raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN returned an unsupported ATen result." ' Use a compatible Torch build or select TORCH_FLASH for Ring.'))
        out, lse = result[:2]
        # torch 2.12 returns the LSE with a trailing unit dim, (B,H,L,1); the
        # Ring merge reads (B,H,L).
        if isinstance(lse, torch.Tensor) and lse.ndim == 4 and lse.shape[-1] == 1:
            lse = lse.squeeze(-1)
        if (not isinstance(out, torch.Tensor) or not isinstance(lse, torch.Tensor)
                or out.shape != qh.shape or out.dtype != q.dtype or out.device != q.device
                or lse.shape != (q.shape[0], q.shape[2], q.shape[1])
                or lse.dtype != torch.float32 or lse.device != q.device):
            raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN requires native FP32 LSE shaped (B,H,L)." ' Use a Torch build providing native FP32 normalization or select TORCH_FLASH for Ring.'))
        return out.transpose(1, 2), lse


class _CudnnRingSentinel:
    pass


def _validate_comfy_surface(q, k, v, heads, mask, attn_precision, skip_reshape,
                            skip_output_reshape, enable_gqa, drop_rows,
                            kv_drop_rows, query_bias_groups, kwargs) -> None:
    # The shared Comfy wrapper forwards its named reshape/GQA/padding contract,
    # but does not pass extra options to xFuser. Refuse those before Ring P2P
    # rather than silently drop a window, scale or deterministic request.
    if kwargs:
        raise UnsupportedModelError(refusal(RefusalClass.PHYSICS, "Ring TORCH_CUDNN cannot apply extra Comfy attention arguments." ' Use the supported named Comfy attention options or run this workflow on a single topology.'))


def _probe_native(processor, *, device: torch.device | None = None) -> None:
    device = device or torch.device("cuda", torch.cuda.current_device())
    for dtype in _DTYPES:
        # Dyadic inputs keep the FP16/BF16 logits exact and bounded. Distinct
        # heads, queries and value coordinates expose layout mistakes without
        # needing a loose tolerance for large matrix products.
        q = torch.zeros((1, 32, 2, 64), device=device, dtype=dtype)
        positions = torch.arange(32, device=device, dtype=torch.float32) / 16 - 1
        q[0, :, 0, 0] = positions
        q[0, :, 1, 0] = -positions
        k = torch.zeros((1, 64, 2, 64), device=device, dtype=dtype)
        k[:, 32:, 0, 0] = 1
        k[:, :32, 1, 0] = 2
        k[:, 32:, 1, 0] = -1
        v = torch.zeros_like(k)
        v[:, :32, :, 0], v[:, 32:, :, 0] = 0.25, 0.75
        v[:, :32, :, 1], v[:, 32:, :, 1] = -0.5, 1.0
        scale = 1.0 / math.sqrt(q.shape[-1])
        out, lse = processor(q, k, v, softmax_scale=scale)
        qh, kh, vh = (value.transpose(1, 2).float() for value in (q, k, v))
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
        expected_lse = torch.logsumexp(scores, dim=-1)
        expected_out = torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(1, 2).to(dtype)
        if not torch.allclose(lse, expected_lse, atol=2e-5, rtol=2e-5):
            raise RuntimeError("Ring TORCH_CUDNN probe returned incorrect native LSE")
        # The output is rounded to the input dtype; LSE above remains FP32.
        epsilon = torch.finfo(dtype).eps
        if (out.shape != q.shape or out.dtype != dtype
                or not torch.allclose(out, expected_out, atol=epsilon / 8, rtol=2 * epsilon)):
            raise RuntimeError("Ring TORCH_CUDNN probe returned incorrect output contract")


def make_cudnn_ring_usp_attention(sync_ulysses: bool = True):
    """Build one custom-processor instance before the guarded readiness exchange."""
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
    from yunchang.kernels import AttnType, select_flash_attn_impl

    sentinel, processor = _CudnnRingSentinel(), TorchCudnnRingProcessor()
    if isinstance(sentinel, AttnType) or select_flash_attn_impl(
        sentinel, stage="fwd-only", attn_processor=processor) is not processor:
        raise RuntimeError("yunchang did not retain the cuDNN Ring processor")
    usp_attn = xFuserLongContextAttention(use_sync=sync_ulysses, attn_type=sentinel,
                                          attn_processor=processor)
    if (getattr(usp_attn, "attn_type", None) is not sentinel
            or getattr(usp_attn, "attn_processor", None) is not processor
            or not callable(getattr(usp_attn, "ring_attn_fn", None))):
        raise RuntimeError("xFuser did not retain the cuDNN Ring binding")
    _probe_native(processor)
    return _make_usp_attention_callable(
        usp_attn, surface_validator=_validate_comfy_surface, qkv_validator=_validate_qkv,
    )
