"""Validate Sol-Attn geometry and options before dispatch.

The package checks dtype, device, contiguity, shape agreement, and options,
but cannot distinguish token and head axes: both ``(B,H,T,128)`` and
``(B,T,H,128)`` are structurally valid. Compare the post-scatter head axis
with the head count declared by the outer call to catch this silent layout
error.

All guards raise class-P refusals with supported alternatives. They neither
repair unsupported inputs nor accept consent as a substitute for capability.
"""
from __future__ import annotations

import threading
from typing import Final, NoReturn

import torch

from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError

# Hard constraints of the released kernel, refused rather than worked around.
SOL_HEAD_DIM: Final = 128
SOL_DTYPE: Final = torch.bfloat16

# The exact install, both packages. apache-tvm-ffi is a runtime dependency of
# the CuTe path that the package's own requirements do not pull in.
SOL_INSTALL_SPEC: Final = (
    "pip install 'sol-attn @ git+https://github.com/NVlabs/Sana"
    "@71350fae#subdirectory=techniques/sparse_backends' apache-tvm-ffi"
)



class CallPreflight:
    """Carry outer-call facts into the per-call layout guard.

    The kernel sees post-scatter tensors and cannot tell a head axis from a token
    axis on its own. The outer surface knows how many heads the model declared,
    so the expected per-rank head count is computed there and read back inside
    the kernel call. Thread-local for the same reason the dense and sink scopes
    are: one worker process can hold more than one forward, and a head count
    leaking between them would disarm the layout guard.
    """

    __slots__ = ("_local", "_ulysses")

    def __init__(self, ulysses: int) -> None:
        self._ulysses = int(ulysses)
        self._local = threading.local()

    def surface(self, q, k, v, heads, mask, attn_precision, skip_reshape,
                skip_output_reshape, enable_gqa, drop_rows, kv_drop_rows,
                query_bias_groups, kwargs) -> None:
        """Validate the outer surface and record the expected head count."""
        if enable_gqa:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                "sol-attn takes one head count for q, k and v, so a "
                "grouped-query model reaches it only by expanding K and V "
                "first, the memory traffic this kernel is built to "
                "avoid. Use a SAGE_* kernel or TORCH_FLASH instead.",
            ))
        if query_bias_groups:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                "this render biases attention by ordered query groups, which "
                "sol-attn cannot express: it takes no mask and no per-group "
                "weighting. Use a SAGE_* kernel or TORCH_FLASH instead.",
            ))
        if mask is not None:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                "sol-attn takes no attention mask, and this render carries an "
                "effective one. Use a SAGE_* kernel or TORCH_FLASH instead.",
            ))
        if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                "sol-attn needs a positive declared head count and this render "
                "declared none. Use a SAGE_* kernel or TORCH_FLASH instead.",
            ))
        if heads % self._ulysses:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"sol-attn under ulysses {self._ulysses} needs the {heads} "
                "declared heads to divide across the ranks. Use a SAGE_* "
                "kernel or TORCH_FLASH instead.",
            ))
        self._local.expected = heads // self._ulysses

    def expected_local_heads(self) -> int:
        """The per-rank head count this thread's outer call declared."""
        expected = getattr(self._local, "expected", None)
        if expected is None:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                "sol-attn reached the kernel before its outer surface declared "
                "a head count, so the layout guard that separates (B, T, H, D) "
                "from (B, H, T, D) cannot arm and the call is refused rather "
                "than run unguarded. Use a SAGE_* kernel or TORCH_FLASH "
                "instead.",
            ))
        return int(expected)


def validate_bthd(q, k, v, expected_local_heads: int) -> None:
    """Own the axis meaning the kernel accepts without checking.

    Shape equality across q, k and v is two checks at once: no grouped-query
    attention (the head counts agree) and no attention between streams of
    different length. It does not prove self-attention; an equal-length cross
    attention would pass.
    """
    for name, tensor in (("query", q), ("key", k), ("value", v)):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"sol-attn needs a 4D {name} tensor shaped (B, T, H, 128). "
                "Use a SAGE_* kernel or TORCH_FLASH instead.",
            ))
    if q.shape != k.shape or q.shape != v.shape:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "sol-attn is self-attention only and needs q, k and v of one shape "
            f"(one head count, one length); this call carries q {list(q.shape)}, "
            f"k {list(k.shape)}, v {list(v.shape)}. Use a SAGE_* kernel or "
            "TORCH_FLASH instead.",
        ))
    if q.shape[-1] != SOL_HEAD_DIM:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"sol-attn is built for head dimension {SOL_HEAD_DIM} and this "
            f"render carries {q.shape[-1]}. Use a SAGE_* kernel or TORCH_FLASH "
            "instead.",
        ))
    if q.shape[2] != expected_local_heads:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"sol-attn expected {expected_local_heads} heads on axis 2 and "
            f"found {q.shape[2]} in {list(q.shape)}. The kernel validates only "
            "the last dimension, so a (B, H, T, D) tensor would be accepted and "
            "would return wrong output that looks valid (measured 2026-08-14). Use a "
            "SAGE_* kernel or TORCH_FLASH instead.",
        ))
    for name, tensor in (("query", q), ("key", k), ("value", v)):
        if tensor.dtype is not SOL_DTYPE:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"sol-attn is bfloat16 only and the {name} tensor is "
                f"{tensor.dtype}. Pick a "
                "SAGE_* kernel or TORCH_FLASH instead.",
            ))
        if tensor.device.type != "cuda":
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"sol-attn needs CUDA tensors and the {name} tensor is on "
                f"{tensor.device}. Use a SAGE_* kernel or TORCH_FLASH instead.",
            ))


def validate_kernel_surface(*, dropout_p, causal, window_size, alibi_slopes,
                            deterministic, return_attn_probs, attn_layer,
                            joint_tensor_key, joint_tensor_value, joint_strategy,
                            q_descale, k_descale, v_descale, softcap=0.0) -> None:
    """Refuse every option this kernel would ignore rather than honor."""
    if causal:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "sol-attn is noncausal only and this call asked for causal "
            "attention. Use a SAGE_* kernel or TORCH_FLASH instead.",
        ))
    if dropout_p:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "sol-attn takes no attention dropout. Use a SAGE_* kernel or "
            "TORCH_FLASH instead.",
        ))
    if return_attn_probs:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "sol-attn returns no attention probabilities. Use a SAGE_* kernel "
            "or TORCH_FLASH instead.",
        ))
    if tuple(window_size or (-1, -1)) != (-1, -1):
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "sol-attn takes no sliding window. Use a SAGE_* kernel or "
            "TORCH_FLASH instead.",
        ))
    if alibi_slopes is not None:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "sol-attn takes no alibi slopes. Use a SAGE_* kernel or "
            "TORCH_FLASH instead.",
        ))
    if softcap:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "sol-attn takes no logit softcap. Use a SAGE_* kernel or "
            "TORCH_FLASH instead.",
        ))
    if deterministic:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "sol-attn has no deterministic mode to request. Use a SAGE_* "
            "kernel or TORCH_FLASH instead.",
        ))
    if (attn_layer is not None or joint_tensor_key is not None
            or joint_tensor_value is not None or joint_strategy != "none"
            or q_descale is not None or k_descale is not None
            or v_descale is not None):
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "sol-attn received an xFuser surface it cannot honor (an attention "
            "layer, joint tensors or descales). Use a SAGE_* kernel or TORCH_FLASH instead.",
        ))


def refuse_ring(world: int) -> NoReturn:
    """Ring cannot host this kernel, for two independent reasons."""
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"sol-attn cannot serve a ring topology (ring degree {world}). It "
        "returns one tensor and no log-sum-exp, so ring ranks have nothing to "
        "merge their partial softmax states with, and its routing threshold "
        "comes from statistics over the whole key row, which a ring rank does "
        "not hold. Run this workflow on a pure uly* topology, or use a SAGE_* "
        "kernel or TORCH_FLASH instead.",
    ))


def refuse_family(family: str, reason: str) -> NoReturn:
    """Refuse a family this lever is not scoped to."""
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"the sol-attn kernel is scoped to minimax_h3 and this render is "
        f"{family}: {reason} Use a SAGE_* kernel or TORCH_FLASH instead.",
    ))


def refuse_missing_package(missing: str) -> NoReturn:
    """Refuse when the optional dependency is absent, naming the install."""
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"the sol-attn kernel needs {missing}, an optional dependency this worker's "
        "environment cannot import. Install it on "
        f"EVERY box in the mesh with `{SOL_INSTALL_SPEC}`, or use a SAGE_* "
        "kernel or TORCH_FLASH instead.",
    ))


def refuse_unselectable_kernel(name: str, selectable: tuple[str, ...]) -> NoReturn:
    """Refuse a sol kernel name this build does not ship."""
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{name} is not a kernel this build ships. Each tau is its own "
        "measured setting, so the widget offers only these sol-attn kernels: "
        f"{', '.join(selectable)}. Use "
        "one of those, a SAGE_* kernel, or TORCH_FLASH instead.",
    ))


def refuse_geometry(detail: str) -> NoReturn:
    """Refuse a packed geometry the recipe cannot be applied to."""
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"sol-attn cannot run this graph: {detail} Use a SAGE_* kernel or "
        "TORCH_FLASH instead.",
    ))
