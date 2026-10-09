"""Exact, fail-closed comparison for Identity Gate latent outputs."""
from __future__ import annotations

from typing import Any


def direct_nested_tensor_parts(value: object) -> tuple[Any, ...] | None:
    """Return direct Comfy ``NestedTensor`` leaves, or ``None`` for another kind.

    An actual Comfy wrapper with a malformed payload is an error rather than an
    ordinary non-match.  Transaction capture uses that distinction to avoid
    silently skipping modality tensors that must remain immutable.
    """
    try:
        from comfy.nested_tensor import NestedTensor
    except (ImportError, AttributeError):
        value_type = type(value)
        if (value_type.__module__ == "comfy.nested_tensor"
                and value_type.__name__ == "NestedTensor"):
            raise TypeError(
                "cannot resolve the direct Comfy NestedTensor type"
            ) from None
        return None

    nested_type: Any = NestedTensor
    if not isinstance(nested_type, type):
        raise TypeError("comfy.nested_tensor.NestedTensor is not a type")
    if isinstance(value, nested_type) and type(value) is not nested_type:
        raise TypeError("NestedTensor subclasses are not identity-gate inputs")
    if type(value) is not nested_type:
        return None
    if getattr(value, "is_nested", None) is not True:
        raise TypeError("Comfy NestedTensor is missing its nested-kind marker")
    unbind = getattr(nested_type, "unbind", None)
    if not callable(unbind):
        raise TypeError("Comfy NestedTensor has no direct unbind method")

    try:
        raw_parts = unbind(value)
    except Exception as exc:
        raise TypeError("Comfy NestedTensor unbind failed") from exc
    if type(raw_parts) is not list:
        raise TypeError("Comfy NestedTensor modalities must use the direct list payload")
    parts = tuple(raw_parts)
    if not parts:
        raise TypeError("Comfy NestedTensor must contain at least one modality")

    import torch

    if any(type(part) is not torch.Tensor for part in parts):
        raise TypeError(
            "Comfy NestedTensor modalities must all be tensors; "
            "tensor subclasses are not supported"
        )
    return parts


def compare_latents(left: object, right: object) -> tuple[bool, float | None]:
    """Compare every finite tensor in a plain or direct Comfy latent exactly.

    Identity-gate PASS requires the same representation and exact values.
    Unsupported wrappers, malformed modalities, incompatible tensor metadata,
    non-finite values, and comparison failures all deny identity without
    raising after the proof renders have completed.
    """
    import torch

    try:
        left_parts: tuple[Any, ...] | None
        right_parts: tuple[Any, ...] | None
        if type(left) is not type(right):
            return False, None
        if type(left) is torch.Tensor:
            left_parts = (left,)
            right_parts = (right,)
        elif isinstance(left, torch.Tensor):
            return False, None
        else:
            left_parts = direct_nested_tensor_parts(left)
            right_parts = direct_nested_tensor_parts(right)
            if left_parts is None or right_parts is None:
                return False, None
        if len(left_parts) != len(right_parts):
            return False, None

        pairs = tuple(zip(left_parts, right_parts, strict=True))
        if any(
            tuple(lhs.shape) != tuple(rhs.shape)
            or lhs.dtype != rhs.dtype
            or lhs.layout != rhs.layout
            or lhs.device != rhs.device
            for lhs, rhs in pairs
        ):
            return False, None
        if any(
            not bool(torch.isfinite(tensor).all().item())
            for pair in pairs
            for tensor in pair
        ):
            return False, None

        identical = True
        max_diff = 0.0
        for lhs, rhs in pairs:
            component_identical = bool(torch.equal(lhs, rhs))
            identical = identical and component_identical
            if component_identical or not lhs.numel():
                continue
            dtype = torch.complex128 if lhs.is_complex() else torch.float64
            component_diff = (
                lhs.to(dtype=dtype) - rhs.to(dtype=dtype)
            ).abs().max()
            if not bool(torch.isfinite(component_diff).item()):
                return False, None
            max_diff = max(max_diff, float(component_diff.item()))
        return identical, max_diff
    except Exception:
        return False, None
