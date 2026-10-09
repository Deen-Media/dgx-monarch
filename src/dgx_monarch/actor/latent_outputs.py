"""Strict actor-side latent contracts: the stock empty-latent fix in, flat and packed outputs out."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from ..sampling_contract import fixed_latent_shape, latent_size_tags


def fixed_empty_latent(model_patcher: Any, latent: dict) -> Any:
    """The latent a stock sampler denoises: ``fix_empty_latent_channels``.

    Stock samplers pass the empty latent's size tags too, so an all-zero
    latent built for another ratio is rescaled onto the model's grid.
    The fix is not idempotent while a tag is present (rescaled zeros are
    still empty), so call it once per path on the raw samples and derive the
    noise and every batch slice from that one tensor.
    """
    import comfy.sample as comfy_sample

    return comfy_sample.fix_empty_latent_channels(
        model_patcher, latent["samples"], *latent_size_tags(latent))


def fixed_latent_grid(model_patcher: Any, latent: dict) -> torch.Tensor:
    """A storage-free (meta) tensor shaped like ``fixed_empty_latent``'s result.

    For readers of the grid only, such as the cfg pad equalizer: the shape
    comes from stock's arithmetic on the format the fix reads, so no second
    fixed latent is built beside the one the sampler makes.
    """
    shape = fixed_latent_shape(
        latent["samples"], model_patcher.get_model_object("latent_format"),
        *latent_size_tags(latent))
    return torch.empty(shape, device="meta")


def _canonical_nested_type() -> type[Any]:
    try:
        import comfy.nested_tensor
    except Exception as exc:
        raise TypeError("Comfy NestedTensor runtime type is unavailable") from exc
    nested_type = getattr(comfy.nested_tensor, "NestedTensor", None)
    if not isinstance(nested_type, type):
        raise TypeError("Comfy NestedTensor runtime type is unavailable")
    return nested_type


def is_direct_nested_tensor(value: Any) -> bool:
    """Return whether value satisfies the exact direct Comfy wrapper contract."""
    try:
        direct_nested_modalities(value, "sampler output")
        return True
    except (TypeError, ValueError):
        return False


def direct_nested_modalities(value: Any, path: str) -> tuple[torch.Tensor, ...]:
    """Return the ordered direct modality tensors in a Comfy NestedTensor."""
    try:
        nested_type = _canonical_nested_type()
    except TypeError as exc:
        raise TypeError(f"{path} must be an exact Comfy NestedTensor") from exc
    if isinstance(value, nested_type) and type(value) is not nested_type:
        raise TypeError(f"{path} NestedTensor subclasses are not supported")
    if type(value) is not nested_type:
        raise TypeError(
            f"{path} must be an exact Comfy NestedTensor, got {type(value).__name__}"
        )
    if getattr(value, "is_nested", None) is not True:
        raise TypeError(f"{path} Comfy NestedTensor is missing its nested-kind marker")
    unbind = getattr(nested_type, "unbind", None)
    if not callable(unbind):
        raise TypeError(f"{path} Comfy NestedTensor has no direct unbind method")
    try:
        raw_parts = unbind(value)
    except Exception as exc:
        raise TypeError(f"{path} NestedTensor.unbind failed") from exc
    if type(raw_parts) is not list:
        raise TypeError(
            f"{path} NestedTensor modalities must use the direct list payload"
        )
    parts = tuple(raw_parts)
    if not parts:
        raise TypeError(f"{path} NestedTensor must contain at least one modality")
    if any(type(part) is not torch.Tensor for part in parts):
        raise TypeError(
            f"{path} NestedTensor modalities must all be Tensors; "
            "Tensor subclasses are not supported"
        )
    return parts


def validate_tensor_tree(
    value: Any,
    path: str,
    *,
    require_cpu: bool = False,
    require_finite: bool = False,
) -> None:
    """Validate one Tensor or a direct-modality Comfy NestedTensor."""
    if isinstance(value, torch.Tensor):
        if type(value) is not torch.Tensor:
            raise TypeError(f"{path} Tensor subclasses are not supported")
        if value.ndim == 0:
            raise ValueError(f"{path} must include a batch dimension")
        if require_cpu and (
            value.device.type != "cpu" or value.requires_grad or not value.is_contiguous()
        ):
            raise RuntimeError(
                f"{path} was not normalized to a detached contiguous CPU tensor"
            )
        if require_finite and (torch.is_floating_point(value) or torch.is_complex(value)):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{path} contains non-finite values")
        return
    parts = direct_nested_modalities(value, path)
    if any(part.ndim == 0 for part in parts):
        raise ValueError(f"{path} NestedTensor modalities must include a batch dimension")
    batch = int(parts[0].shape[0])
    if any(int(part.shape[0]) != batch for part in parts):
        raise ValueError(f"{path} NestedTensor modalities must have matching batches")
    for index, part in enumerate(parts):
        validate_tensor_tree(
            part,
            f"{path}[{index}]",
            require_cpu=require_cpu,
            require_finite=require_finite,
        )


def _build_nested(wrapper: Any, parts: tuple[torch.Tensor, ...], path: str) -> Any:
    try:
        result = type(wrapper)(list(parts))
    except Exception as exc:
        raise TypeError(f"{path} NestedTensor could not be reconstructed") from exc
    direct_nested_modalities(result, path)
    return result


def normalize_tensor_tree(value: Any, path: str) -> Any:
    """Return an independent, detached, contiguous CPU sampler output."""
    validate_tensor_tree(value, path, require_finite=True)
    if type(value) is torch.Tensor:
        normalized = value.detach().to(device="cpu", copy=True).contiguous()
    else:
        parts = direct_nested_modalities(value, path)
        normalized = _build_nested(
            value,
            tuple(
                part.detach().to(device="cpu", copy=True).contiguous()
                for part in parts
            ),
            path,
        )
    validate_tensor_tree(normalized, path, require_cpu=True, require_finite=True)
    return normalized


def zeros_like_latent(value: Any, path: str) -> Any:
    """Implement stock disabled-noise semantics per packed modality."""
    validate_tensor_tree(value, path)

    def zero(part: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            part.shape, dtype=part.dtype, layout=part.layout, device="cpu"
        )

    if type(value) is torch.Tensor:
        result: Any = zero(value)
    else:
        result = _build_nested(
            value,
            tuple(zero(part) for part in direct_nested_modalities(value, path)),
            path,
        )
    validate_tensor_tree(result, path, require_cpu=True, require_finite=True)
    return result


def _clone_tensor_tree(value: Any, path: str) -> Any:
    """Clone a normalized tree without sharing any modality storage."""
    validate_tensor_tree(value, path, require_cpu=True, require_finite=True)
    if type(value) is torch.Tensor:
        result: Any = value.clone()
    else:
        result = _build_nested(
            value,
            tuple(part.clone() for part in direct_nested_modalities(value, path)),
            path,
        )
    validate_tensor_tree(result, path, require_cpu=True, require_finite=True)
    return result


def zero_step_outputs(
    latent_image: Any,
    normalize: Callable[[Any, str], Any],
) -> tuple[Any, Any]:
    """Return the unchanged input and a storage-independent denoised value."""
    normalized_input = normalize(latent_image, "custom sampler zero-step input")
    return normalized_input, _clone_tensor_tree(
        normalized_input, "custom sampler zero-step input"
    )


def _match_sampled_modalities(
    parts: tuple[Any, ...], modalities: tuple[torch.Tensor, ...], source: str
) -> None:
    """Fail closed unless rebuilt modalities mirror the sampled ones exactly."""
    if len(parts) != len(modalities):
        raise RuntimeError(
            f"{source} returned the wrong number of NestedTensor modalities"
        )
    for part, modality in zip(parts, modalities, strict=True):
        if type(part) is not torch.Tensor or tuple(part.shape) != tuple(modality.shape):
            raise RuntimeError(
                f"{source} returned malformed NestedTensor modality shapes"
            )
        if part.dtype != modality.dtype:
            raise RuntimeError(
                f"{source} returned NestedTensor modalities with changed dtype"
            )


def _denoised_from_nested_x0(guider: Any, samples: Any, x0: Any) -> Any:
    """Rebuild the denoised value from the nested x0 a packed render gets.

    Comfy samples on one flat pack but hands every callback the nested view of
    x0 as soon as the latent carries more than one modality, so this is the
    only x0 shape a packed render sees on current comfy (ComfyUI 15196).
    Stock SamplerCustom/Advanced feeds that value straight to
    process_latent_out; this makes the same call on the same shape. The CPU
    move is stock's ``x0.cpu()``, spelled per modality so it never leaves the
    direct-modality contract every other value here is validated through.
    """
    modalities = direct_nested_modalities(samples, "custom sampler samples")
    parts = direct_nested_modalities(x0, "custom sampler x0")
    cpu_x0 = _build_nested(
        x0,
        tuple(part.detach().to(device="cpu") for part in parts),
        "custom sampler x0",
    )
    denoised = guider.model_patcher.model.process_latent_out(cpu_x0)
    _match_sampled_modalities(
        direct_nested_modalities(denoised, "custom sampler denoised output"),
        modalities,
        "process_latent_out",
    )
    return denoised


def _denoised_from_packed_x0(samples: Any, denoised: Any) -> Any:
    """Split one processed flat pack back into the sampled modality shapes."""
    import comfy.nested_tensor
    import comfy.utils

    modalities = direct_nested_modalities(samples, "custom sampler samples")
    if type(denoised) is not torch.Tensor:
        raise TypeError(
            "process_latent_out must return a packed Tensor for NestedTensor samples"
        )
    latent_shapes = [part.shape for part in modalities]
    batch = int(latent_shapes[0][0])
    packed_width = sum(math.prod(shape[1:]) for shape in latent_shapes)
    expected_shape = (batch, 1, packed_width)
    if tuple(denoised.shape) != expected_shape:
        raise RuntimeError(
            "process_latent_out returned malformed packed denoised shape: "
            f"{tuple(denoised.shape)}, expected {expected_shape}"
        )
    unpacked = comfy.utils.unpack_latents(denoised, latent_shapes)
    if type(unpacked) is not list:
        raise RuntimeError("unpack_latents modalities must use the direct list payload")
    _match_sampled_modalities(tuple(unpacked), modalities, "unpack_latents")
    return comfy.nested_tensor.NestedTensor(unpacked)


def custom_denoised_output(
    guider: Any,
    samples: Any,
    x0: Any,
    normalize: Callable[[Any, str], Any],
) -> Any:
    """Apply stock process_latent_out and restore packed modality structure."""
    validate_tensor_tree(
        samples, "custom sampler samples", require_finite=True
    )
    if is_direct_nested_tensor(x0):
        denoised = _denoised_from_nested_x0(guider, samples, x0)
    elif type(x0) is not torch.Tensor:
        raise TypeError(f"custom sampler x0 must be a Tensor, got {type(x0).__name__}")
    else:
        # Stock SamplerCustom/Advanced calls process_latent_out(x0.cpu()). From
        # ComfyUI 33aa808 until 49a7422, a packed render's callback received a
        # flat x0. Stock processed that pack and then unpacked
        # it, which this arm repeats. Current ComfyUI hands the callback a nested
        # x0 (branch above).
        denoised = guider.model_patcher.model.process_latent_out(x0.cpu())
        if is_direct_nested_tensor(samples):
            denoised = _denoised_from_packed_x0(samples, denoised)
        elif type(denoised) is not torch.Tensor:
            raise TypeError(
                f"process_latent_out must return a Tensor, got {type(denoised).__name__}"
            )
    normalized = normalize(denoised, "custom sampler denoised output")
    return _clone_tensor_tree(normalized, "custom sampler denoised output")
