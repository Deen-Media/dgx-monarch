"""Driver-side validation and assembly of sampler leader outputs."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ..transfer_utils import prefer_error, raise_with_distinct_cause
from .latent_identity import direct_nested_tensor_parts


def _failure_evidence(exc: BaseException) -> str:
    try:
        return repr(exc)
    except BaseException:
        return f"<{type(exc).__name__}: repr failed>"


def _leader_failure_label(
    index: int,
    leader_dp_ranks: list[int],
) -> str:
    rank = (leader_dp_ranks[index]
            if index < len(leader_dp_ranks) else f"leader index {index}")
    return f"latent-read failures up to dp rank {rank} also include"


def _note_count(exc: BaseException) -> int | None:
    """Observe helper note publication without consulting exception truthiness."""
    try:
        notes = getattr(exc, "__notes__", None)
        return 0 if notes is None else len(notes)
    except BaseException:
        return None


def _reconcile_leader_failure(
    current: BaseException,
    candidate: BaseException,
    index: int,
    leader_dp_ranks: list[int],
) -> tuple[BaseException, BaseException | None]:
    """Add leader context around the shared cancellation-first policy."""
    current_notes = _note_count(current)
    candidate_notes = _note_count(candidate)
    label = _leader_failure_label(index, leader_dp_ranks)
    strongest = prefer_error(current, candidate, label)
    if current is candidate:
        return strongest, None

    before_notes = candidate_notes if strongest is candidate else current_notes
    if before_notes is not None and _note_count(strongest) == before_notes:
        detail = current if strongest is candidate else candidate
        try:
            strongest.add_note(f"{label}: {_failure_evidence(detail)}")
        except BaseException:
            pass
    return strongest, current if strongest is candidate else None


def _validate_return_tensor(value: Any, path: str) -> None:
    if type(value) is not torch.Tensor:
        raise RuntimeError(
            f"{path} must be an exact tensor, got {type(value).__name__}"
        )
    if value.ndim == 0:
        raise RuntimeError(f"{path} must include a batch dimension")
    if int(value.shape[0]) <= 0:
        raise RuntimeError(f"{path} has an empty batch")
    if (
        value.layout != torch.strided
        or value.device.type != "cpu"
        or value.requires_grad
        or not value.is_contiguous()
    ):
        raise RuntimeError(f"{path} must be a detached contiguous CPU tensor")
    for chunk in value.reshape(-1).split(1_048_576):
        try:
            finite = bool(torch.isfinite(chunk).all().item())
        except Exception as exc:
            raise RuntimeError(f"{path} could not be checked for finite values") from exc
        if not finite:
            raise RuntimeError(f"{path} contains non-finite values")


def _nested_return_modalities(value: Any, path: str) -> tuple[torch.Tensor, ...]:
    """Validate a direct Comfy NestedTensor modality wrapper from an actor."""
    try:
        parts = direct_nested_tensor_parts(value)
    except TypeError as exc:
        raise RuntimeError(
            f"{path} is not a valid direct Comfy NestedTensor: {exc}"
        ) from exc
    if parts is None:
        raise RuntimeError(f"{path} must be a direct Comfy NestedTensor")
    for index, part in enumerate(parts):
        _validate_return_tensor(part, f"{path}[{index}]")
    batch = int(parts[0].shape[0])
    if any(int(part.shape[0]) != batch for part in parts):
        raise RuntimeError(f"{path} NestedTensor modalities have mismatched batches")
    return parts


def is_direct_nested_tensor(value: Any, path: str) -> bool:
    """Return whether value is the exact current Comfy wrapper; reject subclasses."""
    try:
        return direct_nested_tensor_parts(value) is not None
    except TypeError as exc:
        raise RuntimeError(
            f"{path} is not a valid direct Comfy NestedTensor: {exc}"
        ) from exc


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_storage = left.untyped_storage()
    right_storage = right.untyped_storage()
    # Empty tensors report data_ptr() == 0 even when two distinct zero-numel
    # views share a nonempty backing.  Public storage-object identity catches
    # that case without treating every independently allocated empty tensor as
    # aliased; the range check below covers distinct wrappers over one region.
    if left_storage is right_storage:
        return True
    left_nbytes = int(left_storage.nbytes())
    right_nbytes = int(right_storage.nbytes())
    if left_nbytes <= 0 or right_nbytes <= 0:
        return False
    left_pointer = int(left_storage.data_ptr())
    right_pointer = int(right_storage.data_ptr())
    if left_pointer == 0 or right_pointer == 0:
        return False
    return (
        left_pointer < right_pointer + right_nbytes
        and right_pointer < left_pointer + left_nbytes
    )


def _construct_nested(wrapper_type: type, parts: tuple[torch.Tensor, ...], path: str) -> Any:
    try:
        result = wrapper_type(list(parts))
    except Exception as exc:
        raise RuntimeError(f"{path} could not reconstruct NestedTensor") from exc
    if type(result) is not wrapper_type:
        raise RuntimeError(f"{path} changed NestedTensor wrapper type")
    _nested_return_modalities(result, path)
    return result


def clone_latent_samples(value: Any, path: str) -> Any:
    """Clone a flat or direct packed output without preserving storage aliases."""
    if isinstance(value, torch.Tensor):
        _validate_return_tensor(value, path)
        return value.clone()
    parts = _nested_return_modalities(value, path)
    return _construct_nested(type(value), tuple(part.clone() for part in parts), path)


def concatenate_latent_batches(values: list[Any], path: str) -> Any:
    """Concatenate flat or packed batches without losing modality identity."""
    if not values:
        raise RuntimeError(f"{path} returned no latent batches")
    nested = [
        is_direct_nested_tensor(value, f"{path}[{index}]")
        for index, value in enumerate(values)
    ]
    if any(nested) and not all(nested):
        raise RuntimeError(f"{path} mixed Tensor and NestedTensor batches")
    if not any(nested):
        for index, value in enumerate(values):
            _validate_return_tensor(value, f"{path}[{index}]")
        reference = values[0]
        for index, value in enumerate(values[1:], 1):
            if (
                value.shape[1:] != reference.shape[1:]
                or value.dtype != reference.dtype
                or value.layout != reference.layout
            ):
                raise RuntimeError(f"{path}[{index}] changed non-batch Tensor structure")
        return reference if len(values) == 1 else torch.cat(values, dim=0)

    wrapper_type = type(values[0])
    modalities: list[tuple[torch.Tensor, ...]] = []
    for index, value in enumerate(values):
        if type(value) is not wrapper_type:
            raise RuntimeError(f"{path}[{index}] changed NestedTensor wrapper type")
        modalities.append(_nested_return_modalities(value, f"{path}[{index}]"))
    expected_count = len(modalities[0])
    if any(len(parts) != expected_count for parts in modalities):
        raise RuntimeError(f"{path} changed NestedTensor modality count")
    combined: list[torch.Tensor] = []
    for modality_index in range(expected_count):
        modality_parts = [batch[modality_index] for batch in modalities]
        reference = modality_parts[0]
        for batch_index, part in enumerate(modality_parts[1:], 1):
            if (
                part.shape[1:] != reference.shape[1:]
                or part.dtype != reference.dtype
                or part.layout != reference.layout
            ):
                raise RuntimeError(
                    f"{path}[{batch_index}] changed structure for packed modality "
                    f"{modality_index}"
                )
        combined.append(
            reference if len(modality_parts) == 1 else torch.cat(modality_parts, dim=0)
        )
    return _construct_nested(wrapper_type, tuple(combined), f"{path} combined result")


def materialize_leader_samples(
    leaders: list[dict[str, Any]],
    leader_dp_ranks: list[int],
    dp: int,
    reader: Callable[[dict], Any],
) -> tuple[Any, list[Any]]:
    """Read, validate, and DP-assemble leader sample descriptors."""
    tensors: list[Any] = []
    first_error: BaseException | None = None
    error_cause: BaseException | None = None
    for index, result in enumerate(leaders):
        value: Any = None
        try:
            try:
                value = reader(result["latent"])
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                    tensors.clear()
                else:
                    first_error, candidate_cause = _reconcile_leader_failure(
                        first_error, exc, index, leader_dp_ranks)
                    if candidate_cause is not None:
                        error_cause = candidate_cause
            else:
                if first_error is None:
                    tensors.append(value)
        except BaseException as boundary:
            # A one-shot async exception can land after reader() settled its
            # descriptor but before Python publishes the result/error. Recover
            # the active reader error without discarding a stronger boundary.
            context = boundary.__context__
            active = boundary
            active_cause: BaseException | None = None
            if context is not None and context is not boundary:
                active, active_cause = _reconcile_leader_failure(
                    context, boundary, index, leader_dp_ranks)
            if first_error is None:
                first_error = active
                error_cause = active_cause
                tensors.clear()
            elif active is not first_error:
                first_error, candidate_cause = _reconcile_leader_failure(
                    first_error, active, index, leader_dp_ranks)
                if candidate_cause is not None:
                    error_cause = candidate_cause
        finally:
            if first_error is not None:
                value = None
    if first_error is not None:
        raise_with_distinct_cause(first_error, error_cause)
    nested_samples = [
        is_direct_nested_tensor(value, f"leader samples for dp rank {dp_rank}")
        for dp_rank, value in zip(leader_dp_ranks, tensors, strict=True)
    ]
    if any(nested_samples):
        if not all(nested_samples):
            raise RuntimeError(
                "inconsistent leader samples: mixed Tensor and NestedTensor results"
            )
        if int(dp) != 1 or len(tensors) != 1:
            raise RuntimeError(
                "packed NestedTensor samples cannot be reconstructed across DP leaders"
            )
        _nested_return_modalities(tensors[0], "leader samples")
        return tensors[0], tensors
    for dp_rank, tensor in zip(leader_dp_ranks, tensors, strict=True):
        _validate_return_tensor(tensor, f"leader samples for dp rank {dp_rank}")
    samples = concatenate_latent_batches(tensors, "leader samples")
    return samples, tensors


def collect_denoised_output(
    extras: list[dict],
    samples: Any,
    tensors: list[Any],
    leader_dp_ranks: list[int],
) -> Any | None:
    """Validate and assemble custom-sampler denoised output, when present."""
    denoised_present = ["denoised" in extra for extra in extras]
    if not any(denoised_present):
        return None
    if not all(denoised_present):
        missing = [
            leader_dp_ranks[index]
            for index, present in enumerate(denoised_present)
            if not present
        ]
        raise RuntimeError(
            "inconsistent custom-sampler denoised results: missing denoised "
            f"output from dp ranks {missing}"
        )
    denoised = [extra["denoised"] for extra in extras]
    nested_denoised = [
        is_direct_nested_tensor(
            value, f"custom-sampler denoised output for dp rank {dp_rank}"
        )
        for dp_rank, value in zip(leader_dp_ranks, denoised, strict=True)
    ]
    if is_direct_nested_tensor(samples, "leader samples"):
        if not all(nested_denoised):
            raise RuntimeError(
                "custom-sampler denoised structure does not match NestedTensor samples"
            )
        if len(denoised) != 1:
            raise RuntimeError("packed custom-sampler denoised output cannot span DP leaders")
        if type(denoised[0]) is not type(samples):
            raise RuntimeError("custom-sampler denoised changed NestedTensor wrapper type")
        sample_parts = _nested_return_modalities(samples, "leader samples")
        denoised_parts = _nested_return_modalities(
            denoised[0], "custom-sampler denoised output"
        )
        if len(denoised_parts) != len(sample_parts):
            raise RuntimeError(
                "custom-sampler denoised NestedTensor modality count does not match samples"
            )
        for index, (denoised_part, sample_part) in enumerate(
            zip(denoised_parts, sample_parts, strict=True)
        ):
            if denoised_part.shape != sample_part.shape or denoised_part.dtype != sample_part.dtype:
                raise RuntimeError(
                    "custom-sampler denoised modality does not match samples: "
                    f"modality {index} has {tuple(denoised_part.shape)} "
                    f"{denoised_part.dtype}, expected {tuple(sample_part.shape)} "
                    f"{sample_part.dtype}"
                )
        if any(
            _shares_storage(denoised_part, sample_part)
            for denoised_part in denoised_parts
            for sample_part in sample_parts
        ):
            raise RuntimeError("custom-sampler denoised modality aliases the primary samples")
        if any(
            _shares_storage(left, right)
            for index, left in enumerate(denoised_parts)
            for right in denoised_parts[index + 1 :]
        ):
            raise RuntimeError("custom-sampler denoised modalities alias each other")
        return denoised[0]
    if any(nested_denoised):
        raise RuntimeError("custom-sampler denoised structure does not match Tensor samples")
    for dp_rank, tensor, sample in zip(
        leader_dp_ranks, denoised, tensors, strict=True
    ):
        _validate_return_tensor(
            tensor, f"custom-sampler denoised output for dp rank {dp_rank}"
        )
        if tensor.shape != sample.shape or tensor.dtype != sample.dtype:
            raise RuntimeError(
                "custom-sampler denoised output does not match samples for "
                f"dp rank {dp_rank}: {tuple(tensor.shape)} {tensor.dtype} vs "
                f"{tuple(sample.shape)} {sample.dtype}"
            )
    if any(
        _shares_storage(denoised_tensor, sample_tensor)
        for denoised_tensor in denoised
        for sample_tensor in tensors
    ):
        raise RuntimeError("custom-sampler denoised output aliases the primary samples")
    reference = denoised[0]
    return reference if len(denoised) == 1 else torch.cat(denoised, dim=0)
