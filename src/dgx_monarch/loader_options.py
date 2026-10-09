"""Shared, fail-closed contract for diffusion-model loader options."""
from __future__ import annotations

from collections.abc import Mapping

# Append-only: Comfy workflows persist combo values, but retaining the existing
# order also avoids surprising older frontends that still cache combo indices.
SUPPORTED_WEIGHT_DTYPES = (
    "default",
    "fp8_e4m3fn",
    "fp8_e4m3fn_fast",
    "fp8_e5m2",
    "bf16",
)


def validate_weight_dtype(value: object) -> str:
    """Return a supported loader dtype or reject malformed RPC/UI input."""
    if not isinstance(value, str):
        raise TypeError(
            "weight_dtype must be a string "
            f"(got {type(value).__name__})"
        )
    if value not in SUPPORTED_WEIGHT_DTYPES:
        supported = ", ".join(SUPPORTED_WEIGHT_DTYPES)
        display = value if len(value) <= 64 else f"{value[:61]}..."
        raise ValueError(
            f"unsupported weight_dtype {display!r}; expected one of: {supported}"
        )
    return value


def weight_dtype_from_options(options: Mapping[str, object] | None) -> str:
    """Read and validate the worker-side dtype option, defaulting explicitly."""
    if options is None:
        return "default"
    if not isinstance(options, Mapping):
        raise TypeError(
            "model options must be a mapping "
            f"(got {type(options).__name__})"
        )
    return validate_weight_dtype(options.get("weight_dtype", "default"))


def validate_model_spec_weight_dtype(
    model_spec: object,
    *,
    field: str = "model spec",
) -> str:
    """Validate a model spec before the ``cleanup_on_failure`` body runs."""
    if not isinstance(model_spec, Mapping):
        raise TypeError(
            f"{field} must be a mapping "
            f"(got {type(model_spec).__name__})"
        )
    return weight_dtype_from_options(model_spec.get("options"))


def validate_sample_request_weight_dtypes(request: object) -> None:
    """Validate conditional/unconditional loader options before GPU dispatch."""
    if not isinstance(request, Mapping):
        raise TypeError(
            "sample request must be a mapping "
            f"(got {type(request).__name__})"
        )
    if "model" not in request:
        raise ValueError("sample request is missing its model spec")
    validate_model_spec_weight_dtype(request["model"], field="sample model spec")
    uncond_spec = request.get("uncond_model")
    if uncond_spec is not None:
        validate_model_spec_weight_dtype(
            uncond_spec,
            field="sample uncond_model spec",
        )
