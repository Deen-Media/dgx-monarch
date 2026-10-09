"""Pure validation for per-render model-sampling overrides."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Literal, TypedDict


class ModelSamplingWire(TypedDict):
    kind: Literal["sd3"]
    shift: float


@dataclass(frozen=True)
class ModelSamplingSpec:
    """Canonical immutable graph-side representation of a sampling patch."""

    kind: Literal["sd3"]
    shift: float

    def to_wire(self) -> ModelSamplingWire:
        return {"kind": self.kind, "shift": self.shift}


def normalize_model_sampling(value: object | None) -> ModelSamplingWire | None:
    """Validate and copy a model-sampling wire value.

    The exact-key check keeps the protocol fail-closed as new sampling variants
    arrive. JSON integers become floats; booleans are refused although ``bool``
    subclasses ``int``.
    """
    if value is None:
        return None
    if isinstance(value, ModelSamplingSpec):
        value = value.to_wire()
    if not isinstance(value, Mapping):
        raise ValueError("model_sampling must be a mapping or null")
    if set(value) != {"kind", "shift"}:
        raise ValueError("model_sampling must contain exactly 'kind' and 'shift'")
    if value["kind"] != "sd3":
        raise ValueError("model_sampling kind must be 'sd3'")
    shift = value["shift"]
    if isinstance(shift, bool) or not isinstance(shift, Real):
        raise ValueError("model_sampling shift must be a finite number in 0..100")
    normalized_shift = float(shift)
    if not math.isfinite(normalized_shift) or not 0.0 <= normalized_shift <= 100.0:
        raise ValueError("model_sampling shift must be a finite number in 0..100")
    return {"kind": "sd3", "shift": normalized_shift}


def freeze_model_sampling(value: object | None) -> ModelSamplingSpec | None:
    """Return the immutable graph-side form of a validated wire value."""
    normalized = normalize_model_sampling(value)
    if normalized is None:
        return None
    return ModelSamplingSpec(kind="sd3", shift=normalized["shift"])
