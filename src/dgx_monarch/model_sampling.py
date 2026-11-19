"""Pure validation for per-render model-sampling overrides.

Two kinds travel on the wire. ``sd3`` is ComfyUI's ModelSamplingSD3 shift.
``minimax_h3`` is ComfyUI's ModelSamplingMiniMaxH3 node (``MiniMaxH3SigmaShift``),
which sets the video and audio flow shifts together; p-h3-image needs it at 1.0
and 1.0 (h3_sigma_shift.py). Each kind has an exact key set, so the protocol
stays fail-closed as new kinds arrive, and an ``sd3`` wire value is the same
two keys it has always been.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Literal, TypedDict

SD3: Literal["sd3"] = "sd3"
MINIMAX_H3: Literal["minimax_h3"] = "minimax_h3"
# ComfyUI's MiniMaxH3SigmaShift bounds both widgets to 0.01..100.
H3_SHIFT_MIN = 0.01
H3_SHIFT_MAX = 100.0


class ModelSamplingWire(TypedDict):
    kind: Literal["sd3"]
    shift: float


class H3ModelSamplingWire(TypedDict):
    kind: Literal["minimax_h3"]
    shift_video: float
    shift_audio: float


AnyModelSamplingWire = ModelSamplingWire | H3ModelSamplingWire


@dataclass(frozen=True)
class ModelSamplingSpec:
    """Canonical immutable graph-side representation of a sampling patch.

    For ``minimax_h3``, ``shift`` is the video shift and ``shift_audio`` the
    audio shift; ``sd3`` leaves ``shift_audio`` unset.
    """

    kind: Literal["sd3", "minimax_h3"]
    shift: float
    shift_audio: float | None = None

    def to_wire(self) -> AnyModelSamplingWire:
        if self.kind == MINIMAX_H3:
            return {"kind": MINIMAX_H3, "shift_video": self.shift,
                    "shift_audio": float(self.shift_audio or 0.0)}
        return {"kind": self.kind, "shift": self.shift}  # type: ignore[typeddict-item]


def _shift(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"model_sampling {name} must be a finite number in {low:g}..{high:g}")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"model_sampling {name} must be a finite number in {low:g}..{high:g}")
    return number


def normalize_model_sampling(value: object | None) -> AnyModelSamplingWire | None:
    """Validate and copy a model-sampling wire value.

    The exact-key check keeps the protocol fail-closed as new sampling variants
    arrive. JSON integers become floats; booleans are refused although ``bool``
    subclasses ``int``.
    """
    if value is None:
        return None
    if isinstance(value, ModelSamplingSpec):
        if value.kind == MINIMAX_H3 and value.shift_audio is None:
            raise ValueError("model_sampling minimax_h3 needs shift_audio")
        if value.kind == SD3 and value.shift_audio is not None:
            raise ValueError("model_sampling sd3 takes no shift_audio")
        value = value.to_wire()
    if not isinstance(value, Mapping):
        raise ValueError("model_sampling must be a mapping or null")
    kind = value.get("kind")
    if kind == MINIMAX_H3:
        if set(value) != {"kind", "shift_video", "shift_audio"}:
            raise ValueError("model_sampling minimax_h3 must contain exactly 'kind', "
                             "'shift_video' and 'shift_audio'")
        return {"kind": MINIMAX_H3,
                "shift_video": _shift(value["shift_video"], "shift_video", H3_SHIFT_MIN, H3_SHIFT_MAX),
                "shift_audio": _shift(value["shift_audio"], "shift_audio", H3_SHIFT_MIN, H3_SHIFT_MAX)}
    if set(value) != {"kind", "shift"}:
        raise ValueError("model_sampling must contain exactly 'kind' and 'shift'")
    if kind != SD3:
        raise ValueError("model_sampling kind must be 'sd3' or 'minimax_h3'")
    return {"kind": SD3, "shift": _shift(value["shift"], "shift", 0.0, 100.0)}


def freeze_model_sampling(value: object | None) -> ModelSamplingSpec | None:
    """Return the immutable graph-side form of a validated wire value."""
    normalized = normalize_model_sampling(value)
    if normalized is None:
        return None
    if normalized["kind"] == MINIMAX_H3:
        return ModelSamplingSpec(kind=MINIMAX_H3, shift=normalized["shift_video"],
                                 shift_audio=normalized["shift_audio"])
    return ModelSamplingSpec(kind=SD3, shift=normalized["shift"])


def sampling_kind(value: object | None) -> str | None:
    """The kind of a spec or wire value, or None; never raises."""
    if isinstance(value, ModelSamplingSpec):
        return value.kind
    if isinstance(value, Mapping):
        kind = value.get("kind")
        return kind if isinstance(kind, str) else None
    return None
