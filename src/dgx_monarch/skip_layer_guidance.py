"""Pure validation and the shared refusal predicate for per-render Skip Layer Guidance.

The driver node, the driver preflight, the Worker belt and the sweep matrix read
the same functions here, so the four can never disagree about when a skip is
active or when a topology cannot carry it (docs/TROUBLESHOOTING.md #126).
Standard library only, like model_sampling.py.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TypedDict

_FIELD = "skip_layer_guidance"
_KEYS = frozenset({"double_layers", "single_layers", "start_percent", "end_percent"})
_MAX_LAYER_TEXT = 256
# ComfyUI's own parse of the layer widgets (comfy_extras/nodes_slg.py).
_LAYER_NUMBER = re.compile(r"\d+")


class SkipLayerGuidanceWire(TypedDict):
    double_layers: str
    single_layers: str
    start_percent: float
    end_percent: float


@dataclass(frozen=True)
class SkipLayerGuidanceSpec:
    """Canonical immutable graph-side form of the Skip Layer Guidance widgets."""

    double_layers: str
    single_layers: str
    start_percent: float
    end_percent: float

    def to_wire(self) -> SkipLayerGuidanceWire:
        return {
            "double_layers": self.double_layers,
            "single_layers": self.single_layers,
            "start_percent": self.start_percent,
            "end_percent": self.end_percent,
        }


def _layer_text(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) > _MAX_LAYER_TEXT:
        raise ValueError(f"{_FIELD} {name} must be a string of at most {_MAX_LAYER_TEXT} characters")
    return value


def _percent(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{_FIELD} {name} must be a finite number in 0..1")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{_FIELD} {name} must be a finite number in 0..1")
    return number


def normalize_skip_layer_guidance(value: object | None) -> SkipLayerGuidanceWire | None:
    """Validate and copy a Skip Layer Guidance wire value.

    The exact-key check keeps the protocol fail-closed, as model_sampling.py
    does. JSON integers become floats; booleans are refused although ``bool``
    subclasses ``int``.
    """
    if value is None:
        return None
    if isinstance(value, SkipLayerGuidanceSpec):
        value = value.to_wire()
    if not isinstance(value, Mapping):
        raise ValueError(f"{_FIELD} must be a mapping or null")
    if set(value) != _KEYS:
        raise ValueError(f"{_FIELD} must contain exactly {', '.join(sorted(_KEYS))}")
    return {
        "double_layers": _layer_text(value["double_layers"], "double_layers"),
        "single_layers": _layer_text(value["single_layers"], "single_layers"),
        "start_percent": _percent(value["start_percent"], "start_percent"),
        "end_percent": _percent(value["end_percent"], "end_percent"),
    }


def _skips_a_layer(wire: SkipLayerGuidanceWire) -> bool:
    return bool(_LAYER_NUMBER.findall(wire["double_layers"])
                or _LAYER_NUMBER.findall(wire["single_layers"]))


def skip_layer_guidance_active(value: object | None) -> bool:
    """True when either layer list names a layer, by ComfyUI's own parse.

    ComfyUI's node returns the model unchanged when both lists hold no digit, so
    such a value is inactive here too.
    """
    wire = normalize_skip_layer_guidance(value)
    return wire is not None and _skips_a_layer(wire)


def freeze_skip_layer_guidance(value: object | None) -> SkipLayerGuidanceSpec | None:
    """The immutable graph-side form, or None when the value skips no layer.

    None for an inactive value keeps a graph with empty layer lists byte for
    byte the request of a graph without the node.
    """
    wire = normalize_skip_layer_guidance(value)
    if wire is None or not _skips_a_layer(wire):
        return None
    return SkipLayerGuidanceSpec(**wire)


def skip_layer_guidance_refusal_reason(
    *, cfg: int, sp: int, family: str, family_honours: bool, dual_model: bool,
) -> str | None:
    """Why an active skip cannot run on this render, or None when it can.

    Checked in this order: a two-checkpoint guider, a cfg topology, then a
    sequence-parallel family whose sharded forward is not shown to honor
    ComfyUI's block replacement. The cfg remedy names uly2 and ring2 only for
    a family that honors the replacement, and auto only there too, because no
    such family has a cfg auto row (tests/test_skip_layer_guidance.py pins
    that). The raise sites add the common remedy.
    """
    if dual_model:
        return ("Skip Layer Guidance (DGX Monarch) with a two-checkpoint guider is not "
                "carried to the Workers; use a single-model guider to keep the skip.")
    if cfg > 1:
        keep = "auto, uly2, ring2, dp2 or single" if family_honours else "dp2 or single"
        return ("Skip Layer Guidance (DGX Monarch) runs the conditional and unconditional "
                "passes as two separate model calls on every step in its window, and "
                "cfg-parallel can split only a call that carries both; set the Init "
                f"topology to {keep} to keep the skip.")
    if sp > 1 and not family_honours:
        return ("Skip Layer Guidance (DGX Monarch) reaches a sequence-parallel render only "
                "through the family's sharded forward, and dgx-monarch has shown that only "
                f"for moss_image; {family}'s sharded forward could render without the skip "
                "and report nothing; set the Init topology to single or dp2, where ComfyUI's "
                "own forward runs, to keep the skip.")
    return None
