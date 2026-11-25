"""Hold a checkpoint's quantization authority against the live layer ComfyUI built.

The un-bake capture (``actor.unbake.capture_unbake_record``) records a
quantized key only when it can rebuild the layer exactly: the checkpoint's
config (``unbake_file._quant_layer_conf``) must describe the same behaviour as
the live layer's own ``comfy_quant`` marker, and a direct per-layer marker must
name only what the generic restore replays. These helpers answer both.
"""
from __future__ import annotations

import json
from typing import Any

import torch

from .unbake_file import _QUANT_CONF_MAX_BYTES, _decode_quant_conf_bytes

# A direct per-layer ``comfy_quant`` marker is a layer's whole quantization
# authority when neither embedded metadata nor a legacy conversion covers it
# (``_quant_layer_conf``). Restore replays exactly the keys Comfy's loader
# reads, so this allowlist is closed: an unlisted key means the load consumed
# something the replay would not.
_DIRECT_QUANT_FP8_KEYS = frozenset({"format", "full_precision_matrix_mult"})
_DIRECT_QUANT_CONF_KEYS: dict[str, frozenset[str]] = {  # read-only after import
    "float8_e4m3fn": _DIRECT_QUANT_FP8_KEYS,
    "float8_e5m2": _DIRECT_QUANT_FP8_KEYS,
    "int8_tensorwise": _DIRECT_QUANT_FP8_KEYS | {"convrot", "convrot_groupsize", "params"},
}
_DIRECT_QUANT_PARAMS_KEYS = frozenset({"convrot", "convrot_groupsize"})


def _direct_quant_conf_reason(conf: dict[str, Any]) -> str | None:
    """Why generic un-bake cannot replay a direct marker's recipe, or None.
    Comfy reads the two convrot settings flat or from a nested ``params``."""
    fmt = conf.get("format")
    allowed = _DIRECT_QUANT_CONF_KEYS.get(fmt) if isinstance(fmt, str) else None
    if allowed is None:
        return f"format {fmt!r} has no direct-marker restore recipe"
    params = conf.get("params")
    extra = sorted(set(conf) - allowed) + sorted(
        set(params) - _DIRECT_QUANT_PARAMS_KEYS if isinstance(params, dict) else ())
    if extra:
        return (f"the marker names {', '.join(repr(name) for name in extra)}, "
                f"which the {fmt} restore does not replay")
    return None


def _direct_quant_module_supported(
    module: Any,
    conf: dict[str, Any],
    components: dict,
) -> bool:
    """Whether generic restore has the owning-module state it consumes."""
    factory_kwargs = getattr(module, "factory_kwargs", None)
    orig_shape = getattr(module, "_orig_shape", None)
    weight = getattr(module, "weight", None)
    if (
        not isinstance(factory_kwargs, dict)
        or not isinstance(factory_kwargs.get("dtype"), torch.dtype)
        or not isinstance(orig_shape, (tuple, list))
        or not orig_shape
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in orig_shape
        )
        or not isinstance(getattr(weight, "device", None), torch.device)
    ):
        return False
    required = {"weight", "weight_scale"}
    actual_scales = {
        name for name in components if name.startswith("weight_scale")
    }
    return (
        actual_scales == required - {"weight"}
        and all(
            isinstance(components.get(name), torch.Tensor)
            for name in required
        )
    )


def _quant_conf_matches_live(checkpoint: dict, live: dict | None) -> bool:
    """Compare the behavior Comfy retains after loading a quant config."""
    if live is None:
        return False

    def normalized(conf: dict) -> tuple | None:
        fmt = conf.get("format")
        full_precision = conf.get("full_precision_matrix_mult", False)
        if not isinstance(fmt, str) or not isinstance(full_precision, bool):
            return None
        handled = {"format", "full_precision_matrix_mult", "params"}
        if fmt != "int8_tensorwise":
            extras = {
                key: value for key, value in conf.items()
                if key not in handled
            }
            return (
                fmt,
                full_precision,
                json.dumps(extras, sort_keys=True, separators=(",", ":")),
            )
        params = conf.get("params", {})
        if not isinstance(params, dict):
            return None
        handled.update({"convrot", "convrot_groupsize"})
        convrot = conf.get("convrot", params.get("convrot", False))
        if not isinstance(convrot, bool):
            return None
        groupsize = conf.get(
            "convrot_groupsize",
            params.get("convrot_groupsize", 256),
        )
        if (
            isinstance(groupsize, bool)
            or not isinstance(groupsize, int)
            or groupsize <= 0
        ):
            return None
        extras = {
            key: value for key, value in conf.items()
            if key not in handled
        }
        return (
            fmt,
            full_precision,
            convrot,
            groupsize if convrot else None,
            json.dumps(extras, sort_keys=True, separators=(",", ":")),
        )

    checkpoint_normalized = normalized(checkpoint)
    live_normalized = normalized(live)
    return (
        checkpoint_normalized is not None
        and live_normalized is not None
        and checkpoint_normalized == live_normalized
    )


def _decode_live_quant_conf(value: Any) -> dict | None:
    """Decode Comfy's bounded per-layer ``comfy_quant`` state marker."""
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
        or value.numel() > _QUANT_CONF_MAX_BYTES
    ):
        return None
    try:
        raw = bytes(value.detach().to("cpu").contiguous().numpy().tobytes())
    except (TypeError, RuntimeError):
        return None
    return _decode_quant_conf_bytes(raw)
