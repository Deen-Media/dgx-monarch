"""Parse checkpoint slices and verify byte bindings for lazy LoRA restoration.

This module handles safetensors descriptors, sampled byte verification, key
binding, and ComfyUI's per-layer quantization authority: embedded metadata,
legacy scaled-FP8 conversion, and direct markers. Model mutation and restore
execution live in ``actor.unbake``.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import torch

from .. import safetensors_header as _safetensors_header

# Sampled-byte verification windows (start, middle and end of each tensor),
# shared with the slab load path through the torch-free certificate module.
from .slab_certificate import FINGERPRINT_WINDOW as _FINGERPRINT_WINDOW
from .slab_certificate import fingerprint_windows as _fingerprint_windows

_QUANT_CONF_MAX_BYTES = 4096


class _InvalidQuantConfError(ValueError):
    """A selected checkpoint quantization authority is malformed."""

# safetensors dtype tag -> torch dtype, for every dtype the shared parser
# accepts that this torch build has. The parser validates each descriptor
# before this mapping is used.
# Read-only after import.
_ST_TO_TORCH = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
    "C64": torch.complex64,
}
_ST_TO_TORCH.update(
    {
        tag: dtype
        for tag, dtype in {
            "U16": getattr(torch, "uint16", None),
            "U32": getattr(torch, "uint32", None),
            "U64": getattr(torch, "uint64", None),
            "F8_E4M3FNUZ": getattr(torch, "float8_e4m3fnuz", None),
            "F8_E5M2FNUZ": getattr(torch, "float8_e5m2fnuz", None),
            "F8_E8M0": getattr(torch, "float8_e8m0fnu", None),
            "F4": getattr(torch, "float4_e2m1fn_x2", None),
        }.items()
        if dtype is not None
    }
)


@dataclass
class _FileTensor:
    dtype: torch.dtype
    shape: tuple[int, ...]
    start: int  # absolute file offset of the first byte
    nbytes: int
    # When set, the checkpoint stores this tensor in a dtype other than the
    # model's (e.g. F32 on disk, bf16 resident): restore repeats the cast after
    # reading. Capture sets it only when casting sampled file windows
    # reproduces the live tensor's bytes.
    cast_to: torch.dtype | None = None


def read_safetensors_header(
    path: str,
    want_metadata: bool = False,
    want_identity: bool = False,
):
    """Name -> slice map from a .safetensors header, offsets made absolute.

    Reads only the header but validates every descriptor against the complete
    file layout before returning any slice. want_metadata appends the
    __metadata__ dict and want_identity the parsed header, in that order."""
    parsed = _safetensors_header.read_safetensors_header(path)
    out: dict[str, _FileTensor] = {}
    for name, descriptor in parsed.tensors.items():
        dtype = _ST_TO_TORCH.get(descriptor.dtype)
        if dtype is None:
            raise _safetensors_header.UnsupportedSafetensorsDtypeError(
                f"{path}: safetensors dtype {descriptor.dtype!r} is not supported "
                "by this torch build"
            )
        shape = descriptor.shape
        if descriptor.dtype == "F4":
            # safetensors records logical float4 values; torch's x2 dtype
            # records packed bytes, so its final dimension is half as large.
            if not shape or shape[-1] % 2:
                raise _safetensors_header.SafetensorsHeaderError(
                    f"{path}: malformed packed F4 shape for {name!r}: {shape}"
                )
            shape = (*shape[:-1], shape[-1] // 2)
        out[name] = _FileTensor(
            dtype=dtype,
            shape=shape,
            start=descriptor.start,
            nbytes=descriptor.nbytes,
        )
    if want_metadata and want_identity:
        return out, parsed.metadata, parsed
    if want_metadata:
        return out, parsed.metadata
    if want_identity:
        return out, parsed
    return out


def _tensor_bytes_window(t: torch.Tensor, offset: int, length: int) -> bytes:
    """Return raw little-endian tensor bytes starting at ``offset``."""
    flat = t.detach().reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    as_bytes = flat.view(torch.uint8)
    return bytes(as_bytes[offset : offset + length].cpu().numpy().tobytes())


def _candidate_file_bindings(key: str) -> list[tuple[str, str | None]]:
    """Candidate file keys paired with Comfy's legacy model prefix.

    ``convert_old_quants`` looks for ``<model_prefix>scaled_fp8`` before it
    renames ``scale_weight`` components. The optional second item is the
    namespace Comfy's diffusion-only loader can strip before its second
    conversion pass. A bare ``diffusion_model.`` key is a supported weight
    spelling, but the loader in ComfyUI ``bd34f338`` does not strip that prefix,
    so it must not authorize a namespace-local legacy marker.
    """
    if key.startswith("diffusion_model."):
        return [
            (key, None),
            (key[len("diffusion_model.") :], ""),
            ("model." + key, "model.diffusion_model."),
        ]
    return [(key, ""), ("model." + key, "model.")]


def _candidate_file_keys(key: str) -> list[str]:
    """File-key spellings a patcher key may correspond to, likely first."""
    return [candidate for candidate, _model_prefix in _candidate_file_bindings(key)]


def _bytes_match(fd: int, ft: _FileTensor, tensor: torch.Tensor) -> bool:
    """Sampled byte equality between a file slice and a live tensor's value.

    Quant components load through dtype views, so byte identity, not dtype
    equality, decides a match.
    """
    if ft.nbytes != tensor.numel() * tensor.element_size():
        return False
    try:
        return all(
            os.pread(fd, length, ft.start + offset)
            == _tensor_bytes_window(tensor, offset, length)
            for offset, length in _fingerprint_windows(ft.nbytes)
        )
    except (OSError, RuntimeError):
        return False


def _cast_match(fd: int, ft: _FileTensor, tensor: torch.Tensor) -> bool:
    """Whether casting a file tensor reproduces the live tensor's bytes."""
    if ft.shape != tuple(tensor.shape) or ft.dtype == tensor.dtype:
        return False
    numel = tensor.numel()
    if numel == 0:
        return False
    file_itemsize = ft.nbytes // numel
    if file_itemsize * numel != ft.nbytes:
        return False
    live_itemsize = tensor.element_size()
    window_elements = max(
        1, min(_FINGERPRINT_WINDOW // max(file_itemsize, live_itemsize), numel)
    )
    middle = max(
        0, min(numel - window_elements, (numel - window_elements) // 2)
    )
    windows = sorted(
        {(0, window_elements), (middle, window_elements),
         (numel - window_elements, window_elements)}
    )
    try:
        for element, count in windows:
            raw = os.pread(
                fd, count * file_itemsize, ft.start + element * file_itemsize
            )
            if len(raw) != count * file_itemsize:
                return False
            cast = (
                torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                .view(ft.dtype)
                .to(tensor.dtype)
            )
            live = _tensor_bytes_window(
                tensor, element * live_itemsize, count * live_itemsize
            )
            if bytes(cast.view(torch.uint8).numpy().tobytes()) != live:
                return False
    except (OSError, RuntimeError):
        return False
    return True


def _quant_layer_conf(
    metadata: dict,
    header: dict[str, _FileTensor],
    file_layer: str,
    model_prefix: str | None,
    fd: int,
) -> tuple[dict | None, str | None]:
    """Return Comfy's layer config and its checkpoint authority.

    Embedded metadata wins for layers it names and suppresses legacy
    conversion, matching Comfy. A direct per-layer marker remains authoritative
    when valid embedded metadata omits that layer. Without embedded metadata,
    an applicable legacy scaled-FP8 conversion wins over the direct marker.
    """
    if "_quantization_metadata" in metadata:
        raw_metadata = metadata["_quantization_metadata"]
        if not isinstance(raw_metadata, str):
            raise _InvalidQuantConfError("quantization metadata is not text")
        try:
            metadata_bytes = raw_metadata.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise _InvalidQuantConfError(
                "quantization metadata is not valid Unicode"
            ) from exc
        decoded = _decode_quant_conf_bytes(
            metadata_bytes,
            bounded=False,
        )
        if decoded is None:
            raise _InvalidQuantConfError("quantization metadata is malformed")
        layers = decoded.get("layers")
        if not isinstance(layers, dict):
            raise _InvalidQuantConfError("quantization layer map is malformed")
        if file_layer in layers:
            conf = layers[file_layer]
            if not isinstance(conf, dict):
                raise _InvalidQuantConfError(
                    "quantization layer config is malformed"
                )
            return conf, "metadata"
    else:
        # Comfy first converts with an empty prefix, before detecting/stripping
        # a checkpoint namespace. A root marker therefore governs prefixed
        # layers.
        legacy_marker = header.get("scaled_fp8")
        if legacy_marker is None and model_prefix is not None:
            legacy_marker = header.get(f"{model_prefix}scaled_fp8")
        if (
            legacy_marker is not None
            and f"{file_layer}.scale_weight" in header
        ):
            if legacy_marker.dtype not in {
                torch.float8_e4m3fn,
                torch.float32,
            }:
                raise _InvalidQuantConfError(
                    "legacy quant marker dtype is invalid"
                )
            marker_numel = math.prod(legacy_marker.shape)
            # Comfy treats marker presence as the legacy-format authority and
            # uses only cardinality two to enable full-precision matmul.  Its
            # published Wan FP8 repacks use an empty sentinel for the ordinary
            # (non-full-precision) case.
            if marker_numel not in {0, 1, 2}:
                raise _InvalidQuantConfError(
                    "legacy quant marker shape is invalid"
                )
            legacy_conf: dict[str, Any] = {"format": "float8_e4m3fn"}
            if marker_numel == 2:
                legacy_conf["full_precision_matrix_mult"] = True
            return legacy_conf, "legacy"

    marker = header.get(f"{file_layer}.comfy_quant")
    if marker is None:
        return None, None
    if (
        marker.dtype != torch.uint8
        or len(marker.shape) != 1
        or marker.nbytes != math.prod(marker.shape)
        or not 0 < marker.nbytes <= _QUANT_CONF_MAX_BYTES
    ):
        raise _InvalidQuantConfError(
            "direct quant marker descriptor is invalid"
        )
    try:
        raw = os.pread(fd, marker.nbytes, marker.start)
    except OSError as exc:
        raise _InvalidQuantConfError("direct quant marker read failed") from exc
    if len(raw) != marker.nbytes:
        raise _InvalidQuantConfError("direct quant marker read was short")
    conf = _decode_quant_conf_bytes(raw)
    if conf is None:
        raise _InvalidQuantConfError(
            "direct quant marker payload is malformed"
        )
    return conf, "direct"


def _decode_quant_conf_bytes(
    raw: bytes,
    *,
    bounded: bool = True,
) -> dict | None:
    if not raw or (bounded and len(raw) > _QUANT_CONF_MAX_BYTES):
        return None

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError("duplicate JSON key")
            decoded[key] = value
        return decoded

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON constant")

    def valid_value(value: Any, depth: int = 0) -> bool:
        if depth > 64:
            return False
        if value is None or isinstance(value, (str, bool, int)):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, list):
            return all(valid_value(item, depth + 1) for item in value)
        if isinstance(value, dict):
            return all(
                isinstance(key, str) and valid_value(item, depth + 1)
                for key, item in value.items()
            )
        return False

    try:
        text = raw.decode("utf-8", "strict")
        decoded = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
        return None
    return decoded if isinstance(decoded, dict) and valid_value(decoded) else None


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
