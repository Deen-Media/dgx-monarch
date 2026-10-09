"""Preflight ComfyUI's sample-time memory estimate on the driver.

Header-only model detection supplies ``memory_usage_factor`` to the stock
flash-path ``BaseModel.memory_required`` formula. Using two-byte inference
dtype and omitting conditioning extras yields a lower bound. Refuse a
complete over-budget estimate before workers load weights; missing or broken
estimation makes no claim.

The check uses driver-host MemAvailable at every world size. It catches
sample-time overcommit that a successful weight load cannot exclude,
including single-rank renders with no cross-rank divergence guard. See
docs/VALIDATION.md, 2026-08-27, for the measured partial-load failure.
"""
from __future__ import annotations

import json
import os
import struct
from functools import lru_cache
from typing import Any

from .log import get_logger
from .mesh_safety import (
    ACTIVATION_PREFLIGHT_DISABLE_ENV,
    env_enabled,
    gpu_is_integrated,
    mem_available_bytes,
)
from .refusal import RefusalClass, refusal

log = get_logger(__name__)

_MIN_DTYPE_BYTES = 2  # bf16/fp16: the smallest inference dtype comfy selects
_MB = 1024 * 1024
_GUARD = "activation_footprint_preflight"


class RenderMemoryPriceError(RuntimeError):
    """Typed capacity refusal: comfy's own sample-time estimate exceeds the box.

    Never a ``StockLoadCapacityError``: that family routes to the residency
    rescue ladder, to the ceremony's "cross-residency reference cannot load
    under stock residency" verdict (``nodes/gate_cross_mode.py``) and to its
    capacity-stop conversion (``nodes/gate_ceremony.py``), and all three are
    false here. The weights load; what does not fit is ComfyUI's sample-time
    estimate for the latent size asked for (batch, frames and resolution), so a
    ledger row saying stock residency cannot load the checkpoint would mislead.
    """


def _meta_state_dict(path: str) -> dict[str, Any]:
    """Build a shapes-only meta state dict from the safetensors header."""
    import torch

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    out: dict[str, Any] = {}
    for key, spec in header.items():
        if key == "__metadata__" or not isinstance(spec, dict):
            continue
        shape = spec.get("shape")
        if isinstance(shape, list):
            out[key] = torch.empty(shape, device="meta")
    return out


def detect_comfy_model_config(path: str) -> Any:
    """The model config comfy's own detection names for this checkpoint header.

    Shapes only: no tensor is read. Raises when comfy is not importable or the
    header cannot be read, and returns ``None`` when comfy names no config.
    The render-memory price below and ``latent_scale`` (the auto-topology
    latent ratio and the rendered latent shape) need the same answer, so they
    share one detection, cached by path, mtime, ctime and size, the identity
    the header sniff uses (adapters/detect.py).
    """
    stat = os.stat(path)
    return _detect_by_identity(
        str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


@lru_cache(maxsize=64)
def _detect_by_identity(
    path: str, _mtime_ns: int, _ctime_ns: int, _size: int
) -> Any:
    # Exceptions escape uncached, so a transient read failure is retried; a
    # header comfy names no config for is cached as None like any answer.
    return _detect_uncached(path)


def _detect_uncached(path: str) -> Any:
    import comfy.model_detection as model_detection
    import comfy.utils

    sd = _meta_state_dict(path)
    # These steps follow comfy.sd.load_diffusion_model_state_dict's detection
    # and fallbacks, because callers must model what the loader will build: for
    # a drifted header that can be a different class than the artifact intends
    # (measured 2026-08-27: an l2p pixel-space file that comfy detects as latent
    # ZImage, factor 2.8). They do not match comfy exactly: the file's metadata
    # never reaches them, and the quant conversion and fallbacks differ.
    sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=None)
    prefix = model_detection.unet_prefix_from_state_dict(sd)
    stripped = comfy.utils.state_dict_prefix_replace(
        sd, {prefix: ""}, filter_keys=True)
    if stripped:
        sd = stripped
    config = model_detection.model_config_from_unet(sd, "", metadata=metadata)
    if config is None:
        converted = model_detection.convert_diffusers_mmdit(sd, "")
        if converted is not None:
            config = model_detection.model_config_from_unet(converted, "")
    if config is None:
        config = model_detection.model_config_from_diffusers_unet(sd)
    return config


def comfy_render_memory_bound(path: str, latent_shape: Any) -> tuple[int, dict] | None:
    """Comfy's flash-path render-memory estimate, lower-bounded, in bytes.

    Mirrors ``BaseModel.memory_required`` for the flash/xformers branch:
    ``area * dtype_size * 0.01 * memory_usage_factor * 1MB`` on the undoubled
    batch, evaluated on the config comfy's own detection names for this
    header. Returns ``None`` when comfy or the detection is unavailable.
    """
    try:
        shape = [int(s) for s in latent_shape]
        if len(shape) < 3 or any(s < 1 for s in shape):
            return None
        config = detect_comfy_model_config(path)
        if config is None:
            return None
        factor = float(getattr(config, "memory_usage_factor", 0.0))
        if factor <= 0:
            return None
        # The single-batch shape: comfy's placement decision compares free
        # memory against minimum_memory_required (sampler_helpers passes the
        # undoubled batch there), so the price mirrors that quantity; the
        # doubled cond+uncond estimate only sizes the optional headroom.
        area = shape[0]
        for s in shape[2:]:
            area *= s
        bound = int(area * _MIN_DTYPE_BYTES * 0.01 * factor * _MB)
        return bound, {
            "config": type(config).__name__, "memory_usage_factor": factor,
            "area": area, "dtype_bytes": _MIN_DTYPE_BYTES,
        }
    except Exception as exc:
        log.debug("render memory bound unavailable for %s: %r", path, exc)
        return None


def _floors_bytes() -> int:
    """Comfy's reserve-inclusive inference floor, or a fixed floor."""
    try:
        import comfy.model_management as mm

        # Comfy's minimum_inference_memory already includes
        # extra_reserved_memory; adding it again doubles --reserve-vram.
        return int(mm.minimum_inference_memory())
    except Exception:
        return 9 * 2**30


def refuse_render_memory(
    *, unet_name: str, bound: int, detail: dict, floors: int, avail: int,
) -> None:
    """Raise the typed capacity refusal for an over-box render estimate."""
    raise RenderMemoryPriceError(refusal(
        RefusalClass.CAPACITY,
        f"render memory preflight refuses {unet_name}: ComfyUI's own sample-"
        f"time estimate needs at least {bound / 2**30:.1f} GiB for this "
        f"latent size (model config {detail.get('config')}, "
        f"memory_usage_factor {detail.get('memory_usage_factor')}, smallest "
        f"inference dtype) plus {floors / 2**30:.1f} GiB of inference floor "
        f"and reserve, against {avail / 2**30:.1f} GiB available. At sample "
        "time ComfyUI would offload every weight, then stream and cast every "
        "block on every step. On 2026-09-02 a single-box render whose estimate did "
        "not fit filled the host's memory with no OOM kill. Across ranks the partial loads "
        "also diverge, and the cross-rank divergence guard refuses only after "
        "every rank has loaded (issue #339). Reduce the resolution, frame count "
        "or batch size, use a smaller or lower-precision checkpoint, or set "
        f"{ACTIVATION_PREFLIGHT_DISABLE_ENV}=1 to bypass this preflight "
        "(docs/TROUBLESHOOTING.md #79).",
        guard=_GUARD, waivable=False))


def preflight_render_memory(
    path: str, unet_name: str, latent_shape: Any,
) -> None:
    """Refuse before dispatch when comfy's estimate cannot fit the host.

    Ignores the world size on purpose: the estimate is per box, the price reads
    the driver host's own MemAvailable at every world size, and a single-box
    render has no cross-rank guard behind it.
    """
    try:
        if not gpu_is_integrated() or env_enabled(ACTIVATION_PREFLIGHT_DISABLE_ENV):
            return
        found = comfy_render_memory_bound(path, latent_shape)
        if found is None:
            return
        bound, detail = found
        avail = mem_available_bytes()
        if avail is None:
            return
        floors = _floors_bytes()
        if bound + floors <= avail:
            return
        refuse_render_memory(unet_name=unet_name, bound=bound, detail=detail,
                             floors=floors, avail=avail)
    except RenderMemoryPriceError:
        raise
    except Exception as exc:
        log.debug("render memory preflight failed open: %r", exc)
        return
