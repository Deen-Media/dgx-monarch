"""ComfyUI in the driver process reads free memory as the box has it.

On a unified-memory GPU (GB10 in a DGX Spark) the CPU and the GPU share one
pool. ComfyUI's ``model_management.get_free_memory`` asks
``torch.cuda.mem_get_info``, which there follows the kernel's MemFree: it
falls to a few GiB once the page cache is full,
though tens of GiB are available. NVIDIA's note on DGX Spark says the call
does not count memory the kernel can reclaim and points at MemAvailable.

A process that reads this figure has three readers:

- comfy-aimdo's budget, which places every weight of a DynamicVRAM model. On
  an integrated Linux device it already reads MemAvailable from /proc/meminfo
  (its ``control.c``), against a floor of a sixteenth of the device, clamped
  to 2 and 8 GiB.
- ``get_free_memory`` itself: how many images a VAE takes in a batch, whether
  SeedVR2's VAE decodes tiled, whether stock sampling runs cond and uncond in one
  forward, when the allocator cache is flushed, the free figure the UI shows.
- ``comfy.system_memory.virtual_memory_available``: MemAvailable already.

So with DynamicVRAM on the driver's weights were never misplaced; its batch choices
swing with the page cache, and with them the bits of a stock VAE encode
(the UltraTex conditioning of 2026-11-11).

The actors correct the same function for their own reasons
(``actor/comfy_bridge._patch_uma_free_memory``: psutil's available memory over
the actors on the host). This is the driver's correction. It differs in one
thing: nothing prices a VAE batch before it runs, so the reading keeps the
floor aimdo keeps in the same process. Every other device, and the
allocator's own cached bytes, come from ComfyUI's function, so a ComfyUI that
changes those is followed.

``DGXM_DISABLE_DRIVER_UMA_MEMORY=1`` leaves ComfyUI's reading alone.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

from .log import get_logger

log = get_logger(__name__)

_GIB = 2 ** 30

DISABLE_ENV = "DGXM_DISABLE_DRIVER_UMA_MEMORY"

# comfy-aimdo's rule for an integrated Linux GPU (control.c,
# calculate_integrated_ram_headroom): a sixteenth of the device, never under
# 2 GiB or over 8. One process, one floor.
FLOOR_DIVISOR = 16
FLOOR_MIN_BYTES = 2 * _GIB
FLOOR_MAX_BYTES = 8 * _GIB

# This module's marker on the function it installs. The actor's patch carries
# ACTOR_MARK and replaces whatever it finds, so an actor that somehow starts
# with this reading in place still gets its own.
DRIVER_MARK = "_dgxm_uma_driver"
ACTOR_MARK = "_dgxm_uma"

Reading = Callable[..., Any]


def floor_bytes(total_bytes: int) -> int:
    """What stays unclaimed on an integrated device of ``total_bytes``."""
    return min(FLOOR_MAX_BYTES, max(FLOOR_MIN_BYTES, int(total_bytes) // FLOOR_DIVISOR))


def device_free_bytes(available_bytes: int, total_bytes: int) -> int:
    """Free device memory on unified memory: MemAvailable less the floor, never negative."""
    return max(0, int(available_bytes) - floor_bytes(total_bytes))


def host_available_bytes() -> int:
    """MemAvailable, through ComfyUI's own reader where it has one (it knows cgroup limits)."""
    reader = getattr(sys.modules.get("comfy.system_memory"), "virtual_memory_available", None)
    if callable(reader):
        try:
            return int(reader())
        except Exception:
            pass
    import psutil

    return int(psutil.virtual_memory().available)


def integrated_total_bytes(torch: Any, device: Any) -> int | None:
    """Total memory of an integrated CUDA device; ``None`` for any other device."""
    if getattr(device, "type", None) != "cuda":
        return None
    properties = torch.cuda.get_device_properties(device)
    if not getattr(properties, "is_integrated", 0):
        return None
    return int(properties.total_memory)


def corrected_reading(original: Reading, total_of: Callable[[Any], int | None],
                      default_device: Callable[[], Any],
                      available: Callable[[], int] = host_available_bytes) -> Reading:
    """``get_free_memory`` with the device-free term taken from MemAvailable.

    ``total_of(device)`` is the device's total on unified memory and ``None``
    elsewhere; a device it answers ``None`` for gets ``original`` untouched.
    The allocator's cached-free bytes are ``original``'s, added as stock adds
    them. If ``total_of``, ``available`` or ``original`` raises, or
    ``original`` answers in a shape this wrapper does not know, that call and
    every later one return ``original``'s answer, and one WARNING is logged.
    An argument this wrapper does not know goes to ``original`` unread.
    """
    totals: dict[str, int | None] = {}
    failed: list[bool] = []

    def free_memory(dev: Any = None, torch_free_too: bool = False, *args: Any, **kwargs: Any) -> Any:
        # ComfyUI calls this at every batch and load choice of the user's graph.
        # A failure of the correction must cost the correction, never the render,
        # and an argument or answer this wrapper does not know is upstream's own.
        if args or kwargs or failed:
            return original(dev, torch_free_too, *args, **kwargs)
        try:
            device = default_device() if dev is None else dev
            name = str(device)
            if name not in totals:
                totals[name] = total_of(device)
            total = totals[name]
            if total is not None:
                stock = original(device, True)
                if not (isinstance(stock, tuple) and len(stock) == 2):
                    raise TypeError(f"ComfyUI's reading answered {stock!r} for torch_free_too")
                cached_free = stock[1]
                free = device_free_bytes(available(), total) + cached_free
        except Exception as exc:
            failed.append(True)
            log.warning("the unified-memory reading failed (%s: %s); ComfyUI's own free-memory "
                        "reading answers from here on", type(exc).__name__, exc)
            return original(dev, torch_free_too)
        if total is None:
            return original(dev, torch_free_too)
        return (free, cached_free) if torch_free_too else free

    setattr(free_memory, DRIVER_MARK, True)
    free_memory.__wrapped__ = original  # type: ignore[attr-defined]
    return free_memory


def _comfy() -> tuple[Any, Any]:
    """ComfyUI's model management and torch, only if this process already imported them.

    It imports nothing itself: importing ComfyUI's model management here would
    pick its device and read its flags before the process that owns them does.
    The ComfyUI process has imported both long before it loads a node pack.
    """
    model_management = sys.modules.get("comfy.model_management")
    torch = sys.modules.get("torch")
    if model_management is None or torch is None:
        raise ImportError("comfy.model_management is not imported in this process")
    return model_management, torch


def install_driver() -> str:
    """Install the reading in this ComfyUI process, at node import. Never raises.

    Returns what it did: ``installed``, ``off`` (the environment switch),
    ``not_linux``, ``not_integrated`` (a discrete GPU or the CPU), ``already``
    (this reading or an actor's is in place), ``no_comfy`` or ``probe_failed``.
    """
    if os.environ.get(DISABLE_ENV) == "1":
        log.info("%s=1: ComfyUI's own free-memory reading stays in this process", DISABLE_ENV)
        return "off"
    if not sys.platform.startswith("linux"):
        return "not_linux"
    try:
        mm, torch = _comfy()
    except Exception:
        return "no_comfy"
    try:
        current = mm.get_free_memory
        if getattr(current, DRIVER_MARK, False) or getattr(current, ACTOR_MARK, False):
            return "already"
        total = integrated_total_bytes(torch, mm.get_torch_device())
    except Exception as exc:
        log.warning("could not tell whether this GPU shares the host's memory (%s: %s); ComfyUI's own "
                    "free-memory reading stays", type(exc).__name__, exc)
        return "probe_failed"
    if total is None:
        return "not_integrated"
    mm.get_free_memory = corrected_reading(
        current, lambda device: integrated_total_bytes(torch, device), lambda: mm.get_torch_device(),
        host_available_bytes)
    log.info("unified memory: ComfyUI in this process now reads free memory as MemAvailable less a "
             "%.1f GiB floor, not the device figure that follows MemFree (%s=1 turns this off)",
             floor_bytes(total) / _GIB, DISABLE_ENV)
    return "installed"


__all__ = ["ACTOR_MARK", "DISABLE_ENV", "DRIVER_MARK", "corrected_reading", "device_free_bytes",
           "floor_bytes", "host_available_bytes", "install_driver", "integrated_total_bytes"]
