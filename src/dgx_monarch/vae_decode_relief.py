"""Before a VAE decode that ComfyUI prices above its free memory, the driver gives back what is idle.

ComfyUI prices a VAE decode and asks its loader for that much room
(``VAE.decode``, ``load_models_gpu``). With DynamicVRAM on, every model in the
driver is dynamic, ``free_memory`` unloads none of them for another dynamic
model, the loader logs ``0 models unloaded.``, and the decode runs whole
anyway: its batch is at least one item. On a discrete GPU the allocator then
fails at the card's limit and ComfyUI falls back to tiles. On a GPU that shares
the host's memory nothing fails until the kernel does.

Two records from 2026-11-12, both one-GPU video renders on a DGX Spark. A
decode priced at 16.5 GiB with its reserve started with 6.7 GiB read as free,
used 9.3, and took the box to 6.7 GiB available while the driver held 32 GiB
of text encoders no later node needed. A decode priced at 5.6 GiB started with
4.9 read as free and took the box from 12.5 to 7.8.

So on such a GPU, when ComfyUI is about to run a decode whose price and
reserve exceed what it reads as free, this process first does what ComfyUI's
classic loader would have done there. It unloads the models ComfyUI holds in
this process (ComfyUI's own ``/free`` sequence, ``driver_unload.unload``). If
the decode is still priced above the reading and a mesh is live and idle, it
asks the Workers to release their cached allocator blocks
(``driver_unload.trim_actors``, the Worker's ``clear_vram`` at ``soft``, which
unloads no model). Then ComfyUI's decode runs with the caller's own arguments.

Nothing here computes anything that reaches the result. A decode of one item
(every video, every single image) takes a batch of one before and after. A
decode of several images takes ``free / price`` images a batch, as it always
did; the relief only raises the free figure of a tight box towards a roomy
one's. Unloading changes no weight: ComfyUI loads a model again through its
own loader when a node next asks for it.

It never acts while a render this driver started is still in flight
(``driver_unload.busy_reason``): a pipelined render decodes one result while
later ones sample, and their models and the Workers' caches are in use.

What it costs: two calls on every decode (a price and a reading). When it
acts, the text encoder loads again when a prompt's text next changes, a stock
diffusion model the driver held loads again at its next render, and the next
render's first step re-grows the Workers' cache if they were asked.

It adds room. It does not bound a decode: with nothing idle the decode runs as
ComfyUI runs it, and one line says so.

Scope. On by default at node import where the driver's reading is: Linux, with
the driver's CUDA device integrated. ``DGXM_DISABLE_VAE_DECODE_RELIEF=1`` on
the driver turns it off, for diagnosis; no user needs it. ``VAE.decode`` only:
``VAE.decode_tiled`` is already the bounded form, and ``VAE.encode`` runs one
image a call (``vae_encode_item``).

It steps aside, once and for the life of the process, if anything it calls
raises: ComfyUI's own decode answers then and from then on, one WARNING says
why, and ``/dgxm/telemetry`` shows it. The decode in hand still runs.
"""
from __future__ import annotations

import functools
import os
import sys
import threading
import weakref
from collections.abc import Callable
from typing import Any

from . import driver_uma_memory
from .log import get_logger

log = get_logger(__name__)

DISABLE_ENV = "DGXM_DISABLE_VAE_DECODE_RELIEF"

# What this process did, for ``/dgxm/telemetry`` (``driver_rules``) and doctor's row.
INSTALLED = "installed"
OFF = "off"
STEPPED_ASIDE = "stepped_aside"
NOT_LINUX = "not_linux"
NOT_INTEGRATED = "not_integrated"
NO_COMFY = "no_comfy"
PROBE_FAILED = "probe_failed"
UNSET = "unset"

_KEY = "_dgxm_decode_relief"
_GIB = 2 ** 30
_INSTALL_LOCK = threading.Lock()
_STATE = {"state": UNSET}
# The mesh the last synchronous render of this process used, held weakly: the
# relief must never keep a retired mesh alive.
_MESH: dict[str, Any] = {"ref": None}


def state() -> str:
    """What the rule is in this process: one of the names above."""
    return _STATE["state"]


def _set(value: str) -> str:
    _STATE["state"] = value
    return value


def note_mesh(handle: Any) -> None:
    """Remember the mesh a render just used, so a decode after it can ask its Workers.

    Called where a synchronous render has returned its result
    (``nodes/common.run_render``). A handle that cannot be referenced weakly
    is not kept, and the relief then does its own half only. Never raises.
    """
    try:
        _MESH["ref"] = None if handle is None else weakref.ref(handle)
    except Exception:
        _MESH["ref"] = None


def _mesh() -> Any:
    ref = _MESH["ref"]
    return None if ref is None else ref()


def _gib(value: int) -> str:
    return f"{value / _GIB:.1f}"


def decode_price(vae: Any, samples: Any) -> int:
    """ComfyUI's price for this decode, in bytes: ``VAE.memory_used_decode`` on the shape ``VAE.decode`` prices.

    Called, not copied, so a ComfyUI that changes a VAE's price is followed.
    The one line mirrored is the first of ``VAE.decode``: a 2-D VAE given a
    5-D latent drops the frame axis before it prices it.
    """
    shape = tuple(int(size) for size in samples.shape)
    if getattr(vae, "latent_dim", None) == 2 and len(shape) == 5:
        shape = shape[:2] + shape[3:]
    return int(vae.memory_used_decode(shape, vae.vae_dtype))


def loader_need(mm: Any, price: int) -> int:
    """What ``load_models_gpu`` requires free for a call priced at ``price``: its ``minimum_memory_required``."""
    return int(max(mm.minimum_inference_memory(), price + mm.extra_reserved_memory()))


def decode_need(mm: Any, vae: Any, samples: Any) -> tuple[int, int]:
    """``(price, need)`` in bytes for this decode: ComfyUI's price, and what its loader will require."""
    price = decode_price(vae, samples)
    return price, loader_need(mm, price)


def relieve(mm: Any, vae: Any, samples: Any, told: set[str]) -> bool:
    """Give back what is idle if this decode is priced above the reading; True when it acted.

    ``told`` holds what this process has already said about decodes it could
    not help, so a tight session logs each reason once.
    """
    try:
        price = decode_price(vae, samples)
    except Exception as exc:
        # A VAE that cannot price this latent (one with no generic decode, a latent it does not
        # take) fails the same call inside ComfyUI's own decode, which answers for it. That is
        # this decode's matter and no reason to stand down for the process.
        if "unpriced" not in told:
            told.add("unpriced")
            log.info("decode relief: ComfyUI's price for a %s decode could not be read (%s: %s), so that "
                     "decode runs as ComfyUI runs it", type(vae).__name__, type(exc).__name__, exc)
        return False
    need = loader_need(mm, price)
    device = vae.device
    reading = int(mm.get_free_memory(device))
    if reading >= need:
        return False
    from . import driver_unload

    priced = (f"ComfyUI priced this VAE decode at {_gib(price)} GiB ({_gib(need)} GiB with its "
              f"reserve) against {_gib(reading)} GiB free")
    handle = _mesh()
    why = driver_unload.busy_reason(handle)
    if why:
        if why not in told:
            told.add(why)
            log.info("decode relief: %s; the driver kept what it holds because %s", priced, why)
        return False
    held = driver_unload.held_models(mm)
    # The VAE in hand is not idle. With nothing else held, unloading would only re-stage it.
    own = getattr(vae, "patcher", None)
    idle = [pair for pair in held if own is None or getattr(pair[0], "model", None) is not own]
    unloaded = driver_unload.unload(mm, held) if idle else None
    after = int(mm.get_free_memory(device))
    trimmed = None
    trim_failed = ""
    if after < need and driver_unload.trimmable(handle):
        try:
            trimmed = driver_unload.trim_actors(handle)
        except Exception as exc:
            # A mesh another session owns, or a call that did not answer: a skip, not a fault.
            trim_failed = f"{type(exc).__name__}: {exc}"
        after = int(mm.get_free_memory(device))
    if unloaded is None and trimmed is None:
        nothing = "nothing idle" if not trim_failed else f"trim failed: {trim_failed}"
        if nothing not in told:
            told.add(nothing)
            log.info("decode relief: %s; ComfyUI held no idle model here to unload%s, so the decode runs as "
                     "ComfyUI runs it", priced,
                     f" and the Workers' cached memory was not released ({trim_failed})"
                     if trim_failed else " and no idle mesh to ask")
        return False
    steps = []
    if unloaded is not None:
        steps.append(unloaded.readings())
    if trimmed is not None:
        steps.append(f"asked {trimmed.actors} actor{'' if trimmed.actors == 1 else 's'} to release "
                     f"cached memory: MemAvailable read {_gib(trimmed.before)} GiB before that and "
                     f"{_gib(trimmed.after)} GiB after it")
    elif trim_failed:
        steps.append(f"could not ask the Workers to release cached memory ({trim_failed})")
    tail = (f"the decode starts with {_gib(after)} GiB free" if after >= need else
            f"the decode is still priced above the {_gib(after)} GiB free and runs as ComfyUI runs it")
    log.info("decode relief: %s, so the driver %s; %s", priced, "; then ".join(steps), tail)
    return True


def with_relief(original: Callable[..., Any]) -> Callable[..., Any]:
    """``original`` (``VAE.decode``) with the relief run before it."""
    aside: list[str] = []
    told: set[str] = set()

    @functools.wraps(original)
    def decode_after_relief(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not aside:
            samples = args[0] if args else kwargs.get("samples_in")
            mm = sys.modules.get("comfy.model_management")
            if samples is not None and mm is not None:
                try:
                    relieve(mm, self, samples, told)
                except Exception as exc:
                    aside.append(f"{type(exc).__name__}: {exc}")
                    _set(STEPPED_ASIDE)
                    log.warning("decode relief stepped aside (%s); ComfyUI's own decode runs from here on "
                                "in this process, with no idle memory given back before it", aside[0])
        # Stock's own call with the caller's own arguments, in every path.
        return original(self, *args, **kwargs)

    setattr(decode_after_relief, _KEY, True)
    return decode_after_relief


def install_driver() -> str:
    """The ComfyUI process's hook, at node import. Never raises.

    Returns, and keeps for ``state``, what it did: ``installed``, ``off`` (the
    environment switch), ``not_linux``, ``not_integrated`` (a discrete GPU or
    the CPU), ``no_comfy`` or ``probe_failed``. The gate is the reading's
    (``driver_uma_memory``), probed here so that the reading's own switch does
    not turn this rule off.
    """
    if os.environ.get(DISABLE_ENV) == "1":
        log.info("%s=1: ComfyUI's own VAE decode runs in this process with no idle memory given back "
                 "before it", DISABLE_ENV)
        return _set(OFF)
    if not sys.platform.startswith("linux"):
        return _set(NOT_LINUX)
    try:
        model_management, torch = driver_uma_memory.comfy_modules()
    except Exception:
        return _set(NO_COMFY)
    try:
        total = driver_uma_memory.integrated_total_bytes(torch, model_management.get_torch_device())
    except Exception as exc:
        log.warning("could not tell whether this GPU shares the host's memory (%s: %s); ComfyUI's own "
                    "VAE decode stays", type(exc).__name__, exc)
        return _set(PROBE_FAILED)
    if total is None:
        return _set(NOT_INTEGRATED)
    try:
        install()
    except Exception as exc:
        log.warning("the decode relief could not be installed, so ComfyUI's own VAE decode stays "
                    "(%s: %s)", type(exc).__name__, exc)
        return _set(NO_COMFY)
    log.info("unified memory: before a VAE decode that ComfyUI prices above its free memory, this process "
             "gives back the models it holds idle and the Workers' cached memory (%s=1 turns this off)",
             DISABLE_ENV)
    return _set(INSTALLED)


def _vae_class() -> Any:
    """ComfyUI's ``VAE`` class, only if this process already imported ``comfy.sd``.

    It imports nothing itself: importing ComfyUI here would pick its device
    before the process that owns it does.
    """
    module = sys.modules.get("comfy.sd")
    if module is None:
        raise ImportError("comfy.sd is not imported in this process")
    return module.VAE


def install() -> None:
    """Wrap ``comfy.sd.VAE.decode`` once. ``VAE.decode_tiled`` and ``VAE.encode`` are left as they are."""
    vae = _vae_class()
    with _INSTALL_LOCK:
        current = vae.decode
        if not getattr(current, _KEY, False):
            vae.decode = with_relief(current)


def uninstall() -> None:
    """Undo ``install`` (tests)."""
    vae = _vae_class()
    with _INSTALL_LOCK:
        current = vae.decode
        if getattr(current, _KEY, False):
            vae.decode = current.__wrapped__
    _set(UNSET)
    _MESH["ref"] = None


__all__ = ["DISABLE_ENV", "INSTALLED", "NOT_INTEGRATED", "NOT_LINUX", "NO_COMFY", "OFF", "PROBE_FAILED",
           "STEPPED_ASIDE", "UNSET", "decode_need", "decode_price", "install", "install_driver",
           "loader_need", "note_mesh", "relieve", "state", "uninstall", "with_relief"]
