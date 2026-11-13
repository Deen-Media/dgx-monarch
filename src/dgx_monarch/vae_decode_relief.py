"""Before a VAE decode that ComfyUI prices above its free memory, the driver frees what the decode needs.

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

So on such a GPU, with DynamicVRAM on, when ComfyUI is about to run a decode
whose price and reserve exceed what it reads as free, this process first does
what ComfyUI's classic loader does there: it calls ComfyUI's own
``free_memory`` for that requirement, with ``for_dynamic=False``, keeping
every model loaded except the text encoders. ComfyUI then frees just what the
call needs from the text encoders it holds, in its own order, and a
DynamicVRAM model gives up pages without being unloaded where that is enough.
Then ComfyUI's decode runs with the caller's own arguments.

Only text encoders are freed, by an allow-list and not a deny-list: a text
encoder is the one kind of model ComfyUI marks for certain (``is_clip`` on its
patcher, set in ``comfy.sd.CLIP``), nothing samples with one, and every gain
on record is one (24 and 7.7 GiB in the first record above, 6.3 in the
second). A diffusion model, the VAE in hand, another VAE, CLIP vision, a
ControlNet, an adapter or a model patch is never touched, whoever calls and
from whatever sampler.

Nothing here computes anything that reaches the result. A decode of one item
(every video, every single image) takes a batch of one before and after. A
decode of several images takes ``free / price`` images a batch, as it always
did, and that free figure already counted what DynamicVRAM could evict
(``ModelPatcher.get_free_memory``), so the count barely moves. Freeing changes
no weight: ComfyUI faults a page back, or loads a model again, with the same
bytes when a node next asks for it.

When it keeps everything:

- **While a sampler runs.** Preview nodes call ``VAE.decode`` at every step.
  Freeing there would take memory from under a running sampler, so
  ``CFGGuider.sample`` is wrapped with a process-wide depth count and the
  relief does nothing while it is above zero, whichever thread the decode
  comes from. A sampler that builds its own loop and never reaches
  ``CFGGuider.sample`` is not counted; for that one the allow-list above is
  the whole protection, and it is enough: such a sampler does not sample with
  a text encoder either.
- **With ``DGXM_DISABLE_DRIVER_UNLOAD=1``**, whose meaning is that the driver
  unloads none of ComfyUI's models on its own.
- **With DynamicVRAM off.** ComfyUI's classic loader already unloads for the
  decode, and unloading a classic model on unified memory copies it first.

It does not ask the pack's shared busy check (``driver_unload.busy_reason``).
That check guards the render-site unload, which drops every model and calls
the Workers; on 2026-11-12 it read "a call to the Workers is still in flight"
after a render had finished. This rule's own tests of "in use" are narrower
and enough for what it frees: the sampler guard, and the one kind of model it
frees at all. A render on the Workers uses no model in this process.

What it costs: two calls on every decode (a price and a reading). When it
acts, a text encoder that gave up pages faults them back, or loads again, when
a prompt's text next changes.

It adds room. It does not bound a decode: with nothing idle the decode runs as
ComfyUI runs it, and one line says so. The Workers are not asked for their
cached memory here: the one Worker call there is reaches every box, and only
the driver host's Workers share the decode's memory.

Scope. On by default at node import where the driver's reading is: Linux, with
the driver's CUDA device integrated. ``DGXM_DISABLE_VAE_DECODE_RELIEF=1`` on
the driver turns it off, for diagnosis; no user needs it. ``VAE.decode`` only:
``VAE.decode_tiled`` is already the bounded form, and ``VAE.encode`` runs one
image a call (``vae_encode_item``). The trigger leaves out the VAE's own
weights, which DynamicVRAM places against its own budget.

It steps aside, once and for the life of the process, if anything it calls
raises: ComfyUI's own decode answers then and from then on, one WARNING says
why, and ``/dgxm/telemetry`` shows it. The decode in hand still runs. A VAE
that cannot price its latent is that decode's matter alone.
"""
from __future__ import annotations

import functools
import gc
import os
import sys
import threading
import time
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
_GUARD_KEY = "_dgxm_sampler_guard"
_GIB = 2 ** 30
# ComfyUI waits this long after a DynamicVRAM decommit before it reads free memory again.
_SETTLE_S = 0.05
_INSTALL_LOCK = threading.Lock()
_STATE = {"state": UNSET}
# How many of ComfyUI's samplers are running in this process. A decode while one runs is a
# preview, whichever thread makes it. ComfyUI runs one prompt at a time, so a process-wide
# count loses nothing against a per-thread one.
_SAMPLING_LOCK = threading.Lock()
_SAMPLING = {"depth": 0}
# The fleet this driver last submitted a render to, held weakly: never what keeps a mesh alive, and
# never used as it is (``serving_now`` reads the fleet attached now in its place).
_SERVING: dict[str, weakref.ref[Any] | None] = {"handle": None}  # proc lifetime: replaced at every submit

# How much ComfyUI is asked to free. False: the decode's own requirement and no more, so the
# driver holds its floor through the decode with the least taken (decided 2026-11-12). True:
# every text encoder whole, ComfyUI's own "all of it" figure. That is the decided fallback if hardware
# shows a page free returns too little, and this line is the whole change.
FREE_ALL_IDLE = False
_EVERYTHING = 1e30

SAMPLER_RUNNING = "a sampler is running, and a decode made while it runs must not free memory under it"
UNLOAD_SWITCH = "the driver runs with DGXM_DISABLE_DRIVER_UNLOAD=1"


def state() -> str:
    """What the rule is in this process: one of the names above."""
    return _STATE["state"]


def _set(value: str) -> str:
    _STATE["state"] = value
    return value


def _gib(value: int) -> str:
    return f"{value / _GIB:.1f}"


def sampling_depth() -> int:
    """How many of ComfyUI's samplers are running in this process."""
    return int(_SAMPLING["depth"])


def with_sampler_guard(original: Callable[..., Any]) -> Callable[..., Any]:
    """``original`` (``CFGGuider.sample``) counted in and out, so a decode while it runs is known as one."""

    @functools.wraps(original)
    def counted_sample(self: Any, *args: Any, **kwargs: Any) -> Any:
        with _SAMPLING_LOCK:
            _SAMPLING["depth"] += 1
        try:
            return original(self, *args, **kwargs)
        finally:
            with _SAMPLING_LOCK:
                _SAMPLING["depth"] -= 1

    setattr(counted_sample, _GUARD_KEY, True)
    return counted_sample


def note_serving(handle: Any) -> None:
    """Remember, weakly, the fleet a render was just submitted to. Never raises."""
    try:
        _SERVING["handle"] = None if handle is None else weakref.ref(handle)
    except TypeError:
        _SERVING["handle"] = None


def serving_now() -> Any:
    """The live fleet that serves this driver's renders NOW, or None.

    The handle a render was submitted to may have been retired by a reset
    since, with a fresh fleet attached on the same config in its place
    (``driver_unload.serving_fleet``). The note is never called as it is: a
    retired handle reaches Workers that are gone. A mesh that is not set up, or
    whose cleanup failed, gets no call either (``driver_unload.trimmable``).
    """
    from . import driver_unload

    reference = _SERVING["handle"]
    noted = reference() if reference is not None else None
    if noted is None:
        return None
    fleet = driver_unload.serving_fleet(noted)
    return fleet if driver_unload.trimmable(fleet) else None


def release_worker_pools() -> tuple[str, bool]:
    """Ask the serving fleet's actors for their idle device pools: (the relief's clause, whether they released).

    One call of the trim the driver unload already makes
    (``driver_unload.trim_actors``: every actor's ``clear_vram`` at ``pool``,
    which empties its device pool and collects, and drops nothing: not a
    model, and not the Worker's own decode VAE). A Worker keeps the
    pool its sampler grew reserved after the last step, 6.4 GiB on record,
    beside a driver whose decode then asks for 2.4 to 4.4. The Worker runs the
    call under its GPU lock, so never under a sample, and the claim on the
    mesh's render session fails within seconds where a pipelined render owns
    it: the decode then runs as before. It does not ask
    ``driver_unload.busy_reason``, for the reason this module's text gives.
    Never raises; an empty clause where there is no live fleet to ask.
    """
    from . import driver_unload

    try:
        fleet = serving_now()
        if fleet is None:
            return "", False
        trimmed = driver_unload.trim_actors(fleet, driver_unload.POOL_LEVEL)
    except Exception as exc:
        return f"; the Workers were not asked for their idle device pools ({type(exc).__name__}: {exc})", False
    return (f"; {trimmed.actors} Worker actor(s) then released their idle device pools: MemAvailable read "
            f"{_gib(trimmed.before)} GiB before that and {_gib(trimmed.after)} GiB after it"), True


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


def kept_and_idle(mm: Any, vae: Any) -> tuple[list[Any], list[str]]:
    """What ComfyUI holds here, split: the entries the relief keeps loaded, and the text encoders' names.

    Freeable: a text encoder (its patcher carries ``is_clip``, which
    ``comfy.sd.CLIP`` sets) on the decode's own device, since ``free_memory``
    frees on one device. Kept: everything else, the VAE in hand first of all.
    A model this cannot read is kept.
    """
    own = getattr(vae, "patcher", None)
    device = getattr(vae, "device", None)
    keep: list[Any] = []
    idle: list[str] = []
    for loaded in list(mm.current_loaded_models):
        patcher = loaded.model
        if patcher is None:
            continue
        text_encoder = getattr(patcher, "is_clip", False) is True and patcher is not own
        if text_encoder and getattr(loaded, "device", device) == device:
            inner = getattr(patcher, "model", None)
            idle.append(type(patcher if inner is None else inner).__name__)
        else:
            keep.append(loaded)
    return keep, idle


def _dynamic_vram_on() -> bool:
    return bool(getattr(sys.modules.get("comfy.memory_management"), "aimdo_enabled", False))


def relieve(mm: Any, vae: Any, samples: Any, told: set[str]) -> bool:
    """Free what this decode needs if it is priced above the reading; True when it acted.

    ``told`` holds what this process has already said about decodes it could
    not help, so a tight session, or a preview at every step, logs each reason
    once.
    """
    if not _dynamic_vram_on():
        return False
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
    from . import driver_unload, mesh_safety

    priced = (f"ComfyUI priced this VAE decode at {_gib(price)} GiB ({_gib(need)} GiB with its "
              f"reserve) against {_gib(reading)} GiB free")
    why = ""
    if sampling_depth() > 0:
        why = SAMPLER_RUNNING
    elif mesh_safety.env_enabled(driver_unload.DRIVER_UNLOAD_DISABLE_ENV):
        why = UNLOAD_SWITCH
    if why:
        if why not in told:
            told.add(why)
            log.info("decode relief: %s; the driver kept what it holds because %s", priced, why)
        return False
    keep, idle = kept_and_idle(mm, vae)
    if not idle:
        # Nothing of the driver's to free: the Workers' idle pools are still there.
        workers, released = release_worker_pools()
        if released:
            log.info("decode relief: %s; ComfyUI held no text encoder here to free from%s", priced, workers)
            return True
        if "nothing idle" not in told:
            told.add("nothing idle")
            log.info("decode relief: %s; ComfyUI held no text encoder here to free from%s, so the decode "
                     "runs as ComfyUI runs it", priced, workers)
        return False
    before = int(mesh_safety.mem_available_bytes() or 0)
    # ComfyUI's classic loader's own call for this requirement: it frees in its own order until the
    # call fits, and a DynamicVRAM model gives up pages without being unloaded where that is enough.
    mm.free_memory(_EVERYTHING if FREE_ALL_IDLE else need, device, keep_loaded=keep, for_dynamic=False)
    gc.collect()
    mm.soft_empty_cache()
    time.sleep(_SETTLE_S)
    available = int(mesh_safety.mem_available_bytes() or 0)
    after = int(mm.get_free_memory(device))
    workers = ""
    if after < need:
        # Still priced above the reading: the Workers beside this driver hold device pools their last
        # step left reserved and unused. One call of the trim the driver unload already makes.
        workers, released = release_worker_pools()
        if released:
            time.sleep(_SETTLE_S)
            after = int(mm.get_free_memory(device))
    tail = (f"the decode starts with {_gib(after)} GiB free" if after >= need else
            f"the decode is still priced above the {_gib(after)} GiB free and runs as ComfyUI runs it")
    log.info("decode relief: %s, so the driver had ComfyUI free what the decode needs from the %d text "
             "encoder%s it held (%s): MemAvailable read %s GiB before that and %s GiB after it%s; %s",
             priced, len(idle), "" if len(idle) == 1 else "s", ", ".join(idle), _gib(before),
             _gib(available), workers, tail)
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
                                "in this process, with no idle memory freed before it", aside[0])
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
        log.info("%s=1: ComfyUI's own VAE decode runs in this process with no idle memory freed "
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
             "has ComfyUI free what the decode needs from the text encoders it holds (%s=1 turns this off)",
             DISABLE_ENV)
    return _set(INSTALLED)


def _comfy_class(module: str, name: str) -> Any:
    """A ComfyUI class, only if this process already imported its module.

    It imports nothing itself: importing ComfyUI here would pick its device
    before the process that owns it does.
    """
    found = sys.modules.get(module)
    if found is None:
        raise ImportError(f"{module} is not imported in this process")
    return getattr(found, name)


def install() -> None:
    """Wrap ``comfy.samplers.CFGGuider.sample`` (the guard) and then ``comfy.sd.VAE.decode``, each once.

    The guard goes first: the decode wrapper must never be in place without
    it. ``VAE.decode_tiled`` and ``VAE.encode`` are left as they are.
    """
    guider = _comfy_class("comfy.samplers", "CFGGuider")
    vae = _comfy_class("comfy.sd", "VAE")
    with _INSTALL_LOCK:
        if not getattr(guider.sample, _GUARD_KEY, False):
            guider.sample = with_sampler_guard(guider.sample)
        if not getattr(vae.decode, _KEY, False):
            vae.decode = with_relief(vae.decode)


def uninstall() -> None:
    """Undo ``install`` (tests)."""
    vae = _comfy_class("comfy.sd", "VAE")
    guider = _comfy_class("comfy.samplers", "CFGGuider")
    with _INSTALL_LOCK:
        if getattr(vae.decode, _KEY, False):
            vae.decode = vae.decode.__wrapped__
        if getattr(guider.sample, _GUARD_KEY, False):
            guider.sample = guider.sample.__wrapped__
    _set(UNSET)


__all__ = ["DISABLE_ENV", "FREE_ALL_IDLE", "INSTALLED", "NOT_INTEGRATED", "NOT_LINUX", "NO_COMFY", "OFF",
           "PROBE_FAILED", "SAMPLER_RUNNING", "STEPPED_ASIDE", "UNLOAD_SWITCH", "UNSET", "decode_need", "decode_price",
           "install", "install_driver", "kept_and_idle", "loader_need", "note_serving", "release_worker_pools",
           "relieve", "sampling_depth", "serving_now", "state", "uninstall", "with_relief", "with_sampler_guard"]
