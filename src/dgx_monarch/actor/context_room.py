"""A Worker makes room for its CUDA context before it creates one.

A Worker's first CUDA call creates the process's CUDA context, and on a GPU
that shares the host's memory the NVIDIA driver needs free memory for it:
free, not available. With the box's free memory at a few GiB the driver's
request fails although tens of GiB sit in the page cache, and the Worker's
setup ended in a raw ``CUDA error: out of memory`` from
``cuDevicePrimaryCtxRetain``. On a DGX Spark on 2026-11-12 it failed with 1.6
and 4.6 GiB free (45 and 47 GiB available) and worked on every start on
record with 8.9 GiB free or more.

Two rules answer it, both in the Worker process, which is where the context
is made.

**Room first, then the context at once.** Before the bootstrap that makes the
context (``comfy_bridge.ensure_comfy``, whose import of
``comfy.model_management`` is the Worker's first CUDA call), the Worker reads
``/proc/meminfo``. If free memory is under 12 GiB while 17 GiB or more is
available, and the GPU shares the host's memory, it writes to anonymous memory
in steps of 1 GiB up to 12 GiB, holds it, and releases all of it at once. The
kernel gives page cache back to serve those writes, and the release leaves
that memory free for the driver. It checks before every step that the step
leaves 5 GiB available and stops if it would not. It never locks the memory.
Free memory does not stay free: any file read on the box fills it with page
cache again. So torch is in the process before the writes (a Worker has
imported it by then; the step makes sure), and straight after the
release the Worker makes the context itself, by the call ComfyUI's import
makes first (``torch.cuda.mem_get_info``). One log line says what it did, or
why it did nothing in that state. A start with 12 GiB free reads one file and
does nothing else.

Under comfy-managed residency the Worker makes no context of its own: stage A
of that bring-up loads comfy-aimdo in the bootstrap, and it stays ahead of the
context there. The room is still made, the bootstrap's import makes the
context a moment later, and memory freed can be taken again in between; the
bootstrap reads free memory once more just ahead of that import
(``at_bootstrap_context``), so the record has the figure the driver met.

The amount is the mark and not the mark less what is free: the pages a
process writes first are the free ones, so writing 7 GiB on a box with 5 GiB
free leaves about 7 GiB free afterwards, not 12.

Whether the GPU shares the host's memory is asked of the CUDA driver library
itself (``cuDeviceGetAttribute``), which answers without a context. The
pack's other probes ask torch for the device's properties, which is right
where a context exists and is not known to be safe before one.

**A plain refusal behind it.** If the context cannot be made, here or in the
bootstrap before ``comfy.model_management`` has imported, for want of memory
(a CUDA out-of-memory error), the Worker raises a class C refusal (guard
``gpu_context_memory``, no card) that says what failed, what is free and
available, what the make-room step did, and what to do. It carries the
Worker's rank, so the driver can name the box. Any other error passes on
unchanged.

``DGXM_DISABLE_CONTEXT_ROOM=1`` in the Worker's environment turns the
make-room step off, for diagnosis; no user needs it, and the refusal stays.
A fault anywhere in the step leaves the start as it was before this module:
it is logged and the bootstrap goes on. The Worker's status carries what the
step did (``context_room``).
"""
from __future__ import annotations

import ctypes
import importlib
import mmap
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

from .. import mesh_safety, residency_mode
from ..log import get_logger
from ..refusal import RefusalClass, refusal

log = get_logger(__name__)

_GIB = 2 ** 30

DISABLE_ENV = "DGXM_DISABLE_CONTEXT_ROOM"

# Free memory the Worker makes room up to. The lowest Worker start on record
# that made its context had 8.9 GiB free; the two that failed had 4.6 and 1.6.
MARK_BYTES = 12 * _GIB
# What must stay available while the Worker holds what it wrote: the floor the
# pack's prices keep.
FLOOR_BYTES = 5 * _GIB
STEP_BYTES = 1 * _GIB

# cuda.h: CU_DEVICE_ATTRIBUTE_INTEGRATED, "Device is integrated with host memory".
_CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
# The first module whose import makes the Worker's CUDA context.
_CONTEXT_MODULE = "comfy.model_management"
_FILL = b"\x01" * (4 * 2 ** 20)

Reading = tuple[int, int]


class ContextMemoryError(RuntimeError):
    """The NVIDIA driver could not get memory for this Worker's CUDA context."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """What the make-room step did before one bootstrap."""

    state: str = "not_run"
    # What was read before the step, in bytes; None where it was not read.
    free: int | None = None
    available: int | None = None
    touched: int = 0
    seconds: float = 0.0
    free_after: int | None = None
    # Why the step did nothing, or stopped early, in words a refusal can carry.
    why: str = ""
    # The Worker made the context itself, straight after the release.
    context_made: bool = False
    # Free memory as the bootstrap was about to make the context, where the
    # Worker left the context to it; None where that was not read.
    free_at_bootstrap: int | None = None

    def status(self) -> dict[str, Any]:
        """The block the Worker's status publishes."""
        block: dict[str, Any] = {"state": self.state}
        if self.state == "made_room":
            block.update(touched_gib=round(self.touched / _GIB, 1),
                         seconds=round(self.seconds, 1),
                         free_before_gib=_gib_or_none(self.free),
                         free_after_release_gib=_gib_or_none(self.free_after),
                         free_at_context_gib=_gib_or_none(
                             self.free_after if self.context_made
                             else self.free_at_bootstrap),
                         context="made at once" if self.context_made
                         else "left to the bootstrap")
        if self.why:
            block["why"] = self.why
        return block

    def sentence(self) -> str:
        """What a refusal says the Worker did before the context failed."""
        if self.state == "made_room":
            after = ("" if self.free_after is None
                     else f", after which {_gib(self.free_after)} GiB was free")
            early = f" ({self.why})" if self.why else ""
            if self.free_at_bootstrap is not None:
                early += (f"; {_gib(self.free_at_bootstrap)} GiB was free when the context "
                          "was asked for")
            return (f"Before that the Worker made room: it wrote to {_gib(self.touched)} GiB "
                    f"of memory and released it{after}{early}, and the context still could "
                    "not be made, so free memory may not be all the driver lacks.")
        if self.why:
            return f"The Worker made no room first: {self.why}."
        return "The Worker made no room first."


_LAST = Outcome()


def _gib(value: int | float) -> str:
    return f"{value / _GIB:.1f}"


def _gib_or_none(value: int | None) -> float | None:
    return None if value is None else round(value / _GIB, 1)


def meminfo(path: str = "/proc/meminfo") -> Reading | None:
    """MemFree and MemAvailable of this host in bytes, or None where they cannot be read."""
    found: dict[str, int] = {}
    try:
        with open(path) as lines:
            for line in lines:
                name, _, rest = line.partition(":")
                if name in ("MemFree", "MemAvailable"):
                    found[name] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    if len(found) != 2:
        return None
    return found["MemFree"], found["MemAvailable"]


def device_shares_host_memory() -> bool | None:
    """Whether CUDA device 0 shares the host's memory, asked without a context.

    The driver library answers a device attribute from its handle alone, and
    the Worker's device is ordinal 0 once setup has set
    ``CUDA_VISIBLE_DEVICES``. None when the library cannot be asked.
    """
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        device, value = ctypes.c_int(), ctypes.c_int()
        if (cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(device), 0)
                or cuda.cuDeviceGetAttribute(ctypes.byref(value),
                                             _CU_DEVICE_ATTRIBUTE_INTEGRATED, device)):
            return None
        return bool(value.value)
    except Exception:
        return None


def _take(nbytes: int) -> Any:
    """Anonymous memory of ``nbytes`` with every page written. Never locked."""
    held = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try:
        for start in range(0, nbytes, len(_FILL)):
            size = min(len(_FILL), nbytes - start)
            held[start:start + size] = _FILL[:size]
    except BaseException:
        held.close()
        raise
    return held


def _release(held: list[Any]) -> None:
    while held:
        try:
            held.pop().close()
        except Exception:  # one map that will not close must not keep the rest
            pass


def _preload() -> None:
    """Import torch before any memory is freed, so its files are not read after.

    Importing torch makes no CUDA call. A Worker process has it already; this
    is for one that does not.
    """
    importlib.import_module("torch")


def _torch_context() -> None:
    """Make this process's CUDA context, by the call ComfyUI's import makes first."""
    importlib.import_module("torch").cuda.mem_get_info()


def _stepped(read: Callable[[], Reading | None], take: Callable[[int], Any],
             clock: Callable[[], float], free: int, available: int) -> Outcome:
    """Write up to the mark in steps, hold every step, release them together."""
    held: list[Any] = []
    touched, why, started = 0, "", clock()
    try:
        while touched < MARK_BYTES:
            now = read()
            if now is None or now[1] - STEP_BYTES < FLOOR_BYTES:
                why = (f"it stopped there because one more step would have left under "
                       f"{_gib(FLOOR_BYTES)} GiB available")
                break
            size = min(STEP_BYTES, MARK_BYTES - touched)
            held.append(take(size))
            touched += size
    finally:
        _release(held)
    seconds = clock() - started
    after = read()
    return Outcome("made_room" if touched else "skipped", free, available, touched, seconds,
                   None if after is None else after[0],
                   why if touched else f"{_gib(available)} GiB available left no room for a step")


def _decide(read: Callable[[], Reading | None], shares: Callable[[], bool | None],
            take: Callable[[int], Any], clock: Callable[[], float],
            preload: Callable[[], None]) -> Outcome:
    if mesh_safety.env_enabled(DISABLE_ENV):
        return Outcome("off", why=f"the step is off ({DISABLE_ENV}=1 on the Worker)")
    reading = read()
    if reading is None:
        return Outcome("not_run")
    free, available = reading
    if free >= MARK_BYTES:
        return Outcome("roomy", free, available)
    # Free memory is short from here on, so every way out says why in one line.
    if available < MARK_BYTES + FLOOR_BYTES:
        outcome = Outcome("skipped", free, available, why=(
            f"{_gib(available)} GiB available is under the "
            f"{_gib(MARK_BYTES + FLOOR_BYTES)} GiB it needs to do that safely"))
    else:
        shared = shares()
        if shared is None:
            outcome = Outcome("skipped", free, available, why=(
                "the CUDA driver library could not say whether this GPU shares the host's "
                "memory"))
        elif not shared:
            outcome = Outcome("skipped", free, available,
                              why="this GPU has memory of its own")
        else:
            preload()
            outcome = _stepped(read, take, clock, free, available)
    if outcome.state == "made_room":
        log.info("free memory was %s GiB with %s GiB available, so the Worker made room for "
                 "its GPU context: %s GiB touched and released in %.1f seconds%s; %s GiB is "
                 "free now", _gib(free), _gib(available), _gib(outcome.touched),
                 outcome.seconds, f" ({outcome.why})" if outcome.why else "",
                 "an unread amount" if outcome.free_after is None else _gib(outcome.free_after))
    else:
        log.info("free memory is %s GiB with %s GiB available, and the Worker made no room "
                 "for its GPU context: %s", _gib(free), _gib(available), outcome.why)
    return outcome


def make_room(*, read: Callable[[], Reading | None] = meminfo,
              shares: Callable[[], bool | None] = device_shares_host_memory,
              take: Callable[[int], Any] = _take,
              clock: Callable[[], float] = time.monotonic,
              preload: Callable[[], None] = _preload) -> Outcome:
    """Make room for the CUDA context where free memory is short; never raises.

    Any fault leaves the start as it was: one WARNING, and the bootstrap goes
    on to make its context as before.
    """
    global _LAST
    try:
        _LAST = _decide(read, shares, take, clock, preload)
    except Exception as exc:
        log.warning("the Worker could not make room for its GPU context (%r); it goes on "
                    "to make the context as before", exc)
        _LAST = Outcome("failed", why=f"the step failed ({type(exc).__name__})")
    return _LAST


def status_block() -> dict[str, Any]:
    """What the make-room step last did in this Worker process, for its status."""
    return _LAST.status()


def _is_cuda_out_of_memory(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


GUARD = "gpu_context_memory"


def _figures(reading: Reading | None) -> str:
    if reading is None:
        return "The memory of this Worker's box could not be read."
    return (f"{_gib(reading[0])} GiB of memory is free on this Worker's box and "
            f"{_gib(reading[1])} GiB is available: the driver needs free memory, and the "
            "rest of what is available is page cache, files the system has read and still "
            "keeps.")


def refuse_unmade_context(exc: BaseException, outcome: Outcome,
                          rank: tuple[Any, Any] | None = None) -> None:
    """Raise the typed refusal where ``exc`` is the context that could not be made.

    Only a CUDA out-of-memory error counts, raised before the context exists:
    the Worker did not make it itself, and the context module has not finished
    importing. Anything else returns, and the caller raises it as it is.
    ``rank`` is this Worker's rank and world, marked on the refusal so the
    driver can name the box (``failure.mark_rank``).
    """
    if (outcome.context_made or _CONTEXT_MODULE in sys.modules
            or not _is_cuda_out_of_memory(exc)):
        return
    try:
        reading = meminfo()
    except Exception:
        reading = None
    try:
        raise ContextMemoryError(refusal(
            RefusalClass.CAPACITY,
            "this Worker could not start on its GPU: the NVIDIA driver could not get memory "
            f"for a new CUDA context. {_figures(reading)} {outcome.sentence()} Nothing was "
            "loaded. Close a program that holds memory, or wait for a large download or file "
            "copy to end, then click Reset attached mesh and queue again.",
            guard=GUARD, troubleshooting=12)) from exc
    except ContextMemoryError as refused:
        if rank is not None:
            from .failure import mark_rank

            mark_rank(SimpleNamespace(rank=rank[0], world=rank[1]), refused)
        raise


def make_context(outcome: Outcome, rank: tuple[Any, Any] | None = None, *,
                 create: Callable[[], None] = _torch_context) -> Outcome:
    """Make the context straight after the release, before any file read refills memory.

    A CUDA out-of-memory error here is the refusal's cause. Any other fault is
    logged, and the bootstrap goes on to make the context as before.
    """
    global _LAST
    try:
        create()
    except Exception as exc:
        refuse_unmade_context(exc, outcome, rank)
        log.warning("the Worker could not make its GPU context ahead of the bootstrap (%r); "
                    "the bootstrap makes it as before", exc)
        return outcome
    _LAST = replace(outcome, context_made=True)
    return _LAST


def at_bootstrap_context(read: Callable[[], Reading | None] | None = None) -> None:
    """Read free memory as the bootstrap is about to make the context; never raises.

    Stage A of the bootstrap calls it on every start. It reads only where the
    Worker made room and left the context to the bootstrap (comfy-managed
    residency, or an early try that failed for another cause), so the record
    has free memory at the moment the driver is asked and not only at the
    release.
    """
    global _LAST
    try:
        if (_LAST.state != "made_room" or _LAST.context_made
                or _CONTEXT_MODULE in sys.modules):
            return
        reading = (read or meminfo)()
        if reading is None:
            return
        _LAST = replace(_LAST, free_at_bootstrap=reading[0])
        log.info("%s GiB is free as the bootstrap makes the Worker's GPU context",
                 _gib(reading[0]))
    except Exception as exc:
        log.warning("the Worker could not read free memory ahead of its GPU context (%r)", exc)


def ensure_comfy(comfy_dir: str, worker_args: dict | None = None, *,
                 rank: tuple[Any, Any] | None = None, **kwargs: Any) -> None:
    """``comfy_bridge.ensure_comfy``, with room for the context its first call makes.

    The context is made by the first import of ``comfy.model_management``, so
    a process that has imported it already goes straight through. Where room
    was made, the Worker makes the context at once, unless comfy-managed
    residency was asked for: its stage A runs in the bootstrap and stays ahead
    of the context there.
    """
    from . import comfy_bridge

    if _CONTEXT_MODULE in sys.modules:
        comfy_bridge.ensure_comfy(comfy_dir, worker_args, **kwargs)
        return
    outcome = make_room()
    if outcome.state == "made_room" and not residency_mode.requested(worker_args):
        outcome = make_context(outcome, rank)
    try:
        comfy_bridge.ensure_comfy(comfy_dir, worker_args, **kwargs)
    except Exception as exc:
        # The bootstrap's own reading, where it took one for this outcome.
        if replace(_LAST, free_at_bootstrap=None) == outcome:
            outcome = _LAST
        refuse_unmade_context(exc, outcome, rank)
        raise
