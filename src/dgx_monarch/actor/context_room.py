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

**Room first.** Before the bootstrap that makes the context
(``comfy_bridge.ensure_comfy``, whose import of ``comfy.model_management`` is
the Worker's first CUDA call), the Worker reads ``/proc/meminfo``. If free
memory is under 12 GiB while 17 GiB or more is available, and the GPU shares
the host's memory, it writes to anonymous memory in steps of 1 GiB up to 12
GiB, holds it, and releases all of it at once. The kernel gives page cache
back to serve those writes, and the release leaves that memory free for the
driver. It checks before every step that the step leaves 5 GiB available and
stops if it would not. It never locks the memory. One log line says what it
did, or why it did nothing in that state. A start with 12 GiB free reads one
file and does nothing else.

The amount is the mark and not the mark less what is free: the pages a
process writes first are the free ones, so writing 7 GiB on a box with 5 GiB
free leaves about 7 GiB free afterwards, not 12.

Whether the GPU shares the host's memory is asked of the CUDA driver library
itself (``cuDeviceGetAttribute``), which answers without a context. The
pack's other probes ask torch for the device's properties, which is right
where a context exists and is not known to be safe before one.

**A plain refusal behind it.** If the bootstrap still fails with a CUDA
out-of-memory error before ``comfy.model_management`` has imported, the
Worker raises a class C refusal that says what failed, what is free and
available, what the make-room step did, and what to do. Any other error
passes on unchanged.

``DGXM_DISABLE_CONTEXT_ROOM=1`` in the Worker's environment turns the
make-room step off, for diagnosis; no user needs it, and the refusal stays.
A fault anywhere in the step leaves the start as it was before this module:
it is logged and the bootstrap goes on. The Worker's status carries what the
step did (``context_room``).
"""
from __future__ import annotations

import ctypes
import mmap
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .. import mesh_safety
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

    def status(self) -> dict[str, Any]:
        """The block the Worker's status publishes."""
        block: dict[str, Any] = {"state": self.state}
        if self.state == "made_room":
            block.update(touched_gib=round(self.touched / _GIB, 1),
                         seconds=round(self.seconds, 1),
                         free_before_gib=_gib_or_none(self.free),
                         free_after_gib=_gib_or_none(self.free_after))
        if self.why:
            block["why"] = self.why
        return block

    def sentence(self) -> str:
        """What a refusal says the Worker did before the context failed."""
        if self.state == "made_room":
            after = ("" if self.free_after is None
                     else f", after which {_gib(self.free_after)} GiB was free")
            early = f" ({self.why})" if self.why else ""
            return (f"Before that the Worker made room: it wrote to {_gib(self.touched)} GiB "
                    f"of memory and released it{after}{early}, and the context still could "
                    "not be made.")
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
            take: Callable[[int], Any], clock: Callable[[], float]) -> Outcome:
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
              clock: Callable[[], float] = time.monotonic) -> Outcome:
    """Make room for the CUDA context where free memory is short; never raises.

    Any fault leaves the start as it was: one WARNING, and the bootstrap goes
    on to make its context as before.
    """
    global _LAST
    try:
        _LAST = _decide(read, shares, take, clock)
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


def refusal_text(outcome: Outcome, reading: Reading | None) -> str:
    """The class C refusal for a context the driver could not make."""
    figures = ("The memory of this Worker's box could not be read." if reading is None else
               f"{_gib(reading[0])} GiB of memory is free on this Worker's box and "
               f"{_gib(reading[1])} GiB is available: the driver needs free memory, and the "
               "rest of what is available is page cache, files the system has read and still "
               "keeps.")
    return refusal(RefusalClass.CAPACITY, (
        "this Worker could not start on its GPU: the NVIDIA driver could not get memory for "
        f"a new CUDA context. {figures} {outcome.sentence()} Nothing was loaded. Close a "
        "program that holds memory, or wait for a large download or file copy to end, then "
        "click Reset attached mesh and queue again."), troubleshooting=12)


def refuse_unmade_context(exc: BaseException, outcome: Outcome) -> None:
    """Raise the typed refusal where ``exc`` is the context that could not be made.

    Only a CUDA out-of-memory error raised before the context module finished
    importing counts. Anything else returns, and the caller raises it as it is.
    """
    if _CONTEXT_MODULE in sys.modules or not _is_cuda_out_of_memory(exc):
        return
    try:
        reading = meminfo()
    except Exception:
        reading = None
    raise ContextMemoryError(refusal_text(outcome, reading)) from exc


def ensure_comfy(*args: Any, **kwargs: Any) -> None:
    """``comfy_bridge.ensure_comfy``, with room for the context its first call makes.

    The context is made by the first import of ``comfy.model_management``, so
    a process that has imported it already goes straight through.
    """
    from . import comfy_bridge

    if _CONTEXT_MODULE in sys.modules:
        comfy_bridge.ensure_comfy(*args, **kwargs)
        return
    outcome = make_room()
    try:
        comfy_bridge.ensure_comfy(*args, **kwargs)
    except Exception as exc:
        refuse_unmade_context(exc, outcome)
        raise
