"""Unload ComfyUI's own models on the driver host before a capacity refusal.

The render-site capacity checks (``nodes/render_preflight``) read MemAvailable
on the driver host, where the text encoder, the VAEs and any other model
ComfyUI loaded for this graph count as spent. ComfyUI reloads every one of them
on demand. So when a check would refuse and ComfyUI holds a model, the driver
unloads them through ComfyUI's own model management, the call ComfyUI makes
for ``/free``, reads MemAvailable again, and runs the checks again from the one
that refused. Only that second answer refuses, and its text says what was
unloaded and what MemAvailable read before and after the unload
(docs/TROUBLESHOOTING.md #56).

Three things this path never touches. A render still running in this driver:
the driver keeps the models and the refusal says why. The Workers' residents:
they live in other processes, and the model management here holds only this
process's models. ComfyUI's node output cache: the current prompt still needs
its own outputs, and ComfyUI's default cache already drops earlier prompts'
outputs after every node. ``DGXM_DISABLE_DRIVER_UNLOAD=1`` turns the step off,
and the first answer then refuses unchanged.
"""
from __future__ import annotations

import gc
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from . import client_lease, mesh_safety
from .log import get_logger
from .render_memory_price import RenderMemoryPriceError

log = get_logger(__name__)

_GIB = 2 ** 30

DRIVER_UNLOAD_DISABLE_ENV = "DGXM_DISABLE_DRIVER_UNLOAD"

# The render-site refusals priced against this host's MemAvailable. The render
# memory refusal stays outside the stock-load family on purpose
# (render_memory_price), so it is named beside it.
CAPACITY_REFUSALS: tuple[type[Exception], ...] = (
    mesh_safety.StockLoadCapacityError, RenderMemoryPriceError)

# ComfyUI waits this long after a DynamicVRAM decommit before it reads free
# memory again (execution.py, its RAM pressure branch); the re-read waits too.
_SETTLE_S = 0.05


def _gib(value: int) -> str:
    return f"{value / _GIB:.1f}"


def _models(names: Sequence[str]) -> str:
    count = len(names)
    return f"{count} model{'' if count == 1 else 's'} ({', '.join(names)})"


@dataclass(frozen=True, slots=True)
class Unloaded:
    """What one unload removed, with MemAvailable read on each side of it."""

    names: tuple[str, ...]
    before: int
    after: int

    @property
    def gain(self) -> int:
        return self.after - self.before

    def refusal_sentence(self) -> str:
        """The sentence a refusal that still stands after the unload carries."""
        return (f"Before refusing, the driver unloaded the {_models(self.names)} "
                f"ComfyUI held on this host: MemAvailable read "
                f"{_gib(self.before)} GiB before the unload and "
                f"{_gib(self.after)} GiB after it, and the figures above are "
                "the reading after it.")

    def log_line(self, subject: str) -> str:
        """The one line the driver logs when the unload lets a render through."""
        return (f"driver unload: {subject} fits after unloading the "
                f"{_models(self.names)} ComfyUI held on this host; MemAvailable "
                f"went from {_gib(self.before)} GiB to {_gib(self.after)} GiB, "
                f"a gain of {_gib(self.gain)} GiB")


def comfy_model_management() -> Any | None:
    """ComfyUI's model management in this process, or None without ComfyUI."""
    try:
        import comfy.model_management as mm
    except Exception:  # no ComfyUI here means no model to unload
        return None
    return mm


def held_models(mm: Any) -> list[tuple[Any, str]]:
    """Each model ComfyUI holds in this process, with the class name it logs.

    The names are the ones ComfyUI prints in "Requested to load". An entry
    whose model is already gone holds nothing an unload could free.
    """
    held: list[tuple[Any, str]] = []
    for entry in list(getattr(mm, "current_loaded_models", None) or ()):
        try:
            patcher = entry.model
            if patcher is None:
                continue
            inner = getattr(patcher, "model", None)
            held.append((entry, type(patcher if inner is None else inner).__name__))
        except Exception:
            continue
    return held


def _lease_count(handle: Any, attribute: str) -> int:
    return sum(int(count) for count in dict(getattr(handle, attribute, None) or {}).values())


def busy_reason(handle: Any) -> str:
    """Why the driver must keep ComfyUI's models now, or empty when it may unload.

    The driver's progress tracker spans every render this process started,
    from before dispatch to its cleanup, pipelined submissions included. The
    handle's leases and in-flight call cover a result or a Worker call that
    outlives it. State that cannot be read keeps the models.
    """
    try:
        from .telemetry import render_progress

        if render_progress.snapshot().get("active"):
            return "a render is still running in this driver"
        if handle is not None:
            if _lease_count(handle, "sample_leases"):
                return "a render result is still held on the mesh"
            if _lease_count(handle, "abandoned_sample_leases"):
                return "an abandoned render may still be running on the Workers"
            if client_lease.dispatch_in_flight(handle):
                return "a call to the Workers is still in flight"
    except Exception:
        return "the driver could not read whether a render is running"
    return ""


def unload(mm: Any, held: Sequence[tuple[Any, str]]) -> Unloaded:
    """Unload every model ComfyUI holds here, then read MemAvailable once more.

    The calls are ComfyUI's own ``/free`` sequence: unload, collect, return
    the allocator's cached blocks.
    """
    before = mesh_safety.mem_available_bytes() or 0
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    time.sleep(_SETTLE_S)
    after = mesh_safety.mem_available_bytes() or 0
    kept = {id(entry) for entry in list(getattr(mm, "current_loaded_models", None) or ())}
    return Unloaded(tuple(name for entry, name in held if id(entry) not in kept),
                    before, after)


def _amended(exc: Exception, sentence: str) -> Exception:
    """The same refusal with one more sentence at the end of its first line.

    The tag and the budget figures stay at the front, where a reader that
    keeps 200 characters finds them; the type and traceback stay too.
    """
    head, newline, tail = str(exc).partition("\n")
    exc.args = (f"{head.rstrip()} {sentence}{newline}{tail}",)
    return exc


def check_with_unload(steps: Sequence[Callable[[], object]], *, handle: Any = None,
                      subject: str = "this render") -> None:
    """Run ``steps`` in order, unloading ComfyUI's models before one refuses.

    A capacity refusal stops the run. If ComfyUI holds a model in this process
    and nothing is running, the driver unloads every model it holds and runs
    the steps again from the one that refused: the steps before it passed on a
    lower reading. Any other exception, and any refusal the unload cannot
    answer, propagates as it was raised.
    """
    for index, step in enumerate(steps):
        try:
            step()
        except CAPACITY_REFUSALS as refused:
            _after_refusal(steps[index:], refused, handle, subject)
            return


def _after_refusal(steps: Sequence[Callable[[], object]], refused: Exception,
                   handle: Any, subject: str) -> None:
    """Unload and run ``steps`` again, or raise ``refused`` saying why it stands."""
    if mesh_safety.env_enabled(DRIVER_UNLOAD_DISABLE_ENV):
        raise refused
    mm = comfy_model_management()
    held = held_models(mm) if mm is not None else []
    if not held:
        raise refused
    names = tuple(name for _entry, name in held)
    busy = busy_reason(handle)
    if busy:
        raise _amended(refused, (
            f"The driver did not unload the {_models(names)} ComfyUI holds on "
            f"this host before refusing, because {busy}; queue again once that "
            "work ends."))
    try:
        outcome = unload(mm, held)
    except Exception as exc:
        log.warning("driver unload failed before refusing %s: %r", subject, exc)
        raise _amended(refused, (
            f"The driver tried to unload the {_models(names)} ComfyUI holds on "
            f"this host before refusing, and the unload failed "
            f"({type(exc).__name__}), so the figures above are the reading "
            "before it.")) from None
    if not outcome.names:
        log.warning("driver unload: ComfyUI kept every model on this host; "
                    "refusing %s on the first reading", subject)
        raise refused
    try:
        for step in steps:
            step()
    except CAPACITY_REFUSALS as again:
        raise _amended(again, outcome.refusal_sentence()) from None
    log.info(outcome.log_line(subject))
