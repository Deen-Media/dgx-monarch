"""Free what the driver host can get back before a capacity refusal.

The render-site capacity checks (``nodes/render_preflight``) read MemAvailable
on the driver host, where the text encoder, the VAEs and any other model
ComfyUI loaded for this graph count as spent. ComfyUI reloads every one of them
on demand. So when a check would refuse and ComfyUI holds a model, the driver
unloads them through ComfyUI's own model management, the call ComfyUI makes
for ``/free``, reads MemAvailable again, and runs the checks again from the one
that refused.

The actors count there too. One that ran the last render keeps that render's
activations in its allocator cache, so the next render can reuse them, and on
the driver host that cache reads as spent. So when the check still refuses and
the render's mesh is live and set up, the driver asks every actor to release
its allocator cache through the Worker ``clear_vram`` endpoint at ``soft``
(ComfyUI's ``soft_empty_cache`` and a gc, with no model unloaded), reads
MemAvailable again and runs the checks once more from the one that refused.
Only the last answer refuses, and its text says what was unloaded and released
and what MemAvailable read around each step (docs/TROUBLESHOOTING.md #56).
Both steps are rescues: a check that passes runs neither. Before either, a
render whose DiT load is still pending counts the model that load replaces
(``swap_credit.py``). The loader preflight runs the same credit and the same
unload through ``nodes/loader_relief.py``, with this module's calls.

Three things this path never touches. A render still running in this driver:
the driver keeps the models and the caches, and the refusal says why
(``busy_reason`` reads the fleet attached now, not a mesh a reset stopped). The
Workers' residents: the trim releases cached allocator blocks and unloads no
model. ComfyUI's node output cache: the current prompt still needs its own
outputs, and ComfyUI's default cache already drops earlier prompts' outputs
after every node. ``DGXM_DISABLE_DRIVER_UNLOAD=1`` turns both steps off, and
the answer after the swap credit then refuses unchanged. The credit is a
price, what the box will hold at the load, and not a rescue, so that switch
leaves it on at the loader and here alike; ``DGXM_DISABLE_SWAP_CREDIT=1`` turns
it off at both.
"""
from __future__ import annotations

import gc
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from . import client_lease, mesh_safety, swap_credit
from .driver_spend_price import DriverSpendCapacityError
from .log import get_logger
from .render_memory_price import RenderMemoryPriceError

log = get_logger(__name__)

_GIB = 2 ** 30

DRIVER_UNLOAD_DISABLE_ENV = "DGXM_DISABLE_DRIVER_UNLOAD"

# The render-site refusals priced against this host's MemAvailable. The render
# memory refusal stays outside the stock-load family on purpose
# (render_memory_price), and so does a node's spend on the driver
# (driver_spend_price), so both are named beside it.
CAPACITY_REFUSALS: tuple[type[Exception], ...] = (
    mesh_safety.StockLoadCapacityError, RenderMemoryPriceError, DriverSpendCapacityError)

# ComfyUI waits this long after a DynamicVRAM decommit before it reads free
# memory again (execution.py, its RAM pressure branch); the re-read waits too.
_SETTLE_S = 0.05

# The one Worker call this path makes, and the level it makes it at: ``soft``
# frees the allocator cache, where ``hard`` would also unload every resident.
TRIM_ENDPOINT = "clear_vram"
TRIM_LEVEL = "soft"
POOL_LEVEL = "pool"  # the device pool alone: the Worker keeps its resident decode VAE (the decode relief's ask)

# Another render session that owns the mesh (an idle pipeline, say) fails the
# trim's claim this fast instead of after the default claim wait.
_TRIM_CLAIM_S = 2.0

# The trim runs only when nothing is in flight in this driver, so each actor's
# GPU lock is free and the call returns in seconds. A call past this budget
# leaves the mesh DIRTY, as a ClearVRAM node call past its own budget does,
# because clear_vram is a mutation whose completion is then unknown (mesh_rpc).
_TRIM_TIMEOUT_S = 120.0


def _gib(value: int) -> str:
    return f"{value / _GIB:.1f}"


def _models(names: Sequence[str]) -> str:
    count = len(names)
    return f"{count} model{'' if count == 1 else 's'} ({', '.join(names)})"


def _actors(count: int) -> str:
    return f"{count} actor{'' if count == 1 else 's'}"


@dataclass(frozen=True, slots=True)
class Unloaded:
    """What one unload removed, with MemAvailable read on each side of it."""

    names: tuple[str, ...]
    before: int
    after: int

    @property
    def gain(self) -> int:
        return self.after - self.before

    def readings(self) -> str:
        """What the unload removed and the two readings, as one clause.

        With no model named, only the allocator's cached blocks went back
        (``release_cache``): the driver's ComfyUI keeps more of them since its
        free-memory reading follows MemAvailable (``driver_uma_memory.py``).
        """
        if not self.names:
            return ("released the allocator cache ComfyUI kept on this host: MemAvailable read "
                    f"{_gib(self.before)} GiB before that and {_gib(self.after)} GiB after it")
        return (f"unloaded the {_models(self.names)} ComfyUI held on this host: "
                f"MemAvailable read {_gib(self.before)} GiB before the unload and "
                f"{_gib(self.after)} GiB after it")

    def refusal_sentence(self) -> str:
        """The sentence a refusal that still stands after the unload carries."""
        return (f"Before refusing, the driver {self.readings()}, and the figures "
                "above are the reading after it.")

    def log_line(self, subject: str) -> str:
        """The one line the driver logs when the unload lets a render through."""
        what = (f"unloading the {_models(self.names)} ComfyUI held" if self.names
                else "releasing the allocator cache ComfyUI kept")
        return (f"driver unload: {subject} fits after {what} on this host; MemAvailable "
                f"went from {_gib(self.before)} GiB to {_gib(self.after)} GiB, "
                f"a gain of {_gib(self.gain)} GiB")


@dataclass(frozen=True, slots=True)
class Trimmed:
    """How many actors released their caches, with MemAvailable around it."""

    actors: int
    before: int
    after: int

    @property
    def gain(self) -> int:
        return self.after - self.before

    def refusal_sentence(self, unloaded: Unloaded | None) -> str:
        """The sentence a refusal that still stands after the trim carries."""
        trim = (f"asked the {_actors(self.actors)} on the mesh to release their "
                f"allocator caches: MemAvailable read {_gib(self.before)} GiB "
                f"before the trim and {_gib(self.after)} GiB after it, and the "
                "figures above are the reading after it.")
        if unloaded is None:
            return f"Before refusing, the driver {trim}"
        return f"Before refusing, the driver {unloaded.readings()}. It then {trim}"

    def log_line(self, subject: str, unloaded: Unloaded | None, *, fits: bool) -> str:
        """The one line the driver logs for every trim that ran."""
        first = ("" if unloaded is None else
                 f"the driver unloaded the {_models(unloaded.names)} ComfyUI held "
                 f"on this host (MemAvailable {_gib(unloaded.before)} GiB to "
                 f"{_gib(unloaded.after)} GiB) and ")
        return (f"actor trim: {subject} {'fits' if fits else 'still does not fit'} "
                f"after {first}{_actors(self.actors)} released their allocator "
                f"caches; MemAvailable went from {_gib(self.before)} GiB to "
                f"{_gib(self.after)} GiB across the trim, a gain of "
                f"{_gib(self.gain)} GiB")


def _trim_failed_sentence(unloaded: Unloaded | None, exc: BaseException) -> str:
    failed = ("asked the actors on the mesh to release their allocator caches, "
              f"and that request failed ({type(exc).__name__})")
    if unloaded is None:
        return (f"Before refusing, the driver {failed}, so the figures above are "
                "the reading before it.")
    return f"{unloaded.refusal_sentence()} It also {failed}."


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


def serving_fleet(handle: Any) -> Any:
    """The mesh that serves this driver's loads now: ``handle``, or the one
    attached in its place; None when ``handle`` is retired and none is.

    ComfyUI caches the Init node's output, so a loader or a render is handed
    the mesh that node attached. A reset since then has retired it, and a
    fresh fleet on the same config serves the loads.
    """
    if handle is None or not (getattr(handle, "defunct", False)
                              or getattr(handle, "teardown_complete", False)):
        return handle
    from .nodes import recycle_drain

    return recycle_drain.attached_fleet(handle)


def _fleets_at_work(handle: Any) -> list[Any]:
    """The meshes whose work could still need what the driver holds.

    A mesh with a confirmed stop has no process left, so it is left out: a
    reset leaves its "recycle group teardown" mark on the handle it retires,
    which reads as a call in flight for as long as the handle lives. A
    retired mesh with no confirmed stop is still read, since its Workers may
    be alive, and so is the fleet attached in a retired mesh's place.
    """
    if handle is None:
        return []
    fleets = [] if getattr(handle, "teardown_complete", False) else [handle]
    serving = serving_fleet(handle)
    if serving is not None and serving is not handle:
        fleets.append(serving)
    return fleets


def busy_reason(handle: Any) -> str:
    """Why the driver must keep what it holds now, or empty when it may free it.

    The driver's progress tracker spans every render this process started,
    from before dispatch to its cleanup, pipelined submissions included. A
    mesh's leases and in-flight call cover a result or a Worker call that
    outlives it (``_fleets_at_work``). A status read and a capacity quote set
    none of these. State that cannot be read keeps everything.
    """
    try:
        from .telemetry import render_progress

        if render_progress.snapshot().get("active"):
            return "a render is still running in this driver"
        for fleet in _fleets_at_work(handle):
            if _lease_count(fleet, "sample_leases"):
                return "a render result is still held on the mesh"
            if _lease_count(fleet, "abandoned_sample_leases"):
                return "an abandoned render may still be running on the Workers"
            if client_lease.dispatch_in_flight(fleet):
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


def trimmable(handle: Any) -> bool:
    """Whether ``handle`` is a live mesh whose setup completed.

    Only then do its actors hold caches the trim can release; a mesh that is
    gone, not set up, or DIRTY gets no call. Unreadable state counts as not.
    """
    try:
        return (handle is not None
                and callable(getattr(handle, "call_all", None))
                and getattr(handle, "setup_key", None) is not None
                and not getattr(handle, "defunct", False)
                and not getattr(handle, "teardown_complete", False)
                and getattr(handle, "setup_cleanup_state", None) is None)
    except Exception:
        return False


def trim_actors(handle: Any, level: str = TRIM_LEVEL) -> Trimmed:
    """Ask every actor of ``handle`` to release its allocator cache.

    The call claims the mesh's render session, or joins the one active here
    (an identity ceremony's), as every clear_vram call does. Counts the actors
    that answered and reads MemAvailable on each side of the call.
    """
    from .mesh_session import mutation_render_session

    before = mesh_safety.mem_available_bytes() or 0
    with mutation_render_session(handle, timeout_s=_TRIM_CLAIM_S):
        answers = handle.call_all(TRIM_ENDPOINT, level, timeout_s=_TRIM_TIMEOUT_S)
    time.sleep(_SETTLE_S)
    after = mesh_safety.mem_available_bytes() or 0
    return Trimmed(len(list(answers or ())), before, after)


def _amended(exc: Exception, sentence: str) -> Exception:
    """The same refusal with one more sentence at the end of its first line.

    The tag and the budget figures stay at the front, where a reader that
    keeps 200 characters finds them; the type and traceback stay too. A
    refusal that arrived wrapped, a Worker's own refusal inside Monarch's
    ``ActorError``, does not take its text from ``args``, and the wrapper's
    text is "A remote actor call has failed." and the remote traceback, with
    the refusal's tag and figures at its end. So the sentence is added to the
    refusal itself (``consent_observe.refusal_text`` reads the inner
    exception), and it comes back as a typed ``StockLoadCapacityError``
    carrying that text and no traceback; ``_raise_amended`` chains it from
    the wrapper.
    """
    if isinstance(exc, CAPACITY_REFUSALS):
        head, newline, tail = str(exc).partition("\n")
        exc.args = (f"{head.rstrip()} {sentence}{newline}{tail}",)
        return exc
    from .consent_observe import refusal_text

    head, newline, tail = refusal_text(exc).partition("\n")
    return mesh_safety.StockLoadCapacityError(f"{head.rstrip()} {sentence}{newline}{tail}")


def _raise_amended(exc: Exception, sentence: str) -> NoReturn:
    """Raise ``exc`` with ``sentence`` added, chained only where it was rewrapped."""
    amended = _amended(exc, sentence)
    if amended is exc:
        raise exc from None
    raise amended from exc


def is_capacity_refusal(exc: BaseException) -> bool:
    """The default refusal test: a typed render-site capacity refusal."""
    return isinstance(exc, CAPACITY_REFUSALS)


def _first_refusal(steps: Sequence[Callable[[], object]], start: int,
                   is_refusal: Callable[[BaseException], bool] = is_capacity_refusal,
                   ) -> tuple[int, Exception] | None:
    """Run ``steps`` from ``start``; the index and refusal of the first that refuses."""
    for index in range(start, len(steps)):
        try:
            steps[index]()
        except Exception as refused:
            if not is_refusal(refused):
                raise
            return index, refused
    return None


def _swap_credit(swap: tuple[Any, Any] | None) -> swap_credit.SwapCredit:
    """What the Worker on this host drops before the render's pending load."""
    if swap is None:
        return swap_credit.NONE
    try:
        from .nodes import loader_relief

        return loader_relief.render_credit(*swap)
    except Exception as exc:  # no credit is the price as it was
        log.info("render preflight read no swap credit (%r)", exc)
        return swap_credit.NONE


def check_with_unload(steps: Sequence[Callable[[], object]], *, handle: Any = None,
                      subject: str = "this render",
                      is_refusal: Callable[[BaseException], bool] = is_capacity_refusal,
                      swap: tuple[Any, Any] | None = None) -> None:
    """Run ``steps`` in order, counting and freeing what the driver host can
    get back before one refuses.

    A capacity refusal stops the run. ``swap`` is the render's model and
    request: where its DiT load is still pending and replaces a stock resident
    in its own slot, the Worker drops that model first, so its copy is counted
    as available (``swap_credit.py``) and the steps run again from the one that
    refused, and so does every later run. If one still refuses and nothing is
    running, the driver
    unloads the models ComfyUI holds in this process and runs the steps again
    from the one that refused: the steps before it passed on a lower reading.
    If one still refuses and ``handle`` is a live mesh that completed setup,
    the actors release their allocator caches and the steps run once more
    from the one that refused then. Any other exception, and any refusal these
    steps cannot answer, propagates as it was raised. ``is_refusal`` says which
    exceptions count as a refusal; the default takes the typed render-site
    ones, and the identity gate's stock reference also takes a Worker's own
    stock refusal, which reaches the driver as remote actor text.
    """
    steps = list(steps)
    first = _first_refusal(steps, 0, is_refusal)
    if first is None:
        return
    credit = _swap_credit(swap)
    if not credit:
        _after_refusal(steps, *first, handle, subject, is_refusal)
        return
    counted = (f"the Worker on {credit.box or 'this host'} drops "
               f"{credit.dropped or 'the old model'} before this load, which frees at least "
               f"{credit.gib} GiB{credit.note}")
    with swap_credit.applying(credit) as applied:
        again = _first_refusal(steps, first[0], is_refusal)
        if again is None:
            if applied.used:  # a price that charges a load took the credit
                log.info("render preflight: %s fits once %s", subject, counted)
            return
        try:
            _after_refusal(steps, *again, handle, subject, is_refusal)
        except Exception as exc:
            if not is_refusal(exc) or not applied.used:
                raise
            raw = mesh_safety.mem_available_bytes()
            read = "" if raw is None else f" The box itself read {raw / 2**30:.1f} GiB of MemAvailable."
            _raise_amended(exc, f"The figures above already count that {counted}.{read}")


def _after_refusal(steps: Sequence[Callable[[], object]], index: int,
                   refused: Exception, handle: Any, subject: str,
                   is_refusal: Callable[[BaseException], bool] = is_capacity_refusal) -> None:
    """Unload, then trim, running ``steps`` again from ``index`` after each,
    or raise the refusal that stands, saying what was tried."""
    if mesh_safety.env_enabled(DRIVER_UNLOAD_DISABLE_ENV):
        raise refused
    mm = comfy_model_management()
    held = held_models(mm) if mm is not None else []
    busy = busy_reason(handle)
    # The trim goes to the fleet attached now, whose actors hold the caches.
    try:
        handle = serving_fleet(handle)
    except Exception:
        handle = None
    unloaded: Unloaded | None = None
    if held:
        names = tuple(name for _entry, name in held)
        if busy:
            _raise_amended(refused, (
                f"The driver did not unload the {_models(names)} ComfyUI holds on "
                f"this host before refusing, because {busy}; queue again once that "
                "work ends."))
        try:
            outcome = unload(mm, held)
        except Exception as exc:
            log.warning("driver unload failed before refusing %s: %r", subject, exc)
            _raise_amended(refused, (
                f"The driver tried to unload the {_models(names)} ComfyUI holds on "
                f"this host before refusing, and the unload failed "
                f"({type(exc).__name__}), so the figures above are the reading "
                "before it."))
        if outcome.names:
            again = _first_refusal(steps, index, is_refusal)
            if again is None:
                log.info(outcome.log_line(subject))
                return
            index, refused = again
            unloaded = outcome
        elif trimmable(handle):
            log.warning("driver unload: ComfyUI kept every model on this host; "
                        "asking the actors to release their caches before "
                        "refusing %s", subject)
        else:
            log.warning("driver unload: ComfyUI kept every model on this host; "
                        "refusing %s on the first reading", subject)
    elif busy:
        raise refused
    if not trimmable(handle):
        if unloaded is not None:
            _raise_amended(refused, unloaded.refusal_sentence())
        raise refused
    try:
        trimmed = trim_actors(handle)
    except Exception as exc:
        log.warning("actor trim failed before refusing %s: %r", subject, exc)
        _raise_amended(refused, _trim_failed_sentence(unloaded, exc))
    again = _first_refusal(steps, index, is_refusal)
    log.info(trimmed.log_line(subject, unloaded, fits=again is None))
    if again is not None:
        _raise_amended(again[1], trimmed.refusal_sentence(unloaded))
