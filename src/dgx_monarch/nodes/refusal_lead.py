"""Lead the prompt error with a Worker's refusal, and say when the driver forced stock.

A Worker's refusal reaches the driver inside Monarch's ``ActorError``, whose
text is "Actor call ... failed." and the remote traceback, with the tagged
sentence at its end; ComfyUI shows that text as the prompt error. The node
methods that can meet one (the samplers, Fleet KSampler, the UNET loaders) run
under ``lead_with_refusal``. A refusal whose text leads with a dgxm tag is
raised as itself, chained from the wrapper, as ``driver_unload`` reads it
through ``consent_observe.refusal_text`` (#139): the prompt error then leads
with the sentence and the card it names, and ComfyUI's log keeps the remote
traceback through the chain. Everything that decides on the wrapper (the sample
lease, eviction, card registration) has already run below the node method.

When the Init node asks for ``slab_weights=on`` and the combination has no
exact Identity Gate PASS, the driver dispatches on stock residency and notes it
here (``consent_projection.capacity_consent_authorization``). A Worker that then
refuses the stock load saw only the setting arrive off, so its sentence says
slab_weights is off and asks for ``slab_weights=on``. The same seam puts the
cause and the remedy in place of those two clauses.
"""
from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any, TypeVar, cast

from .. import mesh_safety
from ..consent_observe import refusal_text
from ..refusal import RefusalClass, parse_leading_refusal_tag, read_rank

# The Worker's words for a render that arrived on stock (actor/worker_env
# slab_mode_reason and actor/rescue_offer rescue_blocker). The tests hold both
# against those functions, so a one-sided edit fails there, not here.
SLAB_OFF_BLOCKER = ("Slab rescue is unavailable here because slab_weights is off in "
                    "the effective worker args.")
EXPLICIT_STOCK_OPTIONS = ("clear the quarantine with an identity ceremony, or set "
                          "slab_weights=on when the combination is not quarantined, or ")
# The gate's stock reference must load where this one refused, so the remedy
# says how to make it fit: the same model in a lighter graph. No box is named
# here (box_names.py); the Worker's own sentence before it names the load.
GATE_REMEDY = ("run Identity Gate (DGX Monarch) on this model with slab_weights=on in a "
               "graph whose stock reference fits the box that refused, for example with an optional "
               "driver model such as a prompt upsampler switched off, then queue the render "
               "again, or ")
# The state a dual-model render notes: the gate proves one model, so no PASS
# can authorize slab for it and the remedy names none.
DUAL_MODEL = "two models"

# Render lifetime, one per ComfyUI executor thread: set while the render is
# authorized, read by the node method's own wrapper, and cleared at its entry
# and exit, so no note outlives the node call that wrote it.
_NOTE = threading.local()

_Method = TypeVar("_Method", bound=Callable[..., Any])


def note_forced_stock(state: object) -> None:
    """This node call's render runs stock although slab_weights=on: no Gate PASS
    for ``state``, or ``DUAL_MODEL``. The first note of a node call stands."""
    if forced_stock_state() is None:
        _NOTE.state = str(state or "unknown")


def note_if_slab_dropped(requested: Any, dispatched: Any, *, dual: bool) -> None:
    """Note a dispatch that turned off the slab_weights=on the graph asked for.

    Every stock fallback of a normal render comes here (a missing PASS, a
    failed authorization, a second model); the no-PASS path has already noted
    its ledger state, and the first note stands.
    """
    if dict(dispatched or {}).get("slab_weights") is False and dict(
            requested or {}).get("slab_weights") is True:
        note_forced_stock(DUAL_MODEL if dual else "unknown")


def forced_stock_state() -> str | None:
    return getattr(_NOTE, "state", None)


def _clear() -> None:
    _NOTE.state = None


def forced_stock_text(text: str, state: str) -> str:
    """The stock refusal with the real cause and remedy in place of the setting."""
    lead = "Slab residency was not used, although the Init node sets slab_weights=on: "
    if state == DUAL_MODEL:
        cause = (f"{lead}a render with two models runs on stock residency, because the "
                 "Identity Gate proves one model at a time.")
        remedy = ""
    else:
        cause = (f"{lead}this combination has no Identity Gate PASS for its current state "
                 f"({state}), so the driver ran this render on stock residency.")
        remedy = GATE_REMEDY
    return text.replace(SLAB_OFF_BLOCKER, cause, 1).replace(EXPLICIT_STOCK_OPTIONS, remedy, 1)


def surfaced(exc: BaseException, forced: str | None = None) -> BaseException | None:
    """The refusal to raise for ``exc``: the inner one, ``exc`` reworded, or None as it is.

    The point that received the error has already put its box ahead of the
    Worker's sentence (``worker_box.py``); a rank mark that path did not take
    off is dropped here, so no prompt error shows one.
    """
    inner = getattr(exc, "exception", None)
    wrapped = isinstance(inner, Exception)
    raw = refusal_text(exc) if wrapped else str(exc)
    text = read_rank(raw)[0]
    tag = parse_leading_refusal_tag(text)
    if tag is None:
        return None
    worded = (forced_stock_text(text, forced)
              if forced and tag.refusal_class is RefusalClass.CAPACITY else text)
    if not wrapped and worded == raw:
        return None
    target = cast(BaseException, inner if wrapped else exc)
    if str(target) != worded:
        target.args = (worded,)
    if str(target) != worded:  # a type that does not print its args
        target = (mesh_safety.StockLoadCapacityError(worded)
                  if tag.refusal_class is RefusalClass.CAPACITY else RuntimeError(worded))
    return target


def lead_with_refusal(method: _Method) -> _Method:
    """Run a node method so a Worker refusal reaches the prompt error sentence first."""
    @functools.wraps(method)
    def lead(*args: Any, **kwargs: Any) -> Any:
        _clear()
        try:
            return method(*args, **kwargs)
        except Exception as exc:
            refusal = surfaced(exc, forced_stock_state())
            if refusal is None or refusal is exc:
                raise
            raise refusal from exc
        finally:
            _clear()

    return cast(_Method, lead)
