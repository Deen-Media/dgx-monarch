"""Stop an identity ceremony whose request cannot fit the path it would prove.

An ordinary residency ceremony (no FSDP clean-reload scope) runs the render
site's capacity checks for the whole request first, priced for the residency
it is about to prove (:func:`ceremony_activation_preflight`), and only then
writes its RETESTING row (``gate_ceremony``), the same place the FSDP proof
prices its clean reload. A refusal there means no proof render could finish,
so the ceremony runs none. It records one terminal INCONCLUSIVE row in the
vocabulary of the FSDP proof's capacity stop
(``fsdp_reload_price.record_capacity_stop``) and raises
:class:`CeremonyCapacityStop`, with no RETESTING row, no process-local verdict
and no lever taken. On the automatic path the render it gates then dispatches
on stock residency, because no PASS authorizes more, and the next request of
the combination meets the same check (docs/DESIGN.md, "Where the ceremony
cannot help").
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import residency_mode
from ..driver_unload import CAPACITY_REFUSALS
from ..error_utils import failure_text, safe_call
from ..fsdp_reload_price import (
    CAPACITY_CROSS_MODE,
    CAPACITY_STOP_VERDICT,
    capacity_classification,
)
from ..log import get_logger
from . import render_preflight

log = get_logger(__name__)


class CeremonyCapacityStop(RuntimeError):
    """The ceremony stopped before its proof render: the request cannot fit.

    Its text is the refusal's own, class tag and budget first, so the guard's
    class and remedy stay on the wire, with one sentence added. Not a
    ``StockLoadCapacityError``: the refusal it carries may be a render memory
    refusal, which stays outside that family on purpose (render_memory_price).
    """


def dit_already_loaded(model: Any) -> bool:
    """Whether the render site's memos say this DiT is loaded on the Workers:
    a finished render left it there, or a pipelined submission dispatched it.
    The memo of a pending load the driver cleared never counts, because that
    load may not have happened."""
    unet_name = getattr(model, "unet_name", None)
    return isinstance(unet_name, str) and unet_name in (
        render_preflight._LAST_RENDERED_UNET, render_preflight._LAST_SUBMITTED_UNET)


def ceremony_activation_preflight(model: Any, request: dict, latent: dict) -> None:
    """The render site's capacity checks for the path a ceremony would prove.

    ``model`` carries the ceremony's own worker policy, so the driver footprint
    check prices that residency. The DiT is credited when it is loaded already
    and charged as a pending load when it is not: the ceremony's load has not
    happened, and the memo the render site's own check stamped a moment
    earlier for this request records a cleared charge, not a load. An explicit
    topology preset still credits it inside each check, because the loader node
    loaded it. A refusal raises as at the render site, after the same rescue.
    """
    render_preflight.activation_footprint_preflight_for_request(
        model, request, latent, weights_loaded=dit_already_loaded(model))


def stop_unless_request_fits(
    model: Any, request: dict, latent: dict, capacity_row: tuple[Any, ...], *,
    origin: str, run_id: str, loras: int, worker_args: Mapping[str, Any] | None,
) -> None:
    """Raise :class:`CeremonyCapacityStop` when ``request`` cannot fit.

    ``capacity_row`` is the ledger, key, artifacts, commit, capability context
    and checkpoint name the ceremony bound before this check. The light auto
    request keeps the request's conditioning and latent, the only parts the
    checks read. Anything but a capacity refusal fails open, as every preflight
    does: the proof render's own preflight meets the same checks and raises
    what it raises today.
    """
    try:
        ceremony_activation_preflight(model, request, latent)
    except CAPACITY_REFUSALS as refused:
        raise record_capacity_stop(
            *capacity_row, origin=origin, run_id=run_id, loras=loras,
            worker_args=worker_args, refused=refused) from refused
    except Exception as unpriced:
        safe_call(log.warning, "identity gate: the capacity check before the "
                  "proof render failed open (%r)", unpriced)


def proved_path(worker_args: Mapping[str, Any] | None) -> str:
    """The residency path a ceremony under ``worker_args`` would prove."""
    if residency_mode.requested(worker_args):
        return "comfy-managed residency"
    if dict(worker_args or {}).get("slab_weights") is False:
        return "the lazy-swap path"
    return "the slab path"


def record_capacity_stop(
    ledger: Any,
    key: str,
    artifacts: Any,
    commit: Any,
    capability_context: Any,
    unet_name: str,
    *,
    origin: str,
    run_id: str,
    loras: int,
    worker_args: Mapping[str, Any] | None,
    refused: BaseException,
) -> CeremonyCapacityStop:
    """Record the stop's one INCONCLUSIVE row and return the error to raise.

    The row names what refused in the bounded classification the FSDP stop
    uses, never the refusing text (``capacity_classification`` says why); the
    operator reads that text in the error and the driver log. A failed append
    is logged and the stop still stands: the next request prices again.
    """
    path = proved_path(worker_args)
    row = {
        "model": unet_name,
        "loras": int(loras),
        "origin": origin,
        "run_id": run_id,
        "max_abs_latent_diff": None,
        "quarantine_levers": [],
        "cross_mode": CAPACITY_CROSS_MODE,
        "inconclusive_reasons": [
            f"the request's capacity check refused {path} before the proof "
            "render, so the ceremony ran none"],
        "measured": None,
        "capacity_detail": capacity_classification(refused) or None,
    }
    try:
        ledger.record(key, artifacts, commit, CAPACITY_STOP_VERDICT, row,
                      capability_context)
    except Exception as record_exc:
        safe_call(log.warning,
                  "could not persist the identity gate's capacity row: %r",
                  record_exc)
    head, newline, tail = failure_text(refused).partition("\n")
    return CeremonyCapacityStop(
        f"{head.rstrip()} The identity gate ran no proof render: this request "
        f"cannot fit {path}, so it recorded one INCONCLUSIVE capacity row and "
        f"took no residency lever.{newline}{tail}")
