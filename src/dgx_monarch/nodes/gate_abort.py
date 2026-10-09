"""Settle aborted FSDP clean-reload proofs without granting authority.

Typed refusals retain their class; untyped aborts report an INCONCLUSIVE
outcome. A terminal row closes the ceremony's RETESTING transaction.
``gate_fsdp`` re-exports this API.

Builders return exceptions for the ceremony to raise through its runtime
mapping. The raise-site ledger therefore does not discover them; the sweep
collects untyped aborts through ``UNTAGGED_LITERALS``.
"""
from __future__ import annotations

from typing import Any

from ..log import get_logger
from ..refusal import RefusalClass, RefusalTag, escape_body_tags, refusal
from ..transfer_utils import safe_call
from . import gate_process_state as _state

log = get_logger(__name__)

_ABORT_SENTENCE = "FSDP clean-reload proof aborted before a terminal verdict"
_ABORT_REASON = "the FSDP clean-reload proof aborted before a terminal verdict"
_MAX_DETAIL_CHARS = 200

# The operator entry for the abort, the restated boundary and `gate --repair`.
TROUBLESHOOTING = 92

# Every waivable class K guard has its own remedy and its own entry, and they
# do not agree: a sol-attn refusal is answered by an exact kernel, never by
# token counts. Keyed on the family prefix, because every raise site is family
# scoped. Read-only after import.
_FAMILY_REMEDIES: dict[str, tuple[str, tuple[int, ...]]] = {
    "ring_pad": (
        "Render it on a topology without fsdp, or at token counts that need "
        "no divisibility padding.",
        (21, 49),
    ),
    "sol_attn": (
        "Pick a SAGE_* kernel or TORCH_FLASH for exact framing, or render it "
        "on a topology without fsdp.",
        (84,),
    ),
}
# A class K guard added later inherits the boundary, not the ring pad remedy.
_DEFAULT_REMEDY = (
    "Render it on a topology without fsdp, or at a shape this guard does not "
    "refuse.",
    (),
)


def _inner_text(exc: BaseException) -> str:
    """The refused text, without stringifying a wrapper; never raises."""
    try:
        from ..consent_observe import refusal_text

        return refusal_text(exc)
    except BaseException:
        return ""


def settled_refusal_tag(exc: BaseException) -> RefusalTag | None:
    """The class tag of a refusal that answered, or None for anything else.

    Only the strict leading parser: a tag-shaped artifact name inside a crash
    diagnostic never proves a guard decided this render. Never raises, even when
    the diagnostic cannot be read: the caller still has residency to clean up.
    """
    try:
        from ..refusal import parse_leading_refusal_tag

        candidates = [_inner_text(exc)]
        args = getattr(exc, "args", ())
        if args and isinstance(args[0], str):
            candidates.append(args[0])
        for text in candidates:
            if not text:
                continue
            tag = parse_leading_refusal_tag(text)
            if tag is not None:
                return tag
    except BaseException:
        return None
    return None


def aborted_proof_error(exc: BaseException):
    """The untyped abort refusal, carrying what stopped the proof.

    The verdict is INCONCLUSIVE rather than the ERROR default, because that is
    what the retry reports and what the terminal row records.
    """
    from .gate_fsdp import FsdpGateProofError

    try:
        detail = f"{type(exc).__name__}: {_inner_text(exc)}"
    except BaseException:
        detail = "unreadable"
    # Escape first, then truncate: a tag split by the cut is not a tag, and a
    # tag escaped after the cut could still be formed from the tail.
    return FsdpGateProofError(
        f"{_ABORT_SENTENCE}: {escape_body_tags(detail)[:_MAX_DETAIL_CHARS]}",
        verdict="INCONCLUSIVE",
    )


def ceremony_waiver_boundary_error(tag: RefusalTag, exc: BaseException):
    """Restate a waivable class K refusal at the identity-check boundary where waivers do not apply.

    A ceremony strips every accuracy waiver on purpose, so the worker's own
    card would offer a Grant that returns the same card each time. The class,
    the guard and the guard's own remedy stay; only the waiver half is replaced.
    The guard id is interpolated at runtime because this site is not the
    guard's own raise site.
    """
    from .gate_fsdp import FsdpGateProofError

    del exc  # the guard id is the actionable half; the prose is in the log
    guard = tag.guard or "unnamed"
    remedy, guard_entries = _FAMILY_REMEDIES.get(
        guard.split(":")[0], _DEFAULT_REMEDY)
    entries = ", ".join(
        f"#{number}" for number in (TROUBLESHOOTING, 55, *guard_entries))
    text = refusal(
        RefusalClass.KNOWN_WRONG,
        f"the class K guard {guard} refused inside the FSDP clean-reload "
        "proof. A ceremony renders without accuracy waivers on purpose, so a "
        "granted waiver cannot be spent there and this refusal cannot be "
        f"waived. {remedy} See docs/TROUBLESHOOTING.md {entries}.",
        waivable=False,
    )
    return FsdpGateProofError(text, verdict="INCONCLUSIVE")


def _refusal_type(exc: BaseException) -> str:
    """A bounded name for what refused, never the refusing text itself."""
    try:
        from ..fsdp_reload_price import capacity_classification

        return capacity_classification(exc)
    except BaseException:
        return f"{type(exc).__name__[:64]}/unread"


def record_proof_stop(
    ledger: Any,
    key: str,
    artifacts: Any,
    commit: Any,
    capability_context: Any,
    unet_name: str,
    *,
    origin: str,
    run_id: str,
    tag: RefusalTag | None,
    exc: BaseException,
) -> None:
    """Close the ceremony's RETESTING transaction with one terminal row.

    State-neutral by construction: the ledger maps RETESTING and INCONCLUSIVE
    to the same ``inconclusive`` state, so this supersedes the open retest
    without changing what is refused. No ``cross_mode``: an FSDP proof runs
    no cross-residency leg, and the field's ``CAPACITY`` value marks a
    stock-load capacity refusal, which that leg or ``record_capacity_stop``
    records and which never reaches this writer. No ``inconclusive_kind``:
    its ``no_material`` value would turn the row into a skip grant. Never
    raises, and never carries the refusal prose or a remote traceback.
    """
    row: dict[str, Any] = {
        "model": unet_name,
        "loras": 0,
        "origin": origin,
        "run_id": run_id,
        "max_abs_latent_diff": None,
        "quarantine_levers": [],
        "measured": None,
        # Not `capacity_detail`, the capacity row's name for the same value:
        # this stop is usually class P or K, not a capacity refusal.
        "refusal_type": _refusal_type(exc),
        "inconclusive_reasons": [_ABORT_REASON],
    }
    if tag is not None:
        row["inconclusive_reasons"] = [
            "the FSDP clean-reload proof met a typed class "
            f"{tag.refusal_class.value} refusal"
        ]
        row["refusal_class"] = tag.refusal_class.value
        row["refusal_guard"] = tag.guard or "none"
    try:
        ledger.record(key, artifacts, commit, "INCONCLUSIVE", row,
                      capability_context)
    except Exception as record_exc:
        safe_call(log.warning,
                  "could not persist the FSDP clean-reload stop row: %r",
                  record_exc)


def retract_process_gate_verdicts(
    tokens: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
) -> None:
    """Withdraw the provisional denial from normal and Fleet process caches.

    ``record_process_gate_verdicts`` updates both maps and ``maybe_auto_gate``
    blocks every non-PASS entry. Clear both along with the no-material marker so
    no stale classification survives a removed verdict. Removal requires no
    cache-size trimming.
    """
    from .common import gate_inconclusive

    unique_tokens = tuple(dict.fromkeys(tokens))
    with _state._AUTO_GATE_LOCK:
        gate_inconclusive.remember_kind(unique_tokens, None)
        denials = dict(_state._PROCESS_GATE_DENIALS)
        for token in unique_tokens:
            denials.pop(token, None)
        # One copy-on-write store, as the publish does: an async interruption
        # exposes all old or all new revocations.
        _state._PROCESS_GATE_DENIALS = denials
        for token in unique_tokens:
            _state._AUTO_GATE_SESSION.pop(token, None)


def close_aborted_proof(
    runtime: Any,
    *,
    capacity_row: tuple[Any, ...] | None,
    origin: str,
    run_id: str,
    tag: RefusalTag | None,
    exc: BaseException,
    ceremony_error: BaseException,
    retest_tokens: Any,
    row_written: bool,
) -> bool:
    """Close an aborted FSDP proof's ledger and process-cache transaction.

    Return whether the provisional denial was actually removed. No removal occurs
    before publication, for an untyped refusal, or when withdrawal fails.

    Read the recorded guard from the original exception and the withdrawal rule
    from the exception to be raised. Catch branches alone do not identify a typed
    refusal: they also catch missing unload evidence and incomplete proof errors.
    A failed withdrawal leaves denial intact and must not replace the original
    refusal. Already-waiting renders receive an untyped result; later sequential
    attempts receive the settled class.

    ``row_written`` also covers an abort before a transaction was opened.
    """
    settled = settled_refusal_tag(ceremony_error)
    if capacity_row is not None and not row_written:
        runtime["record_proof_stop"](
            *capacity_row, origin=origin, run_id=run_id, tag=tag, exc=exc)
    if settled is None or not retest_tokens:
        return False
    try:
        runtime["_retract_process_gate_verdicts"](retest_tokens)
    except BaseException as retract_exc:
        safe_call(log.warning,
                  "the FSDP clean-reload prejudgement was not withdrawn (%r); "
                  "the process-local denial stands", retract_exc)
        return False
    return True


def log_aborted_proof(logger: Any, tag: RefusalTag | None, retracted: bool,
                      exc: BaseException) -> None:
    """Report the aborted proof's outcome and remaining denial state.

    Only a successfully withdrawn provisional denial allows the next queue to
    reach the original guard again.
    """
    if tag is None:
        safe_call(logger.error,
                  "FSDP clean-reload Gate aborted (%r); denying the exact FSDP "
                  "capability without residency quarantine", exc)
    elif retracted:
        safe_call(logger.error,
                  "FSDP clean-reload Gate met a settled class %s refusal from "
                  "guard %s (%r); that combination is answered and the "
                  "process-local denial was retracted",
                  tag.refusal_class.value, tag.guard or "none", exc)
    else:
        safe_call(logger.error,
                  "FSDP clean-reload Gate met a settled class %s refusal from "
                  "guard %s (%r); that combination is answered and no "
                  "process-local denial was retracted",
                  tag.refusal_class.value, tag.guard or "none", exc)
