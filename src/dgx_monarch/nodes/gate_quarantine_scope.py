"""Scope residency quarantine to the combination whose ceremony aborted.

A waivable class-K accuracy refusal proves nothing about residency, so it
changes no levers or worker policy. Classification uses the refusal tag and
frozen guard vocabulary, never message prose.

Other unproven aborts disable the combination's residency levers. The record
retains the shared graph policy dict and restores its original values when a
different combination uses it; an aborted proof cannot quarantine unrelated
artifact bytes.
"""
from __future__ import annotations

from typing import Any

from ..log import get_logger

log = get_logger(__name__)

# What a quarantine writes. A lever that no longer reads False was set by
# somebody after the quarantine, so it is not this record's to give back.
_OFF = False

# Stands for a lever the graph never named. Absent means worker-side auto, and
# restoring it as an explicit None would name a value no ceremony ever proved
# and would change every capability context the graph builds afterwards.
_ABSENT = object()

# Session lifetime, one record per policy dict. A record retires when a
# different combination takes its levers back. The list is bounded because each
# record holds its graph's policy dict alive; past the limit the oldest record
# drops and its graph keeps the quarantine for every combination.
_RECORDS: list[dict[str, Any]] = []
_RECORD_LIMIT = 16


def measured_abort_guard(cause: BaseException | None) -> str | None:
    """The waivable class K guard an aborted ceremony answered on, or None.

    Never raises: an abort whose class cannot be read counts as unproven, the
    fail-closed answer.
    """
    if cause is None:
        return None
    try:
        from ..adapters.sol_attention import is_sol_waiver_refusal
        from ..refusal import GUARDS, RefusalClass
        from .gate_abort import settled_refusal_tag

        tag = settled_refusal_tag(cause)
        guard = getattr(tag, "guard", None)
        spec = GUARDS.get(guard) if isinstance(guard, str) else None
        if (tag is not None and tag.refusal_class is RefusalClass.KNOWN_WRONG
                and tag.waivable and spec is not None and spec.waivable_now):
            return str(guard)
        if is_sol_waiver_refusal(cause):
            # The kernel exemption this rule generalizes, kept as its own read
            # so a sol abort answers here whatever text carried its tag.
            return "sol_attn"
    except BaseException as exc:  # An abort nobody can read is an unproven one.
        log.warning("an aborted ceremony's refusal class was not read (%r)", exc)
    return None


def abort_class(cause: BaseException | None) -> str:
    """What the operator's log calls the thing that stopped this ceremony."""
    if cause is None:
        return "an unproved verdict"
    try:
        from .gate_abort import settled_refusal_tag

        tag = settled_refusal_tag(cause)
        if tag is not None:
            return f"class {tag.refusal_class.value} {tag.guard or 'no guard'}"
        return f"an untyped abort, {type(cause).__name__}"
    except BaseException:  # A log line never replaces the caller's answer.
        return "an abort of unreadable class"


def combination(model: Any) -> str:
    """This model combination's gate ledger key, or empty when unreadable."""
    try:
        from ..gate_ledger import combo_key

        loras = getattr(model, "loras", ()) or ()
        return combo_key(
            str(model.unet_name), dict(getattr(model, "options", None) or {}),
            [str(entry["name"]) for entry in loras])
    except BaseException:
        return ""


def combination_label(model: Any) -> str:
    """The combination as the driver log names it: file first, then key."""
    name = str(getattr(model, "unet_name", "") or "an unnamed model")
    key = combination(model)
    return f"{name} (combination {key})" if key else name


def _record_for(worker_args: Any) -> dict[str, Any] | None:
    for record in _RECORDS:
        if record["worker_args"] is worker_args:
            return record
    return None


def write_scoped_quarantine(
    model: Any, levers: dict[str, bool], cause: BaseException | None = None,
) -> list[str]:
    """Turn the unproven levers off for one combination, and say whose."""
    guard = measured_abort_guard(cause)
    if guard is not None:
        log.error(
            "auto-gate could not complete for %s: the class K guard %s refused "
            "inside the ceremony. That guard is measured, so it answers this "
            "combination's accuracy and takes no residency lever from the "
            "graph", combination_label(model), guard)
        return []
    worker_args = model.mesh.worker_args
    changed = [lever for lever in levers if worker_args.get(lever) is not _OFF]
    requested = {
        lever: (worker_args[lever] if lever in worker_args else _ABSENT)
        for lever in changed
    }
    # One dict operation is the graph's atomic quarantine publication.  An
    # asynchronous exception at the Python line boundary therefore cannot
    # expose one lever disabled while the other remains risky.
    worker_args.update(**levers)
    if changed:
        _remember(model, worker_args, requested)
        log.error(
            "auto-gate could not complete for %s on %s; %s disabled for that "
            "combination for the rest of this session",
            combination_label(model), abort_class(cause), ", ".join(changed))
    return changed


def _remember(model: Any, worker_args: Any, requested: dict[str, Any]) -> None:
    """Bind this quarantine to the combination whose ceremony aborted."""
    try:
        record = _record_for(worker_args)
        if record is None:
            if len(_RECORDS) >= _RECORD_LIMIT:
                del _RECORDS[0]
            record = {"worker_args": worker_args, "requested": {}}
            _RECORDS.append(record)
        # Keep the first value seen for each lever. A second quarantine on the
        # same graph would otherwise record what the first one wrote and hand
        # the next combination a quarantine back instead of the graph's ask.
        for lever, value in requested.items():
            record["requested"].setdefault(lever, value)
        record["combination"] = combination(model)
    except BaseException as exc:  # An unscoped quarantine is still a quarantine.
        log.warning(
            "an unproven quarantine was not bound to its combination (%r)", exc)


def restore_requested_levers(model: Any) -> list[str]:
    """Give a different combination the levers the graph asked for.

    The aborting combination keeps its quarantine: it is the one thing here
    left unproven, and its own next ceremony re-decides it. Every other
    combination starts from the graph's request, because a ceremony that never
    looked at those bytes proved nothing about them.
    """
    restored: list[str] = []
    try:
        worker_args: Any = getattr(
            getattr(model, "mesh", None), "worker_args", None)
        record = None if worker_args is None else _record_for(worker_args)
        if record is None or record.get("combination") == combination(model):
            return []
        for lever, value in record["requested"].items():
            if worker_args.get(lever) is not _OFF:
                continue  # Somebody set this after the quarantine; leave it.
            if value is _ABSENT:
                worker_args.pop(lever, None)
            else:
                worker_args[lever] = value
            restored.append(lever)
        _RECORDS.remove(record)
    except BaseException as exc:  # A stale quarantine is the safe direction.
        log.warning("the graph's requested levers were not restored (%r)", exc)
        return []
    if restored:
        log.warning(
            "%s asked for %s, which an earlier combination's aborted ceremony "
            "turned off; restoring them, because that abort proved nothing about this combination",
            combination_label(model), ", ".join(restored))
    return restored


def reset_quarantine_scope() -> None:
    """Forget every remembered quarantine (driver teardown, tests)."""
    _RECORDS.clear()
