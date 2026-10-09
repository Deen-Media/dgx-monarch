"""Fail-closed ownership for resources from model loads that did not complete."""
from __future__ import annotations

import gc
import traceback
from typing import Any

from ..log import get_logger
from ..transfer_utils import (
    failure_summary,
    raise_with_distinct_cause,
    reconcile_error,
)

log = get_logger(__name__)

# This module owns uncertain mappings until a later explicit process-wide
# unload proves that no model Parameter can still point into them. Proc
# lifetime, unkeyed, and drained one entry at a time by a confirmed close,
# never by a timer: an entry that ages out is a mapping closed under a live
# tensor.
_RETAINED_FAILED_LOAD_SLABS: list[tuple[Any, BaseException | None]] = []
# Some failed loads own non-slab resources too (for example a pinned checkpoint
# alias). They follow the same rule as slab mappings: do not close them until a
# process-wide unload proves Comfy cannot still refer to them. Same proc
# lifetime and the same confirmed-close drain; `release_after_explicit_unload`
# empties both lists in one step.
_RETAINED_FAILED_LOAD_RESOURCES: list[tuple[Any, BaseException | None]] = []
# Proc lifetime, one slot, and a third piece of state beside those two lists:
# any uncertain cleanup latches it, and only a cleanup that leaves both lists
# empty clears it, so retained ownership and a clear latch cannot coexist.
_FAILED_LOAD_CLEANUP_POISONED = False
# Proc lifetime. A clean outer provisional transaction can contain nested
# owners. If the outer confirms first, clear only when every exact owner is
# gone; a pre-existing poison never grants this authority.
_PROVISIONAL_CLEAR_WHEN_EMPTY = False


def retain(slab: Any, error: BaseException | None = None) -> None:
    for index, (owned, owned_error) in enumerate(_RETAINED_FAILED_LOAD_SLABS):
        if owned is slab:
            if owned_error is None and error is not None:
                _RETAINED_FAILED_LOAD_SLABS[index] = (slab, error)
            return
    _RETAINED_FAILED_LOAD_SLABS.append((slab, error))


def retained_count() -> int:
    return len(_RETAINED_FAILED_LOAD_SLABS)


def retained_resource_count() -> int:
    return len(_RETAINED_FAILED_LOAD_RESOURCES)


def cleanup_poisoned() -> bool:
    return _FAILED_LOAD_CLEANUP_POISONED


def cleanup_pending() -> bool:
    return bool(
        _FAILED_LOAD_CLEANUP_POISONED
        or _RETAINED_FAILED_LOAD_SLABS
        or _RETAINED_FAILED_LOAD_RESOURCES
    )


def _forget(slab: Any) -> None:
    for index, (owned, _error) in enumerate(_RETAINED_FAILED_LOAD_SLABS):
        if owned is slab:
            _RETAINED_FAILED_LOAD_SLABS.pop(index)
            return


def _retain_resource(resource: Any, error: BaseException | None = None) -> None:
    for index, (owned, owned_error) in enumerate(_RETAINED_FAILED_LOAD_RESOURCES):
        if owned is resource:
            if owned_error is None and error is not None:
                _RETAINED_FAILED_LOAD_RESOURCES[index] = (resource, error)
            return
    _RETAINED_FAILED_LOAD_RESOURCES.append((resource, error))


def prepublish_resource(resource: Any) -> bool:
    """Poison reuse and retain an owner at process level before its first side effect."""
    global _FAILED_LOAD_CLEANUP_POISONED

    poisoned_before = _FAILED_LOAD_CLEANUP_POISONED
    _FAILED_LOAD_CLEANUP_POISONED = True
    _retain_resource(resource)
    return poisoned_before


def retain_failed_load_resource(
    resource: Any,
    error: BaseException | None = None,
) -> None:
    """Poison reuse and durably own a resource from partial construction."""
    global _FAILED_LOAD_CLEANUP_POISONED

    _FAILED_LOAD_CLEANUP_POISONED = True
    _retain_resource(resource, error)


def _forget_resource(resource: Any) -> None:
    for index, (owned, _error) in enumerate(_RETAINED_FAILED_LOAD_RESOURCES):
        if owned is resource:
            _RETAINED_FAILED_LOAD_RESOURCES.pop(index)
            return


def confirm_prepublished_resource(
    resource: Any,
    poisoned_before: bool,
) -> None:
    """Forget an adopted/closed provisional owner and clear only our poison."""
    global _FAILED_LOAD_CLEANUP_POISONED, _PROVISIONAL_CLEAR_WHEN_EMPTY

    if not poisoned_before:
        _PROVISIONAL_CLEAR_WHEN_EMPTY = True
    _forget_resource(resource)
    if (
        _PROVISIONAL_CLEAR_WHEN_EMPTY
        and not _RETAINED_FAILED_LOAD_SLABS
        and not _RETAINED_FAILED_LOAD_RESOURCES
    ):
        _FAILED_LOAD_CLEANUP_POISONED = False
        _PROVISIONAL_CLEAR_WHEN_EMPTY = False


def _close_or_retain_after_explicit_unload(
    resource: Any,
    context: str,
    retain_owner: Any,
    forget_owner: Any,
) -> None:
    """Close after global unload under the supplied durable owner registry."""
    global _FAILED_LOAD_CLEANUP_POISONED, _PROVISIONAL_CLEAR_WHEN_EMPTY

    poisoned_before = _FAILED_LOAD_CLEANUP_POISONED
    _FAILED_LOAD_CLEANUP_POISONED = True
    # Publish ownership before close(): an interrupt can arrive after the
    # underlying release but before Python reports its outcome. The registry is
    # removed only after a confirmed return, so no uncertain resource is lost.
    retain_owner(resource)
    try:
        resource.close()
    except BaseException as error:
        retention_error: BaseException | None = None
        try:
            retain_owner(resource, error)
        except BaseException as caught:
            retention_error = caught
        try:
            log.warning("%s close failed (%r); retaining ownership", context, error)
        except BaseException:
            pass
        if retention_error is not None:
            strongest, cause = reconcile_error(
                error,
                retention_error,
                "failed close ownership publication also failed",
            )
            raise_with_distinct_cause(strongest, cause)
        raise
    forget_owner(resource)
    if (
        (not poisoned_before or _PROVISIONAL_CLEAR_WHEN_EMPTY)
        and not _RETAINED_FAILED_LOAD_SLABS
        and not _RETAINED_FAILED_LOAD_RESOURCES
    ):
        _FAILED_LOAD_CLEANUP_POISONED = False
        _PROVISIONAL_CLEAR_WHEN_EMPTY = False


def close_or_retain_after_explicit_unload(resource: Any, context: str) -> None:
    """Close a non-slab resource, retaining ownership on any failure."""
    _close_or_retain_after_explicit_unload(
        resource, context, _retain_resource, _forget_resource
    )


def close_slab_or_retain_after_explicit_unload(slab: Any, context: str) -> None:
    """Close a slab after global unload, retaining ownership on any failure."""
    _close_or_retain_after_explicit_unload(slab, context, retain, _forget)


def _clear_exception_frames(error: BaseException) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend(
            linked for linked in (current.__cause__, current.__context__)
            if linked is not None
        )


def release_after_explicit_unload() -> None:
    """Retry global cleanup and close ownership only after confirmed success."""
    global _FAILED_LOAD_CLEANUP_POISONED, _PROVISIONAL_CLEAR_WHEN_EMPTY

    if not cleanup_pending():
        return
    # Retained-only ownership can predate the poison flag. Latch first so an
    # interruption anywhere in unload, cache cleanup, or close remains blocked.
    _FAILED_LOAD_CLEANUP_POISONED = True
    import comfy.model_management as mm

    for _owned, error in (
        *_RETAINED_FAILED_LOAD_SLABS,
        *_RETAINED_FAILED_LOAD_RESOURCES,
    ):
        if error is not None:
            _clear_exception_frames(error)
    mm.unload_all_models()
    mm.soft_empty_cache()
    gc.collect()
    failure: BaseException | None = None
    failure_cause: BaseException | None = None
    for registry, forget, label in (
        (
            _RETAINED_FAILED_LOAD_SLABS,
            _forget,
            "another retained slab close also failed",
        ),
        (
            _RETAINED_FAILED_LOAD_RESOURCES,
            _forget_resource,
            "another retained resource close also failed",
        ),
    ):
        for resource, _error in reversed(tuple(registry)):
            try:
                resource.close()
                # close() may confirm and remove its own provisional
                # registration. Forget by exact identity instead of position.
                forget(resource)
            except BaseException as close_error:
                failure, failure_cause = reconcile_error(
                    failure,
                    close_error,
                    label,
                )
    if failure is not None:
        raise_with_distinct_cause(failure, failure_cause)
    _FAILED_LOAD_CLEANUP_POISONED = False
    _PROVISIONAL_CLEAR_WHEN_EMPTY = False


def cleanup_failed_load(
    slab: Any,
    error: BaseException,
    context: str,
    *,
    resources: tuple[Any, ...] = (),
) -> bool:
    """Clean a failed load without replacing it with an ordinary cleanup error.

    Returns whether global cleanup and all owned-resource closes were confirmed.
    A false result leaves durable poison for the next explicit unload retry;
    operational cancellation propagates with exact identity.
    """
    global _FAILED_LOAD_CLEANUP_POISONED, _PROVISIONAL_CLEAR_WHEN_EMPTY

    poisoned_before = _FAILED_LOAD_CLEANUP_POISONED
    # Latch even with no concrete owner: global unload/cache/GC itself can be
    # interrupted, and unconfirmed process state must block the next load.
    _FAILED_LOAD_CLEANUP_POISONED = True

    def finish_failure(candidate: BaseException, label: str) -> bool:
        strongest, cause = reconcile_error(error, candidate, label)
        if strongest is not error:
            raise_with_distinct_cause(strongest, cause)
        return False

    try:
        owned_resources = [(resource, False) for resource in resources]
        if slab is not None:
            owned_resources.append((slab, True))
        for resource, is_slab in owned_resources:
            (retain if is_slab else _retain_resource)(resource, error)
    except BaseException as publish_exc:
        # The process latch is already durable even if registry publication was
        # interrupted. Preserve the load's primary exception and require recycle.
        try:
            log.error(
                "%s resource ownership publication failed (%s); recycle required",
                context,
                failure_summary(publish_exc),
            )
        except BaseException:
            pass
        return finish_failure(
            publish_exc,
            "failed-load ownership publication also failed",
        )

    try:
        # Completed callback/loader frames can retain a model via the original
        # traceback. Clearing their locals preserves the traceback itself.
        _clear_exception_frames(error)
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
    except BaseException as cleanup_exc:
        _FAILED_LOAD_CLEANUP_POISONED = True
        suffix = "; retaining failed-load ownership" if slab is not None or resources else ""
        try:
            log.error(
                "%s cleanup failed (%s)%s",
                context,
                failure_summary(cleanup_exc),
                suffix,
            )
        except BaseException:
            pass
        return finish_failure(cleanup_exc, "failed-load global cleanup also failed")
    close_failed = False
    strongest: BaseException = error
    strongest_cause: BaseException | None = None
    boundary_exc: BaseException | None = None
    try:
        for resource, is_slab in owned_resources:
            try:
                resource.close()
                (_forget if is_slab else _forget_resource)(resource)
            except BaseException as close_exc:
                close_failed = True
                _FAILED_LOAD_CLEANUP_POISONED = True
                strongest, strongest_cause = reconcile_error(
                    strongest,
                    close_exc,
                    "failed-load resource close also failed",
                )
                try:
                    log.warning(
                        "%s resource close failed (%s); retaining ownership",
                        context,
                        failure_summary(close_exc),
                    )
                except BaseException:
                    pass
        if (
            not close_failed
            and (not poisoned_before or _PROVISIONAL_CLEAR_WHEN_EMPTY)
            and not _RETAINED_FAILED_LOAD_SLABS
            and not _RETAINED_FAILED_LOAD_RESOURCES
        ):
            _FAILED_LOAD_CLEANUP_POISONED = False
            _PROVISIONAL_CLEAR_WHEN_EMPTY = False
    except BaseException as caught:
        boundary_exc = caught
        # Every unprocessed owner was prepublished, and the process latch must
        # survive an interruption between loop iterations or at the return edge.
        _FAILED_LOAD_CLEANUP_POISONED = True
        strongest, strongest_cause = reconcile_error(
            strongest,
            caught,
            "failed-load resource close boundary also failed",
        )
        try:
            log.error(
                "%s resource close boundary interrupted (%s); recycle required",
                context,
                failure_summary(caught),
            )
        except BaseException:
            pass
    if strongest is not error:
        raise_with_distinct_cause(strongest, strongest_cause)
    return not close_failed and boundary_exc is None
