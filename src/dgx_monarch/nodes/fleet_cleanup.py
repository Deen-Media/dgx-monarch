"""Cancellation-safe cleanup helpers for Fleet-owned resources."""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .. import mesh_setup
from ..transfer_utils import prefer_error
from . import consent_waiver


@dataclass
class _FleetPending:
    job_index: int
    future: object
    progress: object
    render_id: str | None = None
    # The rank the driver submitted this job to; telemetry_fleet.box_label
    # derives the job's box from it. A record rebuilt from a tuple carries
    # none, and the label falls back to the host.
    rank: int | None = None
    collection_started: bool = False
    collected: bool = False
    consumed: bool = False
    # A typed refusal is terminal without a latent, so its lease is consumed.
    refused: bool = False
    retired: bool = False
    audit_retired: bool = False
    progress_closed: bool = False
    cancel_requested: bool = False


def _cancel_fleet_pending(
    handle: Any,
    pending: Iterable[Any],
    *,
    extra_render_id: str | None = None,
) -> BaseException | None:
    """Broadcast cancellation once for each reachable, unfinished Fleet job."""
    first_error: BaseException | None = None
    seen: set[str] = set()
    targets: list[tuple[Any | None, str]] = []
    for item in pending:
        try:
            if item.collected or item.cancel_requested:
                continue
            render_id = item.render_id
            if render_id is not None:
                targets.append((item, str(render_id)))
        except BaseException as exc:
            first_error = prefer_error(
                first_error, exc, "fleet cancellation target inspection failed")
    if extra_render_id is not None:
        targets.append((None, str(extra_render_id)))

    for item, render_id in targets:
        if not render_id or render_id in seen:
            continue
        seen.add(render_id)
        try:
            if item is not None:
                # One request, sent without waiting, is all this side does.
                item.cancel_requested = True
            handle.cancel_sample(render_id, wait=False)
        except BaseException as exc:
            first_error = prefer_error(
                first_error, exc, "additional fleet cancellation failed")
    return first_error


def _drain_fleet_collections(
    handle: Any,
    pending: list[Any],
    timeout_s: float,
    prepare: Callable[[Any, Any], Any],
    publish: Callable[[Any, Any], None],
    classify: Callable[[Any, BaseException], None],
    settle: Callable[[Any], None],
) -> Iterable[tuple[Any | None, BaseException, str]]:
    """Prepare by completion order, then publish and settle in prompt order."""
    completions: queue.Queue[tuple[int, Any, BaseException | None]] = queue.Queue()
    candidates: list[Any] = []
    collectors: dict[int, tuple[Any, threading.Thread, threading.Event]] = {}
    outcomes: dict[int, tuple[Any, BaseException | None]] = {}
    prior_failures: dict[int, list[tuple[BaseException, str]]] = {}
    post_failures: dict[int, list[tuple[BaseException, str]]] = {}
    global_failures: list[tuple[Any | None, BaseException, str]] = []

    def collect(key: int, item: Any, completed: threading.Event) -> None:
        result: Any = None
        error: BaseException | None = None
        try:
            result = handle.collect_one(item.future, timeout_s=timeout_s)
        except BaseException as exc:
            # Hand on the same exception object, never tested for truth: a
            # cancellation may define a __bool__ that is false or raises.
            error = exc
        completed.set()
        completions.put((key, result, error))

    for item in pending:
        if item.collection_started:
            continue
        candidates.append(item)
        thread: threading.Thread | None = None
        try:
            # Set before the thread starts, so a retry never starts a second collector.
            item.collection_started = True
            key = id(item)
            completed = threading.Event()
            thread = threading.Thread(
                target=collect,
                args=(key, item, completed),
                name=f"dgxm-fleet-drain-{item.job_index}",
            )
            collectors[key] = (item, thread, completed)
            thread.start()
        except BaseException as exc:
            started = thread is not None and thread.ident is not None
            if not started:
                collectors.pop(id(item), None)
            prior_failures.setdefault(id(item), []).append(
                (exc, "fleet collector start failed"))

    abort_requested = bool(prior_failures)
    if abort_requested:
        cancel_error = _cancel_fleet_pending(handle, candidates)
        if cancel_error is not None:
            first_key = next(iter(prior_failures))
            post_failures.setdefault(first_key, []).append(
                (cancel_error, "fleet peer cancellation failed"))

    remaining = set(collectors)
    while remaining:
        try:
            key, result, task_error = completions.get()
        except BaseException as exc:
            global_failures.append(
                (None, exc, "fleet completion wait was interrupted"))
            if not abort_requested:
                abort_requested = True
                live = [
                    item
                    for item, _thread, completed in collectors.values()
                    if not completed.is_set()
                ]
                cancel_error = _cancel_fleet_pending(handle, live)
                if cancel_error is not None:
                    global_failures.append(
                        (None, cancel_error, "fleet peer cancellation failed"))
            continue
        if key not in remaining:
            continue
        remaining.remove(key)
        item, _thread, _completed = collectors[key]
        if task_error is None:
            try:
                result = prepare(item, result)
            except BaseException as exc:
                task_error = exc
        outcomes[key] = (result, task_error)
        if task_error is not None and not abort_requested:
            abort_requested = True
            task_was_ambiguous = (
                isinstance(task_error, TimeoutError)
                or not isinstance(task_error, Exception)
            )
            live = [item] if task_was_ambiguous else []
            live.extend(
                peer
                for peer, _thread, completed in collectors.values()
                if id(peer) in remaining and not completed.is_set()
            )
            cancel_error = _cancel_fleet_pending(handle, live)
            if cancel_error is not None:
                post_failures.setdefault(key, []).append(
                    (cancel_error, "fleet peer cancellation failed"))

    yield from global_failures
    for item in candidates:
        key = id(item)
        for stored_error, label in prior_failures.get(key, ()):
            yield item, stored_error, label
        if key in outcomes:
            result, task_error = outcomes[key]
            if task_error is None:
                try:
                    publish(item, result)
                except BaseException as exc:
                    task_error = exc
            if task_error is not None:
                try:
                    classify(item, task_error)
                except BaseException as exc:
                    yield item, exc, "fleet failure classification failed"
                yield item, task_error, "additional fleet job failure"
            try:
                item.collected = True
            except BaseException as exc:
                yield item, exc, "fleet collection completion publication failed"
        for stored_error, label in post_failures.get(key, ()):
            yield item, stored_error, label
        try:
            settle(item)
        except BaseException as exc:
            yield item, exc, "fleet per-job settlement failed"


def _retire_fleet_audit(render_id: str | None) -> BaseException | None:
    """Retry idempotent audit retirement and retain caller cancellation."""
    first_error: BaseException | None = None
    for _attempt in range(2):
        try:
            consent_waiver.retire_audit(render_id)
        except BaseException as exc:
            first_error = prefer_error(
                first_error, exc, "fleet render audit retirement retry failed"
            )
        else:
            return (
                first_error
                if first_error is not None
                and not isinstance(first_error, Exception)
                else None
            )
    return first_error


def _retire_fleet_future(future, *, consumed: bool) -> BaseException | None:
    """Release or abandon the lease through mesh_setup, falling back to the future itself."""
    callback = mesh_setup.release_sample if consumed else mesh_setup.abandon_sample
    first_error: BaseException | None = None
    for _attempt in range(2):
        try:
            callback(future)
        except BaseException as exc:
            first_error = prefer_error(
                first_error, exc, "fleet lease wrapper retry failed"
            )
        else:
            return (
                first_error
                if first_error is not None
                and not isinstance(first_error, Exception)
                else None
            )
    if isinstance(future, mesh_setup.SetupBoundFuture):
        direct = future.release if consumed else future.abandon
        for _attempt in range(2):
            try:
                direct()
            except BaseException as exc:
                first_error = prefer_error(
                    first_error, exc, "fleet direct lease retry failed"
                )
            else:
                return (
                    first_error
                    if first_error is not None
                    and not isinstance(first_error, Exception)
                    else None
                )
    return first_error


def _close_fleet_progress(progress) -> BaseException | None:
    """Close a receiver, retrying once, so an interrupt on the call cannot leak it."""
    first_error: BaseException | None = None
    for _attempt in range(2):
        try:
            progress.__exit__(None, None, None)
        except BaseException as exc:
            first_error = prefer_error(
                first_error, exc, "fleet progress close retry failed"
            )
        else:
            return (
                first_error
                if first_error is not None
                and not isinstance(first_error, Exception)
                else None
            )
    return first_error
