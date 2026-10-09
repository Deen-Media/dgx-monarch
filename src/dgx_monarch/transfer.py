"""Driver/actor tensor transfer (DESIGN.md §5.6; docs/VALIDATION.md)."""
from __future__ import annotations

import os
import threading
from typing import Any

import torch
from monarch.rdma import RDMABuffer, is_ibverbs_available

from . import rdma_ownership, rdma_poison, rdma_settlement, transfer_utils
from .constants import DEFAULT_RDMA_MIN_BYTES
from .log import get_logger
from .rdma_read_driver import run_owned_off_loop as run_blocking_off_loop
from .rdma_receiver import _read_and_release

_byte_split = transfer_utils.byte_split
_failure_summary = transfer_utils.failure_summary
_qp_split = transfer_utils.qp_split
_safe_log = transfer_utils.safe_call
_safe_note = transfer_utils.safe_note
is_nested_tensor = transfer_utils.is_nested_tensor
pack_conditioning = transfer_utils.pack_conditioning
pack_latent = transfer_utils.pack_latent
tree_to_cpu = transfer_utils.tree_to_cpu
subprocess = transfer_utils.subprocess
_prefer_cleanup = rdma_settlement.prefer_error
_drop_cancellation = rdma_settlement.drop_cancellation
_publish_drop_result = rdma_settlement.publish_drop_result
_publish_settlement = rdma_settlement.publish_settlement

log = get_logger(__name__)

RDMA_READ_TIMEOUT_S = 120
RDMA_GET_MARGIN_S = 15
RDMA_READ_OUTER_MARGIN_S = 30
_NATIVE_RDMA_BUFFER = RDMABuffer
_RDMA_PREFLIGHT_LOCK = threading.Lock()
_RDMA_PREFLIGHT_RESULT: bool | None = None


_RDMA_POISON_LOCK = threading.RLock()
_RDMA_POISONED_OWNERS: list[rdma_poison.RDMAPoisonOwner] = []
_RDMA_POISON_REPORTER = rdma_poison.PoisonReporter()
_RDMADropFailure = rdma_poison.RDMADropFailure
_RDMAPoisonOwner = rdma_poison.RDMAPoisonOwner


def _rdma_poisoned() -> bool:
    with _RDMA_POISON_LOCK:
        return bool(_RDMA_POISONED_OWNERS)


def _new_poison_owner(failures: list[_RDMADropFailure], keepalive: Any,
                      phase: str) -> _RDMAPoisonOwner:
    return rdma_poison.new_poison_owner(failures, keepalive, phase)


def _publish_constructor_intent(part: dict, keepalive: Any) -> _RDMAPoisonOwner:
    """Publish a poison owner for the part's backing before upstream RDMABuffer can lose its native handle."""
    owner = rdma_poison.constructor_intent(part, keepalive)
    _RDMA_POISONED_OWNERS.append(owner)
    return owner


def _log_poison_latch() -> None:
    """Claim and report the process's first durable RDMA poison owner."""
    _RDMA_POISON_REPORTER.report(
        _RDMA_POISON_LOCK, _RDMA_POISONED_OWNERS, log.error, _safe_log)


def _report_poison_safely() -> None:
    try:
        _log_poison_latch()
    except BaseException:
        pass


def _unowned_failures(failures: list[_RDMADropFailure]) -> list[_RDMADropFailure]:
    return rdma_poison.unowned_failures(
        failures, _RDMA_POISON_LOCK, _RDMA_POISONED_OWNERS)


def _poison_failed_drops(failures: list[_RDMADropFailure], *, keepalive: Any,
                         phase: str) -> None:
    # Failed or ambiguous registrations stay owned by this process; the poison is process-local.
    with _RDMA_POISON_LOCK:
        failures = _unowned_failures(failures)
        if not failures:
            return
        _RDMA_POISONED_OWNERS.append(
            _new_poison_owner(failures, keepalive, phase))


def _poison_with_retry(failures: list[_RDMADropFailure], keepalive: Any,
                       phase: str) -> BaseException | None:
    try:
        _poison_failed_drops(failures, keepalive=keepalive, phase=phase)
    except BaseException as exc:
        return rdma_settlement.retry_poison_publication(
            exc, failures, keepalive, phase,
            _unowned_failures, _poison_failed_drops)
    return None


def _rdma_usable() -> bool:
    global _RDMA_PREFLIGHT_RESULT
    if _rdma_poisoned():
        return False
    if not is_ibverbs_available():
        return False
    if RDMABuffer is not _NATIVE_RDMA_BUFFER:
        return True
    with _RDMA_PREFLIGHT_LOCK:
        if _RDMA_PREFLIGHT_RESULT is not None:
            return _RDMA_PREFLIGHT_RESULT
        _RDMA_PREFLIGHT_RESULT = transfer_utils.native_rdma_preflight()
        if not _RDMA_PREFLIGHT_RESULT:
            _safe_log(log.error,
                "RDMA native preflight failed; using actor messaging (the probe "
                "ran in a child process, so a native abort could not kill this worker)")
        return _RDMA_PREFLIGHT_RESULT


def _drop_failure(part: dict, exc: BaseException) -> _RDMADropFailure:
    summary = _failure_summary(exc)
    try:
        exc.__traceback__ = exc.__context__ = exc.__cause__ = None
    except BaseException:
        pass
    return _RDMADropFailure(part, exc, summary)


def _part_buffer(part: dict) -> Any:
    owner = part.get("_buffer_owner")
    return part.get("buffer") if owner is None else (owner[0] if owner else None)


def _release_rdma_parts(
    parts: list[dict], outcome_owner: dict | None = None,
) -> list[_RDMADropFailure]:
    failures: list[_RDMADropFailure] = []
    index = -1
    try:
        for index, part in enumerate(parts):  # noqa: B007 - resume after boundary
            try:
                _part_buffer(part).drop().get(timeout=RDMA_READ_TIMEOUT_S)
            except BaseException as exc:
                failures.append(_drop_failure(part, exc))
        _publish_drop_result(outcome_owner, failures, None)
        return failures
    except BaseException as exc:
        if outcome_owner is not None and "drop_outcome" in outcome_owner:
            failures, prior_error = outcome_owner["drop_outcome"]
            error = _prefer_cleanup(
                prior_error, exc, "RDMA drop publication return failed")
            _publish_drop_result(outcome_owner, failures, error)
            return failures
        current = parts[max(0, index)]
        if not any(failure.part is current for failure in failures):
            failures.append(_drop_failure(current, exc))
        for part in parts[index + 1:]:
            try:
                _part_buffer(part).drop().get(timeout=RDMA_READ_TIMEOUT_S)
            except BaseException as later_exc:
                failures.append(_drop_failure(part, later_exc))
        _publish_drop_result(outcome_owner, failures, exc)
        return failures


def _release_rdma_parts_guarded(
    parts: list[dict], outcome_owner: dict | None = None,
) -> list[_RDMADropFailure]:
    owned: list[dict] = []
    try:
        snapshot = list.copy(parts)
        owned = [part for part in snapshot if _part_buffer(part) is not None]
        release_parts = parts if len(owned) == len(snapshot) else owned
        return _release_rdma_parts(release_parts, outcome_owner)
    except BaseException as exc:
        if outcome_owner is not None and "drop_outcome" in outcome_owner:
            failures, prior_error = outcome_owner["drop_outcome"]
            error = _prefer_cleanup(
                prior_error, exc, "RDMA drop helper return failed")
        else:
            if not owned:
                owned = [part for part in list.copy(parts) if _part_buffer(part) is not None]
            failures = [_drop_failure(part, exc) for part in owned]
            error = exc
        _publish_drop_result(outcome_owner, failures, error)
        return failures


def _fail_closed_parts(parts: list[dict], exc: BaseException) -> list[_RDMADropFailure]:
    owned: list[dict] = []
    try:
        owned = [part for part in list.copy(parts) if _part_buffer(part) is not None]
        return [_drop_failure(part, exc) for part in owned]
    except BaseException as recovery_exc:
        if not owned:
            owned = [part for part in list.copy(parts) if _part_buffer(part) is not None]
        return [_drop_failure(part, recovery_exc) for part in owned]


def _settle_parts(
    parts: list[dict], keepalive: Any, phase: str,
    outcome_owner: dict | None = None,
) -> tuple[list[_RDMADropFailure], BaseException | None]:
    drop_owner: dict = {}
    try:
        failures = _release_rdma_parts_guarded(parts, drop_owner)
        drop_error = (
            drop_owner["drop_outcome"][1]
            if "drop_outcome" in drop_owner else None)
        poison_error = _poison_with_retry(failures, keepalive, phase)
    except BaseException as boundary:
        if "drop_outcome" in drop_owner:
            failures, drop_error = drop_owner["drop_outcome"]
            cleanup_error = _prefer_cleanup(
                drop_error, boundary, "RDMA drop return failed")
        else:
            failures = _fail_closed_parts(parts, boundary)
            cleanup_error = boundary
        poison_error = _poison_with_retry(failures, keepalive, phase)
        failures, poison_error, published_cleanup = _publish_settlement(
            outcome_owner, failures, poison_error, cleanup_error)
        if published_cleanup is boundary:
            raise
        if published_cleanup is None:
            published_cleanup = boundary
        raise published_cleanup from boundary
    if drop_error is not None:
        drop_boundary, cause = rdma_settlement.reconcile_drop_error(failures, drop_error)
        if drop_boundary is not None:
            failures, poison_error, drop_cleanup = _publish_settlement(
                outcome_owner, failures, poison_error, drop_boundary)
            transfer_utils.raise_with_distinct_cause(
                drop_cleanup if drop_cleanup is not None else drop_boundary, cause)
    failures, poison_error, settled_cleanup = _publish_settlement(
        outcome_owner, failures, poison_error, None)
    if settled_cleanup is not None:
        raise settled_cleanup
    return failures, poison_error


def _fail_closed(parts: list[dict], keepalive: Any, phase: str,
                 exc: BaseException) -> tuple[list[_RDMADropFailure], BaseException | None]:
    failures = _fail_closed_parts(parts, exc)
    return failures, _poison_with_retry(failures, keepalive, phase)


def _settle_or_fail_closed(parts: list[dict], keepalive: Any, phase: str,
                           handoff: dict | None = None):
    cleanup_error: BaseException | None
    settlement_owner: dict = {}
    try:
        failures, poison_error = _settle_parts(
            parts, keepalive, phase, settlement_owner)
    except BaseException as cleanup_exc:
        cleanup_error = cleanup_exc
        if "outcome" in settlement_owner:
            failures, poison_error, inner_error = settlement_owner["outcome"]
            if inner_error is not None:
                cleanup_error = _prefer_cleanup(
                    cleanup_error, inner_error, "inner RDMA settlement failed")
        else:
            try:
                failures, poison_error = _fail_closed(
                    parts, keepalive, phase, cleanup_error)
            except BaseException as recovery_error:
                cleanup_error = _prefer_cleanup(
                    cleanup_error, recovery_error,
                    "RDMA settlement recovery failed")
                try:
                    failures, poison_error = _fail_closed(
                        parts, keepalive, phase, recovery_error)
                except BaseException as final_error:
                    strongest, cause = transfer_utils.reconcile_error(
                        cleanup_error,
                        final_error,
                        "RDMA settlement final recovery also failed",
                    )
                    transfer_utils.raise_with_distinct_cause(
                        strongest, cause)
    else:
        cleanup_error = None
    if handoff is not None:
        failures, poison_error, cleanup_error = _publish_settlement(
            handoff, failures, poison_error, cleanup_error)
        handoff["state"] = "settled"
    return failures, poison_error, cleanup_error


def _fail_closed_handoff(handoff: dict, phase: str, exc: BaseException):
    return rdma_ownership.fail_closed_handoff(
        handoff, phase, exc, _fail_closed)


def _register_parts(tensor: torch.Tensor, n: int, handoff: dict | None = None) -> list[dict]:
    with _RDMA_POISON_LOCK:  # Spans construction: atomically root/remove tentative intent.
        if _RDMA_POISONED_OWNERS:
            raise RuntimeError("RDMA registration ownership is poisoned")
        flat = tensor.flatten().view(torch.uint8)
        nbytes = flat.numel()
        if handoff is None:
            handoff = {"parts": [], "state": "registering"}
        parts = handoff["parts"]
        try:
            for off, size in _byte_split(nbytes, n):
                part: dict[str, Any] = {
                    "offset": off, "nbytes": size, "_buffer_owner": []}
                parts.append(part)
                intent = _publish_constructor_intent(part, tensor)
                # C list.extend owns the constructor result before Python resumes.
                part["_buffer_owner"].extend(map(RDMABuffer, (flat[off:off + size],)))
                if _part_buffer(part) is None:
                    raise RuntimeError("RDMA registration returned no buffer")
                part["buffer"] = _part_buffer(part)
                part.pop("_buffer_owner")
                _RDMA_POISONED_OWNERS.remove(intent)
            handoff["state"] = "registered"
            return parts
        except BaseException as caught:
            primary = caught
            failures: list[_RDMADropFailure] = []
            try:
                _, poison_error, cleanup_error = _settle_or_fail_closed(
                    parts, tensor, "registration rollback", handoff)
                failures = handoff["outcome"][0]
            except BaseException as boundary_error:
                cleanup_error = boundary_error
                try:
                    failures, poison_error = _fail_closed_handoff(
                        handoff, "registration rollback", boundary_error)
                except BaseException as recovery_error:
                    cleanup_error = _prefer_cleanup(cleanup_error, recovery_error, "registration fail-closed retry failed")
                    try:
                        failures, poison_error = _fail_closed_handoff(handoff, "registration rollback", recovery_error)
                    except BaseException as final_error:
                        poison_error = final_error
            if cleanup_error is not None:
                primary = _prefer_cleanup(primary, cleanup_error, "RDMA registration rollback failed")
            cancellation = _drop_cancellation(failures)
            if cancellation is not None:
                primary = _prefer_cleanup(primary, cancellation, "RDMA registration buffer release failed")
            if poison_error is not None:
                primary = _prefer_cleanup(primary, poison_error, "RDMA poison publication failed")
            if primary is not caught:
                transfer_utils.raise_with_distinct_cause(primary, caught)
            raise


class LatentReturn:
    """Worker-side leader return with message or size-gated RDMA descriptors."""

    def __init__(self, mode: str = "rdma", min_bytes: int | None = None,
                 keepalive_depth: int = 1):
        if mode not in ("message", "rdma"):
            raise ValueError(f"unknown latent return mode {mode!r}")
        if mode == "rdma":
            from .actor import comfy_dynamic  # lazy: the driver has no actor stack

            comfy_dynamic.assert_rdma_compatible()
        self.mode = mode
        if min_bytes is None:
            min_bytes = int(os.environ.get("DGXM_RDMA_MIN_BYTES", DEFAULT_RDMA_MIN_BYTES))
        self.min_bytes = max(1, int(min_bytes))
        self._depth = max(1, int(keepalive_depth))
        self._keepalive: dict[tuple, torch.Tensor] = {}

    def pack(self, key: str, tensor, seq: int = 0, depth: int = 1,
             handoff: dict | None = None) -> dict:
        if is_nested_tensor(tensor):
            return {"kind": "message", "tensor": tree_to_cpu(tensor)}
        tensor = tensor.detach().to("cpu").contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if self.mode == "rdma" and nbytes >= self.min_bytes:
            if not _rdma_usable():
                _report_poison_safely()
                if not _rdma_poisoned():
                    _safe_log(log.warning,
                        "RDMA latent return unavailable; falling back to messaging")
                return {"kind": "message", "tensor": tensor}
            if handoff is None:
                handoff = {}
            handoff.update(parts=[], state="registering", keepalive=tensor)
            try:
                _register_parts(tensor, _qp_split(), handoff)
                if handoff.get("token") is None:
                    self._retain(key, seq, tensor, depth)
                return {
                    "kind": "rdma",
                    "parts": handoff["parts"],
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "owner_token": handoff.get("token"),
                    "setup_generation": handoff.get("setup_generation"),
                }
            except BaseException as caught:
                primary = caught
                failures: list[_RDMADropFailure] = []
                try:
                    if handoff["state"] == "registered":
                        parts = handoff["parts"]
                        _, poison_error, cleanup_error = _settle_or_fail_closed(
                            parts, tensor, "descriptor publication rollback", handoff)
                        failures = handoff["outcome"][0]
                    else:
                        poison_error = cleanup_error = None
                except BaseException as boundary_error:
                    cleanup_error = boundary_error
                    try:
                        failures, poison_error = _fail_closed_handoff(
                            handoff, "descriptor publication rollback", boundary_error)
                    except BaseException as recovery_error:
                        cleanup_error = _prefer_cleanup(cleanup_error, recovery_error, "descriptor fail-closed retry failed")
                        try:
                            failures, poison_error = _fail_closed_handoff(handoff, "descriptor publication rollback", recovery_error)
                        except BaseException as final_error:
                            poison_error = final_error
                if cleanup_error is not None:
                    primary = _prefer_cleanup(primary, cleanup_error, "RDMA descriptor rollback failed")
                cancellation = _drop_cancellation(failures)
                if cancellation is not None:
                    primary = _prefer_cleanup(primary, cancellation, "RDMA descriptor buffer release failed")
                if poison_error is not None:
                    primary = _prefer_cleanup(primary, poison_error, "RDMA poison publication failed")
                self._keepalive.pop((key, seq), None)
                _report_poison_safely()
                if primary is not caught:
                    transfer_utils.raise_with_distinct_cause(primary, caught)
                if not isinstance(caught, Exception):
                    raise
                _safe_log(log.warning,
                    "RDMA latent return unavailable (%r); a caller with no handoff sends this "
                    "latent by messaging, and a caller with one sends it only if its handoff "
                    "holds no RDMA resources", caught)
        return {"kind": "message", "tensor": tensor}

    def abort_handoff(self, handoff: dict, primary: BaseException) -> None:
        try:
            rdma_ownership.abort_handoff(
                handoff, primary, _fail_closed, _safe_note)
        finally:
            _report_poison_safely()

    def _retain(self, key: str, seq: int, tensor, depth: int) -> None:
        window = max(int(depth), self._depth)
        self._keepalive.pop((key, seq), None)
        self._keepalive[(key, seq)] = tensor
        retained = [k for k in self._keepalive if k[0] == key]
        for ks in retained[:-window]:
            self._keepalive.pop(ks, None)

    def drop(self, key: str | None = None) -> None:
        if key is None:
            self._keepalive.clear()
        else:
            for ks in [k for k in self._keepalive if k[0] == key]:
                self._keepalive.pop(ks, None)


def _read_parts_concurrent(parts: list[dict], out_bytes: torch.Tensor) -> None:
    transfer_utils.read_parts_concurrent(
        parts, out_bytes, RDMA_READ_TIMEOUT_S, RDMA_GET_MARGIN_S)


def read_latent_result(desc: dict, read_guard: Any = None) -> Any:
    if desc["kind"] == "message":
        return desc["tensor"]
    if desc["kind"] == "rdma":
        def materialize(
            initial_primary: BaseException | None,
            before_failure_ack: Any,
            mark_operation_entry: Any,
            publish_operation: Any,
        ) -> Any:
            try:
                return _read_and_release(
                    desc, read_guard, read_parts=_read_parts_concurrent,
                    settle=_settle_or_fail_closed, fail_closed=_fail_closed,
                    read_timeout=RDMA_READ_TIMEOUT_S,
                    get_margin=RDMA_GET_MARGIN_S,
                    initial_primary=initial_primary,
                    before_failure_ack=before_failure_ack,
                    mark_operation_entry=mark_operation_entry,
                    publish_operation=publish_operation)
            finally:
                _report_poison_safely()

        try:
            return run_blocking_off_loop(read_guard, materialize,
                timeout_s=lambda: transfer_utils.rdma_outer_timeout(
                    desc, RDMA_READ_TIMEOUT_S, RDMA_GET_MARGIN_S,
                    rdma_ownership.RDMA_ACK_TIMEOUT_S,
                    rdma_ownership.RDMA_ACK_ATTEMPTS, RDMA_READ_OUTER_MARGIN_S),
                thread_name="dgxm-latent-read")
        finally:
            _report_poison_safely()
    raise ValueError(f"unknown latent descriptor kind {desc.get('kind')!r}")
