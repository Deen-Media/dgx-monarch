"""Fail-closed driver-side RDMA read, release, and owner acknowledgement."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch

from . import rdma_ownership
from .transfer_utils import prefer_error, raise_with_distinct_cause, reconcile_error


class _AuthorityExitWitness:
    """Publish proof only after the wrapped authority exit returns."""

    __slots__ = ("authority", "confirmation", "state")

    def __init__(
        self,
        authority: Any,
        state: dict[str, Any],
        confirmation: Callable[[], bool] | None = None,
    ) -> None:
        self.authority = authority
        self.state = state
        self.confirmation = confirmation

    def _confirm(self) -> None:
        confirmed = True if self.confirmation is None else self.confirmation()
        self.state["authority_exit_confirmed"] = confirmed

    def _confirm_after_error(self, error: BaseException) -> None:
        if self.confirmation is None:
            return
        try:
            self._confirm()
        except BaseException as proof_error:
            strongest, cause = reconcile_error(
                error, proof_error, "read-authority exit proof also failed")
            raise_with_distinct_cause(strongest, cause)

    def __enter__(self) -> Any:
        try:
            return self.authority.__enter__()
        except BaseException as error:
            self._confirm_after_error(error)
            raise

    def __exit__(self, *exc_info: Any) -> Any:
        try:
            suppressed = self.authority.__exit__(*exc_info)
        except BaseException as error:
            self._confirm_after_error(error)
            raise
        self._confirm()
        return suppressed

def ack_timeout_s() -> float:
    return rdma_ownership.RDMA_ACK_TIMEOUT_S


def validate_ack_responses(
    rows: Any, generation: int, token: str, expected_world: int
) -> None:
    """Require one exact owner and only explicit non-owner replies."""
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("RDMA handoff ACK returned no worker responses")
    if type(expected_world) is not int or expected_world < 1:
        raise RuntimeError("RDMA handoff ACK has an invalid expected worker count")
    if len(rows) != expected_world:
        raise RuntimeError(
            "RDMA handoff ACK returned a partial worker response set")
    owners = 0
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("RDMA handoff ACK returned a malformed response")
        if (type(row.get("setup_generation")) is not int
                or row["setup_generation"] != generation
                or type(row.get("token")) is not str or row["token"] != token):
            raise RuntimeError("RDMA handoff ACK response identity mismatch")
        status = row.get("status")
        if status not in {"released", "already_released", "unknown"}:
            raise RuntimeError(f"RDMA handoff ACK returned invalid status {status!r}")
        owners += status in {"released", "already_released"}
    if owners != 1:
        raise RuntimeError(
            f"RDMA handoff ACK expected exactly one owner, got {owners}")


def _ack_owner(
    desc: dict,
    read_guard: Any,
    outcome_owner: list[BaseException | None] | None = None,
) -> None:
    primary: BaseException | None = None
    token, generation = desc.get("owner_token"), desc.get("setup_generation")
    try:
        if token is not None or generation is not None:
            if (type(token) is not str or len(token) != 32
                    or any(char not in "0123456789abcdef" for char in token)
                    or type(generation) is not int or generation < 1):
                raise RuntimeError("RDMA descriptor has invalid owner metadata")
            if not hasattr(read_guard, "ack_latent"):
                raise RuntimeError(
                    "RDMA owner token requires an ack-capable read guard")
            rdma_ownership.acknowledge_with_retry(
                read_guard.ack_latent, generation, token)
    except BaseException as error:
        primary = error
    if outcome_owner is not None:
        try:
            if not outcome_owner:
                outcome_owner.append(primary)
        except BaseException as publication_error:
            primary = prefer_error(
                primary, publication_error, "ACK outcome publication failed")
            if outcome_owner:
                outcome_owner[0] = primary
            else:
                outcome_owner.append(primary)
    if primary is not None:
        raise primary


def _phase(state: dict, primary: BaseException | None) -> str:
    if state["read_started"] and not state["read_complete"]:
        return "failed latent read cleanup"
    if primary is None:
        return "latent read release"
    return ("read-authority failure cleanup" if state["prepared"]
            else "latent read preparation cleanup")


def _read_and_release_owned(
    desc: dict,
    read_guard: Any,
    *,
    read_parts: Callable[[list[dict], torch.Tensor], None],
    settle: Callable,
    fail_closed: Callable,
    read_timeout: float,
    get_margin: float,
    initial_primary: BaseException | None = None,
    before_failure_ack: Callable[[], None] | None = None,
    mark_operation_entry: Callable[[], None] | None = None,
) -> Any:
    parts = desc["parts"]
    state = {
        "entry_complete": False,
        "prepared": False, "read_started": False,
        "read_complete": False, "cleanup_complete": False,
        "drops_confirmed": False, "authority_exit_confirmed": True,
        "ack_complete": False,
    }
    ack_outcome: list[BaseException | None] = []
    out: Any = None
    out_bytes: Any = None

    def enter_operation(
        primary: BaseException | None,
    ) -> tuple[BaseException | None, bool]:
        """Reconcile the idempotent entry marker before native settlement."""
        if mark_operation_entry is None:
            return primary, True
        for attempt in range(2):
            try:
                mark_operation_entry()
            except BaseException as entry_error:
                primary = prefer_error(
                    primary, entry_error,
                    "read entry publication failed" if attempt == 0
                    else "read entry publication retry failed")
            else:
                return primary, True
        return primary, False

    def acknowledge_preserving(
        primary: BaseException | None,
    ) -> BaseException | None:
        if (not state["drops_confirmed"]
                or not state["authority_exit_confirmed"]
                or state["ack_complete"]):
            return primary
        if not ack_outcome:
            for _attempt in range(2):
                try:
                    _ack_owner(desc, read_guard, ack_outcome)
                except BaseException as ack_error:
                    primary = prefer_error(
                        primary, ack_error,
                        "RDMA handoff ACK during failure cleanup failed")
                    if ack_outcome:
                        break
                    continue
                break
        if ack_outcome:
            published_ack_error = ack_outcome[0]
            if published_ack_error is None:
                state["ack_complete"] = True
            elif published_ack_error is not primary:
                primary = prefer_error(
                    primary, published_ack_error,
                    "RDMA handoff ACK during failure cleanup failed")
        return primary

    def finish(primary: BaseException | None) -> Any:
        nonlocal out, out_bytes
        keepalive = out if state["read_started"] and not state["read_complete"] else None
        phase = _phase(state, primary)
        failures: Any = None
        poison_error: BaseException | None = None
        settlement_owner: dict[str, Any] = {}
        try:
            if keepalive is not None:
                failures, poison_error = fail_closed(parts, keepalive, phase, primary)
            else:
                failures, poison_error, cleanup_error = settle(
                    parts, None, phase, settlement_owner)
                if cleanup_error is not None:
                    primary = prefer_error(
                        primary, cleanup_error, "latent settlement recovered")
            state["drops_confirmed"] = not failures and keepalive is None
            state["cleanup_complete"] = True
        except BaseException as boundary:
            primary = prefer_error(primary, boundary, "latent cleanup interrupted")
            keepalive = out if state["read_started"] and not state["read_complete"] else None
            if failures is None and "outcome" in settlement_owner:
                failures, poison_error, cleanup_error = settlement_owner["outcome"]
                if cleanup_error is not None:
                    primary = prefer_error(
                        primary, cleanup_error, "latent settlement recovered")
            if failures is not None:
                state["drops_confirmed"] = not failures and keepalive is None
                state["cleanup_complete"] = True
            else:
                try:
                    failures, poison_error = fail_closed(
                        parts, keepalive, "latent cleanup interruption", boundary)
                except BaseException as recovery:
                    primary = prefer_error(
                        primary, recovery, "latent cleanup recovery failed")
                    try:
                        failures, poison_error = fail_closed(
                            parts, keepalive, "latent cleanup interruption", recovery)
                    except BaseException as final_error:
                        primary = prefer_error(
                            primary, final_error, "latent cleanup final retry failed")
                        if primary is final_error:
                            raise
                        raise primary from final_error
                state["drops_confirmed"] = False
                state["cleanup_complete"] = True

        if (primary is not None or failures) and keepalive is None:
            out = out_bytes = None
        strongest = primary
        if strongest is None and failures:
            strongest = RuntimeError(
                f"RDMA latent read completed but {len(failures)} buffer "
                "registration(s) could not be released")
        error_cause: BaseException | None = None
        for failure in failures or ():
            prior = strongest
            strongest, next_cause = reconcile_error(
                strongest, failure.error,
                "RDMA registration cleanup also failed")
            if strongest is not prior:
                error_cause = prior
            elif error_cause is None and next_cause is not strongest:
                error_cause = next_cause
        if poison_error is not None:
            prior = strongest
            strongest, next_cause = reconcile_error(
                strongest, poison_error, "RDMA poison publication also failed")
            if strongest is not prior:
                error_cause = prior
            elif error_cause is None and next_cause is not strongest:
                error_cause = next_cause
        if strongest is not None:
            raise_with_distinct_cause(strongest, error_cause)

        return out

    def finish_before_read(primary: BaseException) -> Any:
        try:
            return finish(primary)
        except BaseException as settlement_error:
            primary = prefer_error(
                primary, settlement_error, "pre-read settlement also failed")
            if before_failure_ack is not None:
                try:
                    before_failure_ack()
                except BaseException as cleanup:
                    state["drops_confirmed"] = False
                    primary = prefer_error(
                        primary, cleanup, "pre-read authority cleanup failed")
            acknowledged_primary = acknowledge_preserving(primary)
            if acknowledged_primary is not None:
                primary = acknowledged_primary
            if primary is settlement_error:
                raise
            raise primary from settlement_error

    def drive() -> Any:
        nonlocal out, out_bytes
        entry_primary, entered = enter_operation(initial_primary)
        if not entered:
            if entry_primary is None:
                entry_primary = RuntimeError(
                    "read entry publication made no attempt")
            raise entry_primary
        state["entry_complete"] = True
        if entry_primary is not None:
            raise entry_primary

        try:
            dtype = getattr(torch, desc["dtype"])
            out = torch.empty(desc["shape"], dtype=dtype)
            out_bytes = out.flatten().view(torch.uint8)
            state["prepared"] = True
        except BaseException as primary:
            return finish_before_read(primary)

        authority_owner: list[Any] = []
        confirmation: Callable[[], bool] | None = None

        def exact_authority_retired() -> bool:
            return bool(authority_owner) and not read_guard.has_read_token(
                authority_owner[0])

        try:
            state["authority_exit_confirmed"] = False
            if hasattr(read_guard, "read_authority"):
                tokenized = (
                    hasattr(read_guard, "begin_read")
                    and hasattr(read_guard, "has_read_token")
                )
                if tokenized:
                    authority = read_guard.read_authority(authority_owner)
                    confirmation = exact_authority_retired
                else:
                    authority = read_guard.read_authority()
            else:
                authority = nullcontext()
                state["authority_exit_confirmed"] = True
        except BaseException as primary:
            return finish_before_read(primary)
        witnessed_authority = _AuthorityExitWitness(
            authority, state, confirmation)

        body_primary: BaseException | None = None
        try:
            with witnessed_authority:
                try:
                    state["read_started"] = True
                    if len(parts) == 1:
                        parts[0]["buffer"].read_into(
                            out_bytes, timeout=read_timeout).get(
                                timeout=read_timeout + get_margin)
                    else:
                        read_parts(parts, out_bytes)
                    state["read_complete"] = True
                    result = finish(None)
                except BaseException as exc:
                    body_primary = exc
                    if not state["cleanup_complete"]:
                        finish(exc)
                    raise
        except BaseException as exc:
            if body_primary is not None:
                if exc is not body_primary:
                    body_primary = prefer_error(
                        body_primary, exc, "RDMA read-authority cleanup also failed")
                acknowledged_primary = acknowledge_preserving(body_primary)
                if acknowledged_primary is not None:
                    body_primary = acknowledged_primary
                if body_primary is exc:
                    raise
                raise body_primary from exc
            if not state["cleanup_complete"]:
                return finish_before_read(exc)
            raise
        if body_primary is not None:  # a foreign context must not suppress it
            acknowledged_primary = acknowledge_preserving(body_primary)
            raise (acknowledged_primary
                   if acknowledged_primary is not None else body_primary)
        ack_error = acknowledge_preserving(None)
        if ack_error is not None:
            raise ack_error
        return result

    try:
        return drive()
    except BaseException as escaped:
        primary = escaped
        if not state["entry_complete"]:
            entry_primary, entered = enter_operation(primary)
            if entry_primary is not None:
                primary = entry_primary
            if not entered:
                if primary is escaped:
                    raise
                raise primary from escaped
            state["entry_complete"] = True
        if not state["cleanup_complete"]:
            return finish_before_read(primary)
        if state["drops_confirmed"] and not state["ack_complete"]:
            acknowledged_primary = acknowledge_preserving(primary)
            if acknowledged_primary is not None:
                primary = acknowledged_primary
        if primary is escaped:
            raise
        raise primary from escaped


def _read_and_release(
    desc: dict,
    read_guard: Any,
    *,
    read_parts: Callable[[list[dict], torch.Tensor], None],
    settle: Callable,
    fail_closed: Callable,
    read_timeout: float,
    get_margin: float,
    initial_primary: BaseException | None = None,
    before_failure_ack: Callable[[], None] | None = None,
    mark_operation_entry: Callable[[], None] | None = None,
    publish_operation: Callable[[Any, BaseException | None], None] | None = None,
) -> Any:
    """Own receiver entry through durable operation-outcome publication."""
    result: Any = None
    primary: BaseException | None = None
    try:
        result = _read_and_release_owned(
            desc, read_guard, read_parts=read_parts, settle=settle,
            fail_closed=fail_closed, read_timeout=read_timeout,
            get_margin=get_margin, initial_primary=initial_primary,
            before_failure_ack=before_failure_ack,
            mark_operation_entry=mark_operation_entry)
    except BaseException as error:
        primary = error
    finally:
        if publish_operation is not None:
            try:
                publish_operation(result, primary)
            except BaseException as publication_error:
                primary = prefer_error(
                    primary, publication_error,
                    "receiver outcome publication failed")
    if primary is not None:
        raise primary
    return result
