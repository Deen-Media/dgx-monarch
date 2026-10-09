"""Exact-once ownership for driver RDMA scratch-thread handoff."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from . import rdma_job_registry as _registry
from . import rdma_read_token
from .rdma_read_attempt import NOT_ENTERED
from .rdma_read_attempt import Claim as _Claim

pending_job_count = _registry.pending_job_count
pending_job_count_for_handle = _registry.pending_job_count_for_handle
pending_job_states = _registry.pending_job_states
_register = _registry.register
_registered = _registry.registered
_unregister = _registry.unregister
_prefer = _registry.prefer
_retire_token = _registry.retire_token
__all__ = ("pending_job_count", "pending_job_count_for_handle", "pending_job_states")
PENDING = "PENDING"
RUNNING = "RUNNING"
RESULT_READY = "RESULT_READY"
CONSUMED = "CONSUMED"
class LatentReadJob:
    """Process-rooted descriptor job with one native-settlement winner."""
    def __init__(
        self,
        guard: Any,
        operation: Callable[
            [BaseException | None, Callable[[], None] | None,
             Callable[[], None], Callable[..., None]],
            Any,
        ],
        holder: list[tuple[Any, BaseException | None]],
        construction_owner: list[LatentReadJob],
    ) -> None:
        self.guard = guard
        self.operation = operation
        self.holder = holder
        self.token_owner: list[Any] = []
        self.runners: list[threading.Thread] = []
        self.sync_recovery_claim = _Claim()
        self.claims: list[_Claim] = [self.sync_recovery_claim]
        self.runner_errors: list[BaseException] = []
        self.outcome_owner: list[tuple[Any, BaseException | None]] = []
        self.confirmation_error: BaseException | None = None
        self.token_retired = False
        self._phase: tuple[str, object | None] = (PENDING, None)
        self._condition = threading.Condition()
        construction_owner.append(self)

    @property
    def state(self) -> str:
        return self._phase[0]

    @property
    def token_capable(self) -> bool:
        return rdma_read_token.token_capable(self)

    def new_runner(
        self, thread_name: str, primary: BaseException | None = None
    ) -> tuple[_Claim, threading.Thread, BaseException | None]:
        """Root a claim and its unstarted Thread before read-token publication."""
        claim = _Claim(primary)
        runner = threading.Thread(
            target=self.run_thread, args=(claim,), name=thread_name)
        evidence: BaseException | None = None
        try:
            self.claims.append(claim)
            self.runners.append(runner)
        except BaseException as error:
            evidence = error
            if not any(existing is claim for existing in self.claims):
                self.claims.append(claim)
            if not any(existing is runner for existing in self.runners):
                self.runners.append(runner)
        return claim, runner, evidence

    def _token_is_confirmed(self) -> bool:
        return rdma_read_token.token_is_confirmed(self)

    def _reconcile_token(
        self, primary: BaseException | None
    ) -> tuple[bool, BaseException | None]:
        return rdma_read_token.reconcile_token(self, primary)

    def prepare(self) -> tuple[bool, BaseException | None]:
        """Publish and prove the exact caller-owned token before any start."""
        if not self.token_capable:
            self.token_retired = True
            return True, None
        primary: BaseException | None = None
        try:
            try:
                self.guard.begin_read(self.token_owner)
            except BaseException as error:
                primary = error
            return self._reconcile_token(primary)
        except BaseException as boundary:
            primary = _prefer(primary, boundary, "read-token preparation interrupted")
        return self._reconcile_token(primary)

    def reconcile_prepare(
        self, primary: BaseException,
    ) -> tuple[bool, BaseException | None]:
        """Reconcile an interrupted prepare call, including tokenless jobs."""
        return rdma_read_token.reconcile_prepare(self, primary)

    def _before_failure_ack(self) -> None:
        rdma_read_token.before_failure_ack(self)

    def _claim(self, claim: _Claim) -> bool:
        with self._condition:
            phase, owner = self._phase
            if phase == PENDING:
                self._phase = (RUNNING, claim)
                return True
            return phase == RUNNING and owner is claim

    def _resolve_claim(self, claim: _Claim) -> bool:
        """Adopt an interrupted atomic claim without creating a second winner."""
        try:
            owns = self._claim(claim)
        except BaseException as error:
            claim.remember(error, "claim publication was interrupted")
            phase, owner = self._phase
            if phase == PENDING:
                try:
                    owns = self._claim(claim)
                except BaseException as retry_error:
                    claim.remember(retry_error, "claim publication retry failed")
                    phase, owner = self._phase
                    owns = phase == RUNNING and owner is claim
            else:
                owns = phase == RUNNING and owner is claim
        return owns

    def _mark_token_retired(self) -> bool:
        return rdma_read_token.mark_token_retired(self)

    def _publish_outcome(
        self, result: Any, primary: BaseException | None
    ) -> tuple[Any, BaseException | None]:
        """Publish one durable outcome; its owner is authoritative over notify."""
        outcome = (result, primary)
        for _attempt in range(2):
            try:
                if not self.outcome_owner:
                    self.outcome_owner.append(outcome)
                prior_result, prior_error = self.outcome_owner[0]
                if primary is not None and primary is not prior_error:
                    prior_error = _prefer(
                        prior_error, primary, "later outcome publication failed")
                    self.outcome_owner[0] = (prior_result, prior_error)
                outcome = self.outcome_owner[0]
                with self._condition:
                    self._phase = (RESULT_READY, outcome)
                    self._condition.notify_all()
                return outcome
            except BaseException as error:
                primary = _prefer(
                    primary, error, "outcome publication was interrupted")
                outcome = (result, primary)
        if not self.outcome_owner:
            self.outcome_owner.append(outcome)
        else:
            prior_result, prior_error = self.outcome_owner[0]
            retry_error = primary if primary is not None else RuntimeError(
                "outcome publication retry failed without evidence")
            self.outcome_owner[0] = (prior_result, _prefer(
                prior_error, retry_error, "outcome publication retry failed"))
        outcome = self.outcome_owner[0]
        with self._condition:
            self._phase = (RESULT_READY, outcome)
            self._condition.notify_all()
        return self.outcome_owner[0]

    def _wait_outcome(self) -> tuple[Any, BaseException | None]:
        while True:
            if self.outcome_owner:
                outcome = self.outcome_owner[0]
                if self._phase[0] == RUNNING:
                    with self._condition:
                        self._phase = (RESULT_READY, outcome)
                        self._condition.notify_all()
                return outcome
            with self._condition:
                self._condition.wait(timeout=0.05)

    def _finish_winner(
        self, result: Any, primary: BaseException | None
    ) -> tuple[Any, BaseException | None]:
        if self.token_owner:
            for attempt in range(2):
                try:
                    evidence = _retire_token(self.guard, self.token_owner[0])
                except BaseException as cleanup:
                    primary = _prefer(
                        primary, cleanup,
                        "scratch read-token cleanup failed" if attempt == 0
                        else "scratch read-token cleanup retry failed")
                    continue
                if evidence is not None:
                    primary = _prefer(
                        primary, evidence, "scratch read-token cleanup recovered")
                break
        try:
            self._mark_token_retired()
        except BaseException as proof_error:
            primary = _prefer(
                primary, proof_error, "scratch read-token retirement proof failed")
        return self._publish_outcome(result, primary)

    def _settle_claim(
        self, claim: _Claim, escaped: BaseException | None = None
    ) -> tuple[Any, BaseException | None]:
        if self.outcome_owner:
            result, error = self.outcome_owner[0]
            if escaped is not None:
                error = _prefer(error, escaped, "winner finalization escaped")
            return self._publish_outcome(result, error)
        if claim.operation_outcome:
            result, error = claim.operation_outcome[0]
            if escaped is not None:
                error = _prefer(error, escaped, "winner finalization escaped")
            try:
                return self._finish_winner(result, error)
            except BaseException as finish_error:
                error = _prefer(error, finish_error, "winner finalization failed")
                try:
                    return self._finish_winner(result, error)
                except BaseException as retry_error:
                    error = _prefer(
                        error, retry_error, "winner finalization retry failed")
                    return self._publish_outcome(result, error)
        if escaped is not None and claim.operation_phase[0] == NOT_ENTERED:
            claim.remember(escaped, "operation entry handoff failed")
            self._invoke_operation(claim)
            if claim.operation_outcome:
                return self._settle_claim(claim)
        error = escaped if escaped is not None else claim.primary
        if error is None:
            error = RuntimeError("RDMA operation outcome was not durably published")
        return self._publish_outcome(None, error)

    def _invoke_operation(self, claim: _Claim) -> None:
        result: Any = None
        try:
            result = self.operation(
                claim.primary,
                self._before_failure_ack if claim.primary is not None else None,
                claim.enter_operation,
                claim.publish_operation,
            )
        except BaseException as error:
            if claim.operation_phase[0] == NOT_ENTERED:
                claim.remember(error, "operation call failed before apply")
                try:
                    result = self.operation(
                        claim.primary, self._before_failure_ack,
                        claim.enter_operation, claim.publish_operation)
                except BaseException as retry_error:
                    claim.remember(retry_error, "operation retry failed before apply")
                    claim.publish_operation(None, retry_error)
                else:
                    claim.publish_operation(result, None)
            else:
                claim.publish_operation(None, error)
        else:
            claim.publish_operation(result, None)

    def _execute_claim(self, claim: _Claim) -> tuple[Any, BaseException | None]:
        self._invoke_operation(claim)
        return self._settle_claim(claim)

    def run_once(self, claim: _Claim) -> Any:
        """Run or wait for the exact claim; native settlement is never retried."""
        try:
            owns = self._resolve_claim(claim)
        except BaseException as claim_error:
            phase, owner = self._phase
            if phase == RUNNING and owner is claim:
                claim.remember(claim_error, "claimed runner handoff failed")
                owns = True
            elif phase == PENDING:
                claim.remember(claim_error, "claim attempt failed")
                owns = self._resolve_claim(claim)
            else:
                raise
        if not owns:
            waited_result, error = self._wait_outcome()
            if error is not None:
                raise error
            return waited_result
        try:
            outcome = self._execute_claim(claim)
        except BaseException as escaped:
            outcome = self._settle_claim(claim, escaped)
        if outcome[1] is not None:
            raise outcome[1]
        return outcome[0]

    def _deliver_outcome(self) -> tuple[Any, BaseException | None]:
        with self._condition:
            return self._deliver_outcome_locked()

    def _deliver_outcome_locked(self) -> tuple[Any, BaseException | None]:
        outcome = self._wait_outcome()
        try:
            if not self.holder:
                self.holder.append(outcome)
        except BaseException as delivery_error:
            result, error = self.outcome_owner[0]
            error = _prefer(error, delivery_error, "caller outcome copy failed")
            outcome = (result, error)
            self.outcome_owner[0] = outcome
            if self.holder:
                self.holder[0] = outcome
            else:
                self.holder.append(outcome)
        outcome = self.outcome_owner[0]
        if self.holder[0] is not outcome:
            self.holder[0] = outcome
        if self._mark_token_retired():
            try:
                with self._condition:
                    self._phase = (CONSUMED, outcome)
                    self._condition.notify_all()
                root_error = _unregister(self)
            except BaseException as root_boundary:
                if not _registered(self):
                    result, error = self.outcome_owner[0]
                    outcome = (result, _prefer(
                        error, root_boundary, "scratch job root retirement failed"))
                    self.outcome_owner[0] = self.holder[0] = outcome
                    return outcome
                raise
            if root_error is not None:
                existing_result, existing_error = outcome
                error = _prefer(
                    existing_error, root_error, "scratch job root retirement recovered")
                outcome = (existing_result, error)
                self.outcome_owner[0] = self.holder[0] = outcome
        return outcome

    def _reconcile_thread(
        self, claim: _Claim, escaped: BaseException | None
    ) -> BaseException | None:
        if (escaped is not None and self.outcome_owner
                and self._phase[0] == RESULT_READY):
            result, outcome_error = self.outcome_owner[0]
            self._publish_outcome(
                result, _prefer(
                    outcome_error, escaped,
                    "runner escaped after durable outcome publication"))
        try:
            owns = self.owns_claim(claim)
        except BaseException as ownership_error:
            escaped = _prefer(
                escaped, ownership_error, "winner ownership proof failed")
            owns = self._phase[0] == RUNNING and self._phase[1] is claim
        if not owns and self._phase[0] == PENDING:
            if escaped is not None:
                claim.remember(escaped, "runner escaped before claim entry")
            owns = self._resolve_claim(claim)
        if owns:
            if not self.outcome_owner and not claim.operation_outcome:
                if claim.operation_phase[0] == NOT_ENTERED:
                    if escaped is not None:
                        claim.remember(
                            escaped, "claimed runner escaped before operation")
                    self._execute_claim(claim)
                else:
                    self._settle_claim(claim, escaped)
            else:
                self._settle_claim(claim, escaped)
        return escaped

    def run_thread(self, claim: _Claim) -> None:
        escaped: BaseException | None = None
        try:
            self.run_once(claim)
        except BaseException as error:
            escaped = error
        finally:
            for _attempt in range(2):
                try:
                    escaped = self._reconcile_thread(claim, escaped)
                    break
                except BaseException as recovery_error:
                    escaped = _prefer(
                        escaped, recovery_error, "winner recovery failed")
            if (not self.outcome_owner and self._phase[0] == RUNNING
                    and self._phase[1] is claim):
                self._publish_outcome(None, escaped if escaped is not None else RuntimeError(
                    "winner recovery exhausted without an outcome"))
            if (escaped is not None
                    and not (self._phase[0] == RUNNING and self._phase[1] is claim)):
                handed_off = escaped
                try:
                    self.runner_errors.append(handed_off)
                except BaseException as handoff_error:
                    escaped = _prefer(
                        handed_off, handoff_error, "runner error handoff failed")
                    for index in range(len(self.runner_errors) - 1, -1, -1):
                        if self.runner_errors[index] is handed_off:
                            self.runner_errors[index] = escaped
                            break
                    else:
                        self.runner_errors.append(escaped)
            try:
                if self.outcome_owner:
                    for _attempt in range(2):
                        try:
                            self._deliver_outcome()
                            break
                        except BaseException as delivery_error:
                            result, delivery_primary = self.outcome_owner[0]
                            self._publish_outcome(result, _prefer(
                                delivery_primary, delivery_error,
                                "outcome delivery failed"))
            finally:
                if self.outcome_owner and _registered(self):
                    try:
                        self._deliver_outcome()
                    except BaseException as final_delivery_error:
                        try:
                            _registry.recover_final_delivery(
                                self, final_delivery_error)
                        except BaseException as recovery_error:
                            _registry.recover_final_delivery(
                                self, _prefer(final_delivery_error, recovery_error,
                                              "final delivery recovery failed"))

    def result_from_holder(self) -> Any:
        outcome = self.holder[0] if self.holder else self._deliver_outcome()
        result, error = outcome
        if error is not None:
            raise error
        return result

    def owns_claim(self, claim: _Claim) -> bool:
        phase, owner = self._phase
        return phase == RUNNING and owner is claim
