"""Driver-side setup-generation leases for sample results and RDMA reads."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from . import mesh_lease_retirement as _retirement
from .mesh_lease_retirement import (
    _RetirementPlan,
    _safe_error,
    log,
)
from .rdma_receiver import ack_timeout_s, validate_ack_responses
from .transfer_utils import prefer_error, raise_with_distinct_cause, reconcile_error
from .transfer_utils import safe_note as _safe_note


class _ReadToken:
    __slots__ = ("key", "lease")

    def __init__(self, lease: SetupBoundFuture) -> None:
        self.lease = lease
        self.key = object()

    def __del__(self) -> None:
        try:
            self.lease.end_read(self.key)
        except BaseException as exc:
            try:
                self.lease.end_read(self.key)
            except BaseException as retry_exc:
                strongest = prefer_error(
                    exc, retry_exc, "lost read-token cleanup retry failed")
                _safe_error(strongest, "lost read-token cleanup failed: %r", strongest)


@dataclass
class SetupBoundFuture:
    """Keep one setup generation live until its result is consumed or abandoned."""

    future: Any
    handle: Any
    generation: int
    state: str = "active"
    retire_to: str | None = None
    finalizers: list[Any] = field(default_factory=list, repr=False)
    _reader_tokens: set[object] = field(default_factory=set, repr=False)
    _registered: bool = field(default=True, repr=False)
    _enqueue_started: bool = field(default=True, repr=False)
    _registration_target: int | None = field(default=None, repr=False)
    _retirement_plan: _RetirementPlan | None = field(default=None, repr=False)
    _pending_completion: tuple[BaseException | None, list[Any]] | None = field(
        default=None, repr=False)
    _pending_deliberate: bool = field(default=False, repr=False)

    @classmethod
    def prepared(cls, handle: Any, generation: int) -> SetupBoundFuture:
        """Create caller-owned authority before an enqueue can have effects."""
        return cls(None, handle, generation, _registered=False,
                   _enqueue_started=False)

    @property
    def readers(self) -> int:
        return len(self._reader_tokens)

    def get(self, timeout: float | None = None) -> Any:
        with self.handle.lock:
            if self.state != "active" or self.retire_to is not None:
                terminal = self.retire_to or self.state
                raise RuntimeError(f"sample result was already {terminal}")
        result = self.future.get(timeout=timeout)
        with self.handle.lock:
            if self.state != "active" or self.retire_to is not None:
                terminal = self.retire_to or self.state
                raise RuntimeError(f"sample result became {terminal} while collecting")
        return result

    def ack_latent(self, setup_generation: int, token: str) -> None:
        """Confirm local drops and read-authority exit for worker retirement."""
        if (type(setup_generation) is not int or setup_generation != self.generation
                or type(token) is not str or len(token) != 32
                or any(char not in "0123456789abcdef" for char in token)):
            raise RuntimeError("RDMA descriptor names a setup generation other than its sample lease's, or a malformed owner token")
        rows = self.handle.call_all(
            "ack_latent_handoff", setup_generation, token, timeout_s=ack_timeout_s())
        validate_ack_responses(rows, setup_generation, token, self.handle.world)

    def begin_read(self, owner: list[Any] | None = None) -> object:
        """Pin backing before a raw RDMA read begins, including scratch threads."""
        token = owner[0] if owner else _ReadToken(self)
        if owner is not None and not owner:
            owner.append(token)
        self.ensure_read_token(token)
        return token

    def ensure_read_token(self, token: object) -> None:
        if not isinstance(token, _ReadToken) or token.lease is not self:
            raise RuntimeError("read token does not belong to this sample lease")
        self._begin_read_token(token.key)

    def has_read_token(self, token: object) -> bool:
        if not isinstance(token, _ReadToken) or token.lease is not self:
            raise RuntimeError("read token does not belong to this sample lease")
        with self.handle.lock:
            return token.key in self._reader_tokens

    def _begin_read_token(self, token: object) -> None:
        """Publish one idempotent read token under the lease lock."""
        with self.handle.lock:
            if self.state != "active" or self.retire_to is not None:
                terminal = self.retire_to or self.state
                raise RuntimeError(f"cannot read a sample result already {terminal}")
            # Copy-on-write in one atomic store. Owner paths put the _ReadToken in the
            # owner list first; read_authority enters both cleanup scopes first.
            self._reader_tokens = {*self._reader_tokens, token}

    @contextmanager
    def read_authority(self, owner: list[Any] | None = None) -> Iterator[None]:
        """Hold one tokenized, idempotently released RDMA read authority."""
        read_token = self.begin_read(owner) if owner is not None else None
        token = read_token.key if isinstance(read_token, _ReadToken) else object()
        primary: BaseException | None = None
        first_cleanup = second_cleanup = primary
        # If inner cleanup is interrupted, outer finally retries this exact
        # token; after inner success that retry is idempotent.
        try:
            try:
                try:
                    self._begin_read_token(token)
                    yield
                except BaseException as exc:
                    primary = exc
            finally:
                try:
                    first_cleanup = self._capture_end_read(token)
                except BaseException as exc:
                    first_cleanup = exc
        finally:
            try:
                second_cleanup = self._capture_end_read(token)
            except BaseException as exc:
                second_cleanup = exc
        strongest, cause = primary, None
        for cleanup in (first_cleanup, second_cleanup):
            if cleanup is None:
                continue
            strongest, cause = reconcile_error(
                strongest, cleanup, "sample read-authority cleanup failed")
        if strongest is not None:
            raise_with_distinct_cause(strongest, cause)

    def _capture_end_read(self, token: object) -> BaseException | None:
        try:
            self.end_read(token)
        except BaseException as exc:
            return exc
        return None

    def add_finalizer(self, callback: Any) -> None:
        """Run ``callback`` after terminal state, outside the handle lock."""
        run_now = drain_pending = False
        with self.handle.lock:
            if self.state != "active" and self.retire_to is None:
                pending = self._pending_completion
                if pending is None:
                    run_now = True
                elif not any(existing is callback for existing in pending[1]):
                    pending[1].append(callback)
                    drain_pending = True
            elif not any(existing is callback for existing in self.finalizers):
                self.finalizers.append(callback)
        if run_now:
            self._run_finalizers([callback])
        elif drain_pending:
            self._drain_finalization()

    def has_finalizer(self, callback: Any) -> bool:
        """Whether this exact callback is installed or awaiting completion."""
        with self.handle.lock:
            if any(existing is callback for existing in self.finalizers):
                return True
            pending = self._pending_completion
            return bool(
                pending is not None
                and any(existing is callback for existing in pending[1])
            )

    def end_read(self, token: object | None = None) -> None:
        """Retire one read token; explicit-token calls are idempotent."""
        token_to_end = token.key if isinstance(token, _ReadToken) else token
        first_error: BaseException | None = None
        error_cause: BaseException | None = None
        try:
            if token_to_end is None:
                with self.handle.lock:
                    if not self._reader_tokens:
                        raise RuntimeError("sample RDMA read lease underflow")
                    token_to_end = next(iter(self._reader_tokens))
            self._end_read_token(token_to_end)
        except BaseException as exc:
            first_error = exc
            if token_to_end is not None:
                try:
                    self._end_read_token(token_to_end)
                except BaseException as retry_exc:
                    first_error, error_cause = reconcile_error(
                        first_error, retry_exc,
                        "sample read-token retirement retry failed")
                    _safe_error(first_error,
                                "read-token retirement failed twice: %r; retry: %r",
                                exc, retry_exc)
        finally:
            try:
                self._drain_finalization()
            except BaseException as drain_exc:
                first_error, error_cause = reconcile_error(
                    first_error, drain_exc, "sample read finalization also failed")
        if first_error is not None:
            raise_with_distinct_cause(first_error, error_cause)

    def _end_read_token(self, token: object) -> None:
        with self.handle.lock:
            if token in self._reader_tokens:
                remaining = set(self._reader_tokens)
                remaining.remove(token)
                self._reader_tokens = remaining
            if (not self._reader_tokens and self.state == "active"
                    and self.retire_to is not None):
                self._finalize_locked(self.retire_to)

    def release(self) -> None:
        self._retire("consumed")

    def abandon(self) -> None:
        self._retire("abandoned")

    def _retire(self, state: str) -> None:
        return _retirement.retire(self, state)

    def _retire_and_drain_once(self, state: str) -> None:
        """Commit retirement and complete its durable owner handoff."""
        try:
            self._retire_once(state)
        finally:
            self._drain_finalization()

    def _retire_once(self, state: str) -> None:
        return _retirement.retire_once(self, state)

    def _finalize_locked(self, state: str) -> None:
        return _retirement.finalize_locked(self, state)

    def _build_retirement_plan_locked(self, state: str) -> _RetirementPlan:
        return _retirement.build_retirement_plan_locked(self, state)

    def _apply_retirement_plan_locked(self) -> None:
        return _retirement.apply_retirement_plan_locked(self)

    def _register_locked(self) -> None:
        """Count prepared authority before the first enqueue instruction."""
        if self.state != "active" or self.retire_to is not None:
            raise RuntimeError("sample authority is already terminal")
        if self._registered:
            raise RuntimeError("sample authority is already registered")
        if self.future is not None:
            raise RuntimeError("prepared sample authority already has a future")
        if self.generation < 0:
            raise RuntimeError("sample authority has an invalid setup generation")
        leases = self.handle.sample_leases
        try:
            self._registration_target = int(leases.get(self.generation, 0)) + 1
            self._apply_registration_locked()
        except BaseException as primary:
            try:
                self._apply_registration_locked()
            except BaseException as retry:
                winner, cause = reconcile_error(
                    primary, retry, "registration commit retry failed")
                raise_with_distinct_cause(winner, cause)
            raise

    def _apply_registration_locked(self) -> None:
        target = self._registration_target
        if target is None:
            return
        self.handle.sample_leases[self.generation] = target
        self._registered = True
        self._registration_target = None

    def _mark_enqueue_started_locked(self) -> None:
        if not self._registered:
            raise RuntimeError("sample authority was not registered")
        # The endpoint may have accepted work; later failure is abandonment.
        self._enqueue_started = True

    def _bind_future_locked(self, future: Any) -> None:
        if future is None:
            raise RuntimeError("sample dispatch returned no future")
        self.future = future

    def abandon_after_dispatch(self, primary: BaseException) -> None:
        """Fail closed without letting cleanup replace a dispatch failure."""
        try:
            self.abandon()
        except BaseException as cleanup_exc:
            try:
                self.abandon()
            except BaseException as retry_exc:
                _safe_note(primary, "sample authority abandonment failed",
                           cleanup_exc)
                _safe_note(primary, "sample authority abandonment retry failed",
                           retry_exc)

    def _drain_finalization(self) -> None:
        return _retirement.drain_finalization(self)

    def _complete_finalization(self, deferred: BaseException | None,
                               finalizers: list[Any]) -> None:
        return _retirement.complete_finalization(self, deferred, finalizers)

    @staticmethod
    def _run_finalizers(finalizers: list[Any]) -> None:
        return _retirement.run_finalizers(finalizers)

    def _resume_deferred_supervision(self, exc: BaseException | None) -> None:
        return _retirement.resume_deferred_supervision(self, exc)


def token_kwargs(token: Any) -> dict[str, Any]:
    return {"setup_token": token} if token is not None else {}


def submit_sample(handle: Any, request: dict, progress_port: Any, token: Any,
                  authority: SetupBoundFuture | None = None) -> Any:
    kwargs = token_kwargs(token)
    if authority is not None:
        kwargs["authority"] = authority
    return handle.submit_sample(request, progress_port=progress_port, **kwargs)


def prepare_sample(handle: Any, token: Any) -> SetupBoundFuture | None:
    """Create production authority before an enqueue can be attempted."""
    from .mesh import MeshHandle
    from .mesh_setup import SetupToken

    if not isinstance(handle, MeshHandle) or not isinstance(token, SetupToken):
        return None
    return SetupBoundFuture.prepared(handle, token.generation)


# A worker that dies mid-collective (a sticky CUDA fault, an OOM-killed rank)
# can leave its peer blocked in NCCL and the reply missing on every rank, so
# one native future.get would wait until the hard timeout. The liveness wait
# below polls in short slices (a timed-out monarch Future.get keeps the result
# collectable) and between slices compares the progress receiver's last
# message stamp with a stall budget. When the fleet is silent for the whole
# budget, the wait raises SampleStallError instead of holding the prompt queue
# (docs/TROUBLESHOOTING.md #86).

SAMPLE_STALL_ENV = "DGXM_SAMPLE_STALL_TIMEOUT_S"
SAMPLE_STALL_DEFAULT_S = 600.0
_STALL_SLICE_S = 20.0


class SampleStallError(RuntimeError):
    """The fleet stopped reporting progress for the whole stall budget."""


def stall_budget_s() -> float:
    """The configured stall budget; 0 or negative disables the guard."""
    import os

    raw = os.environ.get(SAMPLE_STALL_ENV)
    if raw is None:
        return SAMPLE_STALL_DEFAULT_S
    try:
        return float(raw)
    except ValueError:
        log.warning("ignoring non-numeric %s=%r; using the default stall budget", SAMPLE_STALL_ENV, raw)
        return SAMPLE_STALL_DEFAULT_S


def collect_with_liveness(handle: Any, future: Any, timeout_s: float,
                          activity_fn: Any = None) -> Any:
    """Await a sample future under both the hard timeout and the stall budget.

    ``activity_fn`` returns the monotonic stamp of the fleet's last progress
    message (progress.ProgressReceiver.activity). Without one, or with the
    budget disabled, this is exactly the plain off-loop get.
    """
    import time as _time

    from . import mesh_runtime

    budget = stall_budget_s()
    if activity_fn is None or budget <= 0:
        return mesh_runtime.get_off_loop(future, timeout_s, "dgxm-await-sample")
    # Measure idle from the later of the last progress message and the start
    # of this wait. A pipelined or deferred collect can begin long after the
    # receiver was built, while the sample still queues behind an earlier
    # render's GPU lock; anchoring at construction would call that healthy
    # silence a stall.
    wait_start = _time.monotonic()
    deadline = wait_start + float(timeout_s)
    while True:
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            # Let the canonical hard-timeout error surface unchanged.
            return mesh_runtime.get_off_loop(future, 0.0, "dgxm-await-sample")
        try:
            return mesh_runtime.get_off_loop(
                future, min(_STALL_SLICE_S, remaining), "dgxm-await-sample")
        except TimeoutError:
            idle = _time.monotonic() - max(float(activity_fn()), wait_start)
            if idle < budget:
                continue
            raise SampleStallError(
                f"render stalled: the fleet reported no progress for "
                f"{idle:.0f}s (budget {budget:.0f}s; {SAMPLE_STALL_ENV} "
                "sets it, 0 disables). A rank that dies mid-collective "
                "leaves its peers blocked and no rank replies, so the driver "
                "stopped waiting instead of holding the prompt queue. The "
                "mesh is marked defunct and the next render respawns the "
                "workers. Each worker's journal shows what its rank was doing "
                "(docs/TROUBLESHOOTING.md #86).") from None


def collect_sample(handle: Any, future: Any, timeout_s: float,
                   activity_fn: Any = None) -> list:
    """Collect a sample and evict an unusable fleet.

    A stall uses ``mark_defunct_deliberate`` because no supervision event
    occurred. Other failures follow ``MeshHandle._await_or_evict``.
    """
    from . import mesh_helpers

    try:
        value_mesh = collect_with_liveness(handle, future, timeout_s, activity_fn)
    except SampleStallError as exc:
        try:
            mesh_helpers.mark_defunct_deliberate(handle, exc, holding_lock=False)
        except BaseException as publication_exc:
            log.warning("stall eviction publication failed: %r", publication_exc)
        raise
    except Exception as exc:
        mesh_helpers.mark_defunct_preserving_primary(
            handle, exc, holding_lock=False)
        raise
    return [value for _point, value in value_mesh.items()]
