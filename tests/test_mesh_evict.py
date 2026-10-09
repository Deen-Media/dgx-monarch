"""Supervision-failure eviction classifier and shutdown on evict (mesh_helpers.py).

A supervision-class failure evicts and tears the dead fleet down; an ordinary
endpoint error leaves the warm mesh alone; eviction's teardown is best-effort,
so a dead fleet cannot block it.
"""
from dgx_monarch import mesh_helpers, mesh_lease, mesh_recycle, mesh_safety, mesh_setup
from dgx_monarch.mesh import mark_defunct_on_supervision_failure
from dgx_monarch.nodes.render_session import RenderSession
from topology_transition_helpers import (
    _acks,
    _dispatch_sample,
    _Future,
    _handle,
    _ValueMesh,
    _Workers,
)


class _FakeHandle:
    def __init__(self, shutdown_raises: bool = False):
        self.defunct = False
        self.shutdown_called = False
        self._raises = shutdown_raises

    def shutdown(self, timeout_s: float = 60.0, **kwargs) -> None:
        self.shutdown_called = True
        if self._raises:
            raise RuntimeError("procs already gone")


class SupervisionError(Exception):
    """Stand-in whose type name carries the classifier marker."""


def test_supervision_failure_evicts_and_tears_down():
    handle = _FakeHandle()
    mark_defunct_on_supervision_failure(handle, SupervisionError("worker 1 proc exited"))
    assert handle.defunct
    assert handle.shutdown_called  # stops the dead procs and frees the NCCL port


def test_endpoint_error_does_not_evict():
    handle = _FakeHandle()
    mark_defunct_on_supervision_failure(handle, ValueError("bad sampler input"))
    assert not handle.defunct
    assert not handle.shutdown_called  # the actor is alive, so the warm mesh is reused


def test_marker_in_message_also_evicts():
    # Some monarch failures surface as a generic type with the marker in the text.
    handle = _FakeHandle()
    mark_defunct_on_supervision_failure(handle, RuntimeError("connection lost to worker"))
    assert handle.defunct


def test_eviction_shutdown_is_best_effort():
    # A dead fleet's shutdown may itself raise; eviction must still complete.
    handle = _FakeHandle(shutdown_raises=True)
    mark_defunct_on_supervision_failure(handle, SupervisionError("proc died"))
    assert handle.defunct
    assert handle.shutdown_called


def test_a_stall_eviction_heals_instead_of_wedging_the_session():
    """End to end: eviction, lease retirement, then a clean lifecycle again.

    If the deferred eviction never reconciles, the handle stays defunct with no
    teardown outcome: attach refuses it as unresolved and recycle refuses it as
    PRIOR_TEARDOWN_UNKNOWN.
    """
    workers = _Workers([_Future(_acks())])
    handle, _old, _new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)
    session = RenderSession(timeout_s=0)
    session.bind(handle)
    with session.activate():
        lease = _dispatch_sample(
            handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    mesh_helpers.mark_defunct_deliberate(
        handle, mesh_lease.SampleStallError("render stalled"))

    # While the lease is live the handle has no terminal outcome yet.
    assert mesh_safety.coherent_lifecycle_verdict(handle, RuntimeError) == "unresolved"

    mesh_setup.abandon_sample(lease)
    session.close()

    assert mesh_safety.coherent_lifecycle_verdict(handle, RuntimeError) == "completed"
    retired, evicted = [], []
    outcome = mesh_recycle.recycle_detailed_impl(
        handle, lambda: retired.append(1), lambda: evicted.append(1))
    assert outcome.status is mesh_recycle.RecycleStatus.RECYCLED
    assert outcome.already_recycled is True
    assert retired == [1]
