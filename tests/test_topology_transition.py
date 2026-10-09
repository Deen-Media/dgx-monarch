"""Fail-closed same-handle topology transition regressions."""
from __future__ import annotations

import threading

import pytest

from dgx_monarch import mesh as mesh_mod
from dgx_monarch import mesh_lease, mesh_safety, mesh_setup
from dgx_monarch.mesh import MeshHandle
from topology_transition_helpers import (
    _acks,
    _BlockingFuture,
    _Endpoint,
    _Future,
    _handle,
    _ValueMesh,
    _Workers,
)


def test_public_collective_wrapper_prepares_authority_before_preflight(
    monkeypatch,
):
    handle, _old, _new = _handle(_Workers([]))
    token = mesh_setup.current_setup_token(handle)
    events = []
    actual_prepare = mesh_lease.prepare_sample

    def prepare(bound_handle, setup_token):
        events.append("prepare")
        return actual_prepare(bound_handle, setup_token)

    def verify(_request, worker_index=None):
        assert worker_index is None
        events.append("preflight")

    def dispatch(bound_handle, request, progress_port, setup_token, authority):
        assert isinstance(authority, mesh_setup.SetupBoundFuture)
        assert authority.handle is bound_handle
        assert authority.generation == setup_token.generation
        bound_handle._verify_request_artifacts(request)
        return authority

    monkeypatch.setattr(mesh_lease, "prepare_sample", prepare)
    monkeypatch.setattr(handle, "_verify_request_artifacts", verify)
    monkeypatch.setattr(mesh_setup, "dispatch_collective_sample", dispatch)

    result = handle.submit_sample({}, setup_token=token)
    assert result.generation == token.generation
    assert events == ["prepare", "preflight"]


def test_public_fleet_wrapper_prepares_authority_before_preflight(monkeypatch):
    handle, _old, _new = _handle(_Workers([]))
    token = mesh_setup.current_setup_token(handle)
    events = []
    actual_prepare = mesh_lease.prepare_sample

    def prepare(bound_handle, setup_token):
        events.append("prepare")
        return actual_prepare(bound_handle, setup_token)

    def verify(_request, worker_index=None):
        assert worker_index == 1
        events.append("preflight")

    def dispatch(
        bound_handle, index, request, progress_port, setup_token, authority,
    ):
        assert isinstance(authority, mesh_setup.SetupBoundFuture)
        assert authority.handle is bound_handle
        assert authority.generation == setup_token.generation
        bound_handle._verify_request_artifacts(request, worker_index=index)
        return authority

    monkeypatch.setattr(mesh_lease, "prepare_sample", prepare)
    monkeypatch.setattr(handle, "_verify_request_artifacts", verify)
    monkeypatch.setattr(mesh_setup, "dispatch_sample_to_actor", dispatch)

    result = handle.submit_sample_to(1, {}, setup_token=token)
    assert result.generation == token.generation
    assert events == ["prepare", "preflight"]


def test_slow_topology_cleanup_retires_identity_before_wait_and_uses_large_budget():
    entered, release = threading.Event(), threading.Event()
    cleanup = _BlockingFuture(_acks(), entered, release)
    workers = _Workers([cleanup])
    handle, _old, new = _handle(workers)
    result = []

    def transition():
        try:
            result.append(handle.ensure_setup(new, "TORCH_FLASH"))
        except BaseException as exc:  # pragma: no cover - assertion reports it
            result.append(exc)

    thread = threading.Thread(target=transition)
    thread.start()
    assert entered.wait(timeout=1.0)
    assert handle.setup_key is None
    assert handle.topology is None
    assert handle.setup_generation == 2
    assert handle.setup_cleanup_state is not None
    assert handle.setup_cleanup_state.outcome is mesh_setup.SetupCleanupOutcome.IN_PROGRESS
    release.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert result and not isinstance(result[0], BaseException)
    assert cleanup.timeouts == [600.0]
    assert handle.setup_key == mesh_mod._nccl_setup_key(new, "TORCH_FLASH", True)
    assert handle.topology == new
    assert handle.setup_generation == 2
    assert handle.setup_cleanup_state is None
    assert all(len(endpoint.envs) == 1 for endpoint in workers.setup_endpoints)
    assert all(
        endpoint.envs[0]["setup_generation"] == 2
        for endpoint in workers.setup_endpoints
    )


def test_setup_key_retirement_interruption_leaves_driver_cleanup_dirty(monkeypatch):
    class StopNow(BaseException):
        pass

    workers = _Workers([_Future(_acks()), _Future(_acks())])
    handle, old, new = _handle(workers)
    original_setattr = MeshHandle.__setattr__
    armed = True

    def interrupt_after_key_retirement(self, name, value):
        nonlocal armed
        original_setattr(self, name, value)
        if self is handle and name == "setup_key" and value is None and armed:
            armed = False
            raise StopNow("interrupted after setup key retirement")

    monkeypatch.setattr(MeshHandle, "__setattr__", interrupt_after_key_retirement)
    with pytest.raises(StopNow):
        handle.ensure_setup(new, "TORCH_FLASH")

    state = handle.setup_cleanup_state
    assert state is not None
    assert state.outcome is mesh_setup.SetupCleanupOutcome.IN_PROGRESS
    assert state.phase == "topology teardown"
    assert handle.setup_key is None and handle.setup_generation == 2
    assert handle.topology == old
    assert workers.teardown_group.calls == []
    assert all(not endpoint.envs for endpoint in workers.setup_endpoints)

    with pytest.raises(mesh_setup.TopologyTransitionError) as raised:
        handle.ensure_setup(new, "TORCH_FLASH")

    assert raised.value.state == state
    assert workers.teardown_group.calls == []
    assert all(not endpoint.envs for endpoint in workers.setup_endpoints)


def test_setup_generation_interruption_leaves_old_ready_driver_dirty(monkeypatch):
    class StopNow(BaseException):
        pass

    workers = _Workers([_Future(_acks())])
    handle, old, new = _handle(workers)
    original_setattr = MeshHandle.__setattr__
    armed = True

    def interrupt_after_generation_advance(self, name, value):
        nonlocal armed
        original_setattr(self, name, value)
        if self is handle and name == "setup_generation" and value == 2 and armed:
            armed = False
            raise StopNow("interrupted after setup generation advance")

    monkeypatch.setattr(
        MeshHandle, "__setattr__", interrupt_after_generation_advance)
    with pytest.raises(StopNow):
        handle.ensure_setup(new, "TORCH_FLASH")

    state = handle.setup_cleanup_state
    assert state is not None
    assert state.outcome is mesh_setup.SetupCleanupOutcome.IN_PROGRESS
    assert state.generation == 2
    assert state.phase == "topology teardown"
    assert handle.setup_generation == 2
    assert handle.setup_key is not None
    assert handle.topology == old
    assert workers.teardown_group.calls == []
    assert all(not endpoint.envs for endpoint in workers.setup_endpoints)

    with pytest.raises(mesh_setup.TopologyTransitionError) as raised:
        handle.ensure_setup(old, "TORCH_FLASH")

    assert raised.value.state == state
    assert workers.teardown_group.calls == []
    assert all(not endpoint.envs for endpoint in workers.setup_endpoints)


def test_caller_setup_timeout_cannot_shorten_group_cleanup_budget():
    cleanup = _Future(_acks())
    setup_futures = [_Future({"rank": 0}), _Future({"rank": 1})]
    workers = _Workers([cleanup], setup_futures=setup_futures)
    handle, _old, new = _handle(workers)

    handle.ensure_setup(new, "TORCH_FLASH", timeout_s=7.0)

    assert cleanup.timeouts == [600.0]
    assert all(
        future.timeouts and 0.0 < future.timeouts[0] <= 7.0
        for future in setup_futures
    )


@pytest.mark.parametrize("interrupted_field", ["worker_args_key", "setup_key"])
def test_interrupted_ready_publication_never_fast_paths_stale_metadata(
    monkeypatch, interrupted_field
):
    class StopNow(BaseException):
        pass

    workers = _Workers([_Future(_acks()), _Future(_acks())])
    handle, _old, new = _handle(workers)
    new_key = mesh_mod._nccl_setup_key(new, "TORCH_FLASH", True)
    worker_key = MeshHandle._worker_args_key({})
    target = worker_key if interrupted_field == "worker_args_key" else new_key
    original_setattr = MeshHandle.__setattr__
    armed = True

    def interrupt_publication(self, name, value):
        nonlocal armed
        original_setattr(self, name, value)
        if self is handle and name == interrupted_field and value == target and armed:
            armed = False
            raise StopNow(f"{name} publication interrupted")

    monkeypatch.setattr(MeshHandle, "__setattr__", interrupt_publication)
    with pytest.raises(StopNow):
        handle.ensure_setup(new, "TORCH_FLASH")

    assert handle.setup_key is None
    assert handle.worker_args_key is None
    assert handle.active_worker_args == {}
    assert handle.topology is None
    assert handle.setup_cleanup_state is not None
    assert (
        handle.setup_cleanup_state.outcome
        is mesh_setup.SetupCleanupOutcome.INTERRUPTED
    )
    with pytest.raises(mesh_setup.TopologyTransitionError):
        handle.ensure_setup(new, "TORCH_FLASH")
    assert all(len(endpoint.envs) == 1 for endpoint in workers.setup_endpoints)


def test_ready_helper_call_boundary_rolls_back_live_worker_setup(monkeypatch):
    class StopNow(BaseException):
        pass

    workers = _Workers([_Future(_acks()), _Future(_acks())])
    handle, _old, new = _handle(workers)

    def stop_before_helper(*_args, **_kwargs):
        raise StopNow("interrupted before READY helper entry")

    monkeypatch.setattr(
        mesh_mod.mesh_helpers, "publish_setup_ready", stop_before_helper)
    with pytest.raises(StopNow, match="READY helper entry"):
        handle.ensure_setup(new, "TORCH_FLASH")

    assert all(len(endpoint.envs) == 1 for endpoint in workers.setup_endpoints)
    assert len(workers.teardown_group.calls) == 2
    assert handle.setup_key is None
    assert handle.worker_args_key is None
    assert handle.topology is None
    assert handle.setup_cleanup_state is not None
    assert (
        handle.setup_cleanup_state.outcome
        is mesh_setup.SetupCleanupOutcome.INTERRUPTED
    )


def test_initial_setup_dispatch_boundary_is_dirty_before_actor_side_effects(
        monkeypatch):
    class StopNow(BaseException):
        pass

    workers = _Workers([_Future(_acks())])
    handle, _old, new = _handle(workers)
    handle.setup_key = None
    handle.worker_args_key = None
    handle.topology = None
    handle.setup_cleanup_state = None

    def dispatch_then_stop(*_args, **_kwargs):
        assert handle.setup_cleanup_state is not None
        assert (
            handle.setup_cleanup_state.outcome
            is mesh_setup.SetupCleanupOutcome.IN_PROGRESS
        )
        raise StopNow("setup dispatch return lost")

    monkeypatch.setattr(mesh_setup, "dispatch_setup", dispatch_then_stop)

    with pytest.raises(StopNow, match="dispatch return lost"):
        handle.ensure_setup(new, "TORCH_FLASH")

    assert handle.setup_key is None
    assert handle.setup_cleanup_state is not None
    assert (
        handle.setup_cleanup_state.outcome
        is mesh_setup.SetupCleanupOutcome.INTERRUPTED
    )


def test_interrupted_worker_policy_marker_forces_complete_reapply(monkeypatch):
    class StopNow(BaseException):
        pass

    workers = _Workers([])
    values = _ValueMesh([{}, {}])
    workers.apply_worker_args = _Endpoint(_Future(values), _Future(values))
    handle, old, _new = _handle(workers)
    target = {"slab_weights": False}
    target_key = MeshHandle._worker_args_key(target)
    original_setattr = MeshHandle.__setattr__
    armed = True

    def interrupt_marker(self, name, value):
        nonlocal armed
        original_setattr(self, name, value)
        if self is handle and name == "worker_args_key" and value == target_key and armed:
            armed = False
            raise StopNow("worker policy marker interrupted")

    monkeypatch.setattr(MeshHandle, "__setattr__", interrupt_marker)
    with pytest.raises(StopNow):
        handle.apply_worker_args(target, merge_config=False)

    assert handle.worker_args_key is None
    assert handle.active_worker_args == {}
    assert len(workers.apply_worker_args.calls) == 1

    handle.ensure_setup(old, "TORCH_FLASH", worker_args=target)

    assert handle.worker_args_key == target_key
    assert handle.active_worker_args == target
    assert len(workers.apply_worker_args.calls) == 2


def test_topology_cleanup_timeout_is_unknown_blocks_dispatch_and_recycles_once(monkeypatch):
    from dgx_monarch.actor import model_store

    cleanup = _Future(error=TimeoutError("still unloading"))
    workers = _Workers([cleanup])
    handle, _old, new = _handle(workers)

    with pytest.raises(mesh_setup.TopologyTransitionError) as raised:
        handle.ensure_setup(new, "TORCH_FLASH")

    state = raised.value.state
    assert state.outcome is mesh_setup.SetupCleanupOutcome.TIMEOUT_UNKNOWN
    assert state.generation == 2
    assert handle.setup_key is None and handle.topology is None
    assert handle.setup_generation == 2
    assert handle.setup_cleanup_state == state
    assert cleanup.timeouts == [600.0]
    assert all(not endpoint.envs for endpoint in workers.setup_endpoints)

    identity = {"digest": "model", "comfy": "commit", "artifacts": []}
    monkeypatch.setattr(
        model_store, "request_artifact_identity", lambda *_args: identity)
    artifact_future = _Future(_ValueMesh([
        {"artifact_sets": [identity]}, {"artifact_sets": [identity]},
    ]))
    workers.artifact_identity = _Endpoint(artifact_future)
    request = {
        "model": {"unet_name": "model.sft", "options": {}, "loras": []},
        "_dgxm_normal_residency_mode": "operator_off",
        "_dgxm_artifact_preflight": {"generation": 1, "worker_index": None},
    }
    handle._verify_request_artifacts(request)
    assert len(workers.artifact_identity.calls) == 1
    assert request["_dgxm_artifact_preflight"]["generation"] == 2

    with pytest.raises(mesh_setup.TopologyTransitionError):
        handle.ensure_setup(new, "TORCH_FLASH")
    with pytest.raises(mesh_setup.TopologyTransitionError):
        handle.call_all("load_model", "model.sft")
    with pytest.raises(mesh_setup.TopologyTransitionError):
        handle.submit_sample({"model": {"unet_name": "model.sft"}})
    with pytest.raises(mesh_mod.MeshAttachError, match="group cleanup is dirty"):
        mesh_mod.ensure_live(handle)
    assert handle.setup_generation == 2

    mesh_safety.clear_stale_token(id(handle))
    monkeypatch.setattr(mesh_mod, "_MESHES", {("test",): handle})
    assert handle._recycle_impl() is True
    assert handle._recycle_impl() is True
    assert handle.procs.reasons == ["dgx-monarch recycle"]
    assert len(workers.teardown_group.calls) == 1


@pytest.mark.parametrize(
    ("cleanup", "outcome"),
    [
        (_Future(error=RuntimeError("worker cleanup failed")),
         mesh_setup.SetupCleanupOutcome.REPORTED_FAILURE),
        (_Future(error=ValueError("worker rejected cleanup")),
         mesh_setup.SetupCleanupOutcome.REPORTED_FAILURE),
        (_Future(_ValueMesh([{"torn_down": True, "cleanup_state": "UNSETUP"}])),
         mesh_setup.SetupCleanupOutcome.INVALID_RESULT),
        (_Future(_ValueMesh([{"torn_down": "yes", "cleanup_state": "UNSETUP"}] * 2)),
         mesh_setup.SetupCleanupOutcome.INVALID_RESULT),
    ],
)
def test_topology_cleanup_non_successes_latch_distinct_dirty_outcomes(cleanup, outcome):
    workers = _Workers([cleanup])
    handle, _old, new = _handle(workers)

    with pytest.raises(mesh_setup.TopologyTransitionError) as raised:
        handle.ensure_setup(new, "TORCH_FLASH")

    assert raised.value.state.outcome is outcome
    assert handle.setup_cleanup_state == raised.value.state
    assert handle.setup_key is None and handle.setup_generation == 2
    assert all(not endpoint.envs for endpoint in workers.setup_endpoints)


def test_topology_cleanup_baseexception_propagates_and_latches_interrupted():
    class StopNow(BaseException):
        pass

    workers = _Workers([_Future(error=StopNow())])
    handle, _old, new = _handle(workers)

    with pytest.raises(StopNow):
        handle.ensure_setup(new, "TORCH_FLASH")

    assert handle.setup_cleanup_state is not None
    assert handle.setup_cleanup_state.outcome is mesh_setup.SetupCleanupOutcome.INTERRUPTED
    assert handle.setup_key is None and handle.setup_generation == 2


def test_idempotent_unsetup_acknowledgements_are_valid_cleanup_results():
    workers = _Workers([_Future(_ValueMesh([
        {"torn_down": True, "cleanup_state": "UNSETUP"},
        {"torn_down": False, "cleanup_state": "UNSETUP"},
    ]))])
    handle, _old, new = _handle(workers)

    handle.ensure_setup(new, "TORCH_FLASH")

    assert handle.setup_cleanup_state is None
    assert handle.topology == new
    with pytest.raises(RuntimeError, match="lifecycle-internal"):
        handle.call_all("teardown_group")
