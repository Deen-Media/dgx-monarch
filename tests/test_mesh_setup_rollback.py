"""Worker-args refusal, setup rollback, budget and worker teardown regressions."""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from dgx_monarch import mesh as mesh_mod
from dgx_monarch import mesh_safety, mesh_setup
from dgx_monarch.mesh import MeshHandle
from topology_transition_helpers import (
    _acks,
    _dispatch_sample,
    _Endpoint,
    _Future,
    _handle,
    _LoopRefusingFuture,
    _Procs,
    _ValueMesh,
    _Workers,
)


def test_apply_worker_args_refuses_before_setup_ready():
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(_Future(_ValueMesh([{}, {}])))
    handle, _old, _new = _handle(workers)
    handle.setup_key = None

    with pytest.raises(RuntimeError, match="before setup is READY"):
        handle.apply_worker_args({"slab_weights": False})

    assert workers.apply_worker_args.calls == []


def test_apply_worker_args_refuses_while_sample_result_is_live():
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(_Future(_ValueMesh([{}, {}])))
    handle, _old, _new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([])), setup_token=token)

    with pytest.raises(mesh_setup.LifecycleBusyError, match="sample result lease"):
        handle.apply_worker_args({"slab_weights": False})

    assert workers.apply_worker_args.calls == []
    mesh_setup.release_sample(submission)


def test_abandoned_pending_result_is_terminal_and_requires_recycle(monkeypatch):
    from dgx_monarch import telemetry
    from dgx_monarch.nodes.pending import PendingRender

    handle, _old, new = _handle(_Workers([_Future(_acks())]))
    token = mesh_setup.current_setup_token(handle)
    underlying = _Future(_ValueMesh([{"late": True}]))
    submission = _dispatch_sample(
        handle, lambda: underlying, setup_token=token)
    closed = []
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
    pending = PendingRender(
        handle, submission, SimpleNamespace(__exit__=lambda *_args: closed.append(1)),
        None, {}, 10.0, "render", lambda *_args: {})

    pending.abandon()
    pending.abandon()
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {1: 1}
    assert closed == [1] and underlying.timeouts == []
    with pytest.raises(RuntimeError, match="already closed"):
        pending.result()
    with pytest.raises(RuntimeError, match="already abandoned"):
        submission.get()
    with pytest.raises(mesh_setup.LifecycleBusyError, match="abandoned sample"):
        handle.ensure_setup(new, "TORCH_FLASH")
    with pytest.raises(mesh_setup.LifecycleBusyError, match="dispatch another sample"):
        _dispatch_sample(
            handle, lambda: (_ for _ in ()).throw(
                AssertionError("abandoned handle must not send")),
            setup_token=token,
        )


def test_sample_lease_blocks_recycle_and_shutdown_before_destructive_work():
    handle, _old, _new = _handle(_Workers([]))
    handle.sample_leases = {handle.setup_generation: 1}

    assert handle._recycle_impl() is False
    with pytest.raises(mesh_setup.LifecycleBusyError, match="sample result lease"):
        handle._shutdown_impl()

    assert handle.procs.reasons == []
    assert handle.setup_key is not None


def test_lifecycle_lock_timeout_leaves_no_late_stop_waiter():
    handle, _old, _new = _handle(_Workers([]))
    entered, release = threading.Event(), threading.Event()

    def hold_transition_lock():
        with handle.lock:
            entered.set()
            assert release.wait(timeout=2.0)

    holder = threading.Thread(target=hold_transition_lock)
    holder.start()
    assert entered.wait(timeout=1.0)

    assert handle._recycle_impl(lock_timeout_s=0.01) is False
    with pytest.raises(mesh_setup.LifecycleBusyError, match="transition still owns"):
        handle._shutdown_impl(lock_timeout_s=0.01)
    release.set()
    holder.join(timeout=2.0)

    assert not holder.is_alive()
    assert handle.procs.reasons == []
    assert handle.setup_key is not None


def test_synchronous_setup_dispatch_failure_rolls_back_partial_calls():
    workers = _Workers(
        [_Future(_acks()), _Future(_acks())],
        setup_futures=[_Future({"rank": 0}), RuntimeError("dispatch failed")],
    )
    handle, _old, new = _handle(workers)

    with pytest.raises(RuntimeError, match="dispatch failed"):
        handle.ensure_setup(new, "TORCH_FLASH")

    assert len(workers.teardown_group.calls) == 2
    assert all(len(endpoint.envs) == 1 for endpoint in workers.setup_endpoints)
    assert handle.setup_key is None and handle.topology is None
    assert handle.setup_generation == 2
    assert handle.setup_cleanup_state is None
    assert handle.procs.reasons == []


def test_setup_failure_with_unknown_rollback_retires_owned_proc_mesh():
    workers = _Workers(
        [_Future(_acks()), _Future(error=TimeoutError("rollback pending"))],
        setup_futures=[_Future(error=RuntimeError("setup failed")), _Future({"rank": 1})],
    )
    handle, _old, new = _handle(workers)

    with pytest.raises(RuntimeError, match="setup failed"):
        handle.ensure_setup(new, "TORCH_FLASH")

    assert handle.setup_cleanup_state is not None
    assert handle.setup_cleanup_state.phase == "setup rollback"
    assert handle.setup_cleanup_state.outcome is mesh_setup.SetupCleanupOutcome.TIMEOUT_UNKNOWN
    assert handle.defunct is True and handle.teardown_complete is True
    assert handle.replacement_blocked is None
    assert handle.procs.reasons == ["dgx-monarch failed setup rollback"]


def test_setup_rollback_proc_stop_baseexception_preserves_root_and_blocks_restop():
    class StopNow(BaseException):
        pass

    class FailingProcs(_Procs):
        def stop(self, reason):
            self.reasons.append(reason)
            return _Future(error=StopNow("interrupt during proc stop"))

    workers = _Workers(
        [_Future(_acks()), _Future(error=TimeoutError("rollback pending"))],
        setup_futures=[_Future(error=RuntimeError("primary setup failure")),
                       _Future({"rank": 1})],
    )
    handle, _old, new = _handle(workers)
    handle.procs = FailingProcs()

    with pytest.raises(RuntimeError, match="primary setup failure"):
        handle.ensure_setup(new, "TORCH_FLASH")

    assert handle.defunct is True and handle.teardown_complete is False
    assert "process state unknown" in (handle.replacement_blocked or "")
    assert not mesh_safety.token_present(id(handle))
    with pytest.raises(RuntimeError, match="second stop"):
        handle._shutdown_impl()
    assert handle.procs.reasons == ["dgx-monarch failed setup rollback"]


def test_setup_rollback_authority_end_interruption_preserves_setup_root(monkeypatch):
    class StopNow(BaseException):
        pass

    workers = _Workers(
        [_Future(_acks()), _Future(error=TimeoutError("rollback pending"))],
        setup_futures=[_Future(error=RuntimeError("primary setup failure")),
                       _Future({"rank": 1})],
    )
    handle, _old, new = _handle(workers)
    real_end = mesh_safety.end_deliberate_teardown

    def interrupted_end(*_args, **_kwargs):
        raise StopNow("handoff interrupted")

    monkeypatch.setattr(mesh_safety, "end_deliberate_teardown", interrupted_end)
    try:
        with pytest.raises(RuntimeError, match="primary setup failure"):
            handle.ensure_setup(new, "TORCH_FLASH")
    finally:
        real_end(id(handle), stop_confirmed=True)

    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None
    assert handle.procs.reasons == ["dgx-monarch failed setup rollback"]


def test_setup_rollback_group_teardown_runs_off_the_loop(monkeypatch):
    """On comfy's event loop, monarch logs a warning for every Future.get, so a
    rollback ack driven there prints it over the failure the operator is reading
    (docs/TROUBLESHOOTING.md #57).

    Ensure_setup calls teardown_group twice: slot 0 is the pre-dispatch teardown
    of the old groups, slot 1 the rollback this test pins. The three phase caps
    all default to 600 s, so give the rollback its own value; otherwise the
    budget assertion below would hold just as well for the wrong slot.
    """
    budget = mesh_setup.SetupTransitionBudget(rollback_cleanup_s=601.0)
    monkeypatch.setattr(mesh_setup.SetupTransitionBudget, "for_setup_timeout",
                        classmethod(lambda _cls, _timeout_s: budget))
    pre_dispatch = _Future(_acks())
    teardown = _LoopRefusingFuture(_acks())
    workers = _Workers(
        [pre_dispatch, teardown],
        setup_futures=[_Future(error=RuntimeError("setup failed")),
                       _Future({"rank": 1})],
    )
    handle, _old, new = _handle(workers)

    async def transition():
        with pytest.raises(RuntimeError, match="setup failed"):
            handle.ensure_setup(new, "TORCH_FLASH")

    asyncio.run(transition())

    assert handle.setup_cleanup_state is None
    assert handle.setup_key is None and handle.topology is None
    assert workers.teardown_group.futures == []      # both slots consumed, in order
    assert pre_dispatch.timeouts == [budget.group_cleanup_s]
    assert teardown.timeouts == [601.0]              # the rollback slot, not slot 0
    assert handle.procs.reasons == []


def test_get_mesh_refuses_cached_dirty_group_state(monkeypatch):
    workers = _Workers([_Future(error=TimeoutError("still unloading"))])
    handle, _old, new = _handle(workers)
    with pytest.raises(mesh_setup.TopologyTransitionError):
        handle.ensure_setup(new, "TORCH_FLASH")

    monkeypatch.setattr(mesh_mod, "find_config_path", lambda _path: None)
    monkeypatch.setattr(mesh_mod, "local_config", lambda _path=None: handle.config)
    monkeypatch.setattr(mesh_mod, "_detect_comfy_dir", lambda _path: "/comfy")
    monkeypatch.setattr(mesh_mod, "_visible_gpu_count", lambda: 2)
    monkeypatch.setattr(mesh_mod, "_MESHES", {("local", "local"): handle})
    monkeypatch.setattr(mesh_mod, "_MESH_CREATING", {})
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", "local")
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)

    with pytest.raises(mesh_mod.MeshAttachError, match="group or process teardown"):
        mesh_mod.get_mesh(mode="local")


@pytest.mark.parametrize("bad", [True, 0, -1, float("nan"), float("inf"), "600", None])
def test_setup_timeout_budget_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="finite positive"):
        mesh_setup.SetupTransitionBudget.for_setup_timeout(bad)


@pytest.mark.parametrize(
    "field",
    ["group_cleanup_s", "rank_setup_s", "rollback_cleanup_s"],
)
def test_direct_setup_budget_construction_validates_every_phase(field):
    with pytest.raises(ValueError, match="finite positive"):
        mesh_setup.SetupTransitionBudget(**{field: float("nan")})


def test_setup_phase_uses_one_shared_deadline(monkeypatch):
    ticks = iter((100.0, 101.0, 105.5))
    monkeypatch.setattr(mesh_setup.time, "monotonic", lambda: next(ticks))

    deadline = mesh_setup.deadline_after(10.0)

    assert mesh_setup.remaining(deadline, "setup") == pytest.approx(9.0)
    assert mesh_setup.remaining(deadline, "setup") == pytest.approx(4.5)


def test_worker_teardown_retires_setup_before_slow_cleanup():
    from dgx_monarch.actor.worker import GPUWorker

    entered, release = threading.Event(), threading.Event()
    worker = GPUWorker.__new__(GPUWorker)
    worker._setup_key = ("ready",)
    worker._setup_cleanup_failed = False
    worker.rank, worker.world = 0, 2
    worker.topology = {"ulysses": 2}
    old_latent_return = object()
    worker._latent_return = old_latent_return
    result = []

    def slow_cleanup():
        assert worker._setup_key is None
        assert worker.rank is None and worker.world is None and worker.topology == {}
        assert worker._setup_cleanup_failed is True
        assert worker._latent_return is old_latent_return
        entered.set()
        assert release.wait(timeout=2.0)

    worker._teardown_parallel_state = slow_cleanup
    thread = threading.Thread(
        target=lambda: result.append(GPUWorker._teardown_group_impl(worker)))
    thread.start()
    assert entered.wait(timeout=1.0)
    assert worker._setup_key is None
    assert worker._setup_cleanup_failed is True
    with pytest.raises(RuntimeError, match="load_model before setup"):
        GPUWorker._load_model_impl.__wrapped__(worker, "model.sft")
    release.set()
    thread.join(timeout=2.0)

    assert result == [{
        "host": result[0]["host"],
        "torn_down": True,
        "cleanup_state": "UNSETUP",
    }]
    assert worker._setup_cleanup_failed is False
    assert worker._latent_return is old_latent_return


def test_worker_teardown_that_keeps_its_models_stays_dirty_until_a_full_teardown(monkeypatch):
    """Recycle and shutdown skip the model unload because the process stop
    frees the weights. Until that stop the actor still holds models built for
    the retired setup, so it must refuse a new setup and say so in its reply."""
    from dgx_monarch.actor import worker_env
    from dgx_monarch.actor.worker import GPUWorker

    worker = GPUWorker.__new__(GPUWorker)
    worker._setup_key = ("ready",)
    worker._setup_cleanup_failed = False
    worker.rank, worker.world, worker.topology = 0, 2, {"ulysses": 2}
    stages = []
    monkeypatch.setattr(
        worker_env, "teardown_parallel_state",
        lambda _worker, release_models=True: stages.append(release_models))
    worker._teardown_parallel_state = lambda: stages.append("full")

    held = GPUWorker._teardown_group_impl(worker, False)

    assert (held["torn_down"], held["cleanup_state"]) == (True, "DIRTY")
    assert stages == [False] and worker._setup_key is None
    with pytest.raises(RuntimeError, match="tear down or recycle"):
        worker_env.setup_impl(worker, {})

    released = GPUWorker._teardown_group_impl(worker)

    assert (released["torn_down"], released["cleanup_state"]) == (True, "UNSETUP")
    assert stages == [False, "full"] and worker._setup_cleanup_failed is False


def test_worker_key_retirement_interruption_stays_dirty_and_retries(monkeypatch):
    from dgx_monarch.actor.worker import GPUWorker

    class StopNow(BaseException):
        pass

    worker = GPUWorker.__new__(GPUWorker)
    worker._setup_key = (7, "ready")
    worker._setup_generation = 7
    worker._setup_cleanup_failed = False
    worker.rank, worker.world = 0, 2
    worker.topology = {"ulysses": 2}
    worker._latent_return = object()
    cleanup_calls = []
    worker._teardown_parallel_state = lambda: cleanup_calls.append("cleanup")
    original_setattr = GPUWorker.__setattr__
    armed = True

    def interrupt_after_key_retirement(self, name, value):
        nonlocal armed
        original_setattr(self, name, value)
        if self is worker and name == "_setup_key" and value is None and armed:
            armed = False
            raise StopNow("interrupted after worker setup key retirement")

    monkeypatch.setattr(GPUWorker, "__setattr__", interrupt_after_key_retirement)
    with pytest.raises(StopNow):
        GPUWorker._teardown_group_impl(worker)

    assert worker._setup_key is None
    assert worker._setup_cleanup_failed is True
    assert worker._setup_generation == 7
    assert cleanup_calls == []

    result = GPUWorker._teardown_group_impl(worker)

    assert result["cleanup_state"] == "UNSETUP"
    assert result["torn_down"] is True
    assert cleanup_calls == ["cleanup"]
    assert worker._setup_key is None
    assert worker._setup_generation is None
    assert worker.rank is None and worker.world is None and worker.topology == {}
    assert worker._setup_cleanup_failed is False


def test_worker_dirty_teardown_can_be_retried_but_setup_identity_stays_retired():
    from dgx_monarch.actor.worker import GPUWorker

    worker = GPUWorker.__new__(GPUWorker)
    worker._setup_key = ("ready",)
    worker._setup_cleanup_failed = False
    worker.rank, worker.world = 0, 2
    worker.topology = {"ring": 2}
    worker._latent_return = object()
    calls = []

    def cleanup():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("cleanup residue")

    worker._teardown_parallel_state = cleanup
    with pytest.raises(RuntimeError, match="cleanup residue"):
        GPUWorker._teardown_group_impl(worker)
    assert worker._setup_key is None and worker._setup_cleanup_failed is True

    result = GPUWorker._teardown_group_impl(worker)
    assert result["cleanup_state"] == "UNSETUP" and result["torn_down"] is True
    assert calls == [1, 1]
    assert worker._setup_key is None and worker._setup_cleanup_failed is False


def test_worker_unsetup_teardown_is_typed_idempotent_noop():
    from dgx_monarch.actor.worker import GPUWorker

    worker = GPUWorker.__new__(GPUWorker)
    worker._setup_key = None
    worker._setup_cleanup_failed = False

    result = GPUWorker._teardown_group_impl(worker)

    assert result["cleanup_state"] == "UNSETUP"
    assert result["torn_down"] is False


def test_policy_retirement_uses_non_shortenable_large_resident_budget():
    applied = _Future(_ValueMesh([{}, {}]))
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(applied)
    handle, old, _new = _handle(workers)
    target = {"slab_weights": False, "lora_low_rss": False}

    handle.ensure_setup(
        old,
        "TORCH_FLASH",
        worker_args=target,
        timeout_s=7.0,
    )

    assert applied.timeouts == [600.0]
    assert applied.timeouts[0] > 136.39
    assert handle.worker_args_key == MeshHandle._worker_args_key(target)
    assert handle.active_worker_args == target
    assert handle.setup_cleanup_state is None
