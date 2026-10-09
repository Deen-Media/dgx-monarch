"""Lock and event-loop safety regressions for the mesh lifecycle.

1. A supervision failure inside ensure_setup/apply_worker_args while the caller
   holds handle.lock on comfy's event loop must not deadlock against shutdown()'s
   scratch thread: the reconcile stop defers to a detached thread and the
   caller's original exception surfaces immediately.
2. get_mesh keeps generation-scoped ownership of its process-transport claim,
   so one same-mode creator cannot release another creator's live claim and
   admit an incompatible transport.
3. Every driver-side monarch future is driven off comfy's event loop, so monarch
   emits no "Future.get() called from within an active event loop" line into the
   operator's console (docs/TROUBLESHOOTING.md #57).
"""
import ast
import asyncio
import dis
import pathlib
import sys
import threading
import time
import types

import pytest

import dgx_monarch.mesh as mesh_mod
from dgx_monarch import (
    mesh_factory,
    mesh_helpers,
    mesh_lease,
    mesh_runtime,
    mesh_safety,
    mesh_teardown,
)
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.mesh import (
    MeshAttachError,
    MeshHandle,
    mark_defunct_on_supervision_failure,
)
from dgx_monarch.mesh_factory import WorkerSpawnError, spawn_worker_fleet


class _Done:
    def get(self, timeout=None):
        return None


class SupervisionError(Exception):
    """Stand-in whose type name carries the classifier marker."""


def _fake_handle(calls):
    handle = MeshHandle.__new__(MeshHandle)
    handle.lock = threading.RLock()
    handle.owns_hosts = False
    handle.defunct = False
    handle.replacement_blocked = None
    handle.teardown_complete = False
    handle.setup_key = ("topology",)
    handle.workers = types.SimpleNamespace(
        teardown_group=types.SimpleNamespace(call=lambda: _Done()))
    handle.procs = types.SimpleNamespace(
        stop=lambda reason: calls.append(("stop", reason)) or _Done())
    mesh_safety.clear_stale_token(id(handle))  # id() values get reused
    return handle


def test_defunct_under_held_lock_returns_fast_and_reconciles_off_thread():
    """A synchronous reconcile stop here would stall for the 90 s scratch-thread
    join: the on-loop caller holds handle.lock while shutdown()'s scratch
    thread waits on it, and an RLock reenters on its own thread only."""
    calls = []
    handle = _fake_handle(calls)

    async def failing_dispatch():
        t0 = time.monotonic()
        with handle.lock:  # ensure_setup's exact posture at the failure point
            mark_defunct_on_supervision_failure(
                handle, SupervisionError("worker proc exited"), holding_lock=True)
            return time.monotonic() - t0

    elapsed = asyncio.run(failing_dispatch())
    assert elapsed < 5.0                       # no join under the lock
    assert handle.defunct
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not handle.teardown_complete:
        time.sleep(0.05)                       # detached reconcile stop lands
    assert handle.teardown_complete
    assert calls == [("stop", "dgx-monarch client detach")]
    assert handle.replacement_blocked is None


def test_reconcile_thread_launch_retries_one_shot_start_failure(monkeypatch):
    class StartStop(BaseException):
        pass

    calls = []
    handle = _fake_handle(calls)
    real_thread = threading.Thread
    starts = []

    def flaky_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        actual_start = thread.start

        def start():
            starts.append(True)
            if len(starts) == 1:
                raise StartStop("thread start interrupted")
            actual_start()

        thread.start = start
        return thread

    monkeypatch.setattr(threading, "Thread", flaky_thread)
    failure = SupervisionError("worker proc exited")
    with handle.lock:
        mark_defunct_on_supervision_failure(
            handle, failure, holding_lock=True)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not handle.teardown_complete:
        time.sleep(0.01)

    assert starts == [True, True]
    assert handle.teardown_complete is True
    assert calls == [("stop", "dgx-monarch client detach")]
    assert any("thread launch failed" in note for note in failure.__notes__)


def test_supervision_publication_repairs_token_before_defunct_interruption():
    class StopNow(BaseException):
        pass

    class InterruptedHandle:
        def __setattr__(self, name, value):
            if name == "defunct" and getattr(self, "interrupt_defunct", False):
                object.__setattr__(self, "interrupt_defunct", False)
                raise StopNow("defunct publication interrupted")
            object.__setattr__(self, name, value)

    handle = InterruptedHandle()
    handle.lock = threading.RLock()
    handle.defunct = False
    handle.replacement_blocked = None
    handle.teardown_complete = False
    handle.sample_leases = {}
    handle.deferred_supervision_error = None
    handle._supervision_reconcile_token = None
    handle.interrupt_defunct = True

    def shutdown(*, timeout_s, _reconcile_token):
        assert timeout_s == 30
        assert _reconcile_token is handle._supervision_reconcile_token
        handle.teardown_complete = True

    handle.shutdown = shutdown
    with pytest.raises(StopNow, match="defunct publication"):
        mark_defunct_on_supervision_failure(
            handle, SupervisionError("peer closed"), holding_lock=True)

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not handle.teardown_complete:
        time.sleep(0.01)
    assert handle.defunct is True
    assert handle._supervision_reconcile_token is not None
    assert handle.teardown_complete is True


def test_supervision_publication_repairs_deferred_resume_interruption():
    class StopNow(BaseException):
        pass

    class InterruptedHandle:
        def __setattr__(self, name, value):
            if (name == "deferred_supervision_error"
                    and getattr(self, "interrupt_deferred", False)):
                object.__setattr__(self, "interrupt_deferred", False)
                raise StopNow("deferred publication interrupted")
            object.__setattr__(self, name, value)

    handle = InterruptedHandle()
    handle.lock = threading.RLock()
    handle.defunct = False
    handle.replacement_blocked = None
    handle.teardown_complete = False
    handle.sample_leases = {1: 1}
    handle.deferred_supervision_error = None
    handle._supervision_reconcile_token = None
    handle.interrupt_deferred = True
    failure = SupervisionError("peer closed")

    with pytest.raises(StopNow, match="deferred publication"):
        mark_defunct_on_supervision_failure(handle, failure, holding_lock=True)

    assert handle.defunct is True
    assert handle.deferred_supervision_error is failure
    assert getattr(handle, "_supervision_reconcile_thread", None) is None


def test_defunct_without_lock_stays_synchronous():
    """With no lock held (the call_all path), the reconcile stop resolves
    before mark_defunct returns."""
    calls = []
    handle = _fake_handle(calls)
    mark_defunct_on_supervision_failure(handle, SupervisionError("proc exited"))
    assert handle.defunct
    assert handle.teardown_complete            # resolved synchronously
    assert calls == [("stop", "dgx-monarch client detach")]


def test_await_chokepoint_preserves_worker_failure_if_publication_raises(
    monkeypatch,
):
    primary = RuntimeError("worker future failed")
    handle = _fake_handle([])

    class Future:
        @staticmethod
        def get(timeout=None):
            del timeout
            raise primary

    monkeypatch.setattr(
        mesh_helpers,
        "mark_defunct_on_supervision_failure",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("publication interrupted")),
    )

    with pytest.raises(RuntimeError) as caught:
        handle._await_or_evict(Future(), 1.0)

    assert caught.value is primary
    assert any("publication also failed" in note for note in primary.__notes__)


@pytest.mark.parametrize(
    "inner",
    [
        UnsupportedModelError("reference_latent is unsupported"),
        RuntimeError("connection lost to worker"),
    ],
)
def test_await_chokepoint_never_formats_actor_error(monkeypatch, inner):
    from monarch.actor import ActorError

    wrapper = ActorError(inner)
    calls = []
    handle = _fake_handle(calls)
    formatted = []

    class Future:
        @staticmethod
        def get(timeout=None):
            assert timeout == 1.0
            raise wrapper

    def reject_formatting(_self):
        formatted.append(True)
        raise AssertionError("ActorError wrapper was formatted")

    monkeypatch.setattr(ActorError, "__str__", reject_formatting)

    with pytest.raises(ActorError) as caught:
        handle._await_or_evict(Future(), 1.0)

    assert caught.value is wrapper
    assert caught.value.exception is inner
    assert formatted == []
    assert handle.defunct is False
    assert handle.teardown_complete is False
    assert calls == []


def test_await_chokepoint_never_formats_actor_error_subclass(monkeypatch):
    from monarch.actor import ActorError

    class DerivedActorError(ActorError):
        pass

    wrapper = DerivedActorError(RuntimeError("peer closed"))
    calls = []
    handle = _fake_handle(calls)
    formatted = []

    class Future:
        @staticmethod
        def get(timeout=None):
            assert timeout == 1.0
            raise wrapper

    def reject_formatting(_self):
        formatted.append(True)
        raise AssertionError("ActorError subclass was formatted")

    monkeypatch.setattr(ActorError, "__str__", reject_formatting)

    with pytest.raises(ActorError) as caught:
        handle._await_or_evict(Future(), 1.0)

    assert caught.value is wrapper
    assert caught.value.exception.__class__ is RuntimeError
    assert formatted == []
    assert handle.defunct is False
    assert handle.teardown_complete is False
    assert calls == []


@pytest.fixture
def transport_state(isolated_environ):
    # A bring-up plants the driver marker and DGXM_PYTHONPATH in os.environ for
    # the workers it spawns. isolated_environ takes them away again, as the
    # rest of this fixture does for the transport state.
    saved = (mesh_mod._TRANSPORT_MODE, mesh_mod._TRANSPORT_POISON,
             mesh_mod._TRANSPORT_BIND, mesh_mod._TRANSPORT_CLAIM_GENERATION,
             set(mesh_mod._TRANSPORT_CLAIMS),
             mesh_mod._TRANSPORT_COMMITTED_GENERATION,
             dict(mesh_mod._MESHES), dict(mesh_mod._MESH_CREATING),
             dict(mesh_factory._CREATION_CLEANUP_CLAIMS))
    mesh_mod._TRANSPORT_MODE = None
    mesh_mod._TRANSPORT_POISON = None
    mesh_mod._TRANSPORT_BIND = None
    mesh_mod._TRANSPORT_CLAIM_GENERATION = 0
    mesh_mod._TRANSPORT_CLAIMS.clear()
    mesh_mod._TRANSPORT_COMMITTED_GENERATION = None
    mesh_mod._MESHES.clear()
    mesh_mod._MESH_CREATING.clear()
    mesh_factory._CREATION_CLEANUP_CLAIMS.clear()
    yield
    (mesh_mod._TRANSPORT_MODE, mesh_mod._TRANSPORT_POISON,
     mesh_mod._TRANSPORT_BIND, mesh_mod._TRANSPORT_CLAIM_GENERATION) = saved[:4]
    mesh_mod._TRANSPORT_CLAIMS.clear()
    mesh_mod._TRANSPORT_CLAIMS.update(saved[4])
    mesh_mod._TRANSPORT_COMMITTED_GENERATION = saved[5]
    mesh_mod._MESHES.clear()
    mesh_mod._MESHES.update(saved[6])
    mesh_mod._MESH_CREATING.clear()
    mesh_mod._MESH_CREATING.update(saved[7])
    mesh_factory._CREATION_CLEANUP_CLAIMS.clear()
    mesh_factory._CREATION_CLEANUP_CLAIMS.update(saved[8])


def _attach_fakes(monkeypatch):
    monkeypatch.setattr(mesh_mod, "_detect_comfy_dir", lambda _v: "/fake/comfy")
    monkeypatch.setattr(mesh_mod, "_visible_gpu_count", lambda: 1)
    monkeypatch.setattr(mesh_mod, "find_config_path", lambda p: p or None)
    monkeypatch.setattr(
        mesh_mod, "local_config",
        lambda _path=None: types.SimpleNamespace(source="local", comfy_dir=None))
    fake_cluster = types.SimpleNamespace(
        source="/fake/cluster.toml", comfy_dir=None,
        hosts=[types.SimpleNamespace(gpus=1)])
    monkeypatch.setattr(mesh_mod, "load_cluster_config", lambda _p: fake_cluster)


def test_concurrent_local_and_cluster_attach_refuse_typed(transport_state, monkeypatch):
    """While a local bring-up is mid-spawn, a cluster attach must refuse with
    the typed one-transport error instead of proceeding into monarch."""
    _attach_fakes(monkeypatch)
    spawn_entered = threading.Event()
    release_spawn = threading.Event()

    def blocking_spawn(config, use_cluster, gph, attach, bootstrap, handoff):
        spawn_entered.set()
        release_spawn.wait(timeout=30)
        raise WorkerSpawnError(RuntimeError("spawn aborted by test"), None,
                               local_transport_initialized=False)

    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", blocking_spawn)

    local_err: list[str] = []

    def local_attach():
        try:
            mesh_mod.get_mesh(mode="local", gpus_per_host=1)
        except MeshAttachError as exc:
            local_err.append(str(exc))

    worker = threading.Thread(target=local_attach)
    worker.start()
    try:
        assert spawn_entered.wait(timeout=10)
        with pytest.raises(MeshAttachError, match="already brought up a local mesh"):
            mesh_mod.get_mesh(mode="cluster", config_path="/fake/cluster.toml")
    finally:
        release_spawn.set()
        worker.join(timeout=30)
    assert local_err and "bring-up failed" in local_err[0]
    # A local failure before the import touches no native state, so the claim
    # is released and the session is not poisoned: a retry in place can work.
    assert mesh_mod._TRANSPORT_POISON is None
    assert mesh_mod._TRANSPORT_MODE is None
    assert mesh_mod._TRANSPORT_CLAIMS == set()


def test_local_pre_import_failure_is_retryable_without_restart(
        transport_state, monkeypatch):
    """A local WorkerSpawnError that never initialized monarch
    (local_transport_initialized=False, the pre-import path) must not poison
    the session: after the install is fixed, the same session's next get_mesh
    succeeds. A local failure after transport initialization still poisons."""
    _attach_fakes(monkeypatch)
    attempts = {"n": 0}

    def spawn(config, use_cluster, gph, attach, bootstrap, handoff):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise WorkerSpawnError(
                RuntimeError("torchmonarch is not importable"), None,
                local_transport_initialized=False)
        return (types.SimpleNamespace(), types.SimpleNamespace(),
                types.SimpleNamespace(extent=types.SimpleNamespace(labels=[])),
                False)

    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", spawn)
    with pytest.raises(MeshAttachError, match="bring-up failed"):
        mesh_mod.get_mesh(mode="local", gpus_per_host=1)
    assert mesh_mod._TRANSPORT_POISON is None
    assert mesh_mod._TRANSPORT_MODE is None

    handle = mesh_mod.get_mesh(mode="local", gpus_per_host=1)  # retry works
    assert handle is not None
    assert mesh_mod._TRANSPORT_MODE == "local"


def test_local_post_transport_failure_still_poisons(transport_state, monkeypatch):
    _attach_fakes(monkeypatch)

    def spawn(config, use_cluster, gph, attach, bootstrap, handoff):
        raise WorkerSpawnError(RuntimeError("actor spawn exploded"), None,
                               local_transport_initialized=True)

    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", spawn)
    with pytest.raises(MeshAttachError, match="bring-up failed"):
        mesh_mod.get_mesh(mode="local", gpus_per_host=1)
    assert mesh_mod._TRANSPORT_POISON is not None       # native state touched
    assert mesh_mod._TRANSPORT_MODE == "local"          # claim stays committed
    with pytest.raises(MeshAttachError, match="not reusable"):
        mesh_mod.get_mesh(mode="local", gpus_per_host=1)


def test_active_creator_blocks_other_claims_without_releasing_mode_early(
        transport_state, monkeypatch):
    _attach_fakes(monkeypatch)
    monkeypatch.setattr(
        mesh_mod, "load_cluster_config",
        lambda path: types.SimpleNamespace(
            source=path, comfy_dir=None, hosts=[types.SimpleNamespace(gpus=1)]))
    a_entered, b_entered = threading.Event(), threading.Event()
    release_a, release_b = threading.Event(), threading.Event()
    local_spawned = threading.Event()

    def controlled_spawn(config, use_cluster, gph, attach, bootstrap, handoff):
        if config.source == "/fake/a.toml":
            a_entered.set()
            assert release_a.wait(timeout=30)
            raise MeshAttachError("cluster A refused in validation")
        if config.source == "/fake/b.toml":
            b_entered.set()
            assert release_b.wait(timeout=30)
            return object(), object(), object(), True
        local_spawned.set()
        return object(), object(), object(), False

    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", controlled_spawn)
    outcomes: dict[str, object] = {}

    def create(name, **kwargs):
        try:
            outcomes[name] = mesh_mod.get_mesh(**kwargs)
        except BaseException as exc:
            outcomes[name] = exc

    cluster_a = threading.Thread(
        target=create, args=("a",),
        kwargs={"mode": "cluster", "config_path": "/fake/a.toml"})
    refused_b = threading.Thread(
        target=create, args=("b",),
        kwargs={"mode": "cluster", "config_path": "/fake/b.toml"})
    local = threading.Thread(
        target=create, args=("local",),
        kwargs={"mode": "local", "gpus_per_host": 1})
    cluster_a.start()
    try:
        assert a_entered.wait(timeout=10)
        refused_b.start()
        refused_b.join(timeout=3)
        assert not refused_b.is_alive()
        assert isinstance(outcomes.get("b"), MeshAttachError)
        assert "not reusable" in str(outcomes["b"])
        assert not b_entered.is_set()
        assert len(mesh_mod._TRANSPORT_CLAIMS) == 1

        local.start()
        local.join(timeout=3)
        assert not local.is_alive()
        assert isinstance(outcomes.get("local"), MeshAttachError)
        assert "already brought up a cluster mesh" in str(outcomes["local"])
        assert not local_spawned.is_set()

        release_a.set()
        cluster_a.join(timeout=10)
        assert not cluster_a.is_alive()
        assert isinstance(outcomes.get("a"), MeshAttachError)
        assert mesh_mod._TRANSPORT_CLAIMS == set()

        outcomes.pop("b")
        cluster_b = threading.Thread(
            target=create, args=("b",),
            kwargs={"mode": "cluster", "config_path": "/fake/b.toml"})
        cluster_b.start()
        assert b_entered.wait(timeout=10)
        assert len(mesh_mod._TRANSPORT_CLAIMS) == 1
        release_b.set()
        cluster_b.join(timeout=10)
        assert not cluster_b.is_alive()
    finally:
        release_a.set()
        release_b.set()
        cluster_a.join(timeout=30)
        refused_b.join(timeout=30)
        if local.ident is not None:
            local.join(timeout=30)

    assert isinstance(outcomes.get("b"), MeshHandle)
    assert mesh_mod._TRANSPORT_MODE == "cluster"
    assert mesh_mod._TRANSPORT_CLAIMS == set()
    assert (mesh_mod._TRANSPORT_COMMITTED_GENERATION
            == mesh_mod._TRANSPORT_CLAIM_GENERATION)


def test_cluster_attach_is_single_owner_until_its_transport_verdict(
        transport_state, monkeypatch):
    """Different keys cannot race process-global enable/attach initialization."""
    _attach_fakes(monkeypatch)
    monkeypatch.setattr(
        mesh_mod, "load_cluster_config",
        lambda path: types.SimpleNamespace(
            source=path, comfy_dir=None, hosts=[types.SimpleNamespace(gpus=1)]))
    attach_entered = threading.Event()
    release_attach = threading.Event()
    attach_calls = 0

    def blocked_attach(_config, handoff):
        nonlocal attach_calls
        attach_calls += 1
        attach_entered.set()
        assert release_attach.wait(timeout=3)
        mesh_factory.safe_transport_failure(handoff)
        raise MeshAttachError("cluster validation refused before transport enable")

    monkeypatch.setattr(mesh_mod, "_attach_cluster", blocked_attach)
    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", spawn_worker_fleet)
    errors: list[BaseException] = []

    def create_first():
        try:
            mesh_mod.get_mesh(
                mode="cluster", config_path="/fake/a.toml")
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=create_first, name="first-cluster-attach")
    first.start()
    try:
        assert attach_entered.wait(timeout=3)
        assert mesh_factory.creation_cleanup_pending() is True
        with pytest.raises(MeshAttachError, match="not reusable"):
            mesh_mod.get_mesh(
                mode="cluster", config_path="/fake/b.toml")
        assert attach_calls == 1
    finally:
        release_attach.set()
        first.join(timeout=3)

    assert not first.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], MeshAttachError)
    assert "validation refused" in str(errors[0])
    assert mesh_factory.creation_cleanup_pending() is False
    assert mesh_mod._TRANSPORT_POISON is None
    assert mesh_mod._TRANSPORT_MODE is None


def test_simultaneous_different_keys_admit_only_one_transport_acquisition(
        transport_state, monkeypatch):
    _attach_fakes(monkeypatch)
    monkeypatch.setattr(
        mesh_mod, "load_cluster_config",
        lambda path: types.SimpleNamespace(
            source=path, comfy_dir=None, hosts=[types.SimpleNamespace(gpus=1)]))
    ready = threading.Barrier(2)
    attach_entered = threading.Event()
    release_attach = threading.Event()
    one_done = threading.Event()
    attach_calls = 0

    def preflight(_config, _world):
        ready.wait(timeout=3)

    def blocked_attach(_config, handoff):
        nonlocal attach_calls
        attach_calls += 1
        attach_entered.set()
        assert release_attach.wait(timeout=3)
        mesh_factory.safe_transport_failure(handoff)
        raise MeshAttachError("winning attach stopped by test")

    monkeypatch.setattr(mesh_mod, "_attach_cluster", blocked_attach)
    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", spawn_worker_fleet)
    outcomes: dict[str, BaseException] = {}

    def create(name):
        try:
            mesh_mod.get_mesh(
                mode="cluster", config_path=f"/fake/{name}.toml",
                mesh_preflight=preflight)
        except BaseException as exc:
            outcomes[name] = exc
        finally:
            one_done.set()

    creators = [threading.Thread(target=create, args=(name,))
                for name in ("a", "b")]
    for creator in creators:
        creator.start()
    try:
        assert attach_entered.wait(timeout=3)
        assert one_done.wait(timeout=3)
        assert attach_calls == 1
        assert sum("not reusable" in str(exc)
                   for exc in outcomes.values()) == 1
    finally:
        release_attach.set()
        for creator in creators:
            creator.join(timeout=3)

    assert all(not creator.is_alive() for creator in creators)
    assert attach_calls == 1 and len(outcomes) == 2
    assert any("winning attach stopped" in str(exc)
               for exc in outcomes.values())
    assert mesh_factory.creation_cleanup_pending() is False
    assert mesh_mod._TRANSPORT_MODE is None


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_lost_cluster_attach_return_poisons_before_any_same_mode_retry(
        transport_state, monkeypatch):
    class LostAttachReturn(BaseException):
        pass

    _attach_fakes(monkeypatch)
    monkeypatch.setattr(
        mesh_mod, "load_cluster_config",
        lambda path: types.SimpleNamespace(
            source=path, comfy_dir=None, hosts=[types.SimpleNamespace(gpus=1)]))
    attach_calls = 0

    def attached(_config, _handoff):
        nonlocal attach_calls
        attach_calls += 1
        mesh_mod._TRANSPORT_BIND = "tcp://10.0.0.1:0"
        return object()

    monkeypatch.setattr(mesh_mod, "_attach_cluster", attached)
    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", spawn_worker_fleet)
    instructions = list(dis.get_instructions(spawn_worker_fleet))
    load_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == "LOAD_FAST"
        and instruction.argval == "attach_cluster"
    )
    call_index = next(
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    )
    target = instructions[call_index + 1]
    assert target.opname == "STORE_FAST" and target.argval == "hosts"
    interrupted = LostAttachReturn("cluster attach return instruction interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target.offset:
            monitoring.set_local_events(
                tool_id, spawn_worker_fleet.__code__, 0)
            raise interrupted

    monitoring.use_tool_id(tool_id, "dgxm-outer-cluster-attach-handoff-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, spawn_worker_fleet.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(LostAttachReturn) as failure:
        try:
            mesh_mod.get_mesh(
                mode="cluster", config_path="/fake/a.toml")
        finally:
            monitoring.set_local_events(
                tool_id, spawn_worker_fleet.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert failure.value is interrupted
    assert "attach result handoff interrupted" in mesh_mod._TRANSPORT_POISON
    assert mesh_factory.creation_cleanup_pending() is False
    with pytest.raises(MeshAttachError, match="not reusable"):
        mesh_mod.get_mesh(mode="cluster", config_path="/fake/b.toml")
    assert attach_calls == 1
    assert mesh_mod._TRANSPORT_MODE == "cluster"


def test_cluster_validation_failure_releases_provisional_claim(transport_state, monkeypatch):
    """A cluster attach refused in validation (before enable_transport) must
    not leave the process claimed as 'cluster'; the operator can fix the
    config or switch to local without a ComfyUI restart."""
    _attach_fakes(monkeypatch)

    def validation_refusal(config, use_cluster, gph, attach, bootstrap, handoff):
        if use_cluster:
            raise MeshAttachError(
                "cluster attach refused: cluster.transport_security must be 'trusted_fabric'")
        return object(), object(), object(), False

    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", validation_refusal)
    with pytest.raises(MeshAttachError, match="transport_security"):
        mesh_mod.get_mesh(mode="cluster", config_path="/fake/cluster.toml")
    assert mesh_mod._TRANSPORT_MODE is None
    assert mesh_mod._TRANSPORT_POISON is None
    assert mesh_mod._TRANSPORT_CLAIMS == set()
    released_generation = mesh_mod._TRANSPORT_CLAIM_GENERATION

    handle = mesh_mod.get_mesh(mode="local", gpus_per_host=1)
    assert isinstance(handle, MeshHandle)
    assert mesh_mod._TRANSPORT_MODE == "local"
    assert mesh_mod._TRANSPORT_CLAIMS == set()
    assert mesh_mod._TRANSPORT_CLAIM_GENERATION == released_generation + 1
    assert (mesh_mod._TRANSPORT_COMMITTED_GENERATION
            == mesh_mod._TRANSPORT_CLAIM_GENERATION)


def test_safe_claim_registration_failure_allows_opposite_transport(
        transport_state, monkeypatch):
    class ClaimReturnAbort(BaseException):
        pass

    _attach_fakes(monkeypatch)
    real_begin = mesh_factory.begin_creation_cleanup
    interrupted = ClaimReturnAbort("cleanup claim return interrupted")

    def begin_then_interrupt(*args, **kwargs):
        real_begin(*args, **kwargs)
        raise interrupted

    monkeypatch.setattr(
        mesh_factory, "begin_creation_cleanup", begin_then_interrupt)
    with pytest.raises(ClaimReturnAbort) as failure:
        mesh_mod.get_mesh(mode="local", gpus_per_host=1)

    assert failure.value is interrupted
    assert mesh_mod._TRANSPORT_POISON is None
    assert mesh_mod._TRANSPORT_MODE is None
    assert mesh_factory.creation_cleanup_pending() is False

    monkeypatch.setattr(mesh_factory, "begin_creation_cleanup", real_begin)
    monkeypatch.setattr(
        mesh_mod, "spawn_worker_fleet",
        lambda *_args: (object(), object(), object(), True))
    monkeypatch.setattr(mesh_helpers, "start_client_lease", lambda: True)
    handle = mesh_mod.get_mesh(
        mode="cluster", config_path="/fake/cluster.toml")
    assert isinstance(handle, MeshHandle)
    assert mesh_mod._TRANSPORT_MODE == "cluster"


@pytest.mark.parametrize(
    ("native_step", "expected_attach_calls"),
    [("enable", 0), ("first_attach", 1), ("retry_attach", 2)],
)
def test_cluster_native_baseexception_preserves_identity_without_extra_work(
        transport_state, monkeypatch, native_step, expected_attach_calls):
    # This is the ordinary, non-released two-attempt ladder; a liveness
    # release noted by an earlier test in this process must not turn the
    # first attempt's ordinary failure into a released fail-fast here.
    monkeypatch.setattr(mesh_teardown, "released_predecessor_recently",
                        lambda **_k: False)

    class NativeAbort(BaseException):
        def __str__(self):
            raise RuntimeError("native abort text unavailable")

        def __repr__(self):
            raise RuntimeError("native abort representation unavailable")

    interrupted = NativeAbort()
    enable_calls: list[str] = []
    attach_calls: list[list[str]] = []
    restart_calls: list[BaseException] = []
    actor = types.ModuleType("monarch.actor")

    def enable_transport(bind):
        enable_calls.append(bind)
        if native_step == "enable":
            raise interrupted

    def attach_once(addresses):
        attach_calls.append(addresses)
        if native_step == "first_attach" or len(attach_calls) == 2:
            raise interrupted
        raise RuntimeError("ordinary first attach failure")

    actor.enable_transport = enable_transport
    monkeypatch.setitem(sys.modules, "monarch.actor", actor)
    monkeypatch.setattr(mesh_mod, "_attach_once", attach_once)
    monkeypatch.setattr(mesh_mod, "_heal_dead_loops", lambda *_args: None)
    monkeypatch.setattr(
        mesh_mod.mesh_attach, "restart_after_failed_attach",
        lambda _config, exc, _log: restart_calls.append(exc))
    config = types.SimpleNamespace(
        transport_security="trusted_fabric",
        client_bind="tcp://10.0.0.1:0",
        hosts=[types.SimpleNamespace(address="tcp://10.0.0.2:26600")],
        auto_heal=native_step == "retry_attach",
    )
    handoff = mesh_factory.FleetHandoff()

    with pytest.raises(NativeAbort) as failure:
        mesh_mod._attach_cluster(config, handoff)

    assert failure.value is interrupted
    assert enable_calls == ["tcp://10.0.0.1:0"]
    assert len(attach_calls) == expected_attach_calls
    assert restart_calls == []
    assert handoff.transport_failure_safe is False
    # Direct attach owns native bind/poison state; get_mesh owns mode claims.
    assert mesh_mod._TRANSPORT_MODE is None
    assert mesh_mod._TRANSPORT_BIND is (
        None if native_step == "enable" else config.client_bind)
    assert mesh_mod._TRANSPORT_POISON.endswith("<NativeAbort>")


class _LoopRefusingFuture:
    """Turns the docs/TROUBLESHOOTING.md #57 warning into a test failure.

    Monarch keys "Future.get() called from within an active event loop" off the
    calling thread, so a fake that refuses a running loop fails where monarch
    would have logged. tests/transfer_helpers.py's RDMA fakes work the same way.
    """

    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.timeouts: list = []
        self.idents: list = []
        self.thread_names: list = []

    def get(self, timeout=None):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "Future.get must be driven off the event loop (issue #161)")
        self.timeouts.append(timeout)
        self.idents.append(threading.get_ident())
        self.thread_names.append(threading.current_thread().name)
        if self.error is not None:
            raise self.error
        return self.value


def test_await_chokepoint_drives_the_future_off_comfys_loop():
    """_await_or_evict carries every call_all dispatch (loads, sigmas, gate
    legs) and every collect_one, so an on-loop get here alone is most of the
    roughly ten warnings per render cycle in docs/TROUBLESHOOTING.md #57."""
    calls = []
    handle = _fake_handle(calls)
    sentinel = object()
    future = _LoopRefusingFuture(sentinel)

    async def dispatch():
        return handle._await_or_evict(future, 1.0)

    assert asyncio.run(dispatch()) is sentinel
    assert future.timeouts == [1.0]        # the future keeps the caller's deadline
    assert future.idents != [threading.get_ident()]
    assert handle.defunct is False
    assert calls == []


def test_await_chokepoint_stays_inline_without_a_running_loop():
    """CLI, worker-side and test callers pay no thread at all."""
    calls = []
    handle = _fake_handle(calls)
    future = _LoopRefusingFuture("ok")

    assert handle._await_or_evict(future, 1.0) == "ok"
    assert future.idents == [threading.get_ident()]
    assert future.timeouts == [1.0]


def test_await_chokepoint_join_budget_outlasts_the_future_deadline(monkeypatch):
    """A legal 900 s dispatch must time out as the future's TimeoutError, which
    mesh_rpc.call_all_bound classifies as an ambiguous mutation and latches
    DIRTY. An equal join budget would race that and abandon a live thread."""
    captured: list = []

    def capture(fn, timeout_s, thread_name):
        captured.append((timeout_s, thread_name))
        return fn()

    monkeypatch.setattr(mesh_runtime, "run_blocking_off_loop", capture)
    future = _LoopRefusingFuture("value")

    assert mesh_runtime.get_off_loop(future, 900) == "value"
    assert future.timeouts == [900]
    assert captured[0][0] > 900
    assert captured[0][1] == "dgxm-await"


def test_await_chokepoint_still_evicts_on_a_supervision_failure_off_loop():
    """The hop moves only the get: the except clause, and therefore eviction,
    still runs on the caller's own thread."""
    calls = []
    handle = _fake_handle(calls)
    failure = SupervisionError("worker proc exited")
    future = _LoopRefusingFuture(error=failure)

    async def dispatch():
        with pytest.raises(SupervisionError) as caught:
            handle._await_or_evict(future, 1.0)
        return caught.value

    assert asyncio.run(dispatch()) is failure
    assert handle.defunct is True
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not handle.teardown_complete:
        time.sleep(0.05)
    assert handle.teardown_complete
    assert calls == [("stop", "dgx-monarch client detach")]


def test_sample_lease_collection_runs_off_loop_and_takes_the_handle_lock_there():
    """SetupBoundFuture.get takes handle.lock twice, on the scratch thread
    while the caller blocks in join. No collection caller holds that lock
    (nodes/fleet_cleanup.py calls collect_one, nodes/pending.py collect_sample),
    so the RLock's same-thread-only reentrance cannot deadlock it."""
    calls = []
    handle = _fake_handle(calls)
    inner = _LoopRefusingFuture({"latent": None})
    lease = mesh_lease.SetupBoundFuture(inner, handle, 1)

    async def collect():
        started = time.monotonic()
        return handle.collect_one(lease, timeout_s=1.0), time.monotonic() - started

    result, elapsed = asyncio.run(collect())
    assert result == {"latent": None}
    assert elapsed < 5.0                   # no cross-thread lock stall
    assert inner.timeouts == [1.0]


def test_attach_initialization_runs_off_the_loop(monkeypatch):
    """An on-loop get here would print the first warning, before any render."""
    import monarch.actor

    initialized = _LoopRefusingFuture(None)
    hosts = types.SimpleNamespace(initialized=initialized)
    monkeypatch.setattr(
        monarch.actor, "attach_to_workers", lambda **_kwargs: hosts)

    async def attach():
        return mesh_runtime.attach_once(["addr"])

    assert asyncio.run(attach()) is hosts
    # Strictly above monarch's config-push budget; ATTACH_INIT_WAIT_S owns it.
    assert initialized.timeouts == [mesh_runtime.ATTACH_INIT_WAIT_S]
    assert initialized.thread_names == ["dgxm-attach-init"]


def test_partial_bring_up_rollback_stop_runs_off_the_loop():
    """The rollback stop runs while an Init node is still on comfy's loop, and
    it keeps a purpose-named thread: a join timeout here reads as a failed
    bring-up rollback rather than as any other dispatch that ran out of time."""
    stop_future = _LoopRefusingFuture(None)
    spawn_failure = RuntimeError("actor spawn failed")

    class _RollbackProcs:
        def __init__(self):
            self.reasons: list = []

        def spawn(self, *_args, **_kwargs):
            raise spawn_failure

        def stop(self, reason):
            self.reasons.append(reason)
            return stop_future

    procs = _RollbackProcs()
    hosts = types.SimpleNamespace(spawn_procs=lambda **_kwargs: procs)

    async def bring_up():
        with pytest.raises(WorkerSpawnError) as caught:
            spawn_worker_fleet(
                None, True, 1, lambda _config, _handoff: hosts, None)
        return caught.value

    error = asyncio.run(bring_up())
    assert error.original is spawn_failure
    assert error.cleanup is None           # the rollback stop completed
    assert procs.reasons == ["dgx-monarch partial bring-up rollback"]
    assert stop_future.timeouts == [60]
    assert stop_future.thread_names == ["dgxm-bringup-rollback"]


# Every direct <x>.get(...) in the driver tree that could be a blocking wait,
# keyed by (path, enclosing scope) with the reason it is already off comfy's
# loop, or the reason it is not a monarch future at all.
# Duplicate pairs collapse: the pair is pinned, never the call count.
_OFF_LOOP_BY_CONSTRUCTION = {
    ("mesh_recycle.py", "recycle_detailed_impl"):
        "inside recycle()'s dgxm-recycle scratch thread",
    ("mesh.py", "MeshHandle._shutdown_impl"):
        "inside shutdown()'s dgxm-shutdown scratch thread",
    ("mesh_lease.py", "SetupBoundFuture.get"):
        "reached only through _await_or_evict, already off-loop",
    ("mesh_runtime.py", "get_off_loop"):
        "the seam itself: this get IS the scratch-thread body",
    ("mesh_teardown.py", "stop_failed_setup_procs.stop_once"):
        "inside the dgxm-setup-rollback scratch thread",
    ("monarch_surface.py", "_probe_future_get_asyncio.probe"):
        "deliberate on-loop pin of monarch's own semantics",
    ("progress.py", "ProgressReceiver.__enter__.pump"):
        "the dedicated dgxm-progress thread",
    ("transfer.py", "_release_rdma_parts"):
        "inside the dgxm-latent-read scratch thread",
    ("transfer_utils.py", "read_parts_concurrent.read_one"):
        "pool thread under dgxm-latent-read",
    ("rdma_receiver.py", "_read_and_release_owned.drive"):
        "the dgxm-latent-read scratch-thread body",
    ("actor/unbake.py", "restore_pristine"):
        "stdlib queue.Queue.get, not a monarch future",
    ("actor/unbake.py", "restore_pristine._take"):
        "stdlib queue.Queue.get, not a monarch future",
    ("nodes/fleet_cleanup.py", "_drain_fleet_collections"):
        "stdlib queue.Queue.get, not a monarch future",
    ("mesh_liveness.py", "_scan_worker"):
        "the shared dgxm-liveness thread's own body, stdlib queue.Queue.get",
    ("adapters/mage_nvfp4_scale.py", "real_input"):
        "ContextVar read, not a monarch future",
    ("adoption_evidence.py", "active_context_wire"):
        "ContextVar read, not a monarch future",
    ("adoption_evidence.py", "consume_active_context_wire"):
        "ContextVar read, not a monarch future",
    ("adoption_evidence.py", "require_inactive_context"):
        "ContextVar read, not a monarch future",
    ("adoption_evidence.py", "resident_adoption_evidence_context"):
        "ContextVar read, not a monarch future",
    ("adoption_evidence.py", "suspend_active_context"):
        "ContextVar read, not a monarch future",
    ("mesh_session.py", "RenderSession.activate"):
        "ContextVar read, not a monarch future",
    ("mesh_session.py", "_reset_active_session"):
        "ContextVar read, not a monarch future",
    ("mesh_session.py", "claim_render_session"):
        "ContextVar read, not a monarch future",
    ("mesh_session.py", "require_mutation_authority"):
        "ContextVar read, not a monarch future",
}


def _is_blocking_get_shape(node: ast.Call) -> bool:
    """Could this ``.get(...)`` be monarch's blocking wait?

    ``Future.get(self, timeout=None)`` takes its bound three ways, so pinning
    only the keyword form would let ``fut.get(60)`` and the unbounded
    ``fut.get()`` reach comfy's loop unlisted. Any other keyword, a second
    argument or a non-numeric one reads as a mapping lookup.
    """
    if any(kw.arg == "timeout" for kw in node.keywords):
        return True
    if node.keywords:
        return False
    if not node.args:
        return True
    return (len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, int | float)
            and not isinstance(node.args[0].value, bool))


def _scoped_blocking_gets(package_root: pathlib.Path) -> set:
    """Every blocking-shaped ``<x>.get(...)`` as (relative path, dotted scope)."""
    found = set()
    for path in sorted(package_root.rglob("*.py")):
        scope: list = []

        def walk(node, scope=scope, path=path):
            named = isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            if named:
                scope.append(node.name)
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and _is_blocking_get_shape(node)):
                found.add((path.relative_to(package_root).as_posix(),
                           ".".join(scope)))
            for child in ast.iter_child_nodes(node):
                walk(child)
            if named:
                scope.pop()

        walk(ast.parse(path.read_text()))
    return found


def test_every_direct_future_get_is_off_the_event_loop_by_construction():
    """Ledger: a new on-loop ``.get()`` fails the build instead of printing
    monarch's warning into the operator's console again.

    The tree walked is this checkout's ``src``, never the installed package: a
    worktree must be audited against its own source, or the guard passes on
    code it never read. Lambdas are not scopes, so the seam's own
    ``lambda: future.get(...)`` reports as mesh_runtime.get_off_loop.
    MeshHandle.ensure_setup, MeshHandle._await_or_evict, spawn_worker_fleet
    and attach_once moved to get_off_loop on 2026-08-05; none may return.
    """
    package_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"
    actual = _scoped_blocking_gets(package_root)
    listed = set(_OFF_LOOP_BY_CONSTRUCTION)
    assert actual == listed, (
        "blocking-get ledger drift. Drive a new monarch future through "
        "mesh_helpers.get_off_loop; give a new non-future .get() (ContextVar, "
        f"Queue, mapping) a ledger line. unlisted={sorted(actual - listed)}, "
        f"stale={sorted(listed - actual)}"
    )
