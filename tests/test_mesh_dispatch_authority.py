"""Setup-token dispatch authority and mutation dirty-latch regressions."""
from __future__ import annotations

import threading

import pytest

from dgx_monarch import mesh as mesh_mod
from dgx_monarch import mesh_setup
from dgx_monarch.mesh import MeshHandle
from dgx_monarch.nodes.render_session import RenderSession, mutation_render_session
from topology_transition_helpers import (
    _acks,
    _Endpoint,
    _Future,
    _handle,
    _ValueMesh,
    _Workers,
)


@pytest.mark.parametrize(
    "endpoint", ["setup", "teardown_group", "apply_worker_args", "sample"])
def test_generic_dispatch_refuses_lifecycle_internal_endpoints(endpoint):
    handle, _old, _new = _handle(_Workers([]))

    with pytest.raises(RuntimeError, match=r"lifecycle-internal|lease-bound"):
        handle.call_all(endpoint)


def test_setup_bound_dispatch_requires_and_accepts_exact_token():
    workers = _Workers([])
    workers.compute_sigmas = _Endpoint(_Future(_ValueMesh(["rank0", "rank1"])))
    handle, _old, _new = _handle(workers)

    with pytest.raises(RuntimeError, match="requires a SetupToken"):
        handle.call_all("compute_sigmas")
    token = mesh_setup.current_setup_token(handle)

    assert handle.call_all("compute_sigmas", setup_token=token) == ["rank0", "rank1"]
    assert len(workers.compute_sigmas.calls) == 1


def test_provenance_baseline_dispatch_requires_exact_setup_token():
    workers = _Workers([])
    workers.provenance_baseline = _Endpoint(
        _Future(_ValueMesh([
            {"rank": 0, "setup_generation": 1},
            {"rank": 1, "setup_generation": 1},
        ]))
    )
    handle, _old, _new = _handle(workers)

    with pytest.raises(RuntimeError, match="requires a SetupToken"):
        handle.call_all("provenance_baseline", 1)
    token = mesh_setup.current_setup_token(handle)

    assert handle.call_all(
        "provenance_baseline", token.generation, setup_token=token
    ) == [
        {"rank": 0, "setup_generation": 1},
        {"rank": 1, "setup_generation": 1},
    ]
    assert workers.provenance_baseline.calls == [((1,), {})]


def test_timed_out_mutation_latches_dirty_until_recycle():
    running = threading.Event()
    release = threading.Event()

    class TimedOutMutation:
        def get(self, timeout=None):
            raise TimeoutError("mutation completion remains unknown")

    class RunningEndpoint:
        def __init__(self):
            self.thread = None

        def call(self, *args, **kwargs):
            def remote():
                running.set()
                release.wait(timeout=1)

            self.thread = threading.Thread(target=remote)
            self.thread.start()
            assert running.wait(timeout=1)
            return TimedOutMutation()

    workers = _Workers([])
    endpoint = RunningEndpoint()
    workers.load_model = endpoint
    handle, _old, _new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)

    with pytest.raises(TimeoutError, match="completion remains unknown"):
        handle.call_all(
            "load_model", "model.safetensors", {}, [], timeout_s=0.01,
            setup_token=token)

    assert running.is_set() and not release.is_set()
    state = handle.setup_cleanup_state
    assert state is not None
    assert state.outcome is mesh_setup.SetupCleanupOutcome.TIMEOUT_UNKNOWN
    assert state.phase == "load_model RPC completion"
    contender = RenderSession(timeout_s=0)
    with pytest.raises(mesh_setup.TopologyTransitionError):
        contender.bind(handle)
    release.set()
    endpoint.thread.join(timeout=1)


def test_timed_out_provenance_baseline_remains_read_only_and_clean():
    class TimedOutBaseline:
        def get(self, timeout=None):
            raise TimeoutError("baseline completion remains unknown")

    workers = _Workers([])
    workers.provenance_baseline = _Endpoint(TimedOutBaseline())
    handle, _old, _new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)

    with pytest.raises(TimeoutError, match="baseline completion remains unknown"):
        handle.call_all(
            "provenance_baseline", token.generation,
            timeout_s=0.01, setup_token=token)

    assert handle.setup_cleanup_state is None
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_accepted_then_raised_mutation_dispatch_latches_dirty():
    accepted = []

    class AmbiguousEndpoint:
        def call(self, *args, **kwargs):
            accepted.append((args, kwargs))
            raise RuntimeError("dispatch return lost after acceptance")

    workers = _Workers([])
    workers.unload = AmbiguousEndpoint()
    handle, _old, _new = _handle(workers)

    with pytest.raises(RuntimeError, match="return lost"):
        handle.call_all("unload", timeout_s=1)

    assert len(accepted) == 1
    state = handle.setup_cleanup_state
    assert state is not None
    assert state.outcome is mesh_setup.SetupCleanupOutcome.REPORTED_FAILURE
    assert state.phase == "unload RPC completion"
    with pytest.raises(mesh_setup.TopologyTransitionError):
        RenderSession(timeout_s=0).bind(handle)


def test_mutation_latch_publication_interruption_clears_before_dispatch(
    monkeypatch,
):
    """When an interrupt lands just after the pre-dispatch latch is published, dispatch
    clears the latch before any send, retries an interrupted clear and re-raises the interrupt."""
    class StopNow(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    primary = StopNow("interrupted after mutation latch publication")
    workers = _Workers([])
    workers.unload = _Endpoint(_Future(_ValueMesh([{}, {}])))
    handle, _old, _new = _handle(workers)
    original_setattr = MeshHandle.__setattr__
    armed = True

    def interrupt_after_dirty_publication(self, name, value):
        nonlocal armed
        original_setattr(self, name, value)
        if (
            self is handle
            and name == "setup_cleanup_state"
            and value is not None
            and value.outcome is mesh_setup.SetupCleanupOutcome.IN_PROGRESS
            and value.phase == "unload RPC completion"
            and armed
        ):
            armed = False
            raise primary

    monkeypatch.setattr(
        MeshHandle, "__setattr__", interrupt_after_dirty_publication)
    actual_clear = mesh_mod.mesh_rpc._clear_confirmed_mutation
    clear_calls = []

    def interrupt_first_clear(candidate, endpoint_name):
        clear_calls.append(True)
        if len(clear_calls) == 1:
            raise CleanupStop("first mutation latch clear interrupted")
        actual_clear(candidate, endpoint_name)

    monkeypatch.setattr(
        mesh_mod.mesh_rpc,
        "_clear_confirmed_mutation",
        interrupt_first_clear,
    )

    with pytest.raises(StopNow, match="mutation latch publication") as caught:
        handle.call_all("unload", timeout_s=1)

    assert caught.value is primary
    assert clear_calls == [True, True]
    assert any("latch clear attempt 1" in note for note in primary.__notes__)
    assert workers.unload.calls == []
    assert handle.setup_cleanup_state is None
    assert handle.call_all("unload", timeout_s=1) == [{}, {}]
    assert len(workers.unload.calls) == 1
    assert clear_calls == [True, True, True]
    assert handle.setup_cleanup_state is None


def test_successful_mutation_clear_interruption_retries_then_propagates(
    monkeypatch,
):
    """A completed RPC clears DIRTY before its clear interruption escapes."""
    class StopNow(BaseException):
        pass

    primary = StopNow("successful mutation latch clear interrupted")
    values = _ValueMesh([{}, {}])
    workers = _Workers([])
    workers.unload = _Endpoint(_Future(values), _Future(values))
    handle, _old, _new = _handle(workers)
    actual_clear = mesh_mod.mesh_rpc._clear_confirmed_mutation
    clear_calls = []

    def interrupt_first_clear(candidate, endpoint_name):
        clear_calls.append(True)
        if len(clear_calls) == 1:
            raise primary
        actual_clear(candidate, endpoint_name)

    monkeypatch.setattr(
        mesh_mod.mesh_rpc,
        "_clear_confirmed_mutation",
        interrupt_first_clear,
    )

    with pytest.raises(StopNow, match="successful mutation latch clear") as caught:
        handle.call_all("unload", timeout_s=1)

    assert caught.value is primary
    assert clear_calls == [True, True]
    assert len(workers.unload.calls) == 1
    assert handle.setup_cleanup_state is None
    assert handle.call_all("unload", timeout_s=1) == [{}, {}]
    assert len(workers.unload.calls) == 2
    assert handle.setup_cleanup_state is None


def test_successful_mutation_ordinary_clear_failure_recovers_on_retry(
    monkeypatch,
):
    """A recovered ordinary clear failure must not replace RPC success."""
    values = _ValueMesh([{}, {}])
    workers = _Workers([])
    workers.unload = _Endpoint(_Future(values), _Future(values))
    handle, _old, _new = _handle(workers)
    actual_clear = mesh_mod.mesh_rpc._clear_confirmed_mutation
    clear_calls = []

    def fail_first_clear(candidate, endpoint_name):
        clear_calls.append(True)
        if len(clear_calls) == 1:
            raise RuntimeError("first mutation latch clear failed")
        actual_clear(candidate, endpoint_name)

    monkeypatch.setattr(
        mesh_mod.mesh_rpc,
        "_clear_confirmed_mutation",
        fail_first_clear,
    )

    assert handle.call_all("unload", timeout_s=1) == [{}, {}]
    assert clear_calls == [True, True]
    assert len(workers.unload.calls) == 1
    assert handle.setup_cleanup_state is None
    assert handle.call_all("unload", timeout_s=1) == [{}, {}]
    assert len(workers.unload.calls) == 2
    assert handle.setup_cleanup_state is None


def test_successful_mutation_second_clear_cancellation_outranks_first_error(
    monkeypatch,
):
    """A later cancellation cannot be masked by an ordinary clear failure."""
    class StopNow(BaseException):
        pass

    primary = StopNow("second mutation latch clear cancelled")
    values = _ValueMesh([{}, {}])
    workers = _Workers([])
    workers.unload = _Endpoint(_Future(values), _Future(values))
    handle, _old, _new = _handle(workers)
    actual_clear = mesh_mod.mesh_rpc._clear_confirmed_mutation
    clear_calls = []

    def fail_then_cancel(candidate, endpoint_name):
        clear_calls.append(True)
        if len(clear_calls) == 1:
            raise RuntimeError("first mutation latch clear failed")
        actual_clear(candidate, endpoint_name)
        if len(clear_calls) == 2:
            raise primary

    monkeypatch.setattr(
        mesh_mod.mesh_rpc,
        "_clear_confirmed_mutation",
        fail_then_cancel,
    )

    with pytest.raises(StopNow, match="second mutation latch clear") as caught:
        handle.call_all("unload", timeout_s=1)

    assert caught.value is primary
    assert clear_calls == [True, True]
    assert len(workers.unload.calls) == 1
    assert handle.setup_cleanup_state is None
    assert handle.call_all("unload", timeout_s=1) == [{}, {}]
    assert len(workers.unload.calls) == 2
    assert handle.setup_cleanup_state is None


def test_successful_mutation_falsey_first_cancellation_outranks_retry_error(
    monkeypatch,
):
    """A falsey cancellation remains authoritative when both clears fail."""
    class StopNow(BaseException):
        def __bool__(self):
            return False

    primary = StopNow("first mutation latch clear cancelled")
    retry = RuntimeError("second mutation latch clear failed")
    values = _ValueMesh([{}, {}])
    workers = _Workers([])
    workers.unload = _Endpoint(_Future(values))
    handle, _old, _new = _handle(workers)
    failures = [primary, retry]
    clear_calls = []

    def fail_both_clears(_candidate, _endpoint_name):
        clear_calls.append(True)
        raise failures.pop(0)

    monkeypatch.setattr(
        mesh_mod.mesh_rpc,
        "_clear_confirmed_mutation",
        fail_both_clears,
    )

    with pytest.raises(StopNow) as caught:
        handle.call_all("unload", timeout_s=1)

    assert caught.value is primary
    assert clear_calls == [True, True]
    assert len(workers.unload.calls) == 1
    assert handle.setup_cleanup_state is not None
    assert handle.setup_cleanup_state.outcome is mesh_setup.SetupCleanupOutcome.IN_PROGRESS
    assert any(repr(retry) in note for note in primary.__notes__)


def test_successful_mutation_repeated_clear_errors_preserve_first(
    monkeypatch,
):
    """Two ordinary clear failures report the first and retain the latch."""
    primary = RuntimeError("first mutation latch clear failed")
    retry = ValueError("second mutation latch clear failed")
    values = _ValueMesh([{}, {}])
    workers = _Workers([])
    workers.unload = _Endpoint(_Future(values))
    handle, _old, _new = _handle(workers)
    failures = [primary, retry]
    clear_calls = []

    def fail_both_clears(_candidate, _endpoint_name):
        clear_calls.append(True)
        raise failures.pop(0)

    monkeypatch.setattr(
        mesh_mod.mesh_rpc,
        "_clear_confirmed_mutation",
        fail_both_clears,
    )

    with pytest.raises(RuntimeError) as caught:
        handle.call_all("unload", timeout_s=1)

    assert caught.value is primary
    assert clear_calls == [True, True]
    assert len(workers.unload.calls) == 1
    assert handle.setup_cleanup_state is not None
    assert handle.setup_cleanup_state.outcome is mesh_setup.SetupCleanupOutcome.IN_PROGRESS
    assert any(repr(retry) in note for note in primary.__notes__)


def test_dirty_mutation_cannot_be_overwritten_inside_active_session():
    first = TimeoutError("first unload completion unknown")
    workers = _Workers([])
    workers.unload = _Endpoint(
        _Future(error=first),
        _Future(_ValueMesh([{}, {}])),
    )
    handle, _old, _new = _handle(workers)

    with mutation_render_session(handle):
        with pytest.raises(TimeoutError) as caught:
            handle.call_all("unload", timeout_s=0.01)
        original = handle.setup_cleanup_state
        with pytest.raises(mesh_setup.TopologyTransitionError):
            handle.call_all("unload", timeout_s=0.01)

    assert caught.value is first
    assert len(workers.unload.calls) == 1
    assert handle.setup_cleanup_state is original
    assert original is not None
    assert original.phase == "unload RPC completion"


def test_ambiguous_rpc_latch_retries_without_masking_primary(monkeypatch):
    primary = TimeoutError("mutation completion unknown")
    workers = _Workers([])
    workers.unload = _Endpoint(_Future(error=primary))
    handle, _old, _new = _handle(workers)
    actual_latch = handle._latch_ambiguous_mutation
    calls = []

    def interrupt_first_latch(endpoint_name, timeout_s, exc):
        calls.append(True)
        if len(calls) == 1:
            raise KeyboardInterrupt("latch call interrupted")
        actual_latch(endpoint_name, timeout_s, exc)

    monkeypatch.setattr(handle, "_latch_ambiguous_mutation", interrupt_first_latch)

    with pytest.raises(TimeoutError) as caught:
        handle.call_all("unload", timeout_s=0.01)

    assert caught.value is primary
    assert calls == [True, True]
    assert handle.setup_cleanup_state is not None
    assert any("latch attempt 1" in note for note in primary.__notes__)


def test_mutation_is_pessimistically_dirty_before_ambiguous_latch_cleanup(
    monkeypatch,
):
    primary = TimeoutError("mutation completion unknown")
    workers = _Workers([])
    workers.unload = _Endpoint(_Future(error=primary))
    handle, _old, _new = _handle(workers)
    latch_calls = []

    def interrupt_every_latch(*_args):
        latch_calls.append(True)
        raise KeyboardInterrupt("latch cleanup interrupted")

    monkeypatch.setattr(
        handle, "_latch_ambiguous_mutation", interrupt_every_latch)

    with pytest.raises(TimeoutError) as caught:
        handle.call_all("unload", timeout_s=0.01)

    assert caught.value is primary
    assert latch_calls == [True, True]
    state = handle.setup_cleanup_state
    assert state is not None
    assert state.outcome is mesh_setup.SetupCleanupOutcome.IN_PROGRESS
    assert state.phase == "unload RPC completion"
    with pytest.raises(mesh_setup.TopologyTransitionError):
        RenderSession(timeout_s=0).bind(handle)


def test_ambiguous_rpc_logging_never_masks_published_dirty_state():
    primary = TimeoutError("mutation completion unknown")

    class BrokenLogger:
        @staticmethod
        def error(*_args, **_kwargs):
            raise KeyboardInterrupt("logger interrupted")

    workers = _Workers([])
    workers.unload = _Endpoint(_Future(error=primary))
    handle, _old, _new = _handle(workers)
    handle._latch_ambiguous_mutation = lambda endpoint, timeout, exc: (
        mesh_mod.mesh_rpc.latch_ambiguous_mutation(
            handle, endpoint, timeout, exc, logger=BrokenLogger()))

    with pytest.raises(TimeoutError) as caught:
        handle.call_all("unload", timeout_s=0.01)

    assert caught.value is primary
    assert handle.setup_cleanup_state is not None
    assert any("logging also failed" in note for note in primary.__notes__)


def test_setup_token_rejects_replaced_generation_before_send():
    workers = _Workers([_Future(_acks())])
    workers.compute_sigmas = _Endpoint(_Future(_ValueMesh(["rank0", "rank1"])))
    handle, _old, new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)

    handle.ensure_setup(new, "TORCH_FLASH")

    with pytest.raises(mesh_setup.StaleSetupGenerationError):
        handle.call_all("compute_sigmas", setup_token=token)
    assert workers.compute_sigmas.calls == []


def test_setup_token_rejects_policy_drift_without_generation_change():
    workers = _Workers([])
    workers.compute_sigmas = _Endpoint(_Future(_ValueMesh(["rank0", "rank1"])))
    handle, _old, _new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)
    handle.worker_args_key = (("slab_weights", "False"),)

    with pytest.raises(mesh_setup.StaleSetupGenerationError):
        handle.call_all("compute_sigmas", setup_token=token)
    assert workers.compute_sigmas.calls == []


def test_interrupted_policy_broadcast_invalidates_old_setup_token():
    class StopNow(BaseException):
        pass

    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(_Future(error=StopNow()))
    workers.compute_sigmas = _Endpoint(_Future(_ValueMesh(["rank0", "rank1"])))
    handle, _old, _new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)

    with pytest.raises(StopNow):
        handle.apply_worker_args({"slab_weights": False})

    assert handle.worker_args_key is None and handle.active_worker_args == {}
    assert handle.setup_cleanup_state is not None
    assert (
        handle.setup_cleanup_state.outcome
        is mesh_setup.SetupCleanupOutcome.INTERRUPTED
    )
    with pytest.raises(mesh_setup.TopologyTransitionError):
        handle.call_all("compute_sigmas", setup_token=token)
    assert workers.compute_sigmas.calls == []


def test_timed_out_policy_broadcast_latches_dirty_before_remote_completion():
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(
        _Future(error=TimeoutError("policy completion unknown")))
    handle, _old, _new = _handle(workers)

    with pytest.raises(TimeoutError, match="policy completion unknown"):
        handle.apply_worker_args({"slab_weights": False})

    assert handle.worker_args_key is None and handle.active_worker_args == {}
    state = handle.setup_cleanup_state
    assert state is not None
    assert state.outcome is mesh_setup.SetupCleanupOutcome.TIMEOUT_UNKNOWN
    with pytest.raises(mesh_setup.TopologyTransitionError):
        RenderSession(timeout_s=0).bind(handle)


def test_interrupted_policy_restore_does_not_mask_body_failure():
    class StopNow(BaseException):
        pass

    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(
        _Future(_ValueMesh([{}, {}])), _Future(error=StopNow()))
    handle, _old, _new = _handle(workers)

    with pytest.raises(RuntimeError, match="render root failure"):
        with handle.temporary_worker_args({}, {"slab_weights": False}):
            raise RuntimeError("render root failure")

    assert handle.worker_args_key is None and handle.active_worker_args == {}
