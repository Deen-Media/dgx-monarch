"""Cross-render pipeline (RenderPipeline, KSampler Pipeline node): FIFO order, depth bound, abort
and cleanup, auto-gate, preflight and result merging, with fakes in place of GPUs and actors."""
import sys
import threading
import types
from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.nodes.common as common
import dgx_monarch.nodes.consent_waiver as consent_waiver_mod
import dgx_monarch.nodes.pipeline as pipeline_mod
import dgx_monarch.nodes.samplers as samplers_mod
from dgx_monarch import accuracy_waiver
from dgx_monarch.mesh import MeshHandle
from dgx_monarch.nodes.common import RenderPipeline
from dgx_monarch.nodes.render_session import RenderSession
from dgx_monarch.nodes.samplers import DGXMonarchKSamplerPipeline


class _State:
    def __init__(self):
        self.events = []          # Holds (kind, seq) pairs in call order; kind is submit, submit-fail or result.
        self.outstanding = 0
        self.max_outstanding = 0
        self.resulted = []
        self.fail_on = None       # The seq whose result() raises, or None.
        self.submit_fail_on = None


def _fake_submit(state):
    class _FakePending:
        def __init__(self, seq):
            self.seq = seq

        def result(self):
            state.outstanding -= 1
            state.events.append(("result", self.seq))
            state.resulted.append(self.seq)
            if state.fail_on == self.seq:
                raise RuntimeError(f"boom on seq {self.seq}")
            return {"seq": self.seq, "samples": self.seq}

    def fake(model, request, latent, cfg_value, steps_hint, seq=0, depth=1,
             progress_on_step=None, handoff=None):
        if state.submit_fail_on == seq:
            state.events.append(("submit-fail", seq))
            raise RuntimeError(f"submit boom on seq {seq}")
        state.outstanding += 1
        state.max_outstanding = max(state.max_outstanding, state.outstanding)
        state.events.append(("submit", seq))
        pending = _FakePending(seq)
        if handoff is not None:
            handoff.publish(pending)
        return pending

    return fake


def test_pipeline_orders_fifo(monkeypatch):
    state = _State()
    monkeypatch.setattr(common, "submit_render", _fake_submit(state))
    pipe = RenderPipeline(depth=3)
    for _ in range(5):
        pipe.push(None, {}, {}, 4.5, 10)
    outs = pipe.drain()
    assert [o["seq"] for o in outs] == [0, 1, 2, 3, 4]
    assert state.resulted == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_pipeline_depth_bound(monkeypatch, depth):
    state = _State()
    monkeypatch.setattr(common, "submit_render", _fake_submit(state))
    pipe = RenderPipeline(depth=depth)
    for _ in range(9):
        pipe.push(None, {}, {}, 4.5, 10)
    pipe.drain()
    assert state.max_outstanding <= depth
    assert state.resulted == list(range(9))


def test_depth1_is_strictly_sequential(monkeypatch):
    state = _State()
    monkeypatch.setattr(common, "submit_render", _fake_submit(state))
    pipe = RenderPipeline(depth=1)
    for _ in range(3):
        pipe.push(None, {}, {}, 4.5, 10)
    pipe.drain()
    assert state.events == [
        ("submit", 0), ("result", 0),
        ("submit", 1), ("result", 1),
        ("submit", 2), ("result", 2),
    ]
    assert state.max_outstanding == 1


def test_pipeline_aborts_and_drains_on_failure(monkeypatch):
    state = _State()
    state.fail_on = 2
    monkeypatch.setattr(common, "submit_render", _fake_submit(state))
    pipe = RenderPipeline(depth=2)
    with pytest.raises(RuntimeError, match="boom on seq 2"):
        for _ in range(5):
            pipe.push(None, {}, {}, 4.5, 10)
        pipe.drain()
    assert state.outstanding == 0


def test_pipeline_aborts_and_drains_on_submission_failure(monkeypatch):
    state = _State()
    state.submit_fail_on = 2
    monkeypatch.setattr(common, "submit_render", _fake_submit(state))
    pipe = RenderPipeline(depth=3)
    pipe.push(None, {}, {}, 4.5, 10)
    pipe.push(None, {}, {}, 4.5, 10)
    with pytest.raises(RuntimeError, match="submit boom on seq 2"):
        pipe.push(None, {}, {}, 4.5, 10)
    assert state.outstanding == 0
    assert state.resulted == [0, 1]


@pytest.mark.parametrize("operation", ["drain", "push"])
def test_pipeline_session_cleanup_preserves_primary_failure(monkeypatch, operation):
    class Primary(BaseException):
        pass

    class Cleanup(BaseException):
        pass

    primary = Primary("render failed")
    cleanup_calls = []
    pipe = RenderPipeline(depth=1)

    def fail_close():
        cleanup_calls.append(True)
        raise Cleanup("session close failed")

    monkeypatch.setattr(pipe._session, "close", fail_close)
    if operation == "drain":
        class Pending:
            def result(self):
                raise primary

            def abandon(self):
                return None

        pipe._inflight.append(Pending())
        invoke = pipe.drain
    else:
        monkeypatch.setattr(
            common,
            "submit_render",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        )

        def invoke():
            return pipe.push(None, {}, {}, 1.0, 1)

    with pytest.raises(Primary) as raised:
        invoke()

    assert raised.value is primary
    assert cleanup_calls
    assert any("session close failed" in note for note in primary.__notes__)


@pytest.mark.parametrize("cancel_recovers", [True, False])
def test_pipeline_reports_only_terminal_cancel_cleanup_failure(
    monkeypatch, cancel_recovers,
):
    primary = RuntimeError("pipeline submission failed")
    cancel_failures = [
        RuntimeError("first best-effort cancel failed"),
        RuntimeError("second best-effort cancel failed"),
    ]
    cancel_calls = 0
    pipe = RenderPipeline(depth=1)

    def fail_submission(*_args, **_kwargs):
        raise primary

    def cancel_cleanup():
        nonlocal cancel_calls
        failure = cancel_failures[cancel_calls]
        cancel_calls += 1
        if cancel_calls == 1 or not cancel_recovers:
            raise failure

    monkeypatch.setattr(pipe, "_push_bound", fail_submission)
    monkeypatch.setattr(pipe, "_abort", cancel_cleanup)

    with pytest.raises(RuntimeError) as caught:
        pipe.push(None, {}, {}, 1.0, 1)

    notes = getattr(primary, "__notes__", ())
    assert caught.value is primary
    assert cancel_calls == 2
    assert any("pipeline abort also failed" in note for note in notes) is (
        not cancel_recovers)


def test_pipeline_surfaces_session_close_failure_without_primary(monkeypatch):
    class Cleanup(BaseException):
        pass

    pipe = RenderPipeline(depth=1)
    monkeypatch.setattr(
        pipe._session,
        "close",
        lambda: (_ for _ in ()).throw(Cleanup("session close failed")),
    )

    with pytest.raises(Cleanup, match="session close failed"):
        pipe.drain()


@pytest.mark.parametrize("operation", ["drain", "_abort"])
def test_pipeline_retries_first_session_cleanup_interruption(monkeypatch, operation):
    class StopNow(BaseException):
        pass

    handle = object.__new__(MeshHandle)
    handle.lock = threading.RLock()
    pipe = RenderPipeline(depth=1)
    pipe._session.bind(handle)
    actual_close = pipeline_mod._close_pipeline_session
    cleanup_calls = []

    def interrupt_first_cleanup(session, primary):
        cleanup_calls.append(True)
        if len(cleanup_calls) == 1:
            raise StopNow(f"first pipeline {operation} cleanup interrupted")
        actual_close(session, primary)

    monkeypatch.setattr(
        pipeline_mod, "_close_pipeline_session", interrupt_first_cleanup)

    with pytest.raises(StopNow, match=f"first pipeline {operation} cleanup interrupted"):
        getattr(pipe, operation)()

    assert cleanup_calls == [True, True]
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_pipeline_abort_retries_terminal_abandonment_before_releasing_owner(
    monkeypatch,
):
    class StopNow(BaseException):
        pass

    class Pending:
        def __init__(self):
            self.abandoned = 0

        def cancel(self):
            return None

        def result(self):
            raise TimeoutError("still running")

        def abandon(self):
            self.abandoned += 1

    handle = object.__new__(MeshHandle)
    handle.lock = threading.RLock()
    pipe = RenderPipeline(depth=1)
    pipe._session.bind(handle)
    pending = Pending()
    pipe._inflight.append(pending)
    actual_abandon = pipeline_mod._abandon_all_inflight
    calls = []

    def interrupt_first_abandon(inflight):
        calls.append(True)
        if len(calls) == 1:
            raise StopNow("first terminal abandonment interrupted")
        actual_abandon(inflight)

    monkeypatch.setattr(
        pipeline_mod, "_abandon_all_inflight", interrupt_first_abandon)

    with pytest.raises(StopNow, match="first terminal abandonment interrupted"):
        pipe._abort()

    assert calls == [True, True]
    assert pending.abandoned == 1
    assert not pipe._inflight
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


@pytest.mark.parametrize("explosive", [False, True])
def test_pipeline_abandon_preserves_exact_first_baseexception(explosive):
    class FirstFailure(BaseException):
        def __bool__(self):
            if explosive:
                raise AssertionError("exception truthiness must not be evaluated")
            return False

    first = FirstFailure("first abandon failure")

    class Pending:
        def __init__(self):
            self.calls = 0

        def abandon(self):
            self.calls += 1
            if self.calls == 1:
                raise first
            raise BaseException("second abandon failure")

    with pytest.raises(BaseException) as raised:
        pipeline_mod._abandon_all_inflight(pipeline_mod.deque([Pending()]))

    assert raised.value is first


@pytest.mark.parametrize("explosive", [False, True])
def test_pipeline_abort_passes_exact_primary_to_session_cleanup(
    monkeypatch, explosive
):
    class Primary(BaseException):
        def __bool__(self):
            if explosive:
                raise AssertionError("exception truthiness must not be evaluated")
            return False

    primary = Primary("render failed")
    cleanup = BaseException("abandon cleanup failed")
    received = []
    pipe = RenderPipeline(depth=1)
    monkeypatch.setattr(
        pipeline_mod, "_abandon_all_inflight",
        lambda _inflight: (_ for _ in ()).throw(cleanup),
    )
    monkeypatch.setattr(
        pipeline_mod, "_close_pipeline_session",
        lambda _session, active: received.append(active),
    )

    pipeline_mod._abort_cleanup(pipe, primary)

    assert received == [primary]
    assert any("abort cleanup" in note for note in primary.__notes__)


def test_pipeline_drain_primary_survives_first_cleanup_helper_interruption(
    monkeypatch,
):
    class Primary(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    primary = Primary("pipeline collection interrupted")
    handle = object.__new__(MeshHandle)
    handle.lock = threading.RLock()
    pipe = RenderPipeline(depth=1)
    pipe._session.bind(handle)
    pipe._inflight.append(object())
    monkeypatch.setattr(
        pipe, "_collect_oldest", lambda: (_ for _ in ()).throw(primary))
    actual_close = pipeline_mod._close_pipeline_session
    cleanup_calls = []

    def interrupt_first_cleanup(session, received_primary):
        cleanup_calls.append(received_primary)
        if len(cleanup_calls) == 1:
            raise CleanupStop("first pipeline drain cleanup helper call interrupted")
        actual_close(session, received_primary)

    monkeypatch.setattr(
        pipeline_mod, "_close_pipeline_session", interrupt_first_cleanup)

    with pytest.raises(BaseException) as raised:
        pipe.drain()

    assert cleanup_calls == [primary, primary]
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()
    assert raised.value is primary


def test_pipeline_abort_deadline_terminally_abandons_every_remainder(monkeypatch):
    from types import SimpleNamespace

    from dgx_monarch import mesh_setup, telemetry
    from dgx_monarch.nodes.pending import PendingRender

    handle = SimpleNamespace(
        lock=threading.RLock(), sample_leases={4: 3},
        abandoned_sample_leases={},
        collect_sample=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TimeoutError("still running")),
        cancel_sample=lambda *_args, **_kwargs: None,
    )
    closed = []
    pending = []
    for index in range(3):
        future = mesh_setup.SetupBoundFuture(object(), handle, 4)
        progress = SimpleNamespace(
            __exit__=lambda *_args, i=index: closed.append(i))
        pending.append(PendingRender(
            handle, future, progress, None, {}, 10.0, str(index),
            lambda *_args: {}))
    ticks = iter((100.0, 131.0, 131.0))
    monkeypatch.setattr(pipeline_mod.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
    pipe = RenderPipeline(depth=3)
    pipe._inflight.extend(pending)

    pipe._abort()

    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {4: 3}
    assert sorted(closed) == [0, 1, 2]
    assert list(pipe._inflight) == []


def test_pipeline_gates_before_first_unproven_submit_and_quarantines(monkeypatch):
    events = []
    model = type("Model", (), {})()
    model.mesh = type("Mesh", (), {"worker_args": {
        "lora_low_rss": True, "slab_weights": True}})()

    class Pending:
        def result(self):
            return {"samples": 1}

    monkeypatch.setattr(common, "auto_gate_required", lambda *args: True)
    monkeypatch.setattr(
        common, "_maybe_auto_gate",
        lambda *args: events.append("gate") or "INCONCLUSIVE",
    )

    def submit(*args, **kwargs):
        events.append("submit")
        assert model.mesh.worker_args["lora_low_rss"] is False
        assert model.mesh.worker_args["slab_weights"] is False
        return Pending()

    monkeypatch.setattr(common, "submit_render", submit)
    pipe = RenderPipeline(depth=2)
    pipe.push(model, {"kind": "ksampler"}, {}, 0.0, 2)
    assert pipe.drain() == [{"samples": 1}]
    assert events == ["gate", "submit"]


def test_pipeline_lookup_error_fails_closed_before_submit(monkeypatch):
    events = []
    model = type("Model", (), {})()
    model.mesh = type("Mesh", (), {"worker_args": {
        "lora_low_rss": True, "slab_weights": True}})()

    class Pending:
        def result(self):
            return {"samples": 1}

    def broken_context(*args):
        events.append("lookup")
        raise OSError("ledger unavailable")

    monkeypatch.setattr(common, "_auto_gate_context", broken_context)

    def submit(*args, **kwargs):
        events.append("submit")
        assert model.mesh.worker_args == {
            "lora_low_rss": False, "slab_weights": False}
        return Pending()

    monkeypatch.setattr(common, "submit_render", submit)
    pipe = RenderPipeline(depth=2)
    pipe.push(model, {"kind": "ksampler"}, {}, 1.0, 2)
    assert pipe.drain() == [{"samples": 1}]
    assert events == ["lookup", "lookup", "submit"]


def test_run_render_abandons_terminal_timeout(monkeypatch):
    calls = []

    class Pending:
        def result(self):
            raise TimeoutError("terminal wait")

        def abandon(self):
            calls.append("abandon")

    monkeypatch.setattr(common, "_maybe_auto_gate", lambda *_args: None)
    monkeypatch.setattr(common, "submit_render", lambda *_args, **_kwargs: Pending())

    with pytest.raises(TimeoutError, match="terminal wait"):
        common.run_render(object(), {}, {}, None, 1)

    assert calls == ["abandon"]


def test_popped_pipeline_timeout_is_terminally_abandoned():
    calls = []

    class Pending:
        def result(self):
            raise TimeoutError("terminal wait")

        def abandon(self):
            calls.append("abandon")

        def cancel(self):
            calls.append("cancel")

    pipe = RenderPipeline(depth=1)
    pipe._inflight.append(Pending())

    with pytest.raises(TimeoutError, match="terminal wait"):
        pipe._collect_oldest()

    assert calls == ["abandon"]
    assert list(pipe._inflight) == []


def test_collect_oldest_keeps_local_ownership_if_popleft_removes_then_raises():
    from collections import deque

    class StopNow(BaseException):
        pass

    events = []

    class Pending:
        def __init__(self, name):
            self.name = name

        def result(self):
            events.append(("result", self.name))
            return {"name": self.name}

        def abandon(self):
            events.append(("abandon", self.name))

        def cancel(self):
            events.append(("cancel", self.name))

    class InterruptingDeque(deque):
        armed = True

        def popleft(self):
            value = super().popleft()
            if self.armed:
                self.armed = False
                raise StopNow("popleft interrupted after ownership removal")
            return value

    pipe = RenderPipeline(depth=2)
    pipe._inflight = InterruptingDeque([Pending("first"), Pending("second")])

    with pytest.raises(StopNow):
        pipe._collect_oldest()

    assert ("abandon", "first") in events
    assert ("result", "second") in events
    assert list(pipe._inflight) == []


def test_parse_seeds():
    p = DGXMonarchKSamplerPipeline._parse_seeds
    assert p("0, 1, 2, 3") == [0, 1, 2, 3]
    assert p("7\n8\n9") == [7, 8, 9]
    assert p("  42 ") == [42]
    with pytest.raises(ValueError):
        p("   ")


class NestedTensor:
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


class LyingTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is torch.equal:
            return True
        return super().__torch_function__(func, types, args, kwargs or {})


@pytest.fixture
def direct_nested_tensor_surface(monkeypatch):
    comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
    nested = types.ModuleType("comfy.nested_tensor")
    nested.NestedTensor = NestedTensor
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested)
    monkeypatch.setattr(comfy, "nested_tensor", nested, raising=False)


def _run_pipeline_result(monkeypatch, outputs):
    class Pipeline:
        def __init__(self, depth, on_step=None):
            self.depth = depth

        def push(self, *_args, **_kwargs):
            return None

        def drain(self):
            return list(outputs)

    monkeypatch.setattr(samplers_mod, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(samplers_mod, "auto_gate_required", lambda *_args: False)
    monkeypatch.setattr(samplers_mod, "RenderPipeline", Pipeline)
    model = SimpleNamespace(mesh=SimpleNamespace(pipeline_depth=2))
    result = DGXMonarchKSamplerPipeline().sample(
        model,
        "10,11",
        1,
        1.0,
        "euler",
        "simple",
        [],
        [],
        {"samples": torch.zeros(1, 1)},
    )
    return result["result"][0] if isinstance(result, dict) else result[0]


def _run_pipeline_outputs(monkeypatch, outputs):
    return _run_pipeline_result(monkeypatch, outputs)["samples"]


def _run_public_pipeline_source(monkeypatch, source, *, trap_side_effects=False):
    class Pending:
        def __init__(self, latent, seq):
            self.latent = latent
            self.seq = seq

        def result(self):
            return consent_waiver_mod.stamp_result(
                {"samples": torch.full((1, 1), float(self.seq))}, [], self.latent)

        def abandon(self):
            return None

        def cancel(self):
            return None

    def submit(_model, _request, latent, _cfg, _steps, *, seq=0,
               handoff=None, **_kwargs):
        pending = Pending(latent, seq)
        if handoff is not None:
            handoff.publish(pending)
        return pending

    if trap_side_effects:
        from dgx_monarch import adoption_evidence

        def unexpected(*_args, **_kwargs):
            raise AssertionError("pipeline provenance validation ran too late")

        monkeypatch.setattr(common, "submit_render", unexpected)
        monkeypatch.setattr(samplers_mod, "conditioning_for_wire", unexpected)
        monkeypatch.setattr(samplers_mod, "krea2_ref_preflight_summary", unexpected)
        monkeypatch.setattr(samplers_mod, "auto_gate_required", unexpected)
        monkeypatch.setattr(samplers_mod, "RenderPipeline", unexpected)
        monkeypatch.setattr(adoption_evidence, "require_inactive_context", unexpected)
    else:
        monkeypatch.setattr(common, "submit_render", submit)
        monkeypatch.setattr(samplers_mod, "conditioning_for_wire", lambda value: value)
        monkeypatch.setattr(samplers_mod, "auto_gate_required", lambda *_args: False)
    model = SimpleNamespace(mesh=SimpleNamespace(pipeline_depth=2))
    result = DGXMonarchKSamplerPipeline().sample(
        model, "10,11", 1, 1.0, "euler", "simple", [], [], source,
    )
    return result["result"][0] if isinstance(result, dict) else result[0]


def test_pipeline_aggregates_batch_index_and_all_waiver_provenance(monkeypatch):
    stamp_key = accuracy_waiver.STAMPED_RESULT_KEY
    outputs = [
        {
            "samples": torch.full((2, 1), float(index)),
            "batch_index": [4, 9],
            stamp_key: [
                {
                    "run_id": "run-a",
                    "guard": "ring_pad",
                    "stamp": "first",
                    "inherited": True,
                },
                *(
                    [{"run_id": "run-b", "guard": "ring_pad", "stamp": "second"}]
                    if index else []
                ),
            ],
        }
        for index in range(2)
    ]
    outputs[1][stamp_key] = tuple(outputs[1][stamp_key])

    combined = _run_pipeline_result(monkeypatch, outputs)

    assert tuple(combined["samples"].shape) == (4, 1)
    assert combined["batch_index"] == [4, 9, 4, 9]
    assert [entry["run_id"] for entry in combined[stamp_key]] == ["run-a", "run-b"]
    assert combined[stamp_key][0]["inherited"] is True


@pytest.mark.parametrize("malformed", [{}, "", None, {"not": "a sequence"}])
def test_pipeline_rejects_present_malformed_waiver_container(monkeypatch, malformed):
    stamp_key = accuracy_waiver.STAMPED_RESULT_KEY
    with pytest.raises(RuntimeError, match="waiver provenance must be a list or tuple"):
        _run_pipeline_result(
            monkeypatch,
            [{"samples": torch.zeros(1, 1), stamp_key: malformed}],
        )


@pytest.mark.parametrize(
    "entry",
    [
        {"guard": "ring_pad"},
        {"run_id": "run-a"},
        {"run_id": "", "guard": "ring_pad"},
        {"run_id": "run-a", "guard": ""},
        {"run_id": 7, "guard": "ring_pad"},
        {"run_id": "run-a", "guard": 7},
    ],
)
def test_pipeline_rejects_waiver_entry_without_typed_identity(monkeypatch, entry):
    stamp_key = accuracy_waiver.STAMPED_RESULT_KEY
    with pytest.raises(RuntimeError, match="nonempty string 'run_id' and 'guard'"):
        _run_pipeline_result(
            monkeypatch,
            [{"samples": torch.zeros(1, 1), stamp_key: [entry]}],
        )


@pytest.mark.parametrize(
    "malformed",
    [
        {},
        "",
        None,
        {"not": "a sequence"},
        [None],
        [{"guard": "ring_pad"}],
        [{"run_id": "", "guard": "ring_pad"}],
        [{"run_id": 7, "guard": "ring_pad"}],
    ],
)
def test_pipeline_public_entry_rejects_malformed_inherited_provenance(
    monkeypatch, malformed,
):
    source = {
        "samples": torch.zeros(1, 1),
        accuracy_waiver.STAMPED_RESULT_KEY: malformed,
    }
    with pytest.raises(RuntimeError, match="inherited waiver provenance"):
        _run_public_pipeline_source(
            monkeypatch, source, trap_side_effects=True)


def test_pipeline_public_entry_preserves_valid_inherited_provenance(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        consent_waiver_mod, "_record_use_row",
        lambda entry: recorded.append(dict(entry)),
    )
    inherited = {"run_id": "parent-run", "guard": "ring_pad", "stamp": "ancestor"}
    source = {
        "samples": torch.zeros(1, 1),
        accuracy_waiver.STAMPED_RESULT_KEY: [inherited, dict(inherited)],
    }

    result = _run_public_pipeline_source(monkeypatch, source)

    assert result[accuracy_waiver.STAMPED_RESULT_KEY] == [
        {**inherited, "inherited": True}
    ]
    assert recorded == []


def test_pipeline_public_entry_keeps_absent_provenance_absent(monkeypatch):
    result = _run_public_pipeline_source(
        monkeypatch, {"samples": torch.zeros(1, 1)})

    assert accuracy_waiver.STAMPED_RESULT_KEY not in result


def test_sequential_entry_rejects_inherited_provenance_before_side_effects(
    monkeypatch,
):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("sequential provenance validation ran too late")

    monkeypatch.setattr(
        common.render_preflight, "activation_footprint_preflight_for_request",
        unexpected,
    )
    monkeypatch.setattr(common, "_claim_adoption_context", unexpected)
    monkeypatch.setattr(common, "_bind_packed_render_model", unexpected)
    monkeypatch.setattr(common, "_maybe_auto_gate", unexpected)
    monkeypatch.setattr(common, "submit_render", unexpected)

    with pytest.raises(RuntimeError, match="inherited waiver provenance"):
        common.run_render(
            object(), {}, {
                "samples": torch.zeros(1, 1),
                accuracy_waiver.STAMPED_RESULT_KEY: {},
            }, None, 1,
        )


def test_render_pipeline_push_rejects_inherited_provenance_before_bind(
    monkeypatch,
):
    from dgx_monarch import adoption_evidence

    def unexpected(*_args, **_kwargs):
        raise AssertionError("RenderPipeline provenance validation ran too late")

    monkeypatch.setattr(adoption_evidence, "require_inactive_context", unexpected)
    monkeypatch.setattr(
        common.render_preflight, "activation_footprint_preflight_for_request",
        unexpected,
    )
    monkeypatch.setattr(common, "_bind_packed_render_model", unexpected)
    monkeypatch.setattr(common, "auto_gate_required", unexpected)
    monkeypatch.setattr(common, "submit_render", unexpected)
    pipe = RenderPipeline(depth=1)

    with pytest.raises(RuntimeError, match="inherited waiver provenance"):
        pipe.push(
            object(), {}, {
                "samples": torch.zeros(1, 1),
                accuracy_waiver.STAMPED_RESULT_KEY: [None],
            }, None, 1,
        )


def test_pipeline_keeps_missing_batch_index_absent(monkeypatch):
    combined = _run_pipeline_result(
        monkeypatch,
        [
            {"samples": torch.full((2, 1), float(index))}
            for index in range(2)
        ],
    )

    assert tuple(combined["samples"].shape) == (4, 1)
    assert "batch_index" not in combined


def test_pipeline_repeats_explicit_batch_one_index_per_render(monkeypatch):
    combined = _run_pipeline_result(
        monkeypatch,
        [
            {"samples": torch.full((1, 1), float(index)), "batch_index": [7]}
            for index in range(3)
        ],
    )

    assert tuple(combined["samples"].shape) == (3, 1)
    assert combined["batch_index"] == [7, 7, 7]


def test_pipeline_rejects_mixed_batch_index_presence(monkeypatch):
    with pytest.raises(RuntimeError, match="inconsistently returned batch_index"):
        _run_pipeline_result(
            monkeypatch,
            [
                {"samples": torch.zeros(1, 1), "batch_index": [0]},
                {"samples": torch.zeros(1, 1)},
            ],
        )


def test_pipeline_rejects_batch_index_length_mismatch(monkeypatch):
    with pytest.raises(RuntimeError, match="length 1 does not match samples batch 2"):
        _run_pipeline_result(
            monkeypatch,
            [
                {"samples": torch.zeros(2, 1), "batch_index": [0]},
                {"samples": torch.zeros(2, 1), "batch_index": [0, 1]},
            ],
        )


@pytest.mark.parametrize("batch_index", [[True], [0.0], ["x"], [-1]])
def test_pipeline_rejects_invalid_batch_index_entry(monkeypatch, batch_index):
    with pytest.raises(RuntimeError, match="nonnegative integers"):
        _run_pipeline_result(
            monkeypatch,
            [{"samples": torch.zeros(1, 1), "batch_index": batch_index}],
        )


def test_pipeline_concatenates_packed_results_modality_wise(
    monkeypatch, direct_nested_tensor_surface,
):
    outputs = [
        {"samples": NestedTensor((
            torch.full((1, 2, 2), float(index)),
            torch.full((1, 3), float(index + 10)),
        ))}
        for index in range(2)
    ]

    combined = _run_pipeline_outputs(monkeypatch, outputs)

    assert type(combined) is NestedTensor
    video, audio = combined.unbind()
    assert tuple(video.shape) == (2, 2, 2)
    assert tuple(audio.shape) == (2, 3)
    assert video[:, 0, 0].tolist() == [0.0, 1.0]
    assert audio[:, 0].tolist() == [10.0, 11.0]


@pytest.mark.parametrize(
    ("second", "match"),
    [
        ({"samples": torch.zeros(1, 2, 2)}, "mixed Tensor and NestedTensor"),
        (
            {"samples": NestedTensor((torch.zeros(1, 2, 2),))},
            "modality count",
        ),
        (
            {"samples": NestedTensor((
                torch.zeros(1, 2, 3), torch.zeros(1, 3),
            ))},
            "changed structure for packed modality 0",
        ),
        (
            {"samples": NestedTensor((
                torch.zeros(1, 2, 2, dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.float64),
            ))},
            "changed structure for packed modality 0",
        ),
    ],
)
def test_pipeline_rejects_packed_result_structure_drift(
    monkeypatch, direct_nested_tensor_surface, second, match,
):
    first = {"samples": NestedTensor((
        torch.zeros(1, 2, 2), torch.zeros(1, 3),
    ))}
    with pytest.raises(RuntimeError, match=match):
        _run_pipeline_outputs(monkeypatch, [first, second])


def test_pipeline_rejects_nested_subclass(
    monkeypatch, direct_nested_tensor_surface,
):
    class NestedSubclass(NestedTensor):
        pass

    first = {"samples": NestedTensor((torch.zeros(1, 2), torch.zeros(1, 3)))}
    second = {
        "samples": NestedSubclass((torch.zeros(1, 2), torch.zeros(1, 3)))
    }
    with pytest.raises(RuntimeError, match="valid direct Comfy NestedTensor"):
        _run_pipeline_outputs(monkeypatch, [first, second])


@pytest.mark.parametrize("packed", [False, True], ids=["flat", "packed"])
def test_pipeline_rejects_tensor_subclass_outputs(
    monkeypatch, direct_nested_tensor_surface, packed,
):
    lying = torch.zeros(1, 2).as_subclass(LyingTensor)
    samples = NestedTensor((lying, torch.zeros(1, 3))) if packed else lying

    with pytest.raises(RuntimeError, match=r"exact tensor|valid direct Comfy"):
        _run_pipeline_outputs(
            monkeypatch,
            [{"samples": samples}, {"samples": samples}],
        )


def test_pipeline_push_runs_activation_preflight(monkeypatch):
    """A pipelined submission runs the family-scoped activation preflight once,
    as run_render does, so the pipeline path cannot skip that capacity refusal."""
    import dgx_monarch.nodes.render_preflight as render_preflight

    state = _State()
    monkeypatch.setattr(common, "submit_render", _fake_submit(state))
    calls = []
    monkeypatch.setattr(
        render_preflight, "activation_footprint_preflight_for_request",
        lambda model, request, latent: calls.append((model, request, latent)))
    pipe = RenderPipeline(depth=2)
    pipe.push(None, {"kind": "ksampler"}, {"samples": None}, 4.5, 10)
    pipe.drain()
    if len(calls) != 1:
        pytest.fail(f"expected exactly one preflight call, saw {len(calls)}")


def test_pipeline_collect_stamps_residency_memo(monkeypatch):
    """A render completed through the pipeline stamps the residency memo as
    run_render does, so a warm re-submit is not charged again for resident weights."""
    import dgx_monarch.nodes.render_preflight as render_preflight

    state = _State()
    monkeypatch.setattr(common, "submit_render", _fake_submit(state))
    monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", None)
    model = SimpleNamespace(unet_name="scail.safetensors", mesh=None)
    monkeypatch.setattr(common, "model_for_request", lambda m, _r: m)
    monkeypatch.setattr(common, "_bind_packed_render_model", lambda m, _l, _c: (m, None))
    monkeypatch.setattr(common, "auto_gate_required", lambda *_a: False)
    pipe = RenderPipeline(depth=1)
    pipe.push(model, {"kind": "ksampler"}, {"samples": None}, 4.5, 10)
    pipe.drain()
    if render_preflight._LAST_RENDERED_UNET != "scail.safetensors":
        pytest.fail("pipeline completion must stamp the residency memo")
