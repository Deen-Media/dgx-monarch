"""Fleet job assembly, wave collection and cleanup, fail-closed residency, and job rows."""
import sys
import threading
import types
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.gate_ledger as ledger_mod
import dgx_monarch.mesh_safety as mesh_safety_mod
import dgx_monarch.mesh_setup as mesh_setup
import dgx_monarch.nodes.consent_waiver as consent_waiver_mod
import dgx_monarch.nodes.fleet as fleet_mod
import dgx_monarch.nodes.gate as gate_mod
from dgx_monarch import accuracy_waiver, telemetry_fleet
from dgx_monarch.actor import model_store as model_store_mod
from dgx_monarch.mesh import MeshHandle
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.fleet import (
    _collect_fleet_wave,
    _fleet_worker_policy,
    build_jobs,
)
from dgx_monarch.nodes.render_session import RenderSession


def _artifact_identity(*signatures):
    artifacts = [
        {"kind": "diffusion_models", "name": "model.safetensors",
         "signature": signatures[0]},
    ]
    artifacts.extend(
        {"kind": "loras", "name": f"style-{index}.safetensors", "signature": signature}
        for index, signature in enumerate(signatures[1:], 1)
    )
    return {
        "digest": ledger_mod.artifact_set_signature(signatures).current,
        "comfy": "commit",
        "artifacts": artifacts,
    }


def _fleet_spec(handle, worker_args, *, with_lora=True, options=None):
    mesh = MeshSpec(
        handle=handle,
        topology_preset="uly2",
        attention="TORCH_FLASH",
        sync_ulysses=True,
        worker_args=worker_args,
    )
    loras = (({"name": "style-1.safetensors", "strength": 1.0},)
             if with_lora else ())
    return ModelSpec(
        mesh=mesh,
        unet_name="model.safetensors",
        options={} if options is None else options,
        loras=loras,
    )


def test_prompt_lines_become_jobs_with_sequential_seeds():
    jobs = build_jobs("a cat\n\n  a dog  \n", None, ["NEG"], 100)
    assert [j["text"] for j in jobs] == ["a cat", "a dog"]
    assert [j["seed"] for j in jobs] == [100, 101]
    assert all(j["negative"] == "NEG" for j in jobs)


def test_conditioning_list_and_lines_combine():
    jobs = build_jobs("a dog", ["COND_A", "COND_B"], ["NEG"], 7)
    assert len(jobs) == 3
    assert jobs[0]["positive"] == "COND_A" and jobs[2]["text"] == "a dog"
    assert [j["seed"] for j in jobs] == [7, 8, 9]


def test_negative_broadcast_rules():
    jobs = build_jobs("a\nb", None, ["N1", "N2"], 0)
    assert [j["negative"] for j in jobs] == ["N1", "N2"]
    with pytest.raises(ValueError):
        build_jobs("a\nb\nc", None, ["N1", "N2"], 0)


@pytest.mark.parametrize("body_fails", [True, False])
def test_fleet_session_close_preserves_primary_or_surfaces_cleanup(
    monkeypatch, body_fails
):
    class Primary(BaseException):
        pass

    class Cleanup(BaseException):
        pass

    primary = Primary("fleet body failed")

    class FailingSession:
        def bind(self, _handle):
            return None

        def activate(self):
            return nullcontext()

        def close(self):
            raise Cleanup("fleet session close failed")

    handle = object()
    spec = _fleet_spec(handle, {}, with_lora=False)
    monkeypatch.setattr(fleet_mod, "RenderSession", FailingSession)
    monkeypatch.setattr(fleet_mod, "_enforce_persisted_quarantine", lambda *_args: None)
    monkeypatch.setattr(fleet_mod, "ensure_live", lambda value: value)

    def fleet_bound(*_args, **_kwargs):
        if body_fails:
            raise primary
        return {"samples": torch.zeros(1)}

    monkeypatch.setattr(
        fleet_mod.DGXMonarchFleetKSampler, "_fleet_bound", fleet_bound)

    def invoke():
        return fleet_mod.DGXMonarchFleetKSampler().fleet(
            [spec], [object()], ["NEG"], [{"samples": torch.zeros(1)}],
            [""], [42], [1], [1.0], ["euler"], ["simple"], positive=["POS"])

    if body_fails:
        with pytest.raises(Primary) as raised:
            invoke()
        assert raised.value is primary
        assert any("session close failed" in note for note in primary.__notes__)
    else:
        with pytest.raises(Cleanup, match="fleet session close failed"):
            invoke()


def test_fleet_retries_first_session_cleanup_interruption(monkeypatch):
    class StopNow(BaseException):
        pass

    handle = object.__new__(MeshHandle)
    handle.lock = threading.RLock()
    spec = _fleet_spec(handle, {}, with_lora=False)
    monkeypatch.setattr(fleet_mod, "_enforce_persisted_quarantine", lambda *_args: None)
    monkeypatch.setattr(fleet_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        fleet_mod.DGXMonarchFleetKSampler,
        "_fleet_bound",
        lambda *_args, **_kwargs: ({"samples": torch.zeros(1)},),
    )
    actual_close = fleet_mod._close_session_preserving_primary
    cleanup_calls = []

    def interrupt_first_cleanup(session, primary):
        cleanup_calls.append(True)
        if len(cleanup_calls) == 1:
            raise StopNow("first Fleet session cleanup interrupted")
        actual_close(session, primary)

    monkeypatch.setattr(
        fleet_mod, "_close_session_preserving_primary", interrupt_first_cleanup)

    with pytest.raises(StopNow, match="first Fleet session cleanup interrupted"):
        fleet_mod.DGXMonarchFleetKSampler().fleet(
            [spec], [object()], ["NEG"], [{"samples": torch.zeros(1)}],
            [""], [42], [1], [1.0], ["euler"], ["simple"], positive=["POS"],
        )

    assert cleanup_calls == [True, True]
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_fleet_primary_survives_first_cleanup_helper_interruption(monkeypatch):
    class Primary(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    primary = Primary("Fleet body interrupted")
    handle = object.__new__(MeshHandle)
    handle.lock = threading.RLock()
    spec = _fleet_spec(handle, {}, with_lora=False)
    monkeypatch.setattr(fleet_mod, "_enforce_persisted_quarantine", lambda *_args: None)
    monkeypatch.setattr(fleet_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        fleet_mod.DGXMonarchFleetKSampler,
        "_fleet_bound",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    actual_close = fleet_mod._close_session_preserving_primary
    cleanup_calls = []

    def interrupt_first_cleanup(session, received_primary):
        cleanup_calls.append(received_primary)
        if len(cleanup_calls) == 1:
            raise CleanupStop("first Fleet cleanup helper call interrupted")
        actual_close(session, received_primary)

    monkeypatch.setattr(
        fleet_mod, "_close_session_preserving_primary", interrupt_first_cleanup)

    with pytest.raises(BaseException) as raised:
        fleet_mod.DGXMonarchFleetKSampler().fleet(
            [spec], [object()], ["NEG"], [{"samples": torch.zeros(1)}],
            [""], [42], [1], [1.0], ["euler"], ["simple"], positive=["POS"],
        )

    assert cleanup_calls == [primary, primary]
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()
    assert raised.value is primary


def test_wave_failure_still_collects_and_closes_every_job(monkeypatch):
    class Progress:
        def __init__(self):
            self.closed = 0

        def __exit__(self, *args):
            self.closed += 1

    class Handle:
        def __init__(self):
            self.collected = []

        def collect_one(self, future, timeout_s):
            self.collected.append(future)
            if future == "bad":
                raise RuntimeError("collect failed")
            return {"latent": future, "host": "test", "sample_s": 0.1}

    monkeypatch.setattr("dgx_monarch.nodes.fleet.read_latent_result", lambda value: value)
    progress = [Progress(), Progress(), Progress()]
    pending = [(0, "ok-0", progress[0]), (1, "bad", progress[1]),
               (2, "ok-2", progress[2])]
    results = [None, None, None]
    handle = Handle()
    with pytest.raises(RuntimeError, match="collect failed"):
        _collect_fleet_wave(handle, pending, results, 10.0, 3)
    assert sorted(handle.collected) == ["bad", "ok-0", "ok-2"]
    assert [item.closed for item in progress] == [1, 1, 1]
    assert results == ["ok-0", None, "ok-2"]


def test_wave_collects_by_completion_and_cancels_unfinished_peer(monkeypatch):
    slow_started = threading.Event()
    release_slow = threading.Event()
    events = []

    class Progress:
        def __exit__(self, *_args):
            events.append("close")

    class Handle:
        def collect_one(self, future, timeout_s):
            del timeout_s
            if future == "slow":
                events.append("slow-start")
                slow_started.set()
                assert release_slow.wait(1), "younger failure was hidden by FIFO drain"
                events.append("slow-finish")
                return {"latent": future, "host": "test", "sample_s": 0.1}
            assert slow_started.wait(1)
            events.append("fast-failure")
            raise RuntimeError("younger worker failed")

        def cancel_sample(self, render_id, *, wait):
            events.append(("cancel", render_id, wait))
            release_slow.set()

    monkeypatch.setattr(fleet_mod, "read_latent_result", lambda value: value)
    monkeypatch.setattr(
        fleet_mod, "mark_defunct_on_supervision_failure", lambda *_args: None)
    pending = [
        fleet_mod._FleetPending(0, "slow", Progress(), "render-slow"),
        fleet_mod._FleetPending(1, "fast", Progress(), "render-fast"),
    ]
    results = [None, None]

    with pytest.raises(RuntimeError, match="younger worker failed"):
        _collect_fleet_wave(Handle(), pending, results, 10.0, 2)

    cancel = ("cancel", "render-slow", False)
    assert events.index("fast-failure") < events.index(cancel) < events.index("slow-finish")
    assert results == ["slow", None]
    assert pending == []


@pytest.mark.parametrize(
    "failure", ["no-latent", "read-latent", "invalid-extra", "waiver"])
def test_fast_malformed_result_cancels_blocked_peer_before_drain(
    monkeypatch, failure,
):
    slow_started = threading.Event()
    release_slow = threading.Event()
    events = []

    class Progress:
        def __exit__(self, *_args):
            return None

    class Handle:
        def collect_one(self, future, timeout_s):
            del timeout_s
            if future == "slow":
                slow_started.set()
                assert release_slow.wait(1), "malformed peer was not validated promptly"
                events.append("slow-finish")
                return {"latent": "slow", "host": "test", "sample_s": 0.1}
            assert slow_started.wait(1)
            events.append("fast-result")
            result = {"latent": "bad", "host": "test", "sample_s": 0.1}
            if failure == "no-latent":
                result["latent"] = None
            elif failure == "invalid-extra":
                result["latent_extra"] = []
            return result

        def cancel_sample(self, render_id, *, wait):
            events.append(("cancel", render_id, wait))
            release_slow.set()

    def read_latent(value, *_args):
        if failure == "read-latent" and value == "bad":
            raise RuntimeError("latent materialization failed")
        return value

    original_prepare = consent_waiver_mod.prepare_result_stamps

    def prepare_stamps(rows, source=None, *, strict=False):
        if failure == "waiver" and rows[0].get("latent") == "bad":
            assert strict is True
            raise RuntimeError("waiver validation failed")
        return original_prepare(rows, source, strict=strict)

    monkeypatch.setattr(fleet_mod, "read_latent_result", read_latent)
    monkeypatch.setattr(
        consent_waiver_mod, "prepare_result_stamps", prepare_stamps)
    monkeypatch.setattr(
        fleet_mod, "mark_defunct_on_supervision_failure", lambda *_args: None)
    pending = [
        fleet_mod._FleetPending(0, "slow", Progress(), "render-slow"),
        fleet_mod._FleetPending(1, "fast", Progress(), "render-fast"),
    ]

    with pytest.raises(RuntimeError):
        _collect_fleet_wave(
            Handle(), pending, [None, None], 10.0, 2,
            job_outputs=[None, None], latent_template={}, source_latent={})

    cancel = ("cancel", "render-slow", False)
    assert events.index("fast-result") < events.index(cancel) < events.index("slow-finish")
    assert pending == []


def test_wave_supervision_publication_cannot_replace_collect_failure(monkeypatch):
    primary = RuntimeError("collect failed")

    class Progress:
        def __exit__(self, *_args):
            return None

    class Handle:
        @staticmethod
        def collect_one(_future, timeout_s):
            del timeout_s
            raise primary

    def fail_publication(_handle, _exc):
        raise KeyboardInterrupt("supervision publication interrupted")

    monkeypatch.setattr(
        fleet_mod, "mark_defunct_on_supervision_failure", fail_publication)
    with pytest.raises(RuntimeError) as caught:
        _collect_fleet_wave(
            Handle(), [(0, object(), Progress())], [None], 10.0, 1)

    assert caught.value is primary
    assert any(
        "supervision publication" in note for note in primary.__notes__)


def test_wave_baseexception_retires_every_setup_lease(monkeypatch):
    class StopNow(BaseException):
        pass

    class Progress:
        def __init__(self):
            self.closed = 0

        def __exit__(self, *_args):
            self.closed += 1

    class Handle:
        def __init__(self):
            self.lock = threading.RLock()
            self.sample_leases = {7: 2}
            self.abandoned_sample_leases = {}
            self.deferred_supervision_error = None

        def collect_one(self, future, timeout_s):
            if future.future == "stop":
                raise StopNow("comfy stop")
            return {"latent": "wire", "host": "test", "sample_s": 0.1}

    handle = Handle()
    first = mesh_setup.SetupBoundFuture("stop", handle, 7)
    second = mesh_setup.SetupBoundFuture("ok", handle, 7)
    progress = [Progress(), Progress()]
    results = [None, None]
    monkeypatch.setattr(
        fleet_mod, "read_latent_result", lambda value, _guard=None: value)

    with pytest.raises(StopNow, match="comfy stop"):
        _collect_fleet_wave(
            handle,
            [(0, first, progress[0]), (1, second, progress[1])],
            results, 10.0, 2,
        )

    assert first.state == "abandoned" and second.state == "consumed"
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {7: 1}
    assert [item.closed for item in progress] == [1, 1]
    assert results == [None, "wire"]


@pytest.mark.parametrize("consumed", [False, True])
def test_fleet_retirement_retries_wrapper_call_boundary(monkeypatch, consumed):
    class StopNow(BaseException):
        pass

    interruption = StopNow("fleet retirement wrapper call interrupted")
    handle = SimpleNamespace(
        lock=threading.RLock(), sample_leases={3: 1},
        abandoned_sample_leases={}, deferred_supervision_error=None)
    future = mesh_setup.SetupBoundFuture("wire", handle, 3)
    name = "release_sample" if consumed else "abandon_sample"
    actual = getattr(mesh_setup, name)
    calls = []

    def interrupt_first(received):
        calls.append(received)
        if len(calls) == 1:
            raise interruption
        actual(received)

    monkeypatch.setattr(mesh_setup, name, interrupt_first)

    assert fleet_mod._retire_fleet_future(
        future, consumed=consumed) is interruption
    assert calls == [future, future]
    assert future.state == ("consumed" if consumed else "abandoned")
    assert handle.sample_leases == {}


@pytest.mark.parametrize("explosive", [False, True])
def test_fleet_retirement_preserves_exact_first_baseexception(
    monkeypatch, explosive
):
    class FirstFailure(BaseException):
        def __bool__(self):
            if explosive:
                raise AssertionError("exception truthiness must not be evaluated")
            return False

    first = FirstFailure("first wrapper failure")
    wrapper_errors = iter((first, BaseException("second wrapper failure")))
    direct_errors = iter((
        BaseException("first direct failure"),
        BaseException("second direct failure"),
    ))
    handle = SimpleNamespace(
        lock=threading.RLock(), sample_leases={3: 1},
        abandoned_sample_leases={}, deferred_supervision_error=None)
    future = mesh_setup.SetupBoundFuture("wire", handle, 3)

    def fail_wrapper(_future):
        raise next(wrapper_errors)

    def fail_direct():
        raise next(direct_errors)

    monkeypatch.setattr(mesh_setup, "abandon_sample", fail_wrapper)
    monkeypatch.setattr(future, "abandon", fail_direct)

    assert fleet_mod._retire_fleet_future(future, consumed=False) is first


@pytest.mark.parametrize("explosive", [False, True])
def test_fleet_progress_close_preserves_exact_first_baseexception(explosive):
    class FirstFailure(BaseException):
        def __bool__(self):
            if explosive:
                raise AssertionError("exception truthiness must not be evaluated")
            return False

    first = FirstFailure("first progress failure")

    class Progress:
        def __init__(self):
            self.calls = 0

        def __exit__(self, *_args):
            self.calls += 1
            if self.calls == 1:
                raise first
            raise BaseException("second progress failure")

    assert fleet_mod._close_fleet_progress(Progress()) is first


def test_request_construction_baseexception_drains_prior_jobs(monkeypatch):
    class StopNow(BaseException):
        pass

    class Handle:
        world = 2

        def __init__(self):
            self.collected = []
            self.submitted = []
            self.cancelled = []
            self.events = []

        def ensure_setup(self, *_args, **_kwargs):
            return None

        def submit_sample_to(self, index, request, **_kwargs):
            future = f"future-{index}"
            self.submitted.append(future)
            self.events.append(("submit", request["_dgxm_render_id"]))
            return future

        def collect_one(self, future, timeout_s):
            self.collected.append(future)
            self.events.append(("collect", future))
            return {"latent": future, "host": "test", "sample_s": 0.1}

        def cancel_sample(self, render_id, *, wait):
            self.cancelled.append((render_id, wait))
            self.events.append(("cancel", render_id))

    class Progress:
        instances = []

        def __init__(self, *_args, **_kwargs):
            self.port = 1234
            self.closed = 0
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed += 1

    handle = Handle()
    mesh = MeshSpec(
        handle=handle, topology_preset="uly2", attention="TORCH_FLASH",
        sync_ulysses=True, worker_args={})
    spec = ModelSpec(mesh=mesh, unet_name="model.safetensors")
    pack_calls = []

    def pack(value):
        pack_calls.append(value)
        if len(pack_calls) == 2:
            raise StopNow("request build stopped")
        return value

    monkeypatch.setattr(fleet_mod, "_enforce_persisted_quarantine", lambda *_args: None)
    monkeypatch.setattr(fleet_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        fleet_mod, "_fleet_worker_policy",
        lambda *_args: fleet_mod._FleetAuthorization({}, {}, None))
    monkeypatch.setattr(fleet_mod, "pack_latent", pack)
    monkeypatch.setattr(fleet_mod, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(fleet_mod, "ProgressReceiver", Progress)
    monkeypatch.setattr(
        fleet_mod, "read_latent_result", lambda value, _guard=None: value)

    with pytest.raises(StopNow, match="request build stopped"):
        fleet_mod.DGXMonarchFleetKSampler().fleet(
            [spec], [object()], ["NEG"], [{"samples": torch.zeros(1)}],
            [""], [42], [2], [1.0], ["euler"], ["simple"],
            positive=["POS-0", "POS-1"],
        )

    assert handle.submitted == ["future-0"]
    assert handle.collected == ["future-0"]
    assert handle.cancelled == [(handle.events[0][1], False)]
    assert [event[0] for event in handle.events] == ["submit", "cancel", "collect"]
    assert len(Progress.instances) == 1 and Progress.instances[0].closed == 1


def test_wave_handoff_baseexception_after_append_drains_published_job(monkeypatch):
    class StopNow(BaseException):
        pass

    class Handle:
        world = 1

        def __init__(self):
            self.submitted = []
            self.collected = []

        def ensure_setup(self, *_args, **_kwargs):
            return None

        def submit_sample_to(self, _index, _request, **_kwargs):
            future = object()
            self.submitted.append(future)
            return future

        def collect_one(self, future, timeout_s):
            self.collected.append(future)
            return {"latent": "wire", "host": "test", "sample_s": 0.1}

        def cancel_sample(self, *_args, **_kwargs):
            return None

    class Progress:
        instances = []
        port = 1234

        def __init__(self, *_args, **_kwargs):
            self.closed = 0
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed += 1

    handle = Handle()
    spec = ModelSpec(
        mesh=MeshSpec(
            handle=handle, topology_preset="uly1", attention="TORCH_FLASH",
            sync_ulysses=False, worker_args={}),
        unet_name="model.safetensors",
    )
    monkeypatch.setattr(fleet_mod, "_enforce_persisted_quarantine", lambda *_args: None)
    monkeypatch.setattr(fleet_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        fleet_mod, "_fleet_worker_policy",
        lambda *_args: fleet_mod._FleetAuthorization({}, {}, None))
    monkeypatch.setattr(fleet_mod, "ProgressReceiver", Progress)
    monkeypatch.setattr(fleet_mod, "pack_latent", lambda value: value)
    monkeypatch.setattr(fleet_mod, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(
        fleet_mod, "read_latent_result", lambda _value: torch.zeros(1, 1, 1, 1))
    real_collect = fleet_mod._collect_fleet_wave
    calls = []

    def interrupt_once(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise StopNow("wave handoff interrupted after pending append")
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(fleet_mod, "_collect_fleet_wave", interrupt_once)

    with pytest.raises(StopNow):
        fleet_mod.DGXMonarchFleetKSampler().fleet(
            [spec], [object()], ["NEG"], [{"samples": torch.zeros(1)}],
            [""], [42], [2], [1.0], ["euler"], ["simple"], positive=["POS"],
        )

    assert calls == [True, True]
    assert handle.collected == handle.submitted
    assert len(Progress.instances) == 1 and Progress.instances[0].closed == 1


@pytest.mark.parametrize("state", ["unknown", "stale", "inconclusive", "fail"])
def test_fleet_forces_stock_policy_until_exact_context_pass(monkeypatch, state):
    lookups = []
    identity = _artifact_identity("model-sig", "lora-sig")

    class Ledger:
        def __init__(self, directory):
            assert directory == "/ledger"

        def lookup_with_integrity(self, *args):
            lookups.append(args)
            return ledger_mod.GateLedgerLookup(state, None, True)

    handle = SimpleNamespace(effective_worker_args=lambda args: dict(args))
    requested = {"slab_weights": "auto", "lora_low_rss": True}
    spec = _fleet_spec(handle, requested)
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: "/ledger")
    monkeypatch.setattr(ledger_mod, "GateLedger", Ledger)
    monkeypatch.setattr(
        model_store_mod, "request_artifact_identity", lambda *_args: identity)

    authorization = _fleet_worker_policy(spec, handle)

    assert authorization.worker_args == {
        "slab_weights": False, "lora_low_rss": False}
    assert authorization.residency_grant is None
    assert requested == {"slab_weights": "auto", "lora_low_rss": True}
    assert len(lookups) == 1
    key, artifacts, commit, context = lookups[0]
    assert key == mesh_safety_mod.request_combo_key(authorization.model_request)
    assert artifacts.current == identity["digest"] and commit == "commit"
    assert context["capability"] == "fleet_per_rank_residency"
    assert context["rank_world"] == 1
    assert context["worker_args"] == requested
    assert not ({"topology_preset", "attention", "sync_ulysses"} & set(context))


@pytest.mark.parametrize(
    ("session_pass_safe", "expect_grant"),
    [(True, True), (False, False)],
)
def test_fleet_cached_pass_cannot_bridge_unhealed_ledger_damage(
    monkeypatch, session_pass_safe, expect_grant,
):
    from dgx_monarch.nodes import fleet_policy as fleet_policy_mod

    identity = _artifact_identity("model-sig", "lora-sig")

    class Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, *_args):
            return ledger_mod.GateLedgerLookup(
                "unknown", None, session_pass_safe)

    handle = SimpleNamespace(effective_worker_args=lambda args: dict(args))
    requested = {"slab_weights": "auto", "lora_low_rss": True}
    spec = _fleet_spec(handle, requested)
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: "/ledger")
    monkeypatch.setattr(ledger_mod, "GateLedger", Ledger)
    monkeypatch.setattr(
        model_store_mod, "request_artifact_identity", lambda *_args: identity)
    monkeypatch.setattr(
        fleet_policy_mod, "_process_gate_verdict", lambda _token: "PASS")

    authorization = _fleet_worker_policy(spec, handle)

    assert (authorization.residency_grant is not None) is expect_grant
    assert authorization.worker_args == (
        requested if expect_grant else {
            "slab_weights": False, "lora_low_rss": False}
    )


def test_fleet_preserves_requested_policy_only_for_exact_context_pass(monkeypatch):
    identity = _artifact_identity("model-sig", "lora-sig")
    contexts = []

    class Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, key, artifacts, commit, actual_context):
            assert key == mesh_safety_mod.request_combo_key(spec.request_dict())
            assert artifacts.current == identity["digest"] and commit == "commit"
            contexts.append(actual_context)
            return ledger_mod.GateLedgerLookup("pass", None, True)

    requested = {"slab_weights": True, "lora_low_rss": True}
    options = {"weight_dtype": "default", "nested": {"scales": [1.0]}}
    handle = SimpleNamespace(
        effective_worker_args=lambda args: {"reserve_vram_gb": 8.0, **dict(args)},
        config=SimpleNamespace(hosts=(object(), object()), source="/cluster.toml"),
        config_fingerprint="config-a", world=4, n_hosts=2, gpus_per_host=2,
    )
    spec = _fleet_spec(handle, requested, options=options)
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: "/ledger")
    monkeypatch.setattr(ledger_mod, "GateLedger", Ledger)
    monkeypatch.setattr(
        model_store_mod, "request_artifact_identity", lambda *_args: identity)

    authorization = _fleet_worker_policy(spec, handle)
    requested["slab_weights"] = False
    options["weight_dtype"] = "fp8_e4m3fn"
    options["nested"]["scales"].append(2.0)
    spec.loras[0]["strength"] = 0.25

    assert authorization.worker_args == {
        "slab_weights": True, "lora_low_rss": True}
    assert authorization.model_request["options"] == {
        "weight_dtype": "default", "nested": {"scales": [1.0]}}
    assert authorization.model_request["loras"][0]["strength"] == 1.0
    combo = mesh_safety_mod.request_combo_key(authorization.model_request)
    assert authorization.residency_grant == {
        "gate_protocol": ledger_mod.GATE_PROTOCOL_VERSION,
        "dgx_monarch": ledger_mod.__version__,
        "comfy": identity["comfy"],
        "artifact_digest": identity["digest"],
        "gate_token": list(ledger_mod.gate_verdict_token(
            combo, identity["digest"], identity["comfy"], contexts[0])),
        "combo_key": combo,
        "model_request": authorization.model_request,
        "uncond_model_request": None,
        "artifact_sets": [identity],
        "capability_context": contexts[0],
    }
    grant_model = authorization.residency_grant["model_request"]
    assert grant_model is not authorization.model_request
    assert grant_model["options"] is not authorization.model_request["options"]
    authorization.model_request["options"]["nested"]["scales"].append(3.0)
    assert grant_model["options"]["nested"]["scales"] == [1.0]
    assert contexts[0] == {
        "capability": "fleet_per_rank_residency",
        "rank_world": 1,
        "worker_args": {"reserve_vram_gb": 8.0, "slab_weights": True,
                        "lora_low_rss": True},
        "mesh_mode": "cluster",
        "config_source": "/cluster.toml",
        "config_fingerprint": "config-a",
        "physical_world": 4,
        "hosts": 2,
        "gpus_per_host": 2,
    }


def test_fleet_omitted_slab_default_and_lookup_error_both_fail_closed(monkeypatch):
    class Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, *_args):
            raise OSError("ledger unavailable")

    requested = {}
    handle = SimpleNamespace(effective_worker_args=lambda args: dict(args))
    spec = _fleet_spec(handle, requested, with_lora=False)
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: "/ledger")
    monkeypatch.setattr(ledger_mod, "GateLedger", Ledger)
    monkeypatch.setattr(
        model_store_mod, "request_artifact_identity",
        lambda *_args: _artifact_identity("model-sig"),
    )

    assert _fleet_worker_policy(spec, handle).worker_args == {
        "slab_weights": False,
        "lora_low_rss": False,
    }
    assert requested == {}


@pytest.mark.parametrize(
    ("ledger_state", "expected_policy", "expect_grant"),
    [
        ("stale", {"slab_weights": False, "lora_low_rss": False}, False),
        ("pass", {"slab_weights": "auto", "lora_low_rss": True}, True),
    ],
)
def test_fleet_dispatch_binds_authorized_policy_before_setup_and_submit(
        monkeypatch, ledger_state, expected_policy, expect_grant):
    setup_policies = []
    submitted = []

    class Handle:
        world = 1

        def effective_worker_args(self, args):
            return dict(args)

        def ensure_setup(self, _topology, _attention, _sync, worker_args, fleet=False):
            assert fleet is True
            setup_policies.append(dict(worker_args))

        def submit_sample_to(self, _index, request, **_kwargs):
            submitted.append(request)
            return object()

        def collect_one(self, _future, timeout_s):
            assert timeout_s >= 900
            return {"latent": "wire", "host": "test", "sample_s": 0.1}

    class Progress:
        port = 1234

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, *_args):
            return ledger_mod.GateLedgerLookup(ledger_state, None, True)

    handle = Handle()
    mesh = MeshSpec(
        handle=handle,
        topology_preset="uly2",
        attention="TORCH_FLASH",
        sync_ulysses=True,
        worker_args={"slab_weights": "auto", "lora_low_rss": True},
    )
    spec = ModelSpec(
        mesh=mesh,
        unet_name="model.safetensors",
        loras=({"name": "style.safetensors", "strength": 1.0},),
    )
    monkeypatch.setattr(fleet_mod, "_enforce_persisted_quarantine", lambda *_args: None)
    monkeypatch.setattr(fleet_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(fleet_mod, "ProgressReceiver", Progress)
    monkeypatch.setattr(fleet_mod, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(fleet_mod, "pack_latent", lambda value: value)
    monkeypatch.setattr(
        fleet_mod, "read_latent_result", lambda _value: torch.zeros(1, 1, 1, 1))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: "/ledger")
    monkeypatch.setattr(ledger_mod, "GateLedger", Ledger)
    identity = _artifact_identity("model-sig", "lora-sig")
    monkeypatch.setattr(
        model_store_mod, "request_artifact_identity", lambda *_args: identity)

    result, = fleet_mod.DGXMonarchFleetKSampler().fleet(
        [spec], [object()], ["NEG"], [{
            "samples": torch.zeros(1, 1, 1, 1),
            "_dgxm_latent_downscale": 16,
            "downscale_ratio_spacial": 8,
            "batch_index": [0],
        }],
        [""], [42], [2], [1.0], ["euler"], ["simple"], positive=["POS"],
    )

    assert setup_policies == [expected_policy]
    assert mesh.worker_args == {"slab_weights": "auto", "lora_low_rss": True}
    assert tuple(result["samples"].shape) == (1, 1, 1, 1)
    assert result["batch_index"] == [0]
    assert "_dgxm_latent_downscale" not in result
    assert submitted[0]["_dgxm_fleet_job"] is True
    assert "_dgxm_latent_downscale" not in submitted[0]["latent"]
    # The worker's empty-latent fix reads the size tag; like stock, the output drops it.
    assert submitted[0]["latent"]["downscale_ratio_spacial"] == 8
    assert "downscale_ratio_spacial" not in result
    grant = submitted[0].get("_dgxm_fleet_residency_grant")
    assert (grant is not None) is expect_grant
    if grant is not None:
        assert grant["artifact_sets"] == [identity]
        assert grant["combo_key"] == mesh_safety_mod.request_combo_key(
            submitted[0]["model"])


class NestedTensor:
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


@pytest.fixture
def direct_nested_tensor_surface(monkeypatch):
    comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
    nested = types.ModuleType("comfy.nested_tensor")
    nested.NestedTensor = NestedTensor
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested)
    monkeypatch.setattr(comfy, "nested_tensor", nested, raising=False)


def _run_fleet_result(
    monkeypatch, outputs, *, latent=None, worker_rows=None, source_latent=None,
    on_submit=None, public=False, trap_side_effects=False, prompts="", clip=None,
):
    class _Host:
        def __init__(self, name):
            self.name = name

    class _Config:
        hosts = [_Host("worker-a"), _Host("worker-b")]

    class Handle:
        world = 2
        # One GPU per box: rank equals box index, and the box labels the
        # driver publishes read box 1 and box 2.
        gpus_per_host = 1
        # The driver's name for each box, which a row's answering host is
        # checked against.
        config = _Config()

        def submit_sample_to(self, _index, request, **_kwargs):
            if on_submit is not None:
                on_submit(request)
            return request["render_seq"]

        def collect_one(self, future, timeout_s):
            assert timeout_s >= 900
            return {
                "latent": future,
                "host": "test",
                "sample_s": 0.1,
                **({} if worker_rows is None else worker_rows[future]),
            }

        def cancel_sample(self, _render_id):
            return None

    class Progress:
        port = 1234

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class Session:
        def bind(self, _handle):
            return None

        def activate(self):
            return nullcontext()

        def close(self):
            return None

        def track(self, _future):
            return None

    handle = Handle()
    spec = _fleet_spec(handle, {}, with_lora=False)
    jobs = [
        {"positive": f"POS-{index}", "negative": "NEG", "seed": 42 + index}
        for index in range(len(outputs))
    ]
    monkeypatch.setattr(
        fleet_mod,
        "_fleet_worker_policy",
        lambda *_args: fleet_mod._FleetAuthorization({}, spec.request_dict(), None),
    )
    monkeypatch.setattr(fleet_mod, "ProgressReceiver", Progress)
    monkeypatch.setattr(fleet_mod, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(fleet_mod, "pack_latent", lambda value: value)
    monkeypatch.setattr(
        fleet_mod, "read_latent_result", lambda index, *_args: outputs[index]
    )
    monkeypatch.setattr(mesh_setup, "ensure_request_setup", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mesh_setup, "prepare_sample", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mesh_setup, "token_kwargs", lambda _token: {})
    if public:
        if trap_side_effects:
            def unexpected(*_args, **_kwargs):
                raise AssertionError("fleet provenance validation ran too late")

            monkeypatch.setattr(fleet_mod, "RenderSession", unexpected)
            monkeypatch.setattr(fleet_mod, "ensure_live", unexpected)
            monkeypatch.setattr(fleet_mod, "_enforce_persisted_quarantine", unexpected)
            monkeypatch.setattr(
                fleet_mod.DGXMonarchFleetKSampler, "_fleet_bound", unexpected)
        else:
            monkeypatch.setattr(fleet_mod, "RenderSession", Session)
            monkeypatch.setattr(fleet_mod, "ensure_live", lambda value: value)
            monkeypatch.setattr(
                fleet_mod, "_enforce_persisted_quarantine", lambda *_args: None)
        result, = fleet_mod.DGXMonarchFleetKSampler().fleet(
            [spec], [object() if clip is None else clip], ["NEG"], [
                {"samples": torch.zeros(1, 1)} if latent is None else latent
            ], [prompts], [42], [1], [1.0], ["euler"], ["simple"],
            positive=["POS"],
        )
        return result
    result, = fleet_mod.DGXMonarchFleetKSampler()._fleet_bound(
        spec,
        {"samples": torch.zeros(1, 1)} if latent is None else latent,
        jobs,
        1,
        1.0,
        "euler",
        "simple",
        handle,
        Session(),
        source_latent=source_latent,
    )
    return result


def _run_fleet_outputs(monkeypatch, outputs):
    return _run_fleet_result(monkeypatch, outputs)["samples"]


def test_fleet_aggregates_batch_index_and_all_waiver_provenance(monkeypatch, tmp_path):
    stamp_key = accuracy_waiver.STAMPED_RESULT_KEY
    worker_key = accuracy_waiver.RESULT_KEY
    kind = accuracy_waiver.KNOWN_WRONG_GUARDS["ring_pad"]
    submitted = []
    retired = []
    monkeypatch.setattr(consent_waiver_mod, "_PENDING_AUDIT", {})
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.base_path = str(tmp_path)
    folder_paths.get_full_path = lambda *_args: None
    folder_paths.get_output_directory = lambda: str(output_dir)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    ledger = ledger_mod.GateLedger(str(output_dir))

    def use_rows():
        return [row for row in ledger.entries() if row.get("action") == "use"]

    original_retire = consent_waiver_mod.retire_audit

    def retire_audit(render_id):
        retired.append((str(render_id), len(use_rows())))
        original_retire(render_id)

    monkeypatch.setattr(consent_waiver_mod, "retire_audit", retire_audit)
    outputs = [
        torch.full((2, 1), float(index))
        for index in range(2)
    ]
    worker_rows = [
        {
            "latent_extra": {"batch_index": [4, 9]},
            worker_key: [
                {
                    "run_id": "filled from submitted request",
                    "guard": "ring_pad",
                    "kind": kind,
                    "consent_id": f"{index + 1:032x}",
                    "consent_source": "panel",
                    "stamp": accuracy_waiver.stamp_text(kind),
                },
            ]
        }
        for index in range(2)
    ]

    def on_submit(request):
        index = request["render_seq"]
        run_id = request["_dgxm_render_id"]
        submitted.append(run_id)
        worker_rows[index][worker_key][0]["run_id"] = run_id
        consent_waiver_mod._PENDING_AUDIT[run_id] = {
            "combo_key": "combo",
            "artifacts": "a" * 64,
            "memo_context": {"combo_key": "combo", "topology": "single", "world": "1"},
            "capability_context": {"world": 1},
            "unet_name": "model.safetensors",
            "file_identity": "identity",
            "loras": 0,
        }

    combined = _run_fleet_result(
        monkeypatch,
        outputs,
        latent={"samples": torch.zeros(2, 1), "batch_index": [4, 9]},
        worker_rows=worker_rows,
        on_submit=on_submit,
    )

    assert tuple(combined["samples"].shape) == (4, 1)
    assert combined["batch_index"] == [4, 9, 4, 9]
    assert [entry["run_id"] for entry in combined[stamp_key]] == submitted
    rows = use_rows()
    assert [entry["run_id"] for entry in rows] == submitted
    assert [entry["target_guard"] for entry in rows] == ["ring_pad", "ring_pad"]
    assert retired == [(submitted[0], 1), (submitted[1], 2)]
    assert consent_waiver_mod._PENDING_AUDIT == {}


@pytest.mark.parametrize(
    "malformed",
    [
        {},
        "",
        [{"guard": "ring_pad"}],
        [{"run_id": 7, "guard": "ring_pad"}],
    ],
)
def test_fleet_rejects_malformed_aggregated_waiver_provenance(monkeypatch, malformed):
    stamp_key = accuracy_waiver.STAMPED_RESULT_KEY
    with pytest.raises(RuntimeError, match="waiver provenance"):
        _run_fleet_result(
            monkeypatch,
            [torch.zeros(1, 1)],
            latent={"samples": torch.zeros(1, 1), stamp_key: malformed},
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
def test_fleet_public_entry_rejects_malformed_inherited_provenance(
    monkeypatch, malformed,
):
    with pytest.raises(RuntimeError, match="inherited waiver provenance"):
        _run_fleet_result(
            monkeypatch,
            [torch.zeros(1, 1)],
            latent={
                "samples": torch.zeros(1, 1),
                accuracy_waiver.STAMPED_RESULT_KEY: malformed,
            },
            public=True,
            trap_side_effects=True,
        )


def test_fleet_public_entry_preserves_valid_inherited_provenance(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        consent_waiver_mod, "_record_use_row",
        lambda entry: recorded.append(dict(entry)),
    )
    inherited = {"run_id": "parent-run", "guard": "ring_pad", "stamp": "ancestor"}
    result = _run_fleet_result(
        monkeypatch,
        [torch.zeros(1, 1)],
        latent={
            "samples": torch.zeros(1, 1),
            accuracy_waiver.STAMPED_RESULT_KEY: [inherited, dict(inherited)],
        },
        public=True,
    )

    assert result[accuracy_waiver.STAMPED_RESULT_KEY] == [
        {**inherited, "inherited": True}
    ]
    assert recorded == []


def test_fleet_public_entry_keeps_absent_provenance_absent(monkeypatch):
    result = _run_fleet_result(
        monkeypatch, [torch.zeros(1, 1)], public=True)

    assert accuracy_waiver.STAMPED_RESULT_KEY not in result


def test_fleet_deduplicates_legitimate_inherited_source_provenance(monkeypatch):
    stamp_key = accuracy_waiver.STAMPED_RESULT_KEY
    recorded = []
    monkeypatch.setattr(
        consent_waiver_mod,
        "_record_use_row",
        lambda entry: recorded.append(dict(entry)),
    )
    inherited = {
        "run_id": "parent-run",
        "guard": "ring_pad",
        "stamp": "ancestor",
    }

    combined = _run_fleet_result(
        monkeypatch,
        [torch.zeros(1, 1), torch.ones(1, 1)],
        source_latent={"samples": torch.zeros(1, 1), stamp_key: [inherited]},
    )

    assert combined[stamp_key] == [{**inherited, "inherited": True}]
    assert recorded == []


def test_fleet_keeps_missing_batch_index_absent(monkeypatch):
    combined = _run_fleet_result(
        monkeypatch,
        [torch.full((2, 1), float(index)) for index in range(2)],
        latent={"samples": torch.zeros(2, 1)},
    )

    assert tuple(combined["samples"].shape) == (4, 1)
    assert "batch_index" not in combined


def test_fleet_repeats_explicit_batch_one_index_per_render(monkeypatch):
    combined = _run_fleet_result(
        monkeypatch,
        [torch.full((1, 1), float(index)) for index in range(3)],
        latent={"samples": torch.zeros(1, 1), "batch_index": [7]},
    )

    assert tuple(combined["samples"].shape) == (3, 1)
    assert combined["batch_index"] == [7, 7, 7]


def test_fleet_rejects_mixed_worker_batch_index_presence(monkeypatch):
    with pytest.raises(RuntimeError, match="inconsistently returned batch_index"):
        _run_fleet_result(
            monkeypatch,
            [torch.zeros(1, 1), torch.zeros(1, 1)],
            latent={"samples": torch.zeros(1, 1)},
            worker_rows=[{"latent_extra": {"batch_index": [0]}}, {}],
        )


def test_fleet_rejects_worker_batch_index_length_mismatch(monkeypatch):
    with pytest.raises(RuntimeError, match="length 1 does not match samples batch 2"):
        _run_fleet_result(
            monkeypatch,
            [torch.zeros(2, 1), torch.zeros(2, 1)],
            latent={"samples": torch.zeros(2, 1)},
            worker_rows=[
                {"latent_extra": {"batch_index": [0]}},
                {"latent_extra": {"batch_index": [0, 1]}},
            ],
        )


@pytest.mark.parametrize("batch_index", [[True], [0.0], ["x"], [-1]])
def test_fleet_rejects_invalid_worker_batch_index_entry(monkeypatch, batch_index):
    with pytest.raises(RuntimeError, match="nonnegative integers"):
        _run_fleet_result(
            monkeypatch,
            [torch.zeros(1, 1)],
            worker_rows=[{"latent_extra": {"batch_index": batch_index}}],
        )


def test_fleet_concatenates_packed_results_modality_wise(
    monkeypatch, direct_nested_tensor_surface,
):
    outputs = [
        NestedTensor((
            torch.full((1, 2, 2), float(index)),
            torch.full((1, 3), float(index + 10)),
        ))
        for index in range(2)
    ]

    combined = _run_fleet_outputs(monkeypatch, outputs)

    assert type(combined) is NestedTensor
    video, audio = combined.unbind()
    assert tuple(video.shape) == (2, 2, 2)
    assert tuple(audio.shape) == (2, 3)
    assert video[:, 0, 0].tolist() == [0.0, 1.0]
    assert audio[:, 0].tolist() == [10.0, 11.0]


@pytest.mark.parametrize(
    ("second", "match"),
    [
        (torch.zeros(1, 2, 2), "mixed Tensor and NestedTensor"),
        (NestedTensor((torch.zeros(1, 2, 2),)), "modality count"),
        (
            NestedTensor((torch.zeros(1, 2, 3), torch.zeros(1, 3))),
            "changed structure for packed modality 0",
        ),
        (
            NestedTensor((
                torch.zeros(1, 2, 2, dtype=torch.float64),
                torch.zeros(1, 3, dtype=torch.float64),
            )),
            "changed structure for packed modality 0",
        ),
    ],
)
def test_fleet_rejects_packed_result_structure_drift(
    monkeypatch, direct_nested_tensor_surface, second, match,
):
    first = NestedTensor((torch.zeros(1, 2, 2), torch.zeros(1, 3)))
    with pytest.raises(RuntimeError, match=match):
        _run_fleet_outputs(monkeypatch, [first, second])


def test_a_refused_fleet_job_consumes_its_lease_and_leaves_the_wave_dispatchable(
        monkeypatch):
    """One job's typed refusal must not force a recycle of the whole fleet.

    A fleet job is a world-1 render on its own actor with its own lease. A
    tagged refusal from that actor means the render is over with no latent
    packed and no result backing left, so the lease is consumed. An abandoned
    lease blocks every later sample on this handle until a reset, including the
    re-queue the refusal asks the operator for.
    """
    from monarch.actor import ActorError

    from dgx_monarch.refusal import PanelAction, RefusalClass, refusal

    refused = ActorError(RuntimeError(refusal(
        RefusalClass.KNOWN_WRONG, "padded ring on a maskless kernel.",
        guard="ring_pad", waivable=True,
        panel_action=PanelAction("Render under waiver (output stamped)",
                                 "DGXM_WAIVE_KNOWN_WRONG", env_value="ring_pad"),
        troubleshooting=21)))

    class Progress:
        def __exit__(self, *_args):
            return None

    class Handle:
        def __init__(self):
            self.lock = threading.RLock()
            self.sample_leases = {7: 2}
            self.abandoned_sample_leases = {}
            self.deferred_supervision_error = None

        def collect_one(self, future, timeout_s):
            if future.future == "refused":
                raise refused
            return {"latent": "wire", "host": "test", "sample_s": 0.1}

    handle = Handle()
    first = mesh_setup.SetupBoundFuture("refused", handle, 7)
    second = mesh_setup.SetupBoundFuture("ok", handle, 7)
    results = [None, None]
    monkeypatch.setattr(
        fleet_mod, "read_latent_result", lambda value, _guard=None: value)

    with pytest.raises(ActorError):
        _collect_fleet_wave(
            handle, [(0, first, Progress()), (1, second, Progress())],
            results, 10.0, 2)

    assert first.state == "consumed" and second.state == "consumed"
    assert handle.abandoned_sample_leases == {}
    assert results == [None, "wire"]


def test_a_middle_tag_in_a_fleet_failure_abandons_its_lease(monkeypatch):
    """A tag inside a crash message is not a typed refusal: the lease is abandoned."""
    from monarch.actor import ActorError

    crash = ActorError(RuntimeError(
        "CUDA error while loading models/[dgxm:P]checkpoint.safetensors"))

    class Progress:
        def __exit__(self, *_args):
            return None

    class Handle:
        def __init__(self):
            self.lock = threading.RLock()
            self.sample_leases = {7: 2}
            self.abandoned_sample_leases = {}
            self.deferred_supervision_error = None

        def collect_one(self, future, timeout_s):
            if future.future == "crash":
                raise crash
            return {"latent": "wire", "host": "test", "sample_s": 0.1}

    handle = Handle()
    first = mesh_setup.SetupBoundFuture("crash", handle, 7)
    second = mesh_setup.SetupBoundFuture("ok", handle, 7)
    results = [None, None]
    monkeypatch.setattr(
        fleet_mod, "read_latent_result", lambda value, _guard=None: value)

    with pytest.raises(ActorError):
        _collect_fleet_wave(
            handle, [(0, first, Progress()), (1, second, Progress())],
            results, 10.0, 2)

    assert first.state == "abandoned" and second.state == "consumed"
    assert handle.abandoned_sample_leases == {7: 1}
    assert results == [None, "wire"]


def test_fleet_source_manifest_failure_forces_stock_without_a_grant(monkeypatch):
    class Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, *_args):
            return ledger_mod.GateLedgerLookup("pass", {"verdict": "PASS"}, True)

    requested = {"slab_weights": True, "lora_low_rss": True}
    handle = SimpleNamespace(effective_worker_args=lambda args: dict(args))
    spec = _fleet_spec(handle, requested)
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: "/ledger")
    monkeypatch.setattr(ledger_mod, "GateLedger", Ledger)
    monkeypatch.setattr(
        model_store_mod, "request_artifact_identity",
        lambda *_args: _artifact_identity("model-sig", "lora-sig"),
    )
    monkeypatch.setattr(
        ledger_mod.runtime_provenance,
        "cached_dgx_source_manifest_sha256",
        lambda: (_ for _ in ()).throw(RuntimeError("manifest unavailable")),
    )

    authorization = _fleet_worker_policy(spec, handle)

    assert authorization.worker_args == {
        "slab_weights": False, "lora_low_rss": False}
    assert authorization.residency_grant is None
    assert requested == {"slab_weights": True, "lora_low_rss": True}


class _StubClip:
    """Enough CLIP for the public entry: a prompt line in, conditioning out."""

    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return f"COND:{tokens}"


def _fresh_fleet_ledger(monkeypatch):
    ledger = telemetry_fleet.FleetJobs()
    monkeypatch.setattr(telemetry_fleet, "fleet_jobs", ledger)
    return ledger


def test_fleet_publishes_the_host_rank_and_box_of_every_job(monkeypatch):
    # Without a row per job the sidebar timeline draws one row for the whole node.
    ledger = _fresh_fleet_ledger(monkeypatch)
    _run_fleet_result(
        monkeypatch,
        [torch.zeros(1, 1), torch.zeros(1, 1)],
        worker_rows=[
            {"host": "worker-a", "rank": 0, "sample_s": 1.25},
            {"host": "worker-b", "rank": 1, "sample_s": 2.5},
        ],
    )

    block = ledger.snapshot()
    assert block["count"] == 2
    assert block["truncated"] is False
    assert [row["job"] for row in block["jobs"]] == [0, 1]
    assert [row["host"] for row in block["jobs"]] == ["worker-a", "worker-b"]
    assert [row["rank"] for row in block["jobs"]] == [0, 1]
    assert [row["box"] for row in block["jobs"]] == ["box 1", "box 2"]
    assert [row["wall_s"] for row in block["jobs"]] == [1.25, 2.5]
    # A conditioning-list job carries no prompt line to stand for.
    assert [row["prompt"] for row in block["jobs"]] == [None, None]
    assert all("host_reported" not in row for row in block["jobs"])


def test_every_dispatched_job_publishes_the_span_the_driver_timed(monkeypatch):
    # A row that names its box but not its span cannot place the job on the
    # sidebar timeline. The node opens each row when it submits the job, and
    # the answer closes it.
    ledger = _fresh_fleet_ledger(monkeypatch)
    _run_fleet_result(
        monkeypatch,
        [torch.zeros(1, 1), torch.zeros(1, 1)],
        worker_rows=[
            {"host": "worker-a", "sample_s": 1.25},
            {"host": "worker-b", "sample_s": 2.5},
        ],
    )

    rows = ledger.snapshot()["jobs"]
    assert [row["job"] for row in rows] == [0, 1]
    assert all(row["started_s"] is not None for row in rows)
    assert all(row["started_s"] <= row["ended_s"] for row in rows)


def test_a_prompt_line_reaches_the_published_row_as_a_digest(monkeypatch):
    ledger = _fresh_fleet_ledger(monkeypatch)
    _run_fleet_result(
        monkeypatch,
        [torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1)],
        public=True,
        prompts="a cat\na dog",
        clip=_StubClip(),
    )

    block = ledger.snapshot()
    # The conditioning job comes first, then one job per prompt line.
    assert [row["prompt"] for row in block["jobs"]] == [
        None,
        telemetry_fleet.prompt_digest("a cat"),
        telemetry_fleet.prompt_digest("a dog"),
    ]
    assert [row["box"] for row in block["jobs"]] == ["box 1", "box 2", "box 1"]


def test_a_reply_that_names_a_rank_of_its_own_keeps_the_rank_the_driver_sent_to(
        monkeypatch):
    # Every fleet worker is set up as rank 0 of its own world-1 mesh, so a
    # reply's rank names no box and is not compared with anything. The row
    # keeps the rank the driver submitted to and carries no marker.
    ledger = _fresh_fleet_ledger(monkeypatch)
    _run_fleet_result(
        monkeypatch, [torch.zeros(1, 1)],
        worker_rows=[{"host": "worker-a", "rank": 3, "sample_s": 1.0}],
    )

    row, = ledger.snapshot()["jobs"]
    assert row["rank"] == 0
    assert row["box"] == "box 1"
    assert "host_reported" not in row


def test_a_job_answered_by_another_boxs_host_is_marked(monkeypatch):
    # The driver sent job 1 to the second box and the first box's host
    # answered it, so that row is marked.
    ledger = _fresh_fleet_ledger(monkeypatch)
    _run_fleet_result(
        monkeypatch, [torch.zeros(1, 1), torch.zeros(1, 1)],
        worker_rows=[
            {"host": "worker-a", "rank": 0, "sample_s": 1.0},
            {"host": "worker-a", "rank": 0, "sample_s": 1.0},
        ],
    )

    first, second = ledger.snapshot()["jobs"]
    assert "host_reported" not in first
    assert (second["box"], second["host_reported"]) == ("box 2", "worker-a")


def test_a_reply_the_driver_rejected_leaves_its_row_open(monkeypatch):
    # A reply that fails validation leaves its dispatch row open, as a job that
    # never came back does: the row names the box the job went to and when,
    # which is what a Stop mid-fleet leaves the sidebar. The raised error
    # carries the whole reply, host included.
    ledger = _fresh_fleet_ledger(monkeypatch)
    with pytest.raises(RuntimeError, match="returned no latent"):
        _run_fleet_result(
            monkeypatch,
            [torch.zeros(1, 1), torch.zeros(1, 1)],
            worker_rows=[
                {"host": "worker-a", "rank": 0, "sample_s": 1.0},
                {"latent": None},
            ],
        )

    block = ledger.snapshot()
    assert block["count"] == 2
    answered, lost = block["jobs"]
    assert (answered["host"], answered["box"]) == ("worker-a", "box 1")
    assert answered["ended_s"] is not None
    assert (lost["box"], lost["host"]) == ("box 2", None)
    assert lost["started_s"] is not None
    assert (lost["ended_s"], lost["wall_s"]) == (None, None)
