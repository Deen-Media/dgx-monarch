"""Fail-closed automatic execution for FSDP Gate proof scope."""
from __future__ import annotations

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from dgx_monarch import gate_ledger, mesh_rpc, mesh_safety
from dgx_monarch.actor import model_store
from dgx_monarch.nodes import (
    auto_gate,
    common,
    gate,
    gate_identity,
    gate_process_state,
    gate_session,
)
from dgx_monarch.nodes.gate_fsdp import (
    FSDP_PROOF_KIND,
    FSDP_PROOF_SCOPE,
    FsdpGateProofError,
)
from dgx_monarch.topology import Topology


def _fsdp_token() -> tuple[str, ...]:
    return gate_ledger.gate_verdict_token(
        "combo",
        "artifact",
        "known",
        {"proof_scope": FSDP_PROOF_SCOPE},
    )


def _patch_auto_state(monkeypatch, token: tuple[str, ...]) -> None:
    """Give one test its own process-local gate state and context seam.

    Every binding is replaced, the wait included: the two claim-timeout paths
    read it, and the real default is half an hour.
    """
    lock = threading.Lock()
    monkeypatch.setattr(
        gate_process_state, "_AUTO_GATE_ACTIVE", SimpleNamespace(on=False))
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_LOCK", lock)
    monkeypatch.setattr(
        gate_process_state, "_AUTO_GATE_CONDITION", threading.Condition(lock))
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_RUNNING", set())
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_SESSION", {})
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_SESSION_LIMIT", 16)
    monkeypatch.setattr(gate_process_state, "_PROCESS_GATE_DENIALS", {})
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_WAIT_S", 1.0)
    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("unknown", token))


class _CleanupHandle:
    def __init__(self):
        self.unloads = 0
        self.world = 2

    def call_all(self, method, *, timeout_s):
        assert method == "unload"
        assert timeout_s == 600.0
        self.unloads += 1
        return [{"unloaded": True} for _rank in range(self.world)]


def test_auto_fsdp_fail_publishes_denial_then_raises(monkeypatch):
    token = _fsdp_token()
    _patch_auto_state(monkeypatch, token)
    monkeypatch.setattr(
        gate,
        "run_identity_ceremony",
        lambda *_args, **_kwargs: {
            "verdict": "FAIL",
            "proof_kind": FSDP_PROOF_KIND,
            "_gate_tokens": [token],
        },
    )

    with pytest.raises(FsdpGateProofError) as raised:
        auto_gate.maybe_auto_gate(
            SimpleNamespace(),
            {"kind": "ksampler", "steps": 2},
            {},
            1.0,
            2,
        )

    assert raised.value.verdict == "FAIL"
    assert gate_process_state._PROCESS_GATE_DENIALS == {token: "FAIL"}
    assert gate_process_state._AUTO_GATE_SESSION == {token: "FAIL"}
    assert gate_process_state._AUTO_GATE_RUNNING == set()
    assert gate_process_state._AUTO_GATE_ACTIVE.on is False


@pytest.mark.parametrize(
    "rejection",
    ["token_drift", "scope_mismatch", "non_consumable"],
)
def test_auto_rejected_fsdp_pass_cleans_exact_ceremony_handle(
    monkeypatch,
    rejection,
):
    token = _fsdp_token()
    _patch_auto_state(monkeypatch, token)
    exact_handle = _CleanupHandle()
    decoy_handle = _CleanupHandle()
    model = SimpleNamespace(mesh=SimpleNamespace(handle=decoy_handle))
    context_calls = 0

    def context(*_args):
        nonlocal context_calls
        context_calls += 1
        state = (
            "error"
            if rejection == "non_consumable" and context_calls > 1
            else "unknown"
        )
        return state, token

    monkeypatch.setattr(common, "_auto_gate_context", context)
    actual_token = (
        ("changed", *token[1:]) if rejection == "token_drift" else token
    )
    proof_kind = (
        "lora_low_rss_swap"
        if rejection == "scope_mismatch"
        else FSDP_PROOF_KIND
    )
    monkeypatch.setattr(
        gate,
        "run_identity_ceremony",
        lambda *_args, **_kwargs: {
            "verdict": "PASS",
            "proof_kind": proof_kind,
            "_gate_token": actual_token,
            "_gate_tokens": [token],
            "_gate_handle": exact_handle,
            "_fsdp_cleanup_attempted": False,
            "_fsdp_cleanup_confirmed": False,
        },
    )

    with pytest.raises(FsdpGateProofError) as raised:
        auto_gate.maybe_auto_gate(
            model,
            {"kind": "ksampler", "steps": 2},
            {},
            1.0,
            2,
        )

    assert raised.value.verdict == "ERROR"
    assert exact_handle.unloads == 1
    assert decoy_handle.unloads == 0
    assert gate_process_state._PROCESS_GATE_DENIALS == {token: "ERROR"}
    assert gate_process_state._AUTO_GATE_RUNNING == set()
    assert gate_process_state._AUTO_GATE_ACTIVE.on is False


def test_auto_rejected_fsdp_pass_respects_completed_shared_cleanup(monkeypatch):
    token = _fsdp_token()
    _patch_auto_state(monkeypatch, token)
    exact_handle = _CleanupHandle()
    model = SimpleNamespace(mesh=SimpleNamespace(handle=_CleanupHandle()))
    monkeypatch.setattr(
        gate,
        "run_identity_ceremony",
        lambda *_args, **_kwargs: {
            "verdict": "PASS",
            "proof_kind": FSDP_PROOF_KIND,
            "_gate_token": ("changed", *token[1:]),
            "_gate_tokens": [token],
            "_gate_handle": exact_handle,
            "_fsdp_cleanup_attempted": True,
            "_fsdp_cleanup_confirmed": True,
        },
    )

    with pytest.raises(FsdpGateProofError):
        auto_gate.maybe_auto_gate(
            model,
            {"kind": "ksampler", "steps": 2},
            {},
            1.0,
            2,
        )

    assert exact_handle.unloads == 0


def test_auto_fsdp_return_boundary_interruption_cleans_armed_exact_handle(
    monkeypatch,
):
    class ReturnBoundaryInterrupt(BaseException):
        def __bool__(self):
            raise AssertionError("active exception truthiness must not be evaluated")

    token = _fsdp_token()
    _patch_auto_state(monkeypatch, token)
    exact_handle = _CleanupHandle()
    decoy_handle = _CleanupHandle()
    model = SimpleNamespace(
        loras=(),
        mesh=SimpleNamespace(
            handle=decoy_handle,
            topology_preset="uly2+fsdp",
        ),
    )
    monkeypatch.setattr(
        common,
        "_bind_packed_render_model",
        lambda *_args, **_kwargs: (model, exact_handle),
    )
    interruption = ReturnBoundaryInterrupt("interrupted while returning PASS")
    closes = 0

    class Session:
        def bind(self, _handle):
            pass

        def activate(self):
            return nullcontext()

        def close(self):
            nonlocal closes
            closes += 1
            if closes == 1:
                raise interruption

    session_runtime = {
        "ensure_live": lambda value: value,
        "RenderSession": Session,
        "_close_gate_session": lambda session, _primary: session.close(),
        "_force_stock_quarantine": lambda *_args, **_kwargs: None,
        "log": SimpleNamespace(info=lambda *_args: None, error=lambda *_args: None),
        "_run_identity_ceremony_bound": lambda *_args, **_kwargs: {
            "verdict": "PASS",
            "proof_kind": FSDP_PROOF_KIND,
            "_gate_token": token,
            "_gate_tokens": [token],
        },
    }

    def ceremony(
        ceremony_model,
        request,
        latent,
        cfg_value,
        steps_hint,
        origin,
        run_id="",
        provenance_attestor=None,
        _fsdp_return_guard=None,
    ):
        return gate_session.run_identity_ceremony(
            ceremony_model,
            request,
            latent,
            cfg_value,
            steps_hint,
            origin,
            run_id,
            runtime=session_runtime,
            provenance_attestor=provenance_attestor,
            fsdp_return_guard=_fsdp_return_guard,
        )

    monkeypatch.setattr(gate, "run_identity_ceremony", ceremony)

    with pytest.raises(ReturnBoundaryInterrupt) as raised:
        auto_gate.maybe_auto_gate(
            model,
            {"kind": "ksampler", "steps": 2},
            {},
            1.0,
            2,
        )

    assert raised.value is interruption
    assert closes == 2
    assert exact_handle.unloads == 1
    assert decoy_handle.unloads == 0
    assert gate_process_state._AUTO_GATE_RUNNING == set()
    assert gate_process_state._AUTO_GATE_ACTIVE.on is False


@pytest.mark.parametrize("source", ["process", "durable"])
def test_cached_or_durable_fsdp_fail_is_a_typed_refusal(monkeypatch, source):
    token = _fsdp_token()
    _patch_auto_state(monkeypatch, token)
    if source == "process":
        gate_process_state._PROCESS_GATE_DENIALS[token] = "FAIL"
    else:
        monkeypatch.setattr(
            common, "_auto_gate_context", lambda *_args: ("fail", token))

    with pytest.raises(FsdpGateProofError, match="FSDP execution denied"):
        auto_gate.maybe_auto_gate(
            SimpleNamespace(),
            {"kind": "ksampler", "steps": 2},
            {},
            1.0,
            2,
        )


def test_an_unknown_denial_source_names_itself_and_the_sources_that_exist():
    """A mistyped source must not raise a bare KeyError that reads only
    ``'durabel'``, from a helper five calls deep. No call site passes one.
    """
    with pytest.raises(ValueError) as raised:
        auto_gate._deny_automatic_fsdp("durabel", "FAIL")

    message = str(raised.value)
    assert "unknown automatic-FSDP denial source 'durabel'" in message
    for source in auto_gate._FSDP_DENIAL_REASONS:
        assert repr(source) in message


def test_gate_session_never_stock_quarantines_fsdp_result_or_error(monkeypatch):
    handle = object()
    model = SimpleNamespace(mesh=SimpleNamespace(handle=handle))
    monkeypatch.setattr(
        common,
        "_bind_packed_render_model",
        lambda *_args, **_kwargs: (model, handle),
    )
    quarantines = []

    class Session:
        def bind(self, _handle):
            pass

        def activate(self):
            return nullcontext()

        def close(self):
            pass

    runtime = {
        "ensure_live": lambda value: value,
        "RenderSession": Session,
        "_close_gate_session": lambda session, _primary: session.close(),
        "_force_stock_quarantine": lambda *_args, **_kwargs: quarantines.append(True),
        "log": SimpleNamespace(error=lambda *_args: None),
        "_run_identity_ceremony_bound": lambda *_args, **_kwargs: {
            "verdict": "FAIL",
            "proof_kind": FSDP_PROOF_KIND,
        },
    }

    result = gate_session.run_identity_ceremony(
        model, {}, {}, 1.0, 2, "auto_first_use", "", runtime=runtime
    )
    assert result["verdict"] == "FAIL"
    assert quarantines == []

    def refuse(*_args, **_kwargs):
        raise FsdpGateProofError("incomplete", verdict="INCONCLUSIVE")

    runtime["_run_identity_ceremony_bound"] = refuse
    with pytest.raises(FsdpGateProofError):
        gate_session.run_identity_ceremony(
            model, {}, {}, 1.0, 2, "auto_first_use", "", runtime=runtime
        )
    assert quarantines == []


def test_gate_session_preserves_fsdp_base_exception_without_stock_quarantine(
    monkeypatch,
):
    class GateCancelled(BaseException):
        pass

    cancellation = GateCancelled("cancelled")
    unloads = []

    def unload(method, *, timeout_s):
        unloads.append((method, timeout_s))
        return [{"unloaded": True}, {"unloaded": True}]

    handle = SimpleNamespace(
        world=2,
        call_all=unload,
    )
    model = SimpleNamespace(
        loras=(),
        mesh=SimpleNamespace(
            handle=handle,
            topology_preset="uly2+fsdp",
        ),
    )
    monkeypatch.setattr(
        common,
        "_bind_packed_render_model",
        lambda *_args, **_kwargs: (model, handle),
    )
    quarantines = []

    class Session:
        def bind(self, _handle):
            pass

        def activate(self):
            return nullcontext()

        def close(self):
            pass

    def cancel(*_args, **_kwargs):
        raise cancellation

    runtime = {
        "ensure_live": lambda value: value,
        "RenderSession": Session,
        "_close_gate_session": lambda session, _primary: session.close(),
        "_force_stock_quarantine": lambda *_args, **_kwargs: quarantines.append(True),
        "log": SimpleNamespace(error=lambda *_args: None),
        "_run_identity_ceremony_bound": cancel,
    }

    with pytest.raises(GateCancelled) as raised:
        gate_session.run_identity_ceremony(
            model, {}, {}, 1.0, 2, "auto_first_use", "", runtime=runtime
        )

    assert raised.value is cancellation
    assert quarantines == []
    assert unloads == [("unload", 600.0)]


@pytest.mark.parametrize("state", ["pass", "fail"])
def test_fsdp_with_residency_levers_off_still_requires_exact_gate(
    monkeypatch,
    tmp_path,
    state,
):
    handle = SimpleNamespace(world=2)
    model = SimpleNamespace(
        unet_name="wan.safetensors",
        options={},
        loras=(),
        mesh=SimpleNamespace(
            handle=handle,
            auto_gate="first_use",
            worker_args={"lora_low_rss": False, "slab_weights": False},
            topology_preset="uly2+fsdp",
            attention="SDPA",
            sync_ulysses=True,
        ),
    )
    monkeypatch.setattr(common, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        common,
        "topology_from_preset",
        lambda *_args: Topology(world=2, ulysses=2, dp=1, fsdp=True),
    )
    monkeypatch.setattr(
        common,
        "_apply_persisted_quarantine",
        lambda *_args: pytest.fail(
            "FSDP denial must not use residency quarantine"
        ),
    )

    class Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, *_args):
            return gate_ledger.GateLedgerLookup(
                state,
                {"verdict": state.upper()} if state == "fail" else None,
                True,
            )

    monkeypatch.setattr(
        gate,
        "_combo_of",
        lambda _model: (
            "combo",
            gate_ledger.ArtifactSetSignature("artifact", "artifact", True),
        ),
    )
    monkeypatch.setattr(gate, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        gate,
        "_effective_worker_args",
        lambda *_args: {"lora_low_rss": False, "slab_weights": False},
    )
    monkeypatch.setattr(gate_ledger, "GateLedger", Ledger)
    monkeypatch.setattr(gate_ledger, "comfy_commit", lambda: "known")

    context = auto_gate.auto_gate_context(model, "ksampler")

    assert context is not None
    assert context[0] == state
    if state == "fail":
        _patch_auto_state(monkeypatch, context[1])
        monkeypatch.setattr(
            common, "_auto_gate_context", lambda *_args: context)
        with pytest.raises(FsdpGateProofError, match="durable"):
            auto_gate.auto_gate_required(model)


def test_fsdp_pass_attaches_normal_grant_with_proof_scope(monkeypatch, tmp_path):
    topology = Topology(world=2, ulysses=2, dp=1, fsdp=True)
    handle = SimpleNamespace(
        world=2,
        n_hosts=2,
        gpus_per_host=1,
        config=SimpleNamespace(worker_args={}, hosts=(), source=""),
        config_fingerprint="local",
        effective_worker_args=lambda requested: dict(requested),
    )
    request = {"unet_name": "wan.safetensors", "options": {}, "loras": []}
    model = SimpleNamespace(
        request_dict=lambda: dict(request),
        mesh=SimpleNamespace(
            handle=handle,
            auto_gate="first_use",
            worker_args={"lora_low_rss": False, "slab_weights": False},
            topology_preset="uly2+fsdp",
            attention="SDPA",
            sync_ulysses=True,
        ),
    )
    artifacts = gate_ledger.ArtifactSetSignature("artifact", "artifact", True)
    identity = {
        "digest": "artifact",
        "comfy": "known",
        "artifacts": [{"signature": "artifact"}],
    }

    class Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, *_args):
            return gate_ledger.GateLedgerLookup(
                "pass", {"verdict": "PASS"}, True
            )

    monkeypatch.setattr(
        gate_identity,
        "capture",
        lambda _model: (dict(request), "combo", identity, artifacts, {}),
    )
    monkeypatch.setattr(gate_ledger, "GateLedger", Ledger)
    monkeypatch.setattr(gate, "_ledger_dir", lambda: str(tmp_path))

    authorization = gate_identity.authorize_normal_render(
        model,
        handle,
        gate_active=False,
        session_verdict=lambda _token: None,
        resolved_topology=topology,
        resolved_attention="SDPA",
    )

    assert authorization.residency_mode == "required"
    assert authorization.residency_grant is not None
    assert (
        authorization.residency_grant["capability_context"]["proof_scope"]
        == FSDP_PROOF_SCOPE
    )


def _lora_authorization(monkeypatch, tmp_path, topology, *, uncond=None):
    """Authorize a low-RSS LoRA render that has no PASS (the ceremony aborted)."""
    handle = SimpleNamespace(
        world=2,
        n_hosts=2,
        gpus_per_host=1,
        config=SimpleNamespace(worker_args={}, hosts=(), source=""),
        config_fingerprint="local",
        effective_worker_args=lambda requested: dict(requested),
    )
    loras = [{"name": "adapter.safetensors", "strength": 1.0}]
    request = {"unet_name": "wan.safetensors", "options": {}, "loras": loras}
    model = SimpleNamespace(
        loras=tuple(loras),
        request_dict=lambda: dict(request),
        mesh=SimpleNamespace(
            handle=handle,
            auto_gate="first_use",
            worker_args={"lora_low_rss": True, "slab_weights": False},
            topology_preset="uly2+fsdp" if topology.fsdp else "uly2",
            attention="SDPA",
            sync_ulysses=True,
        ),
    )
    artifacts = gate_ledger.ArtifactSetSignature("artifact", "artifact", True)
    identity = {"digest": "artifact", "comfy": "known", "artifacts": [{"signature": "artifact"}]}

    class Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, *_args):
            return gate_ledger.GateLedgerLookup("unknown", None, True)

    monkeypatch.setattr(
        gate_identity, "capture", lambda _model: (dict(request), "combo", identity, artifacts, {}))
    monkeypatch.setattr(gate_ledger, "GateLedger", Ledger)
    monkeypatch.setattr(gate, "_ledger_dir", lambda: str(tmp_path))
    return gate_identity.authorize_normal_render(
        model,
        handle,
        uncond_model_request=uncond,
        gate_active=False,
        session_verdict=lambda _token: "INCONCLUSIVE",
        resolved_topology=topology,
        resolved_attention="SDPA",
    )


def test_fsdp_lora_without_pass_is_refused_typed_not_forced_to_stock(monkeypatch, tmp_path):
    """FSDP has no stock residency: forcing lora_low_rss off only earns a card
    telling the operator to turn on a lever they set on."""
    topology = Topology(world=2, ulysses=2, dp=1, fsdp=True)

    with pytest.raises(FsdpGateProofError, match="no stock residency") as raised:
        _lora_authorization(monkeypatch, tmp_path, topology)

    assert raised.value.verdict == "INCONCLUSIVE"
    assert "first-use Gate PASS" in str(raised.value)
    with pytest.raises(FsdpGateProofError, match="dual-model"):
        _lora_authorization(monkeypatch, tmp_path, topology, uncond={"unet_name": "u.safetensors"})


def test_resident_lora_without_pass_still_falls_back_to_stock(monkeypatch, tmp_path):
    authorization = _lora_authorization(
        monkeypatch, tmp_path, Topology(world=2, ulysses=2, dp=1, fsdp=False))

    assert authorization.residency_mode == "stock"
    assert authorization.worker_args["lora_low_rss"] is False
    assert authorization.worker_args["slab_weights"] is False


def test_fsdp_lora_context_cannot_claim_clean_reload_scope():
    topology = Topology(world=2, ulysses=2, dp=1, fsdp=True)
    handle = SimpleNamespace(
        world=2,
        n_hosts=2,
        gpus_per_host=1,
        config=SimpleNamespace(worker_args={}, hosts=(), source=""),
        config_fingerprint="local",
        effective_worker_args=lambda requested: dict(requested),
    )
    model = SimpleNamespace(
        loras=({"name": "adapter.safetensors", "strength": 1.0},),
        mesh=SimpleNamespace(
            handle=handle,
            worker_args={"lora_low_rss": False, "slab_weights": False},
            topology_preset="uly2+fsdp",
            attention="SDPA",
            sync_ulysses=True,
        ),
    )

    context = gate_identity.gate_capability_context(
        model,
        handle,
        resolved_topology=topology,
        resolved_attention="SDPA",
    )

    assert "proof_scope" not in context


@pytest.mark.parametrize(
    ("loras", "expected_scope"),
    [
        ([], True),
        ([{"name": "adapter.safetensors", "strength": 1.0}], False),
    ],
)
def test_mesh_rpc_reconstructs_only_no_lora_fsdp_proof_scope(
    monkeypatch,
    loras,
    expected_scope,
):
    identity = {"digest": "artifact", "comfy": "known", "artifacts": []}
    captured = {}
    handle = SimpleNamespace(
        setup_generation=3,
        setup_key=("setup", "SDPA", True),
        worker_args_key=("policy",),
        active_worker_args={"lora_low_rss": False, "slab_weights": False},
        topology=SimpleNamespace(
            ulysses=2, ring=1, cfg=1, dp=1, fsdp=True
        ),
        config=SimpleNamespace(hosts=(), source=""),
        config_fingerprint="local",
        world=2,
        n_hosts=2,
        gpus_per_host=1,
        call_all=lambda *_args, **_kwargs: [identity, identity],
    )
    request = {
        "model": {"unet_name": "wan.safetensors", "loras": loras},
        "sage_kernel": "SDPA",
        "sync_ulysses": True,
        "_dgxm_normal_residency_grant": {},
        "_dgxm_normal_policy": {
            "topology_preset": "uly2+fsdp",
            "attention": "SDPA",
            "sync_ulysses": True,
        },
    }
    monkeypatch.setattr(
        model_store, "request_artifact_identity", lambda *_args: identity
    )
    monkeypatch.setattr(
        mesh_safety, "assert_request_artifact_binding", lambda *_args: True
    )
    monkeypatch.setattr(
        mesh_safety, "assert_normal_render_residency_mode", lambda *_args: True
    )
    monkeypatch.setattr(
        mesh_safety,
        "assert_normal_render_residency_grant",
        lambda *_args, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(mesh_safety, "assert_artifact_parity", lambda *_args: True)

    mesh_rpc.verify_request_artifacts(handle, request)

    if expected_scope:
        assert captured["expected_context"]["proof_scope"] == FSDP_PROOF_SCOPE
    else:
        assert "proof_scope" not in captured["expected_context"]


def _priced_out_fsdp_rig(monkeypatch, tmp_path, *, renders=None,
                         mem_available_gib=(30.2,)):
    """The real ceremony, on a fleet whose clean reload provably cannot fit.

    The rig never names the price module, so without pricing these tests fail
    on behaviour, not on import: the whole ceremony runs and the gate PASSes.

    ``mem_available_gib`` is read one entry per ``status`` call and then holds
    at its last value, so a fit-then-short pair drives the reprice stop the
    ceremony makes between the baseline render and the reload cycle.
    """
    import sys

    import torch

    from gate_orchestration_helpers import _cross_mode_rig

    token = _fsdp_token()
    _patch_auto_state(monkeypatch, token)
    handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.zeros(1)] if renders is None else renders,
        {"lora_low_rss": False, "slab_weights": False},
    )
    handle.world = model.mesh.world = 2
    model.loras = ()
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True
    checkpoint = tmp_path / "flux2-dev.safetensors"
    with open(checkpoint, "wb") as fh:  # sparse: no bytes are written
        fh.truncate(60 << 30)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setitem(
        sys.modules, "folder_paths",
        SimpleNamespace(get_full_path=lambda _kind, _name: str(checkpoint)))
    inner = handle.call_all
    remaining = list(mem_available_gib)

    def call_all(method, *args, **kwargs):
        if method == "status":
            handle.calls.append((method, args))
            available = remaining.pop(0) if len(remaining) > 1 else remaining[0]
            return [
                {"rank": rank, "host": {"mem_gib": {"MemAvailable": available}}}
                for rank in range(2)
            ]
        return inner(method, *args, **kwargs)

    handle.call_all = call_all
    return handle, ledger, model, token


def _queue(model):
    return auto_gate.maybe_auto_gate(
        model,
        {"kind": "ksampler", "steps": 2, "noise_seed": 1, "cfg": 1.0},
        {"samples": __import__("torch").zeros(1)},
        1.0,
        2,
    )


def test_a_priced_capacity_stop_publishes_no_process_verdict(monkeypatch, tmp_path):
    """Nothing was proven and nothing was disproven, so nothing is denied.

    A returned result would reach auto_gate's proof-scope reconciliation and be
    rewritten to ERROR, a sticky process denial, so the ceremony records the
    row and raises.
    """
    handle, ledger, model, token = _priced_out_fsdp_rig(monkeypatch, tmp_path)

    with pytest.raises(FsdpGateProofError) as raised:
        _queue(model)

    assert raised.value.verdict == "INCONCLUSIVE"
    assert gate_process_state._PROCESS_GATE_DENIALS == {}
    assert gate_process_state._AUTO_GATE_SESSION == {}
    assert gate_process_state._AUTO_GATE_RUNNING == set()
    assert handle.unload_calls == 0
    assert len(ledger.records) == 1
    assert ledger.records[0][3] == "INCONCLUSIVE"
    assert ledger.records[0][4]["cross_mode"] == "CAPACITY"
    assert token not in gate_process_state._AUTO_GATE_SESSION


def test_the_next_queue_prices_again_instead_of_replaying_a_denial(
    monkeypatch, tmp_path,
):
    """Memory is not a stable fact, so the row grants nothing and re-prices."""
    handle, ledger, model, _token = _priced_out_fsdp_rig(monkeypatch, tmp_path)

    for _attempt in range(2):
        with pytest.raises(FsdpGateProofError):
            _queue(model)

    assert [method for method, _args in handle.calls] == ["status", "status"]
    assert len(ledger.records) == 2
    assert gate_process_state._PROCESS_GATE_DENIALS == {}
    assert gate_process_state._AUTO_GATE_SESSION == {}


def test_an_aborted_fsdp_lora_ceremony_surfaces_its_typed_cause(monkeypatch):
    """LoRA on a resolved FSDP topology has no stock residency to fall back to,
    so a ceremony that met a typed refusal raises that refusal instead of
    forcing stock and handing the render an untyped denial."""
    from monarch.actor import ActorError

    from dgx_monarch.nodes.gate_fsdp import token_resolves_fsdp

    token = gate_ledger.gate_verdict_token(
        "combo", "artifact", "known",
        {"resolved_topology": {"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1, "fsdp": True}})
    assert token_resolves_fsdp(token)
    assert not token_resolves_fsdp(_fsdp_token())  # the no-LoRA scope names no topology here
    _patch_auto_state(monkeypatch, token)

    def ceremony(*_args, **_kwargs):
        raise _class_p_worker_refusal()

    monkeypatch.setattr(gate, "run_identity_ceremony", ceremony)

    with pytest.raises(ActorError) as raised:
        auto_gate.maybe_auto_gate(
            SimpleNamespace(), {"kind": "ksampler", "steps": 2}, {}, 1.0, 2)

    assert "[dgxm:P]" in str(raised.value.exception)


def _watch_published_tokens(monkeypatch) -> list:
    """Record the ceremony's own tokens while still publishing them."""
    seen: list = []
    inner = gate._publish_process_gate_verdicts

    def publish(tokens, verdict, ceremony=None):
        seen.append([tuple(token) for token in tokens])
        return inner(tokens, verdict, ceremony)

    monkeypatch.setattr(gate, "_publish_process_gate_verdicts", publish)
    return seen


def _class_p_worker_refusal():
    from monarch.actor import ActorError

    from dgx_monarch.refusal import RefusalClass, refusal

    return ActorError(RuntimeError(refusal(
        RefusalClass.PHYSICS,
        "cfg-parallel got a model call of batch=1 on 2 ranks, and a batch of 1 "
        "does not split into 2 equal slices. Use a topology without cfg2.")))


def test_a_reprice_stop_leaves_no_process_denial(monkeypatch, tmp_path):
    """The reprice runs after the unload and the baseline render, so unlike the
    preflight it must withdraw the INCONCLUSIVE the ceremony already published."""
    published = _watch_published_tokens(monkeypatch)
    handle, ledger, model, _token = _priced_out_fsdp_rig(
        monkeypatch, tmp_path, mem_available_gib=(96.0, 20.0))

    with pytest.raises(FsdpGateProofError) as raised:
        _queue(model)

    assert raised.value.verdict == "INCONCLUSIVE"
    assert handle.unload_calls  # the reprice stop is in flight, not a preflight
    assert published and published[0]
    assert len(ledger.records) == 1
    assert ledger.records[0][3] == "INCONCLUSIVE"
    assert ledger.records[0][4]["cross_mode"] == "CAPACITY"
    assert gate_process_state._PROCESS_GATE_DENIALS == {}
    assert gate_process_state._AUTO_GATE_SESSION == {}
    assert gate_process_state._AUTO_GATE_RUNNING == set()
    # Memory is not a stable fact, so the next queue prices again.
    with pytest.raises(FsdpGateProofError):
        _queue(model)
    assert len(ledger.records) == 2


def test_a_settled_class_p_refusal_answers_both_attempts(monkeypatch, tmp_path):
    """Attempt 1 meets the class P guard inside the proof. Attempt 2 must meet
    the same guard, not a process-local INCONCLUSIVE left by the aborted proof.
    """
    from monarch.actor import ActorError

    published = _watch_published_tokens(monkeypatch)
    _handle, ledger, model, _token = _priced_out_fsdp_rig(
        monkeypatch, tmp_path,
        renders=[_class_p_worker_refusal(), _class_p_worker_refusal()],
        mem_available_gib=(96.0,))

    with pytest.raises(ActorError) as first:
        _queue(model)

    assert "[dgxm:P]" in str(first.value.exception)
    ceremony_token = published[0][0]
    assert ceremony_token not in gate_process_state._PROCESS_GATE_DENIALS
    assert gate_process_state._PROCESS_GATE_DENIALS == {}
    assert gate_process_state._AUTO_GATE_SESSION == {}
    assert gate_process_state._AUTO_GATE_RUNNING == set()
    assert len(ledger.records) == 1
    assert ledger.records[0][4]["refusal_class"] == "P"

    # Attempt 2 asks about the exact combination the ceremony judged, where a
    # sticky denial would answer.
    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("unknown", ceremony_token))
    with pytest.raises(ActorError) as second:
        _queue(model)

    assert "[dgxm:P]" in str(second.value.exception)
    for reason in auto_gate._FSDP_DENIAL_REASONS.values():
        assert reason.split(" {")[0] not in str(second.value)


def _real_ledger_fsdp_rig(monkeypatch, tmp_path):
    """The production context resolver over a real ledger file.

    ``_priced_out_fsdp_rig`` patches ``_auto_gate_context`` away, so it cannot
    show what the abort arm's terminal row does to the next attempt. This rig
    keeps the real resolver and the real ``GateLedger``, the only place the
    two meet.
    """
    handle = SimpleNamespace(
        config=SimpleNamespace(worker_args={}),
        world=2, n_hosts=2, gpus_per_host=1)
    model = SimpleNamespace(
        unet_name="flux2-dev.safetensors", options={}, loras=(),
        mesh=SimpleNamespace(
            handle=handle, auto_gate="first_use",
            worker_args={"lora_low_rss": False, "slab_weights": False},
            topology_preset="uly2+fsdp", attention="SDPA", sync_ulysses=True),
    )
    artifacts = gate_ledger.artifact_set_signature(["model-sig"])
    monkeypatch.setattr(gate, "_combo_of", lambda _model: ("combo", artifacts))
    monkeypatch.setattr(gate, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(common, "ensure_live", lambda value: value)
    monkeypatch.setattr(gate_ledger, "comfy_commit", lambda: "commit")
    return model, artifacts


def test_the_terminal_row_does_not_itself_deny_the_next_attempt(
    monkeypatch, tmp_path,
):
    """Closing the transaction must not become a new denial.

    The row the abort arm appends reads exactly what the open RETESTING row
    read, so the next queue runs the ceremony again instead of meeting a
    durable stop.
    """
    from dgx_monarch.nodes import gate_abort

    model, artifacts = _real_ledger_fsdp_rig(monkeypatch, tmp_path)
    state, token = common._auto_gate_context(model, "ksampler")
    assert state == "unknown"
    assert auto_gate.token_requires_fsdp_proof(token) is True

    ledger = gate_ledger.GateLedger(str(tmp_path))
    ledger.begin_retest_required(
        "combo", artifacts, "commit", [token[3]],
        {"phase": "ceremony-preflight"})
    assert common._auto_gate_context(model, "ksampler")[0] == "inconclusive"

    gate_abort.record_proof_stop(
        ledger, "combo", artifacts, "commit", token[3], model.unet_name,
        origin="auto_first_use", run_id="", tag=None,
        exc=RuntimeError("the reload cycle exploded"))

    rows = gate_ledger.GateLedger(str(tmp_path)).entries()
    assert [row["verdict"] for row in rows] == ["RETESTING", "INCONCLUSIVE"]
    assert common._auto_gate_context(model, "ksampler")[0] == "inconclusive"

    ceremonies: list = []

    def ceremony(*_args, **_kwargs):
        ceremonies.append(1)
        raise RuntimeError("the reload cycle exploded again")

    monkeypatch.setattr(gate, "run_identity_ceremony", ceremony)
    resolve = common._auto_gate_context
    monkeypatch.setattr(
        common, "_auto_gate_context",
        lambda subject, _kind, *_args: resolve(subject, "ksampler"))

    with pytest.raises(FsdpGateProofError) as raised:
        _queue(model)

    # The ceremony ran, so nothing durable answered ahead of it.
    assert ceremonies == [1]
    for reason in auto_gate._FSDP_DENIAL_REASONS.values():
        assert reason.split(" {")[0] not in str(raised.value)
    # This rig keeps the real process maps, so it also proves the untyped
    # abort left no claim behind and leaks nothing into the next test.
    assert gate_process_state._AUTO_GATE_RUNNING == set()
    assert gate_process_state._AUTO_GATE_SESSION == {}
