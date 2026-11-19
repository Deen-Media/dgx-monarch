"""The driver's permission for the in-place LoRA bake, from the gate ledger to the Worker's store.

A Worker may bake a classic stack in place only for a combination the
identity gate has passed (``actor/in_place_fallback.py``). These tests hold
the trust boundary: ``proven`` against a real ledger, ``stamp`` and the strip
in the real submit path, the one place a Worker binds the field, and the
field's way to the quote. CPU only.
"""
from __future__ import annotations

import dataclasses
import types

import pytest

from dgx_monarch import capacity_agreement
from dgx_monarch.actor import in_place_fallback, store_fsdp
from dgx_monarch.capacity_classic_bake import IN_PLACE_BAKE_PROVEN
from dgx_monarch.gate_ledger import GateLedger
from dgx_monarch.nodes import in_place_bake_permission as permission

COMMIT = "commit-now"
SINGLE, PAIR = ("key-single", "digest-single"), ("key-pair", "digest-pair")
OFF = {"lora_low_rss": False, "slab_weights": "auto"}
UNCOND = {"unet_name": "uncond.safetensors", "loras": None}


def _context(worker_args):
    return {"worker_args": dict(worker_args)}


@pytest.fixture
def gate(monkeypatch, tmp_path):
    """The lookup's seams, with a real ledger in ``tmp_path``."""
    import dgx_monarch.gate_ledger as ledger_mod
    import dgx_monarch.nodes.common as common
    import dgx_monarch.nodes.gate as gate_mod

    state = types.SimpleNamespace(effective=dict(OFF), session={}, ledger=GateLedger(str(tmp_path)))
    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: SINGLE)
    monkeypatch.setattr(gate_mod, "_dual_combo_of", lambda _model, _uncond: PAIR)
    monkeypatch.setattr(gate_mod, "_effective_worker_args", lambda _model, _handle: dict(state.effective))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(gate_mod, "gate_capability_context",
                        lambda _model, _handle, *, effective_worker_args, **_kw: _context(effective_worker_args))
    monkeypatch.setattr(ledger_mod, "comfy_commit", lambda: COMMIT)
    monkeypatch.setattr(common, "_process_gate_verdict", lambda token: state.session.get(token[0]))
    return state


def _record(gate, key, verdict, worker_args, commit=COMMIT):
    assert gate.ledger.record(key[0], key[1], commit, verdict, context=_context(worker_args))


MODEL = types.SimpleNamespace(loras=[{"name": "stack.safetensors", "strength": 0.8}])
HANDLE = object()
ON = {**OFF, "lora_low_rss": True}
AUTO = {**OFF, "lora_low_rss": "auto"}
UNSET = {"slab_weights": "auto"}


def test_no_pass_on_record_is_no_permission(gate):
    assert permission.proven(MODEL, HANDLE) is False


@pytest.mark.parametrize("recorded_under", [ON, AUTO, UNSET])
def test_a_pass_with_the_in_place_bake_on_auto_or_unset_grants_it(gate, recorded_under):
    _record(gate, SINGLE, "PASS", recorded_under)
    assert permission.proven(MODEL, HANDLE) is True


def test_a_pass_recorded_with_the_switch_off_proves_nothing_about_the_in_place_bake(gate):
    _record(gate, SINGLE, "PASS", OFF)
    assert permission.proven(MODEL, HANDLE) is False


def test_a_pass_under_other_worker_arguments_does_not_count(gate):
    _record(gate, SINGLE, "PASS", {"lora_low_rss": True, "slab_weights": True})
    assert permission.proven(MODEL, HANDLE) is False


def test_a_fail_under_any_of_the_three_contexts_withholds_it(gate):
    _record(gate, SINGLE, "PASS", ON)
    _record(gate, SINGLE, "FAIL", UNSET)
    assert permission.proven(MODEL, HANDLE) is False


def test_a_pass_from_another_comfyui_commit_is_stale_and_grants_nothing(gate):
    _record(gate, SINGLE, "PASS", ON, commit="commit-before")
    assert permission.proven(MODEL, HANDLE) is False


def test_a_fail_measured_this_session_withholds_it_whatever_the_ledger_says(gate):
    _record(gate, SINGLE, "PASS", ON)
    gate.session[SINGLE[0]] = "FAIL"
    assert permission.proven(MODEL, HANDLE) is False


def test_a_gate_ceremony_in_progress_gets_none(gate):
    import dgx_monarch.nodes.common as common

    _record(gate, SINGLE, "PASS", ON)
    common._AUTO_GATE_ACTIVE.on = True
    try:
        assert permission.proven(MODEL, HANDLE) is False
    finally:
        common._AUTO_GATE_ACTIVE.on = False
    assert permission.proven(MODEL, HANDLE) is True


def test_a_render_with_no_stack_no_handle_or_the_switch_not_off_gets_none(gate):
    _record(gate, SINGLE, "PASS", ON)
    assert permission.proven(types.SimpleNamespace(loras=[]), HANDLE) is False
    assert permission.proven(MODEL, None) is False
    for effective in (ON, AUTO, UNSET):
        gate.effective = dict(effective)
        assert permission.proven(MODEL, HANDLE) is False  # the default already bakes in place


def test_a_ledger_that_cannot_be_read_is_no_permission(gate, monkeypatch):
    _record(gate, SINGLE, "PASS", ON)

    def broken(*_args, **_kwargs):
        raise OSError("the ledger is unreadable")

    monkeypatch.setattr(GateLedger, "lookup", broken)
    assert permission.proven(MODEL, HANDLE) is False


def test_a_dual_model_render_reads_the_pairs_pass_and_either_keys_fail(gate):
    # The gate's own grant for a pair reads the pair's key alone (nodes/gate_dual.py).
    _record(gate, SINGLE, "PASS", ON)
    assert permission.proven(MODEL, HANDLE, uncond=UNCOND) is False  # the single key's PASS is not the pair's
    _record(gate, PAIR, "PASS", ON)
    assert permission.proven(MODEL, HANDLE, uncond=UNCOND) is True
    _record(gate, SINGLE, "FAIL", ON)
    assert permission.proven(MODEL, HANDLE, uncond=UNCOND) is False  # a FAIL on the model alone withholds


def test_a_stack_on_the_second_model_alone_still_asks(gate):
    _record(gate, PAIR, "PASS", ON)
    bare = types.SimpleNamespace(loras=[])
    assert permission.proven(bare, HANDLE, uncond={**UNCOND, "loras": MODEL.loras}) is True


def test_stamp_sets_true_only_when_proven_and_removes_anything_a_request_arrived_with(gate, monkeypatch):
    request = {IN_PLACE_BAKE_PROVEN: True, "uncond_model": None}
    permission.stamp(request, MODEL, HANDLE)
    assert IN_PLACE_BAKE_PROVEN not in request  # nothing on record: an injected field does not survive
    _record(gate, SINGLE, "PASS", ON)
    permission.stamp(request, MODEL, HANDLE)
    assert request[IN_PLACE_BAKE_PROVEN] is True
    gate.session[SINGLE[0]] = "FAIL"
    permission.stamp(request, MODEL, HANDLE)
    assert IN_PLACE_BAKE_PROVEN not in request  # and a stale True from an earlier stamp goes too


def test_stamp_reads_a_dual_model_renders_own_keys(gate):
    """``stamp`` hands the request's second model to the lookup: the pair's PASS grants, the model's alone does not."""
    dual = {"uncond_model": UNCOND}
    _record(gate, SINGLE, "PASS", ON)
    permission.stamp(dual, MODEL, HANDLE)
    assert IN_PLACE_BAKE_PROVEN not in dual
    _record(gate, PAIR, "PASS", ON)
    permission.stamp(dual, MODEL, HANDLE)
    assert dual[IN_PLACE_BAKE_PROVEN] is True
    single = {"uncond_model": None}
    permission.stamp(single, MODEL, HANDLE)
    assert single[IN_PLACE_BAKE_PROVEN] is True


class _ReachedSetup(Exception):
    """Raised by the setup stub once the submit has stamped the request."""


@pytest.mark.parametrize("is_proven", [False, True, "stamp removed"])
def test_the_real_submit_strips_an_incoming_field_and_stamps_its_own(monkeypatch, tmp_path, is_proven):
    """A request that arrives carrying the field cannot grant itself the permission.

    The third case takes the stamp away: the submit's own strip of internal
    keys must have removed the field before the stamp is reached.
    """
    import dgx_monarch.nodes.common as common
    import dgx_monarch.nodes.render_submit as submit_mod
    from dgx_monarch import mesh_setup
    from dgx_monarch.nodes.pending import PendingRenderHandoff
    from render_sessions_helpers import _real_handle, _stub_direct_submit
    from test_gating_off_posture import _lora_model

    graph, ledger_handle = _lora_model(monkeypatch, tmp_path, auto_gate="off", state="unknown")
    handle = _real_handle()
    for name in ("config", "config_fingerprint", "n_hosts", "gpus_per_host"):
        setattr(handle, name, getattr(ledger_handle, name))
    _stub_direct_submit(monkeypatch, handle, lambda *_args, **_kwargs: object())
    asked = []
    monkeypatch.setattr(permission, "proven", lambda *_args, **_kwargs: asked.append(1) or is_proven is True)
    if is_proven == "stamp removed":
        monkeypatch.setattr(submit_mod, "stamp_in_place_bake_permission",
                            lambda *_args, **_kwargs: asked.append(1))

    def setup(*_args, **_kwargs):
        raise _ReachedSetup()

    monkeypatch.setattr(mesh_setup, "ensure_request_setup", setup)
    spec = dataclasses.replace(graph, mesh=dataclasses.replace(graph.mesh, handle=handle))
    request = {IN_PLACE_BAKE_PROVEN: True}
    with pytest.raises(_ReachedSetup):
        common.submit_render(spec, request, {"samples": object()}, cfg_value=1.0, steps_hint=2,
                             handoff=PendingRenderHandoff())
    assert asked == [1]
    assert request.get(IN_PLACE_BAKE_PROVEN, False) is (is_proven is True)


def _worker(seen):
    def ensure(*args, **kwargs):
        seen.append((in_place_fallback.permitted(), kwargs))
        return "patcher", "load"

    return types.SimpleNamespace(store=types.SimpleNamespace(ensure=ensure), topology={}, world=1)


@pytest.fixture
def bare_ensure(monkeypatch):
    from dgx_monarch import accuracy_waiver

    monkeypatch.setattr(store_fsdp, "_assert_artifact_runnable", lambda _args, _kwargs: None)
    monkeypatch.setattr(store_fsdp, "_assert_comfy_managed_supported", lambda _worker, _args, _kwargs: None)
    monkeypatch.setattr(store_fsdp, "active", lambda _worker: False)
    monkeypatch.setattr(accuracy_waiver, "bind_model", lambda *_args, **_kwargs: None)


def test_the_worker_binds_the_permission_for_the_stores_load_and_nowhere_else(bare_ensure):
    seen: list = []
    worker = _worker(seen)
    store_fsdp.ensure(worker, "m.safetensors", None, None, slot="cond",
                      sample_request={IN_PLACE_BAKE_PROVEN: True})
    assert seen[-1][0] is True and "sample_request" not in seen[-1][1]  # not a store argument
    assert not in_place_fallback.permitted()  # gone when the load returns
    store_fsdp.ensure(worker, "m.safetensors", None, None, slot="cond", sample_request={})
    store_fsdp.ensure(worker, "m.safetensors", None, None, slot="cond")  # a loader node, a gate cycle
    store_fsdp.ensure(worker, "m.safetensors", None, None, slot="cond",
                      sample_request={IN_PLACE_BAKE_PROVEN: "True"})
    assert [granted for granted, _kwargs in seen[1:]] == [False, False, False]


def test_every_ensure_a_sample_dispatch_makes_hands_over_its_request():
    """The three loads of a sample pass the request; no other caller of the store does."""
    import inspect

    from dgx_monarch.actor import gate_cycle, sample_protocol, worker

    source = inspect.getsource(sample_protocol)
    assert source.count("store_fsdp.ensure(") == 3 == source.count("sample_request=request")
    for module in (gate_cycle, worker):
        assert "sample_request" not in inspect.getsource(module)


def test_the_sample_agreement_hands_the_field_to_the_quote(monkeypatch):
    sent = []
    monkeypatch.setattr(capacity_agreement, "agree_or_refuse",
                        lambda _handle, request, **_kwargs: sent.append(request))
    model = {"unet_name": "m.safetensors", "options": None, "loras": MODEL.loras}
    capacity_agreement.agree_sample(object(), {"model": model, IN_PLACE_BAKE_PROVEN: True})
    capacity_agreement.agree_sample(object(), {"model": model})
    capacity_agreement.agree_sample(object(), {"model": model, IN_PLACE_BAKE_PROVEN: "yes"})
    assert [request["in_place_bake_proven"] for request in sent] == [True, False, False]
    assert all("in_place_bake_proven" not in spec for request in sent for spec in request["specs"])
