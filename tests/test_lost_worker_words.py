"""A lost Worker's cause reaches the fault hook and the prompt error in plain words.

The fault hook (mesh_helpers.install_unhandled_fault_hook, installed by
mesh.install_fault_hook) words a fault by the mesh it names: bring-up records
each Worker mesh's name and its boxes (mesh_factory), so a mesh the driver
still holds is worded with its own boxes, one it no longer holds is "a Worker
of an earlier fleet", and a mesh the pack retired or never recorded keeps what
the hook did before. A call's error is recorded where the driver receives it
(worker_box.name) and leads the prompt error at the node method
(nodes.refusal_lead). The texts are the ones captured on CPU against
torchmonarch 0.7.0 and 0.8.0 (tests/test_worker_loss.py).
"""
from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import pytest

import test_worker_loss as captured
from dgx_monarch import (
    driver_footprint,
    mesh_factory,
    mesh_rpc,
    mesh_safety,
    mesh_setup,
    mesh_stop,
    retired_meshes,
    worker_loss,
)
from dgx_monarch.nodes import refusal_lead

PAIR_CONFIG = SimpleNamespace(hosts=tuple(
    SimpleNamespace(name=name, address=f"tcp://{name}:26600") for name in ("box-a", "box-b")))
LIVE_NAME = "dgxm_worker-3Y3EiGJ8Bmi"
HINT = "the kernel's out-of-memory killer sends signal 9, and so does `kill -9`"


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def hook(monkeypatch):
    """The real hook through mesh.install_fault_hook, a fake monarch.actor, toasts captured."""
    import dgx_monarch.mesh as mesh_mod
    from dgx_monarch.log import get_logger

    fake_actor = types.ModuleType("monarch.actor")
    fake_actor.ActorError = type("ActorError", (Exception,), {})
    fake_actor.unhandled_fault_hook = None
    fake_pkg = types.ModuleType("monarch")
    fake_pkg.actor = fake_actor
    monkeypatch.setitem(sys.modules, "monarch", fake_pkg)
    monkeypatch.setitem(sys.modules, "monarch.actor", fake_actor)
    monkeypatch.setattr(mesh_mod, "_FAULT_HOOK_INSTALLED", False)
    monkeypatch.setattr(mesh_mod, "_fault_correlation_state", lambda: (True, False, False))
    monkeypatch.setattr(mesh_mod, "_MESHES", {})
    monkeypatch.setattr(mesh_mod, "_MESH_PENDING", {})
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    monkeypatch.setattr(driver_footprint, "host_is_local", lambda name: name == "box-a")
    toasts: list = []
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = SimpleNamespace(instance=SimpleNamespace(
        send_sync=lambda event, payload: toasts.append((event, payload))))
    monkeypatch.setitem(sys.modules, "server", fake_server)
    retired_meshes.reset()
    worker_loss.reset()
    mesh_mod.install_fault_hook()
    capture = _Capture()
    logger = get_logger("dgx_monarch.mesh")
    logger.addHandler(capture)
    yield SimpleNamespace(fault=fake_actor.unhandled_fault_hook, records=capture.records,
                          toasts=toasts, meshes=mesh_mod._MESHES)
    logger.removeHandler(capture)
    retired_meshes.reset()
    worker_loss.reset()


def _bring_up(name: str, config=PAIR_CONFIG, use_cluster=True, gpus=1) -> SimpleNamespace:
    """What bring-up records, through mesh_factory's own recording step."""
    workers = SimpleNamespace(_name=SimpleNamespace(get=lambda timeout=None: name))
    mesh_factory._record_mesh(workers, config, use_cluster, gpus)
    return SimpleNamespace(workers=workers)


def _errors(records):
    return [r.getMessage() for r in records if r.levelno >= logging.ERROR]


class _Fault(str):
    """A MeshFailure stand-in: its text, and its own mesh_name as Monarch gives it."""

    @property
    def mesh_name(self):
        return self.split("mesh_name=", 1)[1].split(",", 1)[0]


@pytest.mark.parametrize("text", [captured.KILL_FAULT_070.replace("HzZjQgcCSUv", "3Y3EiGJ8Bmi"),
                                  captured.KILL_FAULT_080], ids=["0.7.0", "0.8.0"])
def test_a_held_meshs_killed_worker_opens_with_its_box_and_the_signal(hook, text):
    hook.meshes["cluster"] = _bring_up(LIVE_NAME)
    hook.fault(_Fault(text))
    words = ("The Worker on the sibling was stopped by signal 9 (SIGKILL); " + HINT)
    [error] = _errors(hook.records)
    assert error.startswith("cluster fault: " + words), error
    assert error.endswith(text)  # Monarch's own text still follows, for the log reader
    [(event, payload)] = hook.toasts
    assert event == "dgx-monarch.fault" and payload["message"].startswith(words)
    assert payload["message"].endswith("\n" + text)
    assert not mesh_safety.is_memory_exhaustion(RuntimeError(payload["message"]))


@pytest.mark.parametrize("text, words", [
    (captured.EXIT_FAULT_080.replace("xXfhSbWYyQT", "3Y3EiGJ8Bmi"),
     "The Worker on the sibling exited with code 3."),
    (captured.EXIT_FAULT_070.replace("7kBigQ6FapA", "3Y3EiGJ8Bmi"),
     "The Worker on the sibling exited with code 3."),
    (captured.LIVENESS_FAULT_080.replace("2x9JkQH1rcB", "3Y3EiGJ8Bmi"),
     "The driver lost contact with the sibling."),
])
def test_an_exit_code_or_a_lost_box_reads_plainly(hook, text, words):
    hook.meshes["cluster"] = _bring_up(LIVE_NAME)
    hook.fault(_Fault(text))
    assert _errors(hook.records) == [f"cluster fault: {words} {text}"]
    assert hook.toasts == [("dgx-monarch.fault", {"message": f"{words}\n{text}"})]


def test_a_recorded_mesh_the_driver_no_longer_holds_is_an_earlier_fleets(hook):
    _bring_up(LIVE_NAME)  # recorded at its bring-up; no handle holds it now
    hook.meshes["cluster"] = _bring_up("dgxm_worker-aNewMeshUid")  # today's fleet
    hook.fault(_Fault(captured.KILL_FAULT_080))
    [error] = _errors(hook.records)
    assert error.startswith("cluster fault: A Worker of an earlier fleet was stopped by signal 9"), error
    assert "sibling" not in error.split("MeshFailure(", 1)[0]
    liveness = captured.LIVENESS_FAULT_080.replace("2x9JkQH1rcB", "3Y3EiGJ8Bmi")
    hook.fault(_Fault(liveness))
    assert _errors(hook.records)[-1] == (
        f"cluster fault: The driver lost contact with a Worker of an earlier fleet. {liveness}")


def test_a_retired_meshs_fault_stays_quiet_whatever_it_says(hook):
    handle = _bring_up(LIVE_NAME)
    hook.meshes["cluster"] = handle
    handle.procs = SimpleNamespace(stop=lambda reason: SimpleNamespace(get=lambda timeout=None: None))
    mesh_stop.stop_with_reason(handle, "dgx-monarch recycle", 1)
    hook.fault(_Fault(captured.KILL_FAULT_080))
    assert _errors(hook.records) == [] and hook.toasts == []


@pytest.mark.parametrize("text", [captured.KILL_FAULT_070, captured.SYSTEM_FAULT_080,
                                  captured.UNDELIVERABLE_FAULT_080, captured.CTOR_FAULT_080])
def test_what_is_not_a_held_or_recorded_lost_worker_keeps_the_hooks_words(hook, text):
    hook.meshes["cluster"] = _bring_up(LIVE_NAME)
    hook.fault(text)
    expected = [] if text is captured.SYSTEM_FAULT_080 else [f"cluster fault: {text}"]
    assert _errors(hook.records) == expected
    assert [payload["message"] for _event, payload in hook.toasts] == ([text] if expected else [])


class _NamedFault(str):
    """A MeshFailure whose text carries no mesh name: only its ``mesh_name`` property gives it."""

    def __new__(cls, text, name):
        fault = super().__new__(cls, text)
        fault._name = name
        return fault

    @property
    def mesh_name(self):
        return self._name


def test_inside_the_exit_wrapper_the_setup_mesh_stays_quiet_and_the_worker_mesh_is_worded(hook):
    """The installed hook is the exit wrapper (driver_exit.skipped_at_exit) around the pack's hook.

    Inside it the order holds: Monarch's ``setup-`` mesh's copy of a death is an
    INFO notice (retired_meshes.notice), and the held Worker mesh's fault, named
    only by its ``mesh_name`` property, opens with the words.
    """
    from test_retired_mesh_faults import SETUP_LIVENESS

    assert hook.fault.__wrapped__.__name__ == "hook"  # what mesh.install_fault_hook installed
    hook.meshes["cluster"] = _bring_up(LIVE_NAME)
    hook.fault(_Fault(SETUP_LIVENESS))
    assert _errors(hook.records) == [] and hook.toasts == []
    assert [r.levelno for r in hook.records if "setup-" in r.getMessage()] == [logging.INFO]
    unnamed = captured.KILL_FAULT_080.replace("mesh_name=dgxm_worker-3Y3EiGJ8Bmi", "mesh_name=<none>")
    hook.fault(_NamedFault(unnamed, LIVE_NAME))
    [error] = _errors(hook.records)
    assert error.startswith("cluster fault: The Worker on the sibling was stopped by signal 9"), error


def test_a_local_fleet_names_the_gpu_on_this_box(hook):
    local = SimpleNamespace(hosts=())
    hook.meshes["local"] = _bring_up(LIVE_NAME, config=local, use_cluster=False, gpus=2)
    hook.fault(_Fault(captured.KILL_FAULT_080))
    [error] = _errors(hook.records)
    assert error.startswith("cluster fault: The Worker for GPU 1 on this box was stopped by signal 9"), error


# --- The prompt error ---------------------------------------------------------


def _supervision_error(text: str):
    """Monarch's own SupervisionError, built as Monarch builds it: message and endpoint."""
    pytest.importorskip("monarch.actor")
    from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError

    endpoint, message = text.removeprefix("Endpoint call ").split(" failed, ", 1)
    error = SupervisionError(message, endpoint)
    assert str(error) == text  # the shape the captured text has
    return error


def _call_all_handle(monkeypatch, error):
    monkeypatch.setattr(driver_footprint, "host_is_local", lambda name: name == "box-a")
    handle = SimpleNamespace(world=2, gpus_per_host=1, owns_hosts=True, config=PAIR_CONFIG,
                             active_worker_args={})
    handle.workers = SimpleNamespace(unload=SimpleNamespace(call=lambda *_a, **_k: object()))

    def await_or_evict(_future, _timeout):
        raise error

    handle._await_or_evict = await_or_evict
    monkeypatch.setattr(mesh_setup, "dispatch_endpoint",
                        lambda _handle, _name, send, setup_token=None: send())
    monkeypatch.setattr(mesh_setup, "AMBIGUOUS_MUTATION_ENDPOINTS", frozenset())
    return handle


@pytest.mark.parametrize("text, opening", [
    (captured.KILL_ERROR, "The Worker on the sibling was stopped by signal 9 (SIGKILL); "),
    (captured.EXIT_ERROR, "The Worker on the sibling exited with code 3."),
    (captured.MONITOR_ERROR_080, "The driver lost contact with the sibling."),
    (captured.UNREACHABLE_ERROR_080, "The driver lost contact with the head."),
    (captured.NO_STATUS_ERROR_070, "The driver lost contact with the head."),
    (captured.RESTART_ERROR_080, "The driver lost contact with the head."),
])
def test_the_prompt_error_opens_with_the_lost_workers_cause(monkeypatch, text, opening):
    error = _supervision_error(text)
    handle = _call_all_handle(monkeypatch, error)

    @refusal_lead.lead_with_refusal
    def node_method():
        return mesh_rpc.call_all_bound(handle, "unload")

    with pytest.raises(worker_loss.WorkerLostError) as caught:
        node_method()
    prompt = str(caught.value)
    assert prompt.startswith(opening), prompt
    assert prompt.endswith("\n" + text)
    assert caught.value.__cause__ is error  # Monarch's error stays in the chain
    assert not mesh_safety.is_memory_exhaustion(caught.value)


@pytest.mark.parametrize("text", [captured.RECYCLE_ERROR_070, captured.DRAINING_ERROR_080])
def test_a_deliberate_stops_error_is_left_as_monarch_wrote_it(monkeypatch, text):
    error = _supervision_error(text)
    handle = _call_all_handle(monkeypatch, error)

    @refusal_lead.lead_with_refusal
    def node_method():
        return mesh_rpc.call_all_bound(handle, "unload")

    with pytest.raises(type(error)) as caught:
        node_method()
    assert caught.value is error and str(error) == text


@pytest.mark.parametrize("text", [captured.RECYCLE_ERROR_070, captured.DRAINING_ERROR_080, captured.KILL_ERROR])
def test_the_fleets_boxes_are_resolved_only_for_a_loss(monkeypatch, text):
    """A receive point resolves the fleet's box names (host lookups included) only when the error is a loss."""
    from dgx_monarch import box_names

    error = _supervision_error(text)
    handle = _call_all_handle(monkeypatch, error)
    resolved: list[int] = []
    real = box_names.fleet_labels
    monkeypatch.setattr(box_names, "fleet_labels", lambda h, world: resolved.append(world) or real(h, world))
    with pytest.raises(type(error)):
        mesh_rpc.call_all_bound(handle, "unload")
    assert resolved == ([2] if text == captured.KILL_ERROR else [])


def test_a_render_whose_worker_dies_mid_collect_opens_with_the_cause():
    """The render's own route: a real MeshHandle's sample lease, collect_sample, PendingRender.result.

    The collect re-raises Monarch's error and the render's cleanup chains any
    stronger error ``from`` it, so the words reach the node layer by
    ``__cause__`` alone.
    """
    import torch

    from dgx_monarch.nodes import render_result, render_submit
    from topology_transition_helpers import _acks, _dispatch_sample, _Future, _handle, _Workers

    error = _supervision_error(captured.KILL_ERROR)
    handle, old, _new = _handle(_Workers([_Future(_acks())]))
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(handle, lambda: _Future(error=error), setup_token=token)
    pending = render_submit.PendingRender(
        handle, submission, SimpleNamespace(__exit__=lambda *_args: None), old,
        {"samples": torch.zeros((1, 4, 1, 1))}, 10.0, "render", render_result._finish_render)

    @refusal_lead.lead_with_refusal
    def sample():  # the sampler node's collect
        return pending.result()

    with pytest.raises(worker_loss.WorkerLostError) as caught:
        sample()
    prompt = str(caught.value)
    assert prompt.startswith("The Worker for GPU 1 on this box was stopped by signal 9 (SIGKILL)"), prompt
    assert prompt.endswith("\n" + captured.KILL_ERROR) and caught.value.__cause__ is error
    assert handle.defunct  # the render's eviction ran as before
