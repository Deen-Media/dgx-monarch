"""Late faults from a mesh the pack retired, or from Monarch's own meshes, stay quiet.

On torchmonarch 0.8.0 each heal wrote about three times the "cluster fault"
ERROR lines of 0.7.0 (18 against 5 over three heals): "supervision liveness
failed" for the mesh the heal was retiring, and one per rank for Monarch's
internal ``log_level-<uid>`` mesh; every lost host also wrote two for the
``setup-<uid>`` mesh Monarch spawns when ``spawn_procs`` gets a bootstrap,
which production always passes. The texts below were captured on CPU in those
runs, with uids, host names and paths replaced. A failure of a live
``dgxm_worker`` mesh, or of a name the pack never recorded, must still be an
ERROR with a browser toast, and an internal mesh's notice never calls itself
deliberate.
"""
from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import pytest

from dgx_monarch import mesh_safety, mesh_stop, retired_meshes

LIVENESS = ("MeshFailure(mesh_name={name}, crashed_ranks=[0], event=Supervision event: actor "
            "<root>.<dgx_monarch.actor.worker.GPUWorker dgxm_worker{{'hosts': 0/1, 'gpus': 0/2}}> "
            "failed:\n  supervision liveness failed)")
LOG_LEVEL = ("MeshFailure(mesh_name=log_level-Ab12Cd34Ef5, crashed_ranks=[1], event=Supervision "
             "event: actor <root>.<monarch._src.actor.logging._LogLevelSetter log_level{'hosts': "
             "0/1, 'gpus': 1/2}> failed:\n  supervision liveness failed)")
SETUP_LIVENESS = ("MeshFailure(mesh_name=setup-xE4MMbcoZ2, crashed_ranks=[0], event=Supervision event: "
                  "actor <root>.<monarch._src.actor.proc_mesh.SetupActor setup{'hosts': 0/1, 'gpus': 0/2}> "
                  "failed:\n  supervision liveness failed)")
SETUP_UNREACHABLE = ("MeshFailure(mesh_name=setup-ppfCfjPtUfF, crashed_ranks=[0], event=Supervision event: "
                     "actor <root>.<monarch._src.actor.proc_mesh.SetupActor setup{'hosts': 0/1, 'gpus': "
                     "0/2}> failed:\n  host unreachable: no status received from process within 60s)")
KILLED = ("MeshFailure(mesh_name={name}, crashed_ranks=[1], event=Supervision event: actor "
          "<root>.<dgx_monarch.actor.worker.GPUWorker dgxm_worker{{'gpus': 1/2}}> failed:\n"
          "  the process this actor was running on failed: Killed(sig=9))")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def hook(monkeypatch):
    """The real fault hook against a fake monarch.actor, with toasts and the mesh log captured."""
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
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    toasts: list = []
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = SimpleNamespace(instance=SimpleNamespace(
        send_sync=lambda event, payload: toasts.append((event, payload))))
    monkeypatch.setitem(sys.modules, "server", fake_server)
    retired_meshes.reset()
    mesh_mod.install_fault_hook()
    capture = _Capture()
    logger = get_logger("dgx_monarch.mesh")
    logger.addHandler(capture)
    yield fake_actor.unhandled_fault_hook, capture.records, toasts
    logger.removeHandler(capture)
    retired_meshes.reset()


def _errors(records):
    return [r.getMessage() for r in records if r.levelno >= logging.ERROR]


def _retire(name: str) -> None:
    workers = object()
    retired_meshes.remember(workers, name)
    calls: list = []
    handle = SimpleNamespace(workers=workers, procs=SimpleNamespace(
        stop=lambda reason: SimpleNamespace(get=lambda timeout=None: calls.append(reason))))
    mesh_stop.stop_with_reason(handle, "dgx-monarch recycle", 1)  # the retirement point
    assert calls == ["dgx-monarch recycle"]


def test_a_live_worker_mesh_failure_is_an_error_with_a_toast(hook):
    fault, records, toasts = hook
    retired_meshes.remember(object(), "dgxm_worker-Live0000001")
    text = KILLED.format(name="dgxm_worker-Live0000001")
    fault(text)
    assert _errors(records) == [f"cluster fault: {text}"]
    assert toasts == [("dgx-monarch.fault", {"message": text})]


def test_a_retired_meshs_late_liveness_failure_is_info_without_a_toast(hook):
    fault, records, toasts = hook
    _retire("dgxm_worker-Gone0000001")
    fault(LIVENESS.format(name="dgxm_worker-Gone0000001"))
    assert _errors(records) == [] and toasts == []
    assert any("teardown notice" in r.getMessage() and r.levelno == logging.INFO for r in records)


@pytest.mark.parametrize("text", [LOG_LEVEL, SETUP_LIVENESS, SETUP_UNREACHABLE])
def test_monarchs_internal_meshes_are_info_without_a_toast_and_not_deliberate(hook, text):
    fault, records, toasts = hook
    fault(text)
    assert _errors(records) == [] and toasts == []
    [notice] = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert notice.startswith(retired_meshes.INTERNAL_NOTICE) and "deliberate" not in notice


def test_a_live_worker_failure_beside_a_setup_fault_still_shows(hook):
    """A real death still reaches the user when Monarch's setup mesh fails with it."""
    fault, records, toasts = hook
    retired_meshes.remember(object(), "dgxm_worker-Live0000001")
    worker = KILLED.format(name="dgxm_worker-Live0000001")
    fault(SETUP_LIVENESS)
    fault(worker)
    fault(SETUP_LIVENESS.replace("crashed_ranks=[0]", "crashed_ranks=[1]"))
    assert _errors(records) == [f"cluster fault: {worker}"]
    assert toasts == [("dgx-monarch.fault", {"message": worker})]


CONTROLLER_STOP_FAILURE = (
    "MeshFailure(mesh_name=<none>, crashed_ranks=[], event=Supervision event: actor "
    "proc_mesh_controller_dgxm-procs-testmesh-ckuojgrl1zy<BRGbzmJ9tTX>.local<fg13gHugDYh>@tcp://127.0.0.1:35109 "
    "failed:\n  error stopping proc mesh: statuses: Timeout(30.495408301s)=0..1;Stopped=1..2)")


def test_monarchs_report_on_a_registered_stop_is_info_and_reaches_that_stop(hook):
    from dgx_monarch import proc_stops

    fault, records, toasts = hook
    procs = object()
    proc_stops.remember(procs, "dgxm-procs-testmesh")
    stop = proc_stops.started(procs, keep_s=60)
    fault(CONTROLLER_STOP_FAILURE)
    assert _errors(records) == [] and toasts == []
    [notice] = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert notice.startswith(retired_meshes.STOP_NOTICE)
    assert stop.ended.is_set() and stop.report == CONTROLLER_STOP_FAILURE


@pytest.mark.parametrize("registered", ["none", "another"])
def test_a_stop_report_naming_no_registered_stop_stays_an_error(hook, registered):
    from dgx_monarch import proc_stops

    fault, records, toasts = hook
    if registered == "another":
        procs = object()
        proc_stops.remember(procs, "dgxm-procs-othermesh")
        stop = proc_stops.started(procs, keep_s=60)
    fault(CONTROLLER_STOP_FAILURE)
    assert _errors(records) == [f"cluster fault: {CONTROLLER_STOP_FAILURE}"] and len(toasts) == 1
    if registered == "another":
        assert not stop.ended.is_set()


class _Failure:
    """A MeshFailure stand-in: its own mesh_name property beside its text."""

    def __init__(self, text: str, name):
        self._text, self._name = text, name

    @property
    def mesh_name(self):
        if isinstance(self._name, BaseException):
            raise self._name
        return self._name

    def __str__(self) -> str:
        return self._text


def test_the_hook_reads_the_failures_mesh_name_before_its_text(hook):
    fault, records, toasts = hook
    _retire("dgxm_worker-Gone0000001")
    # A text the regex cannot read: only the property names the retired mesh.
    text = LIVENESS.format(name="dgxm_worker-Gone0000001").replace("MeshFailure(mesh_name=", "Failure(")
    fault(_Failure(text, "dgxm_worker-Gone0000001"))
    assert _errors(records) == [] and toasts == []
    # The property outranks a text naming a retired mesh.
    live = LIVENESS.format(name="dgxm_worker-Gone0000001")
    fault(_Failure(live, "dgxm_worker-Live0000001"))
    assert _errors(records) == [f"cluster fault: {live}"] and len(toasts) == 1


@pytest.mark.parametrize("name", [RuntimeError("binding gone"), None, ""])
def test_an_unreadable_property_falls_back_to_the_text(hook, name):
    fault, records, toasts = hook
    fault(_Failure(SETUP_LIVENESS, name))
    assert _errors(records) == [] and toasts == []


@pytest.mark.parametrize("text", [
    SETUP_LIVENESS.replace("monarch._src.actor.proc_mesh.SetupActor", "pkg.SetupLike"),
    LOG_LEVEL.replace("monarch._src.actor.logging._LogLevelSetter", "pkg.Other"),
])
def test_an_internal_name_without_its_actor_class_stays_an_error(hook, text):
    fault, records, toasts = hook
    fault(text)
    assert _errors(records) == [f"cluster fault: {text}"] and len(toasts) == 1


@pytest.mark.parametrize("text", [
    LIVENESS.format(name="dgxm_worker-Never000001"),  # a name the pack never retired
    LIVENESS.replace("MeshFailure(mesh_name={name},", "MeshFailure(mesh_name=<none>,"),
    "Supervision event: actor <root>.<GPUWorker dgxm_worker> failed: no mesh name here",
])
def test_an_unrecorded_or_unreadable_name_stays_an_error(hook, text):
    fault, records, toasts = hook
    _retire("dgxm_worker-Gone0000001")
    fault(text)
    assert _errors(records) == [f"cluster fault: {text}"]
    assert len(toasts) == 1


def test_only_the_retired_name_goes_quiet():
    retired_meshes.reset()
    workers = object()
    retired_meshes.remember(workers, "dgxm_worker-Gone0000001")
    assert not retired_meshes.quiet(LIVENESS.format(name="dgxm_worker-Gone0000001"))
    retired_meshes.retire(workers)
    assert retired_meshes.quiet(LIVENESS.format(name="dgxm_worker-Gone0000001"))
    assert not retired_meshes.quiet(LIVENESS.format(name="dgxm_worker-Gone00000012"))
    assert not retired_meshes.quiet(None)
    retired_meshes.reset()
