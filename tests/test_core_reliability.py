"""Focused regressions for actor and mesh cleanup failure paths."""

import sys
import threading
import time
import types

import pytest


class _Done:
    def get(self, timeout=None):
        return None


def test_local_mesh_shutdown_stops_owned_proc_mesh():
    from dgx_monarch.mesh import MeshHandle

    calls = []
    handle = MeshHandle.__new__(MeshHandle)
    handle.lock = threading.RLock()
    handle.owns_hosts = False
    handle.setup_key = ("topology",)
    handle.workers = types.SimpleNamespace(
        teardown_group=types.SimpleNamespace(
            call=lambda **flags: calls.append(("teardown", flags)) or _Done()
        )
    )
    handle.procs = types.SimpleNamespace(
        stop=lambda reason: calls.append(("stop", reason)) or _Done()
    )

    handle.shutdown()

    # The stop frees the weights, so the workers are told to skip their unload.
    assert calls == [("teardown", {"release_models": False}),
                     ("stop", "dgx-monarch client detach")]
    assert handle.setup_key is None


def test_mesh_shutdown_surfaces_proc_stop_failure_and_retires_setup():
    from dgx_monarch.mesh import MeshHandle

    handle = MeshHandle.__new__(MeshHandle)
    handle.lock = threading.RLock()
    handle.setup_key = ("topology",)
    handle.replacement_blocked = None
    handle.workers = types.SimpleNamespace(
        teardown_group=types.SimpleNamespace(call=lambda: _Done())
    )

    class Failed:
        def get(self, timeout=None):
            raise RuntimeError("stop failed")

    handle.procs = types.SimpleNamespace(stop=lambda _reason: Failed())
    with pytest.raises(RuntimeError, match="replacement is unsafe"):
        handle.shutdown()
    # The setup identity is retired before the destructive group teardown
    # runs, so it never looks reusable even though the proc stop then fails.
    # The stop failure still blocks replacement.
    assert handle.setup_key is None
    assert handle.replacement_blocked
    assert getattr(handle, "teardown_complete", False) is False


def test_stale_config_retries_cached_shutdown_before_replacement(monkeypatch, isolated_environ):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import monarch_launch

    monkeypatch.setattr(monarch_launch, "prepare_driver", lambda _config: None)

    config = types.SimpleNamespace(
        source="cluster.toml",
        hosts=(types.SimpleNamespace(gpus=1),),
        comfy_dir="",
    )
    calls = []

    def shutdown(timeout_s):
        calls.append(timeout_s)
        if len(calls) == 1:
            raise RuntimeError("old procs still alive")

    handle = types.SimpleNamespace(
        defunct=False,
        config_mtime=1.0,
        config_fingerprint="old",
        n_hosts=1,
        gpus_per_host=1,
        shutdown=shutdown,
    )
    key = mesh_mod._mesh_cache_key("cluster.toml", "/comfy", True)
    monkeypatch.setattr(mesh_mod, "find_config_path", lambda _path: "/cluster.toml")
    monkeypatch.setattr(mesh_mod, "load_cluster_config", lambda _path: config)
    monkeypatch.setattr(mesh_mod, "_detect_comfy_dir", lambda _path: "/comfy")
    monkeypatch.setattr(mesh_mod.os.path, "getmtime", lambda _path: 2.0)
    monkeypatch.setattr(mesh_mod, "_MESHES", {key: handle})
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", "cluster")
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)

    with pytest.raises(mesh_mod.MeshAttachError, match="could not be stopped safely"):
        mesh_mod.get_mesh(mode="cluster")
    assert mesh_mod._MESHES == {key: handle}

    def replacement_reached(_config, _handoff):
        raise RuntimeError("replacement attach reached")

    monkeypatch.setattr(mesh_mod, "_attach_cluster", replacement_reached)
    with pytest.raises(RuntimeError, match="replacement attach reached"):
        mesh_mod.get_mesh(mode="cluster")
    assert calls == [30, 30]
    assert mesh_mod._MESHES == {}


def test_temporary_worker_policy_restores_after_partial_apply_failure():
    from dgx_monarch.mesh import MeshHandle

    calls = []

    class Future:
        def __init__(self, fail):
            self.fail = fail

        def get(self, timeout=None):
            if self.fail:
                raise RuntimeError("rank 1 apply failed")
            return {0: {"ok": True}}

    class Endpoint:
        def call(self, args):
            calls.append(dict(args))
            return Future(fail=len(calls) == 1)

    handle = MeshHandle.__new__(MeshHandle)
    handle.config = types.SimpleNamespace(worker_args={"reserve_vram_gb": 2.0})
    handle.lock = threading.RLock()
    handle.workers = types.SimpleNamespace(apply_worker_args=Endpoint())
    handle.setup_key = ("ready",)
    handle.worker_args_key = ("old",)
    handle.active_worker_args = {}
    original = {"lora_low_rss": True, "slab_weights": True}

    with pytest.raises(RuntimeError, match="rank 1 apply failed"):
        with handle.temporary_worker_args(original, {"slab_weights": False}):
            raise AssertionError("target failure must not enter the body")

    assert calls == [
        {"reserve_vram_gb": 2.0, "lora_low_rss": True, "slab_weights": False},
        {"reserve_vram_gb": 2.0, "lora_low_rss": True, "slab_weights": True},
    ]
    assert handle.active_worker_args == calls[-1]
    assert handle.worker_args_key == handle._worker_args_key(calls[-1])


@pytest.mark.parametrize("value", [True, -1, 65, 257])
def test_programmatic_local_gpu_count_is_bounded_before_spawn(monkeypatch, value):
    from dgx_monarch import mesh as mesh_mod

    monkeypatch.setattr(mesh_mod, "find_config_path", lambda _path: None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)
    with pytest.raises(mesh_mod.MeshAttachError, match="gpus_per_host"):
        mesh_mod.get_mesh(mode="local", gpus_per_host=value)


def test_comfy_dir_only_change_retires_same_source_fleet(monkeypatch, isolated_environ):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import monarch_launch

    monkeypatch.setattr(monarch_launch, "prepare_driver", lambda _config: None)

    config = types.SimpleNamespace(
        source="cluster.toml", hosts=(types.SimpleNamespace(gpus=1),), comfy_dir="/new")
    calls = []
    handle = types.SimpleNamespace(
        defunct=False, replacement_blocked=None, config_fingerprint="unreadable",
        n_hosts=1, gpus_per_host=1, comfy_dir="/old",
        shutdown=lambda timeout_s: calls.append(timeout_s))
    key = mesh_mod._mesh_cache_key(config.source, "/new", True)
    monkeypatch.setattr(mesh_mod, "find_config_path", lambda _path: "/cluster.toml")
    monkeypatch.setattr(mesh_mod, "load_cluster_config", lambda _path: config)
    monkeypatch.setattr(mesh_mod, "_detect_comfy_dir", lambda _path: "/new")
    monkeypatch.setattr(mesh_mod, "_MESHES", {key: handle})
    monkeypatch.setattr(mesh_mod, "_MESH_CREATING", {})
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", "cluster")
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)
    monkeypatch.setattr(
        mesh_mod, "spawn_worker_fleet",
        lambda *args: (_ for _ in ()).throw(RuntimeError("replacement reached")))
    with pytest.raises(RuntimeError, match="replacement reached"):
        mesh_mod.get_mesh(mode="cluster")
    assert calls == [30]
    assert mesh_mod._MESHES == {}


def _replaceable_fleet(monkeypatch, config, events):
    """A live cached fleet whose config digest no longer matches, and no spawn."""
    from dgx_monarch import mesh as mesh_mod

    handle = types.SimpleNamespace(
        defunct=False, replacement_blocked=None, config_fingerprint="edited",
        n_hosts=1, gpus_per_host=1, comfy_dir="/comfy",
        shutdown=lambda timeout_s: events.append(("shutdown", timeout_s)))
    key = mesh_mod._mesh_cache_key(config.source, "/comfy", True)
    monkeypatch.setattr(mesh_mod, "find_config_path", lambda _path: "/cluster.toml")
    monkeypatch.setattr(mesh_mod, "load_cluster_config", lambda _path: config)
    monkeypatch.setattr(mesh_mod, "_detect_comfy_dir", lambda _path: "/comfy")
    monkeypatch.setattr(mesh_mod, "_MESHES", {key: handle})
    monkeypatch.setattr(mesh_mod, "_MESH_CREATING", {})
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", "cluster")
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)
    monkeypatch.setattr(
        mesh_mod, "spawn_worker_fleet",
        lambda *args: events.append("spawn") or (_ for _ in ()).throw(RuntimeError("replacement reached")))
    return key, handle


def test_driver_refusal_keeps_the_live_fleet_a_config_edit_would_replace(monkeypatch, isolated_environ):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import monarch_launch
    from dgx_monarch.config import ClusterConfigError

    config = types.SimpleNamespace(
        source="cluster.toml", hosts=(types.SimpleNamespace(gpus=1),), comfy_dir="/comfy")
    events: list = []
    key, handle = _replaceable_fleet(monkeypatch, config, events)

    def refuse(_config):
        events.append("admit")
        raise ClusterConfigError("native RDMA requires an explicit selected target")

    monkeypatch.setattr(monarch_launch, "prepare_driver", refuse)
    with pytest.raises(ClusterConfigError, match="explicit selected target"):
        mesh_mod.get_mesh(mode="cluster")
    # The refusal precedes the recycle: the working fleet is neither stopped
    # nor evicted, and the creation claim is released for the next Queue.
    assert events == ["admit"]
    assert mesh_mod._MESHES == {key: handle}
    assert mesh_mod._MESH_CREATING == {}
    assert mesh_mod._TRANSPORT_POISON is None

    events.clear()
    monkeypatch.setattr(monarch_launch, "prepare_driver", lambda _config: events.append("admit"))
    with pytest.raises(RuntimeError, match="replacement reached"):
        mesh_mod.get_mesh(mode="cluster")
    assert events == ["admit", ("shutdown", 30), "spawn"]
    assert mesh_mod._MESHES == {}


def test_real_native_admission_refuses_before_recycling_the_fleet(monkeypatch, isolated_environ):
    from dataclasses import replace

    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import monarch_policy
    from dgx_monarch.config import ClusterConfig, ClusterConfigError, HostConfig

    host = HostConfig(name="a", address="tcp://10.0.0.1:26600", gpus=1)
    config = replace(ClusterConfig(hosts=(host,), rdma_latent_return=True), source="cluster.toml")
    events: list = []
    key, handle = _replaceable_fleet(monkeypatch, config, events)
    installed = monarch_policy.snapshot()
    with pytest.raises(ClusterConfigError, match="selected target for every participant"):
        mesh_mod.get_mesh(mode="cluster")
    assert events == []
    assert mesh_mod._MESHES == {key: handle}
    assert monarch_policy.snapshot() == installed  # refused before any policy install


_COLLECTOR = "http://127.0.0.1:4318"
REFUSED_DIAGNOSTICS_EDITS = {
    # Monarch is loaded once a fleet exists, so a new endpoint needs a restart.
    "otlp_restart_required": ({"metrics": True}, {
        "metrics": True, "otlp_endpoint": _COLLECTOR, "otlp_logs": True}, None, "otlp_restart_required"),
    # The driver's pre-check names this one before anything is written.
    "ambient_otlp_endpoint_conflict": ({"otlp_endpoint": _COLLECTOR, "otlp_logs": True}, {
        "otlp_endpoint": "http://127.0.0.1:4319", "otlp_logs": True}, _COLLECTOR,
        "ambient_otlp_endpoint_conflict"),
    "instrument_setup_failed": ({}, {"metrics": True}, None, "instrument_setup_failed"),
}
_EVENT = {"phase": "ack", "outcome": "ok", "return_mode": "rdma", "duration_s": 0.1}


def _meter(recorded, *, blocked=None, entered=None, fail=False):
    """A fake Monarch meter; with ``blocked`` its counter waits inside the backend call."""
    def add(*_args):
        if blocked is not None:
            entered.set()
            blocked.wait(5)

    def get_meter():
        if fail:
            raise RuntimeError("meter unavailable")
        return types.SimpleNamespace(create_counter=lambda _name: types.SimpleNamespace(add=add),
                                     create_histogram=lambda _name: types.SimpleNamespace(
                                         record=lambda *args: recorded.append(args)))
    return get_meter


def _stub_native_driver(monkeypatch, get_meter, ambient=None):
    """Run the real driver admission against stubbed native modules from a clean state."""
    import os

    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import monarch_capabilities
    from dgx_monarch import monarch_diagnostics as diagnostics
    from dgx_monarch import monarch_policy as policy

    monkeypatch.setattr(policy, "_ACTIVE", None)
    monkeypatch.setattr(diagnostics, "_EXPORTER", None)
    monkeypatch.setattr(diagnostics, "_STATE", {"metrics": False, "available": True})
    monkeypatch.setattr(diagnostics, "_POLICY_STATE", {})
    for name in (*policy.policy_environment(policy.MonarchPolicy()), "OTEL_EXPORTER_OTLP_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    if ambient:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = ambient
    native: dict = {}
    actor = types.SimpleNamespace(get_meter=get_meter, span=lambda _name: None)
    monkeypatch.setitem(sys.modules, "monarch", types.SimpleNamespace(actor=actor))
    monkeypatch.setitem(sys.modules, "monarch.actor", actor)
    monkeypatch.setitem(sys.modules, "monarch._rust_bindings", types.ModuleType("native"))
    monkeypatch.setitem(sys.modules, "monarch.config", types.SimpleNamespace(
        configure=lambda **values: native.update(values), get_global_config=lambda: native))
    monkeypatch.setattr(mesh_mod, "install_fault_hook", lambda: None)
    monkeypatch.setattr(monarch_capabilities, "assert_supported_runtime", lambda: None)


def _driver_config(policy_values):
    from dataclasses import replace

    from dgx_monarch.config import ClusterConfig, HostConfig
    from dgx_monarch.monarch_policy import parse_policy

    host = HostConfig(name="a", address="tcp://10.0.0.1:26600", gpus=1)
    return replace(ClusterConfig(hosts=(host,), monarch=parse_policy(policy_values)), source="cluster.toml")


def _driver_state():
    import os

    from dgx_monarch import monarch_diagnostics as diagnostics
    from dgx_monarch import monarch_policy as policy

    return diagnostics.status(), policy.snapshot(), os.environ[policy.POLICY_ENV]


def _live_fleet(monkeypatch, old, edited, shutdown):
    """Register a live fleet admitted from ``old``; cluster.toml now reads ``edited``."""
    from dgx_monarch import mesh as mesh_mod

    handle = types.SimpleNamespace(
        defunct=False, replacement_blocked=None, config_fingerprint="old", n_hosts=1,
        gpus_per_host=1, comfy_dir="/comfy", shutdown=shutdown)
    key = mesh_mod._mesh_cache_key(old.source, "/comfy", True)
    current = {"config": edited, "digest": "edited"}
    monkeypatch.setattr(mesh_mod, "find_config_path", lambda _path: "/cluster.toml")
    monkeypatch.setattr(mesh_mod, "load_cluster_config", lambda _path: current["config"])
    monkeypatch.setattr(mesh_mod, "config_fingerprint", lambda _path: current["digest"])
    monkeypatch.setattr(mesh_mod, "_detect_comfy_dir", lambda _path: "/comfy")
    monkeypatch.setattr(mesh_mod, "_MESHES", {key: handle})
    monkeypatch.setattr(mesh_mod, "_MESH_CREATING", {})
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", "cluster")
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)
    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", lambda *a: pytest.fail("a kept fleet was replaced"))
    return key, handle, current


def _wait_for(predicate, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


@pytest.mark.parametrize("case", sorted(REFUSED_DIAGNOSTICS_EDITS))
def test_a_refused_edit_and_its_revert_leave_the_driver_as_its_fleet_admitted_it(
        monkeypatch, isolated_environ, case):
    """rev-pr-c 1: a refusal must not leave the edited policy or a stopped exporter behind."""
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import monarch_diagnostics as diagnostics
    from dgx_monarch import monarch_launch
    from dgx_monarch.config import ClusterConfigError

    before, edited, ambient, refusal = REFUSED_DIAGNOSTICS_EDITS[case]
    _stub_native_driver(monkeypatch, _meter([], fail=case == "instrument_setup_failed"), ambient)
    old = _driver_config({"diagnostics": before})
    try:
        monarch_launch.prepare_driver(old)  # the live fleet's own admission
        expected = _driver_state()
        assert expected[1]["native_readback"] == "verified"
        events: list = []
        key, handle, current = _live_fleet(monkeypatch, old, _driver_config({"diagnostics": edited}),
                                           lambda timeout_s: events.append("shutdown"))

        with pytest.raises(ClusterConfigError, match=refusal):
            mesh_mod.get_mesh(mode="cluster")
        assert events == [] and mesh_mod._MESHES == {key: handle}
        assert _driver_state() == expected
        assert (diagnostics._EXPORTER is not None) is bool(before.get("metrics"))

        # A byte-identical revert reuses the fleet without a second admission.
        current.update(config=old, digest="old")
        assert mesh_mod.get_mesh(mode="cluster") is handle
        assert events == []
        assert _driver_state() == expected
    finally:
        diagnostics.configure()  # stop any exporter thread this test started


@pytest.mark.parametrize("edited", [
    {"diagnostics": {"metrics": False}},
    {"diagnostics": {"metrics": True}, "startup_timeout_s": 240},
], ids=["metrics_off", "startup_budget"])
def test_an_admitted_edit_whose_recycle_refuses_puts_the_live_fleets_admission_back(
        monkeypatch, isolated_environ, edited):
    """Greptile on #19: the old fleet stays registered, so a revert must find its own policy and export."""
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import monarch_diagnostics as diagnostics
    from dgx_monarch import monarch_launch

    recorded: list = []
    _stub_native_driver(monkeypatch, _meter(recorded))
    old = _driver_config({"diagnostics": {"metrics": True}})
    try:
        monarch_launch.prepare_driver(old)
        expected = _driver_state()
        events: list = []

        def refuse(timeout_s):
            events.append("shutdown")
            raise RuntimeError("a render still holds the fleet")

        key, handle, current = _live_fleet(monkeypatch, old, _driver_config(edited), refuse)
        with pytest.raises(mesh_mod.MeshAttachError, match="could not be stopped safely"):
            mesh_mod.get_mesh(mode="cluster")
        assert events == ["shutdown"] and mesh_mod._MESHES == {key: handle}
        assert _driver_state() == expected
        assert diagnostics._EXPORTER is not None and diagnostics._EXPORTER._thread.is_alive()

        # The revert reuses that fleet with no admission, and its export still runs.
        current.update(config=old, digest="old")
        assert mesh_mod.get_mesh(mode="cluster") is handle
        assert events == ["shutdown"] and _driver_state() == expected
        diagnostics._EXPORTER.submit(_EVENT)
        assert _wait_for(lambda: len(recorded) == 1)
    finally:
        diagnostics.configure()


@pytest.mark.parametrize("case", ["otlp_restart_required", "ambient_otlp_endpoint_conflict"])
def test_a_refused_endpoint_edit_never_stops_the_running_exporter(monkeypatch, isolated_environ, case):
    """Greptile on #19: a refusal decided before admission leaves a stuck exporter free to resume."""
    import threading

    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import monarch_diagnostics as diagnostics
    from dgx_monarch import monarch_launch
    from dgx_monarch.config import ClusterConfigError

    before, edited, ambient, _refusal = REFUSED_DIAGNOSTICS_EDITS[case]
    before, edited = {**before, "metrics": True}, {**edited, "metrics": True}
    recorded: list = []
    entered, backend = threading.Event(), threading.Event()
    _stub_native_driver(monkeypatch, _meter(recorded, blocked=backend, entered=entered), ambient)
    old = _driver_config({"diagnostics": before})
    try:
        monarch_launch.prepare_driver(old)
        exporter = diagnostics._EXPORTER
        thread = exporter._thread
        exporter.submit(_EVENT)
        assert entered.wait(2)  # the exporter thread now sits inside a backend call
        expected = _driver_state()
        stops: list = []
        real_stop = diagnostics._MetricExporter.stop
        monkeypatch.setattr(diagnostics._MetricExporter, "stop",
                            lambda self: stops.append(self) or real_stop(self))
        events: list = []
        key, handle, _current = _live_fleet(monkeypatch, old, _driver_config({"diagnostics": edited}),
                                            lambda timeout_s: events.append("shutdown"))

        with pytest.raises(ClusterConfigError, match=case):
            mesh_mod.get_mesh(mode="cluster")
        assert stops == [] and events == [] and mesh_mod._MESHES == {key: handle}
        assert diagnostics._EXPORTER is exporter and exporter._thread is thread
        assert thread.is_alive() and not exporter._stopped.is_set()
        assert _driver_state() == expected

        # When the backend recovers, the same thread finishes that call and keeps exporting.
        backend.set()
        exporter.submit(_EVENT)
        assert _wait_for(lambda: len(recorded) == 2)
    finally:
        backend.set()
        diagnostics.configure()


@pytest.mark.parametrize("failure", ["recycle_refused", "runtime_refused"])
def test_a_restore_that_cannot_restart_a_stuck_exporter_says_so(monkeypatch, isolated_environ, failure):
    """Greptile on #19: an admitted metrics-off edit stopped a stuck exporter, then the edit failed."""
    import logging
    import threading

    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import monarch_capabilities, monarch_launch
    from dgx_monarch import monarch_diagnostics as diagnostics
    from dgx_monarch import monarch_policy as policy
    from dgx_monarch.log import get_logger

    entered, backend = threading.Event(), threading.Event()
    _stub_native_driver(monkeypatch, _meter([], blocked=backend, entered=entered))
    old = _driver_config({"diagnostics": {"metrics": True}})
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = get_logger("dgx_monarch.monarch_diagnostics")
    thread = None
    try:
        monarch_launch.prepare_driver(old)
        thread = diagnostics._EXPORTER._thread
        diagnostics._EXPORTER.submit(_EVENT)
        assert entered.wait(2)
        expected_policy = _driver_state()[1:]
        events: list = []

        def shutdown(timeout_s):
            events.append("shutdown")
            if failure == "recycle_refused":
                raise RuntimeError("a render still holds the fleet")

        key, handle, _current = _live_fleet(
            monkeypatch, old, _driver_config({"diagnostics": {"metrics": False}}), shutdown)
        if failure == "runtime_refused":
            monkeypatch.setattr(monarch_capabilities, "assert_supported_runtime",
                                lambda: (_ for _ in ()).throw(RuntimeError("runtime refused")))
        logger.addHandler(handler)
        raised, text = ((mesh_mod.MeshAttachError, "could not be stopped safely")
                        if failure == "recycle_refused" else (RuntimeError, "runtime refused"))
        with pytest.raises(raised, match=text):
            mesh_mod.get_mesh(mode="cluster")
        logger.removeHandler(handler)
        assert events == (["shutdown"] if failure == "recycle_refused" else [])
        assert mesh_mod._MESHES == {key: handle}
        assert [(r.levelno, r.getMessage()) for r in records if r.levelno >= logging.ERROR] == [(
            logging.ERROR, "Monarch diagnostics requested for this driver are unavailable "
            "(exporter_still_running); it continues without them")]
        state = diagnostics.status()
        assert state["available"] is False and state["reason"] == "exporter_still_running"
        assert _driver_state()[1:] == expected_policy
        assert policy.snapshot()["metrics_requested"] is True
    finally:
        logger.removeHandler(handler)
        backend.set()
        if thread is not None:
            thread.join(2)
        diagnostics.configure()


def test_concurrent_same_source_creation_publishes_one_fleet(monkeypatch, isolated_environ):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import monarch_launch

    monkeypatch.setattr(monarch_launch, "prepare_driver", lambda _config: None)

    entered, release = threading.Event(), threading.Event()
    calls = []

    def spawn(*args):
        calls.append(1)
        entered.set()
        assert release.wait(timeout=2)
        return object(), object(), object(), False

    monkeypatch.setattr(mesh_mod, "find_config_path", lambda _path: None)
    monkeypatch.setattr(mesh_mod, "_detect_comfy_dir", lambda _path: "/comfy")
    monkeypatch.setattr(mesh_mod, "_visible_gpu_count", lambda: 1)
    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", spawn)
    monkeypatch.setattr(mesh_mod, "_MESHES", {})
    monkeypatch.setattr(mesh_mod, "_MESH_CREATING", {})
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)
    results = []

    def create():
        results.append(mesh_mod.get_mesh(mode="local"))

    first = threading.Thread(target=create)
    second = threading.Thread(target=create)
    first.start()
    assert entered.wait(timeout=1)
    second.start()
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert not first.is_alive() and not second.is_alive()
    assert len(calls) == 1 and results[0] is results[1]


def test_stuck_same_source_creation_wait_is_bounded(monkeypatch):
    from dgx_monarch import mesh as mesh_mod

    key = mesh_mod._mesh_cache_key("local", "/comfy", False)
    monkeypatch.setenv("DGXM_MESH_CREATION_TIMEOUT", "0.05")
    monkeypatch.setattr(mesh_mod, "find_config_path", lambda _path: None)
    monkeypatch.setattr(mesh_mod, "_detect_comfy_dir", lambda _path: "/comfy")
    monkeypatch.setattr(mesh_mod, "_visible_gpu_count", lambda: 1)
    monkeypatch.setattr(mesh_mod, "_MESHES", {})
    monkeypatch.setattr(mesh_mod, "_MESH_CREATING", {key: object()})
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)

    started = time.monotonic()
    with pytest.raises(mesh_mod.MeshAttachError, match=r"timed out.*creating this mesh"):
        mesh_mod.get_mesh(mode="local")
    assert time.monotonic() - started < 1.0


def test_unrelated_mesh_creation_key_does_not_wait(monkeypatch):
    from dgx_monarch import mesh_helpers

    condition = threading.Condition()
    monkeypatch.setenv("DGXM_MESH_CREATION_TIMEOUT", "0.05")
    with condition:
        mesh_helpers.wait_for_mesh_creation(
            condition,
            {("cluster", "/other"): object()},
            ("local", "local"),
            RuntimeError,
        )


def test_mesh_creation_wait_uses_public_timeout_monkeypatch_seam(monkeypatch):
    from dgx_monarch import mesh_helpers

    seen = []

    class Condition:
        def wait_for(self, _predicate, timeout):
            seen.append(timeout)
            return True

    monkeypatch.setattr(mesh_helpers, "mesh_creation_timeout_s", lambda: 7.0)
    mesh_helpers.wait_for_mesh_creation(
        Condition(), {}, ("local", "local"), RuntimeError)
    assert seen == [7.0]


@pytest.mark.parametrize("value", ["0", "nan", "inf", "3600.1", "invalid"])
def test_mesh_creation_wait_timeout_rejects_unsafe_environment(monkeypatch, value):
    from dgx_monarch import mesh as mesh_mod

    key = mesh_mod._mesh_cache_key("local", "/comfy", False)
    monkeypatch.setenv("DGXM_MESH_CREATION_TIMEOUT", value)
    monkeypatch.setattr(mesh_mod, "find_config_path", lambda _path: None)
    monkeypatch.setattr(mesh_mod, "_detect_comfy_dir", lambda _path: "/comfy")
    monkeypatch.setattr(mesh_mod, "_visible_gpu_count", lambda: 1)
    monkeypatch.setattr(mesh_mod, "_MESH_CREATING", {key: object()})
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)

    with pytest.raises(mesh_mod.MeshAttachError, match="must be a number"):
        mesh_mod.get_mesh(mode="local")


@pytest.mark.parametrize(
    "mode",
    [
        "primary_cancel",
        "cleanup_cancel",
        "two_cancellations",
        "same_cancellation",
        "ordinary_pair",
        "success_ordinary_cleanup",
    ],
)
def test_gpu_load_profiling_finalizer_is_cancellation_first(
    monkeypatch,
    mode,
):
    from dgx_monarch.actor import comfy_bridge

    ordinary_primary = RuntimeError("gpu load failed")
    ordinary_cleanup = RuntimeError("profiling finalization failed")
    first_cancel = KeyboardInterrupt("gpu load cancelled")
    second_cancel = SystemExit("profiling cancelled")
    if mode == "primary_cancel":
        inner_error, final_error, expected = (
            first_cancel,
            ordinary_cleanup,
            first_cancel,
        )
    elif mode == "cleanup_cancel":
        inner_error, final_error, expected = (
            ordinary_primary,
            second_cancel,
            second_cancel,
        )
    elif mode == "two_cancellations":
        inner_error, final_error, expected = (
            first_cancel,
            second_cancel,
            first_cancel,
        )
    elif mode == "same_cancellation":
        inner_error = final_error = expected = first_cancel
    elif mode == "ordinary_pair":
        inner_error, final_error, expected = (
            ordinary_primary,
            ordinary_cleanup,
            ordinary_primary,
        )
    else:
        inner_error, final_error, expected = None, ordinary_cleanup, None

    mm = types.ModuleType("comfy.model_management")

    def load_models_gpu(*_args, **_kwargs):
        if inner_error is not None:
            raise inner_error
        return "loaded"

    mm.load_models_gpu = load_models_gpu
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(
        comfy_bridge,
        "_record_gpu_load_profile",
        lambda *_args: (_ for _ in ()).throw(final_error),
    )
    comfy_bridge._instrument_gpu_load()

    caught = None
    try:
        result = mm.load_models_gpu()
    except BaseException as exc:
        caught = exc
        result = None

    if expected is None:
        assert caught is None and result == "loaded"
    else:
        assert caught is expected
        assert caught.__cause__ is not caught


def test_custom_node_import_cancellation_removes_partial_module(
    tmp_path,
    monkeypatch,
):
    from dgx_monarch.actor import comfy_custom_nodes

    pack = tmp_path / "cancelled_pack.py"
    pack.write_text("# loader is replaced by the test\n")
    sys_name = str(pack.with_suffix(""))
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_folder_paths = lambda _kind: [str(tmp_path)]
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(
        comfy_custom_nodes,
        "ensure_prompt_server_stub",
        lambda: True,
    )
    cancellation = KeyboardInterrupt("custom node import cancelled")

    class Loader:
        def exec_module(self, _module):
            raise cancellation

    monkeypatch.setattr(
        comfy_custom_nodes.importlib.util,
        "spec_from_file_location",
        lambda *_args: types.SimpleNamespace(loader=Loader()),
    )
    monkeypatch.setattr(
        comfy_custom_nodes.importlib.util,
        "module_from_spec",
        lambda *_args: types.ModuleType(sys_name),
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        comfy_custom_nodes.load_custom_node_modules()

    assert caught.value is cancellation
    assert sys_name not in sys.modules


def test_parallel_teardown_attempts_world_destroy_after_model_group_failure(monkeypatch):
    calls = []
    worker = types.SimpleNamespace(
        store=types.SimpleNamespace(unload_all=lambda: calls.append("unload")),
        _attn=types.SimpleNamespace(invalidate=lambda: calls.append("attention")),
    )

    def fail_model_parallel():
        calls.append("model")
        raise RuntimeError("model group stuck")

    xfuser = types.ModuleType("xfuser")
    core = types.ModuleType("xfuser.core")
    distributed = types.ModuleType("xfuser.core.distributed")
    parallel_state = types.ModuleType("xfuser.core.distributed.parallel_state")
    parallel_state.destroy_model_parallel = fail_model_parallel
    parallel_state.destroy_distributed_environment = lambda: calls.append("world")
    xfuser.core = core
    core.distributed = distributed
    distributed.parallel_state = parallel_state
    monkeypatch.setitem(sys.modules, "xfuser", xfuser)
    monkeypatch.setitem(sys.modules, "xfuser.core", core)
    monkeypatch.setitem(sys.modules, "xfuser.core.distributed", distributed)
    monkeypatch.setitem(
        sys.modules,
        "xfuser.core.distributed.parallel_state",
        parallel_state,
    )

    from dgx_monarch.actor import worker_env

    with pytest.raises(RuntimeError, match="parallel-state teardown incomplete"):
        worker_env.teardown_parallel_state(worker)
    assert calls == ["unload", "attention", "model", "world"]


def test_parallel_teardown_without_model_release_closes_only_checkpoint_pins(monkeypatch):
    """Recycle and shutdown stop the process next, and its exit frees the
    weights; comfy's unload would first copy them to host RAM (70 s for an
    Ideogram4 pair on hardware, 2026-10-06). The pin still closes, because a
    killed process would leave its alias on disk."""
    calls = []
    pin = types.SimpleNamespace(close=lambda: calls.append("pin"))
    worker = types.SimpleNamespace(
        store=types.SimpleNamespace(
            unload_all=lambda: calls.append("unload"),
            current=types.SimpleNamespace(fsdp_checkpoint_pin=pin),
            uncond=types.SimpleNamespace(fsdp_checkpoint_pin=None)),
        _attn=types.SimpleNamespace(invalidate=lambda: calls.append("attention")),
    )
    parallel_state = types.ModuleType("xfuser.core.distributed.parallel_state")
    parallel_state.destroy_model_parallel = lambda: calls.append("model")
    parallel_state.destroy_distributed_environment = lambda: calls.append("world")
    for name in ("xfuser", "xfuser.core", "xfuser.core.distributed"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, parallel_state.__name__, parallel_state)

    from dgx_monarch.actor import worker_env

    worker_env.teardown_parallel_state(worker, release_models=False)
    assert calls == ["pin", "attention", "model", "world"]

    calls.clear()
    worker_env.teardown_parallel_state(worker)
    assert calls == ["unload", "attention", "model", "world"]


def test_parallel_teardown_baseexception_attempts_every_stage_and_preserves_primary(
        monkeypatch):
    calls = []

    class StopNow(BaseException):
        pass

    primary = StopNow("interrupt teardown")

    def fail_unload():
        calls.append("unload")
        raise primary

    def fail_attention():
        calls.append("attention")
        raise RuntimeError("attention cleanup also failed")

    worker = types.SimpleNamespace(
        store=types.SimpleNamespace(unload_all=fail_unload),
        _attn=types.SimpleNamespace(invalidate=fail_attention),
    )
    xfuser = types.ModuleType("xfuser")
    core = types.ModuleType("xfuser.core")
    distributed = types.ModuleType("xfuser.core.distributed")
    parallel_state = types.ModuleType("xfuser.core.distributed.parallel_state")
    parallel_state.destroy_model_parallel = lambda: calls.append("model")
    parallel_state.destroy_distributed_environment = lambda: calls.append("world")
    xfuser.core = core
    core.distributed = distributed
    distributed.parallel_state = parallel_state
    monkeypatch.setitem(sys.modules, "xfuser", xfuser)
    monkeypatch.setitem(sys.modules, "xfuser.core", core)
    monkeypatch.setitem(sys.modules, "xfuser.core.distributed", distributed)
    monkeypatch.setitem(
        sys.modules, "xfuser.core.distributed.parallel_state", parallel_state)

    from dgx_monarch.actor import worker_env

    with pytest.raises(StopNow) as caught:
        worker_env.teardown_parallel_state(worker)

    assert caught.value is primary
    assert calls == ["unload", "attention", "model", "world"]
    assert any(
        "model unload: StopNow('interrupt teardown')" in note
        and "attention dispatch: RuntimeError('attention cleanup also failed')" in note
        for note in getattr(primary, "__notes__", ())
    )


def test_parallel_teardown_hostile_diagnostics_cannot_mask_cancellation(
    monkeypatch,
):
    calls = []

    class HostileCancellation(KeyboardInterrupt):
        def __repr__(self):
            raise RuntimeError("repr is hostile")

        def add_note(self, _note):
            raise RuntimeError("add_note is hostile")

    ordinary = RuntimeError("ordinary model unload failure")
    cancellation = HostileCancellation("attention teardown cancelled")
    worker = types.SimpleNamespace(
        store=types.SimpleNamespace(
            unload_all=lambda: (_ for _ in ()).throw(ordinary)),
        _attn=types.SimpleNamespace(
            invalidate=lambda: (_ for _ in ()).throw(cancellation)),
    )
    xfuser = types.ModuleType("xfuser")
    core = types.ModuleType("xfuser.core")
    distributed = types.ModuleType("xfuser.core.distributed")
    parallel_state = types.ModuleType("xfuser.core.distributed.parallel_state")
    parallel_state.destroy_model_parallel = lambda: calls.append("model")
    parallel_state.destroy_distributed_environment = lambda: calls.append("world")
    xfuser.core = core
    core.distributed = distributed
    distributed.parallel_state = parallel_state
    monkeypatch.setitem(sys.modules, "xfuser", xfuser)
    monkeypatch.setitem(sys.modules, "xfuser.core", core)
    monkeypatch.setitem(sys.modules, "xfuser.core.distributed", distributed)
    monkeypatch.setitem(
        sys.modules,
        "xfuser.core.distributed.parallel_state",
        parallel_state,
    )

    from dgx_monarch.actor import worker_env

    with pytest.raises(HostileCancellation) as caught:
        worker_env.teardown_parallel_state(worker)

    assert caught.value is cancellation
    assert caught.value.__cause__ is ordinary
    assert calls == ["model", "world"]


def test_fleet_to_collective_setup_reloads_model_and_attention(monkeypatch, isolated_environ):
    """A same-kernel topology switch must not retain world-1 group state."""
    import torch

    from dgx_monarch.actor import comfy_bridge, worker_env, worker_status
    from dgx_monarch.actor import worker as worker_mod

    identity = {"digest": "model", "comfy": "commit", "artifacts": []}
    process_group = {"initialized": False, "generation": 0}
    attention_impls = []

    def init_process_group(*_args, **_kwargs):
        process_group["initialized"] = True
        process_group["generation"] += 1

    def destroy_distributed_environment():
        process_group["initialized"] = False

    class BoundAttention:
        def __init__(self):
            self.group_generation = process_group["generation"]

        def __call__(self):
            if (not process_group["initialized"]
                    or self.group_generation != process_group["generation"]):
                raise ValueError("ProcessGroup is not registered")
            return self.group_generation

    def make_attention(_kernel, _sync):
        impl = BoundAttention()
        attention_impls.append(impl)
        return impl

    monkeypatch.setattr(worker_env, "ensure_comfy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", True)
    provenance_snapshots = []
    monkeypatch.setattr(
        worker_status,
        "runtime_provenance_snapshot",
        lambda: provenance_snapshots.append(
            {"schema": 1, "capture": len(provenance_snapshots) + 1}
        )
        or provenance_snapshots[-1],
    )
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", lambda args: dict(args))
    monkeypatch.setattr("dgx_monarch.config.fixup_fabric_ifaces", lambda env: (env, ""))
    monkeypatch.setattr("dgx_monarch.adapters.make_usp_attention", make_attention)
    monkeypatch.setattr(torch, "set_num_threads", lambda _count: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "test-gpu")
    monkeypatch.setattr(
        torch.distributed, "is_initialized", lambda: process_group["initialized"])
    monkeypatch.setattr(torch.distributed, "init_process_group", init_process_group)
    monkeypatch.setattr(worker_env.rendezvous, "generation_store", lambda *_a, **_k: None)

    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.init_distributed_environment = lambda **_kwargs: None
    distributed.initialize_model_parallel = lambda **_kwargs: None
    parallel_state = types.ModuleType("xfuser.core.distributed.parallel_state")
    parallel_state.destroy_model_parallel = lambda: None
    parallel_state.destroy_distributed_environment = destroy_distributed_environment
    distributed.parallel_state = parallel_state
    core = types.ModuleType("xfuser.core")
    core.distributed = distributed
    xfuser = types.ModuleType("xfuser")
    xfuser.core = core
    monkeypatch.setitem(sys.modules, "xfuser", xfuser)
    monkeypatch.setitem(sys.modules, "xfuser.core", core)
    monkeypatch.setitem(sys.modules, "xfuser.core.distributed", distributed)
    monkeypatch.setitem(
        sys.modules, "xfuser.core.distributed.parallel_state", parallel_state)

    class Store:
        def __init__(self):
            self.current = None
            self.uncond = None
            self.swap_verify = 2
            self.lora_low_rss = False
            self.slab_weights = False
            self.family_override = None
            self.unloads = 0

        def set_lora_mode(self, value):
            self.lora_low_rss = bool(value)

        def set_slab_mode(self, value, reason=""):
            self.slab_weights = value

        def unload_all(self):
            self.unloads += 1
            self.current = self.uncond = None

        def ensure(self, *_args, **_kwargs):
            if self.current is not None:
                return self.current.patcher, "reuse"
            patcher = types.SimpleNamespace(
                model_options={},
                model=types.SimpleNamespace(diffusion_model=types.SimpleNamespace()))
            self.current = types.SimpleNamespace(
                patcher=patcher, artifact_identity=identity,
                attention_impl=worker._attn._impl,
            )
            return patcher, "load"

    worker = worker_mod.GPUWorker.__new__(worker_mod.GPUWorker)
    worker.store = Store()
    worker._setup_key = None
    worker._setup_cleanup_failed = False
    worker._attn = worker_env._AttentionDispatch()
    worker._latent_return = types.SimpleNamespace()
    worker.rank = worker.world = None
    worker.topology = {}
    worker._active_worker_args = {}
    worker._setup_info = lambda: {"rank": worker.rank, "world": worker.world}
    worker._slab_mode_effective = lambda *_args: False
    worker._nccl_launch_order_needed = lambda _topology: False
    worker._check_uma_reserve = lambda: None

    def env(world, port, ulysses):
        return {
            "setup_generation": 1 if world == 1 else 2,
            "world": world, "rank": 0, "master_addr": "127.0.0.1",
            "master_port": port,
            "topology": {"ulysses": ulysses, "ring": 1, "cfg": 1,
                         "dp": 1, "fsdp": False},
            "fabric_env": {}, "comfy_dir": "/comfy", "worker_args": {},
            "local_gpu_index": 0, "gpus_per_host": 2,
            "attention": "TORCH_FLASH", "sync_ulysses": True,
        }

    worker_env.setup_impl(worker, env(1, 23001, 1))
    assert worker._setup_provenance == (
        1,
        {"schema": 1, "capture": 1},
    )
    assert worker.store.unloads == 0
    fleet_attention = worker._attn._impl
    fleet_model = types.SimpleNamespace(
        patcher=types.SimpleNamespace(
            model_options={},
            model=types.SimpleNamespace(diffusion_model=types.SimpleNamespace())),
        artifact_identity=identity, attention_impl=fleet_attention,
    )
    worker.store.current = fleet_model

    worker_env.setup_impl(worker, env(2, 23002, 2))
    assert worker._setup_provenance == (
        2,
        {"schema": 1, "capture": 2},
    )
    collective_attention = worker._attn._impl
    assert worker.store.current is None
    assert worker.store.unloads == 1
    assert collective_attention is not fleet_attention
    assert [impl.group_generation for impl in attention_impls] == [1, 2]
    with pytest.raises(ValueError, match="ProcessGroup is not registered"):
        fleet_attention()
    assert worker._attn() == 2

    monkeypatch.setattr(worker_mod, "request_artifact_identity", lambda *_args: identity)
    monkeypatch.setattr(worker_mod, "model_sampling_render_clone", lambda patcher, _spec: patcher)
    monkeypatch.setattr(worker_mod, "run_ksampler", lambda *_args, **_kwargs: (torch.zeros(1), None))
    monkeypatch.setattr(worker_mod, "_latent_signature", lambda _tensor: {"digest": "latent"})
    monkeypatch.setattr(comfy_bridge, "gpu_load_seconds_reset", lambda: 0.0)
    monkeypatch.setattr("dgx_monarch.actor.sampling._dp_info", lambda: (0, 1))
    result = worker_mod.GPUWorker._sample_impl.__wrapped__(worker, {
        "kind": "ksampler", "model": {"unet_name": "model", "loras": []},
        "_dgxm_artifact_sets": [identity], "sage_kernel": "TORCH_FLASH",
        "_dgxm_normal_residency_mode": "operator_off",
        "sync_ulysses": True,
    })

    assert result["transition"] == "load"
    assert worker.store.current is not fleet_model
    assert worker.store.current.attention_impl is collective_attention


def test_worker_teardown_retires_setup_key_when_parallel_state_has_residue():
    from dgx_monarch.actor.worker import GPUWorker

    worker = GPUWorker.__new__(GPUWorker)
    worker._setup_key = ("live",)
    worker._setup_cleanup_failed = False
    worker.rank, worker.world = 0, 2
    worker.topology = {"ulysses": 2}
    worker._latent_return = object()

    def fail():
        raise RuntimeError("parallel-state teardown incomplete")

    worker._teardown_parallel_state = fail
    with pytest.raises(RuntimeError, match="incomplete"):
        GPUWorker._teardown_group_impl(worker)
    assert worker._setup_key is None
    assert worker.rank is None and worker.world is None and worker.topology == {}
    assert worker._setup_cleanup_failed is True
