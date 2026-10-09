import threading
import time
from dataclasses import replace

from dgx_monarch import operator_readiness as readiness
from dgx_monarch import service_observations as module
from dgx_monarch.config import ClusterConfig, HostConfig


def config():
    return ClusterConfig(hosts=(HostConfig(name="one", address="tcp://127.0.0.1:26600"),))


def fresh(rows):
    return {"state": "fresh", "expected": 1, "expires_at": time.time() + 10,
            "observations": rows}


def verdict(block, actors=None):
    return readiness.readiness_from_telemetry_payload({
        "workers": actors or [{"rank": 0, "world": 1}], "worker_services": block,
        "mesh": {"state": "idle"}, "render": {"active": False},
    })["lifecycle"]["worker_service"]["state"]


def test_service_readiness_requires_fresh_complete_observations():
    row = {"ordinal": 0, "running": True, "listening": True, "healthy": True}
    assert verdict(fresh([row])) == "ready"
    for block in ({"state": "pending"}, {"state": "unavailable"}, fresh([]),
                  {**fresh([row]), "expires_at": time.time() - 1},
                  fresh([{**row, "ordinal": 1}])):
        assert verdict(block) == "unknown"
    assert verdict(fresh([{**row, "healthy": False}])) == "blocked"
    assert verdict(fresh([{**row, "health_error": True}])) == "unknown"
    assert verdict(fresh([row]), [{"setup_cleanup_failed": True}]) == "blocked"
    assert verdict({"state": "pending"}, [{"setup_cleanup_failed": True}]) == "blocked"


def test_single_flight_and_changed_config_discard(monkeypatch):
    cfg = config()
    monkeypatch.setattr(module, "current_config", lambda: (cfg, (1,)))
    started, release = threading.Event(), threading.Event()
    calls = []
    def observe(c):
        calls.append(c)
        started.set()
        assert release.wait(2)
        return [{"ordinal": 0, "healthy": True}]
    monkeypatch.setattr(module, "observe", observe)
    collector = module.ServiceObservations()
    assert collector.snapshot()["state"] == "pending"
    assert started.wait(1)
    for _ in range(20):
        assert collector.snapshot()["state"] == "pending"
    assert len(calls) == 1
    cfg = replace(cfg, source="changed")
    assert collector.snapshot()["state"] == "pending"
    release.set()
    deadline = time.monotonic() + 2
    while collector._inflight and time.monotonic() < deadline:
        time.sleep(.005)
    assert collector._result is None
    assert collector.snapshot()["state"] == "pending"
    while collector._inflight and time.monotonic() < deadline:
        time.sleep(.005)
    assert collector.snapshot()["state"] == "fresh"
    assert len(calls) == 2
    collector.invalidate()
    assert collector._result is None


def test_failure_replaces_prior_success_without_raw_error(monkeypatch):
    monkeypatch.setattr(module, "current_config", lambda: (config(), ()))
    collector = module.ServiceObservations()
    monkeypatch.setattr(module, "observe", lambda cfg: (_ for _ in ()).throw(RuntimeError("secret")))
    collector.snapshot()
    deadline = time.monotonic() + 2
    while collector._inflight and time.monotonic() < deadline:
        time.sleep(.005)
    assert collector.snapshot() == {"state": "unavailable"}


def test_observe_only_calls_passive_health_with_bounded_timeout(monkeypatch):
    from dgx_monarch.cli import lifecycle, worker_health
    seen = []
    def passive(cfg, host, *, runner, timeout):
        seen.append(timeout)
        return {"running": True, "listening": True, "healthy": True, "error": None,
                "raw": "must not leak"}
    monkeypatch.setattr(worker_health, "passive_worker_health", passive)
    for name in ("up", "down", "restart"):
        monkeypatch.setattr(lifecycle, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("mutation")))
    assert module.observe(config()) == [{"ordinal": 0, "running": True, "listening": True,
                                         "healthy": True, "health_error": False}]
    assert seen == [2.0]


def test_expired_cache_refresh_is_pending_not_last_good(monkeypatch):
    monkeypatch.setattr(module, "current_config", lambda: (config(), (1,)))
    started, release = threading.Event(), threading.Event()
    collector = module.ServiceObservations()
    def probe(cfg):
        started.set()
        assert release.wait(2)
        raise RuntimeError("unavailable")
    monkeypatch.setattr(module, "observe", probe)
    collector.snapshot()
    assert started.wait(1)
    with collector._lock:
        collector._result = fresh([{"ordinal": 0, "healthy": True}])
    assert collector.snapshot()["state"] == "fresh"
    with collector._lock:
        collector._result["expires_at"] = time.time() - 1
    assert collector.snapshot() == {"state": "pending"}
    release.set()
    deadline = time.monotonic() + 2
    while collector._inflight and time.monotonic() < deadline:
        time.sleep(.005)
    assert collector.snapshot() == {"state": "unavailable"}


def test_same_config_new_mesh_discards_inflight_result(monkeypatch):
    cohort = (1,)
    monkeypatch.setattr(module, "current_config", lambda: (config(), cohort))
    started, release = threading.Event(), threading.Event()
    def probe(cfg):
        started.set()
        assert release.wait(2)
        return [{"ordinal": 0, "healthy": True}]
    monkeypatch.setattr(module, "observe", probe)
    collector = module.ServiceObservations()
    collector.snapshot()
    assert started.wait(1)
    cohort = (2,)
    assert collector.snapshot() == {"state": "pending"}
    release.set()
    deadline = time.monotonic() + 2
    while collector._inflight and time.monotonic() < deadline:
        time.sleep(.005)
    assert collector._result is None


def test_config_resolves_without_attaching_a_mesh(monkeypatch):
    import sys
    from types import ModuleType

    fake = ModuleType("dgx_monarch.mesh")
    fake._MESHES = {}
    monkeypatch.setitem(sys.modules, "dgx_monarch.mesh", fake)
    monkeypatch.setattr(module, "find_config_path", lambda: "example")
    cfg = config()
    monkeypatch.setattr(module, "load_cluster_config", lambda path: cfg)
    assert module.current_config() == (cfg, ())


def test_actor_errors_and_malformed_cleanup_remain_unknown():
    good = fresh([{"ordinal": 0, "healthy": True}])
    for actor in ({"status_error": "Timeout"}, {"health_error": True},
                  {"error": "failure"}, {"setup_cleanup_failed": "false"},
                  {"rank": True, "world": 1}):
        assert verdict(good, [actor]) == "unknown"
    assert verdict(good, [{"rank": 0, "world": 1, "setup_cleanup_failed": False}]) == "ready"


def test_stuck_probe_becomes_unavailable_without_spawning_replacement(monkeypatch):
    monkeypatch.setattr(module, "current_config", lambda: (config(), (1,)))
    started, release = threading.Event(), threading.Event()
    calls = []
    def probe(cfg):
        calls.append(cfg)
        started.set()
        assert release.wait(2)
        return [{"ordinal": 0, "healthy": True}]
    monkeypatch.setattr(module, "observe", probe)
    collector = module.ServiceObservations()
    collector.snapshot()
    assert started.wait(1)
    with collector._lock:
        collector._started -= 9
    assert collector.snapshot() == {"state": "unavailable"}
    assert len(calls) == 1
    release.set()
    deadline = time.monotonic() + 2
    while collector._inflight and time.monotonic() < deadline:
        time.sleep(.005)


def test_refresh_serves_fresh_prior_result_until_failure(monkeypatch):
    monkeypatch.setattr(module, "current_config", lambda: (config(), (1,)))
    started, release = threading.Event(), threading.Event()
    collector = module.ServiceObservations()
    def probe(cfg):
        started.set()
        assert release.wait(2)
        raise RuntimeError("failure")
    monkeypatch.setattr(module, "observe", probe)
    collector.snapshot()
    assert started.wait(1)
    with collector._lock:
        collector._result = fresh([{"ordinal": 0, "healthy": True}])
    assert collector.snapshot()["state"] == "fresh"
    release.set()
    deadline = time.monotonic() + 2
    while collector._inflight and time.monotonic() < deadline:
        time.sleep(.005)
    assert collector.snapshot() == {"state": "unavailable"}


def test_stale_route_snapshot_cannot_preserve_ready():
    from dgx_monarch.nodes.routes import _stale_snapshot

    payload = {"workers": [{"rank": 0, "world": 1}],
               "worker_services": fresh([{"ordinal": 0, "healthy": True}]),
               "mesh": {"state": "idle", "verdict": "none", "active_leases": 0,
                        "abandoned_samples": 0}, "render": {"active": False}}
    assert _stale_snapshot(payload, "RefreshFailed")["readiness"]["overall"] == "unknown"
    payload["worker_services"]["expires_at"] = time.time() - 1
    assert _stale_snapshot(payload, "RefreshFailed")["readiness"]["lifecycle"]["worker_service"]["state"] == "unknown"
