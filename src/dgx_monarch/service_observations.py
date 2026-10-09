"""Bounded passive Worker service observations for driver telemetry."""
from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict

from .config import ClusterConfig, find_config_path, load_cluster_config


def current_config() -> tuple[ClusterConfig | None, tuple[object, ...]]:
    from .mesh import _MESHES

    handles = list(_MESHES.values())
    if handles:
        configs = [getattr(handle, "config", None) for handle in handles]
        if not all(isinstance(config, ClusterConfig) for config in configs):
            return None, ()
        if any(config != configs[0] for config in configs[1:]):
            return None, ()
        return configs[0], tuple(handles)
    path = find_config_path()
    return (load_cluster_config(path) if path else None), ()


def observe(config: ClusterConfig) -> list[dict]:
    from .cli.lifecycle import run_on_host
    from .cli.worker_health import passive_worker_health

    rows = []
    timeout = min(2.0, 8.0 / len(config.hosts))
    for ordinal, host in enumerate(config.hosts):
        health = passive_worker_health(config, host, runner=run_on_host, timeout=timeout)
        rows.append({
            "ordinal": ordinal,
            "running": health.get("running"),
            "listening": health.get("listening"),
            "healthy": health.get("healthy"),
            "health_error": bool(health.get("error")),
        })
    return rows


class ServiceObservations:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: str | None = None
        self._cohort: tuple[object, ...] = ()
        self._generation = 0
        self._inflight = False
        self._started = float("-inf")
        self._finished = float("-inf")
        self._result: dict | None = None

    def invalidate(self) -> None:
        with self._lock:
            self._generation += 1
            self._key = None
            self._cohort = ()
            self._result = None
            self._finished = float("-inf")

    def snapshot(self) -> dict:
        try:
            config, cohort = current_config()
            if config is None or not 1 <= len(config.hosts) <= 32:
                self.invalidate()
                return {"state": "unavailable"}
            key = hashlib.sha256(repr((asdict(config), tuple(id(handle) for handle in cohort))).encode()).hexdigest()
        except Exception:
            self.invalidate()
            return {"state": "unavailable"}
        with self._lock:
            if key != self._key:
                self._generation += 1
                self._key, self._result = key, None
                self._cohort = cohort
                self._finished = float("-inf")
            if self._inflight:
                if self._result is not None and service_rows(self._result) is not None:
                    return dict(self._result)
                return {"state": "unavailable" if time.monotonic() - self._started > 8.0 else "pending"}
            if time.monotonic() - self._finished < 5.0 and self._result is not None:
                return dict(self._result)
            self._inflight = True
            self._started = time.monotonic()
            generation = self._generation
            try:
                threading.Thread(target=self._refresh, args=(config, key, generation, observe), daemon=True).start()
            except Exception:
                self._inflight = False
                self._finished = time.monotonic()
                self._result = {"state": "unavailable"}
                return dict(self._result)
            if self._result is not None and service_rows(self._result) is not None:
                return dict(self._result)
            return {"state": "pending"}

    def _refresh(self, config: ClusterConfig, key: str, generation: int,
                 probe: Callable[[ClusterConfig], list[dict]]) -> None:
        started = time.monotonic()
        expires = time.time() + 10.0
        try:
            rows = probe(config)
            result = {"state": "fresh", "expected": len(config.hosts),
                      "expires_at": expires, "observations": rows}
            if time.monotonic() - started > 8.0:
                result = {"state": "unavailable"}
        except Exception:
            result = {"state": "unavailable"}
        with self._lock:
            self._inflight = False
            if key == self._key and generation == self._generation:
                self._finished = time.monotonic()
                self._result = result


services = ServiceObservations()

def service_rows(block: object) -> list | None:
    if not isinstance(block, Mapping) or block.get("state") != "fresh":
        return None
    expires = block.get("expires_at")
    expected = block.get("expected")
    rows = block.get("observations")
    if (
        isinstance(expires, bool) or not isinstance(expires, (float, int))
        or not 0 <= expires <= 10**12 or not math.isfinite(expires)
        or not time.time() <= expires <= time.time() + 15
        or type(expected) is not int or not 1 <= expected <= 32
        or not isinstance(rows, list) or len(rows) != expected
        or any(not isinstance(row, Mapping) or type(row.get("ordinal")) is not int
               or row.get("ordinal") != ordinal for ordinal, row in enumerate(rows))
    ):
        return None
    return rows



def actor_problems(actors: object) -> list[dict]:
    """Carry actor errors into readiness without claiming service health."""
    if not isinstance(actors, list):
        return [{"status_error": True}]
    problems = []
    for actor in actors:
        if not isinstance(actor, Mapping):
            problems.append({"status_error": True})
            continue
        if "rank" in actor or "world" in actor:
            rank, world = actor.get("rank"), actor.get("world")
            if type(rank) is not int or type(world) is not int or world < 1 or not 0 <= rank < world:
                problems.append({"status_error": True})
        cleanup = actor.get("setup_cleanup_failed", False)
        if cleanup is True or not isinstance(cleanup, bool):
            problems.append({"setup_cleanup_failed": cleanup})
        if any(value is not None and value is not False and
               (not isinstance(value, str) or bool(value.strip()))
               for value in (actor.get(key) for key in ("status_error", "health_error", "error"))):
            problems.append({"status_error": True})
    return problems
