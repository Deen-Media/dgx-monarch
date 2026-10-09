"""Driver-side telemetry routes on ComfyUI's own web server.

`dgxm top` and other readers (Grafana agents, curl, scripts) read cluster
state over plain HTTP instead of attaching a second Monarch client, which
can wedge the driver.

  GET /dgxm/telemetry   full JSON: worker status (cached ~1 s) + host planes
                        + event rings (driver + workers) + render progress
                        + gate ledger summary + ComfyUI commit + mesh health
                        + readiness
  GET /dgxm/metrics     Prometheus text exposition of the numeric subset
  POST /dgxm/recycle    confirmed attached-mesh reset from the sidebar
  GET  /dgxm/consents   pending consent cards, active consents, the toggle
  POST /dgxm/consent    accept, dismiss, revoke, auto-rescue (consent_routes)

The GET routes are read-only. Worker status is served from a short cache so a
dashboard polling at a few Hz costs the cluster one status call per second
at most; when no mesh is up, the routes still answer with the driver-local
planes (progress, events, ledger) and workers: [].

The reset POST has CSRF checks (action header, same-origin Host and Origin,
Fetch Metadata) against drive-by browser requests, but they do not
authenticate a user: that stays with ComfyUI's authentication middleware.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Mapping

from ..log import get_logger
from . import recycle_guard
from .action_security import _action_request_allowed, _recycle_request_allowed
from .metrics_route import _metrics_text
from .strict_json import _json_payload

__all__ = ["_action_request_allowed", "_recycle_request_allowed", "register"]

log = get_logger(__name__)

# Both caches: process lifetime, unkeyed, one slot each, refreshed in place
# past _CACHE_TTL_S and holding last-observed facts meanwhile. A failed
# telemetry refresh marks readiness unknown. A failed worker poll serves the
# last good rows, or a status_error row (read as unknown) when there are none.
_CACHE: dict = {
    "t": 0.0, "workers": [], "inflight": False, "busy_note": None,
    "generation": 0,
}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_S = 1.0
_TELEMETRY_CACHE: dict = {
    "t": 0.0, "data": None, "inflight": False, "generation": 0,
}  # process lifetime, same rule
_TELEMETRY_CACHE_LOCK = threading.Lock()
# GET routes: the worker status call inside is bounded (timeout_s=10 in
# _workers()); 30 s leaves room for host_stats(), the ledger scan and
# nvidia-smi in _telemetry_uncached() without hanging a poller.
_TELEMETRY_TIMEOUT_S = 30.0
# POST /dgxm/recycle: recycle_detailed() bounds the sequential group teardown
# and ProcMesh stop at 180 s. The other 80 s keep this HTTP deadline from
# racing a slow but legitimate internal resolution.
_RECYCLE_TIMEOUT_S = 260.0
_RECYCLE_MIN_INTERVAL_S = 1.0
_RECYCLE_RATE_LOCK = threading.Lock()
# Process lifetime, one global write gate: reset ownership is driver-wide.
_RECYCLE_RATE: dict[str, object] = {"active": False, "last_finished": float("-inf")}


def _begin_recycle_request(now: float | None = None) -> bool:
    return recycle_guard.begin(
        _RECYCLE_RATE, _RECYCLE_RATE_LOCK, _RECYCLE_MIN_INTERVAL_S, now)


def _finish_recycle_request(now: float | None = None) -> None:
    recycle_guard.finish(_RECYCLE_RATE, _RECYCLE_RATE_LOCK, now)


def invalidate_telemetry_caches() -> None:
    """Drop pre-reset observations and fence refreshes already in flight."""
    from ..service_observations import services

    services.invalidate()
    with _CACHE_LOCK:
        _CACHE.update(
            t=0.0, workers=[], busy_note=None,
            generation=int(_CACHE.get("generation", 0)) + 1,
        )
    with _TELEMETRY_CACHE_LOCK:
        _TELEMETRY_CACHE.update(
            t=0.0, data=None,
            generation=int(_TELEMETRY_CACHE.get("generation", 0)) + 1,
        )


def _busy_render_key() -> str | None:
    """A key for the render this driver has open right now, or None if idle.

    The driver's own progress tracker is the least invasive in-flight signal
    here: it lives in this process, opens before dispatch (render_submit.py
    starts it ahead of submit_sample) and closes in the render's cleanup, so it
    spans every moment a worker can be too busy to answer a status ping. Lease
    and pending-render state would need mesh internals and a lock this
    read-only route must never take.

    The key is the render's start stamp, which start() holds steady across
    overlapping pipeline submissions, so one busy window notes itself once.
    """
    from ..telemetry import render_progress

    try:
        snapshot = render_progress.snapshot()
    except Exception:
        return None
    if not snapshot.get("active"):
        return None
    return f"{snapshot.get('started')}"


def _has_last_good_data(rows: list) -> bool:
    """True when the worker cache holds a real snapshot to fall back on.

    A non-empty cache is not proof of a fallback: _workers() seeds an empty
    cache with a single {"status_error": ...} row and keeps that row until a
    poll succeeds; the panel reads it as unknown and `dgxm top` shows no worker.
    Only rows without that field are data an operator can still trust while a
    refresh is failing.
    """
    return any(isinstance(row, Mapping) and "status_error" not in row
               for row in rows)


def _log_refresh_failure(exc: BaseException) -> None:
    """Log a failed status poll at INFO when the cache absorbs it, else WARNING.

    A worker answers status on the same actor that runs the render, so the
    fixed 10 s ping can expire while the ranks are saturated. The cache
    tolerates that (last good wins, the next tick recovers), so a timeout
    during a render with a last-good snapshot to serve logs one INFO note per
    render, not one per 2.5 s panel poll. Anything else warns: an idle fleet, a
    failure that is not a timeout, or no last-good snapshot, which is a fresh
    fleet's state until its first successful poll, when the panel blanks and
    the operator must be told.

    Placement is part of the contract: this takes _CACHE_LOCK, which is not
    reentrant, so the call site must stay outside both locked blocks.
    """
    key = _busy_render_key()
    if key is not None and isinstance(exc, TimeoutError):
        with _CACHE_LOCK:
            absorbed = _has_last_good_data(_CACHE["workers"])
            noted = _CACHE.get("busy_note") == key
            if absorbed:
                _CACHE["busy_note"] = key
        if absorbed:
            if not noted:
                log.info("worker status poll timed out while a render is in "
                         "flight; this is expected under load and the panel "
                         "keeps its last good data")
            return
    log.warning("worker status refresh failed: %r", exc)


def _workers() -> list:
    """Worker status with a single-flight, last-good-wins cache.

    The telemetry and metrics routes run on the thread pool, so polls overlap
    (panel, dgxm top, scrapers). At most one refresh is in flight; the others
    serve the cached snapshot. A failed refresh never overwrites a good
    snapshot: it only bumps the timestamp, so retries back off for one TTL."""
    from ..mesh import _MESHES  # driver-local registry; read-only peek

    with _CACHE_LOCK:
        if time.time() - _CACHE["t"] < _CACHE_TTL_S or _CACHE["inflight"]:
            return _CACHE["workers"]
        generation = int(_CACHE.get("generation", 0))
        _CACHE["inflight"] = True
    results: list = []
    ok = True
    try:
        handle = next(iter(_MESHES.values()), None)
        if handle is not None and getattr(handle, "setup_key", None) is not None:
            results = handle.call_all("status", timeout_s=10)
    except Exception as exc:
        ok = False
        _log_refresh_failure(exc)
        results = [{"status_error": type(exc).__name__}]
    with _CACHE_LOCK:
        _CACHE["inflight"] = False
        if generation != int(_CACHE.get("generation", 0)):
            return _CACHE["workers"]
        _CACHE["t"] = time.time()
        if ok or not _CACHE["workers"]:
            _CACHE["workers"] = results
        return _CACHE["workers"]


def _ledger_summary() -> dict:
    try:
        import folder_paths

        from ..gate_audit import audit_rows, trust_rows
        from ..gate_ledger import GateLedger

        entries = GateLedger(folder_paths.get_output_directory()).entries()
        # Audit rows (waivers, capacity certificates) are evidence, never
        # verdicts. Counting them here would show WAIVER as a gate verdict in
        # the panel, and `dgxm top` reads the same field.
        latest: dict = {e.get("key"): e for e in trust_rows(entries)}
        counts: dict = {}
        for e in latest.values():
            counts[e.get("verdict", "?")] = counts.get(e.get("verdict", "?"), 0) + 1
        waivers = {e.get("key"): e for e in audit_rows(entries)}
        return {"combinations": counts, "waivers": len(waivers),
                "latest": sorted(latest.values(), key=lambda e: e.get("time", ""))[-8:]}
    except Exception:
        return {}


def _with_readiness(snapshot: dict) -> dict:
    from ..operator_readiness import readiness_from_telemetry_payload

    snapshot["readiness"] = readiness_from_telemetry_payload(snapshot)
    return snapshot


def _stale_snapshot(cached: object, marker: str) -> dict:
    snapshot = dict(cached) if isinstance(cached, dict) else {
        "t": time.time(), "workers": []}
    snapshot["telemetry_error"] = marker
    return _with_readiness(snapshot)


def _telemetry_uncached() -> dict:
    from .. import telemetry, telemetry_fleet
    from ..gate_ledger import comfy_commit
    from ..mesh_health import mesh_health_snapshot
    from ..service_observations import services

    workers = _workers()
    for worker in workers if isinstance(workers, list) else ():
        if isinstance(worker, dict) and "events" in worker:
            worker["events"] = telemetry.public_event_tail(worker.get("events"), 32)
    snapshot = {
        "t": time.time(),
        "comfy": comfy_commit(),
        "driver_host": telemetry.host_stats(),
        # The Fleet block sits inside the render block it belongs to, so every
        # reader of `render` gets the per-job rows without a second key.
        "render": telemetry_fleet.with_fleet(telemetry.render_progress.snapshot()),
        "events": telemetry.events_tail(64),
        "ledger": _ledger_summary(),
        # Lease and teardown state, which _busy_render_key does not read: the
        # accessor reads it in the mesh layer, under that layer's lock order,
        # and hands this route a plain dict.
        "mesh": mesh_health_snapshot(),
        "workers": workers,
        "worker_services": services.snapshot(),
    }
    return _with_readiness(snapshot)


def _telemetry() -> dict:
    """Whole-payload, single-flight cache shared by the JSON and metrics routes.

    Concurrent polls share one snapshot, so host_stats(), the ledger scan and
    nvidia-smi run once per refresh, not once per request. A failed refresh
    keeps the last payload, marked with telemetry_error.
    """
    now = time.time()
    with _TELEMETRY_CACHE_LOCK:
        cached = _TELEMETRY_CACHE["data"]
        if now - _TELEMETRY_CACHE["t"] < _CACHE_TTL_S:
            if isinstance(cached, dict) and "worker_services" in cached:
                return _with_readiness(dict(cached))
            return cached if isinstance(cached, dict) else {}
        if _TELEMETRY_CACHE["inflight"]:
            return _stale_snapshot(cached, "RefreshInFlight")
        generation = int(_TELEMETRY_CACHE.get("generation", 0))
        _TELEMETRY_CACHE["inflight"] = True

    try:
        fresh = _telemetry_uncached()
    except Exception as exc:
        log.warning("telemetry refresh failed: %r", exc)
        with _TELEMETRY_CACHE_LOCK:
            _TELEMETRY_CACHE["inflight"] = False
            if generation != int(_TELEMETRY_CACHE.get("generation", 0)):
                return _stale_snapshot(
                    _TELEMETRY_CACHE["data"], "CacheInvalidated")
            _TELEMETRY_CACHE["t"] = time.time()
            stale = _stale_snapshot(_TELEMETRY_CACHE["data"], type(exc).__name__)
            _TELEMETRY_CACHE["data"] = stale
            return stale
    with _TELEMETRY_CACHE_LOCK:
        if generation != int(_TELEMETRY_CACHE.get("generation", 0)):
            _TELEMETRY_CACHE["inflight"] = False
            return _stale_snapshot(_TELEMETRY_CACHE["data"], "CacheInvalidated")
        _TELEMETRY_CACHE.update(t=time.time(), data=fresh, inflight=False)
        return fresh


async def _await_bounded(fn, timeout_s: float):
    """Run a blocking function off-loop under an HTTP response deadline.

    A Monarch future wait on aiohttp's event-loop thread would stall the server.
    Use a worker thread and ``wait_for`` so an unresponsive call returns a 503
    through the route's timeout handler instead of holding the connection open.
    """
    return await asyncio.wait_for(
        asyncio.get_running_loop().run_in_executor(None, fn), timeout=timeout_s)


async def _await_recycle_bounded(fn, timeout_s: float):
    return await recycle_guard.await_settlement(
        fn, timeout_s, invalidate_telemetry_caches, _finish_recycle_request)


def register() -> None:
    """Attach the routes to PromptServer; a headless import is a no-op."""
    try:
        from aiohttp import web
        from server import PromptServer

        app = PromptServer.instance
    except Exception:
        return
    if getattr(app, "_dgxm_routes", False):
        return

    @app.routes.get("/dgxm/telemetry")
    async def dgxm_telemetry(_request):
        try:
            data = await _await_bounded(_telemetry, _TELEMETRY_TIMEOUT_S)
        except TimeoutError:
            return web.json_response(
                {"t": time.time(), "workers": [], "telemetry_error": "TimeoutError"},
                status=503)
        return web.json_response(_json_payload(data))

    @app.routes.post("/dgxm/recycle")
    async def dgxm_recycle(request):
        """Reset the client-owned mesh (ClearVRAM level=recycle).

        Persistent worker services keep running; blocking work stays off-loop.
        The checks below are CSRF controls, not user authentication.
        """
        from ..mesh import _MESH_LOCK, _MESHES

        allowed, detail = _recycle_request_allowed(request.headers, request.scheme)
        if not allowed:
            raise web.HTTPForbidden(text=detail)
        if not _begin_recycle_request():
            return web.json_response(
                {
                    "ok": False,
                    "status": "rate_limited",
                    "retryable": True,
                    "detail": "another attached-mesh reset is active or just settled; wait before retrying",
                },
                status=429,
                headers={"Retry-After": str(int(_RECYCLE_MIN_INTERVAL_S))},
            )

        def _do():
            with _MESH_LOCK:  # snapshot under lock; recycle is idempotent anyway
                handle = next(iter(_MESHES.values()), None)
            if handle is None:
                return ({
                    "ok": False,
                    "status": "no_live_mesh",
                    "retryable": False,
                    "detail": "no attached mesh to reset; worker services are unchanged",
                }, 200)

            # Do not issue a clear_vram RPC first. recycle_detailed() takes the
            # lifecycle lock and rejects active render/result leases before any
            # destructive mesh action. Actor exit itself returns both loaded
            # models and the retained allocator pool to the OS.
            outcome = handle.recycle_detailed()
            if outcome.ok:
                # Proc exit invalidates worker-residency credits. Drain loader and render
                # memos together before a new fleet can inherit them.
                from . import recycle_drain

                recycle_drain.drop_residency_memos()
            status = 200 if outcome.ok else (409 if outcome.retryable else 503)
            return outcome.as_dict(), status

        try:
            result, status = await _await_recycle_bounded(_do, _RECYCLE_TIMEOUT_S)
        except TimeoutError:
            return web.json_response(
                {
                    "ok": False,
                    "status": "overall_timed_out",
                    "retryable": False,
                    "detail": (
                        "attached-mesh reset passed its server deadline; teardown may "
                        "still be running, so check mesh status before another reset"
                    ),
                },
                status=503)
        except Exception as exc:
            log.error(
                "attached-mesh reset raised an unexpected %s",
                type(exc).__name__,
            )
            return web.json_response(
                {
                    "ok": False,
                    "status": "unexpected_failure",
                    "retryable": False,
                    "detail": (
                        "attached-mesh reset failed unexpectedly and its outcome is "
                        "unknown; check mesh status and the driver log before retrying"
                    ),
                },
                status=503)
        return web.json_response(_json_payload(result), status=status)

    @app.routes.get("/dgxm/metrics")
    async def dgxm_metrics(_request):
        try:
            data = await _await_bounded(_telemetry, _TELEMETRY_TIMEOUT_S)
        except TimeoutError:
            return web.json_response(
                {"t": time.time(), "workers": [], "telemetry_error": "TimeoutError"},
                status=503)
        return web.Response(text=_metrics_text(data), content_type="text/plain")

    from . import consent_routes

    consent_routes.register(app, web)

    # Queue-time graph advice is not a route, so a server that cannot take its
    # handler (an older ComfyUI, or the canary's instance built without
    # __init__) still gets every route above.
    try:
        from ..graph_advisor import register_on_prompt

        register_on_prompt(app)
    except Exception as exc:
        log.warning("graph advisor not registered: %r", exc)

    app._dgxm_routes = True
    log.info("dgxm routes registered: /dgxm/telemetry /dgxm/recycle /dgxm/metrics")
