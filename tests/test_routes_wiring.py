"""Drive the /dgxm/telemetry, /dgxm/metrics and /dgxm/recycle routes over HTTP
through aiohttp's TestClient; tests/test_routes_cache.py tests the telemetry cache
and recycle helpers without a server."""
import asyncio
import math
import sys
import threading
import time
import types

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


async def _build_client(monkeypatch, handle=None):
    """Run register() against a fake PromptServer.instance whose .routes is a real
    RouteTableDef, so the get/post decorators in register() work, then serve that
    table from an Application for TestClient."""
    from dgx_monarch.nodes import routes as routes_mod

    monkeypatch.setattr(
        routes_mod, "_RECYCLE_RATE",
        {"active": False, "last_finished": float("-inf")},
    )

    fake_mesh = types.ModuleType("dgx_monarch.mesh")
    fake_mesh._MESHES = {} if handle is None else {"m": handle}
    fake_mesh._MESH_LOCK = threading.Lock()
    monkeypatch.setitem(sys.modules, "dgx_monarch.mesh", fake_mesh)

    route_table = web.RouteTableDef()
    fake_app = types.SimpleNamespace(routes=route_table)
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = types.SimpleNamespace(instance=fake_app)
    monkeypatch.setitem(sys.modules, "server", fake_server)

    routes_mod.register()
    assert getattr(fake_app, "_dgxm_routes", False) is True

    application = web.Application()
    application.add_routes(route_table)
    client = TestClient(TestServer(application))
    await client.start_server()
    return client


def test_telemetry_route_returns_200_json(monkeypatch):
    from dgx_monarch.nodes import routes as routes_mod

    monkeypatch.setattr(routes_mod, "_telemetry",
                        lambda: {"t": 1.0, "comfy": "abc123", "workers": []})

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await client.get("/dgxm/telemetry")
            assert resp.status == 200
            assert resp.content_type == "application/json"
            assert await resp.json() == {"t": 1.0, "comfy": "abc123", "workers": []}
        finally:
            await client.close()

    asyncio.run(scenario())


def test_telemetry_route_replaces_nonfinite_numbers_with_json_null(monkeypatch):
    from dgx_monarch.nodes import routes as routes_mod

    monkeypatch.setattr(routes_mod, "_telemetry", lambda: {
        "t": math.nan,
        "workers": [{"host": {"gpu": {"power_w": math.inf}}}],
    })

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await client.get("/dgxm/telemetry")
            text = await resp.text()
            assert resp.status == 200
            assert "NaN" not in text
            assert "Infinity" not in text
            assert await resp.json() == {
                "t": None,
                "workers": [{"host": {"gpu": {"power_w": None}}}],
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_telemetry_route_serializes_hostile_event_integers(monkeypatch):
    from dgx_monarch.nodes import routes as routes_mod
    from dgx_monarch.telemetry import public_event_tail

    events = public_event_tail([{
        "kind": "verify",
        "t": 1.0,
        "seq": 10**10000,
        "checked": 10**10000,
        "of": 4,
    }])
    monkeypatch.setattr(
        routes_mod, "_telemetry", lambda: {"t": 1.0, "workers": [], "events": events})

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await client.get("/dgxm/telemetry")
            assert resp.status == 200
            assert await resp.json() == {
                "t": 1.0,
                "workers": [],
                "events": [{
                    "kind": "verify", "t": 1.0, "checked": None, "of": 4,
                }],
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_metrics_route_returns_200_prometheus_text(monkeypatch):
    from dgx_monarch.nodes import routes as routes_mod

    monkeypatch.setattr(routes_mod, "_telemetry",
                        lambda: {"render": {"active": False}, "workers": []})

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await client.get("/dgxm/metrics")
            assert resp.status == 200
            assert resp.content_type == "text/plain"
            assert "dgxm_render_active 0" in await resp.text()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_recycle_route_rejects_without_csrf_headers(monkeypatch):
    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await client.post("/dgxm/recycle")
            assert resp.status == 403
            assert "X-DGXM-Action" in await resp.text()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_recycle_route_reports_no_live_mesh_with_valid_csrf_headers(monkeypatch):
    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            url = client.make_url("/")
            resp = await client.post("/dgxm/recycle", headers={
                "X-DGXM-Action": "recycle",
                "Origin": f"{url.scheme}://{url.host}:{url.port}",
                "Sec-Fetch-Site": "same-origin",
            })
            assert resp.status == 200
            assert await resp.json() == {
                "ok": False,
                "status": "no_live_mesh",
                "retryable": False,
                "detail": "no attached mesh to reset; worker services are unchanged",
            }
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("outcome_status", "detail"),
    [
        ("active_work", "recycle is blocked by active sample/result ownership"),
        ("lifecycle_busy", "recycle is blocked by another lifecycle transition"),
    ],
)
def test_recycle_route_returns_retryable_409_for_busy_work(
        monkeypatch, outcome_status, detail):
    from dgx_monarch import mesh as mesh_mod

    class _BusyHandle:
        def recycle_detailed(self):
            return mesh_mod.RecycleOutcome(
                mesh_mod.RecycleStatus(outcome_status),
                detail,
                retryable=True,
            )

        def call_all(self, *_args, **_kwargs):
            raise AssertionError("route must not mutate a busy worker")

    async def scenario():
        client = await _build_client(monkeypatch, _BusyHandle())
        try:
            url = client.make_url("/")
            resp = await client.post("/dgxm/recycle", headers={
                "X-DGXM-Action": "recycle",
                "Origin": f"{url.scheme}://{url.host}:{url.port}",
                "Sec-Fetch-Site": "same-origin",
            })
            assert resp.status == 409
            assert await resp.json() == {
                "ok": False,
                "status": outcome_status,
                "retryable": True,
                "detail": detail,
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_recycle_route_reports_proc_stop_failure_as_non_retryable_503(monkeypatch):
    from dgx_monarch import mesh as mesh_mod

    class _FailedHandle:
        def recycle_detailed(self):
            return mesh_mod.RecycleOutcome(
                mesh_mod.RecycleStatus.PROC_STOP_FAILED,
                "ProcMesh stop failed without a confirmed outcome",
            )

    async def scenario():
        client = await _build_client(monkeypatch, _FailedHandle())
        try:
            url = client.make_url("/")
            resp = await client.post("/dgxm/recycle", headers={
                "X-DGXM-Action": "recycle",
                "Origin": f"{url.scheme}://{url.host}:{url.port}",
                "Sec-Fetch-Site": "same-origin",
            })
            assert resp.status == 503
            assert await resp.json() == {
                "ok": False,
                "status": "proc_stop_failed",
                "retryable": False,
                "detail": "ProcMesh stop failed without a confirmed outcome",
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_recycle_route_returns_typed_json_for_unexpected_exception(monkeypatch):
    class _BrokenHandle:
        def recycle_detailed(self):
            raise RuntimeError("private reset detail")

    async def scenario():
        client = await _build_client(monkeypatch, _BrokenHandle())
        try:
            url = client.make_url("/")
            resp = await client.post("/dgxm/recycle", headers={
                "X-DGXM-Action": "recycle",
                "Origin": f"{url.scheme}://{url.host}:{url.port}",
                "Sec-Fetch-Site": "same-origin",
            })
            payload = await resp.json()
            assert resp.status == 503
            assert payload["status"] == "unexpected_failure"
            assert payload["retryable"] is False
            assert "private reset detail" not in str(payload)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_recycle_route_rate_limit_is_typed_json(monkeypatch):
    from dgx_monarch.nodes import routes as routes_mod

    async def scenario():
        client = await _build_client(monkeypatch)
        routes_mod._RECYCLE_RATE["active"] = True
        try:
            url = client.make_url("/")
            resp = await client.post("/dgxm/recycle", headers={
                "X-DGXM-Action": "recycle",
                "Origin": f"{url.scheme}://{url.host}:{url.port}",
                "Sec-Fetch-Site": "same-origin",
            })
            payload = await resp.json()
            assert resp.status == 429
            assert resp.headers["Retry-After"] == "1"
            assert payload["status"] == "rate_limited"
            assert payload["retryable"] is True
        finally:
            await client.close()

    asyncio.run(scenario())


def test_await_bounded_raises_timeout_error_on_deadline():
    """A call that outruns its deadline raises TimeoutError instead of hanging the
    caller. The telemetry, metrics and both consent routes rely on this bound."""
    from dgx_monarch.nodes import routes as routes_mod

    def _slow():
        time.sleep(0.15)
        return "done"

    async def scenario():
        with pytest.raises(TimeoutError):
            await routes_mod._await_bounded(_slow, 0.02)

    asyncio.run(scenario())
