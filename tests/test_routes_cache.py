"""The telemetry caches (single-flight, last-good-wins), the public event tail,
and the recycle route's write gate and origin checks."""
import asyncio
import sys
import threading
import types


def test_telemetry_derives_readiness_from_each_existing_snapshot_once(monkeypatch):
    from dgx_monarch import gate_ledger, mesh_health, telemetry
    from dgx_monarch.nodes import routes

    calls = {"workers": 0, "mesh": 0, "render": 0}
    workers = [{
        "healthy": True,
        "events": [{"kind": "notice", "t": 10**1000, "note": "still bounded"}],
    }]
    mesh = {
        "state": "idle", "verdict": "none", "active_leases": 0,
        "abandoned_samples": 0,
    }
    render = {"active": False}

    def observed(name, value):
        def take():
            calls[name] += 1
            return value
        return take

    monkeypatch.setattr(routes, "_workers", observed("workers", workers))
    monkeypatch.setattr(routes, "_ledger_summary", lambda: {})
    monkeypatch.setattr(mesh_health, "mesh_health_snapshot", observed("mesh", mesh))
    monkeypatch.setattr(telemetry.render_progress, "snapshot", observed("render", render))
    monkeypatch.setattr(telemetry, "host_stats", lambda: {})
    monkeypatch.setattr(telemetry, "events_tail", lambda _limit: [])
    monkeypatch.setattr(gate_ledger, "comfy_commit", lambda: "test-commit")

    payload = routes._telemetry_uncached()

    assert calls == {"workers": 1, "mesh": 1, "render": 1}
    assert payload["workers"] is workers
    assert payload["mesh"] is mesh
    assert payload["render"] is render
    assert payload["readiness"]["overall"] == "ready"
    assert payload["readiness"]["lifecycle"]["attached_mesh"]["state"] == "idle"
    assert payload["workers"][0]["events"] == [{
        "kind": "notice", "t": 0.0, "note": "still bounded",
    }]


def _fresh_routes(monkeypatch, call_all):
    from dgx_monarch.nodes import routes

    handle = types.SimpleNamespace(setup_key=("x",), call_all=call_all)
    fake_mesh = types.ModuleType("dgx_monarch.mesh")
    fake_mesh._MESHES = {"m": handle}
    monkeypatch.setitem(sys.modules, "dgx_monarch.mesh", fake_mesh)
    # mirror the module default (routes._CACHE) key for key: a fixture that
    # drifts from production tests a shape production never has
    monkeypatch.setattr(routes, "_CACHE",
                        {"t": 0.0, "workers": [], "inflight": False,
                         "busy_note": None, "generation": 0})
    monkeypatch.setattr(
        routes, "_TELEMETRY_CACHE",
        {"t": 0.0, "data": None, "inflight": False, "generation": 0},
    )
    return routes


def _capture_log(monkeypatch, routes):
    """Capture the module logger's records.

    caplog cannot see these: get_logger sets propagate = False, so records
    never reach pytest's root handler and caplog.text comes back empty
    (a silent pass, not a failure). Stub the logger instead.
    """
    records: list[tuple[str, str]] = []

    def at(level):
        return lambda msg, *args: records.append(
            (level, msg % args if args else msg))

    # every level, so the first future log.debug/error/exception on this path
    # fails as a readable assertion instead of an AttributeError
    monkeypatch.setattr(routes, "log", types.SimpleNamespace(
        debug=at("debug"), info=at("info"), warning=at("warning"),
        error=at("error"), exception=at("exception"), critical=at("critical"),
    ))
    return records


def _fake_render(monkeypatch, state):
    """Point _busy_render_key() at a tracker this test controls."""
    fake_telemetry = types.ModuleType("dgx_monarch.telemetry")
    fake_telemetry.render_progress = types.SimpleNamespace(snapshot=lambda: dict(state))
    monkeypatch.setitem(sys.modules, "dgx_monarch.telemetry", fake_telemetry)
    return state


def test_busy_render_timeout_is_a_note_not_a_warning(monkeypatch):
    """The panel polls every 2.5 s, the status call inside it is bounded at
    10 s, and a worker answers status on the actor running the render. The
    cache absorbs a timeout from that race, so it is a note."""
    calls = {"n": 0}

    def call_all(endpoint, timeout_s):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"host": "good"}]
        raise TimeoutError()

    routes = _fresh_routes(monkeypatch, call_all)
    records = _capture_log(monkeypatch, routes)
    _fake_render(monkeypatch, {"active": False})
    assert routes._workers() == [{"host": "good"}]

    _fake_render(monkeypatch, {"active": True, "started": 100.0})
    routes._CACHE["t"] = 0.0  # expire the TTL
    assert routes._workers() == [{"host": "good"}]  # last good still wins
    assert [level for level, _ in records] == ["info"]
    assert "render is in flight" in records[0][1]


def test_idle_timeout_still_warns(monkeypatch):
    """With no render open, nothing explains a timeout, so the poll warns.

    The cache is primed so a last-good snapshot exists: the idle check has to
    be what forces the warning here, not the absent-fallback check."""
    def call_all(endpoint, timeout_s):
        raise TimeoutError()

    routes = _fresh_routes(monkeypatch, call_all)
    records = _capture_log(monkeypatch, routes)
    routes._CACHE["workers"] = [{"host": "good"}]
    _fake_render(monkeypatch, {"active": False})
    assert routes._workers() == [{"host": "good"}]
    assert [level for level, _ in records] == ["warning"]
    assert records[0][1].startswith("worker status refresh failed:")


def test_non_timeout_failure_warns_even_mid_render(monkeypatch):
    """The downgrade covers timeouts only; a mesh fault mid-render still warns.

    Primed for the same reason as the idle case: with a fallback in hand and a
    render open, only the timeout check can still route this to WARNING."""
    def call_all(endpoint, timeout_s):
        raise RuntimeError("mesh hiccup")

    routes = _fresh_routes(monkeypatch, call_all)
    records = _capture_log(monkeypatch, routes)
    routes._CACHE["workers"] = [{"host": "good"}]
    _fake_render(monkeypatch, {"active": True, "started": 100.0})
    assert routes._workers() == [{"host": "good"}]
    assert [level for level, _ in records] == ["warning"]
    assert records[0][1].startswith("worker status refresh failed:")


def test_busy_note_is_written_once_per_render(monkeypatch):
    """One note per render, keyed on the start stamp: a poll that keeps timing
    out must not trade a WARNING flood for an INFO flood."""
    def call_all(endpoint, timeout_s):
        raise TimeoutError()

    routes = _fresh_routes(monkeypatch, call_all)
    records = _capture_log(monkeypatch, routes)
    routes._CACHE["workers"] = [{"host": "good"}]  # something to fall back on
    state = _fake_render(monkeypatch, {"active": True, "started": 100.0})
    routes._workers()
    routes._CACHE["t"] = 0.0
    routes._workers()
    assert [level for level, _ in records] == ["info"]

    state["started"] = 200.0  # a new render
    routes._CACHE["t"] = 0.0
    routes._workers()
    assert [level for level, _ in records] == ["info", "info"]


def test_busy_render_timeout_with_no_cached_data_still_warns(monkeypatch):
    """The note tells the operator the panel keeps its last good data, so it
    may only be written when there is last good data. A fleet has none until
    its first successful poll, and dispatch can follow READY inside one 2.5 s
    panel tick, so the first status poll of a fleet's life can land mid-render
    against an empty cache. The panel blanks there and `dgxm top` reports zero
    workers, which the operator has to be told about."""
    outcome = {"good": False}

    def call_all(endpoint, timeout_s):
        if outcome["good"]:
            return [{"host": "good"}]
        raise TimeoutError()

    routes = _fresh_routes(monkeypatch, call_all)
    records = _capture_log(monkeypatch, routes)
    _fake_render(monkeypatch, {"active": True, "started": 100.0})
    assert routes._workers() == [{"status_error": "TimeoutError"}]
    assert [level for level, _ in records] == ["warning"]

    # that failure row now sits in the cache, and it is not a fallback either:
    # a non-empty cache alone must not flip the next poll to the note arm
    routes._CACHE["t"] = 0.0
    assert routes._workers() == [{"status_error": "TimeoutError"}]
    assert [level for level, _ in records] == ["warning", "warning"]

    # once a poll succeeds there is something to serve, and the next timeout
    # in the same render is the note
    outcome["good"] = True
    routes._CACHE["t"] = 0.0
    assert routes._workers() == [{"host": "good"}]
    outcome["good"] = False
    routes._CACHE["t"] = 0.0
    assert routes._workers() == [{"host": "good"}]
    assert [level for level, _ in records] == ["warning", "warning", "info"]


def test_failed_refresh_keeps_last_good_snapshot(monkeypatch):
    calls = {"n": 0}

    def call_all(endpoint, timeout_s):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"host": "good"}]
        raise RuntimeError("mesh hiccup")

    routes = _fresh_routes(monkeypatch, call_all)
    assert routes._workers() == [{"host": "good"}]
    routes._CACHE["t"] = 0.0  # expire the TTL
    assert routes._workers() == [{"host": "good"}]  # failure must not clobber
    assert calls["n"] == 2


def test_failed_refresh_reports_exception_type_not_repr(monkeypatch):
    """status_error is an unauthenticated payload field: it must carry only
    the exception's type name, never the full repr (which can embed mesh
    internals, hostnames, or other detail from the raised message)."""
    def call_all(endpoint, timeout_s):
        raise RuntimeError("mesh hiccup with sensitive detail")

    routes = _fresh_routes(monkeypatch, call_all)
    workers = routes._workers()
    assert workers == [{"status_error": "RuntimeError"}]
    assert "sensitive detail" not in str(workers)


def test_single_flight_concurrent_polls(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def call_all(endpoint, timeout_s):
        calls["n"] += 1
        started.set()
        release.wait(timeout=5)
        return [{"host": "fresh"}]

    routes = _fresh_routes(monkeypatch, call_all)
    t = threading.Thread(target=routes._workers)
    t.start()
    started.wait(timeout=5)
    # a second poll while the first is in flight serves the (stale) cache
    # instead of stacking another call_all
    assert routes._workers() == []
    assert calls["n"] == 1
    release.set()
    t.join(timeout=5)
    routes._CACHE["t"] = 0.0
    assert routes._workers() == [{"host": "fresh"}]


def test_recycle_invalidation_fences_an_inflight_worker_refresh(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def call_all(_endpoint, timeout_s=None):
        started.set()
        release.wait(timeout=5)
        return [{"host": "retired-fleet"}]

    routes = _fresh_routes(monkeypatch, call_all)
    result = []
    thread = threading.Thread(target=lambda: result.append(routes._workers()))
    thread.start()
    assert started.wait(timeout=5)

    routes.invalidate_telemetry_caches()
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result == [[]]
    assert routes._CACHE["workers"] == []
    assert routes._CACHE["t"] == 0.0


def test_recycle_invalidation_fences_an_inflight_whole_snapshot(monkeypatch):
    from dgx_monarch.nodes import routes

    started = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        routes, "_CACHE",
        {"t": 0.0, "workers": [], "inflight": False, "busy_note": None,
         "generation": 0},
    )
    monkeypatch.setattr(
        routes, "_TELEMETRY_CACHE",
        {"t": 0.0, "data": None, "inflight": False, "generation": 0},
    )

    def refresh():
        started.set()
        release.wait(timeout=5)
        return {"workers": ["retired-fleet"]}

    monkeypatch.setattr(routes, "_telemetry_uncached", refresh)
    result = []
    thread = threading.Thread(target=lambda: result.append(routes._telemetry()))
    thread.start()
    assert started.wait(timeout=5)

    routes.invalidate_telemetry_caches()
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert routes._TELEMETRY_CACHE["data"] is None
    assert result[0]["telemetry_error"] == "CacheInvalidated"
    assert result[0]["readiness"]["overall"] == "unknown"


def test_recycle_write_gate_is_single_flight_and_rate_limited(monkeypatch):
    from dgx_monarch.nodes import routes

    monkeypatch.setattr(
        routes, "_RECYCLE_RATE",
        {"active": False, "last_finished": float("-inf")},
    )
    assert routes._begin_recycle_request(now=10.0) is True
    assert routes._begin_recycle_request(now=20.0) is False
    routes._finish_recycle_request(now=20.0)
    assert routes._begin_recycle_request(now=20.5) is False
    assert routes._begin_recycle_request(now=21.0) is True
    routes._finish_recycle_request(now=21.0)


def test_timed_out_recycle_keeps_gate_and_final_fence_until_executor_settles(
    monkeypatch,
):
    from dgx_monarch.nodes import routes

    started = threading.Event()
    release = threading.Event()
    fences = []
    monkeypatch.setattr(
        routes, "_RECYCLE_RATE",
        {"active": False, "last_finished": float("-inf")},
    )
    monkeypatch.setattr(
        routes, "invalidate_telemetry_caches", lambda: fences.append("fence"),
    )

    def blocked_reset():
        started.set()
        release.wait(timeout=5)
        return "settled"

    async def scenario():
        assert routes._begin_recycle_request(now=10.0) is True
        try:
            await routes._await_recycle_bounded(blocked_reset, 0.01)
        except TimeoutError:
            pass
        else:  # pragma: no cover - a broken timeout contract
            raise AssertionError("blocked reset did not time out")

        assert started.is_set()
        assert routes._RECYCLE_RATE["active"] is True
        assert routes._begin_recycle_request(now=100.0) is False

        release.set()
        for _ in range(100):
            if routes._RECYCLE_RATE["active"] is False:
                break
            await asyncio.sleep(0.01)
        assert routes._RECYCLE_RATE["active"] is False

    asyncio.run(scenario())
    assert fences == ["fence", "fence"]
    assert routes._begin_recycle_request(now=float("inf")) is True
    routes._finish_recycle_request(now=float("inf"))


def test_route_executor_is_the_real_recycle_wrappers_deepest_owner(monkeypatch):
    from dgx_monarch import mesh_recycle
    from dgx_monarch.nodes import routes

    started = threading.Event()
    release = threading.Event()
    contexts = []
    fences = []
    real_off_loop = mesh_recycle.mesh_helpers.run_blocking_off_loop
    monkeypatch.setattr(
        routes, "_RECYCLE_RATE",
        {"active": False, "last_finished": float("-inf")},
    )
    monkeypatch.setattr(
        routes, "invalidate_telemetry_caches", lambda: fences.append("fence"),
    )

    def observe_context(fn, timeout_s, thread_name):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            contexts.append("no-loop-inline")
        else:  # pragma: no cover - would recreate the audited nested-owner gap
            contexts.append("loop-scratch")
        return real_off_loop(fn, timeout_s, thread_name)

    def inner(*_args, **_kwargs):
        started.set()
        release.wait(timeout=5)
        return mesh_recycle.RecycleOutcome(
            mesh_recycle.RecycleStatus.RECYCLED, "settled")

    monkeypatch.setattr(
        mesh_recycle.mesh_helpers, "run_blocking_off_loop", observe_context)
    monkeypatch.setattr(mesh_recycle, "recycle_detailed_impl", inner)

    async def scenario():
        assert routes._begin_recycle_request(now=10.0) is True
        task = asyncio.create_task(routes._await_recycle_bounded(
            lambda: mesh_recycle.recycle_detailed(
                object(), lambda: None, lambda: None),
            1.0,
        ))
        try:
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()
            assert routes._RECYCLE_RATE["active"] is True
            assert routes._begin_recycle_request(now=100.0) is False
        finally:
            release.set()
        outcome = await task
        assert outcome.status is mesh_recycle.RecycleStatus.RECYCLED
        assert routes._RECYCLE_RATE["active"] is False

    asyncio.run(scenario())
    assert contexts == ["no-loop-inline"]
    assert fences == ["fence", "fence"]


def test_event_tail_has_a_fixed_bounded_json_safe_projection():
    from dgx_monarch.telemetry import public_event_tail

    projected = public_event_tail([
        {
            "kind": "notice", "t": float("nan"), "seq": 7,
            "phase": "gate", "note": "x" * 1000,
            "environment": {"TOKEN": "must-not-leak"},
        },
        {"kind": "unknown", "t": 2.0, "private": "must-not-leak"},
        {"kind": "notice", "t": 10**1000, "note": "huge timestamp"},
        {
            "kind": "verify", "t": 3.0, "seq": 10**10000,
            "checked": 10**10000, "of": 4,
        },
    ])

    assert projected[0] == {
        "kind": "notice", "t": 0.0, "seq": 7,
        "phase": "gate", "note": "x" * 192,
    }
    assert projected[1] == {"kind": "unknown", "t": 2.0}
    assert projected[2] == {
        "kind": "notice", "t": 0.0, "note": "huge timestamp",
    }
    assert projected[3] == {
        "kind": "verify", "t": 3.0, "checked": None, "of": 4,
    }
    assert "must-not-leak" not in str(projected)


def test_recycle_request_requires_matching_host_and_origin():
    from dgx_monarch.nodes.routes import _recycle_request_allowed

    headers = {
        "Host": "127.0.0.1:8188",
        "Origin": "http://127.0.0.1:8188",
        "Sec-Fetch-Site": "same-origin",
        "X-DGXM-Action": "recycle",
    }
    assert _recycle_request_allowed(headers, "http") == (True, "")

    wrong_origin = {**headers, "Origin": "http://attacker.example"}
    allowed, detail = _recycle_request_allowed(wrong_origin, "http")
    assert not allowed
    assert "does not match" in detail


def test_recycle_request_rejects_cross_site_and_missing_csrf_headers():
    from dgx_monarch.nodes.routes import _recycle_request_allowed

    valid = {
        "Host": "comfy.example:443",
        "Origin": "https://comfy.example",
        "Sec-Fetch-Site": "same-origin",
        "X-DGXM-Action": "recycle",
    }
    assert _recycle_request_allowed(valid, "https") == (True, "")

    cross_site = {**valid, "Sec-Fetch-Site": "cross-site"}
    allowed, detail = _recycle_request_allowed(cross_site, "https")
    assert not allowed
    assert "cross-site" in detail

    for missing in ("Host", "Origin", "X-DGXM-Action"):
        incomplete = {key: value for key, value in valid.items() if key != missing}
        assert not _recycle_request_allowed(incomplete, "https")[0]


def test_recycle_request_rejects_scheme_mismatch_and_malformed_origin():
    from dgx_monarch.nodes.routes import _recycle_request_allowed

    headers = {
        "Host": "comfy.example",
        "Origin": "https://comfy.example",
        "X-DGXM-Action": "recycle",
    }
    assert not _recycle_request_allowed(headers, "http")[0]
    assert not _recycle_request_allowed({**headers, "Origin": "null"}, "https")[0]


def test_recycle_request_honors_x_forwarded_proto_behind_ssl_proxy():
    """An SSL-terminating proxy (tailscale serve, nginx) gives aiohttp an http
    socket while the browser's Origin says https; X-Forwarded-Proto must
    reconcile them or every HTTPS deployment loses the recycle button."""
    from dgx_monarch.nodes.routes import _recycle_request_allowed

    headers = {
        "Host": "comfy.example",
        "Origin": "https://comfy.example",
        "Sec-Fetch-Site": "same-origin",
        "X-DGXM-Action": "recycle",
        "X-Forwarded-Proto": "https",
    }
    assert _recycle_request_allowed(headers, "http") == (True, "")

    # multi-hop proxies send a comma list; the first (client-facing) hop wins
    chained = {**headers, "X-Forwarded-Proto": "https, http"}
    assert _recycle_request_allowed(chained, "http") == (True, "")

    # the forwarded scheme participates in the same-origin check: an http
    # Origin against a forwarded-https request is still a mismatch
    downgraded = {**headers, "Origin": "http://comfy.example"}
    assert not _recycle_request_allowed(downgraded, "http")[0]

    # a forged/garbage forwarded proto fails closed
    garbage = {**headers, "X-Forwarded-Proto": "gopher"}
    assert not _recycle_request_allowed(garbage, "http")[0]
