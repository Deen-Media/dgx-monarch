"""Mesh-health reporting in dgxm status and dgxm doctor.

Worker loops and ports can remain live while mesh attachment fails. These tests
check the lifecycle accessor and both CLI rows against the driver's state so
neither command reports such a cluster as healthy.
"""
from __future__ import annotations

import importlib.metadata
import io
import json
import subprocess
import sys
import threading
import types
import urllib.error
from types import SimpleNamespace

import pytest

from dgx_monarch import (
    TORCHMONARCH_PIN,
    mesh_creation,
    mesh_evidence,
    mesh_health,
    mesh_safety,
    mesh_setup_state,
)
from dgx_monarch.cli import doctor, mesh_health_row, probe_certainty
from dgx_monarch.config import ClusterConfig, HostConfig


def _troubleshooting_entry(number: int) -> str:
    """One numbered TROUBLESHOOTING entry, so a docs claim can be read here."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1]
            / "docs" / "TROUBLESHOOTING.md").read_text(encoding="utf-8")
    body = text.split(f"\n## {number}. ", 1)[1]
    return body.split("\n## ", 1)[0]


def _handle(**overrides):
    """A handle carrying every field the accessor reads except `_block_cause`, so it records no block cause."""
    fields = {
        "teardown_complete": False,
        "replacement_blocked": None,
        "defunct": False,
        "setup_cleanup_state": None,
        "setup_generation": 1,
        "sample_leases": {},
        "abandoned_sample_leases": {},
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _in_flight(phase="load_model RPC completion", budget_s=900.0):
    """The latch mesh_rpc publishes before every session-scoped mutation."""
    return mesh_setup_state.cleanup_in_progress(1, phase, budget_s)


def _lost(phase="load_model RPC completion"):
    """The same latch after its budget expired with no answer."""
    return mesh_setup_state.cleanup_failure(1, phase, 900.0, TimeoutError("no answer"))


@pytest.fixture
def teardown_token():
    """Hold a deliberate-teardown token for one handle, then release it."""
    held = []

    def begin(handle):
        mesh_safety.begin_deliberate_teardown(id(handle))
        held.append(id(handle))
        return handle

    yield begin
    for token in held:
        mesh_safety.end_deliberate_teardown(token, stop_confirmed=False)


def test_an_empty_registry_is_idle_not_healthy():
    health = mesh_health.health_of([])
    assert health.state == "idle"
    assert not health.dirty
    assert health.reason == ""
    assert health.remedy == ""


def test_a_clean_cached_fleet_is_ok():
    health = mesh_health.health_of([_handle(setup_generation=3)])
    assert (health.state, health.verdict) == ("ok", "live")
    assert health.setup_generation == 3
    assert not health.dirty


def test_active_leases_are_counted_and_are_not_a_fault():
    health = mesh_health.health_of([_handle(sample_leases={0: 1, 1: 2})])
    assert health.active_leases == 3
    assert health.state == "ok"


def test_one_abandoned_sample_is_dirty_and_names_the_count():
    health = mesh_health.health_of([_handle(abandoned_sample_leases={0: 1})])
    assert health.state == "dirty"
    assert health.abandoned_samples == 1
    assert "1 abandoned sample may still be running" in health.reason
    # The panel button clears this one, so the row must not send the operator
    # to a ComfyUI restart for it.
    assert "Reset attached mesh" in health.remedy
    assert "dgxm restart" in health.remedy


def test_the_abandoned_count_is_summed_across_ranks_and_pluralized():
    health = mesh_health.health_of([_handle(abandoned_sample_leases={0: 1, 1: 1})])
    assert health.abandoned_samples == 2
    assert "2 abandoned samples may still be running" in health.reason


def test_a_blocked_replacement_is_dirty_and_asks_for_a_restart():
    health = mesh_health.health_of([_handle(replacement_blocked="proc stop failed")])
    assert (health.state, health.verdict) == ("dirty", "blocked")
    assert "an earlier attached-mesh process stop failed" in health.reason
    assert "restart ComfyUI" in health.remedy


def test_a_lost_teardown_outcome_is_dirty():
    for handle in (_handle(setup_cleanup_state=_lost()), _handle(defunct=True)):
        health = mesh_health.health_of([handle])
        assert (health.state, health.verdict) == ("dirty", "unresolved")
        assert "teardown outcome is unresolved" in health.reason


def test_a_load_the_driver_is_still_waiting_on_is_busy_not_dirty():
    # mesh_rpc publishes an IN_PROGRESS latch before every session-scoped
    # mutation. `coherent_lifecycle_verdict` reads it as `unresolved` while the
    # RPC runs, up to its budget (the loader's load_model allows 1800 s). A
    # recycle in that window kills the render, so the row must call it busy.
    health = mesh_health.health_of([_handle(setup_cleanup_state=_in_flight())])
    assert (health.state, health.verdict) == ("busy", "unresolved")
    assert health.busy_phase == "load_model RPC completion"
    assert not health.dirty
    assert (health.reason, health.remedy) == ("", "")


def test_a_teardown_in_flight_is_busy_not_dirty(teardown_token):
    health = mesh_health.health_of([teardown_token(_handle())])
    assert health.state == "busy"
    assert health.busy_phase == "worker teardown"


def test_a_token_whose_outcome_already_landed_is_not_busy(teardown_token):
    # Every teardown site publishes its outcome before clearing the token, so a
    # reader can see a live token over a decided failure. That is dirty, not
    # busy.
    handle = teardown_token(_handle(replacement_blocked="proc stop failed"))
    assert mesh_health._busy_phase(handle) == ""
    assert mesh_health.health_of([handle]).state == "dirty"


def test_a_defunct_handle_stays_dirty_under_an_in_flight_latch():
    health = mesh_health.health_of([
        _handle(defunct=True, setup_cleanup_state=_in_flight())])
    assert health.state == "dirty"
    assert "teardown outcome is unresolved" in health.reason


def test_an_abandoned_sample_outranks_a_busy_driver():
    health = mesh_health.health_of([_handle(
        setup_cleanup_state=_in_flight(), abandoned_sample_leases={0: 1})])
    assert health.state == "dirty"
    assert health.reason == "1 abandoned sample may still be running"
    assert "Reset attached mesh" in health.remedy


def test_a_busy_fleet_outranks_a_quiet_one_across_handles():
    health = mesh_health.health_of([
        _handle(), _handle(setup_cleanup_state=_in_flight("gate_swap_cycle RPC completion"))])
    assert health.state == "busy"
    assert health.busy_phase == "gate_swap_cycle RPC completion"


def test_a_confirmed_stop_reads_as_idle_even_holding_abandoned_leases():
    # get_mesh retires a completed handle and builds a fresh fleet, so its
    # leases block nothing: flagging it would be a false alarm.
    health = mesh_health.health_of([
        _handle(teardown_complete=True, abandoned_sample_leases={0: 1})])
    assert (health.state, health.verdict) == ("idle", "completed")
    assert not health.dirty


def test_unreadable_lease_maps_never_raise():
    health = mesh_health.health_of([_handle(
        sample_leases={0: "junk", 1: None, 2: 2},
        abandoned_sample_leases="not a map",
    )])
    assert (health.active_leases, health.abandoned_samples) == (2, 0)


def test_a_handle_the_verdict_cannot_read_is_unknown_not_healthy():
    health = mesh_health.health_of([SimpleNamespace()])
    assert health.state == "unknown"
    assert not health.dirty


def test_one_dirty_fleet_decides_the_whole_verdict():
    health = mesh_health.health_of([
        _handle(sample_leases={0: 1}),
        _handle(abandoned_sample_leases={0: 1}),
    ])
    assert health.state == "dirty"


def test_the_snapshot_takes_the_registry_lock_and_holds_no_lock_to_decide():
    class RecordingLock:
        def __init__(self):
            self.taken = 0
            self.held = False

        def __enter__(self):
            self.taken += 1
            self.held = True

        def __exit__(self, *_exc):
            self.held = False
            return False

    lock = RecordingLock()
    stub = types.ModuleType("dgx_monarch.mesh")
    stub._MESH_LOCK = lock
    stub._MESHES = {("key",): _handle(abandoned_sample_leases={0: 1})}
    original = sys.modules.get("dgx_monarch.mesh")
    sys.modules["dgx_monarch.mesh"] = stub
    try:
        block = mesh_health.mesh_health_snapshot()
    finally:
        if original is None:
            del sys.modules["dgx_monarch.mesh"]
        else:
            sys.modules["dgx_monarch.mesh"] = original
    assert block["state"] == "dirty"
    assert lock.taken == 1
    assert not lock.held


def test_snapshot_reports_pending_or_creating_meshes_without_false_idle():
    lock = type("Lock", (), {
        "__enter__": lambda self: self,
        "__exit__": lambda self, *_exc: False,
    })()
    pending = _handle()
    attempt = SimpleNamespace(handle=pending)
    stub = types.ModuleType("dgx_monarch.mesh")
    stub._MESH_LOCK = lock
    stub._MESHES = {}
    stub._MESH_PENDING = {("key",): pending}
    stub._MESH_CREATING = {("key",): attempt}
    original = sys.modules.get("dgx_monarch.mesh")
    sys.modules["dgx_monarch.mesh"] = stub
    try:
        block = mesh_health.mesh_health_snapshot()
    finally:
        if original is None:
            del sys.modules["dgx_monarch.mesh"]
        else:
            sys.modules["dgx_monarch.mesh"] = original
    assert block["state"] == "busy"
    assert block["pending"] is True
    assert block["busy_phase"] == "mesh creation"


def test_snapshot_reports_stranded_pending_mesh_as_dirty_and_keeps_public_precedence():
    lock = type("Lock", (), {
        "__enter__": lambda self: self,
        "__exit__": lambda self, *_exc: False,
    })()
    public = _handle()
    stub = types.ModuleType("dgx_monarch.mesh")
    stub._MESH_LOCK = lock
    stub._MESHES = {("public",): public}
    stub._MESH_PENDING = {("same",): public, ("stranded",): _handle()}
    stub._MESH_CREATING = {}
    original = sys.modules.get("dgx_monarch.mesh")
    sys.modules["dgx_monarch.mesh"] = stub
    try:
        block = mesh_health.mesh_health_snapshot()
    finally:
        if original is None:
            del sys.modules["dgx_monarch.mesh"]
        else:
            sys.modules["dgx_monarch.mesh"] = original
    assert block["state"] == "dirty"
    assert block["pending"] is True


def _exited_frame():
    """A frame whose function has already returned, so it owns nothing."""
    return sys._getframe()


def _creation_attempt(handle, *, live):
    """A creation attempt owned by this test's frame, or by an exited one."""
    return SimpleNamespace(
        handle=handle,
        owner_thread_id=threading.get_ident(),
        owner_frame=sys._getframe(1) if live else _exited_frame(),
    )


def _snapshot_with(*, meshes, pending, creating, poison=None):
    """Run the snapshot against a stub mesh module holding these registries."""
    lock = type("Lock", (), {
        "__enter__": lambda self: self,
        "__exit__": lambda self, *_exc: False,
    })()
    stub = types.ModuleType("dgx_monarch.mesh")
    stub._MESH_LOCK = lock
    stub._MESHES = meshes
    stub._MESH_PENDING = pending
    stub._MESH_CREATING = creating
    stub._TRANSPORT_POISON = poison
    original = sys.modules.get("dgx_monarch.mesh")
    sys.modules["dgx_monarch.mesh"] = stub
    try:
        return mesh_health.mesh_health_snapshot()
    finally:
        if original is None:
            del sys.modules["dgx_monarch.mesh"]
        else:
            sys.modules["dgx_monarch.mesh"] = original


def test_a_poisoned_transport_is_not_idle_behind_an_empty_cache():
    """`_TRANSPORT_POISON` is the one global that makes every get_mesh raise.

    The snapshot must read it beside the three registries: without it, a
    driver whose transport cannot be reused reports idle, doctor passes, and
    every attach fails.
    """
    block = _snapshot_with(
        meshes={}, pending={}, creating={},
        poison="not reusable after a partial bring-up failure")

    assert block["state"] == "poisoned"
    assert block["poisoned"] is True
    assert "partial bring-up failure" in block["reason"]
    assert "restart ComfyUI" in block["remedy"]
    assert mesh_evidence.mesh_block_dirty(block)
    # A state this reader cannot name is a view, not an observation, and poison
    # proves every attach fails, so `poisoned` must be a named state.
    assert not mesh_evidence.mesh_block_unobserved(block)
    assert mesh_health_row.doctor_verdict(block)[0] == "fail"
    assert "POISONED" in mesh_health_row.status_line(block)


def test_a_poisoned_transport_outranks_a_cached_fleet_that_reads_healthy():
    """Poison is process state, so a clean cached handle does not override it."""
    block = _snapshot_with(
        meshes={("key",): _handle()}, pending={}, creating={}, poison="not reusable")

    assert block["state"] == "poisoned"
    assert block["verdict"] == "live"


def test_a_stranded_pending_handle_is_not_sent_to_a_route_that_cannot_see_it():
    """The reset route reads `_MESHES` only.

    With that registry empty it answers `no_live_mesh` and changes nothing, so
    a remedy that sent the operator to the reset button would leave the row
    where it was.
    """
    block = _snapshot_with(meshes={}, pending={("key",): _handle()}, creating={})

    assert block["state"] == "dirty"
    assert not block["remedy"].startswith("reset the attached mesh")
    assert "the next render" in block["remedy"]
    # The next render has two answers, not one, so the remedy names both.
    assert "typed error" in block["remedy"] and "restart" in block["remedy"]


def test_the_stranded_remedy_names_the_answer_the_next_render_actually_gives():
    """The remedy is a claim about `admission_occupant`, so read it there.

    A stranded fleet that is already gone is dropped and replaced, and the row
    clears on its own. A stranded fleet that is still live is refused, with the
    restart named, because a second fleet on one transport is worse than a
    stopped render. A remedy that promised only the first would send an
    operator to a render that refuses.
    """
    live, dead = _handle(), _handle()
    runtime = {
        "_MESHES": {},
        "_MESH_PENDING": {("key",): live},
        "_coherent_lifecycle_verdict": lambda handle: (
            "live" if handle is live else "completed"),
        "MeshAttachError": RuntimeError,
    }

    with pytest.raises(RuntimeError, match="restart ComfyUI"):
        mesh_creation.admission_occupant(runtime, ("key",))

    runtime["_MESH_PENDING"] = {("key",): dead}
    assert mesh_creation.admission_occupant(runtime, ("key",)) is None


def test_an_abandoned_creation_does_not_hide_its_stranded_pending_handle():
    """An attempt whose owner exited is not work the operator can wait for.

    Reported as busy, it would print "wait for it, nothing to fix" about a
    creation nobody drives and hide the dirty stranded handle beside it.
    get_mesh reaps the attempt before the next attach, so the row reports the
    pending handle.
    """
    stranded = _handle()
    block = _snapshot_with(
        meshes={}, pending={("key",): stranded},
        creating={("key",): _creation_attempt(stranded, live=False)})
    assert block["state"] == "dirty"
    assert block["busy_phase"] == ""
    assert block["pending"] is True


def test_a_creation_whose_owner_is_still_running_stays_busy():
    """The busy row is right while the owning get_mesh frame is on the stack."""
    pending = _handle()
    block = _snapshot_with(
        meshes={}, pending={("key",): pending},
        creating={("key",): _creation_attempt(pending, live=True)})
    assert block["state"] == "busy"
    assert block["busy_phase"] == "mesh creation"


def test_an_abandoned_creation_that_published_no_handle_reports_idle():
    """Nothing was attached and nothing was stranded, so nothing is wrong."""
    block = _snapshot_with(
        meshes={}, pending={},
        creating={("key",): _creation_attempt(None, live=False)})
    assert block["state"] == "idle"


def test_the_snapshot_never_raises_at_the_telemetry_route():
    stub = types.ModuleType("dgx_monarch.mesh")  # no registry on it at all
    original = sys.modules.get("dgx_monarch.mesh")
    sys.modules["dgx_monarch.mesh"] = stub
    try:
        block = mesh_health.mesh_health_snapshot()
    finally:
        if original is None:
            del sys.modules["dgx_monarch.mesh"]
        else:
            sys.modules["dgx_monarch.mesh"] = original
    assert block["state"] == "unknown"


def test_the_telemetry_route_carries_the_mesh_block(monkeypatch):
    from dgx_monarch.nodes import routes

    monkeypatch.setattr(routes, "_workers", lambda: [])
    monkeypatch.setattr(routes, "_ledger_summary", lambda: {})
    payload = routes._telemetry_uncached()
    assert set(payload["mesh"]) == set(mesh_health.MeshHealth().as_dict())


def _block(**overrides) -> dict:
    return mesh_health.health_of([_handle(**overrides)]).as_dict()


def test_status_says_so_when_no_driver_answers():
    assert mesh_health_row.status_line(None) == "no driver reachable (no mesh view)"
    assert mesh_health_row.doctor_verdict(None)[0] == "ok"


def test_status_prints_one_row_under_the_host_rows(monkeypatch, capsys):
    monkeypatch.setattr(
        mesh_health_row.lifecycle, "status",
        lambda _config: [{"host": "worker", "loop": "running", "port": "open"}],
    )
    monkeypatch.setattr(mesh_health_row, "fetch_mesh_block", lambda: _block())
    rows = mesh_health_row.status_with_mesh(ClusterConfig())
    assert rows == [{"host": "worker", "loop": "running", "port": "open"}]
    assert capsys.readouterr().out == "  mesh: ok (cached, 0 active leases)\n"


def test_the_status_row_names_the_abandoned_count_and_the_remedy(monkeypatch, capsys):
    monkeypatch.setattr(mesh_health_row.lifecycle, "status", lambda _config: [])
    monkeypatch.setattr(
        mesh_health_row, "fetch_mesh_block",
        lambda: _block(abandoned_sample_leases={0: 1}),
    )
    mesh_health_row.status_with_mesh(ClusterConfig())
    line = capsys.readouterr().out
    assert line.startswith("  mesh: DIRTY (1 abandoned sample may still be running")
    assert "Reset attached mesh" in line
    assert "dgxm restart" in line
    assert "docs/TROUBLESHOOTING.md #65" in line


def test_the_entry_the_row_points_at_carries_the_two_rows_no_reset_clears():
    """Both rows point at docs/TROUBLESHOOTING.md #65, so that entry must answer them.

    It must cover the two dirty states the reset button cannot clear: a
    poisoned transport, which needs a ComfyUI restart, and a pending handle the
    reset route cannot see.
    """
    entry = _troubleshooting_entry(65)

    assert "POISONED" in entry
    assert "restart ComfyUI" in entry
    assert "no_live_mesh" in entry
    assert "created and never published" in entry


def test_one_active_lease_is_singular():
    assert mesh_health_row.status_line(
        _block(sample_leases={0: 1})) == "ok (cached, 1 active lease)"


def test_a_driver_with_no_mesh_view_is_not_reported_as_healthy():
    assert mesh_health_row.status_line({}) == "no mesh view (this driver serves no mesh block)"
    kind, detail = mesh_health_row.doctor_verdict({})
    assert kind == "warn"
    assert "older driver" in detail


def test_an_unreadable_mesh_state_is_not_worded_as_an_old_driver():
    # A current driver that served a block it could not read must not be sent
    # for an upgrade it already has.
    block = mesh_health.MeshHealth(verdict="unknown").as_dict()
    line, (kind, detail) = (mesh_health_row.status_line(block),
                            mesh_health_row.doctor_verdict(block))
    assert line == "unknown (the driver could not read its own mesh state)"
    assert kind == "warn"
    assert "older driver" not in detail
    assert "could not read its own mesh state" in detail


def test_the_busy_row_passes_doctor_and_names_the_operation():
    block = _block(setup_cleanup_state=_in_flight())
    assert mesh_health_row.status_line(block) == (
        "busy (load_model RPC completion in flight); wait for it, nothing to fix")
    kind, detail = mesh_health_row.doctor_verdict(block)
    assert kind == "ok"
    assert "load_model RPC completion in flight" in detail
    # The destructive advice must not appear next to a working load.
    assert "recycle" not in detail.lower()


def test_a_state_from_a_newer_driver_asks_for_a_cli_upgrade_not_a_driver_one():
    block = {"state": "quarantined"}
    assert "quarantined" in mesh_health_row.status_line(block)
    kind, detail = mesh_health_row.doctor_verdict(block)
    assert kind == "warn"
    assert "upgrade dgxm on this box" in detail


def test_every_state_the_accessor_emits_renders_its_own_row():
    blocks = [
        mesh_health.health_of([]).as_dict(),
        _block(),
        _block(setup_cleanup_state=_in_flight()),
        _block(abandoned_sample_leases={0: 1}),
        mesh_health.MeshHealth(verdict="unknown").as_dict(),
        {},
    ]
    assert len({mesh_health_row.status_line(block) for block in blocks}) == len(blocks)
    assert [mesh_health_row.doctor_verdict(block)[0] for block in blocks] == [
        "ok", "ok", "ok", "fail", "warn", "warn"]


def test_the_row_reads_only_keys_the_accessor_emits():
    emitted = set(mesh_health.MeshHealth().as_dict())
    assert {"state", "active_leases", "reason", "remedy"} <= emitted


def _fake_urlopen(payloads: dict):
    """Answer the named ports; every other one refuses, as a real box does.

    Discovery always offers 8188 and 8191, so the refusal has to be the
    exact shape urllib raises: `_probe_refused` treats only that as proof
    that nothing is there, and anything else as a driver it could not read.
    """
    def urlopen(url, timeout=None):
        for candidate, payload in payloads.items():
            if candidate in url:
                if isinstance(payload, BaseException):
                    raise payload
                return io.BytesIO(json.dumps(payload).encode())
        raise urllib.error.URLError(ConnectionRefusedError(111, "refused"))

    return urlopen


def _driver_at(monkeypatch, *ports):
    """Pin discovery so a test never depends on what this box is running."""
    monkeypatch.setattr(
        mesh_health_row.comfy_ports, "find_comfy_ports",
        lambda: dict.fromkeys(ports))


def test_fetch_returns_none_when_nothing_answers(monkeypatch):
    _driver_at(monkeypatch, 8188, 8191)
    monkeypatch.setattr(mesh_health_row.urllib.request, "urlopen", _fake_urlopen({}))
    assert mesh_health_row.fetch_mesh_block() is None


def test_fetch_prefers_the_driver_that_reports_a_dirty_fleet(monkeypatch):
    _driver_at(monkeypatch, 8188, 8191)
    monkeypatch.setattr(mesh_health_row.urllib.request, "urlopen", _fake_urlopen({
        "8191": {"mesh": mesh_health.health_of([]).as_dict()},
        "8188": {"mesh": _block(abandoned_sample_leases={0: 1})},
    }))
    block = mesh_health_row.fetch_mesh_block()
    assert block is not None
    assert block["state"] == "dirty"


def test_fetch_prefers_a_busy_driver_over_a_quiet_one(monkeypatch):
    _driver_at(monkeypatch, 8188, 8191)
    monkeypatch.setattr(mesh_health_row.urllib.request, "urlopen", _fake_urlopen({
        "8191": {"mesh": _block()},
        "8188": {"mesh": _block(setup_cleanup_state=_in_flight())},
    }))
    block = mesh_health_row.fetch_mesh_block()
    assert block is not None
    assert block["state"] == "busy"


def test_fetch_finds_a_driver_on_a_port_no_convention_names(monkeypatch):
    # `PORT=8195 scripts/comfy-driver.sh` is a supported launch, and the fixed
    # pair would report the dirty fleet on it as no driver reachable.
    _driver_at(monkeypatch, 8188, 8191, 8195)
    monkeypatch.setattr(mesh_health_row.urllib.request, "urlopen", _fake_urlopen({
        "8195": {"mesh": _block(abandoned_sample_leases={0: 1})},
    }))
    block = mesh_health_row.fetch_mesh_block()
    assert block is not None
    assert block["state"] == "dirty"
    assert mesh_health_row.doctor_verdict(block)[0] == "fail"


def test_discovery_is_the_same_one_doctors_frontend_row_uses():
    # One row finding a driver the next one misses is how a real fault reads
    # as green in the same doctor run.
    assert doctor._find_comfy_ports is mesh_health_row.comfy_ports.find_comfy_ports


def test_fetch_distinguishes_no_driver_from_a_driver_with_no_view(monkeypatch):
    _driver_at(monkeypatch, 8188, 8191)
    monkeypatch.setattr(mesh_health_row.urllib.request, "urlopen", _fake_urlopen({
        "8191": {"workers": []},
    }))
    assert mesh_health_row.fetch_mesh_block() == {}


def test_doctor_prints_the_mesh_row(monkeypatch, capsys):
    monkeypatch.setattr(mesh_health_row, "fetch_mesh_block", lambda: _block())
    row = doctor._mesh_health_row()
    assert row["status"] == doctor._OK
    assert "[ ok ] mesh health: attached mesh ready, 0 active leases" in capsys.readouterr().out


def test_a_dirty_mesh_fails_a_doctor_run_that_is_otherwise_green(monkeypatch, capsys):
    """Doctor counts a dirty mesh as a failure even when every host row passes."""
    torch = types.ModuleType("torch")
    torch.__version__ = "2.test"
    torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: TORCHMONARCH_PIN)
    monkeypatch.setattr(
        doctor, "_frontend_skew_row",
        lambda: {"status": doctor._OK, "name": "frontend", "detail": "stub"},
    )
    monkeypatch.setattr(doctor, "_spark_health_rows", lambda _config: [])
    monkeypatch.setattr(doctor.socket, "gethostname", lambda: "driver")
    monkeypatch.setattr(doctor.socket, "gethostbyname", lambda _name: "10.0.0.1")
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/rsync")
    monkeypatch.setattr(doctor, "_tcp_probe", lambda *_a, **_kw: False)
    monkeypatch.setattr(
        doctor, "passive_worker_health",
        lambda *_a, **_kw: {"running": True, "listening": True, "healthy": True},
    )
    monkeypatch.setattr(
        doctor, "run_on_host",
        lambda *_a, **_kw: subprocess.CompletedProcess([], 0, (
            f"torchmonarch={TORCHMONARCH_PIN} torch=2.test comfy=True nccl_proto=unset\n"
            "fabric_iface=up rdma_active=1\n"), ""),
    )
    monkeypatch.setattr(
        mesh_health_row, "fetch_mesh_block",
        lambda: _block(abandoned_sample_leases={0: 1}),
    )
    config = ClusterConfig(
        hosts=(HostConfig(name="worker", address="tcp://10.0.0.2:26600"),),
        client_bind="tcp://10.0.0.1:0",
        fabric_profile="generic-roce",
        transport_security="trusted_fabric",
        source="test.toml",
    )
    assert not doctor.run_doctor(config)
    output = capsys.readouterr().out
    assert "[ ok ] worker worker service" in output
    assert "[FAIL] mesh health: 1 abandoned sample may still be running" in output
    assert "1 failures" in output


def test_an_unanswered_nccl_master_probe_is_never_printed_as_free(monkeypatch, capsys):
    """A connect timeout observes nothing about the port.

    `_tcp_probe` returns None on timeout. Printed as `free`, a host that never
    answered would read as a clean NCCL master port.
    """
    torch = types.ModuleType("torch")
    torch.__version__ = "2.test"
    torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: TORCHMONARCH_PIN)
    monkeypatch.setattr(
        doctor, "_frontend_skew_row",
        lambda: {"status": doctor._OK, "name": "frontend", "detail": "stub"},
    )
    monkeypatch.setattr(doctor, "_spark_health_rows", lambda _config: [])
    monkeypatch.setattr(doctor.socket, "gethostname", lambda: "driver")
    monkeypatch.setattr(doctor.socket, "gethostbyname", lambda _name: "10.0.0.1")
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/rsync")
    monkeypatch.setattr(doctor, "_tcp_probe", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        doctor, "passive_worker_health",
        lambda *_a, **_kw: {"running": True, "listening": True, "healthy": True},
    )
    monkeypatch.setattr(
        doctor, "run_on_host",
        lambda *_a, **_kw: subprocess.CompletedProcess([], 0, (
            f"torchmonarch={TORCHMONARCH_PIN} torch=2.test comfy=True nccl_proto=unset\n"
            "fabric_iface=up rdma_active=1\n"), ""),
    )
    monkeypatch.setattr(mesh_health_row, "fetch_mesh_block", lambda: _block())
    config = ClusterConfig(
        hosts=(HostConfig(name="worker", address="tcp://10.0.0.2:26600"),),
        client_bind="tcp://10.0.0.1:0",
        fabric_profile="generic-roce",
        transport_security="trusted_fabric",
        source="test.toml",
    )
    rows = doctor._doctor_rows(config)
    output = capsys.readouterr().out
    assert "nccl master port: unobserved:" in output
    assert "29500 free" not in output
    row = next(item for item in rows if item["name"] == "nccl master port")
    assert probe_certainty.unobserved(row)
    # Advisory, not gating: a blind port reading does not fail the run.
    assert row.get("critical") is not True


# `dgxm status` exits 0 only for a mesh observed clean, 75 for a view that
# proves nothing, and 1 for a fault.
_GREEN_HOST = {"host": "worker", "loop": "running", "port": "open",
               "address": "10.0.0.2:29500", "mode": "systemd"}


def _status_exit(monkeypatch, block, hosts=(_GREEN_HOST,)):
    from dgx_monarch.cli import main as cli

    monkeypatch.setattr(cli, "_require_config", lambda _args: ClusterConfig())
    monkeypatch.setattr(mesh_health_row.lifecycle, "status", lambda _config: list(hosts))
    monkeypatch.setattr(mesh_health_row, "fetch_mesh_block", lambda: block)
    return cli.cmd_status(SimpleNamespace(json=False, config=None))


def test_status_exits_nonzero_on_a_mesh_it_could_not_observe(monkeypatch, capsys):
    """A view that proves nothing exits 75, neither 0 nor 1.

    No driver, no mesh block, a driver that cannot read its own state, and a
    state this dgxm does not know prove nothing. ComfyUI being down is normal,
    but it is not evidence that the mesh is clean.
    """
    for block in (None, {}, {"state": "unknown"}, {"state": "quiescing"}):
        assert _status_exit(monkeypatch, block) == 75, block
    capsys.readouterr()


def test_status_exits_zero_only_when_the_mesh_was_observed_and_clean(monkeypatch, capsys):
    for block in (_block(), {"state": "idle", "verdict": "none"},
                  {"state": "busy", "verdict": "unresolved", "busy_phase": "a load"}):
        assert _status_exit(monkeypatch, block) == 0, block
    capsys.readouterr()


def test_status_fails_on_every_field_that_makes_a_mesh_unusable(monkeypatch, capsys):
    """Status and readiness both read `mesh_evidence.mesh_block_dirty`, so
    status exits 1 on every field readiness refuses on, not on `state` alone."""
    for block in (
        _block(abandoned_sample_leases={0: 1}),
        {"state": "ok", "verdict": "live", "poisoned": True},
        {"state": "ok", "verdict": "live", "replacement_blocked": "stop failed"},
        {"state": "ok", "verdict": "live", "abandoned_samples": 2},
        {"state": "blocked", "verdict": "live"},
        {"state": "ok", "verdict": "unresolved"},
    ):
        assert _status_exit(monkeypatch, block) == 1, block
    capsys.readouterr()


def test_a_down_worker_loop_still_outranks_an_unobserved_mesh(monkeypatch, capsys):
    down = {**_GREEN_HOST, "loop": "stopped"}
    assert _status_exit(monkeypatch, None, hosts=(down,)) == 1
    capsys.readouterr()


def test_a_worker_loop_nobody_could_read_exits_unknown_not_failed(monkeypatch, capsys):
    """A host command that said nothing is not a stopped loop.

    An unknown loop or port exits 75, not 1, so a flaky SSH hop does not read
    as a dead box.
    """
    for blind in ({**_GREEN_HOST, "loop": "unknown"},
                  {**_GREEN_HOST, "port": "unknown"}):
        assert _status_exit(monkeypatch, _block(), hosts=(blind,)) == 75, blind
    capsys.readouterr()


def test_a_stopped_loop_outranks_a_blind_one(monkeypatch, capsys):
    hosts = ({**_GREEN_HOST, "loop": "unknown"},
             {**_GREEN_HOST, "host": "worker2", "port": "closed"})
    assert _status_exit(monkeypatch, _block(), hosts=hosts) == 1
    capsys.readouterr()


def test_a_driver_that_could_not_be_read_outranks_a_calm_one(monkeypatch):
    """The loaded driver holding a dirty fleet is the one whose telemetry route
    times out; dropping it would let an idle responder answer for it."""
    _driver_at(monkeypatch, 8188, 8191)
    monkeypatch.setattr(mesh_health_row.urllib.request, "urlopen", _fake_urlopen({
        "8188": {"mesh": {"state": "idle", "verdict": "none"}},
        "8191": TimeoutError("telemetry timed out"),
    }))
    blind = mesh_health_row.fetch_mesh_block()
    assert mesh_health_row.unobserved(blind)
    # A driver that did not answer needs looking at; a driver that serves no
    # mesh block needs an upgrade. The row must not send the operator to the
    # wrong one.
    assert mesh_health_row.state_name(blind) == "driver-unreadable"
    assert "no readable answer" in mesh_health_row.doctor_verdict(blind)[1]
    assert "upgrade" not in mesh_health_row.doctor_verdict(blind)[1]
    assert "no readable answer" in mesh_health_row.status_line(blind)
    assert mesh_health_row.doctor_verdict(blind)[0] == "warn"
    empty = mesh_health_row.doctor_verdict({})
    assert "upgrade it" in empty[1] and empty[0] == "warn"
    assert mesh_health_row.state_name({}) == "no-mesh-block"

    # A refused port is still a non-event: discovery offers 8188 and 8191
    # whether or not anything listens there.
    monkeypatch.setattr(mesh_health_row.urllib.request, "urlopen", _fake_urlopen({
        "8191": {"mesh": {"state": "idle", "verdict": "none"}},
    }))
    assert mesh_health_row.fetch_mesh_block() == {"state": "idle", "verdict": "none"}

    # The route's own 503 is a driver, not an absence.
    monkeypatch.setattr(mesh_health_row.urllib.request, "urlopen", _fake_urlopen({
        "8188": {"mesh": {"state": "idle", "verdict": "none"}},
        "8191": urllib.error.HTTPError("http://x", 503, "busy", {}, None),
    }))
    assert mesh_health_row.state_name(mesh_health_row.fetch_mesh_block()) == (
        "driver-unreadable")


def test_a_driver_that_never_answered_outranks_one_that_answered_blank(monkeypatch):
    """Both views are blind, so a tie would fall to port order.

    `find_comfy_ports` returns candidates sorted, so an older responder on a
    lower port would keep the upgrade advice and hide the driver that never
    answered.
    """
    _driver_at(monkeypatch, 8188, 8191)
    monkeypatch.setattr(mesh_health_row.urllib.request, "urlopen", _fake_urlopen({
        "8188": {"mesh": {}},
        "8191": TimeoutError("telemetry timed out"),
    }))
    assert mesh_health_row.state_name(
        mesh_health_row.fetch_mesh_block()) == "driver-unreadable"

    # Reversed ports: the answer must not depend on which was probed first.
    monkeypatch.setattr(mesh_health_row.urllib.request, "urlopen", _fake_urlopen({
        "8188": TimeoutError("telemetry timed out"),
        "8191": {"mesh": {}},
    }))
    assert mesh_health_row.state_name(
        mesh_health_row.fetch_mesh_block()) == "driver-unreadable"

    # A blind view still outranks any healthy answer, and dirty still wins.
    monkeypatch.setattr(mesh_health_row.urllib.request, "urlopen", _fake_urlopen({
        "8188": {"mesh": {"state": "busy", "verdict": "unresolved",
                          "busy_phase": "a load"}},
        "8191": {"mesh": {}},
    }))
    assert mesh_health_row.state_name(
        mesh_health_row.fetch_mesh_block()) == "no-mesh-block"
    monkeypatch.setattr(mesh_health_row.urllib.request, "urlopen", _fake_urlopen({
        "8188": TimeoutError("telemetry timed out"),
        "8191": {"mesh": _block(abandoned_sample_leases={0: 1})},
    }))
    assert mesh_health_row.doctor_verdict(mesh_health_row.fetch_mesh_block())[0] == "fail"
