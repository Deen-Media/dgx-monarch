"""A render, load or node that starts once the driver is exiting holds at the entry.

ComfyUI's prompt worker is a daemon thread that takes every queued prompt until
the interpreter stops it, and prints two ERROR records with a traceback for each
prompt that fails. So a prompt that reaches the pack after the exit notice waits
at the top of ``mesh.get_mesh`` or ``mesh.ensure_live`` (driver_exit.hold_at_entry):
before any lock, registry change, attach, id reading or price, logging nothing,
and the worker takes no other prompt; the process ends and the thread with it.
Only after ``HOLD_AT_EXIT_S`` (30 s; short here) does it raise the exit refusal.

CPU only; no Monarch, no ssh. Each row has its twin with no exit.
"""
from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest

from dgx_monarch import driver_exit, mesh_liveness, mesh_setup
from dgx_monarch import mesh as mesh_mod
from topology_transition_helpers import _handle, _Workers

EXITING = pytest.mark.parametrize("exiting", [True, False], ids=["exiting", "not_exiting"])
BOUND_S = 0.4


class _Stop(Exception):
    """Ends a not-exiting twin at the first step past the entry."""


@pytest.fixture
def flag(monkeypatch):
    """A fresh exit flag and a short hold bound for the test, put back after it."""
    event = threading.Event()
    monkeypatch.setattr(driver_exit, "_EXITING", event)
    monkeypatch.setattr(driver_exit, "HOLD_AT_EXIT_S", BOUND_S)
    return event


def _kept(monkeypatch) -> list[logging.LogRecord]:
    """Every record of the loggers the entry and its callers write through."""
    kept: list[logging.LogRecord] = []
    keep = logging.Handler()
    keep.emit = kept.append  # type: ignore[method-assign]
    for name in ("dgx_monarch.mesh", "dgx_monarch.mesh_liveness", "dgx_monarch.nodes.render_submit",
                 "dgx_monarch.restart_reattach", "dgx_monarch.mesh_attach"):
        logger = logging.getLogger(name)
        monkeypatch.setattr(logger, "handlers", [*logger.handlers, keep])
    return kept


@EXITING
def test_a_get_mesh_that_starts_at_exit_holds_then_refuses_at_the_bound(
        flag, monkeypatch, exiting):
    """Held, it reaches nothing: not the fault hook's install, not the config
    lookup, nor anything after them. Only at the bound does it raise the exit
    refusal, which names the bound. Not exiting, it goes straight on."""
    reached: list[str] = []
    monkeypatch.setattr(mesh_mod, "install_fault_hook", lambda: reached.append("hook"))

    def find(_path):
        reached.append("config")
        raise _Stop

    monkeypatch.setattr(mesh_mod, "find_config_path", find)
    kept = _kept(monkeypatch)
    started = time.monotonic()
    if not exiting:
        with pytest.raises(_Stop):
            mesh_mod.get_mesh(mode="cluster", config_path="/fake/cluster.toml")
        assert reached == ["hook", "config"] and time.monotonic() - started < BOUND_S
        return
    flag.set()
    with pytest.raises(driver_exit.DriverExitingError, match=r"did not start.*waited 0 s") as raised:
        mesh_mod.get_mesh(mode="cluster", config_path="/fake/cluster.toml")
    assert time.monotonic() - started >= BOUND_S and reached == [] and kept == []
    assert "the driver is exiting" in str(raised.value)


@EXITING
def test_an_ensure_live_that_starts_at_exit_holds_before_the_liveness_probe(
        flag, monkeypatch, exiting):
    handle, _old, _new = _handle(_Workers([]))
    probed: list[object] = []

    def try_release(target):
        probed.append(target)
        raise _Stop

    monkeypatch.setattr(mesh_liveness, "try_release", try_release)
    started = time.monotonic()
    if not exiting:
        with pytest.raises(_Stop):
            mesh_mod.ensure_live(handle)
        assert probed == [handle]
        return
    flag.set()
    with pytest.raises(driver_exit.DriverExitingError):
        mesh_mod.ensure_live(handle)
    assert time.monotonic() - started >= BOUND_S and probed == []


def test_a_thread_that_owns_the_handles_lock_refuses_at_once(flag, monkeypatch):
    """No caller enters holding the handle's lock, which a stop takes; were one
    to, it would refuse at once rather than hold that lock until the process ends."""
    handle, _old, _new = _handle(_Workers([]))
    monkeypatch.setattr(driver_exit, "HOLD_AT_EXIT_S", 30.0)
    flag.set()
    started = time.monotonic()
    with handle.lock, pytest.raises(driver_exit.DriverExitingError, match="did not start") as raised:
        mesh_mod.ensure_live(handle)
    assert time.monotonic() - started < 1.0 and "waited" not in str(raised.value)


def test_a_held_thread_holds_no_lock_and_logs_nothing(flag, monkeypatch):
    """While a prompt's thread holds at the entry, every lock a stop, a creation
    or the exit hook takes stays free, and nothing is logged; it ends with the
    refusal at the bound."""
    handle, _old, _new = _handle(_Workers([]))
    monkeypatch.setattr(driver_exit, "HOLD_AT_EXIT_S", 1.5)
    kept = _kept(monkeypatch)
    flag.set()
    outcome: list[BaseException] = []

    def prompt() -> None:
        try:
            mesh_mod.ensure_live(handle)
        except BaseException as exc:  # the row reads what it raised
            outcome.append(exc)

    worker = threading.Thread(target=prompt, daemon=True)
    worker.start()
    time.sleep(0.3)
    assert worker.is_alive() and outcome == []  # held
    for lock in (mesh_mod._MESH_LOCK, handle.lock, driver_exit._LOCK):
        assert lock.acquire(timeout=0.2), lock
        lock.release()
    worker.join(5.0)
    assert not worker.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], driver_exit.DriverExitingError)
    assert kept == []


def test_with_prompts_queued_exactly_one_enters_and_holds(flag, monkeypatch):
    """ComfyUI's prompt worker, four prompts queued behind the cut one: the first
    to reach the pack after the notice holds, and the worker takes no other. None
    raises while the process could still be ending."""
    handle, _old, _new = _handle(_Workers([]))
    monkeypatch.setattr(driver_exit, "HOLD_AT_EXIT_S", 2.0)
    entered: list[int] = []
    raised: list[int] = []
    real_hold = driver_exit.hold_at_entry

    def hold(target=None):
        entered.append(1)
        return real_hold(target)

    monkeypatch.setattr(driver_exit, "hold_at_entry", hold)
    flag.set()

    def prompt_worker() -> None:
        for _prompt in range(4):
            try:
                mesh_mod.ensure_live(handle)
            except driver_exit.DriverExitingError:
                raised.append(1)

    worker = threading.Thread(target=prompt_worker, daemon=True)
    worker.start()
    time.sleep(1.0)
    assert entered == [1] and raised == []  # one prompt holds; the others wait in the queue
    flag.clear()  # the process would have ended; let the worker finish for the test
    worker.join(10.0)


@pytest.mark.parametrize("session", ["owned", "pinned"])
def test_a_render_that_starts_at_exit_holds_and_its_refusal_heals_nothing(
        flag, monkeypatch, session):
    """Through the real submit_render, for a render that owns its session and one
    in a claimed Pipeline or Gate session: both hold (a session is no lock that
    anything at exit waits for), and the refusal at the bound reaches the caller
    as raised, with no eviction, no heal, no pinned retire, no abandoned-lease
    reset and no old-model clear, and no pack log line."""
    from dgx_monarch import mesh_helpers, restart_reattach
    from dgx_monarch.mesh_session import RenderSession
    from dgx_monarch.nodes import abandoned_heal, common, old_model_clear, render_submit

    handle, _old, _new = _handle(_Workers([]))
    model = SimpleNamespace(mesh=SimpleNamespace(handle=handle))
    monkeypatch.setattr(common, "_bind_packed_render_model", lambda m, _latent, _cfg: (m, None))
    monkeypatch.setattr(render_submit, "model_for_request", lambda m, _request: m)
    for module, name in ((mesh_helpers, "_publish_defunct"), (restart_reattach, "heal"),
                         (restart_reattach, "refuse_pinned"),
                         (abandoned_heal, "heal_abandoned_wedge_once"),
                         (old_model_clear, "clear_for_retry")):
        monkeypatch.setattr(module, name, lambda *_a, _name=name, **_k: pytest.fail(f"{_name} ran"))
    kept = _kept(monkeypatch)
    flag.set()
    started = time.monotonic()
    if session == "owned":
        with pytest.raises(driver_exit.DriverExitingError):
            render_submit.submit_render(model, {}, {}, None, 1)
    else:
        pinned = RenderSession()
        pinned.bind(handle)
        try:
            with pinned.activate(), pytest.raises(driver_exit.DriverExitingError):
                render_submit.submit_render(model, {}, {}, None, 1)
        finally:
            flag.clear()
            pinned.close()
    assert time.monotonic() - started >= BOUND_S  # held, not refused at once
    assert handle.defunct is False and kept == []


@pytest.mark.parametrize("failure", ["exit", "other"])
def test_the_first_use_gate_takes_no_fallback_for_the_exits_refusal(monkeypatch, failure):
    """The first-use gate reaches the entry before the render does. Its own
    refusal at the bound is raised as it came: no "ERROR" verdict, so no stock
    fallback, no quarantine of the graph's levers and no second hold at the
    render's entry. Any other failure of the ceremony keeps its fallback."""
    from dgx_monarch.nodes import adaptive_steps, auto_gate, common, h3_ring_pad

    error = (driver_exit.DriverExitingError(driver_exit.NOT_STARTED) if failure == "exit"
             else RuntimeError("the ceremony failed"))

    def context(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(common, "_auto_gate_context", context)
    monkeypatch.setattr(h3_ring_pad, "preflight_for_render", lambda *_a: None)
    monkeypatch.setattr(adaptive_steps, "preflight_for_render", lambda *_a: None)
    model = SimpleNamespace(mesh=SimpleNamespace(worker_args={}))
    if failure == "exit":
        with pytest.raises(driver_exit.DriverExitingError):
            auto_gate.maybe_auto_gate(model, {}, {}, None, 1)
        return
    assert auto_gate.maybe_auto_gate(model, {}, {}, None, 1) == "ERROR"


def test_nothing_past_the_entry_holds(flag, monkeypatch):
    """A new call past the entry raises at once and never holds (R40 item 2)."""
    handle, _old, _new = _handle(_Workers([]))
    monkeypatch.setattr(driver_exit, "HOLD_AT_EXIT_S", 30.0)
    flag.set()
    started = time.monotonic()
    with pytest.raises(driver_exit.DriverExitingError):
        handle.call_all("status", timeout_s=1.0)
    with pytest.raises(driver_exit.DriverExitingError):
        mesh_setup.dispatch_sample(handle, lambda: None, setup_token=None)
    assert time.monotonic() - started < 1.0
