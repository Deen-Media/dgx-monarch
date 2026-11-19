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
    monkeypatch.setattr(driver_exit, "_REFUSED_AT_BOUND", threading.Event())
    monkeypatch.setattr(driver_exit, "_PROCESS_ENDS", threading.Event())
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


def _entering(handle, outcomes: list) -> threading.Thread:
    """One prompt's thread entering ensure_live; what it raises lands in ``outcomes``."""

    def prompt() -> None:
        try:
            mesh_mod.ensure_live(handle)
        except driver_exit.DriverExitingError as exc:
            outcomes.append((time.monotonic(), str(exc)))

    thread = threading.Thread(target=prompt, daemon=True)
    thread.start()
    return thread


@pytest.mark.parametrize("arrival", ["after_the_refusal", "together"])
def test_one_refusal_per_exit_however_long_the_hook_hangs(flag, monkeypatch, arrival):
    """An exit hook hangs past the bound with prompts still queued. The first entry
    to reach the bound raises the one refusal; every entry after it, and any other
    that reached the bound beside it, waits for the process to end with no bound,
    so no second refusal is printed. (Here the process's end is the test setting
    ``_PROCESS_ENDS``; until then nothing more is raised.) Before, each entry
    raised at its own bound: one refusal every 30 s for as long as the hook hung."""
    handle, _old, _new = _handle(_Workers([]))
    monkeypatch.setattr(driver_exit, "HOLD_AFTER_REFUSAL_S", 10.0)  # a backstop: the test ends the wait
    flag.set()
    outcomes: list[tuple[float, str]] = []
    started = time.monotonic()
    first = _entering(handle, outcomes)
    if arrival == "after_the_refusal":
        first.join(5.0)
        assert len(outcomes) == 1 and "waited" in outcomes[0][1]  # the one refusal, at the bound
        late = _entering(handle, outcomes)  # the next queued prompt enters after it
        threads = [late]
    else:
        threads = [first, _entering(handle, outcomes)]  # both reach the bound together
    time.sleep(BOUND_S * 3)
    assert time.monotonic() - started > BOUND_S * 2
    assert len(outcomes) == 1, outcomes  # past its own bound, the later entry still waits
    assert sum("waited" in text for _t, text in outcomes) == 1
    assert any(thread.is_alive() for thread in threads)
    driver_exit._PROCESS_ENDS.set()  # the process ends
    for thread in threads:
        thread.join(5.0)
    assert sum("waited" in text for _t, text in outcomes) == 1  # still the one refusal


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


@EXITING
def test_a_loader_that_starts_at_exit_holds_before_its_own_work(flag, monkeypatch, exiting):
    """The node wrapper (refusal_lead.lead_with_refusal) is the outermost entry: a
    loader that starts once the driver is exiting holds there, before any of its
    own work, so its relief and the requeue price ask the Workers for no quote
    (they run before the mesh entry when the driver's own price does not fit).
    Not exiting, its body runs as before."""
    from dgx_monarch import capacity_agreement
    from dgx_monarch.nodes import loaders
    from dgx_monarch.nodes.common import MeshSpec

    ran: list[str] = []
    monkeypatch.setattr(capacity_agreement, "collect", lambda *_a, **_k: ran.append("quote"))

    def validate(_dtype):
        ran.append("body")
        raise _Stop

    monkeypatch.setattr(loaders, "validate_weight_dtype", validate)
    mesh = MeshSpec(handle=object(), topology_preset="uly2", attention="TORCH_FLASH", sync_ulysses=True)
    started = time.monotonic()
    if not exiting:
        with pytest.raises(_Stop):
            loaders.DGXMonarchUNETLoader().load(mesh, "x.safetensors")
        assert ran == ["body"]
        return
    flag.set()
    with pytest.raises(driver_exit.DriverExitingError):
        loaders.DGXMonarchUNETLoader().load(mesh, "x.safetensors")
    assert ran == [] and time.monotonic() - started >= BOUND_S


def test_a_node_already_running_or_called_from_inside_another_is_not_held(flag, monkeypatch):
    """The wrapper holds only at a thread's outermost node entry. A node already
    running when the notice comes goes on, and a decorated method it calls from
    inside passes straight through; after it ends, the next outermost entry holds."""
    from dgx_monarch.nodes.refusal_lead import lead_with_refusal

    monkeypatch.setattr(driver_exit, "HOLD_AT_EXIT_S", 30.0)
    ran: list[str] = []

    @lead_with_refusal
    def inner() -> str:
        ran.append("inner")
        return "inner"

    @lead_with_refusal
    def outer() -> str:
        ran.append("outer")
        flag.set()  # the notice comes while this node runs
        return inner()

    started = time.monotonic()
    assert outer() == "inner" and ran == ["outer", "inner"]
    assert time.monotonic() - started < 1.0  # neither held
    monkeypatch.setattr(driver_exit, "HOLD_AT_EXIT_S", BOUND_S)
    with pytest.raises(driver_exit.DriverExitingError):
        inner()  # the next outermost entry
    assert ran == ["outer", "inner"]


def test_the_wrapper_puts_its_depth_back_whatever_raises_inside_it(flag, monkeypatch):
    """The wrapper raises the thread's node depth inside its try, so its finally
    puts the depth back whatever raises there, even before the node's own work
    starts (here the note's reset raises, as an asynchronous exception could at
    that line). A depth left at 1 would let the thread's next outermost node
    pass the hold at exit."""
    from dgx_monarch.nodes import refusal_lead

    resets: list[int] = []

    def clear_once() -> None:
        resets.append(1)
        if len(resets) == 1:
            raise _Stop("raised between the depth and the node's own work")

    monkeypatch.setattr(refusal_lead, "_clear", clear_once)

    @refusal_lead.lead_with_refusal
    def node() -> str:
        return "ran"

    with pytest.raises(_Stop):
        node()
    assert getattr(refusal_lead._ENTRY, "depth", 0) == 0
    flag.set()
    with pytest.raises(driver_exit.DriverExitingError, match="waited"):
        node()  # the next outermost entry still holds, then refuses at the bound


def test_reset_forgets_the_one_refusal_so_the_next_exit_refuses_at_its_bound(flag, monkeypatch):
    """``driver_exit.reset`` (the conftest, after each row) clears the exit flag
    and the mark of this exit's one refusal at the bound. Kept, the mark would
    make the next exit's first entry wait for the process to end instead of
    raising at its bound."""
    monkeypatch.setattr(driver_exit, "HOLD_AFTER_REFUSAL_S", 5.0)
    flag.set()
    with pytest.raises(driver_exit.DriverExitingError, match="waited"):
        driver_exit.hold_at_entry()
    assert driver_exit._REFUSED_AT_BOUND.is_set()
    driver_exit.reset()
    assert not flag.is_set() and not driver_exit._REFUSED_AT_BOUND.is_set()
    flag.set()
    started = time.monotonic()
    with pytest.raises(driver_exit.DriverExitingError, match="waited"):
        driver_exit.hold_at_entry()
    assert time.monotonic() - started < BOUND_S + 2.0


@pytest.mark.parametrize("failure", ["exit", "other"])
def test_the_pipelines_gate_lookup_passes_the_exits_refusal_through(monkeypatch, failure):
    """The Pipeline asks auto_gate_required whether a ceremony is due. Its lookup
    reaches the entry; the exit's refusal from there is raised as it came, where
    before the lookup swallowed it, logged a failed lookup and answered True, so
    the Pipeline went on, entered again and held, and the one refusal was never
    printed. Any other lookup failure still answers True (the fail-safe)."""
    from dgx_monarch.nodes import auto_gate, common

    error = (driver_exit.DriverExitingError(driver_exit.NOT_STARTED) if failure == "exit"
             else RuntimeError("the ledger could not be read"))

    def context(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(common, "_auto_gate_context", context)
    model = SimpleNamespace(mesh=SimpleNamespace(worker_args={}))
    if failure == "exit":
        with pytest.raises(driver_exit.DriverExitingError):
            auto_gate.auto_gate_required(model, "ksampler")
        return
    assert auto_gate.auto_gate_required(model, "ksampler") is True


def test_the_shipped_bound_is_the_30_s_the_docs_state():
    """Both fixtures replace the bound before any row reads it, so this row reads
    the shipped values from the source itself: the hold refuses at 30 s, and an
    entry after that one refusal waits with no bound. Every sentence of the docs
    that names the bound states the same number."""
    import ast
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    shipped = {}
    for node in ast.parse(Path(driver_exit.__file__).read_text()).body:  # the module imported
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            shipped[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            shipped[node.target.id] = node.value
    assert ast.literal_eval(shipped["HOLD_AT_EXIT_S"]) == 30.0
    assert ast.literal_eval(shipped["HOLD_AFTER_REFUSAL_S"]) is None
    stated: list[str] = []
    for doc in ("docs/TROUBLESHOOTING.md", "CHANGELOG.md"):
        stated += re.findall(r"still alive (\d+) s", " ".join((root / doc).read_text().split()))
    assert len(stated) >= 3 and set(stated) == {"30"}, stated  # entries 2 and 51, the CHANGELOG


def test_nothing_past_the_entry_holds(flag, monkeypatch):
    """A new call past the entry raises at once and never holds."""
    handle, _old, _new = _handle(_Workers([]))
    monkeypatch.setattr(driver_exit, "HOLD_AT_EXIT_S", 30.0)
    flag.set()
    started = time.monotonic()
    with pytest.raises(driver_exit.DriverExitingError):
        handle.call_all("status", timeout_s=1.0)
    with pytest.raises(driver_exit.DriverExitingError):
        mesh_setup.dispatch_sample(handle, lambda: None, setup_token=None)
    assert time.monotonic() - started < 1.0
