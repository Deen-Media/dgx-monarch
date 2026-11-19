"""Once the driver is exiting, nothing new is sent to the Worker actors.

Monarch's own exit step stops every mesh the driver owns once the exit notice
is out (driver_exit). A new call after that reaches a client that is stopping:
on loopback a price then failed with an EOFError underneath, and a later send
made Monarch panic natively (``SendError``). Each row sets the exit flag before
a new send and shows that the send does not happen, next to its twin with no
exit, which sends as before. A call in flight and a stop already under way are
not changed here.

CPU only. The fleet is a real MeshHandle over the transition tests' fake Worker
endpoints (topology_transition_helpers); Monarch is not started.
"""
from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest

from dgx_monarch import (
    capacity_agreement,
    client_lease,
    driver_exit,
    mesh_rpc,
    mesh_setup,
    vae_decode_remote,
)
from dgx_monarch.actor.capacity_quote import RankPrice
from topology_transition_helpers import _acks, _dispatch_sample, _Endpoint, _Future, _handle, _ValueMesh, _Workers

EXITING = pytest.mark.parametrize("exiting", [True, False], ids=["exiting", "not_exiting"])


@pytest.fixture
def flag(monkeypatch):
    """A fresh exit flag for the test, put back after it."""
    event = threading.Event()
    monkeypatch.setattr(driver_exit, "_EXITING", event)
    return event


def _kept(monkeypatch, *names: str) -> list[logging.LogRecord]:
    """Every record the named loggers emit from here on (the pack's loggers do not propagate)."""
    kept: list[logging.LogRecord] = []
    keep = logging.Handler()
    keep.emit = kept.append  # type: ignore[method-assign]
    for name in names:
        logger = logging.getLogger(name)
        monkeypatch.setattr(logger, "handlers", [*logger.handlers, keep])
    return kept


def _loud(kept: list[logging.LogRecord]) -> list[str]:
    return [record.getMessage() for record in kept if record.levelno >= logging.WARNING]


@EXITING
def test_an_all_rank_call_sends_nothing_at_exit(flag, exiting):
    workers = _Workers([])
    workers.status = _Endpoint(_Future(_ValueMesh([{"rank": 0}, {"rank": 1}])))
    handle, _old, _new = _handle(workers)
    if exiting:
        flag.set()
        with pytest.raises(driver_exit.DriverExitingError, match="sent nothing new"):
            handle.call_all("status", timeout_s=1.0)
        assert workers.status.calls == []
        return
    assert handle.call_all("status", timeout_s=1.0) == [{"rank": 0}, {"rank": 1}]
    assert len(workers.status.calls) == 1


class _LockHolder:
    """Another thread holding the handle's lock (a setup, a stop) until released."""

    def __init__(self, handle) -> None:
        self.handle, self.held, self.release = handle, threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._hold, daemon=True)
        self.thread.start()
        assert self.held.wait(5.0)

    def _hold(self) -> None:
        with self.handle.lock:
            self.held.set()
            self.release.wait(10.0)

    def done(self) -> None:
        self.release.set()
        self.thread.join(5.0)


@pytest.mark.parametrize("call", ["all_rank", "setup_bound", "sample"])
@EXITING
def test_a_call_that_waited_for_the_lock_when_the_exit_began_sends_nothing(flag, call, exiting):
    """The call passed its own check, then waited for the handle's lock (or the
    render session that shares it), and the exit began while it waited. The
    check under the lock, at the send itself (mesh_setup.dispatch_endpoint and
    the sample's enqueue), stops it. Not exiting, it sends once the lock frees."""
    workers = _Workers([])
    workers.clear_vram = _Endpoint(_Future(_ValueMesh([{}, {}])))
    workers.compute_sigmas = _Endpoint(_Future(_ValueMesh([{}, {}])))
    handle, _old, _new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)
    sent: list[int] = []
    outcome: list[object] = []

    def sample_send():
        sent.append(1)
        return _Future(_ValueMesh([]))

    def caller() -> None:
        try:
            if call == "all_rank":  # a session-scoped endpoint that needs no setup
                outcome.append(handle.call_all("clear_vram", "soft", timeout_s=2.0))
            elif call == "setup_bound":
                outcome.append(handle.call_all("compute_sigmas", "x", timeout_s=2.0, setup_token=token))
            else:
                outcome.append(_dispatch_sample(handle, sample_send, setup_token=token))
        except BaseException as exc:
            outcome.append(exc)

    holder = _LockHolder(handle)
    waiting = threading.Thread(target=caller, daemon=True)
    waiting.start()
    time.sleep(0.3)
    assert waiting.is_alive() and outcome == []  # past its own check, waiting for the lock
    if exiting:
        flag.set()
    holder.done()
    waiting.join(10.0)
    calls = {"all_rank": workers.clear_vram.calls, "setup_bound": workers.compute_sigmas.calls,
             "sample": sent}[call]
    if exiting:
        assert isinstance(outcome[0], driver_exit.DriverExitingError) and calls == []
        return
    assert not isinstance(outcome[0], BaseException) and len(calls) == 1
    if call == "sample":
        mesh_setup.release_sample(outcome[0])


@pytest.mark.parametrize("call", ["all_rank", "sample", "reset"])
def test_at_exit_the_check_comes_before_the_lock(flag, call):
    """An all-rank call, a sample dispatch and a reset that start at exit refuse
    at once, while another thread holds the handle's lock: none waits for the
    render session or the lifecycle lock (10 s each) before it sees the exit."""
    workers = _Workers([_Future(_acks())])
    workers.clear_vram = _Endpoint(_Future(_ValueMesh([{}, {}])))
    handle, _old, _new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)
    holder = _LockHolder(handle)
    flag.set()
    started = time.monotonic()
    try:
        with pytest.raises(driver_exit.DriverExitingError):
            if call == "all_rank":
                handle.call_all("clear_vram", "soft", timeout_s=2.0)
            elif call == "sample":
                mesh_setup.dispatch_sample(handle, lambda: _Future(_ValueMesh([])), setup_token=token,
                                           authority=object())
            else:
                handle.recycle_detailed()
        assert time.monotonic() - started < 1.0
    finally:
        holder.done()


@EXITING
def test_a_cancel_at_exit_sends_nothing_and_warns_of_nothing(flag, exiting):
    workers = _Workers([])
    sent: list[str] = []
    workers.cancel_sample = SimpleNamespace(broadcast=sent.append)
    handle, _old, _new = _handle(workers)
    warned: list[tuple] = []
    logger = SimpleNamespace(warning=lambda *args: warned.append(args))
    if exiting:
        flag.set()
    assert mesh_rpc.cancel_sample(handle, "r1", 1.0, False, logger) == []
    assert sent == ([] if exiting else ["r1"]) and warned == []


@EXITING
def test_a_sample_dispatch_sends_nothing_at_exit(flag, exiting):
    handle, _old, _new = _handle(_Workers([]))
    token = mesh_setup.current_setup_token(handle)
    sent: list[int] = []

    def send():
        sent.append(1)
        return _Future(_ValueMesh([]))

    if exiting:
        flag.set()
        with pytest.raises(driver_exit.DriverExitingError):
            _dispatch_sample(handle, send, setup_token=token)
        assert sent == []
        return
    submission = _dispatch_sample(handle, send, setup_token=token)
    assert sent == [1]
    mesh_setup.release_sample(submission)


@EXITING
def test_a_remote_decode_unit_is_not_sent_at_exit(flag, monkeypatch, exiting):
    handle, _old, _new = _handle(_Workers([]))
    dispatched: list[str] = []
    monkeypatch.setattr(mesh_setup, "dispatch_endpoint",
                        lambda _handle, endpoint, _send, **_k: dispatched.append(endpoint))
    send = vae_decode_remote.sender(handle, None, {"schema": 1}, [object()])
    if exiting:
        flag.set()
        with pytest.raises(driver_exit.DriverExitingError):
            send(0, 0)
        assert dispatched == []
        return
    send(0, 0)
    assert dispatched == [vae_decode_remote.ENDPOINT]


@EXITING
def test_a_fleet_jobs_artifact_proof_is_not_sent_at_exit(flag, monkeypatch, exiting):
    """A Fleet job proves artifact parity with the one actor that runs it, by a
    direct call rather than an all-rank one; at exit that call is not sent."""
    from dgx_monarch import mesh_safety
    from dgx_monarch.actor import model_store

    handle, _old, _new = _handle(_Workers([]))
    asked: list[object] = []
    handle.workers.slice = lambda **_coords: SimpleNamespace(artifact_identity=SimpleNamespace(
        call_one=lambda specs: asked.append(specs) or _Future({"m": 1})))
    monkeypatch.setattr(model_store, "request_artifact_identity", lambda name, _loras: {name: 1})
    for check in ("assert_request_artifact_binding", "assert_normal_render_residency_mode",
                  "assert_artifact_parity"):
        monkeypatch.setattr(mesh_safety, check, lambda *_a, **_k: None)
    request = {"model": {"unet_name": "m.safetensors"}}
    if exiting:
        flag.set()
        with pytest.raises(driver_exit.DriverExitingError):
            mesh_rpc.verify_request_artifacts(handle, request, 0)
        assert asked == []
        return
    mesh_rpc.verify_request_artifacts(handle, request, 0)
    assert len(asked) == 1


def _quoting(handle) -> list[int]:
    """Every rank's capacity quote answers at once; returns the ranks asked."""
    asked: list[int] = []

    def slice_(**coords):
        rank = coords["gpus"]
        row = {"host": f"box{rank}", "rank": rank, "quotes": [
            RankPrice(verdict="fits", rung="stock_fits", reason="fits").to_row()]}
        return SimpleNamespace(capacity_quote=SimpleNamespace(
            call_one=lambda _request: asked.append(rank) or _Future(row)))

    handle.workers.slice = slice_
    return asked


@EXITING
def test_a_price_sends_no_quote_at_exit(flag, exiting):
    handle, _old, _new = _handle(_Workers([]))
    asked = _quoting(handle)
    if exiting:
        flag.set()
        with pytest.raises(driver_exit.DriverExitingError):
            capacity_agreement.collect(handle, {}, 2, deadline_s=1.0)
        assert asked == []
        return
    answers = capacity_agreement.collect(handle, {}, 2, deadline_s=1.0)
    assert asked == [0, 1] and [bool(answer.silent) for answer in answers] == [False, False]


@EXITING
def test_worker_arguments_are_not_sent_at_exit(flag, exiting):
    workers = _Workers([])
    workers.apply_worker_args = _Endpoint(_Future(_ValueMesh([{}, {}])))
    handle, _old, _new = _handle(workers)
    key = handle.worker_args_key
    if exiting:
        flag.set()
        with pytest.raises(driver_exit.DriverExitingError):
            handle.apply_worker_args({"slab_weights": False})
        # Refused before the policy marker was retired: the handle is as it was.
        assert workers.apply_worker_args.calls == [] and handle.worker_args_key == key
        assert handle.setup_cleanup_state is None
        return
    handle.apply_worker_args({"slab_weights": False})
    assert len(workers.apply_worker_args.calls) == 1


@EXITING
def test_a_setup_that_starts_at_exit_sends_no_teardown_and_no_setup(flag, exiting):
    workers = _Workers([_Future(_acks())])
    handle, _old, new = _handle(workers)
    key, generation = handle.setup_key, handle.setup_generation
    if exiting:
        flag.set()
        with pytest.raises(driver_exit.DriverExitingError):
            handle.ensure_setup(new, "TORCH_FLASH")
        assert workers.teardown_group.calls == []
        assert all(endpoint.envs == [] for endpoint in workers.setup_endpoints)
        assert (handle.setup_key, handle.setup_generation) == (key, generation)
        return
    handle.ensure_setup(new, "TORCH_FLASH")
    assert len(workers.teardown_group.calls) == 1
    assert all(len(endpoint.envs) == 1 for endpoint in workers.setup_endpoints)


class _ExitAs(_Future):
    """A future whose wait is where the driver's exit begins: it sets the flag, then
    answers or fails as the plain future does."""

    def __init__(self, flag: threading.Event, value=None, error: BaseException | None = None):
        super().__init__(value, error)
        self.flag = flag

    def get(self, timeout=None):
        self.flag.set()
        return super().get(timeout)


@EXITING
def test_an_exit_during_the_topology_teardown_sends_no_setup_and_no_rollback(
        flag, monkeypatch, exiting):
    """The exit begins while the old groups are torn down. The setup that would
    follow is not sent, and neither is the rollback a failed setup sends: no
    second teardown, no stop of the procs and no ERROR. Not exiting, the setup
    follows the teardown as before."""
    teardown = _ExitAs(flag, _acks()) if exiting else _Future(_acks())
    workers = _Workers([teardown, _Future(_acks())])
    handle, _old, new = _handle(workers)
    kept = _kept(monkeypatch, "dgx_monarch.mesh_setup_transition", "dgx_monarch.mesh")
    if not exiting:
        handle.ensure_setup(new, "TORCH_FLASH")
        assert len(workers.teardown_group.calls) == 1
        assert all(len(endpoint.envs) == 1 for endpoint in workers.setup_endpoints)
        return
    with pytest.raises(driver_exit.DriverExitingError):
        handle.ensure_setup(new, "TORCH_FLASH")
    assert len(workers.teardown_group.calls) == 1  # the teardown under way, no rollback
    assert all(endpoint.envs == [] for endpoint in workers.setup_endpoints)
    assert handle.procs.reasons == [] and _loud(kept) == [], _loud(kept)


@EXITING
def test_a_setup_the_exit_cuts_sends_no_rollback(flag, monkeypatch, exiting):
    """A rank's setup fails as the exit begins. The rollback teardown is not
    sent, the procs are not stopped and nothing is logged above INFO; the
    setup's own error is raised. Not exiting, the rollback runs as before."""
    cut = RuntimeError("owner terminated: stopped: client stopped")
    first = _ExitAs(flag, error=cut) if exiting else _Future(error=cut)
    workers = _Workers([_Future(_acks()), _Future(_acks())],
                       setup_futures=[first, _Future({"rank": 1})])
    handle, _old, new = _handle(workers)
    kept = _kept(monkeypatch, "dgx_monarch.mesh_setup_transition", "dgx_monarch.mesh")
    with pytest.raises(RuntimeError, match="client stopped"):
        handle.ensure_setup(new, "TORCH_FLASH")
    if not exiting:
        assert len(workers.teardown_group.calls) == 2  # the topology teardown and the rollback
        return
    assert len(workers.teardown_group.calls) == 1  # the topology teardown only
    assert handle.procs.reasons == [] and _loud(kept) == [], _loud(kept)


@EXITING
def test_a_reset_that_starts_at_exit_stops_nothing(flag, exiting):
    workers = _Workers([_Future(_acks())])
    handle, _old, _new = _handle(workers)
    if exiting:
        flag.set()
        with pytest.raises(driver_exit.DriverExitingError):
            handle.recycle_detailed()
        assert workers.teardown_group.calls == [] and handle.procs.reasons == []
        assert handle.setup_key is not None
        return
    assert handle.recycle_detailed().ok
    assert len(workers.teardown_group.calls) == 1 and handle.procs.reasons


@EXITING
def test_the_lease_renewer_sends_nothing_and_ends_at_exit(flag, exiting):
    workers = _Workers([])
    sent: list[float] = []
    workers.renew_client_lease = SimpleNamespace(broadcast=sent.append)
    handle, _old, _new = _handle(workers)
    passes: list[int] = []

    def sleeper(_seconds):
        passes.append(1)
        if len(passes) == 2:
            flag.set()  # the exit hook ran while the renewer slept
        if len(passes) > 4:
            raise AssertionError("the renewer kept running after the exit")

    if exiting:
        flag.set()
    assert client_lease.renew_pass([handle], 30.0, failures={}) == (0 if exiting else 1)
    assert sent == ([] if exiting else [30.0])
    client_lease._run(lambda: [], 30.0, 0.0, sleeper)  # returns: the renewer ends
    assert passes == ([] if exiting else [1, 1])


@EXITING
def test_always_clear_and_retry_resets_nothing_at_exit(flag, monkeypatch, exiting):
    from dgx_monarch import old_model
    from dgx_monarch.nodes import old_model_clear, recycle_drain

    reset: list[object] = []
    monkeypatch.setattr(old_model, "offer_of", lambda _exc: object())
    monkeypatch.setattr(old_model_clear, "always_on", lambda: True)
    monkeypatch.setattr(recycle_drain, "recycle_and_drain", lambda handle: reset.append(handle)
                        or SimpleNamespace(ok=False, status="refused", detail="stub"))
    kept = _kept(monkeypatch, "dgx_monarch.nodes.old_model_clear")
    handle = object()
    if exiting:
        flag.set()
    assert old_model_clear.clear_for_retry(handle, RuntimeError("old model"), inherited=False) is False
    assert reset == ([] if exiting else [handle])
    assert (_loud(kept) == []) is exiting  # not exiting, the stub's refused reset warns as before


@EXITING
def test_no_pack_log_line_reports_a_call_the_exit_did_not_send(flag, monkeypatch, exiting):
    """A handler that catches the exit's refusal and warns (the panel's status
    poll, an unload before a ceremony) prints nothing once the driver is
    exiting: the pack's loggers drop a record whose arguments or exception carry
    it, as itself or as an explicit cause (``raise ... from``). A record about
    any other failure is kept, through the same handlers, and so is one about an
    error raised while the refusal was being handled, linked to it only as its
    context: that is a different failure. Before the exit nothing is dropped,
    not even a record that carries the error (which only driver_exit raises, at
    exit)."""
    from dgx_monarch.nodes import routes

    kept = _kept(monkeypatch, routes.log.name)
    refused = driver_exit.DriverExitingError(driver_exit.SENT_NOTHING)
    if not exiting:
        routes._log_refresh_failure(refused)
        assert len(_loud(kept)) == 1
        return
    flag.set()
    routes._log_refresh_failure(refused)  # "worker status refresh failed: %r"
    try:
        raise RuntimeError("the unload failed") from refused
    except RuntimeError as wrapped:
        routes.log.warning("unload failed (%r)", wrapped)
        routes.log.error("unload failed", exc_info=True)
    routes.log.warning("unload failed (%(cause)r)", {"cause": refused})  # mapping-style arguments
    assert kept == []
    try:
        try:
            raise refused
        except driver_exit.DriverExitingError:
            {}["fleet"]  # a different failure while the refusal is handled
    except KeyError as other:
        assert other.__context__ is refused and other.__cause__ is None
        routes.log.error("cleanup failed: %r", other)
    assert _loud(kept) == ["cleanup failed: KeyError('fleet')"]
    routes._log_refresh_failure(RuntimeError("status endpoint raised"))
    assert _loud(kept)[-1] == "worker status refresh failed: RuntimeError('status endpoint raised')"
