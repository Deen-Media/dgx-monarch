"""The client-side lease renewer: which handles a pass renews, which it skips,
and how its thread comes up exactly once.

The actor side of the same lease lives in ``test_actor_lifetime_binding.py``.

CPU only: the renewer runs against fake handles and no cast leaves the process.
"""
from __future__ import annotations

import dis
import sys
from types import SimpleNamespace

import pytest

from actor_lifetime_helpers import _Recorder
from dgx_monarch import actor_lifetime, client_lease, mesh_safety
from dgx_monarch.cli import proc_identity


class _FakeWorkers:
    def __init__(self, error: BaseException | None = None) -> None:
        self.casts: list[float] = []
        self.error = error
        self.renew_client_lease = SimpleNamespace(broadcast=self._broadcast)

    def _broadcast(self, lease_s: float) -> None:
        if self.error is not None:
            raise self.error
        self.casts.append(lease_s)


def _handle(**fields):
    base = {
        "workers": _FakeWorkers(),
        "defunct": False,
        "teardown_complete": False,
        "replacement_blocked": None,
        "sample_leases": {},
        "abandoned_sample_leases": {},
    }
    base.update(fields)
    return SimpleNamespace(**base)


def test_a_live_handle_is_renewed_with_the_configured_lease():
    handle = _handle()
    assert client_lease.renew_pass([handle], 300.0, failures={}) == 1
    assert handle.workers.casts == [300.0]


def test_a_deliberate_teardown_token_stands_the_renewer_down():
    handle = _handle()
    mesh_safety.begin_deliberate_teardown(id(handle))
    try:
        assert client_lease.skip_reason(handle) == "deliberate teardown in flight"
        assert client_lease.renew_pass([handle], 300.0, failures={}) == 0
    finally:
        mesh_safety.end_deliberate_teardown(id(handle), stop_confirmed=False)
    assert handle.workers.casts == []
    assert client_lease.skip_reason(handle) is None


def test_a_completed_teardown_stands_the_renewer_down_forever():
    handle = _handle(teardown_complete=True)
    assert client_lease.skip_reason(handle) == "teardown complete"
    assert client_lease.renew_pass([handle], 300.0, failures={}) == 0


def test_a_defunct_handle_with_outstanding_leases_is_still_renewed():
    # mesh_helpers defers reconciliation of a failed fleet until sample/read
    # leases retire; those procs are alive and finishing a render. Standing
    # down here would turn a single-rank fault into a whole-fleet kill.
    handle = _handle(defunct=True, sample_leases={7: 1})
    assert client_lease.skip_reason(handle) is None
    assert client_lease.renew_pass([handle], 300.0, failures={}) == 1
    abandoned = _handle(defunct=True, abandoned_sample_leases={9: 2})
    assert client_lease.renew_pass([abandoned], 300.0, failures={}) == 1


def test_a_defunct_handle_without_leases_stands_down():
    # Bounds the renewer as a fault source: nothing keeps casting at procs
    # somebody already stopped.
    handle = _handle(defunct=True)
    assert client_lease.skip_reason(handle) == "unresolved with no outstanding leases"
    assert client_lease.renew_pass([handle], 300.0, failures={}) == 0


def test_a_blocked_handle_is_renewed_only_while_it_holds_leases():
    # An interrupted stop latches `replacement_blocked` and leaves the handle
    # registered, so renewing it would keep a stranded fleet alive until
    # ComfyUI restarts. While leases are outstanding those ranks may still be
    # finishing a render (a thread-launch failure can latch the same field on
    # healthy procs), so that window keeps its renewals.
    working = _handle(replacement_blocked="could not launch reconcile thread",
                      sample_leases={7: 1})
    assert client_lease.skip_reason(working) is None
    assert client_lease.renew_pass([working], 300.0, failures={}) == 1

    stranded = _handle(
        replacement_blocked="detach interrupted during the proc stop; process state unknown")
    assert client_lease.skip_reason(stranded) == "blocked with no outstanding leases"
    assert client_lease.renew_pass([stranded], 300.0, failures={}) == 0


def test_a_blocked_teardown_latch_cannot_keep_stranded_actors_renewed():
    from dgx_monarch import mesh_setup

    latch = mesh_setup.cleanup_in_progress(3, "recycle group teardown", 60)
    stranded = _handle(
        replacement_blocked="proc stop failed; process state unknown",
        setup_cleanup_state=latch,
    )
    assert client_lease.dispatch_in_flight(stranded) is True
    assert client_lease.skip_reason(stranded) == "blocked with no outstanding leases"
    assert client_lease.renew_pass([stranded], 300.0, failures={}) == 0

    working = _handle(
        replacement_blocked="reconcile launch failed",
        setup_cleanup_state=latch,
        sample_leases={7: 1},
    )
    assert client_lease.skip_reason(working) is None
    assert client_lease.renew_pass([working], 300.0, failures={}) == 1


_DEAD_STOP = "worker ProcMesh stop timed out; its outcome is unknown"


def _dead_latch(**fields):
    from dgx_monarch import mesh_teardown

    return _handle(replacement_blocked=_DEAD_STOP,
                   _block_cause=mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH, **fields)


def test_a_dead_fleet_latch_is_not_renewed_for_abandoned_leases():
    # An abandoned sample lease must not keep renewing against a dead fleet
    # after stop timed out. `client_lease.dead_fleet_latch` defines release.
    dead = _dead_latch(abandoned_sample_leases={9: 1})
    assert client_lease.skip_reason(dead) == client_lease.DEAD_FLEET_SKIP
    for _pass in range(3):
        assert client_lease.renew_pass([dead], 300.0, failures={}) == 0
    assert dead.workers.casts == []

    # A live sample lease still means a rank may be finishing a render.
    working = _dead_latch(sample_leases={7: 1}, abandoned_sample_leases={9: 1})
    assert client_lease.skip_reason(working) is None
    assert client_lease.renew_pass([working], 300.0, failures={}) == 1

    # With no lease at all the general rule stands down, with its own reason.
    bare = _dead_latch()
    assert client_lease.skip_reason(bare) == "blocked with no outstanding leases"


def test_only_the_after_death_timeout_gives_up_abandoned_leases():
    # A latch with no stop issued can sit on healthy processes that a reset can
    # still stop, and the other causes carry no death report. Standing down on
    # any of them would let a fleet the driver may yet stop cleanly reap itself.
    from dgx_monarch import mesh_teardown

    causes = [value for name, value in vars(mesh_teardown).items()
              if name.startswith("TAG_") and value != mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH]
    assert mesh_teardown.TAG_NO_STOP_ISSUED in causes and len(causes) >= 6
    for cause in causes:
        handle = _handle(replacement_blocked="latched", _block_cause=cause,
                         abandoned_sample_leases={9: 1})
        assert client_lease.skip_reason(handle) is None, cause
        assert client_lease.renew_pass([handle], 300.0, failures={}) == 1, cause

    # A latch with no recorded cause keeps renewing.
    uncaused = _handle(replacement_blocked="latched", abandoned_sample_leases={9: 1})
    assert client_lease.skip_reason(uncaused) is None

    # Not yet blocked: reconciliation is still deferred behind the leases.
    unresolved = _handle(defunct=True, abandoned_sample_leases={9: 1},
                         _block_cause=mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH)
    assert client_lease.skip_reason(unresolved) is None


@pytest.mark.parametrize(("lease_s", "said"), [
    (300.0, "exits 300s after the last renewal"),
    (0.0, "actor reaping is disabled"),
])
def test_the_dead_fleet_stand_down_is_said_once_per_handle(monkeypatch, lease_s, said):
    warned: list[str] = []
    monkeypatch.setattr(client_lease, "log", SimpleNamespace(
        warning=lambda message, *args: warned.append(message % args),
        debug=lambda *args: None))
    monkeypatch.setattr(client_lease, "_STOOD_DOWN", set())
    dead = _dead_latch(abandoned_sample_leases={9: 1})
    for _pass in range(4):
        client_lease.renew_pass([dead], lease_s, failures={})
    assert len(warned) == 1
    assert "renewals stopped for a fleet latched after a worker death" in warned[0]
    assert said in warned[0]
    # The note does not outlive the handle, so a later fleet is announced too.
    client_lease.renew_pass([], lease_s, failures={})
    assert client_lease._STOOD_DOWN == set()


# Six of the endpoints mesh_rpc latches as ambiguous mutations, with test
# budgets above the reap grace; the real budgets sit at each call_all site.
# None of these holds a sample lease, so a renewer that stood down on the latch
# alone would kill the fleet in the middle of the operation.
_DISPATCH_BUDGETS = [
    ("load_model", 900.0), ("load_uncond_model", 900.0), ("unload", 600.0),
    ("clear_vram", 600.0), ("gate_swap_cycle", 1800.0),
    ("gate_fsdp_reload_cycle", 1800.0),
]


@pytest.mark.parametrize(("endpoint", "budget_s"), _DISPATCH_BUDGETS)
def test_a_mutation_rpc_in_flight_is_proof_of_life_not_a_stranded_handle(endpoint, budget_s):
    """The renewer must not stand down for the length of a load.

    `mesh_rpc.call_all` publishes an IN_PROGRESS cleanup latch before it sends
    a session-scoped mutation, and `mesh_safety` reads a latched handle as
    `unresolved`. A load holds no sample lease, so a lease-only rule stands the
    renewer down for the whole dispatch and every rank self-exits inside it.
    """
    from dgx_monarch import mesh_setup

    handle = _handle(setup_cleanup_state=mesh_setup.cleanup_in_progress(
        3, f"{endpoint} RPC completion", budget_s))
    assert budget_s > actor_lifetime.REAP_GRACE_S, "the ladder no longer needs this rule"
    assert mesh_safety.coherent_lifecycle_verdict(handle, RuntimeError) == "unresolved"
    assert client_lease.outstanding_leases(handle) == 0
    assert client_lease.dispatch_in_flight(handle) is True
    assert client_lease.skip_reason(handle) is None
    assert client_lease.renew_pass([handle], 300.0, failures={}) == 1


def test_a_decided_dirty_latch_is_not_proof_of_life():
    """Once the outcome is decided nobody is waiting, so the rule stops."""
    from dgx_monarch import mesh_setup

    handle = _handle(setup_cleanup_state=mesh_setup.cleanup_failure(
        3, "load_model RPC completion", 900.0, TimeoutError("no answer")))
    assert client_lease.dispatch_in_flight(handle) is False
    assert client_lease.skip_reason(handle) == "unresolved with no outstanding leases"
    assert client_lease.renew_pass([handle], 300.0, failures={}) == 0


@pytest.mark.parametrize("error", [RuntimeError("cast failed"), KeyboardInterrupt()])
def test_cast_failures_are_swallowed_and_backed_off_per_handle(error):
    broken = _handle(workers=_FakeWorkers(error=error))
    healthy = _handle()
    failures: dict[int, int] = {}
    for _attempt in range(client_lease.MAX_CONSECUTIVE_FAILURES + 2):
        assert client_lease.renew_pass([broken, healthy], 300.0, failures=failures) == 1
    assert len(healthy.workers.casts) == client_lease.MAX_CONSECUTIVE_FAILURES + 2
    # Backed off, not dropped: three failures stand it down, pass four skips it
    # and pass five retries it. Skips and failures both advance the counter, so
    # it reads five either way; the next test proves the retry.
    assert failures[id(broken)] == client_lease.MAX_CONSECUTIVE_FAILURES + 2


def test_a_stood_down_handle_is_renewed_again_once_its_casts_recover():
    # Retiring a handle for good would leave every actor it already armed to
    # exit at the grace, and a cast that raises never arrived, so standing down
    # for good saves nothing. A fleet whose fault clears must be renewed again.
    workers = _FakeWorkers(error=RuntimeError("supervision fault"))
    handle = _handle(workers=workers)
    failures: dict[int, int] = {}
    for _attempt in range(client_lease.MAX_CONSECUTIVE_FAILURES):
        client_lease.renew_pass([handle], 300.0, failures=failures)
    assert workers.casts == []
    workers.error = None
    renewed = 0
    for _pass in range(client_lease.RETIRE_RETRY_PASSES + 1):
        renewed += client_lease.renew_pass([handle], 300.0, failures=failures)
    assert renewed >= 1
    assert failures == {}          # a success clears the counter
    # Within one lease grace, so the actors are never left to time out.
    assert (client_lease.RETIRE_RETRY_PASSES * client_lease.RENEW_INTERVAL_S
            < client_lease.DEFAULT_LEASE_S)


def test_failure_counters_do_not_outlive_their_handles():
    handle = _handle(workers=_FakeWorkers(error=RuntimeError("gone")))
    failures: dict[int, int] = {}
    client_lease.renew_pass([handle], 300.0, failures=failures)
    assert failures
    client_lease.renew_pass([], 300.0, failures=failures)
    assert failures == {}


@pytest.mark.parametrize(("environ", "expected"), [
    ({}, client_lease.DEFAULT_LEASE_S),
    ({"DGXM_REAP_GRACE_S": "900"}, 900.0),
    # Below the floor is refused, not clamped: the knob may lengthen the grace,
    # never put a rank's exit inside the teardown token authority window.
    ({"DGXM_REAP_GRACE_S": "60"}, client_lease.DEFAULT_LEASE_S),
    ({"DGXM_REAP_GRACE_S": ""}, client_lease.DEFAULT_LEASE_S),
    ({"DGXM_REAP_GRACE_S": "0"}, client_lease.DEFAULT_LEASE_S),
    ({"DGXM_REAP_GRACE_S": "-1"}, client_lease.DEFAULT_LEASE_S),
    ({"DGXM_REAP_GRACE_S": "1e9"}, client_lease.DEFAULT_LEASE_S),
    ({"DGXM_REAP_GRACE_S": "abc"}, client_lease.DEFAULT_LEASE_S),
    ({"DGXM_REAP_GRACE_S": "nan"}, client_lease.DEFAULT_LEASE_S),
    ({"DGXM_DISABLE_ACTOR_REAPER": "1"}, 0.0),
    ({"DGXM_DISABLE_ACTOR_REAPER": "1", "DGXM_REAP_GRACE_S": "60"}, 0.0),
    ({"DGXM_DISABLE_ACTOR_REAPER": "0"}, client_lease.DEFAULT_LEASE_S),
])
def test_the_lease_env_knob_is_bounded(environ, expected):
    assert client_lease.lease_seconds(environ) == pytest.approx(expected)


def test_a_refused_lease_value_says_so_out_loud(monkeypatch):
    """Without the warning, an operator who sets 60 waits the full 300 s and
    cannot tell why: the value is replaced, not clamped, so the effective grace
    equals the default."""
    warned: list[tuple] = []
    monkeypatch.setattr(client_lease, "log", SimpleNamespace(
        warning=lambda *args: warned.append(args), debug=lambda *args: None))
    assert client_lease.lease_seconds({"DGXM_REAP_GRACE_S": "60"}) == 300.0
    assert len(warned) == 1
    assert "was not applied" in warned[0][0]
    assert warned[0][1] == client_lease.GRACE_ENV and warned[0][2] == "60"
    assert client_lease.DEFAULT_LEASE_S in warned[0][1:]
    assert client_lease.lease_seconds({"DGXM_REAP_GRACE_S": "900"}) == 900.0
    assert len(warned) == 1


def test_the_driver_marker_is_not_planted_when_it_cannot_be_proved(monkeypatch):
    """An unproved driver identity must plant no marker, or a live driver could
    read as dead and its actors become sweep targets.
    `proc_identity.driver_identity` says why it answers empty, not with a zero
    start time.
    """
    monkeypatch.setattr(proc_identity, "driver_identity", lambda: "")
    environ: dict[str, str] = {}
    assert client_lease.plant_driver_marker(environ) == ""
    assert client_lease.DRIVER_ENV not in environ
    monkeypatch.setattr(proc_identity, "driver_identity", lambda: "4242:99")
    assert client_lease.plant_driver_marker(environ) == "4242:99"
    assert environ[client_lease.DRIVER_ENV] == "4242:99"


def test_the_renewer_is_one_daemon_thread_started_once():
    recorder = _Recorder()
    previous = client_lease._THREAD, client_lease._THREAD_STARTING
    try:
        client_lease._THREAD = None
        client_lease._THREAD_STARTING = False
        assert client_lease.start(lambda: [], thread_factory=recorder) is True
        assert client_lease.start(lambda: [], thread_factory=recorder) is False
    finally:
        client_lease._THREAD, client_lease._THREAD_STARTING = previous
    assert len(recorder.created) == 1
    assert recorder.created[0]["daemon"] is True
    assert recorder.created[0]["name"] == "dgxm-client-lease"


@pytest.mark.parametrize(
    "boundary",
    [RuntimeError("renewer liveness failed"),
     KeyboardInterrupt("renewer liveness cancelled")],
    ids=["ordinary", "cancellation"],
)
def test_established_renewer_requires_proven_liveness(boundary):
    class Existing:
        def is_alive(self):
            raise boundary

    previous = client_lease._THREAD, client_lease._THREAD_STARTING
    try:
        client_lease._THREAD = Existing()
        client_lease._THREAD_STARTING = False
        if isinstance(boundary, Exception):
            with pytest.raises(RuntimeError, match="liveness could not be confirmed"):
                client_lease.start(lambda: [])
        else:
            with pytest.raises(type(boundary)) as failure:
                client_lease.start(lambda: [])
            assert failure.value is boundary
    finally:
        client_lease._THREAD, client_lease._THREAD_STARTING = previous


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_renewer_start_instruction_handoff_never_starts_a_duplicate():
    class StartAbort(BaseException):
        pass

    class Candidate:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.starts = 0
            self.alive = False

        def start(self):
            self.starts += 1
            self.alive = True

        def is_alive(self):
            return self.alive

    created: list[Candidate] = []

    def factory(**kwargs):
        candidate = Candidate(**kwargs)
        created.append(candidate)
        return candidate

    instructions = list(dis.get_instructions(client_lease.start))
    load_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == "LOAD_ATTR" and instruction.argval == "start"
    )
    call_index = next(
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    )
    target = instructions[call_index + 1]
    assert target.opname == "POP_TOP"
    assert any(
        instruction.opname == "STORE_GLOBAL" and instruction.argval == "_THREAD"
        for instruction in instructions[:load_index]
    )
    primary = StartAbort("renewer start return instruction interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target.offset:
            monitoring.set_local_events(tool_id, client_lease.start.__code__, 0)
            raise primary

    previous = client_lease._THREAD, client_lease._THREAD_STARTING
    client_lease._THREAD = None
    client_lease._THREAD_STARTING = False
    monitoring.use_tool_id(tool_id, "dgxm-client-lease-start-handoff-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, client_lease.start.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(StartAbort) as failure:
        try:
            client_lease.start(lambda: [], thread_factory=factory)
        finally:
            monitoring.set_local_events(tool_id, client_lease.start.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)
    try:
        assert failure.value is primary
        assert len(created) == 1 and created[0].starts == 1
        assert client_lease._THREAD is created[0]
        assert client_lease._THREAD_STARTING is False
        assert client_lease.start(lambda: [], thread_factory=factory) is False
        assert len(created) == 1 and created[0].starts == 1
    finally:
        client_lease._THREAD, client_lease._THREAD_STARTING = previous


def test_ambiguous_renewer_start_is_retained_and_blocks_replacement():
    class StartAbort(BaseException):
        pass

    class Candidate:
        def start(self):
            raise StartAbort("native start outcome unavailable")

        def is_alive(self):
            raise RuntimeError("liveness unavailable")

    created: list[Candidate] = []

    def factory(**_kwargs):
        candidate = Candidate()
        created.append(candidate)
        return candidate

    previous = client_lease._THREAD, client_lease._THREAD_STARTING
    try:
        client_lease._THREAD = None
        client_lease._THREAD_STARTING = False
        with pytest.raises(StartAbort):
            client_lease.start(lambda: [], thread_factory=factory)
        assert client_lease._THREAD is created[0]
        assert client_lease._THREAD_STARTING is True
        with pytest.raises(RuntimeError, match="unresolved outcome"):
            client_lease.start(lambda: [], thread_factory=factory)
        assert len(created) == 1
    finally:
        client_lease._THREAD, client_lease._THREAD_STARTING = previous


def test_confirmed_dead_renewer_start_can_retry():
    class Candidate:
        def __init__(self, fails: bool):
            self.fails = fails
            self.alive = False

        def start(self):
            if self.fails:
                raise RuntimeError("thread was not started")
            self.alive = True

        def is_alive(self):
            return self.alive

    created: list[Candidate] = []

    def factory(**_kwargs):
        candidate = Candidate(not created)
        created.append(candidate)
        return candidate

    previous = client_lease._THREAD, client_lease._THREAD_STARTING
    try:
        client_lease._THREAD = None
        client_lease._THREAD_STARTING = False
        with pytest.raises(RuntimeError, match="not started"):
            client_lease.start(lambda: [], thread_factory=factory)
        assert client_lease._THREAD is None
        assert client_lease._THREAD_STARTING is False
        assert client_lease.start(lambda: [], thread_factory=factory) is True
        assert client_lease._THREAD is created[1] and created[1].alive is True
    finally:
        client_lease._THREAD, client_lease._THREAD_STARTING = previous


def test_the_registry_snapshot_casts_outside_the_mesh_lock():
    from dgx_monarch import mesh

    acquired: list[bool] = []

    class _LockProbe(_FakeWorkers):
        def _broadcast(self, lease_s: float) -> None:
            taken = mesh._MESH_LOCK.acquire(blocking=False)
            acquired.append(taken)
            if taken:
                mesh._MESH_LOCK.release()

    handle = _handle(workers=_LockProbe())
    key = ("test-client-lease", "local")
    with mesh._MESH_LOCK:
        mesh._MESHES[key] = handle
    try:
        snapshot = client_lease.registry_snapshot()
        assert handle in snapshot
        assert client_lease.renew_pass(snapshot, 300.0, failures={}) == 1
    finally:
        with mesh._MESH_LOCK:
            mesh._MESHES.pop(key, None)
    assert acquired == [True]
