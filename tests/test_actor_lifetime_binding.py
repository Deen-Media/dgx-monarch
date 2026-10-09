"""Actor lifetime binding: what arms the actor-side reaper and what holds it off.

The sibling file ``test_actor_reaper.py`` covers the other half, the CLI sweep
and its procfs identity rules. The client lease renewer, the reap ledger and
the mesh publication boundary each have a file of their own beside this one.

Everything here is CPU only. No process is ever signalled and no thread ever
sleeps: the reaper's tick body is a pure function over an injected clock and
ppid.
"""
from __future__ import annotations

import dis
import inspect
import io
import sys

import pytest

from actor_lifetime_helpers import _Recorder
from dgx_monarch import actor_lifetime, client_lease, mesh_safety


class _ExplodingSink:
    def write(self, _text: str) -> int:
        raise BrokenPipeError("host agent is gone")

    def flush(self) -> None:
        raise BrokenPipeError("host agent is gone")


def _lifetime() -> actor_lifetime.ActorLifetime:
    return actor_lifetime.ActorLifetime(getpid=lambda: 4242)


def test_an_actor_that_was_never_leased_never_reaps_itself_under_a_live_parent():
    # Version skew fails open: a worker newer than its driver gets no renewals,
    # and only a renewal arms the lease trigger, so it never fires.
    state = _lifetime()
    for moment in (0.0, 1.0, 600.0, 86_400.0):
        assert state.observe(now=moment, ppid=100) is None
    assert state.armed is False


def test_a_never_leased_actor_does_reap_itself_when_its_spawner_dies():
    # Unlike the lease trigger, parent loss arms at bind and fires without any
    # lease. Monarch already asks the kernel to SIGKILL on the same event
    # (pdeathsig, on by default); this covers the window before that prctl runs
    # inside the child.
    state = _lifetime()
    assert state.bind(100) == 100
    assert state.armed is False
    assert state.observe(now=0.0, ppid=100) is None
    assert state.observe(now=actor_lifetime.PARENT_LOSS_GRACE_S, ppid=1) is None
    verdict = state.observe(now=2 * actor_lifetime.PARENT_LOSS_GRACE_S, ppid=1)
    assert verdict is not None and verdict.trigger == actor_lifetime.PARENT_LOSS
    assert state.armed is False


def test_the_first_renewal_arms_and_the_lease_expires_once():
    state = _lifetime()
    state.observe(now=0.0, ppid=100)
    state.renew(actor_lifetime.REAP_GRACE_S, now=0.0)
    assert state.armed is True
    assert state.observe(now=actor_lifetime.REAP_GRACE_S - 0.5, ppid=100) is None
    verdict = state.observe(now=actor_lifetime.REAP_GRACE_S, ppid=100)
    assert verdict is not None
    assert verdict.trigger == actor_lifetime.LEASE_EXPIRY
    assert "300s" in verdict.detail

    exits: list[int] = []
    assert state.fire(verdict, sink=io.StringIO(), exit_hook=exits.append) is True
    # Idempotent: one process exits once, and a later tick returns no verdict.
    assert state.fire(verdict, sink=io.StringIO(), exit_hook=exits.append) is False
    assert state.observe(now=1e6, ppid=100) is None
    assert exits == [actor_lifetime.REAP_EXIT_CODE]


def test_renewals_hold_the_reaper_off_indefinitely():
    state = _lifetime()
    now = 0.0
    for _tick in range(1000):
        state.renew(actor_lifetime.REAP_GRACE_S, now=now)
        for _inner in range(int(client_lease.RENEW_INTERVAL_S // actor_lifetime.TICK_S)):
            now += actor_lifetime.TICK_S
            assert state.observe(now=now, ppid=100) is None


def test_a_zero_lease_disarms_both_triggers():
    state = _lifetime()
    state.observe(now=0.0, ppid=100)
    state.renew(actor_lifetime.REAP_GRACE_S, now=0.0)
    state.renew(0.0, now=1.0)
    assert state.disabled is True
    assert state.armed is False
    # Parent loss is suppressed too: the driver said do not reap this process.
    assert state.observe(now=2.0, ppid=1) is None
    assert state.observe(now=1e6, ppid=1) is None


@pytest.mark.parametrize(("lease", "expected"), [
    (1.0, actor_lifetime.MIN_LEASE_S),
    (actor_lifetime.MIN_LEASE_S, actor_lifetime.MIN_LEASE_S),
    (600.0, 600.0),
    (1e9, actor_lifetime.MAX_LEASE_S),
    ("nonsense", None),
    (float("nan"), None),
])
def test_leases_are_clamped_and_junk_disarms(lease, expected):
    state = _lifetime()
    state.renew(lease, now=100.0)
    if expected is None:
        assert state.armed is False
        assert state.disabled is True
    else:
        assert state.deadline == pytest.approx(100.0 + expected)


def test_parent_loss_fires_after_its_own_grace():
    state = _lifetime()
    # The baseline is bound at install, not at the first tick: if the loop dies
    # inside the first interval, a first-tick baseline would record its adopter
    # and parent loss would never fire. The first bind wins.
    assert state.bind(100) == 100
    assert state.bind(999) == 100
    assert state.observe(now=0.0, ppid=100) is None
    assert state.observe(now=10.0, ppid=1) is None
    assert state.observe(
        now=10.0 + actor_lifetime.PARENT_LOSS_GRACE_S - 1.0, ppid=1) is None
    verdict = state.observe(now=10.0 + actor_lifetime.PARENT_LOSS_GRACE_S, ppid=1)
    assert verdict is not None
    assert verdict.trigger == actor_lifetime.PARENT_LOSS


def test_parent_loss_defers_to_a_client_that_is_still_renewing():
    # A live client renewing this actor outranks a dead spawning process:
    # killing a fleet somebody is using is worse than the leak.
    state = _lifetime()
    state.observe(now=0.0, ppid=100)
    now = 0.0
    while now < 4 * actor_lifetime.PARENT_LOSS_GRACE_S:
        state.renew(actor_lifetime.REAP_GRACE_S, now=now)
        now += client_lease.RENEW_INTERVAL_S
        assert state.observe(now=now, ppid=1) is None
    # Renewals stop; now the latched parent loss resolves it.
    verdict = state.observe(now=now + actor_lifetime.REAP_GRACE_S, ppid=1)
    assert verdict is not None
    assert verdict.trigger == actor_lifetime.PARENT_LOSS


def test_fire_writes_exactly_one_marker_line_naming_pid_and_trigger():
    state = _lifetime()
    sink = io.StringIO()
    state.fire(
        actor_lifetime.Verdict(actor_lifetime.LEASE_EXPIRY, "no client lease for 300s"),
        sink=sink,
        exit_hook=lambda _code: None,
    )
    lines = sink.getvalue().splitlines()
    assert len(lines) == 1
    assert actor_lifetime.REAP_MARKER in lines[0]
    assert "pid=4242" in lines[0]
    assert actor_lifetime.LEASE_EXPIRY in lines[0]


def test_a_broken_log_pipe_still_reaches_the_exit():
    # The stream is inherited from a host agent, which may be the process that
    # died. A BrokenPipeError must not prevent the exit.
    state = _lifetime()
    exits: list[int] = []
    state.fire(
        actor_lifetime.Verdict(actor_lifetime.PARENT_LOSS, "gone"),
        sink=_ExplodingSink(),
        exit_hook=exits.append,
    )
    assert exits == [actor_lifetime.REAP_EXIT_CODE]


def test_the_grace_ladder_keeps_the_armor_and_monarch_ahead_of_us():
    # The protective inequality is the grace minus one renewal interval: the
    # last renewal can land a full interval before a teardown token is minted.
    effective = actor_lifetime.REAP_GRACE_S - client_lease.RENEW_INTERVAL_S
    assert effective > mesh_safety.TOKEN_AUTHORITY_S
    assert effective > 135.0  # dead-peer self-clear (docs/VALIDATION.md)
    # The same inequality must hold for the shortest lease a driver can put on
    # the wire, or DGXM_REAP_GRACE_S could make a rank exit while an in-flight
    # teardown still holds authority to resolve its stop.
    floor = actor_lifetime.MIN_LEASE_S - client_lease.RENEW_INTERVAL_S
    assert floor > mesh_safety.TOKEN_AUTHORITY_S
    assert floor > 135.0
    # Monarch's own reapers (keepalive verdict ~75 s, orphan sweep 60 s) act
    # first on the parent-loss path too.
    assert actor_lifetime.PARENT_LOSS_GRACE_S > 135.0
    assert actor_lifetime.PARENT_LOSS_GRACE_S < actor_lifetime.REAP_GRACE_S
    assert actor_lifetime.TICK_S <= 10.0
    assert actor_lifetime.MIN_LEASE_S > actor_lifetime.TICK_S
    assert client_lease.RENEW_INTERVAL_S * 2 < actor_lifetime.REAP_GRACE_S


def test_the_watcher_is_one_daemon_thread_started_once():
    recorder = _Recorder()
    previous = actor_lifetime._WATCHER, actor_lifetime._WATCHER_STARTING
    try:
        actor_lifetime._WATCHER = None
        actor_lifetime._WATCHER_STARTING = False
        assert actor_lifetime.install(thread_factory=recorder, state=_lifetime()) is True
        assert actor_lifetime.install(thread_factory=recorder, state=_lifetime()) is False
    finally:
        actor_lifetime._WATCHER, actor_lifetime._WATCHER_STARTING = previous
    assert len(recorder.created) == 1
    assert recorder.created[0]["daemon"] is True
    assert recorder.created[0]["name"] == "dgxm-actor-lifetime"


@pytest.mark.parametrize(
    "boundary",
    [RuntimeError("watcher liveness failed"),
     KeyboardInterrupt("watcher liveness cancelled")],
    ids=["ordinary", "cancellation"],
)
def test_established_watcher_requires_proven_liveness(boundary):
    class Existing:
        def is_alive(self):
            raise boundary

    previous = actor_lifetime._WATCHER, actor_lifetime._WATCHER_STARTING
    try:
        actor_lifetime._WATCHER = Existing()
        actor_lifetime._WATCHER_STARTING = False
        if isinstance(boundary, Exception):
            with pytest.raises(RuntimeError, match="liveness could not be confirmed"):
                actor_lifetime.install(state=_lifetime())
        else:
            with pytest.raises(type(boundary)) as failure:
                actor_lifetime.install(state=_lifetime())
            assert failure.value is boundary
    finally:
        actor_lifetime._WATCHER, actor_lifetime._WATCHER_STARTING = previous


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_watcher_start_instruction_handoff_never_starts_a_duplicate():
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

    instructions = list(dis.get_instructions(actor_lifetime.install))
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
        instruction.opname == "STORE_GLOBAL"
        and instruction.argval == "_WATCHER"
        for instruction in instructions[:load_index]
    )
    primary = StartAbort("watcher start return instruction interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target.offset:
            monitoring.set_local_events(
                tool_id, actor_lifetime.install.__code__, 0)
            raise primary

    previous = actor_lifetime._WATCHER, actor_lifetime._WATCHER_STARTING
    actor_lifetime._WATCHER = None
    actor_lifetime._WATCHER_STARTING = False
    monitoring.use_tool_id(tool_id, "dgxm-actor-watcher-start-handoff-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, actor_lifetime.install.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(StartAbort) as failure:
        try:
            actor_lifetime.install(thread_factory=factory, state=_lifetime())
        finally:
            monitoring.set_local_events(
                tool_id, actor_lifetime.install.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)
    try:
        assert failure.value is primary
        assert len(created) == 1 and created[0].starts == 1
        assert actor_lifetime._WATCHER is created[0]
        assert actor_lifetime._WATCHER_STARTING is False
        assert actor_lifetime.install(
            thread_factory=factory, state=_lifetime()) is False
        assert len(created) == 1 and created[0].starts == 1
    finally:
        actor_lifetime._WATCHER, actor_lifetime._WATCHER_STARTING = previous


def test_ambiguous_watcher_start_is_retained_and_blocks_replacement():
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

    previous = actor_lifetime._WATCHER, actor_lifetime._WATCHER_STARTING
    try:
        actor_lifetime._WATCHER = None
        actor_lifetime._WATCHER_STARTING = False
        with pytest.raises(StartAbort):
            actor_lifetime.install(thread_factory=factory, state=_lifetime())
        assert actor_lifetime._WATCHER is created[0]
        assert actor_lifetime._WATCHER_STARTING is True
        with pytest.raises(RuntimeError, match="unresolved outcome"):
            actor_lifetime.install(thread_factory=factory, state=_lifetime())
        assert len(created) == 1
    finally:
        actor_lifetime._WATCHER, actor_lifetime._WATCHER_STARTING = previous


def test_confirmed_dead_watcher_start_can_retry():
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

    previous = actor_lifetime._WATCHER, actor_lifetime._WATCHER_STARTING
    try:
        actor_lifetime._WATCHER = None
        actor_lifetime._WATCHER_STARTING = False
        with pytest.raises(RuntimeError, match="not started"):
            actor_lifetime.install(thread_factory=factory, state=_lifetime())
        assert actor_lifetime._WATCHER is None
        assert actor_lifetime._WATCHER_STARTING is False
        assert actor_lifetime.install(
            thread_factory=factory, state=_lifetime()) is True
        assert actor_lifetime._WATCHER is created[1]
        assert created[1].alive is True
    finally:
        actor_lifetime._WATCHER, actor_lifetime._WATCHER_STARTING = previous


def test_the_watch_loop_survives_a_failing_tick_and_fires_once():
    fired: list[actor_lifetime.Verdict] = []
    verdict = actor_lifetime.Verdict(actor_lifetime.LEASE_EXPIRY, "no client lease")
    answers = [OSError("procfs hiccup"), None, verdict]

    class _Stub:
        def observe(self):
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer

        def fire(self, value):
            fired.append(value)

    naps: list[float] = []
    actor_lifetime._watch(_Stub(), actor_lifetime.TICK_S, naps.append)
    assert fired == [verdict]
    assert naps == [actor_lifetime.TICK_S] * 3


def test_renew_client_lease_is_a_plain_lock_free_endpoint(monkeypatch):
    from dgx_monarch.actor.worker import GPUWorker

    prop = GPUWorker.renew_client_lease
    # Plain @endpoint, like status(): @concurrent_endpoint rewrites the method
    # to an explicit response port and a different property type.
    assert type(prop) is type(GPUWorker.status)
    assert getattr(prop, "_explicit_response_port", False) is False
    body = getattr(prop._method, "__wrapped__", prop._method)
    assert inspect.iscoroutinefunction(body)

    seen: list[float] = []
    monkeypatch.setattr(actor_lifetime, "renew", seen.append)
    worker = GPUWorker.__new__(GPUWorker)  # no _gpu_lock, no store, no state

    import asyncio

    assert asyncio.run(body(worker, 300.0)) is None
    assert seen == [300.0]


def test_bootstrap_proc_arms_the_reaper_after_the_source_path_insertion():
    # The one piece of dgx-monarch that already runs inside every spawned proc.
    from dgx_monarch import mesh_runtime

    source = inspect.getsource(mesh_runtime.bootstrap_proc)
    assert source.index("sys.path.insert") < source.index("actor_lifetime")

    armed: list[bool] = []
    original = actor_lifetime.install
    actor_lifetime.install = lambda: bool(armed.append(True))
    try:
        mesh_runtime.bootstrap_proc()
    finally:
        actor_lifetime.install = original
    assert armed == [True]
