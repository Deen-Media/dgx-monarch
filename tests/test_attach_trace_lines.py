"""Check cluster attach traces and their parser contract.

The format documented in docs/TROUBLESHOOTING.md #101 requires one prefix,
one sequence per attempt, phase timings, listener generations before each
attach, a distinct released-fleet failure stage, and one close line on every
exit. These tests use injected runners and clocks; no SSH or Monarch runs.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from dgx_monarch import attach_trace, mesh_attach, mesh_teardown
from dgx_monarch.cli import listener_generation, worker_health

ADDRESS_A = "tcp://10.0.0.1:26600"
ADDRESS_B = "tcp://10.0.0.2:26600"


def _config(*addresses: str, auto_heal: bool = False):
    from dgx_monarch.config import ClusterConfig, HostConfig

    return ClusterConfig(
        hosts=tuple(HostConfig(name=f"h{index}", address=address)
                    for index, address in enumerate(addresses, 1)),
        client_bind="tcp://10.0.0.1:0", auto_heal=auto_heal,
        transport_security="trusted_fabric")


class _Log:
    """Collects every line, so a test reads exactly what an operator would."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def _record(self, template, *args):
        self.lines.append(template % args if args else template)

    info = warning = error = _record

    def traced(self, event: str) -> list[dict[str, str]]:
        parsed = [attach_trace.parse(line) or {} for line in self.lines]
        return [fields for fields in parsed if fields.get("event") == event]


class _Factory:
    """The mesh_factory seam, reduced to the three calls the attach makes."""

    def __init__(self) -> None:
        self.transport_failures = 0

    def safe_transport_failure(self, _handoff):
        self.transport_failures += 1

    def safe_failure_evidence(self, exc):
        return repr(exc)

    def safe_failure_message(self, exc):
        return repr(exc)


@pytest.fixture
def rig(monkeypatch):
    """A runtime dict standing in for mesh.py's globals, with no ssh in it."""
    import monarch.actor as monarch_actor

    monkeypatch.setattr(monarch_actor, "enable_transport", lambda *_a, **_k: None)
    monkeypatch.setattr(mesh_teardown, "released_predecessor_recently",
                        lambda **_k: False)
    # The last-seen generations outlive one attach on purpose, so a test that
    # inherited another test's readings would compare against the wrong attach.
    monkeypatch.setattr(attach_trace, "_LAST_SEEN", {})
    log = _Log()
    attempts: list[list[str]] = []
    generations = {"gen": "aaaaaaaaaaaa"}

    def fleet(config, **_kwargs):
        return {host.address: {"gen": generations["gen"],
                               "loop_pid": "4242", "age_s": "31",
                               "listening": "true", "unit_restarts": "0",
                               "marker": "match"}
                for host in config.hosts}

    monkeypatch.setattr(listener_generation, "fleet", fleet)

    def build(results):
        def attach_once(addresses):
            attempts.append(list(addresses))
            outcome = results.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return {
            "mesh_factory": _Factory(),
            "MeshAttachError": RuntimeError,
            "log": log,
            "_TRANSPORT_ENABLE_LOCK": threading.Lock(),
            "_TRANSPORT_POISON": None,
            "_TRANSPORT_BIND": None,
            "_attach_once": attach_once,
            "_heal_dead_loops": lambda *_a, **_k: None,
            "_raise_attach": _raise_attach,
            "_raise_released_attach": _raise_released_attach,
        }

    return SimpleNamespace(build=build, log=log, attempts=attempts,
                           generations=generations)


def _raise_attach(addresses, exc):
    raise RuntimeError(f"attach to workers {addresses} failed") from exc


def _raise_released_attach(addresses, exc, *, auto_heal, restarted=None):
    raise RuntimeError(
        f"attach to workers {addresses} failed on a released fleet "
        f"(auto_heal={auto_heal})") from exc


def test_a_line_parses_back_into_the_fields_it_was_written_from():
    text = attach_trace.line("phase", seq=7, phase="attach", wall_s=1.5)
    assert text.startswith("attach_trace event=phase")
    assert attach_trace.parse(text) == {
        "event": "phase", "seq": "7", "phase": "attach", "wall_s": "1.50"}


def test_a_line_survives_the_log_prefix_a_real_driver_puts_in_front_of_it():
    text = attach_trace.line("close", outcome="attached")
    decorated = f"[dgx-monarch box] INFO {text}"
    assert attach_trace.parse(decorated) == {"event": "close",
                                             "outcome": "attached"}


def test_prose_that_is_not_a_trace_line_parses_as_nothing():
    assert attach_trace.parse("attached 2 host(s) in 42.10s") is None
    assert attach_trace.parse("") is None


def test_every_value_becomes_one_greppable_token():
    text = attach_trace.line(
        "open", empty=[], addresses=[ADDRESS_A, ADDRESS_B], missing=None,
        flag=False, note="two words, one comma")
    fields = attach_trace.parse(text) or {}
    assert fields["empty"] == "none"
    assert fields["addresses"] == f"{ADDRESS_A},{ADDRESS_B}"
    assert fields["missing"] == "unknown"
    assert fields["flag"] == "false"
    assert fields["note"] == "two_words;_one_comma"
    assert len(text.split()) == len(fields) + 1  # The prefix comes first, then one token per field.


def test_each_attach_gets_its_own_sequence_number():
    first = attach_trace.AttachTrace(_Log(), [ADDRESS_A])
    second = attach_trace.AttachTrace(_Log(), [ADDRESS_A])
    assert second.seq == first.seq + 1


def test_a_phase_writes_its_wall_whether_the_block_returned_or_raised():
    log = _Log()
    trace = attach_trace.AttachTrace(log, [ADDRESS_A])
    with trace.phase("transport_enable"):
        pass
    with pytest.raises(TimeoutError), trace.phase("attach", attempt=1):
        raise TimeoutError()
    outcomes = [fields["outcome"] for fields in log.traced("phase")]
    assert outcomes == ["ok", "TimeoutError"]
    assert log.traced("phase")[1]["attempt"] == "1"
    assert all("wall_s" in fields for fields in log.traced("phase"))


def test_a_phase_field_the_block_only_learns_at_the_end_still_lands():
    log = _Log()
    trace = attach_trace.AttachTrace(log, [ADDRESS_A])
    with trace.phase("attach") as extra:
        extra["polls"] = 3
    assert log.traced("phase")[0]["polls"] == "3"


def test_the_handshake_halves_are_timed_apart():
    log = _Log()
    with attach_trace.half(log, "attach_to_workers", hosts=2):
        pass
    with pytest.raises(TimeoutError), attach_trace.half(
            log, "initialized_get", bound_s=70):
        raise TimeoutError()
    halves = log.traced("attach_call")
    assert [fields["half"] for fields in halves] == ["attach_to_workers",
                                                     "initialized_get"]
    assert [fields["outcome"] for fields in halves] == ["ok", "TimeoutError"]
    assert halves[1]["bound_s"] == "70"


def test_a_generation_is_compared_against_the_last_attach_not_this_one(
        monkeypatch):
    """The hand-restart and `dgxm restart` cases (docs/TROUBLESHOOTING.md #101)
    fail on the first attempt, so only a comparison against the previous
    attach can show a loop that moved."""
    monkeypatch.setattr(attach_trace, "_LAST_SEEN", {})
    log = _Log()
    attach_trace.AttachTrace(log, [ADDRESS_A]).generation(
        ADDRESS_A, {"gen": "aaaaaaaaaaaa"}, phase="before_attach_1")
    attach_trace.AttachTrace(log, [ADDRESS_A]).generation(
        ADDRESS_A, {"gen": "bbbbbbbbbbbb"}, phase="before_attach_1")
    first, second = log.traced("loop_generation")
    assert first["changed"] == "unknown"
    assert (second["changed"], second["previous"]) == ("true", "aaaaaaaaaaaa")


def test_a_reading_that_did_not_come_back_is_not_a_loop_that_moved(monkeypatch):
    """The probe runs on each host (over ssh when the host is remote) within
    mesh_attach.GENERATION_PROBE_S before every attach attempt, and one that
    times out or fails answers `unknown`. Reading `unknown` as a moved listener
    would write the hand-restart signal, `changed=true`, twice: once going to
    `unknown` and once coming back."""
    monkeypatch.setattr(attach_trace, "_LAST_SEEN", {})
    log = _Log()
    for gen in ("aaaaaaaaaaaa", "unknown", "aaaaaaaaaaaa"):
        attach_trace.AttachTrace(log, [ADDRESS_A]).generation(
            ADDRESS_A, {"gen": gen}, phase="before_attach_1")
    readings = log.traced("loop_generation")
    assert [fields["changed"] for fields in readings] == ["unknown", "unknown",
                                                          "false"]
    assert readings[1]["previous"] == "aaaaaaaaaaaa"


def test_a_loop_that_did_move_across_an_unreadable_attempt_still_says_so(
        monkeypatch):
    monkeypatch.setattr(attach_trace, "_LAST_SEEN", {})
    log = _Log()
    for gen in ("aaaaaaaaaaaa", "unknown", "bbbbbbbbbbbb"):
        attach_trace.AttachTrace(log, [ADDRESS_A]).generation(
            ADDRESS_A, {"gen": gen}, phase="before_attach_1")
    assert log.traced("loop_generation")[-1]["changed"] == "true"


def test_a_loop_whose_socket_nobody_holds_is_a_reading(monkeypatch):
    """`none` is an observation: something did hold that socket and now
    nothing does. Only `unknown` means the probe never answered."""
    monkeypatch.setattr(attach_trace, "_LAST_SEEN", {})
    log = _Log()
    for gen in ("aaaaaaaaaaaa", "none"):
        attach_trace.AttachTrace(log, [ADDRESS_A]).generation(
            ADDRESS_A, {"gen": gen}, phase="before_attach_1")
    assert log.traced("loop_generation")[-1]["changed"] == "true"


def test_a_note_written_outside_an_attach_still_says_which_attach_it_was():
    log = _Log()
    trace = attach_trace.AttachTrace(log, [ADDRESS_A])
    with attach_trace.active(trace):
        attach_trace.note(log, "heal_restart", dead=[ADDRESS_A])
    attach_trace.note(log, "heal_restart", dead=[ADDRESS_B])
    inside, outside = log.traced("heal_restart")
    assert inside["seq"] == str(trace.seq)
    assert outside["seq"] == "0"


def _handshake_rig(monkeypatch, get):
    """The real attach_once over a stub monarch, writing into a caught log."""
    import monarch.actor as monarch_actor

    from dgx_monarch import log as log_mod
    from dgx_monarch import mesh_runtime

    caught = _Log()
    monkeypatch.setattr(log_mod, "get_logger", lambda _name=None: caught)
    monkeypatch.setattr(
        monarch_actor, "attach_to_workers",
        lambda **_kwargs: SimpleNamespace(
            initialized=SimpleNamespace(get=get)))
    return mesh_runtime, caught


def test_the_real_attach_splits_the_handshake_it_makes(monkeypatch):
    """The wall an operator reads first, pinned on the call that spends it."""
    mesh_runtime, caught = _handshake_rig(monkeypatch, lambda timeout: None)
    mesh_runtime.attach_once([ADDRESS_A])
    halves = caught.traced("attach_call")
    assert [fields["half"] for fields in halves] == ["attach_to_workers",
                                                     "initialized_get"]
    assert halves[0]["hosts"] == "1"
    assert halves[1]["bound_s"] == str(mesh_runtime.ATTACH_INIT_WAIT_S)
    assert [fields["outcome"] for fields in halves] == ["ok", "ok"]


def test_a_mesh_that_never_comes_up_marks_the_second_half_only(monkeypatch):
    def never_ready(timeout):
        raise TimeoutError()

    mesh_runtime, caught = _handshake_rig(monkeypatch, never_ready)
    with pytest.raises(TimeoutError):
        mesh_runtime.attach_once([ADDRESS_A])
    halves = caught.traced("attach_call")
    assert [fields["outcome"] for fields in halves] == ["ok", "TimeoutError"]


def test_a_clean_attach_writes_a_header_a_generation_and_a_close(rig):
    runtime = rig.build(["HOSTS"])
    assert mesh_attach.attach_cluster(runtime, _config(ADDRESS_A, ADDRESS_B),
                                      None) == "HOSTS"
    header = rig.log.traced("open")[0]
    assert header["hosts"] == "2"
    assert header["addresses"] == f"{ADDRESS_A},{ADDRESS_B}"
    assert header["transport_bound"] == "false"
    assert header["released_predecessor"] == "false"
    assert header["attach_config_timeout"]
    readings = rig.log.traced("loop_generation")
    assert [fields["address"] for fields in readings] == [ADDRESS_A, ADDRESS_B]
    assert all(fields["phase"] == "before_attach_1" for fields in readings)
    assert all(fields["gen"] == "aaaaaaaaaaaa" for fields in readings)
    assert rig.log.traced("close")[0]["outcome"] == "attached"


def test_a_generation_that_moved_between_attempts_says_so(rig, monkeypatch):
    """A loop that moved between the two attempts reads `changed=true`, the
    disagreement docs/TROUBLESHOOTING.md #101 suspects after a hand restart."""
    runtime = rig.build([RuntimeError("blip"), "HOSTS"])
    readings = iter(["aaaaaaaaaaaa", "bbbbbbbbbbbb"])
    monkeypatch.setattr(
        listener_generation, "fleet",
        lambda config, **_kwargs: {host.address: {"gen": next(readings)}
                                   for host in config.hosts})
    assert mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None) == "HOSTS"
    readings = rig.log.traced("loop_generation")
    assert [fields["phase"] for fields in readings] == ["before_attach_1",
                                                        "before_attach_2"]
    assert [fields["changed"] for fields in readings] == ["unknown", "true"]


def test_an_unreadable_generation_costs_the_attach_a_line_and_nothing_else(
        rig, monkeypatch):
    def refuse(_config, **_kwargs):
        raise OSError("no route")

    monkeypatch.setattr(listener_generation, "fleet", refuse)
    runtime = rig.build(["HOSTS"])
    assert mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None) == "HOSTS"
    assert rig.log.traced("loop_generation")[0]["error"] == "OSError"


def test_the_two_attempts_of_a_failing_attach_are_numbered(rig):
    runtime = rig.build([RuntimeError("blip"), RuntimeError("blip")])
    with pytest.raises(RuntimeError):
        mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None)
    phases = [fields for fields in rig.log.traced("phase")
              if fields.get("phase") == "attach"]
    assert [fields["attempt"] for fields in phases] == ["1", "2"]
    assert [fields["outcome"] for fields in phases] == ["RuntimeError"] * 2
    assert rig.log.traced("close")[0]["stage"] == "attach_2_failed"


def test_a_released_fleet_fails_fast_and_closes_on_its_own_stage(rig, monkeypatch):
    """Once this driver's fleet changes under it, this driver process cannot
    attach to the replacement fleet again (hardware, 2026-10-01, torchmonarch
    0.6.0; the mesh_attach._attach_and_retry docstring lists the four
    reproductions). A fleet whose replacement latch this driver released has
    changed, so it spends exactly one attempt and closes on a stage of its
    own, not the `attach_2_failed` stage a non-released failure uses."""
    monkeypatch.setattr(mesh_teardown, "released_predecessor_recently",
                        lambda **_k: True)
    runtime = rig.build([RuntimeError("worker gone")])
    with pytest.raises(RuntimeError, match="auto_heal=False"):
        mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None)
    assert len(rig.attempts) == 1
    close = rig.log.traced("close")[0]
    assert (close["outcome"], close["stage"]) == ("poisoned", "attach_1_released")
    assert rig.log.traced("post_fail_restart") == []


def test_a_released_fleets_fail_fast_restarts_the_loops_when_auto_heal_is_on(
        rig, monkeypatch):
    """Whether the loops restart together is the operator's `cluster.toml`
    setting, so the raise carries it rather than guessing, and the restart
    still writes its line after the close."""
    from dgx_monarch.cli import lifecycle

    monkeypatch.setattr(mesh_teardown, "released_predecessor_recently",
                        lambda **_k: True)
    monkeypatch.setattr(lifecycle, "restart", lambda *_a, **_k: True)
    runtime = rig.build([RuntimeError("worker gone")])
    with pytest.raises(RuntimeError, match="auto_heal=True"):
        mesh_attach.attach_cluster(
            runtime, _config(ADDRESS_A, auto_heal=True), None)
    assert len(rig.attempts) == 1
    close = rig.log.traced("close")[0]
    assert (close["outcome"], close["stage"]) == ("poisoned", "attach_1_released")
    restart = rig.log.traced("post_fail_restart")[0]
    assert restart["seq"] == close["seq"]


def test_the_heal_names_the_loops_it_saw_down_and_the_loops_it_restarted(
        monkeypatch):
    """The probe names some loops; the restart stops and starts every one."""
    from dgx_monarch.cli import lifecycle

    config = _config(ADDRESS_A, ADDRESS_B, auto_heal=True)
    log = _Log()
    monkeypatch.setattr(
        worker_health, "passive_unhealthy_workers",
        lambda *_a, **_k: worker_health.FleetHealth((ADDRESS_B,), ()))
    monkeypatch.setattr(lifecycle, "restart", lambda *_a, **_k: False)
    mesh_attach.heal_dead_loops(config, [ADDRESS_A, ADDRESS_B], log)
    probe = log.traced("heal_probe")[0]
    assert (probe["dead"], probe["unobserved"]) == (ADDRESS_B, "none")
    restart = log.traced("heal_restart")[0]
    assert restart["dead"] == ADDRESS_B
    assert restart["restarted"] == f"{ADDRESS_A},{ADDRESS_B}"
    assert log.traced("heal_settled")[0]["settled"] == "false"


def test_a_transport_enable_that_poisons_the_session_still_writes_a_close(rig,
                                                                         monkeypatch):
    """The close line names the stage that poisoned the session; the transport
    enable poisons before any attach runs, so it must write its own close."""
    import monarch.actor as monarch_actor

    def refuse(_bind):
        raise RuntimeError("bind refused")

    monkeypatch.setattr(monarch_actor, "enable_transport", refuse)
    runtime = rig.build(["HOSTS"])
    with pytest.raises(RuntimeError, match="restart ComfyUI"):
        mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None)
    close = rig.log.traced("close")[0]
    assert (close["outcome"], close["stage"]) == ("poisoned", "transport_enable")
    assert close["seq"] == rig.log.traced("open")[0]["seq"]
    assert runtime["_TRANSPORT_POISON"].startswith("cluster transport enable failed")


def _one_close(log: _Log) -> dict[str, str]:
    """The attempt's single terminal line, checked to be the only one."""
    closes = log.traced("close")
    assert len(closes) == 1
    assert closes[0]["seq"] == log.traced("open")[0]["seq"]
    return closes[0]


def test_a_session_poisoned_by_an_earlier_attempt_closes_this_one(rig):
    """The refusal is instant and writes no phase, so without a close the
    header is the whole sequence and the runbook has no terminal stage."""
    runtime = rig.build(["HOSTS"])
    runtime["_TRANSPORT_POISON"] = "an earlier attempt"
    with pytest.raises(RuntimeError, match="poisoned by an earlier attempt"):
        mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None)
    close = _one_close(rig.log)
    assert (close["outcome"], close["stage"]) == ("refused", "transport_poisoned")
    assert close["reason"] == "earlier_attempt"
    assert rig.attempts == []


def test_a_bind_this_session_cannot_switch_closes_the_sequence(rig):
    runtime = rig.build(["HOSTS"])
    runtime["_TRANSPORT_BIND"] = "tcp://10.0.0.9:0"
    with pytest.raises(RuntimeError, match="requires a restart"):
        mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None)
    close = _one_close(rig.log)
    assert (close["outcome"], close["stage"]) == ("refused", "transport_bind")
    assert close["reason"] == "client_bind_mismatch"
    assert rig.log.traced("open")[0]["transport_bound"] == "true"


def test_a_pre_attach_heal_that_failed_closes_the_sequence(rig):
    """No attach runs after this, so the heal lines would otherwise be the
    tail of the attempt with nothing saying it ended there."""
    def refuse(*_args, **_kwargs):
        raise OSError("no route")

    runtime = rig.build(["HOSTS"])
    runtime["_heal_dead_loops"] = refuse
    with pytest.raises(RuntimeError, match="auto-heal failed"):
        mesh_attach.attach_cluster(
            runtime, _config(ADDRESS_A, auto_heal=True), None)
    close = _one_close(rig.log)
    assert (close["outcome"], close["stage"]) == ("failed", "auto_heal")
    assert "OSError" in close["evidence"]
    assert rig.attempts == []


def test_an_escape_nobody_foresaw_still_closes_the_sequence(rig, monkeypatch):
    """Close the trace even when an unexpected exception escapes."""
    def surprise(*_args, **_kwargs):
        raise KeyError("_attach_once")

    monkeypatch.setattr(mesh_attach, "_attach_and_retry", surprise)
    runtime = rig.build(["HOSTS"])
    with pytest.raises(KeyError):
        mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None)
    close = _one_close(rig.log)
    assert (close["outcome"], close["stage"]) == ("failed", "unexpected")
    assert close["evidence"] == "KeyError"


def test_an_attach_that_poisoned_the_session_closes_once_and_names_its_stage(
        rig):
    """The guard must not write a second close over a stage that is known."""
    runtime = rig.build([RuntimeError("blip"), RuntimeError("blip")])
    with pytest.raises(RuntimeError):
        mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None)
    close = _one_close(rig.log)
    assert (close["outcome"], close["stage"]) == ("poisoned", "attach_2_failed")


def test_the_post_failure_restart_writes_the_one_line_behind_the_close(
        rig, monkeypatch):
    """The runbook sends an operator to `event=close`, not to the tail of the
    sequence, because this line carries the same seq and lands after it: the
    restart serves the next session, not the attempt that just ended."""
    from dgx_monarch.cli import lifecycle

    monkeypatch.setattr(lifecycle, "restart", lambda *_a, **_k: True)
    runtime = rig.build([RuntimeError("blip"), RuntimeError("blip")])
    with pytest.raises(RuntimeError):
        mesh_attach.attach_cluster(
            runtime, _config(ADDRESS_A, auto_heal=True), None)
    close = _one_close(rig.log)
    assert (close["outcome"], close["stage"]) == ("poisoned", "attach_2_failed")
    restart = rig.log.traced("post_fail_restart")[0]
    assert restart["seq"] == close["seq"]
    events = [fields["event"] for fields in
              (attach_trace.parse(text) or {} for text in rig.log.lines)
              if fields.get("event")]
    assert events.index("post_fail_restart") > events.index("close")
