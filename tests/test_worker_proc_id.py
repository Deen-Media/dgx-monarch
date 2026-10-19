"""The per-start Worker loop id: how it is made, published, read and dialed.

docs/TROUBLESHOOTING.md #2. The loop derives a new id on every start, writes
it into its actor ledger loop record before it serves, and the
listener-generation probe reads it back for the process holding the
listener. These tests run the real probe snippet against a socket this
process holds and a ledger under a temporary HOME; nothing dials a loop.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import socket
from types import SimpleNamespace

import pytest

from dgx_monarch import worker_proc_id
from dgx_monarch.cli import actor_ledger, listener_generation, proc_identity
from dgx_monarch.cli.probe_certainty import FAIL, OK, UNOBSERVED, WARN

ADDRESS_A = "tcp://10.0.0.1:26600"
ADDRESS_B = "tcp://10.0.0.2:26600"
BOOT = "6f1c2d3e-0000-4000-8000-000000000001"


def _host(name: str, address: str):
    return SimpleNamespace(name=name, address=address)


def _probe(address: str) -> dict[str, str]:
    """Run the snippet in this process and parse the line it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(listener_generation.probe_source(address),  # noqa: S102
                     "<listener-generation-probe>", "exec"), {})
    return listener_generation.parse(buffer.getvalue().strip())


@pytest.fixture
def held(monkeypatch, tmp_path):
    """A LISTEN socket this process owns, and a HOME whose ledger the probe reads."""
    monkeypatch.setenv("HOME", str(tmp_path))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        yield SimpleNamespace(address=f"tcp://127.0.0.1:{sock.getsockname()[1]}",
                              home=tmp_path)


def _write_loop(home, loop, *, boot=None, version=actor_ledger.LEDGER_VERSION):
    path = actor_ledger.ledger_path(str(home))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"version": version,
                   "boot_id": proc_identity.boot_id() if boot is None else boot,
                   "written_at": 1.0, "loop": loop, "actors": []}, handle)


def _own_identity() -> tuple[int, int]:
    stat = proc_identity.read_stat(os.getpid())
    assert stat is not None
    return os.getpid(), stat[1]


# Making the id


def test_each_start_gets_a_new_id_and_one_start_always_the_same_one():
    """A restart is a new pid or a new start time, and a reboot a new boot id:
    each must change the id, or a driver that outlives the restart dials the
    old host agent's identity again and stalls (TROUBLESHOOTING #2)."""
    base = worker_proc_id.derive(BOOT, 4242, 31337)
    assert worker_proc_id.is_value(base)
    assert worker_proc_id.derive(BOOT, 4242, 31337) == base
    assert len({base,
                worker_proc_id.derive(BOOT, 4243, 31337),
                worker_proc_id.derive(BOOT, 4242, 31338),
                worker_proc_id.derive(BOOT.replace("1", "2"), 4242, 31337)}) == 4


def test_a_start_whose_identity_will_not_read_still_gets_a_fresh_id(monkeypatch):
    monkeypatch.setattr(proc_identity, "boot_id", lambda *_a: "")
    first, second = worker_proc_id.current(), worker_proc_id.current()
    assert worker_proc_id.is_value(first) and worker_proc_id.is_value(second)
    assert first != second


def test_this_process_id_is_the_derivation_of_its_own_procfs_identity():
    pid, starttime = _own_identity()
    assert worker_proc_id.current() == worker_proc_id.derive(
        proc_identity.boot_id(), pid, starttime)


def test_the_dial_string_is_the_service_proc_addr_the_pinned_wheel_builds():
    from monarch._rust_bindings.monarch_hyperactor.proc import ProcId

    value = worker_proc_id.derive(BOOT, 4242, 31337)
    dial = worker_proc_id.proc_addr(ADDRESS_A, value)
    proc_id, _, address = dial.partition("@")
    assert address == ADDRESS_A
    parsed = ProcId.from_string(proc_id)
    assert parsed.label == worker_proc_id.LABEL == "service"
    assert parsed.uid.is_instance
    assert parsed.uid.instance_value == int(value, 16)


@pytest.mark.parametrize("value", ["bare", "unpublished", "", "XYZ", "0" * 15])
def test_a_reading_that_is_not_an_id_never_becomes_a_dial(value):
    with pytest.raises(ValueError):
        worker_proc_id.proc_addr(ADDRESS_A, value)


def test_the_prime_names_each_address_the_way_monarch_spells_a_channel():
    from monarch._rust_bindings.monarch_hyperactor.proc import ActorAddr

    assert worker_proc_id.channel_addr(ADDRESS_A) == "tcp:10.0.0.1:26600"
    assert worker_proc_id.channel_addr("tcp://[fd00::1]:26600") == "tcp:[fd00::1]:26600"
    for address in (ADDRESS_A, "tcp://[fd00::1]:26600"):
        ActorAddr(addr=worker_proc_id.channel_addr(address), proc_name="p", actor_name="a")


def test_a_prime_without_a_monarch_context_sends_nothing_and_does_not_wait(monkeypatch):
    from monarch._src.actor import actor_mesh

    def no_context():
        raise RuntimeError("no runtime")

    slept: list[float] = []
    monkeypatch.setattr(actor_mesh, "context", no_context)
    assert worker_proc_id.prime([ADDRESS_A, ADDRESS_B], sleep=slept.append) == [
        ADDRESS_A, ADDRESS_B]
    assert slept == []


# Publishing it


def test_the_loop_publishes_its_id_before_it_serves(tmp_path):
    value = worker_proc_id.current()
    assert actor_ledger.publish_loop(ADDRESS_A, proc_id=value, home=str(tmp_path))
    loop = actor_ledger.read_ledger(home=str(tmp_path))["loop"]
    assert (loop["pid"], loop["proc_id"], loop["address"]) == (os.getpid(), value, ADDRESS_A)
    # The tracker's later ticks keep the id rather than dropping it.
    actor_ledger.tick(ADDRESS_A, home=str(tmp_path), proc_id=value, now=10_000.0)
    assert actor_ledger.read_ledger(home=str(tmp_path))["loop"]["proc_id"] == value


def test_a_ledger_the_loop_cannot_write_reports_false(tmp_path):
    blocked = tmp_path / "home"
    blocked.write_text("a file where the home directory should be")
    assert actor_ledger.publish_loop(
        ADDRESS_A, proc_id=worker_proc_id.current(), home=str(blocked)) is False


def test_the_worker_loop_publishes_then_serves_the_id_it_published():
    from dgx_monarch.cli import worker_loop

    source = inspect.getsource(worker_loop.main)
    order = [source.index(token) for token in (
        "proc_id = worker_proc_id.current()",
        "publish_loop(args.address, proc_id=proc_id)",
        "start_tracker(args.address, proc_id=proc_id)",
        "served = worker_proc_id.proc_addr(args.address, proc_id)",
        "run_worker_loop_forever(address=served,",
    )]
    assert order == sorted(order)


def test_the_loop_record_names_the_served_id_before_the_loop_serves(
        isolated_environ, monkeypatch, tmp_path):
    """Run the loop's start with Monarch and the tracker thread stubbed out.
    At the moment it starts serving, its record must already carry the id it
    serves, so a probe that finds the listener can never read the previous
    loop's record. The tracker cannot cover for a skipped publish here."""
    import sys

    import monarch.actor as monarch_actor

    from dgx_monarch import (
        monarch_capabilities,
        monarch_diagnostics,
        monarch_policy,
        runtime_provenance,
    )
    from dgx_monarch.cli import worker_loop, worker_process_env

    address = "tcp://127.0.0.1:27870"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["worker_loop", "--address", address])
    for module, name in ((runtime_provenance, "enforce_source_only_imports"),
                         (worker_process_env, "prepare_worker_process_environment"),
                         (monarch_policy, "apply_policy"),
                         (monarch_policy, "activate_native_policy"),
                         (monarch_diagnostics, "configure_policy_logged"),
                         (monarch_capabilities, "assert_supported_runtime"),
                         (monarch_actor, "enable_transport")):
        monkeypatch.setattr(module, name, lambda *_a, **_k: None)
    tracked: list[str | None] = []
    monkeypatch.setattr(actor_ledger, "start_tracker",
                        lambda _address, **kwargs: tracked.append(kwargs.get("proc_id")))
    served: list[tuple[str, dict]] = []

    def serve(*, address, ca):
        served.append((address, actor_ledger.read_ledger(home=str(tmp_path))["loop"]))

    monkeypatch.setattr(monarch_actor, "run_worker_loop_forever", serve)
    worker_loop.main()

    (dialed, record), = served
    assert worker_proc_id.is_value(record.get("proc_id"))
    assert record["pid"] == os.getpid() and record["address"] == address
    assert dialed == worker_proc_id.proc_addr(address, record["proc_id"])
    assert tracked == [record["proc_id"]]


# Reading it back with the real probe


def test_the_probe_reads_the_id_the_loop_published(held):
    value = worker_proc_id.current()
    assert actor_ledger.publish_loop(held.address, proc_id=value, home=str(held.home))
    fields = _probe(held.address)
    assert fields["proc_id"] == value
    # The same digest input as the listener generation, so the id starts with it.
    assert value.startswith(fields["gen"])


def test_a_new_driver_refuses_an_old_loop_before_dialing(held):
    """An older Worker release writes a loop record without an id and serves
    the bare address. The probe reads `bare`, and the attach refuses with the
    remedy before anything dials, so it never waits out the config push."""
    actor_ledger.tick(held.address, home=str(held.home))
    fields = _probe(held.address)
    assert fields["proc_id"] == worker_proc_id.BARE
    with pytest.raises(worker_proc_id.WorkerProcIdError) as refused:
        worker_proc_id.dial_addresses([_host("h1", held.address)], {held.address: fields})
    text = str(refused.value)
    assert "older release" in text and "`dgxm update`" in text
    assert worker_proc_id.ENTRY in text and "no attach ran" in text


def test_a_record_left_by_the_previous_loop_is_not_this_loops_id(held):
    pid, starttime = _own_identity()
    _write_loop(held.home, {"pid": pid, "starttime": starttime + 1,
                            "address": held.address, "proc_id": "0" * 16})
    assert _probe(held.address)["proc_id"] == worker_proc_id.UNPUBLISHED


def test_a_record_from_another_boot_is_not_this_loops_id(held):
    pid, starttime = _own_identity()
    _write_loop(held.home, {"pid": pid, "starttime": starttime,
                            "address": held.address, "proc_id": "0" * 16},
                boot=BOOT)
    assert _probe(held.address)["proc_id"] == worker_proc_id.UNPUBLISHED


def test_no_ledger_at_all_reads_unpublished(held):
    assert _probe(held.address)["proc_id"] == worker_proc_id.UNPUBLISHED


def test_a_record_that_carries_something_else_reads_malformed(held):
    pid, starttime = _own_identity()
    _write_loop(held.home, {"pid": pid, "starttime": starttime,
                            "address": held.address, "proc_id": "service<x>"})
    assert _probe(held.address)["proc_id"] == worker_proc_id.MALFORMED


def test_a_ledger_that_will_not_read_reads_unreadable(held):
    os.makedirs(actor_ledger.ledger_path(str(held.home)))
    assert _probe(held.address)["proc_id"] == worker_proc_id.UNREADABLE


def test_nothing_listening_reads_none():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    fields = _probe(f"tcp://127.0.0.1:{free}")
    assert (fields["gen"], fields["proc_id"]) == ("none", worker_proc_id.NONE)


# The dial gate


def test_every_host_dials_the_id_its_own_reading_carries():
    a, b = "a" * 16, "b" * 16
    dial = worker_proc_id.dial_addresses(
        [_host("h1", ADDRESS_A), _host("h2", ADDRESS_B)],
        {ADDRESS_A: {"gen": a[:12], "proc_id": a},
         ADDRESS_B: {"gen": b[:12], "proc_id": b}})
    assert dial == [worker_proc_id.proc_addr(ADDRESS_A, a),
                    worker_proc_id.proc_addr(ADDRESS_B, b)]
    assert all(entry.startswith("service<") for entry in dial)


@pytest.mark.parametrize(("fields", "says"), [
    ({"gen": "none", "proc_id": "none"}, "`dgxm up`"),
    ({"gen": "abc", "proc_id": "bare"}, "`dgxm update`"),
    ({"gen": "abc", "proc_id": "unpublished"}, "render again"),
    ({"gen": "abc", "proc_id": "malformed"}, "render again"),
    ({"gen": "abc", "proc_id": "unreadable"}, "actor ledger file would not read (unreadable)"),
    ({"gen": "unknown", "error": "TimeoutExpired"}, "did not read it (TimeoutExpired)"),
    ({}, "did not read it (no reading)"),
])
def test_a_host_without_a_usable_id_refuses_naming_host_reason_and_remedy(fields, says):
    good = {"gen": "a" * 12, "proc_id": "a" * 16}
    with pytest.raises(worker_proc_id.WorkerProcIdError) as refused:
        worker_proc_id.dial_addresses(
            [_host("h1", ADDRESS_A), _host("h2", ADDRESS_B)],
            {ADDRESS_A: good, ADDRESS_B: fields})
    text = str(refused.value)
    assert f"h2 ({ADDRESS_B})" in text and "h1 (" not in text
    assert says in text
    assert "no attach ran" in text and worker_proc_id.ENTRY in text


def test_read_dial_probes_every_configured_loop_through_the_runner():
    calls = []

    def runner(_config, host, script, timeout):
        calls.append((host.address, timeout))
        line = f"{listener_generation.LINE_PREFIX}aaaaaaaaaaaa proc_id={'a' * 16}"
        return SimpleNamespace(returncode=0, stdout=line + "\n")

    config = SimpleNamespace(hosts=(_host("h1", ADDRESS_A), _host("h2", ADDRESS_B)),
                             python_bin="python3")
    dial = worker_proc_id.read_dial(config, runner=runner, timeout=4.0)
    assert [address for address, _ in calls] == [ADDRESS_A, ADDRESS_B]
    assert dial == [worker_proc_id.proc_addr(address, "a" * 16)
                    for address in (ADDRESS_A, ADDRESS_B)]


# The doctor row


def _row(**fields):
    reading = {"gen": "5c4ab03463a9", "loop_pid": "4242", "age_s": "612",
               "listening": "true", "unit_restarts": "0", "marker": "match"}
    reading.update(fields)
    return listener_generation.doctor_row("h1", reading)


def test_a_loop_serving_an_id_reads_ok_and_shows_it():
    status, _name, detail, reason, critical = _row(proc_id="5c4ab03463a91f07")
    assert (status, reason, critical) == (OK, "", False)
    assert detail.endswith("proc_id=5c4ab03463a91f07")


@pytest.mark.parametrize("value", ["bare", "unpublished", "malformed"])
def test_a_listening_loop_that_publishes_no_id_fails_and_names_the_entry(value):
    status, _name, detail, reason, _critical = _row(proc_id=value)
    assert (status, reason) == (FAIL, "")
    assert "the driver refuses to attach" in detail
    assert worker_proc_id.ENTRY in detail


@pytest.mark.parametrize("fields", [{"proc_id": "unreadable"}, {"proc_id": "unknown"}, {}])
def test_an_id_the_probe_could_not_read_stays_an_unobserved_warning(fields):
    status, _name, detail, reason, _critical = _row(**fields)
    assert (status, reason) == (WARN, UNOBSERVED)
    assert worker_proc_id.ENTRY in detail
