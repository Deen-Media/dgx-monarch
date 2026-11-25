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
# The torchmonarch a driver imported, for the gate tests that build readings by hand.
DRIVER = "0.7.0"


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
        "monarch_version = assert_supported_runtime()",
        "proc_id = worker_proc_id.current()",
        "publish_loop(args.address, proc_id=proc_id, monarch=monarch_version)",
        "start_tracker(args.address, proc_id=proc_id, monarch=monarch_version)",
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
                         (monarch_actor, "enable_transport")):
        monkeypatch.setattr(module, name, lambda *_a, **_k: None)
    monkeypatch.setattr(monarch_capabilities, "assert_supported_runtime", lambda: "9.8.7")
    tracked: list[tuple[str | None, str | None]] = []
    monkeypatch.setattr(actor_ledger, "start_tracker", lambda _address, **kwargs: tracked.append(
        (kwargs.get("proc_id"), kwargs.get("monarch"))))
    served: list[tuple[str, dict]] = []

    def serve(*, address, ca):
        served.append((address, actor_ledger.read_ledger(home=str(tmp_path))["loop"]))

    monkeypatch.setattr(monarch_actor, "run_worker_loop_forever", serve)
    worker_loop.main()

    (dialed, record), = served
    assert worker_proc_id.is_value(record.get("proc_id"))
    assert record["pid"] == os.getpid() and record["address"] == address
    # The record names the version the admission returned, and so do the
    # tracker's later ticks.
    assert record["monarch"] == "9.8.7"
    assert dialed == worker_proc_id.proc_addr(address, record["proc_id"])
    assert tracked == [(record["proc_id"], "9.8.7")]


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
        worker_proc_id.dial_addresses([_host("h1", held.address)], {held.address: fields},
                                      monarch=DRIVER)
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
    assert (fields["gen"], fields["proc_id"], fields["monarch"]) == (
        "none", worker_proc_id.NONE, worker_proc_id.NONE)


# The torchmonarch version the loop imported, read back for the process holding the listener


def test_the_probe_reads_the_torchmonarch_version_the_loop_published(held):
    value = worker_proc_id.current()
    assert actor_ledger.publish_loop(held.address, proc_id=value, monarch="9.8.7", home=str(held.home))
    fields = _probe(held.address)
    assert (fields["proc_id"], fields["monarch"]) == (value, "9.8.7")
    # The tracker's later ticks keep the version rather than dropping it.
    actor_ledger.tick(held.address, home=str(held.home), proc_id=value, monarch="9.8.7", now=10_000.0)
    assert _probe(held.address)["monarch"] == "9.8.7"


@pytest.mark.parametrize(("monarch", "reading"), [
    (None, "absent"), ("not a version", "malformed"), ("absent", "malformed"), (7, "malformed")])
def test_a_record_without_a_usable_version_reads_absent_or_malformed(held, monarch, reading):
    pid, starttime = _own_identity()
    loop = {"pid": pid, "starttime": starttime, "address": held.address, "proc_id": "a" * 16}
    if monarch is not None:
        loop["monarch"] = monarch
    _write_loop(held.home, loop)
    assert _probe(held.address)["monarch"] == reading


def test_the_worker_loop_records_the_torchmonarch_it_imported(isolated_environ, monkeypatch, tmp_path):
    """The real admission, against whichever torchmonarch this interpreter
    holds: the record must carry that distribution's version, read after the
    import it checks, so the driver compares what the loop actually runs."""
    import importlib.metadata
    import sys

    import monarch.actor as monarch_actor

    import dgx_monarch
    from dgx_monarch import monarch_diagnostics, monarch_policy, runtime_provenance
    from dgx_monarch.cli import worker_loop, worker_process_env

    installed = importlib.metadata.version("torchmonarch")
    monkeypatch.setattr(dgx_monarch, "TORCHMONARCH_PIN", installed)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["worker_loop", "--address", "tcp://127.0.0.1:27871"])
    for module, name in ((runtime_provenance, "enforce_source_only_imports"),
                         (worker_process_env, "prepare_worker_process_environment"),
                         (monarch_policy, "apply_policy"),
                         (monarch_policy, "activate_native_policy"),
                         (monarch_diagnostics, "configure_policy_logged"),
                         (monarch_actor, "enable_transport")):
        monkeypatch.setattr(module, name, lambda *_a, **_k: None)
    monkeypatch.setattr(actor_ledger, "start_tracker", lambda *_a, **_k: None)
    served: list[dict] = []
    monkeypatch.setattr(monarch_actor, "run_worker_loop_forever", lambda **_kwargs: served.append(
        actor_ledger.read_ledger(home=str(tmp_path))["loop"]))
    worker_loop.main()
    assert served[0]["monarch"] == installed


def _gate(held, monkeypatch):
    """mesh_attach.read_dial, with the real probe script run by bash in place of ssh."""
    import subprocess
    import sys

    from dgx_monarch import mesh_attach
    from dgx_monarch.cli import lifecycle

    def run_on_host(_config, _host, script, timeout):
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                              timeout=max(timeout, 30), env=dict(os.environ))

    from dgx_monarch.monarch_capabilities import loaded_version

    monkeypatch.setattr(lifecycle, "run_on_host", run_on_host)
    config = SimpleNamespace(hosts=(_host("h1", held.address),), python_bin=sys.executable)
    trace = SimpleNamespace(generation=lambda *_a, **_k: "", emit=lambda *_a, **_k: "")
    return lambda: mesh_attach.read_dial(config, trace, attempt=1, monarch=loaded_version())


@pytest.fixture
def driver_monarch():
    """The torchmonarch this test process imported, as a driver has before any attach."""
    import importlib.metadata

    import monarch.actor  # noqa: F401

    return importlib.metadata.version("torchmonarch")


def _record(held, monarch):
    """This process's loop record, as a loop of either release would write it."""
    pid, starttime = _own_identity()
    loop = {"pid": pid, "starttime": starttime, "address": held.address, "proc_id": "a" * 16}
    if monarch is not None:
        loop["monarch"] = monarch
    _write_loop(held.home, loop)


def test_the_attach_dials_a_loop_on_the_driver_s_own_torchmonarch(held, monkeypatch, driver_monarch):
    _record(held, driver_monarch)
    assert _gate(held, monkeypatch)() == ["a" * 16]


def test_the_attach_keeps_each_dialed_loop_s_generation_in_the_scan_s_own_form(
        held, monkeypatch, driver_monarch):
    """A spawn the fleet then queues is named by the loop that holds it (failed_bring_up).

    The real probe runs through bash against a listener this process holds. The
    generation it reads must be the one the liveness scan computes for the same
    process (actor_reaper.liveness), or every loop would read as replaced.
    """
    import sys

    from dgx_monarch import mesh_attach
    from dgx_monarch.monarch_capabilities import loaded_version

    _record(held, driver_monarch)
    _gate(held, monkeypatch)  # installs the bash runner
    config = SimpleNamespace(hosts=(_host("h1", held.address),), python_bin=sys.executable)
    trace = SimpleNamespace(generation=lambda *_a, **_k: "", emit=lambda *_a, **_k: "")
    handoff = SimpleNamespace(loop_gens={})

    assert mesh_attach.read_dial(config, trace, attempt=1, monarch=loaded_version(),
                                 handoff=handoff) == ["a" * 16]

    pid, starttime = _own_identity()
    assert handoff.loop_gens == {
        held.address: worker_proc_id.derive(proc_identity.boot_id(), pid, starttime)[:12]}


@pytest.mark.parametrize("recorded", ["0.0.1", None])
def test_the_attach_refuses_a_loop_on_another_torchmonarch_before_dialing(
        held, monkeypatch, driver_monarch, recorded):
    """The real probe reads the record and the real gate compares it with the
    torchmonarch this process imported. A loop on another version, or one from
    a release that records none, refuses here with the cause and the remedy,
    instead of two config-push timeouts and a restart of healthy loops."""
    _record(held, recorded)
    with pytest.raises(worker_proc_id.WorkerProcIdError) as refused:
        _gate(held, monkeypatch)()
    text = str(refused.value)
    assert text.startswith("no attach ran: h1 (") and "stays usable" in text
    if recorded is None:
        assert "earlier release, which does not report its torchmonarch version" in text
    else:
        assert f"runs torchmonarch 0.0.1 and this driver runs {driver_monarch}" in text
    # Either way this driver is the newer side, which a verified update cannot repair.
    assert "this driver runs the newer release, so bring the Workers to it" in text
    assert f"torchmonarch=={driver_monarch}`) and run `dgxm restart`" in text
    assert f"forward repair in {worker_proc_id.UPDATE_ENTRY}" in text
    assert "`dgxm update --verify` refuses while they differ" in text
    assert worker_proc_id.VERSION_ENTRY in text
    assert refused.value.unknown.startswith("h1 (")


# The dial gate


def test_every_host_dials_the_id_its_own_reading_carries():
    a, b = "a" * 16, "b" * 16
    dial = worker_proc_id.dial_addresses(
        [_host("h1", ADDRESS_A), _host("h2", ADDRESS_B)],
        {ADDRESS_A: {"gen": a[:12], "proc_id": a, "monarch": DRIVER},
         ADDRESS_B: {"gen": b[:12], "proc_id": b, "monarch": DRIVER}}, monarch=DRIVER)
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
    good = {"gen": "a" * 12, "proc_id": "a" * 16, "monarch": DRIVER}
    with pytest.raises(worker_proc_id.WorkerProcIdError) as refused:
        worker_proc_id.dial_addresses(
            [_host("h1", ADDRESS_A), _host("h2", ADDRESS_B)],
            {ADDRESS_A: good, ADDRESS_B: fields}, monarch=DRIVER)
    text = str(refused.value)
    assert f"h2 ({ADDRESS_B})" in text and "h1 (" not in text
    assert says in text
    assert "no attach ran" in text and worker_proc_id.ENTRY in text


def test_read_dial_probes_every_configured_loop_through_the_runner():
    calls = []

    def runner(_config, host, script, timeout):
        calls.append((host.address, timeout))
        line = f"{listener_generation.LINE_PREFIX}aaaaaaaaaaaa proc_id={'a' * 16} monarch={DRIVER}"
        return SimpleNamespace(returncode=0, stdout=line + "\n")

    config = SimpleNamespace(hosts=(_host("h1", ADDRESS_A), _host("h2", ADDRESS_B)),
                             python_bin="python3")
    dial = worker_proc_id.read_dial(config, monarch=DRIVER, runner=runner, timeout=4.0)
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
    from dgx_monarch import TORCHMONARCH_PIN

    status, _name, detail, reason, critical = _row(proc_id="5c4ab03463a91f07", monarch=TORCHMONARCH_PIN)
    assert (status, reason, critical) == (OK, "", False)
    assert detail.endswith(f"proc_id=5c4ab03463a91f07 running torchmonarch={TORCHMONARCH_PIN}")


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


@pytest.mark.parametrize(("monarch", "says"), [
    ("0.0.1", "runs torchmonarch 0.0.1 and this driver runs"),
    ("absent", "earlier release"),
    ("malformed", "could not read the torchmonarch version"),
])
def test_a_loop_running_another_torchmonarch_fails_with_its_running_version(monarch, says):
    """The env row reads the version on disk; this row reads the one the
    running loop imported, which is what the attach gate refuses on."""
    status, _name, detail, reason, _critical = _row(proc_id="5c4ab03463a91f07", monarch=monarch)
    assert (status, reason) == (FAIL, "")
    assert f"running torchmonarch={monarch}" in detail and says in detail
    assert "the driver refuses to attach" in detail and worker_proc_id.VERSION_ENTRY in detail


@pytest.mark.parametrize("fields", [{"monarch": "unknown"}, {}])
def test_a_running_version_the_probe_could_not_read_stays_an_unobserved_warning(fields):
    status, _name, detail, reason, _critical = _row(proc_id="5c4ab03463a91f07", **fields)
    assert (status, reason) == (WARN, UNOBSERVED)
    assert worker_proc_id.VERSION_ENTRY in detail


# What each version refusal asks for. A driver that moved without its Workers
# (a hand pull, a rollback) is the state a loop without a version, or on an
# older version, comes from; there a verified update refuses with
# prior_worker_drift, so the remedy must not be one.

_TO_WORKERS = "this driver runs the newer release, so bring the Workers to it"
_TO_DRIVER = "the Workers run the newer release, so put the driver's checkout and Python on their release"


@pytest.mark.parametrize("reading", ["absent", "0.0.1", "0.7.0"])
def test_a_loop_older_than_this_driver_is_brought_to_it_without_a_verified_update(reading):
    _reason, remedy = worker_proc_id.version_problem(reading, "0.8.0")
    assert remedy.startswith(_TO_WORKERS)
    assert ("on plain source directories install torchmonarch 0.8.0 into the Python cluster.toml "
            "names (`<python> -m pip install --no-deps torchmonarch==0.8.0`) and run `dgxm restart`, "
            "which syncs this driver's source to them") in remedy
    assert f"in release slots use the forward repair in {worker_proc_id.UPDATE_ENTRY}" in remedy
    assert "`dgxm update --verify` refuses while they differ" in remedy
    assert "finish the update" not in remedy and "restart ComfyUI" not in remedy


def test_a_loop_newer_than_this_driver_asks_for_the_driver_and_a_comfyui_restart():
    _reason, remedy = worker_proc_id.version_problem("0.8.0", "0.7.0")
    assert remedy == (f"{_TO_DRIVER} and restart ComfyUI ({worker_proc_id.UPDATE_ENTRY}); "
                      "then render again")


@pytest.mark.parametrize(("reading", "driver"), [("0.8.0+local", "0.7.0"), ("0.8.0rc1", "0.8.0"),
                                                 ("0.8", "0.8.0.0")])
def test_versions_that_do_not_order_name_both_sides(reading, driver):
    _reason, remedy = worker_proc_id.version_problem(reading, driver)
    assert remedy.startswith("the driver and every Worker must run the same release: if this "
                             "driver is the newer side, bring the Workers to it")
    assert f"torchmonarch=={driver}`) and run `dgxm restart`" in remedy
    assert "if the Workers are, put the driver's checkout and Python on their release and " \
           "restart ComfyUI" in remedy
