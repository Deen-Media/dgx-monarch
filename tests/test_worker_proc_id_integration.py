"""A real Worker loop restart under a live driver process, CPU only, on 127.0.0.1.

docs/TROUBLESHOOTING.md #2: a loop restarted at the same address used to keep
Monarch's singleton proc identity, so a driver that stayed up could not attach
again in the same process. Here two real ``worker_loop`` processes serve
per-start ids, a driver process attaches through ``mesh_attach``, one loop
restarts, and the same driver process attaches again through the replacement
path within seconds. A second case dials the bare address, which 0.8.0
resolves to the id the loop serves, and then attaches by id in the same
process.

A third case (issue #47) restarts every loop under a live mesh and sends a
render's capacity price: the wait ends on the delivery failure, and the
re-attach that ``submit_render`` calls attaches by the new ids, far inside the
30 s deadline the price used to wait. The liveness scan, policy admission and
client lease are stood in (procid_integration_helper.py says why).

The third case then recycles the healed mesh across the late stop report
the heal's 2 s stop of the old mesh leaves; the report is that stop's own and
ends no other wait.

Four more cases run the production teardown on 0.8.0: a lost host's stop
finishes inside its bound, a lost host beside a live one ends its stop on
Monarch's report on that very stop, and Recycle and the detach shutdown leave
calls that name their stop reason without an ERROR from the fault hook.

Every loop binds 127.0.0.1 on a port the kernel reports free just before it
starts (a restarted loop takes its old port back), so two runs of this module
on one machine never share a port; the driver binds port 0. Every process
gets its own HOME, so its actor ledger never touches the real one. Skipped
where torchmonarch does not import.
"""
from __future__ import annotations

import json
import logging
import os
import pwd
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("monarch.actor")

from dgx_monarch import worker_proc_id
from dgx_monarch.cli import actor_ledger

HELPER = Path(__file__).with_name("procid_integration_helper.py")
SRC = Path(__file__).resolve().parents[1] / "src"
SUFFIX = f"_dgxm_procid_test_{os.getpid()}"


def _listening(port: int) -> bool:
    want = f"0100007F:{port:04X}"
    with open("/proc/net/tcp") as table:
        return any(row.split()[1] == want and row.split()[3] == "0A"
                   for row in table.readlines()[1:])


def _free_port() -> int:
    """A loopback port nothing holds, named by the kernel, at 40000 or above.

    Fixed ports made two runs of this module on one machine collide. The floor
    keeps the loops clear of service ports such as the Worker's 26600.
    """
    for _attempt in range(200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        if port >= 40000 and not _listening(port):
            return port
    raise AssertionError("the kernel named no free loopback port at 40000 or above")


def _env(home: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ, HOME=str(home), PYTHONPATH=str(SRC),
               MONARCH_LOG_SUFFIX=SUFFIX, PYTHONDONTWRITEBYTECODE="1")
    env.update(extra)
    return env


class _Loop:
    """One real worker_loop process, killed by process group on the way out.

    A new loop takes a free port; a restart passes the old loop's port back.
    """

    def __init__(self, home: Path, port: int | None = None) -> None:
        port = _free_port() if port is None else port
        self.home, self.port = home, port
        self.address = f"tcp://127.0.0.1:{port}"
        home.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [sys.executable, str(HELPER), "host", self.address, SUFFIX],
            env=_env(home), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and not _listening(port):
                assert self.proc.poll() is None, "the worker loop exited at startup"
                time.sleep(0.1)
            assert _listening(port), "the worker loop never listened"
            self.proc_id = self.published()["proc_id"]
        except BaseException:
            # The caller never gets this object to clean up, so a loop that
            # failed its startup checks must not outlive them on the port.
            self.stop()
            raise

    def published(self) -> dict:
        ledger = actor_ledger.read_ledger(home=str(self.home))
        return ledger["loop"]

    def stop(self) -> None:
        if self.proc.poll() is None:
            for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 10)):
                try:
                    os.killpg(self.proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.proc.wait(wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and _listening(self.port):
            time.sleep(0.1)


def _line(proc: subprocess.Popen, timeout: float) -> dict:
    """The driver's next JSON line; on silence, fail with its stderr tail."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], deadline - time.monotonic())
        if not ready:
            break
        text = proc.stdout.readline()
        if not text:
            break
        if text.startswith("{"):
            return json.loads(text)
    errors = getattr(proc, "errors_path", None)
    tail = errors.read_text()[-3000:] if errors is not None and errors.exists() else ""
    raise AssertionError(f"the driver process said nothing within {timeout:.0f}s\n{tail}")


def _client(tmp_path: Path, spec: dict, budget: str) -> subprocess.Popen:
    (tmp_path / "driver").mkdir(exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, str(HELPER), "client", json.dumps(spec)],
        env=_env(tmp_path / "driver", HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT=budget),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=(tmp_path / "driver.err").open("w"), text=True, start_new_session=True)
    proc.errors_path = tmp_path / "driver.err"  # type: ignore[attr-defined]
    return proc


@pytest.fixture
def cleanup():
    owned: list = []
    yield owned
    for item in reversed(owned):
        if isinstance(item, _Loop):
            item.stop()
        elif item.poll() is None:
            os.killpg(item.pid, signal.SIGKILL)
            item.wait(10)
    log_dir = Path("/tmp") / pwd.getpwuid(os.getuid()).pw_name  # noqa: S108
    for path in log_dir.glob(f"monarch_log_*{SUFFIX}*.log"):
        path.unlink(missing_ok=True)


def test_a_driver_attaches_again_in_the_same_process_after_a_loop_restarts(
        tmp_path, cleanup):
    """The re-attach that hung until the config-push budget ran out now
    passes in seconds: the restarted loop serves a new id, the driver read it
    before dialing, and the driver sent its throwaway message first, because
    nothing else has talked to that loop since it restarted. In mutation runs
    of this test, serving the bare address, reusing the id, or skipping the
    throwaway message each made this same attach time out at the 20 s
    budget."""
    loop_a = _Loop(tmp_path / "a")
    cleanup.append(loop_a)
    loop_b = _Loop(tmp_path / "b")
    cleanup.append(loop_b)
    spec = {"mode": "restart",
            "hosts": [["a", loop_a.address], ["b", loop_b.address]],
            "homes": {"a": str(loop_a.home), "b": str(loop_b.home)}}
    driver = _client(tmp_path, spec, "20s")
    cleanup.append(driver)

    first = _line(driver, 120)
    assert first["ok"], first
    assert first["proc_ids"] == {loop_a.address: loop_a.proc_id,
                                 loop_b.address: loop_b.proc_id}

    # Let the loop acknowledge the driver's traffic first (Monarch acks about
    # every 500 ms), as any idle time between renders does. A loop killed
    # inside that window has the driver's old session replayed into its
    # successor, which attaches even with a reused id and so proves nothing.
    time.sleep(3.0)
    old_id = loop_a.proc_id
    loop_a.stop()
    loop_a = _Loop(tmp_path / "a", loop_a.port)  # the same address, a new start
    cleanup.append(loop_a)
    assert loop_a.proc_id != old_id
    driver.stdin.write("go\n")
    driver.stdin.flush()

    second = _line(driver, 120)
    assert second["ok"], second
    assert second["released"] is True
    assert second["proc_ids"] == {loop_a.address: loop_a.proc_id,
                                  loop_b.address: loop_b.proc_id}
    # Only the loop that restarted gets the throwaway message, and one attempt
    # lands, far inside the 20 s config-push budget a lost push would spend.
    assert second["primed"] == [loop_a.address] and second["attempts"] == 1, second
    assert second["wall_s"] < 10.0, second
    assert _line(driver, 30)["step"] == "done"


def test_a_bare_address_dial_finds_the_per_start_id_and_attaches(
        tmp_path, cleanup):
    """A dial of the bare address. On 0.7.0 the loop bounced it and the attach
    failed at the config-push bound; 0.8.0 asks the address for the id the
    loop serves (meta-pytorch/monarch#5066, #5273) and attaches at once. The
    driver still dials by id, and the same process then attaches by id."""
    loop = _Loop(tmp_path / "a")
    cleanup.append(loop)
    assert worker_proc_id.is_value(loop.proc_id)
    spec = {"mode": "bare", "hosts": [["a", loop.address]],
            "homes": {"a": str(loop.home)}}
    driver = _client(tmp_path, spec, "5s")
    cleanup.append(driver)

    bare = _line(driver, 120)
    assert bare["ok"] is True, bare
    assert bare["wall_s"] < 30.0, bare
    by_id = _line(driver, 120)
    assert by_id["ok"], by_id
    assert by_id["proc_ids"] == {loop.address: loop.proc_id}
    assert _line(driver, 30)["step"] == "done"


def test_a_render_after_every_loop_restarts_attaches_again_inside_the_render(
        tmp_path, cleanup):
    """docs/TROUBLESHOOTING.md #2, issue #47. Both loops restart under a live
    mesh, and nothing reaches them before the render, so the render's own price
    request is the first message on each old channel session. Monarch sends it
    again on a fresh session, the new loop reports it undeliverable at once, and
    the wait ends at its first poll instead of at its 30 s deadline, with no
    second request. The re-attach then reads the new ids,
    retires the stale mesh through the deliberate teardown and the (stood-in)
    liveness release, and attaches by the new ids, and the price answers on the
    new mesh. In mutation runs, removing the short circuit made this price wait
    out the deadline, and removing the re-attach left it refused."""
    from dgx_monarch import capacity_agreement

    loop_a = _Loop(tmp_path / "a")
    cleanup.append(loop_a)
    loop_b = _Loop(tmp_path / "b")
    cleanup.append(loop_b)
    spec = {"mode": "render", "recycle": True,
            "hosts": [["a", loop_a.address], ["b", loop_b.address]],
            "homes": {"a": str(loop_a.home), "b": str(loop_b.home)}}
    driver = _client(tmp_path, spec, "20s")
    cleanup.append(driver)
    assert _line(driver, 120)["ok"]

    # Let the loops acknowledge the driver's traffic first (the first case says why).
    time.sleep(3.0)
    old_ids = {loop_a.address: loop_a.proc_id, loop_b.address: loop_b.proc_id}
    loop_a.stop()
    loop_b.stop()
    loop_a = _Loop(tmp_path / "a", loop_a.port)
    cleanup.append(loop_a)
    loop_b = _Loop(tmp_path / "b", loop_b.port)
    cleanup.append(loop_b)
    driver.stdin.write("go\n")
    driver.stdin.flush()

    render = _line(driver, 120)
    assert render["ok"], render
    deadline = capacity_agreement.CAPACITY_QUOTE_DEADLINE_S
    # One poll: on the pin before 0.8.0 only a second request, sent 2 s in,
    # came back, and the wait ended about 2.5 s in.
    assert render["refused_after_s"] < 1.5, render
    assert render["wall_s"] < deadline / 2, render
    assert render["old_retired"] and render["new_mesh"], render
    # The fault text spells the mesh ActorMesh._name reads, so the filter that
    # keeps a retired mesh's stragglers out of the wait matched this one.
    assert render["mesh_uid"] and render["failures_named"], render
    assert render["new_mesh_uid"] not in (None, render["mesh_uid"]), render
    assert render["teardown"] == "liveness_release", render
    [reattach] = render["reattach"]
    assert (reattach["outcome"], reattach["verdict"]) == ("attached", "all"), render
    assert render["proc_ids"] == {loop_a.address: loop_a.proc_id,
                                  loop_b.address: loop_b.proc_id}
    assert set(render["proc_ids"].values()).isdisjoint(old_ids.values())
    assert render["primed"] == [f"{loop_a.address},{loop_b.address}"], render
    # Then a Recycle of the healed mesh, held open about 20 s by a frozen proc,
    # across the report the heal's 2 s stop of the old mesh leaves about 30 s
    # later. That report names the old mesh's controller, so it is the heal's
    # stop's own, logged at INFO, and it ends no other wait: the Recycle
    # returns recycled. Before proc_stops it ended the Recycle as a timeout.
    recycled = _line(driver, 150)
    assert recycled["status"] == "recycled", recycled
    assert recycled["teardown_complete"] is True and recycled["replacement_blocked"] is None, recycled
    assert recycled["overlapped"] is True, recycled
    assert recycled["report_levels"] and set(recycled["report_levels"]) == {logging.INFO}, recycled
    assert recycled["stop_errors"] == [], recycled
    assert _line(driver, 30)["step"] == "done"


def _kill_group(loop: _Loop) -> None:
    """A lost host: the loop and every proc it spawned die at once, with no stop."""
    try:
        os.killpg(loop.proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    loop.proc.wait(10)


def test_a_lost_hosts_fleet_stops_within_its_bound_and_frees_the_replacement(
        tmp_path, cleanup):
    """On torchmonarch 0.8.0 the stop that follows a host's death waits out the
    ProcMesh controller's state query of the dead host agent first, about 52 s
    on CPU. The supervision reconcile's old 30 s wait timed out and left the
    handle blocked until ComfyUI restarted; with the bound
    (mesh_stop.proc_stop_bound_s) the stop completes and nothing is blocked.
    About 75 s of wall time."""
    loop = _Loop(tmp_path / "a")
    cleanup.append(loop)
    spec = {"mode": "lost_host", "hosts": [["a", loop.address]],
            "homes": {"a": str(loop.home)}}
    driver = _client(tmp_path, spec, "20s")
    cleanup.append(driver)
    built = _line(driver, 120)
    assert built["ok"], built
    time.sleep(2.0)
    _kill_group(loop)
    driver.stdin.write("go\n")
    driver.stdin.flush()
    lost = _line(driver, 2 * built["bound_s"] + 120)
    assert lost["ok"], lost
    assert lost["teardown_complete"] is True, lost
    assert lost["replacement_blocked"] is None, lost
    # The name recorded at bring-up is the one Monarch's faults carry, so the
    # retired mesh's late faults, and Monarch's log_level and setup meshes'
    # (the spawn passes a bootstrap, as production does), log at INFO: no ERROR.
    assert lost["retired"] is True, lost
    assert lost["errors"] == [], lost
    assert any("mesh_name=setup-" in line for line in lost["notices"]), lost
    assert _line(driver, 30)["step"] == "done"


def test_an_idle_lost_host_is_reported_in_plain_words_for_the_fleet_the_driver_holds(tmp_path, cleanup):
    """Nothing in flight, one of two loops lost: Monarch's fault for the Worker
    mesh reaches the hook about 30 s later. Bring-up recorded the mesh through
    mesh_factory._record_mesh, as production does, so the hook opens its ERROR
    with the cause in plain words for the box the driver still holds
    (worker_loss.fault_words). About 45 s of wall time."""
    loop_a = _Loop(tmp_path / "a")
    cleanup.append(loop_a)
    loop_b = _Loop(tmp_path / "b")
    cleanup.append(loop_b)
    spec = {"mode": "idle_loss",
            "hosts": [["a", loop_a.address], ["b", loop_b.address]],
            "homes": {"a": str(loop_a.home), "b": str(loop_b.home)}}
    driver = _client(tmp_path, spec, "20s")
    cleanup.append(driver)
    built = _line(driver, 120)
    assert built["ok"], built
    time.sleep(2.0)
    _kill_group(loop_b)
    driver.stdin.write("go\n")
    driver.stdin.flush()
    lost = _line(driver, 120)
    assert lost["ok"], lost
    faults = [line for line in lost["errors"] if line.startswith("cluster fault:")]
    assert faults, lost
    # The box that was lost, named as box_names names a loopback pair's second host.
    assert faults[0].startswith("cluster fault: The driver lost contact with the sibling at b."), faults
    assert "dgxm_worker" not in faults[0].split("MeshFailure", 1)[0], faults  # the words name no mesh
    assert _line(driver, 30)["step"] == "done"


def test_a_lost_host_beside_a_live_one_ends_the_stop_on_monarchs_report(tmp_path, cleanup):
    """One of two loops is lost, and a call meets it about 50 s later, as a
    render did in the lost40 leg. Monarch's ProcMesh controller cannot confirm
    the lost procs, ends the stop with "error stopping proc mesh" after
    proc_stop_max_idle (30 s), and fails, and the stop's own Future never
    completes: the reconcile's wait ran its whole 100 s bound. The report names
    the controller proc_mesh_controller_<the name the pack spawned the mesh
    under>-<uid>, which ties it to this stop, and the wait now ends on it with
    the unknown outcome a timed-out stop gives. A stop sent within about a
    minute of the loss queues first behind the controller's state query of
    the dead host (get_proc_state_max_idle, 1 m), so its report can come up to
    90 s in, still inside the bound. About 110 s of wall time."""
    loop_a = _Loop(tmp_path / "a")
    cleanup.append(loop_a)
    loop_b = _Loop(tmp_path / "b")
    cleanup.append(loop_b)
    spec = {"mode": "lost_beside_live",
            "hosts": [["a", loop_a.address], ["b", loop_b.address]],
            "homes": {"a": str(loop_a.home), "b": str(loop_b.home)}}
    driver = _client(tmp_path, spec, "20s")
    cleanup.append(driver)
    built = _line(driver, 120)
    assert built["ok"], built
    time.sleep(2.0)
    _kill_group(loop_a)
    time.sleep(50.0)
    driver.stdin.write("go\n")
    driver.stdin.flush()
    lost = _line(driver, 2 * built["bound_s"] + 120)
    assert lost["ok"], lost
    assert lost["bound_s"] == built["bound_s"] == 100.0, lost
    assert 25.0 < lost["stop_s"] < 50.0, lost  # well before the bound
    assert "without confirming every proc, so its outcome is unknown" in lost["error"], lost
    assert lost["label"] in lost["error"], lost  # the report named this mesh's controller
    assert lost["teardown_complete"] is False and lost["replacement_blocked"], lost
    assert lost["block_cause"] == "stop_timeout_after_death", lost
    assert lost["stop_errors"] == [], lost  # the report itself logs at INFO
    assert _line(driver, 30)["step"] == "done"


@pytest.mark.parametrize("mode", ["stop_recycle", "stop_detach"])
def test_a_call_in_flight_or_after_a_deliberate_stop_names_the_reason_and_stays_quiet(
        tmp_path, cleanup, mode):
    """Under 0.8.0's direct spawn a ProcMesh stop alone reaches a call in
    flight as "stopped: parent draining". Recycle and the detach shutdown stop
    the actors with their reason first (mesh_stop.stop_with_reason), so both
    the call in flight and one sent after the stop name the pack's reason, and
    the fault hook logs no ERROR for either."""
    loop = _Loop(tmp_path / "a")
    cleanup.append(loop)
    spec = {"mode": mode, "hosts": [["a", loop.address]], "homes": {"a": str(loop.home)}}
    driver = _client(tmp_path, spec, "20s")
    cleanup.append(driver)
    built = _line(driver, 120)
    # 0.8.0 sets the root logger to INFO inside attach_to_workers(); the pack puts it back.
    assert built["ok"] and built["root_level"] == "WARNING", built
    stopped = _line(driver, 300)
    reason = "dgx-monarch recycle" if mode == "stop_recycle" else "dgx-monarch client detach"
    assert stopped["ok"], stopped
    # Both carry the reason, which mesh_safety's teardown markers name.
    assert f"stopped: {reason}" in stopped["in_flight"], stopped
    assert f"stopped: {reason}" in stopped["straggler"], stopped
    assert "parent draining" not in stopped["in_flight"] + stopped["straggler"], stopped
    assert stopped["errors"] == [], stopped
    assert _line(driver, 30)["step"] == "done"
