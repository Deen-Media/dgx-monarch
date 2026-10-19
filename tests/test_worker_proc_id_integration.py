"""A real Worker loop restart under a live driver process, CPU only, on 127.0.0.1.

docs/TROUBLESHOOTING.md #2: a loop restarted at the same address used to keep
Monarch's singleton proc identity, so a driver that stayed up could not attach
again in the same process. Here two real ``worker_loop`` processes serve
per-start ids, a driver process attaches through ``mesh_attach``, one loop
restarts, and the same driver process attaches again through the replacement
path within seconds. A second case pins the upgrade direction a new loop
cannot fix: an older driver that dials the bare address fails at its
config-push bound rather than hanging.

A third case (issue #47) restarts every loop under a live mesh and sends a
render's capacity price: the wait ends on the delivery failure, and the
re-attach that ``submit_render`` calls attaches by the new ids, far inside the
30 s deadline the price used to wait. The liveness scan, policy admission and
client lease are stood in (procid_integration_helper.py says why).

Every process binds 127.0.0.1 on ports 27850 to 27859, or 27920 to 27929 for
the third case, and gets its own HOME, so its actor ledger never touches the
real one. Skipped where torchmonarch does not import.
"""
from __future__ import annotations

import json
import os
import pwd
import select
import signal
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


def _env(home: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ, HOME=str(home), PYTHONPATH=str(SRC),
               MONARCH_LOG_SUFFIX=SUFFIX, PYTHONDONTWRITEBYTECODE="1")
    env.update(extra)
    return env


class _Loop:
    """One real worker_loop process, killed by process group on the way out."""

    def __init__(self, home: Path, port: int) -> None:
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
    loop_a = _Loop(tmp_path / "a", 27851)
    cleanup.append(loop_a)
    loop_b = _Loop(tmp_path / "b", 27852)
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
    loop_a = _Loop(tmp_path / "a", 27851)
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


def test_an_older_driver_dialing_the_bare_address_fails_at_its_bound(
        tmp_path, cleanup):
    """A driver that predates per-start ids dials the bare address. The loop
    bounces it at once, and the attach fails when the config-push budget runs
    out: bounded, never a hang. The same process then attaches by id."""
    loop = _Loop(tmp_path / "a", 27853)
    cleanup.append(loop)
    assert worker_proc_id.is_value(loop.proc_id)
    spec = {"mode": "bare", "hosts": [["a", loop.address]],
            "homes": {"a": str(loop.home)}}
    driver = _client(tmp_path, spec, "5s")
    cleanup.append(driver)

    bare = _line(driver, 120)
    assert bare["ok"] is False, bare
    assert bare["wall_s"] < 30.0, bare
    by_id = _line(driver, 120)
    assert by_id["ok"], by_id
    assert by_id["proc_ids"] == {loop.address: loop.proc_id}
    assert _line(driver, 30)["step"] == "done"


def test_a_render_after_every_loop_restarts_attaches_again_inside_the_render(
        tmp_path, cleanup):
    """docs/TROUBLESHOOTING.md #2, issue #47. Both loops restart under a live
    mesh, and nothing reaches them before the render, so the render's own price
    request is the message each old channel session loses. The wait asks once
    more, Monarch reports that request undeliverable, and the wait ends there
    instead of at its 30 s deadline. The re-attach then reads the new ids,
    retires the stale mesh through the deliberate teardown and the (stood-in)
    liveness release, and attaches by the new ids, and the price answers on the
    new mesh. In mutation runs, removing the short circuit made this price wait
    out the deadline, and removing the re-attach left it refused."""
    from dgx_monarch import capacity_agreement

    loop_a = _Loop(tmp_path / "a", 27921)
    cleanup.append(loop_a)
    loop_b = _Loop(tmp_path / "b", 27922)
    cleanup.append(loop_b)
    spec = {"mode": "render",
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
    loop_a = _Loop(tmp_path / "a", 27921)
    cleanup.append(loop_a)
    loop_b = _Loop(tmp_path / "b", 27922)
    cleanup.append(loop_b)
    driver.stdin.write("go\n")
    driver.stdin.flush()

    render = _line(driver, 120)
    assert render["ok"], render
    deadline = capacity_agreement.CAPACITY_QUOTE_DEADLINE_S
    assert render["refused_after_s"] < 5.0, render
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
    assert _line(driver, 30)["step"] == "done"
