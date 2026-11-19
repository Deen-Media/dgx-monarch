"""A Monarch controller's report on a proc stop ends that stop's wait, and no other.

Under the pinned Monarch a ProcMesh controller that cannot confirm every proc
stopped fails with "error stopping proc mesh", and the stop's Future never
completes. The report names the controller after the ProcMesh's id, which
starts with the name the pack spawned the mesh under (proc_stops). A report
left by another stop, such as a heal's 2 s stop that gave up 20 to 30 s
earlier, must change no wait: fed into a real Recycle it used to end a
healthy stop as a timeout, with the remedy "restart ComfyUI". The texts are
captured on CPU, uids and addresses replaced. tests/stale_report_probe.py and
tests/test_worker_proc_id_integration.py run the real runtime.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from dgx_monarch import mesh_stop, proc_stops, retired_meshes

NAME = "dgxm-procs-testmesh"
OWN_REPORT = (
    "MeshFailure(mesh_name=<none>, crashed_ranks=[], event=Supervision event: actor "
    f"proc_mesh_controller_{NAME}-ckuojgrl1zy<BRGbzmJ9tTX>.local<fg13gHugDYh>@tcp://127.0.0.1:35109 "
    "failed:\n  error stopping proc mesh: statuses: Timeout(30.495408301s)=0..1;Stopped=1..2)")
# A restart heal's 2 s stop left this behind for another ProcMesh (the reviewer's probes).
FOREIGN_REPORT = (
    "MeshFailure(mesh_name=<none>, crashed_ranks=[], event=Supervision event: actor "
    "proc_mesh_controller_anon-wnpcwboqf6n<cVAgKdNmYsM>.local<9yH82kDpLa4>@tcp://127.0.0.1:33255 "
    "failed:\n  error stopping proc mesh: statuses: Timeout(30.000805711s)=0..2)")


class _Procs:
    """A ProcMesh whose stop's Future finishes when ``finish`` is set, or times out."""

    def __init__(self, name: str | None = NAME, finish_after: float | None = None):
        self.finish = threading.Event()
        self.timeouts: list = []
        if name:
            proc_stops.remember(self, name)
        if finish_after is not None:
            threading.Timer(finish_after, self.finish.set).start()

    def stop(self, reason):
        return SimpleNamespace(get=self._get)

    def _get(self, timeout=None):
        self.timeouts.append(timeout)
        if not self.finish.wait(timeout):
            raise TimeoutError("returns_future operation did not finish in time")
        return "stopped"


@pytest.fixture(autouse=True)
def _clean():
    proc_stops.reset()
    yield
    proc_stops.reset()


def _hook_soon(text: str, delay: float = 0.2) -> list:
    """The fault hook's call, from Monarch's thread, while the stop waits; collects its label."""
    labels: list = []
    threading.Timer(delay, lambda: labels.append(retired_meshes.notice(text))).start()
    return labels


def test_a_report_on_this_stop_ends_its_wait_at_once_as_the_unknown_outcome():
    """No grace after the report: the stop's Future cannot complete once its controller reported."""
    procs = _Procs()
    labels = _hook_soon(OWN_REPORT, delay=0.2)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="without confirming every proc, so its outcome is unknown") as ended:
        mesh_stop.stop_procs(procs, "dgx-monarch client detach", 30.0)
    assert time.monotonic() - started < 0.8  # the report came at 0.2 s
    assert "Timeout(30.495408301s)=0..1;Stopped=1..2" in str(ended.value)
    assert labels == [retired_meshes.STOP_NOTICE]
    procs.finish.set()


def test_a_finished_future_keeps_its_outcome_when_a_report_is_also_present(monkeypatch):
    """Both are in when the wait reads them: the Future's own outcome is read first.

    The wait runs on the caller (no thread can start, as at interpreter
    shutdown), so the report lands inside the Future's get, before it returns.
    Reading the report first would end this finished stop as unknown.
    """
    class Reported(_Procs):
        def _get(self, timeout=None):
            assert retired_meshes.notice(OWN_REPORT) == retired_meshes.STOP_NOTICE
            return "stopped"

    def refuse(self):
        raise RuntimeError("can't create new thread at interpreter shutdown")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    assert mesh_stop.stop_procs(Reported(), "dgx-monarch recycle", 30.0) == "stopped"


def test_a_report_without_a_timed_out_rank_is_still_the_unknown_outcome():
    """Upstream sends the same report while a rank is still stopping: never a definite failure."""
    procs = _Procs()
    _hook_soon(OWN_REPORT.replace("Timeout(30.495408301s)=0..1;Stopped=1..2", "Stopping=0..1;Stopped=1..2"))
    with pytest.raises(TimeoutError, match="outcome is unknown"):
        mesh_stop.stop_procs(procs, "dgx-monarch recycle", 30.0)
    procs.finish.set()


def test_a_report_naming_another_controller_changes_no_wait():
    procs = _Procs(finish_after=1.0)
    labels = _hook_soon(FOREIGN_REPORT)
    assert mesh_stop.stop_procs(procs, "dgx-monarch recycle", 30.0) == "stopped"
    assert labels == [None]  # not the pack's: it stays an ERROR with a toast


def test_a_stop_that_already_finished_keeps_its_own_outcome_and_no_later_report_is_its():
    procs = _Procs()
    procs.finish.set()
    assert mesh_stop.stop_procs(procs, "dgx-monarch recycle", 30.0) == "stopped"
    assert retired_meshes.notice(OWN_REPORT) is None  # a confirmed stop is no longer registered


def test_a_stop_whose_waiter_gave_up_still_owns_its_late_report_and_no_other_wait():
    """The heal's 2 s stop gives up; its report 20 to 30 s later reaches a later Recycle."""
    heal = _Procs(name="dgxm-procs-healmesh")
    with pytest.raises(TimeoutError, match="did not finish in time"):
        mesh_stop.stop_procs(heal, "dgx-monarch restart heal", 0.2)
    recycle = _Procs(finish_after=1.0)
    late = OWN_REPORT.replace(NAME, "dgxm-procs-healmesh")
    labels = _hook_soon(late)
    assert mesh_stop.stop_procs(recycle, "dgx-monarch recycle", 30.0) == "stopped"
    assert labels == [retired_meshes.STOP_NOTICE]  # the heal's, kept quiet
    heal.finish.set()


def test_a_given_up_stop_is_forgotten_once_it_can_no_longer_report():
    stop = proc_stops.started(_Procs(), keep_s=0.1)
    proc_stops.left(stop, confirmed=False)
    assert proc_stops.claim(OWN_REPORT) is True
    time.sleep(0.2)
    assert proc_stops.claim(OWN_REPORT) is False


def test_a_confirmed_stop_is_forgotten_at_once_and_a_running_one_never_expires():
    procs = _Procs()
    procs.finish.set()
    assert mesh_stop.stop_procs(procs, "dgx-monarch recycle", 5.0) == "stopped"
    assert proc_stops.claim(OWN_REPORT) is False
    running = proc_stops.started(_Procs(), keep_s=0.0)
    time.sleep(0.05)
    assert proc_stops.claim(OWN_REPORT) is True and running.ended.is_set()


def test_a_proc_mesh_with_no_recorded_name_takes_no_report():
    procs = _Procs(name=None, finish_after=0.5)
    labels = _hook_soon(OWN_REPORT)
    assert mesh_stop.stop_procs(procs, "dgx-monarch recycle", 30.0) == "stopped"
    assert labels == [None]


def test_a_stop_that_raises_keeps_its_own_error():
    class Refusing(_Procs):
        def _get(self, timeout=None):
            raise ValueError("stop refused")

    with pytest.raises(ValueError, match="stop refused"):
        mesh_stop.stop_procs(Refusing(), "dgx-monarch recycle", 5.0)


def test_a_stop_waits_on_its_own_thread_unless_none_can_start(monkeypatch):
    """At interpreter shutdown no thread starts, so the wait runs on the caller."""
    seen: list = []

    class Seen(_Procs):
        def _get(self, timeout=None):
            seen.append(threading.current_thread() is threading.main_thread())

    mesh_stop.stop_procs(Seen(), "r", 5.0)

    def refuse(self):
        raise RuntimeError("can't create new thread at interpreter shutdown")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    mesh_stop.stop_procs(Seen(), "r", 5.0)
    assert seen == [False, True]


def test_the_name_survives_monarchs_label_rules_whole():
    """Monarch lowercases the controller label, keeps [a-z0-9_-] and cuts it at 63 characters."""
    name = proc_stops.fresh_name()
    assert name.startswith(proc_stops.NAME_PREFIX) and name != proc_stops.fresh_name()
    assert name == name.lower() and name.replace("-", "").isalpha()
    uid = "abcdefghijk"  # an 11-character base58 instance uid, lowercased
    assert len(proc_stops.label_prefix(name) + uid) <= 63


def _run_stale_probe(tmp_path, mode: str) -> dict:
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    tests = Path(__file__).resolve().parent
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("MONARCH_", "HYPERACTOR_", "DGXM_MONARCH_POLICY"))}
    environment.update(HOME=str(tmp_path), XDG_RUNTIME_DIR=str(runtime_dir), CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=os.pathsep.join(filter(None, (str(tests.parent / "src"),
                                                                os.environ.get("PYTHONPATH")))))
    proc = subprocess.run([sys.executable, str(tests / "stale_report_probe.py"), mode],
                          capture_output=True, text=True, timeout=240, env=environment, cwd=tmp_path)
    assert proc.returncode == 0, (
        f"probe crashed rc={proc.returncode}\nstdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("mode", ["control", "stale"])
def test_a_real_recycle_with_another_meshs_report_arriving_still_recycles(tmp_path, mode):
    """The reviewer's case: before proc_stops the fed report made this proc_stop_timed_out."""
    pytest.importorskip("monarch.actor")
    result = _run_stale_probe(tmp_path, mode)
    assert result["name"] and result["name"].startswith(proc_stops.NAME_PREFIX), result
    if mode == "stale":
        assert result["stops_in_flight_when_fed"] == 1, result
    assert result["status"] == "recycled", result
    assert result["teardown_complete"] is True and result["replacement_blocked"] is None, result


def _stop_calls_in_source() -> list[tuple[str, str, str]]:
    """Every ``<anything>.stop(...)`` call in src: file, innermost function, call text."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"
    found: list[tuple[str, str, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self, relative: str):
            self.relative, self.function = relative, "<module>"

        def visit_FunctionDef(self, node):
            outer, self.function = self.function, node.name
            self.generic_visit(node)
            self.function = outer

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            if getattr(node.func, "attr", None) == "stop":
                found.append((self.relative, self.function, ast.unparse(node)))
            self.generic_visit(node)

    for path in sorted(src.rglob("*.py")):
        Visitor(str(path.relative_to(src))).visit(ast.parse(path.read_text()))
    return found


def test_every_proc_stop_the_pack_issues_is_registered():
    """Every .stop() call in src, whatever its receiver is called, is listed here.

    The one that stops a ProcMesh is mesh_stop.stop_procs' own, which registers
    first; the others stop a thread or a watcher and take no reason. A new
    .stop() anywhere, a ProcMesh stop under any name included, fails this.
    """
    assert sorted(_stop_calls_in_source()) == sorted([
        ("mesh_stop.py", "stop_procs", "procs.stop(reason)"),
        ("monarch_diagnostics.py", "configure", "_EXPORTER.stop()"),
        ("monarch_diagnostics.py", "configure", "_EXPORTER.stop()"),
        ("studio/server.py", "lifecycle", "media_assets.stop()"),
    ])
