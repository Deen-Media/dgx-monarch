"""The wait for a proc stop that can follow a Worker host's death, and the budgets around it.

On torchmonarch 0.8.0 with direct spawn the pack learns of a lost host at
about 11 s (0.7.0: about 68 s), and its stop then takes about 52 s: the
ProcMesh controller's Stop queues behind its state query of the dead host
agent (get_proc_state_max_idle, 1m) before proc_stop_max_idle (30s) applies.
A 30 s wait timed out and left the handle blocked until ComfyUI restarted.
tests/test_worker_proc_id_integration.py runs the real loss; this file pins
the bound, its cap and admission refusal, the call sites and every sum that
holds the stop.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import mesh_safety, mesh_stop
from dgx_monarch.config import ClusterConfigError

SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"
MEASURED_LOST_HOST_STOP_S = 52.7  # the slowest of the CPU lost-host stops on 0.8.0
# The longest a cleanup's stop needed to confirm on a lone lost host, over stops
# sent 0 to 45 s after the loss: 60.0 s, sent 8 s after it (each confirmed 68.0 s
# after the loss; one sent within about 7.9 s got Monarch's report 30 s in).
LONGEST_CLEANUP_CONFIRMATION_S = 60.0
RECYCLE_DEADLINE_S = 180.0        # mesh_recycle.recycle_detailed
RECYCLE_GROUP_TEARDOWN_S = 60.0   # mesh_recycle's group teardown
LIFECYCLE_LOCK_S = 10.0           # mesh_setup.lifecycle_lock's default wait
SAME_KEY_WAITER_S = 180.0         # mesh_runtime's default mesh creation timeout
SETUP_ROLLBACK_JOIN = "proc_stop_bound_s() + ACTOR_STOP_S + 30"


def _config(monkeypatch, values):
    monkeypatch.setitem(sys.modules, "monarch.config", SimpleNamespace(get_global_config=lambda: values))


def test_the_default_bound_covers_the_measured_stop_and_both_idle_limits():
    assert mesh_stop.PROC_STOP_DEFAULT_S == 60.0 + 30.0 + mesh_stop.PROC_STOP_MARGIN_S == 100.0
    assert mesh_stop.STOP_IDLE_MAX_S == 90.0
    assert mesh_stop.PROC_STOP_DEFAULT_S > MEASURED_LOST_HOST_STOP_S


@pytest.mark.parametrize("text,seconds", [
    ("30s", 30.0), ("1m", 60.0), ("1m 30s", 90.0), ("1m30s", 90.0), ("2min 5sec", 125.0),
    ("500ms", 0.5), ("1h", 3600.0), ("1.5s", 1.5), (" 1m 30s ", 90.0),
    ("", None), ("90", None), ("1x", None), ("1m junk", None), (None, None)])
def test_durations_read_every_form_monarch_reads_back(text, seconds):
    assert mesh_stop.duration_seconds(text) == seconds


def test_the_bound_follows_a_lower_override_and_never_exceeds_the_default(monkeypatch):
    monkeypatch.delitem(sys.modules, "monarch.config", raising=False)
    assert mesh_stop.proc_stop_bound_s() == mesh_stop.PROC_STOP_DEFAULT_S
    _config(monkeypatch, {"get_proc_state_max_idle": "20s", "proc_stop_max_idle": "10s"})
    assert mesh_stop.proc_stop_bound_s() == 40.0
    _config(monkeypatch, {"get_proc_state_max_idle": "1m 10s", "proc_stop_max_idle": "15s"})
    assert mesh_stop.proc_stop_bound_s() == 95.0
    _config(monkeypatch, {"get_proc_state_max_idle": "5m", "proc_stop_max_idle": "45s"})
    assert mesh_stop.proc_stop_bound_s() == mesh_stop.PROC_STOP_DEFAULT_S


@pytest.mark.parametrize("values", [{}, {"get_proc_state_max_idle": "soon", "proc_stop_max_idle": "-3s"},
                                    {"get_proc_state_max_idle": None}])
def test_an_unreadable_limit_keeps_its_default_bound(monkeypatch, values):
    _config(monkeypatch, values)
    assert mesh_stop.proc_stop_bound_s() == mesh_stop.PROC_STOP_DEFAULT_S


def test_a_failing_config_read_keeps_the_defaults(monkeypatch):
    def broken():
        raise RuntimeError("config unavailable")

    monkeypatch.setitem(sys.modules, "monarch.config", SimpleNamespace(get_global_config=broken))
    assert mesh_stop.proc_stop_bound_s() == mesh_stop.PROC_STOP_DEFAULT_S


def test_monarchs_defaults_and_lower_limits_are_admitted(monkeypatch):
    monkeypatch.delitem(sys.modules, "monarch.config", raising=False)
    assert mesh_stop.stop_limits_refusal() is None
    for values in ({"get_proc_state_max_idle": "1m", "proc_stop_max_idle": "30s"},
                   {"get_proc_state_max_idle": "1m 0s", "proc_stop_max_idle": "15s"}):
        _config(monkeypatch, values)
        assert mesh_stop.stop_limits_refusal() is None


def test_an_override_above_the_default_refuses_at_driver_admission(monkeypatch):
    from dgx_monarch import monarch_launch

    _config(monkeypatch, {"get_proc_state_max_idle": "5m", "proc_stop_max_idle": "30s"})
    monkeypatch.setenv("HYPERACTOR_MESH_GET_PROC_STATE_MAX_IDLE", "5m")
    monkeypatch.delenv("HYPERACTOR_MESH_PROC_STOP_MAX_IDLE", raising=False)
    with pytest.raises(ClusterConfigError) as refused:
        monarch_launch._refuse_stop_limits()
    text = str(refused.value)
    assert "get_proc_state_max_idle (5m) and proc_stop_max_idle (30s) add up to 330 s" in text
    # Only the variable that is set, and the lower-it remedy beside unset.
    assert ("unset HYPERACTOR_MESH_GET_PROC_STATE_MAX_IDLE where ComfyUI starts, or lower it so "
            "the two add up to 90 s or less, then restart ComfyUI") in text
    assert "HYPERACTOR_MESH_PROC_STOP_MAX_IDLE" not in text
    assert text.endswith("(docs/TROUBLESHOOTING.md #119)")


def test_with_both_variables_set_the_refusal_names_both(monkeypatch):
    _config(monkeypatch, {"get_proc_state_max_idle": "1m", "proc_stop_max_idle": "1m"})
    monkeypatch.setenv("HYPERACTOR_MESH_GET_PROC_STATE_MAX_IDLE", "1m")
    monkeypatch.setenv("HYPERACTOR_MESH_PROC_STOP_MAX_IDLE", "1m")
    text = mesh_stop.stop_limits_refusal()
    assert ("unset HYPERACTOR_MESH_GET_PROC_STATE_MAX_IDLE or HYPERACTOR_MESH_PROC_STOP_MAX_IDLE where "
            "ComfyUI starts, or lower them to add up to 90 s or less") in text


def test_with_no_variable_set_no_setting_is_offered(monkeypatch):
    _config(monkeypatch, {"get_proc_state_max_idle": "5m", "proc_stop_max_idle": "soon"})
    monkeypatch.delenv("HYPERACTOR_MESH_GET_PROC_STATE_MAX_IDLE", raising=False)
    monkeypatch.delenv("HYPERACTOR_MESH_PROC_STOP_MAX_IDLE", raising=False)
    text = mesh_stop.stop_limits_refusal()
    assert text.startswith("Monarch's proc_stop_max_idle reads 'soon'")
    assert "HYPERACTOR_MESH_PROC_STOP_MAX_IDLE is not set, so reinstall the pinned torchmonarch" in text
    _config(monkeypatch, {"get_proc_state_max_idle": "5m", "proc_stop_max_idle": "30s"})
    assert "neither variable is set, so reinstall" in mesh_stop.stop_limits_refusal()


@pytest.mark.parametrize("values", [{"get_proc_state_max_idle": "soon", "proc_stop_max_idle": "30s"},
                                    {"get_proc_state_max_idle": None, "proc_stop_max_idle": "30s"}])
def test_an_unreadable_limit_refuses_at_driver_admission(monkeypatch, values):
    _config(monkeypatch, values)
    monkeypatch.setenv("HYPERACTOR_MESH_GET_PROC_STATE_MAX_IDLE", "soon")
    text = mesh_stop.stop_limits_refusal()
    assert text is not None and "get_proc_state_max_idle reads" in text
    assert ("unset HYPERACTOR_MESH_GET_PROC_STATE_MAX_IDLE where ComfyUI starts, or set it to a "
            "duration such as 30s") in text and text.endswith("#119)")


def test_a_limit_the_runtime_cannot_report_refuses_and_the_bound_keeps_its_default(monkeypatch):
    def broken():
        raise RuntimeError("config unavailable")

    for config in (SimpleNamespace(get_global_config=lambda: {"proc_stop_max_idle": "30s"}),
                   SimpleNamespace(get_global_config=lambda: ["not", "a", "dict"]),
                   SimpleNamespace(get_global_config=broken)):
        monkeypatch.setitem(sys.modules, "monarch.config", config)
        text = mesh_stop.stop_limits_refusal()
        assert text is not None and text.startswith("Monarch's get_proc_state_max_idle reads nothing,"), text
        assert mesh_stop.proc_stop_bound_s() == mesh_stop.PROC_STOP_DEFAULT_S


def test_the_admission_check_loads_the_config_it_reads():
    """Monarch is loaded by then; the import keeps the check from ever skipping."""
    tree = ast.parse((SRC / "monarch_launch.py").read_text())
    check = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "_refuse_stop_limits")
    assert "import monarch.config" in ast.unparse(check)


_READ_BACK = ("import json, monarch.actor\n"
              "from dgx_monarch import mesh_stop\n"
              "print(json.dumps([mesh_stop.stop_limits_refusal(), mesh_stop.proc_stop_bound_s()]))\n")


@pytest.mark.skipif(importlib.util.find_spec("monarch") is None, reason="needs torchmonarch")
@pytest.mark.parametrize("state,proc,refused,bound", [
    (None, None, None, 100.0),
    ("20s", "10s", None, 40.0),                       # a lower setting is honoured
    ("5m", "1m 30s", "add up to 390 s", 100.0),       # the compound form Monarch reads back
])
def test_the_real_runtime_reads_back_each_override(tmp_path, state, proc, refused, bound):
    """Monarch reads the variables at import, so each case is its own interpreter."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("HYPERACTOR_MESH_")}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(SRC.parent), CUDA_VISIBLE_DEVICES="")
    for key, value in (("HYPERACTOR_MESH_GET_PROC_STATE_MAX_IDLE", state),
                       ("HYPERACTOR_MESH_PROC_STOP_MAX_IDLE", proc)):
        if value is not None:
            env[key] = value
    result = subprocess.run([sys.executable, "-c", _READ_BACK], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    reason, observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed == bound
    if refused is None:
        assert reason is None
    else:
        assert refused in reason and "(5m)" in reason and "(1m 30s)" in reason


def test_the_driver_admission_runs_the_check():
    tree = ast.parse((SRC / "monarch_launch.py").read_text())
    prepare = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "prepare_driver")
    assert "_refuse_stop_limits()" in ast.unparse(prepare)


def test_only_the_proc_stop_takes_the_bound(monkeypatch):
    monkeypatch.delitem(sys.modules, "monarch.config", raising=False)
    bound = mesh_stop.PROC_STOP_DEFAULT_S
    assert mesh_stop.shutdown_phases(30.0, bound) == (30.0, bound)    # reconcile, config change, cleanup
    assert mesh_stop.shutdown_phases(60.0) == (60.0, 60.0)            # a plain shutdown, as before
    assert mesh_stop.shutdown_phases(300.0) == (60.0, bound)
    assert mesh_stop.shutdown_phases(2.0) == (2.0, 2.0)                # restart_reattach's STOP_BOUND_S


def test_every_budget_that_holds_the_stop_still_fits(monkeypatch):
    monkeypatch.delitem(sys.modules, "monarch.config", raising=False)
    bound, actor = mesh_stop.PROC_STOP_DEFAULT_S, mesh_stop.ACTOR_STOP_S
    authority = mesh_safety.TOKEN_AUTHORITY_S
    # MeshHandle.shutdown's join: group + actor stop + proc stop + 30 s, which holds the 10 s lock.
    assert mesh_stop.SHUTDOWN_MARGIN_S >= LIFECYCLE_LOCK_S
    assert mesh_stop.shutdown_envelope_s(60.0) == 60 + actor + 60 + 30 == 155.0
    assert mesh_stop.shutdown_envelope_s(300.0) == 60 + actor + bound + 30 == 195.0 <= authority
    reconcile = mesh_stop.shutdown_envelope_s(mesh_stop.RECONCILE_GROUP_S, bound)
    assert reconcile == 30 + actor + bound + 30 == 165.0 <= authority
    # The config-change stop of the old fleet sits inside the same-key creation waiter.
    assert reconcile <= SAME_KEY_WAITER_S
    # Recycle: lock, group teardown, actor stop and proc stop inside its deadline.
    assert LIFECYCLE_LOCK_S + RECYCLE_GROUP_TEARDOWN_S + actor + bound == 175.0 <= RECYCLE_DEADLINE_S
    assert RECYCLE_DEADLINE_S <= authority
    # The failed-setup rollback's join holds its actor and proc stops.
    assert bound + actor + 30 >= actor + bound
    # The creation cleanup and the bring-up rollback wait the state limit plus
    # 5 s, 65 s at the defaults: past the longest confirmation measured on a lone
    # lost host (60.0 s, a stop sent 8 s after the loss) and its 52 s stop.
    cleanup = mesh_stop.cleanup_stop_s()
    assert cleanup == 60.0 + mesh_stop.CLEANUP_STATE_MARGIN_S == 65.0 <= bound
    assert cleanup > LONGEST_CLEANUP_CONFIRMATION_S > MEASURED_LOST_HOST_STOP_S
    # cli/cluster_smoke ends each handle with one recycle_detailed and no detach
    # after it, so the bound reaches the update's 600 s smoke timeout only
    # through the recycle's own deadline, beside a bring-up at its 180 s default;
    # a bring-up that fails at that deadline rolls back within the cleanup wait.
    assert 180.0 + RECYCLE_DEADLINE_S <= 600.0
    assert 180.0 + cleanup <= 600.0


@pytest.mark.parametrize("values,wait", [
    ({"get_proc_state_max_idle": "1m", "proc_stop_max_idle": "30s"}, 65.0),   # the defaults
    ({"get_proc_state_max_idle": "30s", "proc_stop_max_idle": "30s"}, 35.0),  # confirms 38.0 s after a loss
    ({"get_proc_state_max_idle": "20s", "proc_stop_max_idle": "10s"}, 25.0),
    ({"get_proc_state_max_idle": "1m 20s", "proc_stop_max_idle": "10s"}, 85.0),
    ({"get_proc_state_max_idle": "soon", "proc_stop_max_idle": "30s"}, 65.0),  # unreadable: the default
])
def test_the_cleanup_wait_follows_the_state_limit_and_never_exceeds_the_bound(monkeypatch, values, wait):
    """A stop sent to a lone lost host confirms when Monarch's state query of it ends."""
    _config(monkeypatch, values)
    assert mesh_stop.cleanup_stop_s() == wait <= mesh_stop.proc_stop_bound_s()
    _config(monkeypatch, {"get_proc_state_max_idle": "5m", "proc_stop_max_idle": "30s"})
    assert mesh_stop.cleanup_stop_s() == mesh_stop.proc_stop_bound_s() == mesh_stop.PROC_STOP_DEFAULT_S


def test_the_reconcile_join_holds_the_reconciles_whole_shutdown(monkeypatch):
    monkeypatch.delitem(sys.modules, "monarch.config", raising=False)
    envelope = mesh_stop.shutdown_envelope_s(mesh_stop.RECONCILE_GROUP_S, mesh_stop.proc_stop_bound_s())
    assert mesh_stop.reconcile_join_s() >= envelope
    tree = ast.parse((SRC / "mesh_helpers.py").read_text())
    joins = [ast.unparse(node) for node in ast.walk(tree)
             if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "join"
             and any(keyword.arg == "timeout" for keyword in node.keywords)]
    assert joins == ["thread.join(timeout=mesh_teardown.reconcile_join_s())"], joins


def _calls(relative: str, function: str, callee: str) -> list[str]:
    """The source of each call to ``callee`` (a name or an attribute) inside ``function``."""
    found = []
    for node in ast.walk(ast.parse((SRC / relative).read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and callee in (
                        getattr(call.func, "attr", None), getattr(call.func, "id", None)):
                    found.append(ast.unparse(call))
    return found


@pytest.mark.parametrize("relative,function,callee,expected", [
    # A supervision death's stop, the old fleet on a config change, the creation cleanup.
    ("mesh_teardown.py", "reconcile_stop", "shutdown",
     "stop_timeout_s=proc_stop_bound_s()"),
    ("mesh.py", "get_mesh", "shutdown", "stop_timeout_s=mesh_teardown.proc_stop_bound_s()"),
    ("mesh_factory.py", "reconcile_creation_cleanup", "shutdown",
     "stop_timeout_s=cleanup_stop_s()"),
    ("mesh_factory.py", "reconcile_creation_cleanup", "stop_procs", "cleanup_stop_s()"),
    ("mesh_recycle.py", "recycle_detailed_impl", "stop_with_reason", "proc_stop_bound_s()"),
    ("mesh_factory.py", "spawn_worker_fleet", "stop_procs", "cleanup_stop_s()"),
    ("mesh_teardown.py", "stop_failed_setup_procs", "stop_with_reason", "proc_stop_bound_s()"),
    ("mesh_teardown.py", "stop_failed_setup_procs", "run_off_loop", SETUP_ROLLBACK_JOIN),
])
def test_each_stop_after_a_death_waits_for_the_bound(relative, function, callee, expected):
    calls = _calls(relative, function, callee)
    assert calls, (relative, function, callee)
    assert all(expected in call for call in calls), calls


@pytest.mark.parametrize("relative,function", [
    ("mesh_teardown.py", "reconcile_stop"), ("mesh.py", "get_mesh"),
    ("mesh_factory.py", "reconcile_creation_cleanup")])
def test_the_group_teardown_keeps_its_30_s_where_it_had_it(relative, function):
    calls = _calls(relative, function, "shutdown")
    assert calls and all("timeout_s=30" in call or "timeout_s=RECONCILE_GROUP_S" in call
                         for call in calls), calls


def test_a_short_caller_bound_also_bounds_the_actor_stop():
    """restart_reattach's 2 s stops of a fleet whose loops restarted stay 2 s."""
    waits: list = []

    def stop_for(name):
        return lambda reason: SimpleNamespace(get=lambda timeout=None: waits.append((name, timeout)))

    handle = SimpleNamespace(workers=SimpleNamespace(stop=stop_for("workers")),
                             procs=SimpleNamespace(stop=stop_for("procs")))
    mesh_stop.stop_with_reason(handle, "dgx-monarch client detach", 2.0)
    assert waits == [("workers", 2.0), ("procs", 2.0)]


class _NativePanic(BaseException):
    """Stands in for pyo3's PanicException, a BaseException, raised at interpreter exit."""


@pytest.mark.parametrize("failure", [_NativePanic("actor_mesh.rs panicked"), RuntimeError("gone")])
def test_an_actor_stop_failure_or_panic_still_reaches_the_proc_stop(failure):
    calls: list = []

    def actor_stop(reason):
        raise failure

    handle = SimpleNamespace(workers=SimpleNamespace(stop=actor_stop),
                             procs=SimpleNamespace(stop=lambda reason: SimpleNamespace(
                                 get=lambda timeout=None: calls.append((reason, timeout)))))
    mesh_stop.stop_with_reason(handle, "dgx-monarch client detach", 15.0)
    assert calls == [("dgx-monarch client detach", 15.0)]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_in_the_actor_stop_still_propagates(interrupt):
    def actor_stop(reason):
        raise interrupt()

    handle = SimpleNamespace(workers=SimpleNamespace(stop=actor_stop),
                             procs=SimpleNamespace(stop=lambda reason: pytest.fail("proc stop after an interrupt")))
    with pytest.raises(interrupt):
        mesh_stop.stop_with_reason(handle, "dgx-monarch client detach", 15.0)


def test_the_liveness_proof_never_lengthens_a_stop():
    """A slot is used only when a whole fleet probe fits before the stop's own wait ends.

    The reconcile's stop after a death uses every slot; a restart heal's short
    stop uses none, so every sum this file pins holds with the proof in place.
    """
    from dgx_monarch import mesh_liveness, restart_reattach

    assert mesh_stop.PROOF_BUDGET_S == mesh_liveness.PROBE_FLEET_BUDGET_S
    reconcile_wait = mesh_stop.proc_stop_bound_s() + mesh_stop.ACTOR_STOP_S
    heal_wait = restart_reattach.STOP_BOUND_S + mesh_stop.ACTOR_STOP_S
    used = [at for at in mesh_stop.PROOF_AT_S if at + mesh_stop.PROOF_BUDGET_S <= reconcile_wait]
    assert used == list(mesh_stop.PROOF_AT_S)
    assert all(at + mesh_stop.PROOF_BUDGET_S > heal_wait for at in mesh_stop.PROOF_AT_S)

    class Ended:
        def wait(self, _timeout):
            return False

        def is_set(self):
            return False

    asked: list = []
    assert mesh_stop._wait_or_prove(Ended(), heal_wait, lambda: asked.append(1) or True) is False
    assert asked == []


def test_a_cancellation_in_the_actor_stop_still_propagates():
    import asyncio

    def actor_stop(reason):
        raise asyncio.CancelledError()

    handle = SimpleNamespace(workers=SimpleNamespace(stop=actor_stop),
                             procs=SimpleNamespace(stop=lambda reason: pytest.fail("proc stop after a cancel")))
    with pytest.raises(asyncio.CancelledError):
        mesh_stop.stop_with_reason(handle, "dgx-monarch client detach", 15.0)
