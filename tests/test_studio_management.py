"""Typed management tests never contact a Worker, GPU or service manager."""
from __future__ import annotations

import asyncio
import subprocess
import threading
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from dgx_monarch.cli.setup_config_lock import SetupConfigLock
from dgx_monarch.cli.update_lock import UpdateLock, UpdateLockUnavailable
from dgx_monarch.cli.update_release import ReleaseMetadata
from dgx_monarch.cli.update_types import RepoSnapshot, StagedRelease, UpdateResult
from dgx_monarch.config import load_cluster_config
from dgx_monarch.studio import management_config, management_sol, management_update
from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.management import Management
from dgx_monarch.studio.management_fields import FIELDS, parse
from dgx_monarch.studio.operations import READ_ONLY, Operations
from dgx_monarch.studio.store import Store


@pytest.fixture(autouse=True)
def no_live_commands(monkeypatch, tmp_path):
    def denied(*args, **kwargs):
        raise AssertionError("No live subprocess is permitted")
    monkeypatch.setattr(subprocess, "run", denied)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", denied)
    monkeypatch.setattr(management_config, "SetupConfigLock", lambda path: SetupConfigLock(path, state_root=tmp_path / "locks"))
    monkeypatch.setattr(management_update, "UpdateLock", lambda path: nullcontext(SimpleNamespace(fileno=lambda: 123)))


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state")
    yield value
    value.close()


class Driver:
    origin = "http://127.0.0.1:8188"

    async def request(self, method, path, data=None):
        assert (method, path) == ("GET", "/dgxm/telemetry")
        return {"render": {"active": False}, "workers": [], "driver_host": {"host": "fake"}}


def initial_parameters():
    return {"hosts": [{"name": "fake", "fabric_ip": "127.0.0.1", "gpus": 1}],
            "client_ip": "127.0.0.1", "fabric_profile": "single-node", "transport_security": "trusted_fabric"}


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "cluster.toml"
    params = parse("init", initial_parameters())
    plan = management_config.preview("init", path, params)
    status, _ = management_config.execute("init", path, params, plan)
    assert status == "succeeded"
    return path


def test_all_management_actions_have_typed_fields_and_permission_gate(store, config):
    disabled = {row["id"]: row for row in Operations(store, Driver(), config=config).capabilities()}
    enabled = {row["id"]: row for row in Operations(store, Driver(), config=config, allow_management=True).capabilities()}
    for action in FIELDS:
        assert enabled[action]["available"] and enabled[action]["implementation"] == "typed_adapter"
        assert bool(enabled[action]["parameters"]) is (action != "diagnostics.capabilities")
        if action not in READ_ONLY:
            assert not disabled[action]["available"]


@pytest.mark.parametrize(("action", "values"), [
    ("top.record", {"samples": True}), ("top.record", {"interval_s": float("nan")}),
    ("top.record", {"samples": 601}), ("sol-install", {"allow_change": "torch"}),
    ("setup", {**initial_parameters(), "start_workers": "yes"}),
    ("init", {**initial_parameters(), "arbitrary_argv": ["sh"]}),
])
def test_fields_refuse_untyped_or_unknown_inputs(action, values):
    with pytest.raises(StudioError):
        parse(action, values)


def test_init_review_is_exact_and_has_no_implicit_worker_start(tmp_path, monkeypatch):
    path = tmp_path / "new.toml"
    parameters = parse("init", initial_parameters())
    plan = management_config.preview("init", path, parameters)
    assert not path.exists()
    assert "no Worker" in plan["scope"]
    path.write_text("foreign replacement")
    with pytest.raises(StudioError, match="identity changed"):
        management_config.execute("init", path, parameters, plan)
    assert path.read_text() == "foreign replacement"


def test_launcher_rejects_changed_executable_after_review(config, tmp_path):
    executable = tmp_path / "dgxm-real"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    parameters = {"executable": str(executable), "destination": str(tmp_path / "bin/dgxm")}
    plan = management_config.preview("launcher", config, parameters)
    executable.write_text("#!/bin/sh\nexit 1\n")
    with pytest.raises(StudioError, match="identity changed"):
        management_config.execute("launcher", config, parameters, plan)
    assert not (tmp_path / "bin").exists()


def test_setup_verify_requires_paired_explicit_authority(config):
    values = parse("setup", {**initial_parameters(), "verify": True})
    with pytest.raises(StudioError, match="paired"):
        management_config.setup_request(config, values)


def test_setup_rebuilds_the_reviewed_plan_before_mutation(config, monkeypatch):
    parameters = parse("setup", initial_parameters())
    monkeypatch.setattr(management_config, "build_setup_plan", lambda request: object())
    monkeypatch.setattr(management_config, "setup_view", lambda plan: {"probe": "changed"})
    with pytest.raises(StudioError, match="plan changed"):
        management_config.execute("setup", config, parameters, {"probe": "old"})


@pytest.mark.parametrize(("removable", "stopped", "expected"), [(False, True, []), (True, False, ["down"])])
def test_uninstall_stops_on_unproven_ownership_or_stop(config, monkeypatch, removable, stopped, expected):
    calls = []
    monkeypatch.setattr(management_config.lifecycle, "units_are_removable", lambda config: removable)
    monkeypatch.setattr(management_config.lifecycle, "down", lambda config: calls.append("down") or stopped)
    monkeypatch.setattr(management_config.lifecycle, "uninstall_systemd", lambda config: calls.append("uninstall") or True)
    plan = {"removable": True, "snapshot": management_config.snapshot(config)}
    if not removable:
        with pytest.raises(StudioError, match="ownership"):
            management_config.execute("uninstall", config, {"remove_config": True}, plan)
    else:
        assert management_config.execute("uninstall", config, {"remove_config": True}, plan)[0] == "unknown"
    assert calls == expected and config.exists()


def test_uninstall_config_removal_is_separate_and_exact(config, monkeypatch):
    for name in ("units_are_removable", "down", "uninstall_systemd"):
        monkeypatch.setattr(management_config.lifecycle, name, lambda config: True)
    plan = management_config.preview("uninstall", config, {"remove_config": False})
    state, receipt = management_config.execute("uninstall", config, {"remove_config": False}, plan)
    assert state == "succeeded" and receipt["config_removed"] is False and config.exists()
    state, receipt = management_config.execute("uninstall", config, {"remove_config": True}, plan)
    assert state == "succeeded" and receipt["config_removed"] is True and not config.exists()


@pytest.mark.parametrize("package", ["torch", "torchmonarch", "bad/name"])
def test_sol_never_permits_protected_or_invalid_distributions(package):
    with pytest.raises(StudioError):
        management_sol.allowed_names([package])


def test_sol_exact_plan_and_partial_apply_never_restart(config, monkeypatch):
    calls = []
    plan = {"digest": "a" * 64, "allowed": [], "changes": []}
    def host(config, host, action, token, allowed, expected=""):
        calls.append((action, expected))
        return None if action == "apply" else {"status": "ok", "plan": deepcopy(plan)}
    monkeypatch.setattr(management_sol.sol_install, "_host", host)
    parameters = parse("sol-install", {"idle_maintenance": True})
    cluster = load_cluster_config(config)
    reviewed = management_sol.preview("sol-install", cluster, parameters)
    state, receipt = management_sol.execute("sol-install", cluster, parameters, reviewed)
    assert state == "unknown" and receipt["incomplete_host"] == 0
    assert calls == [("plan", ""), ("read", ""), ("apply", "a" * 64)]


def test_sol_maintenance_requires_explicit_idle_authority(config):
    with pytest.raises(StudioError, match="idle-maintenance"):
        management_sol.execute("sol-smoke", load_cluster_config(config), {"idle_maintenance": False}, {})


def test_incomplete_sol_plan_retains_staging_token_and_cannot_apply(config, monkeypatch):
    monkeypatch.setattr(management_sol.sol_install, "_host", lambda *args: None)
    parameters = parse("sol-install", {"idle_maintenance": True})
    plan = management_sol.preview("sol-install", load_cluster_config(config), parameters)
    assert plan["ready"] is False and len(plan["token"]) == 32
    assert plan["hosts"] == [{"ordinal": 0, "error": "observation_unknown"}]
    with pytest.raises(StudioError, match="incomplete"):
        management_sol.execute("sol-install", load_cluster_config(config), parameters, plan)


def test_record_and_replay_preserve_tui_sample_format_and_detect_tamper(store):
    manager = Management(store, Driver(), None, None)
    store.create("operations", {"id": "record-op", "state": "applying"})
    state, receipt = asyncio.run(manager.record({"samples": 1, "interval_s": 0.1}, "record-op"))
    assert state == "succeeded"
    _, plan = manager.preview("top.replay", {"recording_id": receipt["id"]})
    state, replay = manager.replay(plan)
    assert state == "succeeded" and replay["replay"] is True
    assert replay["samples"][0]["hosts"]["fake"]["source"] == "driver"
    path = store.directory / "recordings" / f"{receipt['id']}.jsonl"
    path.write_text("{}\n")
    with pytest.raises(StudioError, match="changed"):
        manager.replay(plan)


def test_recording_cancellation_retains_replayable_partial_samples(store):
    async def scenario():
        waiting = asyncio.Event()
        class PausingDriver(Driver):
            calls = 0
            async def request(self, *args):
                self.calls += 1
                if self.calls == 2:
                    waiting.set()
                    await asyncio.Event().wait()
                return await super().request(*args)
        manager = Management(store, PausingDriver(), None, None)
        store.create("operations", {"id": "partial", "state": "applying"})
        recording = asyncio.create_task(manager.record({"samples": 2, "interval_s": 0}, "partial"))
        await asyncio.wait_for(waiting.wait(), 2)
        recording.cancel()
        with pytest.raises(asyncio.CancelledError):
            await recording
        receipt = store.get("operations", "partial")["receipt"]
        assert receipt["samples"] == 1 and receipt["complete"] is False
        _, plan = manager.preview("top.replay", {"recording_id": receipt["recording_id"]})
        assert len(manager.replay(plan)[1]["samples"]) == 1
    asyncio.run(scenario())


def test_cancellation_waits_for_owned_effect_before_releasing_caller():
    async def scenario():
        started, finish, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def effect():
            started.set()
            await finish.wait()
            settled.set()
        task = asyncio.create_task(management_update.settle(effect()))
        await started.wait()
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert settled.is_set()
    asyncio.run(scenario())


def test_repeated_cancellation_keeps_real_update_lock_until_thread_settles(tmp_path, store, monkeypatch):
    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release, finished = threading.Event(), threading.Event()
        retained = []
        lock_root = tmp_path / "real-locks"
        ops = Operations(store, Driver(), allow_management=True)
        operation = store.create("operations", {"id": "cancelled-twice", "action": "top.record", "state": "applying",
                                                 "driver_service": None, "config_fingerprint": None})
        def effect():
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5), "test failed to release the synthetic mutator"
            retained.append({"settled": True})
            store.update("operations", operation["id"], {"receipt": {"settled": True}})
            finished.set()
        async def execute_owned_effect(_operation):
            with UpdateLock(tmp_path, state_root=lock_root):
                await management_update.blocking(effect)
            return "succeeded", {"settled": True}
        async def cancellation_delivered():
            barrier = loop.create_future()
            loop.call_soon(barrier.set_result, None)
            await barrier
        monkeypatch.setattr(ops.management, "execute", execute_owned_effect)
        task = asyncio.create_task(ops.execute(operation))
        await asyncio.wait_for(started.wait(), 2)
        try:
            for _ in range(2):
                task.cancel()
                await cancellation_delivered()
                assert not task.done() and not finished.is_set()
                with pytest.raises(UpdateLockUnavailable):
                    with UpdateLock(tmp_path, state_root=lock_root):
                        pytest.fail("a second owner acquired the live mutator's update lock")
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert finished.is_set() and retained == [{"settled": True}]
        result = store.get("operations", operation["id"])
        assert result["state"] == "unknown" and result["receipt"]["settled"] is True
        assert result["receipt"]["error"] == "operation_interrupted"
        with UpdateLock(tmp_path, state_root=lock_root):
            pass
    asyncio.run(scenario())


def update_plan():
    staged = StagedRelease("b" * 40, "1.0", "0.6.0", "c" * 64, "d" * 64, True, True, True, True, 1, 1, 1)
    expected = {key: value for key, value in asdict(staged).items() if key in
                {"target_sha", "version", "torchmonarch_pin", "source_manifest", "dependency_manifest"}}
    prior = ReleaseMetadata("a" * 40, "1.0", "0.6.0", "f" * 64, "d" * 64)
    return staged, {"token": "test-token", "prior_commit": "a" * 40, "prior_release": asdict(prior),
                    "checkout": {"path": "reviewed", "inode": 1}, "release": expected,
                    "driver_identity": {"state": "active", "config_digest": "driver-config"}, "controller": {"exact": "bound"}}


def test_update_confirmation_compares_all_reviewed_release_hashes():
    staged, plan = update_plan()
    proxy = management_update.ReviewedUpdate(object(), plan["release"])
    assert proxy.confirm(staged)
    plan["release"]["source_manifest"] = "e" * 64
    assert not proxy.confirm(staged)
    with pytest.raises(StudioError, match="target differs"):
        proxy.resolve_target("main")


@pytest.mark.parametrize(("stop_ok", "update_ok", "environment_drift", "actions"), [
    (False, True, False, ["stop"]), (True, False, False, ["stop", "update"]),
    (True, True, False, ["stop", "update", "start"]), (True, True, True, ["stop", "update"]),
])
def test_update_restores_only_after_exact_success(store, config, monkeypatch, stop_ok, update_ok, environment_drift, actions):
    _, reviewed = update_plan()
    calls, checkpoints = [], []
    class OwnerDriver:
        async def perform(self, action, identity):
            calls.append(action)
            state = "succeeded" if stop_ok else "unknown"
            return state, {"observed_identity": {"state": "inactive", "config_digest": "driver-config"}}
        def inspect(self):
            return {"state": "inactive" if "stop" in calls else "active", "config_digest": "driver-config"}
    ops = SimpleNamespace(repo=store.directory, repo_snapshot=lambda target: RepoSnapshot("a" * 40, True, True))
    manager = management_update.Updates(config, OwnerDriver(), Driver.origin, store.directory)
    monkeypatch.setattr(manager, "operations", lambda parameters: ops)
    monkeypatch.setattr(management_update, "checkout_identity", lambda path: reviewed["checkout"])
    monkeypatch.setattr(management_update, "inspect_checkout", lambda *args: (ReleaseMetadata(**reviewed["prior_release"]), ()))
    monkeypatch.setattr(management_update.update_controller, "binding",
                        lambda driver: {"exact": "changed"} if environment_drift and "update" in calls else reviewed["controller"])
    monkeypatch.setattr(management_update.update_controller, "prepare", lambda *args: {"private": "controller"})
    monkeypatch.setattr(management_update.update_controller, "cleanup", lambda *args: None)
    def execute(prepared):
        calls.append("update")
        assert prepared == {"private": "controller"}
        return UpdateResult(0 if update_ok else 1, "updated" if update_ok else "stop_unknown", "result", {})
    monkeypatch.setattr(management_update.update_controller, "run", execute)
    state, _ = asyncio.run(manager.execute({"stop_driver": True, "restore_driver": True,
                                           "privileged_process_inspection": False}, reviewed, checkpoints.append))
    assert calls == actions and state == ("succeeded" if "start" in actions else "unknown")
    assert checkpoints[0]["phase"] == "controller_prepared"
    assert checkpoints[1]["phase"] == "driver_stop_requested"


def test_update_accepts_selected_checkout_independent_of_companion_import(config, tmp_path, monkeypatch):
    repo = tmp_path / "separate-source-checkout"
    (repo / "src/dgx_monarch").mkdir(parents=True)
    repo.chmod(0o755)
    (repo / "src/dgx_monarch/__init__.py").write_text("")
    (repo / "pyproject.toml").write_text('[project]\nname="dgx-monarch"\n')
    called = []
    monkeypatch.setattr(management_update, "SystemUpdateOps", lambda *args, **kwargs: called.append((args, kwargs)) or "backend")
    manager = management_update.Updates(config, None, Driver.origin, tmp_path)
    assert manager.operations({"repo": str(repo), "privileged_process_inspection": False}) == "backend"
    assert called[0][0][0] == repo
    assert called[0][1]["driver_host"] == "127.0.0.1:8188"


def test_update_rejects_unrelated_or_writable_checkout(tmp_path):
    repo = tmp_path / "other"
    repo.mkdir()
    repo.chmod(0o755)
    (repo / "pyproject.toml").write_text('[project]\nname="unrelated"\n')
    with pytest.raises(StudioError, match="dgx-monarch source"):
        management_update.checkout_identity(repo)
    repo.chmod(0o777)
    with pytest.raises(StudioError, match="owner-controlled"):
        management_update.checkout_identity(repo)


def test_update_checkout_owner_only_group_write_matches_existing_trust_rule(tmp_path, monkeypatch):
    repo = tmp_path / "source"
    (repo / "src/dgx_monarch").mkdir(parents=True)
    (repo / "src/dgx_monarch/__init__.py").write_text("")
    (repo / "pyproject.toml").write_text('[project]\nname="dgx-monarch"\n')
    repo.chmod(0o775)
    group_id = repo.stat().st_gid
    owner = SimpleNamespace(pw_name="owner", pw_gid=group_id)
    monkeypatch.setattr(management_update.pwd, "getpwuid", lambda uid: owner)
    monkeypatch.setattr(management_update.pwd, "getpwall", lambda: [owner])
    group = SimpleNamespace(gr_mem=["owner"])
    monkeypatch.setattr(management_update.grp, "getgrgid", lambda gid: group)
    assert management_update.checkout_identity(repo)["path"] == str(repo)
    group.gr_mem.append("other-owner")
    with pytest.raises(StudioError, match="owner-controlled"):
        management_update.checkout_identity(repo)
    group.gr_mem.clear()
    monkeypatch.setattr(management_update.pwd, "getpwall", lambda: [owner, SimpleNamespace(pw_name="other-owner", pw_gid=group_id)])
    with pytest.raises(StudioError, match="owner-controlled"):
        management_update.checkout_identity(repo)


@pytest.mark.parametrize("field", ["ssh_user", "comfy_dir"])
def test_init_does_not_silently_ignore_per_host_setup_fields(tmp_path, field):
    values = initial_parameters()
    values["hosts"][0][field] = "explicit-selection"
    with pytest.raises(StudioError, match="guided setup"):
        management_config.preview("init", tmp_path / "cluster.toml", parse("init", values))


def test_management_submission_replays_without_reexecuting(store, config, monkeypatch):
    ops = Operations(store, Driver(), config=config, allow_management=True)
    monkeypatch.setattr(ops.management, "preview", lambda action, params: (params, {"bound": "review"}))
    calls = []
    async def execute(operation):
        calls.append(operation["id"])
        return "succeeded", {"source": "fake_typed_adapter"}
    monkeypatch.setattr(ops.management, "execute", execute)
    async def run():
        review = ops.preview({"action": "launcher", "parameters": {}})
        request = {"digest": review["digest"], "idempotency_key": "same", "confirmed": True}
        first = ops.apply(review["id"], request)
        await asyncio.gather(*ops.tasks)
        second = ops.apply(review["id"], request)
        assert first["id"] == second["id"] and second["state"] == "succeeded"
    asyncio.run(run())
    assert len(calls) == 1


def test_failed_final_receipt_preserves_intermediate_management_evidence(store, config, monkeypatch):
    ops = Operations(store, Driver(), config=config, allow_management=True)
    monkeypatch.setattr(ops.management, "preview", lambda action, params: (params, {"bound": "review"}))
    async def execute(operation):
        store.update("operations", operation["id"], {"receipt": {"driver_stop": "verified", "update": "unknown"}})
        raise OSError("receipt failure")
    monkeypatch.setattr(ops.management, "execute", execute)
    async def run():
        review = ops.preview({"action": "launcher", "parameters": {}})
        job = ops.apply(review["id"], {"digest": review["digest"], "idempotency_key": "failed-receipt", "confirmed": True})
        await asyncio.gather(*ops.tasks)
        result = store.get("operations", job["id"])
        assert result["state"] == "unknown" and result["receipt"]["driver_stop"] == "verified"
        assert result["receipt"]["update"] == "unknown"
    asyncio.run(run())
