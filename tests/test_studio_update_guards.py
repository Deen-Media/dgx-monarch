"""Studio update guards against scratch git checkouts; no Worker, unit, GPU or network is used.

The end-to-end cases run Studio's real review and execution, the real frozen
controller preparation and child admission, and the real verified transaction
with its real repository and activity guards against a scratch checkout and a
local bare origin. Only Worker staging and lifecycle steps are scripted,
because they need real Worker hosts.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import tempfile
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import comfy_ports, legacy_update, update_activity, update_command
from dgx_monarch.cli.update_inspect import inspect_checkout
from dgx_monarch.cli.update_lock import UpdateLock
from dgx_monarch.cli.update_release import ReleaseMetadata
from dgx_monarch.cli.update_repository import git_command, repository_snapshot
from dgx_monarch.cli.update_target import resolve_target
from dgx_monarch.cli.update_types import (
    Certainty,
    ExactReadback,
    OperationResult,
    RepoSnapshot,
    StagedRelease,
    UpdatePreconditionError,
    UpdateResult,
)
from dgx_monarch.cli.update_worktree import remove_worktree
from dgx_monarch.config import render_cluster_toml
from dgx_monarch.studio import management_update, update_controller
from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.management_fields import parse

ORIGIN = "http://127.0.0.1:8188"
GIT_ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}
OK = OperationResult(Certainty.SUCCEEDED)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                           "-c", "commit.gpgsign=false", "-C", str(cwd), *args],
                          check=True, capture_output=True, text=True, env=GIT_ENV, timeout=60).stdout.strip()


def write_package(root: Path, version: str) -> None:
    (root / "src/dgx_monarch").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text('[project]\nname = "dgx-monarch"\ndependencies = ["torchmonarch==0.6.0"]\n')
    (root / "requirements.txt").write_text("torchmonarch==0.6.0\n")
    (root / "src/dgx_monarch/__init__.py").write_text(f'__version__ = "{version}"\nTORCHMONARCH_PIN = "0.6.0"\n')


@pytest.fixture
def scratch(tmp_path):
    """A driver checkout one commit behind a local bare origin, imported by a scratch ComfyUI."""
    origin, seed, checkout = tmp_path / "origin.git", tmp_path / "seed", tmp_path / "checkout"
    subprocess.run(["git", "init", "--bare", "-q", "-b", "main", str(origin)], check=True, env=GIT_ENV, timeout=60)
    subprocess.run(["git", "clone", "-q", str(origin), str(seed)], check=True, env=GIT_ENV, timeout=60, capture_output=True)
    git(seed, "checkout", "-q", "-b", "main")
    write_package(seed, "1.0.0")
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "prior")
    git(seed, "push", "-q", "origin", "main")
    subprocess.run(["git", "clone", "-q", str(origin), str(checkout)], check=True, env=GIT_ENV, timeout=60, capture_output=True)
    checkout.chmod(0o755)
    write_package(seed, "1.0.1")
    git(seed, "commit", "-q", "-am", "target")
    git(seed, "push", "-q", "origin", "main")
    comfy = tmp_path / "comfy"
    (comfy / "custom_nodes").mkdir(parents=True)
    (comfy / "custom_nodes/dgx-monarch").symlink_to(checkout)
    python = tmp_path / "driver-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("synthetic interpreter identity; never executed")
    python.chmod(0o755)
    config = tmp_path / "cluster.toml"
    config.write_text(render_cluster_toml([("worker-a", "127.0.0.1", 1)], "127.0.0.1", "single-node", "127.0.0.1",
                                          transport_security="trusted_fabric"))
    config.chmod(0o600)
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    identity = {"schema_version": 1, "state": "active", "config_digest": "driver-config",
                "config": {"executable": str(python), "source_files": [{"path": str(comfy / "main.py")}]}}
    return SimpleNamespace(root=tmp_path, checkout=checkout.resolve(), config=config, state=state, python=python,
                           identity=identity, prior=git(checkout, "rev-parse", "HEAD"), target=git(seed, "rev-parse", "HEAD"))


class OwnerDriver:
    """A scripted owner unit: Studio's stop and start change only its reported state."""

    def __init__(self, identity):
        self.identity, self.state, self.calls = identity, identity["state"], []

    def inspect(self):
        return {**self.identity, "state": self.state}

    async def perform(self, action, reviewed):
        assert reviewed == self.inspect()
        self.calls.append(action)
        self.state = "inactive" if action == "stop" else "active"
        return "succeeded", {"observed_identity": self.inspect()}


def closed_port() -> int:
    """A loopback port nothing listens on: a stopped driver refuses the connection.

    Another process could take the port between close and probe; that would make
    the guard tests flaky in one run, never silently pass a broken guard.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def stopped_driver_ports(monkeypatch):
    port = closed_port()
    # Local discovery finds only the stopped driver's port and no ComfyUI process,
    # whatever else runs on this machine. The probe itself stays real.
    monkeypatch.setattr(comfy_ports, "find_comfy_ports", lambda: {port: None})
    return port


@pytest.mark.parametrize(("host", "blockers"), [(None, ()), ("named", ("activity_unknown",))])
def test_real_activity_guard_passes_a_stopped_driver_only_without_an_explicit_host(stopped_driver_ports, host, blockers):
    named = f"127.0.0.1:{stopped_driver_ports}" if host else None
    assert comfy_ports.driver_probe(named, comfy_ports.find_comfy_ports()) == (None, True)
    snapshot = update_activity.activity_snapshot(named, 0, comfy_ports, lambda: 0)
    assert snapshot.blockers() == blockers


@pytest.mark.parametrize(("origin", "expected"), [
    ("http://127.0.0.1:8188", "127.0.0.1:8188"), ("https://studio-driver.invalid:8443", None)])
def test_legacy_gets_only_a_plain_http_origin(origin, expected):
    # comfy_ports dials http://host; an https origin would be probed under the wrong scheme.
    assert management_update.legacy_driver_host(origin) == expected


class ScratchOps:
    """The verified transaction's backend for a scratch checkout.

    Repository, target and activity observations and the driver commit are
    real. Worker staging, stop, start, Doctor and smoke are scripted.
    """

    def __init__(self, repo, config, *, driver_host, privileged_process_inspection, refuse_stage=False, log=None):
        self.repo, self.driver_host, self.refuse_stage, self.log = repo, driver_host, refuse_stage, log
        self.privileged_process_inspection = privileged_process_inspection
        self.root = Path(tempfile.mkdtemp(prefix="stage-", dir=repo.parent))
        self.staged_checkout: Path | None = None
        log.append(("driver_host", driver_host))

    def _git(self, *args, cwd=None, timeout=60):
        return git_command(self.repo, subprocess.run, *args, cwd=cwd, timeout=timeout)

    def resolve_target(self, target_ref):
        return resolve_target(target_ref, self._git)

    def repo_snapshot(self, target_sha):
        return repository_snapshot(target_sha, self._git)

    def activity_snapshot(self):
        return update_activity.activity_snapshot(self.driver_host, 0, comfy_ports, lambda: 0)

    def stage(self, target_sha):
        if self.refuse_stage:
            raise UpdatePreconditionError("release_layout_unsupported", "verified update could not prove a supported Worker installation")
        self.staged_checkout = self.root / "update-target"
        assert self._git("worktree", "add", "--detach", str(self.staged_checkout), target_sha).returncode == 0
        metadata, _ = inspect_checkout(self.staged_checkout, target_sha)
        return StagedRelease(metadata.target_sha, metadata.version, metadata.torchmonarch_pin, metadata.source_manifest,
                             metadata.dependency_manifest, True, True, True, True, 1, 1, 1)

    def recheck_dependencies(self):
        return None

    def _step(self, name):
        self.log.append(name)
        return OK

    def stop_workers(self):
        return self._step("stop_workers")

    def activate(self, staged):
        return self._step("activate")

    def start_target_workers(self, staged):
        return self._step("start_target_workers")

    def doctor(self):
        return self._step("doctor")

    def cluster_smoke(self, staged):
        return self._step("cluster_smoke")

    def exact_readback(self, staged):
        return ExactReadback(Certainty.SUCCEEDED, 1, 1, staged.target_sha, staged.version, staged.torchmonarch_pin,
                             staged.source_manifest)

    def commit_driver(self, staged):
        merged = self._git("merge", "--ff-only", staged.target_sha)
        return OK if merged.returncode == 0 else OperationResult(Certainty.UNKNOWN)

    def finalize(self, staged):
        return self._step("finalize")

    def cleanup_stage(self, staged):
        return remove_worktree(self.staged_checkout, self.root, self._git)


def wire_scratch_update(scratch, monkeypatch, *, refuse_stage=False):
    log: list = []
    monkeypatch.setattr(update_controller, "SOURCE", scratch.checkout / "src/dgx_monarch")
    monkeypatch.setattr(update_controller, "sys", SimpleNamespace(executable=str(scratch.python)))
    monkeypatch.setattr(update_controller.signal, "signal", lambda *args: None)
    monkeypatch.setattr(management_update, "UpdateLock", lambda path: UpdateLock(path, state_root=scratch.root / "locks"))
    monkeypatch.setattr(update_command, "SystemUpdateOps", lambda repo, config, **kwargs:
                        ScratchOps(repo, config, refuse_stage=refuse_stage, log=log, **kwargs))

    def run_in_process(prepared):
        # The same frozen manifest and child entrypoint, without a second interpreter.
        root = Path(prepared["path"])
        return update_controller.settled_result(root, update_controller.child(root, json.loads((root / "manifest.json").read_text())))
    monkeypatch.setattr(update_controller, "run", run_in_process)
    return log


@pytest.mark.parametrize("refuse_stage", [False, True])
def test_studio_verified_apply_on_scratch_checkout_and_bare_origin(scratch, monkeypatch, stopped_driver_ports, refuse_stage):
    log = wire_scratch_update(scratch, monkeypatch, refuse_stage=refuse_stage)
    driver = OwnerDriver(scratch.identity)
    updates = management_update.Updates(scratch.config, driver, ORIGIN, scratch.state)
    parameters = parse("update", {"repo": str(scratch.checkout), "stop_driver": True, "restore_driver": True})
    plan = updates.preview(parameters)
    assert plan["mode"] == "verified" and plan["release"]["target_sha"] == scratch.target
    assert plan["driver_checkout"]["resolved"] == str(scratch.checkout) and "activity" in plan["review_scope"]
    checkpoints: list = []
    state, receipt = asyncio.run(updates.execute(parameters, plan, checkpoints.append))
    head = git(scratch.checkout, "rev-parse", "HEAD")
    # The guard saw the stopped driver with no explicit host and let the transaction run.
    assert log[0] == ("driver_host", None)
    assert [step["name"] for step in receipt["update"]["steps"]][:3] == ["target_resolution", "repository_guard", "activity_guard"]
    if refuse_stage:
        # A definite refusal before any Worker change restores the reviewed driver.
        assert (state, receipt["settlement"], head) == ("failed", "refused", scratch.prior)
        assert receipt["update"]["status"] == "failed" and "worker_stop" not in json.dumps(receipt["update"]["steps"])
        assert "could not prove a supported Worker installation" in json.dumps(receipt["update"]["notes"])
    else:
        assert (state, receipt["settlement"], head) == ("succeeded", "succeeded", scratch.target)
        assert receipt["update"]["status"] == "succeeded" and "stop_workers" in log and "finalize" in log
    assert driver.calls == ["stop", "start"] and receipt["controller_cleaned"] is True
    assert not Path(receipt["controller"]["path"]).exists()
    assert [entry.get("phase") for entry in checkpoints if entry.get("phase")][:3] == [
        "controller_prepared", "driver_stop_requested", "verified_update_requested"]


def test_scratch_apply_with_a_named_host_is_refused_by_the_real_guard(scratch, monkeypatch, stopped_driver_ports):
    """The pre-fix shape: naming the stopped driver makes the activity guard refuse, so ComfyUI must not stay down."""
    log = wire_scratch_update(scratch, monkeypatch)
    scratch_ops = update_command.SystemUpdateOps
    monkeypatch.setattr(update_command, "SystemUpdateOps", lambda repo, config, **kwargs:
                        scratch_ops(repo, config, **{**kwargs, "driver_host": f"127.0.0.1:{stopped_driver_ports}"}))
    driver = OwnerDriver(scratch.identity)
    updates = management_update.Updates(scratch.config, driver, ORIGIN, scratch.state)
    parameters = parse("update", {"repo": str(scratch.checkout), "stop_driver": True, "restore_driver": True})
    state, receipt = asyncio.run(updates.execute(parameters, updates.preview(parameters), lambda value: None))
    assert log[0] == ("driver_host", f"127.0.0.1:{stopped_driver_ports}")
    assert (state, receipt["settlement"], receipt["result_code"]) == ("failed", "refused", "activity_unknown")
    assert driver.calls == ["stop", "start"] and git(scratch.checkout, "rev-parse", "HEAD") == scratch.prior


@pytest.fixture
def mocked_execution(tmp_path, monkeypatch):
    """Studio execution with every backend effect scripted, to pin the restore policy."""
    repo = tmp_path / "repo"
    repo.mkdir()
    comfy = tmp_path / "comfy"
    (comfy / "custom_nodes").mkdir(parents=True)
    (comfy / "custom_nodes/dgx-monarch").symlink_to(repo)
    identity = {"state": "active", "config_digest": "driver-config", "config": {"source_files": [{"path": str(comfy / "main.py")}]}}
    prior = ReleaseMetadata("a" * 40, "1.0", "0.6.0", "f" * 64, "d" * 64)
    reviewed = {"token": "token", "mode": "verified", "prior_commit": "a" * 40, "prior_release": asdict(prior),
                "checkout": {"path": "reviewed"}, "release": {"target_sha": "b" * 40}, "driver_identity": identity,
                "driver_checkout": {"link": str(comfy / "custom_nodes/dgx-monarch"), "resolved": str(repo.resolve())},
                "controller": {"executable": "bound"}}
    driver = OwnerDriver(identity)
    ops = SimpleNamespace(repo=repo.resolve(), repo_snapshot=lambda target: RepoSnapshot("a" * 40, True, True))
    updates = management_update.Updates(tmp_path / "cluster.toml", driver, ORIGIN, tmp_path)
    monkeypatch.setattr(updates, "operations", lambda parameters: ops)
    monkeypatch.setattr(management_update, "UpdateLock", lambda path: nullcontext(SimpleNamespace(fileno=lambda: 123)))
    monkeypatch.setattr(management_update, "checkout_identity", lambda path: reviewed["checkout"])
    monkeypatch.setattr(management_update, "inspect_checkout", lambda *args: (prior, ()))
    monkeypatch.setattr(update_controller, "binding", lambda driver: reviewed["controller"])
    monkeypatch.setattr(update_controller, "cleanup", lambda prepared: driver.calls.append("cleanup"))
    monkeypatch.setattr(update_controller, "stderr_tail", lambda prepared: ["refusing to update (release_slot_worker)"])
    hosts: list = []
    def prepare(directory, config, repo, plan, host, privileged, lock_fd):
        hosts.append(host)
        return {"private": "controller"}
    monkeypatch.setattr(update_controller, "prepare", prepare)
    return SimpleNamespace(updates=updates, reviewed=reviewed, driver=driver, hosts=hosts)


def verified(status, *names, code="refused"):
    return {"status": status, "steps": [{"name": name, "status": "succeeded"} for name in names], "notes": [code]}


@pytest.mark.parametrize(("mode", "result", "expected", "calls"), [
    ("verified", UpdateResult(0, "updated", "done", verified("succeeded", "activity_guard", "worker_stop")), "succeeded",
     ["stop", "start", "cleanup"]),
    ("verified", UpdateResult(2, "activity_unknown", "blocked", verified("failed", "target_resolution", "repository_guard")),
     "failed", ["stop", "start", "cleanup"]),
    ("verified", UpdateResult(1, "start_failed", "rolled back", verified("failed", "activity_guard", "worker_stop", "activation")),
     "failed", ["stop"]),
    ("verified", UpdateResult(1, "start_unknown", "unknown", verified("partial", "activity_guard", "worker_stop")), "unknown",
     ["stop"]),
    ("legacy", UpdateResult(2, "legacy_refused", "refused", {"mode": "legacy"}), "failed", ["stop", "start", "cleanup"]),
    ("legacy", UpdateResult(75, "legacy_refused", "refused", {"mode": "legacy"}), "failed", ["stop", "start", "cleanup"]),
    ("legacy", UpdateResult(1, "legacy_incomplete", "incomplete", {"mode": "legacy"}), "unknown", ["stop"]),
    ("legacy", UpdateResult(0, "legacy_updated", "done", {"mode": "legacy"}), "succeeded", ["stop", "start", "cleanup"]),
    ("verified", UpdateResult(2, "controller_refused", "changed", {"mutation_started": False}), "failed",
     ["stop", "start", "cleanup"]),
    ("verified", UpdateResult(1, "controller_unsettled", "missing", {}), "unknown", ["stop"]),
])
def test_restoration_follows_the_classified_receipt(mocked_execution, monkeypatch, mode, result, expected, calls):
    case = mocked_execution
    case.reviewed["mode"] = mode
    monkeypatch.setattr(update_controller, "run", lambda prepared: result)
    parameters = {"mode": mode, "stop_driver": True, "restore_driver": True, "privileged_process_inspection": False}
    state, receipt = asyncio.run(case.updates.execute(parameters, case.reviewed, lambda value: None))
    assert (state, case.driver.calls) == (expected, calls)
    assert case.hosts == [None if mode == "verified" else "127.0.0.1:8188"]
    # The child's own reason is kept, bounded and sanitized, in place of a constant message.
    assert receipt["controller_stderr"] == ["refusing to update (release_slot_worker)"]
    assert receipt["result_message"] == result.message and receipt["result_code"] == result.code
    if receipt["settlement"] == "settled_after_change":
        assert "Driver restoration was not attempted" in receipt["recovery"]


def test_refusal_before_any_change_is_failed_and_never_stops_the_driver(mocked_execution, monkeypatch):
    case = mocked_execution
    monkeypatch.setattr(management_update, "checkout_identity", lambda path: {"path": "changed"})
    monkeypatch.setattr(update_controller, "run", lambda prepared: pytest.fail("controller ran after a refusal"))
    parameters = {"mode": "verified", "stop_driver": True, "restore_driver": True, "privileged_process_inspection": False}
    state, receipt = asyncio.run(case.updates.execute(parameters, case.reviewed, lambda value: None))
    assert state == "failed" and receipt["refusal"]["code"] == "update_changed"
    assert case.driver.calls == [] and case.hosts == []


def test_unauthorized_stop_is_refused_before_the_controller_is_frozen(mocked_execution):
    case = mocked_execution
    parameters = {"mode": "verified", "stop_driver": False, "restore_driver": False, "privileged_process_inspection": False}
    state, receipt = asyncio.run(case.updates.execute(parameters, case.reviewed, lambda value: None))
    assert (state, receipt["refusal"]["code"]) == ("failed", "update_driver") and case.hosts == [] and case.driver.calls == []


@pytest.mark.parametrize(("slots", "code"), [(2, "update_release_slot"), (75, "update_release_slot_unknown"), (None, None)])
def test_legacy_review_runs_the_release_slot_check_before_any_fetch_or_stop(scratch, monkeypatch, slots, code):
    order: list = []
    monkeypatch.setattr(legacy_update, "_release_slot_refusal", lambda config: order.append(("slots", len(config.hosts))) or slots)
    monkeypatch.setattr(legacy_update, "_contract_edited", lambda repo: order.append("contract") or False)
    monkeypatch.setattr(legacy_update, "_fetch_candidate", lambda repo: order.append("fetch"))
    monkeypatch.setattr(update_controller, "SOURCE", scratch.checkout / "src/dgx_monarch")
    monkeypatch.setattr(management_update, "UpdateLock", lambda path: UpdateLock(path, state_root=scratch.root / "locks"))
    driver = OwnerDriver(scratch.identity)
    updates = management_update.Updates(scratch.config, driver, ORIGIN, scratch.state)
    parameters = parse("update", {"repo": str(scratch.checkout), "mode": "legacy", "stop_driver": True, "restore_driver": True})
    with pytest.raises(StudioError) as refused:
        updates.preview(parameters)
    assert driver.calls == []
    if code:
        assert refused.value.code == code and order == [("slots", 1)]
        assert "Nothing was fetched or changed" in str(refused.value)
    else:
        # The slot check passed, so review continued to the fetch, which then refused.
        assert order == [("slots", 1), "contract", "fetch"]
        assert "candidate could not be established" in str(refused.value)


def test_preview_refuses_a_companion_running_other_source_than_the_checkout(scratch, monkeypatch):
    monkeypatch.setattr(management_update, "UpdateLock", lambda path: UpdateLock(path, state_root=scratch.root / "locks"))
    # SOURCE stays the companion's own package, which differs from the scratch checkout.
    updates = management_update.Updates(scratch.config, OwnerDriver(scratch.identity), ORIGIN, scratch.state)
    parameters = parse("update", {"repo": str(scratch.checkout), "stop_driver": True})
    with pytest.raises(StudioError, match="differs from the selected checkout") as refused:
        updates.preview(parameters)
    assert refused.value.code == "update_controller_source"
