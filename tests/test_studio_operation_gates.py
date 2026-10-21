"""Server-side gates on Studio management: confirmation, authority and receipts.

The browser checkbox is a convenience; these tests pin the server refusals
that stand in for `--yes`, plus the receipt each spawned command produces.
No real management command runs: spawned children are fakes, except one real
`dgxm version` child that proves the launch directory cannot shadow the package.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

import dgx_monarch
from dgx_monarch import __version__
from dgx_monarch.cli.probe_certainty import UNKNOWN_EXIT
from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.experiments import GridExperiments
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.management_fields import COMMAND_FIELDS, parse
from dgx_monarch.studio.operations import (
    COMMANDS,
    DRIVER_ACTIONS,
    FIELDS,
    HTTP_ACTIONS,
    READ_ONLY,
    Operations,
    fingerprint,
)
from dgx_monarch.studio.store import Store
from test_studio_integration import companion

# Reviewed 2026-10-20: these actions only read state, so they run without
# --allow-management and without confirmation. A diagnostics request that
# names an output path is the one exception, tested below.
REVIEWED_READ_ONLY = frozenset({
    "version", "status", "doctor", "reap.preview", "gate.inspect", "gate.repair.preview", "top.replay",
    "diagnostics", "diagnostics.capabilities",
})
MUTATING = sorted((set(COMMANDS) | set(HTTP_ACTIONS) | set(DRIVER_ACTIONS) | set(FIELDS)) - READ_ONLY)
UNCONFIRMED = [pytest.param({}, id="absent"), pytest.param({"confirmed": False}, id="false"),
               pytest.param({"confirmed": "true"}, id="string"), pytest.param({"confirmed": 1}, id="one")]
# Every command Studio spawns rather than running in-process.
SPAWNED = sorted(set(COMMANDS) - set(FIELDS))


class Driver:
    origin = "http://127.0.0.1:8188"

    def __init__(self):
        self.calls = []

    async def request(self, method, path, data=None):
        self.calls.append((method, path))
        return {"ok": True}


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state")
    yield value
    value.close()


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "cluster.toml"
    path.write_text("# reviewed configuration\n")
    return path


def gated(store, config, action, parameters=None):
    """An Operations whose review for this action is available with every flag set."""
    ops = Operations(store, Driver(), config=config, allow_management=True, driver_service="comfy.service")
    state = "inactive" if action == "driver.start" else "active"
    ops.driver_binding = SimpleNamespace(inspect=lambda: {"state": state})
    ops._management = SimpleNamespace(preview=lambda name, values: (values, {"ready": True, "synthetic": name}))
    remote = {"gate.last": {"ready": True, "source_prompt_id": "newest"},
              "consent.apply": {"pending": [{"key": "consent-key", "id": "p-1", "title": "Pending consent"}]}}.get(action)
    review = ops.preview({"action": action, "parameters": parameters or {}}, remote)
    return ops, review


def unconsumed(store, review):
    stored = store.get("operations", review["id"])
    assert stored["state"] == "review" and "consumed_by" not in stored
    assert [row["id"] for row in store.all("operations")] == [review["id"]]


CASES = [pytest.param(action, None, id=action) for action in MUTATING]
CASES.append(pytest.param("diagnostics", {"output": "capture.json"}, id="diagnostics-output"))


@pytest.mark.parametrize("variant", UNCONFIRMED)
@pytest.mark.parametrize(("action", "parameters"), CASES)
def test_every_mutating_review_refuses_apply_without_literal_confirmation(store, config, tmp_path, action, parameters, variant):
    if parameters:
        parameters = {"output": str(tmp_path / parameters["output"])}

    async def scenario():
        ops, review = gated(store, config, action, parameters)
        assert review["available"], review["reason"]
        assert review["requires_confirmation"] is True
        request = {"digest": review["digest"], "idempotency_key": "unconfirmed", **variant}
        with pytest.raises(StudioError) as caught:
            ops.apply(review["id"], request)
        assert (caught.value.code, caught.value.status) == ("confirmation", 409)
        assert store.request("operations", "unconfirmed", {**request, "operation_id": review["id"]}) is None
        assert not ops.tasks
        unconsumed(store, review)

        # The same review passes with a literal true, so confirmation alone refused it.
        started = []

        async def execute(operation):
            started.append(operation["id"])
        ops.execute = execute
        accepted = ops.apply(review["id"], {"digest": review["digest"], "idempotency_key": "confirmed", "confirmed": True})
        await asyncio.gather(*ops.tasks)
        assert started == [accepted["id"]]
    asyncio.run(scenario())


def test_every_mutating_action_is_covered_by_the_confirmation_test():
    assert len(MUTATING) >= 20
    assert {"driver.stop", "update", "setup", "actors.recycle", "consent.apply", "doctor.repair"} <= set(MUTATING)


def test_read_only_actions_are_the_reviewed_set():
    assert READ_ONLY == REVIEWED_READ_ONLY


def test_management_http_apply_refuses_unconfirmed_review(tmp_path, monkeypatch, config):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            system.operations.allow_management = True
            system.operations.config = config
            review = await system.request("POST", "/operations/preview", json={"action": "workers.down"})
            assert review["available"] and review["requires_confirmation"]
            request = {"digest": review["digest"], "idempotency_key": "http-unconfirmed"}
            refused = await system.request("POST", f"/operations/{review['id']}/apply", expected=409, json=request)
            assert refused["error"]["code"] == "confirmation"
            assert system.store.request("operations", "http-unconfirmed", {**request, "operation_id": review["id"]}) is None
            assert not system.operations.tasks
            assert system.store.get("operations", review["id"]).get("consumed_by") is None
    asyncio.run(scenario())


@pytest.mark.parametrize("variant", UNCONFIRMED)
def test_job_retry_refuses_without_literal_confirmation(store, variant):
    jobs = Jobs(store, None)
    jobs.launch = lambda coroutine: (coroutine.close(), pytest.fail("an unconfirmed retry dispatched"))
    store.create("jobs", {"id": "failed-job", "state": "failed", "submission": {"prompt": {}}})
    request = {"idempotency_key": "retry-unconfirmed", **variant}
    with pytest.raises(StudioError) as caught:
        jobs.retry("failed-job", request)
    assert (caught.value.code, caught.value.status) == ("retry_scope", 409)
    assert store.request("jobs", "retry-unconfirmed", {**request, "retry_of": "failed-job"}) is None
    assert [row["id"] for row in store.all("jobs")] == ["failed-job"]


@pytest.mark.parametrize("variant", UNCONFIRMED)
def test_grid_retry_refuses_without_literal_confirmation(store, variant):
    jobs = Jobs(store, None)
    jobs.launch = lambda coroutine: (coroutine.close(), pytest.fail("an unconfirmed grid retry dispatched"))
    store.create("jobs", {"id": "cell-job", "state": "failed", "prompt_id": "p", "submission": {"prompt": {}}})
    cells = [{"index": 0, "coordinates": {"x": 1}, "job_id": "cell-job", "attempts": ["cell-job"], "excluded": False}]
    store.create("grid_runs", {"id": "grid", "state": "completed_with_failures", "created_at": 1,
                               "submission": {"request": {"draft": {"workflow": {}}}}, "cells": cells})
    request = {"idempotency_key": "grid-unconfirmed", "indices": [0], **variant}
    with pytest.raises(StudioError) as caught:
        GridExperiments(jobs).retry("grid", request)
    assert (caught.value.code, caught.value.status) == ("grid_retry", 409)
    assert store.request("grid_retries", "grid-unconfirmed", {**request, "grid_id": "grid"}) is None
    assert [row["id"] for row in store.all("jobs")] == ["cell-job"]
    assert store.get("grid_runs", "grid")["cells"] == cells


def test_diagnostics_output_needs_management_authority(store, tmp_path):
    ops = Operations(store, Driver())
    capture = ops.preview({"action": "diagnostics", "parameters": {}})
    assert capture["available"] and capture["requires_confirmation"] is False
    write = ops.preview({"action": "diagnostics", "parameters": {"output": str(tmp_path / "capture.json")}})
    assert write["available"] is False and "--allow-management" in write["reason"]
    assert write["requires_confirmation"] is True and write["adapter_plan"] is None
    request = {"digest": write["digest"], "idempotency_key": "write", "confirmed": True}
    with pytest.raises(StudioError) as caught:
        ops.apply(write["id"], request)
    assert (caught.value.code, caught.value.status) == ("operation_unavailable", 409)
    assert store.request("operations", "write", {**request, "operation_id": write["id"]}) is None
    row = next(row for row in ops.capabilities() if row["id"] == "diagnostics")
    assert "--allow-management" in row["description"]


def test_diagnostics_write_authority_is_rechecked_before_execution(store, tmp_path):
    async def scenario():
        ops = Operations(store, Driver(), allow_management=True)
        review = ops.preview({"action": "diagnostics", "parameters": {"output": str(tmp_path / "capture.json")}})
        assert review["available"] and review["requires_confirmation"]
        await ops.execution_lock.acquire()
        accepted = ops.apply(review["id"], {"digest": review["digest"], "idempotency_key": "w", "confirmed": True})
        ops.allow_management = False
        ops.execution_lock.release()
        await asyncio.gather(*ops.tasks)
        result = store.get("operations", accepted["id"])
        assert result["state"] == "unknown" and result["receipt"]["error"] == "operation_unavailable"
        assert not (tmp_path / "capture.json").exists()
    asyncio.run(scenario())


def fake_child(code, output=b""):
    calls = []

    async def spawn(*argv, **kwargs):
        calls.append(argv)
        stream = asyncio.StreamReader()
        stream.feed_data(output)
        stream.feed_eof()

        async def wait():
            return code
        return SimpleNamespace(pid=4321, stdout=stream, wait=wait)
    return spawn, calls


@pytest.mark.parametrize(("code", "state"), [
    (0, "succeeded"), (1, "failed"), (2, "failed"), (3, "failed"), (UNKNOWN_EXIT, "unknown"), (-9, "unknown"),
])
@pytest.mark.parametrize("action", SPAWNED)
def test_each_spawned_command_receipt_keeps_unknown_distinct_from_failure(store, config, monkeypatch, action, code, state):
    spawn, calls = fake_child(code, f"dgx-monarch {__version__} (torchmonarch pin 0.0.1)\n".encode())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    ops = Operations(store, Driver(), config=config, allow_management=True)
    parameters = parse(action, {}) if action in COMMAND_FIELDS else {}
    operation = store.create("operations", {"id": "spawned", "action": action, "state": "applying", "parameters": parameters,
                                            "driver_service": None, "config_fingerprint": fingerprint(ops.config)})
    asyncio.run(ops.execute(operation))
    result = store.get("operations", "spawned")
    assert (result["state"], result["receipt"]["exit_code"]) == (state, code)
    assert result["receipt"]["source"] == "existing_cli"
    assert calls[0][:6] == (sys.executable, "-P", "-s", "-B", "-m", "dgx_monarch")


@pytest.mark.parametrize("action", ["doctor", "reap.preview", "reap.apply"])
def test_local_adapter_receipt_maps_unknown_exit(store, monkeypatch, action):
    spawn, calls = fake_child(UNKNOWN_EXIT)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    ops = Operations(store, Driver(), allow_management=True)
    operation = store.create("operations", {"id": "local", "action": action, "state": "applying", "parameters": parse(action, {}),
                                            "driver_service": None, "config_fingerprint": None})
    asyncio.run(ops.execute(operation))
    assert store.get("operations", "local")["state"] == "unknown"
    assert calls[0][4:6] == ("-m", "dgx_monarch.studio.local_commands")


def test_version_receipt_comes_from_the_child(store, monkeypatch):
    spawn, _ = fake_child(0, b"some banner\ndgx-monarch 7.7.7 (torchmonarch pin 9.9.9)\n")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    ops = Operations(store, Driver())
    store.create("operations", {"id": "version", "action": "version"})
    state, receipt = asyncio.run(ops.execute_owned({"id": "version", "action": "version", "parameters": {}}))
    assert state == "succeeded"
    assert (receipt["version"], receipt["torchmonarch_pin"], receipt["companion_version"]) == ("7.7.7", "9.9.9", __version__)


def test_version_receipt_without_a_version_line_is_unknown(store, monkeypatch):
    spawn, _ = fake_child(0, b"not a version line\n")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    ops = Operations(store, Driver())
    store.create("operations", {"id": "version", "action": "version"})
    state, receipt = asyncio.run(ops.execute_owned({"id": "version", "action": "version", "parameters": {}}))
    assert state == "unknown" and receipt["version"] is None and receipt["version_reason"]


def test_management_child_ignores_a_dgx_monarch_package_in_the_launch_directory(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    (launch / "dgx_monarch").mkdir(parents=True)
    marker = tmp_path / "shadow-imported"
    (launch / "dgx_monarch" / "__init__.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    (launch / "dgx_monarch" / "__main__.py").write_text("print('dgx-monarch 0.0.0 (torchmonarch pin 0.0.0)')\n")
    # The child imports the code under test through PYTHONPATH, which -P keeps.
    monkeypatch.setenv("PYTHONPATH", str(Path(dgx_monarch.__file__).resolve().parents[1]))
    monkeypatch.chdir(launch)
    with closing(Store(tmp_path / "state")) as store:
        ops = Operations(store, Driver())
        store.create("operations", {"id": "real-version", "action": "version"})
        state, receipt = asyncio.run(ops.execute_owned({"id": "real-version", "action": "version", "parameters": {}}))
    assert not marker.exists()
    assert state == "succeeded" and receipt["version"] == __version__


def test_relative_config_and_repair_receipt_never_depend_on_the_child_directory(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ops = Operations(store, Driver(), config=Path("cluster.toml"), allow_management=True)
    assert ops.config == tmp_path / "cluster.toml" and ops.config.is_absolute()
    with pytest.raises(StudioError) as caught:
        ops.preview({"action": "doctor.repair", "parameters": {"receipt": "repair.json"}})
    assert (caught.value.code, caught.value.status) == ("operation_path", 400)
    monkeypatch.setenv("HOME", str(tmp_path))
    review = ops.preview({"action": "doctor.repair", "parameters": {"receipt": "~/repair.json"}})
    assert review["parameters"]["receipt"] == str(tmp_path / "repair.json")


def test_unexpected_adapter_error_records_unknown_with_a_code(store, caplog):
    async def broken(operation):
        raise KeyError("adapter bug")

    ops = Operations(store, Driver())
    ops._management = SimpleNamespace(execute=broken)
    operation = store.create("operations", {"id": "broken", "action": "top.replay", "state": "applying", "parameters": {},
                                            "driver_service": None, "config_fingerprint": None})
    with caplog.at_level(logging.ERROR, logger="dgx_monarch.studio.operations"):
        asyncio.run(ops.execute(operation))
    result = store.get("operations", "broken")
    assert result["state"] == "unknown" and result["receipt"]["error"] == "operation_internal_error"
    assert "broken" in caplog.text


def test_update_capability_carries_no_mode_specific_config_hint(store, config):
    rows = Operations(store, Driver(), config=config, allow_management=True).capabilities()
    assert all("requires_config_for_modes" not in row for row in rows)
