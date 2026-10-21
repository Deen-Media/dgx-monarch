"""CLI parity is checked against the parser and typed effects, without hardware."""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import subprocess
from collections import defaultdict
from contextlib import closing
from copy import deepcopy
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import legacy_update
from dgx_monarch.cli.arguments import build_parser
from dgx_monarch.studio import management_observe, management_reports
from dgx_monarch.studio.contracts import StudioError, digest
from dgx_monarch.studio.inventory import discover_cli
from dgx_monarch.studio.management import Management
from dgx_monarch.studio.management_fields import FIELDS, parse
from dgx_monarch.studio.management_parity import bindings, validate_bindings
from dgx_monarch.studio.operations import Operations, command_arguments
from dgx_monarch.studio.store import Store


@pytest.fixture(autouse=True)
def no_commands(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("This test must not execute external commands")
    monkeypatch.setattr(subprocess, "run", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", denied)


def test_all_parser_rows_have_explicit_executable_or_surface_mapping():
    parser = build_parser(defaultdict(lambda: lambda args: 0))
    expected, pending = set(), [("global", parser)]
    while pending:
        command, current = pending.pop()
        for action in current._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, child in action.choices.items():
                    expected.add(f"cli.command.{name}")
                    pending.append((name, child))
            else:
                option = action.option_strings[-1].lstrip("-").replace("-", "_")
                expected.add(f"cli.option.{command}.{option}")
    assert expected == bindings().keys()
    rows = discover_cli()
    assert validate_bindings(rows) == bindings()
    assert all(row["studio_binding"] == bindings()[row["id"]] for row in rows)
    for mutation in ({"id": "cli.option.top.new_option"},):
        with pytest.raises(ValueError, match="coverage changed"):
            validate_bindings([*rows, mutation])
    changed = deepcopy(rows)
    next(row for row in changed if row["id"] == "cli.option.top.interval")["default"] = 99
    with pytest.raises(ValueError, match="default differs"):
        validate_bindings(changed)


@pytest.mark.parametrize("action", ["workers.up", "workers.restart"])
def test_sync_is_enabled_by_default_and_no_sync_is_an_explicit_choice(action):
    assert "--no-sync" not in command_arguments(action, parse(action, {}))
    assert command_arguments(action, parse(action, {"no_sync": True}))[-1] == "--no-sync"
    with pytest.raises(StudioError):
        parse(action, {"no_sync": "false"})


@pytest.mark.parametrize("action", ["reap.preview", "reap.apply"])
def test_reap_grace_reaches_existing_backend(action):
    assert command_arguments(action, parse(action, {"grace_s": 12.5}))[-2:] == ["--grace", "12.5"]
    for invalid in (-1, True, float("nan")):
        with pytest.raises(StudioError):
            parse(action, {"grace_s": invalid})


def gate_fixture():
    prompt = {"1": {"class_type": "Inputs", "inputs": {}}, "2": {"class_type": "DGXMonarchKSampler", "inputs": {
        "model": ["1", 0], "positive": ["1", 1], "negative": ["1", 1], "latent_image": ["1", 2],
        "seed": ["1", 3], "cfg": 1.5, "sampler_name": "euler", "scheduler": "simple"}}}
    schema = {"Inputs": {"input": {"required": {}}, "output": ["MODEL", "CONDITIONING", "LATENT", "INT"]},
              "DGXMonarchIdentityGate": {"input": {"required": {
                  "model": ["MODEL", {}], "positive": ["CONDITIONING", {}], "negative": ["CONDITIONING", {}],
                  "latent_image": ["LATENT", {}], "noise_seed": ["INT", {}], "steps": ["INT", {}],
                  "cfg": ["FLOAT", {}], "sampler_name": [["euler"], {}], "scheduler": [["simple"], {}],
                  "run_id": ["STRING", {}]}}, "output": ["LATENT"], "output_node": True}}
    return prompt, schema


def test_gate_prepare_compiles_latest_exact_prompt_and_single_apply(monkeypatch, tmp_path):
    prompt, schema = gate_fixture()
    submitted = []
    class Client:
        origin = "http://127.0.0.1:8188"
        async def request(self, method, path, data=None):
            if path == "/history":
                return {"old": {"prompt": [1, "old", {}]}, "last": {"prompt": [2, "last", prompt]}}
            if path == "/object_info":
                return schema
            assert (method, path) == ("POST", "/prompt")
            submitted.append(data)
            return {"prompt_id": "accepted"}
        async def history_prompt(self, identity):
            return {identity: {"status": {"completed": True, "status_str": "success"}}}
    monkeypatch.setattr(management_observe, "_find_gate_report", lambda path, identity: {"run_id": identity, "verdict": "PASS"})
    async def exercise():
        with closing(Store(tmp_path / "state")) as store:
            ops = Operations(store, Client(), allow_management=True)
            plan = await ops.prepare({"action": "gate.last", "parameters": {}})
            assert plan["adapter_plan"]["source_prompt_id"] == "last"
            assert plan["adapter_plan"]["prompt"]["2"]["inputs"]["noise_seed"] == ["1", 3]
            assert prompt["2"]["class_type"] == "DGXMonarchKSampler"
            request = {"digest": plan["digest"], "confirmed": True, "idempotency_key": "gate-once"}
            accepted = ops.apply(plan["id"], request)
            await asyncio.gather(*ops.tasks)
            repeated = ops.apply(plan["id"], request)
            assert repeated["id"] == accepted["id"] and repeated["state"] == "succeeded"
            assert len(submitted) == 1
    asyncio.run(exercise())


def test_gate_unknown_dispatch_retains_phase_and_does_not_retry(tmp_path):
    _, schema = gate_fixture()
    calls = []
    class Client:
        origin = "http://127.0.0.1:8188"
        async def request(self, method, path, data=None):
            calls.append((method, path))
            if path == "/object_info":
                return schema
            raise StudioError("upstream_unavailable", "connection ended after request", 502)
    async def exercise():
        with closing(Store(tmp_path / "state")) as store:
            ops = Operations(store, Client(), allow_management=True)
            plan = ops.preview({"action": "gate.last"}, {"schema_digest": digest(schema), "run_id": "exact",
                "source_prompt_id": "source", "prompt": {}})
            record = ops.apply(plan["id"], {"digest": plan["digest"], "confirmed": True, "idempotency_key": "unknown-once"})
            await asyncio.gather(*ops.tasks)
            final = store.get("operations", record["id"])
            assert final["state"] == "unknown" and final["receipt"]["phase"] == "submission_requested"
            assert final["receipt"]["run_id"] == "exact"
    asyncio.run(exercise())
    assert calls.count(("POST", "/prompt")) == 1


def test_gate_schema_drift_refuses_before_submission():
    class Client:
        async def request(self, method, path, data=None):
            assert path == "/object_info"
            return {"changed": True}
    with pytest.raises(StudioError, match="schema changed"):
        asyncio.run(management_observe.gate(Client(), {}, {"schema_digest": digest({})}, lambda receipt: None))


def test_diagnostics_uses_sanitized_capture_and_refuses_overwrite(monkeypatch, tmp_path):
    capture = management_observe.diagnostic_session.Capture(duration_s=1)
    monkeypatch.setattr(management_observe.diagnostic_session, "Capture", lambda **kwargs: capture)
    ingest = capture.ingest
    def finish_ingest(value):
        ingest(value)
        capture.deadline = 0
    monkeypatch.setattr(capture, "ingest", finish_ingest)
    class Client:
        async def request(self, method, path, data=None):
            assert (method, path) == ("GET", "/dgxm/telemetry")
            return {"secret": "never-in-report"}
    target = tmp_path / "capture.json"
    parameters = parse("diagnostics", {"duration_s": 1, "output": str(target)})
    state, receipt = asyncio.run(management_observe.diagnostics(Client(), parameters, lambda report: None))
    assert state == "unknown" and receipt["report"]["capture"]["samples"] == 1
    assert "never-in-report" not in target.read_text() and target.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        asyncio.run(management_observe.diagnostics(Client(), parameters, lambda report: None))


def test_legacy_review_target_is_checked_before_installation(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(legacy_update, "_contract_edited", lambda repo: False)
    monkeypatch.setattr(legacy_update, "_fetch_candidate", lambda repo: ("a" * 40, True))
    assert inspect.signature(legacy_update._run_locked).parameters["expected_candidate"].default is None
    code = legacy_update._run_locked(SimpleNamespace(host=None), repo=tmp_path,
        load_config=lambda args: None, driver_probe=lambda host: (None, True), expected_candidate=("b" * 40, True))
    assert code == 2  # the autouse subprocess guard proves no metadata read/install/restart ran


def test_report_parser_retains_structure_but_redacts_diagnostics():
    result = management_reports.command_report("doctor", json.dumps({"checks": [{"check": "ready", "state": "warn",
        "detail": "token=secret-value host=worker-private /private/file"}], "summary": {"checks": 1, "failures": 0},
        "unexpected": "never returned"}).encode())
    assert result["summary"] == {"checks": 1, "failures": 0}
    assert "secret-value" not in json.dumps(result) and "unexpected" not in result
    assert management_reports.command_report("doctor", b'{"truncated":') is None


def test_record_append_and_replay_path_are_review_bound(tmp_path):
    class Client:
        origin = "http://127.0.0.1:8188"
        async def request(self, method, path, data=None):
            return {"render": {"active": False}}
    target = tmp_path / "ticks.jsonl"
    target.write_text('{"t":1}\n')
    target.chmod(0o600)
    with closing(Store(tmp_path / "state")) as store:
        manager = Management(store, Client(), None, None)
        params, plan = manager.preview("top.record", {"samples": 1, "path": str(target)})
        store.create("operations", {"id": "record", "state": "applying"})
        state, receipt = asyncio.run(manager.record(params, "record", plan))
        assert state == "succeeded" and receipt["samples"] == 2
        _, replay = manager.preview("top.replay", {"path": str(target)})
        assert manager.replay(replay)[1]["samples"][0]["t"] == 1
        target.write_text('{"t":9}\n')
        with pytest.raises(StudioError, match="changed"):
            manager.replay(replay)
        with pytest.raises(StudioError, match="changed"):
            asyncio.run(manager.record(params, "record", plan))


@pytest.mark.parametrize(("exit_code", "outcome"), [(0, "legacy_updated"), (1, "legacy_incomplete"), (2, "legacy_refused"),
                                                    (3, "legacy_incomplete"), (75, "legacy_refused")])
@pytest.mark.parametrize("with_config", [True, False])
def test_frozen_child_dispatches_exact_legacy_candidate(tmp_path, monkeypatch, exit_code, outcome, with_config):
    import hashlib
    import os
    from dataclasses import asdict

    from dgx_monarch import config as config_module
    from dgx_monarch.cli import update_command, update_inspect
    from dgx_monarch.cli.update_release import ReleaseMetadata
    from dgx_monarch.cli.update_types import RepoSnapshot, UpdateResult
    from dgx_monarch.studio import management_repository, management_update, update_controller

    config = tmp_path / "cluster.toml"
    config.write_text("selected config fixture")
    metadata = ReleaseMetadata("a" * 40, "1", "0.7.0", "b" * 64, "c" * 64)
    comfy = tmp_path / "comfy"
    (comfy / "custom_nodes").mkdir(parents=True)
    (comfy / "custom_nodes/dgx-monarch").symlink_to(tmp_path)
    reviewed = {"controller": {"bound": True}, "checkout": {"bound": True}, "prior_release": asdict(metadata),
                "release": {"target_sha": "d" * 40, "source_manifest": "e" * 64}, "mode": "legacy", "legacy_candidate": ["d" * 40, True],
                "driver_identity": {"config": {"source_files": [{"path": str(comfy / "main.py")}]}},
                "driver_checkout": {"link": str(comfy / "custom_nodes/dgx-monarch"), "resolved": str(tmp_path.resolve())}}
    calls = []
    monkeypatch.setattr(management_repository, "RepositoryOps", lambda repo:
                        SimpleNamespace(repo_snapshot=lambda target: RepoSnapshot("a" * 40, True, True)))
    monkeypatch.setattr(update_controller, "binding", lambda driver: {"bound": True})
    monkeypatch.setattr(update_controller.signal, "signal", lambda *args: None)
    monkeypatch.setattr(management_update, "checkout_identity", lambda path: {"bound": True})
    monkeypatch.setattr(config_module, "load_cluster_config", lambda path: "selected-config")
    monkeypatch.setattr(update_inspect, "inspect_checkout", lambda *args: (metadata, ()))
    monkeypatch.setattr(update_command, "SystemUpdateOps", lambda *args, **kwargs:
                        SimpleNamespace(repo_snapshot=lambda target: RepoSnapshot("a" * 40, True, True)))
    def legacy(args, *, repo, load_config, driver_probe, expected_candidate, expected_source_manifest):
        calls.append(expected_candidate)
        assert expected_source_manifest == "e" * 64
        assert args.host == "selected:8188" and load_config(args) == "selected-config"
        assert repo == tmp_path
        return exit_code
    monkeypatch.setattr(legacy_update, "_run_locked", legacy)
    descriptor = os.open(tmp_path / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        manifest = {"request": {"reviewed": reviewed, "lock_fd": descriptor, "config": str(config) if with_config else None,
                    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest() if with_config else None, "repo": str(tmp_path),
                    "driver_host": "selected:8188", "privileged_process_inspection": False}}
        returned = update_controller.child(tmp_path, manifest)
    finally:
        os.close(descriptor)
    settled = UpdateResult(**json.loads((tmp_path / "result.json").read_text()))
    assert returned == settled.exit_code
    if not with_config:
        # Both modes need the selected configuration: legacy no longer runs without it.
        assert returned == 2 and calls == [] and settled.code == "controller_refused"
        assert settled.receipt["refusal"] == "update_config"
        assert update_controller.settlement("legacy", settled) == "refused"
        return
    assert settled.exit_code == exit_code and settled.receipt["reviewed_target"] == "d" * 40
    assert calls == [("d" * 40, True)] and settled.code == outcome
    # The receipt says what happened; a failure no longer reads as "completed".
    assert ("completed" in settled.message) is (exit_code == 0)
    expected = "succeeded" if exit_code == 0 else "refused" if exit_code in {2, 75} else "unknown"
    assert update_controller.settlement("legacy", settled) == expected


def test_gate_repair_plan_binds_exact_ledger(monkeypatch):
    rows = []
    class Ledger:
        def __init__(self, path):
            pass
        def entries_with_integrity(self):
            return deepcopy(rows), 0
    monkeypatch.setattr(management_reports, "GateLedger", Ledger)
    monkeypatch.setattr(management_reports.gate_repair, "find_orphans", lambda entries: ([], 0))
    effects = []
    monkeypatch.setattr(management_reports.gate_repair, "repair", lambda *args, **kwargs: effects.append("repair") or 0)
    parameters = parse("gate.repair.apply", {})
    plan = management_reports.gate_plan("gate.repair.apply", parameters)
    rows.append((1, {"changed": True}))
    with pytest.raises(StudioError, match="changed"):
        management_reports.gate_execute("gate.repair.apply", parameters, plan)
    assert effects == []


def test_setup_and_init_default_destinations_preserve_cli_fallback(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with closing(Store(tmp_path / "state")) as store:
        manager = Management(store, SimpleNamespace(origin="http://127.0.0.1:8188"), None, None)
        assert manager.selected_path("setup", {}) == tmp_path / ".config/dgx-monarch/cluster.toml"
        assert manager.selected_path("init", {}) == tmp_path / ".config/dgx-monarch/cluster.toml"
        assert manager.selected_path("setup", {"output": str(tmp_path / "selected")}) == tmp_path / "selected"


@pytest.mark.parametrize("row", discover_cli(), ids=lambda row: row["id"])
def test_each_cli_row_resolves_to_declared_studio_capability_or_explained_surface(row, tmp_path):
    with closing(Store(tmp_path / "state")) as store:
        ops = Operations(store, SimpleNamespace(origin="http://127.0.0.1:8188"), config=tmp_path / "config.toml",
                         allow_management=True, driver_service="comfy.service")
        capabilities = {item["id"]: item for item in ops.capabilities()}
        binding = row["studio_binding"]
        for action in binding.get("actions", []):
            assert capabilities[action]["available"]
        for action, name in binding.get("parameters", []):
            assert name in {field["name"] for field in capabilities[action]["parameters"]}
        if binding["kind"] in {"presentation", "context", "selector"}:
            assert binding["equivalence"]


@pytest.mark.parametrize("mode", ["legacy", "verified"])
def test_update_requires_configuration_in_every_mode(tmp_path, monkeypatch, mode):
    from dgx_monarch.studio import management_update

    monkeypatch.setattr(management_update, "checkout_identity", lambda repo: pytest.fail("checkout inspected without config"))
    manager = management_update.Updates(None, None, "http://127.0.0.1:8188", tmp_path)
    parameters = parse("update", {"repo": str(tmp_path), "mode": mode})
    for review in (manager.operations, manager.preview):
        with pytest.raises(StudioError, match="--config at companion launch, in both modes") as refused:
            review(parameters)
        assert refused.value.code == "update_config"
    with closing(Store(tmp_path / "state")) as store:
        ops = Operations(store, SimpleNamespace(origin="http://127.0.0.1:8188"), allow_management=True)
        row = next(row for row in ops.capabilities() if row["id"] == "update")
        # Availability names the missing --config instead of offering a config-free update.
        assert row["available"] is False and "cluster configuration" in row["reason"]
        review = ops.preview({"action": "update", "parameters": {"repo": str(tmp_path), "mode": mode}})
        assert review["available"] is False and review["adapter_plan"] is None


def test_update_is_unavailable_without_the_driver_service_and_says_why(tmp_path):
    with closing(Store(tmp_path / "state")) as store:
        for service, available in ((None, False), ("comfy.service", True)):
            ops = Operations(store, SimpleNamespace(origin="http://127.0.0.1:8188"), config=tmp_path / "config.toml",
                             allow_management=True, driver_service=service)
            row = next(row for row in ops.capabilities() if row["id"] == "update")
            assert row["available"] is available
            if not available:
                assert "--driver-service" in row["reason"] and "interpreter and checkout" in row["reason"]
                review = ops.preview({"action": "update", "parameters": {"repo": str(tmp_path)}})
                assert review["available"] is False and review["adapter_plan"] is None
        # Only update needs the unit among the actions a configuration enables.
        ops = Operations(store, SimpleNamespace(origin="http://127.0.0.1:8188"), config=tmp_path / "config.toml",
                         allow_management=True)
        assert {row["id"] for row in ops.capabilities() if row["reason"] and "--driver-service" in row["reason"]} == {"update"}


def test_update_defaults_to_verified_mode_and_parity_rows_describe_both_modes():
    assert parse("update", {"repo": "/reviewed/checkout"})["mode"] == "verified"
    rows = bindings()
    assert "Studio default" in rows["cli.option.update.verify"]["equivalence"]
    assert "Both modes need --config" in rows["cli.option.update.verify"]["equivalence"]
    host = rows["cli.option.update.host"]["equivalence"]
    assert "--driver-service" in host and "Verified mode passes no host" in host
    assert "silently probe" in rows["cli.option.top.host"]["equivalence"]
    from dgx_monarch.studio.management_update import REVIEW_SCOPE

    # Each review names the guards its own mode runs, rather than claiming both modes keep the same ones.
    assert set(REVIEW_SCOPE) == set(next(entry for entry in FIELDS["update"] if entry["name"] == "mode")["choices"])
    assert all(word in REVIEW_SCOPE["verified"] for word in ("activity", "stages the release", "exact readback"))
    assert all(word in REVIEW_SCOPE["legacy"] for word in ("release-slot check", "Worker restart", "no rollback"))


@pytest.mark.parametrize("hidden", [False, True])
def test_repository_only_observation_uses_same_clean_and_fastforward_guards(tmp_path, monkeypatch, hidden):
    from dgx_monarch.studio.management_repository import RepositoryOps

    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "rev-parse" in argv:
            output = "a" * 40
        elif "-v" in argv:
            output = "h hidden.py" if hidden else "H normal.py"
        else:
            output = ""
        return SimpleNamespace(returncode=0, stdout=output)
    monkeypatch.setattr(subprocess, "run", run)
    snapshot = RepositoryOps(tmp_path).repo_snapshot("b" * 40)
    assert snapshot.clean is not hidden and snapshot.fast_forward and snapshot.current_sha == "a" * 40
    assert len(calls) == 5
    assert all("core.hooksPath=/dev/null" in argv and kwargs["env"]["GIT_CONFIG_NOSYSTEM"] == "1" for argv, kwargs in calls)


def test_installed_legacy_children_are_isolated_and_source_bound(tmp_path):
    assert legacy_update._installed_command(tmp_path, ["restart"], None)[1:] == ["-m", "dgx_monarch", "restart"]
    command = legacy_update._installed_command(tmp_path, ["--config", "config.toml", "restart"], "a" * 64)
    assert command[1:3] == ["-I", "-c"]
    assert command[4:] == [str(tmp_path / "src/dgx_monarch"), "a" * 64, "--config", "config.toml", "restart"]
    assert 'dgx_source_manifest_sha256(source) != expected' in command[3]
    assert 'Path(spec.origin).resolve() != source / "__init__.py"' in command[3]


@pytest.mark.parametrize("drift", [None, "path", "hash"])
def test_installed_child_refuses_path_or_hash_drift_before_lifecycle(tmp_path, monkeypatch, drift):
    import importlib
    import sys

    from dgx_monarch import runtime_provenance

    cli = importlib.import_module("dgx_monarch.cli.main")
    source = tmp_path / "src/dgx_monarch"
    expected = "a" * 64
    calls = []
    monkeypatch.setattr(sys, "argv", ["-c", str(source), expected, "--config", "selected.toml", "restart"])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name:
                        SimpleNamespace(origin=str(tmp_path / "foreign.py" if drift == "path" else source / "__init__.py")))
    def source_hash(path):
        calls.append("hash")
        return "b" * 64 if drift == "hash" else expected
    monkeypatch.setattr(runtime_provenance, "dgx_source_manifest_sha256", source_hash)
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv) or 0)
    with pytest.raises(SystemExit) as exc:
        exec(legacy_update._INSTALLED_CLI, {})  # noqa: S102 - exercise the fixed, trusted child program with mocked imports
    assert calls == ([] if drift == "path" else ["hash"] if drift == "hash" else ["hash", ["--config", "selected.toml", "restart"]])
    assert (exc.value.code == 0) is (drift is None)
