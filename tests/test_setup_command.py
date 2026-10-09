"""Plan, apply, verification and compensation contracts for the guided setup backend."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dgx_monarch.cli import (
    cluster_smoke,
    operator_receipt,
    setup_command,
    setup_config_io,
    setup_config_lock,
    setup_config_parent,
    setup_config_transaction,
    setup_probe,
    setup_receipt_timing,
    setup_services,
    setup_source,
    setup_verification,
)
from dgx_monarch.config import load_cluster_config


def _payload(
    *,
    artifact_hash: str = "a" * 64,
    dirty: bool = False,
    service_installed: bool | None = False,
    service_active: bool | None = False,
) -> dict[str, object]:
    return {
        "schema": 1,
        "python_available": True,
        "python_version": "3.12.3",
        "torch_version": "2.8.0+cu128",
        "cuda_available": True,
        "gpu_count": 1,
        "integrated": True,
        "gpu_models": ["NVIDIA GB10"],
        "torchmonarch_version": "0.6.0",
        "comfy_exists": True,
        "comfy_runtime_marker": True,
        "comfy_git": True,
        "comfy_dirty": dirty,
        "comfy_commit": "b" * 40,
        "fabric_interface_count": 2,
        "link_layers": ["Ethernet"],
        "rsync_available": True,
        "systemd_user_available": True,
        "linger_enabled": True,
        "service_installed": service_installed,
        "systemd_service_active": service_active,
        "worker_process_active": False if service_active is not None else None,
        "worker_listener_active": False if service_active is not None else None,
        "service_active": service_active,
        "artifacts": [],
    }


def _runner(payloads: list[dict[str, object]] | None = None):
    rows = payloads or [_payload()]
    calls: list[int] = []

    def run(_config, _host, _script, *, timeout):
        index = len(calls)
        calls.append(index)
        return subprocess.CompletedProcess([], 0, setup_probe._MARKER + json.dumps(rows[index % len(rows)]) + "\n", "")

    return run, calls


def _request(tmp_path: Path, **changes) -> setup_command.SetupRequest:
    values = {
        "hosts": (setup_command.CandidateHost("secret-spark", "10.20.30.40"),),
        "client_ip": "10.20.30.40",
        "config_path": tmp_path / "cluster.toml",
        "profile": "safe",
        "python_bin": "python3",
        "transport_security": "trusted_fabric",
    }
    values.update(changes)
    return setup_command.SetupRequest(**values)


def _ops(runner, **changes) -> setup_command.SetupOps:
    written: list[dict[str, object]] = changes.pop("written", [])

    def write_receipt(receipt, _path=None):
        written.append(dict(receipt))
        return Path("/not/exposed/receipt.json")

    selected_writer = changes.pop("write_receipt", write_receipt)
    changes.setdefault(
        "source_snapshot",
        lambda: setup_source.SetupSource(Path("/not/exposed/source"), "d" * 64),
    )
    changes.setdefault("execute_services", _successful_services)
    changes.setdefault("compensate_services", _compensated_services)
    changes.setdefault("config_lock", lambda _path: contextlib.nullcontext())
    return replace(setup_command.SetupOps(), run_host=runner, write_receipt=selected_writer, **changes)


def _service_result(
    request: setup_services.SetupServiceRequest,
    *,
    certainty: setup_services.ServiceCertainty,
    code: str,
    changed: bool,
    compensated: bool,
    interruption: BaseException | None = None,
) -> setup_services.SetupServiceResult:
    final = "succeeded" if certainty is setup_services.ServiceCertainty.SUCCEEDED else str(certainty)
    hosts = tuple(
        setup_services.HostServiceResult(
            ordinal,
            stage="succeeded" if changed else "not_run",
            activate=final if changed else "not_run",
            compensate="succeeded" if compensated else "not_run",
            cleanup="succeeded" if compensated else "not_run",
        )
        for ordinal in range(1, len(request.config.hosts) + 1)
    )
    return setup_services.SetupServiceResult(
        certainty,
        code,
        changed,
        compensated,
        request.source_manifest,
        hosts,
        interruption,
        request.binding(),
    )


def _successful_services(request):
    return _service_result(
        request,
        certainty=setup_services.ServiceCertainty.SUCCEEDED,
        code="succeeded",
        changed=True,
        compensated=False,
    )


def _compensated_services(request, _result):
    return _service_result(
        request,
        certainty=setup_services.ServiceCertainty.SUCCEEDED,
        code="compensated",
        changed=True,
        compensated=True,
    )


def test_setup_source_identity_is_canonical_and_digest_bound():
    with pytest.raises(ValueError, match="absolute Path"):
        setup_source.SetupSource(Path("relative/source"), "d" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        setup_source.SetupSource(Path("/source"), "not-a-digest")


def test_config_snapshot_and_mutation_records_reject_malformed_injected_state(tmp_path: Path):
    with pytest.raises(ValueError, match="snapshot"):
        setup_config_io.ConfigSnapshot(Path("relative"), False, b"", None, None, None)
    with pytest.raises(ValueError, match="snapshot"):
        setup_config_io.ConfigSnapshot(tmp_path / "cluster.toml", True, b"bytes", "d" * 64, 0o600, (1, 2))
    snapshot = setup_config_io.read_snapshot(tmp_path / "cluster.toml")
    with pytest.raises(ValueError, match="mutation"):
        setup_config_io.ConfigMutation(snapshot.path, True, snapshot, "d" * 64, None, None, True)
    with pytest.raises(ValueError, match="mutation"):
        setup_config_io.ConfigMutation(snapshot.path, False, snapshot, "d" * 64, None, None, True)


def test_config_parent_creation_is_private_and_fsyncs_each_new_entry(tmp_path: Path):
    parent = tmp_path / "one" / "two"
    fsynced: list[tuple[int, int]] = []

    setup_config_parent.prepare_config_parent(
        parent,
        fsync=lambda descriptor: fsynced.append((os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)),
    )

    assert fsynced == [
        (tmp_path.stat().st_dev, tmp_path.stat().st_ino),
        ((tmp_path / "one").stat().st_dev, (tmp_path / "one").stat().st_ino),
    ]
    assert stat.S_IMODE((tmp_path / "one").stat().st_mode) == 0o700
    assert stat.S_IMODE(parent.stat().st_mode) == 0o700


def test_config_parent_refuses_an_unsafe_intermediate_directory(tmp_path: Path):
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o700)
    unsafe.chmod(0o777)

    with pytest.raises(OSError, match="unsafe"):
        setup_config_parent.prepare_config_parent(unsafe / "private")

    assert not (unsafe / "private").exists()


def test_config_parent_fd_handoff_preserves_primary_and_closes_children(tmp_path: Path, monkeypatch):
    primary = KeyboardInterrupt()
    real_close = setup_config_parent.os.close
    closed: list[int] = []

    def interrupted_close(descriptor):
        closed.append(descriptor)
        real_close(descriptor)
        if len(closed) == 1:
            raise primary

    monkeypatch.setattr(setup_config_parent.os, "close", interrupted_close)
    with pytest.raises(KeyboardInterrupt) as raised:
        setup_config_parent.prepare_config_parent(tmp_path / "one" / "two")

    assert raised.value is primary
    assert len(set(closed)) == 2
    for descriptor in set(closed):
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_config_parent_final_close_cancellation_preserves_identity_and_closes_fd(tmp_path: Path, monkeypatch):
    primary = KeyboardInterrupt()
    real_close = setup_config_parent.os.close
    expected = tmp_path.stat().st_dev, tmp_path.stat().st_ino
    interrupted_descriptor = -1

    def interrupted_close(descriptor):
        nonlocal interrupted_descriptor
        info = os.fstat(descriptor)
        real_close(descriptor)
        if (info.st_dev, info.st_ino) == expected:
            interrupted_descriptor = descriptor
            raise primary

    monkeypatch.setattr(setup_config_parent.os, "close", interrupted_close)
    with pytest.raises(KeyboardInterrupt) as raised:
        setup_config_parent.prepare_config_parent(tmp_path)

    assert raised.value is primary
    with pytest.raises(OSError):
        os.fstat(interrupted_descriptor)


def test_config_lock_refuses_contention_and_normalizes_dotdot_aliases(tmp_path: Path):
    state = tmp_path / "state"
    alias = tmp_path / "alias"
    alias.mkdir()
    direct = tmp_path / "cluster.toml"
    equivalent = alias / ".." / "cluster.toml"

    with setup_config_lock.SetupConfigLock(direct, state_root=state):
        with pytest.raises(setup_config_lock.SetupConfigLockUnavailable):
            with setup_config_lock.SetupConfigLock(equivalent, state_root=state):
                pass


@pytest.mark.skipif(os.name != "posix", reason="double-slash root aliases are POSIX-specific")
def test_config_lock_serializes_single_and_double_slash_root_aliases(tmp_path: Path):
    state = tmp_path / "state"
    direct = tmp_path / "cluster.toml"
    double_slash = Path(f"//{direct.as_posix().lstrip('/')}")

    with setup_config_lock.SetupConfigLock(direct, state_root=state):
        with pytest.raises(setup_config_lock.SetupConfigLockUnavailable):
            with setup_config_lock.SetupConfigLock(double_slash, state_root=state):
                pass


def test_config_lock_refuses_unsafe_root_and_existing_lock(tmp_path: Path):
    unsafe = tmp_path / "unsafe-locks"
    unsafe.mkdir(mode=0o700)
    unsafe.chmod(0o777)
    with pytest.raises(setup_config_lock.SetupConfigLockUnavailable):
        with setup_config_lock.SetupConfigLock(tmp_path / "cluster.toml", state_root=unsafe):
            pass

    state = tmp_path / "private-locks"
    with setup_config_lock.SetupConfigLock(tmp_path / "cluster.toml", state_root=state):
        pass
    lock_path = next(state.iterdir())
    lock_path.chmod(0o644)
    with pytest.raises(setup_config_lock.SetupConfigLockUnavailable):
        with setup_config_lock.SetupConfigLock(tmp_path / "cluster.toml", state_root=state):
            pass


def test_config_lock_cancellation_after_flock_closes_the_descriptor(tmp_path: Path, monkeypatch):
    state = tmp_path / "locks"
    lock = setup_config_lock.SetupConfigLock(tmp_path / "cluster.toml", state_root=state)
    interrupt = KeyboardInterrupt()
    real_flock = setup_config_lock.fcntl.flock

    def interrupted_flock(descriptor, operation):
        real_flock(descriptor, operation)
        if operation & setup_config_lock.fcntl.LOCK_EX:
            raise interrupt

    monkeypatch.setattr(setup_config_lock.fcntl, "flock", interrupted_flock)
    with pytest.raises(KeyboardInterrupt) as raised:
        lock.__enter__()
    assert raised.value is interrupt
    assert lock._descriptor is None

    monkeypatch.setattr(setup_config_lock.fcntl, "flock", real_flock)
    with setup_config_lock.SetupConfigLock(tmp_path / "cluster.toml", state_root=state):
        pass


def test_config_lock_cleanup_preserves_body_cancellation_and_surfaces_its_own(tmp_path: Path, monkeypatch):
    state = tmp_path / "locks"
    body_interrupt = KeyboardInterrupt()
    cleanup_interrupt = SystemExit(7)
    real_flock = setup_config_lock.fcntl.flock
    lock = setup_config_lock.SetupConfigLock(tmp_path / "cluster.toml", state_root=state)
    descriptor = -1

    with pytest.raises(KeyboardInterrupt) as raised:
        with lock:
            descriptor = lock._descriptor or -1

            def fail_unlock(fd, operation):
                if operation & setup_config_lock.fcntl.LOCK_UN:
                    raise cleanup_interrupt
                return real_flock(fd, operation)

            monkeypatch.setattr(setup_config_lock.fcntl, "flock", fail_unlock)
            raise body_interrupt
    assert raised.value is body_interrupt
    with pytest.raises(OSError):
        os.fstat(descriptor)

    monkeypatch.setattr(setup_config_lock.fcntl, "flock", real_flock)
    lock = setup_config_lock.SetupConfigLock(tmp_path / "other.toml", state_root=state)
    lock.__enter__()
    monkeypatch.setattr(
        setup_config_lock.fcntl,
        "flock",
        lambda _fd, operation: (
            (_ for _ in ()).throw(cleanup_interrupt) if operation & setup_config_lock.fcntl.LOCK_UN else None
        ),
    )
    with pytest.raises(SystemExit) as raised_cleanup:
        lock.__exit__(None, None, None)
    assert raised_cleanup.value is cleanup_interrupt


def test_config_lock_unlock_oserror_is_settled_by_descriptor_close(tmp_path: Path, monkeypatch):
    state = tmp_path / "locks"
    target = tmp_path / "cluster.toml"
    real_flock = setup_config_lock.fcntl.flock

    def fail_unlock(descriptor, operation):
        if operation & setup_config_lock.fcntl.LOCK_UN:
            raise OSError("unlock interrupted")
        return real_flock(descriptor, operation)

    with setup_config_lock.SetupConfigLock(target, state_root=state):
        monkeypatch.setattr(setup_config_lock.fcntl, "flock", fail_unlock)

    monkeypatch.setattr(setup_config_lock.fcntl, "flock", real_flock)
    with setup_config_lock.SetupConfigLock(target, state_root=state):
        pass


def test_plan_is_strict_deterministic_model_aware_and_sanitized(tmp_path: Path):
    current = tmp_path / "cluster.toml"
    current.write_text("# prior config\n", encoding="utf-8")
    runner, calls = _runner()
    request = _request(tmp_path)

    first = setup_command.build_setup_plan(request, ops=_ops(runner))
    second = setup_command.build_setup_plan(request, ops=_ops(runner))

    assert len(calls) == 2  # one explicitly supplied candidate per plan
    assert first.config_digest == second.config_digest
    assert first.diff == second.diff
    assert first.topology == "auto"
    assert first.fabric_profile == "single-node"
    assert first.config.worker_args == {"lora_low_rss": False, "slab_weights": False}
    assert first.config.rdma_latent_return is False
    assert "operator_profile" not in first.config_text
    assert first.gate_context_changes is True
    assert "will be replaced after confirmation" in first.gate_warning
    assert "existing_config_replacement_planned" in first.warnings

    public = json.dumps(first.as_dict(), sort_keys=True)
    for secret in ("secret-spark", "10.20.30.40", str(tmp_path)):
        assert secret not in public
    assert "diff" not in first.as_dict()
    assert "current/cluster.toml" in first.diff


def test_dry_run_has_no_config_service_or_receipt_mutation(tmp_path: Path):
    runner, _calls = _runner()
    touched: list[str] = []
    ops = _ops(
        runner,
        execute_services=lambda _request: touched.append("service") or None,
        written=[],
    )

    outcome = setup_command.run_setup(_request(tmp_path), ops=ops)

    assert outcome.status == "planned"
    assert outcome.applied is False
    assert not (tmp_path / "cluster.toml").exists()
    assert touched == []
    assert outcome.receipt_written is False


def test_reviewed_plan_rechecks_once_then_applies_without_rebuilding(tmp_path: Path):
    runner, calls = _runner()
    ops = _ops(runner)
    preview_request = _request(tmp_path)
    plan = setup_command.build_setup_plan(preview_request, ops=ops)

    outcome = setup_command.run_setup(replace(preview_request, apply=True, assume_yes=True), ops=ops, plan=plan)

    assert outcome.status == "succeeded"
    assert len(calls) == 2
    assert (tmp_path / "cluster.toml").read_text() == plan.config_text


def test_reviewed_plan_refuses_changed_live_evidence_before_mutation(tmp_path: Path):
    changed = _payload(dirty=True)
    runner, calls = _runner([_payload(), changed])
    ops = _ops(runner)
    preview = _request(tmp_path)
    plan = setup_command.build_setup_plan(preview, ops=ops)

    outcome = setup_command.run_setup(replace(preview, apply=True, assume_yes=True), ops=ops, plan=plan)

    assert len(calls) == 2
    assert outcome.failure == "live_readiness_changed"
    assert outcome.applied is False
    assert not (tmp_path / "cluster.toml").exists()


def test_live_recheck_cancellation_is_receipted_and_re_raised_by_identity(tmp_path: Path):
    base_runner, _calls = _runner()
    interrupt = KeyboardInterrupt()
    written: list[dict[str, object]] = []
    calls = 0

    def runner(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interrupt
        return base_runner(*args, **kwargs)

    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(tmp_path, apply=True, assume_yes=True),
            ops=_ops(runner, written=written),
        )

    assert raised.value is interrupt
    assert not (tmp_path / "cluster.toml").exists()
    assert len(written) == 1
    assert written[0]["status"] == "failed"
    steps = {step["name"]: step["status"] for step in written[0]["steps"]}
    assert steps["live_recheck"] == "unknown"
    assert steps["interrupted"] == "failed"


def test_smoke_cancellation_retains_state_and_preserves_identity(tmp_path: Path):
    runner, _calls = _runner()
    interrupt = KeyboardInterrupt()
    calls: list[str] = []
    written: list[dict[str, object]] = []

    def smoke(_path, _comfy):
        calls.append("smoke")
        raise interrupt

    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(
                tmp_path,
                apply=True,
                assume_yes=True,
                install_service=True,
                start_workers=True,
                verify=True,
            ),
            ops=_ops(
                runner,
                execute_services=lambda request: calls.append("service") or _successful_services(request),
                compensate_services=lambda request, result: (
                    calls.append("compensate") or _compensated_services(request, result)
                ),
                run_doctor=lambda _config: calls.append("doctor") or True,
                run_smoke=smoke,
                written=written,
            ),
        )

    assert raised.value is interrupt
    assert calls == ["service", "doctor", "smoke"]
    assert (tmp_path / "cluster.toml").exists()
    assert written[0]["status"] == "partial"
    steps = {step["name"]: step for step in written[0]["steps"]}
    assert steps["doctor_verify"]["status"] == "succeeded"
    assert steps["cluster_smoke"]["status"] == "unknown"
    assert steps["cluster_smoke"]["counts"]["attempted"] == 1
    assert steps["setup_compensation"]["status"] == "unknown"


def test_apply_refuses_when_local_source_changes_after_review(tmp_path: Path):
    runner, _calls = _runner()
    snapshots = iter(
        (
            setup_source.SetupSource(Path("/not/exposed/source"), "a" * 64),
            setup_source.SetupSource(Path("/not/exposed/source"), "b" * 64),
        )
    )
    outcome = setup_command.run_setup(
        _request(tmp_path, apply=True, assume_yes=True),
        ops=_ops(runner, source_snapshot=lambda: next(snapshots)),
    )

    assert outcome.failure == "live_readiness_changed"
    assert outcome.applied is False
    assert not (tmp_path / "cluster.toml").exists()


def test_live_recheck_uses_the_reviewed_deduplicated_artifact_set(tmp_path: Path):
    artifact = {
        "ordinal": 1,
        "exists": True,
        "regular": True,
        "size": 1,
        "sha256": "a" * 64,
    }
    runner, calls = _runner([{**_payload(), "artifacts": [artifact]}] * 2)

    outcome = setup_command.run_setup(
        _request(
            tmp_path,
            artifacts=("models/a.safetensors", "models/a.safetensors"),
            apply=True,
            assume_yes=True,
        ),
        ops=_ops(runner),
    )

    assert len(calls) == 2
    assert outcome.status == "succeeded"
    assert outcome.failure is None


def test_receipt_timing_starts_before_probe_and_spans_config_apply(tmp_path: Path):
    base_runner, _calls = _runner()
    events: list[str] = []
    started_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    clock_values = iter((started_at, started_at + timedelta(seconds=5)))
    timer_values = iter((100.0, 104.25))

    def clock():
        assert events in ([], ["probe", "probe", "apply"])
        return next(clock_values)

    def timer():
        assert events in ([], ["probe", "probe", "apply"])
        return next(timer_values)

    def runner(*args, **kwargs):
        events.append("probe")
        return base_runner(*args, **kwargs)

    def apply(path, text, *, expected, _transaction=None):
        events.append("apply")
        return setup_config_io.apply_config(path, text, expected=expected, _transaction=_transaction)

    outcome = setup_command.run_setup(
        _request(tmp_path, apply=True, assume_yes=True),
        ops=_ops(
            runner,
            apply_config=apply,
            begin_receipt=lambda: setup_receipt_timing.PrestartedReceipt.start(clock=clock, timer=timer),
        ),
    )

    assert outcome.status == "succeeded"
    assert events == ["probe", "probe", "apply"]
    assert outcome.receipt["started_at"] == "2026-08-08T12:00:00.000000Z"
    assert outcome.receipt["duration_ms"] == 4250


def test_reviewed_plan_refuses_changed_request_without_reprobe(tmp_path: Path):
    runner, calls = _runner()
    ops = _ops(runner)
    preview_request = _request(tmp_path)
    plan = setup_command.build_setup_plan(preview_request, ops=ops)

    with pytest.raises(ValueError, match="does not match"):
        setup_command.run_setup(
            replace(
                preview_request,
                client_ip="10.20.30.41",
                apply=True,
                assume_yes=True,
            ),
            ops=ops,
            plan=plan,
        )

    assert len(calls) == 1
    assert not (tmp_path / "cluster.toml").exists()


def test_reviewed_plan_refuses_tampered_config_snapshot_without_reprobe(tmp_path: Path):
    runner, calls = _runner()
    ops = _ops(runner)
    preview = _request(tmp_path)
    plan = setup_command.build_setup_plan(preview, ops=ops)
    tampered = replace(plan, config_text=plan.config_text + "# unreviewed\n")

    with pytest.raises(ValueError, match="does not match"):
        setup_command.run_setup(
            replace(preview, apply=True, assume_yes=True),
            ops=ops,
            plan=tampered,
        )

    assert len(calls) == 1
    assert not (tmp_path / "cluster.toml").exists()


def test_reviewed_plan_refuses_a_tampered_display_diff_without_reprobe(tmp_path: Path):
    runner, calls = _runner()
    ops = _ops(runner)
    preview = _request(tmp_path)
    plan = setup_command.build_setup_plan(preview, ops=ops)

    with pytest.raises(ValueError, match="does not match"):
        setup_command.run_setup(
            replace(preview, apply=True, assume_yes=True),
            ops=ops,
            plan=replace(plan, diff="no changes\n"),
        )

    assert len(calls) == 1
    assert not (tmp_path / "cluster.toml").exists()


def test_apply_requires_confirmation_and_yes_bypasses_only_that_prompt(tmp_path: Path):
    runner, _calls = _runner()
    denied = setup_command.run_setup(_request(tmp_path, apply=True), ops=_ops(runner, confirm=lambda _prompt: False))
    assert denied.failure == "confirmation_declined"
    assert not (tmp_path / "cluster.toml").exists()

    runner, _calls = _runner()
    prompted: list[str] = []
    accepted = setup_command.run_setup(
        _request(tmp_path, apply=True, assume_yes=True),
        ops=_ops(runner, confirm=lambda prompt: prompted.append(prompt) or False),
    )
    assert accepted.status == "succeeded"
    assert prompted == []
    assert stat.S_IMODE((tmp_path / "cluster.toml").stat().st_mode) == 0o600
    assert load_cluster_config(tmp_path / "cluster.toml").worker_args["slab_weights"] is False
    assert accepted.receipt_written is True


def test_malformed_confirmation_value_is_a_sanitized_decline(tmp_path: Path):
    runner, _calls = _runner()
    outcome = setup_command.run_setup(
        _request(tmp_path, apply=True),
        ops=_ops(runner, confirm=lambda _prompt: "/secret/confirmation"),
    )

    assert outcome.failure == "confirmation_declined"
    assert not (tmp_path / "cluster.toml").exists()
    assert "/secret" not in json.dumps(outcome.as_dict())


def test_config_lock_refusal_is_receipted_before_mutation(tmp_path: Path):
    runner, _calls = _runner()
    written: list[dict[str, object]] = []

    def unavailable(_path):
        raise OSError("/secret/lock/path")

    outcome = setup_command.run_setup(
        _request(tmp_path, apply=True, assume_yes=True),
        ops=_ops(runner, config_lock=unavailable, written=written),
    )

    assert outcome.status == "failed"
    assert outcome.failure == "config_lock_unavailable"
    assert outcome.applied is False
    assert not (tmp_path / "cluster.toml").exists()
    assert len(written) == 1
    assert any(step["name"] == "config_lock_unavailable" for step in written[0]["steps"])
    assert "/secret" not in json.dumps(outcome.as_dict())


def test_config_lock_cancellation_is_receipted_and_re_raised_by_identity(tmp_path: Path):
    runner, _calls = _runner()
    interrupt = KeyboardInterrupt()
    written: list[dict[str, object]] = []

    def cancelled(_path):
        raise interrupt

    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(tmp_path, apply=True, assume_yes=True),
            ops=_ops(runner, config_lock=cancelled, written=written),
        )

    assert raised.value is interrupt
    assert not (tmp_path / "cluster.toml").exists()
    assert len(written) == 1
    assert any(step["name"] == "config_lock_unavailable" for step in written[0]["steps"])


def test_denied_replacement_preserves_existing_config_byte_for_byte(tmp_path: Path):
    target = tmp_path / "cluster.toml"
    original = b"# operator-owned formatting and comments\n"
    target.write_bytes(original)
    runner, _calls = _runner()

    outcome = setup_command.run_setup(_request(tmp_path, apply=True), ops=_ops(runner, confirm=lambda _prompt: False))

    assert outcome.failure == "confirmation_declined"
    assert target.read_bytes() == original


def test_plan_refuses_a_lossy_diff_of_non_utf8_existing_bytes(tmp_path: Path):
    target = tmp_path / "cluster.toml"
    original = b"\xffoperator-owned\n"
    target.write_bytes(original)
    runner, _calls = _runner()

    with pytest.raises(OSError, match="not valid UTF-8"):
        setup_command.build_setup_plan(_request(tmp_path), ops=_ops(runner))

    assert target.read_bytes() == original


def test_apply_preserves_private_backup_and_rollback_metadata(tmp_path: Path):
    target = tmp_path / "cluster.toml"
    target.write_text("old private config\n", encoding="utf-8")
    target.chmod(0o640)
    snapshot = setup_config_io.read_snapshot(target)

    mutation = setup_config_io.apply_config(target, "new config\n", expected=snapshot)

    assert mutation.changed is True
    assert mutation.rollback_available is True
    assert mutation.backup_path is not None and mutation.backup_path.read_bytes() == snapshot.content
    assert stat.S_IMODE(mutation.backup_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert setup_config_io.rollback_config(mutation) is True
    assert target.read_bytes() == snapshot.content
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_apply_never_repurposes_an_existing_nonprivate_backup(tmp_path: Path):
    target = tmp_path / "cluster.toml"
    target.write_text("operator config\n", encoding="utf-8")
    snapshot = setup_config_io.read_snapshot(target)
    backup = target.with_name(f".{target.name}.setup-backup-{snapshot.digest[:12]}")
    backup.write_bytes(snapshot.content)
    backup.chmod(0o644)

    with pytest.raises(OSError, match="backup is not private"):
        setup_config_io.apply_config(target, "planned config\n", expected=snapshot)

    assert target.read_bytes() == snapshot.content
    assert stat.S_IMODE(backup.stat().st_mode) == 0o644


def test_atomic_apply_refuses_a_changed_target(tmp_path: Path):
    target = tmp_path / "cluster.toml"
    target.write_text("first\n")
    snapshot = setup_config_io.read_snapshot(target)
    target.write_text("second\n")
    with pytest.raises(OSError, match="changed"):
        setup_config_io.apply_config(target, "planned\n", expected=snapshot)
    assert target.read_text() == "second\n"


def test_absent_config_publication_never_overwrites_a_raced_file(tmp_path: Path, monkeypatch):
    target = tmp_path / "cluster.toml"
    snapshot = setup_config_io.read_snapshot(target)
    real_write = setup_config_io._write_exclusive

    def raced_write(path, payload, mode):
        identity = real_write(path, payload, mode)
        if ".tmp-" in path.name:
            target.write_bytes(b"foreign\n")
        return identity

    monkeypatch.setattr(setup_config_io, "_write_exclusive", raced_write)
    with pytest.raises(OSError, match="appeared immediately"):
        setup_config_io.apply_config(target, "planned\n", expected=snapshot)

    assert target.read_bytes() == b"foreign\n"


def test_existing_config_publication_preserves_a_raced_replacement(tmp_path: Path, monkeypatch):
    target = tmp_path / "cluster.toml"
    target.write_bytes(b"reviewed\n")
    snapshot = setup_config_io.read_snapshot(target)
    real_write = setup_config_io._write_exclusive

    def raced_write(path, payload, mode):
        identity = real_write(path, payload, mode)
        if ".tmp-" in path.name:
            replacement = tmp_path / "foreign"
            replacement.write_bytes(b"foreign\n")
            os.replace(replacement, target)
        return identity

    monkeypatch.setattr(setup_config_io, "_write_exclusive", raced_write)
    with pytest.raises(OSError, match="changed immediately"):
        setup_config_io.apply_config(target, "planned\n", expected=snapshot)

    assert target.read_bytes() == b"foreign\n"


def test_no_change_transaction_recovery_requires_the_reviewed_inode(tmp_path: Path):
    target = tmp_path / "cluster.toml"
    target.write_bytes(b"same\n")
    target.chmod(0o600)
    snapshot = setup_config_io.read_snapshot(target)
    transaction = setup_config_transaction.ConfigTransaction(
        target,
        "same\n",
        snapshot,
        setup_config_io.apply_config,
    )
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"foreign\n")
    os.replace(replacement, target)

    assert transaction.recover_mutation() is None
    assert transaction.unchanged() is False


def test_apply_rejects_a_forged_success_without_published_config(tmp_path: Path):
    runner, _calls = _runner()
    services: list[str] = []

    def forged(path, text, *, expected, _transaction=None):
        return setup_config_io.ConfigMutation(
            path,
            True,
            expected,
            hashlib.sha256(text.encode()).hexdigest(),
            None,
            (1, 2),
            True,
        )

    outcome = setup_command.run_setup(
        _request(tmp_path, apply=True, assume_yes=True),
        ops=_ops(
            runner,
            apply_config=forged,
            execute_services=lambda _request: services.append("service") or None,
        ),
    )

    assert outcome.status == "partial"
    assert outcome.failure == "apply_failed"
    assert outcome.applied is True
    assert services == []
    assert not (tmp_path / "cluster.toml").exists()


def test_custom_config_publisher_cancellation_recovers_and_rolls_back(tmp_path: Path):
    runner, _calls = _runner()
    interrupt = KeyboardInterrupt()
    written: list[dict[str, object]] = []

    def cancelled(path, text, *, expected, _transaction):
        setup_config_io.apply_config(
            path,
            text,
            expected=expected,
            _transaction=_transaction,
        )
        raise interrupt

    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(tmp_path, apply=True, assume_yes=True),
            ops=_ops(runner, apply_config=cancelled, written=written),
        )

    assert raised.value is interrupt
    assert not (tmp_path / "cluster.toml").exists()
    assert len(written) == 1
    assert written[0]["status"] == "failed"
    assert any(step["name"] == "interrupted" for step in written[0]["steps"])


def test_config_mutation_constructor_cancellation_is_recovered_by_intent(tmp_path: Path, monkeypatch):
    runner, _calls = _runner()
    interrupt = KeyboardInterrupt()
    written: list[dict[str, object]] = []

    def cancelled_record(*_args, **_kwargs):
        raise interrupt

    monkeypatch.setattr(setup_config_io, "ConfigMutation", cancelled_record)
    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(tmp_path, apply=True, assume_yes=True),
            ops=_ops(runner, written=written),
        )

    assert raised.value is interrupt
    assert not (tmp_path / "cluster.toml").exists()
    assert len(written) == 1


def test_atomic_apply_restores_prior_bytes_when_cancelled_after_publish(tmp_path: Path, monkeypatch):
    target = tmp_path / "cluster.toml"
    target.write_text("operator config\n", encoding="utf-8")
    target.chmod(0o640)
    snapshot = setup_config_io.read_snapshot(target)
    interrupt = KeyboardInterrupt()
    real_fsync = setup_config_io._fsync_directory
    calls = 0

    def interrupt_publication(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interrupt
        real_fsync(path)

    monkeypatch.setattr(setup_config_io, "_fsync_directory", interrupt_publication)

    with pytest.raises(KeyboardInterrupt) as raised:
        setup_config_io.apply_config(target, "planned config\n", expected=snapshot)

    assert raised.value is interrupt
    assert target.read_bytes() == snapshot.content
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_atomic_apply_preserves_primary_when_temp_cleanup_is_interrupted(tmp_path: Path, monkeypatch):
    target = tmp_path / "cluster.toml"
    snapshot = setup_config_io.read_snapshot(target)
    primary = KeyboardInterrupt()
    secondary = SystemExit(4)
    real_unlink = Path.unlink

    class Transaction:
        def note_prepared(self, _identity, _backup):
            raise primary

    def interrupted_unlink(path, *args, **kwargs):
        if ".tmp-" in path.name:
            raise secondary
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupted_unlink)
    with pytest.raises(KeyboardInterrupt) as raised:
        setup_config_io.apply_config(
            target,
            "planned\n",
            expected=snapshot,
            _transaction=Transaction(),
        )

    assert raised.value is primary
    assert not target.exists()


def test_exclusive_write_fstat_cancellation_closes_and_removes_created_inode(tmp_path: Path, monkeypatch):
    target = tmp_path / "exclusive"
    primary = KeyboardInterrupt()
    real_fstat = setup_config_io.os.fstat
    descriptors: list[int] = []

    def interrupted_fstat(descriptor):
        descriptors.append(descriptor)
        if len(descriptors) == 1:
            raise primary
        return real_fstat(descriptor)

    monkeypatch.setattr(setup_config_io.os, "fstat", interrupted_fstat)
    with pytest.raises(KeyboardInterrupt) as raised:
        setup_config_io._write_exclusive(target, b"planned\n", 0o600)

    assert raised.value is primary
    assert not target.exists()
    monkeypatch.setattr(setup_config_io.os, "fstat", real_fstat)
    with pytest.raises(OSError):
        real_fstat(descriptors[0])


def test_private_roundtrip_cancellation_closes_and_removes_temp(tmp_path: Path, monkeypatch):
    primary = KeyboardInterrupt()

    def interrupted_fchmod(_descriptor, _mode):
        raise primary

    monkeypatch.setattr(setup_config_io.os, "fchmod", interrupted_fchmod)
    with pytest.raises(KeyboardInterrupt) as raised:
        setup_config_io.private_roundtrip("[cluster]\n", temp_dir=tmp_path)

    assert raised.value is primary
    assert list(tmp_path.iterdir()) == []


def test_apply_rolls_back_if_post_publish_strict_load_changes_meaning(tmp_path: Path):
    runner, _calls = _runner()
    calls = 0

    def roundtrip(text):
        nonlocal calls
        calls += 1
        parsed = setup_config_io.private_roundtrip(text)
        return replace(parsed, auto_heal=not parsed.auto_heal) if calls == 3 else parsed

    outcome = setup_command.run_setup(
        _request(tmp_path, apply=True, assume_yes=True),
        ops=_ops(runner, roundtrip=roundtrip),
    )

    assert calls == 3
    assert outcome.status == "failed"
    assert outcome.failure == "apply_failed"
    assert not (tmp_path / "cluster.toml").exists()


def test_rollback_refuses_a_same_byte_replacement_from_another_writer(tmp_path: Path):
    target = tmp_path / "cluster.toml"
    snapshot = setup_config_io.read_snapshot(target)
    mutation = setup_config_io.apply_config(target, "planned\n", expected=snapshot)
    replacement = tmp_path / "replacement"
    replacement.write_text("planned\n")
    os.replace(replacement, target)

    assert setup_config_io.rollback_config(mutation) is False
    assert target.read_text() == "planned\n"


def test_worker_start_requires_install_in_the_same_transaction(tmp_path: Path):
    with pytest.raises(ValueError, match="same setup transaction"):
        _request(tmp_path, start_workers=True)

    assert _request(tmp_path, install_service=True).start_workers is False


@pytest.mark.parametrize(
    "field",
    ["apply", "assume_yes", "install_service", "start_workers", "verify", "privileged_process_inspection"],
)
def test_backend_request_rejects_truthy_non_boolean_authority(tmp_path: Path, field: str):
    with pytest.raises(ValueError, match="booleans"):
        replace(_request(tmp_path), **{field: "/secret/authority"})


@pytest.mark.parametrize("start", [False, True])
def test_service_transaction_receives_exact_opt_in_authority(tmp_path: Path, start: bool):
    runner, _calls = _runner()
    seen: list[tuple[bool, bool, bool]] = []

    def execute(request):
        seen.append((request.install_service, request.start_service, request.privileged_process_inspection))
        return _successful_services(request)

    outcome = setup_command.run_setup(
        _request(
            tmp_path,
            apply=True,
            assume_yes=True,
            install_service=True,
            start_workers=start,
            privileged_process_inspection=True,
        ),
        ops=_ops(runner, execute_services=execute),
    )

    assert outcome.status == "succeeded"
    assert outcome.services_changed is True
    assert seen == [(True, start, True)]
    assert (tmp_path / "cluster.toml").exists()


def test_reviewed_plan_refuses_changed_process_inspection_selection_without_reprobe(tmp_path: Path):
    runner, calls = _runner()
    ops = _ops(runner)
    preview = _request(tmp_path)
    plan = setup_command.build_setup_plan(preview, ops=ops)

    with pytest.raises(ValueError, match="does not match"):
        setup_command.run_setup(
            replace(preview, apply=True, assume_yes=True, privileged_process_inspection=True),
            ops=ops,
            plan=plan,
        )

    assert len(calls) == 1
    assert not (tmp_path / "cluster.toml").exists()


def test_plan_and_receipt_record_process_inspection_selection(tmp_path: Path):
    runner, _calls = _runner()
    outcome = setup_command.run_setup(
        _request(tmp_path, privileged_process_inspection=True),
        ops=_ops(runner),
    )

    assert outcome.plan.as_dict()["privileged_process_inspection"] is True
    config_plan = next(step for step in outcome.receipt["steps"] if step["name"] == "config_plan")
    assert any(note.startswith("Privileged process inspection: requested") for note in config_plan["notes"])


def test_service_safety_transaction_precedes_config_publication(tmp_path: Path) -> None:
    runner, _calls = _runner()
    target = tmp_path / "cluster.toml"
    events: list[str] = []

    def execute(request):
        assert not target.exists()
        events.append("service")
        return _successful_services(request)

    def publish(path, text, *, expected, _transaction=None):
        assert events == ["service"]
        events.append("config")
        return setup_config_io.apply_config(path, text, expected=expected, _transaction=_transaction)

    outcome = setup_command.run_setup(
        _request(tmp_path, apply=True, assume_yes=True, install_service=True),
        ops=_ops(runner, execute_services=execute, apply_config=publish),
    )

    assert outcome.status == "succeeded"
    assert events == ["service", "config"]
    assert target.exists()


@pytest.mark.parametrize(
    ("payload", "changes", "blocker"),
    [
        (
            _payload(service_active=True),
            {"install_service": True, "start_workers": True},
            "host_1_worker_state_not_owned",
        ),
        (
            _payload(service_installed=True),
            {"install_service": True, "start_workers": True},
            "host_1_service_not_owned",
        ),
        (
            _payload(service_installed=True),
            {"install_service": True, "start_workers": True},
            "host_1_service_not_owned",
        ),
        (
            _payload(service_active=None),
            {"install_service": True, "start_workers": True},
            "host_1_worker_state_not_owned",
        ),
    ],
)
def test_setup_never_mutates_preexisting_service_state(
    tmp_path: Path, payload: dict[str, object], changes: dict[str, bool], blocker: str
):
    runner, _calls = _runner([payload])
    mutations: list[str] = []
    outcome = setup_command.run_setup(
        _request(tmp_path, apply=True, assume_yes=True, **changes),
        ops=_ops(
            runner,
            execute_services=lambda _request: mutations.append("service") or None,
        ),
    )
    assert blocker in outcome.plan.blockers
    assert outcome.failure == "readiness_blocked"
    assert mutations == []


def test_apply_requires_explicit_trusted_fabric_acknowledgement(tmp_path: Path):
    runner, _calls = _runner()
    outcome = setup_command.run_setup(
        _request(tmp_path, transport_security="", apply=True, assume_yes=True),
        ops=_ops(runner),
    )
    assert "trusted_fabric_not_acknowledged" in outcome.plan.blockers
    assert outcome.failure == "readiness_blocked"
    assert not (tmp_path / "cluster.toml").exists()


def test_failed_service_transaction_with_exact_cleanup_rolls_back_config(tmp_path: Path):
    runner, _calls = _runner()
    calls: list[str] = []

    def fail(request):
        calls.append("execute")
        return _service_result(
            request,
            certainty=setup_services.ServiceCertainty.FAILED,
            code="activation_failed",
            changed=True,
            compensated=True,
        )

    outcome = setup_command.run_setup(
        _request(tmp_path, apply=True, assume_yes=True, install_service=True, start_workers=True),
        ops=_ops(runner, execute_services=fail),
    )

    assert outcome.status == "failed"
    assert outcome.failure == "service_activation_failed"
    assert calls == ["execute"]
    assert not (tmp_path / "cluster.toml").exists()
    assert "No service change remains, and the config was not published" in json.dumps(outcome.receipt)
    service_step = next(step for step in outcome.receipt["steps"] if step["name"] == "service_transaction")
    assert service_step["status"] == "failed"
    assert service_step["counts"] == {
        "activated": 0,
        "cleaned": 1,
        "compensated": 1,
        "failed": 1,
        "hosts": 1,
        "staged": 1,
        "unknown": 0,
    }
    compensation_step = next(
        step for step in outcome.receipt["steps"] if step["name"] == "setup_compensation"
    )
    assert compensation_step["counts"] == {"config_unchanged": 1, "service_compensated": 1}
    assert any(step["name"] == "service_activation_failed" for step in outcome.receipt["steps"])


def test_unconfirmed_worker_compensation_never_publishes_config_and_reports_partial(tmp_path: Path):
    runner, _calls = _runner()

    def fail(request):
        return _service_result(
            request,
            certainty=setup_services.ServiceCertainty.UNKNOWN,
            code="activation_unknown",
            changed=True,
            compensated=False,
        )

    outcome = setup_command.run_setup(
        _request(tmp_path, apply=True, assume_yes=True, install_service=True, start_workers=True),
        ops=_ops(runner, execute_services=fail),
    )

    assert outcome.status == "partial"
    assert outcome.failure == "service_activation_unknown"
    assert outcome.applied is False
    assert not (tmp_path / "cluster.toml").exists()


def test_ambiguous_worker_start_exception_is_not_compensated_as_owned(tmp_path: Path):
    runner, _calls = _runner()
    calls: list[str] = []

    def ambiguous_start(_request):
        calls.append("execute")
        raise RuntimeError("secret remote state")

    outcome = setup_command.run_setup(
        _request(tmp_path, apply=True, assume_yes=True, install_service=True, start_workers=True),
        ops=_ops(runner, execute_services=ambiguous_start),
    )

    assert outcome.status == "partial"
    assert outcome.failure == "apply_failed"
    assert calls == ["execute"]
    assert not (tmp_path / "cluster.toml").exists()
    assert "secret remote state" not in json.dumps(outcome.as_dict())


def test_unsettled_service_cancellation_is_resettled_before_exact_reraise(tmp_path: Path) -> None:
    runner, _calls = _runner()
    interrupt = KeyboardInterrupt()
    calls: list[str] = []

    def cancelled(request):
        calls.append("execute")
        return _service_result(
            request,
            certainty=setup_services.ServiceCertainty.UNKNOWN,
            code="interrupted",
            changed=True,
            compensated=False,
            interruption=interrupt,
        )

    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(tmp_path, apply=True, assume_yes=True, install_service=True),
            ops=_ops(
                runner,
                execute_services=cancelled,
                compensate_services=lambda request, result: (
                    calls.append("compensate") or _compensated_services(request, result)
                ),
            ),
        )

    assert raised.value is interrupt
    assert calls == ["execute", "compensate"]
    assert not (tmp_path / "cluster.toml").exists()


def test_service_cancellation_is_settled_receipted_and_re_raised_by_identity(tmp_path: Path):
    runner, _calls = _runner()
    interrupt = KeyboardInterrupt()
    written: list[dict[str, object]] = []

    def cancelled(request):
        return _service_result(
            request,
            certainty=setup_services.ServiceCertainty.FAILED,
            code="interrupted",
            changed=True,
            compensated=True,
            interruption=interrupt,
        )

    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(tmp_path, apply=True, assume_yes=True, install_service=True),
            ops=_ops(runner, execute_services=cancelled, written=written),
        )

    assert raised.value is interrupt
    assert not (tmp_path / "cluster.toml").exists()
    assert len(written) == 1
    assert written[0]["status"] == "failed"
    assert any(step["name"] == "interrupted" for step in written[0]["steps"])


def test_cancellation_after_owned_start_compensates_before_re_raise(tmp_path: Path):
    runner, _calls = _runner()
    interrupt = KeyboardInterrupt()
    calls: list[str] = []
    written: list[dict[str, object]] = []

    def doctor(_config):
        calls.append("doctor")
        raise interrupt

    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(
                tmp_path,
                apply=True,
                assume_yes=True,
                install_service=True,
                start_workers=True,
                verify=True,
            ),
            ops=_ops(
                runner,
                execute_services=lambda request: calls.append("service") or _successful_services(request),
                compensate_services=lambda request, result: (
                    calls.append("compensate") or _compensated_services(request, result)
                ),
                run_doctor=doctor,
                written=written,
            ),
        )

    assert raised.value is interrupt
    assert calls == ["service", "doctor", "compensate"]
    assert not (tmp_path / "cluster.toml").exists()
    assert len(written) == 1
    assert written[0]["status"] == "failed"
    steps = {step["name"]: step for step in written[0]["steps"]}
    assert steps["doctor_verify"]["status"] == "unknown"
    assert steps["doctor_verify"]["counts"] == {"attempted": 1, "passed": 0}
    assert steps["cluster_smoke"]["status"] == "planned"
    assert steps["cluster_smoke"]["counts"]["attempted"] == 0


def test_post_compensation_cancellation_rolls_back_config_and_preserves_identity(tmp_path: Path) -> None:
    runner, _calls = _runner()
    interrupt = KeyboardInterrupt()
    calls: list[str] = []

    def interrupted_compensation(request, _result):
        calls.append("compensate")
        return _service_result(
            request,
            certainty=setup_services.ServiceCertainty.FAILED,
            code="interrupted",
            changed=True,
            compensated=True,
            interruption=interrupt,
        )

    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(
                tmp_path,
                apply=True,
                assume_yes=True,
                install_service=True,
                start_workers=True,
                verify=True,
            ),
            ops=_ops(
                runner,
                execute_services=lambda request: calls.append("service") or _successful_services(request),
                compensate_services=interrupted_compensation,
                run_doctor=lambda _config: calls.append("doctor") or False,
            ),
        )

    assert raised.value is interrupt
    assert calls == ["service", "doctor", "compensate"]
    assert not (tmp_path / "cluster.toml").exists()


def test_verification_receipt_error_does_not_replace_primary_cancellation(tmp_path: Path):
    class Builder:
        def __init__(self, *args, **kwargs):
            self.inner = operator_receipt.OperatorReceiptBuilder(*args, **kwargs)

        def add_step(self, name, status, **kwargs):
            if name == "doctor_verify":
                raise RuntimeError("/secret/receipt")
            self.inner.add_step(name, status, **kwargs)

        def finish(self, status, **kwargs):
            return self.inner.finish(status, **kwargs)

    runner, _calls = _runner()
    primary = KeyboardInterrupt()
    written: list[dict[str, object]] = []

    def cancelled_doctor(_config):
        raise primary

    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(
                tmp_path,
                apply=True,
                assume_yes=True,
                install_service=True,
                start_workers=True,
                verify=True,
            ),
            ops=_ops(
                runner,
                receipt_builder=Builder,
                run_doctor=cancelled_doctor,
                written=written,
            ),
        )

    assert raised.value is primary
    assert not (tmp_path / "cluster.toml").exists()
    assert any(step["name"] == "verification_evidence_unavailable" for step in written[0]["steps"])
    assert "/secret" not in json.dumps(written[0])


def test_settlement_receipt_cancellation_does_not_replace_primary_cancellation(tmp_path: Path):
    secondary = SystemExit(9)

    class Builder:
        def __init__(self, *args, **kwargs):
            self.inner = operator_receipt.OperatorReceiptBuilder(*args, **kwargs)

        def add_step(self, name, status, **kwargs):
            if name == "setup_compensation":
                raise secondary
            self.inner.add_step(name, status, **kwargs)

        def finish(self, status, **kwargs):
            return self.inner.finish(status, **kwargs)

    runner, _calls = _runner()
    primary = KeyboardInterrupt()
    written: list[dict[str, object]] = []

    def cancelled_doctor(_config):
        raise primary

    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(
                tmp_path,
                apply=True,
                assume_yes=True,
                install_service=True,
                start_workers=True,
                verify=True,
            ),
            ops=_ops(
                runner,
                receipt_builder=Builder,
                run_doctor=cancelled_doctor,
                written=written,
            ),
        )

    assert raised.value is primary
    assert not (tmp_path / "cluster.toml").exists()
    assert len(written) == 1
    assert any(step["name"] == "receipt_recovery" for step in written[0]["steps"])


def test_receipt_publication_cancellation_recovers_receipt_then_re_raises(tmp_path: Path):
    runner, _calls = _runner()
    interrupt = KeyboardInterrupt()
    calls: list[str] = []
    written: list[dict[str, object]] = []
    rolled_back: list[bool] = []

    def writer(receipt, _path=None):
        calls.append("receipt")
        if len(calls) == 2:
            raise interrupt
        written.append(dict(receipt))
        return Path("/not/exposed/receipt.json")

    def rollback(mutation):
        result = setup_config_io.rollback_config(mutation)
        rolled_back.append(result)
        return result

    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(tmp_path, apply=True, assume_yes=True, install_service=True),
            ops=_ops(
                runner,
                execute_services=lambda request: calls.append("service") or _successful_services(request),
                compensate_services=lambda request, result: (
                    calls.append("compensate") or _compensated_services(request, result)
                ),
                write_receipt=writer,
                rollback_config=rollback,
            ),
        )

    assert raised.value is interrupt
    assert calls == ["service", "receipt", "compensate", "receipt"]
    assert rolled_back == [True]
    assert not (tmp_path / "cluster.toml").exists()
    assert len(written) == 1
    assert written[0]["status"] == "partial"
    assert any(step["name"] == "receipt_recovery" for step in written[0]["steps"])


def test_cancellation_after_success_receipt_link_rolls_back_and_replaces_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _calls = _runner()
    receipt_path = tmp_path / "setup-receipt.json"
    interrupt = KeyboardInterrupt()
    real_link = operator_receipt.os.link
    receipt_links = 0

    def interrupted_first_receipt_link(*args, **kwargs):
        nonlocal receipt_links
        real_link(*args, **kwargs)
        if kwargs.get("src_dir_fd") is not None:
            receipt_links += 1
            if receipt_links == 1:
                raise interrupt

    monkeypatch.setattr(operator_receipt.os, "link", interrupted_first_receipt_link)
    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(tmp_path, apply=True, assume_yes=True, receipt_path=receipt_path),
            ops=_ops(runner, write_receipt=operator_receipt.write_receipt),
        )

    assert raised.value is interrupt
    assert not (tmp_path / "cluster.toml").exists()
    payload = json.loads(receipt_path.read_text(encoding="ascii"))
    assert payload["status"] == "partial"
    assert any(step["name"] == "setup_compensation" for step in payload["steps"])


def test_artifact_or_dirty_source_mismatch_blocks_before_any_mutation(tmp_path: Path):
    dirty = _payload(dirty=True)
    dirty["artifacts"] = [{"ordinal": 1, "exists": True, "regular": True, "size": 1, "sha256": "a" * 64}]
    runner, _calls = _runner([dirty])
    started: list[bool] = []
    request = _request(
        tmp_path,
        artifacts=("models/a.safetensors",),
        apply=True,
        assume_yes=True,
        install_service=True,
        start_workers=True,
    )

    outcome = setup_command.run_setup(
        request,
        ops=_ops(runner, execute_services=lambda _request: started.append(True) or None),
    )

    assert outcome.status == "failed"
    assert "host_1_comfy_dirty" in outcome.plan.blockers
    assert started == []
    assert not (tmp_path / "cluster.toml").exists()


@pytest.mark.parametrize(
    ("payloads", "changes", "blocker"),
    [
        ([_payload()], {"verify": True}, "verify_requires_worker_start"),
        (
            [_payload(service_active=None)],
            {"verify": True, "install_service": True, "start_workers": True},
            "host_1_worker_state_not_owned",
        ),
        (
            [_payload(service_active=False), _payload(service_active=True)],
            {
                "hosts": (
                    setup_command.CandidateHost("secret-a", "10.20.30.40"),
                    setup_command.CandidateHost("secret-b", "10.20.30.41"),
                ),
                "verify": True,
                "install_service": True,
                "start_workers": True,
            },
            "host_2_worker_state_not_owned",
        ),
    ],
)
def test_verify_refuses_without_owned_explicit_start_authority(
    tmp_path: Path,
    payloads: list[dict[str, object]],
    changes: dict[str, object],
    blocker: str,
):
    runner, _calls = _runner(payloads)
    events: list[str] = []
    outcome = setup_command.run_setup(
        _request(tmp_path, apply=True, assume_yes=True, **changes),
        ops=_ops(
            runner,
            execute_services=lambda _request: events.append("service") or None,
            run_doctor=lambda _config: events.append("doctor") or True,
            run_smoke=lambda _path, _comfy: events.append("smoke") or True,
        ),
    )

    assert blocker in outcome.plan.blockers
    assert outcome.failure == "readiness_blocked"
    assert events == []
    assert not (tmp_path / "cluster.toml").exists()


def test_verify_runs_doctor_then_smoke_and_compensates_on_smoke_failure(tmp_path: Path):
    runner, _calls = _runner()
    calls: list[str] = []

    def smoke(path, comfy_dir):
        assert path == tmp_path / "cluster.toml"
        assert comfy_dir == ""
        calls.append("smoke")
        return {"result": "FAIL", "error": "sample_failed"}

    outcome = setup_command.run_setup(
        _request(
            tmp_path,
            apply=True,
            assume_yes=True,
            install_service=True,
            start_workers=True,
            verify=True,
        ),
        ops=_ops(
            runner,
            execute_services=lambda request: calls.append("service") or _successful_services(request),
            compensate_services=lambda request, result: (
                calls.append("compensate") or _compensated_services(request, result)
            ),
            run_smoke=smoke,
            run_doctor=lambda _config: calls.append("doctor") or True,
        ),
    )

    assert calls == ["service", "doctor", "smoke", "compensate"]
    assert outcome.failure == "verification_failed"
    assert outcome.status == "failed"
    assert not (tmp_path / "cluster.toml").exists()
    assert not hasattr(setup_command.SetupOps, "hosts_shutdown")
    verify_steps = [
        step["name"] for step in outcome.receipt["steps"] if step["name"] in {"doctor_verify", "cluster_smoke"}
    ]
    assert verify_steps == [
        "doctor_verify",
        "cluster_smoke",
    ]


def test_typed_smoke_failure_is_sanitized_negative_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = cluster_smoke.ClusterSmokeError("status_failed")

    def fail(_path: Path, *, comfy_dir: str | None) -> object:
        assert comfy_dir is None
        raise failure

    monkeypatch.setattr(cluster_smoke, "run_cluster_smoke", fail)

    assert setup_verification.run_setup_smoke(tmp_path / "cluster.toml", "") == {
        "result": "FAIL",
        "error": "status_failed",
    }


@pytest.mark.parametrize(
    "smoke_result",
    [
        OSError("/secret/transport"),
        {"result": "PASS", "world": 1, "teardown_confirmed": False},
        {"result": "FAIL", "error": "cleanup_not_confirmed"},
    ],
)
def test_unknown_smoke_evidence_retains_config_and_worker_service(
    tmp_path: Path, smoke_result: object
) -> None:
    runner, _calls = _runner()
    calls: list[str] = []

    def smoke(_path, _comfy):
        calls.append("smoke")
        if isinstance(smoke_result, BaseException):
            raise smoke_result
        return smoke_result

    outcome = setup_command.run_setup(
        _request(
            tmp_path,
            apply=True,
            assume_yes=True,
            install_service=True,
            start_workers=True,
            verify=True,
        ),
        ops=_ops(
            runner,
            execute_services=lambda request: calls.append("service") or _successful_services(request),
            compensate_services=lambda request, result: (
                calls.append("compensate") or _compensated_services(request, result)
            ),
            run_doctor=lambda _config: calls.append("doctor") or True,
            run_smoke=smoke,
        ),
    )

    assert calls == ["service", "doctor", "smoke"]
    assert outcome.status == "partial"
    assert outcome.failure == "verification_unknown"
    assert (tmp_path / "cluster.toml").exists()
    smoke_step = next(step for step in outcome.receipt["steps"] if step["name"] == "cluster_smoke")
    assert smoke_step["status"] == "unknown"
    assert "/secret" not in json.dumps(outcome.as_dict())


def test_exact_failed_doctor_never_enters_cluster_smoke(tmp_path: Path):
    runner, _calls = _runner()
    calls: list[str] = []
    outcome = setup_command.run_setup(
        _request(
            tmp_path,
            apply=True,
            assume_yes=True,
            install_service=True,
            start_workers=True,
            verify=True,
        ),
        ops=_ops(
            runner,
            execute_services=lambda request: calls.append("service") or _successful_services(request),
            compensate_services=lambda request, result: (
                calls.append("compensate") or _compensated_services(request, result)
            ),
            run_doctor=lambda _config: calls.append("doctor") or False,
            run_smoke=lambda _path, _comfy: calls.append("smoke") or True,
        ),
    )

    assert calls == ["service", "doctor", "compensate"]
    assert outcome.status == "failed"
    smoke_step = next(step for step in outcome.receipt["steps"] if step["name"] == "cluster_smoke")
    assert smoke_step["counts"]["attempted"] == 0
    assert smoke_step["counts"]["passed"] == 0
    assert "/secret" not in json.dumps(outcome.as_dict())


@pytest.mark.parametrize("doctor_result", ["/secret/doctor/result", OSError("/secret/transport")])
def test_malformed_or_exceptional_doctor_is_unknown_without_rollback(
    tmp_path: Path, doctor_result: object
) -> None:
    runner, _calls = _runner()
    calls: list[str] = []

    def doctor(_config):
        calls.append("doctor")
        if isinstance(doctor_result, BaseException):
            raise doctor_result
        return doctor_result

    outcome = setup_command.run_setup(
        _request(
            tmp_path,
            apply=True,
            assume_yes=True,
            install_service=True,
            start_workers=True,
            verify=True,
        ),
        ops=_ops(
            runner,
            execute_services=lambda request: calls.append("service") or _successful_services(request),
            compensate_services=lambda request, result: (
                calls.append("compensate") or _compensated_services(request, result)
            ),
            run_doctor=doctor,
            run_smoke=lambda _path, _comfy: calls.append("smoke") or True,
        ),
    )

    assert calls == ["service", "doctor"]
    assert outcome.status == "partial"
    assert outcome.failure == "verification_unknown"
    assert (tmp_path / "cluster.toml").exists()
    steps = {step["name"]: step for step in outcome.receipt["steps"]}
    assert steps["doctor_verify"]["status"] == "unknown"
    assert steps["cluster_smoke"]["status"] == "planned"
    assert "/secret" not in json.dumps(outcome.as_dict())


def test_successful_verify_accepts_typed_smoke_result_and_sanitizes_receipt(tmp_path: Path):
    class Result:
        def as_dict(self):
            return {
                "result": "PASS",
                "world": 1,
                "status_rank_coverage": 1,
                "source_manifest_sha256": "d" * 64,
                "source_rank_coverage": 1,
                "teardown_confirmed": True,
            }

    runner, _calls = _runner()
    outcome = setup_command.run_setup(
        _request(
            tmp_path,
            apply=True,
            assume_yes=True,
            install_service=True,
            start_workers=True,
            verify=True,
        ),
        ops=_ops(
            runner,
            run_smoke=lambda _path, _comfy: Result(),
            run_doctor=lambda _config: True,
        ),
    )

    assert outcome.status == "succeeded"
    assert outcome.verified is True
    encoded = json.dumps(outcome.receipt, sort_keys=True)
    for secret in ("secret-spark", "10.20.30.40", str(tmp_path)):
        assert secret not in encoded


def test_smoke_evidence_requires_literal_strings_not_equality_overloads(tmp_path: Path):
    class Hostile:
        def __eq__(self, _other):
            return True

        def __repr__(self):
            return "/secret/evidence"

    runner, _calls = _runner()
    outcome = setup_command.run_setup(
        _request(
            tmp_path,
            apply=True,
            assume_yes=True,
            install_service=True,
            start_workers=True,
            verify=True,
        ),
        ops=_ops(
            runner,
            run_doctor=lambda _config: True,
            run_smoke=lambda _path, _comfy: {
                "result": Hostile(),
                "world": 1,
                "status_rank_coverage": 1,
                "source_manifest_sha256": Hostile(),
                "source_rank_coverage": 1,
                "teardown_confirmed": True,
            },
        ),
    )

    assert outcome.status == "partial"
    assert outcome.failure == "verification_unknown"
    assert (tmp_path / "cluster.toml").exists()
    assert "/secret" not in json.dumps(outcome.as_dict())
    assert setup_verification.smoke_succeeded({"result": Hostile()}) is False


def test_unknown_verification_receipt_cancellation_never_authorizes_rollback(tmp_path: Path) -> None:
    runner, _calls = _runner()
    interrupt = KeyboardInterrupt()
    calls: list[str] = []

    def writer(_receipt, _path=None):
        calls.append("receipt")
        raise interrupt

    with pytest.raises(KeyboardInterrupt) as raised:
        setup_command.run_setup(
            _request(
                tmp_path,
                apply=True,
                assume_yes=True,
                install_service=True,
                start_workers=True,
                verify=True,
            ),
            ops=_ops(
                runner,
                execute_services=lambda request: calls.append("service") or _successful_services(request),
                compensate_services=lambda request, result: (
                    calls.append("compensate") or _compensated_services(request, result)
                ),
                run_doctor=lambda _config: OSError("malformed"),
                write_receipt=writer,
            ),
        )

    assert raised.value is interrupt
    assert "compensate" not in calls
    assert (tmp_path / "cluster.toml").exists()


def test_required_receipt_publication_failure_is_a_partial_outcome(tmp_path: Path):
    runner, _calls = _runner()

    def fail_receipt(_receipt, _path=None):
        raise OSError("state directory unavailable")

    outcome = setup_command.run_setup(
        _request(tmp_path, apply=True, assume_yes=True),
        ops=_ops(runner, write_receipt=fail_receipt),
    )

    assert outcome.status == "partial"
    assert outcome.failure == "receipt_write_failed"
    assert outcome.receipt_written is False
    assert (tmp_path / "cluster.toml").exists()
