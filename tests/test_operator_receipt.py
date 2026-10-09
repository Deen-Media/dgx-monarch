"""Disclosure and durability contract for shared operator receipts."""
from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dgx_monarch.cli import operator_receipt as receipt

SOURCE = "a" * 64
TARGET = "b" * 40


class _Values:
    def __init__(self, *values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


def _built(*, note: str = "all ranks agreed") -> dict[str, object]:
    started = datetime(2026, 8, 9, 1, 2, 3, 4000, tzinfo=UTC)
    builder = receipt.OperatorReceiptBuilder(
        "update",
        profile="balanced",
        target=TARGET,
        source_hashes={"cluster": SOURCE},
        clock=_Values(started, started + timedelta(seconds=1, microseconds=250000)),
        timer=_Values(10.0, 11.25),
    )
    builder.add_step(
        "source_attestation", "succeeded", counts={"ranks": 2}, notes=[note]
    )
    return builder.finish("succeeded", notes=["source-bound grants will revalidate"])


def test_builder_is_versioned_deterministic_and_uses_injected_clocks():
    first = _built()
    second = _built()

    assert first == second
    assert first["schema"] == "dgx-monarch.operator-receipt"
    assert first["schema_version"] == 2
    assert first["started_at"] == "2026-08-09T01:02:03.004000Z"
    assert first["finished_at"] == "2026-08-09T01:02:04.254000Z"
    assert first["duration_ms"] == 1250
    assert first["summary"] == {
        "total_steps": 1,
        "step_counts": {
            "failed": 0,
            "partial": 0,
            "planned": 0,
            "succeeded": 1,
            "unknown": 0,
        },
    }
    assert receipt.canonical_json(first) == receipt.canonical_json(second)
    assert receipt.canonical_json(first).endswith("\n")


@pytest.mark.parametrize("operation", ["", "doctor-repair", "render", "setup\n"])
def test_operation_vocabulary_is_closed(operation):
    with pytest.raises(ValueError, match="operation"):
        receipt.OperatorReceiptBuilder(operation)  # type: ignore[arg-type]


def test_builder_rejects_unstable_identity_and_count_fields():
    with pytest.raises(ValueError, match="profile"):
        receipt.OperatorReceiptBuilder("setup", profile="safe /home/alice")
    with pytest.raises(ValueError, match="target"):
        receipt.OperatorReceiptBuilder("update", target="origin/main")
    with pytest.raises(ValueError, match="source hash"):
        receipt.OperatorReceiptBuilder("cluster_smoke", source_hashes={"cluster": "not-a-hash"})

    builder = receipt.OperatorReceiptBuilder("doctor_repair")
    with pytest.raises(ValueError, match="step name"):
        builder.add_step("Restart Worker A", "planned")
    with pytest.raises(ValueError, match="non-negative"):
        builder.add_step("repair", "planned", counts={"hosts": -1})
    with pytest.raises(ValueError, match="non-negative"):
        builder.add_step("repair", "planned", counts={"hosts": True})


def test_notes_redact_infrastructure_paths_commands_environment_and_secrets():
    unsafe = (
        "host=worker-01 at 10.20.30.40 and fd00::2; path /home/alice/private/config.toml; "
        "API_TOKEN=ghp_123456789abcdef; argv=['ssh','worker-01']; "
        "RuntimeError('password=hunter2')"
    )
    safe = receipt.sanitize_note(unsafe)

    for leaked in (
        "worker-01",
        "10.20.30.40",
        "fd00::2",
        "/home/alice",
        "ghp_",
        "ssh",
        "hunter2",
        "RuntimeError(",
    ):
        assert leaked not in safe
    assert "redacted" in safe


def test_notes_distinguish_dotted_identifiers_from_hosts_and_relative_paths():
    safe = receipt.sanitize_note(
        "source-bound evidence remains; report.json, foo.py, model.safetensors, "
        "torch.float16, torch.nn, and "
        "dgx_monarch.cli are benign; worker.example.internal failed; "
        "inspect private/rig/cluster.toml ../operator/cluster.toml logs/worker.log"
    )

    assert "report.json" in safe
    assert "foo.py" in safe
    assert "model.safetensors" in safe
    assert "torch.float16" in safe
    assert "torch.nn" in safe
    assert "dgx_monarch.cli" in safe
    assert "worker.example.internal" not in safe
    assert "private/rig/cluster.toml" not in safe
    assert "../operator/cluster.toml" not in safe
    assert "logs/worker.log" not in safe
    assert "source-bound" in safe
    assert "[path redacted]" in safe
    assert "[network identity redacted]" in safe


def test_notes_redact_arbitrary_two_label_private_hosts():
    private_hosts = (
        "zeus.lab", "apollo.cluster", "controller.corp", "render.private",
        "box.home", "atlas.uk",
    )
    safe = receipt.sanitize_note(" ".join(private_hosts))

    for private_host in private_hosts:
        assert private_host not in safe
    assert safe.count("[network identity redacted]") == len(private_hosts)


def test_notes_redact_unenumerated_dotted_namespace_values():
    private_hosts = (
        "torch.private", "torch.nn.private",
        "dgx_monarch.private", "dgx_monarch.cli.private",
    )
    safe = receipt.sanitize_note(" ".join(private_hosts))

    for private_host in private_hosts:
        assert private_host not in safe
    assert safe.count("[network identity redacted]") >= len(private_hosts)


def test_schema_v2_counts_unknown_steps_and_v1_receipts_still_validate():
    builder = receipt.OperatorReceiptBuilder("update", target=TARGET)
    builder.add_step("worker_start", "unknown")
    current = builder.finish("partial")

    assert current["schema_version"] == 2
    assert current["summary"]["step_counts"]["unknown"] == 1

    legacy = _built()
    legacy["schema_version"] = 1
    legacy["summary"]["step_counts"].pop("unknown")
    validated = receipt.validate_receipt(legacy)
    assert validated["schema_version"] == 1
    assert "unknown" not in validated["summary"]["step_counts"]

    legacy_with_unknown = json.loads(json.dumps(legacy))
    legacy_with_unknown["steps"][0]["status"] = "unknown"
    legacy_with_unknown["summary"]["step_counts"]["succeeded"] = 0
    with pytest.raises(ValueError, match="step status"):
        receipt.validate_receipt(legacy_with_unknown)


def test_exception_objects_and_scalar_notes_are_refused():
    builder = receipt.OperatorReceiptBuilder("setup")
    with pytest.raises(TypeError, match="sequence"):
        builder.add_step("probe", "failed", notes="not a note list")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="strings"):
        builder.add_step("probe", "failed", notes=[RuntimeError("secret")])  # type: ignore[list-item]


def test_validator_rejects_unknown_fields_and_unsanitized_injected_notes():
    valid = _built()
    unknown = dict(valid, hosts=["worker-01"])
    with pytest.raises(ValueError, match="top-level"):
        receipt.validate_receipt(unknown)

    injected = json.loads(json.dumps(valid))
    injected["steps"][0]["notes"] = ["host=worker-01"]
    with pytest.raises(ValueError, match="unsanitized"):
        receipt.validate_receipt(injected)


def test_finished_builder_is_immutable():
    builder = receipt.OperatorReceiptBuilder("setup")
    builder.finish("planned")
    with pytest.raises(RuntimeError, match="finished"):
        builder.add_step("probe", "planned")
    with pytest.raises(RuntimeError, match="already"):
        builder.finish("planned")


def test_write_is_private_atomic_and_deterministic(tmp_path: Path):
    target = tmp_path / "receipts" / "update.json"
    written = receipt.write_receipt(_built(), target)

    assert written == target
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert target.read_text(encoding="ascii") == receipt.canonical_json(_built())
    assert target.stat().st_nlink == 1
    assert not list(target.parent.glob(".*.tmp-*"))


def test_write_never_replaces_an_existing_file_or_symlink(tmp_path: Path):
    parent = tmp_path / "receipts"
    parent.mkdir()
    existing = parent / "existing.json"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        receipt.write_receipt(_built(), existing)
    assert existing.read_text(encoding="utf-8") == "keep"

    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    linked = parent / "linked.json"
    linked.symlink_to(outside)
    with pytest.raises(FileExistsError):
        receipt.write_receipt(_built(), linked)
    assert linked.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside"


def test_write_refuses_a_symlink_parent(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(OSError, match="real directory"):
        receipt.write_receipt(_built(), linked / "receipt.json")
    assert not list(real.iterdir())


def test_write_refuses_a_symlink_in_the_parent_chain(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(OSError):
        receipt.write_receipt(_built(), linked / "nested" / "receipt.json")
    assert not list(real.iterdir())


def test_prepublication_failure_leaves_no_file_or_temporary_inode(tmp_path: Path, monkeypatch):
    target = tmp_path / "receipts" / "update.json"

    def fail_link(*_args, **_kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(receipt.os, "link", fail_link)
    with pytest.raises(OSError, match="publication"):
        receipt.write_receipt(_built(), target)
    assert not target.exists()
    assert not list(target.parent.glob(".*.tmp-*"))


def test_interruption_after_hard_link_revokes_only_the_owned_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "receipts" / "update.json"
    interrupt = KeyboardInterrupt()
    real_link = os.link

    def interrupted_link(*args, **kwargs):
        real_link(*args, **kwargs)
        raise interrupt

    monkeypatch.setattr(receipt.os, "link", interrupted_link)
    with pytest.raises(KeyboardInterrupt) as raised:
        receipt.write_receipt(_built(), target)

    assert raised.value is interrupt
    assert not target.exists()
    assert not list(target.parent.glob(".*.tmp-*"))


def test_interrupted_publication_does_not_unlink_a_foreign_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "receipts" / "update.json"
    interrupt = SystemExit(17)
    real_link = os.link

    def replaced_link(*args, **kwargs):
        real_link(*args, **kwargs)
        target.unlink()
        target.write_text("foreign", encoding="ascii")
        raise interrupt

    monkeypatch.setattr(receipt.os, "link", replaced_link)
    with pytest.raises(SystemExit) as raised:
        receipt.write_receipt(_built(), target)

    assert raised.value is interrupt
    assert target.read_text(encoding="ascii") == "foreign"


def test_default_path_stays_under_the_private_state_root(tmp_path: Path):
    built = _built()
    expected = (
        tmp_path
        / "dgx-monarch"
        / "receipts"
        / "20260809T010204254000-update-bbbbbbbbbbbb.json"
    )
    assert receipt.default_receipt_path(built, state_home=tmp_path) == expected
    assert receipt.write_receipt(built, state_home=tmp_path) == expected
    assert stat.S_IMODE(expected.parent.stat().st_mode) == 0o700


def test_relative_state_and_destination_paths_are_refused(monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", "relative-state")
    with pytest.raises(ValueError, match="absolute"):
        receipt.default_receipt_path(_built())
    with pytest.raises(ValueError, match="absolute"):
        receipt.write_receipt(_built(), "relative-receipt.json")


def test_file_and_directory_are_both_fsynced(tmp_path: Path, monkeypatch):
    seen_modes: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int):
        seen_modes.append(stat.S_IFMT(os.fstat(fd).st_mode))
        real_fsync(fd)

    monkeypatch.setattr(receipt.os, "fsync", recording_fsync)
    receipt.write_receipt(_built(), tmp_path / "receipts" / "update.json")
    assert stat.S_IFREG in seen_modes
    assert stat.S_IFDIR in seen_modes


def test_every_new_receipt_directory_ancestor_is_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    existing.chmod(0o700)
    target = existing / "first" / "second" / "update.json"
    synced: set[tuple[int, int]] = set()
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            synced.add((info.st_dev, info.st_ino))
        real_fsync(fd)

    monkeypatch.setattr(receipt.os, "fsync", recording_fsync)
    receipt.write_receipt(_built(), target)

    required = {
        (path.stat().st_dev, path.stat().st_ino)
        for path in (existing, existing / "first", existing / "first" / "second")
    }
    assert required <= synced
