"""Versioned operator receipts containing only validated result fields.

Never include raw commands, paths, environments or exceptions.
"""
from __future__ import annotations

import json
import math
import os
import re
import secrets
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

from .receipt_sanitize import sanitize_note

ReceiptOperation: TypeAlias = Literal["setup", "doctor_repair", "update", "cluster_smoke"]
ReceiptStatus: TypeAlias = Literal["planned", "succeeded", "failed", "partial"]
ReceiptStepStatus: TypeAlias = Literal[
    "planned", "succeeded", "failed", "partial", "unknown"
]

SCHEMA_NAME: Final = "dgx-monarch.operator-receipt"
SCHEMA_VERSION: Final = 2
OPERATIONS: Final = frozenset({"setup", "doctor_repair", "update", "cluster_smoke"})
TERMINAL_STATUSES: Final = frozenset(
    {"planned", "succeeded", "failed", "partial"}
)
STEP_STATUSES: Final = frozenset(
    {"planned", "succeeded", "failed", "partial", "unknown"}
)
_V1_STEP_STATUSES: Final = TERMINAL_STATUSES

_SAFE_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_TARGET_HASH = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SOURCE_HASH = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = frozenset(
    "schema schema_version operation status started_at finished_at duration_ms selection steps summary notes".split()
)
_SELECTION_KEYS = frozenset({"profile", "target", "source_hashes"})
_STEP_KEYS = frozenset({"name", "status", "counts", "notes"})
_SUMMARY_KEYS = frozenset({"total_steps", "step_counts"})


def _typed_operation(value: object) -> ReceiptOperation:
    if not isinstance(value, str) or value not in OPERATIONS:
        raise ValueError(f"unsupported receipt operation {value!r}")
    return cast(ReceiptOperation, value)


def _typed_status(value: object) -> ReceiptStatus:
    if not isinstance(value, str) or value not in TERMINAL_STATUSES:
        raise ValueError(f"unsupported receipt status {value!r}")
    return cast(ReceiptStatus, value)


def _typed_step_status(
    value: object, *, schema_version: int = SCHEMA_VERSION
) -> ReceiptStepStatus:
    supported = _V1_STEP_STATUSES if schema_version == 1 else STEP_STATUSES
    if not isinstance(value, str) or value not in supported:
        raise ValueError(f"unsupported receipt step status {value!r}")
    return cast(ReceiptStepStatus, value)


def _safe_name(value: object, field: str) -> str:
    if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase identifier of at most 64 characters")
    return value


def _hash(value: object, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical lowercase hexadecimal hash")
    return value


def _safe_notes(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raise TypeError("receipt notes must be a sequence of strings")
    return [sanitize_note(value) for value in values]


def _safe_counts(values: Mapping[str, int] | None) -> dict[str, int]:
    if values is None:
        return {}
    result: dict[str, int] = {}
    for key, value in values.items():
        name = _safe_name(key, "count name")
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
            raise ValueError(f"count {name!r} must be a non-negative integer")
        result[name] = value
    return dict(sorted(result.items()))


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise ValueError(f"{field} must be a canonical UTC timestamp") from exc
    parsed = _utc(parsed, field)
    if _utc_text(parsed) != value:
        raise ValueError(f"{field} must use six fractional UTC digits")
    return parsed


def _selection(profile: str | None, target: str | None,
               source_hashes: Mapping[str, str] | None) -> dict[str, object]:
    safe_profile = None if profile is None else _safe_name(profile, "profile")
    safe_target = None if target is None else _hash(target, "target", _TARGET_HASH)
    safe_sources: dict[str, str] = {}
    for key, value in (source_hashes or {}).items():
        safe_sources[_safe_name(key, "source hash name")] = _hash(value, f"source hash {key!r}", _SOURCE_HASH)
    return {"profile": safe_profile, "target": safe_target,
            "source_hashes": dict(sorted(safe_sources.items()))}


class OperatorReceiptBuilder:
    """Incrementally build one immutable, validated operator receipt."""
    def __init__(
        self, operation: ReceiptOperation, *,
        profile: str | None = None,
        target: str | None = None,
        source_hashes: Mapping[str, str] | None = None,
        clock: Callable[[], datetime] | None = None,
        timer: Callable[[], float] | None = None,
    ) -> None:
        self._operation = _typed_operation(operation)
        self._selection = _selection(profile, target, source_hashes)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timer = timer or time.monotonic
        self._started_at = _utc(self._clock(), "receipt start")
        self._started_tick = self._tick("receipt start")
        self._steps: list[dict[str, object]] = []
        self._finished = False

    def _tick(self, field: str) -> float:
        value = self._timer()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{field} timer must be finite")
        return float(value)

    def add_step(self, name: str, status: ReceiptStepStatus, *,
                 counts: Mapping[str, int] | None = None,
                 notes: Sequence[str] | None = None) -> None:
        if self._finished:
            raise RuntimeError("cannot add a step to a finished receipt")
        self._steps.append({
            "name": _safe_name(name, "step name"),
            "status": _typed_step_status(status),
            "counts": _safe_counts(counts),
            "notes": _safe_notes(notes),
        })

    def finish(self, status: ReceiptStatus, *,
               notes: Sequence[str] | None = None) -> dict[str, object]:
        if self._finished:
            raise RuntimeError("receipt was already finished")
        finished_at = _utc(self._clock(), "receipt finish")
        finished_tick = self._tick("receipt finish")
        if finished_at < self._started_at:
            raise ValueError("receipt finish timestamp precedes its start")
        elapsed = finished_tick - self._started_tick
        if elapsed < 0:
            raise ValueError("receipt finish timer precedes its start")
        final_notes = _safe_notes(notes)
        self._finished = True
        step_counts = dict.fromkeys(sorted(STEP_STATUSES), 0)
        for step in self._steps:
            step_counts[cast(str, step["status"])] += 1
        receipt: dict[str, object] = {
            "schema": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
            "operation": self._operation, "status": _typed_status(status),
            "started_at": _utc_text(self._started_at),
            "finished_at": _utc_text(finished_at), "duration_ms": round(elapsed * 1000),
            "selection": dict(self._selection), "steps": [dict(step) for step in self._steps],
            "summary": {"total_steps": len(self._steps), "step_counts": step_counts},
            "notes": final_notes}
        return validate_receipt(receipt)


def validate_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    """Validate an exact schema and return a detached, canonicalizable copy."""
    if not isinstance(receipt, Mapping) or set(receipt) != _TOP_LEVEL_KEYS:
        raise ValueError("operator receipt has unknown or missing top-level fields")
    schema_version = receipt.get("schema_version")
    if (
        receipt.get("schema") != SCHEMA_NAME
        or isinstance(schema_version, bool)
        or schema_version not in (1, SCHEMA_VERSION)
    ):
        raise ValueError("operator receipt schema is unsupported")
    operation = _typed_operation(receipt.get("operation"))
    status = _typed_status(receipt.get("status"))
    started = _parse_utc(receipt.get("started_at"), "started_at")
    finished = _parse_utc(receipt.get("finished_at"), "finished_at")
    if finished < started:
        raise ValueError("operator receipt finish precedes its start")
    duration = receipt.get("duration_ms")
    if isinstance(duration, bool) or not isinstance(duration, int) or not 0 <= duration < 2**63:
        raise ValueError("operator receipt duration_ms must be a non-negative integer")
    raw_selection = receipt.get("selection")
    if not isinstance(raw_selection, Mapping) or set(raw_selection) != _SELECTION_KEYS:
        raise ValueError("operator receipt selection has unknown or missing fields")
    raw_sources = raw_selection.get("source_hashes")
    if not isinstance(raw_sources, Mapping):
        raise ValueError("operator receipt source_hashes must be an object")
    selection = _selection(cast(str | None, raw_selection.get("profile")),
                           cast(str | None, raw_selection.get("target")),
                           cast(Mapping[str, str], raw_sources))
    raw_steps = receipt.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) > 256:
        raise ValueError("operator receipt steps must be a list of at most 256 entries")
    steps: list[dict[str, object]] = []
    step_statuses = _V1_STEP_STATUSES if schema_version == 1 else STEP_STATUSES
    actual_counts = dict.fromkeys(sorted(step_statuses), 0)
    for raw_step in raw_steps:
        if not isinstance(raw_step, Mapping) or set(raw_step) != _STEP_KEYS:
            raise ValueError("operator receipt step has unknown or missing fields")
        raw_notes = raw_step.get("notes")
        if not isinstance(raw_notes, list):
            raise ValueError("operator receipt step notes must be a list")
        notes = _safe_notes(cast(Sequence[str], raw_notes))
        if notes != raw_notes:
            raise ValueError("operator receipt contains an unsanitized step note")
        raw_counts = raw_step.get("counts")
        if not isinstance(raw_counts, Mapping):
            raise ValueError("operator receipt step counts must be an object")
        step_status = _typed_step_status(
            raw_step.get("status"), schema_version=cast(int, schema_version)
        )
        actual_counts[step_status] += 1
        steps.append({
            "name": _safe_name(raw_step.get("name"), "step name"),
            "status": step_status,
            "counts": _safe_counts(cast(Mapping[str, int], raw_counts)),
            "notes": notes,
        })
    raw_summary = receipt.get("summary")
    if not isinstance(raw_summary, Mapping) or set(raw_summary) != _SUMMARY_KEYS:
        raise ValueError("operator receipt summary has unknown or missing fields")
    if raw_summary.get("total_steps") != len(steps):
        raise ValueError("operator receipt total_steps disagrees with its steps")
    if raw_summary.get("step_counts") != actual_counts:
        raise ValueError("operator receipt step_counts disagree with its steps")
    raw_notes = receipt.get("notes")
    if not isinstance(raw_notes, list):
        raise ValueError("operator receipt notes must be a list")
    notes = _safe_notes(cast(Sequence[str], raw_notes))
    if notes != raw_notes:
        raise ValueError("operator receipt contains an unsanitized note")
    return {"schema": SCHEMA_NAME, "schema_version": schema_version,
            "operation": operation, "status": status,
            "started_at": _utc_text(started), "finished_at": _utc_text(finished),
            "duration_ms": duration, "selection": selection, "steps": steps,
            "summary": {"total_steps": len(steps), "step_counts": actual_counts},
            "notes": notes}


def canonical_json(receipt: Mapping[str, object]) -> str:
    safe = validate_receipt(receipt)
    return json.dumps(safe, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False) + "\n"


def _state_root(state_home: str | os.PathLike[str] | None) -> Path:
    if state_home is None:
        configured = os.environ.get("XDG_STATE_HOME")
        root = Path(configured).expanduser() if configured else Path.home() / ".local" / "state"
    else:
        root = Path(state_home).expanduser()
    if not root.is_absolute():
        raise ValueError("operator receipt state root must be absolute")
    return root


def default_receipt_path(receipt: Mapping[str, object], *,
                         state_home: str | os.PathLike[str] | None = None) -> Path:
    safe = validate_receipt(receipt)
    stamp = cast(str, safe["finished_at"]).replace("-", "").replace(":", "").replace(".", "")
    stamp = stamp.removesuffix("Z")
    operation = cast(str, safe["operation"])
    selection = cast(dict[str, object], safe["selection"])
    identity = selection.get("target")
    sources = cast(dict[str, str], selection["source_hashes"])
    if identity is None and sources:
        identity = sources[sorted(sources)[0]]
    suffix = str(identity)[:12] if identity is not None else "no-identity"
    return _state_root(state_home) / "dgx-monarch" / "receipts" / f"{stamp}-{operation}-{suffix}.json"


def _private_parent(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(path.anchor, flags)
    pending_fd = -1
    try:
        for index, name in enumerate(path.parts[1:]):
            created = False
            try:
                child_fd = os.open(name, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                os.mkdir(name, 0o700, dir_fd=directory_fd)
                created = True
                os.fsync(directory_fd)
                child_fd = os.open(name, flags, dir_fd=directory_fd)
            except OSError as exc:
                raise OSError("operator receipt parent must be a real directory, not a symlink") from exc
            pending_fd = child_fd
            opened = os.fstat(child_fd)
            final = index == len(path.parts[1:]) - 1
            writable = bool(opened.st_mode & 0o022)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (created and (opened.st_uid != os.geteuid() or opened.st_mode & 0o077))
                or (final and opened.st_uid != os.geteuid())
                or (not final and writable and not opened.st_mode & stat.S_ISVTX)
            ):
                raise OSError("operator receipt parent chain is unsafe")
            os.close(directory_fd)
            directory_fd, pending_fd = child_fd, -1
        os.fchmod(directory_fd, 0o700)
    except BaseException:
        for descriptor in (pending_fd, directory_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass
        raise
    return directory_fd


def _entry_matches(directory_fd: int, name: str, identity: tuple[int, int]) -> bool:
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (current.st_dev, current.st_ino) == identity


def _unlink_owned(directory_fd: int, name: str, identity: tuple[int, int]) -> bool:
    if not _entry_matches(directory_fd, name, identity):
        return False
    os.unlink(name, dir_fd=directory_fd)
    return True


def _same_parent_fd(path: Path, identity: tuple[int, int]) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) == identity:
        return descriptor
    os.close(descriptor)
    raise OSError("operator receipt directory identity changed")


def receipt_destination(path: str | os.PathLike[str]) -> Path:
    """Refuse a relative or nameless receipt path; it needs no receipt to check."""
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        raise ValueError("operator receipt path must be absolute")
    if not destination.name or destination.name in {".", ".."}:
        raise ValueError("operator receipt path needs a filename")
    return destination


def write_receipt(receipt: Mapping[str, object], path: str | os.PathLike[str] | None = None,
                  *, state_home: str | os.PathLike[str] | None = None) -> Path:
    """Publish an immutable 0600 receipt without overwriting; fsync its file and directories."""
    safe = validate_receipt(receipt)
    destination = receipt_destination(default_receipt_path(safe, state_home=state_home) if path is None else path)
    payload = canonical_json(safe).encode("ascii")
    directory_fd = _private_parent(destination.parent)
    parent = os.fstat(directory_fd)
    parent_identity = parent.st_dev, parent.st_ino
    temporary_name: str | None = None
    file_identity: tuple[int, int] | None = None
    link_attempted = False
    try:
        for _attempt in range(16):
            candidate = f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            try:
                file_fd = os.open(candidate, flags, 0o600, dir_fd=directory_fd)
            except FileExistsError:
                continue
            temporary_name = candidate
            opened_file = os.fstat(file_fd)
            file_identity = opened_file.st_dev, opened_file.st_ino
            break
        else:
            raise FileExistsError("could not reserve an operator receipt temporary file")
        open_file_fd = file_fd
        try:
            os.fchmod(file_fd, 0o600)
            handle = os.fdopen(file_fd, "wb", closefd=True)
            open_file_fd = -1
            with handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if open_file_fd >= 0:
                os.close(open_file_fd)
        if temporary_name is None or file_identity is None or not _entry_matches(
            directory_fd, temporary_name, file_identity
        ):
            raise OSError("operator receipt temporary identity changed")
        try:
            os.stat(destination.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("operator receipt destination already exists")
        link_attempted = True
        os.link(temporary_name, destination.name, src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd, follow_symlinks=False)
        if not _unlink_owned(directory_fd, temporary_name, file_identity):
            raise OSError("operator receipt temporary identity changed")
        temporary_name = None
        os.fsync(directory_fd)
        os.close(directory_fd)
        directory_fd = -1
    except BaseException:
        cleanup_fd = directory_fd
        recovered_fd = -1
        try:
            os.fstat(cleanup_fd)
        except BaseException:
            try:
                recovered_fd = _same_parent_fd(destination.parent, parent_identity)
                cleanup_fd = recovered_fd
            except BaseException:
                pass
        if cleanup_fd >= 0 and file_identity is not None:
            removed = False
            for name in ((destination.name,) if link_attempted else ()) + (
                () if temporary_name is None else (temporary_name,)):
                try:
                    removed = _unlink_owned(cleanup_fd, name, file_identity) or removed
                except BaseException:
                    pass
            if removed:
                try:
                    os.fsync(cleanup_fd)
                except BaseException:
                    pass
        for descriptor in (recovered_fd, directory_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass
        raise
    return destination
