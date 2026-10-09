"""Persist consent memos and the standing auto-rescue toggle.

A memo authorizes one bypass for an exact checkpoint identity and narrow
context. Standing and environment grants resolve in ``consent_pending`` and
never create memos. The append-only gate ledger records evidence; it cannot
authorize a bypass because revoked grants remain in its history.

Memo keys bind ``dev:ino:size:mtime_ns:ctime_ns`` so replacement and same-size
rewrites invalidate consent. Residency context binds combination and loader
dtype; accuracy context binds combination, topology, and world. Full trust
context is audit data only: including worker policy would invalidate a rescue
when it changes the residency it authorizes. See docs/DESIGN.md section 5.9.

Mutations raise ``ConsentStoreError`` on persistence failure so the endpoint
can report HTTP 503. The directory is private (0700) and files are 0600.
Consent never overrides quarantine or converts INCONCLUSIVE to PASS.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .log import get_logger

log = get_logger(__name__)

# Fixed path beside family_memo.json, using the same cross-process lock.
# Authorization storage is not operator-relocatable; tests patch this constant.
MEMO_PATH = os.path.expanduser("~/.cache/dgx-monarch/consent_memo.json")
# v2 carries the complete current/legacy artifact join used by API quarantine.
SCHEMA = 2
# FIFO by first insertion: the key granted first is dropped first, and a
# re-grant of a live key keeps that key's place.
CONSENT_LIMIT = 128
# A memo whose checkpoint changed can never match again; drop it once its grant
# is also older than this. Age counts from the grant, not from the file
# change, because nothing watches the file.
STALE_PRUNE_S = 30 * 24 * 3600

_LOCK = threading.Lock()


class ConsentStoreError(OSError):
    """A consent could not be durably persisted or removed."""


def file_identity(path: str) -> str:
    """Return file identity that detects replacement and same-size rewrites.

    Tests pin this spelling to ``actor/store_family.file_identity``. Importing
    that module would load torch and Monarch through ``actor.__init__``; this
    driver-side module must remain dependency-light.
    """
    st = os.stat(path)
    return (
        f"{st.st_dev}:{st.st_ino}:{st.st_size}:"
        f"{st.st_mtime_ns}:{st.st_ctime_ns}"
    )


def canonical_context(context: dict[str, Any]) -> str:
    """Serialize finite JSON-native context or fail closed.

    Same rule as the ledger's canonicalizer, copied in four lines rather than
    imported from a private ledger name: a memo key that silently changed
    shape would revoke every consent on the box.
    """
    serialized = json.dumps(
        context, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if json.loads(serialized) != context:
        raise TypeError("consent context must contain only JSON-native values")
    return serialized


def memo_key(kind: str, path: str, context: dict[str, Any]) -> str:
    """Opaque key for one (kind, file identity, narrow context) authorization."""
    payload = f"{kind}\x00{file_identity(path)}\x00{canonical_context(context)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class ConsentRecord:
    """One persisted authorization; ``list_active`` derives the panel's row."""

    id: str
    kind: str
    target_guard: str
    waiver_class: str
    artifact: str
    path: str
    file_identity: str
    memo_context: dict[str, Any]
    granted_at: str
    granted_epoch: float
    consent_source: str
    reason: str
    ledger_key: str
    combo_key: str | None = None
    artifacts: str | None = None
    artifacts_legacy: str | None = None
    artifacts_legacy_complete: bool = False
    capability_context: dict[str, Any] | None = None
    unet_name: str | None = None
    loras: int = 0

    def as_json(self) -> dict[str, Any]:
        epoch = _as_finite_float(self.granted_epoch)
        if epoch is None:
            raise ValueError("consent grant epoch must be finite")
        return {
            "id": self.id,
            "kind": self.kind,
            "target_guard": self.target_guard,
            "waiver_class": self.waiver_class,
            "artifact": self.artifact,
            "path": self.path,
            "file_identity": self.file_identity,
            "memo_context": dict(self.memo_context),
            "granted_at": self.granted_at,
            "granted_epoch": epoch,
            "consent_source": self.consent_source,
            "reason": self.reason,
            "ledger_key": self.ledger_key,
            "combo_key": self.combo_key,
            "artifacts": self.artifacts,
            "artifacts_legacy": self.artifacts_legacy,
            "artifacts_legacy_complete": self.artifacts_legacy_complete,
            "capability_context": self.capability_context,
            "unet_name": self.unet_name,
            "loras": int(self.loras),
        }

    @classmethod
    def from_json(cls, raw: object) -> ConsentRecord | None:
        """Rebuild a record, or None when the row is not structurally sound.

        A malformed row is treated as absent. Missing evidence never grants.
        """
        if not isinstance(raw, dict):
            return None
        required = ("id", "kind", "target_guard", "waiver_class", "artifact",
                    "path", "file_identity", "granted_at", "consent_source",
                    "reason", "ledger_key")
        if not all(isinstance(raw.get(name), str) for name in required):
            return None
        if (raw.get("waiver_class") == "C"
                and (not all(isinstance(raw.get(name), str) and raw[name]
                             for name in ("combo_key", "artifacts", "artifacts_legacy"))
                     or not isinstance(raw.get("artifacts_legacy_complete"), bool))):
            return None
        context = raw.get("memo_context")
        capability = raw.get("capability_context")
        loras = raw.get("loras", 0)
        epoch = _as_finite_float(raw.get("granted_epoch"))
        if epoch is None:
            return None
        return cls(
            id=str(raw["id"]),
            kind=str(raw["kind"]),
            target_guard=str(raw["target_guard"]),
            waiver_class=str(raw["waiver_class"]),
            artifact=str(raw["artifact"]),
            path=str(raw["path"]),
            file_identity=str(raw["file_identity"]),
            memo_context=dict(context) if isinstance(context, dict) else {},
            granted_at=str(raw["granted_at"]),
            granted_epoch=epoch,
            consent_source=str(raw["consent_source"]),
            reason=str(raw["reason"]),
            ledger_key=str(raw["ledger_key"]),
            combo_key=_as_optional_str(raw.get("combo_key")),
            artifacts=_as_optional_str(raw.get("artifacts")),
            artifacts_legacy=_as_optional_str(raw.get("artifacts_legacy")),
            artifacts_legacy_complete=raw.get("artifacts_legacy_complete") is True,
            capability_context=dict(capability) if isinstance(capability, dict) else None,
            unet_name=_as_optional_str(raw.get("unet_name")),
            loras=int(loras) if isinstance(loras, int) and not isinstance(loras, bool) else 0,
        )


def _as_finite_float(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _empty() -> dict[str, Any]:
    return {"schema": SCHEMA, "auto_rescue": False, "consents": {}}


def _decode(raw: object) -> dict[str, Any]:
    """Coerce one decoded file into the in-memory shape, or start over.

    A file whose ``schema`` is absent or not :data:`SCHEMA` is unreadable and
    is replaced on the next write. There is no migration path: consents are
    cheap to re-grant, and a wrong-shaped authorization must never be honored.
    """
    if not isinstance(raw, dict):
        return _empty()
    if raw.get("schema") != SCHEMA:
        log.warning(
            "consent memo %s has schema %r, not %d, so it is ignored; the next write replaces it. "
            "Grant every consent again, and turn the Auto-allow capacity rescue toggle back on if it was on",
            MEMO_PATH, raw.get("schema"), SCHEMA)
        return _empty()
    consents = raw.get("consents")
    return {
        "schema": SCHEMA,
        "auto_rescue": raw.get("auto_rescue") is True,
        "consents": dict(consents) if isinstance(consents, dict) else {},
    }


def _read_unlocked() -> dict[str, Any]:
    try:
        with open(MEMO_PATH) as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return _empty()
    return _decode(raw)


def read() -> dict[str, Any]:
    """The whole memo file. A read failure degrades to empty, never to a grant."""
    with _LOCK:
        return _read_unlocked()


def _prune(memo: dict[str, Any]) -> None:
    consents = memo["consents"]
    now = time.time()
    for key, raw in list(consents.items()):
        record = ConsentRecord.from_json(raw)
        if record is None:
            consents.pop(key, None)
            continue
        if now - record.granted_epoch < STALE_PRUNE_S:
            continue
        if _current_identity(record.path) != record.file_identity:
            consents.pop(key, None)
    while len(consents) > CONSENT_LIMIT:
        consents.pop(next(iter(consents)))


def _write_atomic(memo: dict[str, Any]) -> None:
    tmp = f"{MEMO_PATH}.{os.getpid()}.tmp"
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        # A previous process died between create and replace. The name is
        # pid-scoped, so this one is ours and stale by definition.
        os.unlink(tmp)
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(memo, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, MEMO_PATH)
        _fsync_parent_directory()
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _fsync_parent_directory() -> None:
    """Persist the rename itself, not only the replacement file's contents.

    A successful ``replace`` without a directory sync can disappear after a
    power failure, restoring a consent the operator had just revoked.  A
    failure here does not undo the ``replace``: the new memo is in place and
    every reader sees it.  Only its survival across a power cut is unknown.
    """
    parent = os.path.dirname(MEMO_PATH) or "."
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(parent, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _mutate(apply: Callable[[dict[str, Any]], bool]) -> None:
    """Read, modify and durably replace the memo, or raise ConsentStoreError.

    ``apply`` returns False when the mutation is a no-op, which skips the
    write. The advisory lock on the sidecar serializes writers across
    ComfyUI processes sharing a cache directory; ``os.replace`` is what makes a
    reader see either the old file or the new one, never a torn one.
    """
    try:
        os.makedirs(os.path.dirname(MEMO_PATH), mode=0o700, exist_ok=True)
        with _LOCK:
            lock_fd = -1
            try:
                try:
                    import fcntl

                    lock_fd = os.open(f"{MEMO_PATH}.lock", os.O_CREAT | os.O_RDWR, 0o600)
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                except (ImportError, OSError):
                    pass  # cross-process serialization is best effort; the replace is not
                memo = _read_unlocked()
                if not apply(memo):
                    return
                _prune(memo)
                _write_atomic(memo)
            finally:
                if lock_fd >= 0:
                    os.close(lock_fd)
    except (OSError, TypeError, ValueError) as exc:
        raise ConsentStoreError(f"consent memo {MEMO_PATH!r} could not be written: {exc}") from exc


def lookup(kind: str, path: str, context: dict[str, Any]) -> ConsentRecord | None:
    """The live authorization for this exact (kind, file, context), or None.

    Every failure mode (missing file, unreadable memo, malformed row, a
    checkpoint that no longer stats) returns None. Missing evidence is never a
    grant.
    """
    try:
        key = memo_key(kind, path, context)
    except (OSError, TypeError, ValueError):
        return None
    record = ConsentRecord.from_json(read()["consents"].get(key))
    if record is None:
        return None
    if record.kind != kind:
        return None
    return record


def grant(record: ConsentRecord, key: str) -> None:
    """Persist one authorization. Raises ConsentStoreError when it did not land."""
    def apply(memo: dict[str, Any]) -> bool:
        memo["consents"][key] = record.as_json()
        return True

    _mutate(apply)


def revoke(key: str) -> bool:
    """Delete one authorization. False when it was already absent."""
    removed = False

    def apply(memo: dict[str, Any]) -> bool:
        nonlocal removed
        removed = memo["consents"].pop(key, None) is not None
        return removed

    _mutate(apply)
    return removed


def _current_identity(path: str) -> str | None:
    try:
        return file_identity(path)
    except OSError:
        return None


def list_active(memo: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Display rows for the panel's active list, in memo order (oldest first).

    No absolute path and no raw file identity crosses the wire: a row carries
    the artifact name and the opaque key, matching the repo's rule that an
    unauthenticated payload field carries the minimum.
    """
    rows: list[dict[str, Any]] = []
    snapshot = read() if memo is None else memo
    for key, raw in snapshot["consents"].items():
        record = ConsentRecord.from_json(raw)
        if record is None:
            continue
        rows.append({
            "id": record.id,
            "key": key,
            "kind": record.kind,
            "class": record.waiver_class,
            "artifact": record.artifact,
            "granted_at": record.granted_at,
            "consent_source": record.consent_source,
            # Expose narrow context fields that distinguish grants for one checkpoint:
            # topology/world for accuracy or loader dtype for residency. Omit the
            # opaque combination hash so the operator can identify the row to revoke.
            "context": {name: str(value)
                        for name, value in (record.memo_context or {}).items()
                        if name != "combo_key"},
            "stale": _current_identity(record.path) != record.file_identity,
        })
    return rows


def auto_rescue() -> bool:
    """The persisted standing-authorization toggle."""
    return read()["auto_rescue"] is True


def set_auto_rescue(value: bool) -> None:
    """Write the standing toggle. Raises ConsentStoreError when it did not land."""
    wanted = bool(value)

    def apply(memo: dict[str, Any]) -> bool:
        if memo["auto_rescue"] is wanted:
            return False
        memo["auto_rescue"] = wanted
        return True

    _mutate(apply)
