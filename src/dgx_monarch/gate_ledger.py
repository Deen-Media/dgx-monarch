"""Driver-side identity-gate verdict ledger and authorization selectors.

The JSONL sits beside Gate reports. Contextual authority is trusted only while
all identity dimensions match:

  * combination key (checkpoint, loader options, and LoRA-name set; strengths
    are excluded because they exercise the same mechanism);
  * bounded sampled-content artifact digest for the checkpoint and every LoRA,
    so a same-name replacement cannot inherit prior authority;
  * ComfyUI commit, gate protocol, dgx-monarch package version, and the
    canonical package-source manifest; and
  * the exact capability context that the ceremony proved.

Mismatches read stale and re-prove. Context-free lookup is diagnostic only;
FAIL quarantine is sticky and source-independent.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass

from . import __version__, runtime_provenance
from .gate_artifacts import artifact_signature as artifact_signature
from .log import get_logger

log = get_logger(__name__)

LEDGER_NAME = "dgxm_gate_ledger.jsonl"
# Bump whenever the ceremony or the meaning of a trusted PASS changes. PASS is
# a capability grant, so an entry from an older protocol or version is evidence
# to re-run the ceremony, never evidence to skip it.
# v4 cannot be reused because of rejected slab grants and bidirectional sibling stamps.
# v5 cannot be reused because it lacked deep transaction binding, exact normal-render
# grants, and one crash-durable multi-context RETESTING boundary.
# v6 cannot be reused because two divergent implementations used it: the
# packed-comparison prototype lacked the transaction and grant hardening merged
# 2026-07-15, and the deployed lineage had that hardening but did not compare
# or mutation-guard packed modalities. v7 was the first protocol to combine both.
# v8 gave resolved no-LoRA FSDP its own exact clean-reload proof scope.
# v9 adds audit/slab checks; v10 binds source and rejects divergent-repeat PASS.
# v10 cannot be reused: a class-K member's audit fields changed
# meaning; v11 for slab proof counting replies without binding each to its exact
# rank/world/setup-generation cohort; v12 for a fourth class-K kind, shard-quant.
# Numbers are single-use, so a semantic change advances rather than redefines one.
GATE_PROTOCOL_VERSION = 13


def _canonical_capability_context(context: dict) -> str:
    """Serialize finite JSON-native capability identity or fail closed."""
    serialized = json.dumps(
        context, sort_keys=True,
        separators=(",", ":"), allow_nan=False)
    if json.loads(serialized) != context:
        raise TypeError("capability context must contain only JSON-native values")
    return serialized


def gate_verdict_token(
    key: str, artifact_digest: str, commit: str, capability_context: dict
) -> tuple[str, ...]:
    """Canonical process-local token for one exact Gate capability."""
    return (
        key, artifact_digest, commit,
        _canonical_capability_context(capability_context), runtime_provenance.cached_dgx_source_manifest_sha256(),
        str(GATE_PROTOCOL_VERSION), str(__version__))


@dataclass(frozen=True, slots=True)
class ArtifactSetSignature:
    """Current artifact-set digest and its pre-0.3 lookup alias."""

    current: str
    legacy: str
    legacy_complete: bool


@dataclass(frozen=True, slots=True)
class GateLedgerLookup:
    """One ledger scan plus whether a process-local PASS may bridge it.

    ``session_pass_safe`` is false when unscoped damage could conceal a newer
    row for an otherwise absent identity.  Callers may still use the canonical
    ``unknown``/``stale`` state to trigger a fresh ceremony, but must not turn a
    pre-existing process cache entry into positive authority until an exact
    valid row heals the damage.
    """

    state: str
    entry: dict | None
    session_pass_safe: bool


class GateLedgerReadError(OSError):
    """Ledger existence is known, but its current verdicts could not be read."""


class GateLedgerWriteError(OSError):
    """A trust-critical ledger row could not be durably appended."""


def artifact_set_signature(signatures: Iterable[str]) -> ArtifactSetSignature:
    """Bind every artifact fingerprint into one untruncated composite digest.

    The caller controls semantic ordering (normally checkpoint first, then the
    canonical LoRA order). Length-prefixed JSON prevents concatenation
    ambiguity and hashing keeps the ledger field compact regardless of stack
    size. The explicit legacy fields let lookup preserve pre-upgrade entries
    without relying on metadata hidden on a string value. A truncated legacy
    aggregate is incomplete and therefore safe only as a FAIL quarantine.
    """
    values = list(signatures)
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    current = hashlib.sha256(payload.encode()).hexdigest()
    legacy_full = "-".join(values)
    return ArtifactSetSignature(
        current=current,
        legacy=legacy_full[:96],
        legacy_complete=len(legacy_full) <= 96,
    )


def comfy_commit() -> str:
    """Best-effort comfy git commit (no subprocess: read .git directly)."""
    try:
        import folder_paths

        git = os.path.join(folder_paths.base_path, ".git")
        head = open(os.path.join(git, "HEAD")).read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            ref_path = os.path.join(git, ref)
            if os.path.exists(ref_path):
                return open(ref_path).read().strip()[:12]
            packed = os.path.join(git, "packed-refs")
            if os.path.exists(packed):
                for line in open(packed):
                    if line.strip().endswith(ref):
                        return line.split()[0][:12]
            return "unknown"
        return head[:12]
    except Exception:
        return "unknown"


def combo_key(unet_name: str, options: dict | None, lora_names) -> str:
    payload = json.dumps(
        [unet_name, sorted((options or {}).items()), sorted(lora_names)], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class GateLedger:
    """Append-only verdict log with last-writer-wins lookup semantics."""

    def __init__(self, directory: str):
        self.path = os.path.join(directory, LEDGER_NAME)

    @staticmethod
    def _valid_entry_schema(entry: object) -> bool:
        """Whether one decoded row has enough identity to affect trust."""
        if not isinstance(entry, dict):
            return False
        if not all(
            isinstance(entry.get(field), str)
            for field in ("key", "artifacts", "comfy", "verdict")
        ):
            return False
        if ("gate_protocol" in entry
                and (isinstance(entry["gate_protocol"], bool)
                     or not isinstance(entry["gate_protocol"], int))):
            return False
        if any(field in entry and not isinstance(entry[field], str) for field in ("dgx_monarch", "dgx_source")):
            return False
        if ("capability_context" in entry
                and not isinstance(entry["capability_context"], str)):
            return False
        if entry.get("verdict") == "RETESTING":
            # Contextual lookup selects a guard only under the exact runtime
            # identity.  A guard without either field must therefore be
            # treated as damage, not skipped in favour of an older PASS.
            if "gate_protocol" not in entry or "dgx_monarch" not in entry:
                return False
            blocked = entry.get("blocked_contexts")
            if (not isinstance(blocked, list) or not blocked
                    or not all(isinstance(value, str) for value in blocked)):
                return False
        return True

    def entries_with_integrity(self) -> tuple[list[tuple[int, dict]], int]:
        """Return intact positioned records and the newest damaged line.

        A process can be killed mid-record. That must not erase the quarantine
        history before the torn tail, so parsing is line-local. Positions let a
        later valid exact row heal older damage while a damaged newer revocation
        still cannot resurrect an older PASS.
        """
        entries: list[tuple[int, dict]] = []
        last_damage_line = 0
        try:
            with open(self.path, "rb") as f:
                for line_number, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        # Decode inside the line-local refusal boundary. Text-mode
                        # iteration decodes ahead of this try and lets one invalid
                        # UTF-8 byte erase every intact verdict in the file.
                        entry = json.loads(line.decode("utf-8"))
                    except (TypeError, ValueError, RecursionError):
                        last_damage_line = line_number
                        log.warning(
                            "gate ledger %s has a damaged record on line %d; "
                            "preserving the remaining intact history",
                            self.path, line_number,
                        )
                        continue
                    if not self._valid_entry_schema(entry):
                        last_damage_line = line_number
                        log.warning(
                            "gate ledger %s has an invalid record schema on line %d; "
                            "preserving the remaining intact history",
                            self.path, line_number,
                        )
                        continue
                    entries.append((line_number, entry))
        except FileNotFoundError:
            return [], 0
        except OSError as exc:
            raise GateLedgerReadError(
                f"could not read identity-gate ledger {self.path!r}") from exc
        return entries, last_damage_line

    def entries(self) -> list[dict]:
        """Public non-authoritative view of intact, structurally valid rows."""
        return [entry for _line, entry in self.entries_with_integrity()[0]]

    @staticmethod
    def _artifact_matches(entry: dict, artifacts: str | ArtifactSetSignature) -> bool:
        recorded = entry.get("artifacts")
        verdict = entry.get("verdict")
        if isinstance(artifacts, ArtifactSetSignature):
            if recorded == artifacts.current:
                return True
            return recorded == artifacts.legacy and (
                artifacts.legacy_complete or verdict == "FAIL"
            )
        return recorded == artifacts

    @staticmethod
    def _context_value(context: dict | str | None) -> str | None:
        if context is None:
            return None
        if isinstance(context, str):
            return context
        return _canonical_capability_context(context)

    def lookup_with_integrity(
        self,
        key: str,
        artifacts: str | ArtifactSetSignature,
        commit: str,
        context: dict | str | None = None,
    ) -> GateLedgerLookup:
        """Verdict, exact row, and process-cache safety from one ledger read.

        ``context=None`` retains legacy diagnostic lookup. Runtime trust callers
        provide a context, so historical context-free PASS rows read stale.
        """
        entries, last_damage_line = self.entries_with_integrity()
        key_entries = [
            (line, entry) for line, entry in entries if entry.get("key") == key
        ]
        if not key_entries:
            # No intact target row can grant; preserve the canonical state for
            # reproof while the integrity bit prevents a process PASS from
            # bridging unscoped damage.
            return GateLedgerLookup("unknown", None, not bool(last_damage_line))
        wanted_context = self._context_value(context)
        artifact_entries = [
            (line, entry) for line, entry in key_entries
            if self._artifact_matches(entry, artifacts)
        ]
        # Last writer wins for this *artifact identity*, not merely this model
        # name/options combination.  A -> B -> A must recover A's quarantine
        # rather than treating B's newer record as proof that A is unknown.
        if wanted_context is None:
            latest_pair = artifact_entries[-1] if artifact_entries else None
        else:
            latest_pair = next((
                (line, entry) for line, entry in reversed(artifact_entries)
                if entry.get("gate_protocol") == GATE_PROTOCOL_VERSION
                and entry.get("dgx_monarch") == __version__
                and (
                    entry.get("capability_context") == wanted_context
                    or (
                        entry.get("verdict") == "RETESTING"
                        and wanted_context in entry.get("blocked_contexts", ())
                    )
                )
            ), None)
        if latest_pair is None:
            return GateLedgerLookup("stale", None, not bool(last_damage_line))
        latest_line, latest = latest_pair
        verdict = latest.get("verdict")
        if wanted_context is not None and verdict in ("PASS", "INCONCLUSIVE", "RETESTING"):
            current_source = runtime_provenance.cached_dgx_source_manifest_sha256()
            if latest.get("dgx_source") != current_source:
                prior_boundary = next((pair for pair in reversed(artifact_entries)
                    if pair[1].get("gate_protocol") == GATE_PROTOCOL_VERSION
                    and pair[1].get("dgx_monarch") == __version__
                    and pair[1].get("capability_context") == wanted_context
                    and (pair[1].get("verdict") == "FAIL" or (
                        pair[1].get("verdict") in ("PASS", "INCONCLUSIVE", "RETESTING")
                        and pair[1].get("dgx_source") == current_source))), None)
                if (prior_boundary is not None
                        and prior_boundary[1].get("verdict") == "FAIL"):
                    return GateLedgerLookup(
                        "fail", prior_boundary[1], last_damage_line <= prior_boundary[0])
                return GateLedgerLookup("stale", latest, False)
        session_pass_safe = last_damage_line <= latest_line
        recorded_commit = latest.get("comfy", "unknown")
        # FAIL is a safety quarantine, not a trust grant. Keep it sticky for
        # the same artifact bytes even when the Comfy commit is unknown or has
        # moved; an explicit identity ceremony can re-test and overwrite it.
        if verdict == "FAIL":
            return GateLedgerLookup("fail", latest, session_pass_safe)
        if verdict == "RETESTING":
            return GateLedgerLookup("inconclusive", latest, session_pass_safe)
        # Only positive authority is vulnerable to a concealed newer row.
        # Parseable negative evidence stays a denial, while a later exact row
        # (its line is newer than the damage) heals the ledger.
        if verdict == "PASS" and last_damage_line > latest_line:
            return GateLedgerLookup("error", latest, False)
        if commit == "unknown" or recorded_commit == "unknown" or recorded_commit != commit:
            return GateLedgerLookup("stale", latest, session_pass_safe)
        if verdict == "PASS":
            return GateLedgerLookup("pass", latest, session_pass_safe)
        if verdict == "INCONCLUSIVE":
            return GateLedgerLookup("inconclusive", latest, session_pass_safe)
        return GateLedgerLookup("unknown", latest, session_pass_safe)

    def lookup_with_entry(self, key: str, artifacts: str | ArtifactSetSignature,
                          commit: str, context: dict | str | None = None,
                          ) -> tuple[str, dict | None]:
        """Verdict state plus the exact row it derives from, in one read."""
        result = self.lookup_with_integrity(key, artifacts, commit, context)
        return result.state, result.entry

    def lookup(self, key: str, artifacts: str | ArtifactSetSignature, commit: str,
               context: dict | str | None = None) -> str:
        """Verdict state only; see :meth:`lookup_with_entry` for the row."""
        return self.lookup_with_entry(key, artifacts, commit, context)[0]

    def matching_entry(self, key: str, artifacts: str | ArtifactSetSignature,
                       context: dict | str | None = None,
                       verdict: str | None = None) -> dict | None:
        """Newest matching row for diagnostics/quarantine lever recovery.

        ``verdict`` narrows to an exact outcome, so a lever reader cannot be
        handed a row from a different verdict than the one it is acting on.
        """
        wanted_context = self._context_value(context)
        for entry in reversed(self.entries()):
            if entry.get("key") != key or not self._artifact_matches(entry, artifacts):
                continue
            if verdict is not None and entry.get("verdict") != verdict:
                continue
            if wanted_context is None:
                return entry
            if (entry.get("gate_protocol") == GATE_PROTOCOL_VERSION
                    and entry.get("dgx_monarch") == __version__
                    and entry.get("capability_context") == wanted_context
                    and (entry.get("verdict") not in ("PASS", "INCONCLUSIVE", "RETESTING") or entry.get("dgx_source") == runtime_provenance.cached_dgx_source_manifest_sha256())):
                return entry
        return None

    def _entry(
        self,
        key: str,
        artifacts: str | ArtifactSetSignature,
        commit: str,
        verdict: str,
        detail: dict | None,
        context: dict | str | None,
    ) -> dict:
        current_artifacts = (artifacts.current
                             if isinstance(artifacts, ArtifactSetSignature) else artifacts)
        entry = {
            **(detail or {}),
            "key": key,
            "artifacts": current_artifacts,
            "comfy": commit,
            "verdict": verdict,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        context_value = self._context_value(context)
        if context_value is not None:
            entry.update({
                "gate_protocol": GATE_PROTOCOL_VERSION,
                "dgx_monarch": __version__,
                "capability_context": context_value,
                **({"dgx_source": runtime_provenance.cached_dgx_source_manifest_sha256()} if verdict in ("PASS", "INCONCLUSIVE", "RETESTING") else {}),
            })
        return entry

    def _append(self, entry: dict, *, strict_lock: bool) -> None:
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        fresh = not os.path.exists(self.path)
        with open(self.path, "a+b") as f:
            # One advisory lock covers the complete JSONL record across
            # ComfyUI processes sharing an output directory. A trust-critical
            # RETESTING transaction refuses to proceed without that lock.
            locked = False
            try:
                import fcntl

                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    locked = True
                except OSError:
                    if strict_lock:
                        raise
            except ImportError as exc:
                # Windows has no fcntl; non-strict records retain O_APPEND plus
                # one write as the best available local-file primitive there.
                if strict_lock:
                    raise OSError(
                        "identity-gate retest requires advisory file locking"
                    ) from exc
            try:
                # If a prior process died mid-record, terminate that damaged
                # line before appending. Otherwise the next valid JSON object
                # would be glued to the torn tail and readers would lose both.
                f.seek(0, os.SEEK_END)
                if f.tell():
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n":
                        f.write(b"\n")
                payload = (json.dumps(entry, separators=(",", ":")) + "\n").encode()
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            finally:
                if locked:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
        if fresh:
            # The file's bytes are on disk, but a new file's name is only in the
            # parent's page cache, and a power cut there takes the whole ledger.
            # The consent memo syncs its parent, and consent_routes writes this
            # audit row before that memo, so without this sync a crash could
            # keep the memo and lose the row. Later appends need no sync: the
            # name is already durable.
            self._fsync_directory(directory)

    @staticmethod
    def _fsync_directory(directory: str) -> None:
        """Persist the new ledger name, or leave the row durable without it."""
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            directory_fd = os.open(directory or ".", flags)
        except OSError as exc:
            log.debug("ledger directory not syncable: %r", exc)
            return
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            log.debug("ledger directory sync failed: %r", exc)
        finally:
            os.close(directory_fd)

    def record(self, key: str, artifacts: str | ArtifactSetSignature, commit: str, verdict: str,
               detail: dict | None = None, context: dict | str | None = None) -> bool:
        entry = self._entry(key, artifacts, commit, verdict, detail, context)
        try:
            self._append(entry, strict_lock=False)
        except OSError as exc:
            log.warning("gate ledger not writable (%r); verdicts will not persist", exc)
            return False
        return True

    def begin_retest_required(
        self,
        key: str,
        artifacts: str | ArtifactSetSignature,
        commit: str,
        contexts: Iterable[dict | str],
        detail: dict | None = None,
    ) -> None:
        """Durably revoke every possibly inherited PASS before a retest.

        One multi-context row is the transaction boundary: a crash cannot
        guard normal authority while leaving Fleet or a deterministic slab-on
        sibling live. Exact final verdict rows supersede this guard one context
        at a time only after the ceremony completes.
        """
        blocked_contexts = list(dict.fromkeys(
            self._context_value(context) for context in contexts))
        if not blocked_contexts or any(value is None for value in blocked_contexts):
            raise ValueError("identity-gate retest requires explicit capability contexts")
        entry = self._entry(
            key,
            artifacts,
            commit,
            "RETESTING",
            {**(detail or {}), "blocked_contexts": blocked_contexts},
            None,
        )
        try:
            entry.update({
                "gate_protocol": GATE_PROTOCOL_VERSION,
                "dgx_monarch": __version__,
                "dgx_source": runtime_provenance.cached_dgx_source_manifest_sha256(),
            })
            self._append(entry, strict_lock=True)
        except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as exc:
            raise GateLedgerWriteError(
                f"could not durably begin identity-gate retest in {self.path!r}"
            ) from exc

    _entries, _scan_entries = entries, entries_with_integrity
