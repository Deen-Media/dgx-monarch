"""Byte-verify certificate for the zero-copy weight slab.

Each tensor region is compared while pristine and under the slab loop's pinned
inode checks. At issue time, the certificate records the verified counts and
digests the tensor layout and the JSON header. Tensors no larger than
``FULL_VERIFY_MAX_BYTES`` are verified whole.

It says nothing about bytes after adoption or rendered pixels. It never grants
identity-gate PASS, converts INCONCLUSIVE, or lifts quarantine. Sampled mode
reads at most three 4096-byte windows per tensor; full mode compares every byte
in one sequential pass. ``SlabCertificateError`` fails the load closed and
remains distinct from dtype, capacity, and header-parse failures so fallback
handlers cannot absorb corruption.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from ..log import get_logger
from ..refusal import RefusalClass, refusal

log = get_logger(__name__)

CERTIFICATE_VERSION = 1

MODE_SAMPLED = "sampled"
MODE_FULL = "full"
SLAB_CERTIFY_ENV = "DGXM_SLAB_CERTIFY"

# Shared with lazy unbake so both paths use one torch-free window policy.
FINGERPRINT_WINDOW = 4096

# Three windows already cover tensors at or below this threshold.
FULL_VERIFY_MAX_BYTES = 3 * FINGERPRINT_WINDOW

_HEADER_CHUNK = 4 << 20

# Bounded full-mode slices avoid two whole-tensor transient copies.
_COMPARE_CHUNK = 4 << 20

# Read-only after import. These strings go into ledger rows, so each must keep
# naming what the check does: a byte compare of checkpoint against arena.
# Renaming one changes the meaning of every row already written.
_ALGORITHMS = {MODE_SAMPLED: "memcmp-tensor-sampled-v1", MODE_FULL: "memcmp-tensor-full-v1"}


class SlabCertificateError(RuntimeError):
    """A slab region does not hold the checkpoint bytes it claims to hold."""


# Class P: consent cannot make mismatched physical bytes valid.
_ABORTED = refusal(
    RefusalClass.PHYSICS,
    "slab byte-verify FAILED. The load was aborted and nothing was adopted. Nothing was "
    "quarantined; this is not a model-correctness finding. Check the file's sha256 against "
    "the source, check the kernel log for IO errors, then use a re-staged copy of the "
    "artifact instead.",
)


def fingerprint_windows(nbytes: int) -> list[tuple[int, int]]:
    """Start / middle / end sampling windows for a tensor of ``nbytes``."""
    width = min(FINGERPRINT_WINDOW, nbytes)
    if width == 0:
        return []
    middle = max(0, min(nbytes - width, (nbytes - width) // 2))
    return sorted({(0, width), (middle, width), (nbytes - width, width)})


def certificate_mode() -> str:
    """Verification mode from the environment; never raises, defaults sampled."""
    raw = (os.environ.get(SLAB_CERTIFY_ENV) or "").strip().lower()
    if not raw:
        return MODE_SAMPLED
    if raw in (MODE_SAMPLED, MODE_FULL):
        return raw
    log.warning(
        "%s=%r is not %r or %r; verifying slab bytes in %s mode",
        SLAB_CERTIFY_ENV, raw, MODE_SAMPLED, MODE_FULL, MODE_SAMPLED,
    )
    return MODE_SAMPLED


@dataclass(frozen=True, slots=True)
class SlabByteCertificate:
    """Attestation that a slab holds its checkpoint's bytes."""

    certificate_version: int
    mode: str
    file_name: str            # basename only: no host paths in logs or ledger rows
    file_identity: str        # store_family.file_identity() spelling, per rank
    header_len: int
    header_digest: str        # sha256 of bytes [0, 8 + header_len)
    layout_digest: str        # sha256 of the canonical per-tensor layout lines
    tensor_count: int
    total_bytes: int
    verified_tensors: int
    fully_verified_tensors: int
    verified_bytes: int
    issued_at: float
    verify_seconds: float
    # Cross-rank binding is content-derived because local inode identity differs.
    checkpoint_bytes: int = 0
    artifact_signature: str = ""

    def digest_summary(self) -> str:
        """Compact greppable form for the load log line and the ledger row."""
        return (
            f"v{self.certificate_version}/{self.mode}/{self.layout_digest[:16]}/"
            f"{self.tensor_count}t/{self.verified_tensors}v/"
            f"{self.fully_verified_tensors}q/"
            f"{self.verified_bytes / (1 << 20):.1f}MiB"
        )

    def public(self) -> dict[str, Any]:
        """Return one JSON-native shape for status, telemetry, and v9 rows.

        The result includes the frozen certificate wire keys consumed by
        ``fold_rank_certificates``.
        """
        out: dict[str, Any] = dict(asdict(self))
        out["issued_at"] = round(self.issued_at, 3)
        out["verify_seconds"] = round(self.verify_seconds, 3)
        out["summary"] = self.digest_summary()
        out["algorithm"] = _ALGORITHMS.get(self.mode, _ALGORITHMS[MODE_SAMPLED])
        out["digest"] = self.layout_digest
        out["complete"] = self.verified_tensors == self.tensor_count
        out["tensors_verified"] = self.verified_tensors
        out["tensors_total"] = self.tensor_count
        out["bytes_verified"] = self.verified_bytes
        return out


class CertificateBuilder:
    """Accumulates per-region evidence during one slab read, then issues."""

    def __init__(
        self,
        *,
        path: str,
        file_identity: str,
        header_len: int,
        tensor_count: int,
        total_bytes: int,
        mode: str | None = None,
    ) -> None:
        self.path = path
        self.file_name = os.path.basename(path)
        self.file_identity = file_identity
        self.header_len = int(header_len)
        self.tensor_count = int(tensor_count)
        self.total_bytes = int(total_bytes)
        self.mode = mode or certificate_mode()
        self.verified_tensors = 0
        self.fully_verified_tensors = 0
        self.verified_bytes = 0
        self.verify_seconds = 0.0
        self._layout = hashlib.sha256()

    def windows_for(self, nbytes: int) -> list[tuple[int, int]]:
        if self.mode == MODE_FULL or nbytes <= FULL_VERIFY_MAX_BYTES:
            return [(0, nbytes)]
        return fingerprint_windows(nbytes)

    def verify_region(self, fd: int, name: str, ft: Any, arena_mm: Any, offset: int) -> None:
        """Compare one just-read region, using ``offset`` in the slab arena."""
        t0 = time.monotonic()
        nbytes = int(ft.nbytes)
        windows = self.windows_for(nbytes)
        for w_off, w_len in windows:
            done = 0
            while done < w_len:
                span = min(_COMPARE_CHUNK, w_len - done)
                at = w_off + done
                raw = os.pread(fd, span, int(ft.start) + at)
                if len(raw) != span:
                    raise SlabCertificateError(
                        _ABORTED + " " + self._short_read_text(name, ft, at, span, len(raw)))
                mine = bytes(memoryview(arena_mm)[offset + at:offset + at + span])
                if raw != mine:
                    raise SlabCertificateError(
                        _ABORTED + " " + self._mismatch_text(name, ft, offset, at, span, raw, mine))
                done += span
            self.verified_bytes += w_len
        self._layout.update(
            f"{name}\0{ft.dtype}\0{tuple(ft.shape)}\0{int(ft.start)}\0{nbytes}\0{offset}\n".encode()
        )
        self.verified_tensors += 1
        if windows and windows[0][1] == nbytes:
            self.fully_verified_tensors += 1
        self.verify_seconds += time.monotonic() - t0

    def issue(self, fd: int) -> SlabByteCertificate:
        """Fold the header in and issue, or refuse if a tensor was skipped."""
        if self.verified_tensors != self.tensor_count:
            raise SlabCertificateError(
                _ABORTED + f" Incomplete for {self.file_name}: {self.verified_tensors} of "
                f"{self.tensor_count} tensors were verified. This is a defect in the slab read "
                "loop, not a checkpoint fault: report it with this message."
            )
        t0 = time.monotonic()
        header_digest = self._header_digest(fd)
        self.verify_seconds += time.monotonic() - t0
        return SlabByteCertificate(
            certificate_version=CERTIFICATE_VERSION,
            mode=self.mode,
            file_name=self.file_name,
            file_identity=self.file_identity,
            header_len=self.header_len,
            header_digest=header_digest,
            layout_digest=self._layout.hexdigest(),
            tensor_count=self.tensor_count,
            total_bytes=self.total_bytes,
            verified_tensors=self.verified_tensors,
            fully_verified_tensors=self.fully_verified_tensors,
            verified_bytes=self.verified_bytes,
            issued_at=time.time(),
            verify_seconds=self.verify_seconds,
            checkpoint_bytes=self._checkpoint_bytes(fd),
            artifact_signature=self._artifact_signature(),
        )

    def _checkpoint_bytes(self, fd: int) -> int:
        """Return verified-inode size for host-stable cross-rank binding."""
        try:
            return int(os.fstat(fd).st_size)
        except OSError:
            return 0

    def _artifact_signature(self) -> str:
        """Return the ledger's content-derived per-file digest for this rank.

        Fold rejects read-failure sentinels, yielding no row instead of false
        cross-rank agreement.
        """
        from ..gate_artifacts import artifact_signature

        return str(artifact_signature(self.path))

    def _header_digest(self, fd: int) -> str:
        """Stream sha256 over the length prefix and header, up to 256 MiB."""
        digest = hashlib.sha256()
        remaining = 8 + self.header_len
        position = 0
        while remaining > 0:
            chunk = os.pread(fd, min(_HEADER_CHUNK, remaining), position)
            if not chunk:
                raise SlabCertificateError(
                    _ABORTED + f" The safetensors header of {self.file_name} ended at byte "
                    f"{position} of a declared {8 + self.header_len}."
                )
            digest.update(chunk)
            position += len(chunk)
            remaining -= len(chunk)
        return digest.hexdigest()

    def _short_read_text(self, name: str, ft: Any, w_off: int, w_len: int, got: int) -> str:
        return (
            f"slab byte-verify FAILED for {self.file_name}: re-reading tensor {name!r} at file "
            f"offset {int(ft.start)}+{w_off} returned {got} bytes of {w_len}. The checkpoint is "
            "being truncated or the storage is failing under it."
        )

    def _mismatch_text(self, name: str, ft: Any, offset: int, w_off: int,
                       w_len: int, raw: bytes, mine: bytes) -> str:
        first = next((i for i in range(w_len) if raw[i] != mine[i]), 0)
        return (
            f"slab byte-verify FAILED for {self.file_name}: tensor {name!r} differs from the "
            f"checkpoint at file offset {int(ft.start)}+{w_off} (slab offset {offset}+{w_off}), "
            f"first differing byte {first} of a {w_len} byte window. The checkpoint or this "
            "machine's storage is not returning the bytes it claims."
        )
