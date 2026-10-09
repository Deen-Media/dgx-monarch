"""Exact-inode checkpoint ownership for FSDP Comfy model loads."""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..transfer_utils import raise_with_distinct_cause, reconcile_error
from . import slab_lifetime

_PROC_FD_DIRECTORY = Path("/proc/self/fd")
# Proc lifetime: last-resort roots if the shared registry handoff is interrupted.
_PIN_POISON_OWNERS: list[Any] = []


def _forget_prepublished_pin(pin: Any) -> None:
    for index, owned in enumerate(_PIN_POISON_OWNERS):
        if owned is pin:
            _PIN_POISON_OWNERS.pop(index)
            return


@dataclass
class PinnedFsdpCheckpoint:
    """A Comfy-compatible path backed by the exact proven checkpoint inode."""

    quant_kind: str
    file_identity: Any
    loader_path: str
    _fd: int
    _directory: str
    _file_owner: Any
    _directory_owner: Any
    _closed: bool = False
    _close_uncertain: bool = False

    def close(self) -> None:
        """Remove the alias before releasing its descriptor for reuse."""
        if self._close_uncertain:
            raise RuntimeError(
                "pinned FSDP checkpoint descriptor close outcome is uncertain; "
                "reset the Attached mesh"
            )
        if self._closed:
            _forget_prepublished_pin(self)
            return
        try:
            os.unlink(self.loader_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            # Keeping the descriptor open is safer than leaving an alias that
            # could later resolve to an unrelated, reused descriptor number.
            raise RuntimeError(
                "failed to remove the pinned FSDP checkpoint alias"
            ) from exc
        # FileIO is a finalizable acquisition owner: if construction handoff is
        # interrupted, refcount cleanup closes it. Closing is still terminal on
        # ambiguity so a later retry can never target a reused descriptor.
        self._close_uncertain = True
        self._fd = -1
        failure: BaseException | None = None
        failure_cause: BaseException | None = None
        try:
            self._file_owner.close()
            self._closed = True
            self._close_uncertain = False
            self._file_owner = None
        except BaseException as file_error:
            failure = file_error
        try:
            if self._directory_owner is not None:
                self._directory_owner.cleanup()
                self._directory_owner = None
        except BaseException as directory_error:
            failure, failure_cause = reconcile_error(
                failure,
                directory_error,
                "pinned checkpoint directory cleanup also failed",
            )
        if self._closed:
            _forget_prepublished_pin(self)
        if failure is not None:
            raise_with_distinct_cause(failure, failure_cause)


def assert_fsdp_checkpoint_identity(path: str, proof: Any, phase: str) -> None:
    """Require the path to retain the exact inode whose header the proof read."""
    from ..mesh_safety import ArtifactBindingError
    from ..safetensors_header import SafetensorsFileIdentity

    expected = getattr(proof, "file_identity", None)
    if expected is None:
        raise ArtifactBindingError(
            "worker-local FSDP checkpoint proof lost its exact file identity"
        )
    try:
        current = SafetensorsFileIdentity.from_stat(os.stat(path))
    except OSError as exc:
        raise ArtifactBindingError(
            f"worker-local FSDP checkpoint changed while {phase}"
        ) from exc
    if current != expected:
        raise ArtifactBindingError(
            f"worker-local FSDP checkpoint changed while {phase}"
        )


def fsdp_reuse_matches_proof(stored: Any, proof: Any) -> bool:
    """Require an FSDP resident to own the exact freshly proved inode."""
    if proof is None:
        return True
    pin = getattr(stored, "fsdp_checkpoint_pin", None)
    return bool(
        pin is not None
        and not pin._closed
        and not pin._close_uncertain
        and pin._fd >= 0
        and pin.file_identity == getattr(proof, "file_identity", None)
    )


def _require_linux_proc_fd() -> None:
    """Require the kernel path used to keep the proven inode loadable."""
    from ..mesh_safety import ArtifactBindingError

    if sys.platform != "linux" or not _PROC_FD_DIRECTORY.is_dir():
        raise ArtifactBindingError(
            "FSDP exact checkpoint pinning requires Linux with procfs "
            "mounted at /proc/self/fd; use a Linux worker or a resident topology"
        )


def _retain_prepublished_pin(
    retained: PinnedFsdpCheckpoint,
    error: BaseException,
) -> BaseException | None:
    """Transfer a prepublished partial pin into the shared poison registry."""
    try:
        slab_lifetime.retain_failed_load_resource(retained, error)
    except BaseException as publish_error:
        # The process root was published before alias creation, so an interrupted
        # shared-registry handoff cannot orphan a live /proc fd alias.
        return publish_error
    else:
        _forget_prepublished_pin(retained)
        return None


def _discard_partial_fsdp_pin(
    pin: PinnedFsdpCheckpoint | None,
    directory_owner: Any,
    file_owner: Any,
) -> BaseException | None:
    """Clean a partial pin without ever replacing its construction failure."""
    if pin is not None:
        try:
            pin.close()
        except BaseException as cleanup_error:
            publish_error = _retain_prepublished_pin(pin, cleanup_error)
            if publish_error is not None:
                strongest, _cause = reconcile_error(
                    cleanup_error,
                    publish_error,
                    "partial pin ownership publication also failed",
                )
                return strongest
            return cleanup_error
        return None
    # No alias can exist before the pin is constructed and prepublished.
    failure: BaseException | None = None
    if file_owner is not None:
        try:
            file_owner.close()
        except BaseException as file_error:
            failure = file_error
    if directory_owner is not None:
        try:
            directory_owner.cleanup()
        except BaseException as directory_error:
            failure, _cause = reconcile_error(
                failure,
                directory_error,
                "partial pin directory cleanup also failed",
            )
    return failure


def pin_fsdp_checkpoint(
    path: str,
    proof: Any,
    *,
    handoff: list[PinnedFsdpCheckpoint] | None = None,
) -> PinnedFsdpCheckpoint | None:
    """Pin the exact proven inode behind a suffix-preserving Comfy alias."""
    if proof is None:
        return None
    from ..mesh_safety import ArtifactBindingError
    from ..safetensors_header import SafetensorsFileIdentity

    _require_linux_proc_fd()
    expected = getattr(proof, "file_identity", None)
    if expected is None:
        raise ArtifactBindingError(
            "worker-local FSDP checkpoint proof lost its exact file identity"
        )
    file_owner = None
    directory_owner = None
    alias = ""
    pin = None
    try:
        file_owner = open(path, "rb", buffering=0)
        fd = file_owner.fileno()
        if SafetensorsFileIdentity.from_stat(os.fstat(fd)) != expected:
            raise ArtifactBindingError(
                "worker-local FSDP checkpoint changed while pinning the Comfy model load"
            )
        directory_owner = tempfile.TemporaryDirectory(prefix="dgxm-fsdp-pin-")
        directory = directory_owner.name
        suffix = Path(path).suffix.lower()
        alias = os.path.join(directory, f"checkpoint{suffix}")
        pin = PinnedFsdpCheckpoint(
            quant_kind=proof.quant_kind,
            file_identity=expected,
            loader_path=alias,
            _fd=fd,
            _directory=directory,
            _file_owner=file_owner,
            _directory_owner=directory_owner,
        )
        # Publish exact ownership before the first instruction that can create
        # a live /proc fd alias. Cleanup only removes this root after confirmed
        # alias removal and descriptor close.
        _PIN_POISON_OWNERS.append(pin)
        os.symlink(f"/proc/{os.getpid()}/fd/{fd}", alias)
        if SafetensorsFileIdentity.from_stat(os.stat(alias)) != expected:
            raise ArtifactBindingError(
                "worker-local FSDP checkpoint changed while pinning the Comfy model load"
            )
        if handoff is not None:
            handoff.append(pin)
            _forget_prepublished_pin(pin)
        return pin
    except BaseException as exc:
        published = bool(
            pin is not None and handoff is not None
            and any(owner is pin for owner in handoff)
        )
        cleanup_error: BaseException | None = None
        if not published:
            cleanup_error = _discard_partial_fsdp_pin(
                pin,
                directory_owner,
                file_owner,
            )
        strongest: BaseException = exc
        cause: BaseException | None = None
        if cleanup_error is not None:
            strongest, cause = reconcile_error(
                exc,
                cleanup_error,
                "partial pinned checkpoint cleanup also failed",
            )
        if strongest is not exc:
            raise_with_distinct_cause(strongest, cause)
        if isinstance(exc, ArtifactBindingError) or not isinstance(exc, Exception):
            if cause is None:
                raise
            raise_with_distinct_cause(exc, cause)
        raise ArtifactBindingError(
            "worker-local FSDP checkpoint could not be pinned for the Comfy model load"
        ) from exc


def close_store_pins(store: Any) -> None:
    """Close each resident's pin without unloading it.

    A worker stopped with its models loaded never runs the unload that
    removes the alias, and a killed process leaves the alias on disk.
    """
    for stored in (getattr(store, "current", None), getattr(store, "uncond", None)):
        pin = getattr(stored, "fsdp_checkpoint_pin", None)
        if pin is not None:
            pin.close()
