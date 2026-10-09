"""Fail-closed descriptor ownership for weight-slab backing arenas."""
from __future__ import annotations

import ctypes
import io
import logging
import mmap
import os
from dataclasses import dataclass
from typing import Any

from ..transfer_utils import raise_with_distinct_cause, reconcile_error
from . import slab_lifetime

log = logging.getLogger("dgx-monarch")

_ALIGN = 256  # cuBLAS/vectorized-load friendly tensor alignment


@dataclass
class _OwnedDescriptor:
    """A numeric FD whose ambiguous release is terminal for this process."""

    _file: io.FileIO
    label: str
    _closed: bool = False
    _close_uncertain: bool = False

    @classmethod
    def open(cls, path: str, label: str) -> _OwnedDescriptor:
        """Acquire through a finalizable C owner, never a bare integer FD."""
        return cls(io.FileIO(path, "r"), label)

    @property
    def _fd(self) -> int:
        return self._file.fileno()

    def close(self) -> None:
        if self._close_uncertain:
            raise RuntimeError(
                f"{self.label} close outcome is uncertain; reset the Attached mesh"
            )
        if self._closed:
            return
        self._close_uncertain = True
        self._file.close()
        self._closed = True
        self._close_uncertain = False


def _discard_identity(handoff: list[Any] | None, owned: Any) -> None:
    if handoff is None:
        return
    for index, candidate in enumerate(handoff):
        if candidate is owned:
            handoff.pop(index)
            return


class _SlabHookRestore:
    """Exact, retryable owner for Comfy's two process-global slab hooks."""

    def __init__(self, utils: Any, base_model: Any, ltf: Any, lmw: Any, slab: Any):
        self.utils = utils
        self.base_model = base_model
        self.ltf = ltf
        self.lmw = lmw
        self.slab = slab
        self._poisoned_before: bool | None = None

    def prepublish(self) -> None:
        self._poisoned_before = slab_lifetime.prepublish_resource(self)

    def bind(self) -> None:
        self.slab._close_guard = self

    def _restore_ltf(self) -> BaseException | None:
        failure: BaseException | None = None
        for _attempt in range(2):
            try:
                self.utils.load_torch_file = self.ltf
                if self.utils.load_torch_file is self.ltf:
                    return (
                        failure
                        if failure is not None
                        and not isinstance(failure, Exception)
                        else None
                    )
            except BaseException as error:
                failure, _cause = reconcile_error(
                    failure,
                    error,
                    "earlier Comfy load_torch_file restoration failure",
                )
        if failure is not None:
            return failure
        return RuntimeError("Comfy load_torch_file restoration was unconfirmed")

    def _restore_lmw(self) -> BaseException | None:
        failure: BaseException | None = None
        for _attempt in range(2):
            try:
                self.base_model.load_model_weights = self.lmw
                if self.base_model.load_model_weights is self.lmw:
                    return (
                        failure
                        if failure is not None
                        and not isinstance(failure, Exception)
                        else None
                    )
            except BaseException as error:
                failure, _cause = reconcile_error(
                    failure,
                    error,
                    "earlier Comfy load_model_weights restoration failure",
                )
        if failure is not None:
            return failure
        return RuntimeError("Comfy load_model_weights restoration was unconfirmed")

    def close(self) -> None:
        ltf_error = None
        lmw_error = None
        try:
            try:
                ltf_error = self._restore_ltf()
            except BaseException as restore_error:
                ltf_error = restore_error
        finally:
            try:
                lmw_error = self._restore_lmw()
            except BaseException as restore_error:
                lmw_error = restore_error
        failure: BaseException | None = None
        failure_cause: BaseException | None = None
        for label, candidate_error in (
            ("Comfy load_torch_file restoration also failed", ltf_error),
            ("Comfy load_model_weights restoration also failed", lmw_error),
        ):
            if candidate_error is not None:
                failure, failure_cause = reconcile_error(
                    failure, candidate_error, label)
        if failure is not None:
            raise_with_distinct_cause(failure, failure_cause)
        poisoned_before = self._poisoned_before
        if poisoned_before is not None:
            slab_lifetime.confirm_prepublished_resource(self, poisoned_before)
            self._poisoned_before = None
        slab = self.slab
        if slab is not None:
            if getattr(slab, "_close_guard", None) is self:
                slab._close_guard = None
            self.slab = None


def _finish_prepublished(
    resource: Any,
    primary: BaseException | None,
) -> None:
    """Confirm close/restore while preserving cancellation precedence."""
    try:
        resource.close()
    except BaseException as error:
        failure: BaseException | None = primary
        failure_cause: BaseException | None = None
        failure, failure_cause = reconcile_error(
            failure,
            error,
            "provisional resource restore also failed",
        )
        try:
            slab_lifetime.retain_failed_load_resource(resource, error)
        except BaseException as retain_error:
            failure, failure_cause = reconcile_error(
                failure,
                retain_error,
                "failed restore ownership publication also failed",
            )
        raise_with_distinct_cause(failure, failure_cause)


class _Arena:
    """One memfd + MAP_SHARED mapping with an append-only aligned layout."""

    def __init__(self, name: str, capacity: int, *, handoff: list[_Arena] | None = None):
        if capacity <= 0:
            raise ValueError(f"slab arena capacity must be positive, got {capacity}")
        self.fd = -1
        self.mm: mmap.mmap | None = None
        self.base = 0
        self.capacity = capacity
        self.cursor = 0
        self.closed = False
        self.close_uncertain = False
        self.acquisition_uncertain = False
        self._provisional_poisoned_before: bool | None = None
        self._handoff = handoff
        if handoff is not None:
            handoff.append(self)
        try:
            self._provisional_poisoned_before = slab_lifetime.prepublish_resource(self)
            # A bare integer FD has no finalizer. If Python is interrupted after
            # memfd_create returns but before STORE_ATTR, retain this prepublished
            # intent and poison the process; the kernel owns the unknown FD until
            # recycle, and cleanup must never guess or retry its number.
            self.fd = os.memfd_create(name, 0)
            os.ftruncate(self.fd, capacity)
            self.mm = mmap.mmap(self.fd, capacity, mmap.MAP_SHARED)
            if self.mm is None:  # defensive for alternate mmap implementations
                raise RuntimeError("mmap returned no mapping")
            # A persistent ctypes view would make mmap.close() raise BufferError.
            self.base = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        except BaseException as error:
            # Preserve the construction exception. The retained owner/poison
            # survives any ambiguous mapping or descriptor cleanup outcome.
            if self.fd < 0:
                self.acquisition_uncertain = True
                try:
                    slab_lifetime.retain_failed_load_resource(self, error)
                except BaseException as retain_error:
                    strongest, cause = reconcile_error(
                        error,
                        retain_error,
                        "slab arena ownership retention also failed",
                    )
                    raise_with_distinct_cause(strongest, cause)
            else:
                cleanup_error: BaseException | None = None
                try:
                    slab_lifetime.close_or_retain_after_explicit_unload(
                        self, "partial slab arena"
                    )
                except BaseException as caught:
                    cleanup_error = caught
                if self.closed:
                    _discard_identity(handoff, self)
                if cleanup_error is not None:
                    strongest, cause = reconcile_error(
                        error,
                        cleanup_error,
                        "partial slab arena cleanup also failed",
                    )
                    raise_with_distinct_cause(strongest, cause)
            raise

    def confirm_handoff(self) -> None:
        poisoned_before = self._provisional_poisoned_before
        if poisoned_before is None:
            return
        slab_lifetime.confirm_prepublished_resource(self, poisoned_before)
        self._provisional_poisoned_before = None

    def place(self, nbytes: int) -> int:
        off = (self.cursor + _ALIGN - 1) // _ALIGN * _ALIGN
        if off + nbytes > self.capacity:
            raise RuntimeError(
                f"slab arena overflow: need {nbytes} at {off}, capacity {self.capacity}"
            )
        self.cursor = off + nbytes
        return off

    def contains(self, ptr: int) -> bool:
        return (
            not self.closed
            and not self.close_uncertain
            and self.base <= ptr < self.base + self.capacity
        )

    def close(self) -> None:
        if self.acquisition_uncertain:
            raise RuntimeError(
                "slab arena memfd acquisition outcome is uncertain; reset the Attached mesh"
            )
        if self.close_uncertain:
            raise RuntimeError(
                "slab arena descriptor close outcome is uncertain; reset the Attached mesh"
            )
        if self.closed:
            self.confirm_handoff()
            return
        try:
            if self.mm is not None:
                self.mm.close()
        except BufferError as exc:  # exported view: retry only after another unload
            log.warning(
                "slab arena %d: mapping still referenced at close; "
                "retaining ownership until retry or process recycle",
                self.fd,
            )
            raise RuntimeError(
                "slab arena mapping is still referenced; reset the Attached mesh "
                "if explicit unload cannot release it"
            ) from exc
        # An error/interrupt can follow close(2) releasing the number. Publish
        # terminal uncertainty before the syscall and never retry that FD.
        self.close_uncertain = True
        fd, self.fd = self.fd, -1
        os.close(fd)
        self.closed = True
        self.close_uncertain = False
        self.confirm_handoff()
