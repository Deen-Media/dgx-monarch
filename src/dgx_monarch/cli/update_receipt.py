"""Verified-update receipts timed over the whole run, and the receipt an interrupting exception carries."""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Protocol

from .operator_receipt import (
    OperatorReceiptBuilder,
    ReceiptStatus,
    ReceiptStepStatus,
)


class ReceiptStep(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def status(self) -> ReceiptStepStatus: ...
    @property
    def counts(self) -> Mapping[str, int] | None: ...
    @property
    def notes(self) -> Sequence[str]: ...


class UpdateReceipt:
    """Capture timing at entry even though target identity resolves later."""

    def __init__(self) -> None:
        self._started_at = datetime.now(UTC)
        self._started_tick = time.monotonic()

    def finish(
        self,
        steps: Sequence[ReceiptStep],
        status: ReceiptStatus,
        *,
        target: str | None,
        source: str | None,
        notes: Sequence[str],
    ) -> Mapping[str, object]:
        clocks = iter((self._started_at, datetime.now(UTC)))
        ticks = iter((self._started_tick, time.monotonic()))
        builder = OperatorReceiptBuilder(
            "update", target=target,
            source_hashes={"dgx_monarch": source} if source is not None else None,
            clock=lambda: next(clocks), timer=lambda: next(ticks),
        )
        for step in steps:
            builder.add_step(
                step.name, step.status, counts=step.counts, notes=step.notes
            )
        return builder.finish(status, notes=notes)


_INTERRUPTED_RECEIPT = "_dgxm_interrupted_update_receipt"


def attach_interrupted_receipt(
    error: BaseException, receipt: Mapping[str, object]
) -> None:
    try:
        object.__setattr__(error, _INTERRUPTED_RECEIPT, receipt)
    except BaseException:
        pass


def interrupted_receipt(error: BaseException) -> Mapping[str, object] | None:
    try:
        value = object.__getattribute__(error, _INTERRUPTED_RECEIPT)
    except BaseException:
        return None
    return value if isinstance(value, Mapping) else None
