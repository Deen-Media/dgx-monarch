"""Run a verified update under the update lock and publish its receipt."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .operator_receipt import receipt_destination, write_receipt
from .update_lock import UpdateLock, UpdateLockUnavailable
from .update_receipt import interrupted_receipt
from .update_transaction import UpdateOps, UpdateRequest, execute_verified_update


def run_serialized_update(
    *,
    repo: Path,
    request: UpdateRequest,
    ops: UpdateOps,
    receipt_path: str | os.PathLike[str] | None,
) -> int:
    # The --receipt check runs here, before the lock and the update. Checked only
    # at write time, a relative or nameless path would fail after the update had
    # run on every host, with no receipt and exit 1.
    if receipt_path is not None:
        try:
            receipt_destination(receipt_path)
        except (TypeError, ValueError) as exc:
            print(f"error: --receipt: {exc}", file=sys.stderr)
            return 2
    try:
        lock = UpdateLock(repo)
        lock.__enter__()
    except (OSError, UpdateLockUnavailable):
        print("another dgxm update is already active, or the update lock could not be taken safely", file=sys.stderr)
        return 2
    try:
        try:
            result = execute_verified_update(request, ops)
        except BaseException as exc:
            receipt = interrupted_receipt(exc)
            if receipt is not None:
                try:
                    write_receipt(receipt, receipt_path)
                except BaseException:
                    pass
            raise
        try:
            published = write_receipt(result.receipt, receipt_path)
        except (OSError, TypeError, ValueError) as exc:
            print(
                f"{result.message}; receipt publication failed: {type(exc).__name__}",
                file=sys.stderr,
            )
            return 1
        stream = sys.stdout if result.succeeded else sys.stderr
        print(result.message, file=stream)
        print(f"receipt: {published}", file=stream)
        return result.exit_code
    finally:
        lock.__exit__(None, None, None)
