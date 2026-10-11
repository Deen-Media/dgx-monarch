"""Bounded settlement of subprocesses created by the combined launcher."""
from __future__ import annotations

import os
import signal
import subprocess
import sys

# Match the bounded mesh recycle window (mesh_recycle.py). This is a
# process shutdown allowance, not proof that mesh or actor teardown succeeded.
DRIVER_GRACE_SECONDS = 180.0
REAP_SECONDS = 5.0


def diagnostic(message: str) -> None:
    """A vanished terminal must never prevent owned-process settlement."""
    try:
        print(message, file=sys.stderr)
    except Exception:
        pass


def signal_driver(driver: subprocess.Popen, signum: int) -> None:
    """Signal the new driver's group only while its unreaped leader is live."""
    if driver.poll() is None:
        try:
            os.killpg(driver.pid, signum)
        except ProcessLookupError:
            pass


def settle_driver(driver: subprocess.Popen, *, grace: float = DRIVER_GRACE_SECONDS) -> int:
    """After a forwarded signal, allow teardown, then bound forced settlement."""
    term_grace = min(5.0, grace)
    try:
        return driver.wait(timeout=max(0.0, grace - term_grace))
    except subprocess.TimeoutExpired:
        signal_driver(driver, signal.SIGTERM)
    try:
        return driver.wait(timeout=term_grace)
    except subprocess.TimeoutExpired:
        diagnostic("ComfyUI did not exit within the shutdown grace; forcing the owned driver group to stop. "
              "Attached-mesh and actor cleanup is NOT confirmed; inspect cluster state before reuse.")
        signal_driver(driver, signal.SIGKILL)
    try:
        return driver.wait(timeout=REAP_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise OSError("owned ComfyUI exit remains unconfirmed after forced shutdown; inspect before reuse") from exc


def close_terminal(terminal: subprocess.Popen, *, grace: float = 3.0) -> None:
    """Reap only the local launch client, never its shared terminal server."""
    try:
        terminal.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        terminal.terminate()
    try:
        terminal.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        diagnostic("Dashboard terminal client did not exit; forcing only the owned client. "
              "Inspect any remaining dashboard window; its shared terminal server was not stopped.")
        terminal.kill()
        try:
            terminal.wait(timeout=REAP_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise OSError("owned dashboard terminal client exit remains unconfirmed") from exc
