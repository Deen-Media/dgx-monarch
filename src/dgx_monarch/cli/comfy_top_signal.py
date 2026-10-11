"""Private pipe observer for terminal signals delivered to the driver group.

The child entry points run as standalone isolated stdlib scripts. Neither reads
terminal stdin. Parent-pipe EOF stops the observer even if the parent is killed.
"""
from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time


class ForegroundDriver:
    """Start a gated driver and ready observer before foreground transfer."""

    def __init__(self, command: list[str], *, timeout: float = 5) -> None:
        self.driver: subprocess.Popen | None = None
        self.observer: subprocess.Popen | None = None
        self.fds: set[int] = set()
        try:
            self.events, events_write = self._pipe()
            life_read, self.life = self._pipe()
            gate_read, self.gate = self._pipe()
            os.set_blocking(self.events, False)
            os.set_blocking(events_write, False)
            script = os.path.abspath(__file__)
            self.driver = subprocess.Popen(
                [sys.executable, "-I", "-S", script, "driver", str(gate_read), *command],
                pass_fds=(gate_read,), process_group=0,
            )
            self.observer = subprocess.Popen(
                [sys.executable, "-I", "-S", script, "observe", str(events_write), str(life_read)],
                pass_fds=(events_write, life_read), process_group=self.driver.pid,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for fd in (events_write, life_read, gate_read):
                self._close(fd)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if select.select([self.events], [], [], .1)[0]:
                    if os.read(self.events, 1) == b"R":
                        return
                    break
            raise OSError("foreground signal observer did not become ready; ComfyUI was not started")
        except BaseException:
            cleanup_failed = False
            try:
                if self.driver is not None and self.driver.poll() is None:
                    os.killpg(self.driver.pid, signal.SIGKILL)
                    self.driver.wait(timeout=5)
            except BaseException:
                cleanup_failed = True
            try:
                self.close()
            except BaseException:
                cleanup_failed = True
            if cleanup_failed:
                from .comfy_top_process import diagnostic

                diagnostic("Foreground startup cleanup remains unconfirmed; inspect owned processes before retrying.")
            raise

    def _pipe(self) -> tuple[int, int]:
        pair = os.pipe2(os.O_CLOEXEC)
        self.fds.update(pair)
        return pair

    def _close(self, fd: int) -> None:
        if fd in self.fds:
            self.fds.remove(fd)
            os.close(fd)

    def release(self) -> None:
        os.write(self.gate, b"G")
        self._close(self.gate)

    def observed_signal(self) -> int:
        try:
            events = os.read(self.events, 1024)
        except BlockingIOError:
            return 0
        if not events:
            raise OSError("foreground signal observer exited; stopping the owned driver safely")
        return next((value for value in events if value in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)), 0)

    def close(self) -> None:
        for fd in tuple(self.fds):
            self._close(fd)
        if self.observer is not None:
            try:
                self.observer.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.observer.kill()  # Exact child handle, never a possibly retired group.
                self.observer.wait(timeout=5)


def _observe(events: int, life: int) -> None:
    def observed(signum, _frame):
        try:
            os.write(events, bytes([signum]))
        except BlockingIOError:
            pass  # A queued observation already makes the parent cancel.
        except OSError:
            os._exit(0)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, observed)
    os.write(events, b"R")
    os.read(life, 1)  # Private parent pipe, never terminal stdin.


def _main() -> None:
    if sys.argv[1] == "observe":
        _observe(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "driver":
        gate = int(sys.argv[2])
        if os.read(gate, 1) != b"G":
            raise SystemExit(2)
        os.close(gate)
        os.execv(sys.argv[3], sys.argv[3:])  # noqa: S606 - exact selected executable, no shell
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    _main()
