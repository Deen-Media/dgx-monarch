"""Own one terminal dashboard beside a foreground ComfyUI subprocess.

A private connection controls only the dashboard launched here. Shared terminal
servers, existing drivers, and Worker services are never cleanup targets.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .comfy_top_process import DRIVER_GRACE_SECONDS, close_terminal, diagnostic, settle_driver, signal_driver
from .comfy_top_signal import ForegroundDriver


def prerequisites(config: str) -> list[str]:
    from ..config import load_cluster_config

    path = Path(config)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("--with-top requires an existing absolute --cluster-config")
    load_cluster_config(path)
    try:
        import textual.app  # noqa: F401

        from ..tui.app import run  # noqa: F401
    except ImportError as exc:
        raise ValueError("--with-top needs the TUI extra in the selected interpreter; use the constrained procedure in docs/INSTALL.md") from exc
    terminal = shutil.which("gnome-terminal")
    if terminal:
        return [terminal, "--wait", "--window", "--title=DGX Monarch dashboard", "--"]
    terminal = shutil.which("xterm")
    if terminal:
        return [terminal, "-T", "DGX Monarch dashboard", "-e"]
    raise ValueError("--with-top requires gnome-terminal or xterm; install one explicitly or launch dgxm top in another terminal")


def dashboard(config: str, host: str, ready: str) -> int:
    prerequisites(config)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError("dashboard terminal did not provide a visible interactive TTY")
    from .main import main

    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(ready)
        finished = threading.Event()
        def monitor_parent():
            try:
                connection.recv(1)
            except OSError:
                pass
            # The control channel can only stop this dashboard process. It
            # never receives a PID, command, or signal number from the parent.
            if not finished.is_set():
                os.kill(os.getpid(), signal.SIGTERM)
                if not finished.wait(3):
                    os._exit(143)  # This dashboard alone, if its main thread ignores TERM.
        threading.Thread(target=monitor_parent, daemon=True).start()
        connection.sendall(b"ready\n")
        try:
            return main(["--config", config, "top", "--host", host])
        finally:
            finished.set()


def _foreground(group: int) -> None:
    previous = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
    try:
        os.tcsetpgrp(sys.stdin.fileno(), group)
    finally:
        signal.signal(signal.SIGTTOU, previous)


def _resume_stopped_driver(driver: subprocess.Popen, foreground: int) -> None:
    stopped = os.waitid(os.P_PID, driver.pid, os.WSTOPPED | os.WNOHANG)
    if stopped is None:
        return
    # Preserve shell job control: hand its job group back before stopping this
    # supervisor. On `fg`, resume the child in the foreground again.
    _foreground(foreground)
    previous = signal.signal(signal.SIGTSTP, signal.SIG_DFL)
    try:
        os.kill(os.getpid(), signal.SIGTSTP)
    finally:
        signal.signal(signal.SIGTSTP, previous)
    if driver.poll() is None:
        _foreground(driver.pid)
        signal_driver(driver, signal.SIGCONT)


def supervise(
    config: str, host: str, command: list[str], *, timeout: float = 10,
    shutdown_grace: float = DRIVER_GRACE_SECONDS, terminal_grace: float = 3,
) -> int:
    terminal_command = prerequisites(config)
    if not sys.stdin.isatty():
        raise ValueError("--with-top must run in your visible terminal; use the desktop shortcut or an interactive terminal")
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ValueError("--with-top requires a graphical display; use a separate dgxm top terminal on headless hosts")
    terminal: subprocess.Popen | None = None
    driver: subprocess.Popen | None = None
    bridge: ForegroundDriver | None = None
    received = 0
    settlement_attempted = False
    channel: socket.socket | None = None
    foreground = os.tcgetpgrp(sys.stdin.fileno())
    handlers = {}

    def forward(signum, _frame):
        nonlocal received
        received = signum
        if driver is not None and driver.poll() is None:
            signal_driver(driver, signum)
        elif channel is not None:
            channel.close()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            handlers[signum] = signal.signal(signum, forward)
        with tempfile.TemporaryDirectory(prefix="dgxm-top-") as temporary:
            with socket.socket(socket.AF_UNIX) as ready:
                ready.bind(str(Path(temporary) / "ready"))
                ready.listen(1)
                ready.settimeout(0.1)
                terminal = subprocess.Popen([
                    *terminal_command, sys.executable,
                    "-m", "dgx_monarch.cli.comfy_top", "--dashboard", "--cluster-config", config,
                    "--host", host, "--ready", str(Path(temporary) / "ready"),
                ], start_new_session=True)
                deadline = time.monotonic() + timeout
                while True:
                    if received:
                        return 128 + received
                    if terminal.poll() is not None:
                        raise ValueError("dashboard terminal exited before startup; ComfyUI was not started")
                    if time.monotonic() >= deadline:
                        raise ValueError("dashboard terminal startup timed out; ComfyUI was not started")
                    try:
                        connection, _ = ready.accept()
                    except TimeoutError:
                        continue
                    channel = connection
                    channel.settimeout(1)
                    greeting = b""
                    while len(greeting) < 6:
                        piece = channel.recv(6 - len(greeting))
                        if not piece:
                            break
                        greeting += piece
                    if greeting != b"ready\n":
                        raise ValueError("dashboard terminal startup was not confirmed")
                    break
                if received:
                    return 128 + received
                if terminal.poll() is not None:
                    raise ValueError("dashboard terminal closed during startup; ComfyUI was not started")
                if not command:
                    # The listener belongs to somebody else. Own only this dashboard.
                    while terminal.poll() is None:
                        if received:
                            return 128 + received
                        try:
                            terminal.wait(timeout=0.2)
                        except subprocess.TimeoutExpired:
                            pass
                    result = terminal.wait(timeout=0)
                    return 128 + received if received else (128 - result if result < 0 else result)
                bridge = ForegroundDriver(command)
                driver = bridge.driver
                if driver is None:
                    raise OSError("foreground driver startup was not confirmed")
                received = received or bridge.observed_signal()
                if received:
                    return 128 + received
                _foreground(driver.pid)
                received = received or bridge.observed_signal()
                if received:
                    return 128 + received
                bridge.release()
                warned = False
                while driver.poll() is None:
                    received = received or bridge.observed_signal()
                    if received:
                        settlement_attempted = True
                        result = settle_driver(driver, grace=shutdown_grace)
                        return 128 - result if result < 0 else result
                    _resume_stopped_driver(driver, foreground)
                    if terminal.poll() is not None and not warned:
                        diagnostic("Dashboard closed; ComfyUI continues in this terminal.")
                        warned = True
                    try:
                        driver.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        pass
                result = driver.wait()
                return 128 - result if result < 0 else result
    finally:
        # Attempt independent cleanup even when one owned process cannot settle.
        pending = sys.exception()
        cleanup_error: BaseException | None = None
        try:
            if driver is not None and driver.poll() is None and not settlement_attempted:
                signal_driver(driver, signal.SIGTERM)
                settle_driver(driver, grace=shutdown_grace)
        except BaseException as exc:
            cleanup_error = exc
        try:
            if bridge is not None:
                bridge.close()
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        try:
            _foreground(foreground)
        except OSError:
            pass  # The controlling terminal may already be gone after SIGHUP.
        try:
            if channel is not None:
                channel.close()
            if terminal is not None:
                close_terminal(terminal, grace=terminal_grace)
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
        if cleanup_error is not None:
            if pending is None:
                raise cleanup_error
            diagnostic("Combined launcher cleanup remains unconfirmed; inspect owned processes before reuse.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--ready")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.check:
            prerequisites(args.cluster_config)
            return 0
        if args.dashboard:
            if not args.ready:
                raise ValueError("missing dashboard startup channel")
            return dashboard(args.cluster_config, args.host, args.ready)
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        return supervise(args.cluster_config, args.host, command)
    except (OSError, ValueError) as exc:
        diagnostic(f"combined launcher: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
