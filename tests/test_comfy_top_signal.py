"""Signal observer startup/lifetime without a GUI or model runtime."""
from __future__ import annotations

import subprocess
import sys

import pytest

from dgx_monarch.cli import comfy_top_signal


def test_driver_cannot_execute_before_ready_foreground_release(tmp_path):
    marker = tmp_path / "driver-executed"
    command = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]
    bridge = comfy_top_signal.ForegroundDriver(command)
    try:
        assert not marker.exists()
        assert bridge.driver is not None
        bridge.release()
        assert bridge.driver.wait(timeout=3) == 0
        assert marker.exists()
    finally:
        bridge.close()
    assert bridge.observer is not None and bridge.observer.returncode == 0


def test_observer_handshake_failure_never_executes_driver_and_reaps_children(tmp_path, monkeypatch):
    marker = tmp_path / "must-not-exist"
    real_popen = subprocess.Popen
    children = []
    def spawn(command, **kwargs):
        if command[4] == "observe":
            command = [sys.executable, "-I", "-S", "-c", "pass"]
        child = real_popen(command, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(comfy_top_signal.subprocess, "Popen", spawn)
    with pytest.raises(OSError, match="observer did not become ready"):
        comfy_top_signal.ForegroundDriver([
            sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ])
    assert not marker.exists()
    assert len(children) == 2 and all(child.poll() is not None for child in children)
