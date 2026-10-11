"""A lost terminal cannot veto forced settlement of an owned child."""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

from dgx_monarch.cli import comfy_top_process


@pytest.mark.parametrize("kind", ["driver", "terminal"])
@pytest.mark.parametrize("error", [OSError, ValueError])
def test_broken_stderr_does_not_skip_kill_or_reap(tmp_path, monkeypatch, kind, error):
    ready = tmp_path / "ready"
    code = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(ready)!r}).touch(); time.sleep(30)"
    )
    child = subprocess.Popen([sys.executable, "-c", code], process_group=0)
    class BrokenTerminal:
        def write(self, _message):
            raise error("terminal disappeared")
    try:
        deadline = time.monotonic() + 3
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        with monkeypatch.context() as patch:
            patch.setattr(comfy_top_process.sys, "stderr", BrokenTerminal())
            if kind == "driver":
                assert comfy_top_process.settle_driver(child, grace=.01) == -9
            else:
                comfy_top_process.close_terminal(child, grace=.01)
        assert child.returncode == -9
        assert child.wait(timeout=0) == -9
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)
