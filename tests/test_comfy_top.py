"""Combined launcher process ownership with fake terminals and CPU-only children."""
from __future__ import annotations

import os
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import comfy_top


@pytest.fixture
def rig(tmp_path, monkeypatch):
    terminal = tmp_path / "terminal"
    terminal.write_text(f"#!{sys.executable}\n" + '''import os,socket,sys,time
from pathlib import Path
mode=os.environ.get("TOP_MODE", "normal")
if mode == "fail": sys.exit(5)
if mode == "stall": time.sleep(30)
args=sys.argv
sock=socket.socket(socket.AF_UNIX)
sock.connect(args[args.index("--ready")+1])
sock.sendall(b"ready\\n")
Path(os.environ["TOP_STARTED"]).write_text(" ".join(args))
if mode == "early":
    time.sleep(.1)
    sys.exit(4)
if mode == "ignore":
    import signal
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    time.sleep(30)
sock.recv(1)
Path(os.environ["TOP_STOPPED"]).write_text("closed")
''')
    terminal.chmod(0o755)
    monkeypatch.setattr(comfy_top, "prerequisites", lambda _config: [str(terminal)])
    monkeypatch.setattr(comfy_top.sys, "stdin", SimpleNamespace(isatty=lambda: True, fileno=lambda: 0))
    monkeypatch.setattr(comfy_top.os, "tcgetpgrp", lambda _fd: 123)
    groups = []
    monkeypatch.setattr(comfy_top, "_foreground", groups.append)
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setenv("TOP_STARTED", str(tmp_path / "started"))
    monkeypatch.setenv("TOP_STOPPED", str(tmp_path / "stopped"))
    return tmp_path, groups


def test_new_driver_preserves_exit_and_closes_only_owned_dashboard(rig):
    root, groups = rig
    assert comfy_top.supervise("/exact/config", "[::1]:8189", [sys.executable, "-c", "raise SystemExit(7)"]) == 7
    assert (root / "stopped").read_text() == "closed"
    assert "--cluster-config /exact/config --host [::1]:8189" in (root / "started").read_text()
    assert len(groups) == 2 and groups[-1] == 123


def test_terminal_failure_prevents_driver_start(rig, monkeypatch):
    root, groups = rig
    monkeypatch.setenv("TOP_MODE", "fail")
    with pytest.raises(ValueError, match="before startup"):
        comfy_top.supervise("/config", "127.0.0.1:8188", [sys.executable, "-c", f"open({str(root / 'driver')!r},'w').close()"])
    assert not (root / "driver").exists()
    assert groups == [123]


def test_reused_driver_branch_only_waits_dashboard(rig, monkeypatch):
    root, groups = rig
    monkeypatch.setenv("TOP_MODE", "early")
    assert comfy_top.supervise("/config", "127.0.0.1:8188", []) == 4
    assert groups == [123]
    assert (root / "started").exists()


def test_closing_dashboard_does_not_stop_new_driver(rig, monkeypatch, capsys):
    root, _groups = rig
    monkeypatch.setenv("TOP_MODE", "early")
    command = [sys.executable, "-c", f"import time; time.sleep(.4); open({str(root / 'done')!r},'w').close()"]
    assert comfy_top.supervise("/config", "127.0.0.1:8188", command) == 0
    assert (root / "done").exists()
    assert "ComfyUI continues" in capsys.readouterr().err


def test_signal_is_forwarded_and_driver_status_preserved(rig):
    root, _groups = rig
    timer = threading.Timer(.3, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        result = comfy_top.supervise("/config", "127.0.0.1:8188", [sys.executable, "-c", "import time; time.sleep(10)"])
    finally:
        timer.cancel()
        timer.join()
    assert result == 143
    assert (root / "stopped").exists()


def test_check_does_not_open_terminal_or_start_children(monkeypatch):
    calls = []
    monkeypatch.setattr(comfy_top, "prerequisites", lambda config: calls.append(config))
    monkeypatch.setattr(comfy_top.subprocess, "Popen", lambda *a, **kw: pytest.fail("check spawned a process"))
    assert comfy_top.main(["--check", "--cluster-config", "/config", "--host", "127.0.0.1:8188"]) == 0
    assert calls == ["/config"]


def test_headless_or_noninteractive_refuses_before_spawn(rig, monkeypatch):
    root, _groups = rig
    monkeypatch.delenv("DISPLAY")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(ValueError, match="graphical"):
        comfy_top.supervise("/config", "127.0.0.1:8188", [])
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(comfy_top.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(ValueError, match="visible terminal"):
        comfy_top.supervise("/config", "127.0.0.1:8188", [])
    assert not (root / "started").exists()


@pytest.mark.parametrize("quit_normally", [False, True])
def test_dashboard_control_channel_stops_only_itself(tmp_path, quit_normally):
    import socket
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="dgxm-top-test-") as directory:
        path = str(Path(directory) / "ready")
        with socket.socket(socket.AF_UNIX) as server:
            server.bind(path)
            server.listen(1)
            server.settimeout(5)
            code = '''import sys,time
from dgx_monarch.cli import comfy_top,main
comfy_top.prerequisites=lambda _config: []
sys.stdin.isatty=lambda: True
sys.stdout.isatty=lambda: True
main.main=lambda args: 4 if sys.argv[2] == 'yes' else time.sleep(30)
raise SystemExit(comfy_top.dashboard('/config','127.0.0.1:8188',sys.argv[1]))
'''
            child = subprocess.Popen([sys.executable, "-c", code, path, "yes" if quit_normally else "no"])
            try:
                connection, _ = server.accept()
                with connection:
                    assert connection.recv(32) == b"ready\n"
                    if quit_normally:
                        assert child.wait(timeout=5) == 4
                if not quit_normally:
                    assert child.wait(timeout=5) == -signal.SIGTERM
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()


def test_terminal_start_timeout_never_starts_driver(rig, monkeypatch):
    root, _groups = rig
    monkeypatch.setenv("TOP_MODE", "stall")
    with pytest.raises(ValueError, match="timed out"):
        comfy_top.supervise("/config", "127.0.0.1:8188", [sys.executable, "-c", f"open({str(root / 'driver')!r},'w').close()"], timeout=.1)
    assert not (root / "driver").exists()


@pytest.mark.parametrize("ignore_interrupt", [False, True])
def test_real_pty_foreground_group_and_exit(tmp_path, monkeypatch, ignore_interrupt):
    terminal = tmp_path / "terminal"
    terminal.write_text(f"#!{sys.executable}\n" + "import socket,sys\ns=socket.socket(socket.AF_UNIX)\ns.connect(sys.argv[sys.argv.index('--ready')+1])\ns.sendall(b'ready\\n')\ns.recv(1)\n")
    terminal.chmod(0o755)
    monkeypatch.setenv("DISPLAY", ":test")
    driver_code = "import os,sys,signal,time; "
    if ignore_interrupt:
        driver_code += "signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); "
    driver_code += "print('FOREGROUND',os.tcgetpgrp(0)==os.getpgrp(),flush=True); "
    driver_code += "time.sleep(30)" if ignore_interrupt else "sys.exit(7)"
    script = (
        "import sys\nfrom dgx_monarch.cli import comfy_top\n"
        f"comfy_top.prerequisites=lambda _config: [{str(terminal)!r}]\n"
        f"raise SystemExit(comfy_top.supervise('/config','127.0.0.1:8188',[sys.executable,'-c',{driver_code!r}],shutdown_grace=.05,terminal_grace=.05))\n"
    )
    # Fork the PTY in a fresh single-threaded interpreter, not pytest's process.
    import subprocess

    harness = "import os,pty,select,signal,sys,time\n" + f"script={script!r}\ninterrupt={ignore_interrupt!r}\n" + """child,master=pty.fork()
if child == 0:
    os.execv(sys.executable,[sys.executable,'-c',script])
output=b''
status=None
sent=False
try:
    deadline=time.monotonic()+7
    while time.monotonic()<deadline:
        if select.select([master],[],[],.1)[0]:
            try: output+=os.read(master,4096)
            except OSError: pass
        if interrupt and not sent and b'FOREGROUND True' in output:
            os.write(master,b'\\x03')
            sent=True
        reaped,result=os.waitpid(child,os.WNOHANG)
        if reaped:
            status=result
            break
    sys.stdout.buffer.write(output)
    sys.stdout.flush()
finally:
    if status is None:
        os.kill(child,signal.SIGKILL)
        os.waitpid(child,0)
    os.close(master)
sys.exit(os.waitstatus_to_exitcode(status) if status is not None else 99)
"""
    result = subprocess.run([sys.executable, "-c", harness], capture_output=True, timeout=10)
    assert result.returncode == (137 if ignore_interrupt else 7), result.stdout + result.stderr
    assert b"FOREGROUND True" in result.stdout
    if ignore_interrupt:
        assert b"actor cleanup is NOT confirmed" in result.stdout


def test_ignored_driver_signal_is_bounded_and_reports_unknown_actor_cleanup(rig, capsys):
    import time

    root, _groups = rig
    command = [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"]
    timer = threading.Timer(.3, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    started = time.monotonic()
    try:
        result = comfy_top.supervise("/config", "127.0.0.1:8188", command, shutdown_grace=.05, terminal_grace=.05)
    finally:
        timer.cancel()
        timer.join()
    assert result == 137
    assert time.monotonic() - started < 3
    assert "actor cleanup is NOT confirmed" in capsys.readouterr().err
    assert (root / "stopped").exists()


def test_reused_driver_cancellation_bounds_an_ignoring_terminal(rig, monkeypatch):
    import time

    _root, groups = rig
    monkeypatch.setenv("TOP_MODE", "ignore")
    timer = threading.Timer(.3, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    started = time.monotonic()
    try:
        assert comfy_top.supervise("/config", "127.0.0.1:8188", [], terminal_grace=.05) == 143
    finally:
        timer.cancel()
        timer.join()
    assert time.monotonic() - started < 3
    assert groups == [123]


def test_resume_continues_stopped_descendants_with_the_driver(tmp_path, monkeypatch):
    import subprocess
    import time

    child_pid = tmp_path / "child"
    child_code = "import time; time.sleep(30)"
    parent_code = f"""import signal,subprocess,sys,time
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c',{child_code!r}])
def cleanup(*_):
    p.terminate()
    p.wait(timeout=3)
    sys.exit(0)
signal.signal(signal.SIGTERM,cleanup)
Path({str(child_pid)!r}).write_text(str(p.pid))
time.sleep(30)
"""
    driver = subprocess.Popen([sys.executable, "-c", parent_code], process_group=0)
    original_kill = os.kill
    monkeypatch.setattr(comfy_top, "_foreground", lambda _group: None)
    def simulate_supervisor_suspend(pid, signum):
        if pid == os.getpid() and signum == signal.SIGTSTP:
            return
        original_kill(pid, signum)
    monkeypatch.setattr(comfy_top.os, "kill", simulate_supervisor_suspend)
    try:
        deadline = time.monotonic() + 3
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        descendant = int(child_pid.read_text())
        os.killpg(driver.pid, signal.SIGSTOP)
        def state(pid):
            return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
        while state(driver.pid) != "T" or state(descendant) != "T":
            assert time.monotonic() < deadline
            time.sleep(.01)
        comfy_top._resume_stopped_driver(driver, 123)
        while state(descendant) == "T":
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert state(driver.pid) != "T"
    finally:
        os.killpg(driver.pid, signal.SIGCONT)
        driver.terminate()
        driver.wait(timeout=3)


def test_exception_cleanup_bounds_an_ignoring_driver_and_preserves_error(rig, monkeypatch):
    import time

    root, _groups = rig
    ready = root / "driver-ready"
    command = [sys.executable, "-c", f"import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path({str(ready)!r}).touch(); time.sleep(30)"]
    def fail_after_driver_ready(*_args):
        deadline = time.monotonic() + 3
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        raise RuntimeError("original supervisor failure")
    monkeypatch.setattr(comfy_top, "_resume_stopped_driver", fail_after_driver_ready)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="original supervisor failure"):
        comfy_top.supervise("/config", "127.0.0.1:8188", command, shutdown_grace=.05, terminal_grace=.05)
    assert time.monotonic() - started < 3
    assert (root / "stopped").exists()


def test_observer_death_settles_owned_driver_instead_of_losing_signal_supervision(rig, monkeypatch):
    root, _groups = rig
    original = comfy_top.ForegroundDriver
    class FailedObserver(original):
        def release(self):
            super().release()
            assert self.observer is not None
            self.observer.kill()
            self.observer.wait(timeout=3)
    monkeypatch.setattr(comfy_top, "ForegroundDriver", FailedObserver)
    with pytest.raises(OSError, match="signal observer exited"):
        comfy_top.supervise("/config", "127.0.0.1:8188", [sys.executable, "-c", "import time; time.sleep(30)"], shutdown_grace=.05, terminal_grace=.05)
    assert (root / "stopped").exists()


@pytest.mark.parametrize("source", ["supervisor", "foreground"])
def test_cancellation_during_observer_handshake_never_releases_driver(rig, monkeypatch, source):
    import select

    root, groups = rig
    marker = root / "must-not-start"
    original = comfy_top.ForegroundDriver
    class InterruptedStartup(original):
        def __init__(self, command):
            super().__init__(command)
            if source == "supervisor":
                os.kill(os.getpid(), signal.SIGTERM)
            else:
                assert self.driver is not None
                os.killpg(self.driver.pid, signal.SIGINT)
                assert select.select([self.events], [], [], 3)[0]
    monkeypatch.setattr(comfy_top, "ForegroundDriver", InterruptedStartup)
    command = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]
    result = comfy_top.supervise("/config", "127.0.0.1:8188", command, shutdown_grace=.05, terminal_grace=.05)
    assert result == (143 if source == "supervisor" else 130)
    assert not marker.exists()
    assert groups == [123]
