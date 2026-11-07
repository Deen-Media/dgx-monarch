"""The battery's ssh guard refuses a spawned ssh call before anything is sent."""
from __future__ import annotations

import subprocess

import pytest

from conftest import _ssh_refusing_run


def _recording_real():
    reached: list[object] = []

    def real(args, *_rest, **_kwargs):
        reached.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    return real, reached


def test_a_spawned_ssh_call_never_reaches_the_real_run():
    real, reached = _recording_real()
    spawned: list[str] = []
    run = _ssh_refusing_run(real, spawned)

    result = run(["ssh", "-o", "BatchMode=yes", "192.0.2.1", "true"],
                 capture_output=True, text=True, timeout=5)
    assert reached == []
    assert (result.returncode, result.stdout, result.stderr) == (255, "", "")
    assert spawned == ["ssh -o BatchMode=yes 192.0.2.1 true"]

    piped = run(["/usr/bin/ssh", "192.0.2.1"], stdout=subprocess.PIPE)
    assert (piped.returncode, piped.stdout, piped.stderr) == (255, b"", None)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        run(("ssh", "192.0.2.1"), check=True, capture_output=True)
    assert raised.value.returncode == 255
    assert reached == [] and len(spawned) == 3


def test_every_other_command_still_runs_for_real():
    real, reached = _recording_real()
    spawned: list[str] = []
    run = _ssh_refusing_run(real, spawned)
    assert run(["true"], check=True).returncode == 0
    assert run(["/usr/bin/ssh-keygen", "-l"]).returncode == 0
    assert reached == [["true"], ["/usr/bin/ssh-keygen", "-l"]] and spawned == []


def test_the_autouse_fixture_installs_the_refusing_run():
    assert subprocess.run.__qualname__ == "_ssh_refusing_run.<locals>.record"
