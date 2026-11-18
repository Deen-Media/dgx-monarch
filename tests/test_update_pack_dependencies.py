"""The verified update's check of the target's own dependencies names what failed.

It runs the real check script, on this interpreter as the driver and through
bash as each configured host, as the update does over ssh. Before, a failed
check printed nothing, and the update read only "release staging failed
before any live change".
"""
from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from types import SimpleNamespace

from dgx_monarch.cli import lifecycle, update_support


def _run(argv, **_kwargs):
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=60)


def _hosts(monkeypatch, count):
    def run_on_host(_config, _host, script, timeout):
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=max(timeout, 60))

    monkeypatch.setattr(lifecycle, "run_on_host", run_on_host)
    return SimpleNamespace(hosts=tuple(SimpleNamespace(name=f"h{n}") for n in range(1, count + 1)),
                           python_bin=sys.executable)


def test_each_unmet_requirement_is_named_with_its_interpreter(monkeypatch, capsys):
    config = _hosts(monkeypatch, 2)
    installed = importlib.metadata.version("pytest")
    satisfied = update_support.dependencies_available(
        config, ["torchmonarch==9.9.9", "dgxm-absent-package==1.0", "pytest<1", "pytest>=1"], _run)
    assert satisfied is False
    printed = capsys.readouterr().err.splitlines()
    for label in ("driver", "host_1", "host_2"):
        assert f"  {label}: dgxm-absent-package: missing" in printed
        assert f"  {label}: pytest: installed {installed}, requires <1" in printed
    # torchmonarch is the candidate contract's to check, and a met requirement is silent.
    assert not any("torchmonarch" in line or "requires >=1" in line for line in printed)


def test_a_satisfied_target_passes_without_a_word(monkeypatch, capsys):
    config = _hosts(monkeypatch, 1)
    assert update_support.dependencies_available(config, ["pytest>=1"], _run) is True
    assert capsys.readouterr().err == ""


def test_a_host_the_check_cannot_reach_is_named(monkeypatch, capsys):
    def unreachable(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ssh", 60)

    monkeypatch.setattr(lifecycle, "run_on_host", unreachable)
    config = SimpleNamespace(hosts=(SimpleNamespace(name="h1"),), python_bin=sys.executable)
    assert update_support.dependencies_available(config, ["pytest>=1"], _run) is False
    assert "  host_1: dependency inspection failed" in capsys.readouterr().err


def test_the_candidate_check_names_every_interpreter_that_lacks_a_requirement(monkeypatch, capsys):
    """The pin's own contract, read from a candidate's METADATA, checked by the
    real probe on every Python. It used to stop at the first that failed, so
    the update named one Python while others lacked the package too."""
    from dgx_monarch.cli.update_dependency_contract import dependency_contract

    metadata = (b"Metadata-Version: 2.1\nName: torchmonarch\nVersion: 9.9.9\n"
                b"Requires-Dist: dgxm-absent-package\n")
    contract = dependency_contract(metadata, "9.9.9", ValueError)
    config = _hosts(monkeypatch, 2)
    assert update_support.candidate_dependencies_available(
        config, ["torchmonarch==9.9.9"], contract, _run) is False
    printed = capsys.readouterr().err.splitlines()
    assert printed == [f"  {label}: dgxm-absent-package: missing"
                       for label in ("driver", "host_1", "host_2")]
