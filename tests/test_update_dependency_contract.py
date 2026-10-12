"""Candidate metadata, target markers and prerequisite drift are pre-mutation gates."""
from __future__ import annotations

import importlib.metadata as metadata
import json
import subprocess
from email.message import Message
from types import SimpleNamespace

import pytest
from pip._vendor.packaging import markers

from dgx_monarch.cli import update_support
from dgx_monarch.cli.update_dependency_contract import check_contract, probe_source
from dgx_monarch.config import ClusterConfig, HostConfig


def contract(*requirements, python="", version="0.7.0", extras=()):
    return {"version": version, "requires_python": python, "requirements": list(requirements),
            "provides_extras": list(extras)}


@pytest.fixture
def installed(monkeypatch):
    distributions = {}

    def add(name, version="1.0", requires=(), extras=(), python=""):
        message = Message()
        message["Requires-Python"] = python
        for extra in extras:
            message["Provides-Extra"] = extra
        distributions[name] = SimpleNamespace(version=version, requires=list(requires), metadata=message)

    def lookup(name):
        if name not in distributions:
            raise metadata.PackageNotFoundError(name)
        return distributions[name]

    monkeypatch.setattr(metadata, "distribution", lookup)
    return add


def test_candidate_dependencies_and_bounded_installed_closure(installed):
    installed("numpy", "1.26.4")
    installed("pyarrow", "17.0.0", requires=("numpy>=1.26",))
    installed("xxhash", "3.6.0")
    assert check_contract(contract("numpy>=1.26", "pyarrow>=17", "xxhash"), ["torchmonarch==0.7.0"]) == []
    assert check_contract(contract("pyarrow>=18", "missing"), []) == [
        "missing: missing", "pyarrow: installed 17.0.0, requires >=18"]


def test_conditional_requirements_use_target_environment_and_selected_extras(installed, monkeypatch):
    installed("parent", extras=("fast-mode",), requires=(
        'optional; extra == "other"', 'child>=2; extra == "fast-mode"',
        'linux-only; sys_platform == "linux"'))
    installed("child", "2")
    installed("linux-only")
    target = markers.default_environment()
    monkeypatch.setattr(markers, "default_environment", lambda: {**target, "sys_platform": "linux"})
    assert check_contract(contract("parent[fast_mode]", 'missing; python_version < "3"'), []) == []
    assert check_contract(contract("parent[absent]"), []) == ["parent: requested extra is unavailable"]
    monkeypatch.setattr(markers, "default_environment", lambda: {**target, "sys_platform": "win32"})
    assert check_contract(contract('missing; sys_platform == "linux"'), []) == []


def test_cycle_and_new_extra_revisit_are_bounded(installed):
    installed("a", requires=("b", 'missing; extra == "more"'), extras=("more",))
    installed("b", requires=("a[more]",))
    assert check_contract(contract("a"), []) == ["missing: missing"]


def test_candidate_version_replaces_ambient_monarch_and_python_bounds(installed):
    installed("torchmonarch", "0.6.0", requires=("old-only",))
    assert check_contract(contract(), ["torchmonarch==0.7.0"]) == []
    assert check_contract(contract(python="<3"), []) == ["torchmonarch: incompatible Python version"]
    installed("child", python="<3")
    assert check_contract(contract("child"), []) == ["child: incompatible Python version"]


def test_candidate_extras_come_from_candidate_metadata(installed):
    installed("torchmonarch", "0.6.0", extras=("obsolete",))
    target = contract('xxhash; extra == "fast"', extras=("fast",))
    assert check_contract(target, ["torchmonarch==0.7.0"]) == []
    assert check_contract(target, ["torchmonarch[fast]==0.7.0"]) == ["xxhash: missing"]


def test_unrelated_broken_packages_are_not_inspected(installed):
    installed("unrelated", requires=("absent",))
    assert check_contract(contract(), []) == []


@pytest.mark.parametrize("requirement", ["broken $$$", "package @ https://example.invalid/private"])
def test_invalid_or_unverifiable_requirement_fails_without_disclosing_url(requirement, installed):
    errors = check_contract(contract(requirement), [])
    assert errors and "https" not in str(errors) and "private" not in str(errors)


def test_dependency_graph_limit_refuses(installed):
    for index in range(515):
        installed(f"dep{index}", requires=(f"dep{index + 1}",))
    assert check_contract(contract("dep0"), []) == ["dependency closure exceeds inspection limit"]


def test_shipped_checker_runs_without_package_import(installed, monkeypatch, capsys):
    import sys

    installed("xxhash")
    monkeypatch.setattr(sys, "argv", ["probe", json.dumps({"contract": contract("xxhash"), "roots": []})])
    with pytest.raises(SystemExit) as error:
        exec(compile(probe_source(), "dependency-probe", "exec"), {})  # noqa: S102 - generated production probe
    assert error.value.code == 0
    assert json.loads(capsys.readouterr().out) == []


def test_driver_and_every_worker_checked_without_installs(monkeypatch):
    calls = []
    config = ClusterConfig(hosts=(HostConfig("a", "tcp://192.0.2.1:26600"),
                                  HostConfig("b", "tcp://192.0.2.2:26600")))

    def driver(argv, **kwargs):
        calls.append("driver")
        assert argv[1] == "-c" and kwargs["timeout"] == 60
        assert json.loads(argv[-1])["contract"] == contract("xxhash")
        return subprocess.CompletedProcess(argv, 0, "[]\n", "")

    def worker(_config, host, script, **kwargs):
        calls.append(host.name)
        assert "def check_contract" in script and kwargs["timeout"] == 60
        return subprocess.CompletedProcess([], 0, "[]\n", "")

    monkeypatch.setattr(update_support.lifecycle, "run_on_host", worker)
    assert update_support.candidate_dependencies_available(config, [], contract("xxhash"), driver)
    assert calls == ["driver", "a", "b"]


@pytest.mark.parametrize("output,code", [("[]", 1), ('["xxhash: missing"]', 0), ("not json", 0)])
def test_remote_failed_or_ambiguous_check_refuses(monkeypatch, output, code):
    config = ClusterConfig(hosts=(HostConfig("a", "tcp://192.0.2.1:26600"),))
    monkeypatch.setattr(update_support.lifecycle, "run_on_host",
                        lambda *_a, **_k: subprocess.CompletedProcess([], code, output, ""))
    assert not update_support.candidate_dependencies_available(
        config, [], contract(), lambda *_a, **_k: subprocess.CompletedProcess([], 0, "[]", ""))


def test_timeout_refuses_but_cancellation_propagates():
    def timeout(*_a, **_k):
        raise subprocess.TimeoutExpired([], 60)

    def interrupt(*_a, **_k):
        raise KeyboardInterrupt

    assert not update_support.candidate_dependencies_available(ClusterConfig(), [], contract(), timeout)
    with pytest.raises(KeyboardInterrupt):
        update_support.candidate_dependencies_available(ClusterConfig(), [], contract(), interrupt)
